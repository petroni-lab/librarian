#!/usr/bin/env bash
# run_sqa_multi.sh — the ScholarQA-Multi target: Citation F1 and both Prometheus judges.
#
# run_sqa.sh calls this for `--only multi`. The bio and neuro targets need no
# container and call run_sqa_new_stack.py directly.
#
# Four steps:
#   1. answer the questions            run_sqa_new_stack.py
#   2. score citations                 scorers/citation_correctness_eval.py (AutoAIS)
#   3. organization + coverage judge   prometheus-bgb-8x7b-v2.0
#   4. relevance judge                 prometheus-8x7b-v2.0
#
# Each judge is served by vLLM inside $APPTAINER_IMAGE at --tensor-parallel-size
# $JUDGE_GPUS, started and stopped around its own pass, so the two never hold
# GPUs at once. Both write into one results.json; prometheus_eval.py merges.
#
# Nothing here serves the librarian: it answers at --librarian-base-url, or on
# the orchestrator under --via-api.
#
# Env: APPTAINER_IMAGE (required), PORT, SERVER_WAIT_SECONDS, MIN_FREE_MB,
#      HF_HOME, SQA_SCRATCH_DIR, PYTHON_BIN, SQA_SCORING_PYTHON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Inherited from run_sqa.sh when launched through it; resolved here for a
# standalone invocation. See ../bench_env.sh.
# shellcheck source=evals/Literature/bench_env.sh
. "$SCRIPT_DIR/../bench_env.sh"
PYTHON_BIN="${PYTHON_BIN:-$(bench_python sqa)}"

DATA_FILE="$SCRIPT_DIR/data/scholarqa_multi/scholar_multi_biomed_eval.json"
OUTPUT_DIR="$SCRIPT_DIR/../results/sqa_multi_bio"
PRED_FILE=""

LIBRARIAN_MODEL=""
LIBRARIAN_BASE_URL=""
SYNTHESIS_MODEL=""
SYNTHESIS_BASE_URL=""
VIA_API=false
PREDICT_ARGS=()

JUDGE_GPUS=4
JUDGE_PASS="all"
SKIP_PREDICTIONS=false
SKIP_CITATION=false
SKIP_JUDGES=false

MAX_WORKERS=1
MAX_RETRIES=2
LIMIT=""
RESUME=false
VERBOSE=false

# vLLM in $APPTAINER_IMAGE serves each judge here in turn.
PORT="${PORT:-8000}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-}"
# The judges are 87 GB each and load from scratch, routinely over 15 minutes.
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-3600}"
# Free VRAM a GPU must have before a judge is placed on it.
MIN_FREE_MB="${MIN_FREE_MB:-0}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# The judge container's pip cache and TMPDIR. Per-user: two people on one node
# must not share them.
SQA_SCRATCH_DIR="${SQA_SCRATCH_DIR:-${TMPDIR:-/tmp}/sqa-multi-${USER:-$(id -un)}}"

# Upstream ScholarQABench's Prometheus recipe: organization and coverage on the
# bgb judge, relevance on the base one.
JUDGE_INSTRUCTION="Answer the question related to the most recent scientific literature."
JUDGE_TOP_N=10
JUDGE_MAX_NEW_TOKENS=512
RUBRIC_PATH="$SCRIPT_DIR/code/rubrics/prometheus_rubrics_v8.json"

usage() { sed -n '2,21p' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --data-file) DATA_FILE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --pred-file) PRED_FILE="$2"; shift 2 ;;
        --librarian-model) LIBRARIAN_MODEL="$2"; shift 2 ;;
        --librarian-base-url) LIBRARIAN_BASE_URL="$2"; shift 2 ;;
        --synthesis-model) SYNTHESIS_MODEL="$2"; shift 2 ;;
        --synthesis-base-url) SYNTHESIS_BASE_URL="$2"; shift 2 ;;
        --via-api) VIA_API=true; shift ;;
        # Arms, forwarded verbatim to run_sqa_new_stack.py.
        --no-librarian|--no-librarian-full-text)
            PREDICT_ARGS+=("$1"); shift ;;
        --librarian-num-subqueries|--librarian-paragraphs-per-subquery)
            PREDICT_ARGS+=("$1" "$2"); shift 2 ;;
        --judge-gpus) JUDGE_GPUS="$2"; shift 2 ;;
        --judge-pass) JUDGE_PASS="$2"; shift 2 ;;
        --skip-predictions) SKIP_PREDICTIONS=true; shift ;;
        --skip-citation) SKIP_CITATION=true; shift ;;
        --skip-judges) SKIP_JUDGES=true; shift ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --resume) RESUME=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "$JUDGE_PASS" in
    all|org_cov|relevance) ;;
    *) echo "ERROR: --judge-pass must be all, org_cov, or relevance" >&2; exit 1 ;;
esac
[ -f "$DATA_FILE" ] || { echo "ERROR: no data file at $DATA_FILE" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
JUDGE_EVAL_DIR="$OUTPUT_DIR/judge_eval"
echo "Output dir: $OUTPUT_DIR"

# Resolved only if a scoring step is reached: the scoring environment is several
# GB, and a run that skips citation eval must not build it.
scoring_python() {
    if [ -z "${SQA_SCORING_PYTHON:-}" ]; then
        SQA_SCORING_PYTHON="$(bench_python sqa-scoring)"
        export SQA_SCORING_PYTHON
    fi
    printf '%s' "$SQA_SCORING_PYTHON"
}

# ── Step 1: predictions ──────────────────────────────────────────────────────
run_predictions() {
    local log_file args=(--bench multi --data-file "$DATA_FILE" --max-retries "$MAX_RETRIES")
    log_file="$(mktemp)"

    [ -n "$SYNTHESIS_MODEL" ] && args+=(--synthesis-model "$SYNTHESIS_MODEL")
    [ -n "$SYNTHESIS_BASE_URL" ] && args+=(--synthesis-base-url "$SYNTHESIS_BASE_URL")
    [ -n "$LIBRARIAN_MODEL" ] && args+=(--librarian-model "$LIBRARIAN_MODEL")
    [ -n "$LIBRARIAN_BASE_URL" ] && args+=(--librarian-base-url "$LIBRARIAN_BASE_URL")
    [ "$VIA_API" = true ] && args+=(--via-api)
    [ "$MAX_WORKERS" -gt 1 ] && args+=(--max-workers "$MAX_WORKERS")
    [ -n "$LIMIT" ] && args+=(--limit "$LIMIT")
    [ "$RESUME" = true ] && args+=(--resume)
    [ "$VERBOSE" = true ] && args+=(--verbose)
    args+=(${PREDICT_ARGS[@]+"${PREDICT_ARGS[@]}"})

    echo "Running predictions..."
    (
        cd "$REPO_ROOT"
        env SCHOLARQA_OUTPUT_DIR="$OUTPUT_DIR" \
            "$PYTHON_BIN" evals/Literature/SQA_bench/run_sqa_new_stack.py "${args[@]}"
    ) | tee "$log_file"

    # run_sqa_new_stack.py prints "  Predictions: <path>" as its last summary line.
    PRED_FILE="$(awk '/Predictions:/ {print $2}' "$log_file" | tail -1)"
    rm -f "$log_file"
    [ -f "$PRED_FILE" ] \
        || { echo "ERROR: run_sqa_new_stack.py produced no predictions file" >&2; exit 1; }
    echo "Predictions file: $PRED_FILE"
}

# ── Step 2: Citation F1 ──────────────────────────────────────────────────────
run_citation_eval() {
    if [ -s "$PRED_FILE.score_post_fix" ]; then
        echo "Citation score already exists: $PRED_FILE.score_post_fix"
        return
    fi
    echo "Running citation correctness (AutoAIS)..."
    (
        cd "$SCRIPT_DIR/scorers"   # citation_correctness_eval.py imports run_utils flat
        "$(scoring_python)" citation_correctness_eval.py \
            --f "$PRED_FILE" --citations \
            --autoais_chunk_size 200 --autoais_reload_model_per_chunk \
            --per_question_output "$PRED_FILE.autoais_per_question.json"
    )
}

# ── Steps 3 and 4: the Prometheus judges ─────────────────────────────────────
# Pick the $1 GPUs with the most free memory, so a judge lands beside someone
# else's work only when the node has nothing better.
select_gpus() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2,2nr \
        | awk -F, -v n="$1" -v min="$MIN_FREE_MB" \
            '$2+0 >= min { printf "%s%s", sep, $1+0; sep=","; if (++c == n) exit }'
}

SERVER_PID=""
stop_judge() {
    [ -n "$SERVER_PID" ] || return 0
    echo "Stopping the judge (process group $SERVER_PID)..."
    kill -TERM "-$SERVER_PID" 2>/dev/null || true
    sleep 10
    kill -KILL "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
trap stop_judge EXIT

start_judge() {
    local hf_model="$1" served_name="$2" gpus
    gpus="$(select_gpus "$JUDGE_GPUS")"
    local found=0
    [ -n "$gpus" ] && found=$(($(tr -cd , <<<"$gpus" | wc -c) + 1))
    [ "$found" -eq "$JUDGE_GPUS" ] \
        || { echo "ERROR: need $JUDGE_GPUS GPUs with >= ${MIN_FREE_MB}MB free; found $found." >&2; exit 1; }

    if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "ERROR: something is already serving on port $PORT. Stop it first." >&2
        exit 1
    fi

    local log="$OUTPUT_DIR/vllm_${served_name}.log"
    echo "Starting $served_name on CUDA_VISIBLE_DEVICES=$gpus (log: $log)"
    # --cleanenv drops the caller's environment, so NCCL tuning has to be
    # forwarded explicitly; each is empty unless exported. PYTHONNOUSERSITE is
    # not optional: --cleanenv does not stop Python reading ~/.local, where a
    # user-site transformers shadows the container's. The bind targets are
    # created first -- apptainer refuses to bind a source that does not exist.
    mkdir -p "$HF_HOME" "$SQA_SCRATCH_DIR/pip-cache" "$SQA_SCRATCH_DIR/tmp"
    setsid apptainer exec --nv --cleanenv \
        --env CUDA_VISIBLE_DEVICES="$gpus" \
        ${NCCL_DEBUG:+--env NCCL_DEBUG="$NCCL_DEBUG"} \
        ${NCCL_P2P_DISABLE:+--env NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE"} \
        ${NCCL_SHM_DISABLE:+--env NCCL_SHM_DISABLE="$NCCL_SHM_DISABLE"} \
        ${NCCL_IB_DISABLE:+--env NCCL_IB_DISABLE="$NCCL_IB_DISABLE"} \
        ${NCCL_CUMEM_ENABLE:+--env NCCL_CUMEM_ENABLE="$NCCL_CUMEM_ENABLE"} \
        --bind "$HF_HOME:/root/.cache/huggingface:rw" \
        --bind "$SQA_SCRATCH_DIR/pip-cache:/opt/pip-cache:rw" \
        --bind "$SQA_SCRATCH_DIR/tmp:/opt/tmp:rw" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env HF_HOME=/root/.cache/huggingface \
        --env HF_HUB_CACHE=/root/.cache/huggingface/hub \
        --env HF_HUB_DISABLE_XET=1 \
        --env PYTHONNOUSERSITE=1 \
        --env XDG_CACHE_HOME=/root/.cache/huggingface \
        --env TMPDIR=/opt/tmp \
        --env PIP_CACHE_DIR=/opt/pip-cache \
        "$APPTAINER_IMAGE" \
        bash -lc "vllm serve $hf_model --served-model-name $served_name \
            --host 0.0.0.0 --port $PORT --tensor-parallel-size $JUDGE_GPUS \
            --gpu-memory-utilization 0.90" >"$log" 2>&1 &
    SERVER_PID=$!

    echo "Waiting up to ${SERVER_WAIT_SECONDS}s for $served_name ..."
    local deadline=$((SECONDS + SERVER_WAIT_SECONDS))
    until curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
        kill -0 "$SERVER_PID" 2>/dev/null \
            || { echo "ERROR: $served_name exited before serving:" >&2; tail -40 "$log" >&2; exit 1; }
        [ "$SECONDS" -lt "$deadline" ] \
            || { echo "ERROR: $served_name did not come up in ${SERVER_WAIT_SECONDS}s." >&2
                 echo "       Raise SERVER_WAIT_SECONDS if it was still loading." >&2
                 tail -40 "$log" >&2; exit 1; }
        sleep 5
    done
    echo "$served_name is ready"
}

# Both passes write into one results.json, which prometheus_eval.py merges.
run_judge() {
    local hf_model="$1" served_name="$2"; shift 2
    start_judge "$hf_model" "$served_name"
    echo "Scoring $* ..."
    "$(scoring_python)" "$SCRIPT_DIR/scorers/prometheus_eval.py" \
        -b "$PRED_FILE" -o "$JUDGE_EVAL_DIR" -f "$DATA_FILE" \
        --model litellm_openai \
        --litellm_api_base "http://localhost:$PORT/v1" \
        --litellm_model "$served_name" \
        --rubric_path "$RUBRIC_PATH" \
        --instruction "$JUDGE_INSTRUCTION" \
        --top_n "$JUDGE_TOP_N" --max_new_tokens "$JUDGE_MAX_NEW_TOKENS" \
        --aspects "$@"
    stop_judge
}

# ── Run ──────────────────────────────────────────────────────────────────────
if [ "$SKIP_PREDICTIONS" = true ]; then
    [ -f "$PRED_FILE" ] || { echo "ERROR: --skip-predictions needs --pred-file" >&2; exit 1; }
    PRED_FILE="$(cd "$(dirname "$PRED_FILE")" && pwd)/$(basename "$PRED_FILE")"
else
    run_predictions
fi

if [ "$SKIP_CITATION" = true ]; then
    echo "Skipping citation correctness."
else
    run_citation_eval
fi

if [ "$SKIP_JUDGES" = true ]; then
    echo "Skipping the Prometheus judges."
else
    [ -n "$APPTAINER_IMAGE" ] \
        || { echo "ERROR: the judges need an image; set [sqa] apptainer_image." >&2; exit 1; }
    mkdir -p "$JUDGE_EVAL_DIR"
    if [ "$JUDGE_PASS" = all ] || [ "$JUDGE_PASS" = org_cov ]; then
        run_judge prometheus-eval/prometheus-bgb-8x7b-v2.0 prometheus-bgb-8x7b-v2.0 \
            organization coverage
    fi
    if [ "$JUDGE_PASS" = all ] || [ "$JUDGE_PASS" = relevance ]; then
        run_judge prometheus-eval/prometheus-8x7b-v2.0 prometheus-8x7b-v2.0 relevance
    fi
fi

echo "Done."
echo "  Predictions:    $PRED_FILE"
[ "$SKIP_CITATION" = true ] || echo "  Citation score: $PRED_FILE.score_post_fix"
[ "$SKIP_JUDGES" = true ] || echo "  Judge scores:   $JUDGE_EVAL_DIR/results.json"
