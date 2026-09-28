#!/usr/bin/env bash
# run_sqa.sh — ScholarQA-Bench, +Librarian row.
#   --only bio | neu | multi ; default: all three.
#
# bio and neu run the full sets (1451 and 1308 questions) and report Citation F1
# only: no Prometheus scores exist for them upstream. They call
# run_sqa_new_stack.py directly, which runs AutoAIS itself at the end, so they
# need one GPU and no container.
#
# multi reports Citation F1 and both Prometheus judges, through run_sqa_multi.sh.
# It needs apptainer and $JUDGE_GPUS GPUs for prometheus-bgb-8x7b-v2.0 and
# prometheus-8x7b-v2.0 at --tensor-parallel-size 4. Without them it is skipped,
# unless it was asked for by name, which is an error.
#
# --via-api runs bio/neu on the orchestrator (POST /run-agent/stream) rather than
# on in-process agents; LIBRARIAN_URL and the ablation flags are then unused,
# because the pods run their deployed configuration. It applies to multi too,
# which still needs its GPUs for the judges.
#
# --skip-citation-eval stops a bio/neu run after the answers. Generation is pure
# network and AutoAIS needs a GPU, so the cheap split is to answer on a CPU box
# and re-run the same command without the flag on a GPU one: --resume goes
# straight to scoring.
#
# --no-librarian answers from the model alone (the parametric baseline row); it
# and the librarian ablation flags force the in-process path.
#
# Env: LIBRARIAN_URL (required unless --via-api), LIBRARIAN_MODEL, SYNTHESIS_MODEL,
#      RESULTS_ROOT, LIMIT, MAX_WORKERS, JUDGE_GPUS, ONLY,
#      VIA_API, LIBRARIAN_API_URL, SKIP_CITATION_EVAL, DRY_RUN.
set -euo pipefail

SQA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SQA_DIR/../../.." && pwd)"

# Two environments: `sqa` generates answers, `sqa-scoring` is the CUDA AutoAIS
# stack. Only the one a run reaches is built.
# shellcheck source=evals/Literature/bench_env.sh
. "$SQA_DIR/../bench_env.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LIBRARIAN_URL="${LIBRARIAN_URL:-}"
LIBRARIAN_MODEL="${LIBRARIAN_MODEL:-glm-5-fp8}"
SYNTHESIS_MODEL="${SYNTHESIS_MODEL:-$LIBRARIAN_MODEL}"
RESULTS_ROOT="${RESULTS_ROOT:-$SQA_DIR/../results}"
JUDGE_GPUS="${JUDGE_GPUS:-4}"
VIA_API="${VIA_API:-false}"
SKIP_CITATION_EVAL="${SKIP_CITATION_EVAL:-false}"
BASELINE_ARGS=()
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-false}"

while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="${ONLY:+$ONLY,}$2"; shift 2 ;;
        --via-api) VIA_API=true; shift ;;
        --skip-citation-eval) SKIP_CITATION_EVAL=true; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        # Baseline and ablation arms, forwarded verbatim to run_sqa_new_stack.py.
        # None has an API equivalent -- the pods run their deployed config -- so
        # each also turns --via-api off rather than quietly measuring the wrong
        # thing. literature_eval.sh does the same for a run started there.
        --no-librarian|--no-librarian-full-text)
            BASELINE_ARGS+=("$1"); VIA_API=false; shift ;;
        --librarian-num-subqueries|--librarian-paragraphs-per-subquery)
            BASELINE_ARGS+=("$1" "$2"); VIA_API=false; shift 2 ;;
        -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Each --via-api worker costs one round trip; each in-process worker also runs
# Stage-2 BM25 on this box, so it gets the lower default. Both change
# wall-clock, not scores -- see [concurrency] in ../literature_eval.toml.
if [ "$VIA_API" = true ]; then
    MAX_WORKERS="${MAX_WORKERS:-16}"
else
    MAX_WORKERS="${MAX_WORKERS:-4}"
    [ -n "$LIBRARIAN_URL" ] || { echo "ERROR: set LIBRARIAN_URL (or pass --via-api)." >&2; exit 1; }
fi

want() { [ -z "$ONLY" ] || grep -qx "$1" <<< "${ONLY//,/$'\n'}"; }

PYTHON_BIN="$(bench_python_maybe sqa "$DRY_RUN")"
# Several GB, so build it only when a run will actually score.
if [ "$SKIP_CITATION_EVAL" != true ]; then
    SQA_SCORING_PYTHON="$(bench_python_maybe sqa-scoring "$DRY_RUN")"
    export SQA_SCORING_PYTHON
fi

# `neu` in the paper table is spelled `neuro` on the command line.
run_autoais_bench() {
    local target="$1" bench="$2"
    local out_dir="$RESULTS_ROOT/sqa_$target"
    local cmd=(
        env SCHOLARQA_OUTPUT_DIR="$out_dir"
        "$PYTHON_BIN" "$SQA_DIR/run_sqa_new_stack.py" --bench "$bench"
        --synthesis-base-url "$LIBRARIAN_URL" --synthesis-model "$SYNTHESIS_MODEL"
        --librarian-base-url "$LIBRARIAN_URL" --librarian-model "$LIBRARIAN_MODEL"
        --max-workers "$MAX_WORKERS" --resume
    )
    [ "$VIA_API" = true ] && cmd+=(--via-api)
    [ "$SKIP_CITATION_EVAL" = true ] && cmd+=(--skip-citation-eval)
    cmd+=(${BASELINE_ARGS[@]+"${BASELINE_ARGS[@]}"})
    [ -n "${LIMIT:-}" ] && cmd+=(--limit "$LIMIT")
    echo "RUN   ${cmd[*]}"
    echo "      -> Citation F1 only for $target (no Prometheus scores exist for bio/neuro)."
    [ "$DRY_RUN" = true ] && return 0
    mkdir -p "$out_dir"
    "${cmd[@]}"
}

if want bio; then run_autoais_bench bio bio; fi
if want neu; then run_autoais_bench neu neuro; fi

# multi alone needs a container image, apptainer and $JUDGE_GPUS GPUs. Asking
# for it by name without them is an error; in a default run it is skipped and
# the Citation F1 rows that did run are kept.
multi_skip_reason=""
if want multi && [ -z "${APPTAINER_IMAGE:-}" ]; then
    multi_skip_reason="no container image set — put one in \`[sqa] apptainer_image\`
       (literature_eval.toml), e.g. docker://vllm/vllm-openai:v0.29.0"
fi

if want multi && [ -n "$multi_skip_reason" ]; then
    if [ -n "$ONLY" ]; then
        echo "ERROR: --only multi needs a container for the Prometheus judges:" >&2
        echo "       $multi_skip_reason" >&2
        exit 1
    fi
    echo "SKIP  multi — $multi_skip_reason"
    echo "      bio and neuro above are unaffected; they need no container."
elif want multi; then
    # Defaults to data/scholarqa_multi/scholar_multi_biomed_eval.json — the
    # paper's Bio+Neu subset of the multidisciplinary questions.
    cmd=(
        bash "$SQA_DIR/run_sqa_multi.sh"
        --librarian-model "$LIBRARIAN_MODEL" --librarian-base-url "${LIBRARIAN_URL%/}/"
        --synthesis-model "$SYNTHESIS_MODEL" --synthesis-base-url "${LIBRARIAN_URL%/}/"
        --judge-gpus "$JUDGE_GPUS" --resume
        --max-workers "$MAX_WORKERS"
        --output-dir "$RESULTS_ROOT/sqa_multi_bio"
    )
    [ "$VIA_API" = true ] && cmd+=(--via-api)
    cmd+=(${BASELINE_ARGS[@]+"${BASELINE_ARGS[@]}"})
    [ -n "${LIMIT:-}" ] && cmd+=(--limit "$LIMIT")
    echo "RUN   ${cmd[*]}"
    if [ "$DRY_RUN" != true ]; then
        # Prometheus at TP=4 will not fit on fewer GPUs; say so now.
        env_problem=""
        command -v apptainer >/dev/null || env_problem="apptainer is not installed"
        if [ -z "$env_problem" ]; then
            gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
            [ "$gpu_count" -ge "$JUDGE_GPUS" ] \
                || env_problem="the Prometheus judges need >= $JUDGE_GPUS GPUs; found $gpu_count"
        fi
        if [ -n "$env_problem" ]; then
            if [ -n "$ONLY" ]; then
                echo "ERROR: --only multi cannot run here: $env_problem." >&2
                exit 1
            fi
            echo "SKIP  multi — $env_problem."
            echo "      bio and neuro above are unaffected; they need neither."
        else
            mkdir -p "$RESULTS_ROOT/sqa_multi_bio"
            "${cmd[@]}"
        fi
    fi
fi
