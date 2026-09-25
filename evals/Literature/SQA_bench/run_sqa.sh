#!/usr/bin/env bash
# run_sqa.sh — ScholarQA-Bench, +Librarian row.
#   --only bio | neu | multi ; default: all three.
#
# What is reported, and what is not:
#   bio, neu   Citation F1 only. There are no Prometheus/LLM scores for Bio and
#              Neuro and there never were — judge_eval/ exists only under
#              scholarqa_multi; bio/neuro hold only .score_post_fix (AutoAIS).
#   multi      Citation F1 + both Prometheus judges.
# Bio and Neu run the FULL sets (1451 and 1308 entries); the --sample 0.2 that
# appears in shell history was an earlier probe, not the paper run.
#
# Bio/Neu call run_sqa_new_stack.py directly — it runs AutoAIS itself at the end,
# no container needed. Only multi goes through the apptainer wrapper, for
# Prometheus. The wrapper is always passed --remote-librarian, so it points at
# $LIBRARIAN_URL rather than trying to serve the librarian model itself.
#
# HARD REQUIREMENT for --only multi: a multi-GPU node. The judges are
# prometheus-eval/prometheus-bgb-8x7b-v2.0 and prometheus-eval/prometheus-8x7b-v2.0
# at --tensor-parallel-size 4. Apptainer alone is not enough. bio/neu need one GPU
# (AutoAIS) and nothing more.
#
# Baseline rows: --no-librarian (parametric), or
#   --bm25-retrieval --es-url ...:9201 --es-fulltext-url ...:9202 (BM25).
# Both are accepted here and forwarded; either also forces the in-process path,
# since the orchestrator only ever runs the full librarian.
#
# --via-api applies to all three targets. multi still needs $JUDGE_GPUS GPUs for
# the Prometheus judges, but no longer needs any for the librarian.
#
# --via-api runs bio/neu on the orchestrator (POST /run-agent/stream) instead of
# in-process agents: Stage-2 BM25 lands on pod CPUs and a question costs one
# round trip instead of one per LLM call. LIBRARIAN_URL is then unused -- the
# pods talk to vLLM in-cluster -- and so are the ablation flags, which the API
# has no equivalent for. --only multi is unaffected (apptainer + Prometheus).
#
# Env: LIBRARIAN_URL (required unless --via-api), LIBRARIAN_MODEL, SYNTHESIS_MODEL,
#      RESULTS_ROOT, LIMIT, MAX_WORKERS, JUDGE_GPUS, ONLY,
#      VIA_API, LIBRARIAN_API_URL, SKIP_CITATION_EVAL, DRY_RUN.
#
# --skip-citation-eval splits the two halves of a bio/neu run by hardware: answer
# generation needs no GPU at all under --via-api, while AutoAIS loads
# google/t5_xxl_true_nli_mixture (11B) and does. Generate on a CPU node, then
# re-run the same command WITHOUT the flag on a GPU node: --resume finds every
# answer present and goes straight to scoring.
set -euo pipefail

SQA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SQA_DIR/../../.." && pwd)"

# Two environments, because the two halves run on different machines: `sqa`
# generates answers and needs no GPU, `sqa-scoring` is torch + transformers
# against a local CUDA. Only the half that is actually about to run is built,
# so answering questions on a laptop never installs a scoring stack.
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
        --no-librarian|--bm25-retrieval|--no-librarian-full-text)
            BASELINE_ARGS+=("$1"); VIA_API=false; shift ;;
        --es-url|--es-fulltext-url|--top-k|--librarian-num-subqueries|--librarian-paragraphs-per-subquery)
            BASELINE_ARGS+=("$1" "$2"); VIA_API=false; shift 2 ;;
        -h|--help) sed -n '1,45p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Concurrency means different things per transport, so the defaults do too.
# 16 is MEASURED, not the 64 agent slots dev exposes (8 replicas x
# MAX_CONCURRENT_AGENTS=8). Those slots are not the ceiling: every agent run
# fans out to _JUDGE_WORKERS=8 concurrent LLM calls, so N runs put up to 8N
# requests on the ONE shared vLLM behind all 8 replicas, and past ~128 its KV
# cache thrashes. 32 questions of scholarqa_bio, 2026-09-04:
#   --max-workers 8  -> 5m56s (11.1 s/q)
#   --max-workers 16 -> 4m49s ( 9.1 s/q)   <- best
#   --max-workers 32 -> 9m32s (17.9 s/q)   <- 1.6x SLOWER than 8
# One sample per point on shared infrastructure, so re-measure if the dev vLLM
# or _JUDGE_WORKERS changes. In-process every worker instead runs Stage-2 BM25
# on this box, so its default stays low.
if [ "$VIA_API" = true ]; then
    MAX_WORKERS="${MAX_WORKERS:-16}"
else
    MAX_WORKERS="${MAX_WORKERS:-4}"
    [ -n "$LIBRARIAN_URL" ] || { echo "ERROR: set LIBRARIAN_URL (or pass --via-api)." >&2; exit 1; }
fi

want() { [ -z "$ONLY" ] || grep -qx "$1" <<< "${ONLY//,/$'\n'}"; }

PYTHON_BIN="$(bench_python_maybe sqa "$DRY_RUN")"
# The scoring environment is several GB and only some runs reach it, so it is
# built when a run will actually score, and not otherwise.
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

# multi is the one target with requirements the other two do not share: a
# container image, apptainer, and $JUDGE_GPUS GPUs. Asking for it explicitly and
# not having them is an error; running the default three on a box that cannot
# host the judges skips it and keeps the Citation F1 rows that did run.
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
        bash "$SQA_DIR/run_local_multieval_new_stack_apptainer.sh"
        --librarian-model "$LIBRARIAN_MODEL" --librarian-base-url "${LIBRARIAN_URL%/}/"
        --synthesis-model "$SYNTHESIS_MODEL" --synthesis-base-url "${LIBRARIAN_URL%/}/"
        --judge-gpus "$JUDGE_GPUS" --resume
        --max-workers "$MAX_WORKERS"
        --output-dir "$RESULTS_ROOT/sqa_multi_bio"
    )
    # --via-api already implies --remote-librarian inside the wrapper.
    [ "$VIA_API" = true ] && cmd+=(--via-api)
    cmd+=(${BASELINE_ARGS[@]+"${BASELINE_ARGS[@]}"})
    # Nothing here serves a librarian, so the wrapper is always told not to
    # start one and to use $LIBRARIAN_URL instead.
    if [ "$VIA_API" != true ]; then cmd+=(--remote-librarian); fi
    [ -n "${LIMIT:-}" ] && cmd+=(--limit "$LIMIT")
    echo "RUN   ${cmd[*]}"
    if [ "$DRY_RUN" != true ]; then
        # Prometheus at TP=4 will not fit on fewer GPUs; say so now, not 40
        # minutes in.
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
