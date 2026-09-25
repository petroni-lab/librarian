#!/usr/bin/env bash
# run_labbench.sh — LAB-Bench, +Librarian row: SeqQA 63.8, ProtocolQA 73.1,
# DbQA 36.2, CloningScenarios 48.5.
#
# Normally invoked through ../literature_eval.sh --bench labbench; also runs
# standalone with LIBRARIAN_URL set.
#
# Loops the paper's four tasks x {gpt-5.4, gpt-4o}. --only takes a task name or a
# model name and narrows to it. TableQA exists in the runner but is not in the
# paper table, so it is excluded by default; reach it with --only TableQA.
#
# No --data-file: the gated HF futurehouse/lab-bench dataset is the paper's ~80%
# public subset (n = 520 / 600 / 108 / 33) and is streamed at run time, so
# `hf auth login` is the only data prerequisite.
#
# Baseline row: the same command with --mode baseline (parametric LLM).
#
# Env: LIBRARIAN_URL (required), LIBRARIAN_MODEL, ANSWER_MODEL,
#      LABBENCH_GPT4O_MODEL, RESULTS_ROOT, LIMIT, MAX_WORKERS, ONLY, DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# This bench runs in its own locked environment, built on first use. The
# repository is not a package, so its code arrives on PYTHONPATH.
# shellcheck source=evals/Literature/bench_env.sh
. "$SCRIPT_DIR/../bench_env.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LIBRARIAN_URL="${LIBRARIAN_URL:-}"
LIBRARIAN_MODEL="${LIBRARIAN_MODEL:-glm-5-fp8}"
ANSWER_MODEL="${ANSWER_MODEL:-gpt-5.4}"
# The plain alias, not the -2024-05-13 snapshot the runner itself defaults to.
LABBENCH_GPT4O_MODEL="${LABBENCH_GPT4O_MODEL:-gpt-4o}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/../results}"
# 16 on the API path; 8 in-process, where each worker also runs Stage-2 BM25
# on this box.
if [ "${VIA_API:-false}" = true ]; then
    MAX_WORKERS="${MAX_WORKERS:-16}"
else
    MAX_WORKERS="${MAX_WORKERS:-8}"
fi
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-false}"

TASKS=(SeqQA ProtocolQA DbQA CloningScenarios)
MODELS=("$ANSWER_MODEL" "$LABBENCH_GPT4O_MODEL")

while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="${ONLY:+$ONLY,}$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) sed -n '/^set -euo/q;p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Only the in-process path opens this endpoint.
[ "${VIA_API:-false}" = true ] || [ -n "$LIBRARIAN_URL" ] || { echo "ERROR: set LIBRARIAN_URL (or use the API path)." >&2; exit 1; }

# --only matches either axis: `--only DbQA` and `--only gpt-4o` both work, and
# repeating or comma-separating it narrows both.
if [ -n "$ONLY" ]; then
    only_tasks=() only_models=()
    IFS=',' read -r -a only_tokens <<< "$ONLY"
    for token in "${only_tokens[@]}"; do
        case "$token" in
            SeqQA|ProtocolQA|DbQA|CloningScenarios|TableQA) only_tasks+=("$token") ;;
            *) only_models+=("$token") ;;
        esac
    done
    [ "${#only_tasks[@]}" -gt 0 ] && TASKS=("${only_tasks[@]}")
    [ "${#only_models[@]}" -gt 0 ] && MODELS=("${only_models[@]}")
fi

# Resolved once; a dry run prints the path and builds nothing.
LABBENCH_PYTHON="$(bench_python_maybe labbench "$DRY_RUN")"

ran_any=false
for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        # The retriever is part of the directory name, which is what --resume
        # keys on; the two paths must not share one.
        if [ "${VIA_API:-false}" = true ]; then
            out_dir="$RESULTS_ROOT/labbench_${task}_${model}_librarian_api"
        else
            out_dir="$RESULTS_ROOT/labbench_${task}_${model}_librarian"
        fi
        cmd=(
            "$LABBENCH_PYTHON" -m evals.Literature.LabBench.run_labbench_eval
            --task "$task" --mode knowledge
            --model "$model"
            --max-workers "$MAX_WORKERS" --resume
            --out-dir "$out_dir"
        )
        # Under --via-api retrieval runs server-side with the model it was
        # deployed with, and the runner rejects these two flags there.
        if [ "${VIA_API:-false}" = true ]; then
            cmd+=(--via-api)
        else
            cmd+=(--agent-model "$LIBRARIAN_MODEL" --agent-base-url "$LIBRARIAN_URL")
        fi
        [ -n "${LIMIT:-}" ] && cmd+=(--max-examples "$LIMIT")
        echo "RUN   (cd $REPO_ROOT && ${cmd[*]})"
        [ "$DRY_RUN" = true ] && continue
        mkdir -p "$out_dir"
        # -m needs the repo on sys.path, hence the cd.
        ( cd "$REPO_ROOT" && "${cmd[@]}" )
        ran_any=true
    done
done

# One audit pass over every result directory, as the paper table was aggregated.
audit=("$LABBENCH_PYTHON" -m evals.Literature.LabBench.audit_results "$RESULTS_ROOT")
echo "RUN   (cd $REPO_ROOT && ${audit[*]})"
if [ "$DRY_RUN" != true ] && [ "$ran_any" = true ]; then
    ( cd "$REPO_ROOT" && "${audit[@]}" )
fi
