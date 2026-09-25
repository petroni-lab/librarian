#!/usr/bin/env bash
# literature_eval.sh — one entry point for the four literature benchmarks of the
# paper, reproducing the **+Librarian row** of each table.
#
#   ./evals/Literature/literature_eval.sh --bench <litqa2|labbench|proclaim|sqa|all> [opts]
#
# --bench takes a comma-separated list: --bench litqa2,labbench,proclaim runs
# those three in order. `all` is the whole suite.
#
# You pass the base URL of an already-running OpenAI-compatible endpoint serving
# the librarian model. Nothing here starts a model server, and nothing here
# needs a scheduler — it all runs from a plain shell.
#
# Expected numbers (paper, +Librarian row):
#   litqa2    Cov 95.6 / Prec 82.6 / Acc 78.9
#   labbench  SeqQA 63.8, ProtocolQA 73.1, DbQA 36.2, Cloning 48.5
#   proclaim  Verifier 0.66, ProClaim 0.80
#   sqa       Citation F1 (Bio, Neu) + Citation F1 & LLM (Multi)
#
# Those are GLM-5 numbers, produced with the retrieval knobs now in
# librarian/config.toml. These scripts pass no knob overrides, so a rerun
# measures the agent as currently configured: expect close, not identical.
#
# START HERE — a smoke run: three questions, no scoring model to download.
#
#   ./evals/Literature/literature_eval.sh --bench litqa2 --limit 3 \
#       --librarian-url http://localhost:8000/v1 --librarian-model my-model
#
# Options:
#   --bench NAMES       litqa2 | labbench | proclaim | sqa | all   (required)
#                       comma-separated for a subset: litqa2,labbench,proclaim
#   --only X            sub-target within a bench (see that bench's -h)
#   --librarian-url U   OpenAI-compatible base URL serving the librarian model
#   --librarian-model M default: glm-5-fp8
#   --answer-model M    answering LLM, default: gpt-5.4
#   --answer-url U      default: https://api.openai.com/v1
#   --results-root DIR  default: evals/Literature/results
#   --limit N           cap the number of questions (smoke runs)
#   --max-workers N     concurrency (bench-specific default)
#   --in-process        the DEFAULT; the agents are built in this process.
#   --via-api           route every question through a running orchestrator.py
#                       instead (env: LIBRARIAN_API_URL, default
#                       http://localhost:8080). Start it from the repo root with
#                       `uv run uvicorn orchestrator:app --port 8080`. A run that
#                       passes a baseline or per-run-knob flag falls back to
#                       in-process, with a note.
#   --dry-run           print the commands and exit
#   -h                  this header
#
# WHAT EACH BENCH NEEDS
# Every bench needs the librarian: --librarian-url must serve --librarian-model,
# and there is no offline fallback. Everything below is *in addition* to that.
#
#   bench             local GPU      other endpoints        API keys
#   litqa2            none           —                      OPENAI_API_KEY
#   labbench          none           —                      OPENAI_API_KEY
#   proclaim          1 x >=24 GB    evidence subagent      ANTHROPIC_API_KEY
#   sqa --only bio    1 x >=8 GB     —                      none
#   sqa --only neu    1 x >=8 GB     —                      none
#   sqa --only multi  4 x H100       2 x Prometheus judge   none
#
# Those GPUs are for *scoring*, not retrieval. ProClaim's evidence subagent and
# SQA's AutoAIS scorer are separate jobs, so one >=24 GB card covers both. Only
# `--only multi` needs four, for the two Prometheus 8x7B judges at
# tensor-parallel size 4; it is a 29-question subset, and `--bench sqa` skips it
# with a note when the judges cannot run here, keeping the Citation F1 rows on
# the full 1451- and 1308-question bio and neuro sets.
#
# Multi-GPU note: those judges need NVLink. On cards without it NCCL falls back
# to PCIe peer-to-peer and they die in initialize_model_parallel with
# "unhandled system error".
#
# Generation is pure network, so it can be split across machines: run
# `--skip-citation-eval` on a CPU box and re-run on the GPU one for scoring,
# where --resume picks up the generated answers.
#
# Baselines are not scripted; each bench's script header gives the exact one-flag
# change that produces its baseline row.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export LITERATURE_DIR="$SCRIPT_DIR"

# An interpreter to read the TOML config with, resolved before load_config.py
# runs. `uv sync` creates .venv without putting it on PATH, so prefer that one.
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    BOOT_PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    BOOT_PYTHON=python3
else
    BOOT_PYTHON=python
fi
command -v "$BOOT_PYTHON" >/dev/null 2>&1 || [ -x "$BOOT_PYTHON" ] || {
    echo "ERROR: no Python interpreter found. Run \`uv sync\` at the" >&2
    echo "       repository root, or put one on PATH." >&2; exit 1; }

# load_config.py maps literature_eval.toml onto the flat environment variables
# the runners read, leaving any variable that is already set alone, so a real
# environment variable and the flags below still win. LITERATURE_EVAL_CONFIG
# points at a personal profile kept outside the repository.
if [ -n "${LITERATURE_EVAL_CONFIG:-}" ]; then
    [ -f "$LITERATURE_EVAL_CONFIG" ] || {
        echo "ERROR: LITERATURE_EVAL_CONFIG not found: $LITERATURE_EVAL_CONFIG" >&2; exit 1; }
    export LITERATURE_EVAL_CONFIG
fi
eval "$("$BOOT_PYTHON" "$SCRIPT_DIR/load_config.py")"
# paths.python names the base interpreter each bench's locked environment is
# created from (see bench_env.sh); the runners never use it directly.
: "${BENCH_BASE_PYTHON:=$BOOT_PYTHON}"
export BENCH_BASE_PYTHON

# shellcheck source=evals/Literature/bench_env.sh
. "$SCRIPT_DIR/bench_env.sh"

BENCH=""
LIBRARIAN_URL_SET=false
LIBRARIAN_MODEL_SET=false
# In-process is the default: the agents are built here and talk to
# $LIBRARIAN_URL directly. --via-api routes through orchestrator.py instead.
VIA_API="${VIA_API:-false}"
PASSTHRU=()

# Paper-exact defaults; every one overridable by env or flag.
export LIBRARIAN_MODEL="${LIBRARIAN_MODEL:-glm-5-fp8}"
export ANSWER_MODEL="${ANSWER_MODEL:-gpt-5.4}"
export ANSWER_URL="${ANSWER_URL:-https://api.openai.com/v1}"
export LIBRARIAN_URL="${LIBRARIAN_URL:-}"
export RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
export DRY_RUN="${DRY_RUN:-false}"
export ONLY="${ONLY:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bench) BENCH="$2"; shift 2 ;;
        --only) ONLY="${ONLY:+$ONLY,}$2"; shift 2 ;;
        --librarian-url) LIBRARIAN_URL="$2"; LIBRARIAN_URL_SET=true; shift 2 ;;
        --librarian-model) LIBRARIAN_MODEL="$2"; LIBRARIAN_MODEL_SET=true; shift 2 ;;
        --answer-model) ANSWER_MODEL="$2"; shift 2 ;;
        --answer-url) ANSWER_URL="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --limit) export LIMIT="$2"; shift 2 ;;
        --max-workers) export MAX_WORKERS="$2"; shift 2 ;;
        --via-api) VIA_API=true; shift ;;
        --in-process|--no-via-api) VIA_API=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) sed -n '/^set -euo/q;p' "$0"; exit 0 ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

# A --via-api run never opens LIBRARIAN_URL itself, so the requirement and the
# preflight below are both skipped for one. Every bench runner reads $VIA_API.
export VIA_API="${VIA_API:-false}"

# The baseline rows (--no-librarian, --bm25-retrieval) and the per-run knobs
# only exist in-process, since a server runs the configuration it was started
# with. --via-api falls back to in-process as soon as one of them appears.
for arg in ${PASSTHRU[@]+"${PASSTHRU[@]}"}; do
    case "$arg" in
        --no-librarian|--bm25-retrieval|--no-librarian-full-text|--librarian-*|--agent-model|--agent-base-url)
            if [ "$VIA_API" = true ]; then
                echo "NOTE  $arg has no API equivalent — using in-process agents for this run."
                VIA_API=false
            fi
            ;;
    esac
done

# The set of benches is whatever has contributed a bench.manifest, ordered by
# each one's `order`; nothing here names them.
known_benches() {
    local m
    for m in "$SCRIPT_DIR"/*/bench.manifest; do
        [ -f "$m" ] || continue
        printf '%s\t%s\n' \
            "$(sed -n 's/^order=//p' "$m" | head -1)" \
            "$(sed -n 's/^bench=//p' "$m" | head -1)"
    done | sort -n | cut -f2
}
KNOWN="$(known_benches | tr '\n' ' ')"
[ -n "${KNOWN// }" ] || { echo "ERROR: no bench.manifest found under $SCRIPT_DIR." >&2; exit 1; }
[ -n "$BENCH" ] || { echo "ERROR: --bench is required (${KNOWN// /|}, comma-separated, or all)." >&2; exit 1; }
BENCHES=()
if [ "$BENCH" = all ]; then
    # shellcheck disable=SC2206
    BENCHES=($KNOWN)
else
    while IFS= read -r b; do
        [ -n "$b" ] || continue
        case " $KNOWN " in
            *" $b "*) BENCHES+=("$b") ;;
            *) echo "ERROR: unknown --bench '$b' (have: ${KNOWN% })." >&2; exit 1 ;;
        esac
    done <<< "${BENCH//,/$'\n'}"
fi
export LIBRARIAN_URL LIBRARIAN_MODEL ANSWER_MODEL ANSWER_URL RESULTS_ROOT ONLY DRY_RUN

# ── Preflight: assert the endpoint actually serves $LIBRARIAN_MODEL ───────────
list_served_models() {
    curl -sf --max-time 10 "${1%/}/models" \
        | python3 -c 'import json,sys; [print(r["id"]) for r in (json.load(sys.stdin).get("data") or []) if r.get("id")]'
}

# A one-token completion, for an endpoint that mounts no /models route: the
# orchestrator is OpenAI-compatible but answers 404 there.
probe_chat_model() {
    local url="$1" want="$2"
    curl -sf --max-time 60 "${1%/}/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$want\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1}" \
        >/dev/null 2>&1
}

preflight_endpoint() {
    local url="$1" want="$2" label="$3" models
    if ! models="$(list_served_models "$url" 2>/dev/null)" || [ -z "$models" ]; then
        # No /models is not "not serving": fall back to asking the model directly.
        if probe_chat_model "$url" "$want"; then
            echo "OK    $label: $want @ $url (no /models route; answered a chat probe)"
            return 0
        fi
        echo "ERROR: $label preflight failed — cannot list models at ${url%/}/models," >&2
        echo "       and '$want' did not answer a chat completion there either." >&2
        exit 1
    fi
    if ! grep -Fxq "$want" <<< "$models"; then
        echo "ERROR: $label preflight failed — '$want' is not served at $url" >&2
        echo "Served there:" >&2
        echo "$models" | sed 's/^/  /' >&2
        exit 1
    fi
    echo "OK    $label: $want @ $url"
}
export -f list_served_models probe_chat_model preflight_endpoint

# --librarian-url/--librarian-model arrive after the config has been read, so
# any value that inherited from the librarian still holds the file's. Re-point
# those, and only those; LITERATURE_INHERITED names them.
inherited() { case " ${LITERATURE_INHERITED:-} " in *" $1 "*) return 0 ;; esac; return 1; }
if [ "$LIBRARIAN_MODEL_SET" = true ]; then
    inherited SYNTHESIS_MODEL          && SYNTHESIS_MODEL="$LIBRARIAN_MODEL"
    inherited PROCLAIM_LIBRARIAN_MODEL && PROCLAIM_LIBRARIAN_MODEL="$LIBRARIAN_MODEL"
    inherited LIBRARIAN_API_MODEL      && LIBRARIAN_API_MODEL="$LIBRARIAN_MODEL"
fi
if [ "$LIBRARIAN_URL_SET" = true ]; then
    inherited PROCLAIM_LIBRARIAN_URL && PROCLAIM_LIBRARIAN_URL="$LIBRARIAN_URL"
fi

if [ -z "$LIBRARIAN_URL" ] && [ "$VIA_API" != true ]; then
    echo "ERROR: set --librarian-url (or LIBRARIAN_URL) to the endpoint serving $LIBRARIAN_MODEL." >&2
    exit 1
fi
if [ "$DRY_RUN" != true ] && [ "$VIA_API" != true ]; then
    preflight_endpoint "$LIBRARIAN_URL" "$LIBRARIAN_MODEL" "librarian"
fi


# A bench with an upstream repository needs it cloned first; setup.sh does that.
# Environments are not checked here, because a run builds its own on first use.
require_setup() {
    local bench="$1" manifest bench_dir clone
    manifest="$(bench_manifest "$bench")" || return 0
    bench_dir="$(dirname "$manifest")"
    clone="$(bench_manifest_get "$manifest" clone)"
    [ -n "$clone" ] || return 0
    [ -d "$bench_dir/$clone/.git" ] && return 0
    echo "ERROR: $bench needs $(bench_manifest_get "$manifest" name) at" >&2
    echo "       ${bench_dir#"$SCRIPT_DIR/"}/$clone, which is not there." >&2
    echo "       Fetch just what this bench needs:" >&2
    echo "         ./evals/Literature/setup.sh --bench $bench" >&2
    echo "       It clones the upstream benchmark at its pinned commit." >&2
    exit 1
}

run_bench() {
    local bench="$1" manifest bench_dir runner
    require_setup "$bench"
    manifest="$(bench_manifest "$bench")" || { echo "ERROR: no manifest for $bench" >&2; exit 1; }
    bench_dir="$(dirname "$manifest")"
    runner="$(bench_manifest_get "$manifest" run)"
    [ -n "$runner" ] || { echo "ERROR: $manifest declares no run= script" >&2; exit 1; }
    bash "$bench_dir/$runner" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
}

for b in "${BENCHES[@]}"; do
    [ "${#BENCHES[@]}" -gt 1 ] && { echo ""; echo "═══ $b ═══"; }
    run_bench "$b"
done
