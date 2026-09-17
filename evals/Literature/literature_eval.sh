#!/usr/bin/env bash
# literature_eval.sh — one entry point for the four literature benchmarks of the
# paper, reproducing the **+Librarian row** of each table.
#
#   ./evals/Literature/literature_eval.sh --bench <litqa2|labbench|proclaim|sqa|all> [opts]
#
# --bench takes a comma-separated list: --bench litqa2,labbench,proclaim runs
# those three in order. `all` is the whole suite.
#
# Endpoint-first: you pass the base URL of an already-running OpenAI-compatible
# endpoint serving the librarian model. Nothing here starts a model server for
# you, and nothing here needs a scheduler — it all runs from a plain shell.
#
# Expected numbers (paper, +Librarian row):
#   litqa2    Cov 95.6 / Prec 82.6 / Acc 78.9
#   labbench  SeqQA 63.8, ProtocolQA 73.1, DbQA 36.2, Cloning 48.5
#   proclaim  Verifier 0.66, ProClaim 0.80
#   sqa       Citation F1 (Bio, Neu) + Citation F1 & LLM (Multi)
#
# CAVEAT — those are GLM-5 numbers, produced with the retrieval knobs now in
# librarian/config.toml. These scripts pass NO knob overrides, so a rerun
# measures the agent as currently configured. Expect close, not identical; a
# different librarian model makes them incomparable rather than wrong. A delta
# is not automatically a bug — report it rather than tuning the scripts to match.
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
#   --via-api           opt into routing every question through a running
#                       orchestrator.py instead (env: LIBRARIAN_API_URL, default
#                       http://localhost:8080). Start it from the repo root with
#                       `uv run uvicorn orchestrator:app --port 8080`. The
#                       baseline and per-run-knob arms have no API equivalent,
#                       and this stands down on its own when it sees one.
#   --dry-run           print the commands and exit
#   -h                  this header
#
# WHAT EACH BENCH NEEDS
# Every bench needs the librarian, which is an agent, not a model: it plans
# Europe PMC sub-queries and judges retrieved paragraphs, so --librarian-url
# must serve --librarian-model. There is no offline fallback. Everything below
# is *in addition* to that.
#
#   bench             local GPU      other endpoints        API keys
#   litqa2            none           —                      OPENAI_API_KEY
#   labbench          none           —                      OPENAI_API_KEY
#   proclaim          1 x >=24 GB    evidence subagent      ANTHROPIC_API_KEY
#   sqa --only bio    1 x >=8 GB     —                      none
#   sqa --only neu    1 x >=8 GB     —                      none
#   sqa --only multi  4 x H100       2 x Prometheus judge   none
#
# Those GPUs are for *scoring*, not retrieval, so pick the tier you need rather
# than the largest: litqa2 and labbench need no GPU at all and are the two to
# start with. One >=24 GB card covers both ProClaim's evidence subagent and
# SQA's AutoAIS scorer, which are separate jobs rather than concurrent ones.
# Only `--only multi` needs four, for the two Prometheus 8x7B judges at
# tensor-parallel size 4 — and it is a 29-question subset, so skipping it still
# leaves Citation F1 on the full 1451- and 1308-question bio and neuro sets.
# `--bench sqa` does exactly that on its own when the judges cannot run here:
# it skips multi with a note and keeps the two rows that did.
#
# Multi-GPU note: those judges need NVLink. On cards without it NCCL falls back
# to PCIe peer-to-peer and they die in initialize_model_parallel with
# "unhandled system error".
#
# Cheaper split: generation is pure network, so run `--skip-citation-eval` on a
# CPU box and re-run on the GPU one for scoring; --resume does the rest.
#
# Baselines are not scripted; each bench's script header gives the exact one-flag
# change that produces its baseline row.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export LITERATURE_DIR="$SCRIPT_DIR"

# An interpreter has to be resolved before the config can be read, because the
# config is TOML and load_config.py is what reads it. `uv sync` creates .venv
# without putting it on PATH, and a bare `python` does not exist on many
# systems, so prefer the one the install just made.
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    BOOT_PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    BOOT_PYTHON=python3
else
    BOOT_PYTHON=python
fi
command -v "$BOOT_PYTHON" >/dev/null 2>&1 || [ -x "$BOOT_PYTHON" ] || {
    echo "ERROR: no Python interpreter found. Run \`uv sync --extra evals\` at the" >&2
    echo "       repository root, or put one on PATH." >&2; exit 1; }

# Every user-specific endpoint, cache and scratch path lives in one TOML file so
# nobody has to edit the per-bench runners. load_config.py maps it onto the flat
# environment variables the runners read, leaving any variable that is already
# set alone -- so a real environment variable, and the flags below, still win.
# LITERATURE_EVAL_CONFIG points at a personal profile kept outside the repo.
if [ -n "${LITERATURE_EVAL_CONFIG:-}" ]; then
    [ -f "$LITERATURE_EVAL_CONFIG" ] || {
        echo "ERROR: LITERATURE_EVAL_CONFIG not found: $LITERATURE_EVAL_CONFIG" >&2; exit 1; }
    export LITERATURE_EVAL_CONFIG
fi
eval "$("$BOOT_PYTHON" "$SCRIPT_DIR/load_config.py")"
# paths.python in the config, if set, is what the runners use; otherwise the
# interpreter resolved above is already the right answer.
: "${PYTHON:=$BOOT_PYTHON}"
export PYTHON

BENCH=""
LIBRARIAN_URL_SET=false
LIBRARIAN_MODEL_SET=false
# In-process is the default: the agents are built in the eval process and talk
# to $LIBRARIAN_URL directly, so a clone needs nothing but that endpoint.
# --via-api opts into routing through a running orchestrator.py instead.
VIA_API="${VIA_API:-false}"
PASSTHRU=()

# Paper-exact defaults; every one overridable by env or flag.
export LIBRARIAN_MODEL="${LIBRARIAN_MODEL:-glm-5-fp8}"
export ANSWER_MODEL="${ANSWER_MODEL:-gpt-5.4}"
export ANSWER_URL="${ANSWER_URL:-https://api.openai.com/v1}"
export LIBRARIAN_URL="${LIBRARIAN_URL:-}"
export RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results_paper}"
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
        -h|--help) sed -n '1,82p' "$0"; exit 0 ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

# --via-api runs every question inside the orchestrator, which reaches vLLM
# in-cluster. Such a run never opens LIBRARIAN_URL, so requiring it -- or
# preflighting it below -- would abort a run that does not need it. Exported
# rather than passed through: every bench runner reads $VIA_API.
export VIA_API="${VIA_API:-false}"

# The documented baseline rows (--no-librarian, --bm25-retrieval) and every
# per-run knob only exist in-process: a server runs the configuration it was
# started with and cannot honour them. Rather than fail on a combination the
# header tells people to use, --via-api stands down as soon as one appears.
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

# --bench takes a comma-separated list so a subset of the suite is one command
# rather than a shell loop around this script: --bench litqa2,labbench,proclaim.
[ -n "$BENCH" ] || { echo "ERROR: --bench is required (litqa2|labbench|proclaim|sqa|all, comma-separated, or all)." >&2; exit 1; }
BENCHES=()
if [ "$BENCH" = all ]; then
    BENCHES=(litqa2 labbench proclaim sqa)
else
    while IFS= read -r b; do
        [ -n "$b" ] || continue
        case "$b" in
            litqa2|labbench|proclaim|sqa) BENCHES+=("$b") ;;
            *) echo "ERROR: unknown --bench '$b'." >&2; exit 1 ;;
        esac
    done <<< "${BENCH//,/$'\n'}"
fi
export LIBRARIAN_URL LIBRARIAN_MODEL ANSWER_MODEL ANSWER_URL RESULTS_ROOT ONLY DRY_RUN

# ── Preflight: assert the endpoint actually serves $LIBRARIAN_MODEL ───────────
# The commonest failure mode is a wrong or stale model on the port. Catching it
# here turns a silent 40-minute wrong-numbers run into an instant error.
list_served_models() {
    curl -sf --max-time 10 "${1%/}/models" \
        | python3 -c 'import json,sys; [print(r["id"]) for r in (json.load(sys.stdin).get("data") or []) if r.get("id")]'
}

# A one-token completion. The orchestrator is a valid OpenAI-compatible endpoint
# but mounts no /models route (404), so listing is not a usable liveness test for
# it — this asks the thing we actually care about: does `$want` answer here.
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

# --librarian-url/--librarian-model arrive after the config has been read, so a
# value that INHERITED from the librarian is still pointing at the file's. Left
# alone, `--librarian-model X` silently synthesises with the config's model
# instead -- which fails at the synthesis step, not at startup. Only values the
# config did not set explicitly are moved; LITERATURE_INHERITED names them.
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


# The upstream benchmark repositories are not committed: setup.sh clones them
# and lays our changes over the clone. Without that, litqa2 and sqa fail deep
# inside an import, a long way from the thing that is actually missing.
require_setup() {
    local want=""
    case "$1" in
        litqa2)   want="AstaBench/vendor/astabench" ;;
        sqa)      want="SQA_bench/code/scripts" ;;
        proclaim) want="ProClaim/ProClaim_src/src/proclaim" ;;
        *)        return 0 ;;
    esac
    [ -d "$SCRIPT_DIR/$want" ] && return 0
    echo "ERROR: $1 needs $want, which is not there." >&2
    echo "       Fetch just what this bench needs:" >&2
    echo "         ./evals/Literature/setup.sh --bench $1" >&2
    echo "       It clones the upstream benchmark at its pinned commit and" >&2
    echo "       applies our changes." >&2
    exit 1
}

run_bench() {
    require_setup "$1"
    case "$1" in
        litqa2)   bash "$SCRIPT_DIR/AstaBench/run_litqa2.sh"   ${PASSTHRU[@]+"${PASSTHRU[@]}"} ;;
        labbench) bash "$SCRIPT_DIR/LabBench/run_labbench.sh"  ${PASSTHRU[@]+"${PASSTHRU[@]}"} ;;
        proclaim) bash "$SCRIPT_DIR/ProClaim/run_proclaim.sh"  ${PASSTHRU[@]+"${PASSTHRU[@]}"} ;;
        sqa)      bash "$SCRIPT_DIR/SQA_bench/run_sqa.sh"      ${PASSTHRU[@]+"${PASSTHRU[@]}"} ;;
    esac
}

for b in "${BENCHES[@]}"; do
    [ "${#BENCHES[@]}" -gt 1 ] && { echo ""; echo "═══ $b ═══"; }
    run_bench "$b"
done
