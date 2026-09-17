#!/usr/bin/env bash
# run_proclaim.sh — ProClaim, +Librarian rows: Verifier Agent 0.66, ProClaim 0.80
# (AGR over 419 claims: 101 SIGNOR + 318 ConnectomeDB).
#
# Two arms, both run by default; --only verifier | --only proclaim picks one.
#   verifier  one-prompt Verifier Agent, librarian retrieval, Sonnet 4.6 verdict
#   proclaim  full ProClaim pipeline with the librarian retrieval backend
# Both write latex_table.tex under their out-dir.
#
# The `proclaim` arm passes NO model flags: librarian_sonnet_config.yaml is the
# single source of truth (librarian glm-5-fp8, evidence programmer
# anthropic/claude-sonnet-4-6 via LiteLLM, evidence subagent qwen3.5-9b on a
# THIRD endpoint, sufficiency_backend: llm, claim_only, no web search).
#
# --proclaim-path takes the benchmark dir, NOT its data/ subdir: the resolver
# probes <path>, <path>/data, <path>/datasets.
#
# Baseline rows: swap the retriever on the verifier arm — --pubmed-s2 or
# --web-search in place of --librarian-agent.
#
# NOTE: results/ is gitignored and the paper's 0.80 run directory is gone from
# disk, so there is no committed baseline to diff against — this script is the
# durable record of how the number was produced.
#
# The `proclaim` arm needs a THIRD endpoint, the evidence subagent. This script
# does not start it for you: if nothing answers at PROCLAIM_SUBAGENT_URL it
# prints the `vllm serve` line to run and stops. Point PROCLAIM_SUBAGENT_URL at
# an existing server to use that instead. `--only verifier` does not need it.
#
# Env: LIBRARIAN_URL (required), LIBRARIAN_MODEL, PROCLAIM_VERDICT_MODEL,
#      PROCLAIM_VERDICT_URL, PROCLAIM_SUBAGENT_URL, PROCLAIM_SUBAGENT_MODEL,
#      PROCLAIM_LIBRARIAN_URL, PROCLAIM_LIBRARIAN_MODEL (arm 2 retrieves
#      in-process; these default to the values pinned in the YAML config),
#      RESULTS_ROOT, LIMIT, ONLY, DRY_RUN. ANTHROPIC_API_KEY must be set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# The full ProClaim pipeline (arm 2) runs ProClaim itself, cloned to ProClaim_src
# beside this script (see its NOTICE). It needs its own environment -- `cd
# ProClaim_src && uv sync` -- because its dependencies are not the harness's.
# PROCLAIM_SRC points the arm at a different checkout. Arm 1, the Verifier
# Agent, needs none of this.
PROCLAIM_SRC="${PROCLAIM_SRC:-$SCRIPT_DIR/ProClaim_src}"
CONFIG_REL="${PROCLAIM_CONFIG_REL:-experiments/configs/librarian_sonnet_config.yaml}"

PYTHON_BIN="${PYTHON:-python}"
LIBRARIAN_URL="${LIBRARIAN_URL:-}"
LIBRARIAN_MODEL="${LIBRARIAN_MODEL:-glm-5-fp8}"
PROCLAIM_VERDICT_MODEL="${PROCLAIM_VERDICT_MODEL:-claude-sonnet-4-6}"
PROCLAIM_VERDICT_URL="${PROCLAIM_VERDICT_URL:-https://api.anthropic.com/v1}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/../results_paper}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-false}"

# The evidence subagent is the endpoint people forget; read what the yaml pins so
# the preflight checks the real thing rather than a hardcoded guess.
# Returns empty when the config is absent, which is the normal state for a
# verifier-only run: every caller below has its own default.
yaml_value() {
    [ -f "$PROCLAIM_SRC/$CONFIG_REL" ] || return 0
    sed -nE "s/^[[:space:]]*$1:[[:space:]]*([^[:space:]#]+).*/\1/p" "$PROCLAIM_SRC/$CONFIG_REL" | head -1
}
PROCLAIM_SUBAGENT_URL="${PROCLAIM_SUBAGENT_URL:-$(yaml_value subagent_base_url)}"
PROCLAIM_SUBAGENT_URL="${PROCLAIM_SUBAGENT_URL:-http://localhost:9900/v1}"
PROCLAIM_SUBAGENT_MODEL="${PROCLAIM_SUBAGENT_MODEL:-$(yaml_value subagent_model)}"
PROCLAIM_SUBAGENT_MODEL="${PROCLAIM_SUBAGENT_MODEL:-qwen3.5-9b}"
PROCLAIM_LIBRARIAN_URL="${PROCLAIM_LIBRARIAN_URL:-$(yaml_value librarian_llm_base_url)}"
PROCLAIM_LIBRARIAN_URL="${PROCLAIM_LIBRARIAN_URL:-$LIBRARIAN_URL}"
PROCLAIM_LIBRARIAN_MODEL="${PROCLAIM_LIBRARIAN_MODEL:-$(yaml_value librarian_llm_model)}"
PROCLAIM_LIBRARIAN_MODEL="${PROCLAIM_LIBRARIAN_MODEL:-$LIBRARIAN_MODEL}"

while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="${ONLY:+$ONLY,}$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) sed -n '1,33p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Keys may live in the repo .env instead of the environment — every runner here
# calls load_dotenv() on it, so accept either source.
have_key() {
    [ -n "${!1:-}" ] && return 0
    grep -qE "^[[:space:]]*(export[[:space:]]+)?$1=." "$REPO_ROOT/.env" 2>/dev/null
}

if [ "${VIA_API:-false}" = true ]; then
    VIA_API_ARGS=(--via-api)
    VIA_API_SUFFIX="_api"
else
    VIA_API_ARGS=()
    VIA_API_SUFFIX=""
    # Only the in-process path opens this endpoint.
    [ -n "$LIBRARIAN_URL" ] || { echo "ERROR: set LIBRARIAN_URL (or use the API path)." >&2; exit 1; }
fi
# The Anthropic key is required only because the DEFAULT verdict model is
# Sonnet. Point PROCLAIM_VERDICT_URL somewhere else (e.g. the orchestrator, to
# smoke-test with glm-5.3-flash) and neither the key nor its preflight applies.
case "$PROCLAIM_VERDICT_URL" in
    *api.anthropic.com*) VERDICT_IS_ANTHROPIC=true ;;
    *) VERDICT_IS_ANTHROPIC=false ;;
esac
if [ "$VERDICT_IS_ANTHROPIC" = true ]; then
    if ! have_key ANTHROPIC_API_KEY && [ "$DRY_RUN" != true ]; then
        echo "ERROR: ANTHROPIC_API_KEY is required (Sonnet 4.6 verdict + evidence programmer); set it or put it in $REPO_ROOT/.env." >&2
        exit 1
    fi
else
    echo "NOTE  verdict endpoint is $PROCLAIM_VERDICT_URL ($PROCLAIM_VERDICT_MODEL) — skipping the Anthropic key check."
fi

# Presence is not validity. A dead key passes have_key, and the 401 then surfaces
# only per-claim, as a verdict that will not parse — every claim scores UNCERTAIN
# with AGR 0.000, which reads as a bad agent rather than a bad key. Worse, it
# does so *after* the evidence subagent has spent ~15 min loading. One request
# here costs a second.
check_anthropic_key() {
    local key status
    key="${ANTHROPIC_API_KEY:-}"
    if [ -z "$key" ]; then
        key="$(grep -E "^[[:space:]]*(export[[:space:]]+)?ANTHROPIC_API_KEY=" "$REPO_ROOT/.env" 2>/dev/null \
            | tail -1 | cut -d= -f2- | tr -d "\"' \t\r\n")"
    fi
    status="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
        https://api.anthropic.com/v1/models \
        -H "x-api-key: $key" -H "anthropic-version: 2023-06-01" || true)"
    case "$status" in
        200) echo "OK    ANTHROPIC_API_KEY accepted by api.anthropic.com" ;;
        # No network is not a dead key; warn rather than block an offline dry run.
        000) echo "WARNING: could not reach api.anthropic.com to validate ANTHROPIC_API_KEY." >&2 ;;
        *)   echo "ERROR: ANTHROPIC_API_KEY rejected by api.anthropic.com (HTTP $status)." >&2
             echo "       Replace it in $REPO_ROOT/.env; a rejected key scores every claim UNCERTAIN." >&2
             exit 1 ;;
    esac
}
if [ "$DRY_RUN" != true ] && [ "$VERDICT_IS_ANTHROPIC" = true ]; then check_anthropic_key; fi

want() { [ -z "$ONLY" ] || grep -qx "$1" <<< "${ONLY//,/$'\n'}"; }

# literature_eval.sh exports this; define a standalone equivalent when it did not.
if ! declare -F preflight_endpoint >/dev/null; then
    preflight_endpoint() {
        if curl -sf --max-time 10 "${1%/}/models" | grep -q "\"id\":[[:space:]]*\"$2\""; then
            echo "OK    $3: $2 @ $1"
            return 0
        fi
        # No /models is not "not serving": the orchestrator mounts no such route
        # but answers chat completions. Same fallback literature_eval.sh uses, so
        # running this script standalone behaves the same as through it.
        if curl -sf --max-time 60 "${1%/}/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1}" \
            >/dev/null 2>&1; then
            echo "OK    $3: $2 @ $1 (no /models route; answered a chat probe)"
            return 0
        fi
        echo "ERROR: $3 preflight failed — '$2' is not served at $1" >&2
        exit 1
    }
fi

# ── Evidence subagent: yours to start ────────────────────────────────────────
# Arm 2 needs a third endpoint beyond the librarian and the verdict model: the
# evidence subagent the ProClaim pipeline programs against. Serving models is
# not this script's job, so it checks and tells you the command rather than
# guessing at your GPUs, container runtime and module stack.
#
# Qwen3.5-9B is 19.3 GB of bf16 weights and measured 23.6 GB of VRAM under the
# settings below, so a >=24 GB card is the floor.
subagent_served() {
    curl -sf --max-time 10 "${PROCLAIM_SUBAGENT_URL%/}/models" 2>/dev/null \
        | grep -q "\"id\":[[:space:]]*\"$PROCLAIM_SUBAGENT_MODEL\""
}

require_subagent() {
    subagent_served && return 0
    cat >&2 <<EOF
ERROR: the ProClaim evidence subagent is not serving '$PROCLAIM_SUBAGENT_MODEL'
       at $PROCLAIM_SUBAGENT_URL. Start it, then re-run:

  vllm serve Qwen/Qwen3.5-9B --served-model-name $PROCLAIM_SUBAGENT_MODEL \\
      --port 9900 --gpu-memory-utilization 0.55 --max-model-len 32768

       Already have one elsewhere? Point PROCLAIM_SUBAGENT_URL at it.
EOF
    exit 1
}

if want proclaim && [ ! -f "$PROCLAIM_SRC/$CONFIG_REL" ]; then
    # Only reachable if PROCLAIM_SRC was pointed somewhere else: the project is
    # vendored at the default path.
    echo "ERROR: no ProClaim config at $PROCLAIM_SRC/$CONFIG_REL." >&2
    echo "       Unset PROCLAIM_SRC to use the vendored copy." >&2
    exit 1
fi

if want proclaim && [ "$DRY_RUN" != true ]; then
    # Fail here, not inside a per-claim subprocess with an unhelpful traceback.
    if ! PYTHONPATH="$PROCLAIM_SRC/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" -c 'import proclaim.verification.evidence_programming_direct' 2>/dev/null; then
        echo "ERROR: proclaim is not importable. Install it:" >&2
        echo "         cd $PROCLAIM_SRC && uv sync" >&2
        exit 1
    fi
    require_subagent
    # Three endpoints, and the two the yaml pins are easy to miss.
    preflight_endpoint "$PROCLAIM_LIBRARIAN_URL" "$PROCLAIM_LIBRARIAN_MODEL" "proclaim librarian"
    preflight_endpoint "$PROCLAIM_SUBAGENT_URL" "$PROCLAIM_SUBAGENT_MODEL" "proclaim evidence subagent"
fi

# ── Arm 1: one-prompt Verifier Agent (0.66) ──────────────────────────────────
if want verifier; then
    # The transport must be in the path: --resume and --cache both key off this
    # directory, so sharing it between the API and in-process paths let a run
    # reuse verdicts produced against the other retriever.
    out_dir="$RESULTS_ROOT/proclaim_verifier_librarian${VIA_API_SUFFIX}"
    cmd=(
        "$PYTHON_BIN" -m evals.Literature.ProClaim.verifier
        --proclaim-path "$SCRIPT_DIR"
        --subset all --librarian-agent
        ${VIA_API_ARGS[@]+"${VIA_API_ARGS[@]}"}
        --llm-base-url "$LIBRARIAN_URL" --llm-model "$LIBRARIAN_MODEL"
        --verdict-base-url "$PROCLAIM_VERDICT_URL"
        --verdict-model "$PROCLAIM_VERDICT_MODEL"
        --verdict-api-key-env ANTHROPIC_API_KEY
        --resume --cache
        --out-dir "$out_dir"
    )
    [ -n "${LIMIT:-}" ] && cmd+=(--max-examples "$LIMIT")
    echo "RUN   (cd $REPO_ROOT && ${cmd[*]})"
    if [ "$DRY_RUN" != true ]; then
        mkdir -p "$out_dir"
        ( cd "$REPO_ROOT" && "${cmd[@]}" )   # -m needs the repo on sys.path
    fi
fi

# ── Arm 2: full ProClaim pipeline on the librarian backend (0.80) ────────────
if want proclaim; then
    # No VIA_API_SUFFIX here on purpose: this arm drives proclaim_librarian.py
    # through a YAML config and never receives --via-api, so it retrieves
    # in-process whatever the transport is. Tagging its directory "_api" would
    # name results after a path they did not take, and would split one arm's
    # resume state across two directories that ran identically.
    out_dir="$RESULTS_ROOT/proclaim_librarian_sonnet46"
    cmd=(
        "$PYTHON_BIN" "$SCRIPT_DIR/proclaim_librarian.py"
        --proclaim-path "$SCRIPT_DIR"
        --subset all
        --config "$CONFIG_REL"
        --out-dir "$out_dir"
        --resume --keep-going
        --librarian-base-url "$PROCLAIM_LIBRARIAN_URL"
        --librarian-model "$PROCLAIM_LIBRARIAN_MODEL"
    )
    [ -n "${LIMIT:-}" ] && cmd+=(--max-examples "$LIMIT")
    echo "RUN   ${cmd[*]}"
    if [ "$DRY_RUN" != true ]; then
        mkdir -p "$out_dir"
        "${cmd[@]}"
    fi
fi
