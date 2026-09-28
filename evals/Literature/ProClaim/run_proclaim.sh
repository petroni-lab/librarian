#!/usr/bin/env bash
# run_proclaim.sh — ProClaim, +Librarian rows: Verifier Agent 0.66, ProClaim 0.80
# (AGR over 419 claims: 101 SIGNOR + 318 ConnectomeDB).
#
# Two arms, both run by default; --only verifier | --only proclaim picks one.
#   verifier  one-prompt Verifier Agent, librarian retrieval, Sonnet 4.6 verdict
#   proclaim  full ProClaim pipeline with the librarian retrieval backend
# Both write latex_table.tex under their out-dir.
#
# The proclaim arm passes no model flags: configs/librarian_sonnet.yaml is the
# single source of truth (librarian glm-5-fp8, evidence programmer
# anthropic/claude-sonnet-4-6 via LiteLLM, evidence subagent qwen3.5-9b on a
# third endpoint, sufficiency_backend: llm, claim_only, no web search).
# PROCLAIM_CONFIG_REL selects another of configs/*.yaml.
#
# That third endpoint, the evidence subagent, is the one people forget. With
# `[proclaim] apptainer_image` set in literature_eval.toml this script starts it
# and stops it on exit; one already answering at PROCLAIM_SUBAGENT_URL is reused
# and left running. Without an image it prints the `vllm serve` line and stops.
# --only verifier needs none of it.
#
# Baseline rows swap the retriever on the verifier arm: --pubmed-s2 or
# --web-search in place of --librarian-agent.
#
# Env: LIBRARIAN_URL (required unless VIA_API), LIBRARIAN_MODEL,
#      PROCLAIM_VERDICT_MODEL, PROCLAIM_VERDICT_URL, PROCLAIM_SUBAGENT_URL,
#      PROCLAIM_SUBAGENT_MODEL, PROCLAIM_LIBRARIAN_URL, PROCLAIM_LIBRARIAN_MODEL
#      (these four default to what the YAML config pins), PROCLAIM_CONFIG_REL,
#      PROCLAIM_APPTAINER_IMAGE, PROCLAIM_SRC, PROCLAIM_SUBAGENT_HF_MODEL,
#      PROCLAIM_SUBAGENT_WAIT_SECONDS, RESULTS_ROOT, LIMIT, ONLY, VIA_API,
#      DRY_RUN. ANTHROPIC_API_KEY must be set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Arm 2 runs ProClaim itself, cloned beside this script. PROCLAIM_SRC points it
# at a different checkout. Arm 1 needs none of this.
PROCLAIM_SRC="${PROCLAIM_SRC:-$SCRIPT_DIR/ProClaim_src}"
# Ours, in this directory: the clone knows nothing about a librarian backend.
CONFIG_REL="${PROCLAIM_CONFIG_REL:-configs/librarian_sonnet.yaml}"

# shellcheck source=evals/Literature/bench_env.sh
. "$SCRIPT_DIR/../bench_env.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LIBRARIAN_URL="${LIBRARIAN_URL:-}"
LIBRARIAN_MODEL="${LIBRARIAN_MODEL:-glm-5-fp8}"
PROCLAIM_VERDICT_MODEL="${PROCLAIM_VERDICT_MODEL:-claude-sonnet-4-6}"
PROCLAIM_VERDICT_URL="${PROCLAIM_VERDICT_URL:-https://api.anthropic.com/v1}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/../results}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-false}"

# Read the endpoints the config pins, so the preflight checks what the run will
# use. Empty when the config is absent, as it is for a verifier-only run; every
# caller below has its own default.
yaml_value() {
    [ -f "$SCRIPT_DIR/$CONFIG_REL" ] || return 0
    sed -nE "s/^[[:space:]]*$1:[[:space:]]*([^[:space:]#]+).*/\1/p" "$SCRIPT_DIR/$CONFIG_REL" | head -1
}
# One rule for every endpoint this bench uses: the YAML config is the default,
# the environment overrides it, and run_proclaim.sh passes the result down so
# the pipeline reads the same value it preflights.
PROCLAIM_SUBAGENT_URL="${PROCLAIM_SUBAGENT_URL:-$(yaml_value subagent_base_url)}"
PROCLAIM_SUBAGENT_URL="${PROCLAIM_SUBAGENT_URL:-http://localhost:9900/v1}"
PROCLAIM_SUBAGENT_MODEL="${PROCLAIM_SUBAGENT_MODEL:-$(yaml_value subagent_model)}"
PROCLAIM_SUBAGENT_MODEL="${PROCLAIM_SUBAGENT_MODEL:-qwen3.5-9b}"
# LIBRARIAN_URL is the harness-wide variable, and it wins over the config so
# that both arms retrieve from one endpoint.
PROCLAIM_LIBRARIAN_URL="${PROCLAIM_LIBRARIAN_URL:-${LIBRARIAN_URL:-$(yaml_value librarian_llm_base_url)}}"
PROCLAIM_LIBRARIAN_MODEL="${PROCLAIM_LIBRARIAN_MODEL:-${LIBRARIAN_MODEL:-$(yaml_value librarian_llm_model)}}"

while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="${ONLY:+$ONLY,}$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) sed -n '1,31p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Every runner here calls load_dotenv(), so a key in .env counts as set.
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
# The key is required only because the default verdict model is Sonnet. Point
# PROCLAIM_VERDICT_URL elsewhere and neither it nor its preflight applies.
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

# Presence is not validity: a rejected key surfaces per-claim as a verdict that
# will not parse, scoring every claim UNCERTAIN at AGR 0.000, and only after the
# subagent has spent ~15 min loading. One request here costs a second.
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
        # No /models route is not "not serving": the orchestrator has none but
        # answers chat completions. Same fallback literature_eval.sh uses.
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

# ── Evidence subagent ────────────────────────────────────────────────────────
# The third endpoint arm 2 needs, beyond the librarian and the verdict model.
# Qwen3.5-9B measured 23.6 GB of VRAM under the settings below, so a >=24 GB
# card is the floor.
subagent_served() {
    curl -sf --max-time 10 "${PROCLAIM_SUBAGENT_URL%/}/models" 2>/dev/null \
        | grep -q "\"id\":[[:space:]]*\"$PROCLAIM_SUBAGENT_MODEL\""
}

# The HF repo behind PROCLAIM_SUBAGENT_MODEL, and how long to wait for it.
SUBAGENT_HF_MODEL="${PROCLAIM_SUBAGENT_HF_MODEL:-Qwen/Qwen3.5-9B}"
SUBAGENT_WAIT_SECONDS="${PROCLAIM_SUBAGENT_WAIT_SECONDS:-1200}"

# The process group of a subagent this script started, so the EXIT trap can
# take it down. Empty when we are reusing someone else's.
SUBAGENT_PGID=""
stop_subagent() {
    [ -n "$SUBAGENT_PGID" ] || return 0
    echo "Stopping the evidence subagent (process group $SUBAGENT_PGID)"
    kill -- "-$SUBAGENT_PGID" 2>/dev/null || true
    SUBAGENT_PGID=""
}
trap stop_subagent EXIT

start_subagent() {
    local port deadline
    port="$(sed -nE 's#.*:([0-9]+).*#\1#p' <<<"$PROCLAIM_SUBAGENT_URL")"
    [[ "$port" =~ ^[0-9]+$ ]] \
        || { echo "ERROR: no port in PROCLAIM_SUBAGENT_URL='$PROCLAIM_SUBAGENT_URL'." >&2; exit 1; }

    echo "Starting $PROCLAIM_SUBAGENT_MODEL via apptainer ($PROCLAIM_APPTAINER_IMAGE)"
    mkdir -p "${HF_HOME:-$HOME/.cache/huggingface}"
    # PYTHONNOUSERSITE is not optional: --cleanenv does not stop Python reading
    # ~/.local, where a user-site transformers shadows the container's.
    setsid apptainer exec --nv --cleanenv \
        --bind "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface:rw" \
        --env HF_HOME=/root/.cache/huggingface \
        --env HF_HUB_CACHE=/root/.cache/huggingface/hub \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env PYTHONNOUSERSITE=1 \
        "$PROCLAIM_APPTAINER_IMAGE" \
        bash -lc "vllm serve $SUBAGENT_HF_MODEL \
            --served-model-name $PROCLAIM_SUBAGENT_MODEL --port $port \
            --gpu-memory-utilization 0.55 --max-model-len 32768" &
    # setsid gives it its own process group, so stop_subagent takes the whole
    # tree. Recorded only here, so a subagent we are reusing is never torn down.
    SUBAGENT_PGID=$!

    echo "Waiting up to ${SUBAGENT_WAIT_SECONDS}s for $PROCLAIM_SUBAGENT_MODEL on port $port ..."
    deadline=$(( SECONDS + SUBAGENT_WAIT_SECONDS ))
    until subagent_served; do
        # A dead launcher means an env/GPU error already printed above; do not
        # sit out the full timeout waiting for a server that will never appear.
        kill -0 "$SUBAGENT_PGID" 2>/dev/null \
            || { echo "ERROR: vLLM exited before serving $PROCLAIM_SUBAGENT_MODEL (see output above)." >&2; exit 1; }
        [ "$SECONDS" -lt "$deadline" ] \
            || { echo "ERROR: the evidence subagent did not come up in ${SUBAGENT_WAIT_SECONDS}s." >&2
                 echo "       Raise PROCLAIM_SUBAGENT_WAIT_SECONDS if it was still loading." >&2
                 exit 1; }
        sleep 10
    done
    echo "Evidence subagent ready at $PROCLAIM_SUBAGENT_URL"
}

# Reuse a running subagent, else start one from the configured image, else say
# what to do. Only `--only proclaim` needs it; `--only verifier` never calls this.
require_subagent() {
    subagent_served && { echo "Reusing the evidence subagent at $PROCLAIM_SUBAGENT_URL"; return 0; }
    if [ -n "${PROCLAIM_APPTAINER_IMAGE:-}" ] && command -v apptainer >/dev/null 2>&1; then
        start_subagent
        return 0
    fi
    local port
    port="$(sed -nE 's#.*:([0-9]+).*#\1#p' <<<"$PROCLAIM_SUBAGENT_URL")"; port="${port:-9900}"
    {
        echo "ERROR: the ProClaim evidence subagent is not serving '$PROCLAIM_SUBAGENT_MODEL'"
        echo "       at $PROCLAIM_SUBAGENT_URL, and it cannot be started for you:"
        [ -n "${PROCLAIM_APPTAINER_IMAGE:-}" ] \
            && echo "       apptainer is not installed." \
            || echo "       no \`[proclaim] apptainer_image\` is set in literature_eval.toml."
        echo
        echo "       Set one (e.g. docker://vllm/vllm-openai:v0.29.0) and this"
        echo "       script starts and stops the subagent itself. Or start it"
        echo "       yourself and re-run:"
        echo
        echo "  vllm serve $SUBAGENT_HF_MODEL --served-model-name $PROCLAIM_SUBAGENT_MODEL \\"
        echo "      --port $port --gpu-memory-utilization 0.55 --max-model-len 32768"
        echo
        echo "       Already have one elsewhere? Point PROCLAIM_SUBAGENT_URL at it."
    } >&2
    exit 1
}

# One environment per arm. Only the arm being run is built, so --only verifier
# never pays for ProClaim's Linux-and-CUDA stack.
PYTHON_BIN="$(bench_python_maybe proclaim "$DRY_RUN")"
if want proclaim; then
    PIPELINE_PYTHON="$(bench_python_maybe proclaim-pipeline "$DRY_RUN")"
fi

if want proclaim && [ ! -f "$SCRIPT_DIR/$CONFIG_REL" ]; then
    echo "ERROR: no librarian config at $SCRIPT_DIR/$CONFIG_REL." >&2
    echo "       PROCLAIM_CONFIG_REL selects one of configs/*.yaml." >&2
    exit 1
fi

if want proclaim && [ "$DRY_RUN" != true ]; then
    # Fail here, not inside a per-claim subprocess. Importing our entry point
    # exercises the seam onto the clone; see check_seam.py.
    if ! PYTHONPATH="$PROCLAIM_SRC/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PIPELINE_PYTHON" -c '
import proclaim.verification.evidence_programming_direct
from evals.Literature.ProClaim.backend import one_shot, config
config._self_check()
' 2>&1; then
        echo "ERROR: the ProClaim + librarian integration is not importable." >&2
        echo "       If proclaim itself is missing, the clone is not set up:" >&2
        echo "         ./evals/Literature/setup.sh --bench proclaim" >&2
        exit 1
    fi
    require_subagent
    # Three endpoints, and the two the yaml pins are easy to miss.
    preflight_endpoint "$PROCLAIM_LIBRARIAN_URL" "$PROCLAIM_LIBRARIAN_MODEL" "proclaim librarian"
    preflight_endpoint "$PROCLAIM_SUBAGENT_URL" "$PROCLAIM_SUBAGENT_MODEL" "proclaim evidence subagent"
fi

# ── Arm 1: one-prompt Verifier Agent (0.66) ──────────────────────────────────
if want verifier; then
    # --resume and --cache key off this directory, so the transport is in the
    # path: the two must not reuse each other's verdicts.
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
    # No VIA_API_SUFFIX: this arm is driven by a YAML config and never receives
    # --via-api, so it retrieves in-process whatever the transport is.
    out_dir="$RESULTS_ROOT/proclaim_librarian_sonnet46"
    cmd=(
        "$PIPELINE_PYTHON" "$SCRIPT_DIR/proclaim_librarian.py"
        --proclaim-path "$SCRIPT_DIR"
        --subset all
        --config "$CONFIG_REL"
        --out-dir "$out_dir"
        --resume --keep-going
        --librarian-base-url "$PROCLAIM_LIBRARIAN_URL"
        --librarian-model "$PROCLAIM_LIBRARIAN_MODEL"
        --subagent-base-url "$PROCLAIM_SUBAGENT_URL"
        --subagent-model "$PROCLAIM_SUBAGENT_MODEL"
    )
    [ -n "${LIMIT:-}" ] && cmd+=(--max-examples "$LIMIT")
    echo "RUN   ${cmd[*]}"
    if [ "$DRY_RUN" != true ]; then
        mkdir -p "$out_dir"
        "${cmd[@]}"
    fi
fi
