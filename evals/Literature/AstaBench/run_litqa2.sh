#!/usr/bin/env bash
# run_litqa2.sh — LitQA2 open-judge, +Librarian row: Cov 95.6 / Prec 82.6 / Acc 78.9.
#
# Normally invoked through ../literature_eval.sh --bench litqa2, which does the
# endpoint preflight; it also runs standalone with LIBRARIAN_URL set.
#
# 91 questions, `europepmc_fulltext` split, thinking OFF, judge gpt-4o-2024-11-20.
# The full split is the default: --limit is what shrinks a run, and run_evals.py's
# --max-samples is Inspect's *concurrency*, not a question count.
# No librarian knob overrides: librarian/config.toml is the single source of truth (7 / 50
# / 16 / 128 as checked in today; the paper ran 7 sub-queries, pre-111039b).
#
# Baseline rows are the same command with one flag changed:
#   --solver llm_only          Parametric
#   --solver llm_web_search    Web Search
#   --solver openscholar_api   OpenScholar-8B
#
# Env: LIBRARIAN_URL (required), LIBRARIAN_MODEL, ANSWER_MODEL, ANSWER_URL,
#      LITQA2_JUDGE_MODEL, RESULTS_ROOT, LIMIT, MAX_CONNECTIONS, DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
LIBRARIAN_URL="${LIBRARIAN_URL:-}"
LIBRARIAN_MODEL="${LIBRARIAN_MODEL:-glm-5-fp8}"
# run_evals.py names the results directory and the .eval file after --llm-model.
# On the API path that flag is inert -- retrieval runs on the pods and the answer
# model comes from LITQA2_ANSWER_LLM_MODEL -- so passing LIBRARIAN_MODEL there
# filed a glm-5.3-flash run under "glm-5-fp8". Label it with what actually ran.
if [ "${VIA_API:-false}" = true ]; then
    LIBRARIAN_MODEL="${LIBRARIAN_API_MODEL:-$LIBRARIAN_MODEL}"
fi
ANSWER_MODEL="${ANSWER_MODEL:-gpt-5.4}"
ANSWER_URL="${ANSWER_URL:-https://api.openai.com/v1}"
LITQA2_JUDGE_MODEL="${LITQA2_JUDGE_MODEL:-openai/gpt-4o-2024-11-20}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/../results_paper}"
# Inspect concurrency knobs — they change wall-clock, never the scores.
MAX_SAMPLES="${MAX_SAMPLES:-10}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-10}"
DRY_RUN="${DRY_RUN:-false}"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        -h|--help) sed -n '1,19p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Keys may live in the repo .env instead of the environment — every runner here
# calls load_dotenv() on it, so accept either source.
read_key() {
    if [ -n "${!1:-}" ]; then printf '%s' "${!1}"; return 0; fi
    sed -nE "s/^[[:space:]]*(export[[:space:]]+)?$1=[\"']?([^\"'#[:space:]]+).*/\2/p" \
        "$REPO_ROOT/.env" 2>/dev/null | head -1
}

if [ "${VIA_API:-false}" = true ]; then
    VIA_API_ENV=(LITERATURE_VIA_API=true)
else
    VIA_API_ENV=()
    # Only the in-process path opens this endpoint.
    [ -n "$LIBRARIAN_URL" ] || { echo "ERROR: set LIBRARIAN_URL (or use the API path)." >&2; exit 1; }
fi

# Judge routing trap. Pointing inspect_ai's openai provider at a LOCAL endpoint
# is a normal thing to do while developing, and OPENAI_BASE_URL is how it is
# done. The paper's judge is the real
# gpt-4o-2024-11-20, so inheriting that export from the shell would self-judge and
# silently change the numbers. Refuse rather than produce wrong numbers.
case "${OPENAI_BASE_URL:-https://api.openai.com/v1}" in
    https://api.openai.com/v1*) ;;
    *) echo "ERROR: OPENAI_BASE_URL='$OPENAI_BASE_URL' routes the judge away from OpenAI." >&2
       echo "       The paper judge is $LITQA2_JUDGE_MODEL on api.openai.com. Unset it." >&2
       exit 1 ;;
esac
OPENAI_KEY="$(read_key OPENAI_API_KEY)"
if [ -z "$OPENAI_KEY" ] && [ "$DRY_RUN" != true ]; then
    echo "ERROR: OPENAI_API_KEY is required (judge + answering model); set it or put it in $REPO_ROOT/.env." >&2
    exit 1
fi
export OPENAI_API_KEY="$OPENAI_KEY"

# librarian/llm_client.py reads LLM_API_KEY — NOT OPENAI_API_KEY — so an OpenAI
# answering model otherwise authenticates with the literal "EMPTY" and 401s on
# every sample. The librarian's own vLLM endpoint ignores the key.
case "$ANSWER_URL" in
    *api.openai.com*) export LLM_API_KEY="${LLM_API_KEY:-$OPENAI_KEY}" ;;
esac

# The task reads this file; build it once if absent (91-question split).
SPLIT_FILE="$SCRIPT_DIR/data/litqa2_europepmc_fulltext/litqa2_full_europepmc_fulltext.json"
if [ ! -f "$SPLIT_FILE" ] && [ "$DRY_RUN" != true ]; then
    echo "Building the europepmc_fulltext split (one-off, queries Europe PMC) ..."
    # --output-dir is explicit: the script's own default still points at the
    # pre-reorg evals/AstaBench/data path, which no longer exists.
    "$PYTHON_BIN" "$SCRIPT_DIR/check_litqa2_europepmc_fulltext.py" \
        --output-dir "$(dirname "$SPLIT_FILE")"
fi

cmd=(
    LITERATURE_USE_LIBRARIAN_AGENT=true
    ${VIA_API_ENV[@]+"${VIA_API_ENV[@]}"}
    ${LIBRARIAN_API_URL:+LIBRARIAN_API_URL="$LIBRARIAN_API_URL"}
    LITQA2_ANSWER_LLM_BASE_URL="$ANSWER_URL"
    LITQA2_ANSWER_LLM_MODEL="$ANSWER_MODEL"
    "$PYTHON_BIN" "$SCRIPT_DIR/run_evals.py"
    --split validation --config custom_tooling --solver bio_agent
    --task LitQA2-FullText-OpenJudge-EuropePMCFullText
    --llm-base-url "$LIBRARIAN_URL" --llm-model "$LIBRARIAN_MODEL"
    --litqa2-open-judge-model "$LITQA2_JUDGE_MODEL"
    --max-samples "$MAX_SAMPLES" --max-connections "$MAX_CONNECTIONS"
    --results-dir "$RESULTS_ROOT/litqa2_librarian"
)
# No --limit means the whole 91-question split, which is the paper run.
[ -n "${LIMIT:-}" ] && cmd+=(--limit "$LIMIT")
echo "RUN   ${cmd[*]}"
[ "$DRY_RUN" = true ] && exit 0
mkdir -p "$RESULTS_ROOT/litqa2_librarian"
"${cmd[@]}"
