#!/usr/bin/env bash
set -euo pipefail

# End-to-end evaluation pipeline for ScholarQA-Multi on Kubernetes
# Runs Inference -> Formatting -> Citation Evaluation -> Prometheus LLM Judge Evaluation
# Usage: ./run_multi_evals_k8s.sh [--judge_model prometheus-eval/prometheus-bgb-8x7b-v2.0] [--judge_relevance_model prometheus-eval/prometheus-8x7b-v2.0] [--judge_pass all|org_cov|relevance] [--skip-citation] [--load_vllm] [--pred_file /abs/path/to/pred.json] [--data-file /abs/path/to/questions.json] [extra run_scholarqa args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVALS_DIR="$REPO_ROOT/evals/Literature/SQA_bench"
DATA_DIR="$EVALS_DIR/data/scholarqa_multi"
OUTPUT_DIR="${SCHOLARQA_OUTPUT_DIR:-$EVALS_DIR/output/osds_bm25/scholarqa_multi}"
SCHOLARQA_SCRIPTS="$EVALS_DIR/scorers"

# Inherited from run_sqa.sh when launched through it; resolved here for a
# standalone invocation. See ../bench_env.sh.
# shellcheck source=evals/Literature/bench_env.sh
. "$EVALS_DIR/../bench_env.sh"
PYTHON_BIN="${PYTHON_BIN:-$(bench_python sqa)}"
# Resolved only if a scoring step is actually reached: the scoring environment
# is several GB, and a run that skips citation eval must not build it. Safe
# under `set -u`, which a bare ${SQA_SCORING_PYTHON} reference would not be.
scoring_python() {
    if [ -z "${SQA_SCORING_PYTHON:-}" ]; then
        SQA_SCORING_PYTHON="$(bench_python sqa-scoring)"
        export SQA_SCORING_PYTHON
    fi
    printf '%s' "$SQA_SCORING_PYTHON"
}

TEST_CONFIG="$DATA_DIR/human_answers.json"
DATA_FILE_OVERRIDE=""
RUBRIC_INPUT_DIR="$OUTPUT_DIR/rubric_input"
JUDGE_EVAL_DIR="$OUTPUT_DIR/judge_eval"
STATE_FILE_PREFIX="$OUTPUT_DIR/.run_multi_evals_k8s.step"
PRED_POINTER_PREFIX="$OUTPUT_DIR/.run_multi_evals_k8s.pred"

# Follow the upstream ScholarQABench Prometheus recipe:
# - organization + coverage use prometheus-bgb-8x7b-v2.0
# - relevance uses prometheus-8x7b-v2.0
JUDGE_MODEL="prometheus-eval/prometheus-bgb-8x7b-v2.0"
JUDGE_RELEVANCE_MODEL="prometheus-eval/prometheus-8x7b-v2.0"
JUDGE_GPU_MEMORY_UTILIZATION=""
JUDGE_BASE_URL=""
JUDGE_RELEVANCE_BASE_URL=""
JUDGE_LITELLM_MODEL=""
JUDGE_RELEVANCE_LITELLM_MODEL=""
JUDGE_INSTRUCTION="Answer the question related to the most recent scientific literature."
JUDGE_TOP_N="10"
JUDGE_MAX_NEW_TOKENS="512"
JUDGE_PASS="all"
FORCED_PRED_FILE=""
SKIP_CITATION=false

# Parse arguments to grab inference args and judge model if passed
INFERENCE_ARGS=()
USE_VLLM_FLAG=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --judge_model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
        --judge_relevance_model)
            JUDGE_RELEVANCE_MODEL="$2"
            shift 2
            ;;
        --judge_gpu_memory_utilization)
            JUDGE_GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --judge_base_url)
            JUDGE_BASE_URL="$2"
            shift 2
            ;;
        --judge_relevance_base_url)
            JUDGE_RELEVANCE_BASE_URL="$2"
            shift 2
            ;;
        --judge_litellm_model)
            JUDGE_LITELLM_MODEL="$2"
            shift 2
            ;;
        --judge_relevance_litellm_model)
            JUDGE_RELEVANCE_LITELLM_MODEL="$2"
            shift 2
            ;;
        --judge_instruction)
            JUDGE_INSTRUCTION="$2"
            shift 2
            ;;
        --judge_top_n)
            JUDGE_TOP_N="$2"
            shift 2
            ;;
        --judge_max_new_tokens)
            JUDGE_MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --judge_pass)
            JUDGE_PASS="$2"
            shift 2
            ;;
        --pred_file)
            FORCED_PRED_FILE="$2"
            shift 2
            ;;
        --data-file|--test-config)
            DATA_FILE_OVERRIDE="$2"
            TEST_CONFIG="$2"
            shift 2
            ;;
    --skip-citation|--skip_citation)
      SKIP_CITATION=true
      shift
      ;;
    --load_vllm)
      USE_VLLM_FLAG=true
      shift
      ;;
    *)
      INFERENCE_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -n "$DATA_FILE_OVERRIDE" ]; then
    if [ -f "$DATA_FILE_OVERRIDE" ]; then
        :
    elif [ -f "$REPO_ROOT/$DATA_FILE_OVERRIDE" ]; then
        DATA_FILE_OVERRIDE="$REPO_ROOT/$DATA_FILE_OVERRIDE"
        TEST_CONFIG="$DATA_FILE_OVERRIDE"
    else
        echo "ERROR: --data-file does not exist: $DATA_FILE_OVERRIDE"
        exit 1
    fi
    INFERENCE_ARGS+=(--data-file "$DATA_FILE_OVERRIDE")
fi

case "$JUDGE_PASS" in
    all|org_cov|relevance)
        ;;
    *)
        echo "ERROR: --judge_pass must be one of: all, org_cov, relevance"
        exit 1
        ;;
esac

REQUESTED_LLM_MODEL=""
REQUESTED_THINKING=false
REQUESTED_FULLTEXT=false
REQUESTED_CTX_TEXT_MODE="abstract_only"
REQUESTED_ELASTIC_SOURCE="abstracts"
HAS_RESUME=false
for ((i = 0; i < ${#INFERENCE_ARGS[@]}; i++)); do
    arg="${INFERENCE_ARGS[$i]}"
    case "$arg" in
        --llm_model)
            if [ $((i + 1)) -lt ${#INFERENCE_ARGS[@]} ]; then
                REQUESTED_LLM_MODEL="${INFERENCE_ARGS[$((i + 1))]}"
            fi
            i=$((i + 1))
            ;;
        --llm_model=*)
            REQUESTED_LLM_MODEL="${arg#--llm_model=}"
            ;;
        --thinking)
            REQUESTED_THINKING=true
            ;;
        --full-text-enrichment)
            REQUESTED_FULLTEXT=true
            ;;
        --ctx-text-mode)
            if [ $((i + 1)) -lt ${#INFERENCE_ARGS[@]} ]; then
                REQUESTED_CTX_TEXT_MODE="${INFERENCE_ARGS[$((i + 1))]}"
            fi
            i=$((i + 1))
            ;;
        --ctx-text-mode=*)
            REQUESTED_CTX_TEXT_MODE="${arg#--ctx-text-mode=}"
            ;;
        --elastic-source)
            if [ $((i + 1)) -lt ${#INFERENCE_ARGS[@]} ]; then
                REQUESTED_ELASTIC_SOURCE="${INFERENCE_ARGS[$((i + 1))]}"
            fi
            i=$((i + 1))
            ;;
        --elastic-source=*)
            REQUESTED_ELASTIC_SOURCE="${arg#--elastic-source=}"
            ;;
        --resume)
            HAS_RESUME=true
            ;;
    esac
done

# Resume convenience: if caller asked for --resume but did not provide --llm_model,
# infer model (+thinking mode) from the most recent completed prediction filename.
if [ "$HAS_RESUME" = true ] && [ -z "$REQUESTED_LLM_MODEL" ]; then
    resume_seed_file="$(ls -t "$OUTPUT_DIR"/pred_elastic_*.json 2>/dev/null | grep -v "_wip" | grep -v "_errors" | head -1 || true)"
    if [ -n "$resume_seed_file" ]; then
        resume_seed_name="$(basename "$resume_seed_file")"
        if [[ "$resume_seed_name" =~ ^pred_elastic_(.+)_([0-9]+)_([0-9]{14}|[0-9]{4}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2})\.json$ ]]; then
            inferred_model_tag="${BASH_REMATCH[1]}"
            inferred_thinking_mode=""
            if [[ "$inferred_model_tag" == *_thinking_on ]]; then
                inferred_thinking_mode="on"
                inferred_model_tag="${inferred_model_tag%_thinking_on}"
            elif [[ "$inferred_model_tag" == *_thinking_off ]]; then
                inferred_thinking_mode="off"
                inferred_model_tag="${inferred_model_tag%_thinking_off}"
            fi
            inferred_ctx_text_mode="abstract_only"
            if [[ "$inferred_model_tag" == *_ctx_abstract_plus_full_text ]]; then
                inferred_ctx_text_mode="abstract_plus_full_text"
                inferred_model_tag="${inferred_model_tag%_ctx_abstract_plus_full_text}"
            elif [[ "$inferred_model_tag" == *_ctx_full_text_or_abstract ]]; then
                inferred_ctx_text_mode="full_text_or_abstract"
                inferred_model_tag="${inferred_model_tag%_ctx_full_text_or_abstract}"
            fi
            inferred_fulltext=false
            if [[ "$inferred_model_tag" == *_fulltext ]]; then
                inferred_fulltext=true
                inferred_model_tag="${inferred_model_tag%_fulltext}"
            fi
            inferred_elastic_source="abstracts"
            if [[ "$inferred_model_tag" == *_elastic_src_both ]]; then
                inferred_elastic_source="abstracts"
                inferred_fulltext=true
                inferred_model_tag="${inferred_model_tag%_elastic_src_both}"
            elif [[ "$inferred_model_tag" == *_elastic_src_fulltext ]]; then
                inferred_elastic_source="fulltext"
                inferred_model_tag="${inferred_model_tag%_elastic_src_fulltext}"
            elif [[ "$inferred_model_tag" == *_elastic_src_abstracts ]]; then
                inferred_elastic_source="abstracts"
                inferred_model_tag="${inferred_model_tag%_elastic_src_abstracts}"
            elif [[ "$inferred_model_tag" == *_elastic_fulltext ]]; then
                # Legacy filename marker for direct full-text retrieval mode.
                inferred_elastic_source="fulltext"
                inferred_model_tag="${inferred_model_tag%_elastic_fulltext}"
            fi
            REQUESTED_LLM_MODEL="$inferred_model_tag"
            INFERENCE_ARGS+=(--llm_model "$REQUESTED_LLM_MODEL")
            if [ "$inferred_thinking_mode" = "on" ] && [ "$REQUESTED_THINKING" != true ]; then
                INFERENCE_ARGS+=(--thinking)
                REQUESTED_THINKING=true
            fi
            if [ "$inferred_fulltext" = true ] && [ "$REQUESTED_FULLTEXT" != true ]; then
                INFERENCE_ARGS+=(--full-text-enrichment)
                REQUESTED_FULLTEXT=true
            fi
            if [ "$inferred_ctx_text_mode" != "abstract_only" ] && [ "$REQUESTED_CTX_TEXT_MODE" = "abstract_only" ]; then
                INFERENCE_ARGS+=(--ctx-text-mode "$inferred_ctx_text_mode")
                REQUESTED_CTX_TEXT_MODE="$inferred_ctx_text_mode"
            fi
            if [ "$inferred_elastic_source" = "fulltext" ] && [ "$REQUESTED_ELASTIC_SOURCE" != "fulltext" ]; then
                INFERENCE_ARGS+=(--elastic-source fulltext)
                REQUESTED_ELASTIC_SOURCE="fulltext"
            fi
            echo "Auto-resume: inferred --llm_model '$REQUESTED_LLM_MODEL' from $resume_seed_name"
        fi
    fi
fi

compute_run_key() {
    local text="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        printf "%s" "$text" | sha256sum | awk '{print substr($1, 1, 16)}'
    elif command -v shasum >/dev/null 2>&1; then
        printf "%s" "$text" | shasum -a 256 | awk '{print substr($1, 1, 16)}'
    else
        printf "%s" "$text" | cksum | awk '{print $1}'
    fi
}

sanitize_model_tag() {
    local model_tag="$1"
    echo "${model_tag//\//_}" | sed 's/ /_/g'
}

pred_file_matches_requested_model() {
    local pred_file="$1"

    if [ -z "$REQUESTED_LLM_MODEL" ]; then
        return 0
    fi

    local pred_name
    local base_model_tag
    pred_name="$(basename "$pred_file")"
    base_model_tag="$(sanitize_model_tag "$REQUESTED_LLM_MODEL")"

    # Require the prediction file to start with the requested model tag.
    # This prevents stale run pointers from crossing models (e.g. glm -> openscholar).
    if [[ "$pred_name" == "pred_elastic_${base_model_tag}_"* ]]; then
        return 0
    fi
    return 1
}

pred_file_matches_requested_run() {
    local pred_file="$1"
    local pred_name

    if ! pred_file_matches_requested_model "$pred_file"; then
        return 1
    fi

    pred_name="$(basename "$pred_file")"

    # When caller explicitly requested a non-default retrieval mode, require
    # filename markers that identify that mode.
    if [ "$REQUESTED_ELASTIC_SOURCE" = "fulltext" ]; then
        if [[ "$pred_name" != *"_elastic_src_fulltext_"* ]] && [[ "$pred_name" != *"_elastic_fulltext_"* ]]; then
            return 1
        fi
    fi

    if [ "$REQUESTED_FULLTEXT" = true ]; then
        if [[ "$pred_name" != *"_elastic_src_both_"* ]] && [[ "$pred_name" != *"_fulltext_"* ]]; then
            return 1
        fi
    fi

    if [ "$REQUESTED_CTX_TEXT_MODE" != "abstract_only" ]; then
        if [[ "$pred_name" != *"_ctx_${REQUESTED_CTX_TEXT_MODE}"* ]]; then
            return 1
        fi
    fi

    return 0
}

build_run_signature() {
    local args_signature=""
    local signature_args=()
    if [ ${#INFERENCE_ARGS[@]} -gt 0 ]; then
        # Keep resume semantics stable: --resume should not create a different
        # pipeline run key, otherwise state tracking restarts from step 0.
        for arg in "${INFERENCE_ARGS[@]}"; do
            if [ "$arg" = "--resume" ]; then
                continue
            fi
            signature_args+=("$arg")
        done
        if [ ${#signature_args[@]} -gt 0 ]; then
            args_signature="$(printf '%q ' "${signature_args[@]}")"
        fi
    fi
    printf "bench=multi|retriever=elastic|args=%s|llm_provider=%s|llm_model=%s|llm_base_url=%s|forced_pred=%s" \
        "$args_signature" \
        "${LLM_PROVIDER:-}" \
        "${LLM_MODEL:-}" \
        "${LLM_BASE_URL:-}" \
        "$FORCED_PRED_FILE"
}

RUN_KEY="$(compute_run_key "$(build_run_signature)")"
STATE_FILE="${STATE_FILE_PREFIX}.${RUN_KEY}"
PRED_POINTER_FILE="${PRED_POINTER_PREFIX}.${RUN_KEY}"

if [ -n "${ES_URL:-}" ]; then
    ES_URL="$ES_URL"
elif [ -n "${KUBERNETES_SERVICE_HOST:-}" ]; then
    ES_URL="http://elasticsearch:9200"
else
    ES_URL="http://localhost:9200"
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$RUBRIC_INPUT_DIR"
mkdir -p "$JUDGE_EVAL_DIR"

find_latest_pred_file() {
    ls -t "$OUTPUT_DIR"/pred_elastic_*.json 2>/dev/null | grep -v "_wip" | grep -v "_errors" | head -1
}

find_latest_pred_file_for_run() {
    if [ -n "$REQUESTED_LLM_MODEL" ]; then
        local base_model_tag
        local model_tag
        local pred_prefix
        local by_model
        base_model_tag="$(sanitize_model_tag "$REQUESTED_LLM_MODEL")"
        model_tag="$base_model_tag"
        if [ "$REQUESTED_ELASTIC_SOURCE" = "fulltext" ]; then
            model_tag="${model_tag}_elastic_src_fulltext"
        elif [ "$REQUESTED_FULLTEXT" = true ]; then
            model_tag="${model_tag}_elastic_src_both"
        else
            model_tag="${model_tag}_elastic_src_abstracts"
        fi
        if [ "$REQUESTED_FULLTEXT" = true ]; then
            model_tag="${model_tag}_fulltext"
        fi
        if [ "$REQUESTED_CTX_TEXT_MODE" != "abstract_only" ]; then
            model_tag="${model_tag}_ctx_${REQUESTED_CTX_TEXT_MODE}"
        fi
        pred_prefix="pred_elastic_${model_tag}"
        by_model="$(ls -t "$OUTPUT_DIR"/"${pred_prefix}"_*.json 2>/dev/null | grep -v "_wip" | grep -v "_errors" | head -1 || true)"
        if [ -n "$by_model" ]; then
            echo "$by_model"
            return
        fi

        # Legacy compatibility: try old naming scheme if the new-source-tag
        # prefix is absent.
        model_tag="$base_model_tag"
        if [ "$REQUESTED_ELASTIC_SOURCE" = "fulltext" ]; then
            model_tag="${model_tag}_elastic_fulltext"
        fi
        if [ "$REQUESTED_FULLTEXT" = true ]; then
            model_tag="${model_tag}_fulltext"
        fi
        if [ "$REQUESTED_CTX_TEXT_MODE" != "abstract_only" ]; then
            model_tag="${model_tag}_ctx_${REQUESTED_CTX_TEXT_MODE}"
        fi
        pred_prefix="pred_elastic_${model_tag}"
        by_model="$(ls -t "$OUTPUT_DIR"/"${pred_prefix}"_*.json 2>/dev/null | grep -v "_wip" | grep -v "_errors" | head -1 || true)"
        if [ -n "$by_model" ]; then
            echo "$by_model"
            return
        fi

        # Extra compatibility fallback: only for default abstract-only mode.
        # For non-default runs (full-text/enrichment/ctx variants), never
        # back off to generic model files.
        if [ "$REQUESTED_ELASTIC_SOURCE" = "abstracts" ] && [ "$REQUESTED_FULLTEXT" != true ] && [ "$REQUESTED_CTX_TEXT_MODE" = "abstract_only" ]; then
            by_model="$(ls -t "$OUTPUT_DIR"/"pred_elastic_${base_model_tag}"_*.json 2>/dev/null | grep -v "_wip" | grep -v "_errors" | head -1 || true)"
            if [ -n "$by_model" ]; then
                echo "$by_model"
                return
            fi
        fi

        # Do not fall back to global latest when a model was explicitly requested:
        # returning another model would silently corrupt resume behavior.
        echo ""
        return
    fi
    find_latest_pred_file
}

write_pred_pointer() {
    local pred_file="$1"
    echo "$pred_file" > "$PRED_POINTER_FILE"
}

read_pred_pointer() {
    if [ -f "$PRED_POINTER_FILE" ]; then
        local pred_file=""
        read -r pred_file < "$PRED_POINTER_FILE" || true
        if [ -n "$pred_file" ] && [ -f "$pred_file" ] && pred_file_matches_requested_run "$pred_file"; then
            echo "$pred_file"
            return
        fi
    fi
    echo ""
}

infer_completed_step_from_pred() {
    local inferred_pred="$1"
    if [ -z "$inferred_pred" ] || [ ! -f "$inferred_pred" ]; then
        echo 0
        return
    fi

    local inferred_name
    local inferred_pred_base_for_results
    local inferred_rubric
    local inferred_citation
    local inferred_judge
    local step=1

    inferred_name="$(basename "$inferred_pred")"
    inferred_pred_base_for_results="$(normalize_pred_base_for_json_results "$inferred_name")"
    inferred_rubric="$RUBRIC_INPUT_DIR/${inferred_name%.*}.jsonl"
    inferred_citation="$OUTPUT_DIR/${inferred_name}.score_post_fix"
    inferred_judge="$JUDGE_EVAL_DIR/results_${inferred_pred_base_for_results}.json"

    # backward compat: fall back to old flat name if new-style file absent
    if [ ! -f "$inferred_judge" ]; then
        inferred_judge="$JUDGE_EVAL_DIR/results.json"
    fi

    if [ -f "$inferred_rubric" ]; then
        step=2
    fi
    if [ -f "$inferred_citation" ]; then
        step=3
    fi
    if [ -f "$inferred_judge" ]; then
        step=4
    fi
    echo "$step"
}

judge_results_cover_requested_pass() {
    local judge_results_file="$1"
    local requested_pass="$2"

    if [ ! -f "$judge_results_file" ]; then
        return 1
    fi

    python3 - "$judge_results_file" "$requested_pass" <<'PY'
import json
import sys

path = sys.argv[1]
requested_pass = sys.argv[2]

with open(path, "r") as f:
    data = json.load(f)

summary = data.get("summary", {}) if isinstance(data, dict) else {}
required = {
    "all": ["organization", "coverage", "relevance"],
    "org_cov": ["organization", "coverage"],
    "relevance": ["relevance"],
}[requested_pass]

missing = [key for key in required if key not in summary]
sys.exit(1 if missing else 0)
PY
}

# Keep legacy filename normalization compatible with both old and new naming.
normalize_pred_base_for_json_results() {
    local pred_filename="$1"
    local pred_base="${pred_filename%.json}"
    local prefix="pred_elastic_"

    if [[ "$pred_base" != ${prefix}* ]]; then
        echo "$pred_base"
        return
    fi

    local suffix="${pred_base#${prefix}}"
    if [[ "$suffix" =~ ^(.+)_([0-9]+)_([0-9]{14}|[0-9]{4}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2})$ ]]; then
        local model_tag="${BASH_REMATCH[1]}"
        local count="${BASH_REMATCH[2]}"
        local run_ts="${BASH_REMATCH[3]}"
        local model_tag_lc="${model_tag,,}"

        if [[ "$model_tag_lc" != *"glm-5"* ]] && [[ "$model_tag_lc" != *"glm5"* ]]; then
            model_tag="${model_tag%_thinking_on}"
            model_tag="${model_tag%_thinking_off}"
        fi

        echo "${prefix}${model_tag}_${count}_${run_ts}"
        return
    fi

    echo "$pred_base"
}

uses_fulltext_eval_context() {
    local pred_name="$1"
    # run_scholarqa.py only includes "_ctx_" in filenames for non-default
    # context export modes, which are the full-text-heavy eval variants.
    [[ "$pred_name" == *_ctx_* ]]
}

mark_step_done() {
    echo "$1" > "$STATE_FILE"
}

completed_step=0
if [ "${RESET_EVAL_PIPELINE:-0}" = "1" ]; then
    echo "Resetting pipeline state for current run key only..."
    reset_pred_file=""

    if [ "${RESET_EVAL_ARTIFACTS:-0}" = "1" ]; then
        # Capture candidate artifact targets before removing run-key pointers.
        reset_pred_file="$(read_pred_pointer || true)"
        if [ -z "$reset_pred_file" ]; then
            reset_pred_file="$(find_latest_pred_file_for_run || true)"
        fi
    fi

    # Only clear state for THIS run key, never all runs in the output directory.
    rm -f "$STATE_FILE"
    rm -f "$PRED_POINTER_FILE"

    if [ "${RESET_EVAL_ARTIFACTS:-0}" = "1" ]; then
        if [ -n "$reset_pred_file" ] && [ -f "$reset_pred_file" ]; then
            reset_pred_name="$(basename "$reset_pred_file")"
            reset_pred_base_for_results="$(normalize_pred_base_for_json_results "$reset_pred_name")"
            reset_pred_errors="${reset_pred_file%.json}_errors.json"
            reset_pred_citation="$OUTPUT_DIR/${reset_pred_name}.score_post_fix"
            reset_rubric_jsonl="$RUBRIC_INPUT_DIR/${reset_pred_name%.*}.jsonl"
            reset_judge_json="$JUDGE_EVAL_DIR/results_${reset_pred_base_for_results}.json"

            echo "  RESET_EVAL_ARTIFACTS=1: removing artifacts tied to: $reset_pred_name"
            rm -f "$reset_pred_file"
            rm -f "$reset_pred_errors"
            rm -f "$reset_pred_citation"
            rm -f "$reset_rubric_jsonl"
            rm -f "$reset_judge_json"
        else
            echo "  RESET_EVAL_ARTIFACTS=1 but no matching prediction file found."
        fi
    else
        echo "  Preserving predictions/scores. Set RESET_EVAL_ARTIFACTS=1 to delete current-run artifacts."
    fi
fi
if [ -n "$FORCED_PRED_FILE" ]; then
    if [ -f "$FORCED_PRED_FILE" ]; then
        :
    elif [ -f "$REPO_ROOT/$FORCED_PRED_FILE" ]; then
        FORCED_PRED_FILE="$REPO_ROOT/$FORCED_PRED_FILE"
    else
        echo "ERROR: --pred_file does not exist: $FORCED_PRED_FILE"
        exit 1
    fi

    write_pred_pointer "$FORCED_PRED_FILE"
    completed_step="$(infer_completed_step_from_pred "$FORCED_PRED_FILE")"
    echo "Forced prediction file: $FORCED_PRED_FILE"
elif [ -f "$STATE_FILE" ]; then
    read -r completed_step < "$STATE_FILE" || completed_step=0
else
    # Resume only from artifacts tied to this run key.
    inferred_pred="$(read_pred_pointer || true)"
    if [ -z "$inferred_pred" ]; then
        # Fallback: recover from the latest predictions that match this run
        # configuration, even if a prior run-key pointer/state file is absent.
        inferred_pred="$(find_latest_pred_file_for_run || true)"
    fi
    if [ -n "$inferred_pred" ]; then
        completed_step="$(infer_completed_step_from_pred "$inferred_pred")"
    fi
fi

# Defensive recovery: stale step files can outlive their matching predictions.
# If resume state says we completed inference but we cannot find a compatible
# prediction file for this run config, reset to step 0 and rerun inference.
if [ "$completed_step" -ge 1 ] && [ -z "${FORCED_PRED_FILE:-}" ]; then
    resume_pred_candidate="$(read_pred_pointer || true)"
    if [ -z "$resume_pred_candidate" ]; then
        resume_pred_candidate="$(find_latest_pred_file_for_run || true)"
    fi
    if [ -z "$resume_pred_candidate" ]; then
        echo "Resume state has no matching prediction for this run config; restarting from inference."
        completed_step=0
        rm -f "$STATE_FILE"
        rm -f "$PRED_POINTER_FILE"
    fi
fi

echo "Run key: $RUN_KEY"
echo "Resume state for this run key: step $completed_step completed (set RESET_EVAL_PIPELINE=1 to force full rerun)"

# Always run from repo root so relative script paths work even when step 1 is skipped on resume.
cd "$REPO_ROOT"

if [ "$completed_step" -lt 1 ]; then
    echo "======================================================"
    echo " 1. INFERENCE ON SCHOLARQA-MULTI"
    echo "======================================================"
    echo "DEBUG: INFERENCE_ARGS has ${#INFERENCE_ARGS[@]} elements"
    [ ${#INFERENCE_ARGS[@]} -gt 0 ] && echo "DEBUG: INFERENCE_ARGS = ${INFERENCE_ARGS[@]}" || echo "DEBUG: INFERENCE_ARGS is empty"
    cd "$REPO_ROOT"
    "$PYTHON_BIN" evals/Literature/SQA_bench/run_scholarqa.py \
        --bench multi \
        --retriever elastic \
        --es-url "$ES_URL" \
        "${INFERENCE_ARGS[@]}"
    PRED_FILE_AFTER_STEP1="$(find_latest_pred_file_for_run)"
    if [ -z "$PRED_FILE_AFTER_STEP1" ]; then
        echo "ERROR: Step 1 completed without generating predictions file."
        exit 1
    fi
    write_pred_pointer "$PRED_FILE_AFTER_STEP1"
    mark_step_done 1
else
    echo "======================================================"
    echo " 1. INFERENCE ON SCHOLARQA-MULTI (SKIPPED)"
    echo "======================================================"
fi

# Find the valid prediction file produced by this run
PRED_FILE="$(read_pred_pointer)"
if [ -z "$PRED_FILE" ]; then
    PRED_FILE="$(find_latest_pred_file_for_run)"
fi
if [ -z "$PRED_FILE" ]; then
    echo "ERROR: No predictions file found in $OUTPUT_DIR"
    exit 1
fi
write_pred_pointer "$PRED_FILE"
echo "Using predictions: $PRED_FILE"
PRED_FILENAME=$(basename "$PRED_FILE")
PRED_BASE_FOR_JSON_RESULTS="$(normalize_pred_base_for_json_results "$PRED_FILENAME")"
RUBRIC_JSONL="$RUBRIC_INPUT_DIR/${PRED_FILENAME%.*}.jsonl"
CITATION_SCORE_FILE="$OUTPUT_DIR/${PRED_FILENAME}.score_post_fix"
JUDGE_RESULTS_FILE="$JUDGE_EVAL_DIR/results_${PRED_BASE_FOR_JSON_RESULTS}.json"

if [ "$completed_step" -ge 2 ] && [ -f "$RUBRIC_JSONL" ]; then
    echo ""
    echo "======================================================"
    echo " 2. FORMATTING PREDICTIONS FOR RUBRIC EVALUATION (SKIPPED)"
    echo "======================================================"
else
    echo ""
    echo "======================================================"
    echo " 2. FORMATTING PREDICTIONS FOR RUBRIC EVALUATION"
    echo "======================================================"
    "$PYTHON_BIN" evals/Literature/SQA_bench/osds/format_rubric.py \
        --predictions "$PRED_FILE" \
        --test-config "$TEST_CONFIG" \
        --output "$RUBRIC_JSONL"
    if [ ! -s "$RUBRIC_JSONL" ]; then
        echo "ERROR: Step 2 did not generate rubric file: $RUBRIC_JSONL"
        exit 1
    fi
    mark_step_done 2
fi

if [ "$SKIP_CITATION" = true ]; then
    echo ""
    echo "======================================================"
    echo " 3. CITATION CORRECTNESS EVALUATION (SKIPPED BY FLAG)"
    echo "======================================================"
    mark_step_done 3
elif [ "$completed_step" -ge 3 ] && [ -f "$CITATION_SCORE_FILE" ]; then
    echo ""
    echo "======================================================"
    echo " 3. CITATION CORRECTNESS EVALUATION (SKIPPED)"
    echo "======================================================"
else
    echo ""
    echo "======================================================"
    echo " 3. CITATION CORRECTNESS EVALUATION"
    echo "======================================================"
    CITATION_ARGS=(--f "$PRED_FILE" --citations)
    if uses_fulltext_eval_context "$PRED_FILENAME"; then
        CITATION_ARGS+=(--autoais_chunk_size 200 --autoais_reload_model_per_chunk)
        echo "Full-text eval detected; running citation eval with chunked AutoAIS."
    fi
    "$(scoring_python)" evals/Literature/SQA_bench/scorers/citation_correctness_eval.py \
        "${CITATION_ARGS[@]}"
    if [ ! -s "$CITATION_SCORE_FILE" ]; then
        echo "ERROR: Step 3 did not generate citation score file: $CITATION_SCORE_FILE"
        exit 1
    fi
    mark_step_done 3
fi

if [ "$completed_step" -ge 4 ] && [ -f "$JUDGE_RESULTS_FILE" ] && judge_results_cover_requested_pass "$JUDGE_RESULTS_FILE" "$JUDGE_PASS"; then
    echo ""
    echo "======================================================"
    echo " 4. QUALITATIVE LLM-AS-A-JUDGE EVALUATION (PROMETHEUS) (SKIPPED)"
    echo "======================================================"
else
    echo ""
    echo "======================================================"
    echo " 4. QUALITATIVE LLM-AS-A-JUDGE EVALUATION (PROMETHEUS)"
    echo "======================================================"
    # Use vLLM if explicitly requested or if GPU is available
    VLLM_FLAG=""
    if [ "$USE_VLLM_FLAG" = true ] || (command -v nvidia-smi &> /dev/null && [ "$JUDGE_MODEL" != "litellm_openai" ]); then
        echo "Enabling --load_vllm for Prometheus eval."
        VLLM_FLAG="--load_vllm"
    fi

    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    echo "Using Prometheus judge pass: $JUDGE_PASS"
    echo "Using Prometheus judge model for organization/coverage: $JUDGE_MODEL"
    echo "Using Prometheus judge model for relevance: $JUDGE_RELEVANCE_MODEL"
    echo "Using Prometheus instruction: $JUDGE_INSTRUCTION"
    echo "Using Prometheus top_n: $JUDGE_TOP_N"
    echo "Using Prometheus max_new_tokens: $JUDGE_MAX_NEW_TOKENS"

    JUDGE_GPU_MEM_ARGS=()
    if [ -n "$JUDGE_GPU_MEMORY_UTILIZATION" ]; then
        echo "Using Prometheus vLLM gpu_memory_utilization: $JUDGE_GPU_MEMORY_UTILIZATION"
        JUDGE_GPU_MEM_ARGS=(--gpu_memory_utilization "$JUDGE_GPU_MEMORY_UTILIZATION")
    fi

    JUDGE_REMOTE_ARGS=()
    if [ -n "$JUDGE_BASE_URL" ]; then
        echo "Using remote judge base URL: $JUDGE_BASE_URL"
        JUDGE_REMOTE_ARGS+=(--litellm_api_base "$JUDGE_BASE_URL")
    fi
    if [ -z "$JUDGE_LITELLM_MODEL" ] && [ "$JUDGE_MODEL" = "litellm_openai" ]; then
        JUDGE_LITELLM_MODEL="prometheus-bgb-8x7b-v2.0"
    fi
    if [ -n "$JUDGE_LITELLM_MODEL" ]; then
        echo "Using remote judge model: $JUDGE_LITELLM_MODEL"
        JUDGE_REMOTE_ARGS+=(--litellm_model "$JUDGE_LITELLM_MODEL")
    fi
    JUDGE_RELEVANCE_REMOTE_ARGS=("${JUDGE_REMOTE_ARGS[@]}")
    if [ -z "$JUDGE_RELEVANCE_LITELLM_MODEL" ] && [ "$JUDGE_RELEVANCE_MODEL" = "litellm_openai" ]; then
        JUDGE_RELEVANCE_LITELLM_MODEL="prometheus-8x7b-v2.0"
    fi
    if [ -n "$JUDGE_RELEVANCE_BASE_URL" ]; then
        echo "Using remote relevance base URL: $JUDGE_RELEVANCE_BASE_URL"
        if [ ${#JUDGE_RELEVANCE_REMOTE_ARGS[@]} -gt 0 ]; then
            filtered_args=()
            skip_next=false
            for arg in "${JUDGE_RELEVANCE_REMOTE_ARGS[@]}"; do
                if [ "$skip_next" = true ]; then
                    skip_next=false
                    continue
                fi
                if [ "$arg" = "--litellm_api_base" ]; then
                    skip_next=true
                    continue
                fi
                filtered_args+=("$arg")
            done
            JUDGE_RELEVANCE_REMOTE_ARGS=("${filtered_args[@]}")
        fi
        JUDGE_RELEVANCE_REMOTE_ARGS+=(--litellm_api_base "$JUDGE_RELEVANCE_BASE_URL")
    fi
    if [ -n "$JUDGE_RELEVANCE_LITELLM_MODEL" ]; then
        echo "Using remote relevance model: $JUDGE_RELEVANCE_LITELLM_MODEL"
        if [ ${#JUDGE_RELEVANCE_REMOTE_ARGS[@]} -gt 0 ]; then
            filtered_args=()
            skip_next=false
            for arg in "${JUDGE_RELEVANCE_REMOTE_ARGS[@]}"; do
                if [ "$skip_next" = true ]; then
                    skip_next=false
                    continue
                fi
                if [ "$arg" = "--litellm_model" ]; then
                    skip_next=true
                    continue
                fi
                filtered_args+=("$arg")
            done
            JUDGE_RELEVANCE_REMOTE_ARGS=("${filtered_args[@]}")
        fi
        JUDGE_RELEVANCE_REMOTE_ARGS+=(--litellm_model "$JUDGE_RELEVANCE_LITELLM_MODEL")
    fi

    if judge_results_cover_requested_pass "$JUDGE_RESULTS_FILE" "$JUDGE_PASS"; then
        echo "Requested Prometheus judge pass already present in $JUDGE_RESULTS_FILE; skipping."
    else
        # Run Prometheus in one or two passes to match the ScholarQABench setup.
        if [ -f "$JUDGE_RESULTS_FILE" ]; then
            cp "$JUDGE_RESULTS_FILE" "$JUDGE_EVAL_DIR/results.json"
        else
            rm -f "$JUDGE_EVAL_DIR/results.json"
        fi

        if [ "$JUDGE_PASS" = "all" ] || [ "$JUDGE_PASS" = "org_cov" ]; then
            echo "Running Prometheus pass for organization and coverage..."
            "$(scoring_python)" evals/Literature/SQA_bench/scorers/prometheus_eval.py \
                -b "$PRED_FILE" \
                -o "$JUDGE_EVAL_DIR" \
                --model "$JUDGE_MODEL" \
                --rubric_path evals/Literature/SQA_bench/code/rubrics/prometheus_rubrics_v8.json \
                --instruction "$JUDGE_INSTRUCTION" \
                --top_n "$JUDGE_TOP_N" \
                --max_new_tokens "$JUDGE_MAX_NEW_TOKENS" \
                -f "$TEST_CONFIG" \
                --aspects organization coverage \
                "${JUDGE_GPU_MEM_ARGS[@]}" \
                "${JUDGE_REMOTE_ARGS[@]}" \
                $VLLM_FLAG
        fi

        if [ "$JUDGE_PASS" = "all" ] || [ "$JUDGE_PASS" = "relevance" ]; then
            echo "Running Prometheus pass for relevance..."
            "$(scoring_python)" evals/Literature/SQA_bench/scorers/prometheus_eval.py \
                -b "$PRED_FILE" \
                -o "$JUDGE_EVAL_DIR" \
                --model "$JUDGE_RELEVANCE_MODEL" \
                --rubric_path evals/Literature/SQA_bench/code/rubrics/prometheus_rubrics_v8.json \
                --instruction "$JUDGE_INSTRUCTION" \
                --top_n "$JUDGE_TOP_N" \
                --max_new_tokens "$JUDGE_MAX_NEW_TOKENS" \
                -f "$TEST_CONFIG" \
                --aspects relevance \
                "${JUDGE_GPU_MEM_ARGS[@]}" \
                "${JUDGE_RELEVANCE_REMOTE_ARGS[@]}" \
                $VLLM_FLAG
        fi

        # prometheus_eval.py always writes results.json; rename to the per-run name.
        if [ -f "$JUDGE_EVAL_DIR/results.json" ]; then
            mv "$JUDGE_EVAL_DIR/results.json" "$JUDGE_RESULTS_FILE"
        fi
        if [ ! -s "$JUDGE_RESULTS_FILE" ]; then
            echo "ERROR: Step 4 did not generate judge results file: $JUDGE_RESULTS_FILE"
            exit 1
        fi
    fi
    mark_step_done 4
fi

echo ""
echo "======================================================"
echo " PIPELINE COMPLETE"
echo "======================================================"
echo "Results available at:"
if [ "$SKIP_CITATION" = true ]; then
    echo "  - Citation Scores: skipped (--skip-citation)"
else
    echo "  - Citation Scores: $OUTPUT_DIR/${PRED_FILENAME}.score_post_fix"
fi
echo "  - LLM Judge Scores: $JUDGE_RESULTS_FILE"
