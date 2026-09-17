#!/usr/bin/env bash
set -euo pipefail

# Local Apptainer/vLLM orchestrator for ScholarQA-Multi using the new two-agent
# stack (LibrarianAgent + LiteratureSynthesisAgent).
#   1. start local librarian-model vLLM
#   2. run predictions via run_sqa_new_stack.py
#   3. stop librarian-model vLLM so AutoAIS/Prometheus can use the GPUs
#   4. run citation correctness
#   5. start organization/coverage judge vLLM and score
#   6. start relevance judge vLLM and score

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVALS_DIR="$REPO_ROOT/evals/Literature/SQA_bench"

DATA_FILE="$EVALS_DIR/data/scholarqa_multi/scholar_multi_biomed_eval.json"
OUTPUT_DIR=""
RUN_NAME=""

LIBRARIAN_MODEL="Qwen/Qwen3.5-27B"
LIBRARIAN_BASE_URL=""
LIBRARIAN_COMMAND=""
LIBRARIAN_GPUS=8
REMOTE_LIBRARIAN=false
VIA_API=false
# Bare `python` is whatever is first on PATH, which in a non-interactive shell
# is often not the environment the harness was installed into -- the symptom is
# `No module named 'dotenv'`. literature_eval.conf documents PYTHON as the one
# knob for this.
PYTHON_BIN="${PYTHON:-python}"
NO_LIBRARIAN=false
BM25_RETRIEVAL=false
ES_URL=""
ES_FULLTEXT_URL=""

SYNTHESIS_MODEL=""
SYNTHESIS_BASE_URL=""
SYNTHESIS_OPEN_AI=false
THINKING=false

JUDGE_GPUS=4
MIN_FREE_MB=0
# Seconds to wait for a vLLM server to answer /models. The judges are 87 GB
# each and load from scratch, which routinely takes >15 min; the old hardcoded
# 900 s killed them mid-load.
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-3600}"
PORT=8000
# No default that is right for everyone: point it at a vllm-openai image you
# can read. literature_eval.conf is where to set it once.
APPTAINER_IMAGE="${APPTAINER_IMAGE:-}"
KILL_OWN_GPU_PROCESSES=false

PRED_FILE=""
SKIP_PREDICTIONS=false
SKIP_CITATION=false
SKIP_JUDGES=false
JUDGE_PASS="all"

TOP_K=""
MAX_WORKERS=1
VERBOSE=false
LIMIT=""
SAMPLE=""
SEED=42
RESUME=false
MAX_RETRIES=2

usage() {
    cat <<'EOF'
Usage:
  bash evals/Literature/SQA_bench/run_local_multieval_new_stack_apptainer.sh [options]

Core options:
  --data-file PATH                    ScholarQA data file.
  --run-name NAME                     Name for the output subdirectory.
  --output-dir PATH                   Override the inferred output directory.
  --pred-file PATH                    Reuse an existing predictions file.

Librarian (local vLLM) options:
  --librarian-model NAME              Model name for the librarian vLLM. Default: Qwen/Qwen3.5-27B.
  --librarian-command CMD             Command run inside Apptainer to serve the librarian model.
  --librarian-gpus N                  Number of GPUs for the librarian vLLM.
  --remote-librarian                  Do not start a local vLLM; read LLM_BASE_URL from the environment.
  --via-api                           Run each question on the orchestrator (POST /run-agent/stream)
                                      instead of an in-process agent. Implies --remote-librarian:
                                      no vLLM is started for the librarian, so the only local GPUs
                                      needed are the ones the Prometheus judges use.
  --no-librarian                      Skip retrieval entirely; answer from synthesis model knowledge only.
  --bm25-retrieval                    Pure BM25 baseline: retrieve from OSDS ES indices, no LLM in retrieval.
  --es-url URL                        Elasticsearch URL for abstracts index (default: http://localhost:9201).
  --es-fulltext-url URL               Elasticsearch URL for fulltext index (default: http://localhost:9202).

Synthesis agent options:
  --synthesis-model NAME              Model name for the synthesis agent.
  --synthesis-base-url URL            Base URL for the synthesis agent.
  --synthesis-open-ai                 Route synthesis to the OpenAI API (default model gpt-4o).
  --thinking                          Enable thinking/reasoning mode on the synthesis agent.

Server / GPU options:
  --judge-gpus N                      Number of GPUs for each judge vLLM.
  --min-free-mb N                     Require selected GPUs to have at least N MB free.
  --port N                            Local vLLM port. Default: 8000.
  --apptainer-image PATH              Apptainer image path.
  --kill-own-gpu-processes            Kill current user's GPU processes on selected GPUs.

Stage options:
  --skip-predictions                  Use --pred-file; skip answer-model vLLM.
  --skip-citation                     Do not run citation correctness.
  --skip-judges                       Do not run Prometheus judges.
  --judge-pass all|org_cov|relevance  Which Prometheus judge pass to run. Default: all.

Prediction options:
  --top-k N                           Passages the librarian returns per question.
  --max-workers N                     Concurrent questions. Default: 1.
  --limit N                           Limit number of questions.
  --sample FRACTION                   Random sample fraction (e.g. 0.1).
  --seed N                            Seed for --sample. Default: 42.
  --resume                            Resume partial predictions when possible.
  --max-retries N                     Auto-retry failed questions. Default: 2.
  --verbose                           Enable agent verbose logging.

Examples:
  # Local Qwen librarian + GPT-4o synthesis on the biomed multi subset.
  bash evals/Literature/SQA_bench/run_local_multieval_new_stack_apptainer.sh \
    --synthesis-open-ai

  # Re-score existing predictions only.
  bash evals/Literature/SQA_bench/run_local_multieval_new_stack_apptainer.sh \
    --skip-predictions \
    --pred-file /path/to/pred_file.json
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-file)
            DATA_FILE="${2:-}"
            shift 2
            ;;
        --run-name)
            RUN_NAME="${2:-}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:-}"
            shift 2
            ;;
        --pred-file|--pred_file)
            PRED_FILE="${2:-}"
            shift 2
            ;;
        --librarian-model)
            LIBRARIAN_MODEL="${2:-}"
            shift 2
            ;;
        --librarian-command)
            LIBRARIAN_COMMAND="${2:-}"
            shift 2
            ;;
        --librarian-gpus|--generator-gpus)
            LIBRARIAN_GPUS="${2:-}"
            shift 2
            ;;
        --librarian-base-url)
            LIBRARIAN_BASE_URL="${2:-}"
            REMOTE_LIBRARIAN=true
            shift 2
            ;;
        --remote-librarian|--remote-generator)
            REMOTE_LIBRARIAN=true
            shift
            ;;
        --via-api)
            # The agent runs inside the orchestrator, so nothing local serves it:
            # imply --remote-librarian so no vLLM is started for the librarian.
            VIA_API=true
            REMOTE_LIBRARIAN=true
            shift
            ;;
        --no-librarian)
            NO_LIBRARIAN=true
            REMOTE_LIBRARIAN=true
            shift
            ;;
        --bm25-retrieval)
            BM25_RETRIEVAL=true
            REMOTE_LIBRARIAN=true
            shift
            ;;
        --es-url)
            ES_URL="${2:-}"
            shift 2
            ;;
        --es-fulltext-url)
            ES_FULLTEXT_URL="${2:-}"
            shift 2
            ;;
        --synthesis-model)
            SYNTHESIS_MODEL="${2:-}"
            shift 2
            ;;
        --synthesis-base-url)
            SYNTHESIS_BASE_URL="${2:-}"
            shift 2
            ;;
        --synthesis-open-ai)
            SYNTHESIS_OPEN_AI=true
            shift
            ;;
        --thinking)
            THINKING=true
            shift
            ;;
        --judge-gpus)
            JUDGE_GPUS="${2:-}"
            shift 2
            ;;
        --min-free-mb)
            MIN_FREE_MB="${2:-}"
            shift 2
            ;;
        --port)
            PORT="${2:-}"
            shift 2
            ;;
        --apptainer-image)
            APPTAINER_IMAGE="${2:-}"
            shift 2
            ;;
        --kill-own-gpu-processes)
            KILL_OWN_GPU_PROCESSES=true
            shift
            ;;
        --skip-predictions)
            SKIP_PREDICTIONS=true
            shift
            ;;
        --skip-citation)
            SKIP_CITATION=true
            shift
            ;;
        --skip-judges)
            SKIP_JUDGES=true
            shift
            ;;
        --judge-pass|--judge_pass)
            JUDGE_PASS="${2:-}"
            shift 2
            ;;
        --top-k)
            TOP_K="${2:-}"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="${2:-}"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --limit)
            LIMIT="${2:-}"
            shift 2
            ;;
        --sample)
            SAMPLE="${2:-}"
            shift 2
            ;;
        --seed)
            SEED="${2:-}"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --max-retries|--max_retries)
            MAX_RETRIES="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

case "$JUDGE_PASS" in
    all|org_cov|relevance) ;;
    *)
        echo "ERROR: --judge-pass must be all, org_cov, or relevance" >&2
        exit 1
        ;;
esac

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: data file does not exist: $DATA_FILE" >&2
    exit 1
fi

if [ -n "$PRED_FILE" ]; then
    if [ ! -f "$PRED_FILE" ]; then
        echo "ERROR: predictions file does not exist: $PRED_FILE" >&2
        exit 1
    fi
    pred_dir="$(cd "$(dirname "$PRED_FILE")" && pwd)"
    PRED_FILE="$pred_dir/$(basename "$PRED_FILE")"
fi

if [ -z "$LIBRARIAN_COMMAND" ]; then
    LIBRARIAN_COMMAND="vllm serve $LIBRARIAN_MODEL --port $PORT --tensor-parallel-size $LIBRARIAN_GPUS --max-model-len 262144 --reasoning-parser qwen3 --language-model-only"
fi

sanitize_path_component() {
    local value="$1"
    value="${value//\//_}"
    value="${value// /_}"
    value="${value//:/_}"
    echo "$value"
}

infer_run_name() {
    local lib_slug synth_slug
    if [ -n "$SYNTHESIS_MODEL" ]; then
        synth_slug="$(sanitize_path_component "$SYNTHESIS_MODEL")"
    elif [ "$SYNTHESIS_OPEN_AI" = true ]; then
        synth_slug="gpt-4o"
    else
        synth_slug="default"
    fi
    if [ "$NO_LIBRARIAN" = true ]; then
        lib_slug="llm-only"
    elif [ "$BM25_RETRIEVAL" = true ]; then
        lib_slug="bm25-osds"
    else
        lib_slug="$(sanitize_path_component "$LIBRARIAN_MODEL")"
    fi
    echo "synth-${synth_slug}_lib-${lib_slug}"
}

infer_dataset_slug() {
    local data_name
    data_name="$(basename "$DATA_FILE")"
    case "$data_name" in
        scholar_multi_biomed_eval.json) echo "scholarqa_multi_bio" ;;
        *)                              echo "scholarqa_multi" ;;
    esac
}

if [ -z "$OUTPUT_DIR" ]; then
    if [ -n "$PRED_FILE" ]; then
        OUTPUT_DIR="$(dirname "$PRED_FILE")"
    else
        DATASET_SLUG="$(infer_dataset_slug)"
        if [ -z "$RUN_NAME" ]; then
            RUN_NAME="$(infer_run_name)"
        fi
        RUN_NAME="$(sanitize_path_component "$RUN_NAME")"
        OUTPUT_DIR="$EVALS_DIR/output/new_stack/scholarqa_multi/${DATASET_SLUG}/${RUN_NAME}"
    fi
fi

mkdir -p "$OUTPUT_DIR"
echo "Output dir: $OUTPUT_DIR"

SERVER_PID=""

select_gpus() {
    local count="$1"
    local min_free_mb="$2"
    local selected=()
    local gpu_index
    local free_mb

    while IFS=, read -r gpu_index free_mb; do
        gpu_index="${gpu_index//[[:space:]]/}"
        free_mb="${free_mb//[[:space:]]/}"
        if [ -z "$gpu_index" ] || [ -z "$free_mb" ]; then
            continue
        fi
        if [ "$free_mb" -lt "$min_free_mb" ]; then
            continue
        fi
        if gpu_has_compute_processes "$gpu_index"; then
            if [ "$KILL_OWN_GPU_PROCESSES" != true ] || ! gpu_has_only_own_compute_processes "$gpu_index"; then
                continue
            fi
        fi
        selected+=("$gpu_index")
        if [ "${#selected[@]}" -ge "$count" ]; then
            break
        fi
    done < <(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
            sort -t, -k2,2nr
    )

    local joined=""
    for gpu_index in "${selected[@]}"; do
        if [ -z "$joined" ]; then
            joined="$gpu_index"
        else
            joined="$joined,$gpu_index"
        fi
    done
    echo "$joined"
}

gpu_uuid_for_index() {
    local gpu_index="$1"

    nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits |
        awk -F, -v target="$gpu_index" '
            {
                gsub(/ /, "", $1)
                gsub(/ /, "", $2)
                if ($1 == target) print $2
            }
        ' |
        head -1
}

gpu_compute_pids_for_index() {
    local gpu_index="$1"
    local gpu_uuid
    gpu_uuid="$(gpu_uuid_for_index "$gpu_index")"
    if [ -z "$gpu_uuid" ]; then
        return
    fi

    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null |
        awk -F, -v target="$gpu_uuid" '
            {
                gsub(/ /, "", $1)
                gsub(/ /, "", $2)
                if ($1 == target && $2 != "") print $2
            }
        ' |
        sort -u
}

gpu_has_compute_processes() {
    local gpu_index="$1"
    [ -n "$(gpu_compute_pids_for_index "$gpu_index")" ]
}

gpu_has_only_own_compute_processes() {
    local gpu_index="$1"
    local current_user
    local pid
    local owner
    local found=0
    current_user="$(id -un)"

    while read -r pid; do
        if [ -z "$pid" ]; then
            continue
        fi
        found=1
        owner="$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')"
        if [ "$owner" != "$current_user" ]; then
            return 1
        fi
    done < <(gpu_compute_pids_for_index "$gpu_index")

    [ "$found" -eq 1 ]
}

print_gpu_compute_processes() {
    echo "Current GPU compute processes:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
}

kill_own_gpu_processes_for_selected_gpus() {
    local gpu_list="$1"
    local current_user
    current_user="$(id -un)"

    IFS=',' read -r -a gpus <<< "$gpu_list"
    for gpu_index in "${gpus[@]}"; do
        while read -r pid; do
            if [ -z "$pid" ]; then
                continue
            fi
            owner="$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')"
            command="$(ps -o comm= -p "$pid" 2>/dev/null | awk '{print $1}')"
            if [ "$owner" = "$current_user" ]; then
                echo "Stopping own GPU process on GPU $gpu_index: pid=$pid command=$command"
                kill -TERM "$pid" >/dev/null 2>&1 || true
            else
                echo "ERROR: GPU $gpu_index has process pid=$pid owned by $owner; refusing to kill another user's process." >&2
                print_gpu_compute_processes >&2
                exit 1
            fi
        done < <(gpu_compute_pids_for_index "$gpu_index")
    done

    sleep 10

    for gpu_index in "${gpus[@]}"; do
        while read -r pid; do
            if [ -z "$pid" ]; then
                continue
            fi
            owner="$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')"
            if [ "$owner" = "$current_user" ]; then
                echo "Force-stopping own lingering GPU process on GPU $gpu_index: pid=$pid"
                kill -KILL "$pid" >/dev/null 2>&1 || true
            else
                echo "ERROR: GPU $gpu_index still has process pid=$pid owned by $owner." >&2
                print_gpu_compute_processes >&2
                exit 1
            fi
        done < <(gpu_compute_pids_for_index "$gpu_index")
    done
}

require_selected_gpus_idle() {
    local gpu_list="$1"
    local busy=0

    IFS=',' read -r -a gpus <<< "$gpu_list"
    for gpu_index in "${gpus[@]}"; do
        if gpu_has_compute_processes "$gpu_index"; then
            echo "ERROR: selected GPU $gpu_index still has active compute processes." >&2
            busy=1
        fi
    done

    if [ "$busy" -ne 0 ]; then
        print_gpu_compute_processes >&2
        exit 1
    fi
}

require_gpu_count() {
    local gpu_list="$1"
    local count="$2"
    local actual=0

    if [ -n "$gpu_list" ]; then
        actual=$(( $(printf "%s" "$gpu_list" | tr -cd "," | wc -c) + 1 ))
    fi

    if [ "$actual" -lt "$count" ]; then
        echo "ERROR: found $actual GPUs, need $count. Selected='$gpu_list'" >&2
        exit 1
    fi
}

wait_for_server() {
    local base_url="$1"
    local label="$2"
    local deadline=$(( SECONDS + SERVER_WAIT_SECONDS ))

    echo "Waiting up to ${SERVER_WAIT_SECONDS}s for $label at $base_url"
    while [ "$SECONDS" -lt "$deadline" ]; do
        if curl -fsS "$base_url/models" >/dev/null 2>&1; then
            echo "$label is ready at $base_url"
            return
        fi
        # A dead server will never answer; fail now rather than at the deadline.
        if [ -n "${SERVER_PID:-}" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: $label vLLM exited before becoming ready." >&2
            [ -n "${SERVER_LOG:-}" ] && tail -n 40 "$SERVER_LOG" >&2
            exit 1
        fi
        sleep 5
    done

    echo "ERROR: timed out waiting for $label at $base_url after ${SERVER_WAIT_SECONDS}s" >&2
    echo "       Raise SERVER_WAIT_SECONDS if it was still loading." >&2
    [ -n "${SERVER_LOG:-}" ] && tail -n 40 "$SERVER_LOG" >&2
    exit 1
}

require_port_free() {
    local base_url="$1"
    local label="$2"

    if curl -fsS "$base_url/models" >/dev/null 2>&1; then
        echo "ERROR: $label port is already serving at $base_url. Stop the existing vLLM server first." >&2
        exit 1
    fi
}

stop_server() {
    local label="$1"
    if [ -z "${SERVER_PID:-}" ]; then
        return
    fi

    echo "Stopping $label vLLM server (process group $SERVER_PID)..."
    kill -TERM "-$SERVER_PID" >/dev/null 2>&1 || kill "$SERVER_PID" >/dev/null 2>&1 || true
    sleep 10
    kill -KILL "-$SERVER_PID" >/dev/null 2>&1 || kill -KILL "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
    SERVER_PID=""

    for _ in $(seq 1 60); do
        if ! curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            return
        fi
        sleep 2
    done

    echo "WARN: $label vLLM endpoint is still responding on port $PORT after stop attempt." >&2
}

cleanup() {
    stop_server "active"
}
trap cleanup EXIT

start_apptainer_vllm() {
    local label="$1"
    local gpu_count="$2"
    local command="$3"

    local selected_gpus
    selected_gpus="$(select_gpus "$gpu_count" "$MIN_FREE_MB")"
    require_gpu_count "$selected_gpus" "$gpu_count"

    if [ "$KILL_OWN_GPU_PROCESSES" = true ]; then
        kill_own_gpu_processes_for_selected_gpus "$selected_gpus"
    fi
    require_selected_gpus_idle "$selected_gpus"

    echo "Starting $label vLLM on CUDA_VISIBLE_DEVICES=$selected_gpus"
    echo "Command: $command"

    require_port_free "http://localhost:$PORT/v1" "$label"

    SERVER_LOG="$OUTPUT_DIR/vllm_$(sanitize_path_component "$label").log"
    echo "Server log: $SERVER_LOG"

    # --cleanenv drops the caller's environment, so NCCL tuning cannot reach the
    # container unless it is forwarded explicitly. Each is empty unless exported,
    # so this changes nothing on a host where the defaults already work. Needed
    # wherever tensor-parallel startup dies in initialize_model_parallel with
    # "NCCL error: unhandled system error" -- typically cards without NVLink.
    # PYTHONNOUSERSITE below is not optional: --cleanenv clears variables but
    # does not stop Python reading ~/.local/lib/python*/site-packages, so a
    # user-site package shadows the container's own. A user-site transformers
    # fails on `is_offline_mode` from a mismatched huggingface_hub.
    setsid apptainer exec --nv --cleanenv \
        --env CUDA_VISIBLE_DEVICES="$selected_gpus" \
        ${NCCL_DEBUG:+--env NCCL_DEBUG="$NCCL_DEBUG"} \
        ${NCCL_P2P_DISABLE:+--env NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE"} \
        ${NCCL_SHM_DISABLE:+--env NCCL_SHM_DISABLE="$NCCL_SHM_DISABLE"} \
        ${NCCL_IB_DISABLE:+--env NCCL_IB_DISABLE="$NCCL_IB_DISABLE"} \
        ${NCCL_CUMEM_ENABLE:+--env NCCL_CUMEM_ENABLE="$NCCL_CUMEM_ENABLE"} \
        --bind "/scratch/sigillo/hf_cache:/root/.cache/huggingface:rw" \
        --bind "/scratch/sigillo/pylibs:/opt/pylibs:rw" \
        --bind "/scratch/sigillo/pip-cache:/opt/pip-cache:rw" \
        --bind "/scratch/sigillo/tmp:/opt/tmp:rw" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env HF_HOME=/root/.cache/huggingface \
        --env HF_HUB_CACHE=/root/.cache/huggingface/hub \
        --env HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
        --env TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
        --env HF_XET_CACHE=/root/.cache/huggingface/xet \
        --env HF_ASSETS_CACHE=/root/.cache/huggingface/assets \
        --env HF_HUB_DISABLE_XET=1 \
        --env PYTHONNOUSERSITE=1 \
        --env XDG_CACHE_HOME=/root/.cache/huggingface \
        --env TMPDIR=/opt/tmp \
        --env PIP_CACHE_DIR=/opt/pip-cache \
        "$APPTAINER_IMAGE" \
        bash -lc "$command" >"$SERVER_LOG" 2>&1 &
    SERVER_PID="$!"
    wait_for_server "http://localhost:$PORT/v1" "$label"
}

run_predictions() {
    local log_file
    log_file="$(mktemp)"

    local py_args=(--bench multi --data-file "$DATA_FILE" --max-retries "$MAX_RETRIES")
    local run_env=(SCHOLARQA_OUTPUT_DIR="$OUTPUT_DIR")

    # Synthesis config
    if [ "$SYNTHESIS_OPEN_AI" = true ]; then
        py_args+=(--synthesis-open-ai)
    fi
    if [ -n "$SYNTHESIS_MODEL" ]; then
        py_args+=(--synthesis-model "$SYNTHESIS_MODEL")
    fi
    if [ -n "$SYNTHESIS_BASE_URL" ]; then
        py_args+=(--synthesis-base-url "$SYNTHESIS_BASE_URL")
    fi

    # Librarian config
    if [ "$NO_LIBRARIAN" = true ]; then
        py_args+=(--no-librarian)
    elif [ "$BM25_RETRIEVAL" = true ]; then
        py_args+=(--bm25-retrieval)
        [ -n "$ES_URL" ] && py_args+=(--es-url "$ES_URL")
        [ -n "$ES_FULLTEXT_URL" ] && py_args+=(--es-fulltext-url "$ES_FULLTEXT_URL")
    else
        py_args+=(--librarian-model "$LIBRARIAN_MODEL")
        if [ -n "$LIBRARIAN_BASE_URL" ]; then
            py_args+=(--librarian-base-url "$LIBRARIAN_BASE_URL")
        elif [ "$REMOTE_LIBRARIAN" != true ]; then
            run_env+=(LLM_BASE_URL="http://localhost:$PORT/v1")
        fi
    fi

    if [ "$VIA_API" = true ]; then
        py_args+=(--via-api)
    fi

    # Optional flags
    if [ "$THINKING" = true ]; then
        py_args+=(--thinking)
    fi
    if [ -n "$TOP_K" ]; then
        py_args+=(--top-k "$TOP_K")
    fi
    if [ "$MAX_WORKERS" -gt 1 ]; then
        py_args+=(--max-workers "$MAX_WORKERS")
    fi
    if [ -n "$LIMIT" ]; then
        py_args+=(--limit "$LIMIT")
    fi
    if [ -n "$SAMPLE" ]; then
        py_args+=(--sample "$SAMPLE" --seed "$SEED")
    fi
    if [ "$RESUME" = true ]; then
        py_args+=(--resume)
    fi
    if [ "$VERBOSE" = true ]; then
        py_args+=(--verbose)
    fi

    echo "Running predictions..."
    (
        cd "$REPO_ROOT"
        env "${run_env[@]}" "$PYTHON_BIN" evals/Literature/SQA_bench/run_sqa_new_stack.py "${py_args[@]}"
    ) | tee "$log_file"

    PRED_FILE="$(awk '/Predictions:/ {print $2}' "$log_file" | tail -1)"
    rm -f "$log_file"

    if [ -z "$PRED_FILE" ] || [ ! -f "$PRED_FILE" ]; then
        echo "ERROR: could not determine predictions file from run_sqa_new_stack.py output" >&2
        exit 1
    fi

    echo "Predictions file: $PRED_FILE"
}

run_citation_eval() {
    if [ "$SKIP_CITATION" = true ]; then
        echo "Skipping citation correctness."
        return
    fi

    if [ -s "$PRED_FILE.score_post_fix" ]; then
        echo "Citation score already exists: $PRED_FILE.score_post_fix"
        return
    fi

    echo "Running citation correctness after answer-model server has been stopped..."
    (
        cd "$REPO_ROOT"
        "$PYTHON_BIN" evals/Literature/SQA_bench/code/scripts/citation_correctness_eval.py \
            --f "$PRED_FILE" \
            --citations \
            --autoais_chunk_size 200 \
            --autoais_reload_model_per_chunk \
            --per_question_output "$PRED_FILE.autoais_per_question.json"
    )
}

run_judge_pass() {
    local pass="$1"
    shift

    local args=(--pred_file "$PRED_FILE" --data-file "$DATA_FILE" --skip-citation --judge_pass "$pass" "$@")
    local run_env=(SCHOLARQA_OUTPUT_DIR="$OUTPUT_DIR" LLM_BASE_URL="http://localhost:$PORT/v1")

    (
        cd "$REPO_ROOT"
        env "${run_env[@]}" ./evals/Literature/SQA_bench/run_multi_evals_k8s.sh "${args[@]}"
    )
}

if [ "$SKIP_PREDICTIONS" = true ]; then
    if [ -z "$PRED_FILE" ] || [ ! -f "$PRED_FILE" ]; then
        echo "ERROR: --skip-predictions requires --pred-file" >&2
        exit 1
    fi
else
    if [ "$REMOTE_LIBRARIAN" = true ]; then
        echo "Using remote/configured librarian; not starting local vLLM."
    else
        start_apptainer_vllm "librarian" "$LIBRARIAN_GPUS" "$LIBRARIAN_COMMAND"
    fi
    run_predictions
    if [ "$REMOTE_LIBRARIAN" != true ]; then
        stop_server "librarian"
    fi
fi

run_citation_eval

if [ "$SKIP_JUDGES" = true ]; then
    echo "Skipping Prometheus judges."
    exit 0
fi

if [ "$JUDGE_PASS" = "all" ] || [ "$JUDGE_PASS" = "org_cov" ]; then
    start_apptainer_vllm \
        "organization/coverage judge" \
        "$JUDGE_GPUS" \
        "vllm serve prometheus-eval/prometheus-bgb-8x7b-v2.0 --served-model-name prometheus-bgb-8x7b-v2.0 --host 0.0.0.0 --port $PORT --tensor-parallel-size $JUDGE_GPUS --gpu-memory-utilization 0.90"
    run_judge_pass \
        org_cov \
        --judge_model litellm_openai \
        --judge_base_url "http://localhost:$PORT/v1" \
        --judge_litellm_model prometheus-bgb-8x7b-v2.0
    stop_server "organization/coverage judge"
fi

if [ "$JUDGE_PASS" = "all" ] || [ "$JUDGE_PASS" = "relevance" ]; then
    start_apptainer_vllm \
        "relevance judge" \
        "$JUDGE_GPUS" \
        "vllm serve prometheus-eval/prometheus-8x7b-v2.0 --served-model-name prometheus-8x7b-v2.0 --host 0.0.0.0 --port $PORT --tensor-parallel-size $JUDGE_GPUS --gpu-memory-utilization 0.90"
    run_judge_pass \
        relevance \
        --judge_relevance_model litellm_openai \
        --judge_relevance_base_url "http://localhost:$PORT/v1" \
        --judge_relevance_litellm_model prometheus-8x7b-v2.0
    stop_server "relevance judge"
fi

echo "Done."
echo "Predictions: $PRED_FILE"
echo "Citation score: $PRED_FILE.score_post_fix"
