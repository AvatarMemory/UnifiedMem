#!/bin/bash
# Run structured memory system evaluation on HaluMem dataset
# This script evaluates the AgenticMemorySystem with memory update operations

set -e  # Exit on error

# ===== Configuration =====
# Compute repo root and default data/cache paths (make script robust to cwd)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$script_dir/.." && pwd)"

get_project_env() {
    local key="$1"
    local default_value="$2"
    python - "$key" "$default_value" "$REPO_ROOT" <<'PY'
import sys

key = sys.argv[1]
default = sys.argv[2]
repo_root = sys.argv[3]

sys.path.insert(0, repo_root)
from src import config as cfg

value = cfg.getenv(key, default)
print("" if value is None else value)
PY
}

# Data paths (default to repo-root `data/HaluMem`)
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/HaluMem}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/HaluMem_structure}"
# Default cache directory (relative to repo root) unless overridden by env var
CACHE_DIR="${CACHE_DIR:-$(get_project_env CACHE_DIR "${REPO_ROOT}/data/cache")}"

# Model configuration
EMBEDDING_MODEL="${EMBEDDING_MODEL:-$(get_project_env EMBEDDING_MODEL "contriever")}"  # stella, contriever, gte, all-MiniLM-L6-v2
LLM_MODEL="${LLM_MODEL:-$(get_project_env LLM_MODEL "gpt-4o-mini")}"
LLM_BACKEND="${LLM_BACKEND:-$(get_project_env LLM_BACKEND "openai")}"
BASE_URL="${BASE_URL:-$(get_project_env LLM_BASE_URL "$(get_project_env OPENAI_BASE_URL "")")}"
API_KEY="${API_KEY:-$(get_project_env LLM_API_KEY "$(get_project_env OPENAI_API_KEY "")")}"
QA_LLM="${QA_LLM:-$(get_project_env QA_LLM "")}"
QA_API_BASE="${QA_API_BASE:-$(get_project_env QA_API_BASE "")}"
QA_API_KEY="${QA_API_KEY:-$(get_project_env QA_API_KEY "")}"

# Retrieval configuration
RETRIEVE_METHOD="${RETRIEVE_METHOD:-$(get_project_env RETRIEVE_METHOD "separate")}"   # merge, merge_raw, or separate
QA_RETRIEVE_METHOD="${QA_RETRIEVE_METHOD:-$(get_project_env QA_RETRIEVE_METHOD "flatten")}"  # flatten or default

TOP_K="${TOP_K:-$(get_project_env TOP_K "20")}"

# Version identifier
VERSION="${VERSION:-$(get_project_env VERSION "tmp")}"

# Resume from checkpoint
RESUME="${RESUME:-$(get_project_env RESUME "true")}"

# Skip QA generation (only run memory extraction and retrieval)
SKIP_QA="${SKIP_QA:-$(get_project_env SKIP_QA "false")}"

ENABLE_UPDATE="${ENABLE_UPDATE:-$(get_project_env ENABLE_UPDATE "false")}"

# Keep original note after update operation
KEEP_UPDATE_NOTE="${KEEP_UPDATE_NOTE:-$(get_project_env KEEP_UPDATE_NOTE "true")}"

# Use cached metadata
USE_METADATA_CACHE="${USE_METADATA_CACHE:-$(get_project_env USE_METADATA_CACHE "true")}"

# Dataset version: medium or long
DATASET="${DATASET:-$(get_project_env DATASET "medium")}"

NUM_WORKERS="${NUM_WORKERS:-$(get_project_env NUM_WORKERS "8")}"

# ===== Parse arguments =====
while [[ $# -gt 0 ]]; do
    case $1 in
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --retrieve-method)
            RETRIEVE_METHOD="$2"
            shift 2
            ;;
        --qa-retrieve-method)
            QA_RETRIEVE_METHOD="$2"
            shift 2
            ;;
        --llm-model)
            LLM_MODEL="$2"
            shift 2
            ;;
        --llm-backend)
            LLM_BACKEND="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --resume)
            RESUME="true"
            shift
            ;;
        --skip-qa)
            SKIP_QA="true"
            shift
            ;;
        --enable-link)
            ENABLE_LINK="true"
            shift
            ;;
        --use-neighbour-memories)
            USE_NEIGHBOUR_MEMORIES="true"
            shift
            ;;
        --use-raw-session-as-key)
            USE_RAW_SESSION_AS_KEY="true"
            shift
            ;;
        --no-metadata-cache)
            USE_METADATA_CACHE="false"
            shift
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --qa-llm)
            QA_LLM="$2"
            shift 2
            ;;
        --qa-api-base)
            QA_API_BASE="$2"
            shift 2
            ;;
        --qa-api-key)
            QA_API_KEY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --embedding-model MODEL    Embedding model (default: contriever)"
            echo "                             Options: stella, contriever, gte, all-MiniLM-L6-v2, all-mpnet-base-v2"
            echo "  --retrieve-method METHOD   Retrieval method (default: merge)"
            echo "                             Options: merge, separate"
            echo "  --llm-model MODEL          LLM model for memory operations (default: gpt-4o-mini)"
            echo "  --llm-backend BACKEND      LLM backend (default: openai)"
            echo "                             Options: openai, vllm"
            echo "  --top-k K                  Number of memories to retrieve (default: 20)"
            echo "  --version VERSION          Version identifier for output (default: default)"
            echo "  --dataset DATASET          Dataset version: medium or long (default: medium)"
            echo "  --resume                   Resume from checkpoint"
            echo "  --skip-qa                  Skip QA generation"
            echo "  --enable-reflection        Enable reflection on memory operations"
            echo "  --keep-update-note         Keep original note after update operation"
            echo "  --no-metadata-cache        Don't use cached metadata"
            echo "  --base-url URL             Base URL for LLM API (for vLLM)"
            echo "  --api-key KEY              API key for LLM"
            echo "  --qa-llm MODEL             QA model (defaults to QA_LLM or llm_model)"
            echo "  --qa-api-base URL          Base URL for QA model API"
            echo "  --qa-api-key KEY           API key for QA model API"
            echo "  --help                     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ===== Set data path based on dataset =====
if [ "$DATASET" == "medium" ]; then
    DATA_BASENAME="HaluMem-Medium.jsonl"
elif [ "$DATASET" == "long" ]; then
    DATA_BASENAME="HaluMem-Long.jsonl"
else
    echo "Unknown dataset: $DATASET. Use 'medium' or 'long'."
    exit 1
fi

DATA_PATH="${DATA_DIR}/${DATA_BASENAME}"

# If the real dataset is missing, try falling back to `sample_data` for smoke tests.
if [ ! -f "$DATA_PATH" ]; then
    SAMPLE_PATH="${REPO_ROOT}/sample_data/${DATA_BASENAME}"
    if [ -f "$SAMPLE_PATH" ]; then
        echo "HaluMem data not found at $DATA_PATH — falling back to sample data: $SAMPLE_PATH"
        DATA_PATH="$SAMPLE_PATH"
    else
        echo "Data file not found: $DATA_PATH" >&2
        echo "Please download the HaluMem dataset and place it at: ${DATA_DIR}/" >&2
        echo "  e.g. download from: https://huggingface.co/datasets/IAAR-Shanghai/HaluMem" >&2
        echo "Or place a small sample at: ${REPO_ROOT}/sample_data/${DATA_BASENAME} for smoke testing." >&2
        exit 1
    fi
fi

# ===== Build command =====
## Ensure repo root is on PYTHONPATH so `import src` works when running as module
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Run as module to ensure package imports resolve
CMD="python -m src.flat.halu_run"
CMD="$CMD --data_path $DATA_PATH"
CMD="$CMD --out_dir $OUT_DIR"
CMD="$CMD --cache_dir $CACHE_DIR"
CMD="$CMD --embedding_model $EMBEDDING_MODEL"
CMD="$CMD --retrieve_method $RETRIEVE_METHOD"
CMD="$CMD --qa_retrieve_method $QA_RETRIEVE_METHOD"
CMD="$CMD --llm_model $LLM_MODEL"
CMD="$CMD --llm_backend $LLM_BACKEND"
CMD="$CMD --top_k $TOP_K"
CMD="$CMD --version ${VERSION}_${DATASET}"

if [ "$RESUME" == "true" ]; then
    CMD="$CMD --resume"
fi

if [ "$SKIP_QA" == "true" ]; then
    CMD="$CMD --skip_qa"
fi

if [ "$KEEP_UPDATE_NOTE" == "true" ]; then
    CMD="$CMD --keep_update_note"
fi

if [ "$ENABLE_LINK" == "true" ]; then
    CMD="$CMD --enable_link"
fi

if [ "$USE_NEIGHBOUR_MEMORIES" == "true" ]; then
    CMD="$CMD --use_neighbour_memories"
fi

if [ "$USE_RAW_SESSION_AS_KEY" == "true" ]; then
    CMD="$CMD --use_raw_session_as_key"
fi

if [ "$ENABLE_UPDATE" == "true" ]; then
    CMD="$CMD --enable_update"
fi
if [ "$USE_METADATA_CACHE" == "true" ]; then
    CMD="$CMD --use_metadata_cache"
fi

if [ -n "$BASE_URL" ]; then
    CMD="$CMD --base_url $BASE_URL"
fi

if [ -n "$API_KEY" ]; then
    CMD="$CMD --api_key $API_KEY"
fi

if [ -n "$QA_LLM" ]; then
    CMD="$CMD --qa_llm $QA_LLM"
fi

if [ -n "$QA_API_BASE" ]; then
    CMD="$CMD --qa_api_base $QA_API_BASE"
fi

if [ -n "$QA_API_KEY" ]; then
    CMD="$CMD --qa_api_key $QA_API_KEY"
fi

if [ -n "$NUM_WORKERS" ]; then
    CMD="$CMD --num_workers $NUM_WORKERS"
fi

# ===== Print configuration =====
echo "=============================================="
echo "Structured Memory System Evaluation on HaluMem"
echo "=============================================="
echo "Dataset:          $DATASET ($DATA_PATH)"
echo "Embedding model:  $EMBEDDING_MODEL"
echo "Retrieve method:  $RETRIEVE_METHOD"
echo "QA retrieve method: $QA_RETRIEVE_METHOD"
echo "LLM model:        $LLM_MODEL"
echo "LLM backend:      $LLM_BACKEND"
echo "Top-k:            $TOP_K"
echo "Version:          ${VERSION}_${DATASET}"
echo "Resume:           $RESUME"
echo "Skip QA:          $SKIP_QA"
echo "Keep update note: $KEEP_UPDATE_NOTE"
echo "Use metadata cache: $USE_METADATA_CACHE"
echo "=============================================="
echo ""
echo "Running command:"
echo "$CMD"
echo ""

# ===== Run evaluation =====
eval $CMD

# ===== Run basic stats computation =====
echo ""
echo "=============================================="
echo "Computing basic statistics..."
echo "=============================================="

FRAME="structure_${EMBEDDING_MODEL}"
RESULTS_DIR="${OUT_DIR}/${FRAME}-${VERSION}_${DATASET}"
RESULTS_FILE="${RESULTS_DIR}/structure_eval_results.jsonl"
