#!/bin/bash
# Launcher for graph-based LongMemEval retrieval using src.graph.lme_run_retrieval
# Usage: ./scripts/graph_lme_run_retrieval.sh [OPTIONS]

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$script_dir/.." && pwd)"
. "$script_dir/_project_env.sh"

# Defaults
DATA_DIR="${DATA_DIR:-$(get_project_env DATA_DIR "${REPO_ROOT}/data" "$REPO_ROOT")}"
MODEL_TAG="${INDEX_MODEL:-$(get_stage_model index "gpt-4o-mini" "$REPO_ROOT")}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/graph_s-${MODEL_TAG}}"
IN_FILE="${IN_FILE:-${DATA_DIR}/longmemeval_s_cleaned.json}"
# Default embedding: prefer environment `EMBEDDING` if set, else `EMBEDDING_MODEL`, else 'auto'
EMBEDDING="${EMBEDDING:-${EMBEDDING_MODEL:-$(get_project_env EMBEDDING_MODEL "contriever" "$REPO_ROOT")}}"
# GRAPHRAG_MODE="${GRAPHRAG_MODE:-entity,chunk,one-hot-expand}"
GRAPHRAG_MODE="${GRAPHRAG_MODE:-$(get_project_env GRAPHRAG_MODE "entity,chunk" "$REPO_ROOT")}"
ONLY_NEED_CONTEXT="${ONLY_NEED_CONTEXT:-$(get_project_env ONLY_NEED_CONTEXT "true" "$REPO_ROOT")}"

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --in-file PATH           Input JSON file (default: ${IN_FILE})
  --out-dir PATH           Output directory for working dirs and results (default: ${OUT_DIR})
  --embedding NAME         Embedding backend or model name (default: ${EMBEDDING})
  --graphrag-mode LIST     Comma-separated graphrag modes (default: ${GRAPHRAG_MODE})
  --only-need-context      Only return context in results
  --help                   Show this help message
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --in-file)
      IN_FILE="$2"; shift 2;;
    --out-dir)
      OUT_DIR="$2"; shift 2;;
    --embedding)
      EMBEDDING="$2"; shift 2;;
    --graphrag-mode)
      GRAPHRAG_MODE="$2"; shift 2;;
    --only-need-context)
      ONLY_NEED_CONTEXT="true"; shift;;
    --help)
      show_help; exit 0;;
    *)
      echo "Unknown option: $1"; show_help; exit 1;;
  esac
done

# Build python command (runs as module to ensure package imports)
CMD="python -m src.graph.lme_run_retrieval"
CMD+=" --in_file $IN_FILE"
CMD+=" --out_dir $OUT_DIR"
CMD+=" --embedding $EMBEDDING"
CMD+=" --graphrag_mode $GRAPHRAG_MODE"
if [ "$ONLY_NEED_CONTEXT" = "true" ]; then
  CMD+=" --only_need_context"
fi

echo "Running: $CMD"
$CMD
