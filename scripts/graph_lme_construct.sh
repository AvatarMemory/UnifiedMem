#!/bin/bash
# Launcher for graph construction for LongMemEval using src.graph.lme_construct_graph
# Usage: ./scripts/graph_lme_construct.sh [OPTIONS]

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$script_dir/.." && pwd)"

# Defaults
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/graph_s-${LLM_MODEL:-llama}}"
IN_FILE="${IN_FILE:-${DATA_DIR}/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json}"
# Default embedding: prefer environment `EMBEDDING` if set, else `EMBEDDING_MODEL`, else 'auto'
# EMBEDDING="${EMBEDDING:-${EMBEDDING_MODEL:-text-embedding-3-small}}"
EMBEDDING="${EMBEDDING:-${EMBEDDING_MODEL:-contriever}}"
ENTITY_NAMESPACE="${ENTITY_NAMESPACE:-}" # if empty, script will choose default

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --in-file PATH           Input JSON file (default: ${IN_FILE})
  --out-dir PATH           Output directory for per-question working dirs (default: ${OUT_DIR})
  --embedding NAME         Embedding backend or model name (default: ${EMBEDDING})
  --entity-namespace NAME  Optional entity namespace (default depends on embedding)
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
    --entity-namespace)
      ENTITY_NAMESPACE="$2"; shift 2;;
    --help)
      show_help; exit 0;;
    *)
      echo "Unknown option: $1"; show_help; exit 1;;
  esac
done

# Build python command (runs as module to ensure package imports)
CMD="python -m src.graph.lme_construct_graph"
CMD+=" --in_file $IN_FILE"
CMD+=" --out_dir $OUT_DIR"
CMD+=" --embedding $EMBEDDING"
if [ -n "$ENTITY_NAMESPACE" ]; then
  CMD+=" --entity_namespace $ENTITY_NAMESPACE"
fi

echo "Running: $CMD"
$CMD
