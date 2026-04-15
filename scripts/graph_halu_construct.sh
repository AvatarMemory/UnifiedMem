#!/bin/bash
# Launcher for graph construction for HaluMem using src.graph.halu_construct_graph
# Usage: ./scripts/graph_halu_construct.sh [OPTIONS]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$script_dir/.." && pwd)"
. "$script_dir/_project_env.sh"

# Defaults
DATA_DIR="${DATA_DIR:-$(get_project_env DATA_DIR "${REPO_ROOT}/data" "$REPO_ROOT")}"
HALU_DATA_PATH="${HALU_DATA_PATH:-}"
if [ -n "$HALU_DATA_PATH" ]; then
  DEFAULT_IN_FILE="$HALU_DATA_PATH"
elif [ -f "${DATA_DIR}/HaluMem/HaluMem-Medium.jsonl" ]; then
  DEFAULT_IN_FILE="${DATA_DIR}/HaluMem/HaluMem-Medium.jsonl"
else
  DEFAULT_IN_FILE="${DATA_DIR}/HaluMem-Medium.jsonl"
fi
IN_FILE="${IN_FILE:-$DEFAULT_IN_FILE}"
LLM_MODEL="${INDEX_MODEL:-$(get_stage_model index "gpt-4o-mini" "$REPO_ROOT")}"
EMBEDDING="${EMBEDDING:-${EMBEDDING_MODEL:-$(get_project_env EMBEDDING_MODEL "auto" "$REPO_ROOT")}}"
GRAPH_ROOT="${GRAPH_ROOT:-${HALU_GRAPH_ROOT:-$(get_project_env GRAPH_ROOT "" "$REPO_ROOT")}}"
HELP_GRAPH_ROOT="${GRAPH_ROOT:-${REPO_ROOT}/data/nc-graph_halu_mem_medium-${LLM_MODEL}}"

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --in-file PATH   Input JSONL file (default: ${IN_FILE})
  --out-dir PATH   Graph output root (default: ${HELP_GRAPH_ROOT})
  --embedding NAME Embedding backend for graph indexing (default: ${EMBEDDING})
  --llm-model NAME Optional model tag used for default graph root naming
  --help           Show this help message
EOF
}

# Parse args (accept both dash and underscore variants to match other scripts)
while [[ $# -gt 0 ]]; do
  case $1 in
    --in-file|--in_file|--input)
      IN_FILE="$2"; shift 2;;
    --out-dir|--out_dir|--outdir|--graph-root|--graph_root)
      OUT_DIR="$2"; shift 2;;
    --embedding)
      EMBEDDING="$2"; shift 2;;
    --llm-model|--llm_model)
      LLM_MODEL="$2"; export LLM_MODEL; shift 2;;
    --help|-h)
      show_help; exit 0;;
    *)
      echo "Unknown option: $1"; show_help; exit 1;;
  esac
done

if [ -z "$GRAPH_ROOT" ]; then
  GRAPH_ROOT="${REPO_ROOT}/data/nc-graph_halu_mem_medium-${LLM_MODEL}"
fi

# If --out-dir provided, pass it through (the python module will derive defaults if omitted)
CMD=(python -m src.graph.halu_construct_graph)
CMD+=(--in_file "$IN_FILE")
CMD+=(--out_dir "${OUT_DIR:-$GRAPH_ROOT}")
CMD+=(--embedding "$EMBEDDING")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
