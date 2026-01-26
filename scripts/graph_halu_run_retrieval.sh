#!/bin/bash
# Launcher for graph-based HaluMem retrieval using src.graph.halu_run_retrieval
# Usage: ./scripts/graph_halu_run_retrieval.sh [OPTIONS]

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$script_dir/.." && pwd)"

# Defaults
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
IN_FILE="${IN_FILE:-${DATA_DIR}/HaluMem-Medium.jsonl}"
OUT_FILE="${OUT_FILE:-""}"
GRAPH_ROOT="${GRAPH_ROOT:-${REPO_ROOT}/data/nc-graph_halu_mem_medium-4o-mini}"

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --in-file PATH   Input JSONL file (default: ${IN_FILE})
  --graph-root PATH Path to graph working-directory root (default: ${GRAPH_ROOT})
  --out-file PATH  Output JSON file for retrieval results (default: under graph-root)
  --help           Show this help message
EOF
}

# Parse args (accept dash/underscore variants and provide optional out_dir/out_file forwarding)
while [[ $# -gt 0 ]]; do
  case $1 in
    --in-file|--in_file|--input)
      IN_FILE="$2"; shift 2;;
    --out-file|--out_file|--out)
      OUT_FILE="$2"; shift 2;;
    --graph-root|--graph_root|--graph-dir|--graph_dir)
      GRAPH_ROOT="$2"; shift 2;;
      --embedding)
        EMBEDDING="$2"; shift 2;;
      --graphrag-mode|--graphrag_mode)
        GRAPHRAG_MODE="$2"; shift 2;;
      --only-need-context)
        ONLY_NEED_CONTEXT="true"; shift;;
    --out-dir|--out_dir)
      OUT_DIR="$2"; shift 2;;
    --help|-h)
      show_help; exit 0;;
    *)
      echo "Unknown option: $1"; show_help; exit 1;;
  esac
done

CMD=(python -m src.graph.halu_run_retrieval)
CMD+=(--in_file "$IN_FILE")
CMD+=(--graph_root "$GRAPH_ROOT")
if [ -n "$EMBEDDING" ]; then
  CMD+=(--embedding "$EMBEDDING")
fi
if [ -n "$GRAPHRAG_MODE" ]; then
  CMD+=(--graphrag_mode "$GRAPHRAG_MODE")
fi
if [ "$ONLY_NEED_CONTEXT" = "true" ]; then
  CMD+=(--only_need_context)
fi
if [ -n "$OUT_FILE" ]; then
  CMD+=(--out_file "$OUT_FILE")
fi
if [ -n "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
  CMD+=(--out_file "${OUT_DIR%/}/graph_retrieval_results.json")
fi

echo "Running: ${CMD[*]}"
eval "${CMD[*]}"
