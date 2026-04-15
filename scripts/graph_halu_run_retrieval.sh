#!/bin/bash
# Launcher for graph-based HaluMem retrieval using src.graph.halu_run_retrieval
# Usage: ./scripts/graph_halu_run_retrieval.sh [OPTIONS]

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
OUT_FILE="${OUT_FILE:-""}"
OUT_DIR="${OUT_DIR:-""}"
LLM_MODEL="${INDEX_MODEL:-$(get_stage_model index "gpt-4o-mini" "$REPO_ROOT")}"
GRAPH_ROOT="${GRAPH_ROOT:-${HALU_GRAPH_ROOT:-$(get_project_env GRAPH_ROOT "" "$REPO_ROOT")}}"
EMBEDDING="${EMBEDDING:-${EMBEDDING_MODEL:-$(get_project_env EMBEDDING_MODEL "" "$REPO_ROOT")}}"
GRAPHRAG_MODE="${GRAPHRAG_MODE:-$(get_project_env GRAPHRAG_MODE "" "$REPO_ROOT")}"
ONLY_NEED_CONTEXT="${ONLY_NEED_CONTEXT:-$(get_project_env ONLY_NEED_CONTEXT "false" "$REPO_ROOT")}"
TOP_K="${TOP_K:-$(get_project_env TOP_K "20" "$REPO_ROOT")}"
HELP_GRAPH_ROOT="${GRAPH_ROOT:-${REPO_ROOT}/data/nc-graph_halu_mem_medium-${LLM_MODEL}}"

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --in-file PATH   Input JSONL file (default: ${IN_FILE})
  --graph-root PATH Path to graph working-directory root (default: ${HELP_GRAPH_ROOT})
  --out-file PATH  Output JSON file for retrieval results (default: under graph-root)
  --top-k N        Top-k documents to retrieve (default: ${TOP_K})
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
    --top-k|--top_k)
      TOP_K="$2"; shift 2;;
    --out-dir|--out_dir)
      OUT_DIR="$2"; shift 2;;
    --help|-h)
      show_help; exit 0;;
    *)
      echo "Unknown option: $1"; show_help; exit 1;;
  esac
done

if [ -z "$GRAPH_ROOT" ]; then
  GRAPH_ROOT="${REPO_ROOT}/data/nc-graph_halu_mem_medium-${LLM_MODEL}"
fi

CMD=(python -m src.graph.halu_run_retrieval)
CMD+=(--in_file "$IN_FILE")
CMD+=(--graph_root "$GRAPH_ROOT")
if [ -n "$EMBEDDING" ]; then
  CMD+=(--embedding "$EMBEDDING")
fi
if [ -n "$GRAPHRAG_MODE" ]; then
  CMD+=(--graphrag_mode "$GRAPHRAG_MODE")
fi
CMD+=(--top_k "$TOP_K")
if [ "$ONLY_NEED_CONTEXT" = "true" ]; then
  CMD+=(--only_need_context)
fi
if [ -n "$OUT_FILE" ]; then
  CMD+=(--out_file "$OUT_FILE")
fi
if [ -n "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
  CMD+=(--out_file "${OUT_DIR%/}/graph_retrieval_results-${EMBEDDING:-auto}.json")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
