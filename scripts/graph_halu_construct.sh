#!/bin/bash
# Launcher for graph construction for HaluMem using src.graph.halu_construct_graph
# Usage: ./scripts/graph_halu_construct.sh [OPTIONS]

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$script_dir/.." && pwd)"

# Defaults
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
IN_FILE="${IN_FILE:-${DATA_DIR}/HaluMem-Medium.jsonl}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --in-file PATH   Input JSONL file (default: ${IN_FILE})
  --llm-model NAME Optional model tag used for naming (not required)
  --help           Show this help message
EOF
}

# Parse args (accept both dash and underscore variants to match other scripts)
while [[ $# -gt 0 ]]; do
  case $1 in
    --in-file|--in_file|--input)
      IN_FILE="$2"; shift 2;;
    --out-dir|--out_dir|--outdir)
      OUT_DIR="$2"; shift 2;;
    --llm-model|--llm_model)
      LLM_MODEL="$2"; export LLM_MODEL; shift 2;;
    --help|-h)
      show_help; exit 0;;
    *)
      echo "Unknown option: $1"; show_help; exit 1;;
  esac
done

# If --out-dir provided, pass it through (the python module will derive defaults if omitted)
CMD=(python -m src.graph.halu_construct_graph)
CMD+=(--in_file "$IN_FILE")
if [ -n "$OUT_DIR" ]; then
  CMD+=(--out_dir "$OUT_DIR")
fi

echo "Running: ${CMD[*]}"
eval "${CMD[*]}"
