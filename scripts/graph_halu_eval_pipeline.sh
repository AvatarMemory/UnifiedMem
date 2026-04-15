#!/bin/bash
# End-to-end offline graph evaluation pipeline for HaluMem.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$script_dir/.." && pwd)"
. "$script_dir/_project_env.sh"

DATA_DIR="${DATA_DIR:-$(get_project_env DATA_DIR "${REPO_ROOT}/data" "$REPO_ROOT")}"
INDEX_MODEL="${INDEX_MODEL:-$(get_stage_model index "gpt-4o-mini" "$REPO_ROOT")}"
EMBEDDING="${EMBEDDING:-${EMBEDDING_MODEL:-$(get_project_env EMBEDDING_MODEL "auto" "$REPO_ROOT")}}"
GRAPHRAG_MODE="${GRAPHRAG_MODE:-$(get_project_env GRAPHRAG_MODE "entity,chunk,one-hot-expand" "$REPO_ROOT")}"
TOP_K="${TOP_K:-$(get_project_env TOP_K "20" "$REPO_ROOT")}"
MAX_WORKERS="${MAX_WORKERS:-$(get_project_env HALU_EVAL_MAX_WORKERS "10" "$REPO_ROOT")}"
USER_NUM="${USER_NUM:-$(get_project_env HALU_EVAL_USER_NUM "" "$REPO_ROOT")}"

if [ -n "${HALU_DATA_PATH:-}" ]; then
  DEFAULT_IN_FILE="${HALU_DATA_PATH}"
elif [ -f "${DATA_DIR}/HaluMem/HaluMem-Medium.jsonl" ]; then
  DEFAULT_IN_FILE="${DATA_DIR}/HaluMem/HaluMem-Medium.jsonl"
else
  DEFAULT_IN_FILE="${DATA_DIR}/HaluMem-Medium.jsonl"
fi

IN_FILE="${IN_FILE:-$DEFAULT_IN_FILE}"
GRAPH_ROOT="${GRAPH_ROOT:-${HALU_GRAPH_ROOT:-$(get_project_env GRAPH_ROOT "" "$REPO_ROOT")}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RETRIEVAL_OUT="${RETRIEVAL_OUT:-}"
ADD_MEMORY_FILE="${ADD_MEMORY_FILE:-}"
MERGED_EVAL_FILE="${MERGED_EVAL_FILE:-}"
SCORE_FILE="${SCORE_FILE:-}"
TMP_DIR="${TMP_DIR:-}"
USE_ENTITY="false"
SKIP_CONSTRUCT="false"
SKIP_RETRIEVAL="false"
SKIP_ADD_MEMORY="false"
SKIP_GEN_EVAL="false"
SKIP_SCORE="false"

refresh_derived_paths() {
  if [ -z "$GRAPH_ROOT" ]; then
    GRAPH_ROOT="${REPO_ROOT}/data/nc-graph_halu_mem_medium-${INDEX_MODEL}"
  fi

  if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$GRAPH_ROOT"
  fi

  if [ -z "$RETRIEVAL_OUT" ]; then
    RETRIEVAL_OUT="${OUTPUT_DIR}/graph_retrieval_results-${EMBEDDING}.json"
  fi

  if [ -z "$ADD_MEMORY_FILE" ]; then
    ADD_MEMORY_FILE="${GRAPH_ROOT}/add_memory_by_session.json"
  fi

  if [ -z "$MERGED_EVAL_FILE" ]; then
    local retrieval_name
    retrieval_name="$(basename "$RETRIEVAL_OUT")"
    if [[ "$retrieval_name" == *.json ]]; then
      MERGED_EVAL_FILE="${OUTPUT_DIR}/${retrieval_name%.json}_eval_results.jsonl"
    else
      MERGED_EVAL_FILE="${OUTPUT_DIR}/${retrieval_name}_eval_results.jsonl"
    fi
  fi

  if [ -z "$SCORE_FILE" ]; then
    SCORE_FILE="${OUTPUT_DIR}/eval_stat_result.json"
  fi

  if [ -z "$TMP_DIR" ]; then
    TMP_DIR="${OUTPUT_DIR}/tmp2"
  fi
}

show_help() {
  refresh_derived_paths
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --in-file PATH         HaluMem JSONL input (default: ${IN_FILE})
  --graph-root PATH      Graph working root (default: ${GRAPH_ROOT})
  --output-dir PATH      Directory for retrieval/eval outputs (default: ${OUTPUT_DIR})
  --retrieval-out PATH   Retrieval JSON output file (default: ${RETRIEVAL_OUT})
  --merged-eval PATH     Merged eval JSONL output file (default: ${MERGED_EVAL_FILE})
  --score-file PATH      Final score JSON output file (default: ${SCORE_FILE})
  --tmp-dir PATH         Temporary cache dir for eval scoring (default: ${TMP_DIR})
  --embedding NAME       Embedding backend (default: ${EMBEDDING})
  --graphrag-mode LIST   Graphrag mode list (default: ${GRAPHRAG_MODE})
  --top-k N              Retrieval top-k (default: ${TOP_K})
  --max-workers N        halu_eval worker count (default: ${MAX_WORKERS})
  --user-num N           Evaluate only the first N users (default: all)
  --use-entity           Use entity context formatting when merging eval input
  --skip-construct       Skip graph construction
  --skip-retrieval       Skip retrieval
  --skip-add-memory      Skip add_memory generation
  --skip-gen-eval        Skip merged eval JSONL generation
  --skip-score           Skip final metric computation
  --help                 Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-file|--in_file)
      IN_FILE="$2"; shift 2;;
    --graph-root|--graph_root)
      GRAPH_ROOT="$2"; shift 2;;
    --output-dir|--output_dir)
      OUTPUT_DIR="$2"; shift 2;;
    --retrieval-out|--retrieval_out)
      RETRIEVAL_OUT="$2"; shift 2;;
    --merged-eval|--merged_eval)
      MERGED_EVAL_FILE="$2"; shift 2;;
    --score-file|--score_file)
      SCORE_FILE="$2"; shift 2;;
    --tmp-dir|--tmp_dir)
      TMP_DIR="$2"; shift 2;;
    --embedding)
      EMBEDDING="$2"; shift 2;;
    --graphrag-mode|--graphrag_mode)
      GRAPHRAG_MODE="$2"; shift 2;;
    --top-k|--top_k)
      TOP_K="$2"; shift 2;;
    --max-workers|--max_workers)
      MAX_WORKERS="$2"; shift 2;;
    --user-num|--user_num)
      USER_NUM="$2"; shift 2;;
    --use-entity)
      USE_ENTITY="true"; shift;;
    --skip-construct)
      SKIP_CONSTRUCT="true"; shift;;
    --skip-retrieval)
      SKIP_RETRIEVAL="true"; shift;;
    --skip-add-memory)
      SKIP_ADD_MEMORY="true"; shift;;
    --skip-gen-eval)
      SKIP_GEN_EVAL="true"; shift;;
    --skip-score)
      SKIP_SCORE="true"; shift;;
    --help|-h)
      show_help; exit 0;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1;;
  esac
done

refresh_derived_paths

mkdir -p "$GRAPH_ROOT" "$OUTPUT_DIR" "$(dirname "$RETRIEVAL_OUT")" "$(dirname "$MERGED_EVAL_FILE")" "$(dirname "$SCORE_FILE")" "$TMP_DIR"

echo "Input dataset: $IN_FILE"
echo "Graph root:    $GRAPH_ROOT"
echo "Output dir:    $OUTPUT_DIR"

if [ "$SKIP_CONSTRUCT" != "true" ]; then
  "$script_dir/graph_halu_construct.sh" --in-file "$IN_FILE" --out-dir "$GRAPH_ROOT" --llm-model "$INDEX_MODEL" --embedding "$EMBEDDING"
fi

if [ "$SKIP_RETRIEVAL" != "true" ]; then
  "$script_dir/graph_halu_run_retrieval.sh" \
    --in-file "$IN_FILE" \
    --graph-root "$GRAPH_ROOT" \
    --out-file "$RETRIEVAL_OUT" \
    --embedding "$EMBEDDING" \
    --graphrag-mode "$GRAPHRAG_MODE" \
    --top-k "$TOP_K" \
    --only-need-context
fi

if [ "$SKIP_ADD_MEMORY" != "true" ]; then
  python -m evals.halu_graph_eval \
    --mode add_memory \
    --graph_root "$GRAPH_ROOT" \
    --output_file "$ADD_MEMORY_FILE"
fi

if [ "$SKIP_GEN_EVAL" != "true" ]; then
  GEN_EVAL_CMD=(
    python -m evals.halu_graph_eval
    --mode gen_eval
    --graph_root "$GRAPH_ROOT"
    --dataset_path "$IN_FILE"
    --retrieve_file_path "$RETRIEVAL_OUT"
    --output_file "$MERGED_EVAL_FILE"
  )
  if [ "$USE_ENTITY" = "true" ]; then
    GEN_EVAL_CMD+=(--use_entity)
  fi
  "${GEN_EVAL_CMD[@]}"
fi

if [ "$SKIP_SCORE" != "true" ]; then
  SCORE_CMD=(
    python -m evals.halu_eval
    --file_path "$MERGED_EVAL_FILE"
    --output_file "$SCORE_FILE"
    --tmp_dir "$TMP_DIR"
    --max_workers "$MAX_WORKERS"
  )
  if [ -n "$USER_NUM" ]; then
    SCORE_CMD+=(--user_num "$USER_NUM")
  fi
  "${SCORE_CMD[@]}"
fi

echo "Pipeline complete."
echo "Retrieval JSON: $RETRIEVAL_OUT"
echo "Merged eval:    $MERGED_EVAL_FILE"
echo "Score JSON:     $SCORE_FILE"
