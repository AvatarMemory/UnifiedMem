#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$script_dir/.." && pwd)"
. "$script_dir/_project_env.sh"

in_file=${1:-"./data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json"}
project_llm_model="${INDEX_MODEL:-$(get_stage_model index "gpt-4o-mini" "$REPO_ROOT")}"
expansion_model_alias=${2:-"$project_llm_model"}
cache_folder_arg=${3:-""}

declare -A model_zoo
model_zoo["llama-3.1-8b-instruct"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_zoo["llama-3.1-8b-instruct-ICL"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_zoo["llama-3.1-70b-instruct"]="meta-llama/Meta-Llama-3.1-70B-Instruct"
model_zoo["gpt-4o-mini"]="gpt-4o-mini"
model_zoo["gpt-5-mini"]="gpt-5-mini"
model_zoo["qwen3-8b"]="qwen3-8b"
model_zoo["qwen3_8b"]="qwen3-8b"

llm_model=${model_zoo["$expansion_model_alias"]:-"$expansion_model_alias"}

sanitize_for_path() {
    case "$1" in
        gpt-4o-mini)
            echo "gpt4o_mini"
            ;;
        gpt-5-mini)
            echo "gpt5_mini"
            ;;
        *)
            echo "$1" | tr '/.-' '___'
            ;;
    esac
}

default_cache_folder="./data/longmemeval-cleaned/expansions-$(sanitize_for_path "$expansion_model_alias")_temp1"
cache_folder=${cache_folder_arg:-"$default_cache_folder"}

mkdir -p "$cache_folder"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export LME_IN_FILE="$in_file"
export LME_CACHE_FOLDER="$cache_folder"
export INDEX_MODEL="$llm_model"

echo "Building LongMemEval flat index expansions"
echo "Input file:     $LME_IN_FILE"
echo "Model alias:    $expansion_model_alias"
echo "LLM model:      $INDEX_MODEL"
echo "Cache folder:   $LME_CACHE_FOLDER"

python "$REPO_ROOT/data_preprocessing/lme_extract_keyphrase.py"
python "$REPO_ROOT/data_preprocessing/lme_extract_summ.py"
python "$REPO_ROOT/data_preprocessing/lme_extract_userfact.py"

echo "Finished. Generated files:"
echo "  ${LME_CACHE_FOLDER}/session-keyphrase_.json"
echo "  ${LME_CACHE_FOLDER}/session-summ_.json"
echo "  ${LME_CACHE_FOLDER}/session-userfact.json"
