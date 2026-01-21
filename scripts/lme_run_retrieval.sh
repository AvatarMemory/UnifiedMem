# in_file=${1:-"../data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json"}
in_file=${1:-"./data/longmemeval_s_cleaned.json"}
retriever=${2:-"flat-contriever"}
granularity=${3:-"session"}
index_expansion=${4:-"none"}  # none, session-summ, session-keyphrase, session-userfact
index_expansion_join_mode=${5:-"none"}   # separate, merge, merge_raw, none
index_expansion_cache=${6:-"./data/longmemeval-cleaned/expansions-llama3.1_8b/session-userfact.json"}  # comma-separated list of cache files for each expansion method
aux_model_alias=${7:-"llama-3.1-8b-instruct-ICL"}
outfile_prefix=${8:-"tmp"}
declare -A model_zoo
model_zoo["none"]="none"
model_zoo["llama-3.1-8b-instruct"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_zoo["llama-3.1-8b-instruct-ICL"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_zoo["llama-3.1-70b-instruct"]="meta-llama/Meta-Llama-3.1-70B-Instruct"
model_zoo["gpt-4o-mini"]="gpt-4o-mini"
model_zoo["gpt-5-mini"]="gpt-5-mini"
aux_model=${model_zoo["$aux_model_alias"]}

## Determine `home_dir` as the repository root (parent of this `scripts/` dir).
## This keeps model cache inside the project by default.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$script_dir/.." && pwd)"
home_dir="$REPO_ROOT"

export HF_HOME=${home_dir}/model_cache/
cache_dir=${home_dir}/model_cache/

# Ensure cache dir exists
mkdir -p "${cache_dir}"
# export HF_HOME=/data/users1/ywei/data/cache
# cache_dir=/data/users1/ywei/data/cache

# Support multiple index expansions (comma-separated)
if [[ $index_expansion == *","* ]]; then
    # Multi-expansion mode
    expansion_label=$(echo $index_expansion | sed 's/,/_/g')
    out_dir=checkpoints/LongMemEval/retrieval_logs_exp/${retriever}_expansion_w_multi_${expansion_label}/${aux_model_alias}/joinmode${index_expansion_join_mode}
elif [[ $index_expansion == "none" ]]; then
    out_dir=checkpoints/LongMemEval/retrieval_logs_exp/${retriever}/${granularity}
elif [[ $index_expansion == "session-summ" || $index_expansion == "session-keyphrase" || $index_expansion == "session-userfact" || $index_expansion == "turn-keyphrase" || $index_expansion == "turn-userfact" ]]; then
    out_dir=checkpoints/LongMemEval/retrieval_logs_exp/${retriever}_expansion_w_${index_expansion}/${aux_model_alias}/joinmode${index_expansion_join_mode}
else
    echo "Unrecognized index expansion method"
    exit 1
fi
mkdir -p $out_dir

# Ensure repo root is on PYTHONPATH so `import src` works when running scripts
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -m src.flat.lme_run_retrieval \
       --in_file $in_file \
       --retriever $retriever \
       --granularity $granularity \
       --index_expansion_method ${index_expansion} \
       --index_expansion_result_join_mode ${index_expansion_join_mode} \
       --index_expansion_llm ${aux_model} \
       --index_expansion_result_cache ${index_expansion_cache} \
       --out_dir ${out_dir} \
       --outfile_prefix ${outfile_prefix} \
       --cache_dir ${cache_dir} \
       --value_expansion_join_mode merge \
    #    --use_raw_session_as_key \


# Check that retrieval output exists before running evaluation
RETRIEVAL_LOG=${out_dir}/${outfile_prefix}_retrievallog_${granularity}_${retriever}
if [ ! -f "$RETRIEVAL_LOG" ]; then
    echo "Retrieval output not found: $RETRIEVAL_LOG" >&2
    exit 1
fi

ORACLE_FILE=${REPO_ROOT}/data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json
HAYSTACK_FILE=${REPO_ROOT}/data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json

if [ ! -f "$ORACLE_FILE" ]; then
    echo "Oracle file not found: $ORACLE_FILE" >&2
    echo "Please generate or place the oracle at that path (or update the script)." >&2
    exit 1
fi
if [ ! -f "$HAYSTACK_FILE" ]; then
    echo "Haystack file not found: $HAYSTACK_FILE" >&2
    echo "Please generate or place the haystack file at that path (or update the script)." >&2
    exit 1
fi

python -m evals.lme_compute_recall \
       --in_file ${out_dir}/${outfile_prefix}_retrievallog_${granularity}_${retriever} \
       --oracle_file ${ORACLE_FILE} \
       --haystack_file ${HAYSTACK_FILE}