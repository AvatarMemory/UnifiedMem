import json
import os
import sys
PROJECT_ROOT = os.path.expanduser("~/UnifiedMem")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

input_file = f"{PROJECT_ROOT}/data/longmemeval_s_cleaned.json"
# graph_retrival_results_file = f"{PROJECT_ROOT}/data/graph_s_all-llama-3.1-8b/graph_retrieval_results-e20.json"
# output_file = f"{PROJECT_ROOT}/data/graph_s_all-llama-3.1-8b/graph_retrieval_results-e20_translated.jsonl"
# graph_retrival_results_file = f"{PROJECT_ROOT}/data/hipporag_s-4o-mini/lmes_woa_hipporag2_retrieve20_1119.json"
# output_file = f"{PROJECT_ROOT}/data/hipporag_s-4o-mini/lmes_woa_hipporag2_retrieve20_1119_translated.jsonl"
# graph_retrival_results_file = f"{PROJECT_ROOT}/data/graph_s_all-llama-3.1-8b/graph_retrieval_results-e20-contriever.json"
# output_file = f"{PROJECT_ROOT}/data/graph_s_all-llama-3.1-8b/graph_retrieval_results-e20-contriever_translated.jsonl"
# graph_retrival_results_file = f"{PROJECT_ROOT}/data/graph_m-llama-3.1-8b/graph_retrieval_results-e20-contriever.json"
# output_file = f"{PROJECT_ROOT}/data/graph_m-llama-3.1-8b/graph_retrieval_results-e20-contriever_translated.jsonl"
graph_retrival_results_file = f"{PROJECT_ROOT}/data/graph_m-4o-mini-results/graph_retrieval_results-e20-contriever.json"
output_file = f"{PROJECT_ROOT}/data/graph_m-4o-mini-results/graph_retrieval_results-e20_translated.jsonl"
# graph_retrival_results_file = f"{PROJECT_ROOT}/data/graph_m-llama-3.1-8b/graph_retrieval_results-e20-contriever.json"
# output_file = f"{PROJECT_ROOT}/data/graph_m-llama-3.1-8b/graph_retrieval_results-e20-contriever_translated.json"
# graph_retrival_results_file = f"{PROJECT_ROOT}/data/graph_s-4o-mini/graph_retrieval_results-e20-rankentitycount.json"
# output_file = f"{PROJECT_ROOT}/data/graph_s-4o-mini/graph_retrieval_results-e20-rankentitycount_translated.jsonl"
# graph_retrival_results_file = f"{PROJECT_ROOT}/data/graph_s-llama-3.1-8b/graph_retrieval_results-e20-contriever.json"
# output_file = f"{PROJECT_ROOT}/data/graph_s-llama-3.1-8b/graph_retrieval_results-e20-contriever_translated.jsonl"
# graph_retrival_results_file = f"{PROJECT_ROOT}/data/graph_s-4o-mini/graph_retrieval_results-e20-c5-rankentitycount.json"
# output_file = f"{PROJECT_ROOT}/data/graph_s-4o-mini/graph_retrieval_results-e20-c5-rankentitycount_translated.jsonl"

qid2retrieval_results = {}
with open(graph_retrival_results_file, 'r') as f:
    data = json.load(f)
    for qid, retrieval_results in data.items():
        qid2retrieval_results[qid] = retrieval_results

with open(input_file, 'r') as f:
    data = json.load(f)
    for item in data:
        if item['question_id'] in qid2retrieval_results:
            graph_retrieval_results = qid2retrieval_results[item['question_id']]
            ranked_items = []
            for graph_retrieval_result in graph_retrieval_results:
                if graph_retrieval_result['res_type'] == 'chunk':
                    ranked_items.append({
                        "res_type": "chunk",
                        "corpus_id": graph_retrieval_result['chunk_id'],
                        "text": graph_retrieval_result['content']
                    })
                    # print("chunk: ",graph_retrieval_result)
                else:
                    ranked_items.append(graph_retrieval_result)
                    # print("other: ",graph_retrieval_result)
            retrieval_results = {
                "query": item['question'],
                "ranked_items": ranked_items
            }
            item["retrieval_results"] = retrieval_results

with open(output_file, 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')