import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.expanduser("~/UnifiedMem")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description='Merge retrieval results into question data.')
    parser.add_argument('--in_file', help='Input retrieval results JSON file')
    parser.add_argument('--out_file', help='Output JSONL file (default: based on in_file with "_translated.jsonl" suffix)')
    parser.add_argument('--data_file',default=f"{PROJECT_ROOT}/data/longmemeval_s_cleaned.json")
    args = parser.parse_args()

    # 默认输出文件名生成规则
    if args.out_file is None:
        base, ext = os.path.splitext(args.in_file)
        args.out_file = f"{base}_translated.jsonl"

    graph_retrieval_results_file = args.in_file
    output_file = args.out_file
    input_file = args.data_file

    # 读取检索结果
    qid2retrieval_results = {}
    with open(graph_retrieval_results_file, 'r') as f:
        data = json.load(f)
        for qid, retrieval_results in data.items():
            qid2retrieval_results[qid] = retrieval_results

    # 读取问题数据并合并检索结果
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
                    else:
                        ranked_items.append(graph_retrieval_result)
                retrieval_results = {
                    "query": item['question'],
                    "ranked_items": ranked_items
                }
                item["retrieval_results"] = retrieval_results

    # 写入输出文件（JSONL格式）
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    main()