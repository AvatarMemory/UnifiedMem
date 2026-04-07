import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evals.lme_generation_utils import convert_graph_retrieval_map_to_entries, default_lme_data_file


def main():
    parser = argparse.ArgumentParser(description='Merge graph retrieval results into LongMemEval question data.')
    parser.add_argument('--in_file', required=True, help='Input graph retrieval JSON file')
    parser.add_argument('--out_file', help='Output JSONL file (default: based on in_file with "_translated.jsonl" suffix)')
    parser.add_argument('--data_file', default=default_lme_data_file(),
                        help='LongMemEval question file used to merge graph retrieval results')
    args = parser.parse_args()

    if args.out_file is None:
        base, _ = os.path.splitext(args.in_file)
        args.out_file = f"{base}_translated.jsonl"

    with open(args.in_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    merged_entries = convert_graph_retrieval_map_to_entries(graph_data, args.data_file)

    with open(args.out_file, 'w', encoding='utf-8') as f:
        for item in merged_entries:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
