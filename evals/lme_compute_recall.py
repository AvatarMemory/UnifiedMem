"""
Created by Sen, 2025-11-1.
This script computes retrieval metrics both for graph retrieval results and flat retrieval results.
The metric logic is adapted from the original longmemeval repo.
"""
import json
import argparse
import numpy as np
from tqdm import tqdm
import os
import sys

from .lme_eval_utils import evaluate_retrieval

def parse_args():
    parser = argparse.ArgumentParser()
    # Set default values - modify these according to your needs
    parser.add_argument('--in_file', type=str, help='Input file containing retrieval results (JSONL format)')
    parser.add_argument('--oracle_file', type=str, default=f'data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json',
                        help='Oracle file containing correct documents')
    parser.add_argument('--haystack_file', type=str, default=f'data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json',  # NOTE: change this to longmemeval_m_cleaned.json to evaluate results for M datasets
                        help='Haystack file containing haystack session ids and sessions')
    parser.add_argument('--out_file', type=str, default=None,
                        help='Output file to save results with recomputed metrics (optional)')
    parser.add_argument('--granularity', type=str, default='session', choices=['session'],
                        help='Granularity level for evaluation')
    parser.add_argument('--print_only', action='store_true',
                        help='Only print metrics without saving to file (default if out_file not specified)')
    return parser.parse_args()

def get_rankings_from_retrieval_results(ranked_items, full_corpus_ids):
    """Extract rankings from ranked_items, mapping corpus_ids to indices"""
    # Create a mapping from corpus_id to index in full_corpus_ids
    corpus_id_to_idx = {corpus_id: idx for idx, corpus_id in enumerate(full_corpus_ids)}
    
    # Build rankings list from ranked_items
    # ranked_items is already sorted by retrieval scores (best first)
    rankings = []
    seen_indices = set()
    
    for item in ranked_items:
        corpus_id = item['corpus_id'].replace('noans_', 'answer_')
        if corpus_id in corpus_id_to_idx:
            idx = corpus_id_to_idx[corpus_id]
            if idx not in seen_indices:  # Avoid duplicates
                rankings.append(idx)
                seen_indices.add(idx)
    
    # Handle any corpus_ids in full_corpus_ids that were not in ranked_items
    # These should be appended at the end (unranked documents)
    for idx in range(len(full_corpus_ids)):
        if idx not in seen_indices:
            rankings.append(idx)
    
    return rankings

def recompute_metrics_for_entry(entry, granularity):
    """Recompute retrieval metrics for a single entry"""
    # Step 1: Rebuild full corpus_ids from haystack data
    full_corpus_ids = entry["haystack_session_ids"]
    
    # Step 2: Identify correct documents (those containing "answer")
    # correct_docs = entry["answer_session_ids"]
    # NOTE: the following logic is same with the LME run_retrieval.py script
    correct_docs = []
    for sess_id,session in zip(entry['haystack_session_ids'], entry['haystack_sessions']):
        if 'answer' in sess_id: 
            # if all([not turn['has_answer'] for turn in [x for x in session if x['role'] == 'user']]):
            if all([not turn['has_answer'] for turn in session]):
                # print(sess_id)
                continue
            else:
                correct_docs.append(sess_id)
    
    # Step 3: Extract rankings from retrieval_results
    ranked_items = entry['retrieval_results']['ranked_items']
    rankings = get_rankings_from_retrieval_results(ranked_items, full_corpus_ids)
    
    # Step 4: Recompute metrics
    metrics = {
        'session': {},
        'turn': {}
    }
    
    # for k in [1, 3, 5, 10, 30, 50]:
    for k in [5, 10, 20, 30]:
        recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, full_corpus_ids, k=k)
        metrics[granularity].update({
            # 'recall_any@{}'.format(k): recall_any,
            'recall_all@{}'.format(k): recall_all,
            'ndcg_any@{}'.format(k): ndcg_any
        })
        
        if granularity == 'turn':
            pass
    
    # if entry['retrieval_results']['metrics']['session']['recall_all@5'] != metrics['session']['recall_all@5']:
    #     print(entry['question_id'])
    #     print(entry['retrieval_results']['metrics']['session']['recall_all@5'])
    #     print(metrics['session']['recall_all@5'])

    # Update entry with recomputed metrics
    entry['retrieval_results']['metrics'] = metrics
    
    return entry

def main(args):
    # Load oracle data to get correct documents
    qid2goldsids = {}
    qid2qtype = {}
    with open(args.oracle_file, 'r') as f:
        data = json.load(f)
        for item in data:
            qid2goldsids[item['question_id']] = item['answer_session_ids']
            qid2qtype[item['question_id']] = item['question_type']
            
    # Load input data to get full corpus ids
    qid2haystack_session_ids = {}
    qid2haystack_sessions = {}
    with open(args.haystack_file, 'r') as f:
        data = json.load(f)
        for item in data:
            qid2haystack_session_ids[item['question_id']] = item['haystack_session_ids']
            qid2haystack_sessions[item['question_id']] = item['haystack_sessions']

    # Load retrieval results
    print(f"Loading retrieval results from {args.in_file}...")
    results = []
    if args.in_file.endswith('.json'):
        with open(args.in_file, 'r') as f:
            data = json.load(f)
            for qid, retrieval_results in data.items():
                results.append({
                    'question_id': qid,
                    'haystack_session_ids': qid2haystack_session_ids[qid],
                    'haystack_sessions': qid2haystack_sessions[qid],
                    'answer_session_ids': qid2goldsids[qid],  # NOTE: this is the original answer session ids (could have WRONG data), we don't use this to compute metrics
                    'retrieval_results': {
                        'ranked_items': [{'corpus_id': item['chunk_id'], 'content': item['content']} for item in retrieval_results if item['res_type'] == 'chunk'],
                        'metrics': {}
                    }
                })
    else:
        with open(args.in_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    qid = data['question_id']
                    data.update({
                        'haystack_session_ids': qid2haystack_session_ids[qid],
                        'haystack_sessions': qid2haystack_sessions[qid],
                        'answer_session_ids': qid2goldsids[qid],
                    })
                    results.append(data)
        
        # Write back to original file
        print(f"Writing updated data back to {args.in_file}...")
        with open(args.in_file, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
    
    print(f"Loaded {len(results)} entries")
    
    # Recompute metrics for each entry
    print(f"Recomputing metrics with granularity={args.granularity}...")
    for entry in tqdm(results):
        try:
            recompute_metrics_for_entry(entry, args.granularity)
        except Exception as e:
            print(f"Error processing entry {entry.get('question_id', 'unknown')}: {e}")
            continue
    
    # Compute and print averaged metrics (same logic as run_retrieval.py)
    if not results:
        print("No results to process!")
        return
    
    averaged_results = {
        'session': {},
        'turn': {}
    }
    # Results grouped by question type
    averaged_results_by_qtype = {}
    # Count samples by question type
    qtype_sample_counts = {}
    ignored_qs_abstention, ignored_qs_no_target = set(), set()
    
    # Get all metric keys from the first successful entry
    metric_keys = {}
    for eval_entry in results:
        if 'retrieval_results' in eval_entry and 'metrics' in eval_entry['retrieval_results']:
            for t in ['session', 'turn']:
                if t in eval_entry['retrieval_results']['metrics']:
                    if t not in metric_keys:
                        metric_keys[t] = list(eval_entry['retrieval_results']['metrics'][t].keys())
                    else:
                        # Update to include all keys
                        metric_keys[t] = list(set(metric_keys[t]) | set(eval_entry['retrieval_results']['metrics'][t].keys()))
            break
    
    # Compute overall metrics and metrics by question type
    for t in ['session', 'turn']:
        if t not in metric_keys:
            continue
        for k in metric_keys[t]:
            results_list = []
            results_by_qtype = {}  # qtype -> list of metric values
            
            for eval_entry in results:
                # [NOTE: raw logic from longmemeval repo] Skip abstention instances for reporting the metric
                if '_abs' in eval_entry['question_id']:
                    ignored_qs_abstention.add(eval_entry['question_id'])
                    continue
                # [NOTE: raw logic from longmemeval repo] Skip instances with no target labels in USER turns
                if not any(('has_answer' in turn) and (turn['has_answer']) 
                            for turn in [x for y in eval_entry['haystack_sessions'] 
                                        for x in y if x['role'] == 'user']):
                    ignored_qs_no_target.add(eval_entry['question_id'])
                    continue
                
                # Get metric value
                if ('retrieval_results' in eval_entry and 
                    'metrics' in eval_entry['retrieval_results'] and
                    t in eval_entry['retrieval_results']['metrics'] and
                    k in eval_entry['retrieval_results']['metrics'][t]):
                    metric_value = eval_entry['retrieval_results']['metrics'][t][k]
                    results_list.append(metric_value)
                    
                    # Group by question type
                    qid = eval_entry['question_id']
                    if qid in qid2qtype:
                        qtype = qid2qtype[qid]
                        if qtype not in results_by_qtype:
                            results_by_qtype[qtype] = []
                        results_by_qtype[qtype].append(metric_value)
                
            # Compute overall average
            if results_list:
                averaged_results[t][k] = np.mean(results_list)
            
            # Compute averages by question type
            for qtype, qtype_results in results_by_qtype.items():
                if qtype not in averaged_results_by_qtype:
                    averaged_results_by_qtype[qtype] = {'session': {}, 'turn': {}}
                averaged_results_by_qtype[qtype][t][k] = np.mean(qtype_results)
                # Record sample count (only once per qtype)
                if qtype not in qtype_sample_counts:
                    qtype_sample_counts[qtype] = len(qtype_results)
    
    print('\n' + '='*60)
    print('Computed Metrics:')
    print('='*60)
    print('Ignored {} instances due to abstention: {}'.format(
        len(ignored_qs_abstention), ignored_qs_abstention
    ))
    print('Additionally ignored {} instances due to no target turns: {}'.format(
        len(ignored_qs_no_target), ignored_qs_no_target
    ))
    print('\nOverall Averaged Metrics:')
    print(json.dumps(averaged_results, indent=2))
    
    print('\n' + '='*60)
    print('Metrics by Question Type:')
    print('='*60)
    for qtype in sorted(averaged_results_by_qtype.keys()):
        sample_count = qtype_sample_counts.get(qtype, 0)
        print(f'Question Type: {qtype} (Sample Count: {sample_count})')
        print(json.dumps(averaged_results_by_qtype[qtype], indent=2))
    print('='*60)
    
    # Print summary of sample counts
    print('\n' + '='*60)
    print('Sample Count Summary by Question Type:')
    print('='*60)
    total_valid_samples = sum(qtype_sample_counts.values())
    for qtype in sorted(qtype_sample_counts.keys()):
        count = qtype_sample_counts[qtype]
        percentage = (count / total_valid_samples * 100) if total_valid_samples > 0 else 0
        print(f'{qtype}: {count} ({percentage:.1f}%)')
    print(f'\nTotal valid samples: {total_valid_samples}')
    print('='*60)
    
    # Save results if output file is specified
    # Default behavior: if out_file is not specified, only print metrics
    if args.out_file:
        print(f"\nSaving results with recomputed metrics to {args.out_file}...")
        with open(args.out_file, 'w') as f:
            for entry in results:
                print(json.dumps(entry), file=f)
        print("Done!")
    else:
        print("\n(Results not saved - use --out_file to save results)")

if __name__ == '__main__':
    args = parse_args()
    main(args)
