import os
import multiprocessing as mp
from functools import partial
import json
from tqdm import tqdm
import numpy as np
import torch
import argparse
import fcntl  # For file locking

from src.flat.lme_utils import Retriever, fetch_expansion_from_cache, resolve_multiple_expansions
from src import config as cfg


def sanitize_for_path(model_name: str) -> str:
    if model_name == "gpt-4o-mini":
        return "gpt4o_mini"
    if model_name == "gpt-5-mini":
        return "gpt5_mini"
    return str(model_name).replace("/", "_").replace(".", "_").replace("-", "_")


def default_expansion_cache_paths() -> str:
    model_alias = cfg.get_stage_model("index", "gpt-4o-mini")
    base_dir = f"data/longmemeval-cleaned/expansions-{sanitize_for_path(model_alias)}_temp1"
    return ",".join(
        [
            f"{base_dir}/session-userfact.json",
            f"{base_dir}/session-keyphrase_.json",
            f"{base_dir}/session-summ_.json",
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='data/longmemeval-cleaned/longmemeval_s_cleaned.json')
    parser.add_argument('--out_dir', type=str, default='checkpoints/tmp')
    parser.add_argument('--outfile_prefix', type=str, default=None, required=False)
    parser.add_argument('--cache_dir', type=str, default=cfg.getenv('CACHE_DIR', 'data/cache'))
    
    # basic parameters
    parser.add_argument('--retriever', type=str, default=cfg.getenv('EMBEDDING_RETRIEVER', 'flat-contriever'),
                        choices=['flat-bm25', 'flat-contriever', 'flat-stella', 'flat-gte', 'flat-openai'])
    parser.add_argument('--granularity', type=str, default='session', choices=['session'])

    parser.add_argument('--use_raw_session_as_key', action='store_true', default=False, 
                        help='Also use the original session conversation as key to retrieve')

    # index expansion - now supports multiple methods and caches (comma-separated)
    parser.add_argument('--index_expansion_method', type=str, default='session-userfact,session-keyphrase,session-summ', 
                        help='Comma-separated list of expansion methods. E.g., "session-summ,session-userfact"')
    parser.add_argument('--index_expansion_llm', type=str, default=None)
    parser.add_argument('--index_expansion_result_cache', type=str, 
                        default=default_expansion_cache_paths(),
                        help='Comma-separated list of cache files corresponding to expansion methods')
    parser.add_argument('--index_expansion_result_join_mode', type=str, default='merge', 
                        choices=['separate', 'none', 'merge', 'merge_raw'])

    parser.add_argument('--value_expansion_join_mode', type=str, default='none', choices=['none', 'merge'], 
                        help='also expand values with expansion (ranked items) after retrieval')
    
    return parser.parse_args()


def check_args(args):
    print(args)
    
    # Parse expansion methods
    expansion_methods = [m.strip() for m in args.index_expansion_method.split(',')]
    
    # Check if any expansion method is used
    if not (len(expansion_methods) == 1 and expansion_methods[0] == 'none'):
        print(f'Note: index expansion methods {expansion_methods} specified')
        assert args.index_expansion_result_join_mode is not None and args.index_expansion_result_join_mode != 'none'
        
        # Validate each expansion method
        valid_methods = ['none', 'session-summ', 'session-keyphrase', 'session-userfact']
        for method in expansion_methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid expansion method: {method}. Must be one of {valid_methods}")
        
        # Parse cache paths
        if args.index_expansion_result_cache is not None and args.index_expansion_result_cache.lower() != 'none':
            cache_paths = [c.strip() for c in args.index_expansion_result_cache.split(',')]
            
            if len(cache_paths) != len(expansion_methods):
                raise ValueError(f"Number of cache paths ({len(cache_paths)}) must match number of expansion methods ({len(expansion_methods)})")
            
            # Validate that each cache path mentions the corresponding expansion method
            for method, cache_path in zip(expansion_methods, cache_paths):
                method_keyword = method.split('-')[1]  # e.g., 'summ', 'userfact', 'keyphrase'
                if method_keyword not in cache_path:
                    print(f"WARNING: Cache path '{cache_path}' may not correspond to method '{method}'")
            
            print(f'Using cached index expansion results:')
            for method, cache_path in zip(expansion_methods, cache_paths):
                print(f"  {method}: {cache_path}")


def get_outfile_prefix(args):
    # If a custom outfile_prefix is provided (and not 'none'), use it directly.
    # Otherwise default to the input filename base (no directories).
    if args.outfile_prefix is not None and args.outfile_prefix.lower() != 'none':
        return args.outfile_prefix
    return os.path.basename(args.in_file)


def process_item_flat_index(data, granularity, sess_id, timestamp):
    """
    Process a session/turn item into corpus format.
    
    Returns:
        corpus: List of text content
        ids: List of session/turn IDs
        timestamps: List of timestamps
        corpus_type: List of types ('original' for all items from this function)
        is_modified: Boolean indicating if session ID was modified
    """
    corpus = []
    is_modified = False

    if granularity == 'session':
        text = ' '.join([interact['content'] for interact in data if interact['role'] == 'user'])
        corpus.append(text)
        ids = [sess_id]
        if 'answer' in sess_id and all([not turn['has_answer'] for turn in [x for x in data if x['role'] == 'user']]):
            ids = [sess_id.replace('answer', 'noans')]
            is_modified = True
    else:
        raise NotImplementedError(f"Granularity '{granularity}' not supported")
    
    # All items from this function are original (not expanded)
    corpus_type = ['original' for _ in corpus]
    
    return corpus, ids, [timestamp for _ in corpus], corpus_type, is_modified


def batch_get_retrieved_context_and_eval(entry_list, args, index_expansion_result_cache=None, out_file=None, processed_ids=None):
    """
    Process a batch of entries: build corpus, run retrieval, evaluate results.
    """
    gpu_id = int(mp.current_process().name.split('-')[-1]) - 1
    if args.retriever in ['flat-bm25', 'flat-contriever', 'flat-stella', 'flat-gte', 'flat-openai']:
        retriever_master = Retriever(args, gpu_id=gpu_id)
    else:
        raise NotImplementedError

    results = []
    
    for entry in tqdm(entry_list):
        # Skip if already processed
        if processed_ids is not None and entry['question_id'] in processed_ids:
            continue
            
        # step 1: prepare corpus index (with potential index expansion)
        corpus, corpus_ids, corpus_timestamps, corpus_type = [], [], [], []
        session_id_mapping = {}  # store mapping from modified session/turn ids to original session ids
        
        for cur_sess_id, sess_entry, ts in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
            cur_items, cur_ids, cur_ts, cur_type, is_modified = process_item_flat_index(sess_entry, args.granularity, cur_sess_id, ts)
            corpus += cur_items
            corpus_ids += cur_ids
            corpus_timestamps += cur_ts
            corpus_type += cur_type
            if is_modified:
                session_id_mapping[cur_sess_id] = cur_ids[0]  # map original sess_id to modified id

        # Keep a copy of the original corpus before expansion
        corpus_raw, corpus_ids_raw, corpus_timestamps_raw = corpus.copy(), corpus_ids.copy(), corpus_timestamps.copy()

        # Apply index expansion if specified
        if args.index_expansion_method != 'none':
            if index_expansion_result_cache is not None:
                # Parse expansion methods and caches
                expansion_methods = [m.strip() for m in args.index_expansion_method.split(',')]
                
                for cur_sess_id, sess_entry, ts in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
                    # Fetch expansions for each method
                    expansions_dict = {}
                    for i, method in enumerate(expansion_methods):
                        cache = index_expansion_result_cache[i]
                        cur_expansion = fetch_expansion_from_cache(cache, cur_sess_id)
                        expansions_dict[method] = cur_expansion
                    
                    cur_sess_id = session_id_mapping.get(cur_sess_id, cur_sess_id)  # map to modified id if applicable
                    
                    # Apply all expansions
                    corpus, corpus_ids, corpus_timestamps, corpus_type = resolve_multiple_expansions(
                        expansion_methods, args.index_expansion_result_join_mode,
                        corpus, corpus_ids, corpus_timestamps, corpus_type,
                        expansions_dict, cur_sess_id, ts
                    )
            else:
                raise NotImplementedError("Index expansion without cache not supported")

        # step 2: run retrieval
        query = entry['question']
        scores = retriever_master.run_flat_retrieval(query, args.retriever, corpus)

        # get the score for each session (session-level ONLY)
        scores_session = {sess_id: {} for sess_id in np.unique(corpus_ids)}
        for i in range(len(corpus_ids)):
            sess_id, sess_type = corpus_ids[i], corpus_type[i]
            if sess_type not in scores_session[sess_id]:
                scores_session[sess_id][sess_type] = scores[i]
            else:
                raise ValueError(f"Duplicate sess_id and sess_type combination found: sess_id={sess_id}, sess_type={sess_type}. Each session should have unique expansion types.")

        if args.index_expansion_result_join_mode == 'merge_raw':
            assert not args.use_raw_session_as_key, "merge_raw mode merges raw session with expansions, thus no need to compute scores for raw session separately"

        # compute final score for each session by averaging scores across different types
        for sess_id in scores_session:
            # When no expansion is used (all items are 'original'), or when use_raw_session_as_key is True,
            # we average all available scores
            if args.use_raw_session_as_key or args.index_expansion_method == 'none':
                scores_session[sess_id]['final'] = np.mean(list(scores_session[sess_id].values()))
            else:
                # When expansion is used and use_raw_session_as_key is False, 
                # we only use expansion scores (excluding 'original')
                expansion_scores = [v for k, v in scores_session[sess_id].items() if k != 'original']
                if not expansion_scores:  # No expansion scores found (shouldn't happen unless only userfact is used, but handle gracefully)
                    print(f"WARNING: No expansion scores found for session {sess_id}, using original score")
                    scores_session[sess_id]['final'] = scores_session[sess_id]['original']
                else:
                    scores_session[sess_id]['final'] = np.mean(expansion_scores)

        # rank sessions based on final scores
        ranked_sessions = sorted(scores_session.items(), key=lambda x: x[1]['final'], reverse=True)
        rankings = []
        # get the id of corpus_ids_raw based on ranked session ids
        for sess_id, sess_scores in ranked_sessions:
            rankings.append(corpus_ids_raw.index(sess_id))

        # do value expansion - merge expansion items into corpus_raw for display
        if args.value_expansion_join_mode != 'none':
            # Build a mapping from session_id to its expansion items
            expansion_items_by_session = {}
            for i, (cid, ctype) in enumerate(zip(corpus_ids, corpus_type)):
                if ctype != 'original':  # This is an expansion
                    if cid not in expansion_items_by_session:
                        expansion_items_by_session[cid] = []
                    expansion_items_by_session[cid].append({
                        'text': corpus[i],
                        'type': ctype,
                        'timestamp': corpus_timestamps[i]
                    })
            
            # Build ranked items with expansions included
            ranked_items_with_expansions = []
            for rid in rankings:
                # Add the original session
                sess_id = corpus_ids_raw[rid]
                ranked_items_with_expansions.append({
                    'corpus_id': sess_id,
                    'text': corpus_raw[rid],
                    'timestamp': corpus_timestamps_raw[rid],
                    'is_original': True,
                    'expansion_type': 'original'
                })
                
                # Add expansions for this session if any exist
                if args.value_expansion_join_mode == 'merge' and sess_id in expansion_items_by_session:
                    for exp_item in expansion_items_by_session[sess_id]:
                        ranked_items_with_expansions.append({
                            'corpus_id': sess_id,
                            'text': exp_item['text'],
                            'timestamp': exp_item['timestamp'],
                            'is_original': False,
                            'expansion_type': exp_item['type']
                        })
        else:
            # No value expansion - just show original items in ranked order
            ranked_items_with_expansions = [
                {
                    'corpus_id': corpus_ids_raw[rid],
                    'text': corpus_raw[rid],
                    'timestamp': corpus_timestamps_raw[rid],
                    'is_original': True,
                    'expansion_type': 'original'
                }
                for rid in rankings
            ]
        
        # step 3: record evaluation metrics
        cur_results = {
            'question_id': entry['question_id'],
            'question_type': entry['question_type'],
            'question': entry['question'],
            'answer': entry['answer'],
            'question_date': entry['question_date'],
            'haystack_dates': entry['haystack_dates'],
            'haystack_session_ids': entry['haystack_session_ids'],
            'answer_session_ids': entry['answer_session_ids'],
            'retrieval_results': {
                'query': query,
                'ranked_items': ranked_items_with_expansions
            }
        }

        results.append(cur_results)
        
        # Save result immediately to file if out_file is provided
        if out_file is not None:
            with open(out_file, 'a') as f:
                # Use file locking to prevent race conditions in multiprocessing
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    print(json.dumps(cur_results), file=f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return results


def load_processed_ids(out_file):
    """Load already processed question IDs from existing output file"""
    processed_ids = set()
    if os.path.exists(out_file):
        print(f"Found existing output file: {out_file}")
        with open(out_file, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    processed_ids.add(result['question_id'])
                except:
                    pass
        print(f"Loaded {len(processed_ids)} already processed samples")
    return processed_ids


def main(args):
    check_args(args)
    
    outfile_prefix = get_outfile_prefix(args)
    out_file = args.out_dir + '/' + outfile_prefix + '_retrievallog_{}_{}'.format(args.granularity, args.retriever)
    
    # Load already processed question IDs if resuming
    processed_ids = load_processed_ids(out_file)
    
    # load data and cache
    in_data = json.load(open(args.in_file))
    
    # Filter out already processed samples
    if processed_ids:
        original_count = len(in_data)
        in_data = [entry for entry in in_data if entry['question_id'] not in processed_ids]
        print(f"Resuming: {original_count - len(in_data)} samples already processed, {len(in_data)} remaining")
    
    if len(in_data) == 0:
        print("All samples already processed! Exiting.")
        return
    
    n_has_abstention = len([x for x in in_data if '_abs' in x['question_id']])
    if n_has_abstention > 0:
        print("Warning: found {} abstention instances within the data".format(n_has_abstention))

    # Load expansion cache only if expansion is actually used
    index_expansion_result_cache = None
    if args.index_expansion_method != 'none':
        if args.index_expansion_result_cache is not None and args.index_expansion_result_cache != 'none':
            cache_paths = [c.strip() for c in args.index_expansion_result_cache.split(',')]
            
            if len(cache_paths) == 1:
                # Single cache file
                index_expansion_result_cache = [json.load(open(cache_paths[0]))]
            else:
                # Multiple cache files
                index_expansion_result_cache = [json.load(open(cp)) for cp in cache_paths]
        else:
            raise ValueError("Index expansion method specified but no cache provided. Please set --index_expansion_result_cache")
    else:
        print("No index expansion method specified. Using only original session content for retrieval.")

    # multiprocessing
    num_processes = torch.cuda.device_count()
    if 'bm25' in args.retriever:
        num_processes = 1
    elif 'openai' in args.retriever:
        num_processes = 2
    print('Setting num processes = {} with retriever {}'.format(num_processes, args.retriever))
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(num_processes)
    worker = partial(batch_get_retrieved_context_and_eval, args=args, 
                     index_expansion_result_cache=index_expansion_result_cache, 
                     out_file=out_file, processed_ids=processed_ids)

    # chunk the data into batches
    in_data_chunked = []
    chunk_size = len(in_data) // num_processes
    remainder = len(in_data) % num_processes
    start = 0
    for i in range(num_processes):
        end = start + chunk_size + (1 if i < remainder else 0)
        in_data_chunked.append(in_data[start:end])
        start = end

    results = []
    
    for d in pool.imap_unordered(worker, in_data_chunked):
        results += d

    pool.close()

    # Load all results from file (including previously processed ones)
    all_results = []
    with open(out_file, 'r') as f:
        for line in f:
            all_results.append(json.loads(line.strip()))

    print(f"All results saved to: {out_file}")
    print(f"Total samples processed: {len(all_results)}")

    
if __name__ == '__main__':
    args = parse_args()
    main(args)
