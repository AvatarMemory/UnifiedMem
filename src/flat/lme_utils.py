import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize

from src.flat.agentic_memory_system import FlexibleEmbeddingRetriever


class Retriever(FlexibleEmbeddingRetriever):
    """
    Stateless retrieval master that computes embeddings on-the-fly for each query.
    Inherits encoding methods from FlexibleEmbeddingRetriever but uses a different workflow.
    """
    def __init__(self, args, gpu_id):
        self.args = args
        
        # Map retriever names to model_type and model_name for FlexibleEmbeddingRetriever
        retriever_mapping = {
            'flat-bm25': ('bm25', 'bm25'),
            'flat-contriever': ('contriever', 'facebook/contriever'),
            'flat-stella': ('stella', 'dunzhang/stella_en_1.5B_v5'),
            'flat-gte': ('gte', 'Alibaba-NLP/gte-Qwen2-7B-instruct'),
            'flat-openai': ('openai', 'text-embedding-3-small'),
        }
        
        if args.retriever not in retriever_mapping:
            raise ValueError(f"Unsupported retriever: {args.retriever}")
        
        model_type, model_name = retriever_mapping[args.retriever]
        
        # Initialize parent class
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            device=device,
            cache_dir=args.cache_dir
        )

    def run_flat_retrieval(self, query, retriever, corpus):
        """
        Stateless retrieval: takes corpus as input and computes similarity scores.
        """
        if retriever == 'flat-bm25':
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(query.split(" "))
            return scores

        elif retriever in ['flat-contriever', 'flat-stella', 'flat-gte']:
            model2bsz = {'flat-contriever': 128, 'flat-stella': 64, 'flat-gte': 1}
            bsz = model2bsz[retriever]
            
            if retriever == 'flat-contriever':
                # Use inherited encoding method
                query_embedding = self._encode_contriever([query], batch_size=bsz)[0]
                doc_embeddings = self._encode_contriever(corpus, batch_size=bsz)
                scores = query_embedding @ doc_embeddings.T
                
            elif retriever == 'flat-stella':
                # Use inherited encoding method
                query_embedding = self._encode_stella([query], batch_size=bsz)[0]
                doc_embeddings = self._encode_stella(corpus, batch_size=bsz)
                scores = torch.tensor(query_embedding @ doc_embeddings.T)

            elif retriever == 'flat-gte':
                # Use inherited encoding methods with separate query/document encoding
                query_embedding = self._encode_gte_query(query)
                doc_embeddings = self._encode_gte_documents(corpus, batch_size=bsz)
                scores = query_embedding @ doc_embeddings.T
            
            return scores

        elif retriever == 'flat-openai':
            # Use inherited OpenAI encoding method
            query_embedding = self._encode_openai([query], batch_size=1)[0]
            doc_embeddings = self._encode_openai(corpus, batch_size=100)
            scores = cosine_similarity([query_embedding], doc_embeddings)[0]
            return scores

        else:
            raise NotImplementedError(f"Retriever {retriever} not implemented")
        

def fetch_expansion_from_cache(index_expansion_result_cache, cur_sess_id):
    """
    Fetch expansion(s) from cache. Can handle both single cache dict or list of cache dicts.
    
    Args:
        index_expansion_result_cache: Either a single dict or a list of dicts
        cur_sess_id: Session ID to look up
        
    Returns:
        If single cache: list or None (original behavior)
        If multiple caches: dict mapping expansion_type -> expansion_value
    """
    # Handle single cache (backward compatibility)
    if isinstance(index_expansion_result_cache, dict):
        processed_id = cur_sess_id
        try:
            cur_expansion = index_expansion_result_cache[processed_id]
        except:
            # failure to generate the expansion; index_expansion_result_cache could also contain None values.
            raise KeyError(f"Session ID {processed_id} not found in index_expansion_result_cache.")
        if type(cur_expansion) == str:
            cur_expansion = [cur_expansion]
        return cur_expansion
    
    # Handle multiple caches - return dict of expansions by type
    elif isinstance(index_expansion_result_cache, list):
        processed_id = cur_sess_id
        expansions_dict = {}
        
        for cache_dict in index_expansion_result_cache:
            try:
                cur_expansion = cache_dict[processed_id]
            except:
                raise KeyError(f"Session ID {processed_id} not found in index_expansion_result_cache.")
            
            if type(cur_expansion) == str:
                cur_expansion = [cur_expansion]
            
            expansions_dict[id(cache_dict)] = cur_expansion
        
        return expansions_dict
    else:
        raise ValueError(f"index_expansion_result_cache must be dict or list, got {type(index_expansion_result_cache)}")

    

def resolve_expansion(expansion_type, resolution_strategy, 
                      existing_corpus, existing_corpus_ids, existing_corpus_timestamps, existing_corpus_type,
                      cur_item_expansions, cur_sess_id, ts):
    """
    Resolve a single expansion type.
    
    Args:
        expansion_type: Type of expansion ('session-summ', 'session-keyphrase', 'session-userfact', etc.)
        resolution_strategy: How to merge expansions ('separate', 'merge', 'merge_raw', 'none')
        existing_corpus: List of corpus texts
        existing_corpus_ids: List of corpus IDs
        existing_corpus_timestamps: List of corpus timestamps
        existing_corpus_type: List of strings indicating type of each corpus item ('original', 'session-summ', etc.)
        cur_item_expansions: Expansion content for current item
        cur_sess_id: Current session ID
        ts: Timestamp
        
    Returns:
        Updated corpus, corpus_ids, corpus_timestamps, corpus_type
    """
    # preprocess and split expansion, if applicable
    if expansion_type == 'session-summ':
        if cur_item_expansions is None or cur_item_expansions == ['']:
            cur_item_expansions = []            
        assert len(cur_item_expansions) <= 1
    elif expansion_type == 'session-keyphrase' or expansion_type == 'turn-keyphrase':
        if cur_item_expansions is None or cur_item_expansions == ['']:
            cur_item_expansions = []
        assert len(cur_item_expansions) <= 1
    elif expansion_type == 'session-userfact' or expansion_type == 'turn-userfact':
        if cur_item_expansions is None or cur_item_expansions == ['']:
            # For failed expansions, we treat it as if the expansion is an empty string
            cur_item_expansions = []
        cur_item_expansions = [str(x) for x in cur_item_expansions]
    else:
        raise NotImplementedError

    # normalize expansions by trimming whitespace and dropping empty strings
    normalized_expansions = []
    for expansion_item in cur_item_expansions:
        cleaned_item = str(expansion_item).strip()
        if cleaned_item:
            normalized_expansions.append(cleaned_item)

    if expansion_type == 'session-userfact' or expansion_type == 'turn-userfact':
        if 'split' not in resolution_strategy:
            if normalized_expansions:
                normalized_expansions = [' '.join(normalized_expansions)]
            else:
                normalized_expansions = []
    elif expansion_type == 'session-summ':
        assert len(normalized_expansions) <= 1
        if 'split' in resolution_strategy:
            normalized_expansions = sent_tokenize(normalized_expansions[0]) if normalized_expansions else []
    elif expansion_type == 'session-keyphrase' or expansion_type == 'turn-keyphrase':
        assert len(normalized_expansions) <= 1
        if 'split' in resolution_strategy:
            normalized_expansions = [x.strip() for x in normalized_expansions[0].split(';')] if normalized_expansions else []
    else:
        raise NotImplementedError
    assert isinstance(normalized_expansions, list)

    # merge expansion with the main items
    # Support 'separate', 'none', 'merge', 'merge_raw' modes
    if resolution_strategy in ['separate', 'merge', 'merge_raw']:
        existing_corpus += normalized_expansions
        existing_corpus_ids += [cur_sess_id for _ in normalized_expansions]
        existing_corpus_timestamps += [ts for _ in normalized_expansions]
        existing_corpus_type += [expansion_type for _ in normalized_expansions]  # Track the expansion type
    elif resolution_strategy == 'none':
        # Don't add expansions
        pass
    else:
        raise NotImplementedError(f"Resolution strategy '{resolution_strategy}' not supported")
    
    return existing_corpus, existing_corpus_ids, existing_corpus_timestamps, existing_corpus_type


def resolve_multiple_expansions(expansion_types, resolution_strategy,
                                existing_corpus, existing_corpus_ids, existing_corpus_timestamps, existing_corpus_type,
                                expansions_dict, cur_sess_id, ts):
    """
    Resolve multiple expansion types for a single session/turn.
    
    Args:
        expansion_types: List of expansion type strings (e.g., ['session-summ', 'session-userfact'])
        resolution_strategy: How to merge expansions ('separate', 'merge', 'merge_raw', 'none')
        existing_corpus, existing_corpus_ids, existing_corpus_timestamps, existing_corpus_type: Current corpus state
        expansions_dict: Dict mapping expansion type to expansion value
        cur_sess_id: Current session/turn ID
        ts: Timestamp
        
    Returns:
        Updated corpus, corpus_ids, corpus_timestamps, corpus_type
    """
    # Apply each expansion type sequentially
    for expansion_type in expansion_types:
        # Get the expansion for this type (may be None if missing)
        cur_item_expansions = expansions_dict.get(expansion_type)
        
        # Apply this expansion
        existing_corpus, existing_corpus_ids, existing_corpus_timestamps, existing_corpus_type = resolve_expansion(
            expansion_type, resolution_strategy,
            existing_corpus, existing_corpus_ids, existing_corpus_timestamps, existing_corpus_type,
            cur_item_expansions, cur_sess_id, ts
        )

    # Merge the expansions of the same session if using merge mode
    if 'merge' in resolution_strategy:
        if resolution_strategy == 'merge_raw':
            # In merge_raw mode, we also need to include the original session text
            existing_indices = [i for i, (cid, ctype) in enumerate(zip(existing_corpus_ids, existing_corpus_type)) if cid == cur_sess_id]
            assert len(existing_indices) <= 4, f"Expected at most 4 items (original + 3 expansions), got {len(existing_indices)}"
        else:
            # Find all indices of the current session ID, exclude the original item
            existing_indices = [i for i, (cid, ctype) in enumerate(zip(existing_corpus_ids, existing_corpus_type)) if cid == cur_sess_id and ctype != 'original']
            assert len(existing_indices) <= 3, f"Expected at most 3 expansion items, got {len(existing_indices)}"

        if existing_indices:
            # Merge all expansions into one item
            merged_expansion = " ".join([existing_corpus[i] for i in existing_indices])

            # Remove individual expansion items
            for i in sorted(existing_indices, reverse=True):
                del existing_corpus[i]
                del existing_corpus_ids[i]
                del existing_corpus_timestamps[i]
                del existing_corpus_type[i]
            
            # Add back the merged item
            existing_corpus.append(merged_expansion)
            existing_corpus_ids.append(cur_sess_id)
            existing_corpus_timestamps.append(ts)
            existing_corpus_type.append(f'merged-{"_".join(expansion_types)}')
    
    return existing_corpus, existing_corpus_ids, existing_corpus_timestamps, existing_corpus_type
