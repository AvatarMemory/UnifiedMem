"""
Use original offline Longmemeval index expansions for memory note
contain userfact, summary keywords
"""

from typing import List, Dict, Optional, Literal, Any, Union
import json
import json_repair
import datetime
from pydantic import BaseModel, Field
import pickle
import os
import numpy as np
import uuid
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from openai import OpenAI
import tiktoken
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

from src.flat.llm_controller import LLMController
from src import config as cfg


class AMSResponse(BaseModel):
    action: str = Field(..., description="The action taken by the memory system. Choices are 'create', 'update', or 'pass'.")
    update_ids: List[str] = Field(..., description="List of IDs of existing memories to update, empty list if not updating")

class UpdatedMemory(BaseModel):
    id: str = Field(..., description="The ID of the existing memory being updated.")
    summary: str = Field(..., description="The comprehensive, updated summary.")
    keywords: List[str] = Field(..., description="The updated list of keywords.")
    facts: List[str] = Field(..., description="The updated list of distinct facts.")

class UpdateMemoryResponse(BaseModel):
    updates: List[UpdatedMemory] = Field(..., description="List of updated memories")

class MemoryOperation(BaseModel):
    """Records a single memory operation"""
    operation_id: str
    operation_index: int  # Sequential index in the trajectory (0-based)
    timestamp: str
    action: str  # 'created', 'merged', 'no_action'
    new_memory_id: str
    new_memory_content: str
    merge_into_id: Optional[str] = None
    neighbor_ids: List[str] = []
    resulting_memory_id: Optional[str] = None
    llm_reasoning: Optional[str] = None

class LinkResponse(BaseModel):
    related_memory_ids: List[str] = Field(..., description="List of IDs of related memories linked to the new memory.")

class MemoryNote:
    def __init__(self, 
                 content: str,
                 keywords: str,
                 summary: str,
                 facts: str,
                 id: str,
                 timestamp: Optional[str] = None,
                 llm_controller: Optional[LLMController] = None): 
        
        self.content = content

        self.id = id or str(uuid.uuid4())
        self.timestamp = timestamp or 'not specified'

        self.summary = summary or ''
        if isinstance(keywords, str): keywords = [keywords]

        if isinstance(facts, dict): facts = []  # TODO inspect why facts is dict in rare occasions
        assert isinstance(facts, list) and all(isinstance(fact, str) for fact in facts), f"Facts provided by LME should be a list of strings, instead got {facts}"
        assert isinstance(keywords, list) and len(keywords) >= 1, f"Keywords provided by LME should be a list of strings, instead got {keywords}"
        if len(keywords) > 1: keywords = ['; '.join(keywords)]  # join multiple keywords into one string
        self.facts = facts or []
        self.keywords = keywords[0] or ''

        # convert facts from list of strings to a single string
        # NOTE only applied to LME's index expansions
        self.facts = [fact.rstrip('.') for fact in self.facts]
        self.facts = '; '.join(self.facts) if self.facts else ''

        # NOTE for now both keywords and facts are a single string with ';' separating different items
        
        if isinstance(self.summary, list):
            assert len(self.summary) == 1, "Summary should be a single string."
            self.summary = self.summary[0]

        self.all_ids = [self.id]  # we need this to record all session_id as merging may replace current id, while we want to get the session_id to compute retrieval metrics
        # Track original content and timestamp for each session ID
        self.id_to_content = {self.id: content}  # Maps session_id -> original content
        self.id_to_timestamp = {self.id: timestamp}  # Maps session_id -> original timestamp

        self.related_memories = []

    def __repr__(self):
        return f"MemoryNote(id={self.id}, timestamp={self.timestamp}, keywords={self.keywords}, summary={self.summary}, userfacts={self.facts})"


class FlexibleEmbeddingRetriever:
    """
    Extended retriever supporting multiple embedding models.
    
    Supported models:
    - SentenceTransformer models: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', etc.
    - Contriever: 'facebook/contriever'
    - Stella: 'dunzhang/stella_en_1.5B_v5'
    - GTE: 'Alibaba-NLP/gte-Qwen2-7B-instruct'
    - OpenAI: 'text-embedding-3-small', 'text-embedding-3-large', etc.
    - BM25: 'bm25' (sparse retrieval)
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 model_type: Literal['sentence-transformer', 'contriever', 'stella', 'gte', 'openai', 'bm25'] = 'sentence-transformer',
                 device: str = None,
                 cache_dir: str = None):
        """
        Initialize the flexible retriever.
        
        Args:
            model_name: Name or path of the model
            model_type: Type of embedding model
            device: Device to run the model on (cuda/cpu)
            cache_dir: Cache directory for model files
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # memory_id -> document index
        self.memory_ids = {}  # document index -> memory_id
        self.bm25 = None  # BM25 index
        self.valid_mask = []  # Tracks whether each document contains non-empty content
        
        # OpenAI-specific attributes
        self.openai_client = None  # For OpenAI
        self.openai_model_name = model_name if model_type == 'openai' else None
        
        # Initialize model based on type
        self._init_model()
    
    def _init_model(self):
        """Initialize the embedding model based on type"""
        if self.model_type == 'sentence-transformer':
            # Standard SentenceTransformer models
            self.model = SentenceTransformer(self.model_name)
            if self.device == 'cuda':
                self.model = self.model.to(self.device)
        
        elif self.model_type == 'contriever':
            # Facebook Contriever
            self.tokenizer = AutoTokenizer.from_pretrained(
                'facebook/contriever',
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                'facebook/contriever',
                cache_dir=self.cache_dir
            ).to(self.device)
            self.model.eval()
        
        elif self.model_type == 'stella':
            # Stella model
            model_dir = self.cache_dir + "/dunzhang_stella_en_1.5B_v5"
            self.vector_dim = 1024
            vector_linear_directory = f"2_Dense_{self.vector_dim}"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)
            self.model.eval()
            
            # Load vector linear layer
            self.vector_linear = torch.nn.Linear(
                in_features=self.model.config.hidden_size, 
                out_features=self.vector_dim
            ).to(self.device)
            
            if os.path.exists(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")):
                vector_linear_dict = {
                    k.replace("linear.", ""): v for k, v in
                    torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
                }
                self.vector_linear.load_state_dict(vector_linear_dict)
            self.vector_linear.to(self.device)
        
        elif self.model_type == 'gte':
            # GTE model
            self.tokenizer = AutoTokenizer.from_pretrained(
                'Alibaba-NLP/gte-Qwen2-7B-instruct',
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                'Alibaba-NLP/gte-Qwen2-7B-instruct',
                trust_remote_code=True,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.model.eval()
        
        elif self.model_type == 'bm25':
            # BM25 doesn't need model initialization
            # Index will be built when documents are added
            self.model = None
        
        elif self.model_type == 'openai':
            # OpenAI embeddings - initialize client
            embedding_key = cfg.get_embedding_api_key(None)
            if not embedding_key:
                raise ValueError(
                    "No API key found for OpenAI embeddings. Set EMBEDDING_API_KEY or OPENAI_API_KEY or LLM_API_KEY in the environment"
                )

            embedding_base = cfg.get_embedding_base_url(None)

            # Initialize OpenAI client for embeddings
            self.openai_client = OpenAI(api_key=embedding_key, base_url=embedding_base or None)
            if not self.openai_model_name:
                self.openai_model_name = cfg.get_embedding_model('text-embedding-3-small')
            self.model = None  # No local model needed
        
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def _encode_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Encode texts using SentenceTransformer"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def _encode_contriever(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Encode texts using Contriever"""
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embeddings = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu().numpy()
                all_embeddings.append(embeddings.astype(np.float32))

        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, self.model.config.hidden_size), dtype=np.float32)
    
    def _encode_stella(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts using Stella"""
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                input_data = self.tokenizer(
                    batch, 
                    padding="longest", 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                )
                input_data = {k: v.to(self.device) for k, v in input_data.items()}
                attention_mask = input_data["attention_mask"]
                last_hidden_state = self.model(**input_data)[0]
                last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                embeddings = normalize(self.vector_linear(embeddings).cpu().numpy())
                all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _encode_gte_documents(self, texts: List[str], batch_size: int = 1) -> np.ndarray:
        """Encode corpus documents using GTE (matches DenseRetrievalMaster)."""
        def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_dict = self.tokenizer(
                    batch,
                    max_length=8192,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, self.model.config.hidden_size), dtype=np.float32)

    def _encode_gte_query(self, text: str, task: str = None) -> np.ndarray:
        """Encode a query with task instruction using GTE."""
        if task is None:
            task = 'Given a query about personal information, retrieve relevant chat history that answer the query.'

        def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        query_text = f'Instruction: {task}\nQuery: {text}'
        with torch.no_grad():
            batch_dict = self.tokenizer(
                [query_text],
                max_length=8192,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings[0].detach().cpu().numpy().astype(np.float32)
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_fixed(10),
        stop=stop_after_attempt(10),
        reraise=True
    )
    def _encode_openai(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Encode texts using OpenAI embeddings API."""
        MAX_TOKENS = 8191
        
        all_embeddings = []
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx: batch_idx + batch_size]
            batch_texts = [str(doc).strip() if doc else " " for doc in batch_texts]
            
            # Truncate to 8191 tokens using tiktoken
            encoding = tiktoken.get_encoding('cl100k_base')
            for i in range(len(batch_texts)):
                token_ids = encoding.encode(batch_texts[i])
                if len(token_ids) > MAX_TOKENS:
                    token_ids = token_ids[:MAX_TOKENS]
                    batch_texts[i] = encoding.decode(token_ids)
            
            # Retry logic for rate limit errors
            batch_embeddings = []
            for _ in range(3):  # try 3 times
                try:
                    batch_response = self.openai_client.embeddings.create(
                        input=batch_texts,
                        model=self.openai_model_name
                    )
                    batch_embeddings = [np.array(item.embedding, dtype=np.float32) for item in batch_response.data]
                    break
                except Exception as e:
                    print(f"Error encoding batch with OpenAI: {str(e)}")
            
            if not batch_embeddings:
                raise RuntimeError("Failed to get embeddings from OpenAI after multiple attempts")
            
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings using the selected model.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings
        """
        if self.model_type == 'sentence-transformer':
            return self._encode_sentence_transformer(texts)
        elif self.model_type == 'contriever':
            return self._encode_contriever(texts)
        elif self.model_type == 'stella':
            return self._encode_stella(texts)
        elif self.model_type == 'gte':
            return self._encode_gte_documents(texts)
        elif self.model_type == 'openai':
            return self._encode_openai(texts)
        elif self.model_type == 'bm25':
            # BM25 doesn't use embeddings
            return None
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def add_documents(self, documents: List[str], memory_ids: List[str]):
        """Add documents to the retriever. Each document corresponds to a MemoryNote's content."""
        # Normalize documents, track emptiness, and replace empties with a placeholder to keep indices aligned
        EMPTY_DOC_PLACEHOLDER = "[empty]"
        processed_documents = []
        new_valid_mask = []
        for doc in documents:
            doc_str = "" if doc is None else str(doc)
            is_valid = bool(doc_str.strip())
            new_valid_mask.append(is_valid)
            processed_documents.append(doc_str if is_valid else EMPTY_DOC_PLACEHOLDER)
        
        if not self.corpus:
            self.corpus = processed_documents
            self.document_ids = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}  # memory_id to local index
            self.memory_ids = {idx: memory_id for idx, memory_id in enumerate(memory_ids)}  # local index to memory_id (session_id)
            self.valid_mask = new_valid_mask.copy()
            if self.model_type == 'bm25':
                tokenized_corpus = [doc.lower().split(" ") for doc in processed_documents]
                self.bm25 = BM25Okapi(tokenized_corpus)
            elif self.model_type == 'gte':
                self.embeddings = self._encode_gte_documents(processed_documents)
            else:
                self.embeddings = self.encode(processed_documents)
        else:
            # Append new documents
            start_idx = len(self.corpus)
            for idx, memory_id in enumerate(memory_ids):
                self.memory_ids[start_idx + idx] = memory_id
                self.document_ids[memory_id] = start_idx + idx
            self.corpus.extend(processed_documents)
            self.valid_mask.extend(new_valid_mask)
            if self.model_type == 'bm25':
                # Rebuild BM25 index with all documents
                tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
            elif self.model_type == 'gte':
                new_embeddings = self._encode_gte_documents(processed_documents)
                if self.embeddings is None or not len(self.embeddings):
                    self.embeddings = new_embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])
            else:
                new_embeddings = self.encode(processed_documents)
                if self.embeddings is None:
                    self.embeddings = new_embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def remove_documents(self, memory_ids: List[str]):
        """Remove documents by memory id."""
        for memory_id in memory_ids:
            idx = self.document_ids[memory_id]
            # Remove from corpus and embeddings
            self.corpus.pop(idx)
            self.valid_mask.pop(idx)
            if self.embeddings is not None:
                self.embeddings = np.delete(self.embeddings, idx, axis=0)
            
            # Update document_ids mapping
            del self.document_ids[memory_id]
            # Adjust indices of remaining documents
            for mid, didx in self.document_ids.items():
                if didx > idx:
                    self.document_ids[mid] = didx - 1
            
            del self.memory_ids[idx]
            for didx in list(self.memory_ids.keys()):
                if didx > idx:
                    self.memory_ids[didx - 1] = self.memory_ids.pop(didx)

        if self.model_type == 'bm25':
            # Rebuild BM25 index
            tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(self, query: str) -> List[int]:
        """
        Search for similar documents using cosine similarity or BM25.
        
        Args:
            query: Query text
            
        Returns:
            Similarity scores for all documents
        """
        if not self.corpus:
            return []

        EMPTY_DOC_PLACEHOLDER = "[empty]"
        processed_query = query if query.strip() else EMPTY_DOC_PLACEHOLDER

        if self.model_type == 'bm25':
            tokenized_query = processed_query.lower().split()
            similarities = self.bm25.get_scores(tokenized_query)
            return list(similarities)

        if self.model_type == 'gte':
            query_embedding = self._encode_gte_query(processed_query)
            similarities = query_embedding @ self.embeddings.T
            return list(similarities)

        query_embedding = self.encode([processed_query])[0]

        if self.model_type == 'contriever':
            similarities = self.embeddings @ query_embedding
            return list(similarities)

        # stella, sentence-transformer, etc. already return normalized vectors
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        return list(similarities)

    @classmethod
    def load_from_local_memory(cls, memories_dict: Dict[str, 'MemoryNote'], model_name: str, model_type: str, 
                              device: str = None, cache_dir: str = None, component: str = None) -> 'FlexibleEmbeddingRetriever':
        """
        Load retriever state from memory objects.
        
        Args:
            memories_dict: Dictionary of {memory_id: MemoryNote}
            model_name: Name of the embedding model
            model_type: Type of embedding model
            device: Device to run on
            cache_dir: Cache directory for models
            component: Which component to load ('topic', 'summary', 'keywords', 'facts', 'events', 'content', or None for full)
            
        Returns:
            Initialized retriever with memories
        """        
        # Create and initialize retriever
        retriever = cls(model_name=model_name, model_type=model_type, device=device, cache_dir=cache_dir)
        
        # Extract memory IDs and format documents based on component
        memory_ids = list(memories_dict.keys())
        documents = []
        for mem_id in memory_ids:
            mem = memories_dict[mem_id]
            # Format based on component (for 'separate' mode) or full (for 'merge' mode)
            if component == 'topic':
                doc = "Topic: \n" + mem.topic
            elif component == 'summary':
                doc = "Summary: \n" + mem.summary
            elif component == 'keywords':
                doc = "Keywords: \n" + ", ".join(mem.keywords)
            elif component == 'facts':
                doc = "Facts: \n" + ", ".join(mem.facts)
            elif component == 'events':
                doc = "Events: \n" + ", ".join(mem.events)
            elif component == 'content':
                doc = "Content: \n" + mem.content
            else:
                # Full format for 'merge' mode
                doc = "Topic: \n" + mem.topic + "\n\nKeywords: \n" + ", ".join(mem.keywords) + "\n\nFacts: \n" + ", ".join(mem.facts) + "\n\nEvents: \n" + ", ".join(mem.events) + "\n\nSummary: \n" + mem.summary + "\n\nContent: \n" + mem.content
            documents.append(doc)
        
        retriever.add_documents(documents, memory_ids)
        return retriever

    def save(self, retriever_cache_file: str):
        """Save retriever state to disk (embeddings will be rebuilt on load)"""
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids,
            'memory_ids': self.memory_ids,
            'valid_mask': self.valid_mask,
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved retriever state to {retriever_cache_file}")

    def load(self, retriever_cache_file: str):
        """Load retriever state from disk and rebuild embeddings/index"""
        print(f"Loading retriever from {retriever_cache_file}")
        
        if not os.path.exists(retriever_cache_file):
            print(f"Cache file not found: {retriever_cache_file}")
            return self
        
        # Load corpus and mappings
        print(f"Loading corpus from {retriever_cache_file}")
        with open(retriever_cache_file, 'rb') as f:
            state = pickle.load(f)
            self.corpus = state['corpus']
            self.document_ids = state['document_ids']
            self.memory_ids = state['memory_ids']
            self.valid_mask = state.get('valid_mask') or [doc != "[empty]" for doc in self.corpus]
            print(f"Loaded corpus with {len(self.corpus)} documents")
        
        # Rebuild embeddings or BM25 index based on model type
        if self.model_type == 'bm25':
            if self.corpus:
                print(f"Rebuilding BM25 index from corpus")
                tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            if self.corpus:
                print(f"Rebuilding embeddings from corpus (this may take a while)...")
                self.embeddings = self.encode(self.corpus)
                print(f"Rebuilt embeddings with shape: {self.embeddings.shape}")
        
        return self
    
    def cleanup(self):
        """Clean up GPU resources"""
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'to'):
                self.model.to('cpu')  # Move to CPU first
            del self.model
            self.model = None
        
        if hasattr(self, 'vector_linear') and self.vector_linear is not None:
            if hasattr(self.vector_linear, 'to'):
                self.vector_linear.to('cpu')
            del self.vector_linear
            self.vector_linear = None
        
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        
        # Clean up OpenAI client
        if hasattr(self, 'openai_client') and self.openai_client is not None:
            del self.openai_client
            self.openai_client = None
        
        # Clear embeddings from memory
        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
        self.valid_mask = []
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


class AgenticMemorySystem:
    """Memory management system with embedding-based retrieval - LME version"""
    def __init__(self,
                 retrieve_method: str = 'merge_raw',  # 'merge', 'separate', 'merge_raw'
                 retriever_name: str = 'all-MiniLM-L6-v2',
                 retriever_type: str = 'sentence-transformer',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 keep_update_note: bool = False,
                 enable_link: bool = False,
                 enable_update: bool = True,
                 use_raw_session_as_key: bool = True,):

        self.memories = {}  # id -> MemoryNote
        self.retrieve_method = retrieve_method
        self.operation_trajectory = []  # List[MemoryOperation] - tracks all memory operations

        if retrieve_method == 'merge_raw':
            self.retriever = FlexibleEmbeddingRetriever(retriever_name, retriever_type, device=device, cache_dir=cache_dir)
        elif retrieve_method == 'merge':
            assert use_raw_session_as_key is True, "In 'merge' retrieve_method, use_raw_session_as_key must be True."
            self.retriever = {
                'merged': FlexibleEmbeddingRetriever(retriever_name, retriever_type, device=device, cache_dir=cache_dir),
                'content': FlexibleEmbeddingRetriever(retriever_name, retriever_type, device=device, cache_dir=cache_dir),
            }
        elif retrieve_method == 'separate':
            self.retriever = {
                "summary": FlexibleEmbeddingRetriever(retriever_name, retriever_type, device=device, cache_dir=cache_dir),
                "keywords": FlexibleEmbeddingRetriever(retriever_name, retriever_type, device=device, cache_dir=cache_dir),
                "facts": FlexibleEmbeddingRetriever(retriever_name, retriever_type, device=device, cache_dir=cache_dir),
            }
            if use_raw_session_as_key:
                self.retriever["content"] = FlexibleEmbeddingRetriever(retriever_name, retriever_type, device=device, cache_dir=cache_dir)
        else:
            raise NotImplementedError(f"Unknown retrieve_method: {retrieve_method}")
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, base_url=base_url)

        self.keep_update_note = keep_update_note  # whether to keep the new note after updating existing memories
        self.enable_link = enable_link  # whether to use LLM to link related memories
        self.enable_update = enable_update  # whether to update memory
        self.use_raw_session_as_key = use_raw_session_as_key  # whether to use raw session content as key in separate retriever or in merge
        
        self.action_system_prompt = '''
            You are an AI memory update agent responsible for managing and evolving a dynamic knowledge base over time.

            Your task is to analyze a new incoming memory note in relation to several of its closest existing memories, and determine how to integrate it appropriately.

            The new memory:
            {content}
            - Keywords: {keywords}
            - Facts: {facts}

            The associated neighbor memories (with their exact IDs):
            {nearest_neighbors_memories}

            Your goal is to maintain both accuracy and temporal consistency of knowledge. 
            When comparing the new memory to existing ones, carefully consider:
            - **Semantic similarity:** Are they about the same topic or entity?
            - **Temporal evolution:** Even if the topic is similar, does this memory reflect a *different timepoint*, *updated fact*, or *changed state* (e.g., an opinion, event, or condition evolving over time)?
            - **Knowledge persistence:** Some facts remain stable, while others evolve—ensure that merging does not overwrite distinct time-based information.

            You must choose one of the following actions:
            1. **"create"** — Add this memory as a separate entry if it introduces new, time-specific, or distinct information (e.g., an updated event, changed belief, or evolution of context).
            2. **"update"** — Update the existing memories if the new memory has certain connections with existing memories but does not fully overlap, and the existing memories should be refined.
            3. **"pass"** — Take no action if the new memory is fully redundant with an existing one (same meaning and timeframe), or it does not contain any meaningful information.

            CRITICAL: If choosing "update", you MUST provide valid memory IDs from the neighbor memories listed above. Do NOT invent or modify IDs.

            Return your decision strictly in JSON format:
            {{
                "action": "<create | update | pass>",
                "update_ids": ["<id1>", "<id2>", "..."]
            }}
            
            Note: update_ids must be an empty array [] if action is "create" or "pass".
            '''
        
        self.merge_system_prompt = '''
            You are an AI memory update agent. Your job in THIS step is ONLY to update the content of existing memories using complementary information from a new memory. You are NOT deciding whether to create or delete memories.

---

Existing memories (these are the ones that must be updated, in the given order):
{existing_memories}

Each existing memory will have an "id" and its current:
- summary
- keywords
- facts

New memory:
- Summary: {new_summary}
- Keywords: {new_keywords}
- Facts: {new_facts}

---

Your task:

1. For each existing memory, integrate any complementary information from the new memory.
   - Preserve important existing information.
   - Add relevant new facts and keywords if they relate to that memory.
   - Optionally refine or rephrase the summary to keep it clear, concise, and up to date.

2. Do NOT:
   - Invent new memories or IDs.
   - Drop existing important facts unless they are clearly contradicted by the new memory.
   - Add information unrelated to that memory.

3. Each updated memory should remain self-contained and understandable on its own.

---

Output format:

Return your result in strict JSON format compatible with this schema:

{{
  "updates": [
    {{
      "id": "<id of existing memory 1>",
      "summary": "<updated summary for memory 1>",
      "keywords": ["<keyword1>", "<keyword2>", "..."],
      "facts": ["<fact1>", "<fact2>", "..."]
    }},
    {{
      "id": "<id of existing memory 2>",
      "summary": "<updated summary for memory 2>",
      "keywords": ["<keyword1>", "<keyword2>", "..."],
      "facts": ["<fact1>", "<fact2>", "..."]
    }}
  ]
}}

Follow these rules:
- Use ONLY valid JSON (no comments or trailing commas).
- Do NOT include any fields other than "updates", "id", "summary", "keywords", and "facts".
- The "updates" array must include one object for each existing memory, in the same order they were provided.
- IMPORTANT: You MUST use the EXACT memory IDs provided above. Do not modify or create new IDs.
            '''
        
        self.memory_str = '''
            Memory ID: {id}
                Timestamp: {timestamp}
                Keywords: {keywords}
                Facts: {facts}
                Summary: {summary}
                {content}

            '''
        
        self.link_system_prompt = '''
            You are an AI memory linking agent. Your task is to identify and link related existing memories to a new incoming memory.
            Analyze the the new memory note according to original content, keyworkds, facts, and summary, also with their several nearest neighbors memory.

            The new memory:
            {content}
            - Keywords: {keywords}
            - Facts: {facts}
            - Summary: {summary}

            The associated neighbor memories (with their exact IDs):
            {nearest_neighbors_memories}

            Based on this information, identify which of the neighbor memories are related to the new memory. Related memories are those that share significant thematic, factual, or contextual overlap with the new memory.
            Return your decision strictly in JSON format:
            {{
                "related_memory_ids": ["<id1>", "<id2>", "..."]
            }}
            '''
        

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)

        note, action, operation_record = self.process_memory(note)  # NOTE
        
        # Set the operation index before recording
        operation_record.operation_index = len(self.operation_trajectory)
        
        # Record the operation in trajectory
        self.operation_trajectory.append(operation_record)
        
        if action == 'no_action':
            return
        elif action == 'updated' and not self.keep_update_note:
            return
        
        assert note.id not in self.memories, "Since we always create new notes and the merged note will have new note's id, the note id should not exist in memories yet."
        self.memories[note.id] = note

        if self.retrieve_method == 'merge_raw':
            if self.use_raw_session_as_key:
                doc = ["Keywords: \n" + note.keywords + "\n\nFacts: \n" + note.facts + "\n\nSummary: \n" + note.summary + "\n\nContent: \n" + note.content]
            else:
                doc = ["Keywords: \n" + note.keywords + "\n\nFacts: \n" + note.facts + "\n\nSummary: \n" + note.summary]
            self.retriever.add_documents(doc, [note.id])
        elif self.retrieve_method == 'separate':
            # add to each retriever - wrap each field in a list
            self.retriever["summary"].add_documents([note.summary], [note.id])
            self.retriever["keywords"].add_documents([note.keywords], [note.id])
            self.retriever["facts"].add_documents([note.facts], [note.id])
            if self.use_raw_session_as_key:
                self.retriever["content"].add_documents([note.content], [note.id])
        elif self.retrieve_method == 'merge':
            doc = ["Keywords: \n" + note.keywords + "\n\nFacts: \n" + note.facts + "\n\nSummary: \n" + note.summary]
            self.retriever["merged"].add_documents(doc, [note.id])
            self.retriever["content"].add_documents([note.content], [note.id])
        

    def _link_memory(self, note: MemoryNote, neighbor_memory_str: str, neighbor_ids: list, max_retries: int = 5) -> list:
        """Link related memories to the new note"""
        if not neighbor_memory_str:
            return []
        
        prompt_memory = self.link_system_prompt.format(
            content=f'- Content: {self.memories[i].content}' if self.use_raw_session_as_key else '',
            keywords=note.keywords,
            facts=note.facts,
            summary=note.summary,
            nearest_neighbors_memories=neighbor_memory_str
        )
        for retry_count in range(max_retries):
            response = self.llm_controller.llm.get_completion(
                prompt_memory,
                response_format={"type": "json_schema", "json_schema": {
                            "name": "link_response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "related_memory_ids": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                },
                                "required": ["related_memory_ids"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }},
                extra_body=LinkResponse,
            )
            # Parse JSON response with error handling
            try:
                response_json = json_repair.loads(response)
                # Validate that all returned IDs exist in neighbor_ids
                invalid_ids = [mem_id for mem_id in response_json["related_memory_ids"] if mem_id not in neighbor_ids]
                if invalid_ids:
                    print(f"Warning: LLM returned invalid memory IDs '{invalid_ids}' not in neighbor memories. Retry {retry_count + 1}/{max_retries}")
                    if retry_count == 1:
                        prompt_memory += f"\n\nAvailable memory IDs are: {neighbor_ids}. Please choose valid IDs from the neighbor memories listed above."
                    continue
                return response_json["related_memory_ids"]
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing failed for link response: {e}. Retry {retry_count + 1}/{max_retries}")
                print(f"Raw response: {response}")
                continue

        print(f"Warning: Failed to link related memories after {max_retries} retries. No related memories linked.")
        return []


    def process_memory(self, note: MemoryNote, max_retries: int = 5) -> tuple:
        """Process a memory note and return (note, action, operation_record)"""        
        neighbor_memory, neighbor_ids = self.find_related_memories(note, k=5)
        if not neighbor_memory:
            neighbor_memory = "No related memories found."
        
        # Initialize operation record
        operation_id = str(uuid.uuid4())
        operation_time = datetime.datetime.now().isoformat()

        # Extract actual memory IDs from neighbor_ids (they are the first ID in each memory's all_ids list)
        # neighbor_ids contains flattened session IDs, but we need the current memory IDs shown in prompts
        # NOTE this maybe redundant since the current code does not add updated memory ids to exsisiting memories if choosing `update`
        actual_memory_ids = []
        seen = set()
        for session_id in neighbor_ids:
            # Find which memory this session_id belongs toz
            for mem_id, mem in self.memories.items():
                if session_id in mem.all_ids and mem.id not in seen:
                    actual_memory_ids.append(mem.id)
                    seen.add(mem.id)
                    break

        if self.enable_link:
            linked_memory_ids = self._link_memory(note, neighbor_memory, actual_memory_ids, max_retries)
            # linked_memory_ids = actual_memory_ids
            note.related_memories = linked_memory_ids

        if not self.enable_update:
            # If not enabling update, always create new memory
            operation_record = MemoryOperation(
                operation_id=operation_id,
                operation_index=0,  # Will be set in add_note
                timestamp=operation_time,
                action='created',
                new_memory_id=note.id,
                new_memory_content=note.content,
                neighbor_ids=neighbor_ids,
                resulting_memory_id=note.id,
                llm_reasoning="N/A - update disabled",
            )
            return note, 'created', operation_record
        
        # Retry loop to handle LLM hallucinating non-existent IDs
        for retry_count in range(max_retries):
            prompt_memory = self.action_system_prompt.format(
                content=f'- Content: {note.content}' if self.use_raw_session_as_key else '',
                keywords=note.keywords,
                facts=note.facts,
                timestamp=note.timestamp,
                nearest_neighbors_memories=neighbor_memory
            )
            # print("prompt_memory", prompt_memory)
            response = self.llm_controller.llm.get_completion(
                prompt_memory,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                    },
                                    "update_ids": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                },
                                "required": ["action", "update_ids"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }},
                extra_body=AMSResponse,
            )
            
            # Parse JSON response with error handling
            try:
                response_json = json_repair.loads(response)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing failed for action response: {e}. Retry {retry_count + 1}/{max_retries}")
                print(f"Raw response: {response}")
                continue

            if response_json["action"] == "create": # NOTE
                operation_record = MemoryOperation(
                    operation_id=operation_id,
                    operation_index=0,  # Will be set in add_note
                    timestamp=operation_time,
                    action='created',
                    new_memory_id=note.id,
                    new_memory_content=note.content,
                    neighbor_ids=neighbor_ids,
                    resulting_memory_id=note.id,
                    llm_reasoning=response,
                )
                return note, 'created', operation_record
            elif response_json["action"] == "update":
                update_memory_ids = response_json.get("update_ids", [])
                
                # Check if update_ids is provided and not empty
                if not update_memory_ids:
                    print(f"Warning: 'update' action chosen but no update_ids provided. Retry {retry_count + 1}/{max_retries}")
                    neighbor_memory += f"\n\n[ERROR] You chose 'update' action but did not provide any update_ids. Please provide valid memory IDs to update from: {neighbor_ids}"
                    continue
                
                # Check if the IDs exist in memories
                if any(update_id not in self.memories for update_id in update_memory_ids):
                    invalid_ids = [update_id for update_id in update_memory_ids if update_id not in self.memories]
                    print(f"Warning: LLM hallucinated non-existent memory IDs '{invalid_ids}'. Retry {retry_count + 1}/{max_retries}")
                    # Add feedback to prompt for next retry
                    neighbor_memory += f"\n\n[ERROR] The IDs '{invalid_ids}' do not exist. Available memory IDs are: {neighbor_ids}. Please choose valid IDs from the neighbor memories listed above."
                    continue

                # Build existing memories string for the merge prompt
                existing_memories_str = ""
                for mem_id in update_memory_ids:
                    mem = self.memories[mem_id]
                    existing_memories_str += f"""
Memory ID: {mem_id}
- Summary: {mem.summary}
- Keywords: {mem.keywords}
- Facts: {mem.facts}

"""
                
                # Call LLM to get the updated memory content
                merge_prompt = self.merge_system_prompt.format(
                    existing_memories=existing_memories_str,
                    new_summary=note.summary,
                    new_keywords=note.keywords,
                    new_facts=note.facts
                )
                
                merge_response = self.llm_controller.llm.get_completion(
                    merge_prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "update_response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "updates": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "summary": {"type": "string"},
                                            "keywords": {"type": "array", "items": {"type": "string"}},
                                            "facts": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["id", "summary", "keywords", "facts"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["updates"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }},
                    extra_body=UpdateMemoryResponse,
                )
                
                # Parse the merge response with robust handling
                try:
                    merge_response_json = json_repair.loads(merge_response)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parsing failed for merge response: {e}. Retry {retry_count + 1}/{max_retries}")
                    print(f"Raw merge response: {merge_response}")
                    continue
                
                # Handle key name mismatch: LLM might return "memories" instead of "updates"
                if "memories" in merge_response_json and "updates" not in merge_response_json:
                    merge_response_json["updates"] = merge_response_json.pop("memories")
                
                # Validate and fix the response structure
                updated_memories = merge_response_json.get("updates", [])
                if not updated_memories:
                    print(f"Warning: No updates in merge response. Retry {retry_count + 1}/{max_retries}")
                    continue
                
                # Validate that all returned IDs exist in the requested update_memory_ids
                valid_updates = []
                for updated_mem in updated_memories:
                    mem_id = updated_mem.get("id")
                    if mem_id not in update_memory_ids:
                        print(f"Warning: Merge response contains unexpected ID '{mem_id}', expected one of {update_memory_ids}. Skipping.")
                        continue
                    if mem_id not in self.memories:
                        print(f"Warning: Updated memory ID '{mem_id}' not found in memories. Skipping.")
                        continue
                    
                    # Ensure keywords and facts are lists
                    keywords = updated_mem.get("keywords", [])
                    facts = updated_mem.get("facts", [])
                    if isinstance(keywords, str):
                        keywords = [k.strip() for k in keywords.split(';') if k.strip()]
                    if isinstance(facts, str):
                        facts = [f.strip() for f in facts.split(';') if f.strip()]
                    
                    updated_mem["keywords"] = keywords
                    updated_mem["facts"] = facts
                    valid_updates.append(updated_mem)
                
                if not valid_updates:
                    print(f"Warning: No valid updates after filtering. Retry {retry_count + 1}/{max_retries}")
                    continue
                
                # Update each memory with the validated content
                for updated_mem in valid_updates:
                    mem_id = updated_mem.get("id")
                    existing_mem = self.memories[mem_id]
                    
                    # Update the memory fields
                    existing_mem.summary = updated_mem.get("summary", existing_mem.summary)
                    new_keywords = updated_mem.get("keywords", existing_mem.keywords)
                    new_facts = updated_mem.get("facts", existing_mem.facts)

                    existing_mem.keywords = '; '.join(new_keywords) if isinstance(new_keywords, list) else new_keywords
                    existing_mem.facts = '; '.join(new_facts) if isinstance(new_facts, list) else new_facts
                    
                    # Track the new note's session ID in the updated memory
                    # existing_mem.all_ids.append(note.id)
                    # existing_mem.id_to_content[note.id] = note.content
                    # existing_mem.id_to_timestamp[note.id] = note.timestamp
                    
                    # Update the retrievers
                    if self.retrieve_method == 'merge_raw':
                        # Remove old document and add updated one
                        self.retriever.remove_documents([mem_id])
                        if self.use_raw_session_as_key:
                            doc = ["Keywords: \n" + existing_mem.keywords + "\n\nFacts: \n" + existing_mem.facts + "\n\nSummary: \n" + existing_mem.summary + "\n\nContent: \n" + existing_mem.content]
                        else:
                            doc = ["Keywords: \n" + existing_mem.keywords + "\n\nFacts: \n" + existing_mem.facts + "\n\nSummary: \n" + existing_mem.summary]
                        self.retriever.add_documents(doc, [mem_id])
                    elif self.retrieve_method == 'separate':
                        # Update each component retriever
                        self.retriever["summary"].remove_documents([mem_id])
                        self.retriever["summary"].add_documents([existing_mem.summary], [mem_id])
                        self.retriever["keywords"].remove_documents([mem_id])
                        self.retriever["keywords"].add_documents([existing_mem.keywords], [mem_id])
                        self.retriever["facts"].remove_documents([mem_id])
                        self.retriever["facts"].add_documents([existing_mem.facts], [mem_id])
                        # Note: content doesn't change, so no need to update content retriever
                    elif self.retrieve_method == 'merge':
                        self.retriever["merged"].remove_documents([mem_id])
                        doc = ["Keywords: \n" + existing_mem.keywords + "\n\nFacts: \n" + existing_mem.facts + "\n\nSummary: \n" + existing_mem.summary]
                        self.retriever["merged"].add_documents(doc, [mem_id])
                        self.retriever["content"].remove_documents([mem_id])
                        self.retriever["content"].add_documents([existing_mem.content], [mem_id])
                
                # Create operation record for update action
                operation_record = MemoryOperation(
                    operation_id=operation_id,
                    operation_index=0,  # Will be set in add_note
                    timestamp=operation_time,
                    action='updated',
                    new_memory_id=note.id,
                    new_memory_content=note.content,
                    merge_into_id=update_memory_ids[0] if len(update_memory_ids) == 1 else None,
                    neighbor_ids=neighbor_ids,
                    resulting_memory_id=update_memory_ids[0] if len(update_memory_ids) == 1 else None,
                    llm_reasoning=response + "\n\nMerge response:\n" + merge_response,
                )
                return note, 'updated', operation_record

            else:
                operation_record = MemoryOperation(
                    operation_id=operation_id,
                    operation_index=0,  # Will be set in add_note
                    timestamp=operation_time,
                    action='no_action',
                    new_memory_id=note.id,
                    new_memory_content=note.content,
                    neighbor_ids=neighbor_ids,
                    llm_reasoning=response,
                )
                return None, 'no_action', operation_record
        
        # If all retries exhausted for merge action, default to create
        print(f"Warning: Failed to find valid merge ID after {max_retries} retries. Creating new memory instead.")
        operation_record = MemoryOperation(
            operation_id=operation_id,
            operation_index=0,  # Will be set in add_note
            timestamp=operation_time,
            action='created',
            new_memory_id=note.id,
            new_memory_content=note.content,
            neighbor_ids=neighbor_ids,
            resulting_memory_id=note.id,
            llm_reasoning="Failed to merge after retries, defaulting to create",
        )
        return note, 'created', operation_record

    def find_related_memories(self, query: Union[str, MemoryNote], k: int = 5, combined_scores=None, use_neighbour_memories=False) -> List[MemoryNote]:
        """Find related memories using hybrid retrieval
        
        Args:
            query: Query string or MemoryNote to find related memories for
            k: Number of memories to retrieve (-1 for all)
            combined_scores: Pre-computed similarity scores (optional)
            use_neighbour_memories: If True, expand results with related_memories. When k=-1, all 
                                   primary and neighbor memories are returned (re-ranked).
        
        Returns:
            Tuple of (memory_str, memory_session_ids)
            - memory_str: Dict (k=-1) or string (k!=-1) of formatted memory strings
            - memory_session_ids: Flattened list of all session IDs from retrieved memories
        """
        if combined_scores is None:
            combined_scores = self.compute_related_memories_similarity(query)

        if use_neighbour_memories: 
            assert self.enable_link, "use_neighbour_memories requires enable_link to be True"

        # Get top k indices
        if k == -1: # use all memories
            indices = np.argsort(combined_scores)[::-1]
        else:
            indices = np.argsort(combined_scores)[-k:][::-1]  # this is the "local" indices in the retriever
    
        if self.retrieve_method in ['separate', 'merge']:
            # Use the first available retriever to get memory_ids
            if self.use_raw_session_as_key and "content" in self.retriever:
                memory_ids = [self.retriever["content"].memory_ids[i] for i in indices]
            else:
                # Fallback to summary retriever when content is not available
                memory_ids = [self.retriever["summary"].memory_ids[i] for i in indices]
        else:
            memory_ids = [self.retriever.memory_ids[i] for i in indices]  # convert back to memory ids (session ids)

        # If use_neighbour_memories is False, use the original simple logic
        if not use_neighbour_memories:
            final_memory_ids = memory_ids
            memory_session_ids = [self.memories[i].all_ids for i in memory_ids if i in self.memories]
            memory_session_ids = sum(memory_session_ids, [])  # flatten list of lists
        else:
            # Expand with neighbor memories
            # Track seen memories to avoid duplicates when expanding with neighbors
            seen_memory_ids = set()
            final_memory_ids = []
            memory_session_ids = []  # session ids of the retrieved memories; each memory can have multiple session ids due to merging

            # Process primary retrieved memories
            for idx, mem_id in enumerate(memory_ids):
                if mem_id not in self.memories:
                    raise Exception(f"Warning: Retrieved memory ID {mem_id} not found in memories.")
                
                # Check if we've reached k (only when k != -1)
                if k != -1 and len(final_memory_ids) >= k:
                    break
                
                # Add the primary memory
                if mem_id not in seen_memory_ids:
                    final_memory_ids.append(mem_id)
                    seen_memory_ids.add(mem_id)
                    memory_session_ids.extend(self.memories[mem_id].all_ids)
                
                # After adding primary memory, add its neighbors
                mem = self.memories[mem_id]
                neighbour_ids = mem.related_memories if hasattr(mem, 'related_memories') else []
                
                for neigh_id in neighbour_ids:
                    # Check if we've reached k (only when k != -1)
                    if k != -1 and len(final_memory_ids) >= k:
                        break
                    
                    # Skip if already added or doesn't exist
                    if neigh_id in seen_memory_ids:
                        continue
                    if neigh_id not in self.memories:
                        print(f"Warning: Neighbour memory ID {neigh_id} not found in memories.")
                        continue
                    
                    # Add the neighbor memory
                    final_memory_ids.append(neigh_id)
                    seen_memory_ids.add(neigh_id)
                    memory_session_ids.extend(self.memories[neigh_id].all_ids)

        # Format memory strings
        if k == -1:  # we want each memory as a separate string
            memory_str = {}
            for i in final_memory_ids:
                if i not in self.memories:
                    continue
                tmp = self.memory_str.format(
                    id=self.memories[i].id,
                    timestamp=self.memories[i].timestamp,
                    keywords=self.memories[i].keywords,
                    facts=self.memories[i].facts,
                    summary=self.memories[i].summary,
                    content=f'Content: {self.memories[i].content}' if self.use_raw_session_as_key else '',
                )
                for sid in self.memories[i].all_ids:
                    memory_str[sid] = tmp
        else:
            memory_str = ""
            for i in final_memory_ids:
                if i not in self.memories:
                    continue
                memory_str += self.memory_str.format(
                    id=self.memories[i].id,
                    timestamp=self.memories[i].timestamp,
                    keywords=self.memories[i].keywords,
                    facts=self.memories[i].facts,
                    summary=self.memories[i].summary,
                    content=f'Content: {self.memories[i].content}' if self.use_raw_session_as_key else '',
                ) + "\n"

        return memory_str, memory_session_ids

    def _check_empty(self, text: str) -> bool:
        """Check if a text is empty or contains only whitespace"""
        return not text or (text.strip() == "" or text.strip() == ',' or text.strip() == ';')

    def compute_related_memories_similarity(self, query: Union[str, MemoryNote]) -> List[float]:
        """Compute similarity scores of related memories to a query"""
        if not self.memories:
            return []
            
        if self.retrieve_method == 'separate':
            # Determine which components to use based on use_raw_session_as_key
            if self.use_raw_session_as_key:
                components = ["summary", "keywords", "facts", "content"]
            else:
                components = ["summary", "keywords", "facts"]
            
            component_scores = []
            # Get num_docs from the first available retriever
            num_docs = len(self.retriever[components[0]].memory_ids)

            for comp in components:
                retriever = self.retriever[comp]
                if isinstance(query, MemoryNote):
                    if comp == "summary":
                        query_text = query.summary
                    elif comp == "keywords":
                        query_text = query.keywords
                    elif comp == "facts":
                        query_text = query.facts
                    else:  # content
                        query_text = query.content
                else:
                    query_text = query

                if self._check_empty(query_text):
                    component_scores.append(np.full(num_docs, np.nan, dtype=np.float32))
                    continue

                scores = retriever.search(query_text)
                if not scores:
                    component_scores.append(np.full(num_docs, np.nan, dtype=np.float32))
                    continue

                scores_array = np.array(scores, dtype=np.float32)
                if scores_array.shape[0] != num_docs:
                    raise ValueError(f"Mismatch between scores ({scores_array.shape[0]}) and documents ({num_docs}) for component '{comp}'")

                mask = np.array(retriever.valid_mask, dtype=bool)
                if mask.shape[0] != num_docs:
                    raise ValueError(f"Mismatch between mask ({mask.shape[0]}) and documents ({num_docs}) for component '{comp}'")
                scores_array[~mask] = np.nan
                component_scores.append(scores_array)

            stacked_scores = np.stack(component_scores, axis=0)
            with np.errstate(invalid='ignore'):
                combined_scores = np.nanmean(stacked_scores, axis=0)
            combined_scores = np.nan_to_num(combined_scores, nan=0.0)
        elif self.retrieve_method == 'merge_raw':
            if isinstance(query, MemoryNote):
                if self.use_raw_session_as_key:
                    tmp = "Keywords: \n" + query.keywords + "\n\nFacts: \n" + query.facts + "\n\nSummary: \n" + query.summary + "\n\nContent: \n" + query.content
                else:
                    tmp = "Keywords: \n" + query.keywords + "\n\nFacts: \n" + query.facts + "\n\nSummary: \n" + query.summary
                combined_scores = self.retriever.search(tmp)
            else:
                combined_scores = self.retriever.search(query)
        else:  # 'merge'
            if isinstance(query, MemoryNote):
                tmp = "Keywords: \n" + query.keywords + "\n\nFacts: \n" + query.facts + "\n\nSummary: \n" + query.summary
                merged_scores = self.retriever["merged"].search(tmp)
                content_scores = self.retriever["content"].search(query.content)
                combined_scores = (np.array(merged_scores, dtype=np.float32) + np.array(content_scores, dtype=np.float32)) / 2
            else:
                merged_scores = self.retriever["merged"].search(query)
                content_scores = self.retriever["content"].search(query)
                combined_scores = (np.array(merged_scores, dtype=np.float32) + np.array(content_scores, dtype=np.float32)) / 2
            
        return list(combined_scores)

    def get_combined_retriever(self, components: List[str] = None) -> FlexibleEmbeddingRetriever:
        """Get a combined retriever from specified components"""
        if self.retrieve_method not in ['separate', 'merge']:
            raise ValueError("Combined retriever is only available in separate or merge retrieve_method")
        
        if components is None:
            # Default components based on use_raw_session_as_key
            if self.retrieve_method == 'merge':
                components = ['merged', 'content']
            else:  # 'separate'
                if self.use_raw_session_as_key:
                    components = ['summary', 'keywords', 'facts', 'content']
                else:
                    components = ['summary', 'keywords', 'facts']
        
        # Filter out components that don't exist in the retriever
        available_components = [comp for comp in components if comp in self.retriever]
        if not available_components:
            raise ValueError(f"None of the requested components {components} are available in the retriever")
        
        combined_retriever = FlexibleEmbeddingRetriever(
            model_name=self.retriever[available_components[0]].model_name,
            model_type=self.retriever[available_components[0]].model_type,
            device=self.retriever[available_components[0]].device,
            cache_dir=self.retriever[available_components[0]].cache_dir
        )

        # Copy corpus structure from one of the retrievers (they should all be aligned)
        base_retriever = self.retriever[available_components[0]]
        combined_retriever.corpus = base_retriever.corpus.copy()
        combined_retriever.document_ids = base_retriever.document_ids.copy()
        combined_retriever.memory_ids = base_retriever.memory_ids.copy()

        # check the document ids and memory ids are aligned
        for comp in available_components[1:]:
            if self.retriever[comp].document_ids != combined_retriever.document_ids:
                raise ValueError(f"Document IDs do not match between components: {available_components[0]} and {comp}")
            if self.retriever[comp].memory_ids != combined_retriever.memory_ids:
                raise ValueError(f"Memory IDs do not match between components: {available_components[0]} and {comp}")

        embeddings_stack = []
        masks_stack = []
        for comp in available_components:
            embeddings = self.retriever[comp].embeddings
            if embeddings is None:
                raise ValueError(f"Embeddings for component '{comp}' are not available; cannot combine retrievers.")
            embeddings_stack.append(embeddings)
            masks_stack.append(np.array(self.retriever[comp].valid_mask, dtype=bool))

        embeddings_stack = np.stack(embeddings_stack, axis=0)
        masks_stack = np.stack(masks_stack, axis=0)

        weights = masks_stack[..., None].astype(np.float32)
        weighted_embeddings = embeddings_stack * weights
        valid_counts = masks_stack.sum(axis=0).astype(np.float32)
        valid_counts_safe = np.where(valid_counts == 0, 1.0, valid_counts)

        combined_embeddings = weighted_embeddings.sum(axis=0) / valid_counts_safe[:, None]
        zero_mask = valid_counts == 0
        if np.any(zero_mask):
            combined_embeddings[zero_mask] = embeddings_stack[0][zero_mask]

        combined_retriever.embeddings = normalize(combined_embeddings)
        combined_retriever.valid_mask = list((valid_counts > 0).astype(bool))
        
        return combined_retriever
    
    def get_operation_trajectory(self) -> List[Dict]:
        """Get the full operation trajectory as a list of dictionaries"""
        return [op.model_dump() for op in self.operation_trajectory]
    
    def save_trajectory(self, filepath: str):
        """Save the operation trajectory to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_operation_trajectory(), f, indent=2)
        print(f"Trajectory saved to {filepath}")
    
    def get_trajectory_summary(self) -> Dict[str, int]:
        """Get a summary of operations performed"""
        summary = {
            'total_operations': len(self.operation_trajectory),
            'created': sum(1 for op in self.operation_trajectory if op.action == 'created'),
            'merged': sum(1 for op in self.operation_trajectory if op.action == 'merged'),
            'updated': sum(1 for op in self.operation_trajectory if op.action == 'updated'),
            'no_action': sum(1 for op in self.operation_trajectory if op.action == 'no_action')
        }
        return summary
    
    def print_trajectory_summary(self):
        """Print a human-readable summary of the trajectory"""
        summary = self.get_trajectory_summary()
        print("\n=== Memory Operation Trajectory Summary ===")
        print(f"Total Operations: {summary['total_operations']}")
        print(f"  - Created: {summary['created']}")
        print(f"  - Merged: {summary['merged']}")
        print(f"  - Updated: {summary['updated']}")
        print(f"  - No Action: {summary['no_action']}")
        print("=" * 43)
    
    def cleanup(self):
        """Clean up GPU resources from retrievers"""
        if self.retrieve_method == 'merge_raw':
            if hasattr(self, 'retriever') and self.retriever is not None:
                self.retriever.cleanup()
        else:
            if hasattr(self, 'retriever') and isinstance(self.retriever, dict):
                for key, ret in self.retriever.items():
                    if ret is not None:
                        ret.cleanup()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
    
    @classmethod
    def load_from_local_memory(cls, memories_dict: Dict[str, 'MemoryNote'], model_name: str, model_type: str, 
                              device: str = None, cache_dir: str = None, component: str = None, use_raw_session_as_key: bool = False) -> 'FlexibleEmbeddingRetriever':
        """
        Load retriever state from memory objects.
        
        Args:
            memories_dict: Dictionary of {memory_id: MemoryNote}
            model_name: Name of the embedding model
            model_type: Type of embedding model
            device: Device to run on
            cache_dir: Cache directory for models
            component: Which component to load ('topic', 'summary', 'keywords', 'facts', 'events', 'content', or None for full)
            
        Returns:
            Initialized retriever with memories
        """        
        # Create and initialize retriever
        retriever = cls(model_name=model_name, model_type=model_type, device=device, cache_dir=cache_dir)
        
        # Extract memory IDs and format documents based on component
        memory_ids = list(memories_dict.keys())
        documents = []
        for mem_id in memory_ids:
            mem = memories_dict[mem_id]
            # Format based on component (for 'separate' mode) or full (for 'merge' mode)
            if component == 'summary':
                doc = "Summary: \n" + mem.summary
            elif component == 'keywords':
                doc = "Keywords: \n" + ", ".join(mem.keywords)
            elif component == 'facts':
                doc = "Facts: \n" + ", ".join(mem.facts)
            elif component == 'content':
                doc = "Content: \n" + mem.content
            else:
                # Full format for 'merge' mode
                if use_raw_session_as_key:
                    doc = "Keywords: \n" + query.keywords + "\n\nFacts: \n" + query.facts + "\n\nSummary: \n" + query.summary + "\n\nContent: \n" + query.content
                else:
                    doc = "Keywords: \n" + mem.keywords + "\n\nFacts: \n" + mem.facts + "\n\nSummary: \n" + mem.summary
            documents.append(doc)
        
        retriever.add_documents(documents, memory_ids)
        return retriever
