"""
Utility functions for HaluMem evaluation with structured memory system.
This module wraps the AgenticMemorySystem for HaluMem evaluation.

The structured memory system supports:
- Memory extraction with summary, keywords, facts
- Memory update/merge operations based on semantic similarity
- Flexible embedding models (Stella, Contriever, GTE, SentenceTransformer)
"""

import os
import sys
import re
import json
import time
import pickle
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.flat.agentic_memory_system import AgenticMemorySystem
from src.flat.llm_controller import LLMController

# Model configuration mapping (aligned with structure_run_retrieval_lme.py)
MODEL_CONFIG = {
    'all-MiniLM-L6-v2': {'model_name': 'all-MiniLM-L6-v2', 'model_type': 'sentence-transformer'},
    'all-mpnet-base-v2': {'model_name': 'all-mpnet-base-v2', 'model_type': 'sentence-transformer'},
    'contriever': {'model_name': 'facebook/contriever', 'model_type': 'contriever'},
    'stella': {'model_name': 'dunzhang_stella_en_1.5B_v5', 'model_type': 'stella'},
    'gte': {'model_name': 'Alibaba-NLP/gte-Qwen2-7B-instruct', 'model_type': 'gte'},
    'openai': {'model_name': 'text-embedding-3-small', 'model_type': 'openai'},
}


@dataclass
class MemoryAddResult:
    """Result of adding a memory to the system."""
    memory_id: str
    action: str  # 'created', 'updated', 'no_action'
    content: str
    summary: str
    keywords: str
    facts: str
    timestamp: str
    session_id: str
    merged_into_ids: List[str] = None
    duration_ms: float = 0.0


class StructuredMemoryRetriever:
    """
    Wrapper around AgenticMemorySystem for HaluMem evaluation.
    
    This class provides:
    - Session-based memory management
    - Memory extraction with structured metadata (summary, keywords, facts)
    - Memory update/merge operations
    - Retrieval with ranking
    """
    
    def __init__(
        self,
        embedding_model: str = 'stella',
        retrieve_method: str = 'merge',
        llm_model: str = 'gpt-4o-mini',
        llm_backend: str = 'openai',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        keep_update_note: bool = False,
        enable_link: bool = False,
        enable_update: bool = True,
        use_raw_session_as_key: bool = False,
    ):
        """
        Initialize the structured memory retriever.
        
        Args:
            embedding_model: Name of embedding model (stella, contriever, gte, etc.)
            retrieve_method: 'merge' (single retriever), 'separate' (per-component), or 'merge_raw' (with raw session)
            llm_model: LLM model for memory operations
            llm_backend: LLM backend ('openai')
            api_key: API key for LLM
            base_url: Base URL for LLM API
            device: Device for embedding model
            cache_dir: Cache directory for models
            keep_update_note: Keep original note after update operation
        """
        # Get model config
        if embedding_model in MODEL_CONFIG:
            model_config = MODEL_CONFIG[embedding_model]
        else:
            model_config = {'model_name': embedding_model, 'model_type': 'sentence-transformer'}
        
        self.embedding_model = embedding_model
        self.model_config = model_config
        self.llm_model = llm_model
        self.temperature = temperature
        
        # Initialize the memory system
        self.memory_system = AgenticMemorySystem(
            retrieve_method=retrieve_method,
            retriever_name=model_config['model_name'],
            retriever_type=model_config['model_type'],
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=api_key,
            base_url=base_url,
            device=device,
            cache_dir=cache_dir,
            keep_update_note=keep_update_note,
            enable_link=enable_link,
            enable_update=enable_update,
            use_raw_session_as_key=use_raw_session_as_key,
        )
        
        # Track session -> memory mapping
        self.session_memories: Dict[str, List[str]] = {}  # session_id -> list of memory_ids
        self.memory_to_session: Dict[str, str] = {}  # memory_id -> session_id
        
        # Track all attempted additions (including rejected)
        self.all_session_ids: List[str] = []
        self.rejected_session_ids: List[str] = []
        
        # Initialize LLM controller for metadata extraction
        self.llm_controller = LLMController(
            backend=llm_backend,
            model=llm_model,
            api_key=api_key,
            base_url=base_url
        )
    
    def extract_metadata_from_dialogue(
        self,
        dialogue: List[Dict],
        user_name: str,
        session_id: str,
        timestamp: str,
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Extract structured metadata (summary, keywords, facts) from dialogue.
        Since HaluMem expect multiple memory points for each session, first use LLM to split dialogue into blocks, then extract metadata per block.
        Uses the same prompts as LongMemEval index expansion.
        
        Args:
            dialogue: List of dialogue turns with 'role', 'content'
            user_name: User's name
            session_id: Session identifier
            timestamp: Session timestamp
            max_retries: Maximum number of retry attempts for each extraction
            
        Returns:
            Dict with 'summary', 'keywords', 'facts', 'content'
            
        Raises:
            RuntimeError: If all retry attempts fail for any extraction
        """
        split_indices = [len(dialogue) - 1]

        memories = []
        
        for idx in split_indices:
            dialogue_block = dialogue[:idx + 1]
            
            # Extract summary with retry
            summary = None
            for attempt in range(max_retries):
                try:
                    summary = self._extract_summ(dialogue_block)
                    if summary:
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to extract summary after {max_retries} attempts: {e}")
                    time.sleep(1)  # Brief pause before retry
            
            # Extract keyphrases with retry
            keyphrases = None
            for attempt in range(max_retries):
                try:
                    keyphrases_str = self._extract_key(dialogue_block)
                    keyphrases = keyphrases_str.split(';')
                    keyphrases = [kp.strip() for kp in keyphrases if kp.strip()]
                    if keyphrases:
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to extract keyphrases after {max_retries} attempts: {e}")
                    time.sleep(1)
            
            # Extract facts with retry
            facts = None
            for attempt in range(max_retries):
                try:
                    facts = self._extract_facts(dialogue_block, user_name, timestamp)
                    if facts:
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to extract facts after {max_retries} attempts: {e}")
                    time.sleep(1)
            memories.append({'summary': summary, 'keywords': keyphrases, 'facts': facts, 'content': self._format_dialogue_content(dialogue_block, user_name)})
        
        return memories
    
    def _format_dialogue_content(self, dialogue: List[Dict], user_name: str) -> str:
        """Format dialogue into a single content string."""
        lines = []
        for turn in dialogue:
            role = turn['role'].capitalize()
            if role.lower() == 'user':
                role = user_name
            lines.append(f"{role}: {turn['content']}")
        return '\n\n'.join(lines)

    def _extract_summ(self, dialogue: List[Dict]) -> str:
        """Extract summary from dialogue using LLM. Adapted from LME's prompts"""
        if 'gpt' in self.llm_model.lower():
            client = OpenAI()
        else:
            client = OpenAI(
                api_key="empty",
                base_url="http://localhost:8001/v1",
            )
        summarization_prompt = "Below is a transcript of a conversation between a human user and an AI assistant. Please summarize the following dialogue as concisely as possible in a short paragraph, extracting the main themes and key information. In your summary, focus more on what the user mentioned or asked for. Dialogue content:\n"
        for turn_entry in dialogue:
            summarization_prompt += f"\n{turn_entry['role']}: {turn_entry['content']}"
        summarization_prompt += '\n\nYour summary (be concise):'

        kwargs = {
            'model': self.llm_model,
            'messages':[
                {"role": "user", "content": summarization_prompt}
            ],
            'n': 1,
            'temperature': 0,
            'max_tokens': 500,
            # 'max_completion_tokens': 500
        }
        completion = client.chat.completions.create(**kwargs) 
        return completion.choices[0].message.content.strip()

    def _extract_key(self, dialogue: List[Dict]) -> List[str]:
        """Extract keyphrases from dialogue using LLM. Copied from LME's batch_expansion_session_keyphrases.py"""
        if 'gpt' in self.llm_model.lower():
            client = OpenAI()
        else:
            client = OpenAI(
                api_key="empty",
                base_url="http://localhost:8001/v1",
            )
        summarization_prompt = "Below is a transcript of a conversation between a human user and an AI assistant. Generate a list of keyphrases for the session. Separate each keyphrase with a semicolon. Dialogue content:\n"
        for turn_entry in dialogue:
            summarization_prompt += f"\n{turn_entry['role']}：{turn_entry['content']}"
        summarization_prompt += '\n\nKeyphrases (separated by semicolon):'

        kwargs = {
            'model': self.llm_model,
            'messages':[
                {"role": "user", "content": summarization_prompt}
            ],
            'n': 1,
            'temperature': 0,
            'max_tokens': 100
        }
        completion = client.chat.completions.create(**kwargs) 
        return completion.choices[0].message.content.strip()
            

    def _extract_facts(self, dialogue: List[Dict], user_name: str, timestamp: str) -> List[str]:
        """Extract facts from dialogue using LLM."""
        FACT_EXTRACTION_PROMPT = """You are a memory extraction system. Your task is to extract key facts from the following dialogue between a user ({user_name}) and an AI assistant.
Extract all the personal information, life events, experience, and preferences related to the user. Make sure you include all details such as life events, personal experience, preferences, specific numbers, locations, or dates. State each piece of information in a simple sentence. Minimize the coreference across the facts, e.g., replace pronouns with actual entities.

# Dialogue:
{dialogue}

# Session Timestamp: {timestamp}

# Output Format
Return one memory per line as plain text. Do not include markdown formatting (like ```json), explanations, quotes, numbering, or conversational text. Each line should be a complete, self-contained fact statement about the user.

Example of valid output:
{user_name} has worked as a software engineer at TechCorp since 2020 and finds the work fulfilling but stressful.
{user_name} enjoys hiking and has a specific plan to visit Yellowstone National Park in summer 2025 with their family.
{user_name} is currently learning to play the guitar, practicing three times a week to manage anxiety.

# Extracted Facts:
"""
        
        dialogue_text = ""
        for turn in dialogue:
            role = turn['role'].capitalize()
            content = turn['content']
            dialogue_text += f"{role}: {content}\n\n"
        
        prompt = FACT_EXTRACTION_PROMPT.format(
            user_name=user_name,
            dialogue=dialogue_text.strip(),
            timestamp=timestamp
        )
        client = OpenAI() if 'gpt' in self.llm_model.lower() else OpenAI(api_key="empty", base_url="http://localhost:8001/v1") 
        
        response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}],
                                         model=self.llm_model,
                                        #  max_tokens=1000,
                                        #  temperature=self.temperature,
                                         )
        response = response.choices[0].message.content.strip()
        # Parse facts from response
        facts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove common prefixes
                line = re.sub(r'^(\d+[\.\)]\s*|-\s*|\*\s*|•\s*)', '', line)
                if line.strip():
                    facts.append(line.strip())
        
        return facts
    
    def add_memory(
        self,
        content: str,
        session_id: str,
        timestamp: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryAddResult:
        """
        Add a memory to the system.
        
        The memory system will:
        1. Compare with existing memories
        2. Decide to create new, update existing, or skip (redundant)
        
        Args:
            content: Memory content (dialogue text)
            session_id: Unique session identifier
            timestamp: Session timestamp
            metadata: Optional pre-extracted metadata with 'summary', 'keywords', 'facts'
            
        Returns:
            MemoryAddResult with operation details
        """
        # Track this session as attempted
        self.all_session_ids.append(session_id)
        
        start_time = time.time()
        
        # Use pre-extracted metadata or provide empty values
        if metadata:
            summary = metadata.get('summary', '')
            keywords = metadata.get('keywords', [])
            facts = metadata.get('facts', [])
        else:
           raise ValueError("Metadata with 'summary', 'keywords', and 'facts' must be provided.")

        # Add memory using AgenticMemorySystem
        # The system will process and decide action (create/update/pass)
        self.memory_system.add_note(
            content=content,
            time=timestamp,
            id=session_id,
            summary=summary,
            keywords=[keywords] if isinstance(keywords, str) else keywords,
            facts=facts
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Get the last operation to determine what action was taken
        last_op = self.memory_system.operation_trajectory[-1] if self.memory_system.operation_trajectory else None
        
        if last_op is None:
            # Shouldn't happen, but handle gracefully
            raise RuntimeError("No operation recorded after add_note call")
        
        # Determine action from the operation record (more reliable than checking memories dict)
        action = last_op.action
        
        if action == 'created':
            # Memory was created - it should be in self.memories
            if session_id in self.memory_system.memories:
                memory = self.memory_system.memories[session_id]
                self.session_memories[session_id] = [session_id]
                self.memory_to_session[session_id] = session_id
                
                return MemoryAddResult(
                    memory_id=session_id,
                    action='created',
                    content=content,
                    summary=memory.summary,
                    keywords=memory.keywords,
                    facts=memory.facts,
                    timestamp=timestamp,
                    session_id=session_id,
                    duration_ms=duration_ms
                )
            else:
                # Created but not in memories - use the metadata we have
                self.session_memories[session_id] = [session_id]
                self.memory_to_session[session_id] = session_id
                
                return MemoryAddResult(
                    memory_id=session_id,
                    action='created',
                    content=content,
                    summary=summary,
                    keywords=keywords if isinstance(keywords, str) else '; '.join(keywords),
                    facts='; '.join(facts) if isinstance(facts, list) else facts,
                    timestamp=timestamp,
                    session_id=session_id,
                    duration_ms=duration_ms
                )
        
        elif action == 'updated':
            # Memory was used to update existing memories
            # Get the IDs of memories that were updated
            # merge_into_id is set when only 1 memory is updated, otherwise check resulting_memory_id
            merged_into_ids = []
            if last_op.merge_into_id:
                merged_into_ids = [last_op.merge_into_id]
            elif last_op.resulting_memory_id:
                merged_into_ids = [last_op.resulting_memory_id]
            
            # Track this session
            self.rejected_session_ids.append(session_id)
            
            return MemoryAddResult(
                memory_id=session_id,
                action='updated',
                content=content,
                summary=summary,
                keywords=keywords if isinstance(keywords, str) else '; '.join(keywords),
                facts='; '.join(facts) if isinstance(facts, list) else facts,
                timestamp=timestamp,
                session_id=session_id,
                merged_into_ids=merged_into_ids,
                duration_ms=duration_ms
            )
        else:
            # action == 'no_action' - Rejected as redundant
            self.rejected_session_ids.append(session_id)
            
            return MemoryAddResult(
                memory_id=session_id,
                action='no_action',
                content=content,
                summary=summary,
                keywords=keywords if isinstance(keywords, str) else '; '.join(keywords),
                facts='; '.join(facts) if isinstance(facts, list) else facts,
                timestamp=timestamp,
                session_id=session_id,
                duration_ms=duration_ms
            )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20
    ) -> Tuple[List[Dict], List[str]]:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: Query string
            top_k: Number of memories to retrieve
            
        Returns:
            Tuple of (memory_str_dict, session_ids)
            - memory_str_dict: Dict mapping memory_id -> formatted memory string
            - session_ids: Ordered list of all session IDs (ranked by relevance)
        """
        if not self.memory_system.memories:
            return {}, []
        
        # Use the memory system's find_related_memories method
        memory_str, session_ids = self.memory_system.find_related_memories(query, k=top_k)
        
        return memory_str, session_ids
    
    def retrieve_with_content(
        self,
        query: str,
        top_k: int = 20,
        use_neighbour_memories: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve relevant memories with original content and timestamps.
        
        Similar to mem0's search_memory, returns memories with:
        - Original content (dialogue)
        - Timestamp
        - Relevance score
        
        Args:
            query: Query string
            top_k: Number of memories to retrieve
            use_neighbour_memories: Use neighbour memories (similiar to A-Mem) to fill top_k
            
        Returns:
            Tuple of (memories_list, session_ids)
            - memories_list: List of dicts with 'memory', 'timestamp', 'memory_id', 'score',
                            'summary', 'keywords', 'facts', 'content' (all fields for flexibility)
            - session_ids: Ordered list of all session IDs (ranked by relevance)
        """
        if not self.memory_system.memories:
            return [], []
        
        # Get similarity scores for ranking
        combined_scores = self.memory_system.compute_related_memories_similarity(query)
        
        if not combined_scores:
            return [], []
        
        # Get top k indices
        combined_scores = np.array(combined_scores)
        if top_k == -1:
            indices = np.argsort(combined_scores)[::-1]
        else:
            indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        # Get memory IDs from retriever
        if self.memory_system.retrieve_method == 'separate':
            memory_ids = [self.memory_system.retriever["summary"].memory_ids[i] for i in indices]
        elif self.memory_system.retrieve_method == 'merge' and self.memory_system.use_raw_session_as_key:
            memory_ids = [self.memory_system.retriever['content'].memory_ids[i] for i in indices]
        else:
            memory_ids = [self.memory_system.retriever.memory_ids[i] for i in indices]
        
        # Build memories list with content and timestamps
        memories_list = []
        all_session_ids = []
        seen_memory_ids = set()  # Track which memories we've already added
        
        for idx, mem_id in enumerate(memory_ids):
            if mem_id not in self.memory_system.memories:
                continue
            
            # Check if we've reached top_k (when use_neighbour_memories is enabled)
            if use_neighbour_memories and top_k > 0 and len(memories_list) >= top_k:
                break
                
            mem = self.memory_system.memories[mem_id]
            score = float(combined_scores[indices[idx]]) if idx < len(indices) else 0.0
            
            memories_list.append({
                'timestamp': mem.timestamp,
                'memory_id': mem_id,
                'score': round(score, 4),
                # Include all structured fields for flexibility
                'summary': mem.summary,
                'keywords': mem.keywords,
                'facts': mem.facts,
                'content': mem.content,

                'related_memories': mem.related_memories if hasattr(mem, 'related_memories') else [],
            })
            seen_memory_ids.add(mem_id)
            
            # Collect all session IDs (for merged memories)
            all_session_ids.extend(mem.all_ids)

            # After adding primary memory, add its neighbors to fill up to top_k
            if use_neighbour_memories and top_k > 0:
                neighbour_ids = mem.related_memories if hasattr(mem, 'related_memories') else []
                for neigh_id in neighbour_ids:
                    # Check if we've reached top_k
                    if len(memories_list) >= top_k:
                        break
                    
                    # Skip if already added or doesn't exist
                    if neigh_id in seen_memory_ids:
                        continue
                    if neigh_id not in self.memory_system.memories:
                        print(f"Warning: Neighbour memory ID {neigh_id} not found in memories.")
                        continue
                    
                    neigh_mem = self.memory_system.memories[neigh_id]

                    memories_list.append({
                        'timestamp': neigh_mem.timestamp,
                        'memory_id': neigh_id,
                        'score': round(score, 4),  # Inherit score from parent for now
                        'summary': neigh_mem.summary,
                        'keywords': neigh_mem.keywords,
                        'facts': neigh_mem.facts,
                        'content': neigh_mem.content,

                        'related_memories': [],  # do not expand further
                    })
                    seen_memory_ids.add(neigh_id)
                    all_session_ids.extend(neigh_mem.all_ids)
        
        return memories_list, all_session_ids
    
    def _flatten_memory_to_points(
        self,
        mem_data: Union[Dict[str, Any], Any],
        mem_id: str,
        use_raw_session: bool = False,
        base_score: Optional[float] = None,
        include_source: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Flatten a single memory into its component points.
        
        This is a unified helper method that works with both:
        - Memory dictionaries (from retrieve_with_content)
        - Memory objects (from self.memory_system.memories)
        
        Args:
            mem_data: Either a dict with memory data or a memory object
            mem_id: Memory ID
            use_raw_session: Whether to include original content as a point
            base_score: Optional base score to assign to all points
            include_source: Whether to include source memory in returned points
            
        Returns:
            List of point dicts with 'point', 'point_type', 'timestamp', 'memory_id', etc.
        """
        points = []
        
        # Handle both dict and object access
        if isinstance(mem_data, dict):
            timestamp = mem_data.get('timestamp', 'unknown')
            facts_str = mem_data['facts']
            summary = mem_data['summary']
            keywords_str = mem_data['keywords']
            content = mem_data['content']
        else:
            # Assume it's a memory object
            timestamp = mem_data.timestamp
            facts_str = mem_data.facts
            summary = mem_data.summary
            keywords_str = mem_data.keywords
            content = mem_data.content
        
        # Add each fact as a separate point
        if facts_str:
            facts = [f.strip() for f in facts_str.split(';') if f.strip()]
            for fact in facts:
                point_data = {
                    'point': fact,
                    'point_type': 'fact',
                    'timestamp': timestamp,
                    'memory_id': mem_id,
                }
                if base_score is not None:
                    point_data['score'] = base_score
                if include_source:
                    point_data['source_memory'] = mem_data
                points.append(point_data)
        
        # Add summary as a point
        if summary and summary.strip():
            point_data = {
                'point': summary.strip(),
                'point_type': 'summary',
                'timestamp': timestamp,
                'memory_id': mem_id,
            }
            if base_score is not None:
                point_data['score'] = base_score
            if include_source:
                point_data['source_memory'] = mem_data
            points.append(point_data)
        
        # Add each keyword as a separate point
        if keywords_str:
            keywords = [k.strip() for k in keywords_str.split(';') if k.strip()]
            for kw in keywords:
                point_data = {
                    'point': kw,
                    'point_type': 'keyword',
                    'timestamp': timestamp,
                    'memory_id': mem_id,
                }
                if base_score is not None:
                    point_data['score'] = base_score
                if include_source:
                    point_data['source_memory'] = mem_data
                points.append(point_data)
        
        # Optionally add content as a point
        if use_raw_session:
            if content and content.strip():
                point_data = {
                    'point': content.strip(),
                    'point_type': 'content',
                    'timestamp': timestamp,
                    'memory_id': mem_id,
                }
                if base_score is not None:
                    point_data['score'] = base_score
                if include_source:
                    point_data['source_memory'] = mem_data
                points.append(point_data)
        
        return points
    
    def retrieve_flattened_points(
        self,
        query: str,
        top_k: int = 20,
        use_raw_session: bool = False,
        use_neighbour_memories: bool = False,
        pre_flatten: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve and flatten memory notes into individual points for ranking.
        
        Each memory note is flattened into separate points:
        - 1 summary point
        - N keyword points (split by ';')
        - M fact points (split by ';')
        - Optionally: 1 content point
        
        All points are ranked together and truncated to top_k.
        
        Args:
            query: Query string
            top_k: Number of flattened points to return
            use_raw_session: Whether to include original content as a point
            use_neighbour_memories: Use neighbour memories to fill top_k
            pre_flatten: If True, use _retrieve_all_flatten method to compute embeddings for all points individually
            
        Returns:
            Tuple of (flattened_points, session_ids)
            - flattened_points: List of dicts with 'point', 'point_type', 'timestamp', 'memory_id', 'score'
            - session_ids: Ordered list of all session IDs from selected points
        """
        if not self.memory_system.memories:
            return [], []
        
        if pre_flatten:
            return self._retrieve_all_flatten(
                query,
                top_k=top_k,
                use_raw_session=use_raw_session,
                use_neighbour_memories=use_neighbour_memories
            )
        
        # First, retrieve all memory notes (no truncation at note level); no use_neighbour_memories here
        memories_list, _ = self.retrieve_with_content(query, top_k=-1, use_neighbour_memories=False)
        
        if not memories_list:
            return [], []
        
        # Flatten each primary memory note into individual points
        all_points = []
        seen_memory_ids = set()  # Track which memories we've already processed
        
        for mem in memories_list:
            mem_id = mem['memory_id']
            base_score = mem['score']
            
            # Flatten primary memory using the unified method
            points = self._flatten_memory_to_points(
                mem_data=mem,
                mem_id=mem_id,
                use_raw_session=use_raw_session,
                base_score=base_score,
                include_source=True
            )
            all_points.extend(points)
            seen_memory_ids.add(mem_id)
            
            # If use_neighbour_memories is enabled, also flatten neighbor memories
            if use_neighbour_memories:
                neighbour_ids = mem.get('related_memories', [])
                for neigh_id in neighbour_ids:
                    # Skip if already processed or doesn't exist
                    if neigh_id in seen_memory_ids:
                        continue
                    if neigh_id not in self.memory_system.memories:
                        print(f"Warning: Neighbour memory ID {neigh_id} not found in memories.")
                        continue
                    
                    neigh_mem_obj = self.memory_system.memories[neigh_id]
                    
                    # Flatten neighbor memory using the unified method
                    neigh_points = self._flatten_memory_to_points(
                        mem_data=neigh_mem_obj,
                        mem_id=neigh_id,
                        use_raw_session=use_raw_session,
                        base_score=base_score,
                        include_source=True
                    )
                    all_points.extend(neigh_points)
                    seen_memory_ids.add(neigh_id)
        
        # Sort all points by score (descending) and truncate to top_k
        all_points.sort(key=lambda x: x['score'], reverse=True)
        
        if top_k > 0:
            all_points = all_points[:top_k]
        
        # Collect session IDs from selected points (preserving order, removing duplicates)
        seen_ids = set()
        session_ids = []
        for point in all_points:
            source_mem = point.get('source_memory', {})
            # Get all_ids from the source memory if available
            mem_id = point['memory_id']
            if mem_id in self.memory_system.memories:
                for sid in self.memory_system.memories[mem_id].all_ids:
                    if sid not in seen_ids:
                        seen_ids.add(sid)
                        session_ids.append(sid)
        
        return all_points, session_ids
    
    def _retrieve_all_flatten(
        self,
        query: str,
        top_k: int = 20,
        use_raw_session: bool = False,
        use_neighbour_memories: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve by directly computing embeddings for all flattened points.
        
        Unlike retrieve_flattened_points which first retrieves notes then flattens,
        this method flattens all points first, computes embeddings on-demand,
        and retrieves directly from the flattened points.
        
        Each memory note is flattened into separate points:
        - 1 summary point
        - N keyword points (split by ';')
        - M fact points (split by ';')
        - Optionally: 1 content point
        
        Args:
            query: Query string
            top_k: Number of flattened points to return
            use_raw_session: Whether to include original content as a point
            use_neighbour_memories: Use neighbour memories to fill top_k
            
        Returns:
            Tuple of (flattened_points, session_ids)
            - flattened_points: List of dicts with 'point', 'point_type', 'timestamp', 'memory_id', 'score'
            - session_ids: Ordered list of all session IDs from selected points
        """
        if not self.memory_system.memories:
            return [], []
        
        # Step 1: Flatten all memories into points
        all_points = []
        seen_memory_ids = set()
        
        for mem_id, mem_obj in self.memory_system.memories.items():
            # Flatten primary memory using the unified method
            points = self._flatten_memory_to_points(
                mem_data=mem_obj,
                mem_id=mem_id,
                use_raw_session=use_raw_session,
                base_score=None,
                include_source=False
            )
            all_points.extend(points)
            seen_memory_ids.add(mem_id)
            
            # If use_neighbour_memories is enabled, also flatten neighbor memories
            if use_neighbour_memories:
                neighbour_ids = mem_obj.related_memories if hasattr(mem_obj, 'related_memories') else []
                for neigh_id in neighbour_ids:
                    # Skip if already processed or doesn't exist
                    if neigh_id in seen_memory_ids:
                        continue
                    if neigh_id not in self.memory_system.memories:
                        continue
                    
                    neigh_mem_obj = self.memory_system.memories[neigh_id]
                    neigh_points = self._flatten_memory_to_points(
                        mem_data=neigh_mem_obj,
                        mem_id=neigh_id,
                        use_raw_session=use_raw_session,
                        base_score=None,
                        include_source=False
                    )
                    all_points.extend(neigh_points)
                    seen_memory_ids.add(neigh_id)
        
        if not all_points:
            return [], []
        
        # Step 2: Get retriever and compute embeddings for all points
        # Determine which retriever to use based on retrieve_method
        if self.memory_system.retrieve_method == 'separate':
            # Use the summary retriever as base (could also use merged retriever)
            base_retriever = self.memory_system.retriever['summary']
        elif self.memory_system.retrieve_method == 'merge':
            base_retriever = self.memory_system.retriever['content']
        else:
            base_retriever = self.memory_system.retriever
        
        # Extract point texts for embedding
        point_texts = [point['point'] for point in all_points]
        
        # Compute embeddings for all points
        point_embeddings = base_retriever.encode(point_texts)
        
        # Compute query embedding
        query_embedding = base_retriever.encode([query])[0]
        
        # Step 3: Compute similarity scores
        similarities = cosine_similarity([query_embedding], point_embeddings)[0]
        
        # Add scores to points
        for i, point in enumerate(all_points):
            point['score'] = round(float(similarities[i]), 4)
        
        # Step 4: Sort by score and truncate to top_k
        all_points.sort(key=lambda x: x['score'], reverse=True)
        
        if top_k > 0:
            all_points = all_points[:top_k]
        
        # Step 5: Collect session IDs from selected points
        seen_ids = set()
        session_ids = []
        for point in all_points:
            mem_id = point['memory_id']
            if mem_id in self.memory_system.memories:
                for sid in self.memory_system.memories[mem_id].all_ids:
                    if sid not in seen_ids:
                        seen_ids.add(sid)
                        session_ids.append(sid)
        
        # Clean up memory_obj from points (not needed in return value)
        for point in all_points:
            point.pop('memory_obj', None)
        
        return all_points, session_ids
    
    def format_flattened_points_as_context(
        self,
        flattened_points: List[Dict[str, Any]],
        user_name: str,
        include_timestamp: bool = True,
        include_point_type: bool = False,
    ) -> str:
        """
        Format flattened memory points into a context string for QA (mem0 style).
        
        Args:
            flattened_points: List of point dicts from retrieve_flattened_points
            user_name: User's name
            include_timestamp: Whether to include timestamps
            include_point_type: Whether to include point type (summary/keyword/fact)
            
        Returns:
            Formatted context string
        """
        if not flattened_points:
            return f"No memories found for {user_name}."
        
        formatted_points = []
        
        for i, point in enumerate(flattened_points, 1):
            point_text = point.get('point', '')
            timestamp = point.get('timestamp', 'unknown')
            point_type = point.get('point_type', 'unknown')
            
            # Optionally prefix with point type
            if include_point_type:
                point_text = f"[{point_type}] {point_text}"
            
            if include_timestamp:
                formatted_points.append(f"{timestamp}: {point_text}")
            else:
                formatted_points.append(point_text)
    
        context = f"""Memories for user {user_name}:

    {json.dumps(formatted_points, indent=4)}
"""
        return context
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        return {
            'total_memories': len(self.memory_system.memories),
            'total_attempted': len(self.all_session_ids),
            'total_rejected': len(self.rejected_session_ids),
            'total_operations': len(self.memory_system.operation_trajectory),
            'embedding_model': self.embedding_model,
        }
    
    def save_state(self, filepath: str):
        """Save memory system state to file."""
        state = {
            'memories': self.memory_system.memories,
            'operation_trajectory': self.memory_system.operation_trajectory,
            'session_memories': self.session_memories,
            'memory_to_session': self.memory_to_session,
            'all_session_ids': self.all_session_ids,
            'rejected_session_ids': self.rejected_session_ids,
        }
        
        # Save retriever state
        if self.memory_system.retrieve_method == 'merge':
            state['retriever'] = {
                'corpus': self.memory_system.retriever.corpus,
                'embeddings': self.memory_system.retriever.embeddings,
                'document_ids': self.memory_system.retriever.document_ids,
                'memory_ids': self.memory_system.retriever.memory_ids,
            }
        else:
            state['retriever'] = {}
            for comp in ['summary', 'keywords', 'facts', 'content']:
                state['retriever'][comp] = {
                    'corpus': self.memory_system.retriever[comp].corpus,
                    'embeddings': self.memory_system.retriever[comp].embeddings,
                    'document_ids': self.memory_system.retriever[comp].document_ids,
                    'memory_ids': self.memory_system.retriever[comp].memory_ids,
                }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load memory system state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.memory_system.memories = state['memories']
        self.memory_system.operation_trajectory = state['operation_trajectory']
        self.session_memories = state['session_memories']
        self.memory_to_session = state['memory_to_session']
        self.all_session_ids = state['all_session_ids']
        self.rejected_session_ids = state['rejected_session_ids']
        
        # Restore retriever state
        if self.memory_system.retrieve_method == 'merge':
            self.memory_system.retriever.corpus = state['retriever']['corpus']
            self.memory_system.retriever.embeddings = state['retriever']['embeddings']
            self.memory_system.retriever.document_ids = state['retriever']['document_ids']
            self.memory_system.retriever.memory_ids = state['retriever']['memory_ids']
        else:
            for comp in ['summary', 'keywords', 'facts', 'content']:
                self.memory_system.retriever[comp].corpus = state['retriever'][comp]['corpus']
                self.memory_system.retriever[comp].embeddings = state['retriever'][comp]['embeddings']
                self.memory_system.retriever[comp].document_ids = state['retriever'][comp]['document_ids']
                self.memory_system.retriever[comp].memory_ids = state['retriever'][comp]['memory_ids']
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.memory_system, 'cleanup'):
            self.memory_system.cleanup()


def format_memories_for_halumem_context(
    memories: Union[str, Dict[str, str]],
    session_ids: List[str],
    user_name: str,
    top_k: int = 20,
    include_timestamp: bool = True
) -> str:
    """
    Format retrieved memories into context string for HaluMem QA.
    
    Args:
        memories: Either formatted string or dict of memory_id -> memory_str
        session_ids: Ordered list of session IDs
        user_name: User's name
        top_k: Maximum memories to include
        include_timestamp: Whether to include timestamps
        
    Returns:
        Formatted context string
    """
    if isinstance(memories, str):
        return memories
    
    if not memories:
        return f"No memories found for {user_name}."
    
    lines = [f"Memories for {user_name}:\n"]
    
    for i, sid in enumerate(session_ids[:top_k]):
        if sid in memories:
            lines.append(f"{i+1}. {memories[sid]}")
    
    return '\n'.join(lines)