"""
- Extracts structured metadata (summary, keywords, facts) from dialogues
- Performs memory update/merge operations based on semantic similarity
- Supports multiple embedding models
- Supports GPU-based parallel processing for faster evaluation
"""

import os
import re
import sys
import time
import json
import copy
import argparse
import traceback
import multiprocessing
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid

import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.flat.halu_utils import StructuredMemoryRetriever, MODEL_CONFIG
from src.flat.halu_prompts import PROMPT_QA_TEMPLATE
from src import config as cfg

MEMORY_EXTRACTION_LLM = {
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'llama_3.1_8b',
    'gpt-4o-mini': 'gpt-4o-mini'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate structured memory system on HaluMem')
    
    # Data paths
    parser.add_argument('--data_path', type=str, default='data/HaluMem/HaluMem-Medium.jsonl',
                        help='Path to HaluMem dataset (JSONL format)')
    parser.add_argument('--out_dir', type=str, default='checkpoints/tmp',
                        help='Output directory for results')
    parser.add_argument('--cache_dir', type=str, default='data/cache',
                        help='Cache directory for models (relative to project root)')
    
    # Memory system configuration
    parser.add_argument('--embedding_model', type=str, default=cfg.getenv('EMBEDDING_MODEL', 'contriever'),
                        choices=list(MODEL_CONFIG.keys()),
                        help='Embedding model for retrieval (overridable via env EMBEDDING_MODEL)')
    parser.add_argument('--retrieve_method', type=str, default=cfg.getenv('RETRIEVE_METHOD', 'merge'),
                        choices=['merge', 'separate', 'merge_raw'],
                        help='How to combine memory components for retrieval')
    parser.add_argument('--use_raw_session_as_key', action='store_true', default=cfg.get_bool('USE_RAW_SESSION_AS_KEY', False),
                        help='Include the raw session for memory operations and final retrieval for QA')
    parser.add_argument('--llm_model', type=str, default=cfg.get_stage_model('index', 'gpt-4o-mini'),
                        help='LLM model for memory operations and QA (overridable via env LLM_MODEL)')
    parser.add_argument('--llm_backend', type=str, default=cfg.getenv('LLM_BACKEND', 'openai'), choices=['openai'], help='LLM backend')
    parser.add_argument('--api_key', type=str, default=cfg.get_stage_api_key('index', None),
                        help='API key for LLM (can be set via env LLM_API_KEY or OPENAI_API_KEY)')
    parser.add_argument('--base_url', type=str, default=cfg.get_stage_base_url('index', None),
                        help='Base URL for LLM API (for vLLM)')
    parser.add_argument('--temperature', type=float, default=cfg.get_float('LLM_TEMPERATURE', 0.0), help='Temperature for LLM generation')
    
    # Retrieval configuration
    parser.add_argument('--top_k', type=int, default=cfg.get_int('TOP_K', 20),
                        help='Number of memories/points to retrieve for QA')
    parser.add_argument('--keep_update_note', action='store_true', default=cfg.get_bool('KEEP_UPDATE_NOTE', False),
                        help='Keep original note after update operation')
    parser.add_argument('--enable_link', action='store_true', default=cfg.get_bool('ENABLE_LINK', False),
                        help='Enable linking related memories during memory operations')
    parser.add_argument('--use_neighbour_memories', action='store_true', default=cfg.get_bool('USE_NEIGHBOUR_MEMORIES', False),
                        help='Use neighbouring memories for retrieval and operations')
    parser.add_argument('--enable_update', action='store_true', default=cfg.get_bool('ENABLE_UPDATE', False),
                        help='Enable memory update operations (create/update/skip)')
    parser.add_argument('--qa_retrieve_method', type=str, default=cfg.getenv('QA_RETRIEVE_METHOD', 'flatten'), choices=['flatten', 'default'], 
                        help='QA retrieval method, flatten all points or default to retrieve_method')
    
    # Flattened retrieval configuration
    parser.add_argument('--include_point_type', action='store_true', default=cfg.get_bool('INCLUDE_POINT_TYPE', False),
                        help='Include point type (summary/keyword/fact) in context')
    
    # QA configuration
    parser.add_argument('--qa_llm', type=str, default=cfg.get_stage_model('qa', None),
                        help='LLM model for QA (defaults to QA_LLM_MODEL/LLM_MODEL, then llm_model)')
    parser.add_argument('--qa_api_base', type=str, default=cfg.get_stage_base_url('qa', None),
                        help='API base URL for QA LLM (defaults to QA_BASE_URL/OPENAI_BASE_URL)')
    parser.add_argument('--qa_api_key', type=str, default=cfg.get_stage_api_key('qa', None),
                        help='API key for QA LLM (defaults to QA_API_KEY/OPENAI_API_KEY)')
    parser.add_argument('--skip_qa', action='store_true',
                        help='Skip QA generation, only run memory extraction and retrieval')
    
    # Processing configuration
    parser.add_argument('--version', type=str, default=cfg.getenv('VERSION', 'default'),
                        help='Version identifier for output files')
    parser.add_argument('--resume', action='store_true', default=cfg.get_bool('RESUME', False),
                        help='Resume from existing progress')
    parser.add_argument('--use_metadata_cache', action='store_true', default=cfg.get_bool('USE_METADATA_CACHE', False),
                        help='Use cached extracted metadata if available')
    parser.add_argument('--metadata_cache_dir', type=str, default=None,
                        help='Directory for metadata extraction cache')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for embedding model (auto-detect if not specified)')
    
    # Multiprocessing configuration
    parser.add_argument('--num_workers', type=int, default=cfg.get_int('NUM_WORKERS', 1),
                        help='Number of parallel workers for processing users')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")')
    
    return parser.parse_args()


def iter_jsonl(file_path: str):
    """Iterate over JSONL file yielding parsed JSON objects."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_user_name(persona_info: str) -> str:
    """Extract username from persona_info string."""
    match = re.search(r'Name:\s*(.*?); Gender:', persona_info)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError(f"No name found in persona_info: {persona_info[:100]}...")


class StructuredHaluMemEvaluator:
    """
    Evaluator for structured memory system on HaluMem.
    
    This evaluator:
    1. Processes sessions sequentially (maintaining temporal order)
    2. Extracts structured memories from each session's dialogue
    3. Memory system decides: create new, update existing, or skip
    4. For each question, retrieves relevant memories using the structured system
    5. Generates answers using retrieved context
    6. Tracks all memory operations for analysis
    """
    
    def __init__(self, args):
        self.args = args
        self.retriever: Optional[StructuredMemoryRetriever] = None
        self.qa_client: Optional[OpenAI] = None
        self._init_qa_client()
    
    def _init_qa_client(self):
        """Initialize QA LLM client."""
        if self.args.skip_qa:
            return
        
        qa_llm = self.args.qa_llm or self.args.llm_model
        qa_api_key = self.args.qa_api_key or self.args.api_key
        qa_api_base = self.args.qa_api_base or self.args.base_url
        
        if 'gpt' in qa_llm.lower():
            api_key = qa_api_key or cfg.get_stage_api_key('qa', None)
            api_base = qa_api_base or cfg.get_stage_base_url('qa', None)
        else:
            api_key = qa_api_key or cfg.get_stage_api_key('qa', None) or 'EMPTY'
            api_base = qa_api_base or cfg.get_stage_base_url('qa', 'http://localhost:8001/v1')
        
        if api_base:
            self.qa_client = OpenAI(api_key=api_key, base_url=api_base)
        else:
            self.qa_client = OpenAI(api_key=api_key)
        
        self.qa_llm = qa_llm
    
    def _init_retriever(self, device: str = None):
        """Initialize the structured memory retriever."""
        if device is None:
            device = self.args.device
            if device is None:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.retriever = StructuredMemoryRetriever(
            embedding_model=self.args.embedding_model,
            retrieve_method=self.args.retrieve_method,
            llm_model=self.args.llm_model,
            llm_backend=self.args.llm_backend,
            api_key=self.args.api_key or cfg.get_stage_api_key('index', None),
            base_url=self.args.base_url,
            temperature=self.args.temperature,
            device=device,
            cache_dir=self.args.cache_dir,
            keep_update_note=self.args.keep_update_note,
            enable_link=self.args.enable_link,
            enable_update=self.args.enable_update,
            use_raw_session_as_key=self.args.use_raw_session_as_key
        )
    
    def extract_session_metadata(
        self,
        dialogue: List[Dict],
        user_name: str,
        session_id: str,
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract structured metadata from a session's dialogue.
        
        Args:
            dialogue: List of dialogue turns
            user_name: User's name
            session_id: Session identifier
            timestamp: Session timestamp
            
        Returns:
            Dict with 'summary', 'keywords', 'facts', 'content'
        """
        return self.retriever.extract_metadata_from_dialogue(
            dialogue=dialogue,
            user_name=user_name,
            session_id=session_id,
            timestamp=timestamp,
        )
    
    def generate_answer(
        self,
        question: str,
        context: str,
        user_name: str
    ) -> str:
        """Generate answer using QA LLM."""
        if self.qa_client is None:
            raise ValueError("QA client not initialized.")
        
        prompt = PROMPT_QA_TEMPLATE.format(
            context=context,
            question=question,
            user_name=user_name
        )
        
        try:
            response = self.qa_client.chat.completions.create(
                model=self.qa_llm,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=256,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"QA generation error: {e}")
            return f"[ERROR: {str(e)}]"
    
    def _convert_fake_ids_to_original(
        self, 
        fake_ids: List[str], 
        fake_to_original: Dict[str, str]
    ) -> List[str]:
        """
        Convert fake memory IDs back to original session IDs.
        
        Args:
            fake_ids: List of fake IDs (e.g., ['a1b2c3_mem_0', 'a1b2c3_mem_1'])
            fake_to_original: Mapping from fake_id -> original_session_id
            
        Returns:
            List of original session IDs (deduplicated while preserving order)
        """
        original_ids = []
        seen = set()
        for fake_id in fake_ids:
            original_id = fake_to_original.get(fake_id, fake_id)  # fallback to fake_id if not found
            if original_id not in seen:
                original_ids.append(original_id)
                seen.add(original_id)
        return original_ids
    
    def process_user(
        self,
        user_data: Dict,
        save_path: str,
        metadata_cache_dir: str = None,
        device: str = None
    ) -> Dict:
        """
        Process a single user's data.
        
        For each session:
        1. Extract metadata (summary, keywords, facts) from dialogue
        2. Add to memory system (which decides create/update/skip)
        3. For questions in this session, retrieve and answer
        
        Args:
            user_data: User data dict with sessions
            save_path: Path to save results
            metadata_cache_dir: Optional directory for metadata cache
            device: Device to use for embedding model (e.g., 'cuda:0')
            
        Returns:
            Result dict with status and path
        """
        user_name = extract_user_name(user_data["persona_info"])
        sessions = user_data["sessions"]
        
        tmp_dir = os.path.join(save_path, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_file = os.path.join(tmp_dir, f"{user_data['uuid']}.json")
        
        # Initialize retriever if not already done
        if self.retriever is None:
            self._init_retriever(device=device)
        else:
            # Reset memory system for new user
            self.retriever = None
            self._init_retriever(device=device)
        
        new_user_data = {
            "uuid": user_data["uuid"],
            "user_name": user_name,
            "sessions": [],
            "memory_stats": {}
        }
        
        # Load cached metadata if available
        cached_metadata = None
        if metadata_cache_dir and self.args.use_metadata_cache:
            cached_metadata = self._load_metadata_cache(metadata_cache_dir, user_data['uuid'])
            if cached_metadata:
                print(f"📦 Loaded metadata cache for user {user_name}")
            else:
                print(f"⚠️ No metadata cache found for user {user_name}")
        
        # Track metadata by session for caching
        metadata_by_session = {}
        
        # Track memory operations
        memory_operations = []

        # Session ID mapping: fake_id -> original_session_id (for reverse lookup)
        # This allows us to use short IDs internally while preserving original IDs for evaluation
        fake_to_original_session_id = {}

        try:
            for session_idx, session in enumerate(tqdm(sessions, desc=f"Processing user {user_name}")):
                new_session = {
                    "memory_points": session.get("memory_points", []),
                    "dialogue": session["dialogue"]
                }
                
                # Skip generated QA sessions (distractor sessions in HaluMem-Long)
                if session.get('is_generated_qa_session', False):
                    new_session["is_generated_qa_session"] = True
                    del new_session["dialogue"]
                    del new_session["memory_points"]
                    new_user_data["sessions"].append(new_session)
                    continue
                
                # Get session metadata
                session_id = f"{user_data['uuid']}_session_{session_idx}"
                timestamp = session['start_time']
                
                # Step 1: Extract or load metadata (returns a list of metadata dicts)
                if cached_metadata and session_id in cached_metadata:
                    metadata_list = cached_metadata[session_id]
                    # Ensure it's a list (backward compatibility with old cache format)
                    if isinstance(metadata_list, dict):
                        metadata_list = [metadata_list]
                    extraction_duration_ms = 0.0
                    new_session["from_cache"] = True
                else:
                    start_time = time.time()
                    metadata_list = self.extract_session_metadata(
                        dialogue=session["dialogue"],
                        user_name=user_name,
                        session_id=session_id,
                        timestamp=timestamp,
                    )
                    extraction_duration_ms = (time.time() - start_time) * 1000
                    new_session["from_cache"] = False
                
                # Store for caching
                metadata_by_session[session_id] = metadata_list
                
                # Step 2: Add each memory point to the system (with update operations)
                # Generate short fake IDs to avoid LLM confusion with long UUIDs
                session_id_fake = str(uuid.uuid4().hex)[:6]

                start_time = time.time()
                add_results = []
                for idx, metadata in enumerate(metadata_list):
                    # Create unique memory_id for each memory point in the session
                    memory_id_fake = f"{session_id_fake}_mem_{idx}"
                    # Store reverse mapping: fake_id -> original_session_id
                    fake_to_original_session_id[memory_id_fake] = session_id

                    add_result = self.retriever.add_memory(
                        content=metadata['content'],
                        session_id=memory_id_fake,  # Use short fake ID for memory system
                        timestamp=timestamp,
                        metadata=metadata
                    )
                    add_results.append(add_result)
                add_duration_ms = (time.time() - start_time) * 1000

                # Record operations for all memory points (use original session_id for traceability)
                for add_result in add_results:
                    # Convert merged_into_ids from fake IDs to original session IDs if present
                    original_merged_into_ids = None
                    if add_result.merged_into_ids:
                        original_merged_into_ids = [
                            fake_to_original_session_id.get(mid, mid) 
                            for mid in add_result.merged_into_ids
                        ]
                    
                    memory_operations.append({
                        'session_id': session_id,  # Original session ID
                        'action': add_result.action,
                        'memory_id_fake': add_result.memory_id,  # Keep fake ID for debugging
                        'merged_into_ids': original_merged_into_ids,
                        'duration_ms': add_result.duration_ms
                    })
                
                # Store results - aggregate all metadata from the session
                new_session["extracted_metadata"] = [
                    {
                        'summary': meta.get('summary', ''),
                        'keywords': meta.get('keywords', ''),
                        'facts': meta.get('facts', [])
                    }
                    for meta in metadata_list
                ]
                
                # Determine overall memory action (prioritize: created > updated > no_action)
                actions = [r.action for r in add_results]
                if 'created' in actions:
                    overall_action = 'created'
                elif 'updated' in actions:
                    overall_action = 'updated'
                else:
                    overall_action = 'no_action'
                new_session["memory_action"] = overall_action
                new_session["extraction_duration_ms"] = extraction_duration_ms
                new_session["add_dialogue_duration_ms"] = add_duration_ms
                
                # For compatibility with HaluMem evaluation.py
                # Collect all extracted memories from all metadata points
                extracted_memories = []
                for meta in metadata_list:
                    if meta['summary']:
                        extracted_memories.append(meta['summary'])
                    if meta['keywords']:
                        extracted_memories.append(meta['keywords'])
                    facts = meta.get('facts', [])
                    if isinstance(facts, list):
                        extracted_memories.extend(facts)
                    elif facts:
                        extracted_memories.append(facts)
                new_session["extracted_memories"] = [m for m in extracted_memories if m]
                
                # Step 3: Handle memory updates for evaluation
                if self.args.enable_update:
                    for memory in new_session.get("memory_points", []):
                        if memory.get("is_update") == "True" and memory.get("original_memories"):
                            start_time = time.time()
                            
                            flattened_points, retrieved_ids = self.retriever.retrieve_flattened_points(
                                query=memory["memory_content"],
                                top_k=10,
                                use_raw_session=self.args.use_raw_session_as_key,
                                pre_flatten=(self.args.qa_retrieve_method == 'flatten'),
                                # use_neighbour_memories=self.args.use_neighbour_memories,
                            )
                            # Format as "{timestamp}: {point}" 
                            memories_from_system = [
                                f"{point['timestamp']}: {point['point']}" 
                                for point in flattened_points
                            ]
                            
                            search_duration_ms = (time.time() - start_time) * 1000
                            memory["memories_from_system"] = memories_from_system

                # Step 4: Process questions
                if "questions" in session and session["questions"]:
                    new_session["questions"] = []
                    
                    for qa in session["questions"]:
                        new_qa = copy.deepcopy(qa)
                        
                        start_time = time.time()
                        flattened_points, retrieved_ids = self.retriever.retrieve_flattened_points(
                            query=qa["question"],
                            top_k=self.args.top_k,
                            use_raw_session=self.args.use_raw_session_as_key,
                            use_neighbour_memories=self.args.use_neighbour_memories,
                            pre_flatten=(self.args.qa_retrieve_method == 'flatten'),
                        )
                        search_duration_ms = (time.time() - start_time) * 1000
                        
                        # Format context using flattened points (mem0 style with timestamps)
                        context = self.retriever.format_flattened_points_as_context(
                            flattened_points=flattened_points,
                            user_name=user_name,
                            include_point_type=self.args.include_point_type,
                        )
                        
                        # Convert fake IDs back to original session IDs for evaluation
                        original_session_ids = self._convert_fake_ids_to_original(
                            retrieved_ids[:self.args.top_k], 
                            fake_to_original_session_id
                        )
                        
                        new_qa["context"] = context
                        new_qa["retrieved_session_ids"] = original_session_ids
                        new_qa["retrieved_points"] = [
                            {k: v for k, v in p.items() if k != 'source_memory'}  # Exclude source_memory to avoid bloat
                            for p in flattened_points
                        ]
                        new_qa["search_duration_ms"] = search_duration_ms
                        
                        # Generate answer
                        if not self.args.skip_qa:
                            start_time = time.time()
                            response = self.generate_answer(
                                question=qa["question"],
                                context=context,
                                user_name=user_name
                            )
                            response_duration_ms = (time.time() - start_time) * 1000
                            
                            new_qa["system_response"] = response
                            new_qa["response_duration_ms"] = response_duration_ms
                        
                        new_session["questions"].append(new_qa)
                
                new_user_data["sessions"].append(new_session)
            
            # Add memory system stats
            new_user_data["memory_stats"] = self.retriever.get_stats()
            new_user_data["memory_operations"] = memory_operations
            
            # Save metadata cache
            if metadata_cache_dir:
                self._save_metadata_cache(metadata_cache_dir, user_data['uuid'], metadata_by_session)
            
            # Save results
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(new_user_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved user {user_name} to {tmp_file}")
            print(f"   Memory stats: {new_user_data['memory_stats']}")
            
            return {"uuid": user_data["uuid"], "status": "ok", "path": tmp_file}
            
        except Exception as e:
            error_path = os.path.join(tmp_dir, f"{user_data['uuid']}_error.log")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            print(f"❌ Error in user {user_name}: {e}")
            traceback.print_exc()
            return {"uuid": user_data["uuid"], "status": "error", "path": error_path}
    
    def _format_dialogue_content(self, dialogue: List[Dict], user_name: str) -> str:
        """Format dialogue into content string."""
        lines = []
        for turn in dialogue:
            role = turn['role'].capitalize()
            if role.lower() == 'user':
                role = user_name
            lines.append(f"{role}: {turn['content']}")
        return '\n\n'.join(lines)
    
    def _load_metadata_cache(self, cache_dir: str, uuid: str) -> Optional[Dict]:
        """Load cached metadata for a user."""
        cache_file = os.path.join(cache_dir, f"{uuid}_metadata.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load metadata cache for {uuid}: {e}")
        return None
    
    def _save_metadata_cache(self, cache_dir: str, uuid: str, metadata_by_session: Dict):
        """Save extracted metadata to cache."""
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{uuid}_metadata.json")
        
        # Convert metadata lists to be JSON serializable
        serializable = {}
        for session_id, metadata_list in metadata_by_session.items():
            # Handle both list format (new) and dict format (old)
            if isinstance(metadata_list, list):
                serializable[session_id] = [
                    {
                        'summary': meta.get('summary', ''),
                        'keywords': meta.get('keywords', ''),
                        'facts': meta.get('facts', []),
                        'content': meta.get('content', '')
                    }
                    for meta in metadata_list
                ]
            else:
                # Backward compatibility: single dict format
                serializable[session_id] = [{
                    'summary': metadata_list.get('summary', ''),
                    'keywords': metadata_list.get('keywords', ''),
                    'facts': metadata_list.get('facts', []),
                    'content': metadata_list.get('content', '')
                }]
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)


def collect_processed_uuids(save_path: str) -> set:
    """Collect already processed UUIDs from tmp directory and output file."""
    processed = set()
    
    # Check output file
    output_file = os.path.join(save_path, "structure_eval_results.jsonl")
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    processed.add(entry['uuid'])
                except:
                    continue
    
    # Check tmp directory
    tmp_dir = os.path.join(save_path, "tmp")
    if os.path.exists(tmp_dir):
        for file in os.listdir(tmp_dir):
            if file.endswith(".json") and not file.endswith("_error.log"):
                file_path = os.path.join(tmp_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if 'uuid' in data and 'sessions' in data:
                            processed.add(data['uuid'])
                except:
                    continue
    
    return processed


def consolidate_results(tmp_dir: str, output_file: str, processed_uuids: set):
    """Consolidate tmp files to output file."""
    if not os.path.exists(tmp_dir):
        return 0
    
    consolidated = 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        for file in os.listdir(tmp_dir):
            if file.endswith(".json") and not file.endswith("_error.log"):
                file_path = os.path.join(tmp_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f_in:
                        data = json.load(f_in)
                        if data['uuid'] not in processed_uuids:
                            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                            consolidated += 1
                except Exception as e:
                    print(f"⚠️ Skipped {file}: {e}")
    
    return consolidated


def print_evaluation_summary(output_file: str):
    """Print summary statistics from evaluation results."""
    if not os.path.exists(output_file):
        return
    
    total_users = 0
    total_sessions = 0
    total_questions = 0
    total_memories = 0
    action_counts = {'created': 0, 'updated': 0, 'no_action': 0}
    
    with open(output_file, 'r') as f:
        for line in f:
            try:
                user_data = json.loads(line.strip())
                total_users += 1
                
                if 'memory_stats' in user_data:
                    total_memories += user_data['memory_stats'].get('total_memories', 0)
                
                for session in user_data.get('sessions', []):
                    if not session.get('is_generated_qa_session', False):
                        total_sessions += 1
                        total_questions += len(session.get('questions', []))
                        
                        action = session.get('memory_action', 'created')
                        if action in action_counts:
                            action_counts[action] += 1
            except:
                continue
    
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total users: {total_users}")
    print(f"Total sessions: {total_sessions}")
    print(f"Total questions: {total_questions}")
    print(f"Total memories stored: {total_memories}")
    print(f"\nMemory Operations:")
    print(f"  - Created: {action_counts['created']}")
    print(f"  - Updated: {action_counts['updated']}")
    print(f"  - Skipped: {action_counts['no_action']}")
    print(f"{'='*60}\n")


def process_user_worker(args_tuple):
    """
    Worker function for multiprocessing. Each worker processes one user.
    
    Args:
        args_tuple: Tuple of (user_data, args, save_path, metadata_cache_dir, device)
    
    Returns:
        Result dict with status and path
    """
    user_data, args, save_path, metadata_cache_dir, device = args_tuple
    
    try:        
        # Set CUDA device for this worker if using GPU
        if device and device.startswith('cuda'):
            torch.cuda.set_device(device)
        
        # Create evaluator for this worker with assigned device
        evaluator = StructuredHaluMemEvaluator(args)
        
        result = evaluator.process_user(
            user_data=user_data,
            save_path=save_path,
            metadata_cache_dir=metadata_cache_dir,
            device=device  # Pass device to process_user
        )
        return result
    except Exception as e:
        # Handle exceptions gracefully
        os.makedirs(os.path.join(save_path, "tmp"), exist_ok=True)
        error_path = os.path.join(save_path, "tmp", f"{user_data['uuid']}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())
        return {"uuid": user_data["uuid"], "status": "error", "path": error_path, "error": str(e)}


def main():
    args = parse_args()

    if args.use_neighbour_memories: assert args.enable_link, "Must enable --enable_link when using --use_neighbour_memories"
    if 'gpt' in args.llm_model.lower():
        args.api_key = args.api_key or cfg.get_stage_api_key('index', None)
    
    # Setup paths
    frame = f"structure_{args.embedding_model}_{MEMORY_EXTRACTION_LLM.get(args.llm_model, args.llm_model)}"
    save_path = os.path.join(args.out_dir, f"{frame}-{args.version}")
    os.makedirs(save_path, exist_ok=True)
    
    output_file = os.path.join(save_path, "structure_eval_results.jsonl")
    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Setup metadata cache directory
    metadata_cache_dir = args.metadata_cache_dir
    if metadata_cache_dir is None:
        metadata_cache_dir = os.path.join(args.out_dir, f"metadata_cache-{MEMORY_EXTRACTION_LLM.get(args.llm_model, args.llm_model)}")
    os.makedirs(metadata_cache_dir, exist_ok=True)

    # check args
    if args.retrieve_method == 'merge_raw':
        assert args.use_raw_session_as_key, "Must use --use_raw_session_as_key with --retrieve_method merge_raw"
    
    print(f"{'='*60}")
    print(f"Structured Memory System Evaluation on HaluMem")
    print(f"{'='*60}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Retrieve method: {args.retrieve_method}")
    print(f"LLM model: {args.llm_model}")
    print(f"Top-k retrieval: {args.top_k}")
    print(f"Output directory: {save_path}")
    print(f"Metadata cache: {metadata_cache_dir}")
    if args.num_workers > 1:
        print(f"Parallel workers: {args.num_workers}")
        if args.gpu_ids:
            print(f"GPU IDs: {args.gpu_ids}")
        else:
            print(f"GPU IDs: auto-detect")
    print(f"{'='*60}\n")
    
    # Check for resume
    processed_uuids = set()
    if args.resume:
        processed_uuids = collect_processed_uuids(save_path)
        print(f"Resuming: found {len(processed_uuids)} already processed users")
        
        # Consolidate any tmp files not in output
        output_uuids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        output_uuids.add(json.loads(line.strip())['uuid'])
                    except:
                        continue
        
        new_from_tmp = processed_uuids - output_uuids
        if new_from_tmp:
            consolidated = consolidate_results(tmp_dir, output_file, output_uuids)
            print(f"Consolidated {consolidated} users from tmp to output file")
    
    # Load data
    all_user_data = list(iter_jsonl(args.data_path))
    print(f"Loaded {len(all_user_data)} users from {args.data_path}")
    
    # Filter already processed
    user_data_to_process = [
        u for u in all_user_data if u['uuid'] not in processed_uuids
    ]
    print(f"Processing {len(user_data_to_process)} users ({len(processed_uuids)} already done)")
    
    if len(user_data_to_process) == 0:
        print("All users already processed!")
        print_evaluation_summary(output_file)
        return
    
    start_time = time.time()
    
    # Determine GPU devices for parallel processing
    if args.num_workers > 1:
        if args.gpu_ids:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        elif torch.cuda.is_available():
            # Auto-detect available GPUs
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            print("⚠️ No GPUs available, falling back to CPU")
            gpu_ids = ['cpu']
        
        # Limit workers to available devices
        if len(gpu_ids) < args.num_workers and args.embedding_model != 'openai':
            print(f"⚠️ Requested {args.num_workers} workers but only {len(gpu_ids)} GPUs available")
            print(f"   Using {len(gpu_ids)} workers instead")
            args.num_workers = len(gpu_ids)
        
        # Assign devices to workers in round-robin fashion
        device_assignments = []
        for i in range(len(user_data_to_process)):
            gpu_idx = gpu_ids[i % len(gpu_ids)]
            device = f"cuda:{gpu_idx}" if isinstance(gpu_idx, int) else gpu_idx
            device_assignments.append(device)
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting parallel processing with {args.num_workers} workers")
        print(f"   GPU IDs: {gpu_ids}")
        print(f"{'='*60}\n")
        
        # Prepare arguments for workers
        worker_args = [
            (user_data, args, save_path, metadata_cache_dir, device_assignments[i])
            for i, user_data in enumerate(user_data_to_process)
        ]
        
        # Process users in parallel
        newly_processed = set()
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_user = {
                executor.submit(process_user_worker, arg): arg[0]['uuid']
                for arg in worker_args
            }
            
            # Process completed tasks as they finish
            completed = 0
            for future in as_completed(future_to_user):
                user_uuid = future_to_user[future]
                try:
                    result = future.result()
                    completed += 1
                    print(f"[{completed}/{len(user_data_to_process)}] ✅ Finished {result['uuid']} ({result['status']})")
                    
                    # Incrementally write to output file
                    if result['status'] == 'ok' and result['uuid'] not in newly_processed:
                        with open(result['path'], 'r', encoding='utf-8') as f:
                            user_result = json.load(f)
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(user_result, ensure_ascii=False) + "\n")
                        newly_processed.add(result['uuid'])
                        print(f"    📝 Written to output file")
                    elif result['status'] == 'error':
                        print(f"    ❌ Error: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"[{completed}/{len(user_data_to_process)}] ❌ Failed {user_uuid}: {e}")
                    traceback.print_exc()
    else:
        # Sequential processing (original behavior)
        print(f"\n{'='*60}")
        print(f"Processing users sequentially (use --num_workers > 1 for parallel)")
        print(f"{'='*60}\n")
        
        evaluator = StructuredHaluMemEvaluator(args)
        newly_processed = set()
        
        for i, user_data in enumerate(user_data_to_process, 1):
            result = evaluator.process_user(
                user_data, 
                save_path, 
                metadata_cache_dir=metadata_cache_dir
            )
            print(f"[{i}/{len(user_data_to_process)}] ✅ Finished {user_data['uuid']} ({result['status']})")
            
            # Incrementally write to output file
            if result['status'] == 'ok' and user_data['uuid'] not in newly_processed:
                with open(result['path'], 'r', encoding='utf-8') as f:
                    user_result = json.load(f)
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(user_result, ensure_ascii=False) + "\n")
                newly_processed.add(user_data['uuid'])
                print(f"    📝 Written to output file")
    
    # Final consolidation
    all_output_uuids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    all_output_uuids.add(json.loads(line.strip())['uuid'])
                except:
                    continue
    
    final_consolidated = consolidate_results(tmp_dir, output_file, all_output_uuids)
    if final_consolidated > 0:
        print(f"Final consolidation: added {final_consolidated} users from tmp")
    
    elapsed = time.time() - start_time
    print(f"\n✅ All done in {elapsed:.2f}s")
    print(f"✅ Final results saved to: {output_file}")
    
    # Print summary
    print_evaluation_summary(output_file)


if __name__ == "__main__":
    main()
