from __future__ import annotations

import os
import re
import time
import json
import logging
import sys
import argparse
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, List, Tuple, Dict
import xml.etree.ElementTree as ET
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log
from tqdm import tqdm
import copy
from dotenv import load_dotenv
from .halu_eval_utils import llm_request

script_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
repo_env = os.path.join(repo_root, '.env')
local_env = os.path.join(script_dir, '.env')

if os.path.exists(repo_env):
    # load repo defaults, do not override existing env
    load_dotenv(repo_env, override=False)

if os.path.exists(local_env):
    # load local overrides and allow them to override repo values
    load_dotenv(local_env, override=True)

# --- Environment-backed configuration (provide sensible defaults) ---
# Retry / wait settings
RETRY_TIMES = int(os.getenv('RETRY_TIMES', '3'))
WAIT_TIME_LOWER = int(os.getenv('WAIT_TIME_LOWER', '10'))
WAIT_TIME_UPPER = int(os.getenv('WAIT_TIME_UPPER', '30'))

# OpenAI / LLM settings
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', os.getenv('LLM_BASE_URL', None))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', os.getenv('MODEL', 'gpt-4o'))

# Optional tuning params
OPENAI_MAX_TOKENS = os.getenv('OPENAI_MAX_TOKENS')
OPENAI_TEMPERATURE = os.getenv('OPENAI_TEMPERATURE', '0.0')
OPENAI_TIMEOUT = os.getenv('OPENAI_TIMEOUT', '300')

# Project root (fallback to repo root)
PROJECT_ROOT = os.getenv('PROJECT_ROOT', repo_root)

logger = logging.getLogger(__name__)

common_params = {}
common_params["max_tokens"] = int(OPENAI_MAX_TOKENS)
common_params["temperature"] = float(OPENAI_TEMPERATURE)
common_params["timeout"] = int(OPENAI_TIMEOUT)

client = OpenAI(
    base_url=OPENAI_BASE_URL or None,
    api_key=OPENAI_API_KEY or None,
)

# Optional shared prompt template used by adapters
PROMPT_ANSWER_COMPLEX = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. 

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    {context}

    Question: {question}

    Answer:
"""

PROMPT_ANSWER_SIMPLE = """
   You are a knowledgeable and helpful AI assistant.

   {context}

   Question: {question}

   Answer:
"""

PROMPT_ANSWER = """
    You are an Intelligent Memory Assistant. Your task is to precisely retrieve and infer information from entity descriptions provided by speakers in a conversation.    

    # CONTEXT:
    You have access to entities descriptions from two speakers in a conversation. 

    # INSTRUCTIONS:
    1. Carefully analyze all provided entities from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the descriptions
    4. If the descriptions contain contradictory information, prioritize the most recent description
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the description timestamp. For example, if a description from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the description
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the descriptions from both speakers. Do not confuse character
       names mentioned in descriptions with the actual users who created those descriptions.
    8. The answer should be less than 1 sentence, please pay careful attention to contain the necessary information for answering question.

    # APPROACH (Think step by step):
    1. First, examine all descriptions that contain information related to the question
    2. Examine the timestamps and content of these descriptions carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the descriptions
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    {context}

    Question: {question}

    Answer:
"""

def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return default if default is not None else {}
    except json.JSONDecodeError as e:
        print(f"JSON decode error for {path}: {e}")
        return default if default is not None else {}

def extract_user_name(persona_info: str):
    match = re.search(r'Name:\s*(.*?); Gender:', persona_info)

    if match:
        username = match.group(1).strip()
        return username
    else:
        raise ValueError("No name found.")

def test_llm_request():
    prompt = "Tell me a joke about computers."
    response = llm_request(prompt)
    print("LLM Response:", response)

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(process)d %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def extract_user_name(persona_info: str) -> str:
    match = re.search(r'Name:\s*(.*?); Gender:', persona_info)
    if match:
        return match.group(1).strip()
    raise ValueError("No name found in persona_info")


def iter_jsonl(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def make_context_from_rawcontext_chunk(raw_context: List[dict], top_k: int = 5) -> str:
    """
    Given a raw context (e.g., from retrieval), format it into a
    structured context block for prompting based on top-k chunk context.

    Args:
        raw_context (List[dict]): The raw context list of dictionaries.
    """
    context_parts = []
    num = 0
    filtered_chunks = []
    for i, item in enumerate(raw_context):
        if raw_context[i].get('res_type', '') == 'chunk':
            num += 1
            filtered_chunks.append(item)
        if num >= top_k:
            break

    for i, item in enumerate(filtered_chunks, start=1):
        chunk_id = item.get('chunk_id', 'N/A')
        content = item.get('content', 'No Content')
        context_parts.append(f"Session {chunk_id}:\n{content}\n")
    return "\n".join(context_parts)

def make_context_from_rawcontext_entity(raw_context: List[dict], top_k: int = 20) -> str:
    """
    Given a raw context (e.g., from retrieval), format it into a
    structured context block for prompting based on top-k chunk context.

    Args:
        raw_context (List[dict]): The raw context list of dictionaries.
    """
    context_parts = []
    num = 0
    filtered_chunks = []
    for i, item in enumerate(raw_context):
        if raw_context[i].get('res_type', '') == 'entity':
            num += 1
            filtered_chunks.append(item)
        if num >= top_k:
            break

    for i, item in enumerate(filtered_chunks, start=1):
        entity_name = item.get('entity_name', 'N/A')
        entity_type = item.get('entity_type', 'N/A')
        content = item.get('content', 'No Content')
        context_parts.append(f"entity: {entity_name} type: ({entity_type}):\n{content}\n")
    return "\n".join(context_parts)


### compute add_memory contents for evaluating
def compute_addmemory():
    filepath = os.path.join(PROJECT_ROOT, "data/nc-graph_halu_mem_medium-4o-mini")
    """
    For each user (a directory under `filepath`), pick the session subdirectory with the
    largest numeric name. Parse that session's GraphML files (e.g. `graph_chunk_entity_relation.graphml`),
    read the `<key>` definitions to map `dX` -> attribute names, then iterate nodes/edges and
    collect entries that reference other `session_id` values in their `<data>` text (values like
    `1b846c59-..._session0`). Group those entries by the referenced `session_id` and emit a JSON
    mapping:

      { "<session_id>": [ {res_type, entity_name, entity_type, description}, ... ], ... }

    Only the chosen (largest) session per user is parsed because it should contain all results.
    """

    out_file = os.path.join(filepath, "add_memory_by_session.json")
    sessions: Dict[str, List[Dict[str, Any]]] = {}

    if not os.path.isdir(filepath):
        logger.warning("compute_addmemory: filepath %s does not exist or is not a directory", filepath)
        return {}

    # Helper to parse a GraphML file and collect entries grouped by referenced session_id
    def parse_graphml_and_group(fpath: str) -> Dict[str, List[Dict[str, Any]]]:
        try:
            tree = ET.parse(fpath)
            xml_root = tree.getroot()
        except Exception as e:
            logger.warning("Failed to parse GraphML %s: %s", fpath, e)
            return {}
        # Helper to strip namespace and get localname
        def local(tag: str) -> str:
            return tag.split('}', 1)[-1] if '}' in tag else tag

        # Build key id -> attr.name mapping from <key> elements (namespace-agnostic)
        key_id_to_name: Dict[str, str] = {}
        for elem in xml_root.iter():
            if local(elem.tag) == 'key':
                kid = elem.attrib.get('id')
                # attr.name may be present literally as 'attr.name'
                name = elem.attrib.get('attr.name') or elem.attrib.get('for') or kid
                if kid:
                    key_id_to_name[kid] = name

        grouped: Dict[str, List[Dict[str, Any]]] = {}


        # Consider node and edge elements irrespective of namespace
        for elem in xml_root.iter():
            tag_local = local(elem.tag)
            if tag_local not in ('node', 'edge'):
                continue

            # small helper to normalize text (strip whitespace and surrounding quotes)
            def clean_text(s: str) -> str:
                if s is None:
                    return ''
                s = s.strip()
                if len(s) >= 2 and ((s[0] == s[-1] and s[0] in ('"', "'"))):
                    s = s[1:-1].strip()
                return s

            # collect data key->value mapping for this element
            data_map: Dict[str, str] = {}
            for child in elem:
                if local(child.tag) == 'data':
                    k = child.attrib.get('key')
                    mapped = key_id_to_name.get(k, k)
                    data_map[mapped] = clean_text(child.text or '')

            # find any values that look like session_id (e.g., uuid_session<number>)
            session_pattern = re.compile(r'[0-9a-fA-F\-]{8,}_[sS]ession\d+')
            referenced_sids = set()
            for v in data_map.values():
                if not v:
                    continue
                for m in session_pattern.findall(v):
                    referenced_sids.add(m)

            if not referenced_sids:
                continue

            # Determine res_type and entity_name from element attributes
            if tag_local == 'node':
                res_type_val = 'entity'
                raw_id = elem.attrib.get('id', '')
                entity_name_val = clean_text(raw_id)
            else:  # edge
                res_type_val = 'relation'
                src = clean_text(elem.attrib.get('source', ''))
                tgt = clean_text(elem.attrib.get('target', ''))
                # if source/target include quotes like "NAME", clean_text removed them
                if src and tgt:
                    entity_name_val = f"{src}->{tgt}"
                else:
                    # fallback to a description if available
                    entity_name_val = data_map.get('description', '')

            # entity_type and description from data_map if present
            entity_type_val = data_map.get('entity_type', '')
            description_val = data_map.get('description', '')

            # Build output entry mapping to requested fields; allow empty strings
            for sid in referenced_sids:
                entry = {
                    'res_type': res_type_val,
                    'entity_name': entity_name_val,
                    'entity_type': entity_type_val,
                    'description': description_val
                }
                grouped.setdefault(sid, []).append(entry)

        return grouped

    # Iterate users (directories directly under filepath)
    for user_name in os.listdir(filepath):
        user_dir = os.path.join(filepath, user_name)
        if not os.path.isdir(user_dir):
            continue

        # find numeric session subdirectories and choose the max
        max_session = None
        max_session_int = -1
        for item in os.listdir(user_dir):
            item_path = os.path.join(user_dir, item)
            if not os.path.isdir(item_path):
                continue
            # session directories are numeric (e.g., '56')
            if item.isdigit():
                val = int(item)
                if val > max_session_int:
                    max_session_int = val
                    max_session = item_path

        if max_session is None:
            logger.debug("No numeric session dirs for user %s, skipping", user_name)
            continue

        # find graphml files under max_session (recursively)
        user_grouped: Dict[str, List[Dict[str, Any]]] = {}
        for root_dir, _, files in os.walk(max_session):
            for fname in files:
                if not fname.lower().endswith('.graphml'):
                    continue
                fpath = os.path.join(root_dir, fname)
                logger.info("Parsing GraphML for user %s from %s", user_name, fpath)
                grouped = parse_graphml_and_group(fpath)
                # merge grouped into user_grouped
                for sid, entries in grouped.items():
                    user_grouped.setdefault(sid, []).extend(entries)

        # Merge user_grouped into global sessions map
        for sid, entries in user_grouped.items():
            sessions.setdefault(sid, []).extend(entries)

    # Write out json
    try:
        with open(out_file, 'w', encoding='utf-8') as wf:
            json.dump(sessions, wf, ensure_ascii=False, indent=2)
        logger.info("Wrote add_memory JSON to %s (%d sessions)", out_file, len(sessions))
    except Exception as e:
        logger.warning("Failed to write add_memory json %s: %s", out_file, e)

    return sessions

# ### compute retrieve memory content
# def compute_retrievememory():
#     retrieve_file_path = os.path.join(PROJECT_ROOT,"data/nc-graph_halu_mem_medium-4o-mini/graph_retrieval_results_entity-chunk-1hop.json")
#     res_filepath = retrieve_file_path.replace(".json", "_grouped_by_question.json")
#     # it seems the retrieve_file_path already satisfying requirements
#     return None
    

### gen file computing metrics
def gen_eval_file(
    retrieve_file_name: str = "graph_retrieval_results_entity-chunk-1hop-top20.json",
    use_entity: bool = False,
    out_suffix: str = "_test_eval_results.jsonl",
    dataset_data_filename: str = "HaluMem-Medium.jsonl",
):
    """
    Generate an offline evaluation file that merges add-memory data (computed by
    compute_addmemory or prewritten add_memory_by_session.json) with retrieval
    results.

    Parameters:
      - retrieve_file_name: filename under the dataset folder to read retrieval results from.
      - use_entity: if True, use `make_context_from_rawcontext_entity`, otherwise use chunk builder.
      - out_suffix: suffix to append to the retrieval filename to build the output path.
      - dataset_data_filename: the JSONL dataset filename under `data/`.

    Output:
      - Writes a JSONL file next to the retrieval input file using `out_suffix`.
    """
    # modify there to adjust base data file paths
    filepath = os.path.join(PROJECT_ROOT, "data/nc-graph_halu_mem_medium-4o-mini")
    add_mem_file = os.path.join(filepath, "add_memory_by_session.json")
    retrieve_file_path = os.path.join(filepath, retrieve_file_name)
    # Create result filepath by replacing .json with provided suffix; fallback if not ending with .json
    if retrieve_file_path.lower().endswith(".json"):
        res_filepath = retrieve_file_path.replace(".json", out_suffix)
    else:
        res_filepath = retrieve_file_path + out_suffix

    dataset_data_filepath = os.path.join(PROJECT_ROOT, "data", dataset_data_filename)

    # read data
    add_mem = read_json(add_mem_file, default={})
    retrieval = read_json(retrieve_file_path, default={})
    dataset = []
    with open(dataset_data_filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSONL line {i} in {dataset_data_filepath}: {e}")

    # read user info from dataset_data
    # results: List[Dict[str, Any]] = []  # collect all users' processed data
    for user_data in dataset:
        user_name = extract_user_name(user_data["persona_info"])
        user_name = user_name.replace(" ", "_")
        sessions = user_data["sessions"]

        new_user_data = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "sessions": []
        }
        session_id = -1
        for session in tqdm(sessions, total=len(sessions), desc=f"Processing user {user_name}"):
            session_id += 1
            sid = f"{user_data['uuid']}_session{session_id}"
            new_session = {"memory_points": session["memory_points"], "dialogue": session["dialogue"]}
            if session.get('is_generated_qa_session', False):
                new_session["add_dialogue_duration_ms"] = 1 # 为了兼容性
                new_session["is_generated_qa_session"] = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue
            new_session["extracted_memories"] = add_mem.get(sid, [])
            new_session["add_dialogue_duration_ms"] = 1

            # we do not calulate memory int
            for memory in new_session["memory_points"]:
                if memory["is_update"] == "False" or not memory["original_memories"]:
                    continue
                # use hippo to get top matching facts
                # context_str, matches, search_dur = search_memory(hippo=hippo, query=memory["memory_content"], user_id=user_name, top_k=10)
                memory["memories_from_system"] = []

            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue
                
            new_session["questions"] = []
            for qd, qa in enumerate(session["questions"]):
                # retrieve top_k docs as context
                # print(f"🔍 Retrieving for question: {qa['question']}")
                qid = f"{sid}_question{qd}"
                context_raw_str = retrieval.get(qid, {})

                new_qa = copy.deepcopy(qa)
                # context_str already contains a formatted TEMPLATE_MEM0_GRAPH string
                # choose context builder based on `use_entity`
                if use_entity:
                    new_qa["context"] = make_context_from_rawcontext_entity(context_raw_str)
                else:
                    new_qa["context"] = make_context_from_rawcontext_chunk(context_raw_str)
                new_qa["search_duration_ms"] = 1

                prompt_template = globals().get('PROMPT_ANSWER', None)
                if not prompt_template:
                    # fallback minimal template
                    prompt_template = "{context}\nQuestion: {question}\nAnswer:"

                prompt = prompt_template.format(context=new_qa["context"], question=qa["question"])

                start_time = time.time()
                response = llm_request(prompt)
                new_qa["system_response"] = response
                new_qa["response_duration_ms"] = (time.time() - start_time) * 1000

                new_session["questions"].append(new_qa)
            new_user_data["sessions"].append(new_session)
        with open(res_filepath, "a", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False)
            f.write("\n")


### evaluating script

if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Utility runner for halu_eval_graph tasks")
    parser.add_argument('--mode', type=str, choices=['gen_eval', 'add_memory', 'retrieve_memory', 'test_llm'], default='gen_eval',
                        help='Which operation to run (default: gen_eval)')
    parser.add_argument('--retrieve_file_name', type=str, default="graph_retrieval_results_entity-chunk-1hop-top20.json")
    parser.add_argument('--use_entity', action='store_true', help='Use entity-based context builder')
    parser.add_argument('--out_suffix', type=str, default="_test_eval_results.jsonl")
    parser.add_argument('--dataset_data_filename', type=str, default="HaluMem-Medium.jsonl")

    args = parser.parse_args()

    if args.mode == 'test_llm':
        test_llm_request()
    elif args.mode == 'add_memory':
        compute_addmemory()
    # elif args.mode == 'retrieve_memory':
    #     compute_retrievememory()
    else:
        gen_eval_file(
            retrieve_file_name=args.retrieve_file_name,
            use_entity=args.use_entity,
            out_suffix=args.out_suffix,
            dataset_data_filename=args.dataset_data_filename,
        )