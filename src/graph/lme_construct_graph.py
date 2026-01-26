import argparse
import json
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Derive repository root from file location (repo_root/src/graph/...)
REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)

from src.graph.graphrag import GraphRAG, QueryParam
from src.graph._llm import contriever_embedding, openai_embedding
from src import config as cfg


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Construct graphs per question for LongMemEval data")
    parser.add_argument("--in_file", type=str, default=os.path.join(REPO_ROOT_STR, "data", "longmemeval_s_cleaned.json"),
                        help="Path to input JSON file with questions")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to store per-question working dirs. If not provided, prefix is chosen as 'graph_s-' or 'graph_m-' based on the input filename.")
    parser.add_argument("--embedding", type=str, default=cfg.getenv('EMBEDDING_MODEL', 'auto'),
                        help="Embedding backend to use. Use 'contriever' or 'openai' to force; 'auto' will follow EMBEDDING_MODEL/EMBEDDING_API_URL config")
    parser.add_argument("--entity_namespace", type=str, default=None,
                        help="Optional namespace for entity vector DB (defaults based on embedding)")
    return parser.parse_args(argv)


def main(args=None):
    args = parse_args(args)

    # Determine model tag early (used when computing default out_dir)
    model_tag = cfg.getenv('LLM_MODEL', 'gpt-4o-mini')

    # If out_dir not explicitly provided, decide prefix based on dataset filename
    if args.out_dir is None:
        in_basename = os.path.basename(args.in_file).lower()
        # Heuristic: filenames that contain '_s' or 'longmemeval_s' are the small/s variant
        if "_s" in in_basename or "-s" in in_basename or "longmemeval_s" in in_basename:
            prefix = "graph_s"
        else:
            prefix = "graph_m"
        args.out_dir = os.path.join(REPO_ROOT_STR, "data", f"{prefix}-{model_tag}")

    # Load entries
    with open(args.in_file, "r") as f:
        entry_list = json.load(f)

    # Choose embedding func and default namespace. If not explicitly set to
    # 'contriever'/'openai', leave embedding_func as None so GraphRAG will use
    # its dispatcher based on EMBEDDING_MODEL/EMBEDDING_API_URL.
    embedding_func = None
    default_ns = None
    if args.embedding == "contriever":
        embedding_func = contriever_embedding
        default_ns = "contriever_name_entities"
    elif args.embedding == "openai":
        embedding_func = openai_embedding
        default_ns = "openai_name_entities"

    entity_ns = args.entity_namespace or default_ns

    # Build graphs
    for entry in tqdm(entry_list, desc="Constructing graphs"):
        question_id = entry["question_id"]
        question = entry["question"]
        chunks = []

        # Include LLM model in per-question workdir name so it reflects the active model
        work_dir = os.path.join(args.out_dir, f"{question_id}-{model_tag}")
        kwargs = {"working_dir": work_dir}
        if entity_ns is not None:
            kwargs["entity_vdb_namespace"] = entity_ns
        if embedding_func is not None:
            kwargs["embedding_func"] = embedding_func

        graph_func = GraphRAG(**kwargs)

        # Each session becomes one chunk (concatenate user messages)
        for cur_sess_id, sess_entry, ts in zip(entry["haystack_session_ids"], entry["haystack_sessions"], entry["haystack_dates"]):
            content = ""
            for turn in sess_entry:
                if turn.get("role") == "user":
                    content += turn.get("content", "") + "\n"
            chunks.append({
                "id": cur_sess_id,
                "content": content,
                "timestamp": ts,
            })

        graph_func.insert(question_id, question, chunks)


if __name__ == "__main__":
    main()
