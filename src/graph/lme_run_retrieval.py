import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Compute repo root from this file's location
REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)

from src.graph.graphrag import GraphRAG, QueryParam
from src.graph._llm import contriever_embedding, openai_embedding
from src import config as cfg


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run graph-based retrieval for LongMemEval data")
    parser.add_argument("--in_file", type=str, default=os.path.join(REPO_ROOT_STR, "data", "longmemeval_s_cleaned.json"),
                        help="Path to input JSON file with questions")
    parser.add_argument("--out_dir", type=str, default=os.path.join(REPO_ROOT_STR, "data", f"graph_m-{cfg.get_stage_model('index', 'gpt-4o-mini')}") ,
                        help="Directory to store per-question working dirs and final output")
    parser.add_argument("--embedding", type=str, default=cfg.getenv('EMBEDDING_MODEL', 'auto'),
                        help="Embedding backend to use. Use 'contriever' or 'openai' to force; 'auto' (default) will follow EMDEDING_MODEL/EMBEDDING_API_URL config")
    parser.add_argument("--only_need_context", action="store_true", default=False,
                        help="Only return context in query results (keeps behavior parity)")
    parser.add_argument("--graphrag_mode", type=str, default="entity,chunk,one-hot-expand",
                        help="Comma-separated graphrag modes to use for QueryParam")
    return parser.parse_args(argv)


def main(args=None):
    args = parse_args(args)

    # Load entries
    with open(args.in_file, "r") as f:
        entry_list = json.load(f)

    retrieval_results = {}

    # Choose embedding function and namespace.
    # If user explicitly requests 'contriever' or 'openai', set them; otherwise
    # leave embedding_func as None so GraphRAG will use its default dispatcher.
    embedding_func = None
    entity_namespace = None
    if args.embedding == "contriever":
        embedding_func = contriever_embedding
        entity_namespace = "contriever_name_entities"
    elif args.embedding == "openai":
        embedding_func = openai_embedding
        entity_namespace = "openai_name_entities"

    # Prepare graphrag_mode list
    graphrag_mode = [m.strip() for m in args.graphrag_mode.split(",") if m.strip()]

    # Iterate and build graph per question
    for entry in tqdm(entry_list, desc="Processing questions"):
        question_id = entry["question_id"]
        if question_id in retrieval_results:
            continue

        question = entry["question"]

        # Include LLM model in per-question workdir name so it reflects the active model
        model_tag = cfg.get_stage_model('index', 'gpt-4o-mini')
        work_dir = os.path.join(args.out_dir, f"{question_id}-{model_tag}")
        kwargs = {"working_dir": work_dir}
        if entity_namespace is not None:
            kwargs["entity_vdb_namespace"] = entity_namespace
        if embedding_func is not None:
            kwargs["embedding_func"] = embedding_func

        graph_func = GraphRAG(**kwargs)

        retrieval_results[question_id] = graph_func.query(
            question,
            param=QueryParam(mode="longmemeval", graphrag_mode=graphrag_mode, only_need_context=args.only_need_context),
        )

    # Write output
    out_fname = f"graph_retrieval_results-{args.embedding}.json"
    retrieval_path = os.path.join(args.out_dir, out_fname)
    os.makedirs(os.path.dirname(retrieval_path), exist_ok=True)
    with open(retrieval_path, "w") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
