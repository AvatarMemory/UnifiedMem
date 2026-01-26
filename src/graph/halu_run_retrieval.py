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


def load_entries(in_path: str):
    entry_list = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # skip invalid lines
                continue
            entry_list.append(entry)
    return entry_list


def retrive_eval(entry_list, graph_root: str, out_path: str | None = None,
                out_dir: str | None = None,
                embedding: str = None,
                graphrag_mode: str = "entity,chunk,one-hot-expand",
                only_need_context: bool = True):
    retrieval_results = {}
    uidx = 0

    # Choose embedding function and default namespace
    embedding_func = None
    entity_namespace = None
    if embedding == "contriever":
        embedding_func = contriever_embedding
        entity_namespace = "contriever_name_entities"
    elif embedding == "openai":
        embedding_func = openai_embedding
        entity_namespace = "openai_name_entities"
    else:
        default_emb = cfg.getenv('EMBEDDING_MODEL', None)
        if default_emb == 'contriever':
            embedding_func = contriever_embedding
            entity_namespace = "contriever_name_entities"
        elif default_emb == 'openai':
            embedding_func = openai_embedding
            entity_namespace = "openai_name_entities"

    graphrag_modes = [m.strip() for m in (graphrag_mode or "").split(",") if m.strip()]

    for entry in tqdm(entry_list, desc="Retrieving for evaluation"):
        user_id = entry.get("uuid")
        uidx += 1
        for session_idx in range(len(entry.get("sessions"))):
            working_dir = os.path.join(graph_root, user_id, str(session_idx))
            kwargs = {"working_dir": working_dir}
            if entity_namespace is not None:
                kwargs["entity_vdb_namespace"] = entity_namespace
            else:
                kwargs["entity_vdb_namespace"] = "text-embedding-3-small_entities"
                kwargs["relation_vdb_namespace"] = "text-embedding-3-small_relations"
            if embedding_func is not None:
                kwargs["embedding_func"] = embedding_func

            graph_func = GraphRAG(**kwargs)

            print(f"[DEBUG]Evaluating user {uidx}, session {session_idx}")
            for question_idx in range(len(entry.get("sessions")[session_idx].get("questions", []))):
                question = entry.get("sessions")[session_idx].get("questions")[question_idx].get("question")
                query_param = QueryParam(mode="halumem", graphrag_mode=graphrag_modes, top_k=20, only_need_context=only_need_context)
                retrieval_chunks = graph_func.query(question, query_param)
                retrieval_results[f"{user_id}_session{session_idx}_question{question_idx}"] = retrieval_chunks

    # determine output path
    if out_path is None:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"graph_retrieval_results-{embedding or 'auto'}.json")
        else:
            out_path = os.path.join(graph_root, "graph_retrieval_results_top20_no_1hop_expand.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=4)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for HaluMem graphs")
    parser.add_argument("--in_file", type=str, default=os.path.join(REPO_ROOT_STR, "data", "HaluMem-Medium.jsonl"),
                        help="Path to input JSONL file (default: data/HaluMem-Medium.jsonl)")
    parser.add_argument("--out_file", type=str, default=None,
                        help="Path to write retrieval results JSON (explicit file)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to write retrieval results (will write graph_retrieval_results-<embedding>.json)")
    parser.add_argument("--embedding", type=str, default=cfg.getenv('EMBEDDING_MODEL', 'auto'),
                        help="Embedding backend to use (e.g., contriever, openai, or auto)")
    parser.add_argument("--graphrag_mode", type=str, default="entity,chunk,one-hot-expand",
                        help="Comma-separated graphrag modes to use for QueryParam")
    parser.add_argument("--only_need_context", action="store_true", default=False,
                        help="Only return context in query results (keeps behavior parity)")
    parser.add_argument("--graph_root", type=str, default=os.path.join(REPO_ROOT_STR, "data", "nc-graph_halu_mem_medium-4o-mini"),
                        help="Path to graph working directory root produced by halu_construct_graph (default: data/nc-graph_halu_mem_medium-4o-mini)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    entry_list = load_entries(args.in_file)
    print(f"Loaded {len(entry_list)} entries from {args.in_file}")
    retrive_eval(entry_list,
                args.graph_root,
                out_path=args.out_file,
                out_dir=args.out_dir,
                embedding=args.embedding,
                graphrag_mode=args.graphrag_mode,
                only_need_context=args.only_need_context)


if __name__ == "__main__":
    main()
