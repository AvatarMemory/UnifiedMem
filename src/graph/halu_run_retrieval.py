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


def retrive_eval(entry_list, project_root: str, out_path: str | None = None):
    retrieval_results = {}
    uidx = 0
    for entry in tqdm(entry_list, desc="Retrieving for evaluation"):
        user_id = entry.get("uuid")
        uidx += 1
        for session_idx in range(len(entry.get("sessions"))):
            working_dir = os.path.join(project_root, "data", "nc-graph_halu_mem_medium-4o-mini", user_id, str(session_idx))
            graph_func = GraphRAG(
                working_dir=working_dir,
                entity_vdb_namespace="text-embedding-3-small_entities",
                relation_vdb_namespace="text-embedding-3-small_relations",
                embedding_func=openai_embedding,
            )
            print(f"[DEBUG]Evaluating user {uidx}, session {session_idx}")
            for question_idx in range(len(entry.get("sessions")[session_idx].get("questions", []))):
                question = entry.get("sessions")[session_idx].get("questions")[question_idx].get("question")
                query_param = QueryParam(mode="halumem", graphrag_mode=["entity", "chunk", "remove-user-node", "rank-entity"], top_k=20, only_need_context=True)
                retrieval_chunks = graph_func.query(question, query_param)
                retrieval_results[f"{user_id}_session{session_idx}_question{question_idx}"] = retrieval_chunks

    if out_path is None:
        out_path = os.path.join(project_root, "data", "nc-graph_halu_mem_medium-4o-mini", "graph_retrieval_results_top20_no_1hop_expand.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=4)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for HaluMem graphs")
    parser.add_argument("--in_file", type=str, default=os.path.join(REPO_ROOT_STR, "data", "HaluMem-Medium.jsonl"),
                        help="Path to input JSONL file (default: data/HaluMem-Medium.jsonl)")
    parser.add_argument("--out_file", type=str, default=None,
                        help="Path to write retrieval results JSON (default under data/nc-graph_halu_mem_medium-4o-mini)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    entry_list = load_entries(args.in_file)
    print(f"Loaded {len(entry_list)} entries from {args.in_file}")
    retrive_eval(entry_list, REPO_ROOT_STR, out_path=args.out_file)


if __name__ == "__main__":
    main()
