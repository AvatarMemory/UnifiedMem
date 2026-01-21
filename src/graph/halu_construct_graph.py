import argparse
import json
from tqdm import tqdm
import os
import sys
from pathlib import Path
import shutil

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


def build_graph(entry_list, project_root: str):
    # 每个用户分别建图，每个session添加到上一个图中，并保存新的储存副本
    for entry in tqdm(entry_list, desc="Building graphs"):
        user_id = entry.get("uuid")
        # 建图代码
        for session_idx in range(len(entry.get("sessions"))):
            working_dir = os.path.join(project_root, "data", "nc-graph_halu_mem_medium-4o-mini", user_id, str(session_idx))
            os.makedirs(working_dir, exist_ok=True)
            if session_idx != 0:
                prev_working_dir = os.path.join(project_root, "data", "nc-graph_halu_mem_medium-4o-mini", user_id, str(session_idx - 1))
                # copy previous working dir files to current dir (preserve if exists)
                try:
                    for fname in os.listdir(prev_working_dir):
                        src = os.path.join(prev_working_dir, fname)
                        dst = os.path.join(working_dir, fname)
                        if os.path.isdir(src):
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)
                except FileNotFoundError:
                    # previous dir might not exist for some users, that's fine
                    pass

            graph_func = GraphRAG(
                working_dir=working_dir,
                entity_vdb_namespace="text-embedding-3-small_entities",
                relation_vdb_namespace="text-embedding-3-small_relations",
                embedding_func=openai_embedding,
            )

            # 每个session作为一个chunk，拼接起来
            session_entry = entry.get("sessions")[session_idx]
            content = ""
            for turn in session_entry.get("dialogue", []):
                timestamp = turn.get("timestamp", "")
                role = turn.get('role', '')
                c = turn.get('content', '')
                content += f"{timestamp}{role}: {c}\n"
            chunks = [{
                "id": f"{user_id}_session{session_idx}",
                "content": content,
                "timestamp": session_entry.get("end_time"),
            }]
            graph_func.insert(session_idx, f"{user_id}_session{session_idx}", chunks)


def retrive_eval(entry_list, project_root: str):
    # (retrieval logic moved to `halu_run_retrieval.py`)
    raise NotImplementedError("Retrieval is handled by halu_run_retrieval.py")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Construct graphs for HaluMem dataset")
    parser.add_argument("--in_file", type=str, default=os.path.join(REPO_ROOT_STR, "data", "HaluMem-Medium.jsonl"),
                        help="Path to input JSONL file (default: data/HaluMem-Medium.jsonl)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    entry_list = load_entries(args.in_file)
    print(f"Loaded {len(entry_list)} entries from {args.in_file}")
    build_graph(entry_list, REPO_ROOT_STR)


if __name__ == "__main__":
    main()

