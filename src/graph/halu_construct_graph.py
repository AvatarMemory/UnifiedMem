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
from src.graph._utils import (
    resolve_halu_dataset_path,
    resolve_halu_graph_root,
)
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


def build_graph(
    entry_list,
    graph_root: str,
    embedding_func=None,
    entity_namespace: str | None = None,
    relation_namespace: str | None = None,
):
    # 每个用户分别建图，每个session添加到上一个图中，并保存新的储存副本
    for entry in tqdm(entry_list, desc="Building graphs"):
        user_id = entry.get("uuid")
        # 建图代码
        for session_idx in range(len(entry.get("sessions"))):
            working_dir = os.path.join(graph_root, user_id, str(session_idx))
            os.makedirs(working_dir, exist_ok=True)
            if session_idx != 0:
                prev_working_dir = os.path.join(graph_root, user_id, str(session_idx - 1))
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

            kwargs = {"working_dir": working_dir}
            if entity_namespace is not None:
                kwargs["entity_vdb_namespace"] = entity_namespace
            if relation_namespace is not None:
                kwargs["relation_vdb_namespace"] = relation_namespace
            if embedding_func is not None:
                kwargs["embedding_func"] = embedding_func
            graph_func = GraphRAG(**kwargs)

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
    parser.add_argument("--in_file", type=str, default=None,
                        help="Path to input JSONL file (defaults to DATA_DIR/HaluMem/HaluMem-Medium.jsonl or HALU_DATA_PATH)")
    parser.add_argument("--out_dir", "--graph_root", dest="graph_root", type=str, default=None,
                        help="Directory to store per-user/session graph outputs (defaults to HALU_GRAPH_ROOT/GRAPH_ROOT or repo data)")
    parser.add_argument("--embedding", type=str, default=cfg.getenv('EMBEDDING_MODEL', 'auto'),
                        help="Embedding backend to use (contriever, openai, or auto)")
    parser.add_argument("--entity_namespace", type=str, default=cfg.getenv('ENTITY_NAMESPACE', None),
                        help="Optional entity namespace override")
    parser.add_argument("--relation_namespace", type=str, default=cfg.getenv('RELATION_NAMESPACE', None),
                        help="Optional relation namespace override")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    in_file = resolve_halu_dataset_path(args.in_file)
    graph_root = resolve_halu_graph_root(args.graph_root, dataset_path=in_file)
    embedding_func = None
    entity_namespace = args.entity_namespace
    relation_namespace = args.relation_namespace

    if args.embedding == "contriever":
        embedding_func = contriever_embedding
        entity_namespace = entity_namespace or "contriever_name_entities"
    elif args.embedding == "openai":
        embedding_func = openai_embedding
        entity_namespace = entity_namespace or "openai_name_entities"
    else:
        default_emb = cfg.getenv('EMBEDDING_MODEL', None)
        if default_emb == 'contriever':
            embedding_func = contriever_embedding
            entity_namespace = entity_namespace or "contriever_name_entities"
        elif default_emb == 'openai':
            embedding_func = openai_embedding
            entity_namespace = entity_namespace or "openai_name_entities"
        else:
            embedding_func = openai_embedding
            entity_namespace = entity_namespace or "text-embedding-3-small_entities"
            relation_namespace = relation_namespace or "text-embedding-3-small_relations"

    entry_list = load_entries(in_file)
    print(f"Loaded {len(entry_list)} entries from {in_file}")
    print(f"Writing graph artifacts to {graph_root}")
    build_graph(
        entry_list,
        graph_root,
        embedding_func=embedding_func,
        entity_namespace=entity_namespace,
        relation_namespace=relation_namespace,
    )


if __name__ == "__main__":
    main()
