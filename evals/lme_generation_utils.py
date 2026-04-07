import json
import os


def default_lme_data_file(repo_root=None):
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "data", "longmemeval-cleaned", "longmemeval_s_cleaned_deduplicate.json"),
        os.path.join(repo_root, "data", "longmemeval-cleaned", "longmemeval_s_cleaned.json"),
        os.path.join(repo_root, "data", "longmemeval_s_cleaned.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def is_graph_retrieval_payload(data):
    if not isinstance(data, dict) or not data:
        return False
    first_value = next(iter(data.values()))
    return isinstance(first_value, list)


def convert_graph_retrieval_map_to_entries(graph_data, data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        question_data = json.load(f)

    merged_entries = []
    for item in question_data:
        graph_retrieval_results = graph_data.get(item["question_id"])
        if graph_retrieval_results is None:
            continue

        ranked_items = []
        for retrieval_result in graph_retrieval_results:
            if retrieval_result.get("res_type") == "chunk":
                ranked_items.append({
                    "res_type": "chunk",
                    "corpus_id": retrieval_result["chunk_id"],
                    "text": retrieval_result["content"],
                })
            else:
                ranked_items.append(retrieval_result)

        merged_item = dict(item)
        merged_item["retrieval_results"] = {
            "query": item["question"],
            "ranked_items": ranked_items,
        }
        merged_entries.append(merged_item)

    return merged_entries


def load_generation_input(in_file, data_file):
    try:
        with open(in_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if is_graph_retrieval_payload(data):
            return convert_graph_retrieval_map_to_entries(data, data_file)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    with open(in_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_expansion_map(ranked_items):
    corpusid2retvalue = {}
    expansion_dict = {
        "session-summ": "summary",
        "session-keyphrase": "keyphrases",
        "session-userfact": "user_facts",
    }

    for ret_result_entry in ranked_items:
        corpus_id = ret_result_entry.get("corpus_id")
        expansion_type = ret_result_entry.get("expansion_type")
        if not corpus_id or expansion_type in (None, "original"):
            continue

        if "text" in ret_result_entry and "content" not in ret_result_entry:
            ret_result_entry["content"] = ret_result_entry["text"]

        label = expansion_dict.get(expansion_type, "Relevant Info")
        value = ret_result_entry.get("content")
        if value in (None, ""):
            continue

        bucket = corpusid2retvalue.setdefault(corpus_id, {})
        if label in bucket and bucket[label] != value:
            bucket[label] += " " + value
        else:
            bucket[label] = value

    return corpusid2retvalue
