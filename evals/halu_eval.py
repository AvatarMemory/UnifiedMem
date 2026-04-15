"""Adapt from Halumem/eval/evaluation.py"""

import os
import json
import time
import copy
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

if __package__ is None and __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src import config as cfg


from evals.halu_eval_utils import (
    evaluation_for_memory_accuracy, 
    evaluation_for_memory_integrity, 
    evaluation_for_question, 
    evaluation_for_update_memory
)

def print_scores(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    overall = data["overall_score"]
    
    print("MEMORY INTEGRITY")
    for key, value in overall["memory_integrity"].items():
        if isinstance(value, (int, float)):
            print(f"{key:>40}: {value:.4f}")
        else:
            print(f"{key:>40}: {value}")
    
    print("\n")
    print("MEMORY ACCURACY")
    for key, value in overall["memory_accuracy"].items():
        if isinstance(value, (int, float)):
            print(f"{key:>40}: {value:.4f}")
        else:
            print(f"{key:>40}: {value}")

    if 'memory_update' in overall:
        print("\n")
        print("MEMORY UPDATE")
        for key, value in overall["memory_update"].items():
            if isinstance(value, (int, float)):
                print(f"{key:>40}: {value:.4f}")
            else:
                print(f"{key:>40}: {value}")
    
    print("\n")
    print("QUESTION ANSWERING")
    for key, value in overall["question_answering"].items():
        if isinstance(value, (int, float)):
            print(f"{key:>40}: {value:.4f}")
        else:
            print(f"{key:>40}: {value}")
    
    print("\n")
    print("MEMORY TYPE ACCURACY")
    for mem_type, scores in overall["memory_type_accuracy"].items():
        print(f"\n{mem_type}:")
        for key, value in scores.items():
            if isinstance(value, (int, float)):
                print(f"  {key:>38}: {value:.4f}")
            else:
                print(f"  {key:>38}: {value}")
    
    print("\n")
    print("TIME CONSUMING (minutes)")
    for key, value in overall["time_consuming"].items():
        print(f"{key:>40}: {value:.4f}")


def process_user(idx: int, user_data: dict, max_workers: int = 10):
    uuid = user_data["uuid"]
    user_name = user_data["user_name"]

    eval_results = {
        "memory_integrity_records":[],
        "memory_accuracy_records": [],
        "memory_update_records": [],
        "question_answering_records": []
    }
    
    memory_integrity_inputs = []
    memory_accuracy_inputs = []
    memory_update_inputs = []
    question_answering_inputs = []

    for sid, session in enumerate(user_data["sessions"]):

        if session.get('is_generated_qa_session', False):
            continue

        golden_memories = session["memory_points"]
        extract_memories = session["extracted_memories"]

        # Normalize extracted memories:
        # - non-graph: list of strings
        # - graph: list of dicts -> convert each dict to a JSON string
        processed_extract_memories = []
        for m in extract_memories:
            if isinstance(m, dict):
                try:
                    s = json.dumps(m, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    s = str(m)
            elif isinstance(m, list):
                # join list elements into a single string
                s = ";".join([str(x) for x in m])
            else:
                s = str(m)
            processed_extract_memories.append(s)

        extract_memories = processed_extract_memories
        extract_memories_str = "\n".join(extract_memories)
        for memory in golden_memories:
            if memory["is_update"] == "True" and memory.get("memories_from_system", []):
                new_update_memory = copy.deepcopy(memory)
                new_update_memory["uuid"] = uuid
                new_update_memory["ssession_id"] = sid
                memory_update_inputs.append(new_update_memory)
            else:
                new_memory = copy.deepcopy(memory)
                new_memory["uuid"] = uuid
                new_memory["ssession_id"] = sid
                memory_integrity_inputs.append(
                    (new_memory, extract_memories_str)
                )

        dialogue = session["dialogue"]
        dialogue_str = []
        for turn in dialogue:
            dialogue_str.append(
                f'[{turn["timestamp"]}]{turn["role"]}: {turn["content"]}'
            )
            if turn["role"] == "assistant":
                dialogue_str.append("")
        
        dialogue_str = "\n".join(dialogue_str)

        golden_memories_str = "\n".join(
            [m["memory_content"] for m in golden_memories if m["memory_source"] != "interference"]
        )

        for memory in extract_memories:
            new_memory = {
                "uuid": uuid,
                "ssession_id": sid,
                "memory_content": memory
            }
            memory_accuracy_inputs.append(
                (dialogue_str, golden_memories_str, new_memory)
            )

        if "questions" in session:
            for qa in session["questions"]:
                new_qa = copy.deepcopy(qa)
                new_qa["uuid"] = uuid
                new_qa["ssession_id"] = sid
                question_answering_inputs.append(new_qa)

    # Memory Integrity Evaluation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for memory, extract_memories_str in memory_integrity_inputs:
            if extract_memories_str.strip() == "":
                memory["memory_integrity_score"] = 0
                eval_results["memory_integrity_records"].append(memory)
                continue
            future = executor.submit(
                evaluation_for_memory_integrity,
                extract_memories_str,
                memory["memory_content"]
            )
            futures[future] = memory

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Memory Integrity Evaluation([{idx}]{user_name})"):
            memory = futures[future]
            try:
                result = future.result()
                score = int(result.get("score"))
            except Exception as e:
                score = None
            memory["memory_integrity_score"] = score
            eval_results["memory_integrity_records"].append(memory)

    # Memory Accuracy Evaluation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for dialogue_str, golden_memories_str, memory in memory_accuracy_inputs:
            future = executor.submit(
                evaluation_for_memory_accuracy,
                dialogue_str,
                golden_memories_str,
                memory["memory_content"]
            )
            futures[future] = memory

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Memory Accuracy Evaluation([{idx}]{user_name})"):
            memory = futures[future]
            try:
                result = future.result()
                score = int(result.get("accuracy_score"))
                is_included_in_golden_memories = result.get("is_included_in_golden_memories", "false")
            except Exception as e:
                score = None
                is_included_in_golden_memories = "false"
            memory["memory_accuracy_score"] = score
            memory["is_included_in_golden_memories"] = is_included_in_golden_memories
            eval_results["memory_accuracy_records"].append(memory)

    # Memory Update Evaluation
    if memory_update_inputs:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for update_memory in memory_update_inputs:
                future = executor.submit(
                    evaluation_for_update_memory,
                    "\n".join(update_memory["memories_from_system"]),
                    update_memory["memory_content"],
                    "\n".join(update_memory["original_memories"])
                )
                futures[future] = update_memory

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Memory Update Evaluation([{idx}]{user_name})"):
                update_memory = futures[future]
                try:
                    result = future.result()
                    update_type = result.get("evaluation_result")
                except Exception as e:
                    update_type = None
                update_memory["memory_update_type"] = update_type
                eval_results["memory_update_records"].append(update_memory)

    # Question-Answering Evaluation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for qa in question_answering_inputs:
            future = executor.submit(
                evaluation_for_question,
                qa["question"], 
                qa["answer"], 
                "\n".join([i["memory_content"] for i in qa["evidence"]]), 
                qa["system_response"]
            )
            futures[future] = qa

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Question-Answering Evaluation([{idx}]{user_name})"):
            qa = futures[future]
            try:
                result = future.result()
                result_type = result.get("evaluation_result")
            except Exception as e:
                result_type = None
            qa["result_type"] = result_type
            eval_results["question_answering_records"].append(qa)

    return eval_results


def aggregate_eval_results(eval_results):

    # Memory Integrity Evaluation
    memory_integrity_scores = 0
    memory_integrity_weighted_scores = 0
    memory_integrity_valid_num = 0
    memory_integrity_num = 0
    memory_integrity_weighted_valid_num = 0
    memory_integrity_weighted_num = 0
    interference_memory_scores = 0
    interference_memory_valid_num = 0
    interference_memory_num = 0
    for item in eval_results["memory_integrity_records"]:

        item["is_valid"] = True

        if item["memory_source"] != "interference":
            memory_integrity_num += 1
            memory_integrity_weighted_num += item["importance"]
        else:
            interference_memory_num += 1

        if item["memory_integrity_score"] is None:
            item["is_valid"] = False
            continue
        
        if item["memory_source"] != "interference":
            if item["memory_integrity_score"] == 2:
                memory_integrity_scores += 1
            memory_integrity_weighted_scores += 0.5 * item["memory_integrity_score"] * item["importance"]
            memory_integrity_valid_num += 1
            memory_integrity_weighted_valid_num += item["importance"]
        else:
            if item["memory_integrity_score"] == 0:
                interference_memory_scores += 1
            interference_memory_valid_num += 1

    eval_results["overall_score"]["memory_integrity"]["recall(all)"] = safe_div(memory_integrity_scores, memory_integrity_num)
    eval_results["overall_score"]["memory_integrity"]["recall(valid)"] = safe_div(memory_integrity_scores, memory_integrity_valid_num)
    eval_results["overall_score"]["memory_integrity"]["weighted_recall(all)"] = safe_div(memory_integrity_weighted_scores, memory_integrity_weighted_num)
    eval_results["overall_score"]["memory_integrity"]["weighted_recall(valid)"] = safe_div(memory_integrity_weighted_scores, memory_integrity_weighted_valid_num)
    eval_results["overall_score"]["memory_integrity"]["memory_valid_importance_sum"] = memory_integrity_weighted_valid_num
    eval_results["overall_score"]["memory_integrity"]["memory_importance_sum"] = memory_integrity_weighted_num
    eval_results["overall_score"]["memory_integrity"]["memory_valid_num"] = memory_integrity_valid_num
    eval_results["overall_score"]["memory_integrity"]["memory_num"] = memory_integrity_num
    eval_results["overall_score"]["memory_accuracy"]["interference_accuracy(all)"] = safe_div(interference_memory_scores, interference_memory_num)
    eval_results["overall_score"]["memory_accuracy"]["interference_accuracy(valid)"] = safe_div(interference_memory_scores, interference_memory_valid_num)
    eval_results["overall_score"]["memory_accuracy"]["interference_memory_valid_num"] = interference_memory_valid_num
    eval_results["overall_score"]["memory_accuracy"]["interference_memory_num"] = interference_memory_num

    # Memory Accuracy Evaluation
    target_memory_accuracy_scores = 0
    memory_accuracy_weighted_scores = 0
    target_memory_accuracy_valid_num = 0
    target_memory_accuracy_num = 0
    memory_accuracy_valid_num = 0
    memory_accuracy_num = 0
    for item in eval_results["memory_accuracy_records"]:

        item["is_valid"] = True

        memory_accuracy_num += 1

        if item["is_included_in_golden_memories"] in ["true", "True"]:
            target_memory_accuracy_num += 1

        if item["memory_accuracy_score"] is None:
            item["is_valid"] = False
            continue
        
        if item["is_included_in_golden_memories"] in ["true", "True"]:
            target_memory_accuracy_scores += 0.5 * item["memory_accuracy_score"]
            target_memory_accuracy_valid_num += 1
        
        memory_accuracy_weighted_scores += 0.5 * item["memory_accuracy_score"]
        memory_accuracy_valid_num += 1

    eval_results["overall_score"]["memory_accuracy"]["target_accuracy(all)"] = safe_div(target_memory_accuracy_scores, target_memory_accuracy_num)
    eval_results["overall_score"]["memory_accuracy"]["target_accuracy(valid)"] = safe_div(target_memory_accuracy_scores, target_memory_accuracy_valid_num)
    eval_results["overall_score"]["memory_accuracy"]["target_memory_valid_num"] = target_memory_accuracy_valid_num
    eval_results["overall_score"]["memory_accuracy"]["target_memory_num"] = target_memory_accuracy_num
    eval_results["overall_score"]["memory_accuracy"]["weighted_accuracy(all)"] = safe_div(memory_accuracy_weighted_scores, memory_accuracy_num)
    eval_results["overall_score"]["memory_accuracy"]["weighted_accuracy(valid)"] = safe_div(memory_accuracy_weighted_scores, memory_accuracy_valid_num)
    eval_results["overall_score"]["memory_accuracy"]["memory_valid_num"] = memory_accuracy_valid_num
    eval_results["overall_score"]["memory_accuracy"]["memory_num"] = memory_accuracy_num

    # Memory Update Evaluation
    correct_update_memory_num = 0
    hallucination_update_memory_num = 0
    omission_update_memory_num = 0
    other_update_memory_num = 0
    update_memory_num = 0
    update_memory_valid_num = 0
    for item in eval_results["memory_update_records"]:

        item["is_valid"] = True
        update_memory_num += 1

        if item["memory_update_type"] not in ["Correct", "Hallucination", "Omission", "Other"]:
            item["is_valid"] = False
            continue

        if item["memory_update_type"] == "Correct":
            correct_update_memory_num += 1
        elif item["memory_update_type"] == "Hallucination":
            hallucination_update_memory_num += 1
        elif item["memory_update_type"] == "Omission":
            omission_update_memory_num += 1
        elif item["memory_update_type"] == "Other":
            other_update_memory_num += 1
        
        update_memory_valid_num += 1

    if update_memory_num > 0:
        eval_results["overall_score"]["memory_update"]["correct_update_memory_ratio(all)"] = safe_div(correct_update_memory_num, update_memory_num)
        eval_results["overall_score"]["memory_update"]["hallucination_update_memory_ratio(all)"] = safe_div(hallucination_update_memory_num, update_memory_num)
        eval_results["overall_score"]["memory_update"]["omission_update_memory_ratio(all)"] = safe_div(omission_update_memory_num, update_memory_num)
        eval_results["overall_score"]["memory_update"]["other_update_memory_ratio(all)"] = safe_div(other_update_memory_num, update_memory_num)
    
    if update_memory_valid_num > 0:
        eval_results["overall_score"]["memory_update"]["correct_update_memory_ratio(valid)"] = safe_div(correct_update_memory_num, update_memory_valid_num)
        eval_results["overall_score"]["memory_update"]["hallucination_update_memory_ratio(valid)"] = safe_div(hallucination_update_memory_num, update_memory_valid_num)
        eval_results["overall_score"]["memory_update"]["omission_update_memory_ratio(valid)"] = safe_div(omission_update_memory_num, update_memory_valid_num)
        eval_results["overall_score"]["memory_update"]["other_update_memory_ratio(valid)"] = safe_div(other_update_memory_num, update_memory_valid_num)
    
    eval_results["overall_score"]["memory_update"]["update_memory_valid_num"] = update_memory_valid_num
    eval_results["overall_score"]["memory_update"]["update_memory_num"] = update_memory_num

    # Question-Answering Evaluation
    correct_qa_num = 0
    hallucination_qa_num = 0
    omission_qa_num = 0
    qa_num = 0
    qa_valid_num = 0
    for item in eval_results["question_answering_records"]:
        item["is_valid"] = True
        qa_num += 1

        if item["result_type"] not in ["Correct", "Hallucination", "Omission"]:
            item["is_valid"] = False
            continue

        if item["result_type"] == "Correct":
            correct_qa_num += 1
        elif item["result_type"] == "Hallucination":
            hallucination_qa_num += 1
        elif item["result_type"] == "Omission":
            omission_qa_num += 1

        qa_valid_num += 1

    eval_results["overall_score"]["question_answering"]["correct_qa_ratio(all)"] = safe_div(correct_qa_num, qa_num)
    eval_results["overall_score"]["question_answering"]["correct_qa_ratio(valid)"] = safe_div(correct_qa_num, qa_valid_num)
    eval_results["overall_score"]["question_answering"]["hallucination_qa_ratio(all)"] = safe_div(hallucination_qa_num, qa_num)
    eval_results["overall_score"]["question_answering"]["hallucination_qa_ratio(valid)"] = safe_div(hallucination_qa_num, qa_valid_num)
    eval_results["overall_score"]["question_answering"]["omission_qa_ratio(all)"] = safe_div(omission_qa_num, qa_num)
    eval_results["overall_score"]["question_answering"]["omission_qa_ratio(valid)"] = safe_div(omission_qa_num, qa_valid_num)
    eval_results["overall_score"]["question_answering"]["qa_valid_num"] = qa_valid_num
    eval_results["overall_score"]["question_answering"]["qa_num"] = qa_num

    # Memory Type Accuracy
    for item in eval_results['memory_integrity_records']:
        if 'memory_integrity_score' not in item or 'importance' not in item:
            continue
        score = 1 if item['memory_integrity_score'] == 2 else 0
        eval_results["overall_score"]["memory_type_accuracy"][item['memory_type']]['memory_integrity_acc'] += score
        eval_results["overall_score"]["memory_type_accuracy"][item['memory_type']]['total_num'] += 1

    if eval_results["memory_update_records"]:
        for item in eval_results["memory_update_records"]:
            if 'memory_update_type' not in item or 'importance' not in item:
                continue
            score = 1 if item['memory_update_type'] == "Correct" else 0
            eval_results["overall_score"]["memory_type_accuracy"][item['memory_type']]['memory_update_acc'] += score
            eval_results["overall_score"]["memory_type_accuracy"][item['memory_type']]['total_num'] += 1


    for key in eval_results["overall_score"]["memory_type_accuracy"]:
        total = eval_results["overall_score"]["memory_type_accuracy"][key]['total_num']
        if total > 0:
            eval_results["overall_score"]["memory_type_accuracy"][key]['memory_integrity_acc'] = safe_div(eval_results["overall_score"]["memory_type_accuracy"][key]['memory_integrity_acc'], total)
            eval_results["overall_score"]["memory_type_accuracy"][key]['memory_update_acc'] = safe_div(eval_results["overall_score"]["memory_type_accuracy"][key]['memory_update_acc'], total)
            eval_results["overall_score"]["memory_type_accuracy"][key]['memory_acc'] = eval_results["overall_score"]["memory_type_accuracy"][key]['memory_integrity_acc'] + eval_results["overall_score"]["memory_type_accuracy"][key]['memory_update_acc']

    return eval_results


def iter_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_input_file(file_path: str | None = None, results_dir: str | None = None) -> str:
    env_file = cfg.getenv("HALU_EVAL_FILE", None)
    env_dir = cfg.getenv("HALU_EVAL_RESULTS_DIR", None)

    if file_path:
        candidate = os.path.abspath(file_path)
        if os.path.isdir(candidate):
            results_dir = candidate
        else:
            return candidate

    results_dir = results_dir or env_dir
    if not results_dir and env_file:
        return os.path.abspath(env_file)

    if not results_dir:
        raise ValueError("Please provide --file_path or --results_dir.")

    results_dir = os.path.abspath(results_dir)
    candidates = []
    for name in os.listdir(results_dir):
        if not name.endswith(".jsonl"):
            continue
        full_path = os.path.join(results_dir, name)
        if name.endswith("_eval_results.jsonl") or name.endswith("_test_eval_results.jsonl"):
            candidates.append(full_path)

    if not candidates:
        candidates = [
            os.path.join(results_dir, name)
            for name in os.listdir(results_dir)
            if name.endswith(".jsonl")
        ]

    if not candidates:
        raise FileNotFoundError(f"No JSONL evaluation input found under: {results_dir}")

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main(
    file_path: str,
    user_num: int | None = None,
    max_workers: int = 10,
    output_file: str | None = None,
    tmp_dir: str | None = None,
):
    # Require a path to the JSONL file containing per-user structures.
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    data_path = os.path.abspath(file_path)
    base_dir = os.path.dirname(data_path)
    if output_file is None:
        output_file = os.path.join(base_dir, "eval_stat_result.json")
    else:
        output_file = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if tmp_dir is None:
        tmp_dir = os.path.join(os.path.dirname(output_file), "tmp2")
    else:
        tmp_dir = os.path.abspath(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    start_time = time.time()

    for idx, user_data in enumerate(iter_jsonl(data_path), 1):
        uuid = user_data["uuid"]
        tmp_file = os.path.join(tmp_dir, f"{uuid}.json")

        if os.path.exists(tmp_file):
            print(f"⚡ Skipping user {uuid} ({idx}) — cached result found.")
        else:
            print(f"Processing user {uuid} ({idx})...")
            user_result = process_user(idx, user_data, max_workers)

            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(user_result, f, ensure_ascii=False, indent=4)

            elapsed = time.time() - start_time
            print(f"✅ Finished user {uuid} ({idx}), elapsed {elapsed:.2f}s.")

        if user_num is not None and idx >= user_num:
            break

    add_dialogue_duration_time = 0
    search_memory_duration_time = 0

    for user_data in iter_jsonl(data_path):
        
        sessions = user_data['sessions']

        for session in sessions:
            if "add_dialogue_duration_ms" in session:
                add_dialogue_duration_time += session["add_dialogue_duration_ms"]
            
            if "questions" in session:
                for question in session["questions"]:
                    if "search_duration_ms" in question:
                        search_memory_duration_time += question["search_duration_ms"]

    add_dialogue_duration_time = add_dialogue_duration_time / 1000 / 60
    search_memory_duration_time = search_memory_duration_time / 1000 / 60

    print("\n🔄 Aggregating all user results...")

    eval_results = {
        "overall_score": {
            "memory_integrity": {},
            "memory_accuracy": {},
            "memory_update": {},
            "question_answering": {},
            "memory_type_accuracy": {
                "Event Memory": {
                    "memory_integrity_acc": 0,
                    "memory_update_acc": 0,
                    "total_num": 0,
                },
                "Persona Memory": {
                    "memory_integrity_acc": 0,
                    "memory_update_acc": 0,
                    "total_num": 0,
                },
                "Relationship Memory": {
                    "memory_integrity_acc": 0,
                    "memory_update_acc": 0,
                    "total_num": 0,
                },
            },
            "time_consuming": {
                "add_dialogue_duration_time": add_dialogue_duration_time,
                "search_memory_duration_time": search_memory_duration_time,
                "total_duration_time": add_dialogue_duration_time + search_memory_duration_time
            }
        },
        "memory_integrity_records": [],
        "memory_accuracy_records": [],
        "memory_update_records": [],
        "question_answering_records": []
    }

    for file_name in os.listdir(tmp_dir):
        if not file_name.endswith(".json"):
            continue
        user_file = os.path.join(tmp_dir, file_name)
        with open(user_file, "r", encoding="utf-8") as f:
            user_result = json.load(f)

        eval_results["memory_accuracy_records"].extend(user_result.get("memory_accuracy_records", []))
        eval_results["memory_integrity_records"].extend(user_result.get("memory_integrity_records", []))
        eval_results["memory_update_records"].extend(user_result.get("memory_update_records", []))
        eval_results["question_answering_records"].extend(user_result.get("question_answering_records", []))

    eval_results = aggregate_eval_results(eval_results)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)

    elapsed = time.time() - start_time
    print(f"✅ All done in {elapsed:.2f}s. Results saved to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Path to the `structure_eval_results.jsonl` file. May also be a directory."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=cfg.getenv("HALU_EVAL_RESULTS_DIR", None),
        help="Directory containing graph eval JSONL files. If set, the newest *_eval_results.jsonl is used."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=cfg.getenv("HALU_EVAL_OUTPUT_FILE", None),
        help="Where to write eval_stat_result.json (defaults next to the input JSONL)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=cfg.getenv("HALU_EVAL_OUTPUT_DIR", None),
        help="Directory where eval_stat_result.json will be written."
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default=cfg.getenv("HALU_EVAL_TMP_DIR", None),
        help="Directory for per-user cached intermediate results."
    )
    parser.add_argument(
        "--user_num",
        type=int,
        default=cfg.get_int("HALU_EVAL_USER_NUM", 0) or None,
        help="How many users to evaluate. Defaults to all users."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=cfg.get_int("HALU_EVAL_MAX_WORKERS", 10),
        help="Worker count used inside per-user evaluation."
    )
    args = parser.parse_args()
    file_path = resolve_input_file(args.file_path, args.results_dir)

    output_file = args.output_file
    if output_file is None and args.output_dir:
        output_file = os.path.join(os.path.abspath(args.output_dir), "eval_stat_result.json")

    output_file = main(
        file_path,
        user_num=args.user_num,
        max_workers=args.max_workers,
        output_file=output_file,
        tmp_dir=args.tmp_dir,
    )

    print_scores(output_file)
