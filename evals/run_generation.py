import sys
import time
import json
import threading
import os
from tqdm import tqdm
import openai
from openai import OpenAI
import backoff
import random
import numpy as np
from datetime import datetime, timedelta
import argparse
from transformers import AutoTokenizer
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

if __package__ is None and __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src import config as cfg

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--out_file_suffix', type=str, default="")
        
    parser.add_argument('--model_name', type=str, default=cfg.getenv('LLM_MODEL', 'meta-llama/Meta-Llama-3.1-8B-Instruct'))
    parser.add_argument('--model_alias', type=str)
    parser.add_argument('--openai_base_url', type=str, default=cfg.getenv('OPENAI_BASE_URL', 'http://localhost:8001/v1'))
    parser.add_argument('--openai_key', type=str, default=cfg.getenv('LLM_API_KEY', cfg.getenv('OPENAI_API_KEY', 'EMPTY')))
    parser.add_argument('--openai_organization', type=str, default=cfg.getenv('OPENAI_ORGANIZATION', None))

    parser.add_argument('--retriever_type', type=str, default='flat-session')
    parser.add_argument('--topk_context', type=int, default=5)
    parser.add_argument('--history_format', type=str, default='json', choices=['json', 'nl'])
    parser.add_argument('--useronly', type=str, default='false', choices=['true', 'false'])
    parser.add_argument('--cot', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--con', type=str, required=False, choices=['true', 'false'], default='false')

    # user fact expansion
    parser.add_argument('--merge_key_expansion_into_value', type=str, choices=['merge', 'replace', 'none'], default='none')     # merge key expansion into value

    parser.add_argument('--gen_length', type=int, default=None)
    
    return parser.parse_args()


def check_args(args):
    print(args)


def prepare_prompt(entry, retriever_type, topk_context: int, useronly: bool, history_format: str, cot: bool, tokenizer, tokenizer_backend, max_retrieval_length, merge_key_expansion_into_value, con=False, con_client=None, con_model=None, tokenizer_lock=None):    
    if retriever_type == 'no-retrieval':
        answer_prompt_template = '{}'
        if cot:
            answer_prompt_template += 'Answer step by step.'
            
    else:
        if merge_key_expansion_into_value is None or merge_key_expansion_into_value == 'none':
            if cot:
                answer_prompt_template = 'I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):'
            else:
                answer_prompt_template = 'I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:'
        elif merge_key_expansion_into_value == 'merge':
            if cot:
                answer_prompt_template = 'I will give you several history chats between you and a user, as well as the relevant user facts extracted from the chat history. Please answer the question based on the relevant chat history and the user facts. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):'
            else:
                answer_prompt_template = 'I will give you several history chats between you and a user, as well as the relevant user facts extracted from the chat history. Please answer the question based on the relevant chat history and the user facts\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:'
        elif merge_key_expansion_into_value == 'replace':
            if cot:
                answer_prompt_template = 'I will give you several facts extracted from history chats between you and a user. Please answer the question based on the relevant facts. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):'
            else:
                answer_prompt_template = 'I will give you several facts extracted from history chats between you and a user. Please answer the question based on the relevant facts.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:'
        else:
            raise NotImplementedError
        
    question_date_string = entry['question_date']
    question_string = entry['question']

    corpusid2date, corpusid2entry = {}, {}
    for session_date, session_id, session_entry in zip(entry['haystack_dates'], entry['haystack_session_ids'], entry['haystack_sessions']):
        corpusid2date[session_id] = session_date
        corpusid2entry[session_id] = session_entry
        for i_turn, turn_entry in enumerate(session_entry):
            corpusid2date[session_id + '_' + str(i_turn+1)] = session_date
            corpusid2entry[session_id + '_' + str(i_turn+1)] = turn_entry

    # corpusid2retvalue = {}
    # try:
    #     for ret_result_entry in entry['retrieval_results']['ranked_items']:
    #         corpusid2retvalue[ret_result_entry['corpus_id']] = ret_result_entry['text']
    # except:
    #     pass
    
    retrieved_chunks = []
    # get chunks in the original order
    if retriever_type == "orig-session":   # no retrieval, session
        for session_date, session_entry in zip(entry['haystack_dates'], entry['haystack_sessions']):
            if useronly:
                retrieved_chunks.append((session_date, [x for x in session_entry if x['role'] == 'user']))
            else:
                retrieved_chunks.append((session_date, session_entry))
    elif retriever_type == "orig-turn":  # no retrieval, turn
        for session_date, session_entry in zip(entry['haystack_dates'], entry['haystack_sessions']):
            if useronly:
                retrieved_chunks += [(session_date, x) for x in session_entry if x['role'] == 'user']
            else:
                retrieved_chunks += [(session_date, x) for x in session_entry]

    # only retain oracle chunks 
    elif retriever_type == "oracle-session":   # no retrieval, session
        for session_date, session_entry in zip(entry['haystack_dates'], entry['haystack_sessions']):
            if useronly:
                retrieved_chunks.append((session_date, [x for x in session_entry if x['role'] == 'user']))
            else:
                retrieved_chunks.append((session_date, session_entry))
    elif retriever_type == "oracle-turn":  # no retrieval, turn
        for session_date, session_entry in zip(entry['haystack_dates'], entry['haystack_sessions']):
            if useronly:
                retrieved_chunks += [(session_date, x) for x in session_entry if x['role'] == 'user']
            else:
                retrieved_chunks += [(session_date, x) for x in session_entry]

    # get retrieved chunks
    # elif retriever_type == "flat-turn":
    #     for ret_result_entry in entry['retrieval_results']['ranked_items']:
    #         converted_corpus_id = '_'.join(ret_result_entry['corpus_id'].replace('noans_', 'answer_').split('_')[:-1])
    #         converted_turn_id = int(ret_result_entry['corpus_id'].replace('noans_', 'answer_').split('_')[-1]) - 1   # we had offset one during retrieval
    #         # automatically expand turn into round
    #         try:
    #             cur_round_data = [corpusid2entry[converted_corpus_id][converted_turn_id]]
    #             converted_next_turn_id = converted_turn_id + 1
    #             if converted_next_turn_id < len(corpusid2entry[converted_corpus_id]):
    #                 cur_round_data.append(corpusid2entry[converted_corpus_id][converted_next_turn_id])
                
    #         except:
    #             continue
            
    #         # handle optional merging key into the value
    #         if merge_key_expansion_into_value is None or merge_key_expansion_into_value == 'none':
    #             retrieved_chunks.append((corpusid2date[converted_corpus_id], cur_round_data))
    #         elif merge_key_expansion_into_value == 'replace':
    #             retrieved_chunks.append((corpusid2date[converted_corpus_id], ret_result_entry['text']))
    #         elif merge_key_expansion_into_value == 'merge':
    #             retrieved_chunks.append((corpusid2date[converted_corpus_id], ret_result_entry['text'], cur_round_data))
    #         else:
    #             raise NotImplementedError

    #     if useronly and not merge_key_expansion_into_value == 'replace':
    #         retrieved_chunks = [x for x in retrieved_chunks if x[-1]['role'] == 'user']     

    elif retriever_type == "flat-session" or retriever_type == "flat-session-split":
        for ret_result_entry in entry['retrieval_results']['ranked_items']:
            # 适配wyx的retrieval结果格式
            if "is_original" in ret_result_entry:
                ret_result_entry["res_type"] = "chunk" if ret_result_entry['is_original'] else "memory_note"
            if "text" in ret_result_entry:
                ret_result_entry["content"] = ret_result_entry['text']

            # handle optional merging key into the value
            if merge_key_expansion_into_value is None or merge_key_expansion_into_value == 'none':
                if "corpus_id" not in ret_result_entry or ret_result_entry.get("res_type", "chunk") != "chunk":
                    continue
                # if useronly:
                #     retrieved_chunks.append((corpusid2date[ret_result_entry['corpus_id'].replace('noans_', 'answer_')],
                #                             [x for x in corpusid2entry[ret_result_entry['corpus_id'].replace('noans_', 'answer_')] if x['role'] == 'user']))
                # else:
                #     retrieved_chunks.append((corpusid2date[ret_result_entry['corpus_id'].replace('noans_', 'answer_')], corpusid2entry[ret_result_entry['corpus_id'].replace('noans_', 'answer_')]))
                target_corpus_id = ret_result_entry['corpus_id'].replace('noans_', 'answer_')
                if target_corpus_id in corpusid2entry:
                    if useronly:
                        retrieved_chunks.append((corpusid2date[target_corpus_id],
                                                [x for x in corpusid2entry[target_corpus_id] if x['role'] == 'user']))
                    else:
                        retrieved_chunks.append((corpusid2date[target_corpus_id], corpusid2entry[target_corpus_id]))
                else:
                    # Fallback or logging if needed when corpus_id is not found in local session map
                    pass
            elif merge_key_expansion_into_value == 'replace':
                if ret_result_entry['res_type'] == 'chunk':
                    continue
                if ret_result_entry['res_type'] == 'entity':
                    retrieved_chunks.append(("unknown", ret_result_entry['entity_name'] + ": " + ret_result_entry['content']))
                # TODO: 现在在split的时候，直接把memory_note用'. '拆开，后面代码要改成检索结果产生时就带特殊符号拼接
                elif ret_result_entry['res_type'] == 'memory_note':
                    if retriever_type == "flat-session-split":
                        mem_points = ret_result_entry['content'].split(". ")
                        for mem_point in mem_points:
                            retrieved_chunks.append(("unknown", mem_point+"."))
                    else:
                        retrieved_chunks.append(("unknown", ret_result_entry['content']))
                else:
                    retrieved_chunks.append(("unknown", ret_result_entry['content']))
                # retrieved_chunks.append((corpusid2date[ret_result_entry['corpus_id'].replace('noans_', 'answer_')], ret_result_entry['text']))
            elif merge_key_expansion_into_value == 'merge':
                if useronly:
                    retrieved_chunks.append((corpusid2date[ret_result_entry['corpus_id'].replace('noans_', 'answer_')], ret_result_entry['text'], [x for x in corpusid2entry[ret_result_entry['corpus_id'].replace('noans_', 'answer_')] if x['role'] == 'user']))
                else:
                    retrieved_chunks.append((corpusid2date[ret_result_entry['corpus_id'].replace('noans_', 'answer_')], ret_result_entry['text'], corpusid2entry[ret_result_entry['corpus_id'].replace('noans_', 'answer_')]))
            else:
                raise NotImplementedError

    elif retriever_type == "no-retrieval":
        retrieved_chunks = []
        
    else:
        raise NotImplementedError

    if retriever_type in ["orig-turn", "orig-session"]:
        retrieved_chunks = retrieved_chunks[-topk_context:]  # keep latest
    else:
        retrieved_chunks = retrieved_chunks[:topk_context]

    # clean up
    # print(f"Cleaning chunks for question_id: {entry.get('question_id', 'N/A')}", flush=True)
    retrieved_chunks_cleaned = []
    for retrieved_item in retrieved_chunks:
        try:
            date, session_entry = retrieved_item
            for turn_entry in session_entry:
                if type(turn_entry) == dict and 'has_answer' in turn_entry:
                    turn_entry.pop('has_answer')
            retrieved_chunks_cleaned.append((date, session_entry))
        except:
            date, expansion_entry, session_entry = retrieved_item
            for turn_entry in session_entry:
                if type(turn_entry) == dict and 'has_answer' in turn_entry:
                    turn_entry.pop('has_answer')
            retrieved_chunks_cleaned.append((date, expansion_entry, session_entry))
    retrieved_chunks = retrieved_chunks_cleaned

    # optional: if CoN is specified, add an information extraction process before feeding into the model
    if con:
        con_prompt = "I will give you a chat history between you and a user, as well as a question from the user. Write reading notes to extract all the relevant user information relevant to answering the answer. If no relevant information is found, just output \"empty\". \n\n\nChat History:\nSession Date: {}\nSession Content:\n{}\n\nQuestion Date: {}\nQuestion: {}\nExtracted note (information relevant to answering the question):"
        retrieved_chunks_with_notes = []
        for i, cur_item in enumerate(retrieved_chunks):
            if merge_key_expansion_into_value == 'merge':
                (chunk_date, chunk_expansion_entry, chunk_entry) = cur_item
                                
            else:
                (chunk_date, chunk_entry) = cur_item
                
            kwargs = {
                'model': con_model,
                'messages':[
                    {"role": "user", "content": con_prompt.format(chunk_date, json.dumps(chunk_entry), question_date_string, question_string)}
                ],
                'n': 1,
                'temperature': 0,
                'max_tokens': 500,
            }
            # print(kwargs)
            completion = chat_completions_with_backoff(con_client, **kwargs) 
            cur_note = completion.choices[0].message.content.strip()
            chunk_entry_con = {'session_summary': cur_note}

            if merge_key_expansion_into_value == 'merge':
                retrieved_chunks_with_notes.append((chunk_date, chunk_expansion_entry, chunk_entry_con))
            else:
                retrieved_chunks_with_notes.append((chunk_date, chunk_entry_con))

        retrieved_chunks = retrieved_chunks_with_notes
                
    # sort sessions by their dates
    retrieved_chunks.sort(key=lambda x: x[0])
    
    history_string = ""
    for i, cur_item in enumerate(retrieved_chunks):
        if merge_key_expansion_into_value == 'merge':
            (chunk_date, chunk_expansion_entry, chunk_entry) = cur_item
        else:
            (chunk_date, chunk_entry) = cur_item

        if history_format == 'json':
            if merge_key_expansion_into_value == 'merge':
                sess_string = '\n' + json.dumps({'session_summary_facts': chunk_expansion_entry, 'original_session': chunk_entry})
            else:
                sess_string = '\n' + json.dumps(chunk_entry)
        elif history_format == 'nl':
            sess_string = ""
            if merge_key_expansion_into_value == 'merge':
                sess_string += "\n\nSession summary and facts:" + chunk_expansion_entry
            if type(chunk_entry) == list:
                for turn_entry in chunk_entry:
                    sess_string += "\n\n{}: {}".format(turn_entry['role'], turn_entry['content'].strip())
            else:
                sess_string += "{}: {}".format(chunk_entry['role'], chunk_entry['content'].strip())    
        else:
            raise NotImplementedError

        if retriever_type in ["orig-session", "flat-session", "oracle-session", "flat-session-split"]:
            history_string += '\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n'.format(i+1, chunk_date, sess_string)
        elif retriever_type in ["orig-turn", "flat-turn", "oracle-turn"]:  
            # history_string += '\n### Round {}:\nDate: {}\nRound Content:\n{}\n'.format(i+1, chunk_date, sess_string)
            history_string += '\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n'.format(i+1, chunk_date, sess_string)  # we include both sides right now
        elif retriever_type == "no-retrieval":
            history_string += ""
        else:
            raise NotImplementedError

    if retriever_type != "no-retrieval" and history_string == "":
        # Debugging information for empty history string
        error_msg = f"History string is empty for retriever_type={retriever_type}, question_id={entry.get('question_id', 'N/A')}. "
        error_msg += f"retrieved_chunks len={len(retrieved_chunks)}. "
        if 'retrieval_results' in entry:
            if 'ranked_items' in entry['retrieval_results']:
                 error_msg += f"ranked_items len={len(entry['retrieval_results']['ranked_items'])}. "
            else:
                 error_msg += "ranked_items key missing. "
        else:
             error_msg += "retrieval_results key missing. "
        raise AssertionError(error_msg)

    # assert retriever_type == "no-retrieval" or history_string != ""
    if retriever_type == "no-retrieval":
        prompt = answer_prompt_template.format(question_string)
    else:
        # truncate history string
        # print(f"Truncating history for question_id: {entry.get('question_id', 'N/A')} with backend {tokenizer_backend}", flush=True)
        if tokenizer_backend == 'openai':
            tokens = tokenizer.encode(history_string, allowed_special={'<|endoftext|>'})
            if len(tokens) > max_retrieval_length:
                print('Truncating from {} to {}'.format(len(tokens), max_retrieval_length), flush=True)
                truncated_tokens = tokens[:max_retrieval_length]
                history_string = tokenizer.decode(truncated_tokens)
        elif tokenizer_backend == 'huggingface':
            # print(f"Starting tokenizer for question_id: {entry.get('question_id', 'N/A')}", flush=True)
            if tokenizer_lock:
                with tokenizer_lock:
                    encoded_input = tokenizer(history_string, max_length=max_retrieval_length, truncation=False, return_tensors="pt")
            else:
                encoded_input = tokenizer(history_string, max_length=max_retrieval_length, truncation=False, return_tensors="pt")
            
            # print(f"Tokenizer finished first pass for question_id: {entry.get('question_id', 'N/A')}", flush=True)
            if len(encoded_input['input_ids'][0]) > max_retrieval_length:
                # print('Truncating from {} to {}'.format(len(encoded_input['input_ids'][0]), max_retrieval_length))
                if tokenizer_lock:
                    with tokenizer_lock:
                        encoded_input = tokenizer(history_string, max_length=max_retrieval_length, truncation=True, return_tensors="pt")
                        history_string = tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)
                else:
                    encoded_input = tokenizer(history_string, max_length=max_retrieval_length, truncation=True, return_tensors="pt")
                    history_string = tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)
        else:
            raise NotImplementedError
        prompt = answer_prompt_template.format(history_string, question_date_string, question_string)

    return prompt
    

def backoff_hdlr(details):
    print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries calling function {details['target'].__name__} at {time.strftime('%X')}", flush=True)

@backoff.on_exception(backoff.constant, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError), 
                      interval=5, on_backoff=backoff_hdlr)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def main(args):
    # setup
    if args.openai_organization:
        openai.organization = args.openai_organization
    client = OpenAI(
        api_key=args.openai_key,
        base_url=args.openai_base_url,
    )

    try:
        in_data = json.load(open(args.in_file))
    except:
        in_data = [json.loads(line) for line in open(args.in_file).readlines()]

    in_file_tmp = args.in_file.split('/')[-1]
    if args.merge_key_expansion_into_value is not None and args.merge_key_expansion_into_value != 'none':
        out_file = args.out_dir + '/' + in_file_tmp + '_testlog_top{}context_{}format_useronly{}_factexpansion{}_{}'.format(
            args.topk_context, args.history_format, args.useronly, args.merge_key_expansion_into_value,
            datetime.now().strftime("%Y%m%d-%H%M"))
    else:
        out_file = args.out_dir + '/' + in_file_tmp + '_testlog_top{}context_{}format_useronly{}_{}'.format(
            args.topk_context, args.history_format, args.useronly, datetime.now().strftime("%Y%m%d-%H%M"))
    if args.out_file_suffix.strip() != "":
        out_file += args.out_file_suffix
    out_f = open(out_file, 'w')

    # model info
    model2maxlength = {
        'gpt-4o': 128000,
        'gpt-4o-2024-08-06': 128000,
        "gpt-4o-mini-2024-07-18": 128000,
        'meta-llama/Meta-Llama-3.1-8B-Instruct': 128000,
        '/mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct': 128000,
        'meta-llama/Meta-Llama-3.1-70B-Instruct': 128000,
        'microsoft/Phi-3-medium-128k-instruct': 120000,
        'microsoft/Phi-3.5-mini-instruct': 120000,
        'microsoft/phi-4': 16000,
        'mistral-7b-instruct-v0.2': 32000,
        'mistral-7b-instruct-v0.3': 32000,
        'In2Training/FILM-7B': 32000,
        'qwen3-8b': 40960,
    }
    model_max_length = model2maxlength[args.model_name]

    # if 'gpt-4' in args.model_name.lower() or 'gpt-3.5' in args.model_name.lower():
    if 1:
        tokenizer = tiktoken.get_encoding('o200k_base')
        tokenizer_backend = 'openai'
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token="hf-xxxx")
        tokenizer_backend = 'huggingface'

    total_prompt_tokens, total_completion_tokens = 0, 0
    tokenizer_lock = threading.Lock()

    # === 优化部分：并行推理 ===
    def run_inference(entry):
        try:
            print(f"Processing question_id: {entry.get('question_id', 'N/A')}", flush=True)
            gen_length = args.gen_length or (500 if not args.cot else 800)
            max_retrieval_length = model_max_length - gen_length - 1000

            if args.con == 'true':
                prompt = prepare_prompt(
                    entry, args.retriever_type, args.topk_context, args.useronly == 'true',
                    args.history_format, args.cot == 'true',
                    tokenizer=tokenizer, tokenizer_backend=tokenizer_backend, max_retrieval_length=max_retrieval_length,
                    merge_key_expansion_into_value=args.merge_key_expansion_into_value,
                    con=True, con_client=client, con_model=args.model_name,
                    tokenizer_lock=tokenizer_lock
                )
            else:
                prompt = prepare_prompt(
                    entry, args.retriever_type, args.topk_context, args.useronly == 'true',
                    args.history_format, args.cot == 'true',
                    tokenizer=tokenizer, tokenizer_backend=tokenizer_backend, max_retrieval_length=max_retrieval_length,
                    merge_key_expansion_into_value=args.merge_key_expansion_into_value,
                    tokenizer_lock=tokenizer_lock
                )
            
            # print(f"Prompt prepared for question_id: {entry.get('question_id', 'N/A')}", flush=True)

            kwargs = {
                'model': args.model_name,
                'messages': [{"role": "user", "content": prompt}],
                'n': 1,
                'temperature': 0,
                'max_tokens': gen_length,
                'timeout': 180,
            }
            # print(kwargs)
            completion = chat_completions_with_backoff(client, **kwargs)
            answer = completion.choices[0].message.content.strip()

            return {
                "question_id": entry['question_id'],
                "question": entry['question'],
                "answer": answer,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "prompt": prompt
            }
        except Exception as e:
            print(f"Error in question_id {entry.get('question_id', 'N/A')}: {repr(e)}")
            traceback.print_exc()
            return None

    results = []
    max_workers = min(8, len(in_data))  # 并行线程数，可根据机器调大
    print(f"Running inference with {max_workers} parallel threads ...", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_inference, entry): entry for entry in in_data}
        print("Tasks submitted.", flush=True)
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                results.append(res)
                total_prompt_tokens += res["prompt_tokens"]
                total_completion_tokens += res["completion_tokens"]
                print(json.dumps({'hypothesis': res["answer"]}), flush=True)
                print(json.dumps({'question_id': res["question_id"], 'hypothesis': res["answer"], 'prompt': res["prompt"]}),
                      file=out_f, flush=True)

    print('Total prompt tokens:', total_prompt_tokens)
    print('Total completion tokens:', total_completion_tokens)
    out_f.close()


if __name__ == '__main__':
    args = parse_args()
    check_args(args)
    main(args)