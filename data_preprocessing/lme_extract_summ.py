import os
import json
from tqdm import tqdm
from openai import OpenAI
import openai
import backoff
from multiprocessing import Pool, Manager, Lock
import time

from pathlib import Path
from dotenv import load_dotenv

# Load env files with precedence: repo-root .env (defaults) then local .env (overrides)
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / '.env', override=False)
load_dotenv(Path(__file__).parent / '.env', override=True)

# Config
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '64'))
SAVE_EVERY = int(os.getenv('SAVE_EVERY', '256'))

# Summarization hyperparameters
SUMMARY_MAX_TOKENS = int(os.getenv('SUMMARY_MAX_TOKENS', '500'))
SUMMARY_TEMPERATURE = float(os.getenv('SUMMARY_TEMPERATURE', '0.0'))
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:8001/v1')

# OpenAI credentials
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', LLM_BASE_URL)
MODEL_NAME = os.getenv('LLM_MODEL', 'gpt-4o-mini')

@backoff.on_exception(backoff.constant, (openai.RateLimitError), 
                      interval=5)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def summarize_session(sess_entry, model_name):
    # Always construct an OpenAI client with configured API key and base URL
    client_api_key = OPENAI_API_KEY or 'empty'
    client = OpenAI(api_key=client_api_key, base_url=OPENAI_BASE_URL)
    # memorybank prompt
    # summarization_prompt = "Below is a transcript of a conversation between a human user and an AI assistant. Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. If there are multiple key events, you may summarize them separately. Dialogue content:\n"
    summarization_prompt = "Below is a transcript of a conversation between a human user and an AI assistant. Please summarize the following dialogue as concisely as possible in a short paragraph, extracting the main themes and key information. In your summary, focus more on what the user mentioned or asked for. Dialogue content:\n"
    for turn_entry in sess_entry:
        summarization_prompt += f"\n{turn_entry['role']}: {turn_entry['content']}"
    summarization_prompt += '\n\nYour summary (be concise):'
    # print(summarization_prompt)

    kwargs = {
        'model': model_name,
        'messages': [
            {"role": "user", "content": summarization_prompt}
        ],
        'n': 1,
        'temperature': SUMMARY_TEMPERATURE,
        'max_tokens': SUMMARY_MAX_TOKENS,
    }
    completion = chat_completions_with_backoff(client,**kwargs) 
    return completion.choices[0].message.content.strip()


def process_single(args):
    """Worker function for multiprocessing"""
    sess_id, sess_entry, model_name = args
    try:
        expansion = summarize_session(sess_entry, model_name)
        return sess_id, expansion, None
    except Exception as e:
        return sess_id, None, str(e)


if __name__ == '__main__':
    # model selection: default from env or fallback
    model_name = MODEL_NAME
    
    in_file = 'data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json'
    # in_file = 'data/longmemeval_s_cleaned.json'
    cache_folder = 'data/longmemeval-cleaned/expansions-gpt4o_mini_temp1'
    os.makedirs(cache_folder, exist_ok=True)
    cache_file = f'{cache_folder}/session-summ_.json'

    if os.path.isfile(cache_file):
        data = json.load(open(cache_file))
        print('Loaded:', cache_file)
    else:
        data = {}

    in_data = json.load(open(in_file))

    todo_sessions = {}
    for entry in in_data:
        for sess_id, sess in zip(entry['haystack_session_ids'], entry['haystack_sessions']):
            if sess_id in todo_sessions:
                assert todo_sessions[sess_id] == sess, "Conflict session content for id: " + sess_id
            todo_sessions[sess_id] = sess

    todo_sessions = [(i, s) for i, s in todo_sessions.items() if i not in data]
    
    print(f"Total sessions to process: {len(todo_sessions)}")
    print(f"Already processed: {len(data)}")
    
    # Prepare args for multiprocessing
    process_args = [(sess_id, sess, model_name) for sess_id, sess in todo_sessions]
    
    # Multiprocessing with periodic saving
    processed_count = 0
    with Pool(NUM_WORKERS) as pool:
        for sess_id, expansion, error in tqdm(pool.imap_unordered(process_single, process_args), total=len(process_args)):
            if error:
                print(f"Error processing {sess_id}: {error}")
                continue
            data[sess_id] = expansion
            processed_count += 1
            
            # Save every SAVE_EVERY samples
            if processed_count % SAVE_EVERY == 0:
                json.dump(data, open(cache_file, 'w'))
                print(f"\nSaved checkpoint at {processed_count} new samples (total: {len(data)})")
        
    json.dump(data, open(cache_file, 'w'))
    print(f"Done! Total entries saved: {len(data)}")
