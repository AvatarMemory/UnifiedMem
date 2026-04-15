"""
Copied from LongMemEval/src/evaluation/evaluate_qa.py
"""


import os
import sys
import json
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI
import numpy as np

if __package__ is None and __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src import config as cfg


model_zoo = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
}


@backoff.on_exception(backoff.expo, (openai.RateLimitError,
                                    openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response) 
    return prompt


def resolve_metric_model(metric_model_short: str | None):
    requested = metric_model_short
    if requested in (None, "", "auto"):
        requested = cfg.get_stage_model('qa_eval', 'gpt-4o-mini')

    if requested in model_zoo:
        source_name, source_type = model_zoo[requested]
        return requested, source_name, source_type

    reverse_map = {full_name: (short_name, source_type) for short_name, (full_name, source_type) in model_zoo.items()}
    if requested in reverse_map:
        short_name, source_type = reverse_map[requested]
        return short_name, requested, source_type

    raise ValueError(f"Requested metric model is not supported: {requested}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('metric_model', nargs='?', default='auto',
                        help='Metric model alias/full name. Use auto to read QA_EVAL_LLM_MODEL/LLM_MODEL.')
    parser.add_argument('hyp_file')
    parser.add_argument('ref_file', nargs='?', default='data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json')
    parser.add_argument('--metric-model', dest='metric_model_kw', default=None,
                        help='Keyword alias for metric model; overrides the positional value when provided.')
    args = parser.parse_args()

    metric_model_short = args.metric_model_kw or args.metric_model
    hyp_file = args.hyp_file
    ref_file = args.ref_file
    verbose = False
    
    metric_model_short, metric_model, metric_model_source = resolve_metric_model(metric_model_short)
    result_file = hyp_file + '.eval-results-{}'.format(metric_model_short)

    if metric_model_source == 'openai':
        openai.organization = cfg.getenv('OPENAI_ORGANIZATION', os.getenv('OPENAI_ORGANIZATION'))
        openai_api_key = cfg.get_stage_api_key('qa_eval', os.getenv('OPENAI_API_KEY'))
        openai_api_base = cfg.get_stage_base_url('qa_eval', 'https://api.openai.com/v1')
    else:
        openai_api_key = cfg.get_stage_api_key('qa_eval', os.getenv('OPENAI_API_KEY'))
        openai_api_base = cfg.get_stage_base_url('qa_eval', 'http://localhost:8001/v1')
    
    metric_client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    try:
        hypotheses = [json.loads(line) for line in open(hyp_file).readlines()]
    except:
        hypotheses = json.load(open(hyp_file))
    try:
        references = json.load(open(ref_file))
    except:
        references = [json.loads(line) for line in open(ref_file).readlines()]
    qid2qdata = {entry['question_id']: entry for entry in references}
    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
    qtypes = set(list(qid2qtype.values()))
    qtype2acc = {t: [] for t in qtypes}
    hypothesis_ids = [entry['question_id'] for entry in hypotheses if 'question_id' in entry]
    reference_ids = [entry['question_id'] for entry in references if 'question_id' in entry]
    if len(hypotheses) != len(references):
        missing_ids = sorted(set(reference_ids) - set(hypothesis_ids))
        extra_ids = sorted(set(hypothesis_ids) - set(reference_ids))
        raise ValueError(
            f"Hypothesis/reference size mismatch: got {len(hypotheses)} hypotheses but {len(references)} references. "
            f"Missing question_ids: {missing_ids[:20]}{'...' if len(missing_ids) > 20 else ''}. "
            f"Extra question_ids: {extra_ids[:20]}{'...' if len(extra_ids) > 20 else ''}."
        )

    with open(result_file, 'w') as out_f:
        logs = []
        for entry in tqdm(hypotheses):

            if entry['question_id'] not in qid2qtype:
                print('Warning: skipping {} as it is not in reference data.'.format(entry['question_id']))
                continue
            
            qtype = qid2qtype[entry['question_id']]
            q = qid2qdata[entry['question_id']]['question']
            ans = qid2qdata[entry['question_id']]['answer']
            hyp = entry['hypothesis']
            
            prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
            kwargs = {
                'model': metric_model,
                'messages':[
                    {"role": "user", "content": prompt}
                ],
                'n': 1,
                'temperature': 0,
                'max_tokens': 10
            }
            completion = chat_completions_with_backoff(metric_client, **kwargs)
            eval_response = completion.choices[0].message.content.strip()
            label = 'yes' in eval_response.lower()
            entry['autoeval_label'] = {
                'model': metric_model,
                'label': label
            }
            logs.append(entry)
            if verbose:
                print(json.dumps({
                    'question': q,
                    'answer': ans,
                    'hypothesis': hyp,
                    'autoeval_label': label
                }, indent=4), flush=True)
            print(json.dumps(entry), file=out_f)
            qtype2acc[qid2qtype[entry['question_id']]].append(1 if label else 0)

            
    print('Accuracy:', round(np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs]).item(), 4))
    for k,v in qtype2acc.items():
        print('\t{}: {} ({})'.format(k, round(np.mean(v), 4), len(v)))

    print('Saved to', result_file)
