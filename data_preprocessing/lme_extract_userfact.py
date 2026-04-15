import os
import json
import json_repair
from tqdm import tqdm
from openai import OpenAI
import openai
import backoff
from multiprocessing import Pool
from pydantic import BaseModel
from typing import List

from pathlib import Path
from dotenv import load_dotenv

# Load env files with precedence: repo-root .env (defaults) then local .env (overrides)
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / '.env', override=False)
load_dotenv(Path(__file__).parent / '.env', override=True)

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config as cfg

# Config (overridable from project .env or a nearer .env in this directory)
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '64'))
SAVE_EVERY = int(os.getenv('SAVE_EVERY', '256'))
MAX_PARSE_RETRIES = int(os.getenv('MAX_PARSE_RETRIES', '5'))  # Max retries for JSON parsing failures

# Userfact extraction hyperparameters
USERFACT_TEMPERATURE = float(os.getenv('USERFACT_TEMPERATURE', '1.0'))
USERFACT_MAX_TOKENS = int(os.getenv('USERFACT_MAX_TOKENS', '2000'))

# OpenAI credentials (can be set in repo `.env` or data_preprocessing `.env`)
OPENAI_API_KEY = cfg.get_stage_api_key('index', '')
OPENAI_BASE_URL = cfg.get_stage_base_url('index', 'http://localhost:8001/v1')
MODEL_NAME = cfg.get_stage_model('index', 'gpt-4o-mini')


class UserFacts(BaseModel):
    """Pydantic model for structured output of user facts."""
    facts: List[str]


@backoff.on_exception(backoff.constant, (openai.RateLimitError), 
                      interval=5)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)



def extract_session_userfact(sess_entry, model_name, examples=None, use_structured_output=True):
    # Always construct an OpenAI client with configured API key and base URL
    client_api_key = OPENAI_API_KEY or 'empty'
    client = OpenAI(api_key=client_api_key, base_url=OPENAI_BASE_URL)
    system_prompt = "You will be given a list of messages from a human user to an AI assistant. Extract all the personal information, life events, experience, and preferences related to the user. Make sure you include all details such as life events, personal experience, preferences, specific numbers, locations, or dates. State each piece of information in a simple sentence. Minimize the coreference across the facts, e.g., replace pronouns with actual entities. If there is no specific events, personal information, or preference mentioned, just generate an empty list."
    
    user_prompt = "Human user messages:\n{}\n\nPersonal facts about the user:"

    dialogue_string = ""
    for turn_entry in sess_entry:
        if turn_entry['role'] == 'user':
            dialogue_string += f"\n{turn_entry['role']}:{turn_entry['content']}"

    summarization_prompt = user_prompt.format(dialogue_string)
    if examples is None:
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summarization_prompt}
            ]
    else:
        messages = [{"role": "system", "content": system_prompt}]
        for example_input_dialogue_string, example_output in examples:
            messages += [
               {"role": "user", "content": user_prompt.format(example_input_dialogue_string)},
               {"role": "assistant", "content": example_output}
            ]
        messages += [{"role": "user", "content": summarization_prompt}]

    if use_structured_output:
        # Use structured output with Pydantic model
        kwargs = {
            'model': model_name,
            'messages': messages,
            'temperature': USERFACT_TEMPERATURE,
            'max_tokens': USERFACT_MAX_TOKENS,
        }
        
        for attempt in range(MAX_PARSE_RETRIES):
            try:
                if 'gpt' in model_name:
                    kwargs['response_format'] = UserFacts
                    completion = client.chat.completions.parse(**kwargs)
                else:
                    completion = client.chat.completions.create(extra_body={'guided_json': UserFacts.model_json_schema()}, **kwargs)
                parsed_response = completion.choices[0].message.content
                parsed_response = json_repair.loads(parsed_response)
                parsed_response = UserFacts(**parsed_response)

                if parsed_response is not None:
                    return parsed_response.facts
                
            except Exception as e:
                print(f"Structured output attempt {attempt + 1}/{MAX_PARSE_RETRIES} failed: {e}")
                if attempt == MAX_PARSE_RETRIES - 1:
                    return None
        return None
    else:
        # Fallback to JSON parsing with retry mechanics
        user_prompt_json = "Human user messages:\n{}\n\nPersonal facts about the user (a list of strings in json format; do not generate anything else):"
        # Update the last user message to use JSON format prompt
        messages[-1] = {"role": "user", "content": user_prompt_json.format(dialogue_string)}
        
        kwargs = {
            'model': model_name,
            'messages': messages,
            'n': 1,
            'temperature': USERFACT_TEMPERATURE,
            'max_tokens': USERFACT_MAX_TOKENS
        }
        
        for attempt in range(MAX_PARSE_RETRIES):
            try:
                completion = chat_completions_with_backoff(client, **kwargs)
                out_string = completion.choices[0].message.content.strip()
                out_string = out_string.replace('```json', '')
                out_string = out_string.replace('```', '')
                result = json.loads(out_string.strip())
                if not isinstance(result, list) or result is None:
                    raise Exception("Parsed result is not a list")
                return result
            except json.JSONDecodeError as e:
                print(f"JSON parsing attempt {attempt + 1}/{MAX_PARSE_RETRIES} failed: {e}")
                # Increase temperature slightly on retry to get different output
                kwargs['temperature'] = min(0.3 * (attempt + 1), 0.9)
                if attempt == MAX_PARSE_RETRIES - 1:
                    return None
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}/{MAX_PARSE_RETRIES}: {e}")
                if attempt == MAX_PARSE_RETRIES - 1:
                    return None
        return None


# Global examples for ICL mode (defined here so workers can access it)
ICL_EXAMPLES = [
   ("\nuser: What impact have recent economic developments had on Oxford's unique blend of old-world charm and modern innovation?\nuser:How has the city goverment responded to the economic challenges faced by the hospitality and tourism industries?\nuser:What specific measures has the city government taken to ensure the safety of tourists and locals amidst the pandemic?", json.dumps([])),   # from sharegpt/ultrachat
    ("\nuser: Could you explain the process of optimizing a website for search engines?\nuser:Do you have any tips for creating high-quality content that can attract more traffic?\nuser:These tips are very helpful. Is there a specific length or format that works best for creating content?\nuser:That makes sense! I'm also curious, how important is it to update old content on my website? Is it worth the effort?", json.dumps(['The user is interested in optimizing their website.'])),   # from sharegpt/ultrachat
   # ("\nuser:What notable religious sites can visitors explore in Udine?\nuser:Wow, there are so many beautiful churches in Udine! Which one would you recommend I visit first?\nuser:I think I'll start with Udine Cathedral since I'm interested in its artwork. Do you know which artist's work I should look out for?\nuser:That sounds amazing! I can't wait to see those frescoes and paintings in Udine Cathedral. Do you know if there are any guided tours available?", json.dumps(['The user is interested in visiting religious sites in Udine.', 'The user plans to visitUdine Cathedral first.', 'The user is interested in the artwork in Udine Cathedral.', 'The user is excited to see the frescoes and paintings in Udine Cathedral.', 'The user is interested in guided tours of Udine Cathedral.'])),
   ("\nuser:How did the British Empire expand and decline over the centuries?\nuser:It's interesting how the legacy of the British Empire still affects many countries today. Do you think it was ultimately a positive or negative force in the world?\nuser:It's interesting to learn about the impact of the British Empire, makes me wonder how different the world would be without it.\nuser:It's fascinating how the British Empire had such a far-reaching impact on the world. I wonder if there are any other empires in history that have left such a mark?", json.dumps([])),   # from sharegpt/ultrachat
   ("What techniques or tools can writers use to create a compelling and authentic character arc for their protagonists? I especially like the idea of using supporting characters to facilitate the protagonist's growth. Do you have any tips for making sure those supporting characters feel authentic and well-rounded? I think having well-rounded supporting characters can really elevate a story. Do you have any favorite examples of stories that have done this well?", json.dumps(["The user likes the idea of using supporting characters to facilitate the protagonist's growth.", 'The user thinks having well-rounded supporting characters can really elevate a story.'])),   # from sharegpt/ultrachat
   ("\nuser:What are the most iconic monuments to see in Washington, D.C.?\nuser:Can you recommend a good burger joint in D.C. near these monuments?\nuser:Hmm, all of those burger joints sound pretty generic. Do you have any recommendations for a more unique burger spot near the monuments in D.C.?\nuser:I don't know, Lucky Buns and Duke's Grocery sounds a little too fancy for a burger joint. I just want a good old-fashioned burger.\nuser:None of those classic burger joints sound good to me. How about something completely out of the box and unique?", json.dumps([])),   # from sharegpt/ultrachat
   ("\nuser:I'm trying to stay on top of my finances and I was wondering if you could help me track my spending on gifts over the past few months. I know I spent a total of $500 on gifts recently, but I'm having trouble breaking it down. By the way, I did get my brother a really nice graduation gift in May - a $100 gift card to his favorite electronics store.\nuser:I remember buying a birthday present for my sister last month, a pair of earrings from that new jewelry store downtown, and it cost $75.\nuser:I also got my best friend a funny meme-themed mug from Amazon for her housewarming party, which was $20.\nuser:I'm still trying to remember if I got anything for my coworker's baby shower last month.\nuser:I'm pretty sure I got something for my coworker's baby shower...\nuser:I think it was a set of baby clothes and toys from Buy Buy Baby, and it cost around $100.", json.dumps(['The user is trying to stay on top of their finances.', 'The user spent a total of $500 on gifts recently.', "The user bought a $100 gift card for their brother's graduation in May.", "The user bought a pair of earrings for their sister's birthday last month for $75.", "The user bought a funny meme-themed mug for their best friend's housewarming party for $20.", "The user bought a set of baby clothes and toys for their coworker's baby shower last month for around $100."])),   # from userinfo
   ("\nuser:I'm looking to plan out my training schedule for the next few months. Can you help me create a calendar to organize my runs, yoga classes, and swimming sessions? By the way, I've been pretty busy with work lately and missed a few events, including a 5K fun run on March 26th.\nuser:I'm training for a local triathlon in July, so I want to focus on building up my endurance for swimming, running, and cycling. I can dedicate around 5-6 days a week for training, with one or two rest days in between. For running, I prefer Saturday mornings and Wednesday evenings. For yoga, I'd like to stick to my current schedule of Tuesdays and Thursdays. And for swimming, I'd like to add an extra session on Fridays, in addition to my existing Wednesday evening sessions.\nuser:How can I incorporate my recreational volleyball sessions on Sundays into this schedule? And also, I'd like to know if you can recommend any local bike shops that offer bike maintenance services, as I want to make sure my bike is in top condition for my training and the triathlon.\nuser:I'm located in the 92101 zip code. Can you provide me with some bike shop recommendations in my area?\nuser:I'd like to get a bike fitting done at one of these shops. Can you recommend which one would be the best for a triathlon bike fitting? And also, do you think I should get my bike fitted before or after I've completed a few more training rides to get more comfortable with it?\nuser:I'll schedule a bike fitting appointment at RIDE Cyclery. Before that, I'll make sure to get a few more training rides in to get a feel for my bike and identify any initial discomfort or issues. Thanks for the recommendation and advice!", json.dumps(['The user is planning a training schedule for the next few months.', 'The user missed a 5K fun run on March 26th due to being busy with work.', 'The user is training for a local triathlon in July.', 'The user wants to focus on building up endurance for swimming, running, and cycling.', 'The user can dedicate around 5-6 days a week for training, with one or two rest days in between.', 'The user prefers running on Saturday mornings and Wednesday evenings.', 'The user wants to stick to their current yoga schedule of Tuesdays and Thursdays.', 'The user wants to add an extra swimming session on Fridays, in addition to their existing Wednesday evening sessions.', 'The user has recreational volleyball sessions on Sundays.', 'The user is looking for local bike shops that offer bike maintenance services.', 'The user is located in the 92101 zip code.', 'The user wants to get a bike fitting done for their triathlon bike.', 'The user plans to schedule a bike fitting appointment at RIDE Cyclery.', 'The user wants to get a few more training rides in before the bike fitting to identify any initial discomfort or issues.'])),   # from userinfo
   ("\nuser:I'm looking for some data on our company's organizational structure. Can you tell me how many departments we have and what are their main responsibilities? By the way, I was thinking about our leadership team and I realized that we have a total of 100 leadership positions across the company.\nuser:I'll reach out to HR to see if they have any information on our organizational structure. Do you know of any resources that provide benchmarks for leadership positions in different industries, so I can better understand our company's leadership structure in comparison?\nuser:I'll look into those resources, thanks. Do you know if any of them provide data on diversity and inclusion in leadership positions, such as demographics or representation of underrepresented groups?\nuser:That's really helpful. I'll definitely check out some of those resources. I'm particularly interested in learning more about the representation of underrepresented groups in leadership positions. Do you think any of these resources would provide data on the average percentage of leadership positions held by women in companies of our size and industry?\nuser:I'll definitely check out those resources. I'm curious to see how our company compares to industry benchmarks. By the way, I was thinking about our leadership team and I realized that having 100 leadership positions is quite significant. I wonder if there are any best practices for structuring leadership teams of that size.\nuser:That's really helpful. I'll definitely consider those best practices when thinking about our leadership team's structure. I'm also curious to learn more about how to create a more inclusive leadership team. Do you have any suggestions on how to increase diversity and inclusion in our leadership team?", json.dumps(["The user is looking for data on their company's organizational structure.", "The user's company has a total of 100 leadership positions.", 'The user is interested in resources that provide benchmarks for leadership positions in different industries.', 'The user is interested in data on diversity and inclusion in leadership positions.', 'The user is curious about the average percentage of leadership positions held by women in companies of their size and industry.', 'The user is interested in best practices for structuring leadership teams of 100 positions.', 'The user is looking for suggestions on how to increase diversity and inclusion in their leadership team.'])),   # from userinfo
   ("\nuser:I'm looking for some yoga classes near my new apartment. Can you recommend any good studios or classes in my area? By the way, I've been enjoying the outdoors a lot lately, just did a 3-mile loop trail at Valley of Fire State Park last weekend.\nuser:I'll try the Google search and yoga apps to find some classes near me. Do you have any recommendations for yoga poses or sequences that can help with flexibility and stress relief?\nuser:I'll try some of those poses and sequences. Do you have any recommendations for bike maintenance or bike shops in my area? I've been commuting by bike three times a week and want to make sure my bike is in good condition.\nuser:Can you also give me some general tips on how to plan a road trip, especially when it comes to mapping out routes and booking accommodations? I've been thinking about taking a road trip to the Grand Canyon in January.\nuser:Can you give me some more information on Monument Valley and Four Corners? I'm considering adding them to my road trip itinerary.\nuser:I'm thinking of visiting the South Rim of the Grand Canyon, then heading to Monument Valley, and finally stopping by Four Corners on my way back. Do you think that's a doable itinerary, or should I consider adding more time to my trip?", json.dumps(['The user is looking for yoga classes near their new apartment.', 'The user has been enjoying the outdoors a lot lately.', 'The user did a 3-mile loop trail at Valley of Fire State Park last weekend.', 'The user is interested in yoga poses or sequences that can help with flexibility and stress relief.', 'The user commutes by bike three times a week.', 'The user wants to make sure their bike is in good condition.', 'The user is thinking about taking a road trip to the Grand Canyon in January.', 'The user is considering adding Monument Valley and Four Corners to their road trip itinerary.', 'The user is thinking of visiting the South Rim of the Grand Canyon, then heading to Monument Valley, and finally stopping by Four Corners on their way back.'])),   # from userinfo
   ("\nuser:I'm planning a night out with friends this weekend and I need some fashion advice. I was thinking of wearing my new Jimmy Choo heels that I got at the outlet mall for $200 - do you have any outfit suggestions that would complement them well?\nuser:I'm thinking of going with the little black dress option. Do you have any suggestions on what kind of accessories I could wear to complement the outfit?\nuser:I really like the idea of a statement necklace. Do you think a bold, colorful necklace would clash with the neutral color of the LBD, or would it add a nice pop of color to the outfit?\nuser:I'm thinking of a bold, colorful necklace with a fun, playful vibe. I have a coral-colored necklace with a geometric design that I think would add a nice pop of color to the outfit. Do you think that would work well with the LBD and Jimmy Choo heels?\nuser:That sounds great! I'm really excited to wear this outfit out with my friends. One more question - do you think I should wear my hair up or down with this outfit? I have long, dark hair and I'm not sure what would look best.\nuser:I think I'll wear my hair down for this outfit. I like the idea of adding a touch of relaxed, effortless glamour to the outfit. Plus, I think it will create a nice contrast with the more formal Jimmy Choo heels. Thanks for the advice!", json.dumps(['The user is planning a night out with friends this weekend.', 'The user owns a pair of Jimmy Choo heels purchased at an outlet mall for $200.', 'The user is considering wearing a little black dress.', 'The user likes the idea of a statement necklace.', 'The user has a coral-colored necklace with a geometric design.', 'The user has long, dark hair.', 'The user plans to wear their hair down for the night out.'])),   # from userinfo
]


def process_single(args):
    """Worker function for multiprocessing"""
    sess_id, sess_entry, model_name, mode, use_structured_output = args
    try:
        if mode == 'zero-shot':
            expansion = extract_session_userfact(sess_entry, model_name, examples=None, use_structured_output=use_structured_output)
        else:
            expansion = extract_session_userfact(sess_entry, model_name, examples=ICL_EXAMPLES, use_structured_output=use_structured_output)
        return sess_id, expansion, None
    except Exception as e:
        return sess_id, None, str(e)


if __name__ == '__main__':
    # model selection: default from env or fallback
    model_name = MODEL_NAME

    mode = 'ICL'   # zero-shot, ICL
    assert mode in ['zero-shot', 'ICL']
    
    in_file = os.getenv('LME_IN_FILE', 'data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json')
    # in_file = 'data/longmemeval_s_cleaned.json'
    cache_folder = os.getenv('LME_CACHE_FOLDER', 'data/longmemeval-cleaned/expansions-gpt4o_mini_temp1')
    os.makedirs(cache_folder, exist_ok=True)
    cache_file = f'{cache_folder}/session-userfact.json'
    
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
    
    # Set to True to use structured output (Pydantic model), False to use JSON parsing with retry
    use_structured_output = True
    
    # Prepare args for multiprocessing
    process_args = [(sess_id, sess, model_name, mode, use_structured_output) for sess_id, sess in todo_sessions]
    
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
