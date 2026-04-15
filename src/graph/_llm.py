import json
import numpy as np
from typing import Optional, List, Any, Callable
import asyncio

import aioboto3
from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os
import sys
import torch
import tiktoken
from transformers import AutoTokenizer, AutoModel
from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

# Use centralized config loader instead of hardcoded PROJECT_ROOT
import src.config as config
# Graph indexing / retrieval uses the index-stage model config.
OPENAI_KEY = config.get_stage_api_key('index', default=config.getenv('OPENAI_KEY', default=None))
LLM_MODEL = config.get_stage_model('index', 'gpt-4o-mini')
OPENAI_BASE_URL = config.get_stage_base_url('index', None)
# Embedding / model names and service URLs from config (with sensible defaults)
CONTRIEVER_MODEL_NAME = config.getenv('CONTRIEVER_MODEL_NAME', 'facebook/contriever')
OPENAI_EMBEDDING_MODEL = config.get_embedding_model('text-embedding-3-small')
# Unified embedding config: single model name env var and optional API URL/key.
# Use `EMBEDDING_MODEL` as the canonical model name for embeddings and also set
# `OPENAI_EMBEDDING_MODEL` for backwards compatibility.
EMBEDDING_MODEL = config.get_embedding_model(OPENAI_EMBEDDING_MODEL)
OPENAI_EMBEDDING_MODEL = EMBEDDING_MODEL
EMBEDDING_API_URL = config.get_embedding_base_url(OPENAI_BASE_URL)
EMBEDDING_API_KEY = config.get_embedding_api_key(OPENAI_KEY)
VLLM_BASE_URL = config.getenv('VLLM_BASE_URL', 'http://localhost:11230/v1')
VLLM_MODEL_PATH = config.getenv('VLLM_MODEL_PATH', '/mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct')

global_openai_async_client = None
global_azure_openai_async_client = None
global_amazon_bedrock_async_client = None
global_vllm_async_client = None
global_contriever_model = None
global_contriever_tokenizer = None
global_openai_async_embedding_client = None


def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        # include base_url if provided (used for vLLM or proxy endpoints)
        if OPENAI_BASE_URL:
            global_openai_async_client = AsyncOpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
        else:
            global_openai_async_client = AsyncOpenAI(api_key=OPENAI_KEY)
    return global_openai_async_client

def get_openai_async_embedding_client_instance():
    global global_openai_async_embedding_client
    if global_openai_async_embedding_client is None:
        # include base_url if provided (used for vLLM or proxy endpoints)
        if EMBEDDING_API_URL:
            global_openai_async_embedding_client = AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_URL)
        else:
            global_openai_async_embedding_client = AsyncOpenAI(api_key=EMBEDDING_API_KEY)
    return global_openai_async_embedding_client

def get_vllm_async_client_instance():
    global global_vllm_async_client
    if global_vllm_async_client is None:
        # Test connection first with synchronous request
        try:
            import urllib.request
            import urllib.error
            test_url = "http://localhost:11230/v1/models"
            print(f"[DEBUG] Testing vLLM service connectivity to {test_url}...")
            with urllib.request.urlopen(test_url, timeout=2) as response:
                print(f"[DEBUG] vLLM service is reachable (status: {response.getcode()})")
        except Exception as e:
            print(f"[WARNING] vLLM service connectivity test failed: {e}")
            print(f"[WARNING] This may cause requests to hang!")
        
        # Use minimal config similar to OpenAI client - let defaults handle timeout/retries
        global_vllm_async_client = AsyncOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="EMPTY",  # vLLM doesn't need real key but AsyncOpenAI requires a value
            timeout=600.0  # Increase timeout to 10 minutes
        )
    return global_vllm_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client


def get_amazon_bedrock_async_client_instance():
    global global_amazon_bedrock_async_client
    if global_amazon_bedrock_async_client is None:
        global_amazon_bedrock_async_client = aioboto3.Session()
    return global_amazon_bedrock_async_client


def get_contriever_instances():
    global global_contriever_model
    global global_contriever_tokenizer
    if global_contriever_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_contriever_tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_MODEL_NAME)
        global_contriever_model = AutoModel.from_pretrained(CONTRIEVER_MODEL_NAME).to(device)
        global_contriever_model.eval()
    return global_contriever_tokenizer, global_contriever_model


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    use_vllm = "gpt" not in model
    if not use_vllm:
        openai_async_client = get_openai_async_client_instance()
    else:
        # openai_async_client = get_vllm_async_client_instance()
        openai_async_client = get_openai_async_client_instance()
    
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    chunk_id = kwargs.pop("chunk_id", None)
    
    try:
        print(f"[DEBUG] Sending request to model {model}...", flush=True)
        import time
        start_time = time.time()
        response = await openai_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        print(f"[DEBUG] Request finished in {time.time() - start_time:.2f}s", flush=True)
    except Exception as e:
        print(f"[ERROR] API call failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model, "chunk_id": chunk_id, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": [{"text": prompt}]})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    inference_config = {
        "temperature": 0,
        "maxTokens": 4096 if "max_tokens" not in kwargs else kwargs["max_tokens"],
    }

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        if system_prompt:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
                system=[{"text": system_prompt}]
            )
        else:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
            )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response["output"]["message"]["content"][0]["text"], "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response["output"]["message"]["content"][0]["text"]


def create_amazon_bedrock_complete_function(model_id: str) -> Callable:
    """
    Factory function to dynamically create completion functions for Amazon Bedrock

    Args:
        model_id (str): Amazon Bedrock model identifier (e.g., "us.anthropic.claude-3-sonnet-20240229-v1:0")

    Returns:
        Callable: Generated completion function
    """
    async def bedrock_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Any] = [],
        **kwargs
    ) -> str:
        return await amazon_bedrock_complete_if_cache(
            model_id,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    # Set function name for easier debugging
    bedrock_complete.__name__ = f"{model_id}_complete"
    
    return bedrock_complete

async def llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_5_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-5-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def vllm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        VLLM_MODEL_PATH,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_embedding(texts: list[str]) -> np.ndarray:
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        embeddings = []
        for text in texts:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": 1024,
                }
            )
            response = await bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0", body=body,
            )
            response_body = await response.get("body").read()
            embeddings.append(json.loads(response_body))
    return np.array([dp["embedding"] for dp in embeddings])


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_embedding_client_instance()
    
    # Truncate texts to 8191 tokens to avoid API errors
    encoding = tiktoken.get_encoding("cl100k_base")
    MAX_TOKENS = 8191
    truncated_texts = []
    for text in texts:
        tokens = encoding.encode(text)
        if len(tokens) > MAX_TOKENS:
            truncated_texts.append(encoding.decode(tokens[:MAX_TOKENS]))
        else:
            truncated_texts.append(text)
            
    response = await openai_async_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL, input=truncated_texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=512)
async def contriever_embedding(texts: list[str]) -> np.ndarray:
    tokenizer, model = get_contriever_instances()
    device = model.device
    
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            all_embeddings.append(embeddings.cpu().numpy())
            
    if not all_embeddings:
        return np.array([])
    return np.concatenate(all_embeddings, axis=0)
