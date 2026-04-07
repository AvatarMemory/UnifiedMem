from ast import Str
from typing import List, Dict, Optional, Literal, Any, Union
from pydantic import ValidationError
import os
import json
import json_repair
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import os
from sklearn.metrics.pairwise import cosine_similarity
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle

from src import config as cfg


class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, use_json_schema: bool = True, max_retries: int = 3, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            self.use_json_schema = use_json_schema
            self.max_retries = max_retries
            if api_key is None or api_key == 'None':
                api_key = cfg.getenv('LLM_API_KEY', cfg.getenv('OPENAI_API_KEY', None))
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set LLM_API_KEY or OPENAI_API_KEY in the project environment.")
            base_url = cfg.getenv('LLM_BASE_URL', cfg.getenv('OPENAI_BASE_URL', None)) if base_url is None or base_url == 'None' else base_url
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def _try_convert_type(self, value: Any, target_type: str) -> tuple[bool, Any]:
        """
        Try to convert a value to the target type.
        
        Args:
            value: The value to convert
            target_type: The target type ('boolean', 'integer', 'number', 'string', 'array', 'object')
            
        Returns:
            Tuple of (success, converted_value)
        """
        try:
            if target_type == "boolean":
                if isinstance(value, bool):
                    return True, value
                elif isinstance(value, str):
                    if value.lower() in ['true', '1', 'yes']:
                        return True, True
                    elif value.lower() in ['false', '0', 'no']:
                        return True, False
                elif isinstance(value, (int, float)):
                    return True, bool(value)
            elif target_type == "integer":
                if isinstance(value, int) and not isinstance(value, bool):
                    return True, value
                elif isinstance(value, str):
                    return True, int(value)
                elif isinstance(value, float):
                    return True, int(value)
            elif target_type == "number":
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return True, float(value)
                elif isinstance(value, str):
                    return True, float(value)
            elif target_type == "string":
                if isinstance(value, str):
                    return True, value
                else:
                    return True, str(value)
            elif target_type == "array":
                if isinstance(value, list):
                    return True, value
            elif target_type == "object":
                if isinstance(value, dict):
                    return True, value
            elif target_type == 'null':
                if value is None or value == 'null':
                    return True, value
        except (ValueError, TypeError):
            pass
        
        return False, value
    
    def _validate_json_schema(self, response_json: dict, schema: dict) -> tuple[bool, str]:
        """
        Validate if the response matches the expected schema, with automatic type conversion.
        
        Args:
            response_json: The parsed JSON response
            schema: The expected JSON schema
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Check required fields
            for field in required:
                if field not in response_json:
                    return False, f"Missing required field: {field}"
            
            # Check and convert field types
            for field, field_schema in properties.items():
                if field not in response_json:
                    continue
                    
                field_type = field_schema.get("type")
                value = response_json[field]
                
                # Try to convert the value to the expected type
                if isinstance(field_type, list):
                    # Handle multiple possible types (e.g., ["string", "null"])
                    conversion_success = False
                    for t in field_type:
                        success, converted_value = self._try_convert_type(value, t)
                        if success:
                            conversion_success = True
                            break
                    if not conversion_success:
                        return False, f"Field '{field}' should be one of {field_type} but got {type(value).__name__} and conversion failed"
                else:
                    success, converted_value = self._try_convert_type(value, field_type)
                
                if not success:
                    return False, f"Field '{field}' should be {field_type} but got {type(value).__name__} and conversion failed"
                
                # Update the response_json with converted value
                response_json[field] = converted_value
                
                # Check array items if specified
                if field_type == "array" and "items" in field_schema:
                    items_type = field_schema["items"].get("type")
                    converted_items = []
                    for i, item in enumerate(converted_value):
                        item_success, converted_item = self._try_convert_type(item, items_type)
                        if not item_success:
                            return False, f"Field '{field}[{i}]' should be {items_type} but got {type(item).__name__} and conversion failed"
                        converted_items.append(converted_item)
                    response_json[field] = converted_items
            
            return True, ""
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7, extra_body=None, **kwargs) -> str:
        """
        Get completion with retry logic for format validation.
        
        Args:
            prompt: The prompt to send to the model
            response_format: Response format specification
            temperature: Temperature for generation
            
        Returns:
            The response content as a JSON string
        """
        schema = None
        if "json_schema" in response_format:
            schema = response_format["json_schema"]["schema"]
        
        for attempt in range(self.max_retries):
            try:
                if self.use_json_schema:
                    # Use json_schema for OpenAI models
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You must respond with a JSON object."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format=response_format,
                        temperature=temperature,
                        max_tokens=5000,
                    )
                else:
                    # Use json_object for other models (like LLaMA via vLLM)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You must respond with a valid JSON object that matches the required schema."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=5000,
                        extra_body={'guided_json': extra_body.model_json_schema()},
                        **kwargs
                    )
                
                response_content = response.choices[0].message.content
                
                # If using json_object mode (non openai models), validate the response
                if not self.use_json_schema:
                    try:
                        response_json = json_repair.loads(response_content)
                        # is_valid, error_msg = self._validate_json_schema(response_json, schema)
                        # try:
                        #     parsed = extra_body(**response_json)
                        #     is_valid = True
                        #     error_msg = ""
                        # except ValidationError as ve:
                        #     is_valid = False
                        #     error_msg = str(ve)
                        
                        # if not is_valid:
                        #     print(f"Attempt {attempt + 1}/{self.max_retries}: Schema validation failed: {error_msg}")
                        #     if attempt < self.max_retries - 1:
                        #         # Add validation error to prompt for retry
                        #         prompt += f"\n\nPrevious response failed validation: {error_msg}. Please ensure the response includes all required fields with correct types."
                        #         continue
                        #     else:
                        #         print(f"Warning: Response validation failed after {self.max_retries} attempts. Using last response.")
                        
                        return response_content
                    except json.JSONDecodeError as e:
                        print(f"Attempt {attempt + 1}/{self.max_retries}: JSON parsing failed: {str(e)}")
                        if attempt < self.max_retries - 1:
                            prompt += f"\n\nPrevious response was not valid JSON. Please ensure the response is a valid JSON object."
                            continue
                        else:
                            print(f"Warning: Failed to parse JSON after {self.max_retries} attempts.")
                            raise
                else:
                    # For json_schema mode, OpenAI guarantees the format
                    return response_content
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.max_retries}: Error during completion: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
        
        # Should not reach here, but return empty response as fallback
        return "{}"


class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None,
                 use_json_schema: bool = True,
                 max_retries: int = 5,
                 base_url: Optional[str] = None):
        use_json_schema = True if backend == "openai" and 'gpt-4' in model else False
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, use_json_schema, max_retries, base_url=base_url)
        else:
            raise ValueError("Backend must be 'openai'")
