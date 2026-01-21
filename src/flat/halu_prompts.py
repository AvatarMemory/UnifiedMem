"""
Copied from HaluMem/eval/prompts.py
"""

# ============================================
# QA Generation Prompts
# ============================================

PROMPT_QA_TEMPLATE = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from conversations with {user_name}. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp
6. Always convert relative time references to specific dates, months, or years
7. Focus only on the content of the memories. Do not make assumptions beyond what is stated.
8. The answer should be brief (less than 5-6 words) and direct.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

# MEMORIES:
{context}

# Question: {question}

# Answer:
"""


# ============================================
# Evaluation Prompts (for LLM-as-judge)
# ============================================

EVALUATION_PROMPT_QA_ACCURACY = """You are evaluating the accuracy of a QA system's response.

# Question: {question}
# Reference Answer: {reference_answer}
# System Response: {system_response}

# Task:
Classify the system response as one of:
- "Correct": The response is semantically equivalent to the reference answer
- "Hallucination": The response contains information that contradicts the reference or is fabricated
- "Omission": The response fails to answer or says it doesn't know when the answer exists

# Output (JSON):
{{"classification": "Correct|Hallucination|Omission", "reasoning": "brief explanation"}}
"""


# ============================================
# Memory Integrity Evaluation (simplified for non-update systems)
# ============================================

EVALUATION_PROMPT_MEMORY_INTEGRITY = """You are evaluating whether a memory system correctly extracted key information.

# Expected Memory Point:
{expected_memory}

# Extracted Memories:
{extracted_memories}

# Task:
Score how well the extracted memories capture the expected memory point:
- 2: Fully covered - the expected information is completely captured
- 1: Partially covered - some aspects are captured but key details are missing
- 0: Not covered - the expected information is not present

# Output (JSON):
{{"score": 2|1|0, "reasoning": "brief explanation"}}
"""
