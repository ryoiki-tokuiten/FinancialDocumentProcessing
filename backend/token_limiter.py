"""
Global Tool Output Token Limiter.
Applies a strict 5000-token cap across all tool outputs using OpenAI's cl100k_base tokenizer.
"""

import logging
from typing import Any
import tiktoken

logger = logging.getLogger(__name__)

# Initialize tokenizer once
try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.warning(f"Failed to load tiktoken cl100k_base, using fallback: {e}")
    _TOKENIZER = None

MAX_TOOL_TOKENS = 5000

def truncate_tool_output(output: Any, max_tokens: int = MAX_TOOL_TOKENS) -> str:
    """
    Limits the tool output string to max_tokens (default 5000 tokens).
    If truncated, appends instructions for the model to redirect to a file and read via Python.
    
    Args:
        output: Raw output string or dictionary/object to truncate.
        max_tokens: Maximum allowable tokens (default 5000).
        
    Returns:
        Token-bounded string safe for LLM context.
    """
    if output is None:
        return ""
        
    text = str(output) if not isinstance(output, str) else output
    
    if not _TOKENIZER:
        # Fallback character approximation (~4 chars per token)
        char_limit = max_tokens * 4
        if len(text) > char_limit:
            return (
                text[:char_limit] +
                f"\n\n[SYSTEM NOTICE: Tool output exceeded {max_tokens} tokens and was truncated. "
                f"If you need to analyze the full output, output/redirect it to a file and inspect it using execute_python.]"
            )
        return text
        
    try:
        tokens = _TOKENIZER.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        truncated_text = _TOKENIZER.decode(tokens[:max_tokens])
        return (
            f"{truncated_text}\n\n"
            f"[SYSTEM NOTICE: Tool output exceeded {max_tokens} tokens and was truncated. "
            f"If you need to analyze the full output, output/redirect it to a file and inspect it using execute_python.]"
        )
    except Exception as e:
        logger.warning(f"Error in token truncation: {e}")
        char_limit = max_tokens * 4
        if len(text) > char_limit:
            return text[:char_limit] + "\n\n[SYSTEM NOTICE: Tool output truncated due to length.]"
        return text
