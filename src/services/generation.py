"""LLM-based answer generation service."""

from typing import AsyncGenerator
from litellm import acompletion, APIError, APIConnectionError
import logging

logger = logging.getLogger("uvicorn.error")


async def generate_answer(
    question: str,
    context: str,
    model: str = "groq/llama-3.3-70b-versatile",
) -> str:
    """Generate an answer using the provided context and question.
    
    Args:
        question: User's question
        context: Retrieved context to base answer on
        model: LLM model to use for generation
        
    Returns:
        Generated answer text
        
    Raises:
        APIError: If the LLM API returns an error
        APIConnectionError: If unable to connect to the LLM API
    """
    system_prompt = (
        "Answer using only the provided context. "
        "If the answer is not in the context, say so."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    
    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    
    return response.choices[0].message.content


async def stream_completion(
    prompt: str,
    model: str = "groq/llama-3.3-70b-versatile",
    max_tokens: int = 500,
) -> AsyncGenerator[str, None]:
    """Stream a direct LLM completion (no context injection).
    
    Args:
        prompt: User prompt
        model: LLM model to use for generation
        max_tokens: Maximum tokens to generate
        
    Yields:
        Streamed completion chunks
    """
    response = await acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


async def stream_answer(
    question: str,
    context: str,
    model: str = "groq/llama-3.3-70b-versatile",
) -> AsyncGenerator[str, None]:
    """Stream an answer using the provided context and question.
    
    Args:
        question: User's question
        context: Retrieved context to base answer on
        model: LLM model to use for generation
        
    Yields:
        Streamed answer chunks
    """
    system_prompt = (
        "Answer using only the provided context. "
        "If the answer is not in the context, say so."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    
    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content
