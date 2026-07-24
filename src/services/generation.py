"""LLM-based answer generation service."""

from typing import AsyncGenerator
from litellm import acompletion, APIError, APIConnectionError
import logging

from src.telemetry import llm_span

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

    with llm_span("generate_answer", model=model) as set_tokens:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = getattr(response, "usage", None)
        if usage:
            set_tokens(usage.prompt_tokens, usage.completion_tokens)
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
    with llm_span("stream_completion", model=model) as set_tokens:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
        )

        completion_tokens_estimate = 0
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                completion_tokens_estimate += (
                    1  # rough proxy; streamed responses don't report usage per-chunk
                )
                yield content

        set_tokens(None, completion_tokens_estimate)


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

    with llm_span("stream_answer", model=model) as set_tokens:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )

        completion_tokens_estimate = 0
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                completion_tokens_estimate += 1
                yield content

        set_tokens(None, completion_tokens_estimate)
