"""LLM completion endpoint."""

import time
import logging
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm.exceptions import APIError, APIConnectionError

from src.services.generation import stream_completion
from src.db import AsyncSessionLocal, Completion

logger = logging.getLogger("uvicorn.error")


class CompletionRequest(BaseModel):
    prompt: str
    model: str = "groq/llama-3.3-70b-versatile"
    max_tokens: int = 500


async def request_llm(request: CompletionRequest):
    """Handle LLM completion requests with streaming and logging.

    Args:
        request: CompletionRequest with prompt, model, and max_tokens

    Returns:
        StreamingResponse with generated text
    """
    start_time = time.time()

    try:

        async def stream_generator():
            full_response_text = ""
            try:
                async for chunk in stream_completion(
                    request.prompt, request.model, request.max_tokens
                ):
                    full_response_text += chunk
                    yield chunk
            except Exception as stream_err:
                logger.error(f"Stream interrupted: {str(stream_err)}", exc_info=True)
                yield f"\n[Error: Stream Interrupted]"
            finally:
                # Log completion to database
                latency_ms = int((time.time() - start_time) * 1000)

                async with AsyncSessionLocal() as session:
                    completion_log = Completion(
                        prompt=request.prompt,
                        response=full_response_text,
                        model=request.model,
                        latency_ms=latency_ms,
                    )
                    session.add(completion_log)
                    await session.commit()

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except APIError as api_err:
        logger.error(
            f"Groq API Error: {api_err.message} (Status Code: {api_err.status_code})"
        )
        raise HTTPException(
            status_code=api_err.status_code, detail=f"LLM API Error: {api_err.message}"
        )

    except APIConnectionError as conn_err:
        logger.error(f"LLM Connection Error: {str(conn_err)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not reach LLM.",
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal server error occurred.",
        )
