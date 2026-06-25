import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm import acompletion
from litellm.exceptions import APIError, APIConnectionError
from dotenv import load_dotenv
import logging

from src.database import engine, AsyncSessionLocal
from src.models import Base, Completion

load_dotenv()

logger = logging.getLogger("uvicorn.error")

# Lifespan context to ensure our table exists on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        # In production, swap this for Alembic migrations
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(lifespan=lifespan)

class CompletionRequest(BaseModel):
    prompt: str
    model: str = "groq/llama-3.3-70b-versatile"
    max_tokens: int = 500

@app.get("/")
async def root():
    return {"Message": "Hello from the root"}

@app.post("/complete")
async def request_llm(request: CompletionRequest):
    start_time = time.time()
    
    try:
        response = await acompletion(
            model=request.model,
            messages=[
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            max_tokens=request.max_tokens,
            stream=True
        )

        async def stream_generator():
            full_response_text = ""
            try:
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response_text += content
                        yield content
            except Exception as stream_err:
                logger.error(f"Stream interrupted: {str(stream_err)}", exc_info=True)
                yield f"\n[Error: Stream Interrupted]"
            finally:
                # Observability: Calculate latency and save to database
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Open a new session locally so it survives the streaming process
                async with AsyncSessionLocal() as session:
                    completion_log = Completion(
                        prompt=request.prompt,
                        response=full_response_text,
                        model=request.model,
                        latency_ms=latency_ms
                    )
                    session.add(completion_log)
                    await session.commit()
        
        return StreamingResponse(stream_generator(), media_type="text/plain")

    except APIError as api_err:
        logger.error(f"Groq API Error: {api_err.message} (Status Code: {api_err.status_code})")
        raise HTTPException(
            status_code=api_err.status_code,
            detail=f"LLM API Error: {api_err.message}" 
        )
    
    except APIConnectionError as conn_err:
        logger.error(f"LLM Connection Error: {str(conn_err)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not reach LLM."
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal server error occurred."
        )