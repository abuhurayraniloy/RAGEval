from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm import acompletion
from litellm.exceptions import APIError, APIConnectionError
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

class CompletionRequest(BaseModel):
    prompt: str
    model: str = "groq/llama-3.3-70b-versatile"
    max_tokens: int = 500

@app.get("/")
async def root():
    return {"Message": "Hello from the root"}

@app.post("/complete")
async def request_llm(request: CompletionRequest):
    try:
        response = await acompletion(
            model = request.model,
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
            try:
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            except Exception as stream_err:
                logger.error(f"Stream interrupted: {str(stream_err)}", exc_info=True)
                yield f"\n[Error: Stream Interrupted]"
        
        return StreamingResponse(stream_generator(), media_type="text/plain")


    except APIError as api_err:
        logger.error(f"Groq API Error: {api_err.message} (Status Code: {api_err.status_code})")
        raise HTTPException(
            status_code=api_err.status_code,
            detail=f"LLM API Error: {api_err.message}" 
        )
    
    except APIConnectionError as conn_err:
        logger.error(f"LLM Connenction Error: {str(conn_err)}")
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

    