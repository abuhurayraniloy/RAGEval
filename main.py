from fastapi import FastAPI
from pydantic import BaseModel
from groq import AsyncGroq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = AsyncGroq(api_key=api_key)

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 500

@app.get("/")
async def root():
    return {"Message": "Hello from the root"}

@app.post("/complete")
async def request_llm(request: CompletionRequest):
    response = await client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": request.prompt
            }
        ],
        max_tokens=request.max_tokens
    )

    return {"response": response.choices[0].message.content}