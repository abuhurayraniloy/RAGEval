import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm import acompletion, aembedding
from litellm.exceptions import APIError, APIConnectionError
from dotenv import load_dotenv
import logging

from src.database import engine, AsyncSessionLocal
from src.models import Base, Completion

from qdrant_client.models import Distance, VectorParams, PointStruct
from src.qdrant import qdrant_client

load_dotenv()

logger = logging.getLogger("uvicorn.error")


# Lifespan context to ensure our table exists on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        # In production, swap this for Alembic migrations
        await conn.run_sync(Base.metadata.create_all)

        collection_name = "embeddings"
        try:
            exists = await qdrant_client.collection_exists(collection_name)
            if not exists:
                await qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            else:
                logger.info(f"Qdrant collection '{collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")

    yield


app = FastAPI(lifespan=lifespan)


class CompletionRequest(BaseModel):
    prompt: str
    model: str = "groq/llama-3.3-70b-versatile"
    max_tokens: int = 500


class EmbedRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class RagRequest(BaseModel):
    question: str


@app.get("/")
async def root():
    return {"Message": "Hello from the root"}


@app.post("/complete")
async def request_llm(request: CompletionRequest):
    start_time = time.time()

    try:
        response = await acompletion(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            stream=True,
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


@app.post("/embed")
async def embed_text(request: EmbedRequest):
    try:
        embedding_response = await aembedding(
            model="gemini/gemini-embedding-001", input=[request.text], dimensions=1536
        )

        vector = embedding_response.data[0].embedding

        point_id = str(uuid.uuid4())

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={"text": request.text, "source": "api_upload"},
        )

        await qdrant_client.upsert(collection_name="embeddings", points=[point])

        return {
            "status": "success",
            "point_id": point_id,
            "message": "Text vectorized via Gemini and indexed in Qdrant successfully.",
        }

    except Exception as e:
        logger.error(f"Failed to process vector embedding: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while computing or saving vector representation.",
        )


@app.post("/search")
async def search_qdrant(request: SearchRequest):
    try:
        embedding_response = await aembedding(
            model="gemini/gemini-embedding-001", input=[request.query], dimensions=1536
        )
        query_vector = embedding_response.data[0].embedding

        search_response = await qdrant_client.query_points(
            collection_name="embeddings",
            query=query_vector,
            limit=request.top_k,
            with_payload=True,
        )

        formatted_results = [
            {
                "id": str(hit.id),
                "score": hit.score,
                "text": hit.payload.get("text", "no text found"),
            }
            for hit in search_response.points
        ]

        return {
            "status": "success",
            "query": request.query,
            "results": formatted_results,
        }

    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching the knowledge base.",
        )


@app.post("/rag")
async def rag_endpoint(request: RagRequest):
    try:
        embedding_response = await aembedding(
            model="gemini/gemini-embedding-001",
            input=[request.question],
            dimensions=1536,
        )

        query_vector = embedding_response.data[0].embedding

        search_result = await qdrant_client.query_points(
            collection_name="embeddings", query=query_vector, limit=5, with_payload=True
        )

        contexts = []
        sources = []

        for hit in search_result.points:
            text = hit.payload.get("text")
            if text:
                contexts.append(text)
                sources.append({"id": str(hit.id), "score": hit.score, "text": text})

        context_string = "\n\n---\n\n".join(contexts)

        system_prompt = "Answer using only the provided context. If the answer is not in the context, say so."
        user_prompt = f"Context: \n{context_string}\n\nQuestion:\n{request.question}"

        completion_response = await acompletion(
            model="groq/llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # logger.info(f"LLM Full Response: {completion_response}")

        answer = completion_response.choices[0].message.content

        return {
            "status": "success",
            "question": request.question,
            "answer": answer,
            "sources": sources,
        }

    except Exception as e:
        logger.error(f"RAG failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred during the RAG process."
        )
