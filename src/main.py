import time
import uuid
import hashlib
import json
import nltk
from src.redis_client import redis_client
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm import acompletion, aembedding
from litellm.exceptions import APIError, APIConnectionError
from dotenv import load_dotenv
import logging

from src.database import engine, AsyncSessionLocal
from src.models import Base, Completion, Chunk
from src.reranker import get_reranker, rerank, RERANK_CANDIDATES_K

from qdrant_client.models import Distance, VectorParams, PointStruct
from src.qdrant import qdrant_client

from src.chunking import ChunkStrategy, chunk_text

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

        try:
            await redis_client.ping()
            logger.info("Connected to Redis cache.")
        except Exception as e:
            logger.error(f"Failed to connect Redis: {str(e)}")

    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            break
        except LookupError:
            continue

    else:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception as e:
            logger.error(f"Failed to download nltk punkt_tab: {str(e)}")

    try:
        get_reranker()
        logger.info("Cross encoder reranker loaded.")
    except Exception as e:
        logger.error(f"Failed to load reranker {str(e)}")

    yield


app = FastAPI(lifespan=lifespan)

CACHE_TTL_SECONDS = 24 * 60 * 60


def make_cache_key(question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return f"rag: {digest}"


class CompletionRequest(BaseModel):
    prompt: str
    model: str = "groq/llama-3.3-70b-versatile"
    max_tokens: int = 500


class EmbedRequest(BaseModel):
    text: str
    strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH
    source: str = "api_upload"


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
        chunks = chunk_text(request.text, request.strategy)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content to embed after chunking (maybe input is empty).",
            )

        embedding_response = await aembedding(
            model="gemini/gemini-embedding-001", input=chunks, dimensions=1536
        )

        points = []
        chunk_rows = []

        for idx, (chunk_text_value, emb_item) in enumerate(
            zip(chunks, embedding_response.data)
        ):
            point_id = str(uuid.uuid4())

            points.append(
                PointStruct(
                    id=point_id,
                    vector=(
                        emb_item["embedding"]
                        if isinstance(emb_item, dict)
                        else emb_item.embedding
                    ),
                    payload={
                        "text": chunk_text_value,
                        "source": request.source,
                        "strategy": request.strategy.value,
                        "chunk_index": idx,
                        "chunk_count": len(chunks),
                    },
                )
            )

            chunk_rows.append(
                Chunk(
                    point_id=point_id,
                    text=chunk_text_value,
                    strategy=request.strategy.value,
                    chunk_index=idx,
                    source=request.source,
                )
            )

        await qdrant_client.upsert(collection_name="embeddings", points=points)

        async with AsyncSessionLocal() as session:
            session.add_all(chunk_rows)
            await session.commit()

        return {
            "status": "success",
            "strategy": request.strategy.value,
            "chunk_count": len(chunks),
            "point_ids": [p.id for p in points],
            "message": (
                f"Text split into {len(chunks)} chunk(s) using "
                f"'{request.strategy.value}' strategy, embedded via Gemini, "
                "and indexed in Qdrant."
            ),
        }

    except HTTPException:
        raise

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
    cache_key = make_cache_key(request.question)

    cached = await redis_client.get(cache_key)
    if cached is not None:
        logger.info(f"Cache hit for key {cache_key}")
        return json.loads(cached)
    try:
        embedding_response = await aembedding(
            model="gemini/gemini-embedding-001",
            input=[request.question],
            dimensions=1536,
        )

        query_vector = embedding_response.data[0].embedding

        search_result = await qdrant_client.query_points(
            collection_name="embeddings",
            query=query_vector,
            limit=RERANK_CANDIDATES_K,
            with_payload=True,
        )

        candidate_hits = [
            hit for hit in search_result.points if hit.payload.get("text")
        ]

        candidate_texts = [hit.payload["text"] for hit in candidate_hits]

        reranked = rerank(request.question, candidate_texts, top_k=5)

        contexts = []
        sources = []

        for original_idx, rerank_score in reranked:
            hit = candidate_hits[original_idx]
            text = hit.payload["text"]
            contexts.append(text)
            sources.append(
                {
                    "id": str(hit.id),
                    "vector_score": hit.score,
                    "rerank_score": rerank_score,
                    "text": text,
                }
            )

        # for hit in search_result.points:
        #     text = hit.payload.get("text")
        #     if text:
        #         contexts.append(text)
        #         sources.append({"id": str(hit.id), "score": hit.score, "text": text})

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

        result = {
            "status": "success",
            "question": request.question,
            "answer": answer,
            "sources": sources,
        }

        await redis_client.set(cache_key, json.dumps(result), ex=CACHE_TTL_SECONDS)

        return result

    except Exception as e:
        logger.error(f"RAG failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred during the RAG process."
        )
