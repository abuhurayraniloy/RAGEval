import os
import nltk
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
import logging

from src.db import engine, Base
from src.clients import qdrant_client, redis_client
from src.services.reranking import get_reranker
from qdrant_client.models import Distance, VectorParams

from src.routers.completions import request_llm, CompletionRequest
from src.routers.embed import embed_text_handler, EmbedRequest
from src.routers.search import search_qdrant, SearchRequest
from src.routers.rag import rag_endpoint, RagRequest
from src.routers.evaluate import evaluate, EvalRequest

load_dotenv()

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Database initialization
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Qdrant collection setup
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

    # Redis connection check
    try:
        await redis_client.ping()
        logger.info("Connected to Redis cache.")
    except Exception as e:
        logger.error(f"Failed to connect Redis: {str(e)}")

    # NLTK punkt tokenizer
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

    # Reranker initialization
    if os.getenv("PRELOAD_RERANKER", "false").lower() == "true":
        try:
            get_reranker()
            logger.info("Cross Encoder reranker loaded.")
        except Exception as e:
            logger.error(f"Failed to load reranker: {str(e)}")
    else:
        logger.info(
            "Skipping reranker preload (PRELOAD_RERANKER not set to true). "
            "It will load lazily on first use."
        )

    yield



app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from RAGEval API"}


@app.post("/complete")
async def completion_endpoint(request: CompletionRequest):
    """Generate LLM completions with streaming support."""
    return await request_llm(request)


@app.post("/embed")
async def embed_endpoint(request: EmbedRequest):
    """Embed text with chunking strategy."""
    return await embed_text_handler(request)


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    """Search for similar documents in the vector store."""
    return await search_qdrant(request)


@app.post("/rag")
async def rag_query_endpoint(request: RagRequest):
    """Run RAG pipeline to answer questions."""
    return await rag_endpoint(request)


@app.post("/evaluate")
async def evaluate_endpoint(request: EvalRequest):
    """Evaluate RAG system on a set of questions."""
    return await evaluate(request)
