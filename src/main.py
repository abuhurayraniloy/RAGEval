import os
import nltk
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException as FastAPIHTTPException

from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from dotenv import load_dotenv
import logging

from src.db import engine, Base
from src.clients import qdrant_client, redis_client
from src.services.reranking import get_reranker
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

from src.routers.completions import request_llm, CompletionRequest
from src.routers.embed import embed_text_handler, EmbedRequest
from src.routers.search import search_qdrant, SearchRequest
from src.routers.rag import rag_endpoint, RagRequest
from src.routers.evaluate import evaluate, EvalRequest
from src.routers.api_keys import create_api_key, CreateApiKeyRequest
from src.services.rate_limiter import limiter, log_rate_limit_hit
from src.services.auth_dependency import require_api_key, require_admin_secret

load_dotenv()

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Database initialization
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)

    # Qdrant collection setup
    collection_name = "embeddings"
    try:
        exists = await qdrant_client.collection_exists(collection_name)
        if not exists:
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=1536, distance=Distance.COSINE)
                },
                sparse_vectors_config={"sparse": SparseVectorParams()},
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

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return 429 with Retry-After, and log the hit to PostgreSQL."""
    await log_rate_limit_hit(request)
    retry_after = getattr(exc, "retry_after", 3600)

    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Try again later."},
        headers={"Retry-After": str(retry_after)},
    )


@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
    """Return raw {"error": ...} bodies for auth failures; fall back to
    the default {"detail": ...} envelope for everything else."""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from RAGEval API"}


@app.post("/api-keys", dependencies=[Depends(require_admin_secret)])
async def api_keys_endpoint(request: CreateApiKeyRequest):
    """Generate a new API key. Unauthenticated by design (bootstrap)."""
    return await create_api_key(request)


@app.post("/complete", dependencies=[Depends(require_api_key)])
@limiter.limit("60/hour")
async def completion_endpoint(request: Request, body: CompletionRequest):
    """Generate LLM completions with streaming support."""
    return await request_llm(body)


@app.post("/embed", dependencies=[Depends(require_api_key)])
@limiter.limit("60/hour")
async def embed_endpoint(request: Request, body: EmbedRequest):
    """Embed text with chunking strategy."""
    return await embed_text_handler(body)


@app.post("/search", dependencies=[Depends(require_api_key)])
@limiter.limit("60/hour")
async def search_endpoint(request: Request, body: SearchRequest):
    """Search for similar documents in the vector store."""
    return await search_qdrant(body)


@app.post("/rag", dependencies=[Depends(require_api_key)])
@limiter.limit("60/hour")
async def rag_query_endpoint(request: Request, body: RagRequest):
    """Run RAG pipeline to answer questions."""
    return await rag_endpoint(body)


@app.post("/evaluate", dependencies=[Depends(require_api_key)])
@limiter.limit("60/hour")
async def evaluate_endpoint(request: Request, body: EvalRequest):
    """Evaluate RAG system on a set of questions."""
    return await evaluate(body)
