"""Complete RAG pipeline orchestration."""

import time
import logging
from typing import List

from src.services.embeddings import embed_text, embed_sparse
from src.services.retrieval import search_hybrid
from src.services.generation import generate_answer
from src.services.reranking import rerank, RERANK_CANDIDATES_K
from src.services.judge import extract_cost, JUDGE_MODEL
from src.services.cache import get_cached, set_cache, make_cache_key

logger = logging.getLogger("uvicorn.error")


async def rag_pipeline(
    question: str,
    model: str = "groq/llama-3.3-70b-versatile",
    embedding_model: str = "gemini/gemini-embedding-001",
    use_cache: bool = True,
) -> dict:
    """Run the complete RAG pipeline.

    Args:
        question: Question to answer
        model: LLM model to use for generation
        embedding_model: Model to use for embeddings
        use_cache: Whether to use Redis cache

    Returns:
        Dictionary with answer, sources, latency, and costs
    """
    t0 = time.time()

    # Check cache
    if use_cache:
        cache_key = make_cache_key(question)
        cached_result = await get_cached(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for key {cache_key}")
            return cached_result
    else:
        cache_key = None

    # 1. Embed question
    query_vector = await embed_text(question, model=embedding_model)
    query_sparse = embed_sparse(question)

    # 2. Search vectors
    search_results = await search_hybrid(
        query_vector=query_vector,
        query_sparse=query_sparse,
        collection_name="embeddings",
        limit=RERANK_CANDIDATES_K,
    )

    candidate_texts = [hit["text"] for hit in search_results if hit["text"]]

    # 3. Rerank
    if not candidate_texts:
        reranked = []
    else:
        reranked = rerank(question, candidate_texts, top_k=5)

    # Build context and sources
    contexts = []
    sources = []

    for orig_idx, rerank_score in reranked:
        if orig_idx < len(search_results):
            hit = search_results[orig_idx]
            text = hit["text"]
            contexts.append(text)
            sources.append(
                {
                    "id": hit["id"],
                    "vector_score": hit["score"],
                    "rerank_score": rerank_score,
                    "text": text,
                }
            )

    # 4. Generate answer
    context_string = "\n\n---\n\n".join(contexts) if contexts else ""
    answer = await generate_answer(question, context_string, model=model)

    latency_ms = int((time.time() - t0) * 1000)

    result = {
        "answer": answer,
        "sources": sources,
        "latency_ms": latency_ms,
        "llm_cost_usd": 0.0,  # Cost tracking would need LiteLLM callbacks
    }

    # Cache the result
    if cache_key:
        await set_cache(cache_key, result)

    return result
