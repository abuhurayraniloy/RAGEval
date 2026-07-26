"""Configurable RAG pipeline execution for A/B testing.

Reuses the same embedding, retrieval, reranking, and generation services as
the standard /rag pipeline, but takes a config dict at call time to vary
model, result count, and whether reranking is applied - so two different
configurations can be run against the identical question and compared.
"""

import time
import logging
from litellm import completion_cost

from src.services.embeddings import embed_text, embed_sparse
from src.services.retrieval import search_hybrid
from src.services.generation import generate_answer
from src.services.reranking import rerank, RERANK_CANDIDATES_K

logger = logging.getLogger("uvicorn.error")

DEFAULT_GENERATION_MODEL = "groq/llama-3.3-70b-versatile"
DEFAULT_TOP_K = 5


async def run_configured_pipeline(question: str, config: dict) -> dict:
    """Run the RAG pipeline once, using the given configuration.

    Args:
            question: The question to answer
            config: Dict optionally containing:
                    - model: generation model to use (default groq/llama-3.3-70b-versatile)
                    - top_k: number of final source chunks to use as context (default 5)
                    - reranking: whether to apply cross-encoder reranking (default True)
                    - chunk_size: accepted but NOT applied at query time - see note below

    Returns:
            Dictionary with answer, sources, latency_ms, and cost_usd

    Note on chunk_size:
            Chunking happens once, at document ingestion, and is permanently
            baked into whatever's already stored in Qdrant. There is no way to
            re-chunk already-indexed documents at query time. If "chunk_size"
            is present in config, it is recorded in the stored result for
            visibility, but has no effect on this run - a genuine chunk_size
            A/B test would require ingesting the same corpus twice, into two
            separate Qdrant collections, one per chunk size, and pointing each
            config at its own collection_name. That's a larger, separate
            feature; flagging it rather than silently ignoring it.
    """
    t0 = time.time()

    model = config.get("model", DEFAULT_GENERATION_MODEL)
    top_k = config.get("top_k", DEFAULT_TOP_K)
    use_reranking = config.get("reranking", True)

    query_vector = await embed_text(question)
    query_sparse = embed_sparse(question)

    search_limit = RERANK_CANDIDATES_K if use_reranking else top_k

    search_results = await search_hybrid(
        query_vector=query_vector,
        query_sparse=query_sparse,
        collection_name="embeddings",
        limit=search_limit,
    )

    candidate_texts = [hit["text"] for hit in search_results if hit["text"]]

    if use_reranking and candidate_texts:
        reranked = rerank(question, candidate_texts, top_k=top_k)
        contexts = []
        sources = []
        for orig_idx, rerank_score in reranked:
            if orig_idx < len(search_results):
                hit = search_results[orig_idx]
                contexts.append(hit["text"])
                sources.append(
                    {
                        "id": hit["id"],
                        "vector_score": hit["score"],
                        "rerank_score": rerank_score,
                        "text": hit["text"],
                    }
                )
    else:
        # No reranking: just take the top_k results straight from hybrid
        # search, in whatever order it returned them.
        top_results = search_results[:top_k]
        contexts = [hit["text"] for hit in top_results]
        sources = [
            {
                "id": hit["id"],
                "vector_score": hit["score"],
                "rerank_score": None,
                "text": hit["text"],
            }
            for hit in top_results
        ]

    context_string = "\n\n---\n\n".join(contexts) if contexts else ""

    answer, cost_usd = await _generate_with_cost(question, context_string, model)

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "config_used": {
            "model": model,
            "top_k": top_k,
            "reranking": use_reranking,
            "chunk_size": config.get(
                "chunk_size"
            ),  # recorded, not applied - see docstring
        },
    }


async def _generate_with_cost(
    question: str, context: str, model: str
) -> tuple[str, float]:
    """Generate an answer and compute its real cost via LiteLLM's cost
    tracking, instead of the existing pipeline's hardcoded 0.0 placeholder."""
    from litellm import acompletion

    system_prompt = (
        "Answer using only the provided context. "
        "If the answer is not in the context, say so."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content

    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = 0.0

    return answer, cost
