"""Reranking service using CrossEncoder."""

import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger("uvicorn.error")

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RERANK_CANDIDATES_K = 20

_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder reranker model.
    
    Returns:
        Loaded CrossEncoder model
    """
    global _model
    if _model is None:
        logger.info(f"Loading cross-encoder reranker: {RERANKER_MODEL_NAME}")
        _model = CrossEncoder(RERANKER_MODEL_NAME)
    return _model


def rerank(
    query: str, candidates: List[str], top_k: int = 5
) -> List[Tuple[int, float]]:
    """Rerank candidates based on relevance to query.
    
    Args:
        query: Query text
        candidates: List of candidate texts to rerank
        top_k: Number of top results to return
        
    Returns:
        List of (index, score) tuples, sorted by score descending
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, candidate) for candidate in candidates]
    scores = model.predict(pairs)

    scored = list(enumerate(scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [(idx, float(score)) for idx, score in scored[:top_k]]
