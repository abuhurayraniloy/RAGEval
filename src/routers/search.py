"""Vector search endpoint."""

import logging
from fastapi import HTTPException, status
from pydantic import BaseModel

from src.services.embeddings import (
    embed_text,
    embed_texts,
    embed_sparse_batch,
    embed_sparse,
)
from src.services.retrieval import search_hybrid

logger = logging.getLogger("uvicorn.error")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


async def search_qdrant(request: SearchRequest):
    """Handle vector search requests.

    Args:
        request: SearchRequest with query and top_k

    Returns:
        Dictionary with search results
    """
    try:
        query_vector = await embed_text(request.query)
        query_sparse = embed_sparse(request.query)

        search_results = await search_hybrid(
            query_vector=query_vector,
            query_sparse=query_sparse,
            collection_name="embeddings",
            limit=request.top_k,
        )

        formatted_results = [
            {
                "id": result["id"],
                "score": result["score"],
                "text": result["text"],
            }
            for result in search_results
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
