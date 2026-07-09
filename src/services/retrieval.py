"""Vector retrieval service for querying Qdrant."""

from typing import List
from src.clients import qdrant_client
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector


async def search_hybrid(
    query_vector: list[float],
    query_sparse: dict,
    collection_name: str = "embeddings",
    limit: int = 5,
) -> list[dict]:
    """Search for similar vectors in Qdrant.

    Args:
        query_vector: Embedding vector to search for
        collection_name: Name of the Qdrant collection
        limit: Maximum number of results to return

    Returns:
        List of search results with id, score, and payload
    """
    search_response = await qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=query_vector, using="dense", limit=limit * 4),
            Prefetch(
                query=SparseVector(
                    indices=query_sparse["indices"], values=query_sparse["values"]
                ),
                using="sparse",
                limit=limit * 4,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            "payload": hit.payload,
            "text": hit.payload.get("text", ""),
        }
        for hit in search_response.points
    ]
