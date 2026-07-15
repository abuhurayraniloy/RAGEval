"""Vector retrieval service for querying Qdrant."""

from src.clients import qdrant_client
from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
)


def build_filter(filters: dict | None = None) -> Filter | None:
    """Build a Qdrant Filter from a simple key-value dict.

    Args:
            filters: Dict of payload field -> exact match value, e.g.
                    {"category": "technical"}. None or empty dict means no filtering.

    Returns:
            A Qdrant Filter object, or None if no filters were provided
    """
    if not filters:
        return None

    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filters.items()
    ]
    return Filter(must=conditions)


async def search_hybrid(
    query_vector: list[float],
    query_sparse: dict,
    collection_name: str = "embeddings",
    limit: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """Hybrid search for similar vectors in Qdrant, with optional metadata filtering.

    Args:
            query_vector: Dense embedding vector to search for
            query_sparse: Sparse embedding dict with "indices" and "values"
            collection_name: Name of the Qdrant collection
            limit: Maximum number of results to return
            filters: Optional dict of payload field -> exact match value,
                    e.g. {"category": "technical"}, applied to both the dense
                    and sparse legs of the search

    Returns:
            List of search results with id, score, and payload
    """
    qdrant_filter = build_filter(filters)

    search_response = await qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_vector,
                using="dense",
                limit=limit * 4,
                filter=qdrant_filter,
            ),
            Prefetch(
                query=SparseVector(
                    indices=query_sparse["indices"], values=query_sparse["values"]
                ),
                using="sparse",
                limit=limit * 4,
                filter=qdrant_filter,
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
