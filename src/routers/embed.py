"""Text embedding endpoint."""

import uuid
import datetime
import logging
from fastapi import HTTPException, status
from pydantic import BaseModel

from src.chunking import ChunkStrategy, chunk_text
from src.services.embeddings import embed_texts, embed_sparse_batch
from src.clients import qdrant_client
from src.db import AsyncSessionLocal, Chunk
from qdrant_client.models import PointStruct, SparseVector

logger = logging.getLogger("uvicorn.error")


class EmbedRequest(BaseModel):
    text: str
    strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH
    source: str = "api_upload"
    category: str = "general"


async def embed_text_handler(request: EmbedRequest):
    """Handle text embedding requests.

    Args:
        request: EmbedRequest with text, strategy, and source

    Returns:
        Dictionary with embedding results and point IDs
    """
    try:
        chunks = chunk_text(request.text, request.strategy)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content to embed after chunking (maybe input is empty).",
            )

        # embedding_response = await embed_texts(chunks)

        dense_vectors = await embed_texts(chunks)
        sparse_vectors = embed_sparse_batch(chunks)

        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        points = []
        chunk_rows = []

        for idx, (chunk_text_value, dense, sparse) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)
        ):
            point_id = str(uuid.uuid4())

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense,
                        "sparse": SparseVector(
                            indices=sparse["indices"], values=sparse["values"]
                        ),
                    },
                    payload={
                        "text": chunk_text_value,
                        "source": request.source,
                        "category": request.category,
                        "date": created_at,
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
            "category": request.category,
            "chunk_count": len(chunks),
            "point_ids": [p.id for p in points],
            "message": (
                f"Text split into {len(chunks)} chunk(s) using "
                f"'{request.strategy.value}' strategy, embedded via Gemini, "
                "+ BM25 (sparse), and indexed in Qdrant."
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
