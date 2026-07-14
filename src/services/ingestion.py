import logging
import uuid

import fitz
from qdrant_client.models import PointStruct, SparseVector

from src.chunking import ChunkStrategy, chunk_text
from src.services.embeddings import embed_sparse_batch, embed_texts
from src.clients import qdrant_client
from src.db import AsyncSessionLocal, Document, Chunk

logger = logging.getLogger("uvicorn.error")

INGEST_CHUNK_STRATEGY = ChunkStrategy.PARAGRAPH

async def _mark_status(
        document_id: str, status: str, total_chunks: int | None = None, error: str | None = None
) -> None:
    from sqlalchemy import update
    from sqlalchemy.sql import func

    async with AsyncSessionLocal() as session:
        values = {"status": status}
        if total_chunks is not None:
            values["total_chunks"] = total_chunks
        if error is not None:
            values["error"] = error
        if status in ("completed", "failed"):
            values["completed_at"] = func.now()

        await session.execute(
            update(Document).where(Document.id == document_id).values(**values)
        )
        await session.commit()

def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                pages.append((page_index + 1, text))
    return pages

async def process_document(
        document_id: int, pdf_bytes: bytes, filename: str
) -> None:
    try:
        pages = extract_pages(pdf_bytes)

        if not pages:
            await _mark_status(
                document_id, "failed", error = "no extractable text found in this pdf."
            )
            return
        
        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for page_number, page_text in pages:
            page_chunks = chunk_text(page_text, INGEST_CHUNK_STRATEGY)
            for idx, chunk in enumerate(page_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({"page_number": page_number, "chunk_index": idx})
        
        if not all_chunks:
            await _mark_status(
                document_id, "failed", error="No content to embed after chunking."
            )
            return

        dense_vectors = await embed_texts(all_chunks)
        sparse_vectors = embed_sparse_batch(all_chunks)

        points = []
        chunk_rows = []

        for i, (chunk_value, dense, sparse, meta) in enumerate(
            zip(all_chunks, dense_vectors, sparse_vectors, chunk_metadata)
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
						"text": chunk_value,
						"source": filename,
						"document_id": document_id,
						"page_number": meta["page_number"],
						"chunk_index": meta["chunk_index"],
						"strategy": INGEST_CHUNK_STRATEGY.value,
					},
				)
            )

            chunk_rows.append(
				Chunk(
					point_id=point_id,
					text=chunk_value,
					strategy=INGEST_CHUNK_STRATEGY.value,
					chunk_index=meta["chunk_index"],
					source=filename,
				)
			)

        await qdrant_client.upsert(collection_name="embeddings", points=points)

        async with AsyncSessionLocal() as session:
            session.add_all(chunk_rows)
            await session.commit()
        
        await _mark_status(document_id, "completed", total_chunks=len(all_chunks))
		
        logger.info(
			f"[ingest] Document {document_id} ({filename}) completed: "
			f"{len(all_chunks)} chunks from {len(pages)} pages."
		)

    except Exception as e:
        logger.error(f"[ingest] Document {document_id} failed: {str(e)}", exc_info=True)
        await _mark_status(document_id, "failed", error=str(e))



  