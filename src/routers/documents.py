"""Document ingestion status endpoint."""

import logging
from fastapi import HTTPException, status
from sqlalchemy import select

from src.db import AsyncSessionLocal, Document

logger = logging.getLogger("uvicorn.error")


async def get_document_status(document_id: str):
	"""Fetch the current ingestion status of a document.

	Args:
		document_id: Document id returned by /ingest

	Returns:
		Dictionary with status, chunk count, and error (if any)
	"""
	async with AsyncSessionLocal() as session:
		result = await session.execute(
			select(Document).where(Document.id == document_id)
		)
		doc = result.scalar_one_or_none()

	if doc is None:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail="Document not found.",
		)

	return {
		"document_id": doc.id,
		"filename": doc.filename,
		"status": doc.status,
		"total_chunks": doc.total_chunks,
		"error": doc.error,
		"created_at": doc.created_at.isoformat() if doc.created_at else None,
		"completed_at": doc.completed_at.isoformat() if doc.completed_at else None,
	}