"""PDF document ingestion endpoint."""

import logging
import uuid

from fastapi import BackgroundTasks, HTTPException, UploadFile, status

from src.db import AsyncSessionLocal, Document
from src.services.ingestion import process_document

logger = logging.getLogger("uvicorn.error")

MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


async def ingest_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    """Accept a PDF upload and kick off background ingestion.

    Args:
            file: Uploaded PDF file
            background_tasks: FastAPI background task runner

    Returns:
            Dictionary with document_id and initial status
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum size of {MAX_PDF_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    document_id = str(uuid.uuid4())

    try:
        async with AsyncSessionLocal() as session:
            session.add(
                Document(id=document_id, filename=file.filename, status="processing")
            )
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to create document record: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while starting document ingestion.",
        )

    background_tasks.add_task(process_document, document_id, pdf_bytes, file.filename)

    return {
        "status": "success",
        "document_id": document_id,
        "filename": file.filename,
        "processing_status": "processing",
        "message": "Document accepted. Processing in the background.",
    }
