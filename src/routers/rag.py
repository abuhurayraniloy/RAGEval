"""RAG pipeline endpoint."""

import logging
from fastapi import HTTPException, status
from pydantic import BaseModel

from src.services.rag_pipeline import rag_pipeline

logger = logging.getLogger("uvicorn.error")


class RagRequest(BaseModel):
    question: str


async def rag_endpoint(request: RagRequest):
    """Handle RAG pipeline requests.
    
    Args:
        request: RagRequest with question
        
    Returns:
        Dictionary with answer, sources, and metadata
    """
    try:
        result = await rag_pipeline(request.question, use_cache=True)
        
        # Add status to response
        result["status"] = "success"
        result["question"] = request.question
        
        return result

    except Exception as e:
        logger.error(f"RAG failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during the RAG process.",
        )
