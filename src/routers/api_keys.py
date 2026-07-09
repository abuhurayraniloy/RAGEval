"""API key management endpoint."""

import logging
from fastapi import HTTPException, status
from pydantic import BaseModel

from src.db import AsyncSessionLocal, ApiKey
from src.services.auth import generate_api_key

logger = logging.getLogger("uvicorn.error")


class CreateApiKeyRequest(BaseModel):
    name: str | None = None


async def create_api_key(request: CreateApiKeyRequest):
    """Generate a new API key and store its hash.

    Args:
            request: Optional friendly name for the key

    Returns:
            The full API key (shown once) plus its id and prefix
    """
    try:
        full_key, key_hash, prefix = generate_api_key()

        async with AsyncSessionLocal() as session:
            record = ApiKey(key_hash=key_hash, prefix=prefix, name=request.name)
            session.add(record)
            await session.commit()
            await session.refresh(record)

        return {
            "status": "success",
            "api_key": full_key,
            "id": record.id,
            "prefix": prefix,
            "message": "Store this key securely — it will not be shown again.",
        }

    except Exception as e:
        logger.error(f"Failed to create API key: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while creating the API key.",
        )
