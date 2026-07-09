"""FastAPI dependencies enforcing auth on protected routes."""

import os
from fastapi import Header, HTTPException, status

from src.services.auth import verify_api_key


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Validate the X-API-Key header on protected routes."""
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "API key required"},
        )

    is_valid = await verify_api_key(x_api_key)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid API key"},
        )

    return x_api_key


async def require_admin_secret(
    x_admin_secret: str | None = Header(default=None),
) -> str:
    """Validate the X-Admin-Secret header for key-management routes.

    Args:
            x_admin_secret: Value of the X-Admin-Secret request header

    Returns:
            The validated admin secret

    Raises:
            HTTPException: 401 if the header is missing or doesn't match
    """
    expected = os.getenv("ADMIN_SECRET")

    if not expected:
        # Fail closed: if the server itself has no admin secret configured,
        # nobody should be able to mint keys at all.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Admin secret not configured on server"},
        )

    if x_admin_secret is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Admin secret required"},
        )

    if x_admin_secret != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid admin secret"},
        )

    return x_admin_secret
