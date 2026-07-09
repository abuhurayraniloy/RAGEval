"""API key generation, hashing, and verification."""

import hashlib
import secrets

from sqlalchemy import select

from src.db import AsyncSessionLocal, ApiKey

API_KEY_PREFIX = "rge"


def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
            Tuple of (full_key, key_hash, prefix). The full_key is shown to the
            user once and never stored; only its hash is persisted.
    """
    raw = secrets.token_urlsafe(32)
    full_key = f"{API_KEY_PREFIX}_{raw}"
    key_hash = hash_key(full_key)
    prefix = full_key[:12]
    return full_key, key_hash, prefix


def hash_key(key: str) -> str:
    """Hash an API key for storage/lookup using SHA-256."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def verify_api_key(key: str) -> bool:
    """Check whether a given API key is valid and not revoked.

    Args:
            key: Raw API key as provided by the client

    Returns:
            True if the key exists and has not been revoked
    """
    key_hash = hash_key(key)

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.revoked == False)
        )
        record = result.scalar_one_or_none()
        return record is not None
