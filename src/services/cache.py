"""Redis caching utilities."""

import hashlib
import json
from typing import Optional, Any
from src.clients import redis_client

CACHE_TTL_SECONDS = 24 * 60 * 60


def make_cache_key(question: str) -> str:
    """Generate a cache key from a question.
    
    Args:
        question: Question to hash
        
    Returns:
        Cache key string
    """
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return f"rag:{digest}"


async def get_cached(key: str) -> Optional[dict]:
    """Retrieve cached value.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value as dict, or None if not found
    """
    cached = await redis_client.get(key)
    if cached is not None:
        return json.loads(cached)
    return None


async def set_cache(
    key: str, 
    value: dict, 
    ttl_seconds: int = CACHE_TTL_SECONDS
) -> None:
    """Store value in cache.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl_seconds: Time to live in seconds
    """
    await redis_client.set(key, json.dumps(value), ex=ttl_seconds)


async def clear_cache(key: str) -> None:
    """Clear a cache entry.
    
    Args:
        key: Cache key to delete
    """
    await redis_client.delete(key)
