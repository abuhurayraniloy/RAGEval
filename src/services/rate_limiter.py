import logging
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.db import AsyncSessionLocal, RateLimitHit

logger = logging.getLogger("uvicorn.error")


def key_func(request: Request):
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key
    return get_remote_address(request)


limiter = Limiter(key_func=key_func)


async def log_rate_limit_hit(request: Request) -> None:
    """Persist a record of a rate limit violation to PostgreSQL.

    Args:
            request: The request that triggered the 429
    """
    api_key = request.headers.get("X-API-Key", "")
    prefix = api_key[:12] if api_key else "unknown"

    try:
        async with AsyncSessionLocal() as session:
            session.add(RateLimitHit(api_key_prefix=prefix, endpoint=request.url.path))
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to log rate limit hit: {str(e)}", exc_info=True)
