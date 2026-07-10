"""Database module for RAGEval.

Exposes:
- engine: AsyncEngine for database connections
- AsyncSessionLocal: Async session factory
- Base: SQLAlchemy declarative base for ORM models
- Completion: Completion ORM model
- Chunk: Chunk ORM model
"""

from src.db.session import engine, AsyncSessionLocal
from src.db.models import Base, Completion, Chunk, ApiKey, RateLimitHit

__all__ = [
    "engine",
    "AsyncSessionLocal",
    "Base",
    "Completion",
    "Chunk",
    "ApiKey",
    "RateLimitHit",
]
