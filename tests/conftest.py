"""Session-wide fixtures shared across the test suite."""

import fakeredis.aioredis
import pytest


@pytest.fixture
def fake_redis() -> fakeredis.aioredis.FakeRedis:
    """Provide an isolated in-memory Redis client for cache tests."""
    return fakeredis.aioredis.FakeRedis(decode_responses=True)
