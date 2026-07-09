import json

from unittest.mock import AsyncMock, patch

import pytest

from src.services.cache import CACHE_TTL_SECONDS, get_cached, make_cache_key, set_cache
from src.services.rag_pipeline import rag_pipeline


@pytest.mark.asyncio
async def test_make_cache_key_is_stable():
    assert make_cache_key("Capital of France?") == make_cache_key("Capital of France?")
    assert make_cache_key("Capital of France?") != make_cache_key("capital of france?")


@pytest.mark.asyncio
async def test_set_and_get_cached_round_trip(fake_redis):
    with patch("src.services.cache.redis_client", fake_redis):
        key = make_cache_key("Capital of France?")
        value = {"answer": "Paris.", "sources": []}

        await set_cache(key, value, ttl_seconds=60)

        assert await get_cached(key) == value
        assert await fake_redis.ttl(key) <= 60


@pytest.mark.asyncio
async def test_rag_pipeline_cache_hit_skips_downstream_work(fake_redis):
    cached_value = {
        "answer": "Paris.",
        "sources": [
            {
                "id": "abc123",
                "vector_score": 0.95,
                "rerank_score": 1.0,
                "text": "Paris is the capital of France.",
            }
        ],
        "latency_ms": 1,
        "llm_cost_usd": 0.0,
    }

    with (
        patch("src.services.cache.redis_client", fake_redis),
        patch(
            "src.services.rag_pipeline.get_cached",
            new=AsyncMock(return_value=cached_value),
        ) as mock_get_cached,
        patch(
            "src.services.rag_pipeline.embed_text",
            new_callable=AsyncMock,
        ) as mock_embed,
        patch(
            "src.services.rag_pipeline.embed_sparse",
            return_value={"indices": [], "values": []},
        ),
        patch(
            "src.services.rag_pipeline.search_hybrid",
            new_callable=AsyncMock,
        ) as mock_search,
        patch(
            "src.services.rag_pipeline.generate_answer",
            new_callable=AsyncMock,
        ) as mock_generate,
        patch("src.services.rag_pipeline.rerank") as mock_rerank,
        patch(
            "src.services.rag_pipeline.set_cache",
            new_callable=AsyncMock,
        ) as mock_set_cache,
    ):
        result = await rag_pipeline("Capital of France?", use_cache=True)

    assert result == cached_value
    mock_get_cached.assert_awaited_once()
    mock_embed.assert_not_awaited()
    mock_search.assert_not_awaited()
    mock_generate.assert_not_awaited()
    mock_rerank.assert_not_called()
    mock_set_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_rag_pipeline_writes_cache_on_miss(fake_redis):
    with (
        patch("src.services.cache.redis_client", fake_redis),
        patch(
            "src.services.rag_pipeline.get_cached",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.services.rag_pipeline.embed_text",
            new_callable=AsyncMock,
            return_value=[0.1] * 1536,
        ),
        patch(
            "src.services.rag_pipeline.embed_sparse",
            return_value={"indices": [], "values": []},
        ),
        patch(
            "src.services.rag_pipeline.search_hybrid",
            new_callable=AsyncMock,
            return_value=[
                {
                    "id": "abc123",
                    "score": 0.95,
                    "text": "Paris is the capital of France.",
                }
            ],
        ),
        patch(
            "src.services.rag_pipeline.generate_answer",
            new_callable=AsyncMock,
            return_value="Paris.",
        ),
        patch(
            "src.services.rag_pipeline.rerank",
            return_value=[(0, 1.0)],
        ),
    ):
        result = await rag_pipeline("Capital of France?", use_cache=True)

    key = make_cache_key("Capital of France?")
    cached_raw = await fake_redis.get(key)

    assert result["answer"] == "Paris."
    assert cached_raw is not None
    assert json.loads(cached_raw)["answer"] == "Paris."
    assert await fake_redis.ttl(key) <= CACHE_TTL_SECONDS
