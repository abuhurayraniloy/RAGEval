import json
import pytest
from unittest.mock import patch, AsyncMock
from httpx import ASGITransport, AsyncClient
import fakeredis.aioredis
import pytest_asyncio

from src.main import app
import src.main as main_module


@pytest_asyncio.fixture(autouse=True)
async def fake_redis():
    """Replace the real Redis client with an in-memory fake for all tests."""
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    original = main_module.redis_client
    main_module.redis_client = fake
    yield fake
    main_module.redis_client = original
    await fake.aclose()


def make_fake_embedding():
    return AsyncMock(data=[type("Item", (), {"embedding": [0.1] * 1536})])


def make_fake_search_result():
    hit = type(
        "Hit",
        (),
        {
            "id": "abc123",
            "score": 0.95,
            "payload": {"text": "Paris is the capital of France."},
        },
    )
    return type("SearchResult", (), {"points": [hit]})


def make_fake_completion(answer_text: str):
    return AsyncMock(choices=[AsyncMock(message=AsyncMock(content=answer_text))])


@pytest.mark.asyncio
async def test_rag_cache_hit_skips_llm_and_embedding(fake_redis):
    with (
        patch("src.main.aembedding", return_value=make_fake_embedding()) as mock_embed,
        patch(
            "src.main.qdrant_client.query_points",
            return_value=make_fake_search_result(),
        ),
        patch(
            "src.main.acompletion", return_value=make_fake_completion("Paris.")
        ) as mock_complete,
    ):

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First call — cache miss
            response1 = await client.post(
                "/rag", json={"question": "Capital of France?"}
            )
            assert response1.status_code == 200
            assert response1.json()["answer"] == "Paris."
            assert mock_embed.call_count == 1
            assert mock_complete.call_count == 1

            # Second call, same question — cache hit
            response2 = await client.post(
                "/rag", json={"question": "Capital of France?"}
            )
            assert response2.status_code == 200
            assert response2.json() == response1.json()

            # Embedding and completion should NOT have been called again
            assert mock_embed.call_count == 1
            assert mock_complete.call_count == 1


@pytest.mark.asyncio
async def test_rag_cache_key_set_with_ttl(fake_redis):
    with (
        patch("src.main.aembedding", return_value=make_fake_embedding()),
        patch(
            "src.main.qdrant_client.query_points",
            return_value=make_fake_search_result(),
        ),
        patch("src.main.acompletion", return_value=make_fake_completion("Paris.")),
    ):

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/rag", json={"question": "Capital of France?"})

        keys = await fake_redis.keys("rag:*")
        assert len(keys) == 1

        ttl = await fake_redis.ttl(keys[0])
        assert 0 < ttl <= 86400

        cached_value = json.loads(await fake_redis.get(keys[0]))
        assert cached_value["answer"] == "Paris."
