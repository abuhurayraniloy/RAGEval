import pytest
import pytest_asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from src.main import app

@pytest_asyncio.fixture
async def test_client():
    """
    A fixture that provides an async HTTP client configured 
    to communicate directly with the FastAPI app instance.
    """
    # Using ASGITransport allows httpx to call FastAPI directly in-memory
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

def test_root_endpoint_sync():
    """
    A simple synchronous test ensuring the baseline root endpoint works.
    We use a standard client wrapper or just a quick request check.
    """
    # For basic endpoints, you can also use standard patterns, 
    # but we will stick to async clients for consistency below.
    pass

@pytest.mark.asyncio
async def test_root_endpoint(test_client: AsyncClient):
    """Test that the baseline root endpoint returns 200."""
    response = await test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Message": "Hello from the root"}

@pytest.mark.asyncio
async def test_embed_endpoint_success(test_client: AsyncClient):
    """
    Test the /embed endpoint by mocking the external Gemini API call 
    and the internal Qdrant database upsert function.
    """
    # Mocking LiteLLM's aembedding response structure
    mock_embedding_data = AsyncMock()
    mock_embedding_data.data = [AsyncMock(embedding=[0.1] * 1536)]
    
    # Target path matches where the functions are imported/used
    with patch("src.main.aembedding", return_value=mock_embedding_data) as mock_embed, \
         patch("src.main.qdrant_client.upsert", new_callable=AsyncMock) as mock_qdrant_upsert:
         
        payload = {"text": "Testing uv and pytest implementation."}
        response = await test_client.post("/embed", json=payload)
        
        # Assertions
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "success"
        assert "point_id" in response_json
        
        # Verify our mocks were actually executed as expected
        mock_embed.assert_called_once()
        mock_qdrant_upsert.assert_called_once()

@pytest.mark.asyncio
async def test_search_endpoint_success(test_client: AsyncClient):
    """
    Test the /search endpoint by mocking the query embedding generation
    and Qdrant vector search retrieval.
    """
    mock_embedding_data = AsyncMock()
    mock_embedding_data.data = [AsyncMock(embedding=[0.1] * 1536)]
    
    # Mocking Qdrant's returned ScoredPoint structure
    mock_hit = SimpleNamespace(
        id="e4f8b9a1-1234-4321-abcd-ef1234567890",
        score=0.9234,
        payload={"text": "Mocked database match string"},
    )
    
    with patch("src.main.aembedding", return_value=mock_embedding_data), \
         patch("src.main.qdrant_client.query_points", return_value=SimpleNamespace(points=[mock_hit])):
         
        payload = {"query": "Find container tools", "top_k": 1}
        response = await test_client.post("/search", json=payload)
        
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "success"
        assert len(response_json["results"]) == 1
        assert response_json["results"][0]["text"] == "Mocked database match string"
        assert response_json["results"][0]["score"] == 0.9234
