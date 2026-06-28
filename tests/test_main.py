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
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
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
    with (
        patch("src.main.aembedding", return_value=mock_embedding_data) as mock_embed,
        patch(
            "src.main.qdrant_client.upsert", new_callable=AsyncMock
        ) as mock_qdrant_upsert,
    ):

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

    with (
        patch("src.main.aembedding", return_value=mock_embedding_data),
        patch(
            "src.main.qdrant_client.query_points",
            return_value=SimpleNamespace(points=[mock_hit]),
        ),
    ):

        payload = {"query": "Find container tools", "top_k": 1}
        response = await test_client.post("/search", json=payload)

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "success"
        assert len(response_json["results"]) == 1
        assert response_json["results"][0]["text"] == "Mocked database match string"
        assert response_json["results"][0]["score"] == 0.9234


# ==========================================
# NEW 5 TEST CASES (Error & Edge Cases)
# ==========================================

@pytest.mark.asyncio
async def test_embed_validation_error(test_client: AsyncClient):
    """Test that missing the required 'text' field triggers a 422 Unprocessable Entity."""
    # Sending 'query' instead of the required 'text' field
    payload = {"query": "This should be text"}
    response = await test_client.post("/embed", json=payload)
    
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_embed_external_api_failure(test_client: AsyncClient):
    """Test that a failure in the LLM embedding provider is handled safely."""
    # Force the mocked aembedding function to throw an exception
    with (
        patch("src.main.aembedding", side_effect=Exception("API Timeout")),
        patch("src.main.qdrant_client.upsert") as mock_qdrant_upsert
    ):
        payload = {"text": "This will fail."}
        response = await test_client.post("/embed", json=payload)
        
        # Should return our 500 error defined in the except block
        assert response.status_code == 500
        assert response.json()["detail"] == "An error occurred while computing or saving vector representation."
        
        # Ensure Qdrant is never called if the embedding fails
        mock_qdrant_upsert.assert_not_called()


@pytest.mark.asyncio
async def test_search_validation_error(test_client: AsyncClient):
    """Test that missing the required 'query' field triggers a 422 Unprocessable Entity."""
    payload = {"top_k": 3} # Missing 'query'
    response = await test_client.post("/search", json=payload)
    
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_empty_results(test_client: AsyncClient):
    """Test search endpoint when Qdrant successfully searches but finds 0 matches."""
    mock_embedding_data = AsyncMock()
    mock_embedding_data.data = [AsyncMock(embedding=[0.1] * 1536)]
    
    with (
        patch("src.main.aembedding", return_value=mock_embedding_data),
        patch("src.main.qdrant_client.query_points", return_value=SimpleNamespace(points=[]))
    ):
        payload = {"query": "Find me stuff", "top_k": 5}
        response = await test_client.post("/search", json=payload)
        
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "success"
        
        # The frontend should receive a clean, empty array rather than an error
        assert response_json["results"] == []


@pytest.mark.asyncio
async def test_rag_endpoint_success(test_client: AsyncClient):
    """Test the complete RAG endpoint by mocking both embedding and generation."""
    # 1. Mock the Embedding Response
    mock_embedding_data = AsyncMock()
    mock_embedding_data.data = [AsyncMock(embedding=[0.1] * 1536)]
    
    # 2. Mock the Qdrant Search Response
    mock_hit = SimpleNamespace(
        id="mock-source-id",
        score=0.98,
        payload={"text": "Sourdough bread needs high hydration."}
    )
    
    # 3. Mock the LLM Completion Response
    mock_completion_data = AsyncMock()
    mock_completion_data.choices = [
        SimpleNamespace(message=SimpleNamespace(content="You need high hydration for sourdough."))
    ]
    
    with (
        patch("src.main.aembedding", return_value=mock_embedding_data),
        patch("src.main.qdrant_client.query_points", return_value=SimpleNamespace(points=[mock_hit])),
        patch("src.main.acompletion", return_value=mock_completion_data)
    ):
        payload = {"question": "How do I make sourdough?"}
        response = await test_client.post("/rag", json=payload)
        
        assert response.status_code == 200
        response_json = response.json()
        
        # Verify the structure matches our endpoint design
        assert response_json["status"] == "success"
        assert response_json["question"] == "How do I make sourdough?"
        assert response_json["answer"] == "You need high hydration for sourdough."
        
        # Verify the sources were passed back properly
        assert len(response_json["sources"]) == 1
        assert response_json["sources"][0]["id"] == "mock-source-id"
        assert response_json["sources"][0]["text"] == "Sourdough bread needs high hydration."