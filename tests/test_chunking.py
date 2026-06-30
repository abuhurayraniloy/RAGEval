"""
tests/test_chunking.py

Unit tests for all three chunking strategies, plus an integration test for
the /embed endpoint verifying that the strategy parameter is respected and
that each chunk is persisted to Postgres.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from src.main import app
from src.chunking import (
    ChunkStrategy,
    chunk_text,
    FIXED_CHUNK_TOKENS,
    FIXED_CHUNK_OVERLAP,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paragraph_text(n: int) -> str:
    """Return n paragraphs separated by blank lines."""
    return "\n\n".join(f"Paragraph {i}. " * 10 for i in range(n))


def _make_long_text(approx_tokens: int) -> str:
    """Return a string that is roughly approx_tokens tokens long.
    Each word is ~1 token with cl100k_base, so we just repeat words."""
    word = "word"
    return (word + " ") * approx_tokens


# ---------------------------------------------------------------------------
# Paragraph strategy
# ---------------------------------------------------------------------------


class TestChunkParagraph:
    def test_splits_on_double_newline(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, ChunkStrategy.PARAGRAPH)
        assert chunks == ["First paragraph.", "Second paragraph.", "Third paragraph."]

    def test_tolerates_extra_whitespace_between_paragraphs(self):
        text = "A.\n\n  \n\nB."
        chunks = chunk_text(text, ChunkStrategy.PARAGRAPH)
        assert chunks == ["A.", "B."]

    def test_single_paragraph_returns_one_chunk(self):
        text = "Just one paragraph with no blank lines at all."
        chunks = chunk_text(text, ChunkStrategy.PARAGRAPH)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("", ChunkStrategy.PARAGRAPH) == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\n   ", ChunkStrategy.PARAGRAPH) == []

    def test_strips_leading_trailing_whitespace_per_chunk(self):
        text = "  Hello world.  \n\n  Goodbye world.  "
        chunks = chunk_text(text, ChunkStrategy.PARAGRAPH)
        assert chunks == ["Hello world.", "Goodbye world."]

    def test_preserves_intra_paragraph_newlines(self):
        text = "Line one.\nLine two.\n\nNew paragraph."
        chunks = chunk_text(text, ChunkStrategy.PARAGRAPH)
        assert "Line one.\nLine two." in chunks[0]


# ---------------------------------------------------------------------------
# Sentence strategy
# ---------------------------------------------------------------------------


class TestChunkSentence:
    def test_splits_into_sentences(self):
        text = "Hello world. This is a test. Here is another sentence."
        chunks = chunk_text(text, ChunkStrategy.SENTENCE)
        assert len(chunks) == 3

    def test_each_chunk_is_a_single_sentence(self):
        text = "First sentence. Second sentence! Third sentence?"
        chunks = chunk_text(text, ChunkStrategy.SENTENCE)
        # Every chunk must end with sentence-terminal punctuation
        for chunk in chunks:
            assert chunk[-1] in ".!?"

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("", ChunkStrategy.SENTENCE) == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   ", ChunkStrategy.SENTENCE) == []

    def test_single_sentence(self):
        text = "Only one sentence here."
        chunks = chunk_text(text, ChunkStrategy.SENTENCE)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multi_sentence_paragraph(self):
        sentences = [f"Sentence number {i}." for i in range(10)]
        text = " ".join(sentences)
        chunks = chunk_text(text, ChunkStrategy.SENTENCE)
        assert len(chunks) == 10


# ---------------------------------------------------------------------------
# Fixed strategy
# ---------------------------------------------------------------------------


class TestChunkFixed:
    def test_short_text_returns_single_chunk(self):
        text = "Short text."
        chunks = chunk_text(text, ChunkStrategy.FIXED)
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        # ~1100 tokens should produce at least 3 chunks at 500/50 settings
        text = _make_long_text(1100)
        chunks = chunk_text(text, ChunkStrategy.FIXED)
        assert len(chunks) >= 3

    def test_chunks_have_overlap(self):
        """
        The last FIXED_CHUNK_OVERLAP tokens of chunk[n] should appear at
        the start of chunk[n+1]. We verify this by checking that consecutive
        chunks share some common text.
        """
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        text = _make_long_text(1100)
        chunks = chunk_text(text, ChunkStrategy.FIXED)

        for i in range(len(chunks) - 1):
            tokens_a = enc.encode(chunks[i])
            tokens_b = enc.encode(chunks[i + 1])
            overlap_tail = tokens_a[-FIXED_CHUNK_OVERLAP:]
            overlap_head = tokens_b[:FIXED_CHUNK_OVERLAP]
            assert (
                overlap_tail == overlap_head
            ), f"Overlap mismatch between chunk {i} and chunk {i+1}"

    def test_no_chunk_exceeds_max_token_size(self):
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        text = _make_long_text(2000)
        chunks = chunk_text(text, ChunkStrategy.FIXED)
        for chunk in chunks:
            assert len(enc.encode(chunk)) <= FIXED_CHUNK_TOKENS

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("", ChunkStrategy.FIXED) == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   ", ChunkStrategy.FIXED) == []

    def test_all_text_is_covered(self):
        """
        Every token in the original text must appear in at least one chunk.
        Reconstruct by decoding and verify nothing was silently dropped.
        """
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        text = _make_long_text(800)
        original_tokens = enc.encode(text)
        chunks = chunk_text(text, ChunkStrategy.FIXED)

        # The first chunk starts at token 0, last chunk ends at the final token.
        # Reconstruct by concatenating non-overlapping spans.
        step = FIXED_CHUNK_TOKENS - FIXED_CHUNK_OVERLAP
        reconstructed = []
        start = 0
        for chunk in chunks:
            chunk_tokens = enc.encode(chunk)
            non_overlap = (
                chunk_tokens if start == 0 else chunk_tokens[FIXED_CHUNK_OVERLAP:]
            )
            reconstructed.extend(non_overlap)
            start += step

        # Allow the reconstruction to be a superset (last chunk may be shorter).
        assert reconstructed[: len(original_tokens)] == original_tokens


# ---------------------------------------------------------------------------
# General / edge-case tests
# ---------------------------------------------------------------------------


class TestChunkTextDispatch:
    def test_invalid_strategy_raises(self):
        with pytest.raises((ValueError, KeyError)):
            chunk_text("some text", "nonexistent_strategy")  # type: ignore[arg-type]

    @pytest.mark.parametrize("strategy", list(ChunkStrategy))
    def test_returns_list(self, strategy):
        result = chunk_text("Hello world. Testing.", strategy)
        assert isinstance(result, list)

    @pytest.mark.parametrize("strategy", list(ChunkStrategy))
    def test_no_empty_chunks(self, strategy):
        text = "  Hello.  \n\n  World.  Second sentence.  "
        chunks = chunk_text(text, strategy)
        for chunk in chunks:
            assert chunk.strip() != "", f"Empty chunk found for strategy {strategy}"


# ---------------------------------------------------------------------------
# Integration tests — /embed endpoint
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 1536


def _make_embedding_response(n: int):
    """Build a minimal litellm-style aembedding response for n chunks."""
    mock = MagicMock()
    mock.data = [MagicMock(embedding=FAKE_EMBEDDING) for _ in range(n)]
    return mock


@pytest_asyncio.fixture
async def async_client():
    """ASGI test client that bypasses the lifespan so we don't need real
    Postgres / Qdrant / Redis connections."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestEmbedEndpoint:
    """Integration tests for POST /embed. All external I/O is mocked."""

    @pytest.mark.asyncio
    async def test_paragraph_strategy_chunks_and_stores(self, async_client):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        expected_chunks = 3

        with (
            patch("src.main.aembedding", new_callable=AsyncMock) as mock_embed,
            patch("src.main.qdrant_client.upsert", new_callable=AsyncMock),
            patch("src.main.AsyncSessionLocal") as mock_session_ctx,
        ):
            mock_embed.return_value = _make_embedding_response(expected_chunks)
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            response = await async_client.post(
                "/embed", json={"text": text, "strategy": "paragraph"}
            )

        assert response.status_code == 200
        body = response.json()
        assert body["strategy"] == "paragraph"
        assert body["chunk_count"] == expected_chunks
        assert len(body["point_ids"]) == expected_chunks
        # Postgres bulk-insert was called once
        mock_session.add_all.assert_called_once()
        rows = mock_session.add_all.call_args[0][0]
        assert len(rows) == expected_chunks
        assert all(r.strategy == "paragraph" for r in rows)

    @pytest.mark.asyncio
    async def test_sentence_strategy_chunks_and_stores(self, async_client):
        text = "First sentence. Second sentence. Third sentence."

        with (
            patch("src.main.aembedding", new_callable=AsyncMock) as mock_embed,
            patch("src.main.qdrant_client.upsert", new_callable=AsyncMock),
            patch("src.main.AsyncSessionLocal") as mock_session_ctx,
        ):
            mock_embed.return_value = _make_embedding_response(3)
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            response = await async_client.post(
                "/embed", json={"text": text, "strategy": "sentence"}
            )

        assert response.status_code == 200
        body = response.json()
        assert body["strategy"] == "sentence"
        assert body["chunk_count"] == 3

    @pytest.mark.asyncio
    async def test_fixed_strategy_chunks_and_stores(self, async_client):
        text = _make_long_text(600)

        with (
            patch("src.main.aembedding", new_callable=AsyncMock) as mock_embed,
            patch("src.main.qdrant_client.upsert", new_callable=AsyncMock),
            patch("src.main.AsyncSessionLocal") as mock_session_ctx,
        ):
            # fixed strategy on 600-token text → 2 chunks
            mock_embed.return_value = _make_embedding_response(2)
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            response = await async_client.post(
                "/embed", json={"text": text, "strategy": "fixed"}
            )

        assert response.status_code == 200
        body = response.json()
        assert body["strategy"] == "fixed"
        assert body["chunk_count"] >= 2

    @pytest.mark.asyncio
    async def test_default_strategy_is_paragraph(self, async_client):
        text = "Only one paragraph, no blanks."

        with (
            patch("src.main.aembedding", new_callable=AsyncMock) as mock_embed,
            patch("src.main.qdrant_client.upsert", new_callable=AsyncMock),
            patch("src.main.AsyncSessionLocal") as mock_session_ctx,
        ):
            mock_embed.return_value = _make_embedding_response(1)
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            response = await async_client.post("/embed", json={"text": text})

        assert response.status_code == 200
        assert response.json()["strategy"] == "paragraph"

    @pytest.mark.asyncio
    async def test_empty_text_returns_400(self, async_client):
        response = await async_client.post(
            "/embed", json={"text": "   ", "strategy": "paragraph"}
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_chunk_index_is_sequential(self, async_client):
        text = "A.\n\nB.\n\nC."

        with (
            patch("src.main.aembedding", new_callable=AsyncMock) as mock_embed,
            patch("src.main.qdrant_client.upsert", new_callable=AsyncMock),
            patch("src.main.AsyncSessionLocal") as mock_session_ctx,
        ):
            mock_embed.return_value = _make_embedding_response(3)
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            await async_client.post(
                "/embed", json={"text": text, "strategy": "paragraph"}
            )

        rows = mock_session.add_all.call_args[0][0]
        assert [r.chunk_index for r in rows] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_invalid_strategy_returns_422(self, async_client):
        response = await async_client.post(
            "/embed", json={"text": "Hello world.", "strategy": "unknown_strategy"}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_qdrant_upsert_called_with_correct_point_count(self, async_client):
        text = "P1.\n\nP2."

        with (
            patch("src.main.aembedding", new_callable=AsyncMock) as mock_embed,
            patch(
                "src.main.qdrant_client.upsert", new_callable=AsyncMock
            ) as mock_upsert,
            patch("src.main.AsyncSessionLocal") as mock_session_ctx,
        ):
            mock_embed.return_value = _make_embedding_response(2)
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            await async_client.post(
                "/embed", json={"text": text, "strategy": "paragraph"}
            )

        mock_upsert.assert_called_once()
        _, kwargs = mock_upsert.call_args
        assert len(kwargs["points"]) == 2

    @pytest.mark.asyncio
    async def test_source_label_propagates_to_chunk_rows(self, async_client):
        text = "Hello.\n\nWorld."

        with (
            patch("src.main.aembedding", new_callable=AsyncMock) as mock_embed,
            patch("src.main.qdrant_client.upsert", new_callable=AsyncMock),
            patch("src.main.AsyncSessionLocal") as mock_session_ctx,
        ):
            mock_embed.return_value = _make_embedding_response(2)
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            await async_client.post(
                "/embed",
                json={"text": text, "strategy": "paragraph", "source": "my_doc.pdf"},
            )

        rows = mock_session.add_all.call_args[0][0]
        assert all(r.source == "my_doc.pdf" for r in rows)
