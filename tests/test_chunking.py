"""Unit tests for chunking strategies and the embed handler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastapi import HTTPException

from src.chunking import ChunkStrategy, chunk_text
from src.chunking.strategies import FIXED_CHUNK_OVERLAP, FIXED_CHUNK_TOKENS
from src.routers.embed import EmbedRequest, embed_text_handler


def _make_long_text(approx_tokens: int) -> str:
    return ("word ") * approx_tokens


def _fake_sparse_batch(n: int):
    return [{"indices": [], "values": []} for _ in range(n)]


class TestChunkParagraph:
    def test_splits_on_double_newline(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        assert chunk_text(text, ChunkStrategy.PARAGRAPH) == [
            "First paragraph.",
            "Second paragraph.",
            "Third paragraph.",
        ]

    def test_ignores_blank_sections(self):
        assert chunk_text("A.\n\n  \n\nB.", ChunkStrategy.PARAGRAPH) == ["A.", "B."]

    def test_empty_input_returns_empty_list(self):
        assert chunk_text("   ", ChunkStrategy.PARAGRAPH) == []


class TestChunkSentence:
    def test_splits_into_sentences(self):
        chunks = chunk_text(
            "Hello world. This is a test. Here is another sentence.",
            ChunkStrategy.SENTENCE,
        )
        assert len(chunks) == 3
        assert all(chunk[-1] in ".!?" for chunk in chunks)

    def test_empty_input_returns_empty_list(self):
        assert chunk_text("", ChunkStrategy.SENTENCE) == []


class TestChunkFixed:
    def test_short_text_returns_single_chunk(self):
        assert chunk_text("Short text.", ChunkStrategy.FIXED) == ["Short text."]

    def test_long_text_produces_multiple_chunks(self):
        assert len(chunk_text(_make_long_text(1100), ChunkStrategy.FIXED)) >= 3

    def test_no_chunk_exceeds_max_token_size(self):
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        chunks = chunk_text(_make_long_text(2000), ChunkStrategy.FIXED)
        assert all(len(enc.encode(chunk)) <= FIXED_CHUNK_TOKENS for chunk in chunks)

    def test_chunks_overlap(self):
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        chunks = chunk_text(_make_long_text(1100), ChunkStrategy.FIXED)

        for index in range(len(chunks) - 1):
            left_tokens = enc.encode(chunks[index])
            right_tokens = enc.encode(chunks[index + 1])
            assert (
                left_tokens[-FIXED_CHUNK_OVERLAP:] == right_tokens[:FIXED_CHUNK_OVERLAP]
            )


class TestChunkDispatch:
    @pytest.mark.parametrize("strategy", list(ChunkStrategy))
    def test_returns_list(self, strategy):
        assert isinstance(chunk_text("Hello world.", strategy), list)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            chunk_text("some text", "nonexistent_strategy")  # type: ignore[arg-type]


class TestEmbedHandler:
    @pytest.mark.asyncio
    async def test_paragraph_strategy_chunks_and_stores(self):
        request = EmbedRequest(
            text="First.\n\nSecond.\n\nThird.", strategy=ChunkStrategy.PARAGRAPH
        )

        with (
            patch(
                "src.routers.embed.embed_texts",
                new_callable=AsyncMock,
            ) as mock_embed,
            patch(
                "src.routers.embed.embed_sparse_batch",
                return_value=_fake_sparse_batch(3),
            ),
            patch(
                "src.routers.embed.qdrant_client.upsert",
                new_callable=AsyncMock,
            ) as mock_upsert,
            patch("src.routers.embed.AsyncSessionLocal") as mock_session_ctx,
        ):
            mock_embed.return_value = [[0.1] * 1536 for _ in range(3)]

            mock_session = MagicMock()
            mock_session.commit = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            response = await embed_text_handler(request)

        assert response["strategy"] == "paragraph"
        assert response["chunk_count"] == 3
        assert len(response["point_ids"]) == 3
        mock_upsert.assert_awaited_once()
        mock_session.add_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_source_is_api_upload(self):
        request = EmbedRequest(text="Just one paragraph.")

        with (
            patch(
                "src.routers.embed.embed_texts",
                new_callable=AsyncMock,
            ) as mock_embed,
            patch(
                "src.routers.embed.embed_sparse_batch",
                return_value=_fake_sparse_batch(1),
            ),
            patch(
                "src.routers.embed.qdrant_client.upsert",
                new_callable=AsyncMock,
            ),
            patch("src.routers.embed.AsyncSessionLocal") as mock_session_ctx,
        ):
            mock_embed.return_value = [[0.1] * 1536]

            mock_session = MagicMock()
            mock_session.commit = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            response = await embed_text_handler(request)

        rows = mock_session.add_all.call_args.args[0]
        assert response["strategy"] == "paragraph"
        assert rows[0].source == "api_upload"

    @pytest.mark.asyncio
    async def test_empty_text_returns_400(self):
        request = EmbedRequest(text="   ", strategy=ChunkStrategy.PARAGRAPH)

        with pytest.raises(HTTPException) as exc_info:
            await embed_text_handler(request)

        assert exc_info.value.status_code == 400
