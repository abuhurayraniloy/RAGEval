"""Embedding generation service using LiteLLM."""

from litellm import aembedding
from fastembed import SparseTextEmbedding

_sparse_model: SparseTextEmbedding | None = None


def get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


def embed_sparse(text: str) -> dict:
    """Generate a sparse (BM25) embedding for a single text"""
    model = get_sparse_embedder()
    result = next(model.embed([text]))
    return {"indices": result.indices.tolist(), "values": result.values.tolist()}


def embed_sparse_batch(texts: list[str]) -> list[dict]:
    model = get_sparse_embedder()
    return [
        {"indices": r.indices.tolist(), "values": r.values.tolist()}
        for r in model.embed(texts)
    ]


async def embed_text(
    text: str, model: str = "gemini/gemini-embedding-001", dimensions: int = 1536
) -> list[float]:
    """Generate embeddings for text using the specified model.

    Args:
        text: Text to embed
        model: Model identifier for embedding
        dimensions: Embedding dimensions

    Returns:
        List of embedding floats
    """
    response = await aembedding(model=model, input=[text], dimensions=dimensions)
    return response.data[0].embedding


async def embed_texts(
    texts: list[str], model: str = "gemini/gemini-embedding-001", dimensions: int = 1536
) -> list[list[float]]:
    """Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        model: Model identifier for embedding
        dimensions: Embedding dimensions

    Returns:
        List of embedding vectors
    """
    response = await aembedding(model=model, input=texts, dimensions=dimensions)
    return [
        item.embedding if hasattr(item, "embedding") else item["embedding"]
        for item in response.data
    ]
