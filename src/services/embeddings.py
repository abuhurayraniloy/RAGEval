"""Embedding generation service using LiteLLM."""

from litellm import aembedding


async def embed_text(text: str, model: str = "gemini/gemini-embedding-001", dimensions: int = 1536) -> list[float]:
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


async def embed_texts(texts: list[str], model: str = "gemini/gemini-embedding-001", dimensions: int = 1536) -> list[list[float]]:
    """Generate embeddings for multiple texts.
    
    Args:
        texts: List of texts to embed
        model: Model identifier for embedding
        dimensions: Embedding dimensions
        
    Returns:
        List of embedding vectors
    """
    response = await aembedding(model=model, input=texts, dimensions=dimensions)
    return [item.embedding if hasattr(item, 'embedding') else item['embedding'] for item in response.data]
