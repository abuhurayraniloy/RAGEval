"""Chunking module for text splitting strategies.

Exposes:
- ChunkStrategy: Enumeration of available chunking strategies
- chunk_text: Main function to chunk text using specified strategy
"""

from src.chunking.strategies import ChunkStrategy, chunk_text

__all__ = ["ChunkStrategy", "chunk_text"]
