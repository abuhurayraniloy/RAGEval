"""Text chunking strategies for splitting documents into smaller pieces."""

import re
from enum import Enum
from typing import List

import nltk
import logging
import tiktoken
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

# Ensure punkt_tab is available for sentence tokenization
for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
    try:
        nltk.data.find(resource)
        break
    except LookupError:
        continue
else:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception as e:
        logger.error(f"Failed to download nltk punkt_tab: {str(e)}")


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


_ENCODING = tiktoken.get_encoding("cl100k_base")

FIXED_CHUNK_TOKENS = 500
FIXED_CHUNK_OVERLAP = 50


def _chunk_fixed(
    text: str, chunk_size: int = FIXED_CHUNK_TOKENS, overlap: int = FIXED_CHUNK_OVERLAP
) -> List[str]:
    if not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk size")

    tokens = _ENCODING.encode(text)
    if not tokens:
        return []

    chunks = []

    step = chunk_size - overlap

    start = 0

    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        chunks.append(_ENCODING.decode(window))

        if start + chunk_size >= len(tokens):
            break
        start += step

    return chunks


def _chunk_sentence(text: str) -> List[str]:
    if not text.strip():
        return []
    sentences = sent_tokenize(text)

    return [s.strip() for s in sentences if s.strip()]


def _chunk_paragraph(text: str) -> List[str]:
    if not text.strip():
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


_STRATEGY_FUNCS = {
    ChunkStrategy.FIXED: _chunk_fixed,
    ChunkStrategy.SENTENCE: _chunk_sentence,
    ChunkStrategy.PARAGRAPH: _chunk_paragraph,
}


def chunk_text(text: str, strategy: ChunkStrategy) -> List[str]:
    func = _STRATEGY_FUNCS.get(strategy)
    if func is None:
        raise ValueError(f"Unkown strategy")
    return func(text)
