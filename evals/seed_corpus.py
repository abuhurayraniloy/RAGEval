"""
seed_corpus.py

Small sample corpus for testing RAGEval's /embed, /search, and /rag endpoints
(and for exercising hybrid search once you've added it).

Each entry is deliberately packed with exact technical terms (Qdrant, RRF,
BM25, asyncpg, etc.) that a pure dense-vector search can under-rank but that
keyword/sparse search hits directly. This makes it easy to see a difference
before/after adding hybrid search.

Usage:
    python seed_corpus.py [--base-url http://localhost:8000] [--strategy paragraph]

Each document is POSTed individually to /embed so chunk_index/source stay
meaningful per-topic.
"""

import argparse
import sys
import time

import httpx

DOCUMENTS: list[dict] = [
    {
        "source": "docs/architecture",
        "text": (
            "RAGEval is a FastAPI application that implements a complete "
            "retrieval augmented generation loop. Text is chunked, embedded, "
            "stored as vectors in Qdrant, retrieved by similarity search, "
            "reranked with a cross-encoder, and finally used as context for "
            "an LLM-generated answer."
        ),
    },
    {
        "source": "docs/caching",
        "text": (
            "The caching layer uses Redis to store RAG answers keyed by a "
            "SHA-256 hash of the question text. Cached entries expire after "
            "24 hours by default. On a cache hit, the pipeline skips "
            "embedding, vector search, reranking, and generation entirely, "
            "returning the stored result immediately."
        ),
    },
    {
        "source": "docs/database",
        "text": (
            "Completion logs and chunk metadata are persisted in PostgreSQL "
            "using SQLAlchemy's async ORM with the asyncpg driver. The "
            "Completion table stores prompt, response, model, and latency. "
            "The Chunk table stores point_id, text, strategy, and source for "
            "every chunk indexed into Qdrant."
        ),
    },
    {
        "source": "docs/embeddings",
        "text": (
            "Embeddings are generated through LiteLLM using the "
            "gemini-embedding-001 model, producing 1536-dimensional vectors. "
            "Both single-text and batch embedding functions are exposed as "
            "embed_text and embed_texts."
        ),
    },
    {
        "source": "docs/cache-miss",
        "text": (
            "When the Redis cache is empty for a given question, the "
            "pipeline runs the full sequence: embed the question, search "
            "Qdrant for candidate chunks, rerank them with a cross-encoder, "
            "generate an answer with the LLM, and finally write the result "
            "back to Redis with a 24 hour TTL."
        ),
    },
    {
        "source": "docs/qdrant-collection",
        "text": (
            "The Qdrant collection named 'embeddings' is created on "
            "application startup if it does not already exist, configured "
            "with 1536 dimensions and cosine distance. Vectors are upserted "
            "as PointStruct objects with a payload containing the chunk "
            "text, source, strategy, and chunk index."
        ),
    },
    {
        "source": "docs/chunking",
        "text": (
            "Three chunking strategies are supported: fixed, sentence, and "
            "paragraph. Fixed chunking splits text into 500-token windows "
            "with 50 tokens of overlap using tiktoken's cl100k_base "
            "encoding. Sentence chunking uses NLTK's punkt tokenizer. "
            "Paragraph chunking splits on blank lines."
        ),
    },
    {
        "source": "docs/embed-errors",
        "text": (
            "The /embed endpoint returns a 400 error if chunking produces no "
            "content, for example when the input text is empty or "
            "whitespace only. Any other failure during embedding or Qdrant "
            "upsert is caught and returned as a 500 error with a generic "
            "message, while the underlying exception is logged."
        ),
    },
    {
        "source": "docs/generation-model",
        "text": (
            "Answer generation defaults to the groq/llama-3.3-70b-versatile "
            "model through LiteLLM's acompletion call. The system prompt "
            "instructs the model to answer using only the provided context "
            "and to say so explicitly if the answer cannot be found there."
        ),
    },
    {
        "source": "docs/cache-ttl",
        "text": (
            "RAG results are cached in Redis for 24 hours, defined by the "
            "CACHE_TTL_SECONDS constant of 24 times 60 times 60 seconds. "
            "The TTL can be overridden per call to set_cache, but the "
            "default RAG pipeline always uses the standard value."
        ),
    },
    {
        "source": "docs/reranking",
        "text": (
            "Reranking is performed with a cross-encoder model, "
            "cross-encoder/ms-marco-MiniLM-L-6-v2, loaded lazily and cached "
            "as a module-level singleton. The rerank function scores "
            "query-candidate pairs and returns the top_k indices sorted by "
            "score descending."
        ),
    },
    {
        "source": "docs/rerank-candidates",
        "text": (
            "The RAG pipeline widens its initial vector search to "
            "RERANK_CANDIDATES_K, currently 20 candidates, before handing "
            "them to the cross-encoder reranker, which narrows the final "
            "result down to the top 5 sources used for answer generation."
        ),
    },
    {
        "source": "docs/evaluation",
        "text": (
            "The /evaluate endpoint runs a batch of question and expected "
            "answer pairs through the RAG pipeline with caching disabled, "
            "then judges each generated answer with a separate LLM call "
            "using the cerebras/gemma-4-31b judge model, returning 1 for "
            "correct or 0 for incorrect."
        ),
    },
    {
        "source": "docs/hybrid-search",
        "text": (
            "Hybrid search combines dense vector similarity with sparse "
            "keyword-based retrieval, typically BM25, and fuses the two "
            "rankings using Reciprocal Rank Fusion, or RRF. Qdrant supports "
            "this natively through named dense and sparse vectors on the "
            "same collection along with a fusion query."
        ),
    },
    {
        "source": "docs/docker",
        "text": (
            "The Docker image installs dependencies with uv sync using the "
            "ml extra so that sentence-transformers and torch are available "
            "for the reranker. The NLTK punkt_tab tokenizer data is "
            "downloaded at build time so sentence chunking works offline in "
            "production."
        ),
    },
]


def main():
    parser = argparse.ArgumentParser(description="Seed RAGEval with sample docs")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--strategy",
        default="paragraph",
        choices=["fixed", "sentence", "paragraph"],
        help="Chunking strategy to use for every document (default: paragraph, "
        "since these docs are already short single-paragraph chunks)",
    )
    args = parser.parse_args()

    ok, failed = 0, 0
    with httpx.Client(timeout=60.0) as client:
        for doc in DOCUMENTS:
            try:
                resp = client.post(
                    f"{args.base_url}/embed",
                    json={
                        "text": doc["text"],
                        "strategy": args.strategy,
                        "source": doc["source"],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                print(f"[ok] {doc['source']:<28} -> {data['chunk_count']} chunk(s)")
                ok += 1
            except httpx.HTTPError as e:
                print(f"[fail] {doc['source']:<28} -> {e}", file=sys.stderr)
                failed += 1
            time.sleep(0.1)  # be gentle on embedding API rate limits

    print(f"\nDone. {ok} succeeded, {failed} failed out of {len(DOCUMENTS)}.")


if __name__ == "__main__":
    main()
