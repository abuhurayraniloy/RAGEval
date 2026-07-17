# RAGEval

RAGEval is a FastAPI service for building, testing, and operating retrieval-augmented generation workflows. It supports streamed LLM completions, document ingestion, hybrid retrieval, reranking, cached RAG answers, and batch evaluation with an LLM judge.

## Features

- Streamed completions through LiteLLM-compatible providers
- PDF ingestion with background chunking, embedding, and indexing
- Hybrid dense and sparse retrieval backed by Qdrant
- Cross-encoder reranking for improved answer quality
- Redis-backed caching for repeated RAG questions
- Batch evaluation for question and answer sets
- PostgreSQL persistence for completions, document metadata, API keys, and rate-limit hits
- API-key authentication with an admin bootstrap flow for key creation

## Tech Stack

- Python 3.14+
- FastAPI
- LiteLLM
- PostgreSQL with SQLAlchemy and asyncpg
- Qdrant for vector search
- Redis for caching
- sentence-transformers for reranking
- NLTK and tiktoken for chunking
- uv for dependency management

## Project Structure

```text
src/
  main.py              FastAPI application and startup lifecycle
  routers/             HTTP route handlers
  services/             Retrieval, generation, embedding, cache, auth, and evaluation logic
  chunking/             Text chunking strategies
  clients/              Shared Qdrant and Redis clients
  db/                  SQLAlchemy models and async session setup
tests/                 Unit and async tests
evals/                 Evaluation scripts and load-testing helpers
alembic/               Database migrations
```

## Prerequisites

- Python 3.14 or newer
- `uv`
- Docker and Docker Compose
- Access credentials for the model providers you configure
- `ADMIN_SECRET` if you plan to mint API keys through the bootstrap endpoint

## Installation

Install dependencies:

```bash
uv sync
```

Start the backing services:

```bash
docker compose up db qdrant redis
```

Run database migrations:

```bash
uv run alembic upgrade head
```

Start the API:

```bash
uv run uvicorn src.main:app --reload
```

The API is available at `http://localhost:8000`, and the interactive docs are at `http://localhost:8000/docs`.

## Configuration

Create a `.env` file in the project root.

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/rageval_logs
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
ADMIN_SECRET=change-me

GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
QDRANT_API_KEY=your_qdrant_api_key
PRELOAD_RERANKER=false
```

### Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL async connection string | `postgresql+asyncpg://user:password@db:5432/rageval_logs` |
| `QDRANT_URL` | Qdrant base URL | required |
| `QDRANT_API_KEY` | Qdrant Cloud API key | unset |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `GROQ_API_KEY` | Used by the default completion and RAG answer model | unset |
| `GEMINI_API_KEY` | Used for embeddings | unset |
| `ADMIN_SECRET` | Required to call `/api-keys` | unset |
| `PRELOAD_RERANKER` | Preloads the cross-encoder during startup when set to `true` | `false` |

On startup, the app ensures the `embeddings` Qdrant collection exists, checks Redis connectivity, and loads the reranker lazily unless `PRELOAD_RERANKER=true`.

## Running With Docker

Build and start the full stack:

```bash
docker compose up --build
```

Compose starts:

- `app` on port `8000`
- `db` on port `5432`
- `qdrant` on ports `6333` and `6334`
- `redis` on port `6379`

## API Overview

### Health Check

```http
GET /
```

Response:

```json
{ "message": "Hello from RAGEval API" }
```

### Authentication

- Protected endpoints require an `X-API-Key` header.
- `/api-keys` requires an `X-Admin-Secret` header that matches `ADMIN_SECRET`.
- Protected routes are rate limited to `60 requests/hour` per key.

### `POST /complete`

Streams a completion for the provided prompt.

Request:

```json
{
  "prompt": "Explain retrieval augmented generation in one paragraph.",
  "model": "groq/llama-3.3-70b-versatile",
  "max_tokens": 500
}
```

### `POST /embed`

Chunks input text, generates embeddings, and indexes the chunks in Qdrant.

Request:

```json
{
  "text": "Qdrant is a vector database for similarity search.",
  "strategy": "paragraph",
  "source": "api_upload"
}
```

Supported chunking strategies:

- `fixed`
- `sentence`
- `paragraph`

### `POST /search`

Runs hybrid retrieval over dense and sparse vectors and returns matching chunks with scores.

Request:

```json
{
  "query": "vector database for semantic search",
  "top_k": 5
}
```

### `POST /rag`

Runs the RAG pipeline with caching, retrieval, reranking, and answer generation.

Request:

```json
{
  "question": "What is Qdrant used for?"
}
```

### `POST /evaluate`

Evaluates a batch of questions against the RAG pipeline and scores outputs with an LLM judge.

Request:

```json
{
  "questions": [
    {
      "question": "What is Qdrant used for?",
      "expected": "A vector database for similarity search."
    }
  ]
}
```

### `POST /api-keys`

Creates a new API key using the admin secret bootstrap flow.

Request:

```json
{
  "name": "local-dev"
}
```

### `POST /ingest`

Uploads a PDF for background extraction, chunking, embedding, and indexing.

### `GET /documents/{document_id}`

Checks ingestion status for a previously uploaded document.

## Development

Run the test suite:

```bash
uv run pytest
```

Focused test runs:

```bash
uv run pytest tests/test_chunking.py
uv run pytest tests/test_reranker.py
uv run pytest tests/test_rag_cache.py
```

## Evaluation

The `evals/` directory includes scripts for reranking and load testing. For example:

```bash
python evals/eval_reranking.py --base-url http://localhost:8000
```

## Notes

- The reranker is loaded lazily by default and can be preloaded with `PRELOAD_RERANKER=true`.
- The RAG cache uses a 24-hour TTL.
- The API stores completion telemetry and ingestion metadata in PostgreSQL.
- Qdrant payload indexes are created for `category` and `source` to support filtering.
