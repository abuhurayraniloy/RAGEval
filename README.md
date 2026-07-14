# RAGEval

[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20search-FF4F64)](https://qdrant.tech/)
[![Redis](https://img.shields.io/badge/Redis-cache-DC382D?logo=redis&logoColor=white)](https://redis.io/)

RAGEval is a FastAPI service for RAG workflows. It chunks text, generates dense and sparse embeddings, stores vectors in Qdrant, reranks retrieved candidates, answers with context, caches repeated questions, and logs operational data to PostgreSQL.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running With Docker](#running-with-docker)
- [API Reference](#api-reference)
- [Development](#development)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Implementation Notes](#implementation-notes)

## Overview

The app is built for practical RAG iteration rather than a toy demo. It exposes endpoints for:

- streaming completions through LiteLLM-compatible providers
- chunking and embedding text before indexing it in Qdrant
- hybrid search over dense and sparse vectors
- reranked RAG answers with source snippets
- cached RAG responses for repeated questions
- batch evaluation of question/answer sets using an LLM judge
- API key minting for bootstrap and local development

Completion telemetry, chunk metadata, API keys, and rate-limit hits go to PostgreSQL. Embeddings are stored in Qdrant, and RAG responses are cached in Redis.

## Features

- FastAPI app served from `src.main:app`
- `/complete` for streamed LLM completions
- `/embed` for chunking and indexing text
- `/search` for hybrid vector retrieval
- `/rag` for reranked answers with cited sources
- `/evaluate` for batched RAG evaluation with an LLM judge
- `/api-keys` for bootstrap key creation behind an admin secret
- PostgreSQL persistence for completions, chunks, API keys, and rate-limit hits
- Redis caching for repeated RAG questions
- Rate limiting on protected routes
- Async tests for chunking, reranking, and cache behavior
- Docker Compose support for the full stack

## Architecture

- `src/main.py` wires the FastAPI app, lifespan startup, auth, and HTTP routes
- `src/routers/` contains the request handlers for completions, embeddings, search, RAG, evaluation, and API key creation
- `src/services/` contains embeddings, retrieval, reranking, generation, caching, auth, rate limiting, and judging logic
- `src/chunking/strategies.py` provides fixed, sentence, and paragraph chunking
- `src/clients/` creates the shared Qdrant and Redis clients
- `src/db/` defines the SQLAlchemy models and async session factory

## Tech Stack

- Python 3.14+
- FastAPI and Uvicorn
- LiteLLM
- SQLAlchemy async ORM with `asyncpg`
- PostgreSQL
- Qdrant
- Redis
- sentence-transformers for reranking
- NLTK and tiktoken for chunking
- uv for dependency management
- pytest, pytest-asyncio, httpx, and fakeredis for tests

## Prerequisites

- Python 3.14+
- uv
- Docker and Docker Compose
- API keys for the model providers you use through LiteLLM
- An `ADMIN_SECRET` value if you want to mint API keys through `/api-keys`

For the simplest local setup, run PostgreSQL, Qdrant, and Redis with Docker Compose and keep the FastAPI app on your host.

## Quick Start

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

Run the API:

```bash
uv run uvicorn src.main:app --reload
```

Open the API at:

```text
http://localhost:8000
```

Interactive docs are available at:

```text
http://localhost:8000/docs
```

## Configuration

Create a `.env` file in the project root.

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/rageval_logs
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
ADMIN_SECRET=<your_admin_secret>

GROQ_API_KEY=<your_groq_api_key>
GEMINI_API_KEY=<your_gemini_api_key>
QDRANT_API_KEY=<your_qdrant_api_key>
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

The app ensures the `embeddings` Qdrant collection exists on startup and checks Redis connectivity during lifespan initialization. The reranker is loaded lazily unless `PRELOAD_RERANKER=true`.

## Running With Docker

Build and start the full stack:

```bash
docker compose up --build
```

Compose starts these services:

- `app`: FastAPI API on port `8000`
- `db`: PostgreSQL on port `5432`
- `qdrant`: Qdrant on ports `6333` and `6334`
- `redis`: Redis on port `6379`

The container image uses `uv` and includes the NLTK tokenizer data needed for sentence chunking.

## API Reference

### Health Check

```http
GET /
```

Returns a simple JSON response:

```json
{
  "message": "Hello from RAGEval API"
}
```

### Authentication

Protected endpoints require an `X-API-Key` header. The `/api-keys` endpoint requires an `X-Admin-Secret` header that matches `ADMIN_SECRET` on the server.

The protected routes are rate limited to 60 requests per hour per key.

### Stream a Completion

```http
POST /complete
Content-Type: application/json
X-API-Key: <api-key>
```

Request:

```json
{
  "prompt": "Explain retrieval augmented generation in one paragraph.",
  "model": "groq/llama-3.3-70b-versatile",
  "max_tokens": 500
}
```

This endpoint streams plain text back to the client and stores the prompt, response, model, and latency in PostgreSQL after the stream finishes.

### Embed Text

```http
POST /embed
Content-Type: application/json
X-API-Key: <api-key>
```

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

The endpoint chunks the text, generates dense Gemini embeddings plus sparse BM25 vectors, stores them in Qdrant, and records chunk metadata in PostgreSQL.

### Search Embeddings

```http
POST /search
Content-Type: application/json
X-API-Key: <api-key>
```

Request:

```json
{
  "query": "vector database for semantic search",
  "top_k": 5
}
```

This endpoint embeds the query, runs hybrid dense + sparse search against Qdrant, and returns the matching chunk text plus scores.

### Generate a RAG Answer

```http
POST /rag
Content-Type: application/json
X-API-Key: <api-key>
```

Request:

```json
{
  "question": "What is Qdrant used for?"
}
```

The `/rag` endpoint:

- checks Redis for a cached answer
- embeds the question with Gemini
- retrieves candidate chunks from Qdrant using dense and sparse vectors
- reranks the candidates with a cross-encoder
- asks the default Groq model to answer using only the retrieved context
- caches the result in Redis for 24 hours

The response includes the answer, source chunks, vector scores, rerank scores, latency, and cache-derived results when available.

### Evaluate a Batch

```http
POST /evaluate
Content-Type: application/json
X-API-Key: <api-key>
```

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

The `/evaluate` endpoint runs each question through the RAG pipeline without cache, scores the answer with `cerebras/gemma-4-31b`, and returns aggregate metrics such as accuracy, latency, cost, and per-question results.

### Mint an API Key

```http
POST /api-keys
Content-Type: application/json
X-Admin-Secret: <admin-secret>
```

Request:

```json
{
  "name": "local-dev"
}
```

The response returns the full API key once, along with its id and prefix. Store the key securely because it is not shown again.

## Development

Run the test suite:

```bash
uv run pytest
```

Useful focused checks:

```bash
uv run pytest tests/test_chunking.py
uv run pytest tests/test_reranker.py
uv run pytest tests/test_rag_cache.py
```

The tests rely on mocked external services, which keeps them fast and deterministic. `fakeredis` covers cache behavior in memory.

## Evaluation

The evaluation script at `evals/eval_reranking.py` compares raw retrieval against reranked RAG answers over an indexed corpus.

Run it against a live API:

```bash
python evals/eval_reranking.py --base-url http://localhost:8000
```

## Project Structure

```text
.
├── alembic/
├── evals/
│   └── eval_reranking.py
├── src/
│   ├── chunking/
│   ├── clients/
│   ├── db/
│   ├── routers/
│   └── services/
├── tests/
│   ├── test_chunking.py
│   ├── test_rag_cache.py
│   └── test_reranker.py
├── Dockerfile
├── README.md
├── alembic.ini
├── docker-compose.yml
└── pyproject.toml
```

## Implementation Notes

- Chunking strategies are `fixed`, `sentence`, and `paragraph`.
- The reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- The judge model is `cerebras/gemma-4-31b`.
- RAG cache keys are SHA-256 hashes of the question with a 24 hour TTL.
- The app uses PostgreSQL models for completions, chunks, API keys, and rate-limit hits.
- The app creates the `embeddings` Qdrant collection on startup and downloads the NLTK punkt tokenizer data when needed.
