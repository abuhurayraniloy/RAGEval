# RAGEval

[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20search-FF4F64)](https://qdrant.tech/)
[![Redis](https://img.shields.io/badge/Redis-cache-DC382D?logo=redis&logoColor=white)](https://redis.io/)

RAGEval is a FastAPI playground for RAG workflows. It covers the full loop: chunk text, embed it, store vectors, retrieve candidates, rerank them, answer with context, and cache the result.

## Table of Contents

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
- semantic search over stored chunks
- reranked RAG answers with source snippets
- cached RAG responses for repeated questions

Completion telemetry and chunk metadata go to PostgreSQL, embeddings go to Qdrant, and RAG responses are cached in Redis.

## Features

- FastAPI app served from `src.main:app`
- `/complete` for streamed LLM completions
- `/embed` for chunking and indexing text
- `/search` for semantic retrieval in Qdrant
- `/rag` for reranked answers with cited sources
- PostgreSQL persistence for completion logs and chunk records
- Redis caching for repeated RAG questions
- Async tests for chunking, reranking, and cache behavior
- Docker Compose support for the full stack

## Architecture

- `src/main.py` wires the FastAPI app, lifespan startup, and HTTP routes
- `src/chunking.py` provides the supported chunking strategies
- `src/reranker.py` loads the cross-encoder reranker and scores candidate chunks
- `src/database.py` configures the async SQLAlchemy engine and session factory
- `src/qdrant.py` creates the async Qdrant client
- `src/redis_client.py` configures the Redis cache client

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
DATABASE_URL=postgresql+asyncpg://<user>:<password>@localhost:5432/rageval_logs
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0

GROQ_API_KEY=<your_groq_api_key>
GEMINI_API_KEY=<your_gemini_api_key>
QDRANT_API_KEY=<your_qdrant_api_key>
```

### Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL async connection string | `postgresql+asyncpg://<user>:<password>@db:5432/rageval_logs` |
| `QDRANT_URL` | Qdrant base URL | required |
| `QDRANT_API_KEY` | Qdrant Cloud API key | unset |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `GROQ_API_KEY` | Used by the default completion and RAG answer model | unset |
| `GEMINI_API_KEY` | Used for embeddings | unset |

The app creates PostgreSQL tables and ensures the `embeddings` Qdrant collection exists on startup. It also loads the reranker and tokenizer resources during lifespan startup when available.

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
  "Message": "Hello from the root"
}
```

### Stream a Completion

```http
POST /complete
Content-Type: application/json
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

The endpoint chunks the text, generates Gemini embeddings, stores vectors in Qdrant, and records chunk metadata in PostgreSQL.

### Search Embeddings

```http
POST /search
Content-Type: application/json
```

Request:

```json
{
  "query": "vector database for semantic search",
  "top_k": 5
}
```

This endpoint embeds the query, searches Qdrant, and returns the matching chunk text plus scores.

### Generate a RAG Answer

```http
POST /rag
Content-Type: application/json
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
- retrieves candidate chunks from Qdrant
- reranks the candidates with a cross-encoder
- asks the default Groq model to answer using only the retrieved context
- caches the result in Redis for 24 hours

The response includes the answer and source chunks with vector and rerank scores.

## Development

Run the tests:

```bash
uv run pytest
```

Useful targeted tests:

```bash
uv run pytest tests/test_chunking.py
uv run pytest tests/test_reranker.py
uv run pytest tests/test_rag_cache.py
```

The test suite uses mocked external services for fast, deterministic coverage. `fakeredis` validates cache behavior in memory.

## Evaluation

An evaluation script is provided at `evals/eval_reranking.py` to compare raw vector search against reranked RAG results on your own indexed corpus.

Run it against a live stack:

```bash
python evals/eval_reranking.py --base-url http://localhost:8000
```

## Project Structure

```text
.
├── evals/
│   └── eval_reranking.py
├── src/
│   ├── chunking.py
│   ├── database.py
│   ├── main.py
│   ├── models.py
│   ├── qdrant.py
│   ├── redis_client.py
│   └── reranker.py
├── tests/
│   ├── test_chunking.py
│   ├── test_rag_cache.py
│   └── test_reranker.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Implementation Notes

- The default completion model is `groq/llama-3.3-70b-versatile`.
- The embedding model is `gemini/gemini-embedding-001` with 1536 dimensions.
- The Qdrant collection name is `embeddings`.
- Chunk metadata is stored in PostgreSQL with the chunk text, strategy, source, and index.
- The reranker uses `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Startup includes a best-effort download or reuse of NLTK tokenizer data for sentence chunking.
- For production, database migrations are preferable to creating tables directly at startup.