# RAGEval

An authenticated FastAPI service for building, operating, and evaluating retrieval-augmented generation pipelines.

RAGEval gives you the backend pieces most RAG prototypes eventually need: document ingestion, dense and sparse retrieval, reranking, answer generation, response caching, API-key auth, rate limiting, and batch evaluation with an LLM judge. It is designed as a service you can run locally with Docker or embed into a larger internal evaluation stack.

The codebase is intentionally small. FastAPI handles the public API, PostgreSQL stores operational records, Qdrant stores dense and sparse vectors, Redis caches repeated RAG answers, and LiteLLM routes model calls to configured providers.

## Features

- **Authenticated RAG API**
  Protected endpoints require `X-API-Key`. API keys are generated through an admin bootstrap endpoint, stored as SHA-256 hashes, and shown only once.

- **Hybrid retrieval by default**
  Text is embedded into Gemini dense vectors and FastEmbed BM25 sparse vectors. Qdrant fuses both retrieval legs with reciprocal rank fusion so lexical matches and semantic matches can both surface.

- **Cross-encoder reranking**
  RAG queries retrieve a wider candidate set, then rerank it with `cross-encoder/ms-marco-MiniLM-L-6-v2`. The final answer receives the top reranked context chunks instead of raw vector-search order.

- **PDF and text ingestion**
  Upload PDFs for background extraction, paragraph chunking, embedding, indexing, and status tracking. Send raw text directly to `/embed` when you already have extracted content.

- **Cached RAG responses**
  Repeated exact questions are cached in Redis for 24 hours using a stable SHA-256 cache key. Evaluation requests bypass the cache so results reflect the current retrieval stack.

- **LLM-judged evaluation**
  Batch evaluation runs the full RAG pipeline for each question and compares the answer against an expected answer with a strict judge model.

- **Operational persistence**
  PostgreSQL stores completions, indexed chunk metadata, document ingestion state, API keys, and rate-limit hits. Alembic owns schema migrations.

## Architecture

```mermaid
flowchart LR
    Client[Client or evaluator] --> API[FastAPI app]
    API --> Routers[HTTP routers]
    Routers --> Services[Service layer]

    Services --> Auth[Auth and rate limiting]
    Services --> Generation[LiteLLM generation]
    Services --> Embeddings[LiteLLM embeddings]
    Services --> Retrieval[Hybrid retrieval]
    Services --> Reranker[Cross-encoder reranker]
    Services --> Cache[Redis cache]
    Services --> DB[(PostgreSQL)]
    Retrieval --> Qdrant[(Qdrant embeddings collection)]

    Generation --> Providers[Model providers]
    Embeddings --> Providers
    Auth --> DB
    Cache --> Redis[(Redis)]
```

RAGEval keeps framework code thin. `src/main.py` wires application startup and route registration, `src/routers/` defines HTTP request/response boundaries, and `src/services/` contains the retrieval, generation, ingestion, caching, auth, and evaluation logic.

### RAG Request Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI
    participant R as RAG pipeline
    participant Redis as Redis
    participant Q as Qdrant
    participant X as CrossEncoder
    participant L as LiteLLM

    C->>A: POST /rag { question }
    A->>R: authenticated request
    R->>Redis: lookup rag:{sha256(question)}
    alt cache hit
        Redis-->>R: cached answer
        R-->>A: answer, sources, latency
    else cache miss
        R->>L: embed question with gemini/gemini-embedding-001
        R->>R: create BM25 sparse embedding
        R->>Q: dense + sparse prefetch with RRF fusion
        Q-->>R: 20 retrieval candidates
        R->>X: rerank query/document pairs
        X-->>R: top 5 contexts
        R->>L: generate answer from context
        R->>Redis: cache result for 24 hours
        R-->>A: answer, sources, latency
    end
    A-->>C: JSON response
```

### PDF Ingestion Lifecycle

```mermaid
flowchart TD
    Upload[POST /ingest PDF] --> Validate[Validate content type, size, and non-empty body]
    Validate --> Row[Create documents row with processing status]
    Row --> Background[Schedule FastAPI background task]
    Background --> Extract[Extract page text with PyMuPDF]
    Extract --> Chunk[Paragraph chunk pages]
    Chunk --> Dense[Create dense embeddings]
    Chunk --> Sparse[Create BM25 sparse embeddings]
    Dense --> Upsert[Upsert named vectors into Qdrant]
    Sparse --> Upsert
    Upsert --> Chunks[Persist chunk metadata in PostgreSQL]
    Chunks --> Status[Mark document completed or failed]
    Status --> Poll[GET /documents/{document_id}]
```

## Tech Stack

| Layer | Technology | Role |
| --- | --- | --- |
| API | FastAPI, Uvicorn | Async HTTP API and OpenAPI docs |
| Model access | LiteLLM | Provider-agnostic completions and embeddings |
| Dense embeddings | `gemini/gemini-embedding-001` | 1536-dimensional semantic vectors |
| Sparse embeddings | FastEmbed `Qdrant/bm25` | Lexical retrieval signal |
| Vector database | Qdrant | Named dense/sparse vectors and RRF hybrid search |
| Reranking | sentence-transformers CrossEncoder | Query/document relevance scoring |
| Cache | Redis | 24-hour exact-question RAG cache |
| Persistence | PostgreSQL, SQLAlchemy, asyncpg | Logs, metadata, API keys, rate-limit hits |
| Migrations | Alembic | Database schema management |
| Chunking | NLTK, tiktoken | Sentence, paragraph, and fixed-token chunking |
| Dependency management | uv | Reproducible Python environment |
| Load testing | Locust | `/rag` traffic simulation |

## Project Structure

```text
src/
  main.py          FastAPI app, startup lifecycle, route registration
  routers/         HTTP endpoints and Pydantic request models
  services/        RAG, retrieval, generation, ingestion, auth, cache, evaluation
  clients/         Qdrant and Redis clients
  chunking/        Fixed, sentence, and paragraph chunking strategies
  db/              SQLAlchemy models and async session setup
alembic/           PostgreSQL migrations
tests/             Unit and async tests for chunking, reranking, and cache behavior
evals/             Reranking comparison, seed helpers, memory experiments, Locust load test
```

Conversation-memory tables and evaluation helpers exist in the repository, but the main FastAPI app does not currently expose a conversation-memory API. They are intentionally not presented as a public feature here.

## Installation

### Prerequisites

- Python 3.14 or newer
- `uv`
- Docker and Docker Compose
- Model-provider credentials for the providers you call through LiteLLM
- An `ADMIN_SECRET` value for creating API keys

### Local Development

Install dependencies, including the optional ML extras required by the reranker:

```bash
uv sync --extra ml
```

Start PostgreSQL, Qdrant, and Redis:

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

The API runs at `http://localhost:8000`. OpenAPI docs are available at `http://localhost:8000/docs`.

### Docker

Build and start the full stack:

```bash
docker compose up --build
```

Compose starts:

| Service | Port(s) | Purpose |
| --- | --- | --- |
| `app` | `8000` | FastAPI service |
| `db` | `5432` | PostgreSQL database |
| `qdrant` | `6333`, `6334` | Vector search |
| `redis` | `6379` | RAG cache |

The application container runs Alembic migrations before starting Uvicorn. `PORT` can override the app port inside the container; it defaults to `8000`.

## Configuration

Create a `.env` file in the repository root:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/rageval_logs
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
REDIS_URL=redis://localhost:6379/0
ADMIN_SECRET=change-me
PRELOAD_RERANKER=false

GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `DATABASE_URL` | No | `postgresql+asyncpg://user:password@db:5432/rageval_logs` | Async PostgreSQL connection string used by the app and Alembic. |
| `QDRANT_URL` | Yes | None | Qdrant URL. Use `http://localhost:6333` locally. |
| `QDRANT_API_KEY` | No | None | API key for Qdrant Cloud or secured Qdrant deployments. |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection string for RAG response caching. |
| `ADMIN_SECRET` | Yes for `/api-keys` | None | Shared secret required in `X-Admin-Secret` to create API keys. |
| `PRELOAD_RERANKER` | No | `false` | Set to `true` to load the cross-encoder during startup instead of on first use. |
| `PORT` | No | `8000` | Container Uvicorn port used by the Docker `CMD`. |
| `GROQ_API_KEY` | Provider-dependent | None | Used by the default `groq/llama-3.3-70b-versatile` generation model. |
| `GEMINI_API_KEY` | Provider-dependent | None | Used by the default `gemini/gemini-embedding-001` embedding model. |
| `LOCUST_API_KEY` | Only for load tests | Empty string | API key sent by `evals/locustfile.py`. |

Provider keys are read by LiteLLM. If you change request models to use different providers, configure the corresponding LiteLLM-supported environment variables.

## Quickstart

After the API and backing services are running, create an API key:

```bash
curl -X POST http://localhost:8000/api-keys \
  -H "Content-Type: application/json" \
  -H "X-Admin-Secret: change-me" \
  -d '{"name":"local-dev"}'
```

Example response:

```json
{
  "status": "success",
  "api_key": "rge_example",
  "id": 1,
  "prefix": "rge_example",
  "message": "Store this key securely - it will not be shown again."
}
```

Store the returned key and send it as `X-API-Key` on protected endpoints.

Embed text into Qdrant:

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -H "X-API-Key: rge_example" \
  -d '{
    "text": "Qdrant stores vectors and supports hybrid retrieval.",
    "strategy": "paragraph",
    "source": "docs",
    "category": "technical"
  }'
```

Ask a RAG question:

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -H "X-API-Key: rge_example" \
  -d '{"question":"What does Qdrant store?"}'
```

Example response shape:

```json
{
  "answer": "Qdrant stores vectors for similarity search.",
  "sources": [
    {
      "id": "4b7f8e8f-7d1a-4b47-bc9f-8d6f6d1e41d5",
      "vector_score": 0.83,
      "rerank_score": 4.12,
      "text": "Qdrant stores vectors and supports hybrid retrieval."
    }
  ],
  "latency_ms": 1240,
  "llm_cost_usd": 0.0,
  "status": "success",
  "question": "What does Qdrant store?"
}
```

`llm_cost_usd` is currently a placeholder in the RAG response. Judge cost is attempted during `/evaluate` through LiteLLM's cost helper.

## API Reference

Protected endpoints require `X-API-Key`. The API-key creation endpoint requires `X-Admin-Secret`.

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/` | None | Health check message. |
| `POST` | `/api-keys` | `X-Admin-Secret` | Create and return a new API key once. |
| `POST` | `/complete` | `X-API-Key` | Stream a direct LLM completion as `text/plain`. |
| `POST` | `/embed` | `X-API-Key` | Chunk text, embed dense/sparse vectors, and index in Qdrant. |
| `POST` | `/search` | `X-API-Key` | Run hybrid vector search with optional exact-match payload filters. |
| `POST` | `/rag` | `X-API-Key` | Run cached retrieval, reranking, and answer generation. |
| `POST` | `/evaluate` | `X-API-Key` | Run uncached RAG evaluation with an LLM judge. |
| `POST` | `/ingest` | `X-API-Key` | Upload a PDF for background ingestion. |
| `GET` | `/documents/{document_id}` | `X-API-Key` | Poll PDF ingestion status. |

Protected routes are rate limited to `60/hour`. Rate-limit violations return `429` and are logged to PostgreSQL.

### `POST /complete`

Streams a direct model completion without retrieval.

```bash
curl -N -X POST http://localhost:8000/complete \
  -H "Content-Type: application/json" \
  -H "X-API-Key: rge_example" \
  -d '{
    "prompt": "Explain retrieval augmented generation in one paragraph.",
    "model": "groq/llama-3.3-70b-versatile",
    "max_tokens": 500
  }'
```

### `POST /embed`

Indexes raw text. Supported chunking strategies are `paragraph`, `sentence`, and `fixed`.

```json
{
  "text": "First paragraph.\n\nSecond paragraph.",
  "strategy": "paragraph",
  "source": "api_upload",
  "category": "general"
}
```

### `POST /search`

Runs hybrid search over the `embeddings` Qdrant collection. Filters are exact-match payload filters and are applied to both dense and sparse retrieval.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: rge_example" \
  -d '{
    "query": "semantic vector search",
    "top_k": 5,
    "filter": { "category": "technical" }
  }'
```

### `POST /ingest`

Uploads a PDF up to 25 MB. The endpoint returns immediately while a background task extracts, chunks, embeds, and indexes the document.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: rge_example" \
  -F "file=@paper.pdf;type=application/pdf"
```

Poll the document status:

```bash
curl http://localhost:8000/documents/00000000-0000-0000-0000-000000000000 \
  -H "X-API-Key: rge_example"
```

### `POST /evaluate`

Runs the RAG pipeline without cache and scores each answer against an expected answer with `cerebras/gemma-4-31b`.

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: rge_example" \
  -d '{
    "questions": [
      {
        "question": "What does Qdrant store?",
        "expected": "Qdrant stores vectors for similarity search."
      }
    ]
  }'
```

The evaluator runs with concurrency `30` and returns accuracy, average latency, wall-clock time, cost breakdown, and per-question results.

## How It Works

1. **Ingest content**
   `/embed` accepts text directly. `/ingest` accepts PDFs, extracts page text with PyMuPDF, and processes the document in a background task.

2. **Chunk text**
   Paragraph chunking is the default. Sentence chunking uses NLTK. Fixed chunking uses `cl100k_base` tokens with 500-token windows and 50-token overlap.

3. **Create retrieval signals**
   Dense embeddings come from `gemini/gemini-embedding-001` through LiteLLM. Sparse vectors come from FastEmbed's `Qdrant/bm25` model.

4. **Index in Qdrant**
   Chunks are stored in the `embeddings` collection with named `dense` and `sparse` vectors. Startup ensures payload indexes exist for `category` and `source`.

5. **Retrieve and rerank**
   Query-time retrieval searches dense and sparse vectors, fuses them with RRF, pulls 20 candidates for RAG, then reranks them down to 5 contexts.

6. **Generate grounded answers**
   The generator asks the model to answer only from provided context and to say when the answer is not in context.

7. **Cache repeated questions**
   `/rag` caches exact-question results in Redis for 24 hours. `/evaluate` disables caching.

## Design Decisions

- **Hybrid search over dense-only retrieval**
  Dense vectors handle semantic similarity. BM25 sparse vectors preserve exact terms, names, and technical phrases that embedding models can blur.

- **RRF inside Qdrant**
  Qdrant performs dense and sparse prefetches and fuses rankings server-side, keeping retrieval logic compact in the application.

- **Reranking after broad retrieval**
  Fetching 20 candidates gives the cross-encoder enough material to reorder before the final 5 chunks are sent to the LLM.

- **Lazy reranker loading**
  The cross-encoder loads on first use by default to keep startup lighter. Set `PRELOAD_RERANKER=true` when predictable first-request latency matters more than startup time.

- **Hash-only API key storage**
  API keys are generated once and stored only as SHA-256 hashes. The first 12 characters are kept as a prefix for identification.

- **Operational data in PostgreSQL**
  Vector payloads live in Qdrant, while PostgreSQL stores records the service needs to audit, poll, or manage.

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

The current test suite covers chunking behavior, embedding-handler persistence boundaries, reranker ordering and lazy loading, and Redis RAG-cache behavior.

No formatter, linter, pre-commit configuration, or CI workflow is currently present in the repository.

## Evaluation And Load Testing

Compare raw hybrid search order against reranked RAG sources on a running stack:

```bash
uv run python evals/eval_reranking.py --base-url http://localhost:8000
```

The script intentionally does not include fabricated benchmark results. Run it against your own indexed corpus and inspect the printed before/after ranking tables.

Run a Locust load test against `/rag`:

```bash
uv run locust -f evals/locustfile.py
```

Set `LOCUST_API_KEY` before starting Locust so requests include `X-API-Key`.

## Roadmap

These are natural next steps based on the current implementation:

- Expose conversation-memory APIs or remove the unused conversation-memory tables.
- Add first-class configuration for retrieval candidate count, final rerank count, model names, and cache TTL.
- Add API-key revocation and listing endpoints.
- Add structured cost tracking for generation responses instead of returning a placeholder RAG cost.
- Add CI, linting, formatting, and a repository license.
- Expand live-stack integration tests for Qdrant, PostgreSQL, Redis, and model-provider failure modes.

## Contributing

1. Create a focused branch.
2. Install dependencies with `uv sync --extra ml`.
3. Run `uv run pytest` before opening a pull request.
4. Keep README/API changes aligned with the Pydantic models and route definitions in `src/routers/`.
5. Avoid documenting benchmark numbers unless they come from a reproducible run against a known corpus.

## License

No license file is currently present in this repository. Add a license before distributing or accepting external contributions.
