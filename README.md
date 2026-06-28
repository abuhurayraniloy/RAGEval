# RAGEval

RAGEval is a FastAPI service for experimenting with retrieval-augmented generation (RAG) infrastructure. It streams LLM completions through LiteLLM, records completion telemetry in PostgreSQL, embeds text with Gemini, stores vectors in Qdrant, searches those vectors, and exposes a simple RAG endpoint.

## Features

- FastAPI application served from `src.main:app`
- Streaming chat completions through LiteLLM-compatible providers
- Completion telemetry stored in PostgreSQL
- Gemini embedding generation through LiteLLM
- Qdrant vector indexing and semantic search
- RAG answer generation from retrieved Qdrant context
- Async endpoint tests with `pytest`, `pytest-asyncio`, and `httpx`

## Tech Stack

- Python 3.14+
- FastAPI and Uvicorn
- LiteLLM
- SQLAlchemy async ORM and `asyncpg`
- PostgreSQL
- Qdrant
- uv
- pytest

## Project Structure

```text
.
|-- src/
|   |-- main.py       # FastAPI app, lifespan setup, routes
|   |-- database.py   # SQLAlchemy async engine/session setup
|   |-- models.py     # PostgreSQL ORM models
|   `-- qdrant.py     # Async Qdrant client
|-- tests/
|   |-- test_main.py          # Async API endpoint tests
|   `-- concurrency_test.py   # Manual concurrency smoke test
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Requirements

- Python 3.14+
- uv
- Docker and Docker Compose, if running PostgreSQL and Qdrant locally
- Provider API keys for the LiteLLM models you call

## Configuration

Create a `.env` file in the project root.

For Docker Compose:

```env
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/rageval_logs
QDRANT_URL=http://qdrant:6333

GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

For running the API directly on your host while PostgreSQL and Qdrant run through Docker Compose:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/rageval_logs
QDRANT_URL=http://localhost:6333

GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Optional Qdrant Cloud configuration:

```env
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your_qdrant_api_key
```

The app creates the `completions` table and the `embeddings` Qdrant collection during FastAPI startup. It does not create the PostgreSQL database itself.

## Local Development

Install dependencies:

```bash
uv sync
```

Start PostgreSQL and Qdrant:

```bash
docker compose up db qdrant
```

Run the API:

```bash
uv run uvicorn src.main:app --reload
```

The API is available at:

```text
http://localhost:8000
```

Interactive API docs are available at:

```text
http://localhost:8000/docs
```

## Docker Compose

Build and run the full stack:

```bash
docker compose up --build
```

Services:

- `app`: FastAPI API on port `8000`
- `db`: PostgreSQL on port `5432`
- `qdrant`: Qdrant on ports `6333` and `6334`

## API

### Health Check

```http
GET /
```

Response:

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

Response:

Returns a `text/plain` streaming response. After streaming finishes, the service stores the prompt, full response, model, latency, and timestamp in PostgreSQL.

### Embed Text

```http
POST /embed
Content-Type: application/json
```

Request:

```json
{
  "text": "Qdrant is a vector database for similarity search."
}
```

Response:

```json
{
  "status": "success",
  "point_id": "generated-uuid",
  "message": "Text vectorized via Gemini and indexed in Qdrant successfully."
}
```

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

Response:

```json
{
  "status": "success",
  "query": "vector database for semantic search",
  "results": [
    {
      "id": "point-id",
      "score": 0.9234,
      "text": "Matched text from Qdrant"
    }
  ]
}
```

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

Response:

```json
{
  "status": "success",
  "question": "What is Qdrant used for?",
  "answer": "Qdrant is used for vector similarity search.",
  "sources": [
    {
      "id": "point-id",
      "score": 0.98,
      "text": "Source text retrieved from Qdrant"
    }
  ]
}
```

The `/rag` endpoint embeds the question, retrieves the top matching vectors from Qdrant, and asks the configured Groq model to answer using only the retrieved context.

## Tests

Run the test suite:

```bash
uv run pytest
```

The endpoint tests use `httpx.AsyncClient` with `ASGITransport`, so they call the FastAPI app in memory. External LiteLLM and Qdrant calls are mocked in the main endpoint tests.

Run the manual concurrency smoke test against a running local API:

```bash
uv run python tests/concurrency_test.py
```

## Implementation Notes

- The Qdrant collection is named `embeddings`.
- Embedding vectors are created with `gemini/gemini-embedding-001` at 1536 dimensions.
- Uploaded text is stored in Qdrant payloads under the `text` key.
- Completion logs are stored in the PostgreSQL `completions` table.
- Startup table creation is useful for local development; production deployments should use migrations, for example Alembic.
