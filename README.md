# RAGEval

RAGEval is a small FastAPI service for experimenting with retrieval-augmented generation infrastructure. It can stream LLM completions through LiteLLM, store completion telemetry in PostgreSQL, generate embeddings for text, write vectors to Qdrant, and search those vectors later.

## What It Does

- Serves a FastAPI API from `src.main:app`
- Streams chat completions from LiteLLM-compatible providers
- Logs completion prompt, response, model, latency, and timestamp to PostgreSQL
- Generates Gemini embeddings through LiteLLM
- Stores embedded text in a Qdrant collection named `embeddings`
- Searches Qdrant using a query embedding
- Includes async endpoint tests with `pytest`, `pytest-asyncio`, and `httpx`

## Tech Stack

- Python 3.14
- FastAPI
- Uvicorn
- LiteLLM
- SQLAlchemy async ORM
- PostgreSQL with `asyncpg`
- Qdrant
- uv for dependency management
- pytest for tests

## Project Structure

```text
.
├── src/
│   ├── main.py       # FastAPI app, lifespan setup, routes
│   ├── database.py   # SQLAlchemy async engine/session setup
│   ├── models.py     # PostgreSQL ORM models
│   └── qdrant.py     # Async Qdrant client
├── tests/
│   └── test_main.py  # Async API tests
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

## Requirements

- Python 3.14+
- uv
- PostgreSQL
- Qdrant
- API keys for the LiteLLM providers you use

For local development, the easiest path is to run PostgreSQL and Qdrant through Docker Compose.

## Environment Variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/rageval_logs
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Provider keys used by LiteLLM, depending on the endpoint/model.
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

For running the app directly on your host machine instead of inside Docker, use localhost service names:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/rageval_logs
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

`DATABASE_URL` is required. The app creates the `completions` table on startup, but it does not create the database itself.

## Setup

Install dependencies:

```bash
uv sync
```

Start PostgreSQL and Qdrant:

```bash
docker compose up db qdrant
```

Run the API locally:

```bash
uv run uvicorn src.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI docs are available at:

```text
http://localhost:8000/docs
```

## Docker Compose

To build and run the full stack:

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

Returns a `text/plain` streaming response. After streaming finishes, the service logs the prompt, full response, model, latency, and timestamp in PostgreSQL.

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

## Tests

Run the test suite:

```bash
uv run pytest
```

The tests use `httpx.AsyncClient` with `ASGITransport`, so they call the FastAPI app in memory. External LiteLLM and Qdrant calls are mocked in the endpoint tests.

## Current Notes

- The embedding collection is named `embeddings`.
- The Qdrant vector size is currently fixed at `1536`.
- Database table creation happens during FastAPI lifespan startup.
- For production, replace startup table creation with migrations, for example Alembic.
- The service currently stores uploaded text as Qdrant payload under the `text` key.
