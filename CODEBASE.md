# RAGEval Codebase Snapshot

Generated as a single-file reading copy of the current codebase.

Excluded from this snapshot:

- `.git/`
- `.venv/`
- `.pytest_cache/`
- `__pycache__/`
- `uv.lock`
- empty log/cache files

## Project Tree

```text
RAGEval/
+-- .gitignore
+-- CODEBASE.md
+-- Dockerfile
+-- README.md
+-- docker-compose.yml
+-- pyproject.toml
+-- uv.lock
+-- src/
|   +-- __init__.py
|   +-- database.py
|   +-- main.py
|   +-- models.py
+-- tests/
    +-- test_api.py
    +-- test_completion.py
    +-- test_streaming.py
```

## Overview

This project is a FastAPI app that streams LLM completions and stores completion logs in Postgres.

- `GET /` returns a basic JSON root response.
- `POST /complete` accepts a prompt, model, and max token limit.
- Completion requests call `litellm.acompletion(..., stream=True)`.
- Completion responses stream text chunks back as `text/plain`.
- Environment variables are loaded through `python-dotenv`.
- `DATABASE_URL` is read from the environment in `src.database`.
- `src.database` creates an async SQLAlchemy engine and async session factory.
- On FastAPI startup, the lifespan handler creates database tables from SQLAlchemy metadata.
- Streamed completion text is accumulated and saved to the `completions` table after streaming finishes.
- Each completion log stores prompt, response, model, latency in milliseconds, and creation time.
- API and connection errors from LiteLLM are converted into FastAPI `HTTPException` responses.
- Tests use `httpx.ASGITransport` to call the app in-process.
- LLM calls are mocked in completion and streaming tests.
- Docker support is present through `Dockerfile` and `docker-compose.yml`.
- Compose defines a Postgres service and an app service with `DATABASE_URL`.
- `uv.lock` is present but omitted from this snapshot because it is generated and large.

## Runtime Notes

- The default model for completion requests is `groq/llama-3.3-70b-versatile`.
- The application expects a valid `DATABASE_URL` when `src.database` is imported.
- The Docker image copies `uv` from `ghcr.io/astral-sh/uv:latest`.
- The app container starts Uvicorn with `uv run uvicorn src.main:app --host 0.0.0.0 --port 8000`.
- `docker-compose.yml` maps Postgres on host port `5432` and the API on host port `8000`.
- `docker-compose.yml` bind-mounts local `./src` into `/src`.
- Tests import the app directly. Because `src.database` creates an engine from `DATABASE_URL` at import time, tests require a valid database URL or database-related mocking/configuration.

## `.gitignore`

```gitignore
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv/
.env

uvicorn.*
```

## `pyproject.toml`

```toml
[project]
name = "rageval"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "asyncpg>=0.31.0",
    "fastapi>=0.138.0",
    "google-genai>=2.9.0",
    "grok>=6.2",
    "groq>=1.5.0",
    "litellm>=1.89.3",
    "pydantic>=2.13.4",
    "python-dotenv>=1.2.2",
    "sqlalchemy>=2.0.51",
    "uvicorn>=0.49.0",
]

[dependency-groups]
dev = [
    "httpx>=0.28.1",
    "pytest>=9.1.1",
    "pytest-asyncio>=1.4.0",
]

[tool.pytest.ini_options]
pythonpath = ["."]
```

## `Dockerfile`

```dockerfile
# 1. Start with the official, stable Python 3.14 slim image
FROM python:3.14-slim

# 2. Core Trick: Copy the pre-compiled uv tool straight into this container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set the working directory
WORKDIR /app

# 4. Copy ONLY the dependency blueprints first
COPY pyproject.toml uv.lock ./

# 5. Download and install all external libraries into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# 6. Copy your actual application source code
COPY src/ src/

# 7. Start your production Uvicorn web server
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## `docker-compose.yml`

```yaml
version: '3.9'

services:
  db:
    image: postgres:16-alpine
    container_name: postgres_db
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: rageval_logs
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://user:password@db:5432/rageval_logs
      
    depends_on:
      - db
    volumes:
      - ./src:/src

volumes:
  postgres_data:
```

## `README.md`

```md

```

## `src/__init__.py`

```python

```

## `src/database.py`

```python
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

DATABASE_URL = os.getenv(
    "DATABASE_URL",
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)
```

## `src/models.py`

```python
from sqlalchemy import String, Integer, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
import datetime

Base = declarative_base()

class Completion(Base):
    __tablename__ = "completions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    prompt: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100))
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    
    
```

## `src/main.py`

```python
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm import acompletion
from litellm.exceptions import APIError, APIConnectionError
from dotenv import load_dotenv
import logging

from src.database import engine, AsyncSessionLocal
from src.models import Base, Completion

load_dotenv()

logger = logging.getLogger("uvicorn.error")

# Lifespan context to ensure our table exists on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        # In production, swap this for Alembic migrations
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(lifespan=lifespan)

class CompletionRequest(BaseModel):
    prompt: str
    model: str = "groq/llama-3.3-70b-versatile"
    max_tokens: int = 500

@app.get("/")
async def root():
    return {"Message": "Hello from the root"}

@app.post("/complete")
async def request_llm(request: CompletionRequest):
    start_time = time.time()
    
    try:
        response = await acompletion(
            model=request.model,
            messages=[
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            max_tokens=request.max_tokens,
            stream=True
        )

        async def stream_generator():
            full_response_text = ""
            try:
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response_text += content
                        yield content
            except Exception as stream_err:
                logger.error(f"Stream interrupted: {str(stream_err)}", exc_info=True)
                yield f"\n[Error: Stream Interrupted]"
            finally:
                # Observability: Calculate latency and save to database
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Open a new session locally so it survives the streaming process
                async with AsyncSessionLocal() as session:
                    completion_log = Completion(
                        prompt=request.prompt,
                        response=full_response_text,
                        model=request.model,
                        latency_ms=latency_ms
                    )
                    session.add(completion_log)
                    await session.commit()
        
        return StreamingResponse(stream_generator(), media_type="text/plain")

    except APIError as api_err:
        logger.error(f"Groq API Error: {api_err.message} (Status Code: {api_err.status_code})")
        raise HTTPException(
            status_code=api_err.status_code,
            detail=f"LLM API Error: {api_err.message}" 
        )
    
    except APIConnectionError as conn_err:
        logger.error(f"LLM Connection Error: {str(conn_err)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not reach LLM."
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal server error occurred."
        )
```

## `tests/test_api.py`

```python
import pytest
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.mark.asyncio
async def test_root():

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:

        response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "Message": "Hello from the root"
    }
```

## `tests/test_completion.py`

```python
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest.mark.asyncio
async def test_completion():

    async def fake_response():
        yield type(
            "Chunk",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "delta": type(
                                "Delta",
                                (),
                                {
                                    "content": "Hello from fake LLM"
                                }
                            )
                        }
                    )
                ]
            }
        )


    with patch(
        "src.main.acompletion",
        return_value=fake_response()
    ):

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test"
        ) as client:

            response = await client.post(
                "/complete",
                json={
                    "prompt": "hello"
                }
            )


    assert response.status_code == 200

    assert response.text == "Hello from fake LLM"
```

## `tests/test_streaming.py`

```python
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch

from src.main import app


@pytest.mark.asyncio
async def test_streaming():

    async def fake_response():
        for content in ("Explain", " AI"):
            yield type(
                "Chunk",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "delta": type(
                                    "Delta",
                                    (),
                                    {
                                        "content": content
                                    }
                                )
                            }
                        )
                    ]
                }
            )

    with patch(
        "src.main.acompletion",
        return_value=fake_response()
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test"
        ) as client:

            response = await client.post(
                "/complete",
                json={
                    "prompt": "Explain AI"
                }
            )


    assert response.status_code == 200

    text = response.text

    assert text == "Explain AI"
```
