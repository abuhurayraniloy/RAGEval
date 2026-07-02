# syntax=docker/dockerfile:1
# The syntax directive above pins the BuildKit frontend version, needed
# for the --mount=type=cache line below to work reliably across Docker
# versions.

# 1. Start with the official, stable Python 3.14 slim image
FROM python:3.14-slim

# 2. Core Trick: Copy the pre-compiled uv tool straight into this container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set the working directory
WORKDIR /app

# 4. Copy ONLY the dependency blueprints first
COPY pyproject.toml uv.lock ./

# 5. Download and install all external libraries into a virtual environment.
# CHANGED: added --extra ml. sentence-transformers/torch live under the
# optional "ml" dependency group now (see pyproject.toml); without this
# flag they are silently skipped and the app crashes on first /rag or
# /evaluate call when get_reranker() tries to import CrossEncoder.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra ml

# 6. Bake nltk punkt_tab tokenizer data into the image so the sentence
#    chunking strategy works offline with no runtime download penalty.
RUN uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# 7. Copy your actual application source code
COPY src/ src/

# 8. Start your production Uvicorn web server
CMD ["sh", "-c", "uv run uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]