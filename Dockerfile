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

# 6. Bake nltk punkt_tab tokenizer data into the image so the sentence
#    chunking strategy works offline with no runtime download penalty.
RUN uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# 7. Copy your actual application source code
COPY src/ src/

# 8. Start your production Uvicorn web server
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]