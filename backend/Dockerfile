FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps, frozen)
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

EXPOSE 8080

# ponytail: single worker keeps the in-process rate limiter (and agent cache)
# accurate — with N workers each keeps its own counters, diluting limits N×.
# Move slowapi to shared storage (redis) before scaling workers.
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
