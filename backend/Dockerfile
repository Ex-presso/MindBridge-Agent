FROM python:3.12-slim

WORKDIR /app

# Match the exact uv release used by CI; `latest` makes old commits drift.
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies without resolving or rewriting the committed lockfile.
RUN uv sync --locked --no-dev

# Copy application code
COPY . .

EXPOSE 8080

# Single worker keeps the in-process rate limiter (and agent cache) accurate:
# with N workers each keeps its own counters, diluting limits N times.
# Move slowapi to shared storage (redis) before scaling workers.
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
