# MindBridge

[![CI](https://github.com/Ex-presso/MindBridge-Agent/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/Ex-presso/MindBridge-Agent/actions/workflows/ci.yml)

A mental-health support chat application with a Next.js frontend and a
FastAPI/LangGraph backend, backed by PostgreSQL. Users can resume conversations,
enable memory across sessions, and inspect or clear what the application remembers.

## Screenshots

Screenshots show fictional, seeded demo data; chat replies come from a local model stub.

![Chat with a streamed assistant reply](docs/img/screenshot-chat.png)

| Memory controls | Model settings |
|---|---|
| ![Saved memories and the consent switch](docs/img/screenshot-memory.png) | ![Configuration for compatible model providers](docs/img/screenshot-settings.png) |

## Features

- **Resumable conversations:** PostgreSQL checkpoints, context compaction for
  longer chats, and streamed replies.
- **Conditional retrieval:** the agent can retrieve counselor-style examples
  for support requests instead of adding retrieval to every turn.
- **User-controlled memory:** mem0 extracts facts from what the user says and
  recalls them in later conversations. Memory is off by default; saved items
  stay inspectable and deletable, one by one or all at once.
- **Background extraction:** a transactional outbox, RabbitMQ with delayed
  retries and a dead-letter queue, and consent checks keep extraction calls
  off the conversation request path.
- **Compatible providers:** users bring their own API keys for OpenAI- or
  Anthropic-compatible endpoints. Keys are encrypted at rest.
- **Observability:** request ids in logs, Prometheus metrics, and optional
  OpenTelemetry traces, none of which carry message text.

## Architecture

```mermaid
flowchart LR
    UI["Next.js client"] -->|"REST + SSE"| API["FastAPI"]
    API --> GRAPH["LangGraph agent"]
    GRAPH -->|"conditional tool call"| RAG["pgvector RAG"]
    GRAPH --> CHECKPOINTS["PostgreSQL checkpoints"]
    API -->|"outbox row"| RELAY["Outbox relay"]
    RELAY --> MQ["RabbitMQ"]
    MQ --> WORKER["Memory worker"]
    WORKER --> MEM["mem0 on pgvector"]
    MEM -->|"recall"| GRAPH
    WORKER -.->|"idempotency keys"| REDIS["Redis"]
    API -.->|"rate limits"| REDIS
```

The model chooses whether to retrieve examples. Application code controls
crisis routing, memory eligibility, and persistence. See the
[architecture and design trade-offs](docs/ARCHITECTURE.md) for the detailed paths.

## Quick Start

Requirements: Docker Compose and an LLM API or local compatible server.

```bash
cp backend/.env.example backend/.env
docker compose up -d
```

Open [localhost:3000](http://localhost:3000), create an account, and configure
a provider under **Settings → Model**. The default is DeepSeek through
`openai_compatible`; for LM Studio, use
`http://host.docker.internal:1234/v1` as the base URL.

Build the retrieval index once before using RAG:

```bash
docker compose exec backend /app/.venv/bin/python -m app.core.rag.knowledge_base
```

This downloads MentalChat16K and Qwen3-Embedding-0.6B (about 1.2 GB for the
model on first use), embeds each client question, and stores it in pgvector
together with the counselor's full answer.

For cross-session memory, set `MEMORY_ENABLED=true` in `backend/.env`,
recreate the backend with `docker compose up -d backend`, and enable memory
in the user's settings. API docs: [localhost:8080/docs](http://localhost:8080/docs).

## Evaluation

Archived results from 2026-08-15, using `deepseek-v4-flash`:

| Check | Scope | Result |
|---|---|---|
| Retrieval tool selection | 45 scored cases; 5 ambiguous cases recorded separately | 45/45 matched expected decisions |
| Safety response markers | 20 probes with required and forbidden phrases | 20/20 passed |

These are small, application-specific regression checks. They do not establish
general answer quality or clinical effectiveness, and they predate the current
retrieval index and the mem0 memory, which have no reported results yet.
Datasets, scoring methods,
reproduction commands, and limitations are in [Evaluation](docs/EVALUATION.md).

CI runs backend tests, frontend lint/build, and a Playwright smoke test with
a local model stub. Live model evaluations run separately.

## Documentation

- [Architecture, memory, and limitations](docs/ARCHITECTURE.md)
- [Evaluation methods and results](docs/EVALUATION.md)

## License

[MIT](LICENSE)
