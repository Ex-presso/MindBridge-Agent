# MindBridge

[![CI](https://github.com/Ex-presso/MindBridge-Agent/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/Ex-presso/MindBridge-Agent/actions/workflows/ci.yml)

A stateful mental-health support agent built with FastAPI, LangGraph, and
PostgreSQL. MindBridge combines conditional RAG, deterministic safety routing,
and consent-gated cross-session memory in a full-stack chat application.

## Highlights

- **Stateful agent runtime:** bounded RAG tool routing, PostgreSQL checkpoints,
  token-aware context compaction, and token-level SSE streaming.
- **Long-term memory:** vector-selected episodes and explicit semantic facts,
  with grounded extraction through a leased transactional outbox.
- **Backend safety:** consistent row-lock ordering, idempotent workers,
  deletion barriers, encrypted BYOK credentials, and retry-safe cleanup.
- **Evaluation:** frozen benchmarks for retrieval, routing, response quality,
  safety, and live cross-session memory behavior.

## Architecture

```mermaid
flowchart LR
    UI["Next.js client"] -->|"REST + SSE"| API["FastAPI"]
    API --> GRAPH["LangGraph agent"]
    GRAPH -->|"conditional tool call"| RAG["pgvector RAG"]
    GRAPH --> CHECKPOINTS["PostgreSQL checkpoints"]
    API -->|"transactional outbox"| WORKER["Memory worker"]
    WORKER --> STORE["LangGraph Store"]
    STORE -->|"semantic + episodic recall"| GRAPH
```

### Agent graph

<img alt="Safety check, summarization, and memory selection run deterministically on every eligible turn; only the chat-to-tool loop is decided by the model." src="docs/img/agent-graph.svg">

Safety routing, context compaction, and memory selection are harness decisions,
so recall cannot be skipped by a model that declines to call a tool. The single
graph-routing choice is whether the turn needs retrieval, measured at F1 =
0.893 on the frozen benchmark.

[Open the self-contained agent diagram](docs/img/skill/agent-graph.html) to use
its PNG/PDF export controls after cloning the repository.

## Memory

MindBridge implements the automatic C-prime memory slice while keeping memory
policy outside model control:

- **Selection** reads exact semantic facts and vector-ranked episodes before
  each eligible turn; recalled content is treated as untrusted user data.
- **Extraction** creates evidence-grounded episodes and promotes only explicit
  user statements into semantic memory.
- **Durability** moves LLM work off the request path through a PostgreSQL
  transactional outbox, leased worker, and idempotent Store writes.
- **Privacy** gates every read and write on consent, data epoch, revision, crisis
  state, and deletion barriers. Memory is disabled by default.

### Write path

<img alt="The eligible assistant reply and outbox job commit atomically; a leased worker re-locks User, Conversation, and Job while it validates, extracts, and writes semantic and episodic memory." src="docs/img/memory-write.svg">

The eligible assistant reply and outbox job commit atomically. A worker claims
the job with `FOR UPDATE SKIP LOCKED`, then re-locks
`User -> Conversation -> Job` while it revalidates privacy gates, calls the
extractor, and writes semantic and episodic memory. The lease is also the
ownership token, so a crashed or stale worker cannot overwrite a newer claim.

[Open the self-contained memory diagram](docs/img/skill/memory-write.html) to
export it after cloning. Full invariants and scope are in
[docs/MEMORY.md](docs/MEMORY.md).

## Quick Start

Requirements: Docker Compose and an LLM API or local OpenAI-compatible server.

```bash
cp backend/.env.example backend/.env
docker compose up -d
```

Open [localhost:3000](http://localhost:3000), create an account, and add a model
under **Settings**. API documentation is available at
[localhost:8080/docs](http://localhost:8080/docs).

For LM Studio, choose `openai_compatible` and use
`http://host.docker.internal:1234/v1`. Enable long-term memory with
`MEMORY_ENABLED=true`; user consent remains off by default.

Detailed setup, migrations, and memory safety: [docs/MEMORY.md](docs/MEMORY.md).

## Evaluation

All reported runs use frozen datasets committed to the repository. Full
methodology and limitations are documented in
[docs/EVALUATION.md](docs/EVALUATION.md).

| Capability | Frozen scope | Result |
|---|---:|---|
| Retrieval ranking | 30 queries x 9 configurations | NDCG@5 = **0.537** |
| RAG tool routing | 45 scored hand-labeled cases | Precision 0.862, recall 0.926, F1 = **0.893** |
| Crisis safety | 20 probes, including 4 self-harm cases | Referral **0/4 -> 4/4** after the deterministic safety node |
| Cross-session memory | 17 live acceptance cases | Recall **4/6 on vs 0/6 off**; grounded writes 5/6; privacy/selection gates 5/5 |

```bash
cd evaluation && uv sync --locked
uv run python eval_routing.py --max-queries 3
uv run python eval_memory.py --max-recall 1 --skip-gates
```

## Repository

```text
backend/        FastAPI, LangGraph, PostgreSQL, memory worker
frontend-next/  Next.js chat and privacy controls
evaluation/     Frozen benchmarks and reproducible runners
analysis/       Evaluation figures and tables
docs/           Architecture, evaluation, and memory design
```

## Documentation

- [Evaluation](docs/EVALUATION.md)
- [Memory architecture](docs/MEMORY.md)

## License

Educational and research use.
