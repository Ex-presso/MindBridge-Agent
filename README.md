# MindBridge

A mental health support chatbot built with **FastAPI**, **LangGraph**, and **RAG**. The agent practices Rogerian (person-centered) therapy by reflecting feelings and creating a safe conversational space. It autonomously decides when to retrieve real counselor-style examples ([MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K)) from a pgvector store. The project ships with a five-layer evaluation framework covering retrieval quality, agent tool-routing, generation alignment, judged response quality, and safety.

## Features

- **LangGraph agent with conditional RAG**: the LLM decides per turn whether to call the retrieval tool (tool-routing F1 = 0.909 on a hand-labeled benchmark), with a bounded tool-call loop.
- **Multi-provider, bring-your-own-key**: OpenAI, Anthropic, Google Gemini, plus any OpenAI- or Anthropic-compatible endpoint (Ollama, LM Studio). Per-user keys are Fernet-encrypted at rest.
- **Server-side conversation memory**: a LangGraph PostgreSQL checkpointer keyed by conversation, so clients send only the new message.
- **Real token-level SSE streaming** with optimistic UI updates.
- **JWT auth**: short-lived access tokens, httpOnly refresh cookies, and bcrypt hashing.
- **Five-layer evaluation framework** with reproducible, seed-frozen benchmarks that run fully on local models.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Next.js Frontend                      │
│          (React 19 · shadcn/ui · Zustand · SWR)         │
└──────────────────────┬──────────────────────────────────┘
                       │ REST + SSE
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                       │
│   Auth (JWT) · Chat (SSE) · API-key vault (Fernet)      │
│                      │                                  │
│              ┌───────▼────────┐      ┌───────────────┐  │
│              │ LangGraph      │─RAG─▶│ pgvector      │  │
│              │ chat ⇄ tool    │ tool │ vector store  │  │
│              └───────┬────────┘      └───────────────┘  │
│              ┌───────▼────────┐                         │
│              │ LLM provider   │  OpenAI / Claude /      │
│              │ factory        │  Gemini / compatible    │
│              └────────────────┘                         │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │ PostgreSQL 16   │  users · conversations ·
              │ (pgvector)      │  messages · api keys ·
              └─────────────────┘  checkpoints · embeddings
```

## Quick Start

Requires [Docker Compose](https://docs.docker.com/get-docker/) and an LLM API key (OpenAI, Anthropic, or Gemini).

```bash
cp backend/.env.example backend/.env   # set SECRET_KEY for production
docker compose up -d
```

Open [http://localhost:3000](http://localhost:3000), create an account, add your API key under **Settings**, and start chatting. The backend runs at `:8080`, with interactive API docs at [http://localhost:8080/docs](http://localhost:8080/docs).

With the default `AUTO_CREATE_TABLES=true`, local and Docker startup runs the
Alembic migration chain automatically. It also recognizes and safely adopts the
older complete schema created by SQLAlchemy `create_all`; partial or unfamiliar
unversioned schemas stop with an actionable error instead of being stamped.
Production deployments should set `AUTO_CREATE_TABLES=false` and run
`cd backend && uv run alembic upgrade head` as an explicit release step before
starting the API.

<details>
<summary>Local development without Docker</summary>

```bash
# Backend (needs PostgreSQL with pgvector running)
cd backend
uv sync
uv run python -m app.core.rag.vector_store   # build vector index (first run)
uv run python -m app.main

# Frontend
cd frontend-next
npm install && npm run dev
```

</details>

## Evaluation

Each layer isolates one variable in the stack. Benchmarks are seed-frozen JSON in git; all runs use local models via LM Studio (no external API keys). Full methodology, statistics, and limitations: **[docs/EVALUATION.md](docs/EVALUATION.md)**.

| Layer | What it measures | Headline result |
|-------|------------------|-----------------|
| Retrieval IR | Recall/NDCG/MRR over 30 queries with cluster-gold chunk labels, 9-config sweep | NDCG@5 = 0.537 (chunk_size = 2000) |
| Agent routing | Does the agent invoke the RAG tool exactly when it should | F1 = 0.909 (n = 45 scored) |
| Reference-based | BERTScore (baseline-rescaled) vs MentalChat16K hold-out | F1 = 0.139, cosine = 0.637 (n = 99) |
| LLM-as-judge | Empathy/safety across 6 prompt × RAG conditions, Wilcoxon + Holm | CBT < Baseline (p = 0.004); Rogerian ≈ Baseline |
| Safety probes | Crisis-referral / refusal on 20 high-risk probes | self-harm crisis-referral **0% → 100%**: probe suite surfaced the gap, an in-graph crisis node closed it (re-run confirmed) |

Two design choices are worth noting. First, the **judge is a different model family than the generator**, because same-model self-judging previously produced ceiling scores with zero significant differences. Second, the safety layer is **diagnostic by design**: it surfaced that the empathy-optimized Rogerian prompt reflected feelings without crisis routing, which an in-graph crisis-routing node then closed (self-harm crisis-referral from 0% to 100% on re-run).

```bash
cd evaluation && uv sync
uv run python eval_routing.py --max-queries 3   # smoke test one layer
python ../analysis/analyze.py                   # regenerate all figures
```

## Project Structure

```
backend/        FastAPI app: LangGraph agent, RAG, auth, async SQLAlchemy
frontend-next/  Next.js 16 App Router chat UI
evaluation/     Five eval layers: scripts, YAML configs, frozen benchmarks, results
analysis/       Figure/table generation from eval results
docs/           EVALUATION.md · system_arch.md · develop.md
legacy/         Retired Gradio frontend
```

## Documentation

- [docs/EVALUATION.md](docs/EVALUATION.md): evaluation methodology, full results, limitations
- [docs/MEMORY.md](docs/MEMORY.md): working-memory baseline and long-term memory architecture

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 16, React 19, shadcn/ui, Zustand, SWR, Tailwind CSS 4 |
| Backend | FastAPI, LangGraph, LangChain, SQLAlchemy (async), Pydantic v2 |
| Data | PostgreSQL 16 + pgvector (documents, checkpoints, embeddings) |
| RAG | Qwen3-Embedding-0.6B (sentence-transformers), LangChain splitters |
| Eval | bert-score, scipy (Wilcoxon + Holm), pandas, matplotlib/seaborn |
| Ops | Docker Compose, uv, npm |

## License

Educational and research use.
