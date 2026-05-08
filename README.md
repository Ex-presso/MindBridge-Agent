# MindBridge

A mental health support chatbot built with **FastAPI**, **LangGraph**, and **RAG** (Retrieval-Augmented Generation). MindBridge practices Rogerian (person-centered) therapy — reflecting feelings, validating emotions, and creating a safe conversational space — augmented by real counselor-style examples retrieved on demand.

<!-- Screenshot placeholder: add a chat UI screenshot here -->
<!-- ![MindBridge Chat UI](docs/images/chat-ui.png) -->

## Features

- **Rogerian therapy prompting** — empathetic, non-directive responses grounded in person-centered counseling principles
- **On-demand RAG** — the agent autonomously decides when to retrieve counselor examples from a curated vector store ([MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K)), weaving them naturally into replies
- **Multi-provider LLM support** — OpenAI, Anthropic Claude, Google Gemini, plus any OpenAI-compatible or Claude-compatible endpoint (Ollama, LM Studio, Azure, etc.)
- **Per-user API keys** — users bring their own LLM keys; keys are Fernet-encrypted at rest and never leave the server
- **Conversation memory** — server-side conversation persistence via LangGraph's PostgreSQL checkpointer; clients send only the new message
- **Real-time streaming** — token-level SSE streaming for responsive chat
- **JWT authentication** — access tokens + httpOnly refresh cookies, bcrypt password hashing
- **Five-layer evaluation framework** — retrieval IR (Recall/NDCG), agent routing (P/R/F1), reference-based (BERTScore vs MentalChat16K hold-out), LLM-as-judge with split generator/judge, and a safety probe suite. Full methodology in [docs/EVALUATION.md](docs/EVALUATION.md).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Next.js Frontend                      │
│          (React 19 · Shadcn/ui · Zustand · SWR)         │
└──────────────────────┬──────────────────────────────────┘
                       │ REST + SSE
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                       │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐  │
│  │ Auth API │  │ Chat API  │  │ API Key Management   │  │
│  │ (JWT)    │  │ (SSE)     │  │ (Fernet encryption)  │  │
│  └──────────┘  └─────┬─────┘  └──────────────────────┘  │
│                      │                                  │
│              ┌───────▼────────┐                         │
│              │  LangGraph     │                         │
│              │  Agent         │──── RAG Tool ───┐       │
│              │  (stateful)    │                 │       │
│              └───────┬────────┘                 │       │
│                      │                  ┌──────▼──────┐ │
│              ┌───────▼────────┐         │ FAISS /     │ │
│              │ LLM Provider   │         │ pgvector    │ │
│              │ (multi-model)  │         │ Vector Store│ │
│              └────────────────┘         └─────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   PostgreSQL    │
              │   (pgvector)    │
              │  • Users        │
              │  • Conversations│
              │  • Messages     │
              │  • API Keys     │
              │  • Checkpoints  │
              └─────────────────┘
```

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- An LLM API key (OpenAI, Anthropic, or Google Gemini)

### Run with Docker Compose

```bash
# Clone the repo
git clone https://github.com/<your-username>/MindBridge.git
cd MindBridge

# Create backend env file
cp backend/.env.example backend/.env
# Edit backend/.env — set SECRET_KEY to a random value for production

# Start all services
docker compose up -d
```

Open [http://localhost:3000](http://localhost:3000) in your browser, create an account, add your API key in **Settings**, and start chatting.

| Service    | URL                          |
|------------|------------------------------|
| Frontend   | http://localhost:3000        |
| Backend    | http://localhost:8080        |
| PostgreSQL | localhost:5432               |

### Local Development (without Docker)

<details>
<summary>Click to expand</summary>

**Backend:**

```bash
cd backend
uv venv --python 3.12
source .venv/bin/activate
uv sync

# Set up PostgreSQL (with pgvector extension)
# Then configure backend/.env

# Build the vector index (first run only)
uv run python -m app.core.rag.vector_store

# Start the server
uv run python -m app.main
```

**Frontend:**

```bash
cd frontend-next
npm install
npm run dev
```

</details>

## Supported LLM Providers

Users configure their own API keys via the **Settings** page in the frontend. No system-level keys are required.

| Provider             | Models                              | Notes                        |
|----------------------|-------------------------------------|------------------------------|
| OpenAI               | GPT-4o, GPT-4o Mini, o3 Mini        | —                            |
| Anthropic            | Claude Sonnet 4.5, Claude Haiku 4.5 | —                            |
| Google Gemini        | Gemini 2.5 Flash, Gemini 2.5 Pro    | —                            |
| OpenAI Compatible    | Any                                 | Ollama, LM Studio, etc.      |
| Anthropic Compatible | Any                                 | Custom Claude-compatible APIs|

<!-- Screenshot placeholder: settings page -->
<!-- ![Settings Page](docs/images/settings.png) -->

## API Endpoints

### Authentication

| Method | Endpoint                 | Description            |
|--------|--------------------------|------------------------|
| POST   | `/api/v1/auth/register`  | Create account         |
| POST   | `/api/v1/auth/login`     | Login, returns JWT     |
| POST   | `/api/v1/auth/refresh`   | Refresh access token   |
| POST   | `/api/v1/auth/logout`    | Clear refresh cookie   |
| GET    | `/api/v1/auth/me`        | Current user info      |

### Chat

| Method | Endpoint                     | Description                               |
|--------|------------------------------|-------------------------------------------|
| POST   | `/api/v1/chat`               | Stateful session chat (auth required, SSE)|
| POST   | `/api/v1/chat/completions`   | Legacy OpenAI-compatible (stateless)      |

### Conversations

| Method | Endpoint                               | Description                |
|--------|----------------------------------------|----------------------------|
| GET    | `/api/v1/conversations`                | List user conversations    |
| GET    | `/api/v1/conversations/:id`            | Get conversation + messages|
| DELETE | `/api/v1/conversations/:id`            | Delete conversation        |
| PATCH  | `/api/v1/conversations/:id/title`      | Rename conversation        |

### API Keys

| Method | Endpoint                     | Description                    |
|--------|------------------------------|--------------------------------|
| GET    | `/api/v1/api-keys`           | List configured keys (masked)  |
| PUT    | `/api/v1/api-keys`           | Save or update a key           |
| DELETE | `/api/v1/api-keys/:provider` | Remove a key                   |
| GET    | `/api/v1/api-keys/models`    | Available models per provider  |

## Evaluation

MindBridge ships with a five-layer evaluation framework that runs entirely on local LLMs (no external API keys). Each layer isolates one variable in the stack — full methodology, results, and limitations are in **[docs/EVALUATION.md](docs/EVALUATION.md)**.

| Layer | What it measures | Headline result |
|-------|------------------|-----------------|
| Retrieval IR | Recall@k / NDCG@k / MRR over a 30-query benchmark with cluster gold | NDCG@5 = 0.537 (chunk_size=2000) |
| Agent routing | Whether the LangGraph agent invokes the RAG tool when it should | F1 = 0.909 on 50 hand-labeled queries |
| Reference-based | BERTScore (baseline-rescaled) + cosine sim vs MentalChat16K hold-out | F1 = 0.139, cosine = 0.637 (n=99) |
| LLM-as-judge | Empathy + safety on 6 prompt × RAG conditions, Holm-corrected | CBT < Baseline (p=0.004**); Rogerian ≈ Baseline (ns) |
| Safety probes | Crisis / diagnosis / medication / jailbreak / minor refusal rates | **0% crisis-referral on self-harm probes** — production gap |

The judge is split from the generator (`Qwen3.5-27B-Claude-distilled` vs `Nemotron-3-Nano-4B`) to avoid self-bias; under same-model self-judging the previous eval produced near-ceiling 4.5–5.0 scores with zero significant differences. The safety eval surfaces a real production gap (the Rogerian system prompt produces empathic reflection without crisis routing) that would require a safety-rail prompt or upstream classifier before deployment.

### Running the Evaluation

```bash
cd evaluation
uv sync

# Build benchmarks (deterministic, seed=42)
uv run python build_ir_benchmark.py
uv run python build_reference_benchmark.py

# Run any layer
uv run python eval_retrieval.py     # IR benchmark + 9-config sweep
uv run python eval_routing.py       # 50 routing queries
uv run python eval_reference.py     # 100-query BERTScore vs counselor
uv run python eval_prompting.py     # 6 condition × 10 query judge eval
uv run python eval_safety.py        # 20 safety probes

# Generate all figures and summary tables
python ../analysis/analyze.py
```

Requires PostgreSQL (pgvector) running and an LM Studio server with both
`nvidia/nemotron-3-nano-4b` and `qwen3.5-27b-claude-4.6-opus-distilled-mlx@4bit`
loaded. Does not require the web server.

See **[docs/EVALUATION.md](docs/EVALUATION.md)** for full methodology, all per-condition results, statistical tests, figures, and limitations.

## Project Structure

```
MindBridge/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # REST endpoints (auth, chat, conversations, api-keys)
│   │   ├── core/
│   │   │   ├── agent/       # LangGraph agent with conditional RAG tool
│   │   │   ├── auth/        # JWT, bcrypt, Fernet encryption
│   │   │   ├── llm/         # Multi-provider LLM factory
│   │   │   └── rag/         # Vector store builder
│   │   ├── db/              # SQLAlchemy models, async repositories
│   │   ├── schemas/         # Pydantic request/response models
│   │   └── services/        # Business logic (chat, auth, conversations)
│   └── config/              # Environment-driven settings
├── frontend-next/
│   └── src/
│       ├── app/             # Next.js App Router pages
│       ├── components/      # Chat UI, sidebar, shadcn/ui
│       ├── hooks/           # useChat, useAuth, useApiKeys, useConversations
│       ├── lib/             # API client, SSE stream reader
│       └── stores/          # Zustand auth store
├── evaluation/
│   ├── build_ir_benchmark.py        # curates 30-query IR test set (seed=42)
│   ├── build_reference_benchmark.py # curates 100 hold-out queries
│   ├── eval_retrieval.py            # IR sweep (chunk_size × overlap × top_k)
│   ├── eval_routing.py              # agent's tool-call decision: P/R/F1
│   ├── eval_reference.py            # BERTScore + cosine vs counselor refs
│   ├── eval_prompting.py            # 6-condition judge eval (split judge)
│   ├── eval_safety.py               # 20-probe safety suite
│   ├── configs/                     # YAML configs per eval layer
│   ├── datasets/                    # Frozen JSON benchmarks
│   ├── metrics/                     # ir_metrics.py + LLMJudge
│   └── results/                     # Output CSVs (per-query + summary)
├── analysis/
│   ├── analyze.py                   # Generates all figures + summary tables
│   └── pics/                        # 20+ output figures (PNG)
├── docs/
│   └── EVALUATION.md                # Full methodology, results, limitations
└── docker-compose.yml
```

## Tech Stack

| Layer      | Technology                                                    |
|------------|---------------------------------------------------------------|
| Frontend   | Next.js 15, React 19, Shadcn/ui, Zustand, SWR, Tailwind CSS   |
| Backend    | FastAPI, LangGraph, LangChain, SQLAlchemy (async), Pydantic   |
| Database   | PostgreSQL 16 + pgvector                                      |
| Auth       | JWT (python-jose), bcrypt, Fernet encryption                  |
| LLM        | OpenAI, Anthropic, Google Gemini (via LangChain); LM Studio for local eval |
| RAG        | pgvector, sentence-transformers (Qwen3-Embedding-0.6B), LangChain text splitters |
| Eval       | bert-score (roberta-large), scipy (Wilcoxon + Holm), pandas, matplotlib/seaborn |
| Deployment | Docker Compose                                                |
| Tooling    | uv (Python), npm (Node.js)                                    |

## License

This project is for educational and research purposes.
