
# Backend Overview

This backend powers a mental-health-oriented chat agent built with FastAPI, LangGraph, and a retrieval-augmented generation (RAG) tool. The agent remains stateless across requests and speaks the OpenAI-compatible chat completions protocol so it can plug directly into OpenWebUI.

## Tech Stack
- Python 3.12.10, managed with [uv](https://github.com/astral-sh/uv)
- FastAPI for the HTTP layer
- LangGraph + LangChain chat models for agent orchestration
- Custom RAG tool that retrieves counselor-style examples from a FAISS vector store

## Project Architecture
- `app/api`: OpenAI-compatible REST endpoints (streaming and non-streaming)
- `app/core/agent`: LangGraph agent that conditionally invokes the RAG tool
- `app/core/rag`: Retriever and vector store helpers
- `app/services`: Provider selection and conversation handling
- `config/settings.py`: Environment-driven configuration

The agent keeps conversation memory by consuming the full message history from the client, while the RAG tool only runs on the latest user turn. Because the API is stateless, front-ends (such as OpenWebUI) can manage sessions on their side.

## Getting Started

1. **Create the virtual environment**
   ```bash
   cd backend
   uv venv --python 3.12.10
   source .venv/bin/activate
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```

3. **Configure environment variables**
   - Copy `backend/.env.example` to `backend/.env`
   - Fill in provider API keys and optional settings

4. **Build the vector index (first run only)**
   ```bash
   uv run python -m app.core.rag.vector_store
   ```

5. **Start the API**
   ```bash
   # from the project root
   uv run python -m app.main
   ```
   The server listens on `http://127.0.0.1:8080` by default.

## Design Principles
- **Stateless API** – all session context is supplied by the caller.
- **OpenAI compatibility** – request/response shapes mirror OpenAI chat completions, enabling drop-in support for OpenWebUI.
- **Tool-mediated RAG** – the agent decides when to call a retrieval tool; the tool returns curated counselor examples which the LLM weaves into the reply.

### Tool Invocation Flow
1. Client sends the complete chat history.
2. The agent checks the latest user turn and decides whether the RAG tool is relevant.
3. If a tool call is emitted, the retriever searches the FAISS index (default top-3) and returns formatted snippets.
4. The LLM receives the tool output and crafts the final response, which is streamed back to the client when requested.

## Pairing with OpenWebUI (Docker)
Run OpenWebUI via Docker and, inside the admin panel (`Settings → Admin → External Links → OpenAI API`), point the base URL to your backend:
```
http://host.docker.internal:8080/api/v1
```
Use the actual port your FastAPI service exposes.

## API Surface
- `POST /api/v1/chat/completions` – OpenAI-compatible chat endpoint supporting streaming SSE chunks and non-stream responses.
- `GET /api/v1/models` – Lightweight model listing for clients that query available providers.

## Roadmap
- Add JWT-based stateless authentication.
- Introduce persistence and an `api/v2` namespace tailored for a custom front-end.
- Expand tooling beyond RAG (e.g., mood trackers, coping strategies) while keeping safety constraints.
