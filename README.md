# MindBridge Agent

MindBridge Agent is a mental-health-focused conversational assistant that blends large language models with retrieval-augmented generation. The backend (FastAPI + LangGraph) exposes an OpenAI-compatible API so tools like OpenWebUI can reuse their existing chat UI while taking advantage of MindBridge’s counseling-centric behaviors.

## Highlights
- **Context-aware support** – consumes the full conversation history supplied by the client, keeping prior exchanges in memory.
- **On-demand RAG** – when the latest user turn warrants it, the agent calls a RAG tool that surfaces counselor-style examples from a curated vector index.
- **Safety-first system prompt** – responses stay empathetic, avoid diagnoses, and gently redirect when conversations drift away from emotional well-being topics.
- **OpenWebUI friendly** – the chat endpoint mirrors OpenAI’s `/chat/completions` contract (including streaming), making MindBridge a drop-in replacement for local UIs.

For backend setup, API details, and deployment tips, see `backend/README.md`.
