# MindBridge Agent - Quick Start Guide

Quick start guide for running MindBridge Agent (Frontend + Backend)

## Quick Start

### 1. Start Backend

```bash
cd backend

# Create virtual environment (first time only)
uv venv --python 3.12.10

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (first time only)
uv sync

# Start the backend server
python3 -m app.main
```

Backend will run at: `http://127.0.0.1:8080`

### 2. Start Frontend (New Terminal)

```bash
cd frontend

# Create virtual environment (first time only)
uv venv --python 3.12.10

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (first time only)
uv sync

# Start the Gradio interface
python3 app.py
```

Frontend will run at: `http://127.0.0.1:7860`

## System Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────┐
│  Gradio UI      │ HTTP    │  FastAPI         │         │  LangGraph  │
│  (Frontend)     │────────▶│  (Backend)       │────────▶│  Agent      │
│  :7860          │   SSE   │  :8080           │         │  + RAG      │
└─────────────────┘         └──────────────────┘         └─────────────┘
                                     │
                                     │
                            ┌────────▼──────────┐
                            │  Vector Store     │
                            │  (FAISS)          │
                            │  MentalChat16K    │
                            └───────────────────┘
```

## Features

### Frontend (Gradio)
- ✅ Real-time streaming responses
- ✅ Model/Provider selector
- ✅ Warm, professional UI theme
- ✅ Automatic session history management
- ✅ Built-in chat history with save/load

### Backend (FastAPI + LangGraph)
- ✅ Rogerian (person-centered) therapy style
- ✅ RAG tool for mental health knowledge retrieval
- ✅ Multi-provider support (Google GenAI, OpenAI)
- ✅ OpenAI-compatible API format
- ✅ Streaming and non-streaming responses

## Configuration

### Backend Configuration (`backend/.env`)
```env
# API Keys (required)
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here

# Model Settings
OPENAI_MODEL=gpt-5
GEMINI_MODEL=gemini-2.5-flash
MODEL_TEMPERATURE=0.3

# Server
HOST=127.0.0.1
PORT=8080
DEBUG=False
```

### Frontend Configuration (`frontend/.env`, optional)
```env
API_BASE_URL=http://127.0.0.1:8080
API_ENDPOINT=/api/v1/chat/completions
GRADIO_SERVER_PORT=7860
```

## Available Models

### Google GenAI (default)
- `gemini-2.5-flash` - Fastest, recommended for daily use

### OpenAI
- `gpt-5` - High-quality responses

## More Information

- Backend documentation: `backend/README.md`
- Frontend documentation: `frontend/README.md`
- Agent architecture: `AGENTS.md`

