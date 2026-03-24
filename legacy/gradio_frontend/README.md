# MindBridge Agent - Frontend Interface

A Gradio-based chat interface for the MindBridge mental health counseling agent.

## Features

- 🎯 **Streaming responses**: Real-time message streaming from the AI agent
- 🔄 **Model selection**: Choose between different AI models (Gemini, GPT-5, etc.)
- 🎨 **Warm UI theme**: Soft, professional design suitable for mental health applications
- 💬 **Chat history**: Automatic conversation history management via Gradio

## Prerequisites

1. **Backend API must be running first**
2. **Python 3.12.10** required
3. **uv package manager** - [Install uv](https://github.com/astral-sh/uv)

## Installation

```bash
# Navigate to frontend directory
cd frontend

# Create virtual environment (first time only)
uv venv --python 3.12.10

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (first time only)
uv sync
```

## Configuration

The `.env` file contains default configuration (optional):
- `API_BASE_URL`: Backend API URL (default: `http://127.0.0.1:8080`)
- `API_ENDPOINT`: Chat endpoint (default: `/api/v1/chat/completions`)
- `GRADIO_SERVER_PORT`: Frontend port (default: `7860`)

## Running the Application

```bash
# Make sure you're in the frontend directory and virtual environment is activated
python3 app.py
```

The interface will be available at:
- **Local**: http://127.0.0.1:7860
- **Network**: Check console output for network URL

## Usage

1. Select your preferred AI provider using the radio buttons (Google GenAI or OpenAI)
2. Choose a model from the available options
3. Type your message in the chat input
4. Press **Enter**
5. Watch the AI response stream in real-time
6. Click **New chat** to start a fresh conversation
7. Access chat history from the Gradio interface

## Model Options

### Google GenAI (default)
- `gemini-2.5-flash` - Fast, recommended for daily use

### OpenAI
- `gpt-5` - High-quality responses

## Troubleshooting

### Common Issues

**Connection Error**
- Ensure the backend API is running at `http://127.0.0.1:8080`
- Check that CORS settings in backend include port `7860`

**Streaming Issues**
- Verify the backend supports SSE (Server-Sent Events)
- Check network connection stability

**Model Not Found**
- Verify your API keys are configured in `backend/.env`
- Ensure the selected provider (Google GenAI or OpenAI) has a valid API key

**Import Errors**
- Run `uv sync` to ensure all dependencies are installed
- Verify you're using Python 3.12.10


## More Information

- Quick Start Guide: `../QUICKSTART.md`
- Backend Documentation: `../backend/README.md`
- Agent Architecture: `../AGENTS.md`
