# Legacy Gradio client

This directory contains the retired Gradio interface. The maintained client is
`frontend-next/`; Docker Compose and the main README use that application.

The legacy client remains for reference and is not part of the release path.
It calls the stateless OpenAI-compatible endpoint and does not support the
current authentication, conversation, BYOK, or memory flows.

## Run locally

Start the backend first, then:

```bash
cd legacy/gradio_frontend
cp .env.example .env
uv sync
uv run python app.py
```

The default address is [http://127.0.0.1:7860](http://127.0.0.1:7860).
