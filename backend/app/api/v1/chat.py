from collections.abc import Iterable
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.conversation import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from app.services.chat import run_chat

router = APIRouter()

DEFAULT_PROVIDER = "google_genai"

def _resolve_provider(request: ChatCompletionRequest) -> str:
    if request.provider:
        return request.provider.lower()
    else:
        return DEFAULT_PROVIDER


def _extract_user_message(request: ChatCompletionRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content
    raise HTTPException(status_code=400, detail="At least one user message is required.")


def _iter_text_chunks(text: str, *, chunk_size: int = 80) -> Iterable[str]:
    for idx in range(0, len(text), chunk_size):
        yield text[idx : idx + chunk_size]


def _build_chunk_payload(chunk_id: str, created_at: int, model: str, content: str | None, *, role_event: bool = False, finish: bool = False) -> ChatCompletionChunk:
    delta = ChatCompletionChunkDelta(
        role="assistant" if role_event else None,
        content=content if content else None,
    )

    choice = ChatCompletionChunkChoice(
        index=0,
        delta=delta,
        finish_reason="stop" if finish else None,
    )

    return ChatCompletionChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created_at,
        model=model,
        choices=[choice],
    )


@router.post("/chat/completions")
def chat(request: ChatCompletionRequest):
    provider = _resolve_provider(request)
    user_message = _extract_user_message(request)

    try:
        reply = run_chat(provider, user_message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if request.stream:

        def event_stream() -> Iterable[bytes]:
            role_chunk = _build_chunk_payload(chunk_id, created, request.model, None, role_event=True)
            yield f"data: {role_chunk.model_dump_json(exclude_none=True)}\n\n".encode("utf-8")

            for piece in _iter_text_chunks(reply):
                data_chunk = _build_chunk_payload(chunk_id, created, request.model, piece)
                yield f"data: {data_chunk.model_dump_json(exclude_none=True)}\n\n".encode("utf-8")

            final_chunk = _build_chunk_payload(chunk_id, created, request.model, None, finish=True)
            yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    response = ChatCompletionResponse(
        id=chunk_id,
        object="chat.completion",
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(role="assistant", content=reply),
            )
        ],
    )
    return response
