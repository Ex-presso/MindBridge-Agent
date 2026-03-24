"""Chat endpoints: legacy OpenAI-compatible + new session-aware."""
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.core.auth.deps import get_current_user
from app.db.engine import get_db
from app.db.models.user import User
from app.db.repositories import conversation_repo, message_repo
from app.schemas.conversation import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    SessionChatRequest,
)
from app.services import chat as chat_service
from app.services import conversation_service
from config.settings import settings

router = APIRouter()


# ── Legacy endpoint (kept for OpenWebUI / third-party clients) ────────────────

@router.post("/chat/completions")
async def chat_legacy(request: Request, body: ChatCompletionRequest):
    """Stateless OpenAI-compatible endpoint. No auth, no persistence."""
    provider = _resolve_provider(body)
    if not body.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")
    if not any(m.role == "user" for m in body.messages):
        raise HTTPException(status_code=400, detail="At least one user message required.")

    agent = request.app.state.stateless_agents.get(provider)
    if agent is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    try:
        reply = await chat_service.run_chat_legacy(agent, body.messages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if body.stream:
        async def event_stream() -> AsyncGenerator[bytes, None]:
            yield _build_sse_chunk(chunk_id, created, body.model, None, role_event=True)
            for i in range(0, len(reply), 80):
                yield _build_sse_chunk(chunk_id, created, body.model, reply[i:i + 80])
            yield _build_sse_chunk(chunk_id, created, body.model, None, finish=True)
            yield b"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return _build_full_response(chunk_id, created, body.model, reply)


# ── New session-aware endpoint ────────────────────────────────────────────────

@router.post("/chat")
async def session_chat(
    request: Request,
    body: SessionChatRequest,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Stateful chat endpoint with conversation persistence and real streaming."""
    if len(body.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(body.message) > settings.MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Message exceeds {settings.MAX_MESSAGE_LENGTH} characters.")

    provider = body.provider.lower()
    agent = request.app.state.stateful_agents.get(provider)
    if agent is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    # Create or validate conversation
    is_new_conversation = body.conversation_id is None
    if is_new_conversation:
        conv = await conversation_service.create_conversation(db, user_id=user.id, model=body.model, provider=provider)
    else:
        try:
            conv_id = uuid.UUID(body.conversation_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation_id.")
        conv = await conversation_repo.get_by_id(db, conv_id)
        if conv is None or conv.user_id != user.id:
            raise HTTPException(status_code=404, detail="Conversation not found.")

    # Save user message
    await message_repo.create(db, conversation_id=conv.id, role="user", content=body.message)
    await conversation_repo.touch(db, conv.id, model=body.model, provider=provider)
    await db.commit()

    thread_id = str(conv.id)
    conv_id_str = str(conv.id)

    if body.stream:
        async def stream_response() -> AsyncGenerator[bytes, None]:
            collected = []
            try:
                async for token in chat_service.stream_chat_session(agent, body.message, thread_id):
                    collected.append(token)
                    data = f"data: {token}\n\n"
                    yield data.encode()
            except Exception as exc:
                yield f"data: [ERROR] {exc}\n\n".encode()
            yield b"data: [DONE]\n\n"

            # Save assistant message after streaming completes
            full_reply = "".join(collected)
            if full_reply:
                async with request.app.state.db_session() as session:
                    await message_repo.create(
                        session,
                        conversation_id=conv.id,
                        role="assistant",
                        content=full_reply,
                        model_used=body.model,
                        provider=provider,
                    )
                    await conversation_repo.touch(session, conv.id)
                    if is_new_conversation:
                        title = await conversation_service.generate_title(agent, body.message)
                        await conversation_repo.update_title(session, conv.id, title)
                    await session.commit()

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"X-Conversation-Id": conv_id_str},
        )

    # Non-streaming
    try:
        reply = await chat_service.run_chat_session(agent, body.message, thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    async with request.app.state.db_session() as session:
        await message_repo.create(
            session,
            conversation_id=conv.id,
            role="assistant",
            content=reply,
            model_used=body.model,
            provider=provider,
        )
        await conversation_repo.touch(session, conv.id)
        if is_new_conversation:
            title = await conversation_service.generate_title(agent, body.message)
            await conversation_repo.update_title(session, conv.id, title)
        await session.commit()

    return {
        "conversation_id": conv_id_str,
        "message": {"role": "assistant", "content": reply},
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_provider(body: ChatCompletionRequest) -> str:
    if body.provider:
        return body.provider.lower()
    model = body.model.lower()
    if "gpt" in model or "openai" in model:
        return "openai"
    if "gemini" in model:
        return "google_genai"
    return "google_genai"


def _build_sse_chunk(chunk_id, created, model, content, *, role_event=False, finish=False) -> bytes:
    delta = ChatCompletionChunkDelta(role="assistant" if role_event else None, content=content)
    choice = ChatCompletionChunkChoice(index=0, delta=delta, finish_reason="stop" if finish else None)
    chunk = ChatCompletionChunk(id=chunk_id, object="chat.completion.chunk", created=created, model=model, choices=[choice])
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()


def _build_full_response(chunk_id, created, model, reply):
    return ChatCompletionResponse(
        id=chunk_id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[ChatCompletionChoice(message=ChatCompletionMessage(role="assistant", content=reply))],
    )
