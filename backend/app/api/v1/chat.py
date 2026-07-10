"""Chat endpoints: legacy OpenAI-compatible + new session-aware."""
import asyncio
import json
import logging
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
logger = logging.getLogger(__name__)

# Cache compiled agents by (provider, model, base_url, key-hash, checkpointer).
# The LangGraph graph is stateless — per-conversation state lives in the
# checkpointer — so one agent is safe to reuse across requests and users that
# share an LLM config. Rebuilding it per request re-binds tools and recompiles
# the graph for nothing.
# ponytail: bounded LRU at 128 entries; bump if you serve many model configs.
import hashlib
from collections import OrderedDict

_AGENT_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_AGENT_CACHE_MAX = 128
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()
_TITLE_UPDATE_TIMEOUT_SECONDS = 15.0


def _get_agent(provider: str, model: str | None, base_url: str | None, api_key: str, checkpointer, store=None):
    from app.core.agent.agent import Agent
    from app.core.llm.provider import get_llm

    key = (provider, model, base_url, hashlib.sha256(api_key.encode()).hexdigest(), id(checkpointer), id(store))
    agent = _AGENT_CACHE.get(key)
    if agent is None:
        llm = get_llm(provider, api_key=api_key, base_url=base_url, model=model)
        agent = Agent(llm, checkpointer=checkpointer, store=store)
        _AGENT_CACHE[key] = agent
        if len(_AGENT_CACHE) > _AGENT_CACHE_MAX:
            _AGENT_CACHE.popitem(last=False)
    else:
        _AGENT_CACHE.move_to_end(key)
    return agent


async def _save_generated_title_if_present(
    request: Request,
    agent,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    first_user_message: str,
) -> None:
    """Best-effort title generation that never changes the chat result."""
    try:
        title = await conversation_service.generate_title(agent, first_user_message)
        async with request.app.state.db_session() as session:
            async with session.begin():
                conv = await conversation_repo.get_owned_for_update(
                    session,
                    conversation_id,
                    user_id,
                )
                if conv is not None:
                    await conversation_repo.update_title(
                        session,
                        conversation_id,
                        title,
                    )
    except Exception:
        logger.exception(
            "Conversation title update failed for conversation %s",
            conversation_id,
        )


def _schedule_title_update(
    request: Request,
    agent,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    first_user_message: str,
) -> None:
    """Run best-effort title work without delaying a successful chat response."""

    async def run_bounded() -> None:
        try:
            await asyncio.wait_for(
                _save_generated_title_if_present(
                    request,
                    agent,
                    conversation_id,
                    user_id,
                    first_user_message,
                ),
                timeout=_TITLE_UPDATE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "Conversation title update timed out for conversation %s",
                conversation_id,
            )

    task = asyncio.create_task(
        run_bounded(),
        name=f"conversation-title-{conversation_id}",
    )
    # asyncio keeps only weak task references; retain this one until completion.
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


# ── Legacy endpoint (kept for OpenWebUI / third-party clients) ────────────────

@router.post("/chat/completions")
async def chat_legacy(request: Request, body: ChatCompletionRequest):
    """Stateless OpenAI-compatible endpoint. No auth, no persistence."""
    if not settings.ENABLE_LEGACY_CHAT_ENDPOINT:
        raise HTTPException(status_code=404, detail="Not found.")
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
    t_start = time.time()
    if len(body.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(body.message) > settings.MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Message exceeds {settings.MAX_MESSAGE_LENGTH} characters.")

    provider = body.provider.lower()
    # Load user's API key for this provider
    from app.db.repositories import api_key_repo
    from app.core.auth.encryption import decrypt_value

    key_record = await api_key_repo.get_by_provider(db, user.id, provider)
    if key_record is None:
        raise HTTPException(status_code=400, detail=f"No API key configured for provider '{provider}'. Add one in Settings.")

    decrypted_key = decrypt_value(key_record.api_key_encrypted)
    checkpointer = getattr(request.app.state, "checkpointer", None)
    store = getattr(request.app.state, "store", None)
    agent = _get_agent(provider, body.model, key_record.base_url, decrypted_key, checkpointer, store)

    # Create or validate conversation
    is_new_conversation = body.conversation_id is None
    if is_new_conversation:
        conv = await conversation_service.create_conversation(db, user_id=user.id, model=body.model, provider=provider)
    else:
        try:
            conv_id = uuid.UUID(body.conversation_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation_id.")
        # Serialize the initial ownership check and user-message insert with
        # deletion. Without this lock, deletion can commit after validation but
        # before the message flush, turning a normal race into an FK error.
        conv = await conversation_repo.get_owned_for_update(db, conv_id, user.id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")

    # Save user message
    await message_repo.create(db, conversation_id=conv.id, role="user", content=body.message)
    await conversation_repo.touch(db, conv.id, model=body.model, provider=provider)
    await db.commit()

    thread_id = str(conv.id)
    conv_id_str = str(conv.id)

    if body.stream:
        async def stream_response() -> AsyncGenerator[bytes, None]:
            collected: list[str] = []
            usage: dict = {}
            completed = False
            missing = False
            generation_failed = False
            full_reply = ""

            try:
                # Hold a PostgreSQL row lock for the graph run. Conversation
                # deletion takes the same lock, so it cannot delete a checkpoint
                # that an in-flight run later recreates.
                try:
                    async with request.app.state.db_session() as session:
                        async with session.begin():
                            locked_conv = await conversation_repo.get_owned_for_update(
                                session,
                                conv.id,
                                user.id,
                            )
                            if locked_conv is None:
                                missing = True
                            else:
                                async for token in chat_service.stream_chat_session(
                                    agent,
                                    body.message,
                                    thread_id,
                                    usage_sink=usage,
                                ):
                                    collected.append(token)
                                    # JSON-encode so token newlines cannot break
                                    # the SSE frame delimiter.
                                    yield f"data: {json.dumps({'delta': token})}\n\n".encode()
                                completed = True

                                full_reply = "".join(collected)
                                if full_reply:
                                    await message_repo.create(
                                        session,
                                        conversation_id=conv.id,
                                        role="assistant",
                                        content=full_reply,
                                        model_used=body.model,
                                        provider=provider,
                                        tokens_used=usage.get("total"),
                                    )
                                    await conversation_repo.touch(session, conv.id)
                except Exception:
                    generation_failed = True
                    logger.exception(
                        "Streaming chat generation or persistence failed for conversation %s",
                        conv_id_str,
                    )

                if missing:
                    yield f"data: {json.dumps({'error': 'Conversation no longer exists.'})}\n\n".encode()
                elif generation_failed:
                    yield f"data: {json.dumps({'error': 'Generation failed.'})}\n\n".encode()
                elif completed and full_reply and is_new_conversation:
                    _schedule_title_update(
                        request,
                        agent,
                        conv.id,
                        user.id,
                        body.message,
                    )

                # Persistence and the generation lock are complete before DONE.
                yield b"data: [DONE]\n\n"
            finally:
                # Request-level observability. A tracer (Langfuse/LangSmith) is
                # the drop-in upgrade; this line already backs a latency/cost panel.
                logger.info(
                    "chat stream: conv=%s provider=%s model=%s tokens=%s latency=%.2fs",
                    conv_id_str,
                    provider,
                    body.model,
                    usage.get("total"),
                    time.time() - t_start,
                )

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"X-Conversation-Id": conv_id_str},
        )

    # Non-streaming
    usage: dict = {}
    try:
        async with request.app.state.db_session() as session:
            async with session.begin():
                locked_conv = await conversation_repo.get_owned_for_update(
                    session,
                    conv.id,
                    user.id,
                )
                if locked_conv is None:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Conversation no longer exists.",
                    )
                reply = await chat_service.run_chat_session(
                    agent,
                    body.message,
                    thread_id,
                    usage_sink=usage,
                )
                await message_repo.create(
                    session,
                    conversation_id=conv.id,
                    role="assistant",
                    content=reply,
                    model_used=body.model,
                    provider=provider,
                    tokens_used=usage.get("total"),
                )
                await conversation_repo.touch(session, conv.id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Non-streaming chat failed for conversation %s", conv_id_str)
        raise HTTPException(status_code=500, detail="Generation failed.") from exc

    if is_new_conversation:
        _schedule_title_update(
            request,
            agent,
            conv.id,
            user.id,
            body.message,
        )

    logger.info(
        "chat: conv=%s provider=%s model=%s tokens=%s latency=%.2fs",
        conv_id_str, provider, body.model, usage.get("total"), time.time() - t_start,
    )
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
