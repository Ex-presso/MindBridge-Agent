"""Chat endpoints: legacy OpenAI-compatible + new session-aware."""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.core.auth.deps import get_current_user
from app.core.agent.safety import CRISIS_DETECTOR_VERSION, detect_crisis
from app.db.engine import get_db
from app.db.models.user import User
from app.db.repositories import (
    api_key_repo,
    conversation_repo,
    memory_job_repo,
    message_repo,
    user_repo,
)
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

if TYPE_CHECKING:
    from app.core.agent.agent import Agent

router = APIRouter()
logger = logging.getLogger(__name__)

# Cache compiled agents by (user, provider, model, base_url, key-hash, runtime).
# The LangGraph graph is stateless — per-conversation state lives in the
# checkpointer — so one agent is safe to reuse across requests for the same
# user and LLM config. User remains part of the key so account deletion can
# evict every object that may retain a decrypted BYOK value.
# Bounded LRU at 128 entries; bump if you serve many model configs.
_AGENT_CACHE: "OrderedDict[tuple, Agent]" = OrderedDict()
_AGENT_CACHE_MAX = 128
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()
_BACKGROUND_TASK_USERS: dict[asyncio.Task[None], str] = {}
_TITLE_UPDATE_TIMEOUT_SECONDS = 15.0
_RUNTIME_EVICTION_TIMEOUT_SECONDS = 1.0


def _episode_guard(db, user_id: uuid.UUID):
    """Build a request-local relational eligibility check for Store episodes."""

    async def guard(raw_ids: tuple[str, ...]) -> set[str]:
        parsed_ids: list[uuid.UUID] = []
        for raw_id in raw_ids:
            try:
                parsed = uuid.UUID(raw_id)
            except (TypeError, ValueError):
                continue
            if str(parsed) == raw_id:
                parsed_ids.append(parsed)
        return await conversation_repo.filter_memory_eligible_ids(
            db,
            user_id,
            tuple(parsed_ids),
        )

    return guard


async def _enqueue_episode_after_success(
    db,
    *,
    user_id: uuid.UUID,
    conversation,
    target_revision: int,
    user_message,
    assistant_message,
    api_key,
    provider: str,
    model: str,
) -> None:
    """Capture a successful turn in the same transaction as its revision."""
    if not settings.MEMORY_ENABLED:
        return
    access = await user_repo.get_memory_access_snapshot(db, user_id)
    if (
        access is None
        or not access.enabled
        or access.account_deletion_pending
        or conversation.memory_crisis_seen
        or not conversation.memory_crisis_reviewed
        or conversation.memory_crisis_review_version
        != CRISIS_DETECTOR_VERSION
    ):
        return
    await memory_job_repo.enqueue_extract_episode(
        db,
        user_id=user_id,
        conversation_id=conversation.id,
        target_revision=target_revision,
        consent_version=access.consent_version,
        data_epoch=access.data_epoch,
        source_user_message_id=user_message.id,
        source_assistant_message_id=assistant_message.id,
        api_key_id=api_key.id,
        provider=provider,
        model=model,
        base_url=api_key.base_url,
    )


def _get_agent(
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key: str,
    checkpointer,
    store=None,
    *,
    user_id: str,
) -> "Agent":
    from app.core.agent.agent import Agent
    from app.core.llm.provider import get_llm

    key = (
        str(user_id),
        provider,
        model,
        base_url,
        hashlib.sha256(api_key.encode()).hexdigest(),
        id(checkpointer),
        id(store),
    )
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


async def evict_user_runtime(user_id: uuid.UUID | str) -> None:
    """Drop decrypted-key agents and drain background work for one user."""
    normalized_user_id = str(user_id)
    for key in tuple(_AGENT_CACHE):
        if key and key[0] == normalized_user_id:
            _AGENT_CACHE.pop(key, None)

    tasks = [
        task
        for task, owner_id in tuple(_BACKGROUND_TASK_USERS.items())
        if owner_id == normalized_user_id
    ]
    for task in tasks:
        task.cancel()
    if tasks:
        _, pending = await asyncio.wait(
            tasks,
            timeout=_RUNTIME_EVICTION_TIMEOUT_SECONDS,
        )
        if pending:
            # A third-party provider coroutine can suppress cancellation. Do not
            # let cache eviction itself hang forever. Its User KEY SHARE lock
            # still prevents the following credential/account mutation from
            # committing until the old-key task actually exits.
            logger.warning(
                "Timed out draining %s background task(s) during user runtime eviction.",
                len(pending),
            )


async def _save_generated_title_if_present(
    request: Request,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    first_user_message: str,
    provider: str,
    model: str | None,
) -> None:
    """Best-effort title generation that never changes the chat result."""
    try:
        async with request.app.state.db_session() as session:
            async with session.begin():
                access = await user_repo.get_memory_access_for_chat(
                    session,
                    user_id,
                )
                if access is None or access.account_deletion_pending:
                    return
                key_record = await api_key_repo.get_by_provider(
                    session,
                    user_id,
                    provider,
                )
                if key_record is None:
                    return
                conv = await conversation_repo.get_owned_for_update(
                    session,
                    conversation_id,
                    user_id,
                )
                if conv is not None:
                    from app.core.auth.encryption import decrypt_value

                    agent = _get_agent(
                        provider,
                        model,
                        key_record.base_url,
                        decrypt_value(key_record.api_key_encrypted),
                        getattr(request.app.state, "checkpointer", None),
                        getattr(request.app.state, "store", None),
                        user_id=str(user_id),
                    )
                    title = await conversation_service.generate_title(
                        agent,
                        first_user_message,
                    )
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
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    first_user_message: str,
    provider: str,
    model: str | None,
) -> None:
    """Run best-effort title work without delaying a successful chat response."""

    async def run_bounded() -> None:
        try:
            await asyncio.wait_for(
                _save_generated_title_if_present(
                    request,
                    conversation_id,
                    user_id,
                    first_user_message,
                    provider,
                    model,
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
    _BACKGROUND_TASK_USERS[task] = str(user_id)

    def release(done: asyncio.Task[None]) -> None:
        _BACKGROUND_TASKS.discard(done)
        _BACKGROUND_TASK_USERS.pop(done, None)

    task.add_done_callback(release)


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
    from app.core.auth.encryption import decrypt_value

    checkpointer = getattr(request.app.state, "checkpointer", None)
    store = getattr(request.app.state, "store", None)

    # Global lock order starts with a User KEY SHARE deletion barrier. Account
    # deletion takes FOR UPDATE, commits its tombstone, and prevents any later
    # conversation/FK work from entering this transaction.
    initial_access = await user_repo.get_memory_access_for_chat(db, user.id)
    if initial_access is None:
        raise HTTPException(status_code=401, detail="User not found.")
    if initial_access.account_deletion_pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account deletion is in progress.",
        )

    key_record = await api_key_repo.get_by_provider(db, user.id, provider)
    if key_record is None:
        raise HTTPException(status_code=400, detail=f"No API key configured for provider '{provider}'. Add one in Settings.")

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
    user_message = await message_repo.create(
        db,
        conversation_id=conv.id,
        role="user",
        content=body.message,
    )
    current_crisis = detect_crisis(body.message) is not None
    transitioned = False
    if (
        not conv.memory_crisis_reviewed
        or conv.memory_crisis_review_version != CRISIS_DETECTOR_VERSION
    ):
        # Migration marks legacy conversations as unknown. Review their complete
        # relational user-message history once, under the Conversation lock,
        # before they can become eligible for Selection or Extraction.
        history = await message_repo.list_by_conversation(db, conv.id)
        crisis_seen = any(
            message.role == "user" and detect_crisis(message.content) is not None
            for message in history
        )
        transitioned = crisis_seen and not conv.memory_crisis_seen
        await conversation_repo.record_memory_crisis_review(
            db,
            conv.id,
            user.id,
            crisis_seen=crisis_seen,
        )
        conv.memory_crisis_seen = conv.memory_crisis_seen or crisis_seen
        conv.memory_crisis_reviewed = True
        conv.memory_crisis_review_version = CRISIS_DETECTOR_VERSION
    elif current_crisis:
        # Sticky relational tombstone: once a crisis turn occurs, no episode
        # from this conversation may be selected again, even if Store cleanup
        # is delayed or temporarily unavailable.
        transitioned = await conversation_repo.mark_memory_crisis_seen(
            db,
            conv.id,
            user.id,
        )
    if transitioned:
        await memory_job_repo.enqueue_delete_episode(
            db,
            user_id=user.id,
            conversation_id=conv.id,
            target_revision=int(conv.memory_revision),
            consent_version=initial_access.consent_version,
            data_epoch=initial_access.data_epoch,
        )
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
            credential_missing = False
            full_reply = ""
            run_agent = None

            try:
                # Hold a PostgreSQL row lock for the graph run. Conversation
                # deletion takes the same lock, so it cannot delete a checkpoint
                # that an in-flight run later recreates.
                try:
                    async with request.app.state.db_session() as session:
                        async with session.begin():
                            memory_access = await user_repo.get_memory_access_for_chat(
                                session,
                                user.id,
                            )
                            if (
                                memory_access is None
                                or memory_access.account_deletion_pending
                            ):
                                missing = True
                            else:
                                run_key_record = await api_key_repo.get_by_provider(
                                    session,
                                    user.id,
                                    provider,
                                )
                                if run_key_record is None:
                                    credential_missing = True
                                else:
                                    locked_conv = await conversation_repo.get_owned_for_update(
                                        session,
                                        conv.id,
                                        user.id,
                                    )
                                    if locked_conv is None:
                                        missing = True
                                    else:
                                        run_agent = _get_agent(
                                            provider,
                                            body.model,
                                            run_key_record.base_url,
                                            decrypt_value(
                                                run_key_record.api_key_encrypted
                                            ),
                                            checkpointer,
                                            store,
                                            user_id=str(user.id),
                                        )
                                        tokens = chat_service.stream_chat_session(
                                            run_agent,
                                            body.message,
                                            thread_id,
                                            user_id=str(user.id),
                                            memory_enabled=memory_access.enabled,
                                            memory_data_epoch=memory_access.data_epoch,
                                            episode_guard=_episode_guard(
                                                session,
                                                user.id,
                                            ),
                                            usage_sink=usage,
                                        )
                                        async with aclosing(tokens) as token_stream:
                                            async for token in token_stream:
                                                collected.append(token)
                                                # JSON-encode so token newlines cannot break
                                                # the SSE frame delimiter.
                                                yield f"data: {json.dumps({'delta': token})}\n\n".encode()
                                        full_reply = "".join(collected)
                                        if not full_reply.strip():
                                            raise RuntimeError(
                                                "Model returned an empty response."
                                            )
                                        assistant_message = await message_repo.create(
                                            session,
                                            conversation_id=conv.id,
                                            role="assistant",
                                            content=full_reply,
                                            model_used=body.model,
                                            provider=provider,
                                            tokens_used=usage.get("total"),
                                        )
                                        await conversation_repo.touch(
                                            session,
                                            conv.id,
                                        )
                                        target_revision = await conversation_repo.increment_memory_revision(
                                            session,
                                            conv.id,
                                        )
                                        await _enqueue_episode_after_success(
                                            session,
                                            user_id=user.id,
                                            conversation=locked_conv,
                                            target_revision=target_revision,
                                            user_message=user_message,
                                            assistant_message=assistant_message,
                                            api_key=run_key_record,
                                            provider=provider,
                                            model=body.model,
                                        )
                                        completed = True
                except Exception:
                    generation_failed = True
                    logger.exception(
                        "Streaming chat generation or persistence failed for conversation %s",
                        conv_id_str,
                    )

                if missing:
                    yield f"data: {json.dumps({'error': 'Conversation no longer exists.'})}\n\n".encode()
                    return
                elif credential_missing:
                    yield f"data: {json.dumps({'error': 'API key no longer available.'})}\n\n".encode()
                    return
                elif generation_failed:
                    yield f"data: {json.dumps({'error': 'Generation failed.'})}\n\n".encode()
                    return
                if not completed:
                    yield f"data: {json.dumps({'error': 'Generation failed.'})}\n\n".encode()
                    return
                if (
                    completed
                    and full_reply
                    and is_new_conversation
                    and run_agent is not None
                ):
                    _schedule_title_update(
                        request,
                        conv.id,
                        user.id,
                        body.message,
                        provider,
                        body.model,
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
    run_agent = None
    try:
        async with request.app.state.db_session() as session:
            async with session.begin():
                memory_access = await user_repo.get_memory_access_for_chat(
                    session,
                    user.id,
                )
                if (
                    memory_access is None
                    or memory_access.account_deletion_pending
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Account or conversation no longer exists.",
                    )
                run_key_record = await api_key_repo.get_by_provider(
                    session,
                    user.id,
                    provider,
                )
                if run_key_record is None:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="API key no longer available.",
                    )
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
                run_agent = _get_agent(
                    provider,
                    body.model,
                    run_key_record.base_url,
                    decrypt_value(run_key_record.api_key_encrypted),
                    checkpointer,
                    store,
                    user_id=str(user.id),
                )
                reply = await chat_service.run_chat_session(
                    run_agent,
                    body.message,
                    thread_id,
                    user_id=str(user.id),
                    memory_enabled=memory_access.enabled,
                    memory_data_epoch=memory_access.data_epoch,
                    episode_guard=_episode_guard(session, user.id),
                    usage_sink=usage,
                )
                if not reply.strip():
                    raise RuntimeError("Model returned an empty response.")
                assistant_message = await message_repo.create(
                    session,
                    conversation_id=conv.id,
                    role="assistant",
                    content=reply,
                    model_used=body.model,
                    provider=provider,
                    tokens_used=usage.get("total"),
                )
                await conversation_repo.touch(session, conv.id)
                target_revision = await conversation_repo.increment_memory_revision(
                    session,
                    conv.id,
                )
                await _enqueue_episode_after_success(
                    session,
                    user_id=user.id,
                    conversation=locked_conv,
                    target_revision=target_revision,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    api_key=run_key_record,
                    provider=provider,
                    model=body.model,
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Non-streaming chat failed for conversation %s", conv_id_str)
        raise HTTPException(status_code=500, detail="Generation failed.") from exc

    if is_new_conversation and run_agent is not None:
        _schedule_title_update(
            request,
            conv.id,
            user.id,
            body.message,
            provider,
            body.model,
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
