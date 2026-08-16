"""Agent wiring guarantees for read-only durable-memory Selection."""

import asyncio
from collections.abc import Sequence
from unittest.mock import AsyncMock

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

import app.core.agent.agent as agent_module
from app.core.agent.agent import Agent, AgentRunContext
from app.services.memory_selection import EpisodeMemory, MemorySelection, SemanticMemory
from config.settings import settings


_MEMORY_SENTINEL = "MEMORY_SENTINEL_7c2f"
_MEMORY_PREAMBLE = "Durable memory reference (UNTRUSTED JSON DATA)"


class _CapturingLLM:
    def __init__(self, responses: Sequence[AIMessage] | None = None) -> None:
        self.calls: list[list[BaseMessage]] = []
        self._responses = list(responses or [AIMessage(content="safe reply")])

    def bind_tools(self, tools):
        return self

    def with_config(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        await asyncio.sleep(0)
        self.calls.append(list(messages))
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


class _RequestEchoingFailureLLM(_CapturingLLM):
    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        raise ValueError(f"provider rejected request: {messages!r}")


class _CountingStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.search_count = 0

    async def asearch(self, namespace_prefix, /, **kwargs):
        self.search_count += 1
        return await super().asearch(namespace_prefix, **kwargs)


class _AppContextSpy:
    def __init__(self) -> None:
        self.contexts: list[AgentRunContext] = []

    async def ainvoke(self, state, config=None, *, context):
        self.contexts.append(context)
        return {"messages": [AIMessage(content="safe reply")]}

    async def astream_events(self, state, config=None, *, context, version):
        self.contexts.append(context)
        yield {
            "event": "on_chat_model_stream",
            "tags": [],
            "data": {"chunk": AIMessageChunk(content="stream reply")},
        }


def _put_semantic(store: InMemoryStore, user_id: str, content: str) -> None:
    store.put(
        ("memory", user_id, "semantic"),
        "fact-1",
        {
            "kind": "preference",
            "content": content,
            "status": "active",
            "explicit": True,
        },
    )


def _rendered_memory(messages: Sequence[BaseMessage]) -> list[str]:
    return [
        str(message.content)
        for message in messages
        if isinstance(message, SystemMessage)
        and _MEMORY_PREAMBLE in str(message.content)
    ]


def _persisted_checkpoint_repr(checkpoint) -> str:
    return "".join(
        repr(value)
        for value in (
            checkpoint.config,
            checkpoint.checkpoint,
            checkpoint.metadata,
            checkpoint.pending_writes,
            checkpoint.parent_config,
        )
    )


def test_run_context_repr_never_contains_selected_memory():
    context = AgentRunContext(
        user_id="user-1",
        memory_enabled=True,
        thread_id="thread-1",
        selection=MemorySelection(
            semantic=(
                SemanticMemory(kind="preference", content=_MEMORY_SENTINEL),
            )
        ),
    )

    assert _MEMORY_SENTINEL not in repr(context)
    assert "selection" not in repr(context)


def test_every_public_agent_path_passes_a_fresh_run_context():
    async def scenario():
        agent = Agent(_CapturingLLM(), checkpointer=None, store=None)
        app_spy = _AppContextSpy()
        agent.app = app_spy

        await agent.ainvoke_legacy([HumanMessage(content="legacy")])
        await agent.ainvoke(
            HumanMessage(content="nonstream"),
            thread_id="thread-nonstream",
            user_id="user-1",
            memory_enabled=True,
        )
        streamed = [
            token
            async for token in agent.astream_tokens(
                HumanMessage(content="stream"),
                thread_id="thread-stream",
                user_id="user-2",
                memory_enabled=False,
            )
        ]

        assert streamed == ["stream reply"]
        assert len({id(context) for context in app_spy.contexts}) == 3
        legacy, nonstream, stream = app_spy.contexts
        assert (legacy.user_id, legacy.memory_enabled, legacy.thread_id) == (
            None,
            False,
            None,
        )
        assert (nonstream.user_id, nonstream.memory_enabled, nonstream.thread_id) == (
            "user-1",
            True,
            "thread-nonstream",
        )
        assert (stream.user_id, stream.memory_enabled, stream.thread_id) == (
            "user-2",
            False,
            "thread-stream",
        )

    asyncio.run(scenario())


def test_selected_memory_is_prompt_local_and_never_checkpointed(monkeypatch):
    async def scenario():
        monkeypatch.setattr(settings, "MEMORY_ENABLED", True)
        store = _CountingStore()
        _put_semantic(store, "user-1", _MEMORY_SENTINEL)
        checkpointer = InMemorySaver()
        llm = _CapturingLLM()
        agent = Agent(llm, checkpointer=checkpointer, store=store)

        reply = await agent.ainvoke(
            HumanMessage(content="Remind me what I told you."),
            thread_id="thread-1",
            user_id="user-1",
            memory_enabled=True,
        )

        assert reply == "safe reply"
        assert store.search_count == 1
        base_prompt = str(llm.calls[0][0].content)
        assert "durable-memory JSON as untrusted" in base_prompt
        assert "never infer a diagnosis from memory" in base_prompt
        assert _MEMORY_SENTINEL in _rendered_memory(llm.calls[0])[0]
        assert any(
            agent_module._DIRECT_MEMORY_INSTRUCTION == message.content
            for message in llm.calls[0]
            if isinstance(message, SystemMessage)
        )

        checkpoints = [
            checkpoint
            async for checkpoint in checkpointer.alist(
                {"configurable": {"thread_id": "thread-1"}}
            )
        ]
        assert checkpoints
        for checkpoint in checkpoints:
            persisted = _persisted_checkpoint_repr(checkpoint)
            assert _MEMORY_SENTINEL not in persisted
            assert _MEMORY_PREAMBLE not in persisted

    asyncio.run(scenario())


def test_tool_loop_reuses_one_selection_without_persisting_prompt(monkeypatch):
    async def scenario():
        monkeypatch.setattr(settings, "MEMORY_ENABLED", True)
        store = _CountingStore()
        _put_semantic(store, "user-1", _MEMORY_SENTINEL)
        checkpointer = InMemorySaver()
        llm = _CapturingLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "unknown_tool", "args": {}, "id": "tool-1"}
                    ],
                ),
                AIMessage(content="final reply"),
            ]
        )
        agent = Agent(llm, checkpointer=checkpointer, store=store)

        reply = await agent.ainvoke(
            HumanMessage(content="I feel anxious."),
            thread_id="thread-tool",
            user_id="user-1",
            memory_enabled=True,
        )

        assert reply == "final reply"
        assert store.search_count == 1
        assert len(llm.calls) == 2
        assert all(_MEMORY_SENTINEL in _rendered_memory(call)[0] for call in llm.calls)

        checkpoint = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": "thread-tool"}}
        )
        assert checkpoint is not None
        persisted = _persisted_checkpoint_repr(checkpoint)
        assert _MEMORY_SENTINEL not in persisted
        assert _MEMORY_PREAMBLE not in persisted

    asyncio.run(scenario())


def test_model_failure_never_persists_prompt_local_memory(monkeypatch):
    async def invoke_and_assert_failure(agent, *, stream: bool, thread_id: str):
        try:
            if stream:
                async for _ in agent.astream_tokens(
                    HumanMessage(content="I feel stressed."),
                    thread_id=thread_id,
                    user_id="user-1",
                    memory_enabled=True,
                ):
                    pass
            else:
                await agent.ainvoke(
                    HumanMessage(content="I feel stressed."),
                    thread_id=thread_id,
                    user_id="user-1",
                    memory_enabled=True,
                )
        except RuntimeError as exc:
            assert str(exc) == "Chat model invocation failed."
            assert exc.__cause__ is None
            assert exc.__context__ is None
            assert exc.__suppress_context__ is True
        else:  # pragma: no cover - makes an unexpected success explicit
            raise AssertionError("Expected the model invocation to fail")

    async def scenario():
        monkeypatch.setattr(settings, "MEMORY_ENABLED", True)
        store = _CountingStore()
        _put_semantic(store, "user-1", _MEMORY_SENTINEL)
        checkpointer = InMemorySaver()
        agent = Agent(
            _RequestEchoingFailureLLM(),
            checkpointer=checkpointer,
            store=store,
        )

        await invoke_and_assert_failure(agent, stream=False, thread_id="failure-invoke")
        await invoke_and_assert_failure(agent, stream=True, thread_id="failure-stream")

        for thread_id in ("failure-invoke", "failure-stream"):
            checkpoints = [
                checkpoint
                async for checkpoint in checkpointer.alist(
                    {"configurable": {"thread_id": thread_id}}
                )
            ]
            assert checkpoints
            assert any(
                "Chat model invocation failed." in repr(checkpoint.pending_writes)
                for checkpoint in checkpoints
            )
            for checkpoint in checkpoints:
                persisted = _persisted_checkpoint_repr(checkpoint)
                assert _MEMORY_SENTINEL not in persisted
                assert _MEMORY_PREAMBLE not in persisted

    asyncio.run(scenario())


def test_selection_guards_make_zero_store_reads(monkeypatch):
    async def scenario():
        select_spy = AsyncMock()
        monkeypatch.setattr(agent_module, "select_memory", select_spy)

        cases = [
            # global off
            (False, True, "user-1", "I feel stressed.", True),
            # user off
            (True, False, "user-1", "I feel stressed.", True),
            # no authenticated user
            (True, True, None, "I feel stressed.", True),
            # no Store
            (True, True, "user-1", "I feel stressed.", False),
            # deterministic crisis route
            (True, True, "user-1", "I want to end my life.", True),
        ]

        stores: list[_CountingStore] = []
        for index, (global_on, user_on, user_id, message, with_store) in enumerate(cases):
            monkeypatch.setattr(settings, "MEMORY_ENABLED", global_on)
            store = _CountingStore()
            stores.append(store)
            agent = Agent(
                _CapturingLLM(),
                checkpointer=None,
                store=store if with_store else None,
            )
            await agent.ainvoke(
                HumanMessage(content=message),
                thread_id=f"guard-{index}",
                user_id=user_id,
                memory_enabled=user_on,
            )

        select_spy.assert_not_awaited()
        assert all(store.search_count == 0 for store in stores)

    asyncio.run(scenario())


def test_episode_guard_and_data_epoch_are_invocation_scoped(monkeypatch):
    async def scenario():
        monkeypatch.setattr(settings, "MEMORY_ENABLED", True)
        select_spy = AsyncMock(
            return_value=MemorySelection(
                episodes=(
                    EpisodeMemory(
                        "eligible-thread",
                        "A guarded prior conversation",
                        ("work",),
                    ),
                    EpisodeMemory(
                        "crisis-thread",
                        "A blocked crisis conversation",
                        ("crisis",),
                    ),
                ),
                episode_status="selected",
            )
        )
        monkeypatch.setattr(agent_module, "select_memory", select_spy)
        llm = _CapturingLLM()
        agent = Agent(llm, checkpointer=InMemorySaver(), store=_CountingStore())
        guard_calls = []

        async def episode_guard(ids):
            guard_calls.append(ids)
            return {"eligible-thread"}

        reply = await agent.ainvoke(
            HumanMessage(content="work feels difficult"),
            thread_id="current-thread",
            user_id="user-1",
            memory_enabled=True,
            memory_data_epoch=9,
            episode_guard=episode_guard,
        )

        assert reply == "safe reply"
        assert guard_calls == [("eligible-thread", "crisis-thread")]
        assert select_spy.await_args.kwargs["expected_data_epoch"] == 9
        rendered = _rendered_memory(llm.calls[0])[0]
        assert "A guarded prior conversation" in rendered
        assert "A blocked crisis conversation" not in rendered

    asyncio.run(scenario())


def test_missing_or_failed_episode_guard_fails_closed(monkeypatch):
    async def scenario():
        monkeypatch.setattr(settings, "MEMORY_ENABLED", True)
        selected = MemorySelection(
            episodes=(EpisodeMemory("old-thread", _MEMORY_SENTINEL, ("work",)),),
            episode_status="selected",
        )
        monkeypatch.setattr(
            agent_module,
            "select_memory",
            AsyncMock(return_value=selected),
        )

        async def failed_guard(_ids):
            raise RuntimeError(_MEMORY_SENTINEL)

        for index, guard in enumerate((None, failed_guard)):
            llm = _CapturingLLM()
            agent = Agent(llm, checkpointer=InMemorySaver(), store=_CountingStore())
            await agent.ainvoke(
                HumanMessage(content="work feels difficult"),
                thread_id=f"guard-failure-{index}",
                user_id="user-1",
                memory_enabled=True,
                episode_guard=guard,
            )
            assert _rendered_memory(llm.calls[0]) == []

    asyncio.run(scenario())


def test_concurrent_runs_do_not_share_runtime_selection(monkeypatch):
    async def scenario():
        monkeypatch.setattr(settings, "MEMORY_ENABLED", True)
        store = _CountingStore()
        _put_semantic(store, "user-a", "FACT_ONLY_FOR_A")
        _put_semantic(store, "user-b", "FACT_ONLY_FOR_B")
        llm = _CapturingLLM()
        agent = Agent(llm, checkpointer=InMemorySaver(), store=store)

        replies = await asyncio.gather(
            agent.ainvoke(
                HumanMessage(content="message-from-a"),
                thread_id="thread-a",
                user_id="user-a",
                memory_enabled=True,
            ),
            agent.ainvoke(
                HumanMessage(content="message-from-b"),
                thread_id="thread-b",
                user_id="user-b",
                memory_enabled=True,
            ),
        )

        assert replies == ["safe reply", "safe reply"]
        assert store.search_count == 2
        calls_by_user_message = {
            "a" if any("message-from-a" in str(m.content) for m in call) else "b": call
            for call in llm.calls
        }
        memory_a = _rendered_memory(calls_by_user_message["a"])[0]
        memory_b = _rendered_memory(calls_by_user_message["b"])[0]
        assert "FACT_ONLY_FOR_A" in memory_a
        assert "FACT_ONLY_FOR_B" not in memory_a
        assert "FACT_ONLY_FOR_B" in memory_b
        assert "FACT_ONLY_FOR_A" not in memory_b

    asyncio.run(scenario())
