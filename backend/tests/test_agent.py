"""Agent guarantees with stub LLMs: crisis-on-failure referral + content normalization."""
import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.core.agent.agent import Agent


class _ExplodingLLM:
    def bind_tools(self, tools):
        return self

    def with_config(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        raise RuntimeError("provider down")


class _BlockContentLLM:
    """Returns Anthropic-style block-list content."""

    def bind_tools(self, tools):
        return self

    def with_config(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content=[{"type": "text", "text": "Hello there"}])


CRISIS_MSG = "I've been thinking about ending it all."


def _collect_stream(agent, msg):
    async def run():
        out = []
        async for tok in agent.astream_tokens(HumanMessage(content=msg), thread_id="t"):
            out.append(tok)
        return "".join(out)

    return asyncio.run(run())


def test_crisis_stream_still_delivers_resources_when_llm_fails():
    agent = Agent(_ExplodingLLM(), checkpointer=None)
    assert "988" in _collect_stream(agent, CRISIS_MSG)


def test_noncrisis_stream_failure_still_raises():
    agent = Agent(_ExplodingLLM(), checkpointer=None)
    with pytest.raises(RuntimeError):
        _collect_stream(agent, "I feel a bit tired.")


def test_crisis_ainvoke_returns_resources_when_llm_fails():
    agent = Agent(_ExplodingLLM(), checkpointer=None)
    out = asyncio.run(agent.ainvoke(HumanMessage(content=CRISIS_MSG), thread_id="t"))
    assert "988" in out


def test_block_content_normalized_not_repred():
    agent = Agent(_BlockContentLLM(), checkpointer=None)
    out = asyncio.run(agent.acomplete([HumanMessage(content="hi")]))
    assert out == "Hello there"


def test_agent_closes_langgraph_event_stream_on_consumer_disconnect():
    async def scenario():
        closed = asyncio.Event()

        class StreamingApp:
            async def astream_events(self, *args, **kwargs):
                try:
                    yield {
                        "event": "on_chat_model_stream",
                        "tags": [],
                        "data": {"chunk": AIMessageChunk(content="token")},
                    }
                    await asyncio.Event().wait()
                finally:
                    closed.set()

        agent = object.__new__(Agent)
        agent.app = StreamingApp()
        stream = agent.astream_tokens(
            HumanMessage(content="hello"),
            thread_id="thread-1",
        )

        assert await anext(stream) == "token"
        assert not closed.is_set()
        await stream.aclose()

        assert closed.is_set()

    asyncio.run(scenario())


def test_tool_limit_answers_every_pending_call_under_its_own_id():
    """Regression: the limit notice used one fixed id, which providers reject."""
    agent = Agent(_ExplodingLLM(), checkpointer=None)
    pending = AIMessage(
        content="",
        tool_calls=[
            {"name": "fetch_mental_health_examples", "args": {"query": "a"}, "id": "call_a"},
            {"name": "fetch_mental_health_examples", "args": {"query": "b"}, "id": "call_b"},
        ],
    )
    state = {
        "messages": [HumanMessage(content="hi"), pending],
        "tool_iterations": agent._max_tool_calls,
    }

    result = asyncio.run(agent._tool_node(state))

    assert [message.tool_call_id for message in result["messages"]] == ["call_a", "call_b"]
    assert result["tool_iterations"] == agent._max_tool_calls + 1
