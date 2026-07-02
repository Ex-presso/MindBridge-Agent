"""Agent guarantees with stub LLMs: crisis-on-failure referral + content normalization."""
import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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
