"""Summarization node: threshold + pruning logic (no real LLM)."""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from app.core.agent.agent import Agent
from config.settings import settings


class _FakeLLM:
    """Minimal stand-in so Agent builds without a provider."""

    def bind_tools(self, tools):
        return self

    def with_config(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="SUMMARY OF EARLIER TURNS")


def _agent():
    return Agent(_FakeLLM(), checkpointer=None)


def _convo(n: int):
    # n alternating messages, each with a stable id so they're prunable.
    out = []
    for i in range(n):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        out.append(cls(content=f"msg {i}", id=f"m{i}"))
    return out


def test_below_threshold_is_noop():
    state = {"messages": _convo(4), "tool_iterations": 0}
    result = asyncio.run(_agent()._summarize_node(state))
    assert result == {}


def test_above_threshold_summarizes_and_prunes():
    n = settings.SUMMARY_TRIGGER_MESSAGES + 2
    state = {"messages": _convo(n), "tool_iterations": 0}
    result = asyncio.run(_agent()._summarize_node(state))

    assert result["summary"] == "SUMMARY OF EARLIER TURNS"
    removals = result["messages"]
    assert all(isinstance(m, RemoveMessage) for m in removals)
    # Everything except the last SUMMARY_KEEP_RECENT is pruned.
    assert len(removals) == n - settings.SUMMARY_KEEP_RECENT
    kept_ids = {f"m{i}" for i in range(n - settings.SUMMARY_KEEP_RECENT, n)}
    removed_ids = {m.id for m in removals}
    assert removed_ids.isdisjoint(kept_ids)
