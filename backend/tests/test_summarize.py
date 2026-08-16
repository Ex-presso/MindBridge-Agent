"""Working-memory compaction: token-based trigger, microCompact, prune (no real LLM)."""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from app.core.agent.agent import Agent, _MICRO_PLACEHOLDER
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
    out = []
    for i in range(n):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        out.append(cls(content=f"message number {i} with some words", id=f"m{i}"))
    return out


def test_below_budget_is_noop():
    # A few short messages are nowhere near the default token budget.
    state = {"messages": _convo(4), "tool_iterations": 0}
    assert asyncio.run(_agent()._summarize_node(state)) == {}


def test_over_budget_summarizes_and_prunes(monkeypatch):
    monkeypatch.setattr(settings, "COMPACT_TRIGGER_TOKENS", 30)
    monkeypatch.setattr(settings, "COMPACT_KEEP_RECENT_TOKENS", 10)
    msgs = _convo(24)
    result = asyncio.run(_agent()._summarize_node({"messages": msgs, "tool_iterations": 0}))

    assert result["summary"] == "SUMMARY OF EARLIER TURNS"
    removals = result["messages"]
    assert removals and all(isinstance(m, RemoveMessage) for m in removals)
    removed_ids = {m.id for m in removals}
    assert "m0" in removed_ids            # oldest pruned
    assert len(removed_ids) < len(msgs)   # recent tail kept


def test_oversized_latest_message_kept_without_index_error(monkeypatch):
    monkeypatch.setattr(settings, "COMPACT_TRIGGER_TOKENS", 20)
    monkeypatch.setattr(settings, "COMPACT_KEEP_RECENT_TOKENS", 5)
    msgs = [
        HumanMessage(content="earlier concern", id="m0"),
        AIMessage(content="earlier response", id="m1"),
        HumanMessage(content="very long current message " * 40, id="m2"),
    ]

    result = asyncio.run(_agent()._summarize_node({"messages": msgs, "tool_iterations": 0}))

    assert result["summary"] == "SUMMARY OF EARLIER TURNS"
    removed_ids = {m.id for m in result["messages"] if isinstance(m, RemoveMessage)}
    assert removed_ids == {"m0", "m1"}
    assert "m2" not in removed_ids


def test_cut_never_strands_a_tool_pair(monkeypatch):
    monkeypatch.setattr(settings, "COMPACT_TRIGGER_TOKENS", 20)
    monkeypatch.setattr(settings, "COMPACT_KEEP_RECENT_TOKENS", 12)
    msgs = _convo(16)
    msgs += [
        HumanMessage(content="user turn about feelings", id="m16"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "call1"}], id="m17"),
        ToolMessage(content="retrieved counseling example text", tool_call_id="call1", id="m18"),
        AIMessage(content="an empathic answer", id="m19"),
        HumanMessage(content="next thing on my mind", id="m20"),
        AIMessage(content="another reply", id="m21"),
    ]
    result = asyncio.run(_agent()._summarize_node({"messages": msgs, "tool_iterations": 0}))
    removed_ids = {m.id for m in result["messages"] if isinstance(m, RemoveMessage)}
    kept = [m for m in msgs if m.id not in removed_ids]
    # Invariant: the kept window never opens on an orphaned ToolMessage.
    assert kept and not isinstance(kept[0], ToolMessage)


def test_micro_compact_replaces_old_tool_results_keeps_recent(monkeypatch):
    monkeypatch.setattr(settings, "COMPACT_MICRO_KEEP_RESULTS", 1)
    monkeypatch.setattr(settings, "COMPACT_MICRO_MIN_CHARS", 10)
    tools = [
        ToolMessage(content="OLD result one " * 5, tool_call_id="c1", id="t1"),
        ToolMessage(content="OLD result two " * 5, tool_call_id="c2", id="t2"),
        ToolMessage(content="RECENT result " * 5, tool_call_id="c3", id="t3"),
    ]
    msgs = [HumanMessage(content="hi", id="h0"), *tools]
    replacements, reclaimed = _agent()._micro_compact(msgs)

    ids = {m.id for m in replacements}
    assert ids == {"t1", "t2"}                        # last tool result (t3) kept verbatim
    assert all(m.content == _MICRO_PLACEHOLDER for m in replacements)
    assert all(m.tool_call_id for m in replacements)  # pairing preserved (id + tool_call_id)
    assert reclaimed > 0


def test_micro_compact_alone_can_avoid_summary(monkeypatch):
    # Over budget, but compacting old tool results brings us back under -> no LLM summary.
    monkeypatch.setattr(settings, "COMPACT_MICRO_KEEP_RESULTS", 0)
    monkeypatch.setattr(settings, "COMPACT_MICRO_MIN_CHARS", 10)
    big = "counseling example paragraph " * 40        # a large, re-derivable tool result
    msgs = [
        HumanMessage(content="I feel anxious", id="h0"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}], id="a0"),
        ToolMessage(content=big, tool_call_id="c1", id="t0"),
        AIMessage(content="short reply", id="a1"),
    ]
    agent = _agent()
    full = agent._estimate_tokens(msgs)
    _, reclaimed = agent._micro_compact(msgs)
    # Trigger between (full - reclaimed) and full so microCompact alone suffices.
    monkeypatch.setattr(settings, "COMPACT_TRIGGER_TOKENS", full - reclaimed // 2)

    result = asyncio.run(agent._summarize_node({"messages": msgs, "tool_iterations": 0}))
    assert "summary" not in result                    # expensive LLM path NOT taken
    assert result["messages"] and all(isinstance(m, ToolMessage) for m in result["messages"])
    assert result["messages"][0].content == _MICRO_PLACEHOLDER
