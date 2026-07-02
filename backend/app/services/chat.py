"""Chat service — orchestrates agent invocation for both legacy and session-aware endpoints."""
from collections.abc import AsyncGenerator, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.agent.agent import Agent
from app.schemas.conversation import ChatMessage


def convert_message(message: ChatMessage) -> BaseMessage:
    role = message.role
    content = message.content
    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    return AIMessage(content=content)


async def run_chat_legacy(agent: Agent, history: Sequence[ChatMessage]) -> str:
    """Stateless: takes full message history. Used by the legacy /chat/completions endpoint."""
    if not history:
        raise ValueError("At least one message is required.")
    messages = [convert_message(m) for m in history]
    if not any(isinstance(m, HumanMessage) for m in messages):
        raise ValueError("History must contain at least one user message.")
    return await agent.ainvoke_legacy(messages)


async def run_chat_session(agent: Agent, new_message: str, thread_id: str, usage_sink: dict | None = None) -> str:
    """Stateful: passes only the new message; history is in the checkpointer."""
    return await agent.ainvoke(HumanMessage(content=new_message), thread_id=thread_id, usage_sink=usage_sink)


async def stream_chat_session(agent: Agent, new_message: str, thread_id: str, usage_sink: dict | None = None) -> AsyncGenerator[str, None]:
    """Real token streaming for the session-aware endpoint."""
    async for token in agent.astream_tokens(HumanMessage(content=new_message), thread_id=thread_id, usage_sink=usage_sink):
        yield token
