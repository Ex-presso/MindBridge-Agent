"""Chat service: invokes the agent for the session-aware endpoint."""
from collections.abc import AsyncGenerator
from contextlib import aclosing

from langchain_core.messages import HumanMessage

from app.core.agent.agent import Agent, EpisodeEligibilityGuard


async def run_chat_session(
    agent: Agent,
    new_message: str,
    thread_id: str,
    *,
    user_id: str,
    memory_enabled: bool,
    memory_data_epoch: int = 0,
    episode_guard: EpisodeEligibilityGuard | None = None,
    usage_sink: dict | None = None,
) -> str:
    """Stateful: passes only the new message; history is in the checkpointer."""
    return await agent.ainvoke(
        HumanMessage(content=new_message),
        thread_id=thread_id,
        usage_sink=usage_sink,
        user_id=user_id,
        memory_enabled=memory_enabled,
        memory_data_epoch=memory_data_epoch,
        episode_guard=episode_guard,
    )


async def stream_chat_session(
    agent: Agent,
    new_message: str,
    thread_id: str,
    *,
    user_id: str,
    memory_enabled: bool,
    memory_data_epoch: int = 0,
    episode_guard: EpisodeEligibilityGuard | None = None,
    usage_sink: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Real token streaming for the session-aware endpoint."""
    tokens = agent.astream_tokens(
        HumanMessage(content=new_message),
        thread_id=thread_id,
        usage_sink=usage_sink,
        user_id=user_id,
        memory_enabled=memory_enabled,
        memory_data_epoch=memory_data_epoch,
        episode_guard=episode_guard,
    )
    async with aclosing(tokens) as token_stream:
        async for token in token_stream:
            yield token
