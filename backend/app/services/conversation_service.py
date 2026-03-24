"""Conversation management: creation, listing, history retrieval, deletion."""
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.agent.agent import Agent
from app.db.models.conversation import Conversation
from app.db.models.message import Message
from app.db.repositories import conversation_repo, message_repo


async def create_conversation(db: AsyncSession, *, user_id: uuid.UUID, model: str | None = None, provider: str | None = None) -> Conversation:
    return await conversation_repo.create(db, user_id=user_id, model=model, provider=provider)


async def get_conversation_with_messages(db: AsyncSession, conv_id: uuid.UUID, user_id: uuid.UUID) -> dict[str, Any] | None:
    conv = await conversation_repo.get_by_id(db, conv_id)
    if conv is None or conv.user_id != user_id:
        return None
    messages = await message_repo.list_by_conversation(db, conv_id)
    return {
        "id": str(conv.id),
        "title": conv.title,
        "model": conv.model,
        "provider": conv.provider,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }


async def list_conversations(db: AsyncSession, user_id: uuid.UUID) -> list[dict[str, Any]]:
    convs = await conversation_repo.list_by_user(db, user_id)
    return [
        {
            "id": str(c.id),
            "title": c.title,
            "model": c.model,
            "provider": c.provider,
            "updated_at": c.updated_at.isoformat(),
            "created_at": c.created_at.isoformat(),
        }
        for c in convs
    ]


async def delete_conversation(db: AsyncSession, conv_id: uuid.UUID, user_id: uuid.UUID, *, checkpointer=None) -> bool:
    conv = await conversation_repo.get_by_id(db, conv_id)
    if conv is None or conv.user_id != user_id:
        return False
    # Delete LangGraph checkpoint if checkpointer available
    if checkpointer is not None:
        try:
            config = {"configurable": {"thread_id": str(conv_id)}}
            await checkpointer.adelete_thread(config)
        except Exception:
            pass
    return await conversation_repo.delete_by_id(db, conv_id)


async def generate_title(agent: Agent, first_user_message: str) -> str:
    """Generate a short conversation title from the first user message."""
    prompt = f'Summarize this message in 5 words or fewer, as a conversation title. Only return the title, no punctuation: "{first_user_message[:200]}"'
    try:
        state = {
            "messages": [
                SystemMessage(content="You generate short conversation titles."),
                HumanMessage(content=prompt),
            ],
            "tool_iterations": 0,
        }
        result = await agent.app.ainvoke(state)
        ai_msg = next((m for m in reversed(result["messages"]) if hasattr(m, "content") and m.content), None)
        title = str(ai_msg.content).strip()[:100] if ai_msg else first_user_message[:50]
        return title
    except Exception:
        return first_user_message[:50]
