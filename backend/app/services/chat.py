from collections.abc import Sequence
from functools import lru_cache
from typing import Final

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.agent.agent import Agent
from app.schemas.conversation import ChatMessage

# --- NEW: imports for emotion detection ---
from transformers import pipeline

# cache the model for efficiency
@lru_cache(maxsize=1)
def get_emotion_analyzer():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)


@lru_cache(maxsize=8)
def _get_agent(provider: str) -> Agent:
    return Agent(provider)


def _convert_message(message: ChatMessage) -> BaseMessage:
    """Map an incoming chat completion message to a LangChain message type."""
    role = message.role
    content = message.content

    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)

    error: Final[str] = f"Unsupported role: {role}"
    raise ValueError(error)


def run_chat(provider: str, history: Sequence[ChatMessage]) -> str:
    """Execute the chat graph for the given provider using the full message history."""
    if not history:
        raise ValueError("At least one message is required.")

    normalized = provider.lower()
    agent = _get_agent(normalized)

    messages = [_convert_message(item) for item in history]

    has_user_message = any(isinstance(msg, HumanMessage) for msg in messages)
    if not has_user_message:
        raise ValueError("Chat history must include at least one user message.")

    # --- NEW: detect emotion from the last user message ---
    emotion_analyzer = get_emotion_analyzer()
    last_user_msg = [m for m in messages if isinstance(m, HumanMessage)][-1]
    emotion_result = emotion_analyzer(last_user_msg.content)[0]
    emotion = emotion_result["label"]

    # pass emotion info to the agent
    return agent.invoke(messages, emotion=emotion)
