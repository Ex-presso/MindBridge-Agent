from functools import lru_cache

from app.core.agent.agent import Agent


@lru_cache(maxsize=8)
def _get_agent(provider: str) -> Agent:
    return Agent(provider)


def run_chat(provider: str, message: str) -> str:
    """Execute the chat graph for the given provider and return the assistant reply."""
    normalized = provider.lower()
    agent = _get_agent(normalized)
    return agent.invoke(message)
