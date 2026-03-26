"""LLM provider factory — creates chat model instances."""
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import settings


# Available models per provider (shown to user in frontend)
PROVIDER_MODELS: dict[str, list[dict[str, str]]] = {
    "openai": [
        {"id": "gpt-4o", "name": "GPT-4o"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
        {"id": "o3-mini", "name": "o3 Mini"},
    ],
    "anthropic": [
        {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5"},
        {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5"},
    ],
    "google_genai": [
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
    ],
    "openai_compatible": [],
    "anthropic_compatible": [],
}


def get_llm(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> BaseChatModel:
    """Create a chat model for the given provider.

    When api_key is provided, it is used directly (per-user keys).
    When api_key is None, falls back to system-wide keys from settings.
    """
    provider = provider.lower()
    temperature = settings.MODEL_TEMPERATURE

    if provider in ("openai", "openai_compatible"):
        from langchain_openai import ChatOpenAI

        key = api_key
        if not key and settings.OPENAI_API_KEY:
            key = settings.OPENAI_API_KEY.get_secret_value()

        kwargs: dict = {
            "model": model or settings.OPENAI_MODEL,
            "temperature": temperature,
        }
        if key:
            kwargs["api_key"] = key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    elif provider in ("anthropic", "anthropic_compatible"):
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": model or "claude-sonnet-4-5",
            "temperature": temperature,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatAnthropic(**kwargs)

    elif provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = api_key
        if not key and settings.GEMINI_API_KEY:
            key = settings.GEMINI_API_KEY.get_secret_value()

        kwargs = {
            "model": model or settings.GEMINI_MODEL,
            "temperature": temperature,
        }
        if key:
            kwargs["api_key"] = key
        return ChatGoogleGenerativeAI(**kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}")
