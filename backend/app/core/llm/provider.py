"""LLM provider factory — creates chat model instances."""

from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import settings

# Available models per provider (shown to user in frontend)
PROVIDER_MODELS: dict[str, list[dict[str, str]]] = {
    "openai_compatible": [],
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
    "anthropic_compatible": [],
}


def get_llm(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
) -> BaseChatModel:
    """Create a chat model for the given provider.

    Callers pass the user's own key. With api_key=None the provider SDK falls
    back to its own environment variable.
    """
    provider = provider.lower()
    resolved_temperature = (
        settings.MODEL_TEMPERATURE if temperature is None else temperature
    )

    if provider in ("openai", "openai_compatible"):
        resolved_model = model or (
            settings.DEFAULT_LLM_MODEL
            if provider == "openai_compatible"
            else settings.OPENAI_MODEL
        )
        resolved_base_url = base_url
        if provider == "openai_compatible" and not resolved_base_url:
            resolved_base_url = settings.DEFAULT_LLM_BASE_URL
        is_deepseek = (
            provider == "openai_compatible"
            and resolved_base_url is not None
            and urlparse(resolved_base_url).hostname == "api.deepseek.com"
        )

        kwargs: dict = {
            "model": resolved_model,
            "temperature": resolved_temperature,
            # Ask for token usage on streamed replies too. LangChain only does
            # this by default for the official OpenAI base URL.
            "stream_usage": True,
        }
        if is_deepseek:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if api_key:
            kwargs["api_key"] = api_key
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        return ChatOpenAI(**kwargs)

    elif provider in ("anthropic", "anthropic_compatible"):
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": model or "claude-sonnet-4-5",
            "temperature": resolved_temperature,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatAnthropic(**kwargs)

    elif provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": model or settings.GEMINI_MODEL,
            "temperature": resolved_temperature,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if api_key:
            kwargs["api_key"] = api_key
        return ChatGoogleGenerativeAI(**kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}")
