"""LLM provider factory — creates chat model instances."""
import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

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


class _OpenAICompatibleChatOpenAI(ChatOpenAI):
    """Adapt structured JSON misplaced in compatible reasoning responses."""

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> Any:
        result = super()._create_chat_result(response, generation_info)
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list):
            return result

        for choice, generation in zip(choices, result.generations, strict=False):
            provider_message = getattr(choice, "message", None)
            if provider_message is None or not hasattr(provider_message, "parsed"):
                continue
            if getattr(provider_message, "parsed", None) is not None:
                continue
            if getattr(provider_message, "content", None) not in (None, ""):
                continue
            if getattr(provider_message, "refusal", None):
                continue

            reasoning_content = getattr(
                provider_message,
                "reasoning_content",
                None,
            )
            if not isinstance(reasoning_content, str):
                continue
            try:
                recovered = json.loads(reasoning_content)
            except json.JSONDecodeError:
                continue
            if not isinstance(recovered, dict):
                continue
            generation.message.additional_kwargs["parsed"] = recovered

        return result


def get_llm(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Create a chat model for the given provider.

    When api_key is provided, it is used directly (per-user keys).
    When api_key is None, falls back to system-wide keys from settings.
    """
    provider = provider.lower()
    resolved_temperature = (
        settings.MODEL_TEMPERATURE if temperature is None else temperature
    )

    if provider in ("openai", "openai_compatible"):
        key = api_key
        if not key and settings.OPENAI_API_KEY:
            key = settings.OPENAI_API_KEY.get_secret_value()

        kwargs: dict = {
            "model": model or settings.OPENAI_MODEL,
            "temperature": resolved_temperature,
        }
        if key:
            kwargs["api_key"] = key
        if base_url:
            kwargs["base_url"] = base_url
        model_class = (
            _OpenAICompatibleChatOpenAI
            if provider == "openai_compatible"
            else ChatOpenAI
        )
        return model_class(**kwargs)

    elif provider in ("anthropic", "anthropic_compatible"):
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": model or "claude-sonnet-4-5",
            "temperature": resolved_temperature,
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
            "temperature": resolved_temperature,
        }
        if key:
            kwargs["api_key"] = key
        return ChatGoogleGenerativeAI(**kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}")
