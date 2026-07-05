"""configure_tracing() must export the exact env var names LangChain reads."""
from pydantic import SecretStr

from app.main import configure_tracing
from config.settings import settings


def test_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    configure_tracing()
    assert "LANGSMITH_TRACING" not in __import__("os").environ


def test_enabled_exports_env(monkeypatch):
    import os
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", SecretStr("lsv2_secret"))
    monkeypatch.setattr(settings, "LANGSMITH_PROJECT", "proj")
    monkeypatch.setattr(settings, "LANGSMITH_ENDPOINT", "https://example.test")
    for k in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
        monkeypatch.delenv(k, raising=False)

    configure_tracing()

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_secret"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://example.test"
