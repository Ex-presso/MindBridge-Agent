"""Shared test configuration.

Settings load from a developer ``.env``, so without a pinned baseline the same
suite can pass or fail depending on local flags (a real incident: setting
``MEMORY_ENABLED=true`` in ``.env`` broke nine chat-lock tests). Pin every
behavior-switching flag to its declared default here; tests that need another
value set it explicitly with ``monkeypatch``.
"""

import pytest

from config.settings import settings


@pytest.fixture(autouse=True)
def _settings_baseline(monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_ENABLED", False)
