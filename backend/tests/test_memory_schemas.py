"""Memory API schemas keep consent strict and Store contents transparent."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.memory import MemoryConsentUpdate, MemoryItemResponse


def test_memory_consent_update_forbids_extra_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MemoryConsentUpdate(memory_enabled=True, delete_existing=True)


def test_memory_item_preserves_transparent_store_fields():
    now = datetime.now(timezone.utc)
    item = MemoryItemResponse(
        namespace=("memory", "user-1", "semantic"),
        key="fact-1",
        value={"kind": "preference", "content": "Prefers concise replies"},
        created_at=now,
        updated_at=now,
        score=None,
    )

    assert item.namespace == ["memory", "user-1", "semantic"]
    assert item.value["content"] == "Prefers concise replies"
    assert item.score is None
