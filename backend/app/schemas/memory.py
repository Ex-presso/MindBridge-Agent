"""Public contracts for inspecting, consenting to, and deleting memory."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MemoryCategory = Literal["semantic", "episodes"]


class MemoryItemResponse(BaseModel):
    """A transparent representation of one LangGraph Store item."""

    namespace: list[str]
    key: str
    value: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    score: float | None = None


class MemoryListResponse(BaseModel):
    memory_enabled: bool
    system_enabled: bool
    items: list[MemoryItemResponse]
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    has_more: bool


class MemoryConsentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_enabled: bool


class MemoryConsentResponse(BaseModel):
    memory_enabled: bool
    system_enabled: bool


class MemoryDeleteResponse(BaseModel):
    memory_enabled: Literal[False] = False
    system_enabled: bool
    deleted_items: int = Field(ge=0)
