"""Public contracts for inspecting, consenting to, and deleting memory."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MemoryItemResponse(BaseModel):
    id: str
    text: str
    # The conversation the memory was extracted from.
    conversation_id: str | None
    created_at: str | None


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
