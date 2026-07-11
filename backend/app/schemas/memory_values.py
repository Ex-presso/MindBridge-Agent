"""Strict value contracts for durable memory stored by MindBridge.

These models describe data at the Store trust boundary. They intentionally
forbid unknown fields so prompt-shaped payloads cannot silently become part of
the context rendered for the chat model.
"""

from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from app.schemas.episode_extraction import EpisodeClaim, EpisodeDraft


NonBlankString: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

SemanticKind: TypeAlias = Literal[
    "preference",
    "ongoing_concern",
    "trigger",
    "helpful_strategy",
    "important_person",
    "goal",
]


class SemanticMemoryValue(BaseModel):
    """One attributable, user-approved semantic fact."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: SemanticKind
    content: NonBlankString
    status: Literal["active", "superseded", "deleted"]
    explicit: bool
    confirmed: bool = False
    source_thread_id: NonBlankString | None = None
    source_message_id: NonBlankString | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sensitivity: Literal["low", "medium", "high"] | None = None
    confirmed_at: NonBlankString | None = None
    created_at: NonBlankString | None = None
    updated_at: NonBlankString | None = None
    version: int | None = Field(default=None, ge=1)
    data_epoch: int = Field(default=0, ge=0)


class EpisodeMemoryValue(BaseModel):
    """One validated conversation synopsis used only for similarity recall."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    conversation_id: NonBlankString
    summary: NonBlankString
    claims: tuple[EpisodeClaim, ...] = Field(min_length=1, max_length=12)
    topics: tuple[NonBlankString, ...] = Field(default_factory=tuple, max_length=12)
    status: Literal["active", "superseded", "deleted"]
    crisis: bool
    target_revision: int = Field(ge=1)
    data_epoch: int = Field(default=0, ge=0)
    started_at: NonBlankString | None = None
    updated_at: NonBlankString | None = None
    message_count: int | None = Field(default=None, ge=0)

    @field_validator("claims", "topics", mode="before")
    @classmethod
    def sequences_must_be_copied_to_immutable_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("topics")
    @classmethod
    def topics_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject ambiguous duplicate topic lists at the persistence boundary."""
        if len(set(value)) != len(value):
            raise ValueError("topics must be unique")
        return value

    @model_validator(mode="after")
    def summary_must_match_claims(self) -> "EpisodeMemoryValue":
        draft = EpisodeDraft(claims=self.claims, topics=self.topics)
        if self.summary != draft.summary:
            raise ValueError("episode summary must be derived from claims")
        return self
