"""Strict contracts at the LLM boundary for episodic-memory Extraction."""

from typing import Annotated, Literal
import unicodedata
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)


_MAX_SUMMARY_CHARS = 800

ClaimText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]
EvidenceQuote = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]
TopicText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]
MessageContent = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
]


def _contains_control_characters(
    value: str,
    *,
    allow_text_whitespace: bool = False,
) -> bool:
    allowed = {"\n", "\r", "\t"} if allow_text_whitespace else set()
    return any(
        character not in allowed
        and unicodedata.category(character).startswith("C")
        for character in value
    )


def _validate_canonical_uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("message ID must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError("message ID must be a canonical UUID")
    return value


def _summary_from_claims(claims: tuple["EpisodeClaim", ...]) -> str:
    """Return the only supported synopsis representation."""
    return " ".join(claim.claim for claim in claims)


class EpisodeSourceMessage(BaseModel):
    """One relational source message supplied to the extractor."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(repr=False)
    role: Literal["user", "assistant"]
    content: MessageContent = Field(repr=False)

    @field_validator("id")
    @classmethod
    def id_must_be_canonical_uuid(cls, value: str) -> str:
        return _validate_canonical_uuid(value)

    @field_validator("content")
    @classmethod
    def content_must_not_contain_controls(cls, value: str) -> str:
        if _contains_control_characters(value, allow_text_whitespace=True):
            raise ValueError("message content contains control characters")
        return value


class EpisodeClaim(BaseModel):
    """One extractive claim with a user-authored, directly checkable citation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    claim: ClaimText = Field(repr=False)
    evidence_message_id: str = Field(repr=False)
    evidence_quote: EvidenceQuote = Field(repr=False)

    @field_validator("evidence_message_id")
    @classmethod
    def evidence_id_must_be_canonical_uuid(cls, value: str) -> str:
        return _validate_canonical_uuid(value)

    @field_validator("claim", "evidence_quote")
    @classmethod
    def text_must_not_contain_controls(cls, value: str) -> str:
        if _contains_control_characters(value, allow_text_whitespace=True):
            raise ValueError("episode claim contains control characters")
        return value


class EpisodeDraft(BaseModel):
    """A claim-grounded synopsis candidate; never written unvalidated."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    claims: tuple[EpisodeClaim, ...] = Field(
        min_length=1,
        max_length=12,
        repr=False,
    )
    topics: tuple[TopicText, ...] = Field(
        default_factory=tuple,
        max_length=12,
        repr=False,
    )

    @field_validator("claims", mode="before")
    @classmethod
    def claims_must_be_revalidated_and_copied(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(
            EpisodeClaim.model_validate(
                claim.model_dump(mode="python", warnings="none")
                if isinstance(claim, EpisodeClaim)
                else claim
            )
            for claim in value
        )

    @field_validator("topics", mode="before")
    @classmethod
    def topics_must_be_copied_to_an_immutable_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("topics")
    @classmethod
    def topics_must_be_safe_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(_contains_control_characters(value) for value in values):
            raise ValueError("topic contains control characters")
        normalized = [unicodedata.normalize("NFKC", value).casefold() for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("topics must be unique")
        return values

    @model_validator(mode="after")
    def claims_must_be_unique_and_fit_summary_budget(self) -> "EpisodeDraft":
        if any(
            claim.claim not in claim.evidence_quote
            for claim in self.claims
        ):
            raise ValueError("episode claim must be an exact span of its evidence quote")
        identities = [
            (
                claim.evidence_message_id,
                claim.evidence_quote,
                unicodedata.normalize("NFKC", claim.claim).casefold(),
            )
            for claim in self.claims
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("episode claims must be unique")
        if len(_summary_from_claims(self.claims)) > _MAX_SUMMARY_CHARS:
            raise ValueError("episode summary exceeds 800 characters")
        for topic in self.topics:
            if not any(topic in claim.claim for claim in self.claims):
                raise ValueError("episode topic must be an exact span of a claim")
        return self

    @computed_field(repr=False, return_type=str)
    @property
    def summary(self) -> str:
        """Build the synopsis deterministically; the model cannot provide it."""
        return _summary_from_claims(self.claims)
