"""Deterministic promotion of explicit user statements into semantic memory."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import unicodedata
import uuid
from typing import Any

from app.schemas.episode_extraction import (
    EpisodeClaim,
    EpisodeDraft,
    EpisodeSourceMessage,
)
from app.schemas.memory_values import SemanticKind, SemanticMemoryValue
from app.services.episode_filters import filter_draft
from app.services.memory_service import memory_namespace


_EXPLICIT_PATTERNS: tuple[tuple[SemanticKind, re.Pattern[str]], ...] = (
    (
        "preference",
        re.compile(
            r"\b(?:i\s+(?:strongly\s+|really\s+)?prefer\b|"
            r"my\s+(?:communication\s+)?preference\s+is\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "goal",
        re.compile(
            r"\b(?:my\s+(?:current\s+)?goal\s+is\b|"
            r"one\s+of\s+my\s+goals\s+is\b|"
            r"i(?:'m|\s+am)\s+(?:trying|working|aiming|planning)\s+to\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "helpful_strategy",
        re.compile(
            r"\b(?:helps?\s+me|calms?\s+me|grounds?\s+me|"
            r"works?\s+for\s+me|makes?\s+me\s+feel\s+"
            r"(?:calm|calmer|steady|better|safe))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "important_person",
        re.compile(
            r"(?i:\bmy\s+(?:partner|spouse|wife|husband|girlfriend|boyfriend|"
            r"sister|brother|mother|father|mom|mum|dad|daughter|son|"
            r"best\s+friend|friend|roommate|therapist|counselor|mentor|"
            r"colleague|coworker)(?:,\s*)?(?:\s+named)?\s+)[A-Z][\w'’-]*\b",
        ),
    ),
    (
        "important_person",
        re.compile(
            r"\bmy\s+(?:partner|spouse|wife|husband|girlfriend|boyfriend|"
            r"sister|brother|mother|father|mom|mum|dad|daughter|son|"
            r"best\s+friend|friend|roommate|therapist|counselor|mentor|"
            r"colleague|coworker)\b.{0,100}\b(?:supports?\s+me|"
            r"is\s+important\s+to\s+me|i\s+trust|i\s+rely\s+on|"
            r"i(?:'m|\s+am)\s+close\s+to)\b",
            re.IGNORECASE,
        ),
    ),
)


class SemanticMemoryConflictError(RuntimeError):
    """An existing item cannot be safely reconciled with this write."""


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    """One explicit, source-attributed fact selected by fixed rules."""

    kind: SemanticKind
    content: str
    source_message_id: str


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _candidate_key(candidate: SemanticCandidate) -> str:
    identity = f"{candidate.kind}:{_normalized(candidate.content)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mindbridge:semantic:{identity}"))


def derive_semantic_candidates(
    source_messages: Sequence[EpisodeSourceMessage],
    *,
    max_content_chars: int,
) -> tuple[SemanticCandidate, ...]:
    """Classify safe, explicit source spans; never infer a user profile."""
    if max_content_chars < 1:
        raise ValueError("max_content_chars must be at least 1")

    candidates: list[SemanticCandidate] = []
    seen: set[tuple[SemanticKind, str]] = set()
    for message in source_messages:
        if message.role != "user":
            continue
        statements = re.split(r"(?<=[.!?])\s+|[\r\n]+", message.content)
        for statement in statements:
            content = statement.strip()
            if not content or len(content) > max_content_chars:
                continue
            candidate_draft = EpisodeDraft(
                claims=(
                    EpisodeClaim(
                        claim=content,
                        evidence_message_id=message.id,
                        evidence_quote=content,
                    ),
                ),
            )
            if filter_draft(candidate_draft) is not None:
                continue
            for kind, pattern in _EXPLICIT_PATTERNS:
                if pattern.search(content) is None:
                    continue
                identity = (kind, _normalized(content))
                if identity in seen:
                    continue
                seen.add(identity)
                candidates.append(
                    SemanticCandidate(
                        kind=kind,
                        content=content,
                        source_message_id=message.id,
                    )
                )
    return tuple(candidates)


def _parse_existing(item: Any) -> SemanticMemoryValue | None:
    if item is None:
        return None
    value = getattr(item, "value", None)
    if not isinstance(value, Mapping):
        raise SemanticMemoryConflictError("semantic_value_invalid")
    try:
        return SemanticMemoryValue.model_validate(dict(value))
    except Exception as exc:
        raise SemanticMemoryConflictError("semantic_value_invalid") from exc


async def write_semantic_candidates(
    store: Any,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    candidates: Sequence[SemanticCandidate],
    data_epoch: int,
    timeout_seconds: float,
) -> int:
    """Idempotently persist explicit facts before the owning episode write."""
    namespace = memory_namespace(user_id, "semantic")
    now = datetime.now(timezone.utc).isoformat()
    writes = 0

    for candidate in candidates:
        key = _candidate_key(candidate)
        async with asyncio.timeout(timeout_seconds):
            existing_item = await store.aget(
                namespace,
                key,
                refresh_ttl=False,
            )
        existing = _parse_existing(existing_item)
        if existing is not None and existing.data_epoch > data_epoch:
            raise SemanticMemoryConflictError("semantic_epoch_ahead")
        if existing is not None and existing.data_epoch == data_epoch:
            if (
                existing.kind != candidate.kind
                or _normalized(existing.content) != _normalized(candidate.content)
            ):
                raise SemanticMemoryConflictError("semantic_identity_mismatch")
            if existing.status != "active":
                continue
            if (
                existing.source_thread_id == str(conversation_id)
                and existing.source_message_id == candidate.source_message_id
            ):
                continue

        value = SemanticMemoryValue(
            kind=candidate.kind,
            content=candidate.content,
            status="active",
            explicit=True,
            confirmed=(existing.confirmed if existing is not None else False),
            source_thread_id=str(conversation_id),
            source_message_id=candidate.source_message_id,
            confidence=(existing.confidence if existing is not None else None),
            sensitivity=(existing.sensitivity if existing is not None else None),
            confirmed_at=(existing.confirmed_at if existing is not None else None),
            created_at=(existing.created_at if existing is not None else now),
            updated_at=now,
            version=((existing.version or 1) + 1 if existing is not None else 1),
            data_epoch=data_epoch,
        )
        async with asyncio.timeout(timeout_seconds):
            await store.aput(
                namespace,
                key,
                value.model_dump(mode="json"),
                index=False,
            )
        writes += 1

    return writes
