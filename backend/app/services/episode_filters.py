"""Deterministic content filters for episodic-memory drafts.

The product's supported language is English, so these vocabularies are
English-only; text in other languages is outside the supported scope and is
not filtered here. They are matching patterns, not user-visible strings.
Filtering runs on NFKC-casefolded text after schema validation and must stay
deterministic: a draft that trips a filter is dropped, never rewritten.
"""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Literal
import unicodedata

from app.schemas.episode_extraction import EpisodeDraft


EpisodeFilterStatus = Literal["filtered_diagnosis", "filtered_instruction"]

_DIAGNOSTIC_TERM_PATTERN = re.compile(
    r"""
    \b(?:
        generalized\s+anxiety(?:\s+disorder)?
        | anxiety\s+disorder
        | gad
        | borderline\s+personality\s+disorder
        | bpd
        | obsessive[\s\-‐-―]+compulsive(?:\s+disorder)?
        | ocd
        | psychotic(?:\s+(?:disorder|episode|symptoms?))?
        | psychosis
        | post[\s\-‐-―]+traumatic\s+stress\s+disorder
        | ptsd
        | attention[\s\-‐-―]+deficit(?:/hyperactivity|\s+hyperactivity)?\s+disorder
        | adhd
        | major\s+depress(?:ion|ive\s+disorder)
        | depression
        | depressive\s+disorder
        | mdd
        | bipolar(?:\s+disorder)?
        | schizophren(?:ia|ic|iform)
        | schizoaffective(?:\s+disorder)?
        | autism(?:\s+spectrum\s+disorder)?
        | autistic
        | asd
        | panic\s+disorder
        | personality\s+disorder
        | eating\s+disorder
        | dissociative\s+disorder
        | anorexia(?:\s+nervosa)?
        | bulimia(?:\s+nervosa)?
        | dysthymia
        | agoraphobia
        | hypomania
        | tourette'?s?(?:\s+syndrome)?
        | tic\s+disorder
        | insomnia
        | hypersomnia
        | narcolepsy
        | gender\s+dysphoria
        | body\s+dysmorph(?:ia|ic\s+disorder)
        | depersonalization
        | derealization
        | cyclothymia
    )\b
    """,
    re.VERBOSE,
)

_DIAGNOSIS_ASSERTION_PATTERN = re.compile(
    r"""
    \b(?:diagnos(?:ed|is)|meets?\s+(?:the\s+)?criteria\s+for
    | clinically\s+consistent\s+with|symptoms?\s+(?:suggest|indicate))\b
    """,
    re.VERBOSE,
)

_GENERIC_DIAGNOSTIC_LABEL_PATTERN = re.compile(
    r"\b[a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*){0,4}\s+(?:disorder|syndrome)\b",
    re.IGNORECASE,
)
_SELF_REPORTED_CONDITION_PATTERN = re.compile(
    r"\b(?:i|the user|they|he|she)\s+(?:have|has|had|"
    r"suffer(?:s|ed|ing)?\s+from|live(?:s|d|ing)?\s+with)\b",
    re.IGNORECASE,
)

_INSTRUCTION_PAYLOAD_PATTERN = re.compile(
    r"(?:ignore (?:all |the )?(?:previous|prior) instructions|follow (?:these|the following) instructions|(?:system|developer) (?:prompt|message|instructions?)|tool call|function call|act as (?:a |an )|call (?:the )?(?:tool|function)|execute (?:this |the )?(?:command|code)|when (?:this (?:memory|text) is )?(?:recalled|remembered)|at recall time|remember to (?:ignore|reveal|disclose|send|call|execute)|(?:reveal|disclose|expose|leak|send).{0,40}(?:hidden (?:rules|prompt|instructions?)|secrets?|credentials?|api keys?|private data))",
    re.IGNORECASE,
)

# Persistent prompt injection is usually phrased as a future memory trigger plus
# an action, not as the literal phrase "ignore previous instructions". Match the
# two signals separately so paraphrases fail closed without treating every
# ordinary mention of remembering as an instruction.
_FUTURE_CONDITION_PATTERN = re.compile(
    r"\b(?:if|when|whenever|each time|every time|next time|later|future|on|upon)\b",
    re.IGNORECASE,
)
_RECALL_PATTERN = re.compile(
    r"\b(?:memory|memories|recall(?:ed|ing)?|remember(?:ed|ing)?|"
    r"comes? up|comes? to mind|think(?:ing)? of)\b",
    re.IGNORECASE,
)
_DIRECTIVE_ACTION_PATTERN = re.compile(
    r"\b(?:email|send|forward|upload|post|share|reveal|disclose|expose|"
    r"leak|tell|show|use|invoke|call|run|execute|follow|ignore|print|output|"
    r"repeat|give|write|return|respond|include|remind)\b",
    re.IGNORECASE,
)
_SENSITIVE_PAYLOAD_PATTERN = re.compile(
    r"\b(?:passwords?|secrets?|credentials?|api[ -]?keys?|access[ -]?tokens?|"
    r"private data|personal data|hidden rules?|internal rules?|system prompts?|"
    r"developer messages?|conversation history|transcripts?)\b",
    re.IGNORECASE,
)
# The action verb only reads as a command aimed at the assistant when the text
# before it addresses "you" or opens a clause imperatively.
_SECOND_PERSON_PATTERN = re.compile(r"\byou\b", re.IGNORECASE)
_CLAUSE_INITIAL_IMPERATIVE_PATTERN = re.compile(
    r"(?:^|[,;:])\s*(?:please\s+)?$",
    re.IGNORECASE,
)


def _normalized_for_filter(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _contains_persistent_instruction(value: str) -> bool:
    if not (
        _FUTURE_CONDITION_PATTERN.search(value)
        and _RECALL_PATTERN.search(value)
    ):
        return False
    # Sensitive data paired with a future recall trigger is unsafe regardless
    # of the verb used. Keeping this ahead of the action vocabulary prevents
    # paraphrases such as "publish", "broadcast", or "copy" from bypassing
    # the guard while ordinary first-person recall statements remain allowed.
    if _SENSITIVE_PAYLOAD_PATTERN.search(value):
        return True
    action = _DIRECTIVE_ACTION_PATTERN.search(value)
    if action is None:
        return False
    prefix = value[: action.start()]
    return bool(
        _SECOND_PERSON_PATTERN.search(prefix)
        or _CLAUSE_INITIAL_IMPERATIVE_PATTERN.search(prefix)
    )


def filter_values(values: Sequence[str]) -> EpisodeFilterStatus | None:
    """Return the filter verdict for raw strings, or None to accept them.

    Callers that hold a standalone statement rather than a draft use this
    directly: routing a single string through ``EpisodeDraft`` would impose
    that schema's unrelated claim-length limit on them.
    """
    normalized = [_normalized_for_filter(value) for value in values]
    if any(
        _DIAGNOSTIC_TERM_PATTERN.search(value)
        or _DIAGNOSIS_ASSERTION_PATTERN.search(value)
        or (
            _GENERIC_DIAGNOSTIC_LABEL_PATTERN.search(value)
            and _SELF_REPORTED_CONDITION_PATTERN.search(value)
        )
        for value in normalized
    ):
        return "filtered_diagnosis"
    if any(_INSTRUCTION_PAYLOAD_PATTERN.search(value) for value in normalized):
        return "filtered_instruction"
    if any(_contains_persistent_instruction(value) for value in normalized):
        return "filtered_instruction"
    return None


def filter_draft(draft: EpisodeDraft) -> EpisodeFilterStatus | None:
    """Return the filter verdict for a schema-valid draft, or None to accept."""
    values = [
        value
        for claim in draft.claims
        for value in (claim.claim, claim.evidence_quote)
    ]
    values.extend(draft.topics)
    return filter_values(values)
