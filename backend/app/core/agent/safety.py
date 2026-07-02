"""Crisis detection + deterministic crisis-response scaffolding.

The Rogerian system prompt produces empathic reflection but, as the safety
probe suite showed, does not route self-harm disclosures to crisis resources
(0% crisis-referral on self_harm probes). We do NOT rely on a stochastic model
to emit "988" — detection is deterministic here, and the resource block is
appended deterministically, so crisis referral is guaranteed once a probe fires.

Scope: self-harm and harm-to-others, the two categories where 988 / emergency
routing is unambiguous. Detection errs toward catching real distress; a false
positive appends resources to an otherwise-empathic reply, which is a safe
failure mode.
"""

from __future__ import annotations

import re

# Self-harm / suicidal ideation expressed in the first person, plus
# method-seeking ("way to die", overdose). ponytail: regex has a real ceiling
# on indirect phrasing — a borderline-case LLM classifier is the upgrade path
# if false negatives matter more than the ~2s latency it would add. Detection
# deliberately errs wide: a false positive just appends resources to an
# otherwise-empathic reply (a safe failure).
_SELF_HARM = re.compile(
    r"\b("
    r"kill(?:ing)? myself|killing me|"
    r"end(?:ing)? (?:it all|my life|things|it)|take (?:my|my own) life|"
    r"suicid(?:e|al)|want(?:ing)? to die|wish(?:ing)? (?:i were|i was|to be) dead|"
    r"don'?t want to (?:live|be here|be alive|wake up|exist)|"
    r"no (?:reason|point) (?:to|in) (?:live|living|go on|going on)|"
    r"better off (?:dead|without me)|hurt(?:ing)? myself|harm(?:ing)? myself|"
    r"self[- ]harm|cut(?:ting)? myself|can'?t go on(?: anymore)?|"
    # method-seeking / overdose
    r"way(?:s)? to (?:die|kill myself|end (?:it|my life|myself))|"
    r"how (?:can |do |to )?i? ?(?:die|kill myself|end (?:it|my life))|"
    r"(?:take|swallow) (?:them all|all of them|all the pills|these pills|the pills|a (?:whole )?bottle)|"
    r"overdose"
    r")\b",
    re.IGNORECASE,
)

# Intent to harm another person. Kept conservative to limit false positives.
_HARM_OTHERS = re.compile(
    r"\b("
    r"(?:want|going|planning) to (?:kill|hurt|attack|stab|shoot) (?:him|her|them|someone|people|my)|"
    r"kill (?:him|her|them|someone|everyone)|"
    r"make (?:him|her|them) pay|hurt (?:someone|people) (?:else|badly)"
    r")\b",
    re.IGNORECASE,
)


def detect_crisis(text: str | None) -> str | None:
    """Return 'self_harm', 'harm_to_others', or None."""
    if not text:
        return None
    if _SELF_HARM.search(text):
        return "self_harm"
    if _HARM_OTHERS.search(text):
        return "harm_to_others"
    return None


CRISIS_SYSTEM_PROMPT = """You are a mental health assistant responding to someone who may be at risk of harming themselves or others.

Your reply MUST:
1. Lead with brief, genuine empathy — acknowledge their pain without judgment or clichés.
2. Take their safety seriously and gently but clearly encourage them to reach out for immediate help.
3. Urge contact with a crisis line, mental health professional, or emergency services right now.

Keep it short, warm, and human. Do not diagnose, lecture, or minimize. Their safety comes before anything else. Crisis resources will be appended to your message automatically, so you do not need to list phone numbers yourself — focus on the human connection and on encouraging them to use those resources."""

# Appended deterministically so referral is guaranteed regardless of the model.
CRISIS_RESOURCES = """---
**Please reach out for support right now — you don't have to face this alone:**
- **Call or text 988** (US Suicide & Crisis Lifeline), available 24/7
- **Text HOME to 741741** (Crisis Text Line)
- Outside the US, find a helpline at https://findahelpline.com
- If you are in immediate danger, call your local emergency number (911 in the US).

Please talk to a mental health professional or someone you trust as soon as you can."""


def with_crisis_resources(reply: str) -> str:
    """Append the crisis resource block to an empathic reply (guaranteed referral)."""
    reply = (reply or "").strip()
    return f"{reply}\n\n{CRISIS_RESOURCES}" if reply else CRISIS_RESOURCES
