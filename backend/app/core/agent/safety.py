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
import unicodedata


# Persisted alongside historical conversation reviews. Increment whenever the
# deterministic patterns change so previously clean conversations fail closed
# until their relational user-message history is reviewed again.
CRISIS_DETECTOR_VERSION = 3

# Self-harm / suicidal ideation in the first person, plus method-seeking
# ("way to die", overdose). Bare injury phrases ("hurt myself", "killing me",
# "end it/things") are deliberately NOT matched — everyday idioms dominate them
# (sports injuries, deadlines, breakups); self-injury requires an ideation verb
# ("want to / thinking about / been ... hurting myself").
# ponytail: regex has a real ceiling on indirect phrasing — a borderline-case
# LLM classifier is the upgrade path if false negatives matter more than the
# ~2s latency it would add.
_SELF_HARM = re.compile(
    r"\b("
    r"kill(?:ing)? myself|"
    r"end(?:ing)? (?:it all|my life)|take (?:my|my own) life|"
    r"suicid(?:e|al)|want(?:ing)? to die|wish(?:ing)? (?:i were|i was|to be) dead|"
    r"(?:don'?t|do not) want to (?:live|be here|be alive|wake up|exist|go on|keep going)|"
    r"no (?:reason|point) (?:to|in) (?:live|living|go on|going on)|"
    r"better off (?:dead|without me)|"
    # self-injury gated on an ideation verb; bare "hurt myself" is usually an accident report
    r"(?:want(?:ed|ing)? to|plan(?:ned|ning)? to|planning on|"
    r"intend(?:ed|ing)? to|decid(?:ed|ing) to|consider(?:ed|ing)?|"
    r"thinking (?:about|of)|thoughts? of|urge to|going to|will|may|might|been) "
    r"(?:hurt|harm)(?:ing)? myself|"
    r"self[- ]harm|cut(?:ting)? myself|can'?t go on(?: anymore)?|"
    # method-seeking / overdose
    r"way(?:s)? to (?:die|kill myself|end (?:it|my life|myself))|"
    r"how (?:can |do |to )?i? ?(?:die|kill myself|end (?:it|my life))|"
    r"(?:take|swallow) (?:them all|all of them|all the pills|these pills|the pills|a (?:whole )?bottle)|"
    r"overdose"
    r")\b",
    re.IGNORECASE,
)

# Chinese does not have whitespace-delimited words, so it needs a separate
# detector instead of being folded into the English ``\b`` expression above.
# Keep desire/intent attached to a self-harm action: matching a bare ``死``
# would route extremely common intensifiers such as ``笑死我了`` and ``累死了``.
_SELF_HARM_ZH = re.compile(
    r"("
    # Explicit intent, including a short time phrase between intent and action.
    r"(?:我|本人|用户)?(?:已经|正在|真的|现在)?"
    r"(?:计划|打算|准备|决定|考虑|想|要|会|将)"
    r"[^。！？!?\n]{0,12}(?:自杀|自殺|轻生|輕生|寻死|尋死|自残|自殘)|"
    # Direct wish to die. The exclusions cover ``想死你了`` and
    # ``死了这条心``, neither of which expresses suicidal intent.
    r"(?:我|本人)?(?:真的|现在|好|很|太|一直)?想(?:去)?死"
    r"(?!(?:你|他|她|大家|了?这条心))|"
    r"(?:我|本人)(?:真的|现在|已经)?(?:准备|打算|决定|想|要)去死|"
    r"(?:我|本人)(?:已经|正在|真的|现在|可能)?"
    r"(?:计划|計劃|打算|准备|準備|决定|決定|考虑|考慮|想|要|会|會|将|將|可能)"
    r"[^。！？!?\n]{0,8}(?:伤害|傷害|弄伤|弄傷|残害|殘害)自己|"
    r"(?:我|本人)?(?:真的|现在|已经|实在)?不想(?:再)?"
    r"(?:活下去|(?:继续|繼續)活[着著]?|活[着著]|活(?:了|下去)|醒来|醒來|存在)(?:了|下去)?|"
    # Explicitly ending one's own life or naming a self-harm method.
    r"(?:结束|結束|终结|終結|了结|了結)(?:我|我的|自己|自己的)(?:这条|這條)?生命|"
    r"(?:我|本人)(?:真的|现在|已经)?(?:准备|打算|决定|想|要)"
    r"(?:结束|結束|终结|終結|了结|了結)(?:我的|自己的|这条|這條)?生命"
    r"(?=$|[。！？!?，,\s])|"
    r"(?:割腕|吞药|吞藥|服毒|跳楼|跳樓|上吊|卧轨|臥軌)(?:自杀|自殺)?|"
    # Common unambiguous ideation descriptions.
    r"(?:自杀|自殺|轻生|輕生|自残|自殘)(?:的)?(?:念头|念頭|想法|计划|計劃|冲动|衝動)|"
    r"(?:活着|活著|人生)(?:没有|沒有|没|沒)(?:意义|意義|意思)|"
    r"(?:没有|沒有|没|沒)(?:(?:继续|繼續)活[着著]?|活下去)的?(?:意义|意義|理由)|"
    r"不想再?醒(?:来|來)|一了百了"
    r")"
)

# Intent to harm another person. Every alternative requires intent phrasing —
# bare "kill him/everyone" was dominated by gaming/venting idioms.
_HARM_OTHERS = re.compile(
    r"\b("
    r"(?:want|going|planning) to (?:kill|hurt|attack|stab|shoot) (?:him|her|them|someone|people|my)|"
    r"i'?ll (?:kill|hurt|stab|shoot) (?:him|her|them|someone)|"
    r"(?:kill|shoot|hurt) everyone (?:at|in) (?:my |the )?(?:school|work|office|class)|"
    r"make (?:him|her|them) pay|hurt (?:someone|people) (?:else|badly)"
    r")\b",
    re.IGNORECASE,
)

# Require both an intent marker and a human target. Besides reducing ordinary
# venting false positives, the explicit target list keeps technical operations
# such as ``杀死进程`` out of crisis routing.
_HARM_OTHERS_ZH = re.compile(
    r"(?:我|本人)?(?:已经|正在|真的|现在)?"
    r"(?:计划|打算|准备|决定|想|要|会|将|一定要)"
    r"[^。！？!?\n]{0,12}"
    r"(?:杀(?:了|死|掉)?|弄死|伤害|攻击|袭击|刺伤|捅死|砍死|枪杀|打死)"
    r"(?:那个|这个|所有|全部|我的)?"
    r"(?:他|她|他们|她们|别人|某人|同事|同学|老师|老板|主管|经理|"
    r"家人|父母|孩子|邻居|室友|前任|男朋友|女朋友|人|大家|所有人)"
)


def detect_crisis(text: str | None) -> str | None:
    """Return 'self_harm', 'harm_to_others', or None."""
    if not text:
        return None
    normalized = unicodedata.normalize("NFKC", text)
    if _SELF_HARM.search(normalized) or _SELF_HARM_ZH.search(normalized):
        return "self_harm"
    if _HARM_OTHERS.search(normalized) or _HARM_OTHERS_ZH.search(normalized):
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
