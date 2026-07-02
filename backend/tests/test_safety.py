"""Crisis detection + deterministic resource appending."""
import pytest

from app.core.agent.safety import (
    CRISIS_RESOURCES,
    detect_crisis,
    with_crisis_resources,
)


@pytest.mark.parametrize(
    "text",
    [
        "I've been thinking about ending it all.",
        "Sometimes I just want to die.",
        "I don't want to be here anymore.",
        "I've been cutting myself again.",
        "There's no reason to go on.",
        "I feel like everyone would be better off without me.",
        # indirect / method-seeking phrasing (previously missed)
        "I have pills in front of me and I want to take them all.",
        "What's the most painless way to die?",
        # ideation-gated self-injury
        "I want to hurt myself.",
        "I've been hurting myself and I can't stop.",
    ],
)
def test_detects_self_harm(text):
    assert detect_crisis(text) == "self_harm"


@pytest.mark.parametrize(
    "text",
    [
        "I'm so anxious about my exam tomorrow.",
        "Work has been really stressful lately.",
        "I had a fight with my partner and feel sad.",
        "",
        None,
        # everyday idioms that must NOT trigger crisis routing
        "My back is killing me after the gym.",
        "This deadline is killing me.",
        "I want to end things with my boyfriend.",
        "I decided to end it with her last week.",
        "I hurt myself at soccer practice yesterday.",
        "This game makes me want to kill everyone lol",
    ],
)
def test_no_false_positive_on_ordinary_distress(text):
    assert detect_crisis(text) is None


def test_detects_harm_to_others():
    assert detect_crisis("I want to hurt someone badly.") == "harm_to_others"


def test_resources_block_always_appended():
    out = with_crisis_resources("It sounds like you're in a lot of pain.")
    assert "988" in out
    assert out.endswith(CRISIS_RESOURCES)


def test_resources_present_even_for_empty_reply():
    assert "988" in with_crisis_resources("")
