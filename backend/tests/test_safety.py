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
        "I do not want to go on anymore.",
        "Ｉ ｗａｎｔ ｔｏ ｄｉｅ.",
        "I've been cutting myself again.",
        "There's no reason to go on.",
        "I feel like everyone would be better off without me.",
        # indirect / method-seeking phrasing (previously missed)
        "I have pills in front of me and I want to take them all.",
        "What's the most painless way to die?",
        # ideation-gated self-injury
        "I want to hurt myself.",
        "I plan to hurt myself.",
        "I am planning to hurt myself.",
        "I am planning on hurting myself.",
        "I intend to harm myself.",
        "I decided to hurt myself.",
        "I will hurt myself tonight.",
        "I may harm myself.",
        "I am considering harming myself.",
        "I've been hurting myself and I can't stop.",
    ],
)
def test_detects_self_harm(text):
    assert detect_crisis(text) == "self_harm"


@pytest.mark.parametrize(
    "text",
    [
        "用户已经计划今晚自杀",
        "我想死",
        "不想活了",
        "结束自己的生命",
        "割腕吞药",
        "我不想活了。",
        "我想结束自己的生命。",
        "我要去死。",
        "我准备结束生命。",
        "我打算伤害自己。",
        "我会伤害自己。",
        "我想結束自己的生命。",
        "我不想活著了。",
        "我想自殘。",
        "我准备割腕吞药。",
        "最近总有自杀的念头。",
        "活着没有意义。",
        "我不想再醒来。",
    ],
)
def test_detects_chinese_self_harm(text):
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
        "I plan to challenge myself at the gym.",
        "This game makes me want to kill everyone lol",
    ],
)
def test_no_false_positive_on_ordinary_distress(text):
    assert detect_crisis(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "笑死我了。",
        "今天真的累死了。",
        "杀死进程后重新启动服务。",
        "这个项目快把我逼死了。",
        "我不想活得这么累。",
        "我想死你了，什么时候回来？",
        "我决定死了这条心。",
    ],
)
def test_no_false_positive_on_chinese_idioms_or_technical_language(text):
    assert detect_crisis(text) is None


def test_detects_harm_to_others():
    assert detect_crisis("I want to hurt someone badly.") == "harm_to_others"


@pytest.mark.parametrize(
    "text",
    [
        "我要杀了他",
        "计划伤害同事",
        "我计划伤害同事。",
        "我已经决定今晚攻击那个邻居。",
    ],
)
def test_detects_chinese_harm_to_others(text):
    assert detect_crisis(text) == "harm_to_others"


def test_resources_block_always_appended():
    out = with_crisis_resources("It sounds like you're in a lot of pain.")
    assert "988" in out
    assert out.endswith(CRISIS_RESOURCES)


def test_resources_present_even_for_empty_reply():
    assert "988" in with_crisis_resources("")
