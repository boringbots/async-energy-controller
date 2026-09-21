"""The thinking label says what was sent, per engine.

Ollama honours only `reasoning_effort`; llama-server honours `enable_thinking`.
Each engine gets exactly the kwargs it needs, and the recorded label is the
lab's own serialization of those kwargs, so a controller row keys with the
lab row that pinned the same thing: `bench reference` with the anchor
(`enable_thinking=false`), Ollama tiers with the lab's Ollama parity wave
(`enable_thinking=false,reasoning_effort=none`).
"""

from __future__ import annotations

from hmasync_controller.bench.quick import (
    THINKING_MODE_OFF,
    THINKING_MODE_OFF_OLLAMA,
    THINKING_MODE_ON,
    thinking_kwargs_for,
    thinking_mode_label,
)


def test_llamacpp_off_matches_the_lab_anchor_label():
    kwargs = thinking_kwargs_for("llama.cpp", thinking=False)
    assert kwargs == {"enable_thinking": False}
    assert thinking_mode_label(kwargs) == THINKING_MODE_OFF == "enable_thinking=false"


def test_ollama_off_sends_reasoning_effort_and_says_so():
    kwargs = thinking_kwargs_for("ollama", thinking=False)
    assert kwargs == {"enable_thinking": False, "reasoning_effort": "none"}
    assert thinking_mode_label(kwargs) == THINKING_MODE_OFF_OLLAMA
    assert THINKING_MODE_OFF_OLLAMA == "enable_thinking=false,reasoning_effort=none"


def test_on_is_one_key_on_every_engine():
    for engine in ("ollama", "llama.cpp", None):
        kwargs = thinking_kwargs_for(engine, thinking=True)
        assert kwargs == {"enable_thinking": True}
        assert thinking_mode_label(kwargs) == THINKING_MODE_ON


def test_unknown_engine_falls_back_to_the_narrow_pin():
    assert thinking_kwargs_for(None, thinking=False) == {"enable_thinking": False}


def test_label_serialization_matches_the_lab_runner():
    # energy-bench orchestrator.runner._serialize_thinking_mode, verbatim rules
    assert thinking_mode_label({"reasoning_effort": "medium"}) == "reasoning_effort=medium"
    assert thinking_mode_label({"a": True, "b": 3}) == "a=true,b=3"
    assert thinking_mode_label({}) is None
    assert thinking_mode_label(None) is None
