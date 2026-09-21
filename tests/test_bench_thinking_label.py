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


# --- effort-ladder models have no "off" ---------------------------------
#
# Measured against Ollama 0.34.2 on gpt-oss:20b: `reasoning_effort: none` is
# not in the vocabulary, so it falls back to the default and is identical to
# `medium`. The generic thinking-off kwargs are therefore a NO-OP there, and a
# row carrying them claims thinking-off while having reasoned at medium.
# energy-bench reached the same shape: its gpt-oss configs are effort-low and
# effort-high rungs ("medium is the reference row"), and it ships no
# thinkoff-gpt-oss config at all.


def test_gpt_oss_maps_the_axis_onto_the_effort_ladder():
    from hmasync_controller.bench.quick import thinking_kwargs_for

    off = thinking_kwargs_for("ollama", False, "gpt-oss:20b")
    on = thinking_kwargs_for("ollama", True, "gpt-oss:20b")
    assert off == {"reasoning_effort": "low"}
    assert on == {"reasoning_effort": "high"}
    # The no-op value must never be sent to a ladder model.
    assert off.get("reasoning_effort") != "none"
    assert "enable_thinking" not in off


def test_gpt_oss_rows_are_labelled_by_their_rung():
    """So they key with the lab's axis-*-effort-* rows instead of pooling
    with genuinely-off rows from models that have an off."""
    from hmasync_controller.bench.quick import thinking_kwargs_for, thinking_mode_label

    assert (
        thinking_mode_label(thinking_kwargs_for("ollama", False, "gpt-oss:20b"))
        == "reasoning_effort=low"
    )
    assert (
        thinking_mode_label(thinking_kwargs_for("ollama", True, "gpt-oss:20b"))
        == "reasoning_effort=high"
    )


def test_switch_models_are_untouched_by_the_ladder_rule():
    from hmasync_controller.bench.quick import (
        THINKING_OFF_CHAT_TEMPLATE_KWARGS,
        thinking_kwargs_for,
    )

    assert (
        thinking_kwargs_for("ollama", False, "qwen3.5:9b-q4_K_M")
        == THINKING_OFF_CHAT_TEMPLATE_KWARGS
    )
    # ... including when no model is named at all.
    assert thinking_kwargs_for("ollama", False) == THINKING_OFF_CHAT_TEMPLATE_KWARGS


def test_the_ladder_matches_on_family_not_exact_tag():
    from hmasync_controller.bench.quick import thinking_kwargs_for

    for tag in ("gpt-oss:20b", "gpt-oss:120b", "gpt-oss:20b-mxfp4"):
        assert thinking_kwargs_for("ollama", False, tag) == {"reasoning_effort": "low"}, tag
