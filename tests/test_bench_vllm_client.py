"""Answer-text extraction from an OpenAI-compatible chat response.

A harmony/reasoning model served by vLLM returns TWO fields on the assistant
message: `content` (the final answer) and `reasoning` / `reasoning_content`
(the chain of thought). Reading `content` alone is correct for an ordinary
model and silently catastrophic for a reasoning one.

Measured on gpt-oss-20b, 2026-08-26: gpqa_diamond items generated 87-512
completion tokens and returned an EMPTY content string, because the whole
budget went to the reasoning channel and no final channel was ever emitted.
Every such item scored incorrect, dragging the model to 0.15 on a 4-choice
task -- BELOW the 0.25 chance floor, which is the signature of a parsing
failure rather than a weak model. The energy numbers were real throughout:
the work happened, only the answer was invisible.
"""

from hmasync_controller.bench.vllm_client import _message_text


class TestReasoningChannelFallback:
    def test_content_wins_when_present(self):
        """The reasoning channel is a scratchpad. When the model emitted a
        real answer the scratchpad must never override it -- reasoning
        routinely names the options it is REJECTING, so preferring it would
        turn a correct answer into a wrong one."""
        assert (
            _message_text({"content": "B", "reasoning": "A is tempting but wrong"})
            == "B"
        )

    def test_falls_back_to_reasoning_when_content_is_empty(self):
        """If the model expressed its conclusion nowhere else, that text is
        the best evidence of its answer -- and the letter/number extractors
        already scan for a concluding statement."""
        assert (
            _message_text({"content": "", "reasoning": "so the answer is B"})
            == "so the answer is B"
        )

    def test_falls_back_to_reasoning_content_alias(self):
        """Field name differs across vLLM versions and reasoning parsers."""
        assert (
            _message_text({"content": None, "reasoning_content": "answer: C"})
            == "answer: C"
        )

    def test_whitespace_only_content_is_treated_as_empty(self):
        assert (
            _message_text({"content": "   \n", "reasoning": "answer: D"}) == "answer: D"
        )

    def test_no_text_anywhere_returns_empty_not_none(self):
        """Callers score a string; None would raise rather than score wrong."""
        assert _message_text({"content": None, "reasoning": None}) == ""
        assert _message_text({}) == ""

    def test_content_is_stripped(self):
        assert _message_text({"content": "  B \n"}) == "B"
