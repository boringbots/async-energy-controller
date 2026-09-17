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

from hmasync_controller.bench.vllm_client import _delta_reasoning, _message_text


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


class TestStreamingReasoningChannel:
    """The STREAMING path is the one `runner.py` actually uses -- every wave
    row carries `streaming_used=True`. Fixing only the non-streaming path
    looked correct in a direct probe and changed nothing in a real run:
    gpt-oss-20b's gpqa_diamond stayed at 0.13, still below the 0.25 chance
    floor, because the runner never went through `message.content` at all.
    """

    def test_reads_the_streamed_reasoning_channel(self):
        assert _delta_reasoning({"delta": {"reasoning": "so the answer is B"}}) == (
            "so the answer is B"
        )

    def test_reads_the_reasoning_content_alias(self):
        assert _delta_reasoning({"delta": {"reasoning_content": "answer: C"}}) == (
            "answer: C"
        )

    def test_no_reasoning_channel_returns_none(self):
        """None, not "" -- the loop distinguishes "no token this chunk" from
        "a token whose text is empty" when timing TTFT."""
        assert _delta_reasoning({"delta": {"content": "B"}}) is None
        assert _delta_reasoning({"delta": {}}) is None
        assert _delta_reasoning({}) is None

    def test_content_chunks_are_not_mistaken_for_reasoning(self):
        assert _delta_reasoning({"delta": {"content": "not reasoning"}}) is None


class TestCompletionTextIsKept:
    """A run's own answers, stored beside its energy numbers.

    Written after two results could not be explained from the stored columns
    alone. Qwen3.5-9B on math500 scored 0.80 with thinking off and 0.12 with
    it on, at 12% truncation -- and no column separated "the model reasoned
    its way to a wrong answer" from "it answered correctly in a format the
    extractor does not read". The same ambiguity had already come up on
    mmlu_redux (0.84 -> 0.48). Both are answerable the moment the text is on
    disk, and neither is answerable without it.
    """

    def test_streaming_keeps_content_and_reasoning_apart(self):
        """`completion_text` is what the scorer read; `reasoning_text` is the
        scratchpad it did NOT read. Kept as two fields because the whole
        point is to tell a wrong answer from a missed extraction."""
        result, text = _run_chat(
            content_chunks=["The answer ", "is B"],
            reasoning_chunks=["A looks right ", "but is not"],
        )
        assert text == "The answer is B"
        assert result.completion_text == "The answer is B"
        assert result.reasoning_text == "A looks right but is not"

    def test_reasoning_is_not_stored_twice(self):
        """When content is empty the reasoning channel IS the scored text
        (`_message_text`'s fallback). Storing it in both fields would double
        a thinking run's largest column -- 7,411 tokens/item measured on
        gpqa_diamond -- to say the same thing twice."""
        result, text = _run_chat(
            content_chunks=[], reasoning_chunks=["so the answer is C"]
        )
        assert text == "so the answer is C"
        assert result.completion_text == "so the answer is C"
        assert result.reasoning_text is None

    def test_ordinary_model_has_no_reasoning_text(self):
        result, _ = _run_chat(content_chunks=["B"], reasoning_chunks=[])
        assert result.completion_text == "B"
        assert result.reasoning_text is None

    def test_non_streaming_path_keeps_the_same_two_fields(self):
        """The two paths must agree: which one a run takes is a config
        detail, not a measurement, and the same run must be diagnosable
        either way."""
        result, text = _run_chat(
            content_chunks=["The answer is B"],
            reasoning_chunks=["A looks right but is not"],
            stream=False,
        )
        assert text == "The answer is B"
        assert result.completion_text == "The answer is B"
        assert result.reasoning_text == "A looks right but is not"


# --- a fake OpenAI-compatible server, streaming and not -----------------------


def _run_chat(*, content_chunks, reasoning_chunks, stream=True):
    """Drive `VLLMClient.chat()` against a throwaway local HTTP server that
    answers with the given channels. A real socket rather than a monkeypatched
    httpx, so the SSE framing is exercised too."""
    import asyncio
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from hmasync_controller.bench.vllm_client import VLLMClient

    content = "".join(content_chunks)
    reasoning = "".join(reasoning_chunks)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence the stderr access log
            pass

        def do_POST(self):  # noqa: N802 - stdlib handler naming
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if stream:
                chunks = [
                    {"choices": [{"delta": {"content": p}, "finish_reason": None}]}
                    for p in content_chunks
                ]
                chunks += [
                    {"choices": [{"delta": {"reasoning": p}, "finish_reason": None}]}
                    for p in reasoning_chunks
                ]
                chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                chunks.append(
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 9},
                    }
                )
                body = "".join(
                    f"data: {json.dumps(c)}\n\n" for c in chunks
                ) + "data: [DONE]\n\n"
                ctype = "text/event-stream"
            else:
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": content,
                                    "reasoning": reasoning,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 9},
                    }
                )
                ctype = "application/json"
            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = VLLMClient("127.0.0.1", server.server_address[1])
        return asyncio.run(
            client.chat(prompt="q", model="m", max_tokens=16, stream=stream)
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestReasoningEffortReachesHarmony:
    """`reasoning_effort` must be sent where the harmony renderer looks.

    vLLM builds a gpt-oss prompt with the harmony encoder, whose system
    message takes the effort from the TOP-LEVEL `reasoning_effort` request
    field; `chat_template_kwargs` is read only by the Jinja template path.
    Sending the dial inside the kwargs dict alone left every "low" and "high"
    row of the 2026-09-09 effort ladder running at the default `medium` --
    142 of 198 gpqa reasoning traces byte-identical between the two rungs
    (energy-bench BENCHMARK-REFERENCE.md F13 / D12).
    """

    def _payload_for(self, chat_template_kwargs):
        import asyncio
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from hmasync_controller.bench.vllm_client import VLLMClient

        seen: dict = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):  # noqa: N802 - stdlib handler naming
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                seen.update(json.loads(raw))
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": "B"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 1},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = VLLMClient("127.0.0.1", server.server_address[1])
            asyncio.run(
                client.chat(
                    prompt="q",
                    model="m",
                    max_tokens=16,
                    stream=False,
                    chat_template_kwargs=chat_template_kwargs,
                )
            )
        finally:
            server.shutdown()
            server.server_close()
        return seen

    def test_effort_is_lifted_to_the_top_level(self):
        payload = self._payload_for({"reasoning_effort": "low"})
        assert payload["reasoning_effort"] == "low"
        # The kwargs copy stays: a Jinja template that reads it there still can.
        assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}

    def test_thinking_pin_is_not_lifted(self):
        """`enable_thinking` has no top-level twin; lifting it would invent
        a request field vLLM does not define."""
        payload = self._payload_for({"enable_thinking": False})
        assert "reasoning_effort" not in payload
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_no_kwargs_sends_neither(self):
        payload = self._payload_for(None)
        assert "reasoning_effort" not in payload
        assert "chat_template_kwargs" not in payload
