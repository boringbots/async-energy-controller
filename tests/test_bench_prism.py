"""`bench prism` -- sub-4-bit Bonsai weights on the PrismML llama.cpp fork."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from hmasync_controller import cli
from hmasync_controller.bench import prism
from hmasync_controller.bench.quick import FULL_TASKS


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestTheRungs:
    def test_every_rung_pins_its_weights(self):
        """A quantization NAME is not a set of weights."""
        for e in prism.PRISM_GGUFS:
            assert e.repo and "/" in e.repo, e.filename
            assert len(e.revision) == 40, e.filename
            assert e.filename.endswith(".gguf"), e.filename
            assert e.quantization and e.size_gb > 0, e.filename

    def test_the_format_control_is_one_checkpoint_in_three_storages(self):
        """Stage 1's premise: the same ternary weights stored dense, group-64
        and group-128 must return the SAME answers, so energy differences are
        bytes-and-kernel with the accuracy confound removed."""
        s1 = [e for e in prism.PRISM_GGUFS if e.stage.startswith("1-format-control")]
        assert len(s1) == 3
        assert {e.quantization for e in s1} == {"F16", "Q2_0_g64", "PQ2_0"}
        # One checkpoint: same repo, same revision, same recorded identity.
        assert len({e.repo for e in s1}) == 1
        assert len({e.revision for e in s1}) == 1
        assert len({e.hf_id for e in s1}) == 1

    def test_the_format_control_is_not_recorded_as_qwen3_8b(self):
        """The model card's base_model is Ternary-Bonsai-8B-unpacked: all
        three storages ARE the ternary model. Recording Qwen3-8B would claim a
        comparison the files do not support -- that is what stage 2 is for."""
        s1 = [e for e in prism.PRISM_GGUFS if e.stage.startswith("1-format-control")]
        for e in s1:
            assert e.hf_id == "prism-ml/Ternary-Bonsai-8B-unpacked", e.filename
        spine = [e for e in prism.PRISM_GGUFS if e.stage == "2-spine"]
        assert spine and all(e.hf_id == "Qwen/Qwen3-8B" for e in spine)

    def test_prism_runs_the_same_tasks_as_full(self):
        """So a Bonsai row and an Ollama row differ in the weights only."""
        assert prism.PRISM_TASKS == FULL_TASKS


class TestServedWeightsIdentification:
    @pytest.mark.parametrize(
        "served,expect",
        [
            ("Ternary-Bonsai-1.7B-PQ2_0.gguf", "PQ2_0"),
            ("/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "PTQ1_0"),
            ("./prism-models/Bonsai-27B-Q1_0.gguf", "Q1_0"),
            ("ternary-bonsai-8b-f16.gguf", "F16"),
        ],
    )
    def test_recognises_a_rung_however_it_is_aliased(self, served, expect):
        got = prism.match_served_gguf(served)
        assert got is not None and got.quantization == expect

    def test_g64_is_not_mistaken_for_the_bare_q2_0_file(self):
        """The model card publishes BOTH `Ternary-Bonsai-8B-Q2_0.gguf` and
        `...-Q2_0_g64.gguf`, and they are different files. Longest-first
        matching keeps the g64 rung from being recorded as the other one."""
        got = prism.match_served_gguf("Ternary-Bonsai-8B-Q2_0_g64.gguf")
        assert got is not None
        assert got.quantization == "Q2_0_g64"

    @pytest.mark.parametrize("served", ["", "qwen3.5:9b-q4_K_M", "some-other.gguf"])
    def test_unknown_weights_are_not_guessed(self, served):
        assert prism.match_served_gguf(served) is None

    def test_unknown_weights_refuse_rather_than_measure(self):
        async def _detect(*a, **kw):
            class _D:
                name = "llamacpp"
                base_url = "http://localhost:8080"

            return _D()

        async def _get_models(self):
            return ["mystery-model.gguf"]

        with (
            patch.object(prism, "detect_engine", _detect),
            patch.object(prism.VLLMClient, "get_models", _get_models),
        ):
            with pytest.raises(prism.PrismWeightsUnknownError) as exc:
                _run(prism.run_prism_suite())
        msg = str(exc.value)
        assert "not a pinned Bonsai rung" in msg
        # The refusal has to be actionable: name the fork and the rungs.
        assert "PrismML-Eng/llama.cpp" in msg
        assert "Ternary-Bonsai-1.7B-PQ2_0.gguf" in msg


class TestCli:
    def test_prism_subcommand_parses(self):
        assert cli._parse_args(["bench", "prism"]).bench_subcommand == "prism"

    def test_prism_has_no_thinking_flag(self):
        """Pinned off to match `full`."""
        with pytest.raises(SystemExit):
            cli._parse_args(["bench", "prism", "--thinking"])

    def test_prism_is_an_accepted_suite(self):
        import json
        from pathlib import Path

        schema = json.loads(
            (Path(cli.__file__).parent / "schemas" / "bench_submission.schema.json").read_text()
        )
        assert "prism" in schema["properties"]["suite"]["enum"]


class TestArchiveOnly:
    """prism rows go to a share drive, not the API.

    The wave measures an engine fork and weights that exist nowhere else in
    the corpus, so its rows have no counterpart to pool with -- and the public
    API validates against a vendored copy of the schema, so upstreaming
    `prism` would need a schema bump plus a redeploy before one row landed."""

    def test_the_submit_seam_is_never_wired_for_prism(self, tmp_path, monkeypatch):
        seen = {}

        def fake_cli(settings, coro, *, suite, submit_fn, now_fn, timeout_s):
            seen["suite"] = suite
            seen["submit_fn"] = submit_fn
            coro.close()
            return 0, "bundle written to x.json"

        monkeypatch.setattr(cli, "_run_bench_suite_cli", fake_cli)
        code, _ = cli.run_bench_prism(cli.Settings(), submit_fn=cli._bench_submit_fn)
        assert code == 0
        assert seen["suite"] == "prism"
        assert seen["submit_fn"] is None, "a prism bundle must not reach the wire"

    def test_opting_in_does_not_change_that(self, tmp_path, monkeypatch):
        seen = {}

        def fake_cli(settings, coro, *, suite, submit_fn, now_fn, timeout_s):
            seen["submit_fn"] = submit_fn
            coro.close()
            return 0, "ok"

        s = cli.Settings()
        s.BENCH_OPTIN = True
        monkeypatch.setattr(cli, "_run_bench_suite_cli", fake_cli)
        cli.run_bench_prism(s, submit_fn=cli._bench_submit_fn)
        assert seen["submit_fn"] is None

class TestTheThinkingAxis:
    # The two templates the wave actually meets, as they are STORED in the
    # GGUF -- a template source, so the newlines are the two characters
    # backslash-n, not newlines.
    TEMPLATE_27B = (
        "{%- if add_generation_prompt %}\n"
        "    {{- '<|im_start|>assistant\\n' }}\n"
        "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
        "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
        "    {%- else %}\n        {{- '<think>\\n' }}\n    {%- endif %}\n{%- endif %}"
    )
    TEMPLATE_4B = (
        "{%- if add_generation_prompt %}\n"
        "    {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}\n{%- endif %}"
    )

    def test_the_27b_template_reads_the_kwarg(self):
        """Its default is thinking ON, so the pin this mode sends is what
        holds the axis -- dropping it turns the 27B rungs into reasoning runs
        at many times the energy."""
        mechanism, _ = prism.thinking_axis_note(self.TEMPLATE_27B)
        assert mechanism == prism.THINKING_AXIS_KWARG

    def test_the_4b_template_hardcodes_it_off(self):
        """No enable_thinking variable at all: the pin is inert here and the
        template is what holds the axis. Recognising the ESCAPED form is the
        whole trick -- a matcher written against rendered newlines would call
        this template unpinned."""
        mechanism, note = prism.thinking_axis_note(self.TEMPLATE_4B)
        assert mechanism == prism.THINKING_AXIS_TEMPLATE
        assert "inert" in note

    def test_a_template_that_does_neither_is_called_out(self):
        mechanism, note = prism.thinking_axis_note(
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
        )
        assert mechanism == prism.THINKING_AXIS_UNKNOWN
        assert "REASONING" in note

    def test_no_template_is_unknown_rather_than_assumed(self):
        assert prism.thinking_axis_note(None)[0] == prism.THINKING_AXIS_UNKNOWN


class TestReadingPropsIsAdvisoryHere:
    """This tier names its rung from `GET /v1/models`, so `/props` is read
    for the thinking axis and the window alone. A server that cannot serve
    it must still produce a rung -- the notes degrade, the row is written."""

    def test_an_unreachable_server_yields_no_props_rather_than_raising(self):
        from hmasync_controller.bench.quick import fetch_llamacpp_props

        # Port 1 on localhost: nothing listens, so the GET fails outright.
        props = _run(fetch_llamacpp_props("http://localhost:1"))
        assert (props.chat_template, props.n_ctx, props.build_info) == (None, None, None)

    def test_a_missing_template_reads_as_unknown_not_as_pinned(self):
        from hmasync_controller.bench.quick import fetch_llamacpp_props

        props = _run(fetch_llamacpp_props("http://localhost:1"))
        assert prism.thinking_axis_note(props.chat_template)[0] == prism.THINKING_AXIS_UNKNOWN

    def test_the_window_and_build_are_read_out_of_the_props_body(self):
        """The fork's own /props shape: n_ctx lives under
        default_generation_settings, build_info at the top."""
        from hmasync_controller.bench.quick import parse_llamacpp_props

        props = parse_llamacpp_props({
            "default_generation_settings": {"n_ctx": 4096, "params": {}},
            "build_info": "b1-5d80cff0",
            "chat_template": "{{ messages }}",
        })
        assert props.n_ctx == 4096
        assert props.build_info == "b1-5d80cff0"
        assert props.chat_template == "{{ messages }}"

    def test_a_malformed_body_is_all_none(self):
        from hmasync_controller.bench.quick import parse_llamacpp_props

        assert parse_llamacpp_props(["not", "a", "dict"]).n_ctx is None
        assert parse_llamacpp_props({"default_generation_settings": "x"}).n_ctx is None

    def test_the_how_to_serves_the_labs_window(self):
        """-c 4096 is what cut the 2026-09-23 wave's math500 items."""
        text = prism._how_to_serve_it()
        command = next(line for line in text.splitlines() if line.strip().startswith("llama-server"))
        assert "-c 16384" in command
        assert "--fit off" in command
