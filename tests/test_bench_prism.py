"""`bench prism` -- the Bonsai sub-4-bit wave on Apple Silicon.

This wave's whole value is that its rungs differ in ONE thing. These tests pin
the table that makes that true, the three refusals that stop a rung being
measured against the wrong file, and the contract that its rows stay local.

The fixture strings here are not invented: the ftype names are
`llama_ftype_name`'s own literals from the pinned fork
(`src/llama-model-loader.cpp`), and the two chat-template shapes are what the
4B and 27B GGUFs actually carry, read out of their metadata on 2026-09-22.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hmasync_controller import cli
from hmasync_controller.bench import prism
from hmasync_controller.bench.quick import QUICK_REFERENCE_N_SHOT, QUICK_REFERENCE_SEED
from hmasync_controller.config import Settings


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _props(rung: prism.PrismRung, **overrides) -> prism.ServedModel:
    """What a correctly-started server would report for this rung."""
    fields = {
        "model_path": f"/Users/x/prism-mac/models/{rung.gguf_file}",
        "ftype": rung.ftype,
        "n_ctx": prism.PRISM_CTX_SIZE,
        "build_info": prism.PRISM_ENGINE_RELEASE,
        "chat_template": "{%- if enable_thinking is defined and enable_thinking is false %}",
    }
    fields.update(overrides)
    return prism.ServedModel(**fields)


class TestTheRungTable:
    def test_keys_are_unique(self):
        keys = prism.rung_keys()
        assert len(keys) == len(set(keys))

    def test_no_rung_names_the_legacy_file(self):
        """prism-ml's own model cards recommend a bare `*-Q2_0.gguf`, which on
        these builds is the deprecated group-128-stored-as-id-42 packing. It
        now LOADS, so nothing downstream would fail -- it would just occupy
        the Q2_0_g64 rung and make the format control compare a file with
        itself."""
        for rung in prism.PRISM_RUNGS:
            assert not rung.gguf_file.endswith(prism.PRISM_LEGACY_FILENAME_SUFFIX), rung.key

    def test_every_rung_pins_a_full_commit(self):
        """A quantization name is not a set of weights; a re-upload under one
        filename is what the revision pins against."""
        for rung in prism.PRISM_RUNGS:
            assert re.fullmatch(r"[0-9a-f]{40}", rung.gguf_revision), rung.key

    def test_the_format_control_differs_only_in_storage(self):
        """Three storages of ONE checkpoint: same model identity, same repo,
        same commit. If any of those three drift apart, the stage stops being
        a control and becomes an ordinary model comparison."""
        control = [r for r in prism.PRISM_RUNGS if r.stage == "format-control"]
        assert len(control) == 3
        assert len({r.model for r in control}) == 1
        assert len({r.gguf_repo for r in control}) == 1
        assert len({r.gguf_revision for r in control}) == 1
        assert len({r.quantization for r in control}) == 3
        assert len({r.gguf_file for r in control}) == 3

    def test_the_reversal_pair_is_one_checkpoint_two_widths(self):
        pair = [r for r in prism.PRISM_RUNGS if r.stage == "reversal"]
        assert len(pair) == 2
        assert len({r.model for r in pair}) == 1
        assert len({r.gguf_revision for r in pair}) == 1
        assert {r.quantization for r in pair} == {"PTQ1_0", "PQ2_0"}

    def test_ftype_strings_are_the_loaders_own_words(self):
        """Matched against `/props.model_ftype` verbatim. The closing
        parenthesis is load-bearing: it is the only thing separating the
        PQ2_0 name from the legacy file's `(group 128, legacy ftype)`."""
        by_quant = {r.quantization: r.ftype for r in prism.PRISM_RUNGS}
        assert by_quant["PQ2_0"] == "PQ2_0 - 2.13 bpw (group 128)"
        assert by_quant["PTQ1_0"] == "PTQ1_0 - 1.75 bpw ternary (group 128)"
        assert by_quant["Q2_0_g64"] == "Q2_0"
        assert by_quant["F16"] == "F16"

    def test_the_f16_control_rung_fits_a_24gb_mac(self):
        """The reason this wave uses the 4B family and not the lab's 8B: the
        8B's F16 rung is 16.38 GB, which on 24 GB of unified memory plus a 16k
        window is close enough to the ceiling to swap -- and a swapping rung
        would silently be the most expensive one in a comparison whose point
        is that only storage differs."""
        f16 = next(r for r in prism.PRISM_RUNGS if r.quantization == "F16")
        assert f16.size_gb < 16.0

    def test_the_wave_plan_names_every_rung(self):
        plan = prism.describe_wave()
        for rung in prism.PRISM_RUNGS:
            assert rung.key in plan
            assert rung.gguf_file in plan

    def test_unknown_rung_names_the_real_ones(self):
        with pytest.raises(prism.UnknownPrismRungError) as exc:
            prism.rung_by_key("bonsai-9000")
        for key in prism.rung_keys():
            assert key in str(exc.value)


class TestTheTaskMatrix:
    def test_two_task_shapes_at_fixed_counts(self):
        """One decode-heavy and one prefill-heavy shape. Fixed, because rungs
        that differ in item count are not a control."""
        assert prism.PRISM_TASKS == [("gsm8k_platinum", 50), ("mmlu_redux", 100)]

    def test_the_configuration_tuple_matches_the_labs(self):
        """n_shot/seed come from the same constants every other mode uses, so
        a Mac rung and a lab prism row differ in hardware and nothing else."""
        assert QUICK_REFERENCE_N_SHOT == 5
        assert QUICK_REFERENCE_SEED == 1234
        assert prism.PRISM_CTX_SIZE == 16384


class TestVerifyingWhatIsServed:
    def test_accepts_the_rung_it_was_asked_for(self):
        rung = prism.PRISM_RUNGS[0]
        prism.verify_served(rung, _props(rung), port=prism.PRISM_PORT)

    def test_refuses_a_different_file(self):
        rung = prism.rung_by_key("bonsai-4b-pq2-0")
        other = prism.rung_by_key("bonsai-4b-f16")
        with pytest.raises(prism.PrismWeightsMismatchError) as exc:
            prism.verify_served(
                rung, _props(rung, model_path=f"/m/{other.gguf_file}"), port=8091
            )
        assert other.gguf_file in str(exc.value)
        assert rung.gguf_file in str(exc.value)

    def test_refuses_when_the_server_reports_no_path(self):
        """Unverifiable weights are refused, not measured: the F16 rung is an
        ordinary GGUF any llama-server would serve, so a wrong row here reads
        as a real result."""
        rung = prism.PRISM_RUNGS[0]
        with pytest.raises(prism.PrismWeightsMismatchError):
            prism.verify_served(rung, _props(rung, model_path=None), port=8091)

    def test_refuses_the_legacy_packing_under_the_right_filename(self):
        """The trap this interlock exists for. The file name is correct, the
        loader accepts it, and the packing is the deprecated one."""
        rung = prism.rung_by_key("bonsai-4b-pq2-0")
        served = _props(rung, ftype="PQ2_0 - 2.13 bpw (group 128, legacy ftype)")
        with pytest.raises(prism.PrismFtypeMismatchError) as exc:
            prism.verify_served(rung, served, port=8091)
        assert "DEPRECATED" in str(exc.value)
        assert rung.gguf_revision in str(exc.value)

    def test_refuses_a_format_that_is_not_this_rungs(self):
        rung = prism.rung_by_key("bonsai-4b-pq2-0")
        with pytest.raises(prism.PrismFtypeMismatchError):
            prism.verify_served(rung, _props(rung, ftype="Q4_K - Medium"), port=8091)

    def test_accepts_a_guessed_ftype_but_it_is_still_the_right_one(self):
        """`llama_ftype_name` prefixes "(guessed) " when the file declared no
        general.file_type. The weights are still pinned by repo+revision, so
        this warns rather than refuses."""
        rung = prism.rung_by_key("bonsai-4b-f16")
        prism.verify_served(rung, _props(rung, ftype="(guessed) F16"), port=8091)

    def test_refuses_a_shrunken_window(self):
        """llama.cpp's auto-fit reduces the context rather than failing. A
        control whose rungs differ in both storage and window measures
        nothing."""
        rung = prism.PRISM_RUNGS[0]
        with pytest.raises(prism.PrismContextMismatchError) as exc:
            prism.verify_served(rung, _props(rung, n_ctx=4096), port=8091)
        assert "--fit off" in str(exc.value)

    def test_an_unreported_window_is_not_treated_as_wrong(self):
        rung = prism.PRISM_RUNGS[0]
        prism.verify_served(rung, _props(rung, n_ctx=None), port=8091)

    def test_every_refusal_says_how_to_fix_it(self):
        rung = prism.rung_by_key("bonsai-4b-pq2-0")
        with pytest.raises(prism.PrismWeightsMismatchError) as exc:
            prism.verify_served(rung, _props(rung, model_path="/m/other.gguf"), port=8091)
        message = str(exc.value)
        assert rung.gguf_repo in message
        assert rung.gguf_revision in message
        assert "llama-server" in message


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


class TestReadingPropsOverASocket:
    """The parsing runs against a REAL socket serving the fork's own /props
    shape (tools/server/server-context.cpp's `get_res_props`), because every
    interlock in this mode is a read of that payload: a field renamed
    upstream must break here, not in the field at 2am."""

    PROPS = {
        "default_generation_settings": {"n_ctx": prism.PRISM_CTX_SIZE, "params": {}},
        "total_slots": 1,
        "model_alias": "Ternary-Bonsai-4B-PQ2_0.gguf",
        "model_ftype": "PQ2_0 - 2.13 bpw (group 128)",
        "model_path": "/Users/x/prism-mac/models/Ternary-Bonsai-4B-PQ2_0.gguf",
        "chat_template": "{%- if enable_thinking is defined %}",
        "build_info": "prism-b10709-9a9394a",
    }

    def test_reads_the_file_format_and_window(self):
        import http.server
        import threading

        payload = json.dumps(self.PROPS).encode()

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib's spelling
                body = payload if self.path == "/props" else b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            served = _run(prism.fetch_props(base_url))
        finally:
            server.shutdown()
            server.server_close()

        assert served.model_path.endswith("Ternary-Bonsai-4B-PQ2_0.gguf")
        assert served.ftype == "PQ2_0 - 2.13 bpw (group 128)"
        assert served.n_ctx == prism.PRISM_CTX_SIZE
        assert served.build_info == "prism-b10709-9a9394a"
        prism.verify_served(
            prism.rung_by_key("bonsai-4b-pq2-0"), served, port=prism.PRISM_PORT
        )

    def test_an_unreachable_server_refuses_rather_than_guesses(self):
        with pytest.raises(prism.ModelNotAvailableError) as exc:
            # Port 1 is reserved and never listening.
            _run(prism.fetch_props("http://127.0.0.1:1"))
        assert "cannot proceed" in str(exc.value)


class TestRunningARung:
    """The suite call itself: what it pins, and that nothing is measured
    before the weights are verified."""

    def _patch_engine(self, monkeypatch, served, recorder):
        class _Detected:
            name = "llamacpp"
            base_url = "http://localhost:8091"

        async def _detect(*a, **kw):
            return _Detected()

        async def _props_fn(base_url):
            return served

        async def _suite(**kwargs):
            recorder.update(kwargs)
            return "measured"

        monkeypatch.setattr(prism, "detect_engine", _detect)
        monkeypatch.setattr(prism, "fetch_props", _props_fn)
        monkeypatch.setattr(prism, "_run_bench_suite", _suite)

    def test_pins_every_field_the_comparison_depends_on(self, monkeypatch):
        rung = prism.rung_by_key("bonsai2-27b-pq2-0")
        called: dict = {}
        self._patch_engine(monkeypatch, _props(rung), called)

        result = _run(prism.run_prism_suite(rung_key=rung.key))

        assert result.rung is rung
        assert called["engine_choice"] == "llamacpp"
        assert called["tasks"] == prism.PRISM_TASKS
        # Thinking off, pinned: it is the largest energy lever these models
        # have and would otherwise sit inside a storage comparison.
        assert called["thinking"] is False
        # Stock power only. Apple exposes no power-limit control, and a sweep
        # would be a second axis.
        assert called["max_sweep_points"] == 0
        assert called["model_id"] == rung.model
        assert called["quantization"] == rung.quantization
        assert called["gguf_repo"] == rung.gguf_repo
        assert called["gguf_revision"] == rung.gguf_revision

    def test_measures_nothing_when_the_wrong_file_is_served(self, monkeypatch):
        rung = prism.rung_by_key("bonsai-4b-f16")
        called: dict = {}
        self._patch_engine(monkeypatch, _props(rung, model_path="/m/something-else.gguf"), called)

        with pytest.raises(prism.PrismWeightsMismatchError):
            _run(prism.run_prism_suite(rung_key=rung.key))
        assert called == {}, "verification must precede measurement"

    def test_no_server_names_the_port_and_the_remedy(self, monkeypatch):
        async def _detect(*a, **kw):
            return None

        monkeypatch.setattr(prism, "detect_engine", _detect)
        with pytest.raises(prism.NoEngineDetectedError) as exc:
            _run(prism.run_prism_suite(rung_key="bonsai-4b-pq2-0", llamacpp_port=8091))
        assert "8091" in str(exc.value)
        assert "never launches an engine itself" in str(exc.value)

    def test_defaults_to_the_interlock_port_not_8080(self):
        """Three of these rungs are files another llama-server could serve, so
        a shared port could attach a prism rung to a mainline server and
        produce a plausible wrong row."""
        assert prism.PRISM_PORT == 8091
        signature = inspect.signature(prism.run_prism_suite)
        assert signature.parameters["llamacpp_port"].default == prism.PRISM_PORT


@dataclass
class _FakeRun:
    run_id: str = "prism-gsm8k_platinum-stock-1"
    task: str = "gsm8k_platinum"
    n_items: int = 50
    n_correct: int = 41
    n_shot: int = 5
    seed: int = 1234
    max_tokens: int = 8192
    thinking_mode: str = "enable_thinking=false"
    accuracy: float = 0.82
    truncated_pct: float = 0.0
    total_joules_gpu: float = 4200.0
    total_joules_gpu_best: float = 4180.0
    total_joules_cpu: float | None = 900.0
    total_joules_cpu_dram: float | None = 120.0
    joules_per_correct_answer: float = 102.0
    joules_per_token: float = 0.31
    total_completion_tokens: int = 13_500
    run_duration_s: float = 2400.0
    mean_tokens_per_second: float = 5.6
    energy_source: str = "ioreport"
    engine_version: str = "prism-b10709-9a9394a"
    gpu_name: str = "Apple M3"
    measurement_tier: str = "C"


@dataclass
class _FakeSuite:
    runs: list = field(default_factory=lambda: [_FakeRun()])
    task_runs: list = field(default_factory=list)


class TestCli:
    def test_the_subcommand_parses(self):
        args = cli._parse_args(["bench", "prism", "--rung", "bonsai-4b-pq2-0"])
        assert args.bench_subcommand == "prism"
        assert args.rung == "bonsai-4b-pq2-0"
        assert args.port == prism.PRISM_PORT

    def test_an_unknown_rung_is_refused_at_parse_time(self):
        with pytest.raises(SystemExit):
            cli._parse_args(["bench", "prism", "--rung", "bonsai-9000"])

    def test_list_prints_the_plan(self, capsys):
        code = cli.main(["bench", "prism", "--list"])
        assert code == 0
        out = capsys.readouterr().out
        for key in prism.rung_keys():
            assert key in out

    def test_writes_a_summary_and_never_a_bundle(self, tmp_path, monkeypatch):
        """The contract in `bench/prism.py`'s docstring: the submission
        schema's `suite` enum and the leaderboard's config key both live in
        the lab repo, and neither has a field for WHICH BUILD of llama.cpp
        served the row. So these rows stay local."""
        rung = prism.rung_by_key("bonsai-4b-pq2-0")
        result = prism.PrismSuiteResult(rung=rung, served=_props(rung), suite=_FakeSuite())

        def _fake_await(coro, **kwargs):
            coro.close()  # the real helper awaits it; nothing to run here
            return result

        monkeypatch.setattr(cli, "_await_bench_suite", _fake_await)
        monkeypatch.setattr(cli, "_write_bench_artifacts", lambda *a, **kw: None)

        settings = Settings(
            BENCH_DATA_DIR=str(tmp_path / "data"),
            BENCH_BUNDLE_DIR=str(tmp_path / "bundles"),
            BENCH_OPTIN=True,  # even opted in, nothing is built to submit
        )
        code, message = cli.run_bench_prism(settings, rung.key)

        assert code == 0
        assert not (tmp_path / "bundles").exists()
        summaries = list((tmp_path / "data" / "prism").glob("*.json"))
        assert len(summaries) == 1
        assert rung.key in summaries[0].name
        assert str(summaries[0]) in message

        summary = json.loads(summaries[0].read_text())
        assert summary["rung"]["quantization"] == "PQ2_0"
        assert summary["rung"]["gguf_revision"] == rung.gguf_revision
        assert summary["ctx_size"] == prism.PRISM_CTX_SIZE
        # The join key for the per-item format-control check.
        assert summary["tasks"][0]["run_id"] == "prism-gsm8k_platinum-stock-1"
        # Which mechanism actually held thinking off, not just what was sent.
        assert summary["thinking"]["mechanism"] == prism.THINKING_AXIS_KWARG
        # A home directory is not part of the measurement.
        assert summary["served"]["gguf_file"] == rung.gguf_file
        assert "/Users/" not in json.dumps(summary)

    def test_a_mismatch_is_an_operator_error_not_a_dead_box(self, tmp_path, monkeypatch):
        def _raise(coro, **kwargs):
            coro.close()
            raise prism.PrismFtypeMismatchError("legacy packing")

        monkeypatch.setattr(cli, "_await_bench_suite", _raise)
        settings = Settings(BENCH_DATA_DIR=str(tmp_path), BENCH_BUNDLE_DIR=str(tmp_path))
        code, message = cli.run_bench_prism(settings, "bonsai-4b-pq2-0")
        assert code == 2
        assert "legacy packing" in message

    def test_the_source_builds_no_bundle(self):
        """A tripwire, not a restatement: the bundle path is one `build_bundle`
        call away, and adding it would publish fork rows into a config key
        that cannot tell them from mainline llama.cpp."""
        source = inspect.getsource(cli.run_bench_prism)
        # Docstring stripped: it explains at length WHY there is no bundle,
        # and the words it uses to do that are the words being searched for.
        body = source.split('"""')[-1]
        assert "build_bundle" not in body
        assert "submit_bundle" not in body
        assert "submit_fn" not in body

    def test_prism_is_not_a_submittable_suite(self):
        schema = json.loads(
            (Path(cli.__file__).parent / "schemas" / "bench_submission.schema.json").read_text()
        )
        assert "prism" not in schema["properties"]["suite"]["enum"]


class TestSharedSuiteHelpers:
    """`_await_bench_suite`/`_write_bench_artifacts` were split out of
    `_run_bench_suite_cli` so a suite that writes no bundle could reuse them.
    The five bundle-writing suites must be unaffected."""

    def test_a_timeout_still_names_the_way_out(self):
        async def _forever():
            await asyncio.sleep(10)

        with pytest.raises(cli._BenchSuiteAborted) as exc:
            cli._await_bench_suite(_forever(), suite="prism", timeout_s=0.01)
        assert exc.value.code == 1
        assert "--timeout" in exc.value.message
        assert "BENCH_PRISM_TIMEOUT_S" in exc.value.message

    def test_a_missing_model_is_exit_two(self):
        from hmasync_controller.bench.quick import ModelNotAvailableError

        async def _raise():
            raise ModelNotAvailableError("pull it first")

        with pytest.raises(cli._BenchSuiteAborted) as exc:
            cli._await_bench_suite(_raise(), suite="quick", timeout_s=5)
        assert exc.value.code == 2

    def test_nothing_measurable_is_exit_one(self):
        from hmasync_controller.bench.quick import AllTasksFailedError

        async def _raise():
            raise AllTasksFailedError("every task failed")

        with pytest.raises(cli._BenchSuiteAborted) as exc:
            cli._await_bench_suite(_raise(), suite="quick", timeout_s=5)
        assert exc.value.code == 1


class TestRecordedProvenance:
    """`resolve_quick_model`'s llama.cpp branch records no quantization by
    design -- attach mode has nothing to read one from. `bench prism` may
    override it because it verified both against `/props` first."""

    def _engine(self):
        class _Engine:
            name = "llamacpp"
            base_url = "http://localhost:8091"
            adapter = None

        return _Engine()

    def _patch_models(self, monkeypatch, served="Ternary-Bonsai-4B-PQ2_0.gguf"):
        from hmasync_controller.bench import quick

        async def _get_models(self):
            return [served]

        monkeypatch.setattr(quick.VLLMClient, "get_models", _get_models)

    def test_defaults_are_unchanged_without_an_override(self, monkeypatch):
        from hmasync_controller.bench import quick

        self._patch_models(monkeypatch)
        model = _run(quick.resolve_quick_model(self._engine()))
        assert model.record_model == "Ternary-Bonsai-4B-PQ2_0.gguf"
        assert model.record_quantization is None

    def test_a_verified_caller_may_record_both(self, monkeypatch):
        from hmasync_controller.bench import quick

        self._patch_models(monkeypatch)
        model = _run(
            quick.resolve_quick_model(
                self._engine(),
                model_id="prism-ml/Ternary-Bonsai-4B",
                quantization="PQ2_0",
            )
        )
        # The three format-control rungs must share a model identity and
        # differ only in quantization, or they group as three models.
        assert model.record_model == "prism-ml/Ternary-Bonsai-4B"
        assert model.record_quantization == "PQ2_0"
        # What is SENT to the server stays the server's own id.
        assert model.name == "Ternary-Bonsai-4B-PQ2_0.gguf"
