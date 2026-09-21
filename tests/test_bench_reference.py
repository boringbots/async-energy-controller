"""`bench reference` -- the Efficiency Index anchor.

The whole value of this mode is that its number is comparable to a lab row,
which is true only while every field of the anchor holds. These tests pin the
fields and the refusals.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from hmasync_controller import cli
from hmasync_controller.bench import reference as ref
from hmasync_controller.bench.quick import (
    QUICK_REFERENCE_MODELS,
    QUICK_REFERENCE_MODEL_HF_ID,
    QUICK_REFERENCE_N_SHOT,
    QUICK_REFERENCE_SEED,
)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestTheAnchorTuple:
    def test_matches_energy_benchs_reference_config(self):
        """TEST-MATRIX.md section 1, field for field. Changing any of these
        invalidates every efficiency_index ever computed against it."""
        assert ref.REFERENCE_TASK == "gsm8k_platinum"
        assert ref.REFERENCE_N_ITEMS == 100
        assert QUICK_REFERENCE_SEED == 1234
        assert QUICK_REFERENCE_N_SHOT == 5
        assert QUICK_REFERENCE_MODEL_HF_ID == "Qwen/Qwen3.5-9B"
        spec = QUICK_REFERENCE_MODELS["llama.cpp"]
        assert spec["gguf_repo"] == "lmstudio-community/Qwen3.5-9B-GGUF"
        assert spec["revision"] == "1379f25c6b505a3fc737bd7818cb09389cf807c1"

    def test_item_count_is_not_quicks(self):
        """25 items of the same task is a different measurement, not a
        cheaper one."""
        from hmasync_controller.bench.quick import QUICK_TASKS

        quick_gsm8k = dict(QUICK_TASKS)["gsm8k_platinum"]
        assert quick_gsm8k != ref.REFERENCE_N_ITEMS

    def test_published_anchor_figures_are_recorded(self):
        """The .114 row this reproduces, so a run can be read against it."""
        a = ref.REFERENCE_ANCHOR
        assert a.joules_per_correct == 975.0
        assert a.accuracy == 0.81
        assert a.joules_total == 83_348.0


class TestWeightsVerification:
    @pytest.mark.parametrize(
        "served",
        [
            "Qwen3.5-9B-Q4_K_M.gguf",
            "/models/Qwen3.5-9B-Q4_K_M.gguf",
            "qwen3.5-9b-q4_k_m",
        ],
    )
    def test_accepts_the_reference_gguf_however_it_is_aliased(self, served):
        """llama-server reports whatever alias it was started with."""
        assert ref._served_model_matches(served) is True

    @pytest.mark.parametrize(
        "served",
        [
            "Qwen3.5-9B-Q8_0.gguf",       # different quant
            "Qwen3.5-4B-Q4_K_M.gguf",     # different size
            "Meta-Llama-3.1-8B-Q4_K_M.gguf",  # different model
            "unknown",
        ],
    )
    def test_rejects_anything_else(self, served):
        assert ref._served_model_matches(served) is False

    def test_wrong_weights_refuse_rather_than_measure(self):
        """An anchor computed against the wrong weights is worse than no
        anchor -- it looks like a comparable number."""

        async def _detect(*a, **kw):
            class _D:
                name = "llamacpp"
                base_url = "http://localhost:8080"

            return _D()

        async def _get_models(self):
            return ["Qwen3.5-4B-Q4_K_M.gguf"]

        with (
            patch.object(ref, "detect_engine", _detect),
            patch.object(ref.VLLMClient, "get_models", _get_models),
        ):
            with pytest.raises(ref.ReferenceWeightsMismatchError) as exc:
                _run(ref.run_reference_suite())
        msg = str(exc.value)
        assert "not the reference GGUF" in msg
        # The refusal must be actionable.
        assert "lmstudio-community/Qwen3.5-9B-GGUF" in msg
        assert "1379f25c6b505a3fc737bd7818cb09389cf807c1" in msg


class TestCli:
    def test_reference_subcommand_parses(self):
        args = cli._parse_args(["bench", "reference"])
        assert args.bench_subcommand == "reference"

    def test_reference_has_no_thinking_flag(self):
        """The anchor pins enable_thinking=false; offering the knob would
        advertise something that must not be turned."""
        with pytest.raises(SystemExit):
            cli._parse_args(["bench", "reference", "--thinking"])

    def test_reference_is_an_accepted_suite(self):
        import json
        from pathlib import Path

        schema = json.loads(
            (
                Path(cli.__file__).parent / "schemas" / "bench_submission.schema.json"
            ).read_text()
        )
        assert "reference" in schema["properties"]["suite"]["enum"]


class TestComparisonOutput:
    def test_flags_an_accuracy_gap_as_configuration_not_hardware(self):
        class _Run:
            total_joules_gpu_best = 5000.0
            joules_per_correct_answer = 200.0
            accuracy = 0.20
            energy_source = "counter"

        class _Result:
            runs = [_Run()]

        text = ref.format_reference_comparison(_Result())
        assert "accuracy is 0.61 off the anchor" in text
        assert "configuration differs, not the machine" in text

    def test_flags_a_cross_vendor_energy_source(self):
        class _Run:
            total_joules_gpu_best = 5000.0
            joules_per_correct_answer = 900.0
            accuracy = 0.80
            energy_source = "ioreport"

        class _Result:
            runs = [_Run()]

        text = ref.format_reference_comparison(_Result())
        assert "compare the accuracy, not the joules" in text


class TestAnchorRowIdentity:
    """The anchor exists to be joined against the lab's row. It can only be
    joined on the keys REFERENCE_CONFIG names."""

    def test_verified_weights_record_the_hf_join_key_not_the_server_alias(self):
        import asyncio

        from unittest.mock import patch

        from hmasync_controller.bench import quick as q
        from hmasync_controller.bench.roster import REFERENCE_ENTRY

        async def _get_models(self):
            # llama-server reports whatever `-m` was given: often a local path.
            return ["./ref/Qwen3.5-9B-Q4_K_M.gguf"]

        class _Engine:
            name = "llamacpp"
            base_url = "http://localhost:8080"

        with patch.object(q.VLLMClient, "get_models", _get_models):
            model = asyncio.run(q.resolve_quick_model(_Engine(), REFERENCE_ENTRY))

        # Recorded identity is the anchor's, not the alias.
        assert model.record_model == "Qwen/Qwen3.5-9B"
        assert model.record_quantization == "Q4_K_M"
        assert "./ref/" not in model.record_model
        # ... while the wire still addresses whatever the server calls it.
        assert model.name == "./ref/Qwen3.5-9B-Q4_K_M.gguf"

    def test_unverified_llamacpp_still_records_what_was_served(self):
        """`bench quick` against llama.cpp measures whatever is loaded and has
        no basis to claim an identity, so it keeps reporting the alias."""
        import asyncio

        from unittest.mock import patch

        from hmasync_controller.bench import quick as q

        async def _get_models(self):
            return ["some-random-model.gguf"]

        class _Engine:
            name = "llamacpp"
            base_url = "http://localhost:8080"

        with patch.object(q.VLLMClient, "get_models", _get_models):
            model = asyncio.run(q.resolve_quick_model(_Engine(), None))

        assert model.record_model == "some-random-model.gguf"
        assert model.record_quantization is None
