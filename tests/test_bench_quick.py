"""Unit tests for `hmasync_controller.bench.quick` (US-MERGE-04): engine
detection, model resolution (never pulls), the mini power sweep's skip
logic, and the `run_quick_suite` orchestrator. Mocked engines/telemetry
throughout -- no real Ollama, llama-server, or GPU hardware touched.

Ported from energy-bench's `tests/unit/test_quick.py`, minus
`TestLocalNvmlSampler`/`TestCollectorTelemetrySource` -- the sampler this
package uses (`bench.sampler.LocalNvmlSampler`) is already exhaustively
covered by `tests/test_bench_sampler.py` (US-MERGE-02), and there is no
collector fallback here (see `bench/quick.py`'s module docstring). Every
`@pytest.mark.asyncio async def` from the reference file is converted to a
plain sync `def` driving `asyncio.run(...)` -- this suite's pytest-asyncio
is installed but UNCONFIGURED (no `asyncio_mode`), same convention
`tests/test_bench_sampler.py` established.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hmasync_controller.bench.engines import DEFAULT_LLAMACPP_PORT, DEFAULT_OLLAMA_PORT, OllamaModelNotPulledError
from hmasync_controller.bench.metrics.models import InferenceResult
from hmasync_controller.bench.quick import (
    QUICK_REFERENCE_MODELS,
    QUICK_REFERENCE_N_SHOT,
    QUICK_REFERENCE_SEED,
    QUICK_TASKS,
    AllTasksFailedError,
    DetectedEngine,
    ModelNotAvailableError,
    NoEngineDetectedError,
    NvmlUnavailableError,
    QuickModel,
    QuickTaskRun,
    _derive_power_sweep_caps_w,
    detect_engine,
    resolve_quick_model,
    run_power_sweep,
    run_quick_suite,
    run_quick_task,
)
from hmasync_controller.bench.sampler import TelemetrySample
from hmasync_controller.bench.tasks import TaskItem
from hmasync_controller.bench.vllm_client import VLLMUnavailableError


def _run(coro):
    return asyncio.run(coro)


def _adapter_mock(*, ready: bool, base_url: str) -> MagicMock:
    adapter = MagicMock()
    adapter.ready = AsyncMock(return_value=ready)
    adapter.base_url = MagicMock(return_value=base_url)
    adapter.version = AsyncMock(return_value="1.2.3")
    adapter.launch_args = MagicMock(return_value=[])
    return adapter


class TestDetectEngine:
    def test_auto_prefers_ollama_when_ready(self):
        ollama = _adapter_mock(ready=True, base_url="http://localhost:11434")
        llamacpp = _adapter_mock(ready=True, base_url="http://localhost:8080")
        with (
            patch("hmasync_controller.bench.quick.OllamaAdapter", return_value=ollama),
            patch("hmasync_controller.bench.quick.AttachLlamaCppAdapter", return_value=llamacpp),
        ):
            result = _run(detect_engine(None))
        assert result == DetectedEngine(
            name="ollama", base_url="http://localhost:11434", adapter=ollama
        )
        llamacpp.ready.assert_not_called()  # short-circuits once ollama answers

    def test_auto_falls_back_to_llamacpp(self):
        ollama = _adapter_mock(ready=False, base_url="http://localhost:11434")
        llamacpp = _adapter_mock(ready=True, base_url="http://localhost:8080")
        with (
            patch("hmasync_controller.bench.quick.OllamaAdapter", return_value=ollama),
            patch("hmasync_controller.bench.quick.AttachLlamaCppAdapter", return_value=llamacpp),
        ):
            result = _run(detect_engine(None))
        assert result is not None
        assert result.name == "llama.cpp"

    def test_auto_returns_none_when_neither_ready(self):
        ollama = _adapter_mock(ready=False, base_url="http://localhost:11434")
        llamacpp = _adapter_mock(ready=False, base_url="http://localhost:8080")
        with (
            patch("hmasync_controller.bench.quick.OllamaAdapter", return_value=ollama),
            patch("hmasync_controller.bench.quick.AttachLlamaCppAdapter", return_value=llamacpp),
        ):
            result = _run(detect_engine(None))
        assert result is None

    def test_explicit_engine_never_tries_the_other(self):
        ollama = _adapter_mock(ready=True, base_url="http://localhost:11434")
        llamacpp = _adapter_mock(ready=False, base_url="http://localhost:8080")
        with (
            patch("hmasync_controller.bench.quick.OllamaAdapter", return_value=ollama),
            patch("hmasync_controller.bench.quick.AttachLlamaCppAdapter", return_value=llamacpp),
        ):
            result = _run(detect_engine("llamacpp"))
        assert result is None
        ollama.ready.assert_not_called()

    def test_ports_and_host_threaded_into_adapters(self):
        ollama = _adapter_mock(ready=True, base_url="http://myhost:9999")
        with patch(
            "hmasync_controller.bench.quick.OllamaAdapter", return_value=ollama
        ) as ollama_cls:
            _run(detect_engine("ollama", host="myhost", ollama_port=9999))
        ollama_cls.assert_called_once_with(
            host="myhost", port=9999, base_url="http://myhost:9999"
        )


class TestResolveQuickModel:
    def test_ollama_pulled_records_reference_model(self):
        adapter = _adapter_mock(ready=True, base_url="http://localhost:11434")
        adapter.verify_model_pulled = AsyncMock(return_value="qwen2")
        engine = DetectedEngine(
            name="ollama", base_url="http://localhost:11434", adapter=adapter
        )

        model = _run(resolve_quick_model(engine))

        adapter.verify_model_pulled.assert_awaited_once_with(
            QUICK_REFERENCE_MODELS["ollama"]["tag"]
        )
        assert model.name == QUICK_REFERENCE_MODELS["ollama"]["tag"]
        assert model.record_quantization == "Q4_K_M"
        assert "qwen2" in model.note

    def test_ollama_not_pulled_raises_with_pull_command(self):
        adapter = _adapter_mock(ready=True, base_url="http://localhost:11434")
        adapter.verify_model_pulled = AsyncMock(
            side_effect=OllamaModelNotPulledError(
                f"Run `ollama pull {QUICK_REFERENCE_MODELS['ollama']['tag']}`"
            )
        )
        engine = DetectedEngine(
            name="ollama", base_url="http://localhost:11434", adapter=adapter
        )

        with pytest.raises(ModelNotAvailableError) as exc_info:
            _run(resolve_quick_model(engine))
        assert "ollama pull" in str(exc_info.value)
        assert QUICK_REFERENCE_MODELS["ollama"]["tag"] in str(exc_info.value)

    def test_llamacpp_records_whatever_is_loaded(self):
        adapter = _adapter_mock(ready=True, base_url="http://localhost:8080")
        client = AsyncMock()
        client.get_models = AsyncMock(return_value=["/models/some-other-model.gguf"])
        with patch("hmasync_controller.bench.quick.VLLMClient", return_value=client):
            engine = DetectedEngine(
                name="llama.cpp", base_url="http://localhost:8080", adapter=adapter
            )
            model = _run(resolve_quick_model(engine))

        assert model.name == "/models/some-other-model.gguf"
        assert model.record_model == "/models/some-other-model.gguf"
        assert model.record_quantization is None  # unknown, not guessed

    def test_llamacpp_no_model_loaded_raises(self):
        adapter = _adapter_mock(ready=True, base_url="http://localhost:8080")
        client = AsyncMock()
        client.get_models = AsyncMock(return_value=[])
        with patch("hmasync_controller.bench.quick.VLLMClient", return_value=client):
            engine = DetectedEngine(
                name="llama.cpp", base_url="http://localhost:8080", adapter=adapter
            )
            with pytest.raises(ModelNotAvailableError):
                _run(resolve_quick_model(engine))


def _make_items(n: int) -> list[TaskItem]:
    return [
        TaskItem(item_id=f"t:{i}", prompt=f"prompt {i}", target="42")
        for i in range(n)
    ]


class _FakeTask:
    name = "fake_task"
    shape = "decode"
    is_canary = False
    revision = "refs/convert/parquet"
    default_max_tokens = 64
    stop = ["\n"]

    def __init__(self, items):
        self._items = items

    def load(self, n_items, n_shot, seed):
        return self._items[:n_items]

    def score(self, completion, item):
        return completion.strip() == item.target


class _FakeTelemetry:
    def __init__(self, samples, stock_w=None, min_w=None, max_w=None):
        self._samples = samples
        self.rapl_max_energy_range_uj = None
        self.rapl_dram_max_energy_range_uj = None
        self.start_calls: list[str] = []
        self._stock_w = stock_w
        self._min_w = min_w
        self._max_w = max_w

    async def start(self, run_id):
        self.start_calls.append(run_id)

    async def stop(self):
        return self._samples

    async def get_power_limit_w(self):
        return self._stock_w

    async def get_power_limit_constraints_w(self):
        return self._min_w, self._max_w


def _samples(n: int) -> list[TelemetrySample]:
    return [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
        )
        for i in range(n)
    ]


def _inference_result(text: str = "42") -> InferenceResult:
    return InferenceResult(
        request_id="r", prompt_tokens=10, completion_tokens=5,
        ttft_s=0.1, total_s=0.5, tokens_per_second=10.0,
    )


class TestRunQuickTask:
    def test_streaming_path_scores_items(self):
        items = _make_items(2)
        task = _FakeTask(items)
        telemetry = _FakeTelemetry(_samples(3))
        vllm_client = AsyncMock()
        vllm_client.chat = AsyncMock(
            side_effect=[
                (_inference_result(), "42"),
                (_inference_result(), "wrong"),
            ]
        )
        with patch("hmasync_controller.bench.quick.load_task", return_value=task):
            run = _run(run_quick_task(vllm_client, telemetry, "some-model", "fake_task", 2))

        assert run.n_items == 2
        assert run.streaming_used is True
        assert [r.correct for r in run.inference_results] == [True, False]
        assert run.task_shape == "decode"
        assert run.dataset_revision == "refs/convert/parquet"
        assert telemetry.start_calls  # telemetry was started

    def test_falls_back_to_non_streaming_on_failure(self):
        items = _make_items(1)
        task = _FakeTask(items)
        telemetry = _FakeTelemetry(_samples(2))
        vllm_client = AsyncMock()
        vllm_client.chat = AsyncMock(
            side_effect=[
                VLLMUnavailableError("connection reset"),
                (_inference_result(), "42"),
            ]
        )
        with patch("hmasync_controller.bench.quick.load_task", return_value=task):
            run = _run(run_quick_task(vllm_client, telemetry, "some-model", "fake_task", 1))

        assert run.streaming_used is False
        assert vllm_client.chat.await_count == 2

    def test_reference_defaults_used(self):
        items = _make_items(1)
        task = _FakeTask(items)
        telemetry = _FakeTelemetry(_samples(2))
        vllm_client = AsyncMock()
        vllm_client.chat = AsyncMock(return_value=(_inference_result(), "42"))
        with patch("hmasync_controller.bench.quick.load_task", return_value=task):
            run = _run(run_quick_task(vllm_client, telemetry, "some-model", "fake_task", 1))
        assert run.n_shot == QUICK_REFERENCE_N_SHOT
        assert run.seed == QUICK_REFERENCE_SEED

    def test_rapl_fields_read_defensively_when_absent(self):
        """`bench.sampler.LocalNvmlSampler` has no `rapl_max_energy_range_uj`
        attribute at all (unlike energy-bench's own Tier-C sampler, which
        always sets it to None) -- `run_quick_task` must not crash reading
        it off a telemetry source that simply doesn't define it."""
        items = _make_items(1)
        task = _FakeTask(items)

        class _BareTelemetry:
            async def start(self, run_id):
                pass

            async def stop(self):
                return _samples(2)

        vllm_client = AsyncMock()
        vllm_client.chat = AsyncMock(return_value=(_inference_result(), "42"))
        with patch("hmasync_controller.bench.quick.load_task", return_value=task):
            run = _run(run_quick_task(vllm_client, _BareTelemetry(), "some-model", "fake_task", 1))
        assert run.rapl_max_energy_range_uj is None
        assert run.rapl_dram_max_energy_range_uj is None


class TestRunPowerSweep:
    """`caps_w` is passed explicitly in these tests (the escape hatch that
    keeps working unchanged for explicit points) so they exercise the
    confirm/skip/partial loop in isolation from derivation, which
    `TestDerivePowerSweepCapsW` and `TestRunPowerSweepDerivesFromCard` below
    cover directly."""

    def test_first_cap_failing_skips_entire_sweep(self):
        telemetry = _FakeTelemetry(_samples(2))
        telemetry.set_power_limit_w = AsyncMock(return_value=None)
        vllm_client = AsyncMock()

        points, reason = _run(
            run_power_sweep(vllm_client, telemetry, "some-model", n_items=25, caps_w=[280, 250, 225])
        )

        assert points == []
        assert reason is not None
        assert "SetPowerManagementLimit" in reason
        telemetry.set_power_limit_w.assert_awaited_once()  # only tried the first cap

    def test_all_caps_succeed(self):
        telemetry = _FakeTelemetry(_samples(2))
        telemetry.set_power_limit_w = AsyncMock(side_effect=[280, 250, 225])
        fake_run = MagicMock()

        with patch(
            "hmasync_controller.bench.quick.run_quick_task", new=AsyncMock(return_value=fake_run)
        ) as run_task_mock:
            points, reason = _run(
                run_power_sweep(AsyncMock(), telemetry, "some-model", n_items=25, caps_w=[280, 250, 225])
            )

        assert reason is None
        assert [p.confirmed_w for p in points] == [280, 250, 225]
        assert run_task_mock.await_count == 3

    def test_partial_sweep_keeps_earlier_points(self):
        """A LATER cap failing (not the first) keeps what already ran --
        partial sweep beats none, distinct from the first-cap-fails case."""
        telemetry = _FakeTelemetry(_samples(2))
        telemetry.set_power_limit_w = AsyncMock(side_effect=[280, None])
        fake_run = MagicMock()

        with patch(
            "hmasync_controller.bench.quick.run_quick_task", new=AsyncMock(return_value=fake_run)
        ):
            points, reason = _run(
                run_power_sweep(AsyncMock(), telemetry, "some-model", n_items=25, caps_w=[280, 250, 225])
            )

        assert reason is None
        assert len(points) == 1
        assert points[0].confirmed_w == 280

    def test_unreadable_stock_limit_skips_entirely_no_fixed_fallback(self):
        """When the stock limit can't be read, the sweep must be skipped --
        never silently fall back to a fixed wattage ladder that might not
        even fit the card."""
        telemetry = _FakeTelemetry(_samples(2), stock_w=None)
        telemetry.set_power_limit_w = AsyncMock()

        points, reason = _run(run_power_sweep(AsyncMock(), telemetry, "some-model", n_items=25))

        assert points == []
        assert reason is not None
        assert "stock power limit" in reason
        telemetry.set_power_limit_w.assert_not_awaited()


class TestDerivePowerSweepCapsW:
    """`_derive_power_sweep_caps_w` pure-function tests -- ported verbatim,
    the arithmetic is unchanged from energy-bench's US-COMM-04."""

    def test_320w_card_320w_3090_class(self):
        caps = _derive_power_sweep_caps_w(320, None, None)
        assert caps == [272, 240, 208]
        assert len(set(caps)) == len(caps)  # all distinct

    def test_575w_card_575w_5090_class(self):
        caps = _derive_power_sweep_caps_w(575, None, None)
        assert caps == [489, 431, 374]
        assert len(set(caps)) == len(caps)

    def test_115w_laptop_card_all_points_below_stock(self):
        caps = _derive_power_sweep_caps_w(115, None, None)
        assert caps == [98, 86, 75]
        assert all(w < 115 for w in caps)
        assert len(set(caps)) == len(caps)

    def test_colliding_points_after_clamping_are_dropped(self):
        caps = _derive_power_sweep_caps_w(200, min_w=150, max_w=170)
        assert caps == [170, 150]

    def test_unknown_range_derives_unclamped(self):
        caps = _derive_power_sweep_caps_w(300, None, None)
        assert caps == [255, 225, 195]

    def test_a_card_with_no_headroom_below_stock_derives_nothing(self):
        assert _derive_power_sweep_caps_w(115, min_w=115, max_w=115) == []

    def test_points_clamped_up_to_a_high_floor_collapse_to_the_survivors(self):
        assert _derive_power_sweep_caps_w(115, min_w=100, max_w=115) == [100]


class TestRunPowerSweepDerivesFromCard:
    """End-to-end through `run_power_sweep` (no `caps_w` override) --
    proves the derived watts are what actually gets requested."""

    def test_320w_card(self):
        telemetry = _FakeTelemetry(_samples(2), stock_w=320)
        telemetry.set_power_limit_w = AsyncMock(side_effect=[272, 240, 208])
        fake_run = MagicMock()

        with patch(
            "hmasync_controller.bench.quick.run_quick_task", new=AsyncMock(return_value=fake_run)
        ):
            points, reason = _run(
                run_power_sweep(AsyncMock(), telemetry, "some-model", n_items=25)
            )

        assert reason is None
        assert [p.requested_w for p in points] == [272, 240, 208]

    def test_575w_card(self):
        telemetry = _FakeTelemetry(_samples(2), stock_w=575)
        telemetry.set_power_limit_w = AsyncMock(side_effect=[489, 431, 374])
        fake_run = MagicMock()

        with patch(
            "hmasync_controller.bench.quick.run_quick_task", new=AsyncMock(return_value=fake_run)
        ):
            points, reason = _run(
                run_power_sweep(AsyncMock(), telemetry, "some-model", n_items=25)
            )

        assert reason is None
        assert [p.requested_w for p in points] == [489, 431, 374]

    def test_115w_laptop_card(self):
        telemetry = _FakeTelemetry(_samples(2), stock_w=115)
        telemetry.set_power_limit_w = AsyncMock(side_effect=[98, 86, 75])
        fake_run = MagicMock()

        with patch(
            "hmasync_controller.bench.quick.run_quick_task", new=AsyncMock(return_value=fake_run)
        ):
            points, reason = _run(
                run_power_sweep(AsyncMock(), telemetry, "some-model", n_items=25)
            )

        assert reason is None
        assert [p.requested_w for p in points] == [98, 86, 75]
        assert all(w < 115 for w in (p.requested_w for p in points))

    def test_clamps_into_known_supported_range(self):
        telemetry = _FakeTelemetry(_samples(2), stock_w=575, min_w=300, max_w=450)
        telemetry.set_power_limit_w = AsyncMock(side_effect=[450, 431, 374])
        fake_run = MagicMock()

        with patch(
            "hmasync_controller.bench.quick.run_quick_task", new=AsyncMock(return_value=fake_run)
        ):
            points, reason = _run(
                run_power_sweep(AsyncMock(), telemetry, "some-model", n_items=25)
            )

        assert reason is None
        # 0.85*575=489 clamped down to max_w=450; the other two are already
        # inside [300, 450].
        assert [p.requested_w for p in points] == [450, 431, 374]


class TestQuickModelResolution:
    def test_llamacpp_gguf_spec_matches_verified_repo(self):
        spec = QUICK_REFERENCE_MODELS["llama.cpp"]
        assert spec["gguf_repo"] == "unsloth/Qwen3.5-9B-GGUF"
        assert spec["gguf_file"] == "Qwen3.5-9B-Q4_K_M.gguf"
        assert len(spec["revision"]) == 40  # a real git commit sha

    def test_ollama_tag_matches_verified_library_listing(self):
        assert QUICK_REFERENCE_MODELS["ollama"]["tag"] == "qwen3.5:9b-q4_K_M"

    def test_default_ports_match_engine_adapters(self):
        assert DEFAULT_OLLAMA_PORT == 11434
        assert DEFAULT_LLAMACPP_PORT == 8080


# ============================================================
# run_quick_suite (US-MERGE-04) -- the new orchestrator this story adds.
# Every collaborator (detect_engine, resolve_quick_model, LocalNvmlSampler,
# VLLMClient, run_quick_task) is mocked at the module-namespace level, same
# pattern the classes above already use.
# ============================================================


def _suite_engine_and_model():
    adapter = _adapter_mock(ready=True, base_url="http://localhost:11434")
    engine = DetectedEngine(name="ollama", base_url="http://localhost:11434", adapter=adapter)
    model = QuickModel(
        name="qwen3.5:9b-q4_K_M", note="ollama:qwen3.5:9b-q4_K_M",
        record_model="Qwen/Qwen3.5-9B", record_quantization="Q4_K_M",
    )
    return engine, model


def _suite_telemetry(*, stock_w=None, min_w=None, max_w=None):
    telemetry = MagicMock()
    telemetry.gpu_info = AsyncMock(
        return_value={
            "gpu_name": "NVIDIA GeForce RTX 3090", "gpu_mem_total_mib": 24576.0,
            "driver_version": "550.90.07", "cuda_version": "12.4",
        }
    )
    telemetry.get_power_limit_w = AsyncMock(return_value=stock_w)
    telemetry.get_power_limit_constraints_w = AsyncMock(return_value=(min_w, max_w))
    # Echoes the requested watts back, like a real NVML readback confirming
    # an already-clamped request -- NOT a fixed `stock_w` return, which would
    # make every "confirmed" cap equal to stock regardless of what was asked.
    telemetry.set_power_limit_w = AsyncMock(side_effect=lambda watts: watts)
    telemetry.close = MagicMock()
    return telemetry


def _fake_task_run(task_name: str, n_items: int, *, power_limit_w=None, n_shot=None, seed=None) -> QuickTaskRun:
    return QuickTaskRun(
        task_name=task_name, task_shape="decode", is_canary=False, dataset_revision=None,
        n_items=n_items, n_shot=n_shot or QUICK_REFERENCE_N_SHOT, seed=seed or QUICK_REFERENCE_SEED,
        max_tokens=64, power_limit_w=power_limit_w,
        inference_results=[_inference_result()],
        telemetry_samples=_samples(2),
        streaming_used=True,
    )


class TestRunQuickSuite:
    def test_no_engine_detected_raises(self):
        async def _none(*a, **kw):
            return None

        with patch("hmasync_controller.bench.quick.detect_engine", _none):
            with pytest.raises(NoEngineDetectedError):
                _run(run_quick_suite())

    def test_model_not_available_propagates(self):
        engine, _model = _suite_engine_and_model()

        async def _detect(*a, **kw):
            return engine

        async def _resolve(*a, **kw):
            raise ModelNotAvailableError("Run `ollama pull qwen3.5:9b-q4_K_M` first.")

        with (
            patch("hmasync_controller.bench.quick.detect_engine", _detect),
            patch("hmasync_controller.bench.quick.resolve_quick_model", _resolve),
        ):
            with pytest.raises(ModelNotAvailableError):
                _run(run_quick_suite())

    def test_nvml_unavailable_propagates(self):
        engine, model = _suite_engine_and_model()
        telemetry = MagicMock()
        telemetry.gpu_info = AsyncMock(side_effect=NvmlUnavailableError("no NVIDIA GPU on this box"))

        async def _detect(*a, **kw):
            return engine

        async def _resolve(*a, **kw):
            return model

        with (
            patch("hmasync_controller.bench.quick.detect_engine", _detect),
            patch("hmasync_controller.bench.quick.resolve_quick_model", _resolve),
            patch("hmasync_controller.bench.quick.LocalNvmlSampler", return_value=telemetry),
            patch("hmasync_controller.bench.quick.VLLMClient"),
        ):
            with pytest.raises(NvmlUnavailableError):
                _run(run_quick_suite())

    def test_all_tasks_failed_raises_and_still_restores_power_limit(self):
        """The restore-in-finally guarantee: even when every task raises
        (nothing to build a bundle from), the original power limit is still
        restored and telemetry is still closed -- the `finally` block must
        fire on the way OUT via an exception too, not only on success."""
        engine, model = _suite_engine_and_model()
        telemetry = _suite_telemetry(stock_w=300)

        async def _detect(*a, **kw):
            return engine

        async def _resolve(*a, **kw):
            return model

        async def _always_fail(*a, **kw):
            raise RuntimeError("engine went away mid-task")

        with (
            patch("hmasync_controller.bench.quick.detect_engine", _detect),
            patch("hmasync_controller.bench.quick.resolve_quick_model", _resolve),
            patch("hmasync_controller.bench.quick.LocalNvmlSampler", return_value=telemetry),
            patch("hmasync_controller.bench.quick.VLLMClient"),
            patch("hmasync_controller.bench.quick.run_quick_task", new=AsyncMock(side_effect=_always_fail)),
        ):
            with pytest.raises(AllTasksFailedError):
                _run(run_quick_suite())

        telemetry.set_power_limit_w.assert_awaited_once_with(300)
        telemetry.close.assert_called_once()

    def test_success_pairs_every_run_with_its_task_run_in_order(self):
        """No power sweep in this scenario (stock limit unreadable) -- keeps
        the assertion to exactly `len(QUICK_TASKS)` pairs, one per task."""
        engine, model = _suite_engine_and_model()
        telemetry = _suite_telemetry(stock_w=None)  # sweep skips: stock unreadable

        async def _detect(*a, **kw):
            return engine

        async def _resolve(*a, **kw):
            return model

        async def _fake_run_quick_task(vllm_client, telemetry_arg, model_name, task_name, n_items, **kw):
            return _fake_task_run(task_name, n_items, **kw)

        with (
            patch("hmasync_controller.bench.quick.detect_engine", _detect),
            patch("hmasync_controller.bench.quick.resolve_quick_model", _resolve),
            patch("hmasync_controller.bench.quick.LocalNvmlSampler", return_value=telemetry),
            patch("hmasync_controller.bench.quick.VLLMClient"),
            patch(
                "hmasync_controller.bench.quick.run_quick_task",
                new=AsyncMock(side_effect=_fake_run_quick_task),
            ),
        ):
            result = _run(run_quick_suite())

        assert result.engine_name == "ollama"
        assert result.model is model
        assert result.power_sweep_skipped_reason is not None
        assert len(result.runs) == len(QUICK_TASKS)
        assert len(result.task_runs) == len(result.runs)
        for run_metrics, task_run in zip(result.runs, result.task_runs, strict=True):
            assert run_metrics.task == task_run.task_name
            assert run_metrics.model == model.record_model
            assert run_metrics.quantization == model.record_quantization
        # Original power limit was never readable, so nothing to restore --
        # `set_power_limit_w` is called only inside a (skipped, in this
        # case) sweep, never here.
        telemetry.set_power_limit_w.assert_not_awaited()
        telemetry.close.assert_called_once()

    def test_success_includes_sweep_points_paired_with_their_own_task_runs(self):
        """A readable, clampable stock limit: the mini power sweep runs for
        real (3 derived caps, all confirmed), and each capped point's
        RunMetrics/QuickTaskRun pair lands alongside the 3 baseline tasks."""
        engine, model = _suite_engine_and_model()
        telemetry = _suite_telemetry(stock_w=300, min_w=100, max_w=350)

        async def _detect(*a, **kw):
            return engine

        async def _resolve(*a, **kw):
            return model

        async def _fake_run_quick_task(vllm_client, telemetry_arg, model_name, task_name, n_items, **kw):
            return _fake_task_run(task_name, n_items, **kw)

        with (
            patch("hmasync_controller.bench.quick.detect_engine", _detect),
            patch("hmasync_controller.bench.quick.resolve_quick_model", _resolve),
            patch("hmasync_controller.bench.quick.LocalNvmlSampler", return_value=telemetry),
            patch("hmasync_controller.bench.quick.VLLMClient"),
            patch(
                "hmasync_controller.bench.quick.run_quick_task",
                new=AsyncMock(side_effect=_fake_run_quick_task),
            ),
        ):
            result = _run(run_quick_suite())

        # 3 QUICK_TASKS + 3 derived capped points (0.85/0.75/0.65 of 300 W,
        # all distinct and below stock, all within [100, 350]) = 6.
        assert len(result.runs) == len(QUICK_TASKS) + 3
        assert len(result.task_runs) == len(result.runs)
        capped_runs = [r for r in result.runs if r.power_limit_w is not None]
        assert sorted(r.power_limit_w for r in capped_runs) == [195, 225, 255]
        assert result.power_sweep_skipped_reason is None

        # Restore-in-finally: the 3 sweep caps, then the original 300 W last.
        assert telemetry.set_power_limit_w.await_count == 4
        assert telemetry.set_power_limit_w.await_args_list[-1].args == (300,)
        telemetry.close.assert_called_once()
