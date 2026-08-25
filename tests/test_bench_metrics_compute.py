"""Unit tests for the bench metrics compute module (US-MERGE-03, ported from
energy-bench's tests/unit/test_compute.py)."""

import pytest

from hmasync_controller.bench.metrics.compute import (
    MetricsComputeError,
    compute_cpu_dram_energy,
    compute_metrics,
    compute_streaming_latency,
)
from hmasync_controller.bench.metrics.models import SCHEMA_VERSION, InferenceResult, WallPowerSample
from hmasync_controller.bench.sampler import TelemetrySample


def _basic_inputs():
    """Minimal valid (samples, inference_results) for compute_metrics."""
    samples = [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
        )
        for i in range(3)
    ]
    inference_results = [
        InferenceResult(
            request_id="r", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=2.0, tokens_per_second=50.0,
        )
    ]
    return samples, inference_results


def test_wall_rollups_and_humidity():
    """Wall-power samples produce mean/peak rollups; humidity flows through."""
    samples, inference_results = _basic_inputs()
    wall = [
        WallPowerSample(ts=1000.0, wall_power_w=300.0),
        WallPowerSample(ts=1001.0, wall_power_w=360.0),
        WallPowerSample(ts=1002.0, wall_power_w=240.0),
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5, ambient_rh_pct_start=48.0, wall_samples=wall,
    )
    assert metrics.ambient_rh_pct_start == 48.0
    assert metrics.peak_wall_w == 360.0
    assert metrics.mean_wall_w == pytest.approx(300.0)  # (300+360+240)/3


def test_wall_rollups_none_when_no_samples():
    """No wall samples / no humidity -> the new fields default to None."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.ambient_rh_pct_start is None
    assert metrics.mean_wall_w is None
    assert metrics.peak_wall_w is None


def test_repeat_index_defaults_to_zero():
    """A cell without n_repeats produces metrics with repeat_index 0."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.repeat_index == 0


def test_repeat_index_passes_through():
    """repeat_index flows from the caller straight onto RunMetrics."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5, repeat_index=2,
    )
    assert metrics.repeat_index == 2


def test_pooled_tokens_per_second_weights_by_duration():
    """Pooled throughput weights by request duration, unlike the mean of rates.

    Two requests: one short/fast (10 tokens / 0.1s = 100 tok/s), one
    long/slow (100 tokens / 10s = 10 tok/s). The mean of rates is 55 tok/s;
    pooled is 110 tokens / 10.1s ~= 10.9 tok/s, dominated by the slow request
    since it consumed nearly all the wall time.
    """
    samples = [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
        )
        for i in range(3)
    ]
    inference_results = [
        InferenceResult(
            request_id="fast", prompt_tokens=5, completion_tokens=10,
            ttft_s=0.05, total_s=0.1, tokens_per_second=100.0,
        ),
        InferenceResult(
            request_id="slow", prompt_tokens=5, completion_tokens=100,
            ttft_s=0.5, total_s=10.0, tokens_per_second=10.0,
        ),
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.mean_tokens_per_second == pytest.approx(55.0)
    assert metrics.pooled_tokens_per_second == pytest.approx(110 / 10.1)


def test_pooled_tokens_per_second_zero_when_no_duration():
    """All-zero total_s (e.g. a degenerate fixture) guards against div-by-zero."""
    samples, _ = _basic_inputs()
    inference_results = [
        InferenceResult(
            request_id="r", prompt_tokens=5, completion_tokens=10,
            ttft_s=0.0, total_s=0.0, tokens_per_second=0.0,
        ),
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.pooled_tokens_per_second == 0.0


def test_accuracy_and_n_correct_from_bool_scores():
    """Bool `correct` values produce the familiar fraction-correct accuracy
    and an exact n_correct tally (the pre-existing, unchanged behavior)."""
    samples, _ = _basic_inputs()
    inference_results = [
        InferenceResult(
            request_id=f"r{i}", prompt_tokens=5, completion_tokens=5,
            ttft_s=0.1, total_s=1.0, tokens_per_second=5.0, correct=c,
        )
        for i, c in enumerate([True, True, False, None])
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.n_items == 4
    assert metrics.n_correct == 2
    assert metrics.accuracy == pytest.approx(2 / 3)  # None is excluded, not a 0


def test_accuracy_and_n_correct_from_continuous_scores():
    """A continuous-scored task (e.g. longctx_summary's ROUGE-L F1) stores a
    float `correct` per item; accuracy becomes the mean of those floats
    rather than a fraction of exact matches."""
    samples, _ = _basic_inputs()
    inference_results = [
        InferenceResult(
            request_id=f"r{i}", prompt_tokens=9000, completion_tokens=200,
            ttft_s=1.0, total_s=5.0, tokens_per_second=40.0, correct=c,
        )
        for i, c in enumerate([0.8, 0.4, 0.2])
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.accuracy == pytest.approx((0.8 + 0.4 + 0.2) / 3)
    assert metrics.n_correct == round(0.8 + 0.4 + 0.2)  # accuracy-weighted, not a tally


def test_accuracy_none_when_nothing_scored():
    """Every result unscored (None) -> accuracy stays None, n_correct stays 0."""
    samples, inference_results = _basic_inputs()
    inference_results[0].correct = None
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.accuracy is None
    assert metrics.n_correct == 0


def test_truncated_pct_counts_length_finish_reason():
    """truncated_pct is the percent of items that hit finish_reason == 'length'."""
    samples, _ = _basic_inputs()
    inference_results = [
        InferenceResult(
            request_id=f"r{i}", prompt_tokens=5, completion_tokens=5,
            ttft_s=0.1, total_s=1.0, tokens_per_second=5.0, finish_reason=fr,
        )
        for i, fr in enumerate(["length", "stop", "length", None])
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.truncated_pct == pytest.approx(50.0)  # 2 of 4


def test_truncated_pct_zero_not_none_when_nothing_truncated():
    """finish_reason absent on every item -> 0.0 (a real measurement), never None."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.truncated_pct == 0.0


def test_provenance_fields_default_to_none():
    """Run metadata defaults to None/base values when unset."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.schema_version == SCHEMA_VERSION
    assert metrics.engine == "vllm"
    assert metrics.engine_version is None
    assert metrics.driver_version is None
    assert metrics.cuda_version is None
    assert metrics.gpu_name is None
    assert metrics.has_vision_tower is False


def test_provenance_fields_pass_through():
    """Run metadata flows from the caller straight onto RunMetrics."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
        engine="llama.cpp",
        engine_version="0.6.3.post1",
        driver_version="550.90.07",
        cuda_version="12.4",
        gpu_name="NVIDIA GeForce RTX 4090",
        has_vision_tower=True,
    )
    assert metrics.engine == "llama.cpp"
    assert metrics.engine_version == "0.6.3.post1"
    assert metrics.driver_version == "550.90.07"
    assert metrics.cuda_version == "12.4"
    assert metrics.gpu_name == "NVIDIA GeForce RTX 4090"
    assert metrics.has_vision_tower is True


def test_power_limit_w_defaults_to_none():
    """No cap applied (stock run) -> power_limit_w stays None."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.power_limit_w is None


def test_power_limit_w_passes_through():
    """The confirmed cap from the collector flows onto RunMetrics."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
        power_limit_w=280,
    )
    assert metrics.power_limit_w == 280


def test_clock_lock_mhz_defaults_to_none():
    """No lock applied -> clock_lock_mhz stays None."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.clock_lock_mhz is None


def test_clock_lock_mhz_passes_through():
    """The confirmed lock from the collector flows onto RunMetrics."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
        clock_lock_mhz=1400,
    )
    assert metrics.clock_lock_mhz == 1400


def _samples_with_sm_clock(sm_clock_mhz):
    """Same shape as `_basic_inputs()`'s samples, plus a fixed SM clock reading."""
    return [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, gpu_sm_clock_mhz=sm_clock_mhz,
        )
        for i in range(3)
    ]


def test_clock_lock_achieved_none_when_no_lock_requested():
    """No lock requested -> clock_lock_achieved stays None regardless of telemetry."""
    samples = _samples_with_sm_clock(1400)
    _, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.clock_lock_mhz is None
    assert metrics.clock_lock_achieved is None


def test_clock_lock_achieved_none_when_sm_clock_unavailable():
    """No gpu_sm_clock_mhz telemetry -> withheld, not assumed False."""
    samples, inference_results = _basic_inputs()  # no gpu_sm_clock_mhz on these samples
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
        clock_lock_mhz=1400,
    )
    assert metrics.clock_lock_achieved is None


def test_clock_lock_achieved_true_within_five_percent():
    """Measured mean SM clock within 5% of the requested lock -> achieved."""
    samples = _samples_with_sm_clock(1380)  # 1.4% below the 1400 MHz request
    _, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
        clock_lock_mhz=1400,
    )
    assert metrics.clock_lock_achieved is True


def test_clock_lock_achieved_false_outside_five_percent():
    """Measured SM clock diverges from the requested lock -> a suspect cell."""
    samples = _samples_with_sm_clock(1200)  # ~14.3% below the 1400 MHz request
    _, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
        clock_lock_mhz=1400,
    )
    assert metrics.clock_lock_achieved is False


def test_is_canary_defaults_to_false():
    """A free-text probe or a graded task -> is_canary stays False."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.is_canary is False


def test_is_canary_passes_through():
    """A canary task's flag flows onto RunMetrics."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
        task="hellaswag", task_shape="prefill", is_canary=True,
    )
    assert metrics.is_canary is True


def test_costmodel_fields_default_none_without_counter_or_timestamps():
    """_basic_inputs() has no gpu_energy_mj and no t_start_s/t_end_s -> the
    cost-model fit has nothing to interpolate against."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.alpha_j_per_prompt_token is None
    assert metrics.beta_j_per_completion_token is None
    assert metrics.e_fixed_j is None
    assert metrics.costmodel_r2 is None
    assert metrics.costmodel_n is None


def test_costmodel_fields_wired_from_fit():
    """With a gpu_energy_mj counter series and per-item timestamps, the fit
    runs and its results land on RunMetrics. Uses a known linear energy
    model so this also confirms compute_metrics threads its own
    samples/inference_results into the fit unmodified, not just that *some*
    numbers come back — the fit's edge cases are covered in depth by
    tests/test_bench_metrics_costmodel.py."""
    true_e_fixed, true_alpha, true_beta = 5.0, 0.02, 0.08
    items = [(50, 20), (200, 90), (80, 40), (150, 60)]

    samples = [
        TelemetrySample(
            ts=0.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, gpu_energy_mj=100_000.0,
        )
    ]
    inference_results = []
    cumulative_mj = 100_000.0
    t = 0.0
    for i, (prompt, completion) in enumerate(items):
        t_start, t_end = t + 1.0, t + 2.0
        energy_j = true_e_fixed + true_alpha * prompt + true_beta * completion
        samples.append(
            TelemetrySample(
                ts=t_start, gpu_power_w=200.0, gpu_util_pct=80.0,
                gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, gpu_energy_mj=cumulative_mj,
            )
        )
        cumulative_mj += energy_j * 1000.0
        samples.append(
            TelemetrySample(
                ts=t_end, gpu_power_w=200.0, gpu_util_pct=80.0,
                gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, gpu_energy_mj=cumulative_mj,
            )
        )
        inference_results.append(
            InferenceResult(
                request_id=f"r{i}", prompt_tokens=prompt, completion_tokens=completion,
                ttft_s=0.1, total_s=1.0, tokens_per_second=float(completion),
                t_start_s=t_start, t_end_s=t_end,
            )
        )
        t = t_end + 1.0

    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.costmodel_n == 4
    assert metrics.alpha_j_per_prompt_token == pytest.approx(true_alpha, rel=0.05)
    assert metrics.beta_j_per_completion_token == pytest.approx(true_beta, rel=0.05)
    assert metrics.e_fixed_j == pytest.approx(true_e_fixed, rel=0.05)
    assert metrics.costmodel_r2 > 0.99


def _ci_fixture(correct_flags):
    """Same known-linear-energy shape as `test_costmodel_fields_wired_from_fit`,
    plus a `correct` score per item so both within-run CIs have something to
    attribute."""
    true_e_fixed, true_alpha, true_beta = 5.0, 0.02, 0.08
    items = [(50, 20), (200, 90), (80, 40), (150, 60)]
    samples = [
        TelemetrySample(
            ts=0.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, gpu_energy_mj=100_000.0,
        )
    ]
    inference_results = []
    cumulative_mj = 100_000.0
    t = 0.0
    for i, ((prompt, completion), correct) in enumerate(zip(items, correct_flags)):
        t_start, t_end = t + 1.0, t + 2.0
        energy_j = true_e_fixed + true_alpha * prompt + true_beta * completion
        samples.append(
            TelemetrySample(
                ts=t_start, gpu_power_w=200.0, gpu_util_pct=80.0,
                gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, gpu_energy_mj=cumulative_mj,
            )
        )
        cumulative_mj += energy_j * 1000.0
        samples.append(
            TelemetrySample(
                ts=t_end, gpu_power_w=200.0, gpu_util_pct=80.0,
                gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, gpu_energy_mj=cumulative_mj,
            )
        )
        inference_results.append(
            InferenceResult(
                request_id=f"r{i}", prompt_tokens=prompt, completion_tokens=completion,
                ttft_s=0.1, total_s=1.0, tokens_per_second=float(completion),
                t_start_s=t_start, t_end_s=t_end, correct=correct,
            )
        )
        t = t_end + 1.0
    return samples, inference_results


def test_ci_fields_none_without_energy_or_correctness():
    """_basic_inputs() has no gpu_energy_mj and no correctness scores, so both
    within-run CIs stay None."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.jpc_ci_low is None
    assert metrics.jpc_ci_high is None
    assert metrics.accuracy_ci_low is None
    assert metrics.accuracy_ci_high is None


def test_accuracy_ci_computed_without_energy_counter():
    """Accuracy CI only needs correctness, not the energy counter -- it must
    not be withheld just because jpc_ci_* is (no gpu_energy_mj anywhere)."""
    samples, _ = _basic_inputs()
    inference_results = [
        InferenceResult(
            request_id=f"r{i}", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=2.0, tokens_per_second=50.0, correct=(i % 2 == 0),
        )
        for i in range(10)
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.jpc_ci_low is None
    assert metrics.jpc_ci_high is None
    assert metrics.accuracy_ci_low is not None
    assert metrics.accuracy_ci_high is not None
    assert metrics.accuracy_ci_low <= metrics.accuracy <= metrics.accuracy_ci_high


def test_ci_fields_wired_when_attribution_available():
    """With a gpu_energy_mj counter series, per-item timestamps, and
    correctness scores, both within-run CIs bracket their point estimate."""
    samples, inference_results = _ci_fixture([True, False, True, True])
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5, seed=1234,
    )
    assert metrics.jpc_ci_low is not None
    assert metrics.jpc_ci_high is not None
    assert metrics.jpc_ci_low <= metrics.joules_per_correct_answer <= metrics.jpc_ci_high
    assert metrics.accuracy_ci_low is not None
    assert metrics.accuracy_ci_high is not None
    assert metrics.accuracy_ci_low <= metrics.accuracy <= metrics.accuracy_ci_high


def test_jpc_ci_deterministic_for_same_seed():
    """Same inputs + same seed -> identical bounds -- `seed` threads straight
    to `bootstrap_jpc_ci`, so two runs from the same `probe.seed` reproduce
    the same CI."""
    kwargs = dict(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5, seed=99,
    )
    samples_a, results_a = _ci_fixture([True, False, True, True])
    samples_b, results_b = _ci_fixture([True, False, True, True])
    metrics_a = compute_metrics(samples=samples_a, inference_results=results_a, **kwargs)
    metrics_b = compute_metrics(samples=samples_b, inference_results=results_b, **kwargs)
    assert metrics_a.jpc_ci_low == metrics_b.jpc_ci_low
    assert metrics_a.jpc_ci_high == metrics_b.jpc_ci_high


def test_compute_metrics_basic():
    """Test basic metrics computation with simple data."""
    samples = [
        TelemetrySample(
            ts=1000.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
        ),
        TelemetrySample(
            ts=1001.0, gpu_power_w=210.0, gpu_util_pct=85.0,
            gpu_mem_used_mib=8100.0, gpu_temp_c=66.0,
        ),
        TelemetrySample(
            ts=1002.0, gpu_power_w=190.0, gpu_util_pct=75.0,
            gpu_mem_used_mib=7900.0, gpu_temp_c=64.0,
        ),
    ]

    inference_results = [
        InferenceResult(
            request_id="req1", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=2.0, tokens_per_second=50.0,
        ),
    ]

    metrics = compute_metrics(
        run_id="test_run_1", label="test_label", model="test/model",
        quantization=None, target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=10.5, kwh_after=10.6,
        ambient_c_start=22.5,
    )

    assert metrics.run_id == "test_run_1"
    assert metrics.label == "test_label"
    assert metrics.model == "test/model"
    assert metrics.quantization is None
    assert metrics.target_host == "192.168.0.114"
    assert metrics.kwh_delta == pytest.approx(0.1)
    assert metrics.peak_gpu_w == 210.0
    assert metrics.mean_gpu_w == 200.0  # (200 + 210 + 190) / 3
    assert metrics.mean_tokens_per_second == 50.0
    assert metrics.run_duration_s == 2.0  # 1002.0 - 1000.0
    assert metrics.ambient_c_start == 22.5
    assert metrics.total_completion_tokens == 100


def test_compute_metrics_total_completion_tokens():
    """Test that total_completion_tokens sums completion tokens across requests."""
    samples = [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
        )
        for i in range(4)
    ]

    inference_results = [
        InferenceResult(
            request_id="req1", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=2.0, tokens_per_second=50.0,
        ),
        InferenceResult(
            request_id="req2", prompt_tokens=60, completion_tokens=150,
            ttft_s=0.6, total_s=3.0, tokens_per_second=50.0,
        ),
        InferenceResult(
            request_id="req3", prompt_tokens=70, completion_tokens=250,
            ttft_s=0.7, total_s=5.0, tokens_per_second=50.0,
        ),
    ]

    metrics = compute_metrics(
        run_id="test_run_tokens", label="token_sum", model="test/model",
        quantization=None, target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=10.0, kwh_after=10.1,
        ambient_c_start=22.0,
    )

    # 100 + 150 + 250 = 500
    assert metrics.total_completion_tokens == 500
    # joules_per_token must divide by the same total
    assert metrics.joules_per_token == metrics.total_joules_gpu / 500


def test_compute_metrics_constant_power_trapezoidal():
    """Test trapezoidal integration with constant 200W over 10 seconds."""
    samples = [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
        )
        for i in range(11)
    ]

    inference_results = [
        InferenceResult(
            request_id="req1", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=10.0, tokens_per_second=10.0,
        ),
    ]

    metrics = compute_metrics(
        run_id="test_run_2", label="constant_power", model="test/model",
        quantization=None, target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=10.0, kwh_after=10.1,
        ambient_c_start=22.0,
    )

    # Trapezoidal integration: 10 intervals of 1 second each, avg power = 200W
    # Total energy = 10 * 1.0 * 200.0 = 2000.0 joules
    assert abs(metrics.total_joules_gpu - 2000.0) < 2000.0 * 0.01  # Within 1% tolerance
    assert metrics.joules_per_token == metrics.total_joules_gpu / 100  # 100 completion tokens
    assert metrics.peak_gpu_w == 200.0
    assert metrics.mean_gpu_w == 200.0


def test_compute_metrics_multiple_inference_results():
    """Test metrics with multiple inference results."""
    samples = [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
        )
        for i in range(6)
    ]

    inference_results = [
        InferenceResult(
            request_id="req1", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=2.0, tokens_per_second=50.0,
        ),
        InferenceResult(
            request_id="req2", prompt_tokens=60, completion_tokens=150,
            ttft_s=0.6, total_s=3.0, tokens_per_second=50.0,
        ),
    ]

    metrics = compute_metrics(
        run_id="test_run_3", label="multi_request", model="test/model",
        quantization="awq", target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=10.0, kwh_after=10.2,
        ambient_c_start=23.0,
    )

    # Total completion tokens = 100 + 150 = 250
    assert metrics.joules_per_token == metrics.total_joules_gpu / 250
    # Mean tokens per second = (50.0 + 50.0) / 2 = 50.0
    assert metrics.mean_tokens_per_second == 50.0
    assert metrics.quantization == "awq"


def test_compute_metrics_zero_tokens_raises():
    """Test that zero completion tokens raises MetricsComputeError."""
    samples = [
        TelemetrySample(
            ts=1000.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
        ),
        TelemetrySample(
            ts=1001.0, gpu_power_w=210.0, gpu_util_pct=85.0,
            gpu_mem_used_mib=8100.0, gpu_temp_c=66.0,
        ),
    ]

    inference_results = [
        InferenceResult(
            request_id="req1", prompt_tokens=50, completion_tokens=0,  # Zero completion tokens
            ttft_s=0.5, total_s=2.0, tokens_per_second=0.0,
        ),
    ]

    with pytest.raises(MetricsComputeError, match="Total completion tokens is zero"):
        compute_metrics(
            run_id="test_run_4", label="zero_tokens", model="test/model",
            quantization=None, target_host="192.168.0.114", samples=samples,
            inference_results=inference_results, kwh_before=10.0, kwh_after=10.1,
            ambient_c_start=22.0,
        )


def test_compute_metrics_no_samples_raises():
    """Test that empty samples list raises MetricsComputeError."""
    inference_results = [
        InferenceResult(
            request_id="req1", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=2.0, tokens_per_second=50.0,
        ),
    ]

    with pytest.raises(MetricsComputeError, match="No telemetry samples provided"):
        compute_metrics(
            run_id="test_run_5", label="no_samples", model="test/model",
            quantization=None, target_host="192.168.0.114", samples=[],
            inference_results=inference_results, kwh_before=10.0, kwh_after=10.1,
            ambient_c_start=22.0,
        )


def test_compute_metrics_no_inference_results_raises():
    """Test that empty inference results list raises MetricsComputeError."""
    samples = [
        TelemetrySample(
            ts=1000.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
        ),
        TelemetrySample(
            ts=1001.0, gpu_power_w=210.0, gpu_util_pct=85.0,
            gpu_mem_used_mib=8100.0, gpu_temp_c=66.0,
        ),
    ]

    with pytest.raises(MetricsComputeError, match="No inference results provided"):
        compute_metrics(
            run_id="test_run_6", label="no_results", model="test/model",
            quantization=None, target_host="192.168.0.114", samples=samples,
            inference_results=[], kwh_before=10.0, kwh_after=10.1,
            ambient_c_start=22.0,
        )


def test_compute_metrics_varying_power():
    """Test trapezoidal integration with varying power levels."""
    samples = [
        TelemetrySample(
            ts=1000.0, gpu_power_w=100.0, gpu_util_pct=50.0,
            gpu_mem_used_mib=6000.0, gpu_temp_c=60.0,
        ),
        TelemetrySample(
            ts=1001.0, gpu_power_w=200.0, gpu_util_pct=75.0,
            gpu_mem_used_mib=7000.0, gpu_temp_c=65.0,
        ),
        TelemetrySample(
            ts=1002.0, gpu_power_w=300.0, gpu_util_pct=95.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=70.0,
        ),
        TelemetrySample(
            ts=1003.0, gpu_power_w=200.0, gpu_util_pct=75.0,
            gpu_mem_used_mib=7000.0, gpu_temp_c=65.0,
        ),
        TelemetrySample(
            ts=1004.0, gpu_power_w=100.0, gpu_util_pct=50.0,
            gpu_mem_used_mib=6000.0, gpu_temp_c=60.0,
        ),
    ]

    inference_results = [
        InferenceResult(
            request_id="req1", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=4.0, tokens_per_second=25.0,
        ),
    ]

    metrics = compute_metrics(
        run_id="test_run_7", label="varying_power", model="test/model",
        quantization=None, target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=10.0, kwh_after=10.1,
        ambient_c_start=22.0,
    )

    # Trapezoidal integration: 150 + 250 + 250 + 150 = 800J
    expected_joules = 800.0
    assert abs(metrics.total_joules_gpu - expected_joules) < expected_joules * 0.01
    assert metrics.peak_gpu_w == 300.0
    assert metrics.mean_gpu_w == 180.0  # (100 + 200 + 300 + 200 + 100) / 5
    assert metrics.joules_per_token == metrics.total_joules_gpu / 100


def test_compute_metrics_kwh_delta():
    """Test that kWh delta is computed correctly."""
    samples = [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
        )
        for i in range(3)
    ]

    inference_results = [
        InferenceResult(
            request_id="req1", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=2.0, tokens_per_second=50.0,
        ),
    ]

    metrics = compute_metrics(
        run_id="test_run_8", label="kwh_test", model="test/model",
        quantization=None, target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=12.345, kwh_after=12.678,
        ambient_c_start=21.5,
    )

    assert metrics.kwh_delta == 12.678 - 12.345
    assert abs(metrics.kwh_delta - 0.333) < 0.001


def test_compute_metrics_wall_counter_ticks_five():
    """A 0.05 kWh delta is 5 meter ticks; kwh_delta is preserved."""
    samples, inference_results = _basic_inputs()

    metrics = compute_metrics(
        run_id="ticks_5", label="test_label", model="test/model",
        quantization=None, target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=10.00, kwh_after=10.05,
        ambient_c_start=22.5,
    )

    assert metrics.wall_counter_ticks == 5
    assert metrics.kwh_delta == pytest.approx(0.05)


def test_compute_metrics_wall_counter_ticks_zero_nulls_kwh_delta():
    """A delta below the 0.01 kWh meter resolution rounds to 0 ticks; the
    literal delta is not stored as a false-precision 0.0, it's None."""
    samples, inference_results = _basic_inputs()

    metrics = compute_metrics(
        run_id="ticks_0", label="test_label", model="test/model",
        quantization=None, target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=10.000, kwh_after=10.004,
        ambient_c_start=22.5,
    )

    assert metrics.wall_counter_ticks == 0
    assert metrics.kwh_delta is None


def test_compute_metrics_no_plug_leaves_ticks_and_delta_none():
    """No smart plug entity resolved: both kwh readings are None, so ticks
    stays None too (distinct from the 0-tick case above, where the plug
    resolved but the counter never advanced)."""
    samples, inference_results = _basic_inputs()

    metrics = compute_metrics(
        run_id="no_plug", label="test_label", model="test/model",
        quantization=None, target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=None, kwh_after=None,
        ambient_c_start=22.5,
    )

    assert metrics.wall_counter_ticks is None
    assert metrics.kwh_delta is None


def test_compute_metrics_with_cpu_rapl():
    """Test that metrics computation works with CPU RAPL data present."""
    samples = [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
            cpu_rapl_uj=1000000.0 + i * 50000,  # RAPL data present
        )
        for i in range(3)
    ]

    inference_results = [
        InferenceResult(
            request_id="req1", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=2.0, tokens_per_second=50.0,
        ),
    ]

    metrics = compute_metrics(
        run_id="test_run_9", label="rapl_test", model="test/model",
        quantization=None, target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=10.0, kwh_after=10.1,
        ambient_c_start=22.0,
    )

    # Verify metrics are computed correctly regardless of RAPL data
    assert metrics.total_joules_gpu > 0
    assert metrics.joules_per_token > 0
    # 3 readings, +50000 uj each step, no wrap -> 100000 uj = 0.1 J total.
    assert metrics.total_joules_cpu == pytest.approx(0.1)


def test_config_as_data_fields_default():
    """temperature/max_tokens/seed/n_shot/thinking_mode/dataset_revision all
    default sensibly when the caller doesn't pass them."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.temperature == 0.0
    assert metrics.max_tokens == 0
    assert metrics.seed == 0
    assert metrics.n_shot is None
    assert metrics.thinking_mode is None
    assert metrics.dataset_revision is None


def test_config_as_data_fields_pass_through():
    """Every config-as-data field flows from the caller straight onto
    RunMetrics."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
        temperature=0.7, max_tokens=1024, seed=42, n_shot=5,
        thinking_mode="enable_thinking=false", dataset_revision="refs/convert/parquet",
    )
    assert metrics.temperature == 0.7
    assert metrics.max_tokens == 1024
    assert metrics.seed == 42
    assert metrics.n_shot == 5
    assert metrics.thinking_mode == "enable_thinking=false"
    assert metrics.dataset_revision == "refs/convert/parquet"


def test_total_joules_cpu_none_when_rapl_absent():
    """No cpu_rapl_uj on any sample -> total_joules_cpu stays None."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.total_joules_cpu is None


def test_total_joules_cpu_corrects_wrap_via_rapl_max_energy_range():
    """A wrapped RAPL counter recovers the true delta when
    rapl_max_energy_range_uj is supplied."""
    samples = [
        TelemetrySample(
            ts=1000.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, cpu_rapl_uj=55_000_000.0,
        ),
        TelemetrySample(
            ts=1001.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, cpu_rapl_uj=5_000_000.0,
        ),
    ]
    inference_results = [
        InferenceResult(
            request_id="r", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=2.0, tokens_per_second=50.0,
        )
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5, rapl_max_energy_range_uj=60_000_000.0,
    )
    assert metrics.total_joules_cpu == pytest.approx(10.0)


def test_compute_cpu_dram_energy_none_when_no_readings():
    """Fewer than 2 dram readings (e.g. host has no dram RAPL zone) ->
    withheld, not guessed."""
    samples = [
        TelemetrySample(
            ts=1000.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, cpu_rapl_uj=None,
        ),
    ]
    assert compute_cpu_dram_energy(samples, None) is None


def test_compute_cpu_dram_energy_sums_deltas_without_wrap():
    samples = [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, cpu_rapl_uj=None,
            cpu_rapl_dram_uj=1_000_000.0 + i * 50_000,
        )
        for i in range(3)
    ]
    # 3 readings, +50000 uj each step, no wrap -> 100000 uj = 0.1 J total.
    assert compute_cpu_dram_energy(samples, None) == pytest.approx(0.1)


def test_compute_cpu_dram_energy_corrects_wrap_via_range():
    samples = [
        TelemetrySample(
            ts=1000.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, cpu_rapl_uj=None,
            cpu_rapl_dram_uj=55_000_000.0,
        ),
        TelemetrySample(
            ts=1001.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, cpu_rapl_uj=None,
            cpu_rapl_dram_uj=5_000_000.0,
        ),
    ]
    assert compute_cpu_dram_energy(samples, 60_000_000.0) == pytest.approx(10.0)


def test_compute_cpu_dram_energy_withheld_on_uncorrectable_wrap():
    """A wrap with no range supplied to correct it withholds the whole
    result rather than reporting a partially-corrected total."""
    samples = [
        TelemetrySample(
            ts=1000.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, cpu_rapl_uj=None,
            cpu_rapl_dram_uj=55_000_000.0,
        ),
        TelemetrySample(
            ts=1001.0, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0, cpu_rapl_uj=None,
            cpu_rapl_dram_uj=5_000_000.0,
        ),
    ]
    assert compute_cpu_dram_energy(samples, None) is None


def test_compute_metrics_total_joules_cpu_dram_none_when_absent():
    """No cpu_rapl_dram_uj on any sample -> total_joules_cpu_dram stays None,
    independently of the package-domain total_joules_cpu."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.total_joules_cpu_dram is None


def test_compute_metrics_total_joules_cpu_dram_populated():
    """cpu_rapl_dram_uj on samples flows through compute_metrics into
    RunMetrics.total_joules_cpu_dram, wrap-corrected via
    rapl_dram_max_energy_range_uj -- independent of the package-domain RAPL
    fields on the same samples."""
    samples = [
        TelemetrySample(
            ts=1000.0 + i, gpu_power_w=200.0, gpu_util_pct=80.0,
            gpu_mem_used_mib=8000.0, gpu_temp_c=65.0,
            cpu_rapl_uj=2_000_000.0 + i * 10_000,
            cpu_rapl_dram_uj=1_000_000.0 + i * 50_000,
        )
        for i in range(3)
    ]
    inference_results = [
        InferenceResult(
            request_id="req1", prompt_tokens=50, completion_tokens=100,
            ttft_s=0.5, total_s=2.0, tokens_per_second=50.0,
        ),
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=10.0, kwh_after=10.1,
        ambient_c_start=22.0,
    )
    assert metrics.total_joules_cpu == pytest.approx(0.02)
    assert metrics.total_joules_cpu_dram == pytest.approx(0.1)


def test_node_overhead_ratio_none_without_wall_data():
    """No wall-power samples -> node_overhead_ratio stays None."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.node_overhead_ratio is None


def test_node_overhead_ratio_is_mean_wall_over_mean_gpu():
    """node_overhead_ratio = mean_wall_w / mean_gpu_w when wall data exists."""
    samples, inference_results = _basic_inputs()  # mean_gpu_w == 200.0
    wall = [
        WallPowerSample(ts=1000.0, wall_power_w=250.0),
        WallPowerSample(ts=1001.0, wall_power_w=250.0),
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5, wall_samples=wall,
    )
    assert metrics.mean_gpu_w == pytest.approx(200.0)
    assert metrics.node_overhead_ratio == pytest.approx(1.25)


def test_measurement_tier_c_when_no_wall_samples():
    """No wall-power samples -> measurement_tier is 'C', the Tier-C default."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.measurement_tier == "C"


def test_measurement_tier_b_when_wall_samples_present():
    """Wall-power samples -> measurement_tier is 'B', set automatically from
    wall-sample presence alone -- not special-cased to any one caller."""
    samples, inference_results = _basic_inputs()
    wall = [
        WallPowerSample(ts=1000.0, wall_power_w=250.0),
        WallPowerSample(ts=1001.0, wall_power_w=250.0),
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5, wall_samples=wall,
    )
    assert metrics.measurement_tier == "B"


def test_ambient_c_start_none_accepted():
    """ambient_c_start is nullable: this package never queries Home
    Assistant, so it always passes None here rather than a hardcoded 0.0
    that would look like a real freezing-cold reading."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=None, kwh_after=None,
        ambient_c_start=None,
    )
    assert metrics.ambient_c_start is None
    assert metrics.measurement_tier == "C"


class TestComputeStreamingLatency:
    """Unit tests for compute_streaming_latency."""

    def _results(self, ttfts, itl_gaps):
        return [
            InferenceResult(
                request_id=f"r{i}", prompt_tokens=10, completion_tokens=5,
                ttft_s=ttft, total_s=1.0, tokens_per_second=5.0, itl_gaps_ms=gaps,
            )
            for i, (ttft, gaps) in enumerate(zip(ttfts, itl_gaps))
        ]

    def test_none_when_streaming_not_used(self):
        """A run with a fallback (streaming_used=False) reports all four
        fields as None even though real ttft/itl data is present -- a
        mix of measured and placeholder values can't give a real percentile."""
        results = self._results([0.1, 0.2, 0.3], [[10.0], [20.0], [30.0]])
        out = compute_streaming_latency(results, streaming_used=False)
        assert out == {
            "ttft_p50_s": None,
            "ttft_p95_s": None,
            "itl_mean_ms": None,
            "itl_p95_ms": None,
        }

    def test_none_when_no_inference_results(self):
        out = compute_streaming_latency([], streaming_used=True)
        assert out["ttft_p50_s"] is None
        assert out["itl_mean_ms"] is None

    def test_percentiles_computed_when_streaming_used(self):
        results = self._results([0.1, 0.2, 0.3, 0.4, 0.5], [[], [], [], [], []])
        out = compute_streaming_latency(results, streaming_used=True)
        assert out["ttft_p50_s"] == pytest.approx(0.3)
        assert out["ttft_p95_s"] == pytest.approx(0.48)

    def test_itl_gaps_pooled_across_requests(self):
        results = self._results([0.1, 0.1], [[10.0, 20.0], [30.0]])
        out = compute_streaming_latency(results, streaming_used=True)
        assert out["itl_mean_ms"] == pytest.approx(20.0)  # mean(10, 20, 30)
        assert out["itl_p95_ms"] == pytest.approx(29.0)

    def test_itl_none_when_no_requests_have_gaps(self):
        """Every request generated <=1 content token -> no gaps to pool, but
        ttft is still computed."""
        results = self._results([0.1, 0.2], [[], []])
        out = compute_streaming_latency(results, streaming_used=True)
        assert out["ttft_p50_s"] is not None
        assert out["itl_mean_ms"] is None
        assert out["itl_p95_ms"] is None


def test_compute_metrics_streaming_fields_default_false_and_none():
    """streaming_used defaults False and every percentile field stays None
    when compute_metrics is called without opting in."""
    samples, inference_results = _basic_inputs()
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5,
    )
    assert metrics.streaming_used is False
    assert metrics.ttft_p50_s is None
    assert metrics.ttft_p95_s is None
    assert metrics.itl_mean_ms is None
    assert metrics.itl_p95_ms is None


def test_compute_metrics_streaming_fields_wired_when_used():
    """streaming_used=True with real per-item ttft/itl data lands on
    RunMetrics via compute_streaming_latency."""
    samples, _ = _basic_inputs()
    inference_results = [
        InferenceResult(
            request_id="r1", prompt_tokens=10, completion_tokens=5,
            ttft_s=0.2, total_s=1.0, tokens_per_second=5.0, itl_gaps_ms=[15.0, 25.0],
        ),
        InferenceResult(
            request_id="r2", prompt_tokens=10, completion_tokens=5,
            ttft_s=0.4, total_s=1.0, tokens_per_second=5.0, itl_gaps_ms=[35.0],
        ),
    ]
    metrics = compute_metrics(
        run_id="r", label="l", model="m", quantization=None,
        target_host="192.168.0.114", samples=samples,
        inference_results=inference_results, kwh_before=1.0, kwh_after=1.05,
        ambient_c_start=22.5, streaming_used=True,
    )
    assert metrics.streaming_used is True
    assert metrics.ttft_p50_s == pytest.approx(0.3)  # midpoint of [0.2, 0.4]
    assert metrics.itl_mean_ms == pytest.approx(25.0)  # mean(15, 25, 35)


def test_no_numpy_import_in_metrics_package():
    """Acceptance criterion: the ported metrics package is pure Python.
    `rouge-score` (bench.tasks.longctx_summary, US-MERGE-01) pulls numpy
    transitively into this package's base install, so this checks the
    ported modules' own source -- not whether numpy is importable at all."""
    import ast
    import inspect

    from hmasync_controller.bench.metrics import compute, costmodel, derived, flexibility, stats

    for module in (compute, costmodel, derived, flexibility, stats):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            assert not any(n and n.split(".")[0] == "numpy" for n in names), (
                f"{module.__name__} imports numpy: {names}"
            )
