"""Derived metrics computation: J/token, J/correct answer, peak W, power shape.

Ported near-verbatim from energy-bench's `metrics/compute.py` (US-MERGE-03).
See `metrics/models.py`'s module docstring for how this package's
`RunMetrics` relates to energy-bench's authoritative one -- the function
below builds exactly the subset of fields this package's `RunMetrics`
carries; the lab-only fields (`model_type`, `params_b`,
`reasoning_mode_class`, `efficiency_index`) don't exist here because nothing
downstream of this package sets them.
"""

from __future__ import annotations

import statistics
from typing import Literal

from hmasync_controller.bench.metrics.costmodel import compute_item_energies_j, fit_cost_model
from hmasync_controller.bench.metrics.models import (
    THROTTLE_POWER_CAP_MASK,
    THROTTLE_THERMAL_MASK,
    InferenceResult,
    RunMetrics,
    WallPowerSample,
)
from hmasync_controller.bench.metrics.stats import accuracy_ci, bootstrap_jpc_ci
from hmasync_controller.bench.sampler import TelemetrySample

# Smart-plug meter resolution: readings advance in fixed 0.01 kWh increments,
# so a delta below this cannot be distinguished from "drew nothing" -- see
# `RunMetrics.wall_counter_ticks`.
_WALL_METER_RESOLUTION_KWH = 0.01


def _mean_of(values: list[float]) -> float | None:
    """Mean of the non-None readings, or None if a field was never populated."""
    return (sum(values) / len(values)) if values else None


def compute_hardware_health(
    samples: list[TelemetrySample], gpu_mem_total_mib: float | None = None
) -> dict[str, float | None]:
    """Roll the per-sample GPU series up into queryable run-level facts.

    Covers thermals, VRAM, utilization, clocks, and throttle validity. Every
    field degrades to None rather than raising when a channel wasn't
    reported, so a sample from an older/limited driver still loads.
    """
    out: dict[str, float | None] = {}
    if not samples:
        return out

    temps = [s.gpu_temp_c for s in samples]
    out["peak_gpu_temp_c"] = max(temps)
    out["mean_gpu_temp_c"] = sum(temps) / len(temps)

    mem = [s.gpu_mem_used_mib for s in samples]
    peak_mem = max(mem)
    out["peak_gpu_mem_used_mib"] = peak_mem
    out["mean_gpu_mem_used_mib"] = sum(mem) / len(mem)
    out["gpu_mem_used_pct_of_total"] = (
        100.0 * peak_mem / gpu_mem_total_mib if gpu_mem_total_mib else None
    )

    out["mean_gpu_util_pct"] = _mean_of([s.gpu_util_pct for s in samples])
    out["mean_gpu_mem_util_pct"] = _mean_of(
        [s.gpu_mem_util_pct for s in samples if s.gpu_mem_util_pct is not None]
    )
    out["mean_gpu_sm_clock_mhz"] = _mean_of(
        [float(s.gpu_sm_clock_mhz) for s in samples if s.gpu_sm_clock_mhz is not None]
    )
    out["mean_gpu_fan_pct"] = _mean_of(
        [float(s.gpu_fan_pct) for s in samples if s.gpu_fan_pct is not None]
    )

    # Throttle validity. Thermal and power-cap throttles are counted separately:
    # one invalidates the cell, the other is the point of a power-limit sweep.
    flagged = [s.gpu_throttle_reasons for s in samples if s.gpu_throttle_reasons is not None]
    if flagged:
        out["thermal_throttle_pct"] = 100.0 * sum(
            1 for r in flagged if r & THROTTLE_THERMAL_MASK
        ) / len(flagged)
        out["power_cap_throttle_pct"] = 100.0 * sum(
            1 for r in flagged if r & THROTTLE_POWER_CAP_MASK
        ) / len(flagged)
    else:
        out["thermal_throttle_pct"] = None
        out["power_cap_throttle_pct"] = None

    return out


def compute_counter_energy(
    samples: list[TelemetrySample], total_joules_integrated: float
) -> dict[str, float | None]:
    """Derive run energy from NVML's hardware counter, when available.

    The counter is cumulative since driver load, so the run's energy is just
    the last reading minus the first. This is exact, whereas integrating 5 Hz
    power samples approximates the curve between samples and misses anything
    faster than 200 ms. Reporting both, plus their difference, keeps the
    sampling error visible instead of assumed.
    """
    readings = [s.gpu_energy_mj for s in samples if s.gpu_energy_mj is not None]
    if len(readings) < 2:
        return {
            "total_joules_gpu_counter": None,
            "counter_vs_integration_pct_diff": None,
        }

    joules = (readings[-1] - readings[0]) / 1000.0
    if joules < 0:  # Counter reset (driver reload) mid-run; unusable.
        return {
            "total_joules_gpu_counter": None,
            "counter_vs_integration_pct_diff": None,
        }

    pct_diff: float | None = None
    if joules > 0:
        pct_diff = 100.0 * (total_joules_integrated - joules) / joules

    return {
        "total_joules_gpu_counter": joules,
        "counter_vs_integration_pct_diff": pct_diff,
    }


def _wrap_corrected_rapl_energy_j(
    readings: list[float], max_energy_range_uj: float | None
) -> float | None:
    """Shared wrap-correction arithmetic for any RAPL domain (package, dram):
    sum consecutive sample-to-sample deltas, adding `max_energy_range_uj` back
    per negative step so multiple wraps within one run are each corrected
    independently. Returns None when fewer than 2 readings were collected, or
    a wrap occurred but no range was supplied to correct it -- a
    partially-corrected total would be silently wrong, so the whole result is
    withheld rather than guessed at.
    """
    if len(readings) < 2:
        return None

    total_uj = 0.0
    for prev, cur in zip(readings, readings[1:]):
        delta = cur - prev
        if delta < 0:
            if max_energy_range_uj is None:
                return None
            delta += max_energy_range_uj
        total_uj += delta

    return total_uj / 1_000_000.0


def compute_cpu_energy(
    samples: list[TelemetrySample], rapl_max_energy_range_uj: float | None
) -> float | None:
    """Derive run CPU package energy from RAPL's cumulative counter,
    correcting wrap. Nothing in this package's `LocalNvmlSampler` populates
    `cpu_rapl_uj` today (see `TelemetrySample`'s docstring) -- this stays
    ported for parity and for a future local RAPL reader."""
    readings = [s.cpu_rapl_uj for s in samples if s.cpu_rapl_uj is not None]
    return _wrap_corrected_rapl_energy_j(readings, rapl_max_energy_range_uj)


def compute_cpu_dram_energy(
    samples: list[TelemetrySample], rapl_dram_max_energy_range_uj: float | None
) -> float | None:
    """DRAM-domain counterpart to `compute_cpu_energy`. Same "nothing sets
    this yet" caveat."""
    readings = [
        s.cpu_rapl_dram_uj for s in samples if s.cpu_rapl_dram_uj is not None
    ]
    return _wrap_corrected_rapl_energy_j(readings, rapl_dram_max_energy_range_uj)


class MetricsComputeError(Exception):
    """Raised when metrics computation fails (e.g., zero tokens)."""


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile over a pre-sorted list."""
    if not sorted_values:
        raise MetricsComputeError("Cannot take a percentile of an empty series")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def compute_power_shape(samples: list[TelemetrySample]) -> dict[str, float | None]:
    """Quantify how bursty ("lumpy") the GPU power draw was.

    These are all independent of total energy -- they describe the *shape*
    of the curve, not its area. Returns a dict of shape metrics; values are
    None when the series is too short to characterise (fewer than 2 samples).
    """
    empty: dict[str, float | None] = {
        "gpu_power_std_w": None,
        "gpu_power_cv": None,
        "gpu_power_crest_factor": None,
        "gpu_power_p95_p50_ratio": None,
        "gpu_power_jaggedness_w_per_s": None,
    }
    if len(samples) < 2:
        return empty

    watts = [s.gpu_power_w for s in samples]
    mean_w = sum(watts) / len(watts)
    std_w = statistics.stdev(watts)

    ordered = sorted(watts)
    p50 = _percentile(ordered, 0.50)
    p95 = _percentile(ordered, 0.95)

    # Mean absolute rate of change: how violently power swings sample-to-sample.
    deltas: list[float] = []
    for i in range(len(samples) - 1):
        dt = samples[i + 1].ts - samples[i].ts
        if dt > 0:
            deltas.append(abs(samples[i + 1].gpu_power_w - samples[i].gpu_power_w) / dt)

    return {
        "gpu_power_std_w": std_w,
        "gpu_power_cv": (std_w / mean_w) if mean_w > 0 else None,
        "gpu_power_crest_factor": (max(watts) / mean_w) if mean_w > 0 else None,
        "gpu_power_p95_p50_ratio": (p95 / p50) if p50 > 0 else None,
        "gpu_power_jaggedness_w_per_s": (sum(deltas) / len(deltas)) if deltas else None,
    }


def compute_streaming_latency(
    inference_results: list[InferenceResult], streaming_used: bool
) -> dict[str, float | None]:
    """Roll per-request TTFT and inter-token gaps into run-level percentiles.

    Only meaningful when every request in the run completed on the real
    streaming path -- a run where even one request fell back to the
    non-streaming path mixes measured TTFTs with that path's hardcoded 0.0
    placeholder, so all four fields collapse to None rather than reporting a
    skewed percentile.
    """
    empty: dict[str, float | None] = {
        "ttft_p50_s": None,
        "ttft_p95_s": None,
        "itl_mean_ms": None,
        "itl_p95_ms": None,
    }
    if not streaming_used or not inference_results:
        return empty

    ttfts = sorted(r.ttft_s for r in inference_results)
    itl_gaps_ms = sorted(gap for r in inference_results for gap in r.itl_gaps_ms)

    return {
        "ttft_p50_s": _percentile(ttfts, 0.50),
        "ttft_p95_s": _percentile(ttfts, 0.95),
        "itl_mean_ms": (sum(itl_gaps_ms) / len(itl_gaps_ms)) if itl_gaps_ms else None,
        "itl_p95_ms": _percentile(itl_gaps_ms, 0.95) if itl_gaps_ms else None,
    }


def compute_metrics(
    run_id: str,
    label: str,
    model: str,
    quantization: str | None,
    target_host: str,
    samples: list[TelemetrySample],
    inference_results: list[InferenceResult],
    kwh_before: float | None,
    kwh_after: float | None,
    ambient_c_start: float | None,
    ambient_rh_pct_start: float | None = None,
    wall_samples: list[WallPowerSample] | None = None,
    rapl_max_energy_range_uj: float | None = None,
    rapl_dram_max_energy_range_uj: float | None = None,
    task: str | None = None,
    task_shape: str | None = None,
    is_canary: bool = False,
    gpu_mem_total_mib: float | None = None,
    repeat_index: int = 0,
    engine: str = "vllm",
    engine_version: str | None = None,
    driver_version: str | None = None,
    cuda_version: str | None = None,
    gpu_name: str | None = None,
    has_vision_tower: bool = False,
    power_limit_w: int | None = None,
    clock_lock_mhz: int | None = None,
    temperature: float = 0.0,
    max_tokens: int = 0,
    seed: int = 0,
    n_shot: int | None = None,
    thinking_mode: str | None = None,
    dataset_revision: str | None = None,
    streaming_used: bool = False,
) -> RunMetrics:
    """Compute derived energy metrics from raw telemetry and inference results.

    Args:
        run_id: Unique identifier for this run
        label: User-provided experiment label
        model: Model identifier used
        quantization: Quantization method if applied (e.g., 'awq', 'gptq')
        target_host: Host this run measured
        samples: Telemetry samples from the local NVML sampler
        inference_results: Per-request results from the engine client
        kwh_before: Smart plug kWh reading before the run, or None (the
            common case here -- this package never queries Home Assistant)
        kwh_after: Smart plug kWh reading after the run, or None
        ambient_c_start: Ambient temperature at run start in Celsius, or None

    Returns:
        RunMetrics with computed energy efficiency metrics

    Raises:
        MetricsComputeError: If computation fails (e.g., zero tokens)
    """
    if not samples:
        raise MetricsComputeError("No telemetry samples provided")

    if not inference_results:
        raise MetricsComputeError("No inference results provided")

    total_completion_tokens = sum(r.completion_tokens for r in inference_results)
    if total_completion_tokens == 0:
        raise MetricsComputeError("Total completion tokens is zero")

    # Trapezoidal integration of gpu_power_w over sample timestamps.
    total_joules_gpu = 0.0
    for i in range(len(samples) - 1):
        dt = samples[i + 1].ts - samples[i].ts
        avg_power = (samples[i].gpu_power_w + samples[i + 1].gpu_power_w) / 2.0
        total_joules_gpu += avg_power * dt

    peak_gpu_w = max(s.gpu_power_w for s in samples)
    mean_gpu_w = sum(s.gpu_power_w for s in samples) / len(samples)

    total_joules_cpu = compute_cpu_energy(samples, rapl_max_energy_range_uj)
    total_joules_cpu_dram = compute_cpu_dram_energy(samples, rapl_dram_max_energy_range_uj)

    # Prefer NVML's hardware energy counter over integrating 5 Hz power
    # samples. Every per-unit metric below derives from `joules_best`, and
    # `energy_source` records which it was.
    counter = compute_counter_energy(samples, total_joules_gpu)
    counter_joules = counter["total_joules_gpu_counter"]
    joules_best = counter_joules if counter_joules is not None else total_joules_gpu
    energy_source = "counter" if counter_joules is not None else "integrated"

    joules_per_token = joules_best / total_completion_tokens

    # Wall-counter honesty: a reading of exactly 0.0 kWh is not necessarily
    # zero energy drawn -- it can just be below the plug's meter resolution.
    wall_counter_ticks: int | None = None
    kwh_delta: float | None = None
    if kwh_before is not None and kwh_after is not None:
        kwh_delta = kwh_after - kwh_before
        wall_counter_ticks = round(kwh_delta / _WALL_METER_RESOLUTION_KWH)
        if wall_counter_ticks == 0:
            kwh_delta = None

    mean_tokens_per_second = sum(r.tokens_per_second for r in inference_results) / len(
        inference_results
    )

    # Pooled throughput: total tokens / total wall time, rather than an
    # average of per-request rates.
    sum_total_s = sum(r.total_s for r in inference_results)
    pooled_tokens_per_second = (
        total_completion_tokens / sum_total_s if sum_total_s > 0 else 0.0
    )

    run_duration_s = samples[-1].ts - samples[0].ts

    mean_wall_w: float | None = None
    peak_wall_w: float | None = None
    if wall_samples:
        wall_watts = [s.wall_power_w for s in wall_samples]
        mean_wall_w = sum(wall_watts) / len(wall_watts)
        peak_wall_w = max(wall_watts)

    node_overhead_ratio: float | None = None
    if mean_wall_w is not None and mean_gpu_w > 0:
        node_overhead_ratio = mean_wall_w / mean_gpu_w

    # 'B' the moment wall-power samples were collected (never special-cased
    # to any one caller); this package's callers never attach a smart plug
    # today, so this is always 'C' in practice.
    measurement_tier: Literal["A", "B", "C"] = "B" if mean_wall_w is not None else "C"

    # Accuracy. Only scored items count -- an unscored probe leaves
    # `correct` as None throughout. `correct` is bool for most tasks and
    # float in [0, 1] for a continuous-scored one (e.g. longctx_summary's
    # ROUGE-L F1).
    scored = [r for r in inference_results if r.correct is not None]
    n_items = len(inference_results)
    correct_values = [float(r.correct) for r in scored]
    n_correct = round(sum(correct_values)) if scored else 0
    accuracy: float | None = (sum(correct_values) / len(scored)) if scored else None

    n_truncated = sum(1 for r in inference_results if r.finish_reason == "length")
    truncated_pct = 100.0 * n_truncated / n_items

    joules_per_correct_answer: float | None = None
    if scored and n_correct > 0:
        joules_per_correct_answer = joules_best / n_correct
    joules_per_item: float | None = joules_best / n_items if n_items > 0 else None

    # Within-run confidence intervals. Same per-item energy attribution
    # `fit_cost_model` uses, so both CIs and the cost-model fit see
    # identical per-item numbers.
    item_energies_j = compute_item_energies_j(samples, inference_results)
    item_correct = [
        float(r.correct) if r.correct is not None else None for r in inference_results
    ]
    jpc_ci_low, jpc_ci_high = bootstrap_jpc_ci(
        item_energies_j, item_correct, joules_best, seed=seed
    )
    accuracy_ci_low, accuracy_ci_high = accuracy_ci(n_correct, len(scored))

    shape = compute_power_shape(samples)
    health = compute_hardware_health(samples, gpu_mem_total_mib=gpu_mem_total_mib)
    costmodel = fit_cost_model(samples, inference_results)
    streaming = compute_streaming_latency(inference_results, streaming_used)

    # `/set-locked-clocks` can only ECHO the request (no NVML query for the
    # lock target exists), so confirmation must come from telemetry -- the
    # mean SM clock health already computed above, not a new sample pass.
    clock_lock_achieved: bool | None = None
    mean_gpu_sm_clock_mhz = health.get("mean_gpu_sm_clock_mhz")
    if clock_lock_mhz is not None and mean_gpu_sm_clock_mhz is not None:
        clock_lock_achieved = abs(mean_gpu_sm_clock_mhz - clock_lock_mhz) <= 0.05 * clock_lock_mhz

    return RunMetrics(
        run_id=run_id,
        label=label,
        repeat_index=repeat_index,
        model=model,
        quantization=quantization,
        target_host=target_host,
        engine=engine,
        engine_version=engine_version,
        driver_version=driver_version,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        has_vision_tower=has_vision_tower,
        power_limit_w=power_limit_w,
        clock_lock_mhz=clock_lock_mhz,
        clock_lock_achieved=clock_lock_achieved,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        n_shot=n_shot,
        thinking_mode=thinking_mode,
        dataset_revision=dataset_revision,
        joules_per_token=joules_per_token,
        total_joules_gpu=total_joules_gpu,
        total_joules_cpu=total_joules_cpu,
        total_joules_cpu_dram=total_joules_cpu_dram,
        kwh_delta=kwh_delta,
        wall_counter_ticks=wall_counter_ticks,
        peak_gpu_w=peak_gpu_w,
        mean_gpu_w=mean_gpu_w,
        mean_tokens_per_second=mean_tokens_per_second,
        pooled_tokens_per_second=pooled_tokens_per_second,
        total_completion_tokens=total_completion_tokens,
        streaming_used=streaming_used,
        run_duration_s=run_duration_s,
        ambient_c_start=ambient_c_start,
        ambient_rh_pct_start=ambient_rh_pct_start,
        mean_wall_w=mean_wall_w,
        peak_wall_w=peak_wall_w,
        node_overhead_ratio=node_overhead_ratio,
        measurement_tier=measurement_tier,
        gpu_temp_c_start=samples[0].gpu_temp_c,
        gpu_temp_c_end=samples[-1].gpu_temp_c,
        task=task,
        task_shape=task_shape,
        is_canary=is_canary,
        n_items=n_items,
        n_correct=n_correct,
        accuracy=accuracy,
        truncated_pct=truncated_pct,
        joules_per_correct_answer=joules_per_correct_answer,
        joules_per_item=joules_per_item,
        jpc_ci_low=jpc_ci_low,
        jpc_ci_high=jpc_ci_high,
        accuracy_ci_low=accuracy_ci_low,
        accuracy_ci_high=accuracy_ci_high,
        energy_source=energy_source,
        total_joules_gpu_best=joules_best,
        **shape,
        **health,
        **counter,
        **costmodel,
        **streaming,
    )
