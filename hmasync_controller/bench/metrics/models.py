"""Plain-dataclass data carriers for the bench metrics pipeline (US-MERGE-03).

`InferenceResult` and `RunMetrics` here are ported from energy-bench's
pydantic `energy_bench.models` -- but **this is not a second schema
authority**. energy-bench's `RunMetrics` (88 fields, `schema_version`
bumped in lockstep with the DuckDB `runs` table and the vendored bundle
JSON schema) stays authoritative there; the lab layer's DuckDB index,
`grading/efficiency.py`'s Efficiency Index, and `model_meta.yaml`'s
architecture lookup are all lab-only concerns this package never touches.
This `RunMetrics` is deliberately narrower: exactly the fields
`metrics.compute.compute_metrics()` itself populates, which is what a
`bench quick`/`bench calibrate` submission bundle's "run" entry needs
(`hmasync_controller/schemas/bench_submission.schema.json`'s `$defs.run`).
Four energy-bench fields are dropped because nothing in this package ever
sets them -- they come from the lab-only grading/model_meta pipeline:
`model_type`, `params_b`, `reasoning_mode_class` (config/model_meta.yaml
lookup) and `efficiency_index` (`eb reindex-efficiency`, needs a reference
run on the same target_host, which only exists in the DuckDB index).

`TelemetrySample` stays where US-MERGE-02 put it (`bench.sampler`) --
importing it from here would create the exact "two competing definitions"
problem that story's `nvml_reader` unification was about avoiding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --- NVML clock-throttle bitmask reasons ------------------------------------
# Ported from energy_bench.models -- fixed NVML register bit values, not
# schema, so copying them here carries no drift risk. Not every nonzero mask
# is bad: 0x1 is simply "GPU is idle".

THROTTLE_GPU_IDLE = 0x1
THROTTLE_SW_POWER_CAP = 0x4
THROTTLE_HW_SLOWDOWN = 0x8
THROTTLE_SW_THERMAL = 0x20
THROTTLE_HW_THERMAL = 0x40
THROTTLE_HW_POWER_BRAKE = 0x80

THROTTLE_THERMAL_MASK = (
    THROTTLE_HW_SLOWDOWN | THROTTLE_SW_THERMAL | THROTTLE_HW_THERMAL | THROTTLE_HW_POWER_BRAKE
)
"""Throttles that INVALIDATE a measurement -- the card was forcibly slowed to
protect itself. See energy_bench.models for the full rationale."""

THROTTLE_POWER_CAP_MASK = THROTTLE_SW_POWER_CAP
"""Hitting the configured power limit -- informative during a power sweep,
not invalidating. See energy_bench.models for the full rationale."""

SCHEMA_VERSION = "4"
"""Mirrors energy-bench's `RunMetrics.schema_version` -- both packages
compute the same run-level shape from the same telemetry/inference-result
inputs, so a bundle's `run.schema_version` means the same thing regardless
of which one produced it. Kept in sync by hand, same as the vendored bundle
schema copy in `bench/submission.py`."""


@dataclass
class InferenceResult:
    """Result from a single inference request to an OpenAI-compatible engine.

    Ported from `energy_bench.models.InferenceResult` (pydantic ->
    dataclass; same fields, same defaults). Produced by the vLLM-compatible
    HTTP client US-MERGE-04 ports.
    """

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_s: float
    tokens_per_second: float
    item_id: str | None = None
    correct: bool | float | None = None
    t_start_s: float | None = None
    t_end_s: float | None = None
    finish_reason: str | None = None
    itl_gaps_ms: list[float] = field(default_factory=list)


@dataclass
class WallPowerSample:
    """Lab-only in energy-bench (smart-plug wall-power time series), but the
    bundle's `run` schema still carries `mean_wall_w`/`peak_wall_w` fields
    for a lab box that *does* attach one -- so `compute_metrics()` keeps
    accepting an optional list of these rather than dropping the parameter."""

    ts: float
    wall_power_w: float


@dataclass
class RunMetrics:
    """One run's derived metrics -- see module docstring for scope vs.
    energy-bench's authoritative `RunMetrics`."""

    # --- Required (no sensible default) --------------------------------------
    run_id: str
    label: str
    model: str
    target_host: str
    joules_per_token: float
    total_joules_gpu: float
    kwh_delta: float | None
    peak_gpu_w: float
    mean_gpu_w: float
    mean_tokens_per_second: float
    run_duration_s: float

    # --- Identity / provenance ------------------------------------------------
    repeat_index: int = 0
    quantization: str | None = None
    schema_version: str = SCHEMA_VERSION
    engine: str = "vllm"
    engine_version: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    has_vision_tower: bool = False
    power_limit_w: int | None = None
    clock_lock_mhz: int | None = None
    clock_lock_achieved: bool | None = None

    # --- Probe / sampling configuration ---------------------------------------
    temperature: float = 0.0
    max_tokens: int = 0
    seed: int = 0
    n_shot: int | None = None
    thinking_mode: str | None = None
    dataset_revision: str | None = None

    # --- Energy -----------------------------------------------------------
    total_joules_cpu: float | None = None
    total_joules_cpu_dram: float | None = None
    wall_counter_ticks: int | None = None
    total_joules_gpu_counter: float | None = None
    energy_source: str = "integrated"
    total_joules_gpu_best: float | None = None
    counter_vs_integration_pct_diff: float | None = None

    # --- Throughput / latency --------------------------------------------
    pooled_tokens_per_second: float = 0.0
    total_completion_tokens: int = 0
    streaming_used: bool = False
    ttft_p50_s: float | None = None
    ttft_p95_s: float | None = None
    itl_mean_ms: float | None = None
    itl_p95_ms: float | None = None

    # --- Thermal / VRAM / clocks ----------------------------------------
    ambient_c_start: float | None = None
    ambient_rh_pct_start: float | None = None
    gpu_temp_c_start: float | None = None
    gpu_temp_c_end: float | None = None
    peak_gpu_temp_c: float | None = None
    mean_gpu_temp_c: float | None = None
    peak_gpu_mem_used_mib: float | None = None
    mean_gpu_mem_used_mib: float | None = None
    gpu_mem_used_pct_of_total: float | None = None
    mean_gpu_util_pct: float | None = None
    mean_gpu_mem_util_pct: float | None = None
    mean_gpu_sm_clock_mhz: float | None = None
    mean_gpu_fan_pct: float | None = None
    thermal_throttle_pct: float | None = None
    power_cap_throttle_pct: float | None = None

    # --- Wall power (lab-only in practice, still nullable here) --------------
    mean_wall_w: float | None = None
    peak_wall_w: float | None = None
    node_overhead_ratio: float | None = None
    measurement_tier: Literal["A", "B", "C"] = "C"

    # --- Task / accuracy ----------------------------------------------------
    task: str | None = None
    task_shape: str | None = None
    is_canary: bool = False
    n_items: int = 0
    n_correct: int = 0
    accuracy: float | None = None
    truncated_pct: float = 0.0
    joules_per_correct_answer: float | None = None
    joules_per_item: float | None = None
    jpc_ci_low: float | None = None
    jpc_ci_high: float | None = None
    accuracy_ci_low: float | None = None
    accuracy_ci_high: float | None = None

    # --- Cost-model fit -------------------------------------------------
    alpha_j_per_prompt_token: float | None = None
    beta_j_per_completion_token: float | None = None
    e_fixed_j: float | None = None
    costmodel_r2: float | None = None
    costmodel_n: int | None = None

    # --- Power shape ("lumpiness") ---------------------------------------
    gpu_power_std_w: float | None = None
    gpu_power_cv: float | None = None
    gpu_power_crest_factor: float | None = None
    gpu_power_p95_p50_ratio: float | None = None
    gpu_power_jaggedness_w_per_s: float | None = None
