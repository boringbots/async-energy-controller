"""Bench metrics: derived energy/accuracy figures from raw telemetry +
inference results (US-MERGE-03, ported from energy-bench's
`metrics/compute.py` + `metrics/stats.py` + `metrics/derived.py` +
`metrics/costmodel.py` + `grading/flexibility.py`).

  - models.py — `RunMetrics`/`InferenceResult`/`WallPowerSample` (plain
    dataclasses; see that module's docstring for why this is NOT a second
    schema authority) + the NVML throttle-mask constants
  - compute.py — `compute_metrics()`, the top-level entry point, plus its
    per-facet helpers (power shape, hardware health, streaming latency, RAPL
    energy)
  - stats.py — within-run bootstrap/Wilson confidence intervals (pure)
  - derived.py — read-layer-only formulas: IPJ, accuracy-per-watt,
    net-of-idle (pure)
  - costmodel.py — per-run linear cost-model fit, pure-Python least squares
    (pure)
  - flexibility.py — power-sweep / clock-sweep Flexibility metrics (pure)

`TelemetrySample` stays in `bench.sampler` (US-MERGE-02) -- import it from
there, not from here.
"""

from hmasync_controller.bench.metrics.compute import (
    MetricsComputeError,
    compute_counter_energy,
    compute_cpu_dram_energy,
    compute_cpu_energy,
    compute_hardware_health,
    compute_metrics,
    compute_power_shape,
    compute_streaming_latency,
)
from hmasync_controller.bench.metrics.costmodel import compute_item_energies_j, fit_cost_model
from hmasync_controller.bench.metrics.derived import (
    accuracy_per_watt,
    ipj,
    net_joules,
    wall_accuracy_per_watt,
)
from hmasync_controller.bench.metrics.flexibility import (
    MIN_SWEEP_POINTS,
    compute_clock_flexibility_metrics,
    compute_flexibility_metrics,
    sweep_config_key,
)
from hmasync_controller.bench.metrics.models import (
    SCHEMA_VERSION,
    THROTTLE_GPU_IDLE,
    THROTTLE_HW_POWER_BRAKE,
    THROTTLE_HW_SLOWDOWN,
    THROTTLE_HW_THERMAL,
    THROTTLE_POWER_CAP_MASK,
    THROTTLE_SW_POWER_CAP,
    THROTTLE_SW_THERMAL,
    THROTTLE_THERMAL_MASK,
    InferenceResult,
    RunMetrics,
    WallPowerSample,
)
from hmasync_controller.bench.metrics.stats import accuracy_ci, bootstrap_jpc_ci, pooled_mean_sigma

__all__ = [
    "MIN_SWEEP_POINTS",
    "SCHEMA_VERSION",
    "THROTTLE_GPU_IDLE",
    "THROTTLE_HW_POWER_BRAKE",
    "THROTTLE_HW_SLOWDOWN",
    "THROTTLE_HW_THERMAL",
    "THROTTLE_POWER_CAP_MASK",
    "THROTTLE_SW_POWER_CAP",
    "THROTTLE_SW_THERMAL",
    "THROTTLE_THERMAL_MASK",
    "InferenceResult",
    "MetricsComputeError",
    "RunMetrics",
    "WallPowerSample",
    "accuracy_ci",
    "accuracy_per_watt",
    "bootstrap_jpc_ci",
    "compute_clock_flexibility_metrics",
    "compute_counter_energy",
    "compute_cpu_dram_energy",
    "compute_cpu_energy",
    "compute_flexibility_metrics",
    "compute_hardware_health",
    "compute_item_energies_j",
    "compute_metrics",
    "compute_power_shape",
    "compute_streaming_latency",
    "fit_cost_model",
    "ipj",
    "net_joules",
    "pooled_mean_sigma",
    "sweep_config_key",
    "wall_accuracy_per_watt",
]
