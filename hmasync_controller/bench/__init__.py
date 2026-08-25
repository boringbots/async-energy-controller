"""Community benchmark suite (US-MERGE-01..05).

Ported from the energy-bench LAN-lab tool so a bare `pip install
async-energy-controller` gives scheduling AND native benchmarking
(`bench quick`, `bench calibrate`) with no second install and no subprocess
hand-off. Everything in here runs with zero optional hardware: no Home
Assistant, no smart plug, no collector container — only local NVML (or a
graceful null) and an OpenAI-compatible engine on this box.

  - tasks/ — benchmark dataset loading + scoring (the task registry;
    extensible via `tasks.register_task` so a downstream package, e.g.
    energy-bench's lab layer re-registering `humaneval_plus`, can add its own
    without touching this module)
  - sampler.py — `LocalNvmlSampler`, the 5 Hz bench telemetry source (ported
    from energy-bench's `quick.py`, US-MERGE-02); shares its per-tick NVML
    register reads with `hmasync_controller.profiler` via
    `hmasync_controller.nvml_reader` so there is exactly one NVML sampling
    implementation in this repo
  - submission.py — validate/redact/spool/submit a bundle (formerly the flat
    `hmasync_controller/bench.py`, moved here when `bench` became a package;
    its public names are re-exported below so existing call sites —
    `from hmasync_controller import bench; bench.<name>` and
    `from hmasync_controller.bench import <name>` — are unaffected)
  - metrics/ — derived energy/accuracy figures from raw telemetry + inference
    results (`compute_metrics()`, within-run confidence intervals, the
    per-run cost-model fit, power/clock-sweep Flexibility; ported from
    energy-bench's `metrics/*.py` + `grading/flexibility.py`, US-MERGE-03).
    Not re-exported here (same as `tasks/`) — import from
    `hmasync_controller.bench.metrics` directly.

energy-bench keeps the LAN-lab layer on top: Home Assistant, smart plugs, the
collector service, multi-machine fleet sweeps, and the dashboard. None of
that lives here.
"""

from hmasync_controller.bench.sampler import (
    LocalNvmlSampler,
    NvmlUnavailableError,
    TelemetrySample,
)
from hmasync_controller.bench.submission import (
    DENYLISTED_KEY_SUBSTRINGS,
    SCHEMA_PATH,
    BenchDrainResult,
    SubmitResult,
    denylisted_keys,
    drain_bench_spool,
    load_schema,
    submit_bundle_file,
    validate_bundle,
)

__all__ = [
    "DENYLISTED_KEY_SUBSTRINGS",
    "SCHEMA_PATH",
    "BenchDrainResult",
    "LocalNvmlSampler",
    "NvmlUnavailableError",
    "SubmitResult",
    "TelemetrySample",
    "denylisted_keys",
    "drain_bench_spool",
    "load_schema",
    "submit_bundle_file",
    "validate_bundle",
]
