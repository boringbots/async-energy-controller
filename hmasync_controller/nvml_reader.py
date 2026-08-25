"""hmasync_controller.nvml_reader — the single per-tick NVML register-read
implementation shared by `profiler.NVMLProfiler` (1 Hz `RunTelemetry` for
scheduled jobs) and `bench.sampler.LocalNvmlSampler` (5 Hz bench trace,
US-MERGE-02). Before this module existed the two callers each had their own
copy of the same `nvmlDeviceGet*` call sequence — exactly the drift
`profiler.py`'s docstring used to warn about. Now neither owns the raw
calls; both shape this module's output to their own contract (cadence,
field names, decoded vs. raw throttle reasons).

Each channel degrades independently: a getter this driver/GPU doesn't
support (or a fake/stub `nvml` used in tests doesn't implement) yields
`None` for that one channel, never a fabricated value and never an
exception out of these functions (Rule 3, energy-bench/AGENTS.md).

`read_nvml_channels` deliberately excludes the cumulative energy counter —
see its docstring for why energy stays a separate call.

Duck-typed on purpose: nothing here imports `pynvml` itself. Callers pass an
already-initialized module (real or fake) plus a device handle, so this
module carries no import-time GPU dependency of its own.
"""

from __future__ import annotations

from typing import Any, Callable


def _try(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - any NVML failure means "not available here"
        return None


def read_nvml_channels(nvml: Any, handle: Any) -> dict[str, Any]:
    """One per-tick GPU sample: every channel either caller needs.

    Excludes the cumulative energy counter on purpose (see
    `read_energy_counter_mj`): `NVMLProfiler` reads it only at start/stop for
    one counter delta per run, while `LocalNvmlSampler` reads it every tick —
    folding it in here would change how often either caller consumes it.
    """
    power_mw = _try(lambda: nvml.nvmlDeviceGetPowerUsage(handle))
    util = _try(lambda: nvml.nvmlDeviceGetUtilizationRates(handle))
    mem = _try(lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
    throttle_fn = getattr(nvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None) or getattr(
        nvml, "nvmlDeviceGetCurrentClocksEventReasons", None
    )
    return {
        "power_w": power_mw / 1000.0 if power_mw is not None else None,
        "util_gpu": float(util.gpu) if util is not None else None,
        "util_mem": float(util.memory) if util is not None else None,
        "mem_used_mib": mem.used / (1024 * 1024) if mem is not None else None,
        "temp_c": _try(
            lambda: float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
        ),
        "sm_clock_mhz": _try(
            lambda: float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM))
        ),
        "mem_clock_mhz": _try(
            lambda: float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM))
        ),
        "fan_pct": _try(lambda: float(nvml.nvmlDeviceGetFanSpeed(handle))),
        "perf_state": _try(lambda: int(nvml.nvmlDeviceGetPerformanceState(handle))),
        "throttle_mask": _try(lambda: throttle_fn(handle)) if throttle_fn is not None else None,
    }


def read_energy_counter_mj(nvml: Any, handle: Any) -> float | None:
    """The cumulative GPU energy counter in millijoules, or None if unreadable."""
    return _try(lambda: float(nvml.nvmlDeviceGetTotalEnergyConsumption(handle)))
