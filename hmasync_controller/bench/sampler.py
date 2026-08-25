"""LocalNvmlSampler — the bench's 5 Hz telemetry source (ported from
energy-bench's `quick.py`, US-MERGE-02). Samples `pynvml` directly on this
box: no collector container, no HTTP round trip, no separate process.
`nvidia-ml-py` is a base runtime dependency of this package specifically for
this (see `pyproject.toml`'s `dependencies` comment on the pin).

Shares its per-tick NVML register reads with `hmasync_controller.profiler`
via `hmasync_controller.nvml_reader` — see that module's docstring for why
one implementation now serves both. The two callers differ only in cadence
and output shape: this sampler ticks at `sample_hz` (default 5) and emits
one `TelemetrySample` per tick for the bench trace; `profiler.NVMLProfiler`
ticks at 1 Hz and folds samples into a `RunTelemetry` summary for scheduled
jobs.

Construction performs no I/O — NVML only initializes on the first
`start()`/`gpu_info()`/`get_power_limit_w()`/`set_power_limit_w()` call, via
`_ensure_handle()`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from hmasync_controller.nvml_reader import read_energy_counter_mj, read_nvml_channels

logger = logging.getLogger(__name__)


class NvmlUnavailableError(Exception):
    """pynvml is not importable, or NVML init/handle lookup fails on this box."""


@dataclass
class TelemetrySample:
    """One 5 Hz bench-trace tick. Every field but `ts` degrades
    independently to `None` on a channel this driver/GPU doesn't support —
    never a fabricated value (Rule 3, energy-bench/AGENTS.md).

    `cpu_rapl_uj`/`cpu_rapl_dram_uj` (US-MERGE-03) mirror
    `energy_bench.models.TelemetrySample`'s RAPL fields for full parity with
    the ported `metrics.compute.compute_cpu_energy`/`compute_cpu_dram_energy`
    -- but `LocalNvmlSampler.sample()` below never sets them (this box's
    NVML-only sampler has no local RAPL reader, same as energy-bench's own
    Tier-C `LocalNvmlSampler.rapl_max_energy_range_uj`, which always stays
    None). They exist so a future local RAPL reader can populate them
    without a second `TelemetrySample` definition -- the whole point of
    unifying on one sampler."""

    ts: float
    gpu_power_w: float | None
    gpu_util_pct: float | None
    gpu_mem_used_mib: float | None
    gpu_temp_c: float | None
    gpu_mem_util_pct: float | None = None
    gpu_energy_mj: float | None = None
    gpu_throttle_reasons: int | None = None
    gpu_sm_clock_mhz: int | None = None
    gpu_mem_clock_mhz: int | None = None
    gpu_fan_pct: int | None = None
    gpu_perf_state: int | None = None
    cpu_rapl_uj: float | None = None
    cpu_rapl_dram_uj: float | None = None


def _format_cuda_version(raw: int) -> str:
    """Format NVML's packed CUDA driver version (e.g. 12040) as 'major.minor'."""
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def _try_nvml(fn: Callable[[], Any]) -> Any:
    """Read one optional NVML value, tolerating an unsupported query."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 - any NVML failure means "not available here"
        return None


class LocalNvmlSampler:
    """Tier C's default telemetry source: samples `pynvml` directly on this
    box, in-process, at `sample_hz`."""

    def __init__(self, sample_hz: int = 5) -> None:
        self.sample_hz = sample_hz
        self._pynvml = None
        self._handle = None
        self._task: asyncio.Task[None] | None = None
        self._samples: list[TelemetrySample] = []
        self._sampling = False

    def _ensure_handle(self):
        if self._handle is not None:
            return self._handle
        try:
            import pynvml
        except ImportError as e:
            raise NvmlUnavailableError(
                "pynvml is not importable -- bench telemetry needs local NVML "
                "access (it is a base async-energy-controller dependency; "
                "`python -c 'import pynvml'` to check the active environment)."
            ) from e
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            raise NvmlUnavailableError(
                f"NVML unavailable on this box ({e}) -- bench telemetry needs "
                "a local NVIDIA GPU."
            ) from e
        self._pynvml = pynvml
        self._handle = handle
        return handle

    def sample(self) -> TelemetrySample:
        """One telemetry sample, via the shared `nvml_reader` register reads."""
        pynvml, h = self._pynvml, self._handle
        ts = time.time()
        ch = read_nvml_channels(pynvml, h)
        sm_clock = ch["sm_clock_mhz"]
        mem_clock = ch["mem_clock_mhz"]
        fan_pct = ch["fan_pct"]
        throttle = ch["throttle_mask"]
        return TelemetrySample(
            ts=ts,
            gpu_power_w=ch["power_w"],
            gpu_util_pct=ch["util_gpu"],
            gpu_mem_used_mib=ch["mem_used_mib"],
            gpu_temp_c=ch["temp_c"],
            gpu_mem_util_pct=ch["util_mem"],
            gpu_energy_mj=read_energy_counter_mj(pynvml, h),
            gpu_throttle_reasons=int(throttle) if throttle is not None else None,
            gpu_sm_clock_mhz=int(sm_clock) if sm_clock is not None else None,
            gpu_mem_clock_mhz=int(mem_clock) if mem_clock is not None else None,
            gpu_fan_pct=int(fan_pct) if fan_pct is not None else None,
            gpu_perf_state=ch["perf_state"],
        )

    async def start(self, run_id: str = "") -> None:
        """Begin background sampling. `run_id` is accepted (unused) only to
        keep the same call shape as a future collector-backed telemetry
        source."""
        self._ensure_handle()
        self._samples = []
        self._sampling = True
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        interval = 1.0 / self.sample_hz
        while self._sampling:
            self._samples.append(self.sample())
            await asyncio.sleep(interval)

    async def stop(self) -> list[TelemetrySample]:
        self._sampling = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        samples, self._samples = self._samples, []
        return samples

    async def gpu_info(self) -> dict[str, object]:
        h = self._ensure_handle()
        pynvml = self._pynvml
        return {
            "gpu_name": _try_nvml(lambda: str(pynvml.nvmlDeviceGetName(h))),
            "gpu_mem_total_mib": _try_nvml(
                lambda: float(pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 * 1024))
            ),
            "driver_version": _try_nvml(lambda: str(pynvml.nvmlSystemGetDriverVersion())),
            "cuda_version": _try_nvml(
                lambda: _format_cuda_version(pynvml.nvmlSystemGetCudaDriverVersion())
            ),
        }

    async def get_power_limit_w(self) -> int | None:
        try:
            h = self._ensure_handle()
            return int(self._pynvml.nvmlDeviceGetPowerManagementLimit(h) // 1000)
        except Exception:  # noqa: BLE001 - best-effort, mirrors runner's original-limit fetch
            return None

    async def get_power_limit_constraints_w(self) -> tuple[int | None, int | None]:
        """Card's supported power-limit range in watts, via
        `nvmlDeviceGetPowerManagementLimitConstraints`. Both None on any NVML
        failure (Rule 3 -- never guess a range); a future adaptive power
        sweep uses this to clamp its derived caps."""
        try:
            h = self._ensure_handle()
            min_mw, max_mw = self._pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
            return int(min_mw) // 1000, int(max_mw) // 1000
        except Exception:  # noqa: BLE001 - best-effort, see get_power_limit_w
            return None, None

    async def set_power_limit_w(self, watts: int) -> int | None:
        """Best-effort: None on ANY failure -- most commonly a permission
        error, since NVML's SetPowerManagementLimit needs root on most
        drivers. Never raises: a future mini power sweep's whole point is to
        skip gracefully rather than abort the bench over this."""
        try:
            h = self._ensure_handle()
            self._pynvml.nvmlDeviceSetPowerManagementLimit(h, watts * 1000)
            actual_mw = self._pynvml.nvmlDeviceGetPowerManagementLimit(h)
            return int(actual_mw // 1000)
        except Exception as e:  # noqa: BLE001 - see docstring
            logger.info(f"NVML SetPowerManagementLimit({watts}W) failed: {e}")
            return None

    def close(self) -> None:
        if self._handle is not None and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._handle = None
