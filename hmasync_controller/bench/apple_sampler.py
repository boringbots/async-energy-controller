"""AppleSiliconSampler -- the bench's telemetry source on Apple Silicon.

WHY A SECOND SAMPLER
--------------------
`bench quick` is the one thing a stranger runs after `pip install
async-energy-controller`, and until now it could only run on a box with an
NVIDIA GPU: `LocalNvmlSampler._ensure_handle()` raises `NvmlUnavailableError`
the moment NVML is missing, which on a Mac it always is. Every other layer
already worked there -- the quick suite attaches to Ollama or llama-server
over HTTP, and both run natively on Apple Silicon with Metal -- so the
sampler was the whole blocker.

WHY IOReport AND NOT `powermetrics`
-----------------------------------
`powermetrics` is the obvious tool and the wrong one here. It needs root, and
asking a stranger to `sudo` a freshly-installed Python package to measure
their own laptop is a bad trade. It is also a subprocess emitting text on its
own cadence, so the sampler would be parsing someone else's timing rather
than controlling its own.

IOReport is the interface `powermetrics` itself reads, it needs **no
elevated privileges**, and it is counter-based: energy over a window is the
difference of two counter reads, which is the same shape as NVML's
`nvmlDeviceGetTotalEnergyConsumption` that this project already prefers over
integrating a power trace. We reach it through `zeus-apple-silicon`
(`pip install zeus-apple-silicon`, from the ml-energy group) rather than
hand-rolling `ctypes` bindings into a private Apple framework.

WHAT AN APPLE ENERGY NUMBER IS, AND IS NOT
------------------------------------------
Upstream is explicit that IOReport's energy values "are believed to be
model-based estimates" rather than readings off a shunt. So are NVML's, to a
degree -- but they are different models, on different silicon, calibrated by
different vendors. **A Mac row and an NVIDIA row are not the same
measurement**, and this project already has F5 in its findings: two
*identical* RTX 3090s disagree by 22% on the same work. Two different
instruments on different architectures will disagree by more, and nothing
here can correct for it.

That is fenced, not just documented: this sampler tags its runs
`energy_source="ioreport"`, a third value beside `"counter"` and
`"integrated"`, so anything that groups or pools by provenance sees a
distinct instrument instead of silently averaging a Mac into an NVIDIA
population. Accuracy comparisons across the two are fine -- accuracy is
hardware-independent. Joule comparisons are not.

WHAT IS MEASURED, AND WHAT IS WITHHELD
--------------------------------------
Measured: GPU energy, CPU package energy, DRAM energy, all as monotonic
counters in the units the rest of the pipeline already expects.

Withheld as `None`, because Apple exposes no unprivileged equivalent and
Rule 3 says an unrecorded fact is unknown, never guessed or zeroed:

    gpu_temp_c            no unprivileged sensor read
    gpu_util_pct          IOReport has utilization channels; zeus does not
                          expose them
    gpu_mem_used_mib      THERE IS NO VRAM. Memory is unified, so a
                          "GPU memory used" figure would be an invention.
                          A model's footprint on a Mac is a system-memory
                          question, not a card-capacity one.
    gpu_sm_clock_mhz      not exposed
    gpu_mem_clock_mhz     not exposed
    gpu_fan_pct           meaningless on a fanless Air, unread elsewhere
    gpu_perf_state        no NVML analogue
    gpu_throttle_reasons  see the warning below

THE THERMAL GUARD IS INERT HERE, AND THAT MATTERS
-------------------------------------------------
`bench.thermal` trips on NVML's `hw_thermal` clocks-throttle bit and treats
a `None` mask as "not throttled" (`consecutive_hw_thermal_seconds`). Apple
exposes no such bit unprivileged, so on this sampler the circuit-breaker
never fires and `thermal_throttle_pct` stays `None`.

    UPDATE 2026-09-24: it has a signal now. `pmset -g therm` reports
    `CPU_Speed_Limit` without sudo, 100 when the SoC is unthrottled; the
    sampler polls it every THERM_POLL_S and stamps `cpu_speed_limit_pct` on
    each tick. `compute_hardware_health` turns it into `thermal_throttle_pct`
    (percent of samples under 100) and `bench.thermal` treats a limit under
    100 as `hw_thermal`, so the breaker arms on a Mac. It is the SoC's
    pressure, not a GPU clock -- a proxy, and the only one available.
    `pmset -g batt` is read once per run into `power_source` for the same
    reason: on battery macOS caps GPU clocks, and the row has to say so.

This is not a small caveat on a laptop. A fanless MacBook Air under a
sustained benchmark throttles hard, and the run will complete looking
perfectly healthy. Inferring throttling from a falling token rate was
considered and rejected: that is a guess dressed as a measurement, and this
package does not do that. The honest statement is that a Mac run carries no
thermal validity signal at all, which is stated in APPLE-SILICON.md next to
the advice that actually helps: mains power, a hard surface, a short suite.

POWER LIMITS
------------
Apple offers no analogue to NVML's power-limit control, so every
`*_power_limit_*` method below returns `None`. The quick suite already
handles exactly this: `_run_mini_power_sweep` skips with a stated reason
rather than inventing a wattage ladder, so no caller needed changing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import platform
from collections.abc import Callable
import time
from typing import Any

from hmasync_controller.bench.sampler import TelemetrySample

logger = logging.getLogger(__name__)

ENERGY_SOURCE_IOREPORT = "ioreport"
"""`RunMetrics.energy_source` for a run measured through IOReport. A third
value beside 'counter' (NVML) and 'integrated' (5 Hz trapezoid), so a Mac row
never pools with an NVIDIA one by accident."""


class AppleEnergyUnavailableError(Exception):
    """`zeus-apple-silicon` is not importable, or IOReport will not start."""


def is_apple_silicon() -> bool:
    """True on an arm64 Mac. False on Intel Macs, which have no IOReport
    energy model (their power path is Intel RAPL via a different route this
    sampler does not implement)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class AppleSiliconSampler:
    """Apple Silicon's telemetry source, interface-compatible with
    `LocalNvmlSampler` (see `GpuSampler` in `bench.sampler`).

    Construction performs no I/O; IOReport only opens on the first
    `start()`/`gpu_info()` call, mirroring `LocalNvmlSampler._ensure_handle`
    so that importing this module on a non-Mac box is harmless.
    """

    energy_source = ENERGY_SOURCE_IOREPORT

    def __init__(self, sample_hz: int = 5) -> None:
        self.sample_hz = sample_hz
        self._monitor: Any = None
        self._task: asyncio.Task[None] | None = None
        self._samples: list[TelemetrySample] = []
        self._sampling = False
        # Monotonic running totals, so `gpu_energy_mj` behaves like NVML's
        # counter: `metrics.compute` takes last-minus-first over the run and
        # would read a per-tick value as a counter reset.
        self._cum_gpu_mj = 0.0
        self._cum_cpu_mj = 0.0
        self._cum_dram_mj = 0.0
        self._window = 0
        self._tick_opened_at: float | None = None
        self._sram_seen = False
        # `pmset -g therm` is a subprocess, so it is polled every
        # THERM_POLL_S and the last reading is stamped on every tick between
        # polls. `_therm_reader` is an attribute so a test can hand in a
        # canned reading without a Mac.
        self._therm_reader: Callable[[], int | None] = read_cpu_speed_limit_pct
        self._therm_read_at: float = 0.0
        self._cpu_speed_limit_pct: int | None = None

    # --- lifecycle ---------------------------------------------------------

    def _ensure_monitor(self) -> Any:
        if self._monitor is not None:
            return self._monitor
        if not is_apple_silicon():
            raise AppleEnergyUnavailableError(
                "AppleSiliconSampler needs an arm64 Mac; this box reports "
                f"{platform.system()}/{platform.machine()}."
            )
        try:
            from zeus_apple_silicon import AppleEnergyMonitor
        except ImportError as e:
            raise AppleEnergyUnavailableError(
                "zeus-apple-silicon is not importable, so this Mac's energy "
                "cannot be measured. It ships WITH this package on Apple "
                "Silicon, so seeing this means the install did not take: try "
                "`pip install zeus-apple-silicon`, and check you are on the "
                "Python you think you are "
                "(`python -c 'import zeus_apple_silicon'`). "
                "See APPLE-SILICON.md."
            ) from e
        try:
            self._monitor = AppleEnergyMonitor()
        except Exception as e:  # any failure here means "no IOReport on this box"
            raise AppleEnergyUnavailableError(
                f"IOReport would not start on this Mac ({e}). No sudo is "
                "required for it, so this is not a permissions problem -- "
                "most likely an unsupported chip or macOS build."
            ) from e
        return self._monitor

    def _label(self) -> str:
        return f"bench-tick-{self._window}"

    def _open_window(self) -> None:
        """Start the next measurement window and remember when it opened."""
        monitor = self._monitor
        self._window += 1
        monitor.begin_window(self._label())
        self._tick_opened_at = time.monotonic()

    def _close_window(self) -> tuple[Any, float]:
        """Close the current window, returning its metrics and its true duration.

        Windows run BACK TO BACK: the next one opens microseconds after this
        one closes, rather than around the 200 ms sleep. Energy in the gap
        between them is the only thing lost, and it is the cost of two Python
        calls -- not a fifth of every second, which is what a naive
        open-sample-close-sleep loop would drop on the floor.
        """
        monitor = self._monitor
        metrics = monitor.end_window(self._label())
        opened = self._tick_opened_at
        elapsed = (time.monotonic() - opened) if opened is not None else 0.0
        return metrics, elapsed

    # --- sampling ----------------------------------------------------------

    @staticmethod
    def _gpu_mj(metrics: Any) -> float | None:
        """GPU-block energy for one window: the GPU core plus its SRAM.

        NVML's counter covers the whole board, so the nearest Apple analogue
        is the GPU plus its own cache. Unified DRAM is deliberately excluded
        -- it is shared with the CPU and cannot be attributed to the GPU --
        and is reported separately as the DRAM channel.

        A `None` SRAM channel (older silicon) is NOT read as zero: the GPU
        core figure is returned alone and `_sram_seen` stays False, so
        `gpu_info()` can say which of the two shapes produced the number.
        """
        core = getattr(metrics, "gpu_mj", None)
        if core is None:
            return None
        sram = getattr(metrics, "gpu_sram_mj", None)
        return float(core) + float(sram) if sram is not None else float(core)

    def sample(self) -> TelemetrySample:
        """Close the open window, emit its tick, and open the next one."""
        metrics, elapsed_s = self._close_window()
        ts = time.time()

        if ts - self._therm_read_at >= THERM_POLL_S:
            self._cpu_speed_limit_pct = self._therm_reader()
            self._therm_read_at = ts

        gpu_mj = self._gpu_mj(metrics)
        if getattr(metrics, "gpu_sram_mj", None) is not None:
            self._sram_seen = True
        cpu_mj = getattr(metrics, "cpu_total_mj", None)
        dram_mj = getattr(metrics, "dram_mj", None)

        gpu_power_w: float | None = None
        if gpu_mj is not None and elapsed_s > 0:
            gpu_power_w = (gpu_mj / 1000.0) / elapsed_s

        if gpu_mj is not None:
            self._cum_gpu_mj += gpu_mj
        if cpu_mj is not None:
            self._cum_cpu_mj += float(cpu_mj)
        if dram_mj is not None:
            self._cum_dram_mj += float(dram_mj)

        self._open_window()

        return TelemetrySample(
            ts=ts,
            gpu_power_w=gpu_power_w,
            # Every field below is withheld rather than guessed -- see the
            # module docstring's table for why each one is unavailable.
            gpu_util_pct=None,
            gpu_mem_used_mib=None,
            gpu_temp_c=None,
            gpu_mem_util_pct=None,
            gpu_energy_mj=self._cum_gpu_mj if gpu_mj is not None else None,
            gpu_throttle_reasons=None,
            gpu_sm_clock_mhz=None,
            gpu_mem_clock_mhz=None,
            gpu_fan_pct=None,
            gpu_perf_state=None,
            # The pipeline's CPU channels are named for Intel RAPL because
            # that is where they came from; the quantity is CPU package and
            # DRAM energy, which is exactly what IOReport reports. Units
            # match the RAPL contract (microjoules), and the value is
            # cumulative, so the wrap-correction in `compute_cpu_energy`
            # -- which only ever adds on a NEGATIVE step -- never triggers.
            cpu_rapl_uj=self._cum_cpu_mj * 1000.0 if cpu_mj is not None else None,
            cpu_rapl_dram_uj=self._cum_dram_mj * 1000.0 if dram_mj is not None else None,
            cpu_speed_limit_pct=self._cpu_speed_limit_pct,
        )

    async def start(self, run_id: str = "") -> None:
        self._ensure_monitor()
        self._samples = []
        self._cum_gpu_mj = 0.0
        self._cum_cpu_mj = 0.0
        self._cum_dram_mj = 0.0
        self._sampling = True
        self._open_window()
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        interval = 1.0 / self.sample_hz
        while self._sampling:
            await asyncio.sleep(interval)
            if not self._sampling:
                break
            self._samples.append(self.sample())

    async def stop(self) -> list[TelemetrySample]:
        self._sampling = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._monitor is not None and self._tick_opened_at is not None:
            # Close the window left open by the last tick, so IOReport is not
            # left holding a subscription for a run that has ended.
            try:
                self._monitor.end_window(self._label())
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._tick_opened_at = None
        samples, self._samples = self._samples, []
        return samples

    def current_samples(self) -> list[TelemetrySample]:
        """Snapshot for `bench.thermal`'s mid-run read. There is no NVML
        throttle mask here; the breaker reads `cpu_speed_limit_pct` instead
        (see the module docstring)."""
        return list(self._samples)

    # --- identity and power limits ----------------------------------------

    async def gpu_info(self) -> dict[str, object]:
        self._ensure_monitor()
        chip = _chip_name()
        return {
            "gpu_name": chip,
            # No VRAM to report: memory is unified. Total system memory is a
            # different quantity and is already carried by the machine
            # fingerprint, so filling this with it would be a category error.
            "gpu_mem_total_mib": None,
            "driver_version": _macos_version(),
            "cuda_version": None,
            "energy_source": self.energy_source,
            "gpu_energy_channels": "gpu+sram" if self._sram_seen else "gpu",
            # macOS caps GPU clocks on battery; a row has to say which.
            "power_source": read_power_source(),
        }

    async def get_power_limit_w(self) -> int | None:
        """Apple exposes no power-limit control. None, so the quick suite's
        mini power sweep skips with its own stated reason."""
        return None

    async def get_power_limit_default_w(self) -> int | None:
        return None

    async def get_power_limit_constraints_w(self) -> tuple[int | None, int | None]:
        return None, None

    async def set_power_limit_w(self, watts: int) -> int | None:
        """Never succeeds, never raises -- same contract as the NVML sampler
        when it lacks privilege."""
        logger.info(
            "Power limiting is not available on Apple Silicon; "
            "ignoring a request for %s W.",
            watts,
        )
        return None

    def close(self) -> None:
        self._monitor = None
        self._tick_opened_at = None


def _macos_version() -> str | None:
    """'macOS 15.3', or None when the platform will not say.

    There is no driver version on a Mac. The OS build is the closest thing to
    the NVIDIA driver string this field carries elsewhere, and an empty
    `mac_ver()` has to mean unknown -- not the string "macOS ".
    """
    release = platform.mac_ver()[0]
    return f"macOS {release}" if release else None


def _chip_name() -> str | None:
    """The marketing chip name (e.g. 'Apple M3 Pro') from sysctl, or None.

    `platform.processor()` answers 'arm' on macOS, which identifies nothing.
    `machdep.cpu.brand_string` is the only unprivileged spelling of the chip
    a reader would recognise, and a run whose `gpu_name` is 'arm' is a run
    nobody can interpret later.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name = out.stdout.strip()
    return name or None


THERM_POLL_S = 5.0
"""How often `sample()` re-reads `pmset -g therm`. A subprocess at 5 Hz
would cost more than the reading is worth; thermal pressure does not change
in 200 ms."""

_CPU_SPEED_LIMIT = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")


def _pmset(arg: str) -> str | None:
    """`pmset -g <arg>` as text, or None when it cannot be run."""
    import subprocess

    try:
        out = subprocess.run(
            ["pmset", "-g", arg], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def parse_cpu_speed_limit_pct(text: str | None) -> int | None:
    """`CPU_Speed_Limit = 100` out of `pmset -g therm`, or None."""
    if not text:
        return None
    m = _CPU_SPEED_LIMIT.search(text)
    return int(m.group(1)) if m else None


def parse_power_source(text: str | None) -> str | None:
    """'ac' or 'battery' from `pmset -g batt`'s first line
    (`Now drawing from 'AC Power'`), or None when it says neither."""
    if not text:
        return None
    if "'AC Power'" in text:
        return "ac"
    if "'Battery Power'" in text:
        return "battery"
    return None


def read_cpu_speed_limit_pct() -> int | None:
    return parse_cpu_speed_limit_pct(_pmset("therm"))


def read_power_source() -> str | None:
    return parse_power_source(_pmset("batt"))
