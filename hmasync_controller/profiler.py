"""
Profiler — the GPU/CPU telemetry seam (lives in the controller).

    start(run_id)      -> None            begin 1 Hz sampling in a background thread
    stop(run_id)       -> RunTelemetry    summary stats + the raw trace
    capabilities()     -> set[str]        what this box can actually measure

`get_profiler()` probes the hardware and returns the best backend:

    NVMLProfiler  (primary)   — NVIDIA driver's NVML via nvidia-ml-py (lazy import)
    SmiProfiler   (fallback)  — parses `nvidia-smi --query-gpu` CSV over subprocess
    NullProfiler  (last)      — no GPU telemetry; duration-only scheduling still works

Contract: implementations probe what the hardware offers and degrade
**per capability** — a channel the box can't read reports `None`, never a
fabricated value. `energy_source` records how `energy_wh` was obtained:
`counter` (NVML cumulative energy delta, exact), `integrated` (trapezoidal
integration of the power trace), or `None` (no GPU power at all).

CPU RAPL is **independent of the GPU backend**: the shared sampler reads the
Linux RAPL package-energy counter every tick, so a NullProfiler box with RAPL
still reports `cpu_rapl_uj` in its trace (the API turns that into `cpu_energy_wh`).

Historical provenance: this backend's design started as a read of the
energy-bench collector's lazy-pynvml sampler loop and dual-path (Intel/AMD)
RAPL reader
(`AI-Energy-Experimentation/Benchmarking/energy-bench/collector/collector.py`,
a separate repo). As of US-MERGE-02 that is history, not the current shape:
the actual per-tick NVML register reads live in ONE place in THIS repo,
`nvml_reader.py`, shared by this profiler's 1 Hz `RunTelemetry` and
`bench.sampler.LocalNvmlSampler`'s 5 Hz bench trace (itself ported from
energy-bench's `quick.py`). Neither caller re-implements the pynvml call
sequence; each just shapes `nvml_reader.read_nvml_channels()`'s output to
its own contract (cadence, field names, decoded vs. raw throttle reasons).

Import safety: nothing here touches pynvml or a GPU at import time. `pynvml` is
imported lazily inside `NVMLProfiler`/`_nvml_available`, so this module imports
and `get_profiler()` runs on a box with no NVIDIA GPU (→ SmiProfiler / NullProfiler).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hmasync_controller.nvml_reader import read_energy_counter_mj, read_nvml_channels

# --- unit conversions (nail these down; they are the whole game — see rollup.py) ---
# NVML nvmlDeviceGetTotalEnergyConsumption returns MILLIJOULES; 1 Wh = 3.6e6 mJ.
MJ_PER_WH = 3.6e6
# Trapezoidal power integration is in JOULES; 1 Wh = 3600 J.
J_PER_WH = 3600.0

# --- energy_source values (mirror models.EnergySource) ---
ENERGY_COUNTER = "counter"
ENERGY_INTEGRATED = "integrated"

# --- capability tokens returned by capabilities() ---
CAP_POWER = "power"
CAP_UTIL = "util"
CAP_MEMORY = "memory"
CAP_TEMP = "temp"
CAP_CLOCKS = "clocks"
CAP_ENERGY_COUNTER = "energy_counter"
CAP_THROTTLE = "throttle"
CAP_CPU_RAPL = "cpu_rapl"

# The GPU channels of a run_samples row (cpu_rapl_uj + ts are added by the sampler).
_GPU_CHANNELS = (
    "power_w",
    "util_gpu",
    "util_mem",
    "mem_used_mb",
    "temp_c",
    "sm_clock_mhz",
    "throttle_reasons",
)

# Linux RAPL sysfs paths (Intel first, then AMD amd_energy driver) — copy-source's
# read_rapl_energy_uj supports both. Module-level so tests can point them at a fake.
_INTEL_RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
_AMD_RAPL = Path("/sys/class/powercap/amd-rapl:0/energy_uj")

# NVML clocks-throttle-reason bits → short names. Decoded so the trace carries
# rollup-recognizable tokens (rollup._is_throttled treats 'none'/'gpuidle' as NOT
# throttled): a mask of only GPU_IDLE reads as idle, a thermal/power bit reads as a
# genuine throttle. getattr-with-None tolerates constants absent in a given
# nvidia-ml-py version.
_THROTTLE_BITS = (
    ("nvmlClocksThrottleReasonGpuIdle", "gpuidle"),
    ("nvmlClocksThrottleReasonApplicationsClocksSetting", "applications_clocks"),
    ("nvmlClocksThrottleReasonSwPowerCap", "sw_power_cap"),
    ("nvmlClocksThrottleReasonHwSlowdown", "hw_slowdown"),
    ("nvmlClocksThrottleReasonSyncBoost", "sync_boost"),
    ("nvmlClocksThrottleReasonSwThermalSlowdown", "sw_thermal"),
    ("nvmlClocksThrottleReasonHwThermalSlowdown", "hw_thermal"),
    ("nvmlClocksThrottleReasonHwPowerBrakeSlowdown", "hw_power_brake"),
    ("nvmlClocksThrottleReasonDisplayClockSetting", "display_clock"),
)

# nvidia-smi --query-gpu fields, in the order they map to _GPU_CHANNELS (minus
# throttle_reasons, which the basic query does not expose → stays null).
_SMI_QUERY_FIELDS = (
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "temperature.gpu",
    "clocks.sm",
)


class ProfilerUnavailable(Exception):
    """A requested backend cannot be constructed on this box (e.g. no pynvml)."""


class PowerCapPermissionError(Exception):
    """`NVMLProfiler.set_power_limit_w` refused for lack of privilege.

    Setting the power-management limit needs root/CAP_SYS_ADMIN (or the vendor
    equivalent) on most boxes. Raised as its own type, distinct from every other
    NVMLError, so powercap.py can log "skipped-no-permission" instead of a
    generic failure — the two need different operator-facing messages.
    """


def _utcnow() -> datetime:
    """Timezone-aware now — no naive datetime ever enters a sample."""
    return datetime.now(timezone.utc)


def read_rapl_energy_uj() -> float | None:
    """Read the CPU package RAPL energy counter in microjoules.

    Tries the Intel path (`intel-rapl:0`) then the AMD path (`amd-rapl:0`), matching
    the copy-source collector. Returns None when neither is present or readable
    (missing telemetry → null, never fabricated). The counter is cumulative and
    wraps; the API distills a wrap-aware delta into `cpu_energy_wh`.
    """
    for path in (_INTEL_RAPL, _AMD_RAPL):
        try:
            if path.exists():
                return float(path.read_text().strip())
        except (OSError, ValueError):
            return None
    return None


def _non_null(values: list[Any]) -> list[Any]:
    return [v for v in values if v is not None]


def _avg_peak(values: list[Any]) -> tuple[float | None, float | None]:
    vals = _non_null(values)
    if not vals:
        return None, None
    return sum(vals) / len(vals), max(vals)


def _max_non_null(values: list[Any]) -> float | None:
    vals = _non_null(values)
    return max(vals) if vals else None


def _integrate_power_wh(samples: list[dict]) -> float | None:
    """Trapezoidal integration of power_w over sample ts → Wh, or None.

    Needs at least two samples with non-null power and a positive elapsed time.
    Independent of, but consistent with, the API-side rollup integration.
    """
    pts = [
        (s["ts"], s.get("power_w"))
        for s in samples
        if s.get("power_w") is not None and s.get("ts") is not None
    ]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[0])
    joules = 0.0
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        dt = (t1 - t0).total_seconds()
        if dt <= 0:
            continue
        joules += 0.5 * (p0 + p1) * dt
    if joules <= 0:
        return None
    return joules / J_PER_WH


@dataclass
class RunTelemetry:
    """The result of a profiled run — summary stats plus the raw ~1 Hz trace.

    `samples` are `run_samples`-shaped dicts (tz-aware `ts` datetimes + the eight
    telemetry channels); the executor pushes them verbatim to the samples endpoint,
    where the API distills p95/power_profile/throttled_s/cpu_energy_wh. The summary
    fields are what the API cannot recompute or wants authoritatively: `energy_wh`
    with `energy_source='counter'` (the NVML counter delta lives only on-box) plus
    convenience rollups the executor can log.
    """

    run_id: str
    duration_s: float
    samples: list[dict] = field(default_factory=list)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    energy_wh: float | None = None
    energy_source: str | None = None
    avg_w: float | None = None
    peak_w: float | None = None
    gpu_mem_mb: float | None = None

    def to_record_fields(self) -> dict[str, Any]:
        """The RunRecord column subset this run measured, for the executor to merge.

        Leaves p95_w/power_profile/throttled_s/cpu_energy_wh to the API rollup
        (computed from the trace) and identity/provenance
        (controller_id/run_id/workflow_id/fingerprint/framework/exit_status/work_units)
        to the executor.
        """
        return {
            "duration_s": self.duration_s,
            "energy_wh": self.energy_wh,
            "energy_source": self.energy_source,
            "avg_w": self.avg_w,
            "peak_w": self.peak_w,
            "gpu_mem_mb": self.gpu_mem_mb,
        }


class Profiler(ABC):
    """The telemetry seam. Backends probe capabilities and degrade per channel."""

    @abstractmethod
    def start(self, run_id: str) -> None:
        """Begin sampling telemetry for `run_id`."""

    @abstractmethod
    def stop(self, run_id: str | None = None) -> RunTelemetry:
        """Stop sampling and return the run's summary + raw trace."""

    @abstractmethod
    def capabilities(self) -> set[str]:
        """What this box can measure, e.g. {'power', 'energy_counter', 'throttle'}."""


class _SampledProfiler(Profiler):
    """Shared 1 Hz sampler machinery; backends supply the per-tick GPU read.

    A daemon thread samples at `sample_interval_s` while the (blocking) job runs,
    so profiling never delays execution. Every tick reads the GPU
    channels (backend-specific) and the CPU RAPL counter (shared), timestamped with
    a tz-aware clock. Duration is measured with a monotonic clock (immune to
    wall-clock jumps). The thread never raises — a channel read that fails yields
    null for that channel, not a crash.
    """

    def __init__(
        self,
        *,
        sample_interval_s: float = 1.0,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        rapl_reader: Callable[[], float | None] | None = None,
    ):
        self._interval = float(sample_interval_s)
        self._clock = clock or _utcnow
        self._monotonic = monotonic or time.monotonic
        self._rapl = rapl_reader or read_rapl_energy_uj
        self._samples: list[dict] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._run_id: str | None = None
        self._t0: float | None = None
        self._energy_start_mj: float | None = None

    # --- Profiler interface ----------------------------------------------

    def start(self, run_id: str) -> None:
        # Defensively stop any run still sampling (a prior stop() that never ran).
        if self._thread is not None and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=self._interval + 1.0)
        self._run_id = run_id
        self._samples = []
        self._stop = threading.Event()
        self._t0 = self._monotonic()
        self._energy_start_mj = self._read_energy_counter_mj()
        self._thread = threading.Thread(
            target=self._loop, name=f"hmasync-profiler-{run_id}", daemon=True
        )
        self._thread.start()

    def stop(self, run_id: str | None = None) -> RunTelemetry:
        rid = run_id or self._run_id or ""
        if self._thread is None:
            # Never started — return a valid duration-only telemetry, no crash.
            return RunTelemetry(
                run_id=rid, duration_s=0.0, capabilities=frozenset(self.capabilities())
            )
        self._stop.set()
        self._thread.join(timeout=self._interval + 5.0)
        t1 = self._monotonic()
        duration_s = max(0.0, t1 - (self._t0 if self._t0 is not None else t1))
        energy_end_mj = self._read_energy_counter_mj()
        energy_wh, source = self._compute_energy(self._energy_start_mj, energy_end_mj)
        samples = self._samples
        avg_w, peak_w = _avg_peak([s.get("power_w") for s in samples])
        gpu_mem_mb = _max_non_null([s.get("mem_used_mb") for s in samples])
        return RunTelemetry(
            run_id=rid,
            duration_s=duration_s,
            samples=samples,
            capabilities=frozenset(self.capabilities()),
            energy_wh=energy_wh,
            energy_source=source,
            avg_w=avg_w,
            peak_w=peak_w,
            gpu_mem_mb=gpu_mem_mb,
        )

    # --- sampler internals -----------------------------------------------

    def _loop(self) -> None:
        # Sample immediately so even a sub-interval run captures at least one point,
        # then every interval until stop() sets the event (which interrupts wait()).
        while not self._stop.is_set():
            self._collect_one()
            self._stop.wait(self._interval)

    def _collect_one(self) -> None:
        """Take one sample: backend GPU channels + shared RAPL + tz-aware ts."""
        sample: dict[str, Any] = {ch: None for ch in _GPU_CHANNELS}
        try:
            gpu = self._read_gpu_sample()
            if gpu:
                sample.update(gpu)
        except Exception:
            pass  # never crash the sampler; leave the GPU channels null
        try:
            sample["cpu_rapl_uj"] = self._rapl()
        except Exception:
            sample["cpu_rapl_uj"] = None
        sample["ts"] = self._clock()
        self._samples.append(sample)

    def _compute_energy(
        self, start_mj: float | None, end_mj: float | None
    ) -> tuple[float | None, str | None]:
        """Prefer the NVML counter delta; else integrate; else null (never fabricate)."""
        if start_mj is not None and end_mj is not None:
            delta = end_mj - start_mj
            if delta >= 0:  # a negative delta = counter reset/wrap → fall through
                return delta / MJ_PER_WH, ENERGY_COUNTER
        wh = _integrate_power_wh(self._samples)
        if wh is not None:
            return wh, ENERGY_INTEGRATED
        return None, None

    # --- backend hooks (overridden per backend) --------------------------

    def _read_gpu_sample(self) -> dict:
        """Read the GPU telemetry channels for one tick. Base = no GPU (all null)."""
        return {}

    def _read_energy_counter_mj(self) -> float | None:
        """Read the cumulative GPU energy counter in millijoules, or None."""
        return None


# ============================================================
# NVML backend (primary)
# ============================================================
def _import_pynvml():
    try:
        import pynvml  # noqa: PLC0415 (lazy on purpose — optional GPU dep)

        return pynvml
    except ImportError as exc:  # pragma: no cover - env without nvidia-ml-py
        raise ProfilerUnavailable("pynvml (nvidia-ml-py) is not installed") from exc


def _try(fn: Callable[[], Any]) -> Any:
    """Call fn, returning its result or None on any exception (per-channel null)."""
    try:
        return fn()
    except Exception:
        return None


def _is_no_permission_error(nvml: Any, exc: Exception) -> bool:
    """Best-effort detection of NVML's "not privileged" error across versions.

    Prefers the library's own `NVMLError_NoPermission` class or
    `NVML_ERROR_NO_PERMISSION` code when present (real pynvml exposes both via
    its error registry); falls back to the error text, which is what a test
    double or an nvidia-ml-py build lacking either exposes.
    """
    cls = getattr(nvml, "NVMLError_NoPermission", None)
    if cls is not None and isinstance(exc, cls):
        return True
    no_perm_code = getattr(nvml, "NVML_ERROR_NO_PERMISSION", None)
    exc_code = getattr(exc, "value", None)
    if no_perm_code is not None and exc_code == no_perm_code:
        return True
    return "permission" in str(exc).lower()


def _throttle_fn(nvml) -> Callable | None:
    """The available throttle-reasons getter across nvidia-ml-py versions."""
    return getattr(nvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None) or getattr(
        nvml, "nvmlDeviceGetCurrentClocksEventReasons", None
    )


def _decode_throttle(nvml, mask: Any) -> str | None:
    """Decode an NVML throttle bitmask into rollup-recognizable short names.

    0 → 'none'; only the idle bit → 'gpuidle' (both read as NOT throttled by the
    API rollup); any thermal/power bit → its name (read as throttled). Unknown
    non-zero bits fall back to a hex string (rollup treats as throttled — safe).
    """
    try:
        mask = int(mask)
    except (TypeError, ValueError):
        return None
    if mask == 0:
        return "none"
    active = [
        name
        for attr, name in _THROTTLE_BITS
        if (bit:= getattr(nvml, attr, None)) and (mask & int(bit))
    ]
    return ",".join(active) if active else f"0x{mask:x}"


class NVMLProfiler(_SampledProfiler):
    """1 Hz GPU sampling via the NVIDIA driver's NVML (nvidia-ml-py).

    Prefers the exact cumulative energy counter (`nvmlDeviceGetTotalEnergyConsumption`,
    `energy_source='counter'`); falls back to integrating the power trace. Each
    channel degrades independently — a getter the driver/GPU doesn't support yields
    null for that channel and drops from `capabilities()`.
    """

    def __init__(self, *, nvml: Any = None, index: int = 0, **kw):
        super().__init__(**kw)
        # Import eagerly so a missing pynvml fails at construction (get_profiler
        # only builds this after _nvml_available() confirmed the driver); tests
        # inject a fake nvml module.
        self._nvml = nvml if nvml is not None else _import_pynvml()
        self._index = index
        self._handle: Any = None
        self._initialized = False
        self._caps: set[str] | None = None

    def _ensure_nvml(self) -> None:
        if self._initialized:
            return
        self._nvml.nvmlInit()
        self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self._index)
        self._initialized = True

    def close(self) -> None:
        """Release NVML. Safe to call more than once."""
        if self._initialized:
            _try(self._nvml.nvmlShutdown)
            self._initialized = False

    def _read_gpu_sample(self) -> dict:
        self._ensure_nvml()
        ch = read_nvml_channels(self._nvml, self._handle)
        mask = ch["throttle_mask"]
        return {
            "power_w": ch["power_w"],
            "util_gpu": ch["util_gpu"],
            "util_mem": ch["util_mem"],
            "mem_used_mb": ch["mem_used_mib"],
            "temp_c": ch["temp_c"],
            "sm_clock_mhz": ch["sm_clock_mhz"],
            "throttle_reasons": _decode_throttle(self._nvml, mask) if mask is not None else None,
        }

    def _read_energy_counter_mj(self) -> float | None:
        self._ensure_nvml()
        return read_energy_counter_mj(self._nvml, self._handle)

    def device_fingerprint(self) -> dict[str, Any]:
        """GPU model name, driver version, and total VRAM (GB) via THIS handle.

        Independent of sampling/capabilities() — usable right after construction,
        no run in progress. Each field degrades on its own (a getter this
        driver/GPU doesn't support is simply omitted), matching every other
        per-channel contract in this module: never a fabricated value.
        """
        self._ensure_nvml()
        nv, h = self._nvml, self._handle
        out: dict[str, Any] = {}

        name = _try(lambda: nv.nvmlDeviceGetName(h))
        if name:
            out["gpu_name"] = name.decode() if isinstance(name, bytes) else name

        driver = _try(lambda: nv.nvmlSystemGetDriverVersion())
        if driver:
            out["driver_version"] = driver.decode() if isinstance(driver, bytes) else driver

        mem = _try(lambda: nv.nvmlDeviceGetMemoryInfo(h))
        total = getattr(mem, "total", None) if mem is not None else None
        if total is not None:
            out["vram_gb"] = round(total / (1024 ** 3), 1)

        return out

    def get_power_limit_w(self) -> float | None:
        """Current power-management limit in watts, or None if unreadable.

        Independent of sampling/capabilities(), like `device_fingerprint` —
        usable right after construction. Used by powercap.py to capture the
        prior limit before applying a cap, so it can always be restored.
        """
        self._ensure_nvml()
        mw = _try(lambda: self._nvml.nvmlDeviceGetPowerManagementLimit(self._handle))
        return mw / 1000.0 if mw is not None else None

    def get_power_limit_default_w(self) -> float | None:
        """The card's FACTORY DEFAULT power limit in watts, or None if
        unreadable.

        Distinct from `get_power_limit_w`, which reports whatever the card is
        set to right now. powercap.py compares the two before applying a cap:
        a current limit BELOW the factory default means the card was already
        capped, which is either something the operator did deliberately or a
        cap left behind by a job that was killed before it could restore. The
        two cases look identical from here, so this is used to WARN rather
        than to silently override -- see `PowerCapManager.apply`.
        """
        self._ensure_nvml()
        mw = _try(lambda: self._nvml.nvmlDeviceGetPowerManagementDefaultLimit(self._handle))
        return mw / 1000.0 if mw is not None else None

    def set_power_limit_w(self, watts: float) -> None:
        """Set the power-management limit in watts.

        Raises `PowerCapPermissionError` when the driver refuses for lack of
        privilege (the common case on an unprivileged box), or the driver's
        own error for anything else (e.g. a value outside
        `nvmlDeviceGetPowerManagementLimitConstraints`). Deliberately not
        swallowed here — powercap.py is the layer that decides what a
        failure means for a job in progress.
        """
        self._ensure_nvml()
        limit_mw = int(round(watts * 1000))
        try:
            self._nvml.nvmlDeviceSetPowerManagementLimit(self._handle, limit_mw)
        except Exception as exc:
            if _is_no_permission_error(self._nvml, exc):
                raise PowerCapPermissionError(str(exc)) from exc
            raise

    def capabilities(self) -> set[str]:
        if self._caps is not None:
            return set(self._caps)
        self._ensure_nvml()
        nv, h = self._nvml, self._handle
        caps: set[str] = set()
        probes = [
            (CAP_POWER, lambda: nv.nvmlDeviceGetPowerUsage(h)),
            (CAP_UTIL, lambda: nv.nvmlDeviceGetUtilizationRates(h)),
            (CAP_MEMORY, lambda: nv.nvmlDeviceGetMemoryInfo(h)),
            (CAP_TEMP, lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
            (CAP_CLOCKS, lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
            (CAP_ENERGY_COUNTER, lambda: nv.nvmlDeviceGetTotalEnergyConsumption(h)),
        ]
        for cap, fn in probes:
            if _try(fn) is not None:
                caps.add(cap)
        fn = _throttle_fn(nv)
        if fn is not None and _try(lambda: fn(h)) is not None:
            caps.add(CAP_THROTTLE)
        if _try(self._rapl) is not None:
            caps.add(CAP_CPU_RAPL)
        self._caps = caps
        return set(caps)


# ============================================================
# nvidia-smi backend (fallback)
# ============================================================
def _parse_smi_field(raw: str) -> float | None:
    """Parse one nvidia-smi CSV cell to a float, or None for [N/A]/[Not Supported]."""
    token = raw.strip()
    if not token or token.startswith("["):  # [N/A], [Not Supported], [Unknown Error]
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _parse_smi_csv(line: str) -> dict:
    """Map one `--query-gpu` CSV row (nounits) to the GPU trace channels.

    Field order matches _SMI_QUERY_FIELDS: power.draw(W), utilization.gpu(%),
    utilization.memory(%), memory.used(MiB), temperature.gpu(C), clocks.sm(MHz).
    nvidia-smi has no cumulative-energy or throttle-reason column in this query, so
    those channels stay null (→ energy integrated, throttle unknown).
    """
    cells = [c for c in line.split(",")]
    vals = [_parse_smi_field(cells[i]) if i < len(cells) else None for i in range(6)]
    return {
        "power_w": vals[0],
        "util_gpu": vals[1],
        "util_mem": vals[2],
        "mem_used_mb": vals[3],
        "temp_c": vals[4],
        "sm_clock_mhz": vals[5],
        "throttle_reasons": None,
    }


class SmiProfiler(_SampledProfiler):
    """Fallback GPU sampling by parsing `nvidia-smi --query-gpu` CSV per tick.

    nvidia-smi is a CLI wrapper over the same NVML the primary backend uses
    directly; this exists only for boxes where nvidia-ml-py is unavailable but the
    binary is present. No cumulative energy counter → energy is always integrated
    from the power trace (or null).
    """

    def __init__(self, *, smi_bin: str | None = None, index: int = 0, **kw):
        super().__init__(**kw)
        self._smi_bin = smi_bin or shutil.which("nvidia-smi") or "nvidia-smi"
        self._index = index

    def _run_smi(self) -> str | None:
        query = ",".join(_SMI_QUERY_FIELDS)
        try:
            proc = subprocess.run(
                [
                    self._smi_bin,
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self._index),
                ],
                capture_output=True,
                text=True,
                timeout=max(2.0, self._interval),
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        out = (proc.stdout or "").strip()
        return out.splitlines()[0] if out else None

    def _read_gpu_sample(self) -> dict:
        line = self._run_smi()
        return _parse_smi_csv(line) if line else {}

    def capabilities(self) -> set[str]:
        # The --query-gpu fields this backend reads (no energy counter, no throttle).
        caps = {CAP_POWER, CAP_UTIL, CAP_MEMORY, CAP_TEMP, CAP_CLOCKS}
        if _try(self._rapl) is not None:
            caps.add(CAP_CPU_RAPL)
        return caps


# ============================================================
# Null backend (no GPU telemetry; duration-only)
# ============================================================
class NullProfiler(_SampledProfiler):
    """No GPU telemetry — duration-only, GPU energy null.

    Still runs the shared sampler so CPU RAPL (`cpu_rapl_uj`) is captured on a box
    with a RAPL counter but no NVIDIA GPU (RAPL is independent of the GPU backend).
    Its GPU channels are always null; `energy_wh`/`energy_source` stay null.
    """

    def _read_gpu_sample(self) -> dict:
        return {}

    def capabilities(self) -> set[str]:
        caps: set[str] = set()
        if _try(self._rapl) is not None:
            caps.add(CAP_CPU_RAPL)
        return caps


# ============================================================
# Backend selection
# ============================================================
def _nvml_available() -> bool:
    """True when nvidia-ml-py imports, NVML initializes, and a GPU is present.

    Initializes and shuts NVML down as a probe (nvmlInit is refcounted, so the
    profiler re-initializes cleanly afterwards). Never raises.
    """
    try:
        nv = _import_pynvml()
    except ProfilerUnavailable:
        return False
    try:
        nv.nvmlInit()
    except Exception:
        return False
    try:
        return nv.nvmlDeviceGetCount() > 0
    except Exception:
        return False
    finally:
        _try(nv.nvmlShutdown)


def _smi_available() -> bool:
    """True when the nvidia-smi binary is on PATH and answers a cheap query."""
    binpath = shutil.which("nvidia-smi")
    if not binpath:
        return False
    try:
        proc = subprocess.run(
            [binpath, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        return False


def get_profiler(
    *,
    sample_interval_s: float = 1.0,
    clock: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] | None = None,
    rapl_reader: Callable[[], float | None] | None = None,
) -> Profiler:
    """Probe the box and return the best available profiler backend.

    Order: NVMLProfiler → SmiProfiler → NullProfiler. The NullProfiler is always a
    valid choice, so this never fails — a box with no NVIDIA GPU profiles duration
    (and CPU RAPL) and schedules on that alone.
    """
    kw = dict(
        sample_interval_s=sample_interval_s,
        clock=clock,
        monotonic=monotonic,
        rapl_reader=rapl_reader,
    )
    if _nvml_available():
        try:
            return NVMLProfiler(**kw)
        except ProfilerUnavailable:
            pass  # driver vanished between probe and construct — fall back
    if _smi_available():
        return SmiProfiler(**kw)
    return NullProfiler(**kw)
