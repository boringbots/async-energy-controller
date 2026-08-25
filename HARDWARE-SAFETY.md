# Hardware safety

This document enumerates every call this package makes that touches GPU
hardware state, what bounds it, how it is undone, and how it degrades on a
card that cannot do what was asked. It exists because `bench quick`/`bench
calibrate` run on a base install (`pip install async-energy-controller`, no
extra) -- meaning on any GPU a stranger happens to have, not a curated lab
fleet. "Safe on any GPU -- old, new, small" is a requirement, not an
assumption, and this is the audit trail for it.

## What is NEVER done

None of the following NVML/driver capabilities are called anywhere in this
codebase (verified by grep, re-verified by `tests/test_base_install_audit.py`'s
AST sweep and this repo's own test suite):

- **No overclocking.** No `nvmlDeviceSetApplicationsClocks`, no core/memory
  offset APIs.
- **No overvolting, no voltage/frequency curve offsets.** No V/f-offset NVML
  calls of any kind.
- **No fan control.** No `nvmlDeviceSetFanSpeed`/`_v2`.
- **No memory-clock control.** The clock-lock APIs this package's sibling
  project (energy-bench) uses for its clock-sweep research feature are not
  part of the community bench and are not called here at all.
- **No disabling of the GPU's own protections.** No persistence-mode
  toggling, no ECC toggling, no touching `nvmlDeviceSetTemperatureThreshold`
  or any other protection-circuit configuration call.

The one and only *write* this package ever issues to a GPU is a power-limit
**decrease**, described below. Everything else it does to a GPU is a read.

## Every hardware-touching call

| Call | Where | What it does | Bound / clamp | Restore path |
|---|---|---|---|---|
| `nvmlDeviceSetPowerManagementLimit` | `bench/sampler.py::LocalNvmlSampler.set_power_limit_w` (bench mini power sweep) | Sets the board power cap | Caller-derived caps only (`quick.py::_derive_power_sweep_caps_w`: fractions of the card's OWN stock limit -- 0.85/0.75/0.65 -- clamped into `nvmlDeviceGetPowerManagementLimitConstraints`'s reported range; a candidate that is not below stock is dropped, never measured as a fake "capped" point); NVML itself additionally refuses anything outside its own constraint range regardless of what was asked | `bench/quick.py::_run_bench_suite`'s `finally` restores `original_power_limit` (read before any task ran) unconditionally -- fires on success, on any task exception, and on a sweep abort |
| `nvmlDeviceSetPowerManagementLimit` | `profiler.py::NVMLProfiler.set_power_limit_w` (`powercap.py`, scheduled jobs) | Sets the board power cap to the server-recommended value | The server names a wattage; this call does not independently verify it is below stock before applying (that check lives server-side, out of this repo). Raises `PowerCapPermissionError` distinctly from other NVML errors, so the caller can log a clean skip | `powercap.PowerCapManager.restore()` in the executor's `finally`, and ONLY when `apply()` both read the prior limit and successfully wrote the new one first -- `restore()` never fires on a limit this process never actually changed |
| `nvmlDeviceGetPowerUsage`, `GetUtilizationRates`, `GetMemoryInfo`, `GetTemperature`, `GetClockInfo` (SM + mem), `GetFanSpeed`, `GetPerformanceState`, `GetCurrentClocksThrottleReasons`/`GetCurrentClocksEventReasons`, `GetTotalEnergyConsumption` | `nvml_reader.py::read_nvml_channels`/`read_energy_counter_mj` (the one shared per-tick reader -- see its docstring) | Per-tick telemetry, at 1 Hz (`profiler.NVMLProfiler`) or 5 Hz (`bench.sampler.LocalNvmlSampler`) | Read-only; no hardware state changes | N/A |
| `nvmlDeviceGetPowerManagementLimit`, `GetPowerManagementLimitConstraints` | `bench/sampler.py`, `profiler.py` | Reads the current limit and the card's supported range | Read-only | N/A |
| `nvmlDeviceGetName`, `GetMemoryInfo` (`.total`), `nvmlSystemGetDriverVersion`, `nvmlSystemGetCudaDriverVersion` | `bench/sampler.py::gpu_info`, `profiler.py::device_fingerprint` | Device identity/capability probes | Read-only | N/A |
| `nvidia-smi --query-gpu` over `subprocess.run` | `profiler.py::SmiProfiler` (fallback when `nvidia-ml-py` is unavailable but the binary is present) | Same read-only channels as the NVML backend, via the CLI | Read-only; a fixed `--query-gpu` field list, no write flags | N/A |

Every read degrades **per channel**: a getter this driver/GPU/library
version doesn't support returns `None` for that one field, never a
fabricated value, and never crashes the sampler (`nvml_reader.py`'s `_try`,
`profiler.py`'s `_try`). The whole package's controlling rule (energy-bench
`AGENTS.md` Rule 3, which this repo inherited at the merge): **withhold,
never approximate.**

## Per-GPU degradation

| Situation | What happens |
|---|---|
| No NVIDIA GPU at all | `get_profiler()` falls back NVML -> `nvidia-smi` -> `NullProfiler` (scheduled jobs keep working on duration alone). `bench quick`/`bench calibrate` have no such fallback -- local NVML is their one hard hardware requirement, so `LocalNvmlSampler._ensure_handle()` raises `NvmlUnavailableError` naming pynvml, and the bench command exits cleanly with that reason. |
| Laptop / OEM GPU that refuses `SetPowerManagementLimit` (common -- many mobile parts don't expose it, or it needs elevated privilege) | `set_power_limit_w` returns `None` rather than raising (`bench/sampler.py`) or raises `PowerCapPermissionError` (`profiler.py`, caught by `powercap.py`). The mini power sweep's first cap failing skips the WHOLE sweep with a stated reason rather than silently reporting stock as "capped"; a scheduled job just runs uncapped (`STATUS_SKIPPED_NO_PERMISSION`). |
| A card whose own supported floor is at or above its stock limit (some OEM/laptop parts) | `_derive_power_sweep_caps_w` drops every candidate that isn't strictly below stock; if that empties the ladder, the sweep is skipped with the exact reason ("this card's supported power range leaves no point below its stock limit") rather than re-measuring stock and mislabeling it a capped point. |
| Pre-Volta card, or any GPU/driver combination without a cumulative energy counter | `nvmlDeviceGetTotalEnergyConsumption` fails; `gpu_energy_mj`/the profiler's `energy_wh` fall back to trapezoidal integration of the power trace (`energy_source='integrated'`), or `None` if power itself is unreadable. Never fabricated. |
| Old driver lacking `nvmlDeviceGetCurrentClocksThrottleReasons` | `nvml_reader.read_nvml_channels` falls back to the newer `nvmlDeviceGetCurrentClocksEventReasons` name; if neither exists, `throttle_mask` is `None` for every sample. This is also the one gap in the thermal reaction below: with no throttle-reason channel at all, there is nothing to sustain-check, so the bench loop cannot react to it -- the GPU's own thermal protection (see "By construction," below) is still the backstop. |
| `nvidia-smi` fallback (`SmiProfiler`) | No cumulative-energy or throttle-reason column in the basic `--query-gpu` this backend uses -- energy is always integrated (or null), and `throttle_reasons` is always `None`. The thermal reaction below only ever runs against the NVML-backed bench sampler (`bench.sampler.LocalNvmlSampler`), which does not have an `nvidia-smi` fallback of its own (see that module's docstring: local NVML is the one hard requirement). |

## Sustained hardware thermal throttle reaction (`bench/thermal.py`)

Because this package never raises a protection ceiling (section above), the
GPU's own hardware thermal-protection circuit -- NVML's `hw_thermal`
clocks-throttle-reason bit -- is the only thing standing between a
marginal-cooling box and a genuinely unsafe run. Before this existed, `bench
quick`/`bench calibrate` recorded that bit in every telemetry sample but
never *looked* at it while a task was still running: a run could finish
having spent its whole middle throttled, with nothing but a buried
per-sample field to show it.

`bench/quick.py::run_quick_task`'s item loop now checks between items
(never mid-request -- there is no second background monitor; the existing
5 Hz `LocalNvmlSampler` trace already IS the monitor, this only reads it):

1. **Below 10 consecutive seconds of `hw_thermal`:** no-op. A blip -- a fan
   ramping, a momentary ambient spike -- is normal.
2. **Sustained past 10s:** the loop PAUSES (no new item sent), polling the
   sampler's live trace every 5s.
3. **Clears during the pause:** resumes normally.
4. **Still set after 300s (5 min) of pausing:** the task ABORTS with
   `SustainedThermalThrottleError`, which states the exact duration. The
   caller (`_run_bench_suite`) treats one task raising as "skip it, keep
   the rest of the suite" -- the same posture a task failing for any other
   reason already gets -- so an aborted task never gets silently reported
   as a normal, comparable result.

Deliberately checks `THROTTLE_HW_THERMAL` alone, not the broader
`THROTTLE_THERMAL_MASK` `compute.py` uses for post-hoc reporting.
`hw_thermal` specifically means the GPU's hardware protection circuit
tripped. The mask also includes `sw_thermal` (the driver's own boost-clock
backoff -- routine on plenty of healthy cards under sustained load) and
`hw_power_brake` (an external power-supply assertion, not thermal at all);
reacting to either would pause runs that were never actually in danger.

Full rationale and the three named constants
(`HW_THERMAL_SUSTAINED_S`/`THERMAL_PAUSE_POLL_S`/`THERMAL_PAUSE_TIMEOUT_S`):
`hmasync_controller/bench/thermal.py`'s module docstring. Tests for both the
pause-then-resume and the abort-after-timeout paths, with synthetic throttle
telemetry: `tests/test_bench_thermal.py` (the pure duration function and the
pause/abort control flow in isolation) and
`tests/test_bench_quick.py::TestRunQuickTaskThermalReaction` (wired through
the real `run_quick_task` item loop, proving this is not just a function
that exists but unused).

## By construction

Because no cap this package ever applies can raise a protection ceiling,
and no fan/clock/voltage control exists at all, the worst case on a
marginal-cooling machine is the GPU's own hardware self-throttling --
exactly what happens under any other sustained workload (a game, a render),
not something this package introduces. The thermal reaction above is this
package choosing to notice that and back off, on top of a hardware floor
that was already going to hold regardless.
