"""Thermal reaction for the bench item loop (US-MERGE-07).

`HARDWARE-SAFETY.md` establishes that this package never raises a
protection ceiling: no overclock, no overvolt, no fan control, no
memory-clock control. That means the GPU's own hardware thermal protection
circuit (NVML's `hw_thermal` clocks-throttle-reason bit,
`THROTTLE_HW_THERMAL`) is the ONLY thing standing between a marginal-cooling
box and a genuinely unsafe run -- and until this module existed, `bench
quick`/`bench calibrate` recorded that bit in every telemetry sample but
never looked at it while a task was still running. A run could complete
having spent its whole middle throttled, with nothing but a buried
per-sample field to say so.

This module turns "the card has been in hw_thermal for a while" into an
explicit reaction, checked BETWEEN items (not per-item mid-request, and not
a second background monitor loop -- the existing 5 Hz `LocalNvmlSampler`
already IS the monitor; this only reads its trace):

  - A single blip (a fan ramping, a brief ambient spike) is normal and not
    worth reacting to -- only a SUSTAINED run of the bit, longer than
    `HW_THERMAL_SUSTAINED_S`, triggers anything.
  - Once triggered, the loop PAUSES (no new item is sent) and polls the
    sampler's own trace every `THERMAL_PAUSE_POLL_S` until the bit clears.
  - If it has not cleared after `THERMAL_PAUSE_TIMEOUT_S`, the task ABORTS
    with `SustainedThermalThrottleError`, which states the exact duration --
    never a silent continue through a throttled window that would make the
    resulting numbers look comparable to an unthrottled run when they are
    not.

Deliberately checks `THROTTLE_HW_THERMAL` alone, not the broader
`THROTTLE_THERMAL_MASK` (`compute.py`'s `thermal_throttle_pct` uses the
mask, on purpose, for post-hoc reporting of anything thermal-adjacent).
`hw_thermal` specifically means the GPU's own hardware protection circuit
tripped -- a real "this chip is protecting itself" signal. The mask also
includes `sw_thermal` (the driver's own boost-clock backoff, routine on
plenty of healthy cards under sustained load) and `hw_power_brake`
(external power-supply assertion, not thermal at all); reacting to either
of those would pause runs that are not actually in danger.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from hmasync_controller.bench.metrics import THROTTLE_HW_THERMAL
from hmasync_controller.bench.sampler import TelemetrySample

logger = logging.getLogger(__name__)

HW_THERMAL_SUSTAINED_S = 10.0
"""Consecutive seconds of the `hw_thermal` reason bit before the item loop
reacts at all. Matches the >10s threshold this story's acceptance criteria
name."""

THERMAL_PAUSE_POLL_S = 5.0
"""How often the pause re-checks the sampler's trace for the bit clearing."""

THERMAL_PAUSE_TIMEOUT_S = 300.0
"""Hard cap on one pause -- mirrors `orchestrator.cooldown`'s
`thermal_timeout_s` never-wait-forever posture (energy-bench). Past this,
the task aborts rather than pause indefinitely; a card still tripping
`hw_thermal` after 5 minutes is not going to clear on its own within a bench
run's time budget."""


class SustainedThermalThrottleError(Exception):
    """A task aborted because the GPU's own `hw_thermal` bit stayed set
    through an entire pause window. The message states the exact duration,
    never a silent continue."""


def consecutive_hw_thermal_seconds(samples: list[TelemetrySample]) -> float:
    """How long the MOST RECENT run of `samples` has had `hw_thermal`
    continuously set, in wall-clock seconds (last sample's `ts` minus the
    first `ts` of that trailing run).

    Returns 0.0 when `samples` is empty, when the last sample doesn't have
    the bit set, or when `gpu_throttle_reasons` is `None` (unread on this
    driver/GPU) -- an unread channel is never treated as "throttling" (Rule
    3, energy-bench/AGENTS.md: withhold, never fabricate)."""
    if not samples:
        return 0.0
    last = samples[-1]
    if last.gpu_throttle_reasons is None or not (
        last.gpu_throttle_reasons & THROTTLE_HW_THERMAL
    ):
        return 0.0
    run_start_ts = last.ts
    for sample in reversed(samples):
        if sample.gpu_throttle_reasons is None or not (
            sample.gpu_throttle_reasons & THROTTLE_HW_THERMAL
        ):
            break
        run_start_ts = sample.ts
    return last.ts - run_start_ts


async def maybe_pause_for_thermal_throttle(
    get_samples: Callable[[], list[TelemetrySample]],
    *,
    sustained_threshold_s: float | None = None,
    poll_interval_s: float | None = None,
    timeout_s: float | None = None,
) -> None:
    """Called between items in the bench loop. A no-op unless `get_samples()`
    shows `hw_thermal` sustained past `sustained_threshold_s`; otherwise
    pauses (polling `get_samples()` again every `poll_interval_s`) until the
    bit clears, or raises `SustainedThermalThrottleError` after `timeout_s`.

    `get_samples` is a callable rather than a samples list so this can
    re-read the sampler's live trace on every poll -- the whole point is
    noticing the bit clear without stopping the sampler.

    The three thresholds default to `None`, resolved from this module's
    globals (`HW_THERMAL_SUSTAINED_S`/`THERMAL_PAUSE_POLL_S`/
    `THERMAL_PAUSE_TIMEOUT_S`) INSIDE the function body rather than bound as
    parameter defaults -- so a test that monkeypatches those module
    attributes changes what `bench.quick.run_quick_task`'s bare,
    no-override call actually does, without needing quick.py to thread
    three new parameters through its own signature just for tests."""
    if sustained_threshold_s is None:
        sustained_threshold_s = HW_THERMAL_SUSTAINED_S
    if poll_interval_s is None:
        poll_interval_s = THERMAL_PAUSE_POLL_S
    if timeout_s is None:
        timeout_s = THERMAL_PAUSE_TIMEOUT_S

    throttled_s = consecutive_hw_thermal_seconds(get_samples())
    if throttled_s <= sustained_threshold_s:
        return

    logger.warning(
        "sustained hardware thermal throttling: hw_thermal has been set for "
        "%.1fs -- pausing between items (checking every %.0fs, aborting the "
        "task after %.0fs if it does not clear).",
        throttled_s,
        poll_interval_s,
        timeout_s,
    )
    pause_start = time.monotonic()
    while True:
        elapsed = time.monotonic() - pause_start
        if elapsed >= timeout_s:
            raise SustainedThermalThrottleError(
                f"GPU hardware thermal throttle (hw_thermal) did not clear "
                f"after pausing {elapsed:.0f}s -- aborting this task rather "
                "than keep measuring a card that is still protecting "
                "itself. This is expected behavior on a marginal-cooling "
                "box, not a bug: see HARDWARE-SAFETY.md."
            )
        await asyncio.sleep(poll_interval_s)
        if consecutive_hw_thermal_seconds(get_samples()) == 0.0:
            logger.info("hw_thermal cleared after a %.1fs pause -- resuming.", elapsed)
            return
