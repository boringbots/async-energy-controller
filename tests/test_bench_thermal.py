"""Tests for hmasync_controller.bench.thermal (US-MERGE-07): the bench item
loop's reaction to sustained NVML `hw_thermal` throttling.

`consecutive_hw_thermal_seconds` is pure and tested directly against
synthetic `TelemetrySample` lists -- no GPU, no sleep, no asyncio needed.
`maybe_pause_for_thermal_throttle` is async; driven with `asyncio.run(...)`
in plain sync tests (same convention as `test_bench_sampler.py`/
`test_bench_quick.py` -- this suite's pytest-asyncio is installed but
unconfigured). Its thresholds are passed as small, explicit overrides so the
pause/timeout paths run in milliseconds, not the real 10s/300s defaults.
"""

from __future__ import annotations

import asyncio

import pytest
from hmasync_controller.bench.metrics import THROTTLE_HW_THERMAL, THROTTLE_SW_THERMAL
from hmasync_controller.bench.sampler import TelemetrySample
from hmasync_controller.bench.thermal import (
    HW_THERMAL_SUSTAINED_S,
    THERMAL_PAUSE_POLL_S,
    THERMAL_PAUSE_TIMEOUT_S,
    SustainedThermalThrottleError,
    consecutive_hw_thermal_seconds,
    maybe_pause_for_thermal_throttle,
)


def _sample(ts: float, throttle_reasons: int | None) -> TelemetrySample:
    return TelemetrySample(
        ts=ts,
        gpu_power_w=200.0,
        gpu_util_pct=90.0,
        gpu_mem_used_mib=8000.0,
        gpu_temp_c=88.0,
        gpu_throttle_reasons=throttle_reasons,
    )


class TestConsecutiveHwThermalSeconds:
    def test_empty_samples_is_zero(self):
        assert consecutive_hw_thermal_seconds([]) == 0.0

    def test_not_throttled_is_zero(self):
        samples = [_sample(1000.0 + i, 0) for i in range(5)]
        assert consecutive_hw_thermal_seconds(samples) == 0.0

    def test_none_throttle_reasons_is_zero_not_fabricated(self):
        """An unread channel is never treated as throttling (Rule 3)."""
        samples = [_sample(1000.0 + i, None) for i in range(5)]
        assert consecutive_hw_thermal_seconds(samples) == 0.0

    def test_sustained_run_measures_full_duration(self):
        samples = [_sample(1000.0 + i, THROTTLE_HW_THERMAL) for i in range(11)]
        assert consecutive_hw_thermal_seconds(samples) == pytest.approx(10.0)

    def test_only_counts_the_trailing_run(self):
        # Throttled 1000-1002, clear 1003-1005, throttled again 1006-1010.
        samples = (
            [_sample(1000.0 + i, THROTTLE_HW_THERMAL) for i in range(3)]
            + [_sample(1003.0 + i, 0) for i in range(3)]
            + [_sample(1006.0 + i, THROTTLE_HW_THERMAL) for i in range(5)]
        )
        assert consecutive_hw_thermal_seconds(samples) == pytest.approx(4.0)

    def test_most_recent_sample_clear_is_zero_even_after_a_long_throttled_run(self):
        samples = [_sample(1000.0 + i, THROTTLE_HW_THERMAL) for i in range(20)] + [
            _sample(1020.0, 0)
        ]
        assert consecutive_hw_thermal_seconds(samples) == 0.0

    def test_other_thermal_bits_alone_do_not_count(self):
        """sw_thermal is routine boost-clock backoff, not the hardware
        protection circuit tripping -- only hw_thermal reacts."""
        samples = [_sample(1000.0 + i, THROTTLE_SW_THERMAL) for i in range(20)]
        assert consecutive_hw_thermal_seconds(samples) == 0.0

    def test_combined_mask_including_hw_thermal_still_counts(self):
        combined = THROTTLE_HW_THERMAL | THROTTLE_SW_THERMAL
        samples = [_sample(1000.0 + i, combined) for i in range(11)]
        assert consecutive_hw_thermal_seconds(samples) == pytest.approx(10.0)

    def test_real_defaults_trigger_threshold(self):
        """Sanity check against the real module constant, not just an
        overridden test threshold."""
        n = int(HW_THERMAL_SUSTAINED_S) + 2
        samples = [_sample(1000.0 + i, THROTTLE_HW_THERMAL) for i in range(n)]
        assert consecutive_hw_thermal_seconds(samples) > HW_THERMAL_SUSTAINED_S


class _Clearing:
    """`get_samples()` double: a sustained-throttle trace for `throttled_reads`
    calls, then a trace whose most recent sample is clear."""

    def __init__(self, throttled_reads: int):
        self._remaining = throttled_reads
        self.calls = 0

    def __call__(self) -> list[TelemetrySample]:
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            return [_sample(1000.0 + i, THROTTLE_HW_THERMAL) for i in range(20)]
        return [_sample(1000.0 + i, THROTTLE_HW_THERMAL) for i in range(19)] + [
            _sample(1019.0, 0)
        ]


class TestMaybePauseForThermalThrottle:
    def test_no_op_when_not_sustained(self):
        calls = 0

        def get_samples():
            nonlocal calls
            calls += 1
            return [_sample(1000.0, 0)]

        asyncio.run(
            maybe_pause_for_thermal_throttle(
                get_samples, sustained_threshold_s=10.0, poll_interval_s=0.01, timeout_s=0.05
            )
        )
        # Checked once (the initial trigger read) and never entered the poll loop.
        assert calls == 1

    def test_pauses_then_resumes_once_it_clears(self):
        source = _Clearing(throttled_reads=2)

        asyncio.run(
            maybe_pause_for_thermal_throttle(
                source, sustained_threshold_s=1.0, poll_interval_s=0.01, timeout_s=1.0
            )
        )
        # 2 throttled reads (the initial trigger check + 1 poll) + 1 clearing poll.
        assert source.calls == 3

    def test_aborts_with_stated_reason_after_timeout(self):
        def always_throttled():
            return [_sample(1000.0 + i, THROTTLE_HW_THERMAL) for i in range(20)]

        with pytest.raises(SustainedThermalThrottleError) as exc_info:
            asyncio.run(
                maybe_pause_for_thermal_throttle(
                    always_throttled,
                    sustained_threshold_s=1.0,
                    poll_interval_s=0.01,
                    timeout_s=0.03,
                )
            )
        message = str(exc_info.value)
        assert "hw_thermal" in message
        assert "did not clear" in message

    def test_real_defaults_are_sane_pause_before_abort(self):
        """The real timeout is meaningfully longer than the real poll
        interval, and the poll interval is meaningfully longer than the
        trigger threshold's own granularity -- a config mistake here would
        make the pause either instant-abort or effectively infinite."""
        assert THERMAL_PAUSE_TIMEOUT_S > THERMAL_PAUSE_POLL_S
        assert THERMAL_PAUSE_POLL_S > 0
        assert HW_THERMAL_SUSTAINED_S > 0
