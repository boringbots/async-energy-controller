"""Unit tests for the pure read-layer derived-metrics module (US-MERGE-03,
ported from energy-bench's tests/unit/test_derived.py)."""

import pytest

from hmasync_controller.bench.metrics.derived import (
    accuracy_per_watt,
    ipj,
    net_joules,
    net_wall_joules,
    wall_accuracy_per_watt,
    wall_joules,
)


class TestIpj:
    def test_computes_correct_per_joule(self) -> None:
        assert ipj(80, 400.0) == 0.2

    def test_n_correct_none_returns_none(self) -> None:
        assert ipj(None, 400.0) is None

    def test_total_joules_none_returns_none(self) -> None:
        assert ipj(80, None) is None

    def test_total_joules_zero_returns_none(self) -> None:
        assert ipj(80, 0.0) is None

    def test_zero_correct_is_valid_zero(self) -> None:
        assert ipj(0, 400.0) == 0.0


class TestAccuracyPerWatt:
    def test_computes_accuracy_over_mean_gpu_w(self) -> None:
        assert accuracy_per_watt(0.8, 200.0) == 0.004

    def test_accuracy_none_returns_none(self) -> None:
        assert accuracy_per_watt(None, 200.0) is None

    def test_mean_gpu_w_none_returns_none(self) -> None:
        assert accuracy_per_watt(0.8, None) is None

    def test_mean_gpu_w_zero_returns_none(self) -> None:
        assert accuracy_per_watt(0.8, 0.0) is None

    def test_zero_accuracy_is_valid_zero(self) -> None:
        assert accuracy_per_watt(0.0, 200.0) == 0.0


class TestWallAccuracyPerWatt:
    def test_computes_accuracy_over_mean_wall_w(self) -> None:
        assert wall_accuracy_per_watt(0.8, 320.0) == 0.8 / 320.0

    def test_accuracy_none_returns_none(self) -> None:
        assert wall_accuracy_per_watt(None, 320.0) is None

    def test_mean_wall_w_none_returns_none(self) -> None:
        assert wall_accuracy_per_watt(0.8, None) is None

    def test_mean_wall_w_zero_returns_none(self) -> None:
        assert wall_accuracy_per_watt(0.8, 0.0) is None


class TestNetJoules:
    def test_subtracts_idle_draw_over_run(self) -> None:
        assert net_joules(1000.0, 50.0, 10.0) == 500.0

    def test_total_joules_none_returns_none(self) -> None:
        assert net_joules(None, 50.0, 10.0) is None

    def test_loaded_idle_w_none_returns_none(self) -> None:
        assert net_joules(1000.0, None, 10.0) is None

    def test_duration_s_none_returns_none(self) -> None:
        assert net_joules(1000.0, 50.0, None) is None

    def test_can_go_negative_when_idle_exceeds_gross(self) -> None:
        assert net_joules(10.0, 50.0, 1.0) == -40.0

    def test_zero_duration_returns_gross_joules(self) -> None:
        assert net_joules(1000.0, 50.0, 0.0) == 1000.0


class TestWallJoules:
    def test_power_times_duration(self) -> None:
        assert wall_joules(500.0, 100.0) == 50_000.0

    def test_mean_wall_w_none_returns_none(self) -> None:
        assert wall_joules(None, 100.0) is None

    def test_duration_none_returns_none(self) -> None:
        assert wall_joules(500.0, None) is None


class TestNetWallJoules:
    """The whole-machine floor (2026-09-22).

    `net_joules` subtracts a `kind='loaded'` GPU baseline. The wall variant
    subtracts `kind='empty'` instead, and the difference is deliberate: when
    engines disagree about where weights live, "resident" is the variable
    under test, so a loaded wall floor would measure each rung of a residency
    ladder against a different zero.
    """

    def test_subtracts_idle_over_the_run(self) -> None:
        # 500 W for 100 s against a 68.9 W floor
        assert net_wall_joules(500.0, 68.9, 100.0) == (500.0 - 68.9) * 100.0

    def test_is_less_than_gross_by_exactly_the_idle_energy(self) -> None:
        gross = wall_joules(500.0, 100.0)
        net = net_wall_joules(500.0, 68.9, 100.0)
        assert gross - net == pytest.approx(68.9 * 100.0)

    def test_a_long_slow_run_loses_more_to_idle_than_a_short_fast_one(self) -> None:
        """Why gross favours whichever engine finishes sooner: the same work
        at lower power over a longer window carries a bigger idle share."""
        fast_share = 1 - net_wall_joules(513.0, 68.9, 97.0) / wall_joules(513.0, 97.0)
        slow_share = 1 - net_wall_joules(359.0, 68.9, 1284.0) / wall_joules(359.0, 1284.0)
        assert fast_share < slow_share
        assert round(fast_share, 3) == 0.134
        assert round(slow_share, 3) == 0.192

    def test_any_none_returns_none(self) -> None:
        assert net_wall_joules(None, 68.9, 100.0) is None
        assert net_wall_joules(500.0, None, 100.0) is None
        assert net_wall_joules(500.0, 68.9, None) is None

    def test_negative_is_shown_as_measured_never_clamped(self) -> None:
        assert net_wall_joules(50.0, 68.9, 10.0) < 0
