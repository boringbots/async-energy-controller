"""Unit tests for the pure read-layer derived-metrics module (US-MERGE-03,
ported from energy-bench's tests/unit/test_derived.py)."""

from hmasync_controller.bench.metrics.derived import (
    accuracy_per_watt,
    ipj,
    net_joules,
    wall_accuracy_per_watt,
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
