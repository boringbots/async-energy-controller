"""Unit tests for the per-run cost-model fit (US-MERGE-03, ported from
energy-bench's tests/unit/test_costmodel.py)."""

import pytest

from hmasync_controller.bench.metrics.costmodel import (
    _interpolate,
    _item_energy_j,
    _solve_linear_system,
    compute_item_energies_j,
    fit_cost_model,
)
from hmasync_controller.bench.metrics.models import InferenceResult
from hmasync_controller.bench.sampler import TelemetrySample


def _sample(ts: float, gpu_energy_mj: float | None = None) -> TelemetrySample:
    return TelemetrySample(
        ts=ts, gpu_power_w=200.0, gpu_util_pct=80.0, gpu_mem_used_mib=8000.0,
        gpu_temp_c=65.0, gpu_energy_mj=gpu_energy_mj,
    )


def _result(
    prompt_tokens: int,
    completion_tokens: int,
    t_start_s: float | None = None,
    t_end_s: float | None = None,
    request_id: str = "r",
) -> InferenceResult:
    return InferenceResult(
        request_id=request_id, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        ttft_s=0.1, total_s=1.0, tokens_per_second=float(completion_tokens),
        t_start_s=t_start_s, t_end_s=t_end_s,
    )


class TestInterpolate:
    def test_exact_match_at_sample(self) -> None:
        times, energies = [0.0, 1.0, 2.0], [1000.0, 2000.0, 4000.0]
        assert _interpolate(1.0, times, energies) == 2000.0

    def test_linear_midpoint(self) -> None:
        times, energies = [0.0, 2.0], [1000.0, 3000.0]
        assert _interpolate(1.0, times, energies) == pytest.approx(2000.0)

    def test_before_range_returns_none(self) -> None:
        assert _interpolate(-1.0, [0.0, 1.0], [1000.0, 2000.0]) is None

    def test_after_range_returns_none(self) -> None:
        assert _interpolate(5.0, [0.0, 1.0], [1000.0, 2000.0]) is None


class TestItemEnergy:
    times = [0.0, 1.0, 2.0, 3.0]
    energies = [10_000.0, 12_000.0, 15_000.0, 15_500.0]

    def test_normal_window(self) -> None:
        r = _result(100, 50, t_start_s=1.0, t_end_s=2.0)
        # (15000 - 12000) mJ / 1000 == 3.0 J
        assert _item_energy_j(r, self.times, self.energies) == pytest.approx(3.0)

    def test_missing_timestamps_returns_none(self) -> None:
        r = _result(100, 50)
        assert _item_energy_j(r, self.times, self.energies) is None

    def test_outside_counter_range_returns_none(self) -> None:
        r = _result(100, 50, t_start_s=1.0, t_end_s=10.0)
        assert _item_energy_j(r, self.times, self.energies) is None

    def test_counter_reset_within_window_returns_none(self) -> None:
        times = [0.0, 1.0, 2.0]
        energies = [10_000.0, 500.0, 1_000.0]  # reset between t=0 and t=1
        r = _result(100, 50, t_start_s=0.0, t_end_s=1.0)
        assert _item_energy_j(r, times, energies) is None


class TestComputeItemEnergiesJ:
    """The shared attribution helper both `fit_cost_model` and
    `metrics.compute`'s within-run CIs call."""

    times = [0.0, 1.0, 2.0, 3.0]
    energies = [10_000.0, 12_000.0, 15_000.0, 15_500.0]

    def test_counter_series_missing_returns_all_none(self) -> None:
        samples = [_sample(0.0), _sample(1.0)]  # no gpu_energy_mj
        results = [_result(100, 50, t_start_s=0.0, t_end_s=1.0) for _ in range(3)]
        assert compute_item_energies_j(samples, results) == [None, None, None]

    def test_same_order_and_values_as_item_energy_j(self) -> None:
        samples = [
            TelemetrySample(
                ts=t, gpu_power_w=200.0, gpu_util_pct=80.0, gpu_mem_used_mib=8000.0,
                gpu_temp_c=65.0, gpu_energy_mj=e,
            )
            for t, e in zip(self.times, self.energies)
        ]
        results = [
            _result(100, 50, t_start_s=1.0, t_end_s=2.0, request_id="a"),
            _result(100, 50, request_id="b"),  # no timestamps -> None
            _result(100, 50, t_start_s=1.0, t_end_s=10.0, request_id="c"),  # out of range -> None
        ]
        expected = [
            _item_energy_j(r, self.times, self.energies) for r in results
        ]
        assert compute_item_energies_j(samples, results) == expected
        assert expected[0] == pytest.approx(3.0)
        assert expected[1] is None
        assert expected[2] is None


class TestSolveLinearSystem:
    def test_known_solution(self) -> None:
        # x + y + z = 6, 2y + 5z = -4, 2x + 5y - z = 27 -> x=5, y=3, z=-2
        matrix = [[1.0, 1.0, 1.0], [0.0, 2.0, 5.0], [2.0, 5.0, -1.0]]
        rhs = [6.0, -4.0, 27.0]
        x = _solve_linear_system(matrix, rhs)
        assert x is not None
        assert x == pytest.approx([5.0, 3.0, -2.0])

    def test_singular_matrix_returns_none(self) -> None:
        matrix = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]
        assert _solve_linear_system(matrix, [1.0, 2.0, 3.0]) is None


class TestFitCostModel:
    def test_counter_series_missing_returns_all_none(self) -> None:
        samples = [_sample(0.0), _sample(1.0)]  # no gpu_energy_mj
        results = [_result(100, 50, t_start_s=0.0, t_end_s=1.0) for _ in range(5)]
        fit = fit_cost_model(samples, results)
        assert fit == {
            "alpha_j_per_prompt_token": None,
            "beta_j_per_completion_token": None,
            "e_fixed_j": None,
            "costmodel_r2": None,
            "costmodel_n": None,
        }

    def test_fewer_than_three_usable_items_returns_all_none(self) -> None:
        samples = [_sample(0.0, 0.0), _sample(10.0, 100_000.0)]
        results = [
            _result(100, 50, t_start_s=1.0, t_end_s=2.0),
            _result(200, 100, t_start_s=3.0, t_end_s=4.0),
        ]
        fit = fit_cost_model(samples, results)
        assert fit["costmodel_n"] is None
        assert fit["alpha_j_per_prompt_token"] is None

    def test_items_without_timestamps_are_excluded_not_fatal(self) -> None:
        # 3 usable items + 2 with no timestamps -> fit runs on the 3 usable ones.
        e_fixed, alpha, beta = 5.0, 0.02, 0.08
        items = [(100, 50), (200, 100), (50, 200)]
        samples = [_sample(0.0, 0.0)]
        t = 0.0
        results = []
        cumulative = 0.0
        for prompt, completion in items:
            t_start, t_end = t + 1.0, t + 2.0
            energy_j = e_fixed + alpha * prompt + beta * completion
            samples.append(_sample(t_start, cumulative * 1000.0))
            cumulative += energy_j
            samples.append(_sample(t_end, cumulative * 1000.0))
            results.append(_result(prompt, completion, t_start_s=t_start, t_end_s=t_end))
            t = t_end + 1.0
        results.append(_result(999, 999))  # no timestamps
        results.append(_result(1, 1))  # no timestamps
        fit = fit_cost_model(samples, results)
        assert fit["costmodel_n"] == 3

    def test_recovers_known_alpha_beta_within_5_percent(self) -> None:
        """The story's named acceptance criterion: synthetic data with a known
        linear cost model recovers alpha/beta/e_fixed within 5%."""
        true_e_fixed, true_alpha, true_beta = 5.0, 0.02, 0.08
        items = [
            (100, 50), (200, 100), (50, 200), (300, 20), (10, 10), (150, 150),
        ]

        samples: list[TelemetrySample] = [_sample(0.0, 100_000.0)]
        results: list[InferenceResult] = []
        cumulative_mj = 100_000.0
        t = 0.0
        idle_rate_mj_per_s = 50.0
        for i, (prompt, completion) in enumerate(items):
            gap = 2.0
            cumulative_mj += idle_rate_mj_per_s * gap
            t += gap
            samples.append(_sample(t, cumulative_mj))

            t_start = t
            duration = 1.0 + i * 0.3
            t_end = t_start + duration
            energy_j = true_e_fixed + true_alpha * prompt + true_beta * completion
            cumulative_mj += energy_j * 1000.0
            t = t_end
            samples.append(_sample(t, cumulative_mj))

            results.append(
                _result(prompt, completion, t_start_s=t_start, t_end_s=t_end, request_id=f"r{i}")
            )

        fit = fit_cost_model(samples, results)

        assert fit["costmodel_n"] == len(items)
        assert fit["alpha_j_per_prompt_token"] == pytest.approx(true_alpha, rel=0.05)
        assert fit["beta_j_per_completion_token"] == pytest.approx(true_beta, rel=0.05)
        assert fit["e_fixed_j"] == pytest.approx(true_e_fixed, rel=0.05)
        assert fit["costmodel_r2"] > 0.99

    def test_collinear_token_counts_return_none(self) -> None:
        # prompt_tokens constant across every item -> that column is a scalar
        # multiple of the intercept column -> singular normal equations.
        samples = [_sample(0.0, 0.0)]
        results = []
        cumulative = 0.0
        t = 0.0
        for completion in (10, 20, 30, 40):
            t_start, t_end = t + 1.0, t + 2.0
            energy_j = 1.0 + 0.05 * 100 + 0.1 * completion
            samples.append(_sample(t_start, cumulative * 1000.0))
            cumulative += energy_j
            samples.append(_sample(t_end, cumulative * 1000.0))
            results.append(_result(100, completion, t_start_s=t_start, t_end_s=t_end))
            t = t_end + 1.0

        fit = fit_cost_model(samples, results)
        assert fit["costmodel_n"] is None
        assert fit["alpha_j_per_prompt_token"] is None
