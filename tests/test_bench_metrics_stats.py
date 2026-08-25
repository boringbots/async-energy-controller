"""Unit tests for the pure within-run statistics module (US-MERGE-03, ported
from energy-bench's tests/unit/test_stats.py)."""

import pytest

from hmasync_controller.bench.metrics.stats import accuracy_ci, bootstrap_jpc_ci, pooled_mean_sigma


class TestBootstrapJpcCi:
    def test_deterministic_for_same_seed(self) -> None:
        energies = [1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2, 1.9]
        corrects = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        a = bootstrap_jpc_ci(energies, corrects, total_joules=20.0, n_boot=500, seed=1234)
        b = bootstrap_jpc_ci(energies, corrects, total_joules=20.0, n_boot=500, seed=1234)
        assert a == b
        assert a[0] is not None and a[1] is not None
        assert a[0] <= a[1]

    def test_different_seed_can_differ(self) -> None:
        energies = [1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2, 1.9]
        corrects = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        a = bootstrap_jpc_ci(energies, corrects, total_joules=20.0, n_boot=500, seed=1)
        b = bootstrap_jpc_ci(energies, corrects, total_joules=20.0, n_boot=500, seed=2)
        assert a != b

    def test_no_attributable_items_returns_none_pair(self) -> None:
        result = bootstrap_jpc_ci([None, None], [1.0, 0.0], total_joules=10.0, n_boot=100, seed=1)
        assert result == (None, None)

    def test_empty_input_returns_none_pair(self) -> None:
        assert bootstrap_jpc_ci([], [], total_joules=10.0, n_boot=100, seed=1) == (None, None)

    def test_total_joules_none_returns_none_pair(self) -> None:
        assert bootstrap_jpc_ci([1.0, 2.0], [1.0, 1.0], total_joules=None, n_boot=100, seed=1) == (
            None,
            None,
        )

    def test_all_incorrect_withholds_ci(self) -> None:
        # Every item scores 0 -> every resample has n_correct == 0 -> every
        # resample is invalid -> far more than 5% invalid -> withhold.
        energies = [1.0, 2.0, 1.5, 3.0]
        corrects = [0.0, 0.0, 0.0, 0.0]
        assert bootstrap_jpc_ci(energies, corrects, total_joules=10.0, n_boot=200, seed=1) == (
            None,
            None,
        )

    def test_partial_attribution_excludes_unattributed_items(self) -> None:
        # Items 2 and 4 lack an energy or correctness value and should be
        # dropped from resampling, not treated as zero.
        energies = [1.0, None, 1.5, 3.0, None]
        corrects = [1.0, 1.0, 1.0, 1.0, None]
        low, high = bootstrap_jpc_ci(energies, corrects, total_joules=10.0, n_boot=500, seed=42)
        assert low is not None and high is not None

    def test_remainder_amortized_uniformly_widens_with_unattributed_energy(self) -> None:
        # With total_joules well above the attributed sum, the CI should
        # sit near total_joules / n_correct-ish territory, not near the
        # bare attributed-only ratio.
        energies = [1.0, 1.0, 1.0, 1.0]
        corrects = [1.0, 1.0, 1.0, 1.0]
        low, high = bootstrap_jpc_ci(energies, corrects, total_joules=4.0, n_boot=1, seed=0)
        assert low == pytest.approx(1.0)
        assert high == pytest.approx(1.0)

        low2, high2 = bootstrap_jpc_ci(energies, corrects, total_joules=8.0, n_boot=1, seed=0)
        assert low2 == pytest.approx(2.0)
        assert high2 == pytest.approx(2.0)


class TestAccuracyCi:
    def test_zero_scored_returns_none_pair(self) -> None:
        assert accuracy_ci(0, 0) == (None, None)

    def test_known_wilson_interval(self) -> None:
        low, high = accuracy_ci(8, 10)
        assert low == pytest.approx(0.49016, abs=1e-4)
        assert high == pytest.approx(0.94332, abs=1e-4)

    def test_perfect_accuracy_stays_within_bounds(self) -> None:
        low, high = accuracy_ci(10, 10)
        assert 0.0 <= low <= 1.0
        assert high == pytest.approx(1.0)

    def test_zero_accuracy_stays_within_bounds(self) -> None:
        low, high = accuracy_ci(0, 10)
        assert low == pytest.approx(0.0, abs=1e-9)
        assert 0.0 <= high <= 1.0

    def test_wider_n_narrows_interval(self) -> None:
        low_small, high_small = accuracy_ci(50, 100)
        low_large, high_large = accuracy_ci(500, 1000)
        assert (high_large - low_large) < (high_small - low_small)


class TestPooledMeanSigma:
    def test_empty_returns_all_none(self) -> None:
        assert pooled_mean_sigma([]) == (None, None, 0)

    def test_single_value_sigma_is_none(self) -> None:
        mean, sigma, n = pooled_mean_sigma([5.0])
        assert mean == pytest.approx(5.0)
        assert sigma is None
        assert n == 1

    def test_known_mean_and_sample_sigma(self) -> None:
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        mean, sigma, n = pooled_mean_sigma(values)
        assert mean == pytest.approx(5.0)
        assert sigma == pytest.approx(2.13809, abs=1e-4)
        assert n == 8

    def test_constant_values_have_zero_sigma(self) -> None:
        mean, sigma, n = pooled_mean_sigma([3.0, 3.0, 3.0])
        assert mean == pytest.approx(3.0)
        assert sigma == pytest.approx(0.0)
        assert n == 3
