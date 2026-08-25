"""Unit tests for `hmasync_controller.bench.metrics.flexibility` (US-MERGE-03,
ported from energy-bench's tests/unit/test_flexibility.py).

`compute_flexibility_metrics` is a pure function over a list of same-group
runs -- these tests build `RunMetrics` (and, for one test each, plain dicts
via `dataclasses.asdict`) directly.
"""

import dataclasses

import pytest

from hmasync_controller.bench.metrics.flexibility import (
    MIN_SWEEP_POINTS,
    compute_clock_flexibility_metrics,
    compute_flexibility_metrics,
    sweep_config_key,
)
from hmasync_controller.bench.metrics.models import RunMetrics


def _run(
    run_id: str,
    power_limit_w: int | None,
    mean_gpu_w: float,
    mean_tokens_per_second: float,
    joules_per_token: float,
    **overrides,
) -> RunMetrics:
    fields = dict(
        run_id=run_id,
        label=run_id,
        model="Qwen/Qwen3.5-9B",
        quantization="Q4_K_M",
        engine="llama.cpp",
        target_host="192.168.0.114",
        task="gsm8k_platinum",
        power_limit_w=power_limit_w,
        joules_per_token=joules_per_token,
        total_joules_gpu=1000.0,
        kwh_delta=None,
        peak_gpu_w=mean_gpu_w + 10.0,
        mean_gpu_w=mean_gpu_w,
        mean_tokens_per_second=mean_tokens_per_second,
        run_duration_s=10.0,
        ambient_c_start=22.5,
    )
    fields.update(overrides)
    return RunMetrics(**fields)


def _clock_run(
    run_id: str,
    clock_lock_mhz: int | None,
    mean_gpu_sm_clock_mhz: float | None,
    mean_tokens_per_second: float,
    joules_per_token: float,
    **overrides,
) -> RunMetrics:
    fields = dict(
        run_id=run_id,
        label=run_id,
        model="Qwen/Qwen3.5-9B",
        quantization="Q4_K_M",
        engine="llama.cpp",
        target_host="192.168.0.114",
        task="gsm8k_platinum",
        clock_lock_mhz=clock_lock_mhz,
        joules_per_token=joules_per_token,
        total_joules_gpu=1000.0,
        kwh_delta=None,
        mean_gpu_w=300.0,
        peak_gpu_w=310.0,
        mean_gpu_sm_clock_mhz=mean_gpu_sm_clock_mhz,
        mean_tokens_per_second=mean_tokens_per_second,
        run_duration_s=10.0,
        ambient_c_start=22.5,
    )
    fields.update(overrides)
    return RunMetrics(**fields)


class TestSweepConfigKey:
    def test_joins_the_four_fields(self):
        key = sweep_config_key("Qwen/Qwen3.5-9B", "Q4_K_M", "llama.cpp", "192.168.0.114")
        assert key == "Qwen/Qwen3.5-9B|Q4_K_M|llama.cpp|192.168.0.114"

    def test_none_quantization_becomes_a_placeholder_not_the_string_none(self):
        key = sweep_config_key("Qwen/Qwen2.5-7B-Instruct", None, "vllm", "192.168.0.115")
        assert key == "Qwen/Qwen2.5-7B-Instruct|-|vllm|192.168.0.115"


class TestComputeFlexibilityMetricsExplicitNumericStock:
    """The example ladder: 390 (stock) / 350 / 300 / 280 / 250 / 225 / 200 /
    150 W -- stock recorded as an explicit numeric cap (no run in the group
    has power_limit_w is None), so the highest cap stands in for stock."""

    @pytest.fixture
    def sweep(self):
        return [
            _run("r390", 390, 385.0, 100.0, 0.50),
            _run("r350", 350, 345.0, 99.0, 0.47),
            _run("r300", 300, 295.0, 97.0, 0.44),
            _run("r280", 280, 276.0, 96.0, 0.42),  # the knee: lowest J/token
            _run("r250", 250, 246.0, 93.0, 0.43),
            _run("r225", 225, 222.0, 85.0, 0.46),
            _run("r200", 200, 198.0, 60.0, 0.55),
            _run("r150", 150, 149.0, 40.0, 0.75),
        ]

    def test_flex_band_w_is_the_deepest_cut_still_at_95pct_throughput(self, sweep):
        result = compute_flexibility_metrics(sweep)
        # 280W is the lowest cap still >= 95% of stock's 100 tok/s (96%);
        # 250W drops to 93%, disqualifying it and everything below.
        assert result["flex_band_w"] == pytest.approx(110.0)  # 390 - 280

    def test_knee_savings_pct_is_the_best_capped_point_vs_stock(self, sweep):
        result = compute_flexibility_metrics(sweep)
        # best J/token among capped points is 0.42 (280W) vs stock's 0.50.
        assert result["knee_savings_pct"] == pytest.approx(16.0)

    def test_turndown_ratio_is_stock_w_over_cliff_edge_w(self, sweep):
        result = compute_flexibility_metrics(sweep)
        # lowest cap still >= 50% of stock's 100 tok/s is 200W (60%); 150W
        # drops to 40%, disqualifying it.
        assert result["turndown_ratio"] == pytest.approx(385.0 / 198.0)

    def test_n_points_is_the_full_group_size(self, sweep):
        assert compute_flexibility_metrics(sweep)["n_points"] == 8


class TestComputeFlexibilityMetricsUncappedStock:
    """The mini-sweep convention: stock is genuinely uncapped (power_limit_w
    is None), so its "cap reduction" reference is its measured mean_gpu_w
    instead of a numeric cap."""

    @pytest.fixture
    def sweep(self):
        return [
            _run("stock", None, 300.0, 60.0, 0.55),
            _run("r280", 280, 278.0, 59.0, 0.50),
            _run("r250", 250, 248.0, 56.0, 0.48),  # the knee
            _run("r225", 225, 223.0, 45.0, 0.52),
        ]

    def test_flex_band_w_uses_stock_mean_gpu_w_as_the_reference(self, sweep):
        result = compute_flexibility_metrics(sweep)
        # only 280W clears 95% of stock's 60 tok/s (59/60 = 98.3%).
        assert result["flex_band_w"] == pytest.approx(20.0)  # 300 - 280

    def test_knee_savings_pct(self, sweep):
        result = compute_flexibility_metrics(sweep)
        assert result["knee_savings_pct"] == pytest.approx((0.55 - 0.48) / 0.55 * 100.0)

    def test_turndown_ratio(self, sweep):
        result = compute_flexibility_metrics(sweep)
        # every capped point clears 50% of stock's 60 tok/s (30); the
        # lowest cap among them, 225W, is the cliff edge.
        assert result["turndown_ratio"] == pytest.approx(300.0 / 223.0)

    def test_accepts_runs_row_dicts_not_just_runmetrics(self, sweep):
        dict_result = compute_flexibility_metrics([dataclasses.asdict(r) for r in sweep])
        model_result = compute_flexibility_metrics(sweep)
        assert dict_result == model_result


class TestComputeFlexibilityMetricsEdgeCases:
    def test_fewer_than_min_sweep_points_withholds_everything(self):
        sweep = [
            _run("stock", None, 300.0, 60.0, 0.55),
            _run("r280", 280, 278.0, 59.0, 0.50),
            _run("r250", 250, 248.0, 56.0, 0.48),
        ]
        assert len(sweep) < MIN_SWEEP_POINTS
        result = compute_flexibility_metrics(sweep)
        assert result == {
            "flex_band_w": None,
            "knee_savings_pct": None,
            "turndown_ratio": None,
            "n_points": 3,
        }

    def test_ambiguous_stock_point_withholds_everything(self):
        # Two uncapped runs in the same group -- which is "stock" is not
        # decidable, so the whole result is withheld rather than guessed.
        sweep = [
            _run("stock_a", None, 300.0, 60.0, 0.55),
            _run("stock_b", None, 298.0, 61.0, 0.54),
            _run("r250", 250, 248.0, 56.0, 0.48),
            _run("r225", 225, 223.0, 45.0, 0.52),
        ]
        result = compute_flexibility_metrics(sweep)
        assert result == {
            "flex_band_w": None,
            "knee_savings_pct": None,
            "turndown_ratio": None,
            "n_points": 4,
        }

    def test_no_cap_holds_95pct_throughput_gives_zero_width_band_not_none(self):
        sweep = [
            _run("stock", None, 300.0, 100.0, 0.50),
            _run("r250", 250, 245.0, 80.0, 0.45),
            _run("r225", 225, 222.0, 70.0, 0.42),  # the knee
            _run("r200", 200, 198.0, 55.0, 0.50),
        ]
        result = compute_flexibility_metrics(sweep)
        assert result["flex_band_w"] == 0.0
        assert result["knee_savings_pct"] == pytest.approx(16.0)
        # 200W is the lowest cap still >= 50% of stock's 100 tok/s (55%).
        assert result["turndown_ratio"] == pytest.approx(300.0 / 198.0)

    def test_no_cap_sustains_50pct_throughput_withholds_turndown_ratio_only(self):
        sweep = [
            _run("stock", None, 300.0, 100.0, 0.50),
            _run("r200", 200, 195.0, 45.0, 0.60),
            _run("r175", 175, 170.0, 35.0, 0.65),
            _run("r150", 150, 145.0, 20.0, 0.70),
        ]
        result = compute_flexibility_metrics(sweep)
        assert result["turndown_ratio"] is None
        assert result["flex_band_w"] == 0.0
        # Knee savings can be negative -- a modest cap can be genuinely less
        # efficient than stock, and that's a real finding, not an error.
        assert result["knee_savings_pct"] == pytest.approx(-20.0)

    def test_zero_stock_joules_per_token_withholds_knee_savings_only(self):
        sweep = [
            _run("stock", None, 300.0, 100.0, 0.0),
            _run("r250", 250, 245.0, 96.0, 0.45),
            _run("r225", 225, 222.0, 90.0, 0.42),
            _run("r200", 200, 198.0, 60.0, 0.50),
        ]
        result = compute_flexibility_metrics(sweep)
        assert result["knee_savings_pct"] is None
        # The other two metrics are unaffected by the zero-guard.
        assert result["flex_band_w"] is not None
        assert result["turndown_ratio"] is not None


class TestComputeClockFlexibilityMetricsExplicitNumericStock:
    """The analogue of the ladder-style explicit-numeric-stock case above:
    no run in the group has clock_lock_mhz is None, so the highest lock
    stands in for stock."""

    @pytest.fixture
    def sweep(self):
        return [
            _clock_run("r2100", 2100, 2098.0, 100.0, 0.50),
            _clock_run("r1900", 1900, 1895.0, 99.0, 0.47),
            _clock_run("r1700", 1700, 1690.0, 97.0, 0.44),
            _clock_run("r1500", 1500, 1480.0, 96.0, 0.42),  # the knee: lowest J/token
            _clock_run("r1300", 1300, 1280.0, 93.0, 0.43),
        ]

    def test_clock_band_mhz_is_the_deepest_cut_still_at_95pct_throughput(self, sweep):
        result = compute_clock_flexibility_metrics(sweep)
        # 1500MHz is the lowest lock still >= 95% of stock's 100 tok/s (96%);
        # 1300MHz drops to 93%, disqualifying it.
        assert result["clock_band_mhz"] == pytest.approx(600.0)  # 2100 - 1500

    def test_clock_knee_savings_pct_is_the_best_locked_point_vs_stock(self, sweep):
        result = compute_clock_flexibility_metrics(sweep)
        assert result["clock_knee_savings_pct"] == pytest.approx(16.0)

    def test_n_points_is_the_full_group_size(self, sweep):
        assert compute_clock_flexibility_metrics(sweep)["n_points"] == 5

    def test_has_no_turndown_ratio_key(self, sweep):
        assert set(compute_clock_flexibility_metrics(sweep).keys()) == {
            "clock_band_mhz",
            "clock_knee_savings_pct",
            "n_points",
        }


class TestComputeClockFlexibilityMetricsUncappedStock:
    """Stock is genuinely unlocked (clock_lock_mhz is None), so its "cut"
    reference is its measured mean_gpu_sm_clock_mhz instead of a numeric
    lock."""

    @pytest.fixture
    def sweep(self):
        return [
            _clock_run("stock", None, 2100.0, 60.0, 0.55),
            _clock_run("r1900", 1900, 1895.0, 59.0, 0.50),
            _clock_run("r1700", 1700, 1690.0, 56.0, 0.48),  # the knee
            _clock_run("r1500", 1500, 1480.0, 45.0, 0.52),
        ]

    def test_clock_band_mhz_uses_stock_mean_sm_clock_as_the_reference(self, sweep):
        result = compute_clock_flexibility_metrics(sweep)
        # only 1900MHz clears 95% of stock's 60 tok/s (59/60 = 98.3%).
        assert result["clock_band_mhz"] == pytest.approx(200.0)  # 2100 - 1900

    def test_clock_knee_savings_pct(self, sweep):
        result = compute_clock_flexibility_metrics(sweep)
        assert result["clock_knee_savings_pct"] == pytest.approx((0.55 - 0.48) / 0.55 * 100.0)

    def test_accepts_runs_row_dicts_not_just_runmetrics(self, sweep):
        dict_result = compute_clock_flexibility_metrics([dataclasses.asdict(r) for r in sweep])
        model_result = compute_clock_flexibility_metrics(sweep)
        assert dict_result == model_result


class TestComputeClockFlexibilityMetricsEdgeCases:
    def test_fewer_than_min_sweep_points_withholds_everything(self):
        sweep = [
            _clock_run("stock", None, 2100.0, 60.0, 0.55),
            _clock_run("r1900", 1900, 1895.0, 59.0, 0.50),
            _clock_run("r1700", 1700, 1690.0, 56.0, 0.48),
        ]
        assert len(sweep) < MIN_SWEEP_POINTS
        result = compute_clock_flexibility_metrics(sweep)
        assert result == {
            "clock_band_mhz": None,
            "clock_knee_savings_pct": None,
            "n_points": 3,
        }

    def test_ambiguous_stock_point_withholds_everything(self):
        # Two unlocked runs in the same group -- which is "stock" is not
        # decidable, so the whole result is withheld rather than guessed.
        sweep = [
            _clock_run("stock_a", None, 2100.0, 60.0, 0.55),
            _clock_run("stock_b", None, 2080.0, 61.0, 0.54),
            _clock_run("r1500", 1500, 1480.0, 56.0, 0.48),
            _clock_run("r1300", 1300, 1280.0, 45.0, 0.52),
        ]
        result = compute_clock_flexibility_metrics(sweep)
        assert result == {
            "clock_band_mhz": None,
            "clock_knee_savings_pct": None,
            "n_points": 4,
        }

    def test_no_lock_holds_95pct_throughput_gives_zero_width_band_not_none(self):
        sweep = [
            _clock_run("stock", None, 2100.0, 100.0, 0.50),
            _clock_run("r1700", 1700, 1690.0, 80.0, 0.45),
            _clock_run("r1500", 1500, 1480.0, 70.0, 0.42),  # the knee
            _clock_run("r1300", 1300, 1280.0, 55.0, 0.50),
        ]
        result = compute_clock_flexibility_metrics(sweep)
        assert result["clock_band_mhz"] == 0.0
        assert result["clock_knee_savings_pct"] == pytest.approx(16.0)

    def test_zero_stock_joules_per_token_withholds_knee_savings_only(self):
        sweep = [
            _clock_run("stock", None, 2100.0, 100.0, 0.0),
            _clock_run("r1500", 1500, 1480.0, 96.0, 0.45),
            _clock_run("r1300", 1300, 1280.0, 90.0, 0.42),
            _clock_run("r1100", 1100, 1080.0, 60.0, 0.50),
        ]
        result = compute_clock_flexibility_metrics(sweep)
        assert result["clock_knee_savings_pct"] is None
        # clock_band_mhz is unaffected by the zero-J/token guard.
        assert result["clock_band_mhz"] is not None

    def test_unresolvable_stock_reference_clock_withholds_band_only(self):
        # Stock is unlocked AND its telemetry never captured
        # mean_gpu_sm_clock_mhz (older collector) -- unlike mean_gpu_w on the
        # power side, this field IS nullable, so the reference clock itself
        # can be unavailable even though stock is otherwise resolvable.
        sweep = [
            _clock_run("stock", None, None, 60.0, 0.55),
            _clock_run("r1900", 1900, 1895.0, 59.0, 0.50),
            _clock_run("r1700", 1700, 1690.0, 56.0, 0.48),  # the knee
            _clock_run("r1500", 1500, 1480.0, 45.0, 0.52),
        ]
        result = compute_clock_flexibility_metrics(sweep)
        assert result["clock_band_mhz"] is None
        # clock_knee_savings_pct doesn't depend on the reference clock, so
        # it's still computed.
        assert result["clock_knee_savings_pct"] == pytest.approx((0.55 - 0.48) / 0.55 * 100.0)
        assert result["n_points"] == 4
