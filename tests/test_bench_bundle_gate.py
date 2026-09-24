"""The submission gate: a row whose two energy totals disagree stays home."""

from __future__ import annotations

from hmasync_controller.bench.bundle import COUNTER_INTEGRATION_MAX_PCT_DIFF, submission_gate_reason
from hmasync_controller.bench.metrics.models import RunMetrics


def _row(**overrides) -> RunMetrics:
    base = dict(
        run_id="r", label="l", model="m", target_host="h", joules_per_token=1.0,
        total_joules_gpu=10.0, kwh_delta=None, peak_gpu_w=5.0, mean_gpu_w=3.0,
        mean_tokens_per_second=1.0, run_duration_s=10.0,
    )
    base.update(overrides)
    return RunMetrics(**base)


def test_an_awake_row_passes():
    assert submission_gate_reason(_row(counter_vs_integration_pct_diff=1.2)) is None


def test_the_labs_known_integration_bias_passes():
    """+5.7-7.3% on the NVIDIA cards is the 5 Hz integral's own bias, not a
    sleeping sampler; the limit sits above it."""
    assert COUNTER_INTEGRATION_MAX_PCT_DIFF > 7.3
    assert submission_gate_reason(_row(counter_vs_integration_pct_diff=7.3)) is None


def test_a_slept_row_is_refused_and_says_why():
    reason = submission_gate_reason(_row(counter_vs_integration_pct_diff=573.0, sampler_gap_s=17331.0))
    assert reason is not None
    assert "573%" in reason and "17331s" in reason


def test_an_unread_counter_is_not_a_reason():
    assert submission_gate_reason(_row(counter_vs_integration_pct_diff=None)) is None
