"""Flexibility metrics from a power-limit or clock-lock sweep.

Ported near-verbatim from energy-bench's `grading/flexibility.py`
(US-MERGE-03) -- physically filed under this package's `metrics/` rather
than a separate `grading/` layer, since the lab-only grading concepts
(Efficiency Index, reference-config normalization, the six-axis grade card)
all stay in energy-bench per the merge PRD's GROUND TRUTH.

Three numbers computed from all runs of ONE (model, quantization, engine,
target_host, task) configuration measured across >=4 distinct
`power_limit_w` points (a power sweep -- everything else held fixed):

* `flex_band_w` -- the widest cap reduction (in Watts) that still keeps
  throughput >= 95% of stock's.
* `knee_savings_pct` -- percent energy-per-token saved at the single best
  (lowest J/token) cap point, versus stock.
* `turndown_ratio` -- stock mean GPU Watts / mean GPU Watts at the lowest
  cap that still sustains >= 50% of stock throughput (the "cliff edge").

"Stock" is the sweep's uncapped point (`power_limit_w is None`). Some
sweeps instead pin an explicit numeric cap equal to the card's own default
as their "stock" point -- when no run in the group has `power_limit_w is
None`, the run with the HIGHEST `power_limit_w` stands in for stock instead.

`compute_clock_flexibility_metrics()` is the analogue for a
`clock_lock_mhz` sweep -- same grouping contract, same stock-resolution
rules, same `MIN_SWEEP_POINTS` guard, returning `clock_band_mhz` /
`clock_knee_savings_pct` / `n_points`. It has deliberately no
`turndown_ratio` analogue.

Not wired into `metrics.compute.compute_metrics()` -- like energy-bench's
Efficiency Index, this needs every OTHER run in the same sweep group, which
doesn't exist yet when any single run's metrics are computed. This ships
the pure computation only, not any grouping/persistence around it (that is
lab-side, `eb grades rebuild`).
"""

from __future__ import annotations

from hmasync_controller.bench.metrics.models import RunMetrics

MIN_SWEEP_POINTS = 4
"""Fewer than this many power_limit_w points and there isn't a real sweep to
derive Flexibility from."""

_FLEX_BAND_THRESHOLD = 0.95
_TURNDOWN_THRESHOLD = 0.50

_EMPTY_RESULT = {
    "flex_band_w": None,
    "knee_savings_pct": None,
    "turndown_ratio": None,
}


def sweep_config_key(
    model: str, quantization: str | None, engine: str, target_host: str
) -> str:
    """The sweep-group identifying key. Deliberately excludes `task` -- a
    single config_key spans every task measured against it."""
    return f"{model}|{quantization or '-'}|{engine}|{target_host}"


def _field(run: RunMetrics | dict, name: str) -> object:
    """`run[name]` or `run.name`, whichever `run` is -- lets callers pass
    either a freshly built `RunMetrics` or a plain dict (e.g. a bundle's
    `run` entry, or a lab-side `runs` row)."""
    return run.get(name) if isinstance(run, dict) else getattr(run, name, None)


def compute_flexibility_metrics(runs: list[RunMetrics] | list[dict]) -> dict:
    """Flexibility metrics for one power-sweep group.

    Args:
        runs: Every run of ONE (model, quantization, engine, target_host,
            task) configuration, differing only in `power_limit_w`. Order
            does not matter.

    Returns:
        `{"flex_band_w": ..., "knee_savings_pct": ..., "turndown_ratio":
        ..., "n_points": len(runs)}`. The three metrics are `None` when they
        can't be determined at all (too few points, an ambiguous stock
        point, or -- for `turndown_ratio` only -- no capped point sustains
        50% of stock's throughput). `flex_band_w` is `0.0`, not `None`, when
        no capped point keeps 95% of stock's throughput: "the widest
        reduction that still qualifies" is a real answer in that case --
        stock itself, i.e. no reduction at all.
    """
    n_points = len(runs)
    if n_points < MIN_SWEEP_POINTS:
        return {**_EMPTY_RESULT, "n_points": n_points}

    # COLD-CACHE ONLY (2026-08-26). vLLM runs with `enable_prefix_caching=True`
    # and the hit rate climbs across successive runs against one server
    # (measured: 0.0% -> 73.5% -> 80.4% -> 83.1% -> 85.1% -> 85.8%), so
    # repeats are not independent samples -- they measure cache warmth. Two
    # concrete failures that filtering fixes:
    #
    #   * `knee_savings_pct` takes the SINGLE lowest-J/token capped run, which
    #     with repeats is always the most cache-discounted warm one. On the
    #     first real sweep it reported 58.6% where the cold-cache figure is
    #     ~27%, and that number is quoted verbatim in a "publishable claim".
    #   * `stock_candidates` counts every uncapped REPEAT, so an uncapped
    #     point measured 3 times looked like 3 ambiguous stock points and made
    #     the whole computation withhold.
    #
    # Falls back to all runs when none carry a repeat index, so a
    # single-run-per-cap sweep (the community `bench quick` shape) is
    # unaffected.
    cold_runs = [r for r in runs if (_field(r, "repeat_index") or 0) == 0]
    if cold_runs:
        runs = cold_runs

    stock_candidates = [r for r in runs if _field(r, "power_limit_w") is None]
    if len(stock_candidates) > 1:
        # More than one uncapped point in the group -- which one is "stock"
        # is ambiguous. Withhold rather than guess.
        return {**_EMPTY_RESULT, "n_points": n_points}
    elif len(stock_candidates) == 1:
        stock_run = stock_candidates[0]
    else:
        stock_run = max(runs, key=lambda r: _field(r, "power_limit_w"))

    capped_runs = [r for r in runs if r is not stock_run]
    if len(capped_runs) < MIN_SWEEP_POINTS - 1:
        return {**_EMPTY_RESULT, "n_points": n_points}

    stock_power_limit_w = _field(stock_run, "power_limit_w")
    stock_reference_w = (
        float(stock_power_limit_w)
        if stock_power_limit_w is not None
        else _field(stock_run, "mean_gpu_w")
    )
    stock_tok_s = _field(stock_run, "mean_tokens_per_second")
    stock_mean_w = _field(stock_run, "mean_gpu_w")
    stock_j_per_token = _field(stock_run, "joules_per_token")

    # flex_band_w: among capped runs still >= 95% of stock throughput, the
    # deepest cut (lowest power_limit_w) sets the band's width. No
    # qualifying point means the band is zero-width -- stock is the only
    # point that clears the bar.
    flex_band_floor = _FLEX_BAND_THRESHOLD * stock_tok_s
    qualifying_95 = [
        r for r in capped_runs if _field(r, "mean_tokens_per_second") >= flex_band_floor
    ]
    if qualifying_95:
        deepest = min(qualifying_95, key=lambda r: _field(r, "power_limit_w"))
        flex_band_w = stock_reference_w - _field(deepest, "power_limit_w")
    else:
        flex_band_w = 0.0

    # knee_savings_pct: the single most efficient capped point (lowest
    # J/token), regardless of whether it also clears the 95%/50% throughput
    # bars used above/below.
    knee_savings_pct = None
    if stock_j_per_token:
        best_cap = min(capped_runs, key=lambda r: _field(r, "joules_per_token"))
        best_j_per_token = _field(best_cap, "joules_per_token")
        knee_savings_pct = (stock_j_per_token - best_j_per_token) / stock_j_per_token * 100.0

    # turndown_ratio: the lowest cap that still sustains >= 50% of stock
    # throughput is the cliff edge. Unlike flex_band_w, there is no
    # zero-width fallback here -- if nothing sustains half of stock's
    # throughput, the cliff edge was never observed, so the ratio is
    # withheld rather than guessed.
    turndown_ratio = None
    turndown_floor = _TURNDOWN_THRESHOLD * stock_tok_s
    qualifying_50 = [
        r for r in capped_runs if _field(r, "mean_tokens_per_second") >= turndown_floor
    ]
    if qualifying_50:
        cliff = min(qualifying_50, key=lambda r: _field(r, "power_limit_w"))
        cliff_mean_w = _field(cliff, "mean_gpu_w")
        if cliff_mean_w:
            turndown_ratio = stock_mean_w / cliff_mean_w

    return {
        "flex_band_w": flex_band_w,
        "knee_savings_pct": knee_savings_pct,
        "turndown_ratio": turndown_ratio,
        "n_points": n_points,
    }


_EMPTY_CLOCK_RESULT = {
    "clock_band_mhz": None,
    "clock_knee_savings_pct": None,
}


def compute_clock_flexibility_metrics(runs: list[RunMetrics] | list[dict]) -> dict:
    """Flexibility metrics for one clock-lock-sweep group.

    Args:
        runs: Every run of ONE (model, quantization, engine, target_host,
            task) configuration, differing only in `clock_lock_mhz`. Order
            does not matter.

    Returns:
        `{"clock_band_mhz": ..., "clock_knee_savings_pct": ..., "n_points":
        len(runs)}`. `clock_band_mhz` is `0.0`, not `None`, when no locked
        point keeps 95% of stock's throughput -- same "stock itself is the
        widest qualifying reduction" reading `compute_flexibility_metrics`
        uses for `flex_band_w`. Both metrics are `None` when they can't be
        determined: too few points, an ambiguous stock point, or (for
        `clock_band_mhz` only) stock's reference clock is itself unavailable
        (neither an explicit `clock_lock_mhz` nor a measured
        `mean_gpu_sm_clock_mhz`).
    """
    n_points = len(runs)
    if n_points < MIN_SWEEP_POINTS:
        return {**_EMPTY_CLOCK_RESULT, "n_points": n_points}

    stock_candidates = [r for r in runs if _field(r, "clock_lock_mhz") is None]
    if len(stock_candidates) > 1:
        return {**_EMPTY_CLOCK_RESULT, "n_points": n_points}
    elif len(stock_candidates) == 1:
        stock_run = stock_candidates[0]
    else:
        stock_run = max(runs, key=lambda r: _field(r, "clock_lock_mhz"))

    capped_runs = [r for r in runs if r is not stock_run]
    if len(capped_runs) < MIN_SWEEP_POINTS - 1:
        return {**_EMPTY_CLOCK_RESULT, "n_points": n_points}

    stock_clock_lock_mhz = _field(stock_run, "clock_lock_mhz")
    stock_reference_mhz = (
        float(stock_clock_lock_mhz)
        if stock_clock_lock_mhz is not None
        else _field(stock_run, "mean_gpu_sm_clock_mhz")
    )
    stock_tok_s = _field(stock_run, "mean_tokens_per_second")
    stock_j_per_token = _field(stock_run, "joules_per_token")

    # clock_band_mhz: among locked runs still >= 95% of stock throughput, the
    # deepest cut (lowest clock_lock_mhz) sets the band's width. Withheld
    # (not zero-width) when stock's own reference clock can't be resolved --
    # unlike mean_gpu_w, mean_gpu_sm_clock_mhz is nullable.
    clock_band_mhz = None
    if stock_reference_mhz is not None:
        flex_band_floor = _FLEX_BAND_THRESHOLD * stock_tok_s
        qualifying_95 = [
            r for r in capped_runs if _field(r, "mean_tokens_per_second") >= flex_band_floor
        ]
        if qualifying_95:
            deepest = min(qualifying_95, key=lambda r: _field(r, "clock_lock_mhz"))
            clock_band_mhz = stock_reference_mhz - _field(deepest, "clock_lock_mhz")
        else:
            clock_band_mhz = 0.0

    # clock_knee_savings_pct: the single most efficient locked point (lowest
    # J/token), regardless of whether it also clears the 95% throughput bar
    # used above.
    clock_knee_savings_pct = None
    if stock_j_per_token:
        best_lock = min(capped_runs, key=lambda r: _field(r, "joules_per_token"))
        best_j_per_token = _field(best_lock, "joules_per_token")
        clock_knee_savings_pct = (
            (stock_j_per_token - best_j_per_token) / stock_j_per_token * 100.0
        )

    return {
        "clock_band_mhz": clock_band_mhz,
        "clock_knee_savings_pct": clock_knee_savings_pct,
        "n_points": n_points,
    }
