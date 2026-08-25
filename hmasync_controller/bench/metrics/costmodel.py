"""Per-run cost-model fit: E_i = e_fixed + alpha*prompt_i + beta*completion_i.

Ported near-verbatim from energy-bench's `metrics/costmodel.py`
(US-MERGE-03). Given a candidate config's prompt/completion token counts for
a request, alpha/beta/e_fixed predict its GPU energy without having to run
it. Per-item energy comes from linearly interpolating the GPU's cumulative
NVML energy counter (`TelemetrySample.gpu_energy_mj`) at each request's
`t_start_s`/`t_end_s` boundaries -- the counter is exact, unlike integrating
5 Hz power samples, so this is the same "prefer the counter" discipline as
`compute.compute_counter_energy`.

No numpy: solved via pure-Python normal equations (Gauss-Jordan elimination
with partial pivoting) rather than `numpy.linalg.lstsq`, so this module adds
no dependency of its own (`rouge-score`, needed by `bench.tasks.longctx_summary`,
pulls numpy transitively into this package's base install anyway -- see
US-MERGE-01's discrepancy note in `pyproject.toml` -- but this module never
imports it).
"""

from __future__ import annotations

import bisect

from hmasync_controller.bench.metrics.models import InferenceResult
from hmasync_controller.bench.sampler import TelemetrySample

_EMPTY_FIT: dict[str, float | int | None] = {
    "alpha_j_per_prompt_token": None,
    "beta_j_per_completion_token": None,
    "e_fixed_j": None,
    "costmodel_r2": None,
    "costmodel_n": None,
}

# 3 free parameters (e_fixed, alpha, beta): fewer usable items than that makes
# the normal equations rank-deficient by construction, before pivoting even
# gets a chance to detect it.
_MIN_ITEMS = 3

_SINGULAR_PIVOT_EPS = 1e-9


def _interpolate(ts: float, times: list[float], energies: list[float]) -> float | None:
    """Linearly interpolate the cumulative energy counter at `ts`.

    `times` must be sorted ascending. Returns None if `ts` falls outside the
    sampled range -- the counter's value there is unknown, not extrapolatable.
    """
    if ts < times[0] or ts > times[-1]:
        return None
    idx = bisect.bisect_left(times, ts)
    if times[idx] == ts:
        return energies[idx]
    lo, hi = idx - 1, idx
    t0, t1 = times[lo], times[hi]
    e0, e1 = energies[lo], energies[hi]
    frac = (ts - t0) / (t1 - t0)
    return e0 + frac * (e1 - e0)


def _item_energy_j(
    result: InferenceResult, times: list[float], energies: list[float]
) -> float | None:
    """Energy counter delta over one request's [t_start_s, t_end_s] window, in
    joules. None if the request has no timestamps, either falls outside the
    counter's sampled range, or the counter went backwards (a driver-reload
    reset mid-window -- the same case `compute.compute_counter_energy` rejects).

    Assumes requests are effectively serialized: with concurrent in-flight
    requests, this attributes the whole telemetry window's energy delta to
    each overlapping request, which double-counts shared draw.
    """
    if result.t_start_s is None or result.t_end_s is None:
        return None
    e_start = _interpolate(result.t_start_s, times, energies)
    e_end = _interpolate(result.t_end_s, times, energies)
    if e_start is None or e_end is None:
        return None
    energy_j = (e_end - e_start) / 1000.0
    return energy_j if energy_j >= 0 else None


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Solve `matrix @ x = rhs` via Gauss-Jordan elimination with partial
    pivoting. Returns None if `matrix` is singular (or near enough that
    pivoting can't find a usable row) rather than dividing by ~0.
    """
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < _SINGULAR_PIVOT_EPS:
            return None
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / pivot
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]

    return [aug[i][n] / aug[i][i] for i in range(n)]


def compute_item_energies_j(
    samples: list[TelemetrySample], inference_results: list[InferenceResult]
) -> list[float | None]:
    """Per-item GPU energy (joules), same order as `inference_results`.

    The shared attribution step behind both `fit_cost_model` and
    `metrics.compute`'s within-run confidence intervals -- interpolates the
    NVML energy counter at each request's `[t_start_s, t_end_s]` window (see
    `_item_energy_j`). An entry is None wherever that interpolation is
    unavailable for that item, or every entry is None when the counter
    itself has fewer than 2 readings (nothing to interpolate against).
    """
    readings = sorted(
        ((s.ts, s.gpu_energy_mj) for s in samples if s.gpu_energy_mj is not None),
        key=lambda pair: pair[0],
    )
    if len(readings) < 2:
        return [None] * len(inference_results)

    times = [t for t, _ in readings]
    energies = [e for _, e in readings]
    return [_item_energy_j(result, times, energies) for result in inference_results]


def fit_cost_model(
    samples: list[TelemetrySample], inference_results: list[InferenceResult]
) -> dict[str, float | int | None]:
    """Least-squares fit of E_i = e_fixed + alpha*prompt_i + beta*completion_i
    over one run's items.

    Returns a dict with keys `alpha_j_per_prompt_token`,
    `beta_j_per_completion_token`, `e_fixed_j`, `costmodel_r2`, `costmodel_n`
    -- spread straight onto `RunMetrics`. All None when the GPU energy
    counter has fewer than 2 readings (nothing to interpolate against), or
    fewer than `_MIN_ITEMS` requests have a usable per-item energy (not
    enough degrees of freedom for 3 parameters, or the token counts are too
    collinear to solve).
    """
    energies_j = compute_item_energies_j(samples, inference_results)
    rows: list[tuple[int, int, float]] = [
        (result.prompt_tokens, result.completion_tokens, energy_j)
        for result, energy_j in zip(inference_results, energies_j)
        if energy_j is not None
    ]

    if len(rows) < _MIN_ITEMS:
        return dict(_EMPTY_FIT)

    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for prompt_tokens, completion_tokens, energy_j in rows:
        x = [1.0, float(prompt_tokens), float(completion_tokens)]
        for i in range(3):
            xty[i] += x[i] * energy_j
            for j in range(3):
                xtx[i][j] += x[i] * x[j]

    coeffs = _solve_linear_system(xtx, xty)
    if coeffs is None:
        return dict(_EMPTY_FIT)
    e_fixed, alpha, beta = coeffs

    n = len(rows)
    mean_energy = sum(energy_j for _, _, energy_j in rows) / n
    ss_res = 0.0
    ss_tot = 0.0
    for prompt_tokens, completion_tokens, energy_j in rows:
        predicted = e_fixed + alpha * prompt_tokens + beta * completion_tokens
        ss_res += (energy_j - predicted) ** 2
        ss_tot += (energy_j - mean_energy) ** 2

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return {
        "alpha_j_per_prompt_token": alpha,
        "beta_j_per_completion_token": beta,
        "e_fixed_j": e_fixed,
        "costmodel_r2": r2,
        "costmodel_n": n,
    }
