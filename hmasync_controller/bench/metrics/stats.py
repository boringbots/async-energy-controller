"""Within-run statistics: bootstrap CIs, Wilson intervals, pooled sigma.

Ported verbatim from energy-bench's `metrics/stats.py` (US-MERGE-03) --
pure module, no dependency on `models.py` or anything else in this package.
Callers supply already-computed per-item energies
(`metrics/costmodel.py`'s attribution) and correctness; this module only
does the resampling and interval arithmetic.
"""

import random
import statistics

_Z_95 = statistics.NormalDist().inv_cdf(0.975)

# J/correct is unbounded when a resample scores zero correct answers; a
# handful of such resamples are normal noise, but past this fraction the
# percentile CI itself would be dominated by which resamples happened to
# avoid the singularity.
_MAX_INVALID_FRACTION = 0.05


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted list."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = (pct / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def bootstrap_jpc_ci(
    item_energies: list[float | None],
    item_correct: list[float | None],
    total_joules: float | None,
    n_boot: int = 10000,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Percentile 95% bootstrap CI on joules-per-correct-answer.

    Jointly resamples (energy, correctness) pairs for items where BOTH are
    known. `total_joules` minus the sum of attributed item energies is the
    run's non-attributed remainder (telemetry overhead the per-item
    interpolation window doesn't cover); it is amortized uniformly across the
    attributable items so each resample's energy sum stays consistent with
    the run's actual measured total.

    Deterministic for a given `seed` (callers pass the run's `probe.seed`).
    Returns `(None, None)` when attribution is impossible (no item has both
    an energy and a correctness value), or when more than 5% of resamples
    score zero correct answers -- J/correct is unbounded there, so the
    percentile is withheld rather than let it be dominated by whichever
    resamples happened to dodge the singularity.
    """
    if total_joules is None:
        return (None, None)

    attributable = [
        (energy, correct)
        for energy, correct in zip(item_energies, item_correct)
        if energy is not None and correct is not None
    ]
    m = len(attributable)
    if m == 0:
        return (None, None)

    attributed_sum = sum(energy for energy, _ in attributable)
    remainder_share = (total_joules - attributed_sum) / m
    full_energies = [energy + remainder_share for energy, _ in attributable]
    corrects = [correct for _, correct in attributable]

    rng = random.Random(seed)
    indices = range(m)
    ratios: list[float] = []
    invalid = 0
    for _ in range(n_boot):
        sample = rng.choices(indices, k=m)
        energy_sum = sum(full_energies[i] for i in sample)
        n_correct = round(sum(corrects[i] for i in sample))
        if n_correct >= 1:
            ratios.append(energy_sum / n_correct)
        else:
            invalid += 1

    if invalid / n_boot > _MAX_INVALID_FRACTION:
        return (None, None)

    ratios.sort()
    return (_percentile(ratios, 2.5), _percentile(ratios, 97.5))


def accuracy_ci(n_correct: int, n_scored: int) -> tuple[float | None, float | None]:
    """Wilson score 95% interval on accuracy.

    Returns `(None, None)` when `n_scored` is zero -- accuracy itself is
    undefined there, so no interval can be.
    """
    if n_scored <= 0:
        return (None, None)

    n = n_scored
    phat = n_correct / n
    denom = 1 + _Z_95**2 / n
    center = phat + _Z_95**2 / (2 * n)
    margin = _Z_95 * ((phat * (1 - phat) / n) + (_Z_95**2 / (4 * n**2))) ** 0.5
    low = max(0.0, (center - margin) / denom)
    high = min(1.0, (center + margin) / denom)
    return (low, high)


def pooled_mean_sigma(values: list[float]) -> tuple[float | None, float | None, int]:
    """Mean and sample standard deviation (ddof=1) of `values`.

    Sigma is `None` for n<2 -- a sample standard deviation needs at least two
    observations. Mean is `None` only when `values` is empty.
    """
    n = len(values)
    if n == 0:
        return (None, None, 0)
    mean = statistics.mean(values)
    if n < 2:
        return (mean, None, n)
    return (mean, statistics.stdev(values), n)
