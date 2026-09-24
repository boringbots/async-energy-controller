"""Read-layer derived metrics: IPJ, per-watt, net-of-idle.

Ported verbatim from energy-bench's `metrics/derived.py` (US-MERGE-03) --
pure module, no dependency on anything else in this package. energy-bench's
METHODOLOGY.md quotes these formulas verbatim and hm-async's server-side
bench API implements the same formulas independently; the two must not
drift, which is why this is a straight port rather than a rewrite.

None of these are stored on `RunMetrics` -- every function None-propagates
(withhold rather than approximate); `joules_per_correct_answer` remains the
headline metric wherever these are displayed alongside it.
"""


def ipj(n_correct: int | None, total_joules: float | None) -> float | None:
    """Correct answers per joule: `n_correct / total_joules`.

    The literature-comparable inverse of `joules_per_correct_answer`.
    `None` when either input is `None` or `total_joules` is zero.
    """
    if n_correct is None or total_joules is None or total_joules == 0:
        return None
    return n_correct / total_joules


def accuracy_per_watt(accuracy: float | None, mean_gpu_w: float | None) -> float | None:
    """Accuracy per watt of mean GPU power draw: `accuracy / mean_gpu_w`.

    Time-free by construction (a slower run at equal power and accuracy
    scores identically) -- never the headline, `joules_per_correct_answer`
    is. `None` when either input is `None` or `mean_gpu_w` is zero.
    """
    if accuracy is None or mean_gpu_w is None or mean_gpu_w == 0:
        return None
    return accuracy / mean_gpu_w


def wall_accuracy_per_watt(accuracy: float | None, mean_wall_w: float | None) -> float | None:
    """Accuracy per watt of mean WALL power draw: `accuracy / mean_wall_w`.

    The deployment-honest variant of `accuracy_per_watt`: wall power
    includes PSU loss and node overhead an accelerator-only study can't see.
    `None` when either input is `None` or `mean_wall_w` is zero.
    """
    if accuracy is None or mean_wall_w is None or mean_wall_w == 0:
        return None
    return accuracy / mean_wall_w


def net_joules(
    total_joules: float | None,
    loaded_idle_w: float | None,
    duration_s: float | None,
) -> float | None:
    """Gross joules minus idle draw over the run: `total_joules - loaded_idle_w * duration_s`.

    `loaded_idle_w` must come from a `kind='loaded'` baseline (idle WITH the
    model resident) -- never `kind='empty'`, and never a fallback to gross
    joules when no loaded baseline exists for the node. `None` when any
    input is `None`. Can be negative if idle draw over the run exceeds gross
    joules (a very short run against a noisy baseline); callers display it
    as measured, never clamped to zero.
    """
    if total_joules is None or loaded_idle_w is None or duration_s is None:
        return None
    return total_joules - loaded_idle_w * duration_s


def wall_joules(mean_wall_w: float | None, duration_s: float | None) -> float | None:
    """Gross whole-machine joules: `mean_wall_w * duration_s`.

    The smart plug reports power, not energy (its cumulative kWh counter
    ticks in 0.01 kWh steps, coarser than most runs), so the whole-machine
    figure is always this product. `None` when either input is `None`.
    """
    if mean_wall_w is None or duration_s is None:
        return None
    return mean_wall_w * duration_s


def net_wall_joules(
    mean_wall_w: float | None,
    idle_wall_w: float | None,
    duration_s: float | None,
) -> float | None:
    """Whole-machine joules above the machine's own idle draw.

    `(mean_wall_w - idle_wall_w) * duration_s`.

    **The floor here is the `kind='empty'` baseline, not `kind='loaded'` --
    the opposite of `net_joules`.** The GPU metric subtracts loaded idle
    because a serving deployment keeps the model resident and the work sits
    on top of that. The wall metric cannot: "resident" is exactly the
    variable under test when engines place weights differently, so a loaded
    wall floor would be engine-specific and the rungs of a residency ladder
    would each be measured against a different zero. The empty baseline is
    the one thing every engine on a node shares -- what the box costs
    switched on and doing nothing.

    So this answers "what did choosing to run this workload cost, above
    owning the machine", while gross answers "what did the machine draw
    while doing it". Report gross as the headline and this beside it:
    on .114 idle is 68.9 W, which is 13 % of a vLLM run's draw and 19-20 %
    of a CPU-offloaded run's, so gross quietly favours whichever engine
    finishes sooner.

    `None` when any input is `None`. Can be negative if a run somehow drew
    less than idle; shown as measured, never clamped.
    """
    if mean_wall_w is None or idle_wall_w is None or duration_s is None:
        return None
    return (mean_wall_w - idle_wall_w) * duration_s
