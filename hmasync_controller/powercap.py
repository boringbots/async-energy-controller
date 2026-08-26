"""
powercap — apply the server-recommended power cap around one scheduled GPU job.

Gated by a single flag, `config.Settings.APPLY_POWER_CAP` (default False, same
opt-in shape as `BENCH_OPTIN`): off, nothing here is ever constructed or
called. On, and only on a box with an NVML-backed profiler (no GPU → nothing
to cap), `cli.build_executor` wires a `PowerCapManager` into the executor,
which brackets every scheduled job's `adapter.run()` (executor._execute):

    apply()   — before the job starts
    restore() — in a `finally`, so a crash, a timeout, or an adapter that
                raises still gets the prior limit put back

The recommendation comes from
`GET /api/v1/bench/nodes/{node_hash}/recommended-cap?tolerance_pct=<pct>`
(prd.json GROUND TRUTH (2): "null when no flexibility data — normal, not an
error", which is the normal state until this node has contributed bench data
via `register_node`). Fetched at most once per `cache_ttl_s` (default one day,
"refresh daily" per the AC) so a job never blocks on a network round-trip it
does not need — most ticks reuse the cached value.

**Never fails a job.** Every NVML call and every wire call here is caught;
`apply`/`restore` always return a status string describing what happened, for
the executor to log, and never raise past this module.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from hmasync_controller.profiler import NVMLProfiler, PowerCapPermissionError, Profiler

logger = logging.getLogger("hmasync.powercap")

# "refresh daily" (AC). A cap that changed server-side mid-day is picked up on
# the next refresh, not instantly — the tradeoff in exchange for never
# blocking a job on a network round-trip.
DEFAULT_CACHE_TTL_S = 86400.0

STATUS_APPLIED = "applied"
STATUS_RESTORED = "restored"
STATUS_NO_RECOMMENDATION = "no-recommendation"
STATUS_SKIPPED_NO_PERMISSION = "skipped-no-permission"
STATUS_SKIPPED_NO_GPU = "skipped-no-gpu"
STATUS_SKIPPED_ERROR = "skipped-error"
STATUS_NOTHING_TO_RESTORE = "nothing-to-restore"

#: Who owns the GPU's power limit. See `config.Settings.POWER_CAP_POLICY`.
POLICY_PRESERVE = "preserve"
POLICY_MANAGED = "managed"
_VALID_POLICIES = (POLICY_PRESERVE, POLICY_MANAGED)


def _extract_cap_w(data: Any) -> float | None:
    """Pull the recommended cap (watts) out of a recommended-cap response body.

    The response shape is not pinned down anywhere reachable from this repo —
    prd.json's GROUND TRUTH only says the endpoint exists and that a null
    means no recommendation. `recommended_cap_w` is this client's assumed
    field name, chosen to match the endpoint's own name; a bare numeric body
    is also accepted defensively. Anything else (missing field, non-numeric,
    a null body from a 204) is "no recommendation", never a fabricated cap.
    """
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return float(data)
    if isinstance(data, dict):
        value = data.get("recommended_cap_w")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


class PowerCapManager:
    """Fetches (and caches) the recommended cap; applies/restores it via NVML."""

    def __init__(
        self,
        *,
        client: Any,
        profiler: Profiler,
        node_hash: str,
        tolerance_pct: float = 5.0,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        clock: Callable[[], float] | None = None,
        policy: str = POLICY_PRESERVE,
    ):
        # An unrecognised policy falls back to "preserve" with a warning
        # rather than failing the job: a typo in .env must not stop scheduled
        # work, and "put back exactly what was there" is the choice that
        # cannot surprise anyone.
        if policy not in _VALID_POLICIES:
            logger.warning(
                "power cap: unknown POWER_CAP_POLICY %r; falling back to %r. Valid values: %s",
                policy,
                POLICY_PRESERVE,
                ", ".join(_VALID_POLICIES),
            )
            policy = POLICY_PRESERVE
        self._policy = policy
        self._client = client
        self._profiler = profiler
        self._node_hash = node_hash
        self._tolerance_pct = tolerance_pct
        self._cache_ttl_s = cache_ttl_s
        self._clock = clock or time.monotonic
        self._cached_cap_w: float | None = None
        self._cached_at: float | None = None
        # Set only once apply() has both read the prior limit AND successfully
        # written the new one — so restore() never fires on a limit we never
        # actually changed, and apply() never changes a limit it could not
        # later put back.
        self._prior_limit_w: float | None = None

    def _recommended_cap_w(self) -> float | None:
        """The cached recommendation, refreshed at most once per `cache_ttl_s`."""
        now = self._clock()
        if self._cached_at is None or (now - self._cached_at) >= self._cache_ttl_s:
            self._cached_cap_w = self._fetch_recommended_cap_w()
            self._cached_at = now
        return self._cached_cap_w

    def _fetch_recommended_cap_w(self) -> float | None:
        try:
            result = self._client.get_recommended_cap(
                self._node_hash, tolerance_pct=self._tolerance_pct
            )
        except Exception:
            logger.warning("power cap: fetching the recommendation raised", exc_info=True)
            return None
        if not result.ok:
            logger.info("power cap: could not fetch a recommendation (%s)", result.error)
            return None
        return _extract_cap_w(result.data)

    def apply(self) -> str:
        """Apply the cached/fetched recommendation before a job runs. Never raises."""
        if not isinstance(self._profiler, NVMLProfiler):
            return STATUS_SKIPPED_NO_GPU

        cap_w = self._recommended_cap_w()
        if cap_w is None:
            logger.info(
                "power cap: no recommendation for node %s; running uncapped", self._node_hash
            )
            return STATUS_NO_RECOMMENDATION

        prior = self._profiler.get_power_limit_w()
        if prior is None:
            logger.warning(
                "power cap: could not read the current power limit; not applying a cap"
            )
            return STATUS_SKIPPED_ERROR

        try:
            self._profiler.set_power_limit_w(cap_w)
        except PowerCapPermissionError:
            logger.warning(
                "power cap: no permission to set the power limit on this box; running uncapped"
            )
            return STATUS_SKIPPED_NO_PERMISSION
        except Exception:
            logger.warning("power cap: failed to apply a %.0fW cap", cap_w, exc_info=True)
            return STATUS_SKIPPED_ERROR

        self._prior_limit_w = prior
        logger.info("power cap: applied %.0fW (previous limit %.0fW)", cap_w, prior)

        # A limit BELOW the card's factory default means it was already capped
        # before this job started. Two causes look identical from here: the
        # operator set it deliberately, or an earlier job was killed (reboot,
        # OOM, SIGKILL) before its `finally` could restore. Because `restore()`
        # puts back what it FOUND, the second case is self-perpetuating -- the
        # card never returns to factory and nothing says so. Warn rather than
        # override: silently undoing a deliberate cap would be its own
        # surprise, and the operator is the only one who can tell the cases
        # apart. See HARDWARE-SAFETY.md, "Known limitation".
        default_w = self._profiler.get_power_limit_default_w()
        if default_w is not None and prior < default_w:
            if self._policy == POLICY_MANAGED:
                logger.info(
                    "power cap: GPU was at %.0fW before this job (factory default %.0fW); "
                    "POWER_CAP_POLICY=managed, so it will be restored to %.0fW afterward.",
                    prior,
                    default_w,
                    default_w,
                )
            else:
                logger.warning(
                    "power cap: this GPU was ALREADY capped at %.0fW before the job started "
                    "(card's factory default is %.0fW), and POWER_CAP_POLICY=preserve means "
                    "that %.0fW is what gets put back. If you set that cap yourself, nothing "
                    "is wrong. If you did not, it was probably left behind by a job killed "
                    "before it could restore, and every run since has been throttled -- set "
                    "POWER_CAP_POLICY=managed to have the controller restore the factory "
                    "default from now on, or fix it once with: nvidia-smi -pl %.0f",
                    prior,
                    default_w,
                    prior,
                    default_w,
                )
        return STATUS_APPLIED

    def restore(self) -> str:
        """Put the power limit back, per `POWER_CAP_POLICY`. Never raises.

        "preserve" restores the exact limit `apply()` found -- a cap the
        operator set themselves survives. "managed" restores the card's
        factory default instead, so a cap left behind by a killed job heals
        on the next run; the trade is that a deliberate operator cap is reset
        too, because the two are indistinguishable from here.

        Falls back to the found limit whenever the factory default cannot be
        read: restoring something known beats guessing a wattage.
        """
        prior = self._prior_limit_w
        self._prior_limit_w = None
        if prior is None:
            return STATUS_NOTHING_TO_RESTORE

        target = prior
        if self._policy == POLICY_MANAGED:
            default_w = self._profiler.get_power_limit_default_w()
            if default_w is not None:
                target = default_w

        try:
            self._profiler.set_power_limit_w(target)
        except Exception:
            logger.warning(
                "power cap: failed to restore the %.0fW limit", target, exc_info=True
            )
            return STATUS_SKIPPED_ERROR

        logger.info("power cap: restored %.0fW (policy=%s)", target, self._policy)
        return STATUS_RESTORED
