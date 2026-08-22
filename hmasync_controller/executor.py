"""
ScheduleExecutor — the controller's execution loop.

This is the piece that turns a pulled schedule into actual runs. Each tick:

  1. Poll `GET /schedule?after=<version>`; adopt a newer version when one lands.
  2. Opportunistically drain the offline spool when the API is reachable again.
  3. Execute every placement whose start (or fallback target) has arrived, one at a
     time (single-GPU serialization), each wrapped in the profiler and reported
     upstream (or spooled) with started/finished/failed acks.

**Degrade explicitly.** The executor follows the *last known* schedule
regardless of whether the API is reachable:
  - **Within `valid_until`** — run each placement at its planned `start`, even while
    the API is down (run records spool, acks are best-effort).
  - **Past `valid_until` with no newer version** — apply the schedule's embedded
    `fallback_policy`. v1 default `deadline_latest_start`: run each remaining job at
    its *latest feasible start* (`deadline - duration`), deadline-safe and
    price-blind. It never invents placements and never silently skips a feasible
    deadlined job.

**Pre-flight.** Before running a placement the executor calls the
adapter's `preflight()`; a failure sends a `failed` ack immediately (so the API can
replan) instead of burning the window on a doomed run.

**Overrun containment.** Every job is bounded by its window:
a job declaring no `timeout` gets one filled in from the time left in its
placement (`_bounded_request`), so an overrun becomes a `failed` ack the API can
replan around instead of an unbounded run that burns through the cheap window
into peak pricing. Within its bound a job still runs to completion — `run()`
blocks — so a slow-but-finishing job delays its successors. True preemption
(SIGTERM at the boundary, resume in the next window) is the remaining upgrade.

**Clock discipline.** The clock is injected (`now_fn`) so tests are
deterministic; every timestamp that leaves the executor is timezone-aware.

The executor is a pure orchestrator over the already-built seams — ApiClient
(wire), RunReporter (report/spool/drain), Profiler (telemetry), and the Adapter
registry (run/fingerprint/preflight). It resolves a placement's `workflow_id` to a
local job definition (framework + request) via an injected `job_source`; the wire
contract stays the fixed four endpoints (widening it is a design
conversation, not a quiet edit), so the controller carries its own small job
catalog rather than growing a fifth "fetch workflow" route.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from hmasync_controller.adapters import (
    Adapter,
    AdapterError,
    AdapterRunResult,
    EXIT_ERROR,
    declared_timeout,
    get_adapter,
)
from hmasync_controller.profiler import Profiler, RunTelemetry, get_profiler
from hmasync_controller.reporter import RunReporter

logger = logging.getLogger("hmasync.executor")

# The only fallback policy implemented in v1. An unknown policy string
# degrades to this deadline-safe behaviour rather than doing nothing.
FALLBACK_DEADLINE_LATEST_START = "deadline_latest_start"

# Floor for a window-derived timeout. A job starting near the end of its window
# (a late tick, a long-running predecessor) would otherwise get a timeout of
# seconds and be killed before it could ever succeed. One minute is short enough
# to still contain a hang and long enough that a punctual job is never the
# casualty of arithmetic.
MIN_WINDOW_TIMEOUT_S = 60.0


def _utcnow() -> datetime:
    """Timezone-aware now — no naive datetime ever leaves the executor."""
    return datetime.now(timezone.utc)


def _ensure_aware(dt: datetime) -> datetime:
    """Attach UTC to a naive datetime; leave an aware one untouched."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_dt(value: Any) -> datetime | None:
    """Coerce an ISO string / datetime / None into a tz-aware datetime or None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _ensure_aware(value)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):  # tolerate a trailing 'Z' (fromisoformat pre-3.11)
            text = text[:-1] + "+00:00"
        try:
            return _ensure_aware(datetime.fromisoformat(text))
        except ValueError:
            return None
    return None


# ============================================================
# Local job catalog
# ============================================================
@dataclass
class JobDef:
    """The controller's local knowledge of a workflow — how to actually run it.

    A pulled `Placement` only carries a `workflow_id` (the wire contract is
    deliberately narrow), so the executor resolves it to a `JobDef` to learn the
    `framework` + adapter `request`. `deadline` (resolved, tz-aware) is optional but
    lets the `deadline_latest_start` fallback compute a genuine latest start; absent,
    the fallback falls back to the placement's planned `start` (still deadline-safe).
    """

    framework: str
    request: dict[str, Any] = field(default_factory=dict)
    workflow_id: str | None = None
    deadline: datetime | None = None
    earliest_start: datetime | None = None


# A job source maps a workflow_id to its JobDef (or None if the controller has no
# local definition for it). A plain dict is accepted for convenience.
JobSource = Callable[[str | None], "JobDef | dict | None"]


def _normalize_job_source(source: JobSource | Mapping | None) -> JobSource:
    if source is None:
        return lambda _wid: None
    if callable(source):
        return source
    if isinstance(source, Mapping):
        return lambda wid: source.get(wid)
    raise TypeError("job_source must be a callable or a mapping")


# ============================================================
# Parsed schedule
# ============================================================
@dataclass
class Placement:
    """A placement parsed from a pulled schedule (tz-aware windows)."""

    workflow_id: str | None
    start: datetime | None
    end: datetime | None
    predicted_wh: float | None
    feasible: bool
    reason: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveSchedule:
    """The schedule the executor is currently following."""

    version: int
    valid_until: datetime | None
    fallback_policy: str
    degraded: bool
    placements: list[Placement]


def _parse_placement(data: dict[str, Any]) -> Placement:
    return Placement(
        workflow_id=data.get("workflow_id"),
        start=_parse_dt(data.get("start")),
        end=_parse_dt(data.get("end")),
        predicted_wh=data.get("predicted_wh"),
        feasible=bool(data.get("feasible", True)),
        reason=data.get("reason"),
        raw=data,
    )


def _parse_schedule(data: dict[str, Any]) -> ActiveSchedule:
    placements = [
        _parse_placement(p) for p in (data.get("placements") or []) if isinstance(p, dict)
    ]
    return ActiveSchedule(
        version=int(data.get("version", 0)),
        valid_until=_parse_dt(data.get("valid_until")),
        fallback_policy=data.get("fallback_policy") or FALLBACK_DEADLINE_LATEST_START,
        degraded=bool(data.get("degraded", False)),
        placements=placements,
    )


# ============================================================
# Tick results (observability + tests)
# ============================================================
@dataclass
class PlacementOutcome:
    """What happened to one placement this tick."""

    workflow_id: str | None
    run_id: str | None
    status: str  # executed | preflight_failed | unknown_job | unknown_framework | error
    exit_status: str | None = None
    spooled: bool = False
    reason: str | None = None


@dataclass
class TickResult:
    """The result of one executor tick.

    `placements` and `pending` exist because `outcomes=0` is otherwise ambiguous:
    it reads identically whether the schedule is empty, the catalog is empty, a
    workflow id is mistyped, or it is simply 16:00 and the window opens at 20:00.
    Counting what the executor is *holding* separates "nothing to do yet" from
    "nothing to do, ever" without the operator having to guess.
    """

    version: int
    adopted: bool  # a newer schedule was adopted this tick
    mode: str  # normal | fallback | idle
    outcomes: list[PlacementOutcome] = field(default_factory=list)
    drained: int = 0
    reachable: bool = True
    # Placements in the schedule currently held (0 before the first pull).
    placements: int = 0
    # Feasible placements not yet attempted — work still ahead of this controller.
    pending: int = 0
    # The next placement's target start, for the "why is nothing running?" question.
    next_start: datetime | None = None


# ============================================================
# Executor
# ============================================================
class ScheduleExecutor:
    """Executes pulled schedules, degrading explicitly when the API is down."""

    def __init__(
        self,
        *,
        client: Any,
        reporter: RunReporter,
        job_source: JobSource | Mapping | None,
        profiler: Profiler | None = None,
        controller_id: str = "",
        now_fn: Callable[[], datetime] | None = None,
        adapter_provider: Callable[[str], Adapter] | None = None,
        run_id_factory: Callable[[], str] | None = None,
        extra_drain: Callable[[], int] | None = None,
        power_cap: Any = None,
    ):
        self.client = client
        self.reporter = reporter
        self.profiler = profiler or get_profiler()
        self.controller_id = controller_id
        # An optional second drain, run on the SAME reconnect trigger as the
        # run-report spool (reachable again → flush). Wired by cli.py to the
        # bench-bundle spool when bench opt-in is on; None otherwise, so an
        # un-opted-in controller never touches it. Never allowed to break a
        # tick — see `_safe_extra_drain`.
        self._extra_drain = extra_drain
        # An optional powercap.PowerCapManager (duck-typed as `apply()`/
        # `restore()` -> str, like `client`/`reporter` above), wired by cli.py
        # only when APPLY_POWER_CAP is set on a box with an NVML-backed
        # profiler. None otherwise, so an un-opted-in controller never polls
        # or touches an NVML power limit. Never allowed to break a job — see
        # `_safe_power_cap_apply`/`_safe_power_cap_restore`.
        self.power_cap = power_cap
        # The source as handed in (mapping, watcher, or callable). `_job_source` is
        # the normalized lookup the executor calls; a mapping normalizes into a
        # closure, which would hide `len()` from anything wanting to report the
        # catalog size — so the original is kept as the public one.
        self.job_source = job_source
        self._job_source = _normalize_job_source(job_source)
        self._now_fn = now_fn or _utcnow
        self._adapter_provider = adapter_provider or (lambda fw: get_adapter(fw))
        self._run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)

        self._adapters: dict[str, Adapter] = {}
        self._current: ActiveSchedule | None = None
        self._version = -1
        # Placements already attempted, keyed by (workflow_id, start-iso). Persists
        # across version adoptions so a recurring instance is attempted at most once;
        # a replan that re-schedules the same workflow at a *new* start gets a new key
        # and is run again. Bounded by jobs actually seen.
        self._executed: set[tuple[str | None, str | None]] = set()

    # --- public surface ---------------------------------------------------

    @property
    def version(self) -> int:
        """The highest schedule version adopted so far (-1 before the first pull)."""
        return self._version

    @property
    def current(self) -> ActiveSchedule | None:
        return self._current

    def sync_schedule(self) -> tuple[bool, bool]:
        """Pull the newest schedule; adopt it if newer. Returns (reachable, adopted).

        `reachable` is False only on a transport failure (the API is unreachable) —
        a 204 (no newer schedule) or an HTTP error still counts as reachable. Adopting
        never resets the executed set, so jobs already run are not re-run.
        """
        result = self.client.pull_schedule(after=self._version)
        reachable = not getattr(result, "transport_error", False)
        if result.ok and result.status_code == 200 and isinstance(result.data, dict):
            self._adopt(result.data)
            return reachable, True
        return reachable, False

    def tick(self, now: datetime | None = None) -> TickResult:
        """One executor step: sync, drain, then execute everything due at `now`."""
        now = self._now(now)
        reachable, adopted = self.sync_schedule()

        drained = 0
        if reachable:
            # Back in contact — flush anything spooled during the outage.
            drained = self.reporter.drain_spool().drained
            if self._extra_drain is not None:
                drained += self._safe_extra_drain()

        if self._current is None:
            return TickResult(self._version, adopted, "idle", [], drained, reachable)

        mode = self._mode(now)
        outcomes: list[PlacementOutcome] = []
        for placement in self._due(now, mode):
            outcome = self._execute(placement, now)
            self._executed.add(self._key(placement))
            outcomes.append(outcome)

        # Counted after execution so `pending`/`next_start` describe what is still
        # ahead, not what was ahead when the tick began.
        pending = self._pending(mode)
        return TickResult(
            self._version, adopted, mode, outcomes, drained, reachable,
            placements=len(self._current.placements),
            pending=len(pending),
            next_start=pending[0][0] if pending else None,
        )

    def run_forever(
        self,
        *,
        poll_interval_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        stop: Callable[[], bool] | None = None,
        max_ticks: int | None = None,
        on_tick: Callable[[TickResult], None] | None = None,
    ) -> int:
        """Tick on a fixed cadence until `stop()` is true or `max_ticks` is reached.

        Returns the number of ticks run. Thin wrapper over `tick()`; the per-tick
        logic is what the tests exercise.

        `on_tick` receives each `TickResult`. The loop otherwise consumes its own
        results and drops them, which left `--once` (which logs its one tick) and
        the daemon (which logged nothing per tick) reporting differently — the
        long-running shape being the quieter of the two. A raising callback must
        never take the loop down with it: observability is not worth a job.
        """
        ticks = 0
        while True:
            if stop is not None and stop():
                break
            result = self.tick()
            ticks += 1
            if on_tick is not None:
                try:
                    on_tick(result)
                except Exception:
                    logger.exception("on_tick callback raised; continuing")
            if max_ticks is not None and ticks >= max_ticks:
                break
            sleep(poll_interval_s)
        return ticks

    def close(self) -> None:
        """Release any adapter-owned HTTP clients."""
        for adapter in self._adapters.values():
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass

    def _safe_extra_drain(self) -> int:
        """Run the injected extra_drain callback; a raise must never take a tick down."""
        try:
            return int(self._extra_drain())
        except Exception:
            logger.warning("extra_drain callback failed", exc_info=True)
            return 0

    def _safe_power_cap_apply(self) -> None:
        """Apply the power cap; PowerCapManager already never raises, but a raise
        here must still never block the job it is meant to be invisible to."""
        try:
            self.power_cap.apply()
        except Exception:
            logger.warning("power_cap.apply() raised; running uncapped", exc_info=True)

    def _safe_power_cap_restore(self) -> None:
        """Restore the power cap; a raise here must never mask the job's own result."""
        try:
            self.power_cap.restore()
        except Exception:
            logger.warning("power_cap.restore() raised", exc_info=True)

    # --- scheduling internals --------------------------------------------

    def _adopt(self, data: dict[str, Any]) -> None:
        self._current = _parse_schedule(data)
        self._version = self._current.version

    def _mode(self, now: datetime) -> str:
        """`normal` within the validity window, `fallback` past it."""
        vu = self._current.valid_until if self._current else None
        if vu is None or now <= vu:
            return "normal"
        return "fallback"

    def _pending(self, mode: str) -> list[tuple[datetime, Placement]]:
        """Feasible, un-attempted placements as (target_start, placement), earliest first.

        Everything this controller still intends to run, whether or not its time has
        come. `_due` is this list filtered to what has arrived; the tick line reports
        the rest, which is what distinguishes "waiting for 20:00" from "nothing here".
        """
        pending: list[tuple[datetime, Placement]] = []
        for placement in self._current.placements:
            if not placement.feasible or placement.start is None:
                continue  # infeasible / unplaced jobs have no window — never invent one
            if self._key(placement) in self._executed:
                continue
            target = self._target_start(placement, mode)
            if target is None:
                continue
            pending.append((target, placement))
        # Serialize deterministically: earliest target first, workflow_id as tiebreak.
        pending.sort(key=lambda item: (item[0], item[1].workflow_id or ""))
        return pending

    def _due(self, now: datetime, mode: str) -> list[Placement]:
        """Feasible, un-attempted placements whose target start has arrived, in order."""
        return [
            placement for target, placement in self._pending(mode) if now >= target
        ]

    def _target_start(self, placement: Placement, mode: str) -> datetime | None:
        """The time this placement should start: planned start, or fallback latest-start."""
        if mode != "fallback":
            return placement.start

        # deadline_latest_start: run as late as still meets the deadline.
        duration = None
        if placement.start is not None and placement.end is not None:
            duration = placement.end - placement.start
        job = self._resolve_job(placement.workflow_id)
        # Prefer the locally-known resolved deadline; fall back to the placement's
        # planned end (the optimizer guarantees end <= deadline, so this is safe).
        deadline = job.deadline if (job and job.deadline) else placement.end
        if deadline is not None and duration is not None:
            return deadline - duration
        return placement.start

    @staticmethod
    def _key(placement: Placement) -> tuple[str | None, str | None]:
        start_iso = placement.start.isoformat() if placement.start else None
        return (placement.workflow_id, start_iso)

    # --- execution -------------------------------------------------------

    def _execute(self, placement: Placement, now: datetime) -> PlacementOutcome:
        """Preflight → profile → run → report → ack for one placement. Never raises."""
        wid = placement.workflow_id
        version = self._current.version if self._current else 0

        job = self._resolve_job(wid)
        if job is None:
            # No local definition — a controller-config gap, not a schedule problem;
            # a replan won't help, so log and move on without acking failed.
            logger.warning("no local job for workflow_id=%s; skipping placement", wid)
            return PlacementOutcome(wid, None, "unknown_job", reason="no local job definition")

        try:
            adapter = self._adapter_for(job.framework)
        except AdapterError as exc:
            logger.warning("no adapter for framework=%r (workflow %s): %s", job.framework, wid, exc)
            return PlacementOutcome(wid, None, "unknown_framework", reason=str(exc))

        run_id = self._run_id_factory()

        # Pre-flight before the window burns: a doomed run gets a failed
        # ack immediately so the API can replan, rather than executing anyway.
        if not self._preflight(adapter, job, wid):
            self._ack(version, "failed", wid, now)
            return PlacementOutcome(wid, run_id, "preflight_failed", reason="preflight failed")

        fingerprint = self._fingerprint(adapter, job, wid)

        self._ack(version, "started", wid, now)

        # Profile the blocking run (the profiler samples in a background thread so it
        # never delays execution). The power cap (if wired) brackets the SAME run:
        # applied just before, restored in a `finally` so a crash/timeout/raise still
        # puts the prior limit back — the job's own outcome is decided independently.
        self.profiler.start(run_id)
        if self.power_cap is not None:
            self._safe_power_cap_apply()
        try:
            result = adapter.run(self._bounded_request(job, placement, now))
        except AdapterError as exc:
            result = AdapterRunResult(EXIT_ERROR, detail=str(exc))
        except Exception as exc:  # defensive: one bad job must not kill the loop
            logger.exception("adapter.run crashed for workflow %s", wid)
            result = AdapterRunResult(EXIT_ERROR, detail=f"adapter crashed: {exc}")
        finally:
            if self.power_cap is not None:
                self._safe_power_cap_restore()
        telemetry = self.profiler.stop(run_id)

        record = self._build_record(job, placement, run_id, fingerprint, result, now, telemetry)
        report = self.reporter.report_run(record, telemetry.samples)

        event = "finished" if result.ok else "failed"
        self._ack(version, event, wid, now)
        return PlacementOutcome(
            wid,
            run_id,
            "executed",
            exit_status=result.exit_status,
            spooled=report.spooled,
            reason=result.detail,
        )

    def _bounded_request(
        self, job: JobDef, placement: Placement, now: datetime
    ) -> dict[str, Any]:
        """The job's request with a `timeout` filled in from its window, if it has none.

        A job that declares no timeout used to run unbounded, and an unbounded job
        is the one failure this product cannot tolerate: it does not merely delay
        its successors, it runs out of the cheap window it was placed in and into
        peak pricing — the exact outcome the schedule existed to avoid — while the
        run record still reports a correctly-planned placement.

        The bound is the time left in the placement window, floored at
        `MIN_WINDOW_TIMEOUT_S`; a window with no end falls back to the local
        deadline. Overrunning now produces a bounded `EXIT_ERROR` and a `failed`
        ack, which the API already knows how to replan around.

        An explicitly declared `timeout` is always honoured as-is — a job that
        genuinely needs to exceed its window is the operator's call to make, and
        this must not quietly override it. Returns the original dict untouched in
        that case, so nothing is copied unless a value is actually being added.
        """
        budget = self._window_budget_s(job, placement, now)
        if budget is None or declared_timeout(job.request) is not None:
            return job.request
        return {**job.request, "timeout": budget}

    def _window_budget_s(
        self, job: JobDef, placement: Placement, now: datetime
    ) -> float | None:
        """Seconds this job may run before it overruns its window. None if unknowable."""
        limit = placement.end or job.deadline
        if limit is None:
            return None
        return max((limit - now).total_seconds(), MIN_WINDOW_TIMEOUT_S)

    def _build_record(
        self,
        job: JobDef,
        placement: Placement,
        run_id: str,
        fingerprint: str | None,
        result: AdapterRunResult,
        now: datetime,
        telemetry: RunTelemetry,
    ) -> dict[str, Any]:
        """Merge identity/provenance with the profiler's telemetry into a RunRecord dict.

        The API rollup derives p95/power_profile/throttled_s/cpu_energy_wh
        from the pushed trace, so we leave those out. Datetimes stay tz-aware; the
        ApiClient/Spool serialize them to ISO on the way out.
        """
        record: dict[str, Any] = {
            "controller_id": self.controller_id,
            "run_id": run_id,
            "workflow_id": job.workflow_id or placement.workflow_id,
            "fingerprint": fingerprint,
            "framework": job.framework,
            "exit_status": result.exit_status,
            "work_units": result.work_units,
            "work_unit_kind": result.work_unit_kind,
            "scheduled_start": placement.start,
            "actual_start": now,
            "ts": now,
        }
        record.update(telemetry.to_record_fields())
        return record

    # --- helpers ---------------------------------------------------------

    def _resolve_job(self, workflow_id: str | None) -> JobDef | None:
        job = self._job_source(workflow_id)
        if job is None or isinstance(job, JobDef):
            return job
        if isinstance(job, Mapping):
            return JobDef(
                framework=job.get("framework", ""),
                request=dict(job.get("request") or {}),
                workflow_id=job.get("workflow_id", workflow_id),
                deadline=_parse_dt(job.get("deadline")),
                earliest_start=_parse_dt(job.get("earliest_start")),
            )
        return job  # duck-typed object exposing.framework/.request/.deadline

    def _adapter_for(self, framework: str) -> Adapter:
        adapter = self._adapters.get(framework)
        if adapter is None:
            adapter = self._adapter_provider(framework)  # may raise AdapterError
            self._adapters[framework] = adapter
        return adapter

    def _preflight(self, adapter: Adapter, job: JobDef, wid: str | None) -> bool:
        try:
            return bool(adapter.preflight(job.request))
        except Exception as exc:  # a preflight that raises is a failed preflight
            logger.warning("preflight raised for workflow %s: %s", wid, exc)
            return False

    def _fingerprint(self, adapter: Adapter, job: JobDef, wid: str | None) -> str | None:
        try:
            fp, _features = adapter.fingerprint(job.request)
            return fp
        except Exception as exc:  # a malformed request → null fingerprint, still run
            logger.warning("fingerprint failed for workflow %s: %s", wid, exc)
            return None

    def _ack(self, version: int, event: str, wid: str | None, at: datetime) -> None:
        """Best-effort applied-state echo — a failed ack (API down) is swallowed."""
        try:
            self.client.ack(version, event, workflow_id=wid, at=at)
        except Exception as exc:  # pragma: no cover - client.ack already clean-errors
            logger.debug("ack %s for workflow %s failed: %s", event, wid, exc)

    def _now(self, now: datetime | None) -> datetime:
        return _ensure_aware(now if now is not None else self._now_fn())
