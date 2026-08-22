"""
Tests for the schedule executor loop.

Wiring: the real ApiClient/RunReporter/Spool drive against the conftest
`FakeApiServer` (httpx MockTransport — no live vendor), a `StubProfiler` supplies
canned telemetry, and the framework layer is either the real `CommandAdapter`
(running `true`/`false`) or a `StubAdapter` for the preflight/framework cases. The
clock is injected as a fixed tz-aware `now`, so every timing assertion is
deterministic.

Covers the six acceptance scenarios: normal window execution with acks, version
adoption mid-loop, API-down-within-validity continuation, expired-schedule fallback
firing at the latest feasible start, the failure-ack path, and preflight-failure
acking without executing — plus edges (unknown job/framework, infeasible skip,
serialization order, spool drain on reconnect, no naive datetimes).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hmasync_controller.adapters import (
    Adapter,
    AdapterRunResult,
    EXIT_ERROR,
    EXIT_SUCCESS,
)
from hmasync_controller.apiclient import ApiClient
from hmasync_controller.executor import JobDef, ScheduleExecutor
from hmasync_controller.profiler import Profiler, RunTelemetry
from hmasync_controller.reporter import RunReporter

# --- shared datetimes (all tz-aware) ------------------------------
UTC = timezone.utc
DAY = "2026-07-11"
FAR_FUTURE = "2026-07-12T00:00:00+00:00"


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 11, hour, minute, tzinfo=UTC)


def iso(hour: int, minute: int = 0) -> str:
    return at(hour, minute).isoformat()


# --- doubles ----------------------------------------------------------------
def _sample(ts: datetime) -> dict:
    return {
        "ts": ts,
        "power_w": 100.0,
        "util_gpu": 50.0,
        "util_mem": 10.0,
        "mem_used_mb": 2048.0,
        "temp_c": 60.0,
        "sm_clock_mhz": 1500.0,
        "throttle_reasons": "none",
        "cpu_rapl_uj": 1_000_000.0,
    }


class StubProfiler(Profiler):
    """Canned telemetry — records start/stop so we can assert profiling happened."""

    def __init__(self, *, samples=None, duration_s: float = 2.0):
        self._samples = samples if samples is not None else [_sample(at(2, 0))]
        self.duration_s = duration_s
        self.start_calls: list[str] = []
        self.stop_calls: list[str | None] = []

    def start(self, run_id: str) -> None:
        self.start_calls.append(run_id)

    def stop(self, run_id: str | None = None) -> RunTelemetry:
        self.stop_calls.append(run_id)
        return RunTelemetry(
            run_id=run_id or "",
            duration_s=self.duration_s,
            samples=list(self._samples),
            capabilities=frozenset(),
            energy_wh=12.5,
            energy_source="counter",
            avg_w=100.0,
            peak_w=150.0,
            gpu_mem_mb=4096.0,
        )

    def capabilities(self) -> set[str]:
        return set()


class StubAdapter(Adapter):
    """Configurable adapter for preflight / unknown-framework paths."""

    FRAMEWORK = "stub"

    def __init__(self, *, preflight_result=True, run_result=None, fp="fp-stub"):
        self.preflight_result = preflight_result
        self.run_result = run_result or AdapterRunResult(EXIT_SUCCESS)
        self.fp = fp
        self.run_calls = 0
        self.preflight_calls = 0

    def run(self, request):
        self.run_calls += 1
        return self.run_result

    def fingerprint(self, request):
        return self.fp, {"framework": self.FRAMEWORK}

    def preflight(self, request=None):
        self.preflight_calls += 1
        return self.preflight_result


def counter_ids(prefix="run"):
    box = {"i": 0}

    def _next() -> str:
        box["i"] += 1
        return f"{prefix}-{box['i']}"

    return _next


# --- builders ---------------------------------------------------------------
def make_executor(
    fake_api,
    spool,
    catalog,
    *,
    profiler=None,
    adapter_provider=None,
    controller_id="box-1",
    power_cap=None,
):
    client = ApiClient(
        base_url="https://api.hm-async.test",
        email="owner@example.com",
        password="s3cret",
        controller_id=controller_id,
        http_client=fake_api.client(),
    )
    reporter = RunReporter(client, spool)
    return ScheduleExecutor(
        client=client,
        reporter=reporter,
        job_source=catalog,
        profiler=profiler or StubProfiler(),
        controller_id=controller_id,
        adapter_provider=adapter_provider,
        run_id_factory=counter_ids(),
        power_cap=power_cap,
    )


def placement(wid, start, end=None, *, feasible=True, **extra):
    p = {
        "workflow_id": wid,
        "start": start,
        "end": end,
        "predicted_wh": 100.0,
        "grid_cost": 0.1,
        "now_cost": 0.2,
        "feasible": feasible,
        "reason": None,
    }
    p.update(extra)
    return p


def command_job(wid, cmd, **extra):
    return JobDef(framework="command", request={"command": cmd}, workflow_id=wid, **extra)


# --- request inspectors -----------------------------------------------------
def ack_events(fake_api):
    out = []
    for req in fake_api.requests:
        if req.url.path == "/api/v1/schedule/ack" and req.method == "POST":
            out.append(json.loads(req.content.decode()).get("event"))
    return out


def run_bodies(fake_api):
    out = []
    for req in fake_api.requests:
        if req.url.path == "/api/v1/runs" and req.method == "POST":
            out.append(json.loads(req.content.decode()))
    return out


# ============================================================
# 1. Normal window execution with acks
# ============================================================
def test_normal_window_execution_with_acks(fake_api, spool):
    fake_api.set_schedule(
        1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE
    )
    profiler = StubProfiler()
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])}, profiler=profiler)

    result = ex.tick(now=at(2, 30))

    assert result.mode == "normal"
    assert result.version == 1
    assert len(result.outcomes) == 1
    out = result.outcomes[0]
    assert out.status == "executed"
    assert out.exit_status == "success"
    assert out.spooled is False
    # started + finished acks, in that order.
    assert ack_events(fake_api) == ["started", "finished"]
    # The run reached the API with the profiler's telemetry merged into the record.
    bodies = run_bodies(fake_api)
    assert len(bodies) == 1
    rec = bodies[0]
    assert rec["workflow_id"] == "wf-1"
    assert rec["controller_id"] == "box-1"
    assert rec["exit_status"] == "success"
    assert rec["energy_wh"] == 12.5
    assert rec["energy_source"] == "counter"
    assert rec["duration_s"] == 2.0
    # Trace pushed to the run's server id.
    assert sum(len(v) for v in fake_api.samples.values()) == 1
    assert profiler.start_calls and profiler.stop_calls


def test_placement_not_due_before_start(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0))], valid_until=FAR_FUTURE)
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})

    result = ex.tick(now=at(1, 0))  # before the 02:00 window

    assert result.mode == "normal"
    assert result.outcomes == []
    assert run_bodies(fake_api) == []


def test_no_schedule_is_idle(fake_api, spool):
    ex = make_executor(fake_api, spool, {})  # fake_api.schedule is None → 204
    result = ex.tick(now=at(2, 30))
    assert result.mode == "idle"
    assert result.version == -1
    assert result.outcomes == []


# ============================================================
# 2. Version adoption mid-loop
# ============================================================
def test_version_adoption_mid_loop(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    catalog = {"wf-1": command_job("wf-1", ["true"]), "wf-2": command_job("wf-2", ["true"])}
    ex = make_executor(fake_api, spool, catalog)

    first = ex.tick(now=at(2, 30))
    assert [o.workflow_id for o in first.outcomes] == ["wf-1"]
    fake_api.requests.clear()

    # A replan publishes v2: wf-1 unchanged (same start) + a new wf-2.
    fake_api.set_schedule(
        2,
        placements=[
            placement("wf-1", iso(2, 0), iso(3, 0)),
            placement("wf-2", iso(2, 0), iso(3, 0)),
        ],
        valid_until=FAR_FUTURE,
    )

    second = ex.tick(now=at(2, 30))
    assert second.adopted is True
    assert second.version == 2
    # wf-1 already ran under v1 (same identity) → skipped; only wf-2 runs.
    assert [o.workflow_id for o in second.outcomes] == ["wf-2"]
    assert [b["workflow_id"] for b in run_bodies(fake_api)] == ["wf-2"]


# ============================================================
# 3. API-down-within-validity continues (spool instead of reaching the API)
# ============================================================
def test_api_down_within_validity_continues(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})

    # Seed v1 while the API is up (before the window opens → nothing runs yet).
    ex.tick(now=at(1, 0))
    assert ex.version == 1
    fake_api.go_down()

    result = ex.tick(now=at(2, 30))  # still within validity, API unreachable

    assert result.reachable is False
    assert result.mode == "normal"
    assert len(result.outcomes) == 1
    out = result.outcomes[0]
    assert out.status == "executed"  # the job still ran locally
    assert out.spooled is True  #...but its report was buffered, not delivered
    assert spool.count() == 1
    assert fake_api.runs == {}  # nothing reached the API store


# ============================================================
# 4. Expired-schedule fallback fires at latest feasible start
# ============================================================
def test_expired_schedule_fallback_at_latest_feasible_start(fake_api, spool):
    # Planned cheap window 02:00–03:00 (1 h), deadline 07:00 → latest start 06:00.
    fake_api.set_schedule(
        1,
        placements=[placement("wf-1", iso(2, 0), iso(3, 0))],
        valid_until=iso(4, 0),
    )
    catalog = {"wf-1": command_job("wf-1", ["true"], deadline=at(7, 0))}
    ex = make_executor(fake_api, spool, catalog)

    # Seed v1 while valid.
    ex.tick(now=at(1, 0))
    fake_api.requests.clear()

    # Past valid_until, no newer version (204): fallback mode, but before the latest
    # feasible start (06:00) the job must NOT run — it is neither its planned start
    # nor the deadline-safe latest start.
    early = ex.tick(now=at(5, 0))
    assert early.mode == "fallback"
    assert early.outcomes == []
    assert run_bodies(fake_api) == []

    # At/after 06:00 the fallback fires.
    fired = ex.tick(now=at(6, 30))
    assert fired.mode == "fallback"
    assert [o.workflow_id for o in fired.outcomes] == ["wf-1"]
    assert ack_events(fake_api) == ["started", "finished"]


def test_fallback_without_known_deadline_uses_planned_window(fake_api, spool):
    # No deadline in the local job → fallback falls back to the placement's planned
    # end as the deadline proxy, so latest start == planned start (deadline-safe).
    fake_api.set_schedule(
        1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=iso(4, 0)
    )
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})
    ex.tick(now=at(1, 0))

    result = ex.tick(now=at(5, 0))  # expired; planned start (02:00) already passed
    assert result.mode == "fallback"
    assert [o.workflow_id for o in result.outcomes] == ["wf-1"]


# ============================================================
# 5. Failure ack path
# ============================================================
def test_failure_ack_path(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["false"])})  # exits 1

    result = ex.tick(now=at(2, 30))

    out = result.outcomes[0]
    assert out.status == "executed"
    assert out.exit_status == "error"
    # started then failed (not finished) — the API replans on the failed event.
    assert ack_events(fake_api) == ["started", "failed"]
    # A failed run is still recorded upstream (it's training data / audit).
    assert run_bodies(fake_api)[0]["exit_status"] == "error"


# ============================================================
# 6. Preflight failure acks without executing
# ============================================================
def test_preflight_failure_acks_without_executing(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    stub = StubAdapter(preflight_result=False)
    profiler = StubProfiler()
    ex = make_executor(
        fake_api,
        spool,
        {"wf-1": JobDef(framework="stub", request={"model": "x"}, workflow_id="wf-1")},
        profiler=profiler,
        adapter_provider=lambda fw: stub,
    )

    result = ex.tick(now=at(2, 30))

    out = result.outcomes[0]
    assert out.status == "preflight_failed"
    assert stub.preflight_calls == 1
    assert stub.run_calls == 0  # never executed
    assert profiler.start_calls == []  # never profiled
    assert ack_events(fake_api) == ["failed"]  # only a failed ack, no started/finished
    assert run_bodies(fake_api) == []  # nothing reported upstream


# ============================================================
# Edges
# ============================================================
def test_unknown_job_skipped_without_ack(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("ghost", iso(2, 0))], valid_until=FAR_FUTURE)
    ex = make_executor(fake_api, spool, {})  # no catalog entry for "ghost"

    result = ex.tick(now=at(2, 30))

    assert result.outcomes[0].status == "unknown_job"
    assert ack_events(fake_api) == []  # a local-config gap, no replan spam
    assert run_bodies(fake_api) == []


def test_unknown_framework(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0))], valid_until=FAR_FUTURE)
    ex = make_executor(
        fake_api,
        spool,
        {"wf-1": JobDef(framework="does-not-exist", request={}, workflow_id="wf-1")},
    )

    result = ex.tick(now=at(2, 30))

    assert result.outcomes[0].status == "unknown_framework"
    assert run_bodies(fake_api) == []


def test_infeasible_placement_never_run(fake_api, spool):
    fake_api.set_schedule(
        1,
        placements=[placement("wf-1", None, None, feasible=False, reason="cannot meet deadline")],
        valid_until=FAR_FUTURE,
    )
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})

    result = ex.tick(now=at(2, 30))

    assert result.outcomes == []
    assert run_bodies(fake_api) == []


def test_serialization_two_due_placements_run_in_start_order(fake_api, spool):
    fake_api.set_schedule(
        1,
        placements=[
            placement("wf-late", iso(2, 30), iso(3, 0)),
            placement("wf-early", iso(2, 0), iso(2, 30)),
        ],
        valid_until=FAR_FUTURE,
    )
    catalog = {
        "wf-early": command_job("wf-early", ["true"]),
        "wf-late": command_job("wf-late", ["true"]),
    }
    ex = make_executor(fake_api, spool, catalog)

    result = ex.tick(now=at(3, 0))  # both due

    assert [o.workflow_id for o in result.outcomes] == ["wf-early", "wf-late"]
    assert [b["workflow_id"] for b in run_bodies(fake_api)] == ["wf-early", "wf-late"]


def test_already_executed_not_rerun(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})

    first = ex.tick(now=at(2, 30))
    assert len(first.outcomes) == 1
    second = ex.tick(now=at(2, 45))  # same schedule, same placement
    assert second.outcomes == []
    assert len(run_bodies(fake_api)) == 1  # exactly one run, not two


def test_drain_spool_on_reconnect(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})

    ex.tick(now=at(1, 0))  # seed v1 while up
    fake_api.go_down()
    ex.tick(now=at(2, 30))  # runs + spools during the outage
    assert spool.count() == 1

    fake_api.go_up()
    result = ex.tick(now=at(2, 45))  # reconnect → drain flushes the buffered report

    assert result.reachable is True
    assert result.drained == 1
    assert spool.count() == 0
    assert len(fake_api.runs) == 1  # the spooled run finally reached the API


def test_extra_drain_runs_on_reconnect_and_adds_to_drained_count(fake_api, spool):
    fake_api.set_schedule(1, placements=[], valid_until=FAR_FUTURE)
    client = ApiClient(
        base_url="https://api.hm-async.test", email="owner@example.com",
        password="s3cret", controller_id="box-1", http_client=fake_api.client(),
    )
    reporter = RunReporter(client, spool)
    calls = []
    ex = ScheduleExecutor(
        client=client, reporter=reporter, job_source={},
        profiler=StubProfiler(), controller_id="box-1",
        extra_drain=lambda: (calls.append(1), 3)[1],
    )

    result = ex.tick(now=at(1, 0))

    assert calls == [1]
    assert result.drained == 3
    ex.client.close()


def test_extra_drain_is_not_called_when_unreachable(fake_api, spool):
    fake_api.go_down()
    client = ApiClient(
        base_url="https://api.hm-async.test", email="owner@example.com",
        password="s3cret", controller_id="box-1", http_client=fake_api.client(),
    )
    reporter = RunReporter(client, spool)
    calls = []
    ex = ScheduleExecutor(
        client=client, reporter=reporter, job_source={},
        profiler=StubProfiler(), controller_id="box-1",
        extra_drain=lambda: (calls.append(1), 1)[1],
    )

    ex.tick(now=at(1, 0))

    assert calls == []
    ex.client.close()


def test_extra_drain_raising_never_breaks_a_tick(fake_api, spool):
    fake_api.set_schedule(1, placements=[], valid_until=FAR_FUTURE)
    client = ApiClient(
        base_url="https://api.hm-async.test", email="owner@example.com",
        password="s3cret", controller_id="box-1", http_client=fake_api.client(),
    )
    reporter = RunReporter(client, spool)

    def _boom():
        raise RuntimeError("bench spool db is locked")

    ex = ScheduleExecutor(
        client=client, reporter=reporter, job_source={},
        profiler=StubProfiler(), controller_id="box-1",
        extra_drain=_boom,
    )

    result = ex.tick(now=at(1, 0))  # must not raise

    assert result.reachable is True
    assert result.drained == 0
    ex.client.close()


# ============================================================
# Power cap bracketing (US-ONB-06): apply before, restore after, always
# ============================================================
class FakePowerCap:
    """Records apply()/restore() order without touching real NVML."""

    def __init__(self):
        self.calls: list[str] = []

    def apply(self):
        self.calls.append("apply")
        return "applied"

    def restore(self):
        self.calls.append("restore")
        return "restored"


def test_power_cap_apply_and_restore_bracket_the_job(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    power_cap = FakePowerCap()
    ex = make_executor(
        fake_api, spool, {"wf-1": command_job("wf-1", ["true"])}, power_cap=power_cap
    )

    ex.tick(now=at(2, 30))

    assert power_cap.calls == ["apply", "restore"]


def test_power_cap_not_touched_when_not_wired(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})  # power_cap=None

    result = ex.tick(now=at(2, 30))  # must not raise (no power_cap.apply/restore to call)

    assert result.outcomes[0].status == "executed"


def test_power_cap_restore_runs_even_when_the_adapter_raises(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    power_cap = FakePowerCap()

    class BoomAdapter(StubAdapter):
        def run(self, request):
            raise RuntimeError("gpu driver crashed")

    ex = make_executor(
        fake_api, spool,
        {"wf-1": JobDef(framework="stub", request={}, workflow_id="wf-1")},
        adapter_provider=lambda fw: BoomAdapter(),
        power_cap=power_cap,
    )

    result = ex.tick(now=at(2, 30))

    assert power_cap.calls == ["apply", "restore"]
    # The defensive catch in _execute turned the crash into a failed run, not
    # an unhandled exception — power_cap.restore() still ran either way.
    assert result.outcomes[0].exit_status == EXIT_ERROR


def test_power_cap_apply_and_restore_raising_never_breaks_a_tick(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)

    class BoomPowerCap:
        def apply(self):
            raise RuntimeError("power_cap.apply blew up")

        def restore(self):
            raise RuntimeError("power_cap.restore blew up")

    ex = make_executor(
        fake_api, spool, {"wf-1": command_job("wf-1", ["true"])}, power_cap=BoomPowerCap()
    )

    result = ex.tick(now=at(2, 30))  # must not raise

    assert result.outcomes[0].status == "executed"
    assert result.outcomes[0].exit_status == EXIT_SUCCESS


def test_no_naive_datetimes_in_pushed_record(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})

    ex.tick(now=at(2, 30))

    rec = run_bodies(fake_api)[0]
    for field_name in ("scheduled_start", "actual_start", "ts"):
        value = rec[field_name]
        assert isinstance(value, str)
        assert value.endswith("+00:00") or value.endswith("Z")  # offset-carrying, tz-aware
        # And it round-trips back to an aware datetime.
        assert datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None


def test_tick_uses_injected_clock_when_now_omitted(fake_api, spool):
    fake_api.set_schedule(1, placements=[placement("wf-1", iso(2, 0), iso(3, 0))], valid_until=FAR_FUTURE)
    clock = {"t": at(1, 0)}  # before the window
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})
    ex._now_fn = lambda: clock["t"]  # inject a controllable clock

    assert ex.tick().outcomes == []  # 01:00 → not due
    clock["t"] = at(2, 30)
    assert [o.workflow_id for o in ex.tick().outcomes] == ["wf-1"]  # 02:30 → due


# ============================================================
# Window-bounded runs (overrun containment v1)
# ============================================================
#
# A job declaring no `timeout` used to run unbounded. That is the one overrun this
# product cannot tolerate: it does not merely delay its successors, it runs out of
# the cheap window it was placed in and into peak pricing — the exact outcome the
# schedule existed to avoid — while the run record still reports a correctly-planned
# placement and the user is billed for the difference.


class RecordingAdapter(StubAdapter):
    """Captures the request dict `run()` was actually handed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests: list[dict] = []

    def run(self, request):
        self.requests.append(dict(request))
        return super().run(request)


def _run_one(fake_api, spool, job, place, now):
    adapter = RecordingAdapter()
    ex = make_executor(
        fake_api, spool, {"wf-1": job}, adapter_provider=lambda fw: adapter
    )
    fake_api.set_schedule(1, placements=[place], valid_until=FAR_FUTURE)
    ex.tick(now=now)
    return adapter


def test_undeclared_timeout_is_bounded_by_the_remaining_window(fake_api, spool):
    """02:00–04:00, ticking at 02:00 → two hours of budget, not None."""
    job = JobDef(framework="stub", request={"command": ["slow"]}, workflow_id="wf-1")
    adapter = _run_one(
        fake_api, spool, job, placement("wf-1", iso(2), iso(4)), at(2, 0)
    )
    assert adapter.requests[0]["timeout"] == 7200.0


def test_a_late_start_shrinks_the_budget_to_what_is_left(fake_api, spool):
    """Ticking at 03:00 in a 02:00–04:00 window leaves one hour, not two."""
    job = JobDef(framework="stub", request={"command": ["slow"]}, workflow_id="wf-1")
    adapter = _run_one(
        fake_api, spool, job, placement("wf-1", iso(2), iso(4)), at(3, 0)
    )
    assert adapter.requests[0]["timeout"] == 3600.0


def test_a_declared_timeout_is_never_overridden(fake_api, spool):
    """Exceeding the window on purpose is the operator's call to make."""
    job = JobDef(
        framework="stub",
        request={"command": ["slow"], "timeout": 30},
        workflow_id="wf-1",
    )
    adapter = _run_one(
        fake_api, spool, job, placement("wf-1", iso(2), iso(4)), at(2, 0)
    )
    assert adapter.requests[0]["timeout"] == 30


def test_a_nearly_expired_window_still_gets_the_floor(fake_api, spool):
    """A job starting seconds before its window closes must not insta-fail."""
    job = JobDef(framework="stub", request={"command": ["slow"]}, workflow_id="wf-1")
    adapter = _run_one(
        fake_api, spool, job, placement("wf-1", iso(2), iso(4)), at(3, 59)
    )
    assert adapter.requests[0]["timeout"] == 60.0  # MIN_WINDOW_TIMEOUT_S


def test_a_placement_with_no_end_falls_back_to_the_local_deadline(fake_api, spool):
    job = JobDef(
        framework="stub",
        request={"command": ["slow"]},
        workflow_id="wf-1",
        deadline=at(7, 0),
    )
    adapter = _run_one(
        fake_api, spool, job, placement("wf-1", iso(2), None), at(2, 0)
    )
    assert adapter.requests[0]["timeout"] == 18000.0  # 02:00 → 07:00


def test_no_window_and_no_deadline_leaves_the_request_untouched(fake_api, spool):
    """Nothing to derive a bound from — do not invent one."""
    job = JobDef(framework="stub", request={"command": ["slow"]}, workflow_id="wf-1")
    adapter = _run_one(
        fake_api, spool, job, placement("wf-1", iso(2), None), at(2, 0)
    )
    assert "timeout" not in adapter.requests[0]


def test_bounding_does_not_mutate_the_catalog_entry(fake_api, spool):
    """The catalog is the operator's file; a per-run bound must not leak into it."""
    request = {"command": ["slow"]}
    job = JobDef(framework="stub", request=request, workflow_id="wf-1")
    _run_one(fake_api, spool, job, placement("wf-1", iso(2), iso(4)), at(2, 0))
    assert request == {"command": ["slow"]}


def test_an_overrunning_command_becomes_a_failed_ack(fake_api, spool, monkeypatch):
    """End to end: the bound fires, and the API is told so it can replan."""
    from hmasync_controller import adapters

    def _raise(*a, **k):
        assert k.get("timeout") == 7200.0, "the window budget must reach subprocess"
        raise adapters.subprocess.TimeoutExpired(cmd="slow", timeout=k.get("timeout"))

    monkeypatch.setattr(adapters.subprocess, "run", _raise)
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["slow"])})
    fake_api.set_schedule(
        1, placements=[placement("wf-1", iso(2), iso(4))], valid_until=FAR_FUTURE
    )

    result = ex.tick(now=at(2, 0))

    assert result.outcomes[0].exit_status == EXIT_ERROR
    assert result.outcomes[0].reason == "command timed out"
    assert ack_events(fake_api) == ["started", "failed"]


# ============================================================
# Tick counters (telling "idle" apart from "misconfigured")
# ============================================================


def test_tick_reports_placements_and_pending(fake_api, spool):
    """`outcomes=0` alone cannot distinguish these cases; the counters can."""
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})
    fake_api.set_schedule(
        1,
        placements=[placement("wf-1", iso(20)), placement("wf-2", iso(22))],
        valid_until=FAR_FUTURE,
    )

    result = ex.tick(now=at(16, 0))  # both windows still ahead

    assert result.placements == 2
    assert result.pending == 2
    assert result.next_start == at(20, 0)
    assert len(result.outcomes) == 0


def test_pending_counts_down_as_jobs_run(fake_api, spool):
    ex = make_executor(
        fake_api, spool,
        {"wf-1": command_job("wf-1", ["true"]), "wf-2": command_job("wf-2", ["true"])},
    )
    fake_api.set_schedule(
        1,
        placements=[placement("wf-1", iso(2)), placement("wf-2", iso(20))],
        valid_until=FAR_FUTURE,
    )

    result = ex.tick(now=at(2, 0))

    assert result.placements == 2
    assert len(result.outcomes) == 1
    assert result.pending == 1, "wf-2 is still ahead"
    assert result.next_start == at(20, 0)


def test_an_empty_schedule_reports_zero_placements(fake_api, spool):
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})
    fake_api.set_schedule(1, placements=[], valid_until=FAR_FUTURE)

    result = ex.tick(now=at(2, 0))

    assert result.placements == 0 and result.pending == 0
    assert result.next_start is None


def test_infeasible_placements_are_not_counted_as_pending(fake_api, spool):
    """They have no window; the controller will never run them."""
    ex = make_executor(fake_api, spool, {"wf-1": command_job("wf-1", ["true"])})
    fake_api.set_schedule(
        1,
        placements=[placement("wf-1", iso(20), feasible=False)],
        valid_until=FAR_FUTURE,
    )

    result = ex.tick(now=at(16, 0))

    assert result.placements == 1 and result.pending == 0


# --- run_forever's on_tick hook --------------------------------------------


def test_run_forever_reports_every_tick(fake_api, spool):
    ex = make_executor(fake_api, spool, {})
    fake_api.set_schedule(1, placements=[], valid_until=FAR_FUTURE)
    seen = []

    ex.run_forever(max_ticks=3, sleep=lambda _s: None, on_tick=seen.append)

    assert len(seen) == 3
    assert all(t.version == 1 for t in seen)


def test_a_raising_on_tick_never_stops_the_loop(fake_api, spool):
    """Observability is not worth a job."""
    ex = make_executor(fake_api, spool, {})
    fake_api.set_schedule(1, placements=[], valid_until=FAR_FUTURE)

    def _boom(_result):
        raise RuntimeError("logging blew up")

    assert ex.run_forever(max_ticks=2, sleep=lambda _s: None, on_tick=_boom) == 2
