"""
Tests for the library face (hmasync_controller/sdk.py).

Everything is mocked: httpx via MockTransport, the profiler via a stub. No
network, no GPU, no account.

The behaviors worth pinning down are the ones a caller would be burned by:
an infeasible window must not silently sleep forever, a failed report must not
turn a successful job into a failure, and an exception inside the measured block
must still be raised after being recorded.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from hmasync_controller.apiclient import ApiClient
from hmasync_controller.profiler import RunTelemetry
from hmasync_controller.sdk import AsyncEnergy, AsyncEnergyError, Window

FUTURE = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
FUTURE_END = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()

ADVICE = {
    "start": FUTURE,
    "end": FUTURE_END,
    "grid_cost": 0.18,
    "now_cost": 0.83,
    "savings": 0.65,
    "predicted_wh": 2100.0,
    "feasible": True,
    "reason": None,
    "prediction_method": "prior",
    "degraded": False,
}


class StubProfiler:
    """Records start/stop and returns fixed telemetry."""

    def __init__(self, telemetry: RunTelemetry | None = None, *, raise_on_stop: bool = False):
        self.telemetry = telemetry
        self.started: list[str] = []
        self.stopped: list[str] = []
        self._raise_on_stop = raise_on_stop

    def start(self, run_id):
        self.started.append(run_id)

    def stop(self, run_id=None):
        self.stopped.append(run_id)
        if self._raise_on_stop:
            raise RuntimeError("nvml exploded")
        return self.telemetry or RunTelemetry(run_id=run_id or "r", duration_s=1.0)

    def capabilities(self):
        return set()


def _ae(handler, *, profiler=None) -> AsyncEnergy:
    """An AsyncEnergy wired to a MockTransport server."""
    client = ApiClient(
        base_url="https://api.test",
        api_key="k_test",
        controller_id="box-1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return AsyncEnergy(client=client, profiler=profiler or StubProfiler())


# --- credentials ----------------------------------------------------------

def test_requires_some_credential(monkeypatch):
    for var in ("HM_ASYNC_API_KEY", "HM_ASYNC_EMAIL", "HM_ASYNC_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AsyncEnergyError, match="No credentials"):
        AsyncEnergy()


def test_api_key_is_sent_as_a_header_not_a_bearer_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["api_key"] = request.headers.get("X-API-Key")
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=ADVICE)

    ae = _ae(handler)
    ae.next_window(est_duration_s=1800, deadline="by 7am")
    assert seen["api_key"] == "k_test"
    assert seen["auth"] is None, "an API key must not be sent as a bearer token"


def test_api_key_401_is_not_retried_as_a_login():
    """A key is either valid or revoked — a login round-trip cannot fix it and
    would just hide the real error."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(401, json={"detail": "Invalid API key"})

    ae = _ae(handler)
    with pytest.raises(AsyncEnergyError):
        ae.next_window(est_duration_s=60, deadline="by 7am")
    assert calls == ["/api/v1/advise"], f"expected no login attempt, got {calls}"


# --- next_window ----------------------------------------------------------

def test_next_window_parses_the_answer():
    ae = _ae(lambda r: httpx.Response(200, json=ADVICE))
    w = ae.next_window(est_duration_s=1800, deadline="by 7am", nameplate_watts=400)
    assert w.feasible is True
    assert w.savings == 0.65
    assert w.predicted_wh == 2100.0
    assert w.start is not None and w.start.tzinfo is not None


def test_next_window_sends_only_the_fields_given():
    """Optional fields are omitted rather than sent as null, so server-side
    defaults stay in charge."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=ADVICE)

    ae = _ae(handler)
    ae.next_window(est_duration_s=1800, deadline="by 7am")
    assert "earliest_start" not in seen
    assert "nameplate_watts" not in seen
    assert seen["est_duration_s"] == 1800


def test_infeasible_window_is_returned_not_raised():
    body = {**ADVICE, "feasible": False, "reason": "cannot fit before deadline", "start": None}
    ae = _ae(lambda r: httpx.Response(200, json=body))
    w = ae.next_window(est_duration_s=99999, deadline="by 7am")
    assert w.feasible is False
    assert w.reason == "cannot fit before deadline"


def test_waiting_on_an_infeasible_window_raises_rather_than_sleeping():
    """The dangerous failure mode: a caller blindly calling.wait() on a window
    that will never open."""
    w = Window(start=None, end=None, feasible=False, reason="cannot fit")
    with pytest.raises(AsyncEnergyError, match="infeasible"):
        w.wait(sleep=lambda s: None)


def test_wait_sleeps_until_the_window_and_is_capped():
    w = Window(start=datetime.now(timezone.utc) + timedelta(hours=2), end=None)
    slept: list[float] = []
    assert w.wait(sleep=slept.append, max_wait_s=5) == 5
    assert slept == [5]


def test_wait_returns_immediately_for_an_open_window():
    w = Window(start=datetime.now(timezone.utc) - timedelta(minutes=1), end=None)
    slept: list[float] = []
    assert w.wait(sleep=slept.append) == 0.0
    assert slept == []


def test_api_error_becomes_a_clean_exception():
    ae = _ae(lambda r: httpx.Response(503, json={"detail": "Database unavailable"}))
    with pytest.raises(AsyncEnergyError):
        ae.next_window(est_duration_s=60, deadline="by 7am")


# --- measure --------------------------------------------------------------

def _run_capture():
    """A handler that records pushed runs and returns a server run id."""
    pushed: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runs":
            pushed.append(json.loads(request.content))
            return httpx.Response(200, json={"runs": [{"id": "server-uuid", "status": "inserted"}]})
        if request.url.path.endswith("/samples"):
            pushed.append({"_samples": json.loads(request.content)})
            return httpx.Response(200, json={"inserted": 1})
        return httpx.Response(404)

    return handler, pushed


def test_measure_reports_a_successful_run():
    handler, pushed = _run_capture()
    prof = StubProfiler(RunTelemetry(run_id="r", duration_s=2.0, energy_wh=12.5,
                                     energy_source="counter", avg_w=250.0))
    ae = _ae(handler, profiler=prof)

    with ae.measure(fingerprint="fp-1") as run:
        run["work_units"] = 4096
        run["work_unit_kind"] = "tokens"

    record = pushed[0]
    assert record["exit_status"] == "success"
    assert record["energy_wh"] == 12.5
    assert record["energy_source"] == "counter"
    assert record["work_units"] == 4096
    assert record["fingerprint"] == "fp-1"
    assert prof.started and prof.stopped, "the profiler must wrap the block"


def test_measure_records_a_failure_and_still_raises():
    """Error handling must not be swallowed to make the report tidy."""
    handler, pushed = _run_capture()
    ae = _ae(handler, profiler=StubProfiler())

    with pytest.raises(ZeroDivisionError):
        with ae.measure():
            1 / 0

    assert pushed[0]["exit_status"] == "failed"


def test_measure_pushes_the_trace_when_there_is_one():
    handler, pushed = _run_capture()
    telemetry = RunTelemetry(
        run_id="r", duration_s=2.0,
        samples=[{"ts": datetime.now(timezone.utc), "power_w": 200.0}],
    )
    ae = _ae(handler, profiler=StubProfiler(telemetry))
    with ae.measure():
        pass
    assert any("_samples" in p for p in pushed), "the 1 Hz trace should be pushed"


def test_a_failed_report_does_not_fail_the_job():
    """Losing a measurement is not a reason to fail work that succeeded."""
    ae = _ae(lambda r: httpx.Response(500, json={"detail": "boom"}),
             profiler=StubProfiler())
    with ae.measure():   # must not raise
        pass


def test_a_broken_profiler_does_not_fail_the_job():
    handler, pushed = _run_capture()
    ae = _ae(handler, profiler=StubProfiler(raise_on_stop=True))
    with ae.measure():   # must not raise
        pass
    # The run is still reported, just without telemetry — never a fabricated one.
    assert pushed[0]["exit_status"] == "success"
    assert "energy_wh" not in pushed[0] or pushed[0]["energy_wh"] is None


def test_null_telemetry_reports_null_energy_not_zero():
    handler, pushed = _run_capture()
    ae = _ae(handler, profiler=StubProfiler(RunTelemetry(run_id="r", duration_s=1.0)))
    with ae.measure():
        pass
    assert pushed[0]["energy_wh"] is None, "a box that cannot measure must report null, not 0"


def test_report_can_be_disabled():
    handler, pushed = _run_capture()
    ae = _ae(handler, profiler=StubProfiler())
    with ae.measure(report=False):
        pass
    assert pushed == []
