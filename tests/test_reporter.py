"""
RunReporter tests — the store-and-forward flow that is the point
of this story:

    push success  → nothing spooled
    API down      → report spooled
    reconnect     → drain flushes in FIFO order, spool empties
    idempotency   → a re-drain (or partial drain) is safe

Driven through the FakeApiServer (real httpx via MockTransport); its
`(controller_id, run_id)` idempotency and simulated outage make the assertions
meaningful rather than mock theatre.
"""

from __future__ import annotations

from hmasync_controller.reporter import RunReporter, _chunks
from hmasync_controller.spool import Spool


def _run(run_id: str, controller_id: str = "box-1") -> dict:
    return {"controller_id": controller_id, "run_id": run_id, "duration_s": 5.0}


def _reporter(make_client, spool, **client_kwargs) -> RunReporter:
    return RunReporter(make_client(**client_kwargs), spool)


# --- live push ------------------------------------------------------------

def test_report_live_success_nothing_spooled(make_client, spool, fake_api):
    reporter = _reporter(make_client, spool)
    result = reporter.report_run(_run("r-1"))
    assert result.ok
    assert result.spooled is False
    assert result.server_run_id is not None
    assert spool.count() == 0
    assert ("box-1", "r-1") in fake_api.runs


def test_report_live_pushes_samples(make_client, spool, fake_api):
    reporter = _reporter(make_client, spool)
    samples = [
        {"ts": "2026-07-11T03:00:00+00:00", "power_w": 100.0},
        {"ts": "2026-07-11T03:00:01+00:00", "power_w": 120.0},
    ]
    result = reporter.report_run(_run("r-1"), samples)
    assert result.ok and not result.spooled
    assert len(fake_api.samples[result.server_run_id]) == 2


# --- outage → spool -------------------------------------------------------

def test_report_when_api_down_spools(make_client, spool, fake_api):
    reporter = _reporter(make_client, spool)
    fake_api.go_down()
    result = reporter.report_run(_run("r-1"))
    assert not result.ok
    assert result.spooled is True
    assert spool.count() == 1
    # Nothing reached the (down) API.
    assert fake_api.runs == {}


# --- reconnect → drain in order ------------------------------------------

def test_drain_flushes_in_fifo_order(make_client, spool, fake_api):
    reporter = _reporter(make_client, spool)
    fake_api.go_down()
    for rid in ("r-1", "r-2", "r-3"):
        reporter.report_run(_run(rid))
    assert spool.count() == 3

    fake_api.go_up()
    drain = reporter.drain_spool()
    assert drain.drained == 3
    assert drain.remaining == 0
    assert spool.count() == 0
    # Pushed to the API oldest-first.
    assert fake_api.run_push_order == [("box-1", "r-1"), ("box-1", "r-2"), ("box-1", "r-3")]


def test_drain_pushes_spooled_samples(make_client, spool, fake_api):
    reporter = _reporter(make_client, spool)
    fake_api.go_down()
    samples = [{"ts": "2026-07-11T03:00:00+00:00", "power_w": 90.0}]
    reporter.report_run(_run("r-1"), samples)

    fake_api.go_up()
    reporter.drain_spool()
    sid = fake_api.runs[("box-1", "r-1")]
    assert len(fake_api.samples[sid]) == 1


def test_drain_stops_early_when_still_down(make_client, spool, fake_api):
    reporter = _reporter(make_client, spool)
    fake_api.go_down()
    reporter.report_run(_run("r-1"))
    reporter.report_run(_run("r-2"))

    # Still unreachable at drain time → nothing flushes, order preserved.
    drain = reporter.drain_spool()
    assert drain.drained == 0
    assert drain.stopped_early is True
    assert spool.count() == 2
    assert [item.run_id for item in spool.pending()] == ["r-1", "r-2"]


def test_redrain_after_partial_is_idempotent(make_client, spool, fake_api):
    # First report lands live; a second is spooled during an outage.
    reporter = _reporter(make_client, spool)
    reporter.report_run(_run("r-live"))
    fake_api.go_down()
    reporter.report_run(_run("r-spooled"))
    assert spool.count() == 1

    fake_api.go_up()
    first_drain = reporter.drain_spool()
    assert first_drain.drained == 1
    assert spool.count() == 0
    # A redundant second drain is a no-op (nothing queued), and the run store is
    # unchanged — idempotency means no duplicate rows were created.
    second_drain = reporter.drain_spool()
    assert second_drain.drained == 0
    assert len(fake_api.runs) == 2  # r-live + r-spooled, one each


def test_spool_survives_restart_then_drains(make_client, tmp_path, fake_api):
    path = str(tmp_path / "spool.db")

    # Process 1: API down, one report buffered, then process exits.
    spool1 = Spool(path)
    RunReporter(make_client(), spool1).report_run(_run("r-1"))  # live push ok
    fake_api.go_down()
    RunReporter(make_client(), spool1).report_run(_run("r-2"))  # spooled
    assert spool1.count() == 1
    spool1.close()

    # Process 2: fresh spool on the same file, API back up → the buffered report drains.
    fake_api.go_up()
    spool2 = Spool(path)
    drain = RunReporter(make_client(), spool2).drain_spool()
    assert drain.drained == 1
    assert spool2.count() == 0
    assert ("box-1", "r-2") in fake_api.runs
    spool2.close()


# --- helpers --------------------------------------------------------------

def test_chunks_splits_preserving_order():
    assert _chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert _chunks([], 2) == []
    assert _chunks([1, 2], 10) == [[1, 2]]
