"""
Spool tests: FIFO ordering, datetime-safe storage, and — the
load-bearing property — durability across a process restart (a new Spool on the
same file sees the queued reports).
"""

from __future__ import annotations

from datetime import datetime, timezone

from hmasync_controller.spool import Spool


def _run(run_id: str, controller_id: str = "box-1") -> dict:
    return {"controller_id": controller_id, "run_id": run_id, "duration_s": 5.0}


def test_enqueue_returns_id_and_counts(spool):
    assert spool.count() == 0
    item_id = spool.enqueue(_run("r-1"))
    assert item_id > 0
    assert spool.count() == 1


def test_pending_is_fifo(spool):
    spool.enqueue(_run("r-1"))
    spool.enqueue(_run("r-2"))
    spool.enqueue(_run("r-3"))
    order = [item.run_id for item in spool.pending()]
    assert order == ["r-1", "r-2", "r-3"]


def test_remove_drops_one_item(spool):
    a = spool.enqueue(_run("r-1"))
    spool.enqueue(_run("r-2"))
    spool.remove(a)
    remaining = [item.run_id for item in spool.pending()]
    assert remaining == ["r-2"]
    assert spool.count() == 1


def test_stores_record_and_samples(spool):
    samples = [{"ts": "2026-07-11T03:00:00+00:00", "power_w": 100.0}]
    spool.enqueue(_run("r-1"), samples)
    item = spool.pending()[0]
    assert item.record["run_id"] == "r-1"
    assert item.record["duration_s"] == 5.0
    assert item.samples == samples
    assert item.controller_id == "box-1"


def test_defaults_samples_to_empty_list(spool):
    spool.enqueue(_run("r-1"))
    assert spool.pending()[0].samples == []


def test_datetime_values_are_serialized(spool):
    ts = datetime(2026, 7, 11, 3, 0, tzinfo=timezone.utc)
    spool.enqueue({"controller_id": "box-1", "run_id": "r-1", "ts": ts})
    item = spool.pending()[0]
    # Round-trips as an ISO string, not a datetime that JSON would reject.
    assert item.record["ts"] == "2026-07-11T03:00:00+00:00"


def test_survives_restart(tmp_path):
    path = str(tmp_path / "spool.db")
    s1 = Spool(path)
    s1.enqueue(_run("r-1"))
    s1.enqueue(_run("r-2"))
    s1.close()

    # Fresh process → fresh Spool on the same file must still see both reports.
    s2 = Spool(path)
    order = [item.run_id for item in s2.pending()]
    assert order == ["r-1", "r-2"]
    assert s2.count() == 2
    s2.close()


def test_clear_empties_the_queue(spool):
    spool.enqueue(_run("r-1"))
    spool.enqueue(_run("r-2"))
    spool.clear()
    assert spool.count() == 0
    assert spool.pending() == []


def test_pending_limit(spool):
    for i in range(5):
        spool.enqueue(_run(f"r-{i}"))
    limited = spool.pending(limit=2)
    assert [item.run_id for item in limited] == ["r-0", "r-1"]
