"""
Spool — the controller's offline store-and-forward buffer.

The controller keeps only a small local spool for offline resilience: when the
optimizer API is unreachable, a run report (the run record plus its telemetry
trace) is enqueued here and forwarded on reconnect. A single SQLite file backs
it, so the buffer survives a process restart — an outage that outlives the
controller process must not lose run history.

**Why a run+trace bundle is the unit.** The samples endpoint is addressed by the
run's SERVER id, which only exists after the run is pushed. Storing the run and
its trace together as one FIFO item means the drain can push the run, learn its
server id, and push the trace within a single step — no cross-item server-id
bookkeeping, and because runs are idempotent server-side by
`(controller_id, run_id)`, re-pushing on a retry (or after a restart) is safe.

This module is pure storage: enqueue / read FIFO / remove / count. The replay
lives in reporter.py (it needs an ApiClient). Nothing here touches the network.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spool (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    controller_id TEXT,
    run_id        TEXT,
    record        TEXT NOT NULL,
    samples       TEXT NOT NULL DEFAULT '[]'
)
"""


@dataclass
class SpoolItem:
    """One queued run report: the run record plus its full telemetry trace."""

    id: int
    controller_id: str | None
    run_id: str | None
    record: dict[str, Any]
    samples: list[dict[str, Any]] = field(default_factory=list)
    created_at: str | None = None


def _json_default(value: Any) -> Any:
    """JSON encoder hook: serialize tz-aware datetimes to ISO-8601 strings."""
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class Spool:
    """A single-file, FIFO, restart-durable buffer of pending run reports."""

    def __init__(self, path: str):
        self.path = path
        # check_same_thread=False: the controller is single-threaded today, but a
        # background sampler/executor split may touch the spool
        # from a helper thread; the connection is only ever used serially.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # --- lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Spool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- write ------------------------------------------------------------

    def enqueue(
        self,
        record: dict[str, Any],
        samples: list[dict[str, Any]] | None = None,
    ) -> int:
        """Append a run report to the tail of the queue; return its spool id.

        `record` is a RunRecord-shaped dict; `samples` is its (possibly empty)
        list of TraceSample-shaped dicts. Both are stored as JSON with datetimes
        coerced to ISO strings, so the item round-trips unchanged after a restart.
        """
        controller_id = record.get("controller_id") if isinstance(record, dict) else None
        run_id = record.get("run_id") if isinstance(record, dict) else None
        created_at = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO spool (created_at, controller_id, run_id, record, samples)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                created_at,
                controller_id,
                run_id,
                json.dumps(record, default=_json_default),
                json.dumps(samples or [], default=_json_default),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def remove(self, item_id: int) -> None:
        """Delete a drained item by its spool id."""
        self._conn.execute("DELETE FROM spool WHERE id = ?", (item_id,))
        self._conn.commit()

    def clear(self) -> None:
        """Drop every queued item (test/maintenance helper)."""
        self._conn.execute("DELETE FROM spool")
        self._conn.commit()

    # --- read -------------------------------------------------------------

    def pending(self, limit: int | None = None) -> list[SpoolItem]:
        """Return queued items in FIFO order (oldest first), oldest = smallest id."""
        sql = "SELECT * FROM spool ORDER BY id ASC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_item(row) for row in rows]

    def count(self) -> int:
        """Number of items still queued."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM spool").fetchone()
        return int(row["n"])

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> SpoolItem:
        return SpoolItem(
            id=int(row["id"]),
            controller_id=row["controller_id"],
            run_id=row["run_id"],
            record=json.loads(row["record"]),
            samples=json.loads(row["samples"]),
            created_at=row["created_at"],
        )
