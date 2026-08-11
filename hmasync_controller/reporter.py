"""
RunReporter — store-and-forward reporting on top of ApiClient + Spool.

This is the seam the executor loop reports through. It turns "push
this finished run upstream" into a single call that is correct whether the API is
up or down:

  - `report_run(record, samples)` pushes the run and its trace live; if the API
    is unreachable it enqueues the whole report to the spool and returns cleanly.
  - `drain_spool()` replays queued reports in FIFO order on reconnect, stopping at
    the first still-failing push so ordering is preserved and nothing is dropped.

Correctness rests on two properties already built into the API:
  - **Idempotency** by `(controller_id, run_id)`: re-pushing a run
    on a retry or after a restart returns the same server id and inserts no
    duplicate row — so a re-drain is always safe.
  - **Server-id resolution is local to one drain step**: push the run, read its
    server id from the response, push the trace to it. No server id is persisted
    across restarts; it is re-learned on each drain via the idempotent run push.

Known v1 limitation: if a run pushes live but its trace push then fails midway,
the whole report is spooled and re-drained, which re-pushes the run (idempotent,
harmless) but may re-append trace samples (the samples endpoint appends, not
idempotent). The common outage case — the run push itself fails, so no samples
were sent — re-drains with zero duplication. The executor's checkpoint model
 refines the partial-trace case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hmasync_controller.apiclient import ApiClient, ApiResult, server_run_id
from hmasync_controller.spool import Spool

# Max samples per POST /runs/{id}/samples request. The API caps a batch at 10000
# (MAX_SAMPLES_BATCH); stay well under so a long run's trace is checkpointed
# across several requests rather than rejected as one oversized body.
SAMPLE_BATCH_SIZE = 5000


@dataclass
class ReportResult:
    """Outcome of report_run: pushed live, or spooled for later."""

    ok: bool
    spooled: bool = False
    server_run_id: str | None = None
    error: str | None = None


@dataclass
class DrainResult:
    """Outcome of drain_spool: how many reports flushed and how many remain."""

    drained: int
    remaining: int
    # True when the drain stopped early because a push was still failing.
    stopped_early: bool = False
    error: str | None = None


def _chunks(items: list, size: int) -> list[list]:
    """Split a list into contiguous chunks of at most `size` (order preserved)."""
    if size <= 0:
        return [items] if items else []
    return [items[i: i + size] for i in range(0, len(items), size)]


class RunReporter:
    """Reports runs upstream with a spool fallback for offline resilience."""

    def __init__(
        self,
        client: ApiClient,
        spool: Spool,
        *,
        sample_batch_size: int = SAMPLE_BATCH_SIZE,
    ):
        self.client = client
        self.spool = spool
        self.sample_batch_size = sample_batch_size

    # --- report -----------------------------------------------------------

    def report_run(
        self,
        record: dict[str, Any],
        samples: list[dict[str, Any]] | None = None,
    ) -> ReportResult:
        """Push a finished run + its trace live; spool the whole report on outage.

        Returns `spooled=True` when the report was buffered for a later drain
        (the API was unreachable). Never raises.
        """
        samples = samples or []
        ok, sid, err, transport = self._push_bundle(record, samples)
        if ok:
            return ReportResult(ok=True, spooled=False, server_run_id=sid)

        # Buffer for a later drain. We spool on any push failure so the report is
        # never lost; the idempotent re-push on drain makes that safe.
        self.spool.enqueue(record, samples)
        return ReportResult(ok=False, spooled=True, error=err)

    # --- drain ------------------------------------------------------------

    def drain_spool(self, max_items: int | None = None) -> DrainResult:
        """Flush queued reports in FIFO order; stop at the first still-failing push.

        Stopping early (rather than skipping) preserves ordering and guarantees a
        report is only removed from the spool once it (and its trace) are durably
        upstream. Safe to call repeatedly — idempotency covers a partial re-drain.
        """
        items = self.spool.pending(limit=max_items)
        drained = 0
        for item in items:
            ok, _sid, err, _transport = self._push_bundle(item.record, item.samples)
            if not ok:
                return DrainResult(
                    drained=drained,
                    remaining=self.spool.count(),
                    stopped_early=True,
                    error=err,
                )
            self.spool.remove(item.id)
            drained += 1
        return DrainResult(drained=drained, remaining=self.spool.count())

    # --- internals --------------------------------------------------------

    def _push_bundle(
        self,
        record: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> tuple[bool, str | None, str | None, bool]:
        """Push a run then its trace. Returns (ok, server_id, error, transport_error)."""
        run_result: ApiResult = self.client.push_run(record)
        if not run_result.ok:
            return False, None, run_result.error, run_result.transport_error

        sid = server_run_id(run_result)
        if sid is None:
            # A 2xx with no run id in the body means we can't address the trace —
            # treat as a (non-transport) failure so it spools and is retried.
            return False, None, "run push returned no server id", False

        for batch in _chunks(samples, self.sample_batch_size):
            sample_result = self.client.push_samples(sid, batch)
            if not sample_result.ok:
                return False, sid, sample_result.error, sample_result.transport_error

        return True, sid, None, False
