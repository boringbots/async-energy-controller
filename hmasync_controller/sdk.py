"""
The library face of Async Energy — for callers who already have an orchestrator.

The daemon in this package is one valid client shape: it is the thing that is
awake at 3am on a box you own. But if your work is already being driven by
something — an Airflow DAG, a CI job, a cron entry, an agent assembling a
pipeline — a second scheduler is redundant, and installing a systemd unit may
not even be possible.

This module is the other shape. Control stays with your process; Async Energy is
something you ask:

    from hmasync_controller.sdk import AsyncEnergy

    ae = AsyncEnergy(api_key="...")

    window = ae.next_window(est_duration_s=1800, deadline="by 7am")
    window.wait()                       # sleeps until the cheap hour
    with ae.measure(fingerprint="nightly-embeddings"):
        do_the_work()                   # your code, your process
    # the run reports itself on exit — energy included, if the box can measure it

No daemon, no `jobs.json`, no systemd, and no inbound trust boundary: nothing
calls into your machine, your code calls out. That whole class of concern simply
does not arise here.

What you give up versus the daemon: nothing restarts your job if it fails, and
nothing is awake to run it if your process is not. Pick the shape that matches
who is holding the clock.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from hmasync_controller.apiclient import ApiClient, server_run_id
from hmasync_controller.profiler import Profiler, get_profiler

logger = logging.getLogger("hmasync.sdk")

DEFAULT_API_URL = "https://api.async.energy"


class AsyncEnergyError(Exception):
    """A call could not be completed. Carries the API's own message where there is one."""


@dataclass
class Window:
    """When to run, and what it should cost.

    `feasible=False` means the work cannot finish before its deadline. That is a
    real answer, not an error — `reason` says what to change. `start` is None in
    that case, and `wait()` refuses rather than sleeping forever.
    """

    start: datetime | None
    end: datetime | None
    grid_cost: float | None = None
    now_cost: float | None = None
    savings: float | None = None
    predicted_wh: float | None = None
    feasible: bool = True
    reason: str | None = None
    prediction_method: str = "prior"
    degraded: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def seconds_until_start(self) -> float:
        """Seconds to wait now. 0 when the window is open or already past."""
        if self.start is None:
            return 0.0
        return max(0.0, (self.start - datetime.now(timezone.utc)).total_seconds())

    def wait(self, *, sleep=time.sleep, max_wait_s: float | None = None) -> float:
        """Block until the window opens. Returns the seconds actually slept.

        `sleep` is injectable so tests (and async callers wrapping this) do not
        have to really wait. `max_wait_s` caps the sleep for a caller that would
        rather re-ask than hold a process open for hours.
        """
        if not self.feasible:
            raise AsyncEnergyError(
                f"cannot wait for an infeasible window: {self.reason or 'no reason given'}"
            )
        delay = self.seconds_until_start
        if max_wait_s is not None:
            delay = min(delay, max_wait_s)
        if delay > 0:
            sleep(delay)
        return delay


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class AsyncEnergy:
    """A thin, synchronous client. One object, two things worth doing."""

    def __init__(
        self,
        *,
        api_key: str = "",
        api_url: str = "",
        email: str = "",
        password: str = "",
        controller_id: str = "",
        timeout: float = 10.0,
        client: ApiClient | None = None,
        profiler: Profiler | None = None,
    ):
        """Credentials come from the arguments, else the environment.

        `HM_ASYNC_API_KEY` is the one to prefer for automation: scoped to this
        caller and revocable without touching a human's password. Falls back to
        `HM_ASYNC_EMAIL`/`HM_ASYNC_PASSWORD` so an existing controller `.env`
        works unchanged.
        """
        self._controller_id = (
            controller_id or os.environ.get("CONTROLLER_ID", "") or _default_controller_id()
        )

        if client is not None:
            # An injected client carries its own credentials and transport. Do
            # NOT re-derive them from the environment and do NOT demand them
            # here — the caller has already made those choices, and second-
            # guessing would make a fully-configured client unusable.
            self._client = client
        else:
            api_key = api_key or os.environ.get("HM_ASYNC_API_KEY", "")
            api_url = api_url or os.environ.get("HM_ASYNC_API_URL", "") or DEFAULT_API_URL
            email = email or os.environ.get("HM_ASYNC_EMAIL", "")
            password = password or os.environ.get("HM_ASYNC_PASSWORD", "")
            if not api_key and not (email and password):
                raise AsyncEnergyError(
                    "No credentials: pass api_key= (preferred for automation), or "
                    "email=/password=, or set HM_ASYNC_API_KEY."
                )
            self._client = ApiClient(
                base_url=api_url,
                email=email,
                password=password,
                controller_id=self._controller_id,
                api_key=api_key,
                timeout=timeout,
            )
        self._profiler = profiler
        self._owns_client = client is None

    # --- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "AsyncEnergy":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- the two calls that matter ---------------------------------------

    def next_window(
        self,
        *,
        est_duration_s: float,
        deadline: str,
        earliest_start: str | None = None,
        nameplate_watts: float | None = None,
        framework: str = "command",
        fingerprint: str | None = None,
    ) -> Window:
        """Ask when this work should run. Nothing is registered or stored.

        `deadline` and `earliest_start` take human strings read in your account's
        timezone — "by 7am", "22:00", "2026-08-20T09:00".
        """
        result = self._client.advise(
            est_duration_s=est_duration_s,
            deadline=deadline,
            earliest_start=earliest_start,
            nameplate_watts=nameplate_watts,
            framework=framework,
            fingerprint=fingerprint,
        )
        if not result.ok:
            raise AsyncEnergyError(result.error or "advise failed")

        data = result.data or {}
        return Window(
            start=_parse_dt(data.get("start")),
            end=_parse_dt(data.get("end")),
            grid_cost=data.get("grid_cost"),
            now_cost=data.get("now_cost"),
            savings=data.get("savings"),
            predicted_wh=data.get("predicted_wh"),
            feasible=bool(data.get("feasible", True)),
            reason=data.get("reason"),
            prediction_method=data.get("prediction_method", "prior"),
            degraded=bool(data.get("degraded", False)),
            raw=data,
        )

    @contextmanager
    def measure(
        self,
        *,
        workflow_id: str | None = None,
        fingerprint: str | None = None,
        run_id: str | None = None,
        report: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Wrap a block of work, measure its energy, and report the run.

        Yields a mutable dict the caller can annotate before the block exits —
        set `work_units` / `work_unit_kind` if you know how much work was done,
        and `exit_status` to record a failure without raising.

        An exception inside the block is recorded as a failed run and then
        re-raised: your error handling is not swallowed to make a report tidy.

        Reporting is best-effort. A telemetry push must never be the reason a
        successful job is treated as failed, so a failed report is logged, not
        raised. Energy stays `null` when the box cannot measure it — never a
        fabricated number.
        """
        run_id = run_id or str(uuid.uuid4())
        profiler = self._profiler or get_profiler()
        started = datetime.now(timezone.utc)
        monotonic_start = time.monotonic()

        handle: dict[str, Any] = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "fingerprint": fingerprint,
            "work_units": None,
            "work_unit_kind": None,
            "exit_status": "success",
        }

        profiler.start(run_id)
        try:
            yield handle
        except BaseException:
            handle["exit_status"] = "failed"
            raise
        finally:
            try:
                telemetry = profiler.stop(run_id)
            except Exception:
                logger.warning("profiler.stop failed; reporting without telemetry", exc_info=True)
                telemetry = None

            duration_s = time.monotonic() - monotonic_start
            if report:
                self._report(handle, started, duration_s, telemetry)

    # --- internals --------------------------------------------------------

    def _report(
        self,
        handle: dict[str, Any],
        started: datetime,
        duration_s: float,
        telemetry: Any,
    ) -> None:
        record: dict[str, Any] = {
            "run_id": handle["run_id"],
            "controller_id": self._controller_id,
            "workflow_id": handle.get("workflow_id"),
            "fingerprint": handle.get("fingerprint"),
            "ts": started,
            "duration_s": duration_s,
            "exit_status": handle.get("exit_status", "success"),
            "work_units": handle.get("work_units"),
            "work_unit_kind": handle.get("work_unit_kind"),
        }
        samples: list = []
        if telemetry is not None:
            record.update(telemetry.to_record_fields())
            samples = list(getattr(telemetry, "samples", []) or [])

        result = self._client.push_run(record)
        if not result.ok:
            # Deliberately not raised: the work succeeded, and losing a
            # measurement is not a reason to fail the caller's job.
            logger.warning("run report failed (%s); the measurement is lost", result.error)
            return

        # The ~1 Hz trace, so the dashboard can draw real measured power. Best
        # effort for the same reason, and keyed by the SERVER run id (a distinct
        # id from our client-side run_id).
        if samples:
            sid = server_run_id(result)
            if sid:
                sample_result = self._client.push_samples(sid, samples)
                if not sample_result.ok:
                    logger.warning("sample push failed (%s); the trace is lost",
                                   sample_result.error)


def _default_controller_id() -> str:
    import socket

    try:
        return socket.gethostname() or "unknown"
    except Exception:  # pragma: no cover - defensive
        return "unknown"
