"""
hm-async controller — console entrypoint (`hm-async-controller`).

Wires the already-built seams into a runnable process:

    Settings → ApiClient (login) → Spool → RunReporter → ScheduleExecutor → run_forever

The executor needs to turn a pulled `Placement` (which carries only a
`workflow_id`) into something runnable (framework + adapter request). The wire
contract is the fixed four endpoints (no "fetch workflow" route), so
the controller carries its own **local job catalog**: a JSON file mapping each
`workflow_id` to `{framework, request, deadline?, earliest_start?}`. This module
loads that file and hands it to the executor as its `job_source`.

Everything here is constructed lazily and boots with an empty `.env`:
building the client/spool/executor touches no socket and reads
no file until `run`/`--once` actually runs. A missing job catalog is a warning,
not an error — the executor still pulls schedules and drains the spool; it just
skips workflows it has no local definition for.

Three modes beyond the loop, each answering a question the loop cannot:

    register    create a workflow AND write its catalog entry, so the server's
                workflow id is never copy-pasted by hand
    --check     resolve every layer (catalog, auth, schedule, and the match
                between them) and exit non-zero on anything wrong
    --once      one tick, for cron

`--check` exists because `--once` prints `outcomes=0` whether the catalog is
empty, a workflow id is mistyped, or it is simply 16:00 and the window opens at
20:00 — a smoke test whose output cannot answer what a smoke test is asked.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hmasync_controller.adapters import (
    AdapterError,
    get_adapter,
    list_frameworks,
    normalize_command,
)
from hmasync_controller.apiclient import ApiClient
from hmasync_controller.bench import denylisted_keys, drain_bench_spool, submit_bundle_file
from hmasync_controller.config import (
    BENCH_CONSENT_TEXT,
    Settings,
    resolve_controller_id,
    set_bench_optin,
)
from hmasync_controller.executor import ScheduleExecutor
from hmasync_controller.fingerprint import (
    collect_fingerprint,
    compute_node_hash,
    load_or_create_salt,
)
from hmasync_controller.powercap import PowerCapManager
from hmasync_controller.profiler import NVMLProfiler, Profiler, get_profiler
from hmasync_controller.reporter import RunReporter
from hmasync_controller.spool import Spool

logger = logging.getLogger("hmasync.cli")

# Where the local job catalog lives. The value is resolved through `Settings`
# (see `resolve_catalog_path`) so `.env`, a real environment variable, and
# `--job-catalog` all work and have an obvious precedence.
JOB_CATALOG_ENV = "HM_ASYNC_JOB_CATALOG"
DEFAULT_JOB_CATALOG = "jobs.json"

# Ticks between repeats of the empty-catalog warning. At the default 30s poll
# that is roughly hourly: often enough that `journalctl -n 50` on a box that has
# been running for a week still shows it, rare enough to stay ignorable noise-wise.
EMPTY_CATALOG_WARN_EVERY = 120

# The recurrence values the API accepts. Checked here so `register` rejects a
# typo at the command line, where the message can name the valid values, rather
# than sending it and surfacing a validation error.
RECURRENCES = ("none", "daily", "weekly")


def resolve_catalog_path(
    settings: Settings, explicit: str | os.PathLike[str] | None = None
) -> str:
    """Resolve the job-catalog path, highest precedence first.

    1. `--job-catalog PATH` (the flag the operator just typed)
    2. `HM_ASYNC_JOB_CATALOG` — a real environment variable, then `.env`
       (pydantic-settings' own ordering; both land on the same field)
    3. `jobs.json` in the working directory

    Kept as a named function because "which file did it actually read?" is the
    question `--check` and the startup log both need to answer.
    """
    if explicit:
        return str(explicit)
    return settings.HM_ASYNC_JOB_CATALOG or DEFAULT_JOB_CATALOG


def _parse_catalog_data(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Filter a parsed JSON object into the workflow_id → spec catalog.

    Keeps only object-valued entries; the executor coerces each spec to a JobDef.
    """
    return {str(wid): spec for wid, spec in data.items() if isinstance(spec, dict)}


def _read_catalog(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]] | None:
    """Read + parse the catalog file, returning None on ANY failure.

    A `None` return (missing file, unreadable/malformed JSON, or a non-object top
    level) is deliberately distinct from a valid-but-empty `{}` so a watcher can
    tell "the read failed, keep the last good catalog" from "the file validly has
    no jobs". `load_job_catalog` maps None → `{}` for the load-once path.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return _parse_catalog_data(data)


def load_job_catalog(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Load the local workflow_id → job-definition mapping from a JSON file.

    Expected shape (each value coerced to a `JobDef` by the executor)::

        {
          "<workflow_id>": {
            "framework": "command|ollama|openai",
            "request": {... adapter-specific... },
            "deadline": "2026-07-11T07:00:00-04:00",     # optional, resolved+tz-aware
            "earliest_start": "2026-07-10T22:00:00-04:00" # optional
          }
        }

    A missing file returns an empty catalog (a warning, not an error): the
    executor still runs, pulls schedules, and drains the spool — it just has no
    local job to execute yet. A malformed file also degrades to empty (logged) so
    a typo never crashes the loop.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("job catalog %s not found; running with an empty catalog", p)
        return {}
    try:
        data = json.loads(p.read_text())
    except (ValueError, OSError) as exc:
        logger.warning("job catalog %s is unreadable (%s); running with an empty catalog", p, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("job catalog %s is not a JSON object; running with an empty catalog", p)
        return {}
    return _parse_catalog_data(data)


class CatalogWatcher:
    """A callable `job_source` that re-reads the catalog file when its mtime changes.

    Wired by `--watch-catalog`. The executor's `job_source` seam already accepts a
    callable ``(workflow_id) -> spec | None`` (`executor.JobSource`), so this is a
    drop-in that keeps the wire contract and the seam unchanged — an agent that
    appends a new workflow to `jobs.json` gets it run without restarting the daemon.

    Each lookup cheaply `stat()`s the file; only an mtime change triggers a re-read.
    A file that goes **missing or malformed while watching keeps the LAST GOOD
    catalog** (a bad edit must never silently stop every job) — distinguished from a
    validly-empty `{}`, which IS adopted, via `_read_catalog`'s None-on-failure
    contract. The initial load mirrors `load_job_catalog` (missing/malformed → empty
    + warning) so `--watch-catalog` on a not-yet-created file behaves like load-once.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self._path = Path(path)
        self._catalog: dict[str, dict[str, Any]] = {}
        self._mtime: float | None = None
        self._primed = False
        self._refresh(initial=True)

    def __call__(self, workflow_id: str | None) -> dict[str, Any] | None:
        self._refresh()
        return self._catalog.get(workflow_id)

    def __len__(self) -> int:
        """Current entry count, so the tick line can report it like a plain dict."""
        self._refresh()
        return len(self._catalog)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """The catalog as it stands right now (for `--check` and logging)."""
        self._refresh()
        return dict(self._catalog)

    def _stat_mtime(self) -> float | None:
        try:
            return self._path.stat().st_mtime
        except OSError:
            return None

    def _refresh(self, *, initial: bool = False) -> None:
        mtime = self._stat_mtime()
        # Unchanged (including still-missing: None == None) → no re-read, no re-warn.
        if self._primed and mtime == self._mtime:
            return

        parsed = _read_catalog(self._path)
        if parsed is not None:
            # Good read — possibly a legitimately smaller/empty catalog. Adopt it.
            if self._primed:
                logger.info("job catalog %s changed; reloaded %d entr%s",
                            self._path, len(parsed), "y" if len(parsed) == 1 else "ies")
            self._catalog = parsed
        elif initial:
            # First load of a missing/malformed file → load-once semantics (empty).
            logger.warning("job catalog %s not usable; starting with an empty catalog", self._path)
            self._catalog = {}
        else:
            # Missing/malformed WHILE watching → keep the last good catalog.
            logger.warning(
                "job catalog %s became missing or malformed; keeping the last good catalog (%d entries)",
                self._path, len(self._catalog),
            )
        # Record the observed mtime either way so a persistently-broken file is not
        # re-read/re-warned every tick; a later fix bumps the mtime and re-reads.
        self._mtime = mtime
        self._primed = True


def catalog_snapshot(job_source: Any) -> dict[str, dict[str, Any]]:
    """The catalog behind a job_source, whichever shape `build_executor` chose."""
    if isinstance(job_source, CatalogWatcher):
        return job_source.snapshot()
    if isinstance(job_source, dict):
        return dict(job_source)
    return {}


class TickLogger:
    """Logs one line per tick, and re-warns about an empty catalog periodically.

    Two things a long-running daemon got wrong before:

    * `tick: … outcomes=0 drained=0` is byte-identical whether the controller is
      correctly idle at 16:00 with a job due at 20:00, or holding nothing at all
      because the catalog never loaded. `catalog=`/`placements=`/`next=` are the
      numbers that tell those apart at a glance — and they also confirm that
      tonight's plan actually arrived.
    * The empty-catalog warning was emitted once, at startup. In a systemd unit
      that line scrolls out of `journalctl -n 20` within minutes, leaving an
      operator staring at a healthy-looking loop that will never run anything. So
      it repeats, on a slow cadence that will not drown the log.
    """

    def __init__(self, job_source: Any, *, warn_every: int = EMPTY_CATALOG_WARN_EVERY):
        self._job_source = job_source
        self._warn_every = warn_every
        self._ticks = 0

    def __call__(self, result: Any) -> None:
        self._ticks += 1
        size = self._catalog_size()
        logger.info(
            "tick: version=%s mode=%s reachable=%s catalog=%s placements=%d "
            "pending=%d next=%s outcomes=%d drained=%d",
            result.version, result.mode, result.reachable,
            "?" if size is None else size,
            result.placements, result.pending, _fmt_time(result.next_start),
            len(result.outcomes), result.drained,
        )
        for outcome in result.outcomes:
            logger.info(
                "  ran workflow=%s status=%s exit=%s%s",
                outcome.workflow_id, outcome.status, outcome.exit_status,
                f" ({outcome.reason})" if outcome.reason else "",
            )
        # Re-warn on the first tick and every `warn_every` ticks thereafter.
        if size == 0 and (self._ticks - 1) % self._warn_every == 0:
            logger.warning(
                "the job catalog is empty — this controller cannot run anything. "
                "Check --job-catalog / $%s, or run `--check` to see what is wrong.",
                JOB_CATALOG_ENV,
            )

    def _catalog_size(self) -> int | None:
        """Entry count, or None for a bare-callable job_source (size is unknowable)."""
        try:
            return len(self._job_source)
        except TypeError:
            return None


def _fmt_time(value: Any) -> str:
    """`HH:MM` in local time for the tick line, or `-` when there is nothing next."""
    if value is None:
        return "-"
    try:
        return value.astimezone().strftime("%H:%M")
    except (ValueError, OSError, AttributeError):  # pragma: no cover - defensive
        return str(value)


def build_executor(
    settings: Settings | None = None,
    *,
    job_catalog_path: str | os.PathLike[str] | None = None,
    watch_catalog: bool = False,
) -> ScheduleExecutor:
    """Construct a wired ScheduleExecutor from settings (no network, no login yet).

    Login and the run loop happen in `main`; this stays side-effect-free so it is
    unit-testable and empty-`.env`-safe.

    `watch_catalog` picks which shape of `job_source` the executor gets:

    * ``False`` (default) — the catalog is read ONCE and passed as a plain
      mapping. Unchanged behavior; an existing deployment is unaffected.
    * ``True`` — a `CatalogWatcher` callable that re-reads on mtime change, so a
      workflow appended to `jobs.json` after start-up runs without a restart.

    Both satisfy `executor.JobSource`, which already accepts a callable or a
    mapping — the seam is used as-is, not widened.
    """
    settings = settings or Settings()
    controller_id = resolve_controller_id(settings.CONTROLLER_ID)

    client = ApiClient(
        base_url=settings.HM_ASYNC_API_URL,
        email=settings.HM_ASYNC_EMAIL,
        password=settings.HM_ASYNC_PASSWORD,
        controller_id=controller_id,
        timeout=settings.HTTP_TIMEOUT_S,
    )
    spool = Spool(settings.SPOOL_PATH)
    reporter = RunReporter(client, spool)

    catalog_path = resolve_catalog_path(settings, job_catalog_path)
    job_source: CatalogWatcher | dict[str, dict[str, Any]] = (
        CatalogWatcher(catalog_path) if watch_catalog else load_job_catalog(catalog_path)
    )

    # Bench-bundle retries ride the SAME reconnect trigger as the run-report
    # spool (see ScheduleExecutor.tick). Only wired when opted in, so an
    # un-opted-in controller never opens the bench spool file or attempts a
    # bench-related request.
    extra_drain = None
    if settings.BENCH_OPTIN:
        bench_spool = Spool(settings.BENCH_SPOOL_PATH)
        extra_drain = lambda: drain_bench_spool(client, bench_spool).drained  # noqa: E731

    # A single profiler instance for BOTH telemetry sampling and the power cap
    # (when wired) — they share the same NVML handle, not two separate inits.
    profiler = get_profiler()

    # Independent of BENCH_OPTIN (see config.APPLY_POWER_CAP): only requires
    # this box to actually have an NVML-backed GPU, since there is nothing to
    # cap otherwise.
    power_cap = None
    if settings.APPLY_POWER_CAP and isinstance(profiler, NVMLProfiler):
        node_hash = compute_node_hash(load_or_create_salt(settings.NODE_SALT_PATH))
        power_cap = PowerCapManager(client=client, profiler=profiler, node_hash=node_hash)

    return ScheduleExecutor(
        client=client,
        reporter=reporter,
        job_source=job_source,
        profiler=profiler,
        controller_id=controller_id,
        extra_drain=extra_drain,
        power_cap=power_cap,
    )


# ============================================================
# --check — answer the question a smoke test is actually asked
# ============================================================
#
# `--once` prints `outcomes=0`, which is the same output whether the catalog is
# empty, the schedule is empty, a workflow id is mistyped, or it is 16:00 and the
# window opens at 20:00. Those need different fixes, so they need different
# output. `--check` resolves every layer and names what is wrong.
#
# The two set differences carry most of the value:
#   unmatched — scheduled here, but absent from the catalog. The optimizer plans
#               it, this box silently skips it. Always a fault.
#   orphaned  — in the catalog, but not in tonight's schedule. Usually benign (a
#               weekly job, or one not yet planned), sometimes a mistyped or
#               deleted id. Reported, never fatal — a check that cries wolf on a
#               normal Tuesday stops being run.

CHECK_OK = 0
CHECK_PROBLEMS = 1
CHECK_CONFIG_ERROR = 2


@dataclass
class CheckReport:
    """Accumulated `--check` output plus whether anything is actually wrong."""

    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def row(self, label: str, text: str) -> None:
        self.lines.append(f"{label:<12} {text}")

    def fault(self, label: str, text: str) -> None:
        self.row(label, text)
        # The label is indented in the table to show nesting; the summary list is
        # flat, so strip it back before repeating it there.
        self.problems.append(f"{label.strip()}: {text}")

    @property
    def exit_code(self) -> int:
        return CHECK_PROBLEMS if self.problems else CHECK_OK

    def render(self) -> str:
        out = list(self.lines)
        if self.problems:
            out.append("")
            out.append(f"{len(self.problems)} problem(s) found:")
            out.extend(f"  - {p}" for p in self.problems)
        else:
            out.append("")
            out.append("ok — catalog, auth, and schedule all resolve.")
        return "\n".join(out)


def _check_catalog(report: CheckReport, path: str) -> dict[str, dict[str, Any]]:
    """Resolve + validate the catalog file; every entry is checked, not just parsed."""
    p = Path(path)
    if not p.exists():
        report.fault("catalog", f"{p} does not exist")
        return {}

    parsed = _read_catalog(p)
    if parsed is None:
        report.fault("catalog", f"{p} is not readable as a JSON object")
        return {}
    if not parsed:
        report.fault("catalog", f"{p} (0 jobs) — nothing can run")
        return {}

    report.row("catalog", f"{p} ({len(parsed)} job{'s' if len(parsed) != 1 else ''})")
    for wid, spec in sorted(parsed.items()):
        framework = spec.get("framework") or ""
        try:
            adapter = get_adapter(framework)
        except AdapterError as exc:
            report.fault("  entry", f"{_short(wid)} {exc}")
            continue
        # fingerprint() is the cheapest total validation of a request there is: it
        # already raises on every field the adapter requires to run, and it costs
        # nothing. Reusing it means --check can never drift from what run() needs.
        try:
            adapter.fingerprint(dict(spec.get("request") or {}))
        except AdapterError as exc:
            report.fault("  entry", f"{_short(wid)} ({framework}) {exc}")
    return parsed


def _check_auth(report: CheckReport, client: ApiClient, settings: Settings) -> bool:
    login = client.login()
    if not login.ok:
        report.fault("auth", f"login failed — {login.error}")
        return False
    report.row(
        "auth",
        f"ok — {settings.HM_ASYNC_EMAIL or '(api key)'}, "
        f"controller_id={client.controller_id}",
    )
    return True


def _check_schedule(report: CheckReport, client: ApiClient) -> dict[str, Any] | None:
    result = client.pull_schedule(after=-1)
    if not result.ok:
        report.fault("schedule", f"could not be pulled — {result.error}")
        return None
    if result.status_code == 204 or not isinstance(result.data, dict):
        report.fault("schedule", "none published yet for this account")
        return None
    data = result.data
    placements = [p for p in (data.get("placements") or []) if isinstance(p, dict)]
    report.row(
        "schedule",
        f"version {data.get('version')}, {len(placements)} placement"
        f"{'s' if len(placements) != 1 else ''}, valid until {data.get('valid_until')}",
    )
    if data.get("degraded"):
        report.row("", "note: degraded=true — prices were stale when this was planned")
    return data


def _check_matching(
    report: CheckReport, catalog: dict[str, dict[str, Any]], schedule: dict[str, Any]
) -> None:
    placements = [p for p in (schedule.get("placements") or []) if isinstance(p, dict)]
    scheduled = {str(p.get("workflow_id")): p for p in placements if p.get("workflow_id")}

    for wid, placement in sorted(scheduled.items()):
        window = _fmt_window(placement)
        wh = placement.get("predicted_wh")
        wh_text = f"{wh:.1f} Wh" if isinstance(wh, (int, float)) else "— Wh"
        name = (catalog.get(wid) or {}).get("name") or ""
        if wid in catalog:
            report.row("matched", f"{_short(wid)} {name:<22} {window}  {wh_text}")
        elif not placement.get("feasible", True):
            # Infeasible AND unknown locally: the optimizer already declined to
            # place it, so the missing entry is not what is blocking it.
            report.row("unmatched", f"{_short(wid)} {name:<22} infeasible: "
                                    f"{placement.get('reason') or 'no reason given'}")
        else:
            report.fault(
                "unmatched",
                f"{_short(wid)} is scheduled for {window} but has no catalog entry — "
                "this box will skip it",
            )

    for wid in sorted(set(catalog) - set(scheduled)):
        name = (catalog.get(wid) or {}).get("name") or ""
        report.row(
            "orphaned",
            f"{_short(wid)} {name:<22} in the catalog, not in this schedule "
            "(not planned tonight, disabled, or a mistyped id)",
        )


def run_check(
    settings: Settings, *, job_catalog_path: str | os.PathLike[str] | None = None
) -> tuple[int, str]:
    """Resolve every layer the controller depends on. Returns (exit_code, text)."""
    report = CheckReport()
    catalog_path = resolve_catalog_path(settings, job_catalog_path)
    catalog = _check_catalog(report, catalog_path)

    if not settings.HM_ASYNC_API_URL:
        report.fault("auth", "HM_ASYNC_API_URL is not set — nothing to connect to")
        return CHECK_CONFIG_ERROR, report.render()

    client = _client_from(settings)
    try:
        if not _check_auth(report, client, settings):
            return report.exit_code, report.render()
        schedule = _check_schedule(report, client)
        if schedule is not None:
            _check_matching(report, catalog, schedule)
    finally:
        client.close()
    return report.exit_code, report.render()


def _short(workflow_id: str) -> str:
    """First segment of a UUID — enough to identify a row, narrow enough to scan."""
    return str(workflow_id).split("-")[0]


def _fmt_window(placement: dict[str, Any]) -> str:
    """`HH:MM–HH:MM` in local time, or a placeholder when unplaced."""
    start, end = placement.get("start"), placement.get("end")
    if not start:
        return "unplaced"
    return f"{_fmt_time(_parse_iso(start))}–{_fmt_time(_parse_iso(end))}"


def _parse_iso(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _client_from(settings: Settings) -> ApiClient:
    return ApiClient(
        base_url=settings.HM_ASYNC_API_URL,
        email=settings.HM_ASYNC_EMAIL,
        password=settings.HM_ASYNC_PASSWORD,
        controller_id=resolve_controller_id(settings.CONTROLLER_ID),
        timeout=settings.HTTP_TIMEOUT_S,
    )


# ============================================================
# register — create the workflow AND write the catalog entry
# ============================================================
#
# Registering used to be the one step with no tooling: hand-rolled `curl` against
# /api/v1/workflows, then a UUID copy-pasted into jobs.json. That is both the most
# error-prone step in the setup and the only one where a typo fails silently — a
# mistyped id is simply a workflow this box never runs, with no log line naming it.
#
# Doing both halves in one command is the point. The server's id goes straight
# into the catalog, so the two cannot disagree at creation time.


def _workflow_id_from(data: Any) -> str | None:
    """Pull the new workflow's id out of a create response, whatever it is nested in."""
    if not isinstance(data, dict):
        return None
    for key in ("workflow_id", "id"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    nested = data.get("workflow")
    if isinstance(nested, dict):
        return _workflow_id_from(nested)
    return None


def _build_request(args: argparse.Namespace) -> dict[str, Any]:
    """Turn the register flags into the adapter request stored in the catalog."""
    request: dict[str, Any] = {}
    if args.command:
        # Stored as an argv LIST, not a string: it is what actually runs, and it
        # sidesteps shell-quoting entirely for arguments containing spaces or
        # quotes. A string would be shlex.split at run time to reach the same
        # place, with one more chance to be wrong.
        request["command"] = normalize_command(args.command)
    if args.model:
        request["model"] = args.model
    if args.prompt:
        request["prompt"] = args.prompt
    if args.prompt_file:
        request["prompt_file"] = args.prompt_file
    if args.base_url:
        request["base_url"] = args.base_url
    if args.timeout:
        request["timeout"] = args.timeout
    if args.cwd:
        request["cwd"] = args.cwd
    return request


def write_catalog_entry(
    path: str | os.PathLike[str], workflow_id: str, entry: dict[str, Any]
) -> None:
    """Add one workflow to the catalog file, preserving everything already in it.

    Refuses to write over a file that exists but does not parse: an operator's
    hand-edited catalog is not something to replace with a fresh one because a
    trailing comma made it unreadable for a moment. Comment keys and formatting
    choices survive because the whole parsed document is written back, not just
    the entries the executor cares about.

    Written via a temp file + `os.replace`, so a `--watch-catalog` daemon reading
    concurrently sees either the old catalog or the new one, never a half-file.
    """
    p = Path(path)
    document: dict[str, Any] = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except (ValueError, OSError) as exc:
            raise RegisterError(
                f"{p} exists but could not be parsed ({exc}); "
                "fix or move it before registering, so nothing already there is lost"
            ) from exc
        if not isinstance(existing, dict):
            raise RegisterError(f"{p} is not a JSON object; refusing to overwrite it")
        document = existing

    document[workflow_id] = entry
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(tmp, p)


class RegisterError(Exception):
    """A register run could not be completed (bad flags, API refusal, unsafe write)."""


def run_register(
    settings: Settings,
    args: argparse.Namespace,
    *,
    client: ApiClient | None = None,
) -> tuple[int, str]:
    """Create the workflow server-side and append it to the local catalog."""
    request = _build_request(args)
    if not request:
        return 2, "nothing to run: pass --command, or --model with --prompt/--prompt-file"

    entry: dict[str, Any] = {"framework": args.framework, "request": request}
    # `name` is not read by the executor — it is there so `--check` and the catalog
    # itself are legible to a human six months later, when a bare UUID is not.
    entry["name"] = args.name

    # Deliberately NOT writing deadline/earliest_start into the entry. The two
    # sides take different formats: the API takes human strings ("by 7am") and
    # resolves them itself, while the catalog's optional `deadline` is a fallback
    # hint that must be a tz-aware ISO datetime — `_parse_dt("by 7am")` is None.
    # Writing the human string would produce a field that looks like it sets a
    # deadline and silently does nothing, which is the exact class of bug this
    # command exists to remove. The schedule carries the resolved window.

    catalog_path = resolve_catalog_path(settings, args.job_catalog)

    if args.dry_run:
        preview = json.dumps({"<workflow_id>": entry}, indent=2)
        return 0, f"dry run — would register {args.name!r} and write to {catalog_path}:\n{preview}"

    if not settings.HM_ASYNC_API_URL:
        return 2, "HM_ASYNC_API_URL is not set — nothing to connect to. See .env.example."

    owned = client is None
    client = client or _client_from(settings)
    try:
        result = client.create_workflow(
            name=args.name,
            framework=args.framework,
            # Off by default: what this box runs is a local concern, and the
            # catalog already describes it.
            request=request if args.share_request else None,
            est_duration_s=args.est_duration,
            deadline=args.deadline,
            earliest_start=args.earliest_start,
            recurrence=args.recurrence,
            nameplate_watts=args.nameplate_watts,
            enabled=False if args.disabled else None,
            bench_gpu_class=args.bench_gpu_class,
            bench_model_size_class=args.bench_model_size_class,
            bench_quant=args.bench_quant,
        )
        if not result.ok:
            return 1, f"could not register {args.name!r}: {result.error}"

        workflow_id = _workflow_id_from(result.data)
        if not workflow_id:
            # The workflow may well exist server-side now; say so rather than
            # implying nothing happened.
            return 1, (
                f"the API accepted {args.name!r} but returned no workflow id; "
                f"nothing was written to {catalog_path}. Response: {result.data!r}"
            )

        try:
            write_catalog_entry(catalog_path, workflow_id, entry)
        except (RegisterError, OSError) as exc:
            return 1, (
                f"registered {args.name!r} as {workflow_id}, but the catalog was NOT "
                f"updated: {exc}\nAdd it by hand so the two do not drift."
            )

        # Best-effort: opted-in, so also upsert this box's hardware fingerprint.
        # A hiccup here must never undo a successful workflow registration,
        # which is the primary action of this command — hence the note rather
        # than a non-zero exit.
        node_note = ""
        if settings.BENCH_OPTIN:
            _node_code, node_message = _register_node(client, get_profiler(), settings)
            node_note = (
                f"\n{node_message}" if _node_code == 0
                else f"\nnode fingerprint not sent: {node_message}"
            )
    finally:
        if owned:
            client.close()

    return 0, (
        f"registered {args.name!r}\n"
        f"  workflow_id  {workflow_id}\n"
        f"  catalog      {catalog_path}\n"
        f"  framework    {args.framework}\n"
        "Run `--check` to confirm it lands in the next schedule.\n"
        "Tip: contribute ~25 minutes of benchmark data to improve your own "
        f"scheduling — run `bench opt-in` to see what is shared.{node_note}"
    )


# ============================================================
# bench opt-in / opt-out — explicit, headless-safe consent
# ============================================================
#
# BENCH_OPTIN gates everything the later bench stories do (running the suite,
# submitting a bundle, sending hardware-profile hints). Nothing bench-related
# ever leaves the box until an operator runs `bench opt-in`, and doing so
# always prints BENCH_CONSENT_TEXT first — there is no flag to flip blind. No
# interactive (y/n) prompt: a systemd box or a script has no TTY to answer one.

DEFAULT_ENV_PATH = ".env"


def run_bench(
    args: argparse.Namespace, *, env_path: str | os.PathLike[str] = DEFAULT_ENV_PATH
) -> tuple[int, str]:
    """Dispatch `bench opt-in` / `bench opt-out`. Persists BENCH_OPTIN; no prompt."""
    sub = getattr(args, "bench_subcommand", None)
    if sub == "opt-in":
        set_bench_optin(True, env_path)
        return 0, f"{BENCH_CONSENT_TEXT}\nOpted in — BENCH_OPTIN=true written to {env_path}."
    if sub == "opt-out":
        set_bench_optin(False, env_path)
        return 0, (
            f"Opted out — BENCH_OPTIN=false written to {env_path}. "
            "No further benchmark data will be sent."
        )
    return 2, "usage: async-energy-controller bench {opt-in,opt-out,register-node,quick,submit}"


# ============================================================
# bench register-node — send this box's hardware-CLASS fingerprint
# ============================================================
#
# Distinct from `bench submit`: this sends no benchmark data, just a hardware
# fingerprint (GPU model/driver/VRAM via NVML when present, CPU model + RAM
# from /proc, and a salted node_hash — see fingerprint.py) so the server can
# upsert a bench_nodes row and give this box bench-prior cold-start estimates.
# Reached two ways: the standalone `bench register-node` command, and as a
# best-effort side effect of a successful `register` when BENCH_OPTIN is set
# (see run_register below) — both funnel through `_register_node`.


def _register_node(client: ApiClient, profiler: Profiler, settings: Settings) -> tuple[int, str]:
    """Collect + send the fingerprint. Refuses locally if it ever carries a
    denylisted key (should never happen — collect_fingerprint only gathers
    gpu/cpu/ram class fields — but this is the same defense-in-depth bench.py
    applies to a submission bundle before it goes out)."""
    fingerprint = collect_fingerprint(profiler, settings.NODE_SALT_PATH)
    leaked = denylisted_keys(fingerprint)
    if leaked:
        return 1, "refusing to send — denylisted key(s) present: " + ", ".join(leaked)

    result = client.register_node(fingerprint)
    if result.ok:
        return 0, f"registered node {fingerprint['node_hash']}"
    return 1, f"could not register node: {result.error}"


def run_bench_register_node(
    settings: Settings, *, client: ApiClient | None = None, profiler: Profiler | None = None
) -> tuple[int, str]:
    """`bench register-node` — send this box's hardware-class fingerprint.

    Requires BENCH_OPTIN, the same single gate as every other bench upload —
    the fingerprint is one of the things BENCH_CONSENT_TEXT names as shared.
    """
    if not settings.BENCH_OPTIN:
        return 2, "not opted in — run `bench opt-in` first; nothing was sent."
    if not settings.HM_ASYNC_API_URL:
        return 2, "HM_ASYNC_API_URL is not set — nothing to connect to. See .env.example."

    owned = client is None
    client = client or _client_from(settings)
    try:
        return _register_node(client, profiler or get_profiler(), settings)
    finally:
        if owned:
            client.close()


# ============================================================
# bench submit — validate, redact-check, POST one bundle by hand
# ============================================================
#
# The seam `run_bench_quick`'s `submit_fn` calls into (see below) and the
# manual `bench submit <bundle.json>` command are the SAME function: whether
# a bundle came from this run of `bench quick` or an old one sitting on disk,
# submitting it goes through identical validation, redaction, and spool-on-
# outage handling (bench.submit_bundle_file).


def run_bench_submit(
    settings: Settings, bundle_path: str, *, client: ApiClient | None = None
) -> tuple[int, str]:
    """`bench submit <bundle.json>` — validate, redact-check, and POST one bundle.

    Requires BENCH_OPTIN — the same single gate that governs every other
    bench-related upload — so this command cannot send data the operator has
    not consented to share, even when invoked by hand.
    """
    if not settings.BENCH_OPTIN:
        return 2, "not opted in — run `bench opt-in` first; nothing was sent."
    if not settings.HM_ASYNC_API_URL:
        return 2, "HM_ASYNC_API_URL is not set — nothing to connect to. See .env.example."

    owned = client is None
    client = client or _client_from(settings)
    bench_spool = Spool(settings.BENCH_SPOOL_PATH)
    try:
        result = submit_bundle_file(client, bench_spool, bundle_path)
    finally:
        bench_spool.close()
        if owned:
            client.close()

    # A spooled outcome is a normal, expected result of the offline-resilience
    # design (the bundle is durably queued, not lost) — not an operator error.
    return (0 if (result.ok or result.spooled) else 1), result.message


def _bench_submit_fn(bundle_path: str, settings: Settings) -> tuple[int, str]:
    """The `submit_fn` seam `run_bench_quick` hands a fresh bundle off to."""
    return run_bench_submit(settings, bundle_path)


# ============================================================
# bench quick — run the ~25-minute suite through the operator's eb
# ============================================================
#
# Foreground, output NOT captured: stdout/stderr are inherited so the suite's
# own progress prints straight to the operator's terminal, live, the same as
# if they had typed `eb quick` themselves. The 45-minute timeout is a
# backstop against a wedged run, not a target, and is injectable for tests.

BENCH_QUICK_TIMEOUT_S = 45 * 60.0
BENCH_PREFLIGHT_TIMEOUT_S = 10.0


def _bench_utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _bench_install_hint(cmd: str) -> str:
    return (
        f"`{cmd}` was not found on this box. Install energy-bench (see its README "
        f"for the current install method) so `{cmd}` runs, or point ENERGY_BENCH_CMD "
        "at the right command."
    )


def _bench_cmd_available(cmd: str) -> bool:
    """Preflight: can `cmd --help` even be exec'd?

    The exit status is not checked — a `--help` that itself exits non-zero is
    still evidence the binary runs. Only "cannot be executed at all" (missing
    binary, not executable, hangs past the preflight timeout) counts as absent.
    """
    try:
        subprocess.run([cmd, "--help"], capture_output=True, timeout=BENCH_PREFLIGHT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return True


def _bench_bundle_path(
    bundle_dir: str | os.PathLike[str], *, now_fn: Callable[[], datetime]
) -> Path:
    ts = now_fn().strftime("%Y%m%dT%H%M%SZ")
    return Path(bundle_dir) / f"bundle-{ts}.json"


def run_bench_quick(
    settings: Settings,
    *,
    submit_fn: Callable[[str, Settings], tuple[int, str]] | None = None,
    now_fn: Callable[[], datetime] | None = None,
    timeout_s: float = BENCH_QUICK_TIMEOUT_S,
) -> tuple[int, str]:
    """Run `eb quick --share-out <bundle>` in the foreground; hand off on opt-in.

    `submit_fn` is the seam a later story wires to the real submission client
    (POST /api/v1/bench/submissions, spool-on-failure). Until that exists, an
    opted-in operator sees the bundle path and a plain note that nothing is
    wired up yet, rather than a pointer to a command that does not exist.
    """
    cmd = settings.ENERGY_BENCH_CMD
    if not _bench_cmd_available(cmd):
        return 2, _bench_install_hint(cmd)

    bundle_dir = Path(settings.BENCH_BUNDLE_DIR)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = _bench_bundle_path(bundle_dir, now_fn=now_fn or _bench_utcnow)

    try:
        proc = subprocess.run([cmd, "quick", "--share-out", str(bundle_path)], timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return 1, f"{cmd} quick did not finish within {int(timeout_s)}s; no bundle was written"
    except OSError as exc:
        return 1, f"failed to run {cmd} quick: {exc}"

    if proc.returncode != 0:
        return 1, f"{cmd} quick exited {proc.returncode}; no bundle was written"
    if not bundle_path.exists():
        return 1, f"{cmd} quick exited 0 but {bundle_path} was not written"

    message = f"bundle written to {bundle_path}"
    if not settings.BENCH_OPTIN:
        return 0, message
    if submit_fn is None:
        return 0, f"{message}\nbench_optin is set, but no submitter is wired up on this build yet."
    sub_code, sub_message = submit_fn(str(bundle_path), settings)
    return sub_code, f"{message}\n{sub_message}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        # No explicit `prog`: two console scripts point here
        # (`async-energy-controller`, plus the legacy `hm-async-controller`
        # alias), so let argparse take the name the user actually typed rather
        # than hardcoding one and printing usage for a command they did not run.
        description="Execute the Async Energy optimizer's schedule on this box.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single executor tick and exit (useful for cron / smoke tests).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Seconds between schedule polls in the run loop (default: 30).",
    )
    parser.add_argument(
        "--job-catalog",
        default=None,
        metavar="PATH",
        help=f"Path to the local job catalog JSON (default: ${JOB_CATALOG_ENV} or {DEFAULT_JOB_CATALOG}).",
    )
    parser.add_argument(
        "--watch-catalog",
        action="store_true",
        help=(
            "Re-read the job catalog when it changes on disk, instead of loading it "
            "once at start-up. Use when something registers workflows while the "
            "controller is running, so a new entry runs without a restart."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Validate the setup and exit: catalog path and every entry, login, the "
            "current schedule, and which workflows match between them. Exits non-zero "
            "if anything is wrong. This is the real smoke test — prefer it to --once."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )

    # `register` is a subcommand rather than another flag because it takes a dozen
    # of its own. Subparsers stay OPTIONAL, so every existing invocation
    # (`--once`, `--poll-interval 30`, a bare run) parses exactly as before.
    sub = parser.add_subparsers(dest="subcommand")
    reg = sub.add_parser(
        "register",
        help="Create a workflow on the API and add it to the local job catalog.",
        description=(
            "Register a workload and write its catalog entry in one step, so the "
            "server's workflow id is never copy-pasted by hand."
        ),
    )
    reg.add_argument("--name", required=True, help="Human-readable workload name.")
    reg.add_argument(
        "--framework",
        default="command",
        choices=list_frameworks(),
        help="Which adapter runs it locally (default: command).",
    )
    reg.add_argument(
        "--command",
        help="The command this box runs, e.g. 'docker exec box python job.py --years 1'.",
    )
    reg.add_argument("--model", help="Model name, for the ollama / openai frameworks.")
    reg.add_argument("--prompt", help="Inline prompt text.")
    reg.add_argument("--prompt-file", help="Path to a file holding the prompt.")
    reg.add_argument("--base-url", help="Override the framework's local endpoint.")
    reg.add_argument("--cwd", help="Working directory for a command job.")
    reg.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help=(
            "Hard limit on one run. Omit and the controller bounds the job by the "
            "time left in its placement window, which is usually what you want."
        ),
    )
    reg.add_argument(
        "--est-duration",
        type=float,
        metavar="SECONDS",
        help="Rough runtime, so the first plan has something to place. Measured runs replace it.",
    )
    # Human strings the API resolves in your account's timezone. An unparseable
    # one is refused at registration with a message naming the field, so a typo
    # surfaces immediately rather than as a job that quietly never fits.
    reg.add_argument("--deadline", help="When it must be done, e.g. 'by 7am' or '2026-08-20T09:00'.")
    reg.add_argument("--earliest-start", help="Not before this, e.g. '20:00'.")
    reg.add_argument(
        "--recurrence",
        choices=RECURRENCES,
        help="How often it repeats (default: none, a one-shot workload).",
    )
    reg.add_argument(
        "--disabled",
        action="store_true",
        help="Register it without scheduling it yet. Enable later via the API or dashboard.",
    )
    reg.add_argument(
        "--share-request",
        action="store_true",
        help=(
            "Also send the request payload (your command line) to the API. Off by "
            "default: what this box runs stays on this box."
        ),
    )
    reg.add_argument(
        "--nameplate-watts",
        type=float,
        help="Rated draw of the hardware, used to predict cost before any run is measured.",
    )
    # Bench-prior hint fields (the api's migration 008): optional classification
    # metadata the operator types explicitly, unlocking bench-prior cold-start
    # estimates for this workflow. No BENCH_OPTIN gate — distinct from the
    # hardware fingerprint `bench register-node` sends.
    reg.add_argument(
        "--bench-gpu-class",
        help="GPU class hint for bench-prior cold-start estimates, e.g. 'rtx4090'.",
    )
    reg.add_argument(
        "--bench-model-size-class",
        help="Model-size class hint, e.g. '7b' or '70b'.",
    )
    reg.add_argument(
        "--bench-quant",
        help="Quantization hint, e.g. 'int4' or 'fp16'.",
    )
    # SUPPRESS, not a default value: a subparser writes its defaults into the SAME
    # namespace and would otherwise silently overwrite the parent's — so
    # `--job-catalog x.json register …` would quietly write to jobs.json instead.
    # With SUPPRESS the attribute is only set when the flag is actually typed.
    reg.add_argument(
        "--job-catalog",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help=f"Catalog to write to (default: ${JOB_CATALOG_ENV} or {DEFAULT_JOB_CATALOG}).",
    )
    reg.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the catalog entry that would be written; call no API and touch no file.",
    )
    reg.add_argument("--log-level", default=argparse.SUPPRESS, help="Logging level (default: INFO).")

    # `bench` gathers opt-in/opt-out/register-node/quick/submit.
    bench = sub.add_parser(
        "bench",
        help="Opt in/out of contributing benchmark data to Async Energy.",
        description=(
            "Contribute measured benchmark data to Async Energy's shared model. "
            "Off by default — `bench opt-in` prints exactly what is shared "
            "before turning it on. No interactive prompt, safe to run headless."
        ),
    )
    bench_sub = bench.add_subparsers(dest="bench_subcommand")
    bench_sub.add_parser(
        "opt-in",
        help="Opt in to sharing benchmark data. Prints what is shared first.",
    )
    bench_sub.add_parser("opt-out", help="Opt out of sharing benchmark data.")
    bench_sub.add_parser(
        "register-node",
        help=(
            "Send this box's hardware-class fingerprint (GPU/CPU/RAM, no "
            "identifiers) so the server can give it bench-prior cold-start "
            "estimates. Requires prior `bench opt-in`."
        ),
    )
    bench_sub.add_parser(
        "quick",
        help="Run the ~25-minute energy-bench quick suite and write a bundle.",
    )
    bench_submit = bench_sub.add_parser(
        "submit",
        help="Manually submit a bench bundle file (requires prior `bench opt-in`).",
    )
    bench_submit.add_argument(
        "bundle_path",
        metavar="BUNDLE_JSON",
        help="Path to a bundle JSON file, e.g. one written by `bench quick`.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Console entrypoint. Returns a process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = Settings()

    if getattr(args, "subcommand", None) == "register":
        code, message = run_register(settings, args)
        print(message)
        return code

    if getattr(args, "subcommand", None) == "bench":
        bench_sub = getattr(args, "bench_subcommand", None)
        if bench_sub == "quick":
            code, message = run_bench_quick(settings, submit_fn=_bench_submit_fn)
        elif bench_sub == "submit":
            code, message = run_bench_submit(settings, args.bundle_path)
        elif bench_sub == "register-node":
            code, message = run_bench_register_node(settings)
        else:
            code, message = run_bench(args)
        print(message)
        return code

    if args.check:
        # --check reports a missing API URL as one row among the others rather
        # than bailing early: the catalog rows above it are still worth printing,
        # and they are often where the actual mistake is.
        code, text = run_check(settings, job_catalog_path=args.job_catalog)
        print(text)
        return code

    if not settings.HM_ASYNC_API_URL:
        logger.error("HM_ASYNC_API_URL is not set — nothing to connect to. See.env.example.")
        return 2

    executor = build_executor(
        settings,
        job_catalog_path=args.job_catalog,
        watch_catalog=args.watch_catalog,
    )
    catalog_path = resolve_catalog_path(settings, args.job_catalog)
    logger.info("job catalog: %s", catalog_path)
    if args.watch_catalog:
        logger.info("watching the job catalog for changes; new entries run without a restart")

    # Best-effort login; ApiClient auto-(re)authenticates on the first 401 too, so a
    # transient failure here is not fatal — the loop retries on the next tick.
    login = executor.client.login()
    if not login.ok:
        logger.warning("initial login failed (%s); the loop will retry on demand", login.error)

    # One logger for both shapes, so a single tick and the daemon's Nth tick print
    # the same line and can be compared directly.
    log_tick = TickLogger(executor.job_source)
    try:
        if args.once:
            log_tick(executor.tick())
        else:
            logger.info("starting executor loop (poll every %ss); Ctrl-C to stop", args.poll_interval)
            executor.run_forever(poll_interval_s=args.poll_interval, on_tick=log_tick)
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down")
    finally:
        executor.close()
        executor.client.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
