"""
Tests for the console entrypoint wiring (hmasync_controller/cli.py).

No live network: build_executor is side-effect-free (no login), the job catalog
is a tmp JSON file, and get_profiler is stubbed to a NullProfiler so the test
never shells out to nvidia-smi.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from hmasync_controller import cli
from hmasync_controller.apiclient import ApiClient
from hmasync_controller.config import Settings
from hmasync_controller.executor import JobDef, ScheduleExecutor
from hmasync_controller.powercap import PowerCapManager
from hmasync_controller.profiler import NullProfiler, NVMLProfiler


@pytest.fixture(autouse=True)
def _stub_profiler(monkeypatch):
    """Keep build_executor hermetic — never probe real GPU/smi in unit tests."""
    monkeypatch.setattr(cli, "get_profiler", lambda: NullProfiler())


def _settings(tmp_path, **overrides) -> Settings:
    kwargs = dict(HM_ASYNC_API_URL="", SPOOL_PATH=str(tmp_path / "spool.db"))
    kwargs.update(overrides)
    return Settings(**kwargs)


# --- load_job_catalog -----------------------------------------------------

def test_load_job_catalog_missing_file_is_empty(tmp_path):
    assert cli.load_job_catalog(tmp_path / "nope.json") == {}


def test_load_job_catalog_reads_object_entries(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({
        "wf-1": {"framework": "command", "request": {"command": ["true"]}},
        "wf-2": {"framework": "ollama", "request": {"model": "llama3"}},
    }))
    catalog = cli.load_job_catalog(path)
    assert set(catalog) == {"wf-1", "wf-2"}
    assert catalog["wf-1"]["framework"] == "command"


def test_load_job_catalog_drops_non_object_entries(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"wf-1": {"framework": "command"}, "wf-bad": "nope"}))
    catalog = cli.load_job_catalog(path)
    assert set(catalog) == {"wf-1"}


def test_load_job_catalog_malformed_json_is_empty(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text("{not valid json")
    assert cli.load_job_catalog(path) == {}


def test_load_job_catalog_non_dict_top_level_is_empty(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(["wf-1", "wf-2"]))
    assert cli.load_job_catalog(path) == {}


# --- build_executor -------------------------------------------------------

def test_build_executor_constructs_without_network(tmp_path):
    executor = cli.build_executor(_settings(tmp_path), job_catalog_path=tmp_path / "absent.json")
    assert isinstance(executor, ScheduleExecutor)
    # No catalog → unknown workflow resolves to None (executor skips it cleanly).
    assert executor._resolve_job("wf-x") is None
    executor.client.close()


def test_build_executor_wires_catalog_as_job_source(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({
        "wf-1": {
            "framework": "command",
            "request": {"command": ["echo", "hi"]},
            "deadline": "2026-07-11T07:00:00-04:00",
        }
    }))
    executor = cli.build_executor(_settings(tmp_path), job_catalog_path=path)
    job = executor._resolve_job("wf-1")
    assert isinstance(job, JobDef)
    assert job.framework == "command"
    assert job.request == {"command": ["echo", "hi"]}
    # The human-friendly deadline string was parsed into a tz-aware datetime.
    assert job.deadline is not None and job.deadline.tzinfo is not None
    executor.client.close()


def test_build_executor_uses_resolved_controller_id(tmp_path):
    executor = cli.build_executor(_settings(tmp_path, CONTROLLER_ID="box-42"), job_catalog_path=tmp_path / "x.json")
    assert executor.controller_id == "box-42"


def test_build_executor_no_extra_drain_when_not_opted_in(tmp_path):
    executor = cli.build_executor(_settings(tmp_path, BENCH_OPTIN=False), job_catalog_path=tmp_path / "x.json")
    assert executor._extra_drain is None
    executor.client.close()


def test_build_executor_wires_extra_drain_when_opted_in(tmp_path):
    executor = cli.build_executor(
        _settings(tmp_path, BENCH_OPTIN=True, BENCH_SPOOL_PATH=str(tmp_path / "bench_spool.db")),
        job_catalog_path=tmp_path / "x.json",
    )
    assert callable(executor._extra_drain)
    assert executor._extra_drain() == 0  # empty bench spool, no network needed
    executor.client.close()


# --- build_executor: power cap wiring (US-ONB-06) --------------------------
#
# The autouse `_stub_profiler` fixture above returns a NullProfiler, so by
# default (no GPU) build_executor must never wire a power cap even when
# APPLY_POWER_CAP is set — there is nothing to cap.

class _FakeNvmlForCli:
    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"handle-{index}"


def test_build_executor_no_power_cap_when_not_opted_in(tmp_path):
    executor = cli.build_executor(
        _settings(tmp_path, APPLY_POWER_CAP=False), job_catalog_path=tmp_path / "x.json"
    )
    assert executor.power_cap is None
    executor.client.close()


def test_build_executor_no_power_cap_without_a_gpu(tmp_path):
    """Opted in, but this box's profiler is Null (no NVML GPU) — nothing to cap."""
    executor = cli.build_executor(
        _settings(tmp_path, APPLY_POWER_CAP=True), job_catalog_path=tmp_path / "x.json"
    )
    assert executor.power_cap is None
    executor.client.close()


def test_build_executor_wires_power_cap_when_opted_in_with_a_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "get_profiler", lambda: NVMLProfiler(nvml=_FakeNvmlForCli()))
    executor = cli.build_executor(
        _settings(tmp_path, APPLY_POWER_CAP=True, NODE_SALT_PATH=str(tmp_path / "salt")),
        job_catalog_path=tmp_path / "x.json",
    )
    assert isinstance(executor.power_cap, PowerCapManager)
    executor.client.close()


# --- main -----------------------------------------------------------------

def test_main_without_api_url_returns_2(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    assert cli.main([]) == 2


def test_main_once_runs_a_single_tick(tmp_path, monkeypatch, fake_api):
    """--once runs exactly one tick against the fake API and returns 0."""
    fake_api.set_schedule(1)  # a schedule with no placements → idle-but-reachable
    settings = _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test",
                         HM_ASYNC_EMAIL="owner@example.com", HM_ASYNC_PASSWORD="s3cret")
    monkeypatch.setattr(cli, "Settings", lambda: settings)

    # Wire the executor's client to the fake transport by patching ApiClient's
    # default http client via build_executor → then swap the transport.
    real_build = cli.build_executor

    def build_with_fake(s, **kw):
        ex = real_build(s, **kw)
        ex.client._http = fake_api.client()
        return ex

    monkeypatch.setattr(cli, "build_executor", build_with_fake)

    assert cli.main(["--once"]) == 0
    # The single tick pulled the schedule (login + at least one GET /schedule).
    paths = [r.url.path for r in fake_api.requests]
    assert "/auth/login" in paths
    assert "/api/v1/schedule" in paths


# --- CatalogWatcher (--watch-catalog) -------------------------------------
#
# The point of the watcher is that something else (an agent, a deploy script)
# can append a workflow to jobs.json while the controller is running and have it
# execute without a restart. mtime is the change signal, so these tests set it
# explicitly rather than sleeping — a same-second rewrite is exactly the case a
# naive implementation gets wrong.

def _write_catalog(path, mapping, *, mtime=None):
    path.write_text(json.dumps(mapping))
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_watcher_picks_up_a_workflow_added_after_start(tmp_path):
    path = tmp_path / "jobs.json"
    _write_catalog(path, {"wf-1": {"framework": "command", "request": {"command": ["true"]}}},
                   mtime=1_000_000)
    watcher = cli.CatalogWatcher(path)
    assert watcher("wf-1") is not None
    assert watcher("wf-2") is None, "not registered locally yet"

    # An agent appends a second workflow while the controller runs.
    _write_catalog(path, {
        "wf-1": {"framework": "command", "request": {"command": ["true"]}},
        "wf-2": {"framework": "ollama", "request": {"model": "llama3"}},
    }, mtime=1_000_060)

    assert watcher("wf-2") == {"framework": "ollama", "request": {"model": "llama3"}}


def test_watcher_does_not_reread_when_mtime_is_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "jobs.json"
    _write_catalog(path, {"wf-1": {"framework": "command", "request": {}}}, mtime=1_000_000)
    watcher = cli.CatalogWatcher(path)

    reads = {"n": 0}
    real = cli._read_catalog

    def counting(p):
        reads["n"] += 1
        return real(p)

    monkeypatch.setattr(cli, "_read_catalog", counting)
    for _ in range(5):
        watcher("wf-1")
    assert reads["n"] == 0, "unchanged mtime must not trigger a re-read every lookup"


def test_watcher_keeps_last_good_catalog_when_file_becomes_malformed(tmp_path):
    path = tmp_path / "jobs.json"
    _write_catalog(path, {"wf-1": {"framework": "command", "request": {"command": ["true"]}}},
                   mtime=1_000_000)
    watcher = cli.CatalogWatcher(path)
    assert watcher("wf-1") is not None

    # A bad edit must NOT silently stop every job.
    path.write_text("{ this is not json")
    os.utime(path, (1_000_060, 1_000_060))

    assert watcher("wf-1") is not None, "a typo must not wipe the running catalog"


def test_watcher_keeps_last_good_catalog_when_file_is_deleted(tmp_path):
    path = tmp_path / "jobs.json"
    _write_catalog(path, {"wf-1": {"framework": "command", "request": {"command": ["true"]}}},
                   mtime=1_000_000)
    watcher = cli.CatalogWatcher(path)
    assert watcher("wf-1") is not None

    path.unlink()
    assert watcher("wf-1") is not None, "a deleted file must not wipe the running catalog"


def test_watcher_adopts_a_validly_empty_catalog(tmp_path):
    """Empty-but-valid is a real intention and must be honored — this is the
    case the keep-last-good rule must NOT swallow."""
    path = tmp_path / "jobs.json"
    _write_catalog(path, {"wf-1": {"framework": "command", "request": {}}}, mtime=1_000_000)
    watcher = cli.CatalogWatcher(path)
    assert watcher("wf-1") is not None

    _write_catalog(path, {}, mtime=1_000_060)
    assert watcher("wf-1") is None, "an intentionally emptied catalog must be adopted"


def test_watcher_on_missing_file_matches_load_once_semantics(tmp_path):
    """Starting with no catalog is a warning, not an error (same as load-once)."""
    watcher = cli.CatalogWatcher(tmp_path / "nope.json")
    assert watcher("wf-1") is None


def test_watcher_recovers_when_a_broken_file_is_fixed(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text("{ broken")
    os.utime(path, (1_000_000, 1_000_000))
    watcher = cli.CatalogWatcher(path)
    assert watcher("wf-1") is None

    _write_catalog(path, {"wf-1": {"framework": "command", "request": {}}}, mtime=1_000_060)
    assert watcher("wf-1") is not None, "fixing the file must recover without a restart"


# --- build_executor wiring for the flag -----------------------------------

def test_build_executor_default_passes_a_plain_mapping(tmp_path):
    """Default behavior is unchanged — existing deployments are unaffected."""
    path = tmp_path / "jobs.json"
    _write_catalog(path, {"wf-1": {"framework": "command", "request": {}}})
    executor = cli.build_executor(_settings(tmp_path), job_catalog_path=path)
    # A later edit is NOT seen without --watch-catalog.
    _write_catalog(path, {
        "wf-1": {"framework": "command", "request": {}},
        "wf-2": {"framework": "command", "request": {}},
    }, mtime=1_000_060)
    assert executor._resolve_job("wf-2") is None
    executor.client.close()


def test_build_executor_watch_catalog_sees_later_edits(tmp_path):
    path = tmp_path / "jobs.json"
    _write_catalog(path, {"wf-1": {"framework": "command", "request": {}}}, mtime=1_000_000)
    executor = cli.build_executor(_settings(tmp_path), job_catalog_path=path, watch_catalog=True)
    assert executor._resolve_job("wf-2") is None

    _write_catalog(path, {
        "wf-1": {"framework": "command", "request": {}},
        "wf-2": {"framework": "command", "request": {"command": ["true"]}},
    }, mtime=1_000_060)

    job = executor._resolve_job("wf-2")
    assert isinstance(job, JobDef) and job.framework == "command"
    executor.client.close()


def test_watch_catalog_flag_parses_and_defaults_off():
    assert cli._parse_args([]).watch_catalog is False
    assert cli._parse_args(["--watch-catalog"]).watch_catalog is True


# ============================================================
# Catalog path resolution (#1)
# ============================================================
#
# HM_ASYNC_JOB_CATALOG was read straight off os.environ, so a line in `.env` —
# where the quickstart puts every other knob — was silently dropped. Credentials
# in the same file DID work, so the controller logged in, polled happily, and
# executed nothing: the failure presented as "the optimizer isn't scheduling
# anything" rather than as a config error.


def test_catalog_path_comes_from_settings(tmp_path):
    settings = _settings(tmp_path, HM_ASYNC_JOB_CATALOG="/srv/from-dotenv.json")
    assert cli.resolve_catalog_path(settings) == "/srv/from-dotenv.json"


def test_explicit_flag_beats_settings(tmp_path):
    settings = _settings(tmp_path, HM_ASYNC_JOB_CATALOG="/srv/from-dotenv.json")
    assert cli.resolve_catalog_path(settings, "/cli/flag.json") == "/cli/flag.json"


def test_catalog_path_falls_back_to_the_default(tmp_path):
    settings = _settings(tmp_path, HM_ASYNC_JOB_CATALOG="")
    assert cli.resolve_catalog_path(settings) == cli.DEFAULT_JOB_CATALOG


def test_build_executor_loads_the_catalog_named_in_settings(tmp_path):
    """The end-to-end shape of the bug: a `.env`-sourced path must load."""
    path = tmp_path / "from-dotenv.json"
    path.write_text(json.dumps({"wf-1": {"framework": "command", "request": {"command": ["true"]}}}))

    executor = cli.build_executor(_settings(tmp_path, HM_ASYNC_JOB_CATALOG=str(path)))

    assert executor._resolve_job("wf-1") is not None
    executor.client.close()


# ============================================================
# TickLogger (#2)
# ============================================================


class _Result:
    """A minimal stand-in for TickResult."""

    def __init__(self, **kw):
        self.version = kw.get("version", 1)
        self.mode = kw.get("mode", "normal")
        self.reachable = kw.get("reachable", True)
        self.placements = kw.get("placements", 0)
        self.pending = kw.get("pending", 0)
        self.next_start = kw.get("next_start")
        self.outcomes = kw.get("outcomes", [])
        self.drained = kw.get("drained", 0)


def test_tick_line_reports_catalog_and_placement_counts(caplog):
    log = cli.TickLogger({"wf-1": {}, "wf-2": {}})
    with caplog.at_level("INFO"):
        log(_Result(placements=2, pending=1))
    line = caplog.text
    assert "catalog=2" in line and "placements=2" in line and "pending=1" in line


def test_tick_line_counts_a_watcher_like_a_dict(tmp_path, caplog):
    path = tmp_path / "jobs.json"
    _write_catalog(path, {"wf-1": {"framework": "command", "request": {}}}, mtime=1_000_000)
    log = cli.TickLogger(cli.CatalogWatcher(path))
    with caplog.at_level("INFO"):
        log(_Result())
    assert "catalog=1" in caplog.text


def test_empty_catalog_warning_repeats_on_a_slow_cadence(caplog):
    """Warned once at startup, it scrolls out of `journalctl -n 20` in minutes."""
    log = cli.TickLogger({}, warn_every=5)
    with caplog.at_level("INFO"):
        for _ in range(11):
            log(_Result())
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 3, "ticks 1, 6 and 11"
    assert "catalog is empty" in warnings[0].message


def test_a_populated_catalog_never_warns(caplog):
    log = cli.TickLogger({"wf-1": {}}, warn_every=2)
    with caplog.at_level("INFO"):
        for _ in range(6):
            log(_Result())
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_tick_line_shows_the_next_start(caplog):
    from datetime import datetime, timezone

    log = cli.TickLogger({"wf-1": {}})
    with caplog.at_level("INFO"):
        log(_Result(next_start=datetime(2026, 8, 12, 0, 10, tzinfo=timezone.utc)))
    assert "next=" in caplog.text and "next=-" not in caplog.text


def test_tick_line_shows_a_dash_when_nothing_is_next(caplog):
    log = cli.TickLogger({"wf-1": {}})
    with caplog.at_level("INFO"):
        log(_Result())
    assert "next=-" in caplog.text


# ============================================================
# --check (#3)
# ============================================================
#
# `--once` prints `outcomes=0` whether the catalog is empty, the schedule is
# empty, a workflow id is mistyped, or it is simply 16:00 and the window opens at
# 20:00. Those need different fixes, so they need different output.


def _catalog(tmp_path, mapping) -> str:
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(mapping))
    return str(path)


def test_check_reports_a_missing_catalog(tmp_path):
    code, text = cli.run_check(_settings(tmp_path), job_catalog_path=tmp_path / "gone.json")
    assert code != 0
    assert "does not exist" in text


def test_check_reports_an_empty_catalog(tmp_path):
    code, text = cli.run_check(_settings(tmp_path), job_catalog_path=_catalog(tmp_path, {}))
    assert code != 0
    assert "0 jobs" in text and "nothing can run" in text


def test_check_names_an_unknown_framework(tmp_path):
    path = _catalog(tmp_path, {"wf-1": {"framework": "nope", "request": {}}})
    _code, text = cli.run_check(_settings(tmp_path), job_catalog_path=path)
    assert "unknown framework" in text


def test_check_names_a_request_missing_a_required_field(tmp_path):
    path = _catalog(tmp_path, {"wf-1": {"framework": "ollama", "request": {}}})
    _code, text = cli.run_check(_settings(tmp_path), job_catalog_path=path)
    assert "missing required field 'model'" in text


def test_check_without_an_api_url_is_a_config_error(tmp_path):
    path = _catalog(tmp_path, {"wf-1": {"framework": "command", "request": {"command": ["true"]}}})
    code, text = cli.run_check(_settings(tmp_path, HM_ASYNC_API_URL=""), job_catalog_path=path)
    assert code == cli.CHECK_CONFIG_ERROR
    assert "HM_ASYNC_API_URL is not set" in text


def _check_against(fake_api, tmp_path, monkeypatch, catalog, **schedule_fields):
    """Run run_check with its ApiClient wired to the fake transport."""
    settings = _settings(
        tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test",
        HM_ASYNC_EMAIL="owner@example.com", HM_ASYNC_PASSWORD="s3cret",
    )
    if schedule_fields:
        fake_api.set_schedule(4, **schedule_fields)

    real = cli._client_from

    def _wired(s):
        client = real(s)
        client._http = fake_api.client()
        return client

    monkeypatch.setattr(cli, "_client_from", _wired)
    return cli.run_check(settings, job_catalog_path=_catalog(tmp_path, catalog))


def _placement(wid, start, end, **extra):
    p = {"workflow_id": wid, "start": start, "end": end,
         "predicted_wh": 8.3, "feasible": True, "reason": None}
    p.update(extra)
    return p


def test_check_matches_catalog_entries_against_the_schedule(fake_api, tmp_path, monkeypatch):
    code, text = _check_against(
        fake_api, tmp_path, monkeypatch,
        {"wf-1": {"framework": "command", "request": {"command": ["true"]},
                  "name": "stonks-fed-update"}},
        placements=[_placement("wf-1", "2026-08-12T00:10:00+00:00",
                               "2026-08-12T00:12:00+00:00")],
    )
    assert code == cli.CHECK_OK
    assert "matched" in text and "stonks-fed-update" in text
    assert "auth" in text and "owner@example.com" in text
    assert "version 4" in text


def test_check_flags_a_scheduled_workflow_with_no_catalog_entry(fake_api, tmp_path, monkeypatch):
    """The optimizer plans it; this box silently skips it. Always a fault."""
    code, text = _check_against(
        fake_api, tmp_path, monkeypatch,
        {"wf-known": {"framework": "command", "request": {"command": ["true"]}}},
        placements=[_placement("wf-typo", "2026-08-12T00:10:00+00:00",
                               "2026-08-12T00:12:00+00:00")],
    )
    assert code == cli.CHECK_PROBLEMS
    assert "unmatched" in text and "will skip it" in text


def test_check_reports_an_orphan_without_failing(fake_api, tmp_path, monkeypatch):
    """A weekly job not in tonight's plan is normal; crying wolf would kill the check."""
    code, text = _check_against(
        fake_api, tmp_path, monkeypatch,
        {"wf-weekly": {"framework": "command", "request": {"command": ["true"]}}},
        placements=[],
    )
    assert code == cli.CHECK_OK
    assert "orphaned" in text


def test_check_reports_no_schedule_published(fake_api, tmp_path, monkeypatch):
    code, text = _check_against(
        fake_api, tmp_path, monkeypatch,
        {"wf-1": {"framework": "command", "request": {"command": ["true"]}}},
    )
    assert code == cli.CHECK_PROBLEMS
    assert "none published yet" in text


def test_check_reports_a_failed_login(fake_api, tmp_path, monkeypatch):
    fake_api.go_down()
    code, text = _check_against(
        fake_api, tmp_path, monkeypatch,
        {"wf-1": {"framework": "command", "request": {"command": ["true"]}}},
    )
    assert code == cli.CHECK_PROBLEMS
    assert "login failed" in text


def test_check_flag_parses_and_defaults_off():
    assert cli._parse_args([]).check is False
    assert cli._parse_args(["--check"]).check is True


# ============================================================
# register (#4)
# ============================================================
#
# Creating a workflow meant hand-rolled curl, then a UUID copy-pasted into
# jobs.json — the most error-prone step in the setup, and the only one with no
# tooling. A mistyped id is simply a workflow this box never runs.


def _register_args(**overrides):
    argv = ["register", "--name", overrides.pop("name", "nightly")]
    for key, value in overrides.items():
        if value is True:
            argv.append(f"--{key.replace('_', '-')}")
        elif value is not None:
            argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return cli._parse_args(argv)


def test_register_creates_the_workflow_and_writes_the_catalog(tmp_path, make_client, fake_api):
    fake_api.workflows_response = {"id": "29812d27-aaaa-bbbb-cccc-000000000001"}
    path = tmp_path / "jobs.json"
    args = _register_args(
        name="stonks-fed-extend",
        command="docker exec box python job.py --years 1",
        deadline="by 7am", earliest_start="20:00", recurrence="daily",
        nameplate_watts=250, est_duration=700, job_catalog=str(path),
    )
    settings = _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test")

    code, message = cli.run_register(settings, args, client=make_client())

    assert code == 0, message
    written = json.loads(path.read_text())
    entry = written["29812d27-aaaa-bbbb-cccc-000000000001"]
    assert entry["framework"] == "command"
    # Stored as argv, which is what sidesteps shell-quoting entirely.
    assert entry["request"]["command"] == [
        "docker", "exec", "box", "python", "job.py", "--years", "1"
    ]
    assert entry["name"] == "stonks-fed-extend"
    # The human deadline goes to the API, NOT into the catalog: the catalog's
    # optional `deadline` must be tz-aware ISO, and "by 7am" parses to None there
    # — a field that looks set and silently does nothing.
    assert "deadline" not in entry
    assert "earliest_start" not in entry


def test_register_sends_the_scheduling_fields(tmp_path, make_client, fake_api):
    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(
        name="nightly", command="true", deadline="by 7am", recurrence="daily",
        nameplate_watts=250, est_duration=700, job_catalog=str(tmp_path / "j.json"),
    )
    cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )

    body = json.loads(
        [r for r in fake_api.requests if r.url.path == "/api/v1/workflows"][0].content
    )
    assert body["name"] == "nightly"
    assert body["deadline"] == "by 7am"
    assert body["recurrence"] == "daily"
    assert body["nameplate_watts"] == 250
    assert body["est_duration_s"] == 700
    # The local command line is not uploaded unless asked for.
    assert "request" not in body


def test_register_preserves_existing_catalog_entries(tmp_path, make_client, fake_api):
    fake_api.workflows_response = {"id": "wf-2"}
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({
        "_comment": "a header block a human wrote",
        "wf-1": {"framework": "command", "request": {"command": ["true"]}},
    }))
    args = _register_args(name="second", command="true", job_catalog=str(path))

    cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )

    written = json.loads(path.read_text())
    assert set(written) == {"_comment", "wf-1", "wf-2"}
    assert written["_comment"] == "a header block a human wrote"


def test_register_refuses_to_clobber_an_unparseable_catalog(tmp_path, make_client, fake_api):
    """A trailing comma is not a reason to replace an operator's file."""
    fake_api.workflows_response = {"id": "wf-2"}
    path = tmp_path / "jobs.json"
    path.write_text("{ this is not json")
    args = _register_args(name="second", command="true", job_catalog=str(path))

    code, message = cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )

    assert code == 1
    assert "could not be parsed" in message
    assert "Add it by hand" in message, "the workflow DOES exist server-side now"
    assert path.read_text() == "{ this is not json"


def test_register_reports_an_api_refusal(tmp_path, make_client, fake_api):
    fake_api.workflows_status = 422
    fake_api.workflows_response = {"detail": "deadline could not be parsed"}
    path = tmp_path / "jobs.json"
    args = _register_args(name="bad", command="true", job_catalog=str(path))

    code, message = cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )

    assert code == 1
    assert "deadline could not be parsed" in message
    assert not path.exists(), "nothing local should change when the API refused"


def test_register_dry_run_touches_nothing(tmp_path, make_client, fake_api):
    path = tmp_path / "jobs.json"
    args = _register_args(name="nightly", command="true", job_catalog=str(path), dry_run=True)

    code, message = cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )

    assert code == 0
    assert "dry run" in message
    assert not path.exists()
    assert [r for r in fake_api.requests if r.url.path == "/api/v1/workflows"] == []


def test_register_needs_something_to_run(tmp_path, make_client):
    args = _register_args(name="empty", job_catalog=str(tmp_path / "j.json"))
    code, message = cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )
    assert code == 2
    assert "--command" in message


def test_register_handles_a_response_with_no_id(tmp_path, make_client, fake_api):
    fake_api.workflows_response = {"status": "created"}
    path = tmp_path / "jobs.json"
    args = _register_args(name="nightly", command="true", job_catalog=str(path))

    code, message = cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )

    assert code == 1
    assert "returned no workflow id" in message
    assert not path.exists()


def test_register_finds_the_id_however_it_is_nested():
    assert cli._workflow_id_from({"id": "a"}) == "a"
    assert cli._workflow_id_from({"workflow_id": "b"}) == "b"
    assert cli._workflow_id_from({"workflow": {"id": "c"}}) == "c"
    assert cli._workflow_id_from({"status": "ok"}) is None
    assert cli._workflow_id_from(None) is None


def test_register_writes_atomically(tmp_path):
    """A --watch-catalog daemon must never observe a half-written file."""
    path = tmp_path / "jobs.json"
    cli.write_catalog_entry(path, "wf-1", {"framework": "command", "request": {}})
    assert json.loads(path.read_text())["wf-1"]["framework"] == "command"
    assert not (tmp_path / "jobs.json.tmp").exists()


def test_run_flags_still_parse_with_the_subcommand_present():
    """Adding `register` must not change any existing invocation."""
    assert cli._parse_args(["--once"]).once is True
    assert cli._parse_args(["--poll-interval", "15"]).poll_interval == 15.0
    assert cli._parse_args([]).subcommand is None
    assert cli._parse_args(["register", "--name", "x"]).subcommand == "register"


def test_a_parent_flag_survives_the_register_subcommand():
    """argparse writes subparser defaults into the same namespace; SUPPRESS stops
    them silently overwriting a value the operator typed before the subcommand."""
    args = cli._parse_args(
        ["--job-catalog", "top.json", "--log-level", "DEBUG", "register", "--name", "x"]
    )
    assert args.job_catalog == "top.json"
    assert args.log_level == "DEBUG"


def test_register_own_job_catalog_flag_still_wins():
    args = cli._parse_args(
        ["--job-catalog", "top.json", "register", "--name", "x", "--job-catalog", "sub.json"]
    )
    assert args.job_catalog == "sub.json"


def test_register_does_not_upload_the_command_by_default(tmp_path, make_client, fake_api):
    """What this box runs is a local concern; register sends constraints only."""
    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(
        name="nightly", command="docker exec box secret-script.sh --token abc",
        job_catalog=str(tmp_path / "j.json"),
    )
    cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )
    body = fake_api.created_workflows[0]
    assert "request" not in body
    assert "secret-script.sh" not in json.dumps(body)


def test_share_request_opts_in(tmp_path, make_client, fake_api):
    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(
        name="nightly", command="true", share_request=True,
        job_catalog=str(tmp_path / "j.json"),
    )
    cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )
    assert fake_api.created_workflows[0]["request"] == {"command": ["true"]}


def test_recurrence_is_restricted_to_what_the_api_accepts():
    """A typo should fail at the command line, where the message can name the
    valid values, rather than as a validation error from the API."""
    assert cli._parse_args(["register", "--name", "x", "--recurrence", "daily"]).recurrence == "daily"
    with pytest.raises(SystemExit):
        cli._parse_args(["register", "--name", "x", "--recurrence", "nightly"])


def test_disabled_registers_without_scheduling(tmp_path, make_client, fake_api):
    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(
        name="nightly", command="true", disabled=True, job_catalog=str(tmp_path / "j.json"),
    )
    cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )
    assert fake_api.created_workflows[0]["enabled"] is False


def test_enabled_is_omitted_when_not_disabled(tmp_path, make_client, fake_api):
    """Let the server's own default (True) apply rather than restating it."""
    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(name="nightly", command="true", job_catalog=str(tmp_path / "j.json"))
    cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )
    assert "enabled" not in fake_api.created_workflows[0]


def test_register_success_points_at_bench_optin(tmp_path, make_client, fake_api):
    """A no-interactive-prompt pointer, printed once, after a successful register."""
    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(name="nightly", command="true", job_catalog=str(tmp_path / "j.json"))

    _code, message = cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )

    assert "bench opt-in" in message


def test_register_sends_bench_prior_hint_flags(tmp_path, make_client, fake_api):
    """US-ONB-05: --bench-gpu-class/--bench-model-size-class/--bench-quant pass
    through to the workflow payload (the api's bench-prior hint fields)."""
    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(
        name="nightly", command="true", job_catalog=str(tmp_path / "j.json"),
        bench_gpu_class="rtx4090", bench_model_size_class="7b", bench_quant="int4",
    )
    cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )
    body = fake_api.created_workflows[0]
    assert body["bench_gpu_class"] == "rtx4090"
    assert body["bench_model_size_class"] == "7b"
    assert body["bench_quant"] == "int4"


def test_register_omits_bench_prior_hints_when_not_passed(tmp_path, make_client, fake_api):
    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(name="nightly", command="true", job_catalog=str(tmp_path / "j.json"))
    cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )
    body = fake_api.created_workflows[0]
    assert "bench_gpu_class" not in body
    assert "bench_model_size_class" not in body
    assert "bench_quant" not in body


def test_register_sends_node_fingerprint_when_opted_in(tmp_path, make_client, fake_api):
    """US-ONB-05: a successful register ALSO upserts the node fingerprint, but
    only when the operator has opted into bench (the fingerprint is one of the
    things BENCH_CONSENT_TEXT names as shared)."""
    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(name="nightly", command="true", job_catalog=str(tmp_path / "j.json"))
    settings = _settings(
        tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test",
        BENCH_OPTIN=True, NODE_SALT_PATH=str(tmp_path / "salt"),
    )

    code, message = cli.run_register(settings, args, client=make_client())

    assert code == 0
    assert len(fake_api.registered_nodes) == 1
    assert "node_hash" in fake_api.registered_nodes[0]
    assert "registered node" in message


def test_register_does_not_send_node_fingerprint_without_optin(tmp_path, make_client, fake_api):
    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(name="nightly", command="true", job_catalog=str(tmp_path / "j.json"))
    settings = _settings(
        tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test",
        BENCH_OPTIN=False, NODE_SALT_PATH=str(tmp_path / "salt"),
    )

    code, message = cli.run_register(settings, args, client=make_client())

    assert code == 0
    assert fake_api.registered_nodes == []
    assert not (tmp_path / "salt").exists()


def test_register_node_fingerprint_failure_does_not_fail_the_register(tmp_path, make_client, fake_api):
    """Registering the workflow is the primary action; a node-fingerprint
    hiccup is secondary and must never undo it."""
    fake_api.workflows_response = {"id": "wf-new"}
    fake_api.register_node_status = 500
    fake_api.register_node_response = {"detail": "server error"}
    args = _register_args(name="nightly", command="true", job_catalog=str(tmp_path / "j.json"))
    settings = _settings(
        tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test",
        BENCH_OPTIN=True, NODE_SALT_PATH=str(tmp_path / "salt"),
    )

    code, message = cli.run_register(settings, args, client=make_client())

    # code == 0: the workflow registration itself still succeeded.
    assert code == 0
    assert "node fingerprint not sent" in message
    assert (tmp_path / "j.json").exists()


def test_register_node_fingerprint_has_no_denylisted_keys(tmp_path, make_client, fake_api):
    """The explicit denylist assertion the PRD's acceptance criteria calls
    for, exercised through the real register path (not a hand-built dict)."""
    from hmasync_controller.bench import denylisted_keys

    fake_api.workflows_response = {"id": "wf-new"}
    args = _register_args(name="nightly", command="true", job_catalog=str(tmp_path / "j.json"))
    settings = _settings(
        tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test",
        BENCH_OPTIN=True, NODE_SALT_PATH=str(tmp_path / "salt"),
    )

    cli.run_register(settings, args, client=make_client())

    assert len(fake_api.registered_nodes) == 1
    assert denylisted_keys(fake_api.registered_nodes[0]) == []


def test_register_surfaces_a_400_naming_the_bad_field(tmp_path, make_client, fake_api):
    """The API refuses an unparseable deadline; that message is the useful one."""
    fake_api.workflows_status = 400
    fake_api.workflows_response = {"detail": "deadline: could not parse 'whenever'"}
    path = tmp_path / "jobs.json"
    args = _register_args(name="bad", command="true", deadline="whenever", job_catalog=str(path))

    code, message = cli.run_register(
        _settings(tmp_path, HM_ASYNC_API_URL="https://api.hm-async.test"),
        args, client=make_client(),
    )

    assert code == 1
    assert "deadline: could not parse" in message
    assert not path.exists()


# ============================================================
# bench opt-in / opt-out (US-ONB-02)
# ============================================================
#
# Explicit, headless-safe consent: no interactive (y/n) prompt (a systemd box
# has no TTY to answer one), and opting in always prints what is shared before
# the flag is persisted.


def test_bench_subcommand_parses():
    args = cli._parse_args(["bench", "opt-in"])
    assert args.subcommand == "bench"
    assert args.bench_subcommand == "opt-in"

    args = cli._parse_args(["bench", "opt-out"])
    assert args.bench_subcommand == "opt-out"


def test_bench_optin_persists_the_flag_and_prints_consent(tmp_path):
    env = tmp_path / ".env"
    args = cli._parse_args(["bench", "opt-in"])

    code, message = cli.run_bench(args, env_path=env)

    assert code == 0
    assert "BENCH_OPTIN=true" in env.read_text()
    assert "hardware fingerprint" in message
    assert "Opted in" in message


def test_bench_optout_persists_the_flag(tmp_path):
    env = tmp_path / ".env"
    env.write_text("BENCH_OPTIN=true\n")
    args = cli._parse_args(["bench", "opt-out"])

    code, message = cli.run_bench(args, env_path=env)

    assert code == 0
    assert "BENCH_OPTIN=false" in env.read_text()
    assert "Opted out" in message


def test_bench_with_no_subcommand_is_a_usage_error(tmp_path):
    args = cli._parse_args(["bench"])
    code, message = cli.run_bench(args, env_path=tmp_path / ".env")
    assert code == 2
    assert "opt-in" in message and "opt-out" in message
    assert not (tmp_path / ".env").exists()


def test_main_dispatches_bench_optin(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    assert cli.main(["bench", "opt-in"]) == 0
    assert "BENCH_OPTIN=true" in (tmp_path / ".env").read_text()
    assert "hardware fingerprint" in capsys.readouterr().out


# ============================================================
# bench quick (US-ONB-03)
# ============================================================
#
# `eb` is stubbed with a real (tiny) executable script rather than a
# monkeypatched subprocess.run, so the preflight check, argument plumbing,
# and foreground/timeout behavior all run for real — just against a fake
# binary instead of the real energy-bench, and with no network involved.

_EB_STUB_SUCCESS = """#!/bin/sh
if [ "$1" = "--help" ]; then
  echo "eb (stub)"
  exit 0
fi
if [ "$1" = "quick" ]; then
  shift
  out=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --share-out) out="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  echo "stub: running quick suite"
  printf '{"tier": "C", "stub": true}' > "$out"
  exit 0
fi
exit 1
"""

_EB_STUB_QUICK_FAILS = """#!/bin/sh
if [ "$1" = "--help" ]; then
  exit 0
fi
if [ "$1" = "quick" ]; then
  echo boom 1>&2
  exit 1
fi
exit 1
"""

_EB_STUB_QUICK_HANGS = """#!/bin/sh
if [ "$1" = "--help" ]; then
  exit 0
fi
if [ "$1" = "quick" ]; then
  sleep 5
  exit 0
fi
exit 1
"""


def _write_eb_stub(tmp_path, body: str, name: str = "eb-stub") -> str:
    script = tmp_path / name
    script.write_text(body)
    script.chmod(0o755)
    return str(script)


def test_bench_quick_subcommand_parses():
    args = cli._parse_args(["bench", "quick"])
    assert args.subcommand == "bench"
    assert args.bench_subcommand == "quick"


def test_bench_quick_preflight_missing_binary_prints_install_hint(tmp_path):
    settings = _settings(tmp_path, ENERGY_BENCH_CMD=str(tmp_path / "no-such-eb"))

    code, message = cli.run_bench_quick(settings)

    assert code == 2
    assert "not found" in message
    assert "ENERGY_BENCH_CMD" in message


def test_bench_quick_writes_bundle_on_success(tmp_path):
    eb = _write_eb_stub(tmp_path, _EB_STUB_SUCCESS)
    bundle_dir = tmp_path / "bundles"
    settings = _settings(
        tmp_path, ENERGY_BENCH_CMD=eb, BENCH_BUNDLE_DIR=str(bundle_dir)
    )
    fixed_now = datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc)

    code, message = cli.run_bench_quick(settings, now_fn=lambda: fixed_now)

    assert code == 0
    bundle_path = bundle_dir / "bundle-20260822T120000Z.json"
    assert bundle_path.exists()
    assert json.loads(bundle_path.read_text()) == {"tier": "C", "stub": True}
    assert str(bundle_path) in message


def test_bench_quick_no_optin_does_not_hand_off(tmp_path):
    eb = _write_eb_stub(tmp_path, _EB_STUB_SUCCESS)
    settings = _settings(
        tmp_path,
        ENERGY_BENCH_CMD=eb,
        BENCH_BUNDLE_DIR=str(tmp_path / "bundles"),
        BENCH_OPTIN=False,
    )
    calls = []

    code, message = cli.run_bench_quick(
        settings, submit_fn=lambda path, s: (calls.append((path, s)), (0, "submitted"))[1]
    )

    assert code == 0
    assert calls == []
    assert "submitted" not in message


def test_bench_quick_optin_hands_off_to_submitter(tmp_path):
    eb = _write_eb_stub(tmp_path, _EB_STUB_SUCCESS)
    settings = _settings(
        tmp_path,
        ENERGY_BENCH_CMD=eb,
        BENCH_BUNDLE_DIR=str(tmp_path / "bundles"),
        BENCH_OPTIN=True,
    )
    calls = []

    def fake_submit(path, s):
        calls.append((path, s))
        return 0, "submitted ok"

    code, message = cli.run_bench_quick(settings, submit_fn=fake_submit)

    assert code == 0
    assert len(calls) == 1
    assert calls[0][0].endswith(".json")
    assert calls[0][1] is settings
    assert "submitted ok" in message


def test_bench_quick_optin_without_submitter_notes_it_honestly(tmp_path):
    eb = _write_eb_stub(tmp_path, _EB_STUB_SUCCESS)
    settings = _settings(
        tmp_path,
        ENERGY_BENCH_CMD=eb,
        BENCH_BUNDLE_DIR=str(tmp_path / "bundles"),
        BENCH_OPTIN=True,
    )

    code, message = cli.run_bench_quick(settings)

    assert code == 0
    assert "no submitter is wired up" in message


def test_bench_quick_nonzero_exit_is_an_error_and_writes_no_bundle(tmp_path):
    eb = _write_eb_stub(tmp_path, _EB_STUB_QUICK_FAILS)
    bundle_dir = tmp_path / "bundles"
    settings = _settings(tmp_path, ENERGY_BENCH_CMD=eb, BENCH_BUNDLE_DIR=str(bundle_dir))

    code, message = cli.run_bench_quick(settings)

    assert code == 1
    assert "exited 1" in message
    assert not list(bundle_dir.glob("*.json"))


def test_bench_quick_timeout_is_a_clean_error(tmp_path):
    eb = _write_eb_stub(tmp_path, _EB_STUB_QUICK_HANGS)
    settings = _settings(
        tmp_path, ENERGY_BENCH_CMD=eb, BENCH_BUNDLE_DIR=str(tmp_path / "bundles")
    )

    code, message = cli.run_bench_quick(settings, timeout_s=0.3)

    assert code == 1
    assert "did not finish" in message


def test_main_dispatches_bench_quick(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        cli,
        "run_bench_quick",
        lambda settings, submit_fn=None: (0, "bundle written to x.json"),
    )

    assert cli.main(["bench", "quick"]) == 0
    assert "bundle written" in capsys.readouterr().out


def test_main_wires_bench_submit_fn_into_bench_quick(tmp_path, monkeypatch, capsys):
    """`main()` hands `run_bench_quick` a real submit_fn (bound to run_bench_submit)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    seen = {}

    def fake_run_bench_quick(settings, submit_fn=None):
        seen["submit_fn"] = submit_fn
        return 0, "bundle written to x.json"

    monkeypatch.setattr(cli, "run_bench_quick", fake_run_bench_quick)
    assert cli.main(["bench", "quick"]) == 0
    assert seen["submit_fn"] is cli._bench_submit_fn


# ============================================================
# bench submit (US-ONB-04)
# ============================================================


def test_bench_submit_subcommand_parses():
    args = cli._parse_args(["bench", "submit", "bundle.json"])
    assert args.subcommand == "bench"
    assert args.bench_subcommand == "submit"
    assert args.bundle_path == "bundle.json"


def test_run_bench_submit_requires_optin(tmp_path):
    settings = _settings(tmp_path, BENCH_OPTIN=False)
    code, message = cli.run_bench_submit(settings, str(tmp_path / "bundle.json"))
    assert code == 2
    assert "opted in" in message


def test_run_bench_submit_requires_api_url(tmp_path):
    settings = _settings(tmp_path, BENCH_OPTIN=True, HM_ASYNC_API_URL="")
    code, message = cli.run_bench_submit(settings, str(tmp_path / "bundle.json"))
    assert code == 2
    assert "HM_ASYNC_API_URL" in message


def test_run_bench_submit_success(tmp_path, fake_api, monkeypatch):
    bundle = {
        "schema_version": "1", "generated_at": "2026-08-22T00:00:00Z",
        "nodes": [], "runs": [], "grades": [], "load_profiles": [],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    settings = _settings(
        tmp_path,
        BENCH_OPTIN=True,
        HM_ASYNC_API_URL="https://api.hm-async.test",
        BENCH_SPOOL_PATH=str(tmp_path / "bench_spool.db"),
    )
    client = ApiClient(
        base_url="https://api.hm-async.test", email="owner@example.com",
        password="s3cret", http_client=fake_api.client(),
    )

    code, message = cli.run_bench_submit(settings, str(bundle_path), client=client)

    assert code == 0
    assert "submitted" in message
    assert len(fake_api.bench_submissions) == 1


def test_run_bench_submit_spooled_on_outage_is_not_a_hard_failure(tmp_path, fake_api):
    bundle = {
        "schema_version": "1", "generated_at": "2026-08-22T00:00:00Z",
        "nodes": [], "runs": [], "grades": [], "load_profiles": [],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    fake_api.go_down()
    settings = _settings(
        tmp_path,
        BENCH_OPTIN=True,
        HM_ASYNC_API_URL="https://api.hm-async.test",
        BENCH_SPOOL_PATH=str(tmp_path / "bench_spool.db"),
    )
    client = ApiClient(
        base_url="https://api.hm-async.test", email="owner@example.com",
        password="s3cret", http_client=fake_api.client(),
    )

    code, message = cli.run_bench_submit(settings, str(bundle_path), client=client)

    assert code == 0
    assert "spooled" in message


def test_main_dispatches_bench_submit(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    seen = {}

    def fake_run_bench_submit(settings, bundle_path):
        seen["bundle_path"] = bundle_path
        return 0, "submitted bundle.json"

    monkeypatch.setattr(cli, "run_bench_submit", fake_run_bench_submit)
    assert cli.main(["bench", "submit", "bundle.json"]) == 0
    assert seen["bundle_path"] == "bundle.json"
    assert "submitted" in capsys.readouterr().out


# ============================================================
# bench register-node (US-ONB-05)
# ============================================================
#
# Sends this box's hardware-class fingerprint (GPU/CPU/RAM + a salted
# node_hash — never an identifier) so the server can give it bench-prior
# cold-start estimates. Gated on BENCH_OPTIN, same as every other bench
# upload; NullProfiler is passed explicitly so these never probe a real GPU.


def test_bench_register_node_subcommand_parses():
    args = cli._parse_args(["bench", "register-node"])
    assert args.subcommand == "bench"
    assert args.bench_subcommand == "register-node"


def test_run_bench_register_node_requires_optin(tmp_path):
    settings = _settings(tmp_path, BENCH_OPTIN=False)
    code, message = cli.run_bench_register_node(settings)
    assert code == 2
    assert "opted in" in message


def test_run_bench_register_node_requires_api_url(tmp_path):
    settings = _settings(tmp_path, BENCH_OPTIN=True, HM_ASYNC_API_URL="")
    code, message = cli.run_bench_register_node(settings)
    assert code == 2
    assert "HM_ASYNC_API_URL" in message


def test_run_bench_register_node_success(tmp_path, make_client, fake_api):
    settings = _settings(
        tmp_path, BENCH_OPTIN=True, HM_ASYNC_API_URL="https://api.hm-async.test",
        NODE_SALT_PATH=str(tmp_path / "salt"),
    )

    code, message = cli.run_bench_register_node(
        settings, client=make_client(), profiler=NullProfiler()
    )

    assert code == 0
    assert "registered node" in message
    assert len(fake_api.registered_nodes) == 1
    assert "node_hash" in fake_api.registered_nodes[0]
    assert (tmp_path / "salt").exists()


def test_run_bench_register_node_no_gpu_degrades_gracefully(tmp_path, make_client, fake_api):
    settings = _settings(
        tmp_path, BENCH_OPTIN=True, HM_ASYNC_API_URL="https://api.hm-async.test",
        NODE_SALT_PATH=str(tmp_path / "salt"),
    )

    code, _message = cli.run_bench_register_node(
        settings, client=make_client(), profiler=NullProfiler()
    )

    assert code == 0
    payload = fake_api.registered_nodes[0]
    assert "gpu_name" not in payload
    assert "driver_version" not in payload
    assert "vram_gb" not in payload


def test_run_bench_register_node_api_refusal_is_reported(tmp_path, make_client, fake_api):
    fake_api.register_node_status = 500
    fake_api.register_node_response = {"detail": "server error"}
    settings = _settings(
        tmp_path, BENCH_OPTIN=True, HM_ASYNC_API_URL="https://api.hm-async.test",
        NODE_SALT_PATH=str(tmp_path / "salt"),
    )

    code, message = cli.run_bench_register_node(
        settings, client=make_client(), profiler=NullProfiler()
    )

    assert code == 1
    assert "could not register node" in message


def test_main_dispatches_bench_register_node(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        cli, "run_bench_register_node", lambda settings: (0, "registered node abc123")
    )
    assert cli.main(["bench", "register-node"]) == 0
    assert "registered node abc123" in capsys.readouterr().out
