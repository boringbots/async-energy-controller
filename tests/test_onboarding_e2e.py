"""
The onboarding path end to end (US-ONB-08), mocked at both ends.

Every other bench test exercises one seam. This file walks the whole operator
path in the order an operator walks it, so the ORDER is the assertion:

    bench opt-in  ->  bench quick (a stub `eb`)  ->  auto-submit (mocked API)
                  ->  API unreachable, bundle spools
                  ->  API back, the daemon's own tick flushes it

Both ends are fakes, and neither is a monkeypatched internal: `eb` is a real
tiny executable on PATH-by-absolute-path (so the preflight, the `--share-out`
plumbing, and the foreground exec all run for real), and the API is the shared
`FakeApiServer` behind `httpx.MockTransport` (so the real request, auth,
retry, and spool code runs for real). The only thing stubbed inside the
package is the profiler, which would otherwise probe a GPU that CI does not
have.

The flush deliberately goes through `cli.build_executor(...).tick()` rather
than calling `drain_bench_spool` directly: "retries on the existing flush
cadence" is a claim about the daemon, and calling the drain by hand would not
test it.
"""

from __future__ import annotations

import json

import pytest

from hmasync_controller import cli
from hmasync_controller.apiclient import ApiClient
from hmasync_controller.config import Settings
from hmasync_controller.profiler import NullProfiler
from hmasync_controller.spool import Spool

# --- the two fakes ---------------------------------------------------------

_EB_STUB = """#!/bin/sh
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
  cat "__PAYLOAD__" > "$out"
  exit 0
fi
exit 1
"""

# Settings names pydantic-settings reads from the real environment IN PREFERENCE
# to the `.env` file under test. Cleared so a developer box with any of these
# exported cannot silently take over the file this test is asserting about.
_ENV_KEYS = (
    "HM_ASYNC_API_URL", "HM_ASYNC_EMAIL", "HM_ASYNC_PASSWORD", "CONTROLLER_ID",
    "HM_ASYNC_JOB_CATALOG", "SPOOL_PATH", "BENCH_OPTIN", "BENCH_BUNDLE_DIR",
    "ENERGY_BENCH_CMD", "BENCH_SPOOL_PATH", "NODE_SALT_PATH", "APPLY_POWER_CAP",
)


def _valid_bundle(**overrides) -> dict:
    """A schema-valid bundle shaped like something `eb quick` would really emit."""
    bundle = {
        "schema_version": "1",
        "generated_at": "2026-08-22T00:00:00Z",
        "nodes": [
            {
                "node_hash": "0" * 64,
                "cpu_model": "AMD Ryzen 9 5950X 16-Core Processor",
                "ram_gb": 64,
            }
        ],
        "runs": [
            {
                "node_hash": "0" * 64,
                "created_at": "2026-08-22T00:00:00Z",
                "repeat_index": 0,
                "schema_version": "3",
                "engine": "ollama",
                "engine_version": "0.5.7",
                "model": "llama3.1:8b",
                "quantization": "Q4_K_M",
                "measurement_tier": "C",
                "run_duration_s": 41.2,
                "joules_per_token": 1.83,
                "total_joules_gpu": 4210.0,
                "energy_source": "nvml_counter",
                "mean_tokens_per_second": 55.1,
                "pooled_tokens_per_second": 54.4,
                "total_completion_tokens": 2300,
                "streaming_used": True,
                "peak_gpu_w": 318.0,
                "mean_gpu_w": 271.5,
                "temperature": 0.0,
                "max_tokens": 512,
                "seed": 1234,
                "is_canary": False,
                "has_vision_tower": False,
                "n_items": 20,
                "n_correct": 17,
            }
        ],
        "grades": [
            {
                "config_key": "ollama/llama3.1:8b/Q4_K_M",
                "model": "llama3.1:8b",
                "engine": "ollama",
                "n_runs": 1,
            }
        ],
        "load_profiles": [
            {
                "model": "llama3.1:8b",
                "engine": "ollama",
                "load_time_s": 3.4,
                "vram_after_load_mib": 6100.0,
                "loaded_idle_mean_w": 34.0,
            }
        ],
    }
    bundle.update(overrides)
    return bundle


@pytest.fixture(autouse=True)
def _stub_profiler(monkeypatch):
    """build_executor probes the GPU; this box (and CI) may not have one."""
    monkeypatch.setattr(cli, "get_profiler", lambda: NullProfiler())


@pytest.fixture
def onboard(tmp_path, monkeypatch, fake_api):
    """A box mid-onboarding: a real `.env`, a stub `eb`, a mocked API.

    Returns a small handle rather than a tuple so each step below reads as the
    operator action it stands for.
    """

    def _setup(bundle: dict | None = None):
        for key in _ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

        payload = tmp_path / "eb-payload.json"
        payload.write_text(json.dumps(bundle if bundle is not None else _valid_bundle()))
        eb = tmp_path / "eb-stub"
        eb.write_text(_EB_STUB.replace("__PAYLOAD__", str(payload)))
        eb.chmod(0o755)

        # A plausible operator `.env` — the four documented knobs plus the one
        # pointer at this box's energy-bench. Note what is NOT here: BENCH_OPTIN.
        # `bench opt-in` is what puts it in the file, which is exactly the
        # round trip (CLI writes .env -> Settings reads it back) under test.
        env = tmp_path / ".env"
        env.write_text(
            "# operator notes stay put\n"
            "HM_ASYNC_API_URL=https://api.hm-async.test\n"
            "HM_ASYNC_EMAIL=owner@example.com\n"
            "HM_ASYNC_PASSWORD=s3cret\n"
            "CONTROLLER_ID=box-e2e\n"
            f"ENERGY_BENCH_CMD={eb}\n"
        )

        # cwd-relative paths (.env, bench_bundles/, the two spool files) all
        # land in tmp_path, the same way they land in an operator's install dir.
        monkeypatch.chdir(tmp_path)

        # Every ApiClient the CLI builds — `bench submit`'s and the executor's —
        # is routed at the transport layer, so the client's own auth/retry/
        # error-mapping code is the code under test.
        real_client = ApiClient

        def _fake_transport_client(*args, **kwargs):
            kwargs["http_client"] = fake_api.client()
            return real_client(*args, **kwargs)

        monkeypatch.setattr(cli, "ApiClient", _fake_transport_client)
        return env

    return _setup


def _bench_spool_count(tmp_path) -> int:
    spool = Spool(str(tmp_path / "hmasync_bench_spool.db"))
    try:
        return spool.count()
    finally:
        spool.close()


# --- the path ---------------------------------------------------------------


def test_optin_quick_submit_spool_then_daemon_flush(tmp_path, monkeypatch, capsys, fake_api, onboard):
    env = onboard()

    # 1. opt in. The consent text is printed before the flag is written, and
    #    the flag lands in .env without disturbing what was already there.
    assert cli.main(["bench", "opt-in"]) == 0
    out = capsys.readouterr().out
    assert "Opting in shares, per benchmark submission" in out
    assert "salted local hash" in out
    env_text = env.read_text()
    assert "BENCH_OPTIN=true" in env_text
    assert "# operator notes stay put" in env_text
    assert "HM_ASYNC_API_URL=https://api.hm-async.test" in env_text

    # The flag round-trips: a freshly-constructed Settings reads it back off disk.
    assert Settings().BENCH_OPTIN is True

    # 2. the API goes away before the suite finishes — the outage case is the
    #    one the whole store-and-forward design exists for.
    fake_api.go_down()

    # 3. run the suite. Opted in, so `bench quick` hands the bundle straight to
    #    the submitter; the submitter cannot reach the API, so it spools.
    assert cli.main(["bench", "quick"]) == 0  # spooled is not an operator error
    out = capsys.readouterr().out
    assert "bundle written to" in out
    assert "spooled" in out

    bundles = sorted((tmp_path / "bench_bundles").glob("bundle-*.json"))
    assert len(bundles) == 1
    written = json.loads(bundles[0].read_text())
    assert written == _valid_bundle()

    assert fake_api.bench_submissions == []  # nothing reached the wire
    assert _bench_spool_count(tmp_path) == 1  # ...and nothing was lost

    # 4. the API comes back. No re-run, no manual retry: the daemon's ordinary
    #    tick flushes the bench spool on the same reconnect trigger that
    #    flushes the run-report spool.
    fake_api.go_up()
    executor = cli.build_executor(Settings(), job_catalog_path=tmp_path / "jobs.json")
    try:
        result = executor.tick()
    finally:
        executor.close()
        executor.client.close()

    assert result.reachable
    assert result.drained == 1
    assert len(fake_api.bench_submissions) == 1
    assert fake_api.bench_submissions[0] == _valid_bundle()
    assert _bench_spool_count(tmp_path) == 0


def test_opted_out_box_runs_the_suite_and_sends_nothing(tmp_path, capsys, fake_api, onboard):
    """The consent gate, end to end: same suite, same reachable API, no opt-in."""
    onboard()

    assert cli.main(["bench", "quick"]) == 0
    out = capsys.readouterr().out
    assert "bundle written to" in out

    # The bundle exists locally — running the suite is never gated, only sharing is.
    assert len(sorted((tmp_path / "bench_bundles").glob("bundle-*.json"))) == 1

    assert fake_api.requests == []  # not one request, not even a login
    assert not (tmp_path / "hmasync_bench_spool.db").exists()

    # And the daemon does not open the bench spool or drain it either.
    executor = cli.build_executor(Settings(), job_catalog_path=tmp_path / "jobs.json")
    try:
        assert executor._extra_drain is None
    finally:
        executor.close()
        executor.client.close()


def test_a_leaky_bundle_never_reaches_the_wire_or_the_spool(tmp_path, capsys, fake_api, onboard):
    """Redaction, end to end: a denylisted key refuses locally, with the API up.

    Not spooled either — a retry would resubmit the identical leak, so the
    refusal has to be terminal rather than deferred.
    """
    leaky = _valid_bundle()
    leaky["nodes"][0]["gpu_uuid"] = "GPU-deadbeef-0000-1111-2222-333344445555"
    onboard(bundle=leaky)

    assert cli.main(["bench", "opt-in"]) == 0
    capsys.readouterr()

    assert cli.main(["bench", "quick"]) == 1
    out = capsys.readouterr().out
    assert "denylisted" in out
    assert "gpu_uuid" in out

    assert fake_api.bench_submissions == []
    assert _bench_spool_count(tmp_path) == 0


# --- release prep -----------------------------------------------------------


def test_package_version_matches_pyproject():
    """The version lives in two files; nothing but this notices when they drift.

    Deliberately not pinned to a literal — pinning would make every future bump
    a two-file edit plus a test edit. What matters is that `pip show` and
    `hmasync_controller.__version__` never disagree about which build this is.
    """
    import re
    from pathlib import Path

    from hmasync_controller import __version__

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text())
    assert declared, "no version = \"...\" line in pyproject.toml"
    assert declared.group(1) == __version__
