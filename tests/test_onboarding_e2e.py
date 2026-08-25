"""
The onboarding path end to end (US-ONB-08; updated for US-MERGE-04's
in-process `bench quick`), mocked at both ends.

Every other bench test exercises one seam. This file walks the whole operator
path in the order an operator walks it, so the ORDER is the assertion:

    bench opt-in  ->  bench quick (the REAL in-process suite)  ->  auto-submit (mocked API)
                  ->  API unreachable, bundle spools
                  ->  API back, the daemon's own tick flushes it

Before US-MERGE-04, `bench quick` shelled out to a stubbed `eb` executable.
That subprocess seam is gone: `cli.run_bench_quick` now calls
`hmasync_controller.bench.quick.run_quick_suite` directly, which drives the
REAL engine-detection / model-resolution / telemetry / compute_metrics /
artifact-write / bundle-build code path. The only things mocked in THIS file
are the two things a real box can't fake in CI: the engine on the wire
(`OllamaAdapter`/`VLLMClient`, module-level in `bench.quick`'s own
namespace) and NVML (`sys.modules["pynvml"]`, same pattern
`tests/test_bench_quick.py` uses, itself mirroring energy-bench's
`tests/unit/test_quick.py`). The optimizer API is still the shared
`FakeApiServer` behind `httpx.MockTransport`, so the real request, auth,
retry, and spool code runs for real. The only thing stubbed beyond the
engine/NVML boundary is the profiler used for node fingerprinting, which
would otherwise probe a GPU CI does not have.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from hmasync_controller import cli
from hmasync_controller.apiclient import ApiClient
from hmasync_controller.bench.metrics.models import InferenceResult
from hmasync_controller.bench.submission import validate_bundle
from hmasync_controller.bench.tasks.base import TaskItem
from hmasync_controller.config import Settings
from hmasync_controller.profiler import NullProfiler
from hmasync_controller.spool import Spool

# --- pynvml mock, module-level -- installed before any
# LocalNvmlSampler._ensure_handle() call in this file. Mirrors
# tests/test_bench_quick.py's mock (itself ported from energy-bench's
# tests/unit/test_quick.py).

mock_pynvml = MagicMock()
mock_pynvml.NVML_TEMPERATURE_GPU = 0
mock_pynvml.NVML_CLOCK_SM = 0
mock_pynvml.NVML_CLOCK_MEM = 1

mock_handle = MagicMock()
mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 200_000  # 200W in mW
mock_util = MagicMock()
mock_util.gpu = 75.0
mock_util.memory = 40.0
mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
mock_mem = MagicMock()
mock_mem.used = 8 * 1024 * 1024
mock_mem.total = 24 * 1024 * 1024
mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 300_000  # 300W
mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 3090"
mock_pynvml.nvmlSystemGetDriverVersion.return_value = "550.90.07"
mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12040  # -> "12.4"
mock_pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (100_000, 350_000)
# SetPowerManagementLimit needs root on most drivers -- simulate that so the
# mini power sweep skips gracefully (exercising the SAME restore-in-finally
# path a successful sweep would, without needing a second telemetry
# fixture). The suite's headline path (3 measured tasks, a bundle written)
# does not depend on the sweep succeeding.
mock_pynvml.nvmlDeviceSetPowerManagementLimit.side_effect = Exception("Insufficient Permissions")

sys.modules["pynvml"] = mock_pynvml


# --- the engine/task boundary -----------------------------------------------


class _FakeOnboardingTask:
    """A tiny scored task standing in for whichever of QUICK_TASKS is
    requested -- `load_task` is patched to always return this, so no real
    HuggingFace dataset fetch happens anywhere in this file."""

    name = "fake_onboarding_task"
    shape = "decode"
    is_canary = False
    revision = None
    default_max_tokens = 64
    stop: list[str] = []

    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        n = min(n_items, 3)  # keep the suite fast; QUICK_TASKS' own n_items don't matter here
        return [TaskItem(item_id=f"t:{i}", prompt=f"prompt {i}", target="42") for i in range(n)]

    def score(self, completion: str, item: TaskItem) -> bool:
        return completion.strip().endswith(item.target)


@pytest.fixture(autouse=True)
def _mock_engine_and_tasks(monkeypatch):
    """Attach-mode Ollama (always ready, always pulled), a fake scored task,
    and a fake OpenAI-compatible client -- the network/dataset boundary
    `run_quick_suite` measures against. No real HTTP request and no real HF
    dataset fetch happens anywhere in this file."""
    ollama_adapter = MagicMock()
    ollama_adapter.ready = AsyncMock(return_value=True)
    ollama_adapter.verify_model_pulled = AsyncMock(return_value="qwen2")
    ollama_adapter.version = AsyncMock(return_value="0.5.7")
    ollama_adapter.launch_args = MagicMock(return_value=[])
    monkeypatch.setattr(
        "hmasync_controller.bench.quick.OllamaAdapter",
        MagicMock(return_value=ollama_adapter),
    )

    async def _fake_chat(*args, **kwargs):
        # A real (tiny) suspension point: `LocalNvmlSampler`'s background
        # sampling task is only ever scheduled by `asyncio.create_task`,
        # never actually run, until SOMETHING in the current task truly
        # yields to the event loop -- a plain AsyncMock's coroutine resolves
        # without one, which would starve the sampler and leave every task
        # run with zero telemetry samples. A genuine `asyncio.sleep` here is
        # what makes this an in-process integration test of the real
        # sampler, not a bypass of it.
        await asyncio.sleep(0.01)
        return (
            InferenceResult(
                request_id="r", prompt_tokens=20, completion_tokens=10,
                ttft_s=0.05, total_s=0.3, tokens_per_second=33.3,
            ),
            "the answer is #### 42",
        )

    fake_client = MagicMock()
    fake_client.chat = AsyncMock(side_effect=_fake_chat)
    monkeypatch.setattr(
        "hmasync_controller.bench.quick.VLLMClient",
        MagicMock(return_value=fake_client),
    )

    monkeypatch.setattr(
        "hmasync_controller.bench.quick.load_task", lambda name: _FakeOnboardingTask()
    )


@pytest.fixture(autouse=True)
def _stub_profiler(monkeypatch):
    """build_executor (and node fingerprinting) probe the GPU; this box (and
    CI) may not have one -- unrelated to the NVML mock above, which only
    covers `bench.sampler.LocalNvmlSampler`'s OWN lazy `import pynvml`."""
    monkeypatch.setattr(cli, "get_profiler", lambda: NullProfiler())


# Settings names pydantic-settings reads from the real environment IN PREFERENCE
# to the `.env` file under test. Cleared so a developer box with any of these
# exported cannot silently take over the file this test is asserting about.
_ENV_KEYS = (
    "HM_ASYNC_API_URL", "HM_ASYNC_EMAIL", "HM_ASYNC_PASSWORD", "CONTROLLER_ID",
    "HM_ASYNC_JOB_CATALOG", "SPOOL_PATH", "BENCH_OPTIN", "BENCH_BUNDLE_DIR",
    "BENCH_DATA_DIR", "BENCH_SPOOL_PATH", "NODE_SALT_PATH", "APPLY_POWER_CAP",
)


@pytest.fixture
def onboard(tmp_path, monkeypatch, fake_api):
    """A box mid-onboarding: a real `.env`, a mocked engine, a mocked API.

    Returns a small handle rather than a tuple so each step below reads as the
    operator action it stands for.
    """

    def _setup():
        for key in _ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

        # A plausible operator `.env` — the four documented knobs. Note what
        # is NOT here: BENCH_OPTIN. `bench opt-in` is what puts it in the
        # file, which is exactly the round trip (CLI writes .env -> Settings
        # reads it back) under test.
        env = tmp_path / ".env"
        env.write_text(
            "# operator notes stay put\n"
            "HM_ASYNC_API_URL=https://api.hm-async.test\n"
            "HM_ASYNC_EMAIL=owner@example.com\n"
            "HM_ASYNC_PASSWORD=s3cret\n"
            "CONTROLLER_ID=box-e2e\n"
        )

        # cwd-relative paths (.env, bench_bundles/, bench_data/, the two
        # spool files) all land in tmp_path, the same way they land in an
        # operator's install dir.
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

    # 3. run the REAL in-process suite (engine + NVML mocked at the boundary,
    #    see module docstring). Opted in, so `bench quick` hands the bundle
    #    straight to the submitter; the submitter cannot reach the API, so
    #    it spools.
    assert cli.main(["bench", "quick"]) == 0  # spooled is not an operator error
    out = capsys.readouterr().out
    assert "bundle written to" in out
    assert "spooled" in out

    bundles = sorted((tmp_path / "bench_bundles").glob("bundle-*.json"))
    assert len(bundles) == 1
    written = json.loads(bundles[0].read_text())
    assert written["schema_version"] == "2"
    assert len(written["runs"]) >= 1
    assert written["runs"][0]["model"] == "Qwen/Qwen3.5-9B"  # the ollama reference model
    # No prompts, commands, or hosts anywhere in what got written.
    text = bundles[0].read_text()
    assert "localhost" not in text
    assert validate_bundle(written) == []  # schema-valid, the real gate submit_bundle_file applies

    # Each measured run's raw trace also landed on disk (bench quick's own
    # artifact folder, distinct from the bundle).
    assert list((tmp_path / "bench_data").iterdir())

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
    assert fake_api.bench_submissions[0] == written
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


def test_a_denylist_violation_refuses_locally_before_anything_is_written(
    tmp_path, capsys, fake_api, onboard, monkeypatch
):
    """Defense in depth, end to end: if `build_bundle`'s own allowlist ever
    let a denylisted field through (it shouldn't -- see bench/bundle.py's
    module docstring; the denylist logic itself is unit-tested directly in
    tests/test_bench_bundle.py), `run_bench_quick` must refuse LOCALLY, with
    the API up, rather than crash or write/submit anything. Not spooled
    either — a retry would resubmit the identical leak, so the refusal has
    to be terminal rather than deferred.
    """
    onboard()

    assert cli.main(["bench", "opt-in"]) == 0
    capsys.readouterr()

    def _raise(*args, **kwargs):
        raise cli.ExportDenylistViolation(
            "Field 'bundle.nodes[0].gpu_uuid' matches the export denylist"
        )

    monkeypatch.setattr(cli, "build_bundle", _raise)

    assert cli.main(["bench", "quick"]) == 1
    out = capsys.readouterr().out
    assert "denylist" in out.lower()
    assert "gpu_uuid" in out

    assert not list((tmp_path / "bench_bundles").glob("*.json"))
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
