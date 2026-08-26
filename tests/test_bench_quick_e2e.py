"""`bench quick`, end to end, on fixtures — the controller-side gate (US-MERGE-06).

The MILESTONE criterion this file answers: *a fixture `bench quick` end to end
with zero network beyond a local engine mock*. So the boundary here is drawn
one notch further out than in any other bench test:

- The engine is a **real HTTP server on 127.0.0.1** (`http.server`, ephemeral
  port), speaking Ollama's `/api/version` + `/api/show` and the
  OpenAI-compatible `/v1/chat/completions` with a genuine SSE stream. The real
  `OllamaAdapter` and the real `VLLMClient` — connection handling, streaming
  parse, TTFT/ITL timing, non-streaming fallback — all run against it over an
  actual socket. `tests/test_onboarding_e2e.py` mocks both of those classes;
  this file does not.
- The datasets are **real parquet files on disk**, written by the fixture and
  handed back through a patched `hf_hub_download`. The tasks' own loaders,
  prompt formatting, few-shot assembly and scorers run unchanged, so the
  accuracy in the bundle is computed, not stubbed.
- **A socket guard asserts the "zero network" half.** Every `connect` /
  `connect_ex` (and `socket.getaddrinfo`, which covers the synchronous
  resolution paths) is checked; anything that is not loopback fails the test
  at the moment it is attempted, and the recorded log is asserted afterwards.
  Without it, "no network" would be a claim about what the mocks happen to
  cover rather than a property of the run. Verified against a live outbound
  request while writing this file: the guard fires on both the IPv4 and IPv6
  happy-eyeballs attempts, surfacing as an `ExceptionGroup` wrapping the
  `AssertionError` because anyio raises it from inside a task group.

NVML is the one thing left mocked (CI has no GPU), installed via
`monkeypatch.setitem(sys.modules, ...)` rather than the module-level
`sys.modules["pynvml"] = ...` that `test_bench_quick.py`/`test_onboarding_e2e.py`
use — a module-level assignment leaks into every later test file in the run,
and this file needs its own NVML behaviour (a sweep that SUCCEEDS) rather
than whichever mock happened to be imported last.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hmasync_controller import cli
from hmasync_controller.bench.submission import validate_bundle
from hmasync_controller.bench.tasks.base import fetch_parquet_rows
from hmasync_controller.config import Settings

# --- the engine on the wire -------------------------------------------------

# Answers chosen so each task's REAL scorer produces a non-trivial accuracy:
# a bundle showing 0.0 or 1.0 everywhere would pass a "did it run" assertion
# while proving nothing about the scoring path.
GSM8K_ANSWER = "Let's work through it. The total comes to 7.\n#### 7"
MMLU_ANSWER = "Answer: B"
IFEVAL_ANSWER = "here is a reply written entirely in lowercase with no punctuation"

PROMPT_TOKENS = 120
COMPLETION_TOKENS = 24


def _completion_for(prompt: str) -> str:
    """Route by what the task's own prompt builder produced."""
    if "#### <answer>" in prompt:
        return GSM8K_ANSWER
    if "multiple choice questions" in prompt:
        return MMLU_ANSWER
    return IFEVAL_ANSWER


class _EngineHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        # Silence the stdlib access log — 175 request lines per run.
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self):  # stdlib handler naming
        self.server.requests.append(("GET", self.path))
        if self.path == "/api/version":
            self._send_json(200, {"version": "0.5.7-fixture"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):  # stdlib handler naming
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(("POST", self.path))

        if self.path == "/api/show":
            self._send_json(200, {"details": {"family": "qwen3"}})
            return
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        prompt = body["messages"][0]["content"]
        text = _completion_for(prompt)
        usage = {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        }

        if not body.get("stream"):
            self.server.non_streaming_requests += 1
            self._send_json(200, {
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": usage,
            })
            return

        # A real SSE body: one chunk per whitespace-delimited piece, then a
        # final chunk carrying finish_reason + usage, then [DONE]. This is what
        # makes `VLLMClient`'s streaming parse (and therefore the run's TTFT
        # and inter-token gaps) real measurements rather than placeholders.
        pieces = [p + " " for p in text.split(" ")]
        pieces[-1] = pieces[-1].rstrip()
        lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': p}, 'finish_reason': None}]})}\n\n"
            for p in pieces
        ]
        lines.append(
            "data: "
            + json.dumps({
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            })
            + "\n\n"
        )
        lines.append("data: [DONE]\n\n")
        self._send(200, "".join(lines).encode(), "text/event-stream")


class _EngineServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _EngineHandler)
        self.requests: list[tuple[str, str]] = []
        self.non_streaming_requests = 0

    @property
    def port(self) -> int:
        return self.server_address[1]


@pytest.fixture(scope="module")
def engine_mock():
    server = _EngineServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- the datasets on disk ---------------------------------------------------

MMLU_SUBJECT_CHOICES = ["first option", "second option", "third option", "fourth option"]


def _fixture_table(repo_id: str, filename: str) -> pa.Table:
    """A parquet fixture with the real column names/types each loader reads."""
    if repo_id == "madrylab/gsm8k-platinum":
        # Alternating gold answers: the mock always answers 7, so exactly half
        # the items are correct and the accuracy in the bundle is 0.5-ish.
        return pa.table({
            "question": [f"Fixture question {i}?" for i in range(40)],
            "answer": [f"Some reasoning.\n#### {7 if i % 2 == 0 else 99}" for i in range(40)],
        })
    if repo_id == "openai/gsm8k":
        return pa.table({
            "question": [f"Few-shot question {i}?" for i in range(8)],
            "answer": [f"Few-shot reasoning.\n#### {i}" for i in range(8)],
        })
    if repo_id == "edinburgh-dawg/mmlu-redux-2.0":
        subject = filename.split("/")[0]
        # Two scorable rows per subject plus one the loader must DROP — the
        # error_type filter is part of what this run exercises.
        return pa.table({
            "question": [f"{subject} q0?", f"{subject} q1?", f"{subject} dropped?"],
            "choices": [MMLU_SUBJECT_CHOICES] * 3,
            "answer": [1, 0, 1],
            "error_type": ["ok", "ok", "wrong_groundtruth"],
        })
    if repo_id == "cais/mmlu":
        return pa.table({
            "question": [f"Dev question {i}?" for i in range(5)],
            "choices": [MMLU_SUBJECT_CHOICES] * 5,
            "answer": [0, 1, 2, 3, 0],
            "subject": ["abstract_algebra"] * 5,
        })
    if repo_id == "google/IFEval":
        # Alternating instructions: the mock's lowercase, comma-free answer
        # satisfies the first and violates the second.
        ids = [
            ["punctuation:no_comma"] if i % 2 == 0 else ["change_case:english_capital"]
            for i in range(30)
        ]
        return pa.table(
            {
                "key": list(range(30)),
                "prompt": [f"Fixture instruction-following prompt {i}." for i in range(30)],
                "instruction_id_list": ids,
                # Every IFEval row carries one kwargs dict per instruction id.
                # These two checkers take none, but parquet has no zero-field
                # struct type, so the real schema's nullable fields stand in.
                "kwargs": [[{"num_words": None}] for _ in range(30)],
            },
            schema=pa.schema([
                ("key", pa.int64()),
                ("prompt", pa.string()),
                ("instruction_id_list", pa.list_(pa.string())),
                ("kwargs", pa.list_(pa.struct([("num_words", pa.int64())]))),
            ]),
        )
    raise AssertionError(f"unexpected dataset fetch: {repo_id}/{filename}")


@pytest.fixture(scope="module")
def dataset_fixtures(run_dir, patch):
    """Serve every `fetch_parquet_rows` call from a locally-written parquet."""
    root = run_dir / "hf-fixtures"
    root.mkdir()
    fetched: list[tuple[str, str, str]] = []

    def _fake_hf_hub_download(*, repo_id, filename, repo_type, revision):
        assert repo_type == "dataset"
        fetched.append((repo_id, filename, revision))
        path = root / f"{repo_id}--{filename}".replace("/", "__")
        if not path.exists():
            pq.write_table(_fixture_table(repo_id, filename), path)
        return str(path)

    patch.setattr("huggingface_hub.hf_hub_download", _fake_hf_hub_download)
    fetch_parquet_rows.cache_clear()
    try:
        yield fetched
    finally:
        fetch_parquet_rows.cache_clear()


# --- NVML ------------------------------------------------------------------


@pytest.fixture(scope="module")
def nvml_mock(patch):
    """A 300 W card that ACCEPTS power-limit changes, so the mini power sweep
    measures its full derived ladder rather than skipping."""
    pynvml = MagicMock()
    pynvml.NVML_TEMPERATURE_GPU = 0
    pynvml.NVML_CLOCK_SM = 0
    pynvml.NVML_CLOCK_MEM = 1

    handle = MagicMock()
    pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    pynvml.nvmlDeviceGetPowerUsage.return_value = 200_000
    util = MagicMock(gpu=75.0, memory=40.0)
    pynvml.nvmlDeviceGetUtilizationRates.return_value = util
    pynvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(
        used=8 * 1024 * 1024, total=24 * 1024 * 1024
    )
    pynvml.nvmlDeviceGetTemperature.return_value = 65
    pynvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 3090"
    pynvml.nvmlSystemGetDriverVersion.return_value = "550.90.07"
    pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12040
    pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (100_000, 350_000)

    # A REAL readback: the cap NVML reports is whatever was last set, so a
    # "confirmed" sweep point is the watts that were actually requested. A
    # fixed return value here would silently collapse the whole ladder onto
    # stock (the bug US-MERGE-04 found in a sibling fixture).
    state = {"limit_mw": 300_000}
    pynvml.nvmlDeviceGetPowerManagementLimit.side_effect = lambda h: state["limit_mw"]
    # The card's FACTORY DEFAULT, which is what the suite restores TO and what
    # the fraction ladder is derived FROM -- deliberately a fixed value, unlike
    # the readback above, because a factory default does not move when a cap is
    # applied. That is the whole point: restoring the CURRENT limit would put
    # back a leftover cap from a hard-killed previous run.
    pynvml.nvmlDeviceGetPowerManagementDefaultLimit.side_effect = lambda h: 300_000

    def _set_limit(h, mw):
        state["limit_mw"] = mw

    pynvml.nvmlDeviceSetPowerManagementLimit.side_effect = _set_limit

    patch.setitem(sys.modules, "pynvml", pynvml)
    return state


# --- the "zero network" assertion -------------------------------------------

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(scope="module")
def loopback_only(patch):
    """Fail the moment anything tries to leave this machine."""
    connections: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def _check(host: object, where: str) -> None:
        if isinstance(host, str) and host not in _LOOPBACK_HOSTS:
            raise AssertionError(f"{where} left loopback: {host!r}")

    def _guarded_connect(self, address):
        _check(address[0] if isinstance(address, tuple) else address, "connect()")
        connections.append(str(address))
        return real_connect(self, address)

    def _guarded_connect_ex(self, address):
        _check(address[0] if isinstance(address, tuple) else address, "connect_ex()")
        connections.append(str(address))
        return real_connect_ex(self, address)

    def _guarded_getaddrinfo(host, port, *args, **kwargs):
        _check(host, "getaddrinfo()")
        return real_getaddrinfo(host, port, *args, **kwargs)

    patch.setattr(socket.socket, "connect", _guarded_connect)
    patch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    patch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)
    return connections


# --- the run ----------------------------------------------------------------

_ENV_KEYS = (
    "BENCH_OPTIN", "BENCH_BUNDLE_DIR", "BENCH_DATA_DIR", "NODE_SALT_PATH",
    "HM_ASYNC_API_URL", "HM_ASYNC_EMAIL", "HM_ASYNC_PASSWORD", "CONTROLLER_ID",
)


@pytest.fixture(scope="module")
def patch():
    """A module-scoped `monkeypatch`. The whole file describes ONE run of
    `bench quick` (~6 s of real HTTP, real parquet reads and real 5 Hz
    sampling); re-running it per test would be twelve identical runs."""
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("bench-quick-e2e")


@pytest.fixture(scope="module")
def quick_run(run_dir, patch, engine_mock, dataset_fixtures, nvml_mock, loopback_only):
    """Run the real `bench quick` CLI body against the fixtures above.

    The ONLY rewiring is where the engine lives: `run_quick_suite` is called
    with the mock server's loopback host/port instead of the default
    `localhost:11434`. Everything downstream of detection — model resolution,
    the task loop, 5 Hz sampling, the mini power sweep, `compute_metrics`,
    artifact writing, `build_bundle` — is the shipped code path.
    """
    for key in _ENV_KEYS:
        patch.delenv(key, raising=False)
    patch.chdir(run_dir)
    patch.setattr(cli, "get_profiler", lambda: _no_gpu_profiler())

    real_run_quick_suite = cli.run_quick_suite
    patch.setattr(
        cli,
        "run_quick_suite",
        # **kw so the stub keeps accepting whatever the CLI passes through
        # (currently `restore_to_factory_default`, from POWER_CAP_POLICY) --
        # a positional-only lambda made this fixture a second place to edit
        # every time the suite gained a parameter.
        lambda **kw: real_run_quick_suite(
            engine_choice="ollama",
            host="127.0.0.1",
            ollama_port=engine_mock.port,
            **kw,
        ),
    )

    settings = Settings(
        _env_file=None,
        BENCH_BUNDLE_DIR=str(run_dir / "bundles"),
        BENCH_DATA_DIR=str(run_dir / "data"),
        NODE_SALT_PATH=str(run_dir / "node_salt"),
    )
    code, message = cli.run_bench_quick(settings)
    return code, message, settings


def _no_gpu_profiler():
    from hmasync_controller.profiler import NullProfiler

    return NullProfiler()


def _bundle(settings: Settings) -> dict:
    bundles = sorted(Path(settings.BENCH_BUNDLE_DIR).glob("bundle-*.json"))
    assert len(bundles) == 1, bundles
    return json.loads(bundles[0].read_text())


class TestFixtureBenchQuickEndToEnd:
    def test_exits_clean_and_writes_one_bundle(self, quick_run):
        code, message, settings = quick_run
        assert code == 0, message
        assert "bundle written to" in message
        assert _bundle(settings)["suite"] == "quick"

    def test_bundle_validates_against_the_vendored_schema(self, quick_run):
        _, _, settings = quick_run
        assert validate_bundle(_bundle(settings)) == []

    def test_every_quick_task_is_measured(self, quick_run):
        _, _, settings = quick_run
        tasks = {run["task"] for run in _bundle(settings)["runs"]}
        assert tasks == {"gsm8k_platinum", "mmlu_redux", "ifeval"}

    def test_the_mini_power_sweep_measured_its_full_derived_ladder(self, quick_run):
        """0.85/0.75/0.65 of a 300 W card, clamped into [100, 350] W."""
        _, _, settings = quick_run
        caps = sorted(
            run["power_limit_w"]
            for run in _bundle(settings)["runs"]
            if run["power_limit_w"] is not None
        )
        assert caps == [195, 225, 255]

    def test_the_card_is_left_at_its_stock_limit(self, quick_run, nvml_mock):
        """Restore-in-finally, observed through the fake card's own state."""
        assert nvml_mock["limit_mw"] == 300_000

    def test_accuracy_is_computed_by_the_real_scorers(self, quick_run):
        _, _, settings = quick_run
        by_task = {run["task"]: run for run in _bundle(settings)["runs"]}

        # gsm8k_platinum: gold alternates 7/99, the engine always answers 7.
        gsm = by_task["gsm8k_platinum"]
        assert gsm["n_items"] == 25
        assert 0.0 < gsm["accuracy"] < 1.0
        assert 0 < gsm["n_correct"] < gsm["n_items"]
        assert gsm["accuracy"] == pytest.approx(gsm["n_correct"] / gsm["n_items"])

        # ifeval: half the rows demand ALL-CAPS, which the engine never sends.
        ifeval = by_task["ifeval"]
        assert 0.0 < ifeval["accuracy"] < 1.0

        # mmlu_redux: the engine always answers B; golds alternate B/A, and the
        # `error_type != "ok"` rows never reach the model at all.
        mmlu = by_task["mmlu_redux"]
        assert 0.0 < mmlu["accuracy"] < 1.0

    def test_streaming_was_real_not_a_fallback(self, quick_run, engine_mock):
        _, _, settings = quick_run
        assert engine_mock.non_streaming_requests == 0
        for run in _bundle(settings)["runs"]:
            assert run["streaming_used"] is True
            assert run["ttft_p50_s"] is not None and run["ttft_p50_s"] > 0
            assert run["itl_mean_ms"] is not None

    def test_energy_and_telemetry_came_from_the_sampler(self, quick_run):
        _, _, settings = quick_run
        for run in _bundle(settings)["runs"]:
            assert run["total_joules_gpu"] > 0
            assert run["mean_gpu_util_pct"] == pytest.approx(75.0)
            assert run["gpu_name"] == "NVIDIA GeForce RTX 3090"

    def test_per_run_artifacts_land_on_disk(self, quick_run):
        _, _, settings = quick_run
        bundle = _bundle(settings)
        folders = sorted(p for p in Path(settings.BENCH_DATA_DIR).iterdir() if p.is_dir())
        assert len(folders) == len(bundle["runs"])
        for folder in folders:
            assert (folder / "metrics.json").exists()
            assert pq.read_table(folder / "telemetry.parquet").num_rows > 0
            assert pq.read_table(folder / "items.parquet").num_rows > 0

    def test_the_bundle_carries_no_local_run_id(self, quick_run):
        """Artifacts are keyed by `run_id` on disk; the bundle is not. The
        submission side joins on `node_hash` alone (bundle.py's allowlist)."""
        for run in _bundle(quick_run[2])["runs"]:
            assert "run_id" not in run

    def test_the_real_http_client_drove_the_real_engine(self, quick_run, engine_mock):
        """Detection, model verification and every item went over a socket."""
        paths = [path for _, path in engine_mock.requests]
        assert "/api/version" in paths       # OllamaAdapter.ready()/version()
        assert "/api/show" in paths          # resolve_quick_model, never pulls
        # 25 + 50 + 25 scored items, plus 25 more per capped sweep point.
        assert paths.count("/v1/chat/completions") == 100 + 3 * 25

    def test_nothing_left_loopback(self, quick_run, engine_mock, loopback_only):
        """The `loopback_only` guard raises in-flight; this pins the log too."""
        assert loopback_only, "no connections recorded — the guard never ran"
        for address in loopback_only:
            assert "127.0.0.1" in address or "::1" in address, address
        assert any(str(engine_mock.port) in address for address in loopback_only)

    def test_the_only_datasets_read_were_the_fixtures(self, quick_run, dataset_fixtures):
        repos = {repo for repo, _, _ in dataset_fixtures}
        assert repos == {
            "madrylab/gsm8k-platinum", "openai/gsm8k",
            "edinburgh-dawg/mmlu-redux-2.0", "cais/mmlu", "google/IFEval",
        }
