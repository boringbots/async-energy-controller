"""
Framework adapter tests.

The two contracts under test:

  - **fingerprint stability + volatile exclusion**: two jobs that cost
    the same hash identically; seeds/timestamps never change the hash.
  - **run() capturing exit status + work units**: subprocess/httpx fully
    mocked (repo mock-all-vendors rule — no live subprocess for a real binary, no
    live Ollama/OpenAI server); token counts flow into work_units.

HTTP backends are driven through a real `httpx.Client` wired to `httpx.MockTransport`
(the same posture the ApiClient tests use), so the adapter's real request/response
code runs against a scripted fake. CommandAdapter's subprocess is monkeypatched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from hmasync_controller import adapters
from hmasync_controller.adapters import (
    EXIT_ERROR,
    EXIT_SUCCESS,
    WORK_UNIT_TOKENS,
    AdapterError,
    CommandAdapter,
    OllamaAdapter,
    OpenAICompatAdapter,
    UnknownFrameworkError,
    _bucket,
    get_adapter,
    list_frameworks,
)


# --- helpers ---------------------------------------------------------------


def _mock_client(handler) -> httpx.Client:
    """A real httpx.Client whose requests are answered by `handler`."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def _fake_completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# --- registry --------------------------------------------------------------


def test_get_adapter_returns_registered_classes():
    assert isinstance(get_adapter("command"), CommandAdapter)
    assert isinstance(get_adapter("ollama"), OllamaAdapter)
    assert isinstance(get_adapter("openai"), OpenAICompatAdapter)


def test_get_adapter_aliases():
    assert isinstance(get_adapter("openai-compat"), OpenAICompatAdapter)
    assert isinstance(get_adapter("openai_compatible"), OpenAICompatAdapter)
    assert isinstance(get_adapter("CMD"), CommandAdapter)  # case-insensitive


def test_get_adapter_unknown_framework_raises_clear_error():
    with pytest.raises(UnknownFrameworkError) as exc:
        get_adapter("tensorflow")
    # The message names what IS registered so the operator can self-correct.
    assert "tensorflow" in str(exc.value)
    for known in list_frameworks():
        assert known in str(exc.value)


def test_get_adapter_empty_framework_raises():
    with pytest.raises(UnknownFrameworkError):
        get_adapter("")


def test_get_adapter_passes_kwargs_through():
    adapter = get_adapter("ollama", base_url="http://box:11434/")
    assert adapter._base_url == "http://box:11434"  # trailing slash stripped


# --- _bucket ---------------------------------------------------------------


def test_bucket_groups_near_values_splits_far():
    assert _bucket(100) == _bucket(120)  # both in [64, 128) → bucket 6
    assert _bucket(100) != _bucket(300)  # 6 vs 8
    assert _bucket(None) is None
    assert _bucket(0) == 0
    assert _bucket(1) == 0
    assert _bucket("nope") is None


# --- CommandAdapter: fingerprint -------------------------------------------


def test_command_fingerprint_stable_and_normalized():
    a = CommandAdapter()
    # String and list forms of the same command normalize to the same argv → same hash.
    h_str, _ = a.fingerprint({"command": "python train.py --epochs 3"})
    h_list, feats = a.fingerprint({"command": ["python", "train.py", "--epochs", "3"]})
    assert h_str == h_list
    assert feats["command"] == ["python", "train.py", "--epochs", "3"]
    # Deterministic across calls.
    assert a.fingerprint({"command": "python train.py --epochs 3"})[0] == h_str


def test_command_fingerprint_excludes_volatile_top_level_fields():
    a = CommandAdapter()
    base = {"command": ["run.sh"]}
    h0, _ = a.fingerprint(base)
    # Seeds / timestamps / request ids at the top level must not change the hash.
    h1, _ = a.fingerprint({**base, "seed": 42, "timestamp": "2026-07-11T00:00:00Z"})
    h2, _ = a.fingerprint({**base, "seed": 999, "request_id": "abc"})
    assert h0 == h1 == h2


def test_command_fingerprint_changes_with_command_and_cost_features():
    a = CommandAdapter()
    h_base, _ = a.fingerprint({"command": ["run.sh"]})
    assert h_base != a.fingerprint({"command": ["run.sh", "--big"]})[0]
    # Declared cost features are part of the identity.
    h_cf, _ = a.fingerprint({"command": ["run.sh"], "cost_features": {"size": "large"}})
    assert h_cf != h_base
    #...but order-independent within the cost_features dict.
    h_cf2, _ = a.fingerprint(
        {"command": ["run.sh"], "cost_features": {"size": "large", "gpu": 1}}
    )
    h_cf3, _ = a.fingerprint(
        {"command": ["run.sh"], "cost_features": {"gpu": 1, "size": "large"}}
    )
    assert h_cf2 == h_cf3


def test_command_fingerprint_missing_command_raises():
    with pytest.raises(AdapterError):
        CommandAdapter().fingerprint({})


# --- CommandAdapter: run ----------------------------------------------------


def test_command_run_success(monkeypatch):
    monkeypatch.setattr(
        adapters.subprocess, "run", lambda *a, **k: _fake_completed(0, "done\n")
    )
    result = CommandAdapter().run({"command": ["echo", "hi"]})
    assert result.exit_status == EXIT_SUCCESS
    assert result.ok
    assert result.exit_code == 0
    assert result.output["stdout"] == "done\n"
    # No declared work → null work units (the command is opaque).
    assert result.work_units is None
    assert result.work_unit_kind is None


def test_command_run_declared_work_units(monkeypatch):
    monkeypatch.setattr(adapters.subprocess, "run", lambda *a, **k: _fake_completed(0))
    result = CommandAdapter().run(
        {"command": ["gen.sh"], "work_units": 512, "work_unit_kind": "tokens"}
    )
    assert result.work_units == 512.0
    assert result.work_unit_kind == "tokens"


def test_command_run_nonzero_exit_is_error(monkeypatch):
    monkeypatch.setattr(
        adapters.subprocess, "run", lambda *a, **k: _fake_completed(2, "", "boom")
    )
    result = CommandAdapter().run({"command": ["false"]})
    assert result.exit_status == EXIT_ERROR
    assert not result.ok
    assert result.exit_code == 2
    assert "exit code 2" in result.detail


def test_command_run_missing_binary_is_clean_error(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("no such file: nope")

    monkeypatch.setattr(adapters.subprocess, "run", _raise)
    result = CommandAdapter().run({"command": ["nope"]})
    assert result.exit_status == EXIT_ERROR
    assert "not found" in result.detail


def test_command_run_timeout_is_clean_error(monkeypatch):
    def _raise(*a, **k):
        raise adapters.subprocess.TimeoutExpired(cmd="slow", timeout=1)

    monkeypatch.setattr(adapters.subprocess, "run", _raise)
    result = CommandAdapter().run({"command": ["slow"], "timeout": 1})
    assert result.exit_status == EXIT_ERROR
    assert "timed out" in result.detail


def test_command_run_missing_command_raises():
    with pytest.raises(AdapterError):
        CommandAdapter().run({})


def test_command_preflight_always_true():
    # No server to check — CommandAdapter is always ready (a missing binary shows
    # up as EXIT_ERROR at run() time, not preflight).
    assert CommandAdapter().preflight() is True
    assert CommandAdapter().preflight({"command": ["anything"]}) is True


# --- OllamaAdapter: fingerprint --------------------------------------------


def test_ollama_fingerprint_stable_buckets_and_excludes_seed():
    a = OllamaAdapter()
    r1 = {"model": "llama3", "prompt": "x" * 100, "options": {"num_predict": 128, "seed": 1}}
    r2 = {"model": "llama3", "prompt": "y" * 110, "options": {"num_predict": 128, "seed": 999}}
    # Near-identical prompt sizes (100 vs 110 chars → same bucket) + same model +
    # same num_predict, different seed/content → identical fingerprint.
    assert a.fingerprint(r1)[0] == a.fingerprint(r2)[0]


def test_ollama_fingerprint_changes_with_model_and_size():
    a = OllamaAdapter()
    base = {"model": "llama3", "prompt": "x" * 100}
    assert a.fingerprint(base)[0] != a.fingerprint({**base, "model": "mistral"})[0]
    # A large prompt-size jump crosses a bucket boundary.
    assert a.fingerprint(base)[0] != a.fingerprint({"model": "llama3", "prompt": "x" * 5000})[0]


def test_ollama_fingerprint_uses_messages_length_when_no_prompt():
    a = OllamaAdapter()
    msgs = {"model": "llama3", "messages": [{"role": "user", "content": "x" * 100}]}
    prompt = {"model": "llama3", "prompt": "z" * 100}
    # Same model + same content length via either shape → same bucket → same hash.
    assert a.fingerprint(msgs)[0] == a.fingerprint(prompt)[0]


def test_ollama_fingerprint_missing_model_raises():
    with pytest.raises(AdapterError):
        OllamaAdapter().fingerprint({"prompt": "hi"})


# --- OllamaAdapter: run -----------------------------------------------------


def test_ollama_run_generate_captures_eval_count():
    def handler(request):
        assert request.url.path == "/api/generate"
        body = request.read().decode()
        assert '"stream":false' in body.replace(" ", "")
        return httpx.Response(200, json={"response": "hi", "eval_count": 42, "done": True})

    a = OllamaAdapter(http_client=_mock_client(handler))
    result = a.run({"model": "llama3", "prompt": "hello"})
    assert result.exit_status == EXIT_SUCCESS
    assert result.work_units == 42.0
    assert result.work_unit_kind == WORK_UNIT_TOKENS


def test_ollama_run_chat_path():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, json={"message": {"content": "hi"}, "eval_count": 7})

    a = OllamaAdapter(http_client=_mock_client(handler))
    result = a.run({"model": "llama3", "messages": [{"role": "user", "content": "hi"}]})
    assert seen["path"] == "/api/chat"
    assert result.work_units == 7.0


def test_ollama_run_no_eval_count_leaves_work_null():
    def handler(request):
        return httpx.Response(200, json={"response": "hi"})

    a = OllamaAdapter(http_client=_mock_client(handler))
    result = a.run({"model": "llama3", "prompt": "hi"})
    assert result.exit_status == EXIT_SUCCESS
    assert result.work_units is None
    assert result.work_unit_kind is None


def test_ollama_run_http_error_is_error_result():
    def handler(request):
        return httpx.Response(500, json={"error": "model not loaded"})

    a = OllamaAdapter(http_client=_mock_client(handler))
    result = a.run({"model": "llama3", "prompt": "hi"})
    assert result.exit_status == EXIT_ERROR
    assert "HTTP 500" in result.detail
    assert result.exit_code == 500


def test_ollama_run_network_error_is_error_result():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    a = OllamaAdapter(http_client=_mock_client(handler))
    result = a.run({"model": "llama3", "prompt": "hi"})
    assert result.exit_status == EXIT_ERROR
    assert "network error" in result.detail


def test_ollama_run_missing_model_raises():
    with pytest.raises(AdapterError):
        OllamaAdapter().run({"prompt": "hi"})


def test_ollama_run_uses_request_base_url_override():
    seen = {}

    def handler(request):
        seen["host"] = request.url.host
        return httpx.Response(200, json={"eval_count": 1})

    a = OllamaAdapter(http_client=_mock_client(handler))
    a.run({"model": "llama3", "prompt": "hi", "base_url": "http://gpu-box:11434"})
    assert seen["host"] == "gpu-box"


# --- OllamaAdapter: preflight ----------------------------------------------


def _ollama_tags_handler(models):
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": models})

    return handler


def test_ollama_preflight_model_installed_true():
    handler = _ollama_tags_handler([{"name": "llama3:latest"}, {"name": "mistral:7b"}])
    a = OllamaAdapter(http_client=_mock_client(handler))
    assert a.preflight({"model": "llama3"}) is True  # base-name match
    assert a.preflight({"model": "mistral:7b"}) is True  # exact match


def test_ollama_preflight_model_not_installed_false():
    handler = _ollama_tags_handler([{"name": "llama3:latest"}])
    a = OllamaAdapter(http_client=_mock_client(handler))
    assert a.preflight({"model": "qwen"}) is False


def test_ollama_preflight_no_model_just_checks_reachable():
    handler = _ollama_tags_handler([])
    a = OllamaAdapter(http_client=_mock_client(handler))
    assert a.preflight() is True


def test_ollama_preflight_server_down_false():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    a = OllamaAdapter(http_client=_mock_client(handler))
    assert a.preflight({"model": "llama3"}) is False


# --- OpenAICompatAdapter: fingerprint --------------------------------------


def test_openai_fingerprint_stable_excludes_seed():
    a = OpenAICompatAdapter()
    r1 = {"model": "gpt-oss", "messages": [{"role": "user", "content": "x" * 100}],
          "max_tokens": 256, "seed": 1}
    r2 = {"model": "gpt-oss", "messages": [{"role": "user", "content": "y" * 110}],
          "max_tokens": 256, "seed": 42}
    assert a.fingerprint(r1)[0] == a.fingerprint(r2)[0]


def test_openai_fingerprint_changes_with_model_and_max_tokens():
    a = OpenAICompatAdapter()
    base = {"model": "gpt-oss", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    assert a.fingerprint(base)[0] != a.fingerprint({**base, "model": "other"})[0]
    assert a.fingerprint(base)[0] != a.fingerprint({**base, "max_tokens": 4000})[0]


def test_openai_and_ollama_fingerprints_differ_by_framework():
    # Even with identical-looking inputs, the framework is part of the identity.
    req = {"model": "m", "prompt": "x" * 100}
    assert OpenAICompatAdapter().fingerprint(req)[0] != OllamaAdapter().fingerprint(req)[0]


# --- OpenAICompatAdapter: run ----------------------------------------------


def test_openai_run_captures_completion_tokens():
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"completion_tokens": 88, "prompt_tokens": 5},
            },
        )

    a = OpenAICompatAdapter(base_url="http://localhost:8000/v1", http_client=_mock_client(handler))
    result = a.run({"model": "gpt-oss", "messages": [{"role": "user", "content": "hi"}]})
    assert result.exit_status == EXIT_SUCCESS
    assert result.work_units == 88.0
    assert result.work_unit_kind == WORK_UNIT_TOKENS


def test_openai_run_coerces_prompt_to_messages():
    seen = {}

    def handler(request):
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"usage": {"completion_tokens": 1}})

    a = OpenAICompatAdapter(http_client=_mock_client(handler))
    a.run({"model": "gpt-oss", "prompt": "hello there"})
    assert "hello there" in seen["body"]
    assert "messages" in seen["body"]


def test_openai_run_sends_bearer_when_api_key_set():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"usage": {"completion_tokens": 1}})

    a = OpenAICompatAdapter(api_key="sk-local", http_client=_mock_client(handler))
    a.run({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert seen["auth"] == "Bearer sk-local"


def test_openai_run_http_error_is_error_result():
    def handler(request):
        return httpx.Response(400, json={"error": "bad request"})

    a = OpenAICompatAdapter(http_client=_mock_client(handler))
    result = a.run({"model": "m", "messages": []})
    assert result.exit_status == EXIT_ERROR
    assert result.exit_code == 400


def test_openai_run_missing_model_raises():
    with pytest.raises(AdapterError):
        OpenAICompatAdapter().run({"messages": []})


# --- OpenAICompatAdapter: preflight ----------------------------------------


def test_openai_preflight_model_listed_true():
    def handler(request):
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "gpt-oss"}, {"id": "other"}]})

    a = OpenAICompatAdapter(http_client=_mock_client(handler))
    assert a.preflight({"model": "gpt-oss"}) is True


def test_openai_preflight_model_not_listed_false():
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "other"}]})

    a = OpenAICompatAdapter(http_client=_mock_client(handler))
    assert a.preflight({"model": "gpt-oss"}) is False


def test_openai_preflight_listing_unsupported_but_reachable_true():
    def handler(request):
        return httpx.Response(404, json={"detail": "not found"})

    a = OpenAICompatAdapter(http_client=_mock_client(handler))
    # A local server that 404s /models is still reachable → usable.
    assert a.preflight({"model": "gpt-oss"}) is True


def test_openai_preflight_server_down_false():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    a = OpenAICompatAdapter(http_client=_mock_client(handler))
    assert a.preflight({"model": "gpt-oss"}) is False


# --- prompt_file ------------------------------------------------------------
#
# `prompt_file` is documented in the README's framework table and used in
# jobs.example.json, but nothing read it: OllamaAdapter.run sent
# `request.get("prompt", "")`. A job copied verbatim from the docs therefore sent
# an EMPTY prompt, got a 200 back, profiled cleanly, and reported real watt-hours
# for work that never happened — a successful-looking run that poisons the energy
# history for its fingerprint. That is worse than a hard failure, which is why an
# unreadable file now raises instead of degrading to "".


def test_ollama_run_reads_prompt_file(tmp_path):
    prompt = tmp_path / "batch.txt"
    prompt.write_text("summarize these documents")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"response": "ok", "eval_count": 3})

    a = OllamaAdapter(http_client=_mock_client(handler))
    result = a.run({"model": "llama3", "prompt_file": str(prompt)})

    assert result.exit_status == EXIT_SUCCESS
    assert seen["body"]["prompt"] == "summarize these documents", (
        "the documented prompt_file key must actually reach the model"
    )


def test_ollama_run_inline_prompt_wins_over_prompt_file(tmp_path):
    prompt = tmp_path / "batch.txt"
    prompt.write_text("from the file")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"response": "ok"})

    a = OllamaAdapter(http_client=_mock_client(handler))
    a.run({"model": "llama3", "prompt": "inline", "prompt_file": str(prompt)})
    assert seen["body"]["prompt"] == "inline"


def test_ollama_run_missing_prompt_file_raises_rather_than_sending_empty(tmp_path):
    """The whole point: a silently-empty prompt would report as a successful run."""
    a = OllamaAdapter(http_client=_mock_client(lambda r: httpx.Response(200, json={})))
    with pytest.raises(AdapterError, match="prompt_file"):
        a.run({"model": "llama3", "prompt_file": str(tmp_path / "gone.txt")})


def test_openai_run_reads_prompt_file(tmp_path):
    prompt = tmp_path / "batch.txt"
    prompt.write_text("score this filing")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"choices": [], "usage": {"completion_tokens": 5}})

    a = OpenAICompatAdapter(http_client=_mock_client(handler))
    a.run({"model": "qwen", "prompt_file": str(prompt)})
    assert seen["body"]["messages"] == [{"role": "user", "content": "score this filing"}]


def test_prompt_file_and_inline_prompt_fingerprint_identically(tmp_path):
    """Same work, same cost, same history key — regardless of how it was passed."""
    text = "x" * 500
    prompt = tmp_path / "p.txt"
    prompt.write_text(text)

    inline, _ = OllamaAdapter().fingerprint({"model": "llama3", "prompt": text})
    from_file, _ = OllamaAdapter().fingerprint({"model": "llama3", "prompt_file": str(prompt)})
    assert inline == from_file


def test_unreadable_prompt_file_does_not_break_fingerprinting(tmp_path):
    """run() raises the real error; fingerprint must not also blow up first."""
    fp, features = OllamaAdapter().fingerprint(
        {"model": "llama3", "prompt_file": str(tmp_path / "gone.txt")}
    )
    assert fp and features["prompt_bucket"] == 0


# --- timeouts ---------------------------------------------------------------
#
# An undeclared timeout meant subprocess.run(..., timeout=None) — blocking
# indefinitely. A hung job does not merely delay its successors: it runs through
# the cheap window and into peak pricing, the exact outcome the product exists to
# prevent, while the run record still reports a correctly-planned placement.


def test_declared_timeout_reads_numbers_and_numeric_strings():
    assert adapters.declared_timeout({"timeout": 30}) == 30.0
    assert adapters.declared_timeout({"timeout": "45"}) == 45.0
    assert adapters.declared_timeout({"timeout": 12.5}) == 12.5


def test_declared_timeout_treats_nonsense_as_undeclared():
    """A zero-second timeout would fail every job instantly; no typo means that."""
    assert adapters.declared_timeout({}) is None
    assert adapters.declared_timeout({"timeout": 0}) is None
    assert adapters.declared_timeout({"timeout": -5}) is None
    assert adapters.declared_timeout({"timeout": "soon"}) is None
    assert adapters.declared_timeout({"timeout": True}) is None


def test_command_run_passes_declared_timeout_to_subprocess(monkeypatch):
    seen = {}

    def _run(*a, **k):
        seen["timeout"] = k.get("timeout")
        return _fake_completed(0)

    monkeypatch.setattr(adapters.subprocess, "run", _run)
    CommandAdapter().run({"command": ["slow"], "timeout": 90})
    assert seen["timeout"] == 90.0


def test_http_adapter_prefers_request_timeout_over_its_default():
    seen = {}

    def handler(request):
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={})

    a = OllamaAdapter(http_client=_mock_client(handler), timeout=600.0)
    a.run({"model": "llama3", "prompt": "hi", "timeout": 42})
    # httpx records the effective per-request timeout in the request extensions.
    assert seen["timeout"]["read"] == 42.0


def test_http_adapter_falls_back_to_its_default_timeout():
    seen = {}

    def handler(request):
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={})

    a = OllamaAdapter(http_client=_mock_client(handler), timeout=600.0)
    a.run({"model": "llama3", "prompt": "hi"})
    assert seen["timeout"]["read"] == 600.0


def test_timeout_is_not_part_of_the_fingerprint():
    """A limit on the call, never a cost feature — like base_url."""
    bare, _ = CommandAdapter().fingerprint({"command": ["job.sh"]})
    bounded, _ = CommandAdapter().fingerprint({"command": ["job.sh"], "timeout": 300})
    assert bare == bounded

    o_bare, _ = OllamaAdapter().fingerprint({"model": "llama3", "prompt": "hi"})
    o_bounded, _ = OllamaAdapter().fingerprint(
        {"model": "llama3", "prompt": "hi", "timeout": 300}
    )
    assert o_bare == o_bounded
