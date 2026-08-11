"""
Adapter — the framework seam (lives in the controller).

    run(request: dict)         -> AdapterRunResult   execute, blocking until complete
    fingerprint(request: dict) -> (str, dict)        (stable hash, cost-relevant features)
    preflight(request=None)    -> bool               framework reachable + model loaded

`get_adapter(framework)` returns the backend for a workflow's declared framework:

    CommandAdapter       — runs an arbitrary local command via subprocess
    OllamaAdapter        — blocking HTTP call to a local Ollama server
    OpenAICompatAdapter  — blocking HTTP call to a local OpenAI-compatible server

Contract: two jobs that cost the same to run must produce the same
fingerprint hash; volatile inputs (seeds, timestamps, request ids) are excluded so
history accumulates per *workload identity*, not per invocation. `run` blocks so the
profiler can wrap it; the executor merges the result's
`exit_status`/`work_units` into the RunRecord. New frameworks = one registered class.

Work-unit capture: when the framework reports work done — Ollama's
`eval_count`, OpenAI's `usage.completion_tokens` — the adapter fills `work_units`
with `work_unit_kind='tokens'`, which feeds energy-per-unit-work (J/token)
prediction. CommandAdapter leaves them null unless the request declares a value.

Bounded by construction: every `run()` honours a `timeout` in the request (the
executor fills one in from the placement window when the job declares none), so
no adapter can hold the GPU past the window it was planned into. HTTP backends
fall back to `DEFAULT_RUN_TIMEOUT_S`; `command` has no fallback of its own
because a subprocess timeout of None blocks forever.

Seam signature note: the brief writes `preflight() -> bool`, but "the requested
model loaded" needs the request, and the executor has it at each placement — so
`preflight` takes an optional `request` (None → reachability only). Same
seam-widening discipline the API services used (predict/optimize).

Import safety: httpx clients are built lazily; nothing here opens a socket or runs
a subprocess at import time, so the module imports with an empty `.env` and no
local AI server running.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# --- exit_status values (the vocabulary the API accepts on a run record) ---
EXIT_SUCCESS = "success"
EXIT_ERROR = "error"
EXIT_PREEMPTED = "preempted"

# Token work is the unit every LLM framework reports; a job that generated N
# completion tokens carries work_units=N, work_unit_kind='tokens'.
WORK_UNIT_TOKENS = "tokens"

# Default local endpoints (override via constructor or request['base_url']).
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_URL = "http://localhost:8000/v1"

# run() blocks for the full generation, so its HTTP timeout is generous (a local
# model can take minutes). preflight() is a cheap liveness poke, so it's short.
DEFAULT_RUN_TIMEOUT_S = 600.0
DEFAULT_PREFLIGHT_TIMEOUT_S = 5.0


class AdapterError(Exception):
    """A request is malformed for this adapter (e.g. missing a required field)."""


class UnknownFrameworkError(AdapterError):
    """A workflow declared a framework with no registered adapter."""


@dataclass
class AdapterRunResult:
    """The outcome of one `run()` — what the executor turns into a RunRecord.

    `exit_status` is one of EXIT_SUCCESS / EXIT_ERROR / EXIT_PREEMPTED. `work_units`
    (+ `work_unit_kind`) are the framework-reported work done, null when unknown.
    `output` carries the raw framework response for logging/debugging; `detail` is a
    short human-readable failure reason on a non-success exit.
    """

    exit_status: str
    work_units: float | None = None
    work_unit_kind: str | None = None
    exit_code: int | None = None
    output: Any = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_status == EXIT_SUCCESS


# --- fingerprint helpers -----------------------------------------------------


def _stable_hash(features: dict[str, Any]) -> str:
    """Deterministic hash of a feature dict (sorted keys → stable across runs)."""
    canonical = json.dumps(features, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bucket(value: Any) -> int | None:
    """Coarse log-scale bucket so near-identical sizes share one history key.

    Returns floor(log2(n)) — 100 and 120 both land in bucket 6 ([64,128)), while
    100 and 300 split (6 vs 8). None/0/1 → 0. Non-numeric → None. Bucketing (not
    raw counts) is what lets a 100-token and a 110-token job pool their measured
    energy; the trade-off is that values straddling a power-of-two boundary split.
    """
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if n <= 1:
        return 0
    return int(math.floor(math.log2(n)))


def _messages_text(messages: Any) -> str:
    """Concatenate the text of chat messages (for prompt-size bucketing only).

    Content is never hashed directly — only its *length* is bucketed — so the
    prompt text stays out of the fingerprint (two same-length prompts share a key).
    """
    if not messages:
        return ""
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # OpenAI multi-part content: [{"type":"text","text":"..."},...]
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "".join(parts)


def _require(request: dict[str, Any], key: str) -> Any:
    if key not in request or request[key] in (None, ""):
        raise AdapterError(f"request is missing required field '{key}'")
    return request[key]


def declared_timeout(request: dict[str, Any]) -> float | None:
    """The request's `timeout` in seconds, or None when it declares none.

    Tolerant of a numeric string (a catalog is hand-edited JSON). A non-positive
    or unparseable value is treated as "not declared" rather than as zero — a
    zero-second timeout would fail every job instantly, which is never what a
    typo meant.
    """
    raw = request.get("timeout")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _prompt_length(request: dict[str, Any]) -> int:
    """Prompt size for fingerprint bucketing — inline, from `prompt_file`, or chat.

    Fingerprints only ever see the *length*, never the text, so a file-backed
    prompt buckets exactly like the same prompt passed inline. An unreadable file
    is length 0 here rather than an error: a null fingerprint would lose the run's
    history key, and `run()` is about to raise the real, specific failure anyway.
    """
    if request.get("prompt") or request.get("prompt_file"):
        try:
            return len(_resolve_prompt(request))
        except AdapterError:
            return 0
    return len(_messages_text(request.get("messages")))


def _resolve_prompt(request: dict[str, Any]) -> str:
    """Return the request's prompt text, reading `prompt_file` when that's the form used.

    An inline `prompt` wins; `prompt_file` is read from disk. Both are documented,
    and a batch prompt is usually a file — it is the natural shape for the queue
    of work an overnight job chews through.

    An unreadable `prompt_file` raises `AdapterError` rather than degrading to an
    empty prompt. That is the whole point: a silently-empty prompt still returns
    200, still profiles, and still reports watt-hours, so it enters the run history
    as a successful job that did no work and poisons the energy prediction for that
    fingerprint. A hard failure gets a `failed` ack and a replan instead.
    """
    prompt = request.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt

    path = request.get("prompt_file")
    if not path:
        return prompt if isinstance(prompt, str) else ""
    try:
        return Path(path).read_text()
    except OSError as exc:
        raise AdapterError(f"prompt_file {path!r} could not be read: {exc}") from exc


# --- base ------------------------------------------------------------------


class Adapter(ABC):
    """The framework seam. One registered class per framework."""

    FRAMEWORK: str = ""

    @abstractmethod
    def run(self, request: dict[str, Any]) -> AdapterRunResult:
        """Execute the request, blocking until it completes."""

    @abstractmethod
    def fingerprint(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Return (stable_hash, cost-relevant features) for this request."""

    def preflight(self, request: dict[str, Any] | None = None) -> bool:
        """Whether the framework is ready to run `request` right now.

        Default True — a backend with nothing to check (CommandAdapter) is always
        ready. HTTP backends override to poke the server and check the model.
        """
        return True


class _HttpAdapter(Adapter):
    """Shared HTTP plumbing for the Ollama / OpenAI-compatible backends."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        timeout: float | None = DEFAULT_RUN_TIMEOUT_S,
        preflight_timeout: float = DEFAULT_PREFLIGHT_TIMEOUT_S,
    ):
        self._base_url = (base_url or self._default_base_url()).rstrip("/")
        self._injected_client = http_client
        self._owned_client: httpx.Client | None = None
        self.timeout = timeout
        self.preflight_timeout = preflight_timeout

    def _default_base_url(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _client(self) -> httpx.Client:
        if self._injected_client is not None:
            return self._injected_client
        if self._owned_client is None:
            self._owned_client = httpx.Client()
        return self._owned_client

    def _base(self, request: dict[str, Any]) -> str:
        # A request may target a different endpoint than the adapter default;
        # base_url is a routing detail, never part of the cost fingerprint.
        override = request.get("base_url")
        return override.rstrip("/") if isinstance(override, str) and override else self._base_url

    def _timeout_for(self, request: dict[str, Any]) -> float | None:
        """Per-request timeout, falling back to this adapter's default.

        The executor injects the placement window's remaining time as `timeout`
        when the job declares none, so a hung generation is bounded by the window
        it was planned into rather than by a flat 600s that can run past it. An
        explicitly declared `timeout` always wins. Like `base_url`, this is a limit
        on the call, never a cost feature — it stays out of the fingerprint.
        """
        return declared_timeout(request) or self.timeout

    def close(self) -> None:
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None


# --- command ----------------------------------------------------------------


def normalize_command(command: Any) -> list[str]:
    """Normalize a command to an argv list (string → shlex.split, list → str-ified)."""
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, (list, tuple)):
        return [str(token) for token in command]
    raise AdapterError("command must be a string or a list of arguments")


class CommandAdapter(Adapter):
    """Runs an arbitrary local command via subprocess, blocking, capturing status.

    The fingerprint is the normalized command plus any declared `cost_features`;
    top-level volatile fields (`seed`, `timestamp`,...) are never read, so two
    runs of the same command with different seeds share one history key. Work units
    stay null unless the request declares `work_units` (the command is opaque —
    only the user knows what work it did).
    """

    FRAMEWORK = "command"

    def run(self, request: dict[str, Any]) -> AdapterRunResult:
        argv = normalize_command(_require(request, "command"))
        if not argv:
            raise AdapterError("command resolved to an empty argument list")

        cwd = request.get("cwd")
        # None → subprocess.run blocks forever. The executor fills this in from the
        # placement window for any job that declares no timeout, so an unbounded
        # run is only reachable by calling the adapter directly.
        timeout = declared_timeout(request)
        env = None
        if isinstance(request.get("env"), dict):
            env = {**os.environ, **{str(k): str(v) for k, v in request["env"].items()}}

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env,
            )
        except FileNotFoundError as exc:
            return AdapterRunResult(EXIT_ERROR, detail=f"command not found: {exc}")
        except subprocess.TimeoutExpired:
            return AdapterRunResult(EXIT_ERROR, detail="command timed out")
        except OSError as exc:
            return AdapterRunResult(EXIT_ERROR, detail=f"exec error: {exc}")

        exit_status = EXIT_SUCCESS if proc.returncode == 0 else EXIT_ERROR
        work_units, work_unit_kind = _declared_work(request)
        return AdapterRunResult(
            exit_status=exit_status,
            work_units=work_units,
            work_unit_kind=work_unit_kind,
            exit_code=proc.returncode,
            output={"stdout": proc.stdout, "stderr": proc.stderr},
            detail=None if exit_status == EXIT_SUCCESS else f"exit code {proc.returncode}",
        )

    def fingerprint(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        features: dict[str, Any] = {
            "framework": self.FRAMEWORK,
            "command": normalize_command(_require(request, "command")),
        }
        declared = request.get("cost_features")
        if isinstance(declared, dict) and declared:
            # Serialize nested dicts stably so the feature dict stays hashable and
            # order-independent.
            features["cost_features"] = json.loads(
                json.dumps(declared, sort_keys=True, default=str)
            )
        return _stable_hash(features), features

    # preflight() inherits the always-True default — a local command has no server
    # to check; a missing binary surfaces as EXIT_ERROR at run() time instead.


def _declared_work(request: dict[str, Any]) -> tuple[float | None, str | None]:
    """Read declared work_units/work_unit_kind off a request (both null if absent)."""
    raw = request.get("work_units")
    if raw is None:
        return None, None
    try:
        units = float(raw)
    except (TypeError, ValueError):
        return None, None
    kind = request.get("work_unit_kind") or WORK_UNIT_TOKENS
    return units, str(kind)


# --- ollama -----------------------------------------------------------------


class OllamaAdapter(_HttpAdapter):
    """Blocking HTTP call to a local Ollama server (`/api/generate` or `/api/chat`).

    Fingerprint = model + bucketed prompt / num_predict / batch sizes; the seed and
    prompt *content* are excluded. Work units come from the response `eval_count`
    (completion tokens), tagged `work_unit_kind='tokens'`.
    """

    FRAMEWORK = "ollama"

    def _default_base_url(self) -> str:
        return DEFAULT_OLLAMA_URL

    def run(self, request: dict[str, Any]) -> AdapterRunResult:
        model = _require(request, "model")
        base = self._base(request)
        options = request.get("options") if isinstance(request.get("options"), dict) else None

        messages = request.get("messages")
        if messages is not None:
            url = f"{base}/api/chat"
            payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        else:
            url = f"{base}/api/generate"
            payload = {"model": model, "prompt": _resolve_prompt(request), "stream": False}
        if options:
            payload["options"] = options

        data, err = self._post_json(url, payload, self._timeout_for(request))
        if err is not None:
            return err
        eval_count = data.get("eval_count")
        units = float(eval_count) if isinstance(eval_count, (int, float)) else None
        return AdapterRunResult(
            exit_status=EXIT_SUCCESS,
            work_units=units,
            work_unit_kind=WORK_UNIT_TOKENS if units is not None else None,
            output=data,
        )

    def fingerprint(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        model = _require(request, "model")
        options = request.get("options") if isinstance(request.get("options"), dict) else {}
        prompt_len = _prompt_length(request)
        features = {
            "framework": self.FRAMEWORK,
            "model": model,
            "prompt_bucket": _bucket(prompt_len),
            "num_predict_bucket": _bucket(options.get("num_predict")),
            "num_ctx_bucket": _bucket(options.get("num_ctx")),
            "batch_bucket": _bucket(options.get("num_batch")),
        }
        return _stable_hash(features), features

    def preflight(self, request: dict[str, Any] | None = None) -> bool:
        base = self._base(request or {})
        try:
            resp = self._client().get(f"{base}/api/tags", timeout=self.preflight_timeout)
        except httpx.HTTPError:
            return False
        if resp.status_code >= 400:
            return False
        model = (request or {}).get("model")
        if not model:
            return True  # no specific model to check → reachable is enough
        try:
            installed = resp.json().get("models") or []
        except (ValueError, AttributeError):
            return False
        return _model_installed(model, installed)

    def _post_json(
        self, url: str, payload: dict, timeout: float | None = None
    ) -> tuple[dict, AdapterRunResult | None]:
        return _blocking_post(self._client(), url, payload, timeout or self.timeout)


def _model_installed(model: str, installed: list) -> bool:
    """Whether `model` matches any Ollama tag (exact, or base name before the ':')."""
    names: set[str] = set()
    for entry in installed:
        if isinstance(entry, dict):
            for key in ("name", "model"):
                val = entry.get(key)
                if isinstance(val, str):
                    names.add(val)
        elif isinstance(entry, str):
            names.add(entry)
    if model in names:
        return True
    base = model.split(":")[0]
    return any(name == base or name.split(":")[0] == base for name in names)


# --- openai-compatible ------------------------------------------------------


class OpenAICompatAdapter(_HttpAdapter):
    """Blocking HTTP call to a local OpenAI-compatible server (`/chat/completions`).

    Covers llama.cpp server, vLLM, LM Studio, etc. Fingerprint = model + bucketed
    prompt / max_tokens / n sizes; seed and prompt content excluded. Work units come
    from `usage.completion_tokens`, tagged `work_unit_kind='tokens'`.
    """

    FRAMEWORK = "openai"

    def __init__(self, *, api_key: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.api_key = api_key

    def _default_base_url(self) -> str:
        return DEFAULT_OPENAI_URL

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def run(self, request: dict[str, Any]) -> AdapterRunResult:
        model = _require(request, "model")
        base = self._base(request)
        messages = request.get("messages")
        if messages is None and ("prompt" in request or "prompt_file" in request):
            messages = [{"role": "user", "content": _resolve_prompt(request)}]
        payload: dict[str, Any] = {"model": model, "messages": messages or []}
        for key in ("max_tokens", "temperature", "top_p", "n", "stop"):
            if key in request:
                payload[key] = request[key]

        data, err = _blocking_post(
            self._client(),
            f"{base}/chat/completions",
            payload,
            self._timeout_for(request),
            headers=self._headers(),
        )
        if err is not None:
            return err
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        completion = usage.get("completion_tokens")
        units = float(completion) if isinstance(completion, (int, float)) else None
        return AdapterRunResult(
            exit_status=EXIT_SUCCESS,
            work_units=units,
            work_unit_kind=WORK_UNIT_TOKENS if units is not None else None,
            output=data,
        )

    def fingerprint(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        model = _require(request, "model")
        features = {
            "framework": self.FRAMEWORK,
            "model": model,
            "prompt_bucket": _bucket(_prompt_length(request)),
            "max_tokens_bucket": _bucket(request.get("max_tokens")),
            "n_bucket": _bucket(request.get("n")),
        }
        return _stable_hash(features), features

    def preflight(self, request: dict[str, Any] | None = None) -> bool:
        base = self._base(request or {})
        try:
            resp = self._client().get(
                f"{base}/models", timeout=self.preflight_timeout, headers=self._headers()
            )
        except httpx.HTTPError:
            return False
        model = (request or {}).get("model")
        if resp.status_code >= 400:
            # Reachable but listing unsupported (many local servers 404 /models) —
            # reachability is the most we can assert, so accept it.
            return resp.status_code in (404, 405, 501)
        if not model:
            return True
        try:
            data = resp.json().get("data") or []
        except (ValueError, AttributeError):
            return True  # answered 200 but no parseable list → treat as reachable
        ids = {entry.get("id") for entry in data if isinstance(entry, dict)}
        # A server that lists no models but is up is still usable; only fail when it
        # lists models and ours isn't among them.
        return not ids or model in ids


# --- shared HTTP post -------------------------------------------------------


def _blocking_post(
    client: httpx.Client,
    url: str,
    payload: dict,
    timeout: float | None,
    headers: dict[str, str] | None = None,
) -> tuple[dict, AdapterRunResult | None]:
    """POST JSON and parse the response; return (data, None) or ({}, error_result).

    Transport failures and non-2xx statuses become an EXIT_ERROR result rather than
    an exception, so the executor gets a uniform outcome and can emit a failed ack.
    """
    try:
        resp = client.post(url, json=payload, timeout=timeout, headers=headers)
    except httpx.TimeoutException:
        return {}, AdapterRunResult(EXIT_ERROR, detail="request timed out")
    except httpx.HTTPError as exc:
        return {}, AdapterRunResult(EXIT_ERROR, detail=f"network error: {exc}")
    if resp.status_code >= 400:
        return {}, AdapterRunResult(
            EXIT_ERROR,
            exit_code=resp.status_code,
            detail=f"HTTP {resp.status_code}",
            output=_safe_json(resp),
        )
    try:
        return resp.json(), None
    except ValueError:
        return {}, AdapterRunResult(EXIT_ERROR, detail="response was not JSON")


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text


# --- registry ---------------------------------------------------------------

_REGISTRY: dict[str, type[Adapter]] = {
    CommandAdapter.FRAMEWORK: CommandAdapter,
    OllamaAdapter.FRAMEWORK: OllamaAdapter,
    OpenAICompatAdapter.FRAMEWORK: OpenAICompatAdapter,
}

# Friendly aliases → canonical framework key.
_ALIASES = {
    "cmd": "command",
    "shell": "command",
    "openai-compat": "openai",
    "openai_compatible": "openai",
    "openai-compatible": "openai",
}


def list_frameworks() -> list[str]:
    """The registered framework keys (sorted), for error messages and docs."""
    return sorted(_REGISTRY)


def get_adapter(framework: str, **kwargs: Any) -> Adapter:
    """Construct the adapter for `framework`, raising a clear error if unknown.

    `kwargs` pass through to the adapter constructor (base_url, http_client,...).
    """
    if not framework or not isinstance(framework, str):
        raise UnknownFrameworkError(
            f"framework must be one of {list_frameworks()}, got {framework!r}"
        )
    key = _ALIASES.get(framework.strip().lower(), framework.strip().lower())
    cls = _REGISTRY.get(key)
    if cls is None:
        raise UnknownFrameworkError(
            f"unknown framework {framework!r}; registered: {list_frameworks()}"
        )
    return cls(**kwargs)
