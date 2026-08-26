"""HTTP client for an OpenAI-compatible inference server (US-MERGE-04).

Ported near-verbatim from energy-bench's `clients/vllm.py` -- despite the
name, this client only ever speaks the OpenAI-compatible subset every engine
this package attaches to shares (`GET /health`, `GET /v1/models`,
`POST /v1/chat/completions`, `POST /v1/completions`), so it drives Ollama and
llama-server exactly as it drove vLLM in the lab tool (`bench/engines.py`'s
`OllamaAdapter`/`AttachLlamaCppAdapter` both construct one). Kept the
`VLLM*` naming rather than inventing a generic one -- it is the name every
ported caller (`bench/quick.py`, the two adapters) already imports, and
renaming it would just be churn with no behavior change.

Points at `bench.metrics.models.InferenceResult` (the plain-dataclass
carrier US-MERGE-03 ported) instead of energy-bench's pydantic
`energy_bench.models.InferenceResult` -- same fields, same defaults, see
that module's docstring for why this package doesn't share a schema
authority with energy-bench.
"""

import json
import time
import uuid
from collections.abc import Callable

import httpx

from hmasync_controller.bench.metrics.models import InferenceResult


class VLLMUnavailableError(Exception):
    """Raised when the server is unreachable or returns non-200."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class VLLMTimeoutError(Exception):
    """Raised when a request times out."""

    pass


class VLLMStreamingError(VLLMUnavailableError):
    """Raised when the SSE stream could not be parsed (malformed chunk,
    unexpected shape) -- distinct from a plain connection/timeout failure so
    a log line can say why streaming specifically failed. Still a
    `VLLMUnavailableError` so an existing broad catch on that type also
    catches this."""

    pass


def _delta_reasoning(choice: dict) -> str | None:
    """The reasoning-channel text of one streamed chunk, if any.

    Mirrors `_message_text`'s fallback for the streaming path. vLLM streams a
    reasoning model's chain of thought on `delta.reasoning` (or
    `delta.reasoning_content` depending on version/parser) and may never emit
    a `delta.content` token at all -- which is how gpt-oss-20b streamed
    hundreds of tokens and returned an empty string.
    """
    delta = choice.get("delta") or {}
    for key in ("reasoning", "reasoning_content"):
        piece = delta.get(key)
        if piece:
            return piece
    return None


def _message_text(message: dict) -> str:
    """The assistant's answer text, tolerating reasoning models that split
    their output across channels.

    A harmony/reasoning model served by vLLM returns TWO fields: `content`
    (the final answer) and `reasoning` / `reasoning_content` (the chain of
    thought). Reading `content` alone is correct for an ordinary model and
    silently catastrophic for a reasoning one -- measured on gpt-oss-20b
    2026-08-26, gpqa_diamond items generated 87-512 tokens and returned an
    EMPTY content string, because the whole budget went to the reasoning
    channel and no final channel was ever emitted. Every such item scored
    incorrect, dragging the model to 0.15 on a 4-choice task: BELOW the 0.25
    chance floor, which is the signature of a parsing failure rather than a
    weak model. The energy numbers were real throughout -- the work happened,
    only the answer was invisible.

    Prefer `content`. Fall back to the reasoning channel ONLY when content is
    empty: if the model expressed its conclusion nowhere else, that text is
    the best evidence of its answer, and the letter/number extractors already
    scan for a concluding statement ("...so the answer is B"). When content
    has anything at all, the reasoning is a scratchpad and must NOT override
    it.
    """
    content = (message.get("content") or "").strip()
    if content:
        return content
    for key in ("reasoning", "reasoning_content"):
        fallback = (message.get(key) or "").strip()
        if fallback:
            # Deliberately silent: this module has no logger, and a per-item
            # log line would fire on every request of a reasoning-model run.
            # The condition is visible in the data instead -- a run whose
            # answers came only from the reasoning channel shows up as
            # accuracy at or below chance if the fallback is ever removed.
            return fallback
    return ""


class VLLMClient:
    """Async HTTP client for an OpenAI-compatible inference server.

    vLLM, Ollama, and llama-server all expose an OpenAI-compatible
    completions API; this client sends completion requests and measures
    timing against whichever one `base_url` points at.
    """

    def __init__(self, host: str, port: int = 8000, timeout: float = 120.0) -> None:
        """Initialize the client.

        Args:
            host: Target host IP address.
            port: Server port (default 8000, vLLM's default -- callers
                targeting another engine override `base_url` directly, same
                as energy-bench's engine adapters do).
            timeout: HTTP request timeout in seconds (default 120s).
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    async def health_check(self) -> None:
        """Check server health.

        Raises:
            VLLMUnavailableError: If server is unreachable or unhealthy.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code != 200:
                    raise VLLMUnavailableError(
                        f"health check failed: HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
        except httpx.TimeoutException as e:
            raise VLLMUnavailableError(
                f"health check timed out at {self.base_url}"
            ) from e
        except httpx.RequestError as e:
            raise VLLMUnavailableError(
                f"unreachable at {self.base_url}: {e}"
            ) from e

    async def get_models(self) -> list[str]:
        """Get list of loaded models from the server.

        Returns:
            List of model names loaded on the server.

        Raises:
            VLLMUnavailableError: If server is unreachable.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                if response.status_code != 200:
                    raise VLLMUnavailableError(
                        f"Failed to get models: HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                data = response.json()
                # OpenAI-compatible API returns {data: [{id: "model_name"}, ...]}
                models = [model["id"] for model in data.get("data", [])]
                return models
        except httpx.TimeoutException as e:
            raise VLLMUnavailableError(
                f"Get models request timed out at {self.base_url}"
            ) from e
        except httpx.RequestError as e:
            raise VLLMUnavailableError(
                f"unreachable at {self.base_url}: {e}"
            ) from e

    async def get_version(self) -> str | None:
        """Get the server version from GET /version.

        Best-effort run metadata, not a health check: returns None on any
        error (no such endpoint, network failure, malformed response)
        rather than raising, so a version lookup never aborts a run.

        Returns:
            The version string, or None if it could not be determined.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/version")
                if response.status_code != 200:
                    return None
                return response.json().get("version")
        except Exception:
            return None

    async def _stream_sse(
        self,
        url: str,
        payload: dict[str, object],
        extract_text: Callable[[dict], str | None],
    ) -> tuple[str, str | None, int, int, float | None, list[float]]:
        """POST a `stream: true` request and parse the SSE response body.

        `extract_text(choice)` pulls the token text out of one `choices[0]`
        entry -- chat completions nest it under `delta.content`, legacy
        completions put it directly on `text`, so the two callers share this
        parsing loop but differ only in that extraction.

        Requests `stream_options: {"include_usage": true}` so the final
        chunk carries `usage`, same token counts the non-streaming path gets
        for free from the top-level response body.

        Returns (text, finish_reason, prompt_tokens, completion_tokens,
        ttft_s, itl_gaps_ms). `ttft_s` is None if no content token ever
        arrived (e.g. max_tokens=0) -- the caller substitutes 0.0 to match
        the non-streaming path's placeholder value.

        Raises:
            VLLMTimeoutError: On a request timeout.
            VLLMUnavailableError: On a non-200 response or connection failure.
            VLLMStreamingError: On a malformed/unparseable SSE chunk.
        """
        stream_payload = {
            **payload,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        text_parts: list[str] = []
        # Reasoning models stream their chain of thought on a SEPARATE channel
        # (`delta.reasoning` / `delta.reasoning_content`) and may never emit a
        # `delta.content` token at all. Accumulated separately so the primary
        # channel still wins when it exists -- the reasoning channel is a
        # scratchpad that routinely names options the model is REJECTING.
        # See `_message_text` for the non-streaming counterpart.
        fallback_parts: list[str] = []
        finish_reason: str | None = None
        prompt_tokens = 0
        completion_tokens = 0
        ttft_s: float | None = None
        itl_gaps_ms: list[float] = []
        last_token_perf: float | None = None
        send_perf = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=stream_payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise VLLMUnavailableError(
                            f"Streaming request failed: HTTP {response.status_code}: "
                            f"{body[:200].decode('utf-8', errors='replace')}",
                            status_code=response.status_code,
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data_str)
                        except ValueError as e:
                            raise VLLMStreamingError(
                                f"Malformed SSE chunk from {url}: {e}"
                            ) from e

                        usage = chunk.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = usage.get(
                                "completion_tokens", completion_tokens
                            )

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        reason = choice.get("finish_reason")
                        if reason:
                            finish_reason = reason
                        piece = extract_text(choice)
                        fallback_piece = _delta_reasoning(choice)
                        # TTFT/ITL time GENERATION, not one channel of it: a
                        # reasoning token is a token the GPU produced and is
                        # counted in `completion_tokens`. Timing only content
                        # tokens reported ttft=None (-> 0.0) for a response
                        # that streamed hundreds of reasoning tokens first.
                        if piece or fallback_piece:
                            now = time.perf_counter()
                            if ttft_s is None:
                                ttft_s = now - send_perf
                            elif last_token_perf is not None:
                                itl_gaps_ms.append((now - last_token_perf) * 1000.0)
                            last_token_perf = now
                        if piece:
                            text_parts.append(piece)
                        if fallback_piece:
                            fallback_parts.append(fallback_piece)
        except httpx.TimeoutException as e:
            raise VLLMTimeoutError(
                f"Streaming request timed out after {self.timeout}s"
            ) from e
        except httpx.RequestError as e:
            raise VLLMUnavailableError(f"unreachable at {self.base_url}: {e}") from e

        # Same precedence as the non-streaming `_message_text`: the primary
        # channel wins whenever it produced anything, and the reasoning channel
        # is used only when it did not. Keeping the two paths identical matters
        # -- otherwise the same run scores differently depending on whether
        # streaming was used, which is a config detail, not a measurement.
        text = "".join(text_parts).strip()
        if not text:
            text = "".join(fallback_parts).strip()

        return (
            text,
            finish_reason,
            prompt_tokens,
            completion_tokens,
            ttft_s,
            itl_gaps_ms,
        )

    async def chat(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        chat_template_kwargs: dict | None = None,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> tuple[InferenceResult, str]:
        """Send a chat-completion request and return the result plus its text.

        Uses /v1/chat/completions so the server applies the model's own chat
        template. For a cross-model benchmark this is the fair comparison:
        each model sees the prompt formatted the way it was trained to
        expect, rather than one model's format imposed on all of them.

        Args:
            chat_template_kwargs: Passed through as the top-level
                `chat_template_kwargs` (e.g. `{"enable_thinking": False}`) only
                when set, so an unset value is byte-identical to the payload
                before this parameter existed.
            temperature: Sampling temperature. Default 0.0 (deterministic)
                matches the value this was hardcoded to before it became a
                parameter.
            stream: When True, use the SSE streaming path so
                `InferenceResult.ttft_s`/`itl_gaps_ms` are real measurements
                instead of the 0.0/empty placeholders. Defaults to False so
                this method's behavior is unchanged for any caller that
                doesn't opt in; `run_quick_task` requests streaming
                explicitly and falls back to `stream=False` on a
                `VLLMUnavailableError`/`VLLMTimeoutError`.

        Returns:
            (InferenceResult, completion_text) — the text is needed for scoring.

        Raises:
            VLLMTimeoutError: If the request times out.
            VLLMUnavailableError: If the request fails (incl.
                `VLLMStreamingError` for a malformed stream when
                `stream=True`).
        """
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        t_start_s = time.time()

        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs

        if stream:
            (
                text,
                finish_reason,
                prompt_tokens,
                completion_tokens,
                ttft_s,
                itl_gaps_ms,
            ) = await self._stream_sse(
                f"{self.base_url}/v1/chat/completions",
                payload,
                lambda choice: choice.get("delta", {}).get("content"),
            )
            total_time = time.perf_counter() - start_time
            t_end_s = time.time()
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0.0
            return (
                InferenceResult(
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    ttft_s=ttft_s if ttft_s is not None else 0.0,
                    total_s=total_time,
                    tokens_per_second=tokens_per_second,
                    t_start_s=t_start_s,
                    t_end_s=t_end_s,
                    finish_reason=finish_reason,
                    itl_gaps_ms=itl_gaps_ms,
                ),
                text,
            )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )

                if response.status_code != 200:
                    raise VLLMUnavailableError(
                        f"Chat completion failed: HTTP {response.status_code}: "
                        f"{response.text[:200]}",
                        status_code=response.status_code,
                    )

                total_time = time.perf_counter() - start_time
                data = response.json()

                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                choices = data.get("choices", [])
                text = _message_text(choices[0].get("message", {})) if choices else ""
                finish_reason = choices[0].get("finish_reason") if choices else None
                t_end_s = time.time()

                tokens_per_second = (
                    completion_tokens / total_time if total_time > 0 else 0.0
                )

                return (
                    InferenceResult(
                        request_id=request_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        ttft_s=0.0,  # TTFT requires streaming; deferred
                        total_s=total_time,
                        tokens_per_second=tokens_per_second,
                        t_start_s=t_start_s,
                        t_end_s=t_end_s,
                        finish_reason=finish_reason,
                    ),
                    text or "",
                )

        except httpx.TimeoutException as e:
            raise VLLMTimeoutError(
                f"Chat completion timed out after {self.timeout}s"
            ) from e
        except httpx.RequestError as e:
            raise VLLMUnavailableError(
                f"unreachable at {self.base_url}: {e}"
            ) from e

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> InferenceResult:
        """Send a completion request and measure timing.

        Args:
            prompt: The input prompt text.
            model: Model name to use for inference.
            max_tokens: Maximum tokens to generate (default 512).
            temperature: Sampling temperature. Default 0.0 (deterministic)
                matches the value this was hardcoded to before it became a
                parameter.
            stream: When True, use the SSE streaming path so
                `ttft_s`/`itl_gaps_ms` are real measurements instead of the
                0.0/empty placeholders. See `chat()`'s docstring for the
                default-False rationale.

        Returns:
            InferenceResult with token counts and timing metrics.

        Raises:
            VLLMTimeoutError: If the request times out.
            VLLMUnavailableError: If the request fails (incl.
                `VLLMStreamingError` for a malformed stream when
                `stream=True`).
        """
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        t_start_s = time.time()

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stream:
            (
                _text,
                finish_reason,
                prompt_tokens,
                completion_tokens,
                ttft_s,
                itl_gaps_ms,
            ) = await self._stream_sse(
                f"{self.base_url}/v1/completions",
                payload,
                lambda choice: choice.get("text"),
            )
            total_time = time.perf_counter() - start_time
            t_end_s = time.time()
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0.0
            return InferenceResult(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_s=ttft_s if ttft_s is not None else 0.0,
                total_s=total_time,
                tokens_per_second=tokens_per_second,
                t_start_s=t_start_s,
                t_end_s=t_end_s,
                finish_reason=finish_reason,
                itl_gaps_ms=itl_gaps_ms,
            )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/completions",
                    json=payload,
                )

                if response.status_code != 200:
                    raise VLLMUnavailableError(
                        f"Completion request failed: HTTP {response.status_code}",
                        status_code=response.status_code,
                    )

                total_time = time.perf_counter() - start_time
                data = response.json()

                # Extract token counts from OpenAI-compatible response
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                choices = data.get("choices", [])
                finish_reason = choices[0].get("finish_reason") if choices else None
                t_end_s = time.time()

                # TTFT is deferred (not available without streaming)
                ttft_s = 0.0

                tokens_per_second = (
                    completion_tokens / total_time if total_time > 0 else 0.0
                )

                return InferenceResult(
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    ttft_s=ttft_s,
                    total_s=total_time,
                    tokens_per_second=tokens_per_second,
                    t_start_s=t_start_s,
                    t_end_s=t_end_s,
                    finish_reason=finish_reason,
                )

        except httpx.TimeoutException as e:
            raise VLLMTimeoutError(
                f"Completion request timed out after {self.timeout}s"
            ) from e
        except httpx.RequestError as e:
            raise VLLMUnavailableError(
                f"unreachable at {self.base_url}: {e}"
            ) from e
