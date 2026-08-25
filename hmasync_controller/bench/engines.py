"""Engine adapters `bench quick` attaches to: Ollama and llama.cpp's
`llama-server` (US-MERGE-04). Both ported from energy-bench's
`engines/ollama_adapter.py` and `engines/llamacpp_adapter.py` -- ATTACH MODE
ONLY, which is the whole surface this package needs.

energy-bench's `LlamaCppAdapter` has a second, Docker-launch mode
(`LlamaCppServerManager`: inspect/recreate a container via the `docker`
package) that this package deliberately does NOT port. This package has no
`docker` dependency and never will -- it measures whatever server an
operator already runs, exactly like `OllamaAdapter` always has (Ollama has
no launch mode here either; see that class's own docstring). The class below
is named `AttachLlamaCppAdapter` rather than reusing `LlamaCppAdapter`'s bare
name, so a future reader never has to check which mode it is -- there is
only one.

Neither adapter performs I/O at construction (matches
`bench.sampler.LocalNvmlSampler`'s convention) -- a network call only
happens on `health()`/`ready()`/`version()`/`verify_model_pulled()`.
"""

import logging

import httpx

from hmasync_controller.bench.vllm_client import VLLMClient, VLLMUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_PORT = 11434
DEFAULT_LLAMACPP_PORT = 8080


class OllamaModelNotPulledError(Exception):
    """Raised when `POST /api/show` reports a model is not pulled.

    The message names the exact `ollama pull <model>` command -- this
    adapter never runs it itself (pulling a model is an explicit operator
    act, never something that happens inside a measured window).
    """


class OllamaAdapter:
    """Attach-only adapter for a running Ollama server.

    Ollama has no `GET /health` -- its documented liveness probe is
    `GET /api/version`, which responds whether or not any model is loaded.
    `launch`/`swap`/`stop` are permanent no-ops: Ollama's own model
    lifecycle goes through `ollama pull`/`ollama rm`, never through this
    package.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_OLLAMA_PORT,
        *,
        base_url: str | None = None,
    ) -> None:
        self._base_url = base_url or f"http://{host}:{port}"

    async def launch(self, model_cfg: object) -> bool:
        """No-op -- this adapter never manages the server. Returns False."""
        return False

    async def health(self) -> None:
        """Raises `VLLMUnavailableError` if Ollama is unreachable.

        Polls `GET /api/version` rather than `/health` -- see class
        docstring.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._base_url}/api/version")
        except httpx.TimeoutException as e:
            raise VLLMUnavailableError(
                f"Ollama health check timed out at {self._base_url}"
            ) from e
        except httpx.RequestError as e:
            raise VLLMUnavailableError(f"Ollama unreachable at {self._base_url}: {e}") from e

        if response.status_code != 200:
            raise VLLMUnavailableError(
                f"Ollama health check failed: HTTP {response.status_code}",
                status_code=response.status_code,
            )

    async def ready(self) -> bool:
        """Best-effort readiness check: True if `health()` doesn't raise."""
        try:
            await self.health()
            return True
        except Exception:  # noqa: BLE001 - not ready is a normal pre-launch state
            return False

    async def swap(self, model_cfg: object) -> bool:
        """No-op -- Ollama's own model lifecycle is out of scope for this
        adapter (see class docstring). Returns False."""
        return False

    async def stop(self) -> bool:
        """No-op -- this adapter never stops the operator's server. Returns False."""
        return False

    async def version(self) -> str | None:
        """Ollama's own version from `GET /api/version`.

        Best-effort like `VLLMClient.get_version()` -- returns None on any
        error rather than raising, so a version lookup never aborts a run.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._base_url}/api/version")
                if response.status_code != 200:
                    return None
                return response.json().get("version")
        except Exception:
            return None

    def base_url(self) -> str:
        return self._base_url

    def launch_args(self) -> list[str]:
        """Always empty -- this adapter never launches anything to have args."""
        return []

    async def verify_model_pulled(self, model: str) -> str | None:
        """Pre-flight gate: confirm `model` is already pulled via `POST /api/show`.

        Args:
            model: The Ollama model tag (e.g. `"qwen2.5:7b"`).

        Returns:
            A best-effort "runner backend" label -- `details.family` from
            the `/api/show` response, or None if the response doesn't carry
            one.

        Raises:
            OllamaModelNotPulledError: If `model` is not pulled -- the
                message names the exact `ollama pull <model>` command.
            VLLMUnavailableError: If the Ollama server itself is
                unreachable.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/show", json={"model": model}
                )
        except httpx.TimeoutException as e:
            raise VLLMUnavailableError(
                f"Ollama unreachable at {self._base_url}: timed out"
            ) from e
        except httpx.RequestError as e:
            raise VLLMUnavailableError(f"Ollama unreachable at {self._base_url}: {e}") from e

        if response.status_code == 404:
            raise OllamaModelNotPulledError(
                f"Model '{model}' is not pulled on this Ollama server "
                f"({self._base_url}). Run `ollama pull {model}` on the target "
                f"box first -- this package never pulls a model itself "
                f"(security standing rule: model pulls are an explicit "
                f"pre-flight operator act, never inside a measured window)."
            )
        if response.status_code != 200:
            raise VLLMUnavailableError(
                f"Ollama /api/show failed: HTTP {response.status_code}",
                status_code=response.status_code,
            )

        data = response.json()
        details = data.get("details") or {}
        family = details.get("family")
        return str(family) if family else None


class AttachLlamaCppAdapter:
    """Attach-only adapter for a running `llama-server`.

    Unlike `OllamaAdapter`, `GET /health` here is an honest readiness
    signal (503 while loading, 200 once serving), and health/models reuse
    `VLLMClient` since llama-server implements those two endpoints in the
    same OpenAI-compatible shape. `launch`/`swap`/`stop` are permanent
    no-ops -- there is nothing to launch, swap, or stop on a server this
    process does not own (energy-bench's Docker-launch mode for this same
    engine is deliberately NOT ported here; see module docstring).
    """

    def __init__(self, host: str, port: int = DEFAULT_LLAMACPP_PORT, *, base_url: str | None = None) -> None:
        self._base_url = base_url or f"http://{host}:{port}"
        # VLLMClient is reused for health/models -- see class docstring. It
        # only builds base_url from host/port at construction, so an
        # explicit base_url is patched in afterward.
        self._client = VLLMClient(host, port)
        self._client.base_url = self._base_url

    async def launch(self, model_cfg: object) -> bool:
        """No-op -- attach mode never launches anything. Returns False."""
        return False

    async def health(self) -> None:
        """Raises if the server is unreachable or unhealthy."""
        await self._client.health_check()

    async def ready(self) -> bool:
        """Best-effort readiness check: True if the health endpoint responds."""
        try:
            await self._client.health_check()
            return True
        except Exception:  # noqa: BLE001 - not ready is a normal pre-launch state
            return False

    async def swap(self, model_cfg: object) -> bool:
        """No-op -- the operator's server is not ours to reconfigure. Returns False."""
        return False

    async def stop(self) -> bool:
        """No-op -- this adapter never stops the operator's server. Returns False."""
        return False

    async def version(self) -> str | None:
        """llama-server's version, best-effort from `GET /props` (whichever
        version-ish field it reports -- the endpoint is not stably
        schematized across llama.cpp releases). None on any failure; unlike
        energy-bench's Docker-mode adapter there is no pinned image digest
        to fall back to in attach mode."""
        return await self._fetch_props_version()

    async def _fetch_props_version(self) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._base_url}/props")
                if response.status_code != 200:
                    return None
                data = response.json()
        except Exception:  # noqa: BLE001 - best-effort, never raises
            return None

        for key in ("build_info", "version", "commit"):
            value = data.get(key)
            if value:
                return str(value)
        return None

    def base_url(self) -> str:
        return self._base_url

    def launch_args(self) -> list[str]:
        """Always empty -- attach mode never launches anything to have args."""
        return []
