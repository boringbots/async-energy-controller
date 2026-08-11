"""
Controller configuration (the controller authenticates as its owner).

Every value defaults to empty/benign so the package imports and constructs with
an empty `.env` (never block startup on missing credentials). Nothing here opens a socket or reads a file at import time; the
ApiClient and Spool are built lazily by whatever wires them together.
"""

from __future__ import annotations

import socket

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Controller settings, sourced from the environment / a local `.env`.

    The four operator-facing knobs are the ones in `.env.example`:
    `HM_ASYNC_API_URL`, `HM_ASYNC_EMAIL`, `HM_ASYNC_PASSWORD`, `CONTROLLER_ID`.
    The rest have sensible defaults.

    **Every documented knob is a field here.** `extra="ignore"` means an
    undeclared name in `.env` is dropped without a word, so a knob read from
    anywhere else (`os.environ` in particular) is one a `.env` line cannot set —
    the failure is silent and the config file looks correct. If you add a setting,
    add it here; that is the whole reason this class is the single source.
    """

    # --- optimizer API endpoint + owner credentials (JWT + refresh) ---
    HM_ASYNC_API_URL: str = ""
    HM_ASYNC_EMAIL: str = ""
    HM_ASYNC_PASSWORD: str = ""

    # Stable identity for this box; half of the (controller_id, run_id) server-side
    # idempotency key. Empty → resolve_controller_id() falls back to the hostname
    # so a fresh box still has a stable id without extra config.
    CONTROLLER_ID: str = ""

    # Path to the local job catalog (workflow_id → what THIS box runs).
    # Declared here rather than read straight off `os.environ` so it behaves like
    # every other knob: `.env` works, a real exported variable overrides it, and
    # `--job-catalog` overrides both. Reading it straight off `os.environ` instead
    # meant a `.env` line was silently dropped — pydantic-settings reads `.env`
    # into this object, it does NOT export to the process environment, so an
    # undeclared name has nowhere to land.
    HM_ASYNC_JOB_CATALOG: str = "jobs.json"

    # Single local SQLite file backing the offline spool (the only
    # durable state the controller keeps). Relative paths resolve against the
    # controller's working directory.
    SPOOL_PATH: str = "hmasync_spool.db"

    # Per-request HTTP timeout (seconds). Every wire call is bounded so a hung
    # API can never stall the executor loop (sampling/overhead).
    HTTP_TIMEOUT_S: float = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def resolve_controller_id(configured: str | None) -> str:
    """Return a stable controller id, falling back to the hostname when unset.

    `CONTROLLER_ID` is the operator's explicit choice; when it's empty we use the
    machine hostname so the id is stable across restarts (a random id per boot
    would defeat the server-side idempotency key). Never returns an empty string.
    """
    if configured:
        return configured
    host = socket.gethostname()
    return host or "hm-async-controller"


# A module-level instance is convenient for callers, but constructing your own
# `Settings()` (e.g. in tests) is equally valid — nothing here is a singleton by
# necessity.
settings = Settings()
