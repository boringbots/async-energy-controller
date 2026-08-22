"""
Controller configuration (the controller authenticates as its owner).

Every value defaults to empty/benign so the package imports and constructs with
an empty `.env` (never block startup on missing credentials). Nothing here opens a socket or reads a file at import time; the
ApiClient and Spool are built lazily by whatever wires them together.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

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

    # --- bench opt-in (contribute measured benchmark data upstream) ---
    # Off by default: nothing bench-related is ever sent until the operator runs
    # `bench opt-in`, which persists this flag (see set_bench_optin below).
    BENCH_OPTIN: bool = False
    # Where `bench quick` writes its dated bundle files.
    BENCH_BUNDLE_DIR: str = "bench_bundles"
    # The energy-bench CLI command installed on this box. Not on PyPI yet, so
    # this stays a plain command name rather than a package pin.
    ENERGY_BENCH_CMD: str = "eb"
    # Where a bench-bundle submission is queued when the API is unreachable at
    # send time — same store-and-forward pattern as SPOOL_PATH, kept in its own
    # file so a stuck bench submission never blocks or mixes into the
    # run-report spool's drain.
    BENCH_SPOOL_PATH: str = "hmasync_bench_spool.db"

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


# What `bench opt-in` actually turns on. Printed verbatim by the CLI before the
# flag is set, so consent is informed rather than a name on a flag nobody read.
BENCH_CONSENT_TEXT = """\
Opting in shares, per benchmark submission:
  - a hardware fingerprint: GPU model name, VRAM (GB), driver version, CPU
    model, and RAM (GB)
  - software versions: this controller and energy-bench
  - benchmark metrics: energy (Wh), duration, throughput, and related numbers
    produced by the suite

It never shares prompts, commands, workflow definitions, or any workflow data.
It never shares GPU UUIDs, serial numbers, MAC addresses, hostnames, or Home
Assistant entity ids — this box is identified only by a salted local hash,
generated once and never transmitted in raw form.

Data license: submitted results feed Async Energy's shared routing-table and
cold-start-prediction aggregates. Opting out stops future submissions; it does
not withdraw data already submitted.
"""


def set_bench_optin(value: bool, env_path: str | os.PathLike[str] = ".env") -> None:
    """Persist BENCH_OPTIN by upserting its line in the `.env` file.

    Follows the same care as `cli.write_catalog_entry`: every other line
    (including an operator's comments) is preserved untouched, and the write is
    atomic (temp file + `os.replace`) so a running process never reads a
    half-written `.env`. A missing file is created with just this one line —
    the empty-environment policy means every other value still resolves from
    its own default or a real environment variable.
    """
    path = Path(env_path)
    lines = path.read_text().splitlines() if path.exists() else []
    new_line = f"BENCH_OPTIN={'true' if value else 'false'}"

    out: list[str] = []
    updated = False
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key == "BENCH_OPTIN":
            out.append(new_line)
            updated = True
        else:
            out.append(line)
    if not updated:
        out.append(new_line)

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n")
    os.replace(tmp, path)
