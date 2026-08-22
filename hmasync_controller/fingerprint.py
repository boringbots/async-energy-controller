"""
fingerprint — this box's hardware-CLASS fingerprint, for bench_nodes registration.

Collected on `register` (when opted into bench) and by the standalone `bench
register-node` command (cli.py). Fields:

    node_hash        a stable, opaque id — see below. The ONLY thing derived
                      from the salt that is ever sent.
    gpu_name          via the profiler's OWN NVML handle, and ONLY when the
    driver_version    active backend is NVMLProfiler — the nvidia-smi fallback
    vram_gb           is deliberately not read for this (profiler.NVMLProfiler
                      .device_fingerprint does the actual NVML calls).
    cpu_model         from /proc/cpuinfo's `model name`
    ram_gb            from /proc/meminfo's `MemTotal`

Every field but node_hash is best-effort: a channel this box cannot read (no
GPU, /proc missing — e.g. non-Linux) is omitted, never guessed, matching the
profiler's own null-not-fabricated contract.

**Identity, not fingerprinting-for-tracking.** `node_hash` exists so the server
can recognize "the same box" across submissions without ever learning what that
box is. The salt is a random value generated once on first use and persisted
locally (see `load_or_create_salt`); node_hash is its hash and is the only
thing that leaves the box — the salt itself is never transmitted (per
config.BENCH_CONSENT_TEXT). Deleting the salt file resets this box's identity.

This module never reads GPU UUIDs, serial numbers, MAC addresses, hostnames, or
Home Assistant entity ids — those simply never enter a fingerprint dict, so
there is nothing here for bench.denylisted_keys to catch (that check exists for
externally-generated bench bundles; this module's own payload is denylist-clean
by construction, which is what tests/test_fingerprint.py asserts).
"""

from __future__ import annotations

import hashlib
import secrets
from pathlib import Path
from typing import Any

from hmasync_controller.profiler import NVMLProfiler, Profiler

# Module-level so tests can point them at a fixture file, same pattern as
# profiler.py's _INTEL_RAPL / _AMD_RAPL.
PROC_CPUINFO = Path("/proc/cpuinfo")
PROC_MEMINFO = Path("/proc/meminfo")


def read_cpu_model(path: Path = PROC_CPUINFO) -> str | None:
    """The `model name` line from /proc/cpuinfo, or None (missing/unreadable/non-Linux)."""
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            _, _, value = line.partition(":")
            value = value.strip()
            return value or None
    return None


def read_ram_gb(path: Path = PROC_MEMINFO) -> float | None:
    """Total RAM in GB from /proc/meminfo's `MemTotal` (kB), or None."""
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) < 2:
                return None
            try:
                kb = float(parts[1])
            except ValueError:
                return None
            return round(kb / (1024 * 1024), 1)
    return None


def read_gpu_fields(profiler: Profiler) -> dict[str, Any]:
    """gpu_name/driver_version/vram_gb via the profiler's own NVML handle.

    Deliberately restricted to the NVMLProfiler backend — the nvidia-smi
    fallback is not read here. No GPU (or NVML unavailable) degrades to an
    empty dict, never a fabricated value.
    """
    if not isinstance(profiler, NVMLProfiler):
        return {}
    try:
        return profiler.device_fingerprint()
    except Exception:
        return {}


def load_or_create_salt(path: str | Path) -> str:
    """This box's random local identity seed — generated once, never transmitted.

    An existing file is read as-is, so node_hash stays stable across restarts.
    A missing/empty one gets a fresh 32-byte hex token. Deleting the file resets
    this box's identity on the next call.
    """
    p = Path(path)
    if p.exists():
        existing = p.read_text().strip()
        if existing:
            return existing
    salt = secrets.token_hex(32)
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(salt + "\n")
    return salt


def compute_node_hash(salt: str) -> str:
    """A stable, opaque id for this box — the only value derived from the salt
    that is ever sent. The salt itself never leaves this function's callers."""
    return hashlib.sha256(f"hmasync-node:{salt}".encode()).hexdigest()


def collect_fingerprint(profiler: Profiler, salt_path: str | Path) -> dict[str, Any]:
    """Build the payload `register`/`bench register-node` POST to upsert bench_nodes.

    Every field but node_hash is best-effort (see module docstring); node_hash
    is always present.
    """
    salt = load_or_create_salt(salt_path)
    payload: dict[str, Any] = {"node_hash": compute_node_hash(salt)}

    # Passed explicitly (rather than relying on read_cpu_model/read_ram_gb's own
    # defaults, which bind at function-DEFINITION time) so a test's
    # monkeypatch.setattr(fingerprint, "PROC_CPUINFO", ...) is honored here too.
    cpu_model = read_cpu_model(PROC_CPUINFO)
    if cpu_model:
        payload["cpu_model"] = cpu_model
    ram_gb = read_ram_gb(PROC_MEMINFO)
    if ram_gb is not None:
        payload["ram_gb"] = ram_gb

    payload.update(read_gpu_fields(profiler))
    return payload
