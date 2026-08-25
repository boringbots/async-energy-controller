"""
submission — validate, redact-check, and store-and-forward a bench submission bundle.

`bench quick` (cli.py) writes a bundle file (US-MERGE-04 replaces the old
energy-bench subprocess hand-off with an in-process call). This module is
what stands between that file and the wire: it validates the bundle against
the vendored copy of energy-bench's submission schema, refuses anything
carrying a denylisted key (defense in depth alongside the schema's own
`additionalProperties: false`), and — on a network failure — spools it using
the same pattern as spool.py, in its own SQLite file so a stuck bench
submission never blocks or mixes into the run-report spool.

**Not a `jsonschema` dependency.** The package is deliberately a thin API
client (see pyproject's dependency comment); the JSON-Schema subset the
vendored file actually uses (type/properties/required/additionalProperties/
items/$ref/const/enum) is small enough to walk directly against the schema
loaded from disk, so validation can never drift from what is vendored.

Formerly `hmasync_controller/bench.py`; moved under `bench/` (US-MERGE-01)
when that name became a package (tasks/, and the rest of the ported
benchmark suite). Its public names are re-exported from `bench/__init__.py`
so every existing `from hmasync_controller import bench; bench.<name>` /
`from hmasync_controller.bench import <name>` call site is unaffected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hmasync_controller.apiclient import ApiClient
from hmasync_controller.spool import Spool

# Vendored copy of energy-bench's bench_submission.schema.json. Kept in sync by
# hand (`cp` from the energy-bench repo) and checked for drift by
# tests/test_bench.py, which skips when no local energy-bench checkout is
# present to compare against.
SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "bench_submission.schema.json"

# Key names that must never appear anywhere in a submission, checked
# independently of schema validation — a redaction net that still catches a
# hand-edited or future-schema bundle even if it slips past a shape check.
DENYLISTED_KEY_SUBSTRINGS = ("uuid", "serial", "mac", "hostname", "entity_id")


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


# --- schema validation (small, generic JSON-Schema-subset validator) -------


def validate_bundle(bundle: Any, schema: dict[str, Any] | None = None) -> list[str]:
    """Validate `bundle` against the vendored schema. Empty list = valid."""
    schema = schema if schema is not None else load_schema()
    errors: list[str] = []
    _validate(bundle, schema, schema.get("$defs", {}), "$", errors)
    return errors


def _validate(instance: Any, node: dict[str, Any], defs: dict[str, Any], path: str, errors: list[str]) -> None:
    if "$ref" in node:
        node = _resolve_ref(node["$ref"], defs)

    if "const" in node and instance != node["const"]:
        errors.append(f"{path}: expected {node['const']!r}, got {instance!r}")
        return
    if "enum" in node and instance not in node["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {node['enum']!r}")
        return

    types = node.get("type")
    if types is not None:
        expected = types if isinstance(types, list) else [types]
        if not any(_type_matches(instance, t) for t in expected):
            errors.append(f"{path}: expected type {expected!r}, got {_json_type(instance)!r}")
            return

    if isinstance(instance, dict) and "properties" in node:
        properties = node["properties"]
        for key in node.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        if node.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{path}: unexpected property {key!r}")
        for key, sub_schema in properties.items():
            if key in instance:
                _validate(instance[key], sub_schema, defs, f"{path}.{key}", errors)

    if isinstance(instance, list) and "items" in node:
        for i, item in enumerate(instance):
            _validate(item, node["items"], defs, f"{path}[{i}]", errors)


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    # Only local "#/$defs/name" refs appear in this schema.
    return defs[ref.rsplit("/", 1)[-1]]


_TYPE_MAP = {"object": dict, "array": list, "string": str, "boolean": bool}


def _type_matches(instance: Any, expected: str) -> bool:
    if expected == "null":
        return instance is None
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    py_type = _TYPE_MAP.get(expected)
    return py_type is not None and isinstance(instance, py_type)


def _json_type(instance: Any) -> str:
    if instance is None:
        return "null"
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    if isinstance(instance, str):
        return "string"
    if isinstance(instance, list):
        return "array"
    if isinstance(instance, dict):
        return "object"
    return type(instance).__name__  # pragma: no cover - defensive


# --- redaction --------------------------------------------------------


def denylisted_keys(bundle: Any, *, _path: str = "$") -> list[str]:
    """Every dict key anywhere in `bundle` that contains a denylisted substring."""
    found: list[str] = []
    if isinstance(bundle, dict):
        for key, value in bundle.items():
            if any(tok in str(key).lower() for tok in DENYLISTED_KEY_SUBSTRINGS):
                found.append(f"{_path}.{key}")
            found.extend(denylisted_keys(value, _path=f"{_path}.{key}"))
    elif isinstance(bundle, list):
        for i, item in enumerate(bundle):
            found.extend(denylisted_keys(item, _path=f"{_path}[{i}]"))
    return found


# --- submit + spool -----------------------------------------------------


@dataclass
class SubmitResult:
    """Outcome of submitting one bundle file."""

    ok: bool
    spooled: bool = False
    message: str = ""


@dataclass
class BenchDrainResult:
    """Outcome of draining the bench spool."""

    drained: int
    remaining: int
    stopped_early: bool = False


def _load_bundle(path: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    p = Path(path)
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return None, f"could not read {p}: {exc}"
    if not isinstance(data, dict):
        return None, f"{p} is not a JSON object"
    return data, None


def submit_bundle_file(
    client: ApiClient,
    spool: Spool,
    bundle_path: str | Path,
    *,
    schema: dict[str, Any] | None = None,
) -> SubmitResult:
    """Validate, redaction-check, then submit one bundle file; spool on outage.

    A read failure, a schema violation, or a denylisted key all refuse LOCALLY
    (no request is sent) — none is retryable by spooling, since resubmitting
    the same bad bundle would fail the exact same way every time. A network
    failure is the one outcome that DOES spool, since a later retry can help.

    Opportunistically drains anything already queued first, so a string of
    manual `bench submit` calls (or repeated opt-in `bench quick` hand-offs)
    keeps the queue from growing unbounded between daemon ticks.
    """
    drain = drain_bench_spool(client, spool)

    bundle, err = _load_bundle(bundle_path)
    if err:
        return SubmitResult(ok=False, message=err)

    # Redaction before shape: every object in the vendored schema already sets
    # additionalProperties: false, so a denylisted key would usually also be an
    # "unexpected property" schema error — but the redaction check exists
    # precisely to not depend on that (a future looser schema, or a bug in the
    # hand-rolled validator, must not open a path for one of these to slip
    # through), so it is checked first and reported on its own terms.
    denylisted = denylisted_keys(bundle)
    if denylisted:
        return SubmitResult(
            ok=False,
            message="refusing to submit — denylisted key(s) present: " + ", ".join(denylisted),
        )

    errors = validate_bundle(bundle, schema)
    if errors:
        return SubmitResult(
            ok=False,
            message="bundle failed schema validation:\n  " + "\n  ".join(errors),
        )

    result = client.submit_bench_bundle(bundle)
    if result.ok:
        note = f" ({drain.drained} queued bundle(s) also flushed)" if drain.drained else ""
        return SubmitResult(ok=True, message=f"submitted {Path(bundle_path).name}{note}")

    if result.transport_error:
        spool.enqueue({"bundle": bundle}, [])
        return SubmitResult(
            ok=False, spooled=True,
            message=f"API unreachable ({result.error}); bundle spooled for retry",
        )

    return SubmitResult(ok=False, message=f"submission refused: {result.error}")


def drain_bench_spool(client: ApiClient, spool: Spool, max_items: int | None = None) -> BenchDrainResult:
    """Flush queued bundles in FIFO order; stop at the first still-failing push.

    Mirrors reporter.RunReporter.drain_spool's stop-early behaviour: safe to
    call repeatedly, and a malformed queued item (there is no code path that
    should produce one) is dropped rather than blocking every submission after
    it forever.
    """
    items = spool.pending(limit=max_items)
    drained = 0
    for item in items:
        bundle = item.record.get("bundle") if isinstance(item.record, dict) else None
        if not isinstance(bundle, dict):
            spool.remove(item.id)
            continue
        result = client.submit_bench_bundle(bundle)
        if not result.ok:
            return BenchDrainResult(drained=drained, remaining=spool.count(), stopped_early=True)
        spool.remove(item.id)
        drained += 1
    return BenchDrainResult(drained=drained, remaining=spool.count())
