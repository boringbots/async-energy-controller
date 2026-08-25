"""Submission bundle construction (US-MERGE-04): `bench quick`'s
`--share-out`-equivalent bundle output.

Ported from energy-bench's `export.py` (`eb export --bundle`), but adapted
to this package's shape in two ways:

1. **In-memory, not DB-backed.** energy-bench's `build_bundle` assembles a
   bundle from already-filtered DuckDB rows (dicts) across possibly many
   hosts (`nodes_by_host`). This package has no DuckDB and measures exactly
   ONE box per `bench quick` invocation, so `build_bundle` here takes a
   list of freshly-computed `RunMetrics` dataclass instances (converted via
   `dataclasses.asdict`) and a single node fingerprint dict.
2. **Grades/load_profiles/baselines are always empty.** `bench quick` never
   touches a grades table, a load-profiles table, or a baselines table --
   this package has none of the three. They still appear as empty arrays in
   every bundle, since the vendored schema (`bench/submission.py::SCHEMA_PATH`)
   requires the `grades`/`load_profiles` keys (though not `baselines`,
   which is optional).

Privacy stays three independent layers, same as energy-bench:

1. **Allowlist by construction.** `_export_run`/`_export_node` each
   hand-pick exactly which fields cross into the bundle -- never a
   wholesale dict passthrough. `run_id`/`label` are never in
   `_RUN_EXPORT_FIELDS`, so `RunMetrics.run_id`'s embedded free-text label
   (`generate_run_id`: `f"{label}_{timestamp}_{uuid}"`) never reaches the
   bundle. `target_host` never crosses either -- `node_hash` stands in for
   it everywhere.
2. **Denylist, enforced in code, over the CONSTRUCTED bundle.**
   `_check_denylist` walks every key/value in the assembled dict
   recursively -- deliberately redundant with (1): a field added straight
   to an allowlist later, without re-reading this docstring, still gets
   caught here before it ever reaches disk. This is a SEPARATE, stricter
   check from `bench.submission.denylisted_keys` (used again at SUBMIT
   time, `# noqa` intentional duplication) -- this one also blocks
   `token`/`password` substrings (guarded by `_LLM_TOKEN_FIELD_ALLOWLIST`
   for the LLM-token metric field names that would otherwise false-positive
   on "token"), which the submit-time check does not.
3. **Schema validation** happens at the CALLER's discretion, via
   `bench.submission.validate_bundle` (this package's own hand-rolled
   JSON-Schema-subset validator against the vendored schema copy -- no
   `jsonschema` dependency; see that module's docstring) -- not called from
   here, so `build_bundle` stays usable in a test that wants to inspect an
   intentionally-invalid bundle before validation.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime, timezone

from hmasync_controller.bench.metrics.models import RunMetrics

BUNDLE_SCHEMA_VERSION = "2"
"""Version of the *bundle format* (`bench/submission.py::SCHEMA_PATH`)
itself -- independent of `RunMetrics.schema_version` (the local run schema,
carried per-run inside the bundle as its own field). Mirrors energy-bench's
`export.BUNDLE_SCHEMA_VERSION`; kept in sync by hand, same as the vendored
schema file itself."""

# This package's own Home Assistant awareness is zero (GROUND TRUTH: no HA
# anywhere in this package) -- this regex exists purely so a bundle built
# here and one built by energy-bench apply the identical redaction rule,
# in case a future caller feeds this module a node/run dict sourced from
# there. Anchored to the `sensor.` domain specifically (energy-bench's own
# HA integration only ever names `sensor.*` entities) to avoid false-positive
# on legitimate dotted identifiers this bundle needs to carry, e.g. the
# engine name "llama.cpp".
_HA_ENTITY_ID_RE = re.compile(r"^sensor\.[a-z0-9_]+$", re.IGNORECASE)

DENYLIST_KEY_SUBSTRINGS = ("uuid", "serial", "mac", "hostname", "entity_id", "token", "password")

# The "token" denylist term is aimed at credentials (api_token, ha_token,
# access_token, ...); it collides with this project's own vocabulary, where
# "token" pervasively means an LLM token, not a secret --
# `joules_per_token` is a headline metric. Exact-name allowlist for the
# fields that are known, by direct inspection, to be LLM-token metrics
# rather than credentials. Any OTHER field containing "token" -- including a
# new field nobody added here -- still trips the denylist below.
_LLM_TOKEN_FIELD_ALLOWLIST = frozenset(
    {
        "max_tokens",
        "joules_per_token",
        "total_completion_tokens",
        "mean_tokens_per_second",
        "pooled_tokens_per_second",
        "alpha_j_per_prompt_token",
        "beta_j_per_completion_token",
    }
)


class ExportDenylistViolation(Exception):
    """A constructed bundle failed the redaction denylist check."""


def _check_denylist(value: object, path: str = "bundle") -> None:
    """Recursively assert `value` (a JSON-shaped dict/list/scalar tree)
    carries no denylisted field name or Home-Assistant-entity-id-shaped
    string value anywhere. Raises `ExportDenylistViolation` on the first
    match found."""
    if isinstance(value, dict):
        for key, sub_value in value.items():
            lowered = str(key).lower()
            is_allowlisted_token_field = lowered in _LLM_TOKEN_FIELD_ALLOWLIST
            if not is_allowlisted_token_field and any(
                term in lowered for term in DENYLIST_KEY_SUBSTRINGS
            ):
                raise ExportDenylistViolation(
                    f"Field '{path}.{key}' matches the export denylist "
                    f"({'/'.join(DENYLIST_KEY_SUBSTRINGS)}) -- never allowed in a "
                    "submission bundle."
                )
            _check_denylist(sub_value, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_denylist(item, f"{path}[{index}]")
    elif isinstance(value, str) and _HA_ENTITY_ID_RE.match(value):
        raise ExportDenylistViolation(
            f"Value at '{path}' looks like a Home Assistant entity id ('{value}') "
            "-- never allowed in a submission bundle."
        )


def _bundle_config_key(
    model: str,
    quantization: str | None,
    engine: str,
    node_hash: str | None,
    power_limit_w: int | None,
) -> str:
    """Ported for parity with energy-bench's `export._bundle_config_key`
    (a future `bench calibrate`/grades feature would need it) -- unused by
    `build_bundle` today, since this package never builds a `grades` entry
    (see module docstring). `node_hash` stands in for `target_host`, same
    reasoning as everywhere else in this module: the raw host never crosses
    into a submission."""
    power_part = str(power_limit_w) if power_limit_w is not None else "stock"
    return f"{model}|{quantization or '-'}|{engine}|{node_hash or 'unknown'}|{power_part}"


_NODE_EXPORT_FIELDS = ("cpu_model", "mem_total_mib", "ram_gb", "pcie_gen")

# energy-bench's export._RUN_EXPORT_FIELDS, minus "efficiency_index": that
# field is filled lab-side by `eb reindex-efficiency` against a DuckDB
# reference run this package never has (see bench/metrics/models.py's
# module docstring) -- there is nothing to export because this package's
# RunMetrics dataclass has no such field to begin with, so it is omitted
# entirely (a real value of `null` would misleadingly suggest "computed but
# absent" rather than "never applicable here"). `ambient_c_start`/
# `ambient_rh_pct_start`/`truncated_pct` are likewise absent -- they are not
# part of energy-bench's own `_RUN_EXPORT_FIELDS` either, so this is not a
# narrowing specific to this package.
_RUN_EXPORT_FIELDS = (
    "schema_version",
    "engine",
    "engine_version",
    "driver_version",
    "cuda_version",
    "gpu_name",
    "has_vision_tower",
    "power_limit_w",
    "clock_lock_mhz",
    "temperature",
    "max_tokens",
    "seed",
    "n_shot",
    "thinking_mode",
    "dataset_revision",
    "model",
    "quantization",
    "task",
    "task_shape",
    "is_canary",
    "n_items",
    "n_correct",
    "accuracy",
    "joules_per_token",
    "joules_per_item",
    "joules_per_correct_answer",
    "jpc_ci_low",
    "jpc_ci_high",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "total_joules_gpu",
    "total_joules_gpu_counter",
    "total_joules_gpu_best",
    "energy_source",
    "counter_vs_integration_pct_diff",
    "total_joules_cpu",
    "total_joules_cpu_dram",
    "kwh_delta",
    "wall_counter_ticks",
    "mean_tokens_per_second",
    "pooled_tokens_per_second",
    "total_completion_tokens",
    "streaming_used",
    "ttft_p50_s",
    "ttft_p95_s",
    "itl_mean_ms",
    "itl_p95_ms",
    "run_duration_s",
    "peak_gpu_w",
    "mean_gpu_w",
    "gpu_temp_c_start",
    "gpu_temp_c_end",
    "peak_gpu_temp_c",
    "mean_gpu_temp_c",
    "peak_gpu_mem_used_mib",
    "mean_gpu_mem_used_mib",
    "gpu_mem_used_pct_of_total",
    "mean_gpu_util_pct",
    "mean_gpu_mem_util_pct",
    "mean_gpu_sm_clock_mhz",
    "mean_gpu_fan_pct",
    "thermal_throttle_pct",
    "power_cap_throttle_pct",
    "mean_wall_w",
    "peak_wall_w",
    "node_overhead_ratio",
    "measurement_tier",
    "alpha_j_per_prompt_token",
    "beta_j_per_completion_token",
    "e_fixed_j",
    "costmodel_r2",
    "costmodel_n",
    "gpu_power_std_w",
    "gpu_power_cv",
    "gpu_power_crest_factor",
    "gpu_power_p95_p50_ratio",
    "gpu_power_jaggedness_w_per_s",
)


def _export_node(node: dict[str, object]) -> dict[str, object] | None:
    """One node fingerprint -> its bundle entry, or None when it has no
    `node_hash` -- a node entry keyed by nothing can't be joined against
    anyway (mirrors energy-bench's `export._export_node`).

    `ram_gb` is coerced to an int: `fingerprint.read_ram_gb()` returns a
    float rounded to 1 decimal (e.g. `31.9`), but the vendored schema's
    node def types it `["integer", "null"]`. This coercion is local to the
    BUNDLE shape -- `fingerprint.py` itself is untouched, so
    `bench register-node`'s own path (which sends the float form) is
    unaffected.
    """
    node_hash = node.get("node_hash")
    if not node_hash:
        return None
    exported: dict[str, object] = {"node_hash": node_hash}
    for field_name in _NODE_EXPORT_FIELDS:
        value = node.get(field_name)
        if field_name == "ram_gb" and value is not None:
            value = round(value)
        exported[field_name] = value
    return exported


def _export_run(run: RunMetrics, node_hash: str | None, created_at: str) -> dict[str, object]:
    """One computed `RunMetrics` -> its bundle entry.

    `created_at` is not a field on the plain `RunMetrics` dataclass (unlike
    energy-bench's DB-backed `runs` rows, which get it from an INSERT
    timestamp) -- the caller stamps it once per `build_bundle` call and it
    is applied uniformly to every run in the bundle, which is accurate
    enough: a `bench quick` run's bundle is built immediately after the
    suite finishes, not read back from storage hours or days later.
    """
    run_dict = asdict(run)
    exported: dict[str, object] = {
        "node_hash": node_hash,
        "created_at": created_at,
        "repeat_index": run_dict.get("repeat_index", 0),
    }
    exported.update({field_name: run_dict.get(field_name) for field_name in _RUN_EXPORT_FIELDS})
    return exported


def build_bundle(
    runs: list[RunMetrics],
    node: dict[str, object] | None,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Assemble the submission bundle from a `bench quick` run's freshly
    computed `RunMetrics` and this box's own fingerprint dict.

    Pure -- no I/O, no DuckDB (this package has none). `node` is the dict
    `fingerprint.collect_fingerprint()` returns (or `None` if fingerprinting
    failed entirely, in which case the bundle carries no node at all and
    every run's `node_hash` is `None`). `grades`/`load_profiles`/
    `baselines` are always empty arrays -- see module docstring.

    Raises:
        ExportDenylistViolation: If the assembled bundle fails the
            redaction check (defense in depth -- see module docstring).
    """
    stamp = generated_at or datetime.now(timezone.utc).isoformat()
    node_hash = node.get("node_hash") if node else None
    node_hash = str(node_hash) if node_hash else None

    exported_runs = [_export_run(run, node_hash, stamp) for run in runs]

    exported_nodes: list[dict[str, object]] = []
    if node is not None:
        exported_node = _export_node(node)
        if exported_node is not None:
            exported_nodes.append(exported_node)

    bundle: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": stamp,
        "nodes": exported_nodes,
        "runs": exported_runs,
        "grades": [],
        "load_profiles": [],
        "baselines": [],
    }

    _check_denylist(bundle)
    return bundle
