"""
Tests for hmasync_controller/bench.py — schema validation, redaction, and the
bench-bundle submit/spool/drain flow.

Network is exercised only through the ApiClient + FakeApiServer fixtures
already shared with test_apiclient.py/test_cli.py (httpx.MockTransport) — no
live vendor call.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from hmasync_controller import bench
from hmasync_controller.spool import Spool


def _minimal_bundle(**overrides) -> dict:
    bundle = {
        "schema_version": "1",
        "generated_at": "2026-08-22T00:00:00Z",
        "nodes": [],
        "runs": [],
        "grades": [],
        "load_profiles": [],
    }
    bundle.update(overrides)
    return bundle


# --- validate_bundle --------------------------------------------------


def test_minimal_v1_bundle_is_valid():
    assert bench.validate_bundle(_minimal_bundle()) == []


def test_minimal_v2_bundle_is_valid():
    # v2's additions (CI fields, total_joules_cpu_dram, baselines) are all
    # optional per the schema, so a bundle carrying nothing but the bumped
    # schema_version — no baselines array, no new run fields — must still
    # validate exactly like a v1 bundle does.
    assert bench.validate_bundle(_minimal_bundle(schema_version="2")) == []


def test_v2_bundle_with_baselines_is_valid():
    bundle = _minimal_bundle(
        schema_version="2",
        baselines=[
            {
                "node_hash": "abc123",
                "kind": "loaded",
                "model": "some-model",
                "mean_gpu_w": 45.0,
                "peak_gpu_w": 60.0,
                "gpu_mem_used_mib": 8192.0,
                "duration_s": 120.0,
                "driver_version": "550.1",
            }
        ],
    )
    assert bench.validate_bundle(bundle) == []


def test_bundle_with_suite_calibrate_is_valid():
    # US-MERGE-05: `suite` is an OPTIONAL top-level field -- a bundle from
    # before this addition (no `suite` key at all, like `_minimal_bundle()`)
    # keeps validating unchanged (test_minimal_v2_bundle_is_valid above), and
    # a bundle stamping either allowed value validates too.
    assert bench.validate_bundle(_minimal_bundle(schema_version="2", suite="calibrate")) == []
    assert bench.validate_bundle(_minimal_bundle(schema_version="2", suite="quick")) == []


def test_bundle_with_unknown_suite_value_is_rejected():
    errors = bench.validate_bundle(_minimal_bundle(schema_version="2", suite="full"))
    assert any("suite" in e for e in errors)


def test_missing_required_field_is_reported():
    bundle = _minimal_bundle()
    del bundle["runs"]
    errors = bench.validate_bundle(bundle)
    assert any("runs" in e for e in errors)


def test_unknown_schema_version_is_rejected():
    errors = bench.validate_bundle(_minimal_bundle(schema_version="3"))
    assert any("schema_version" in e for e in errors)


def test_wrong_type_is_reported():
    errors = bench.validate_bundle(_minimal_bundle(nodes="not-a-list"))
    assert any("nodes" in e for e in errors)


def test_unexpected_top_level_property_is_reported():
    errors = bench.validate_bundle(_minimal_bundle(unexpected_field="x"))
    assert any("unexpected_field" in e for e in errors)


def test_nested_run_entry_is_validated():
    bundle = _minimal_bundle(runs=[{"node_hash": "abc"}])  # missing required run fields
    errors = bench.validate_bundle(bundle)
    assert any("engine" in e or "model" in e for e in errors)


# --- denylisted_keys ----------------------------------------------------


def test_clean_bundle_has_no_denylisted_keys():
    assert bench.denylisted_keys(_minimal_bundle()) == []


@pytest.mark.parametrize(
    "bad_key",
    ["gpu_uuid", "serial_number", "mac_address", "hostname", "ha_entity_id"],
)
def test_denylisted_key_is_found_when_nested(bad_key):
    bundle = _minimal_bundle(nodes=[{"node_hash": "abc", bad_key: "leak"}])
    found = bench.denylisted_keys(bundle)
    assert any(bad_key in f for f in found)


def test_denylisted_key_found_inside_a_list_item():
    bundle = _minimal_bundle(runs=[{"node_hash": "abc"}, {"uuid": "leak"}])
    found = bench.denylisted_keys(bundle)
    assert any("uuid" in f for f in found)


# --- submit_bundle_file ---------------------------------------------------


@pytest.fixture
def bench_spool(tmp_path) -> Spool:
    s = Spool(str(tmp_path / "bench_spool.db"))
    yield s
    s.close()


def _write_bundle(tmp_path, name="bundle.json", **overrides) -> Path:
    import json

    path = tmp_path / name
    path.write_text(json.dumps(_minimal_bundle(**overrides)))
    return path


def test_submit_success(tmp_path, make_client, fake_api, bench_spool):
    client = make_client()
    path = _write_bundle(tmp_path)
    result = bench.submit_bundle_file(client, bench_spool, path)
    assert result.ok
    assert not result.spooled
    assert len(fake_api.bench_submissions) == 1
    assert bench_spool.count() == 0


def test_submit_quarantined_2xx_is_success(tmp_path, make_client, fake_api, bench_spool):
    fake_api.bench_submission_status = 202
    fake_api.bench_submission_response = {"status": "quarantined"}
    client = make_client()
    path = _write_bundle(tmp_path)
    result = bench.submit_bundle_file(client, bench_spool, path)
    assert result.ok
    assert not result.spooled


def test_submit_network_failure_spools(tmp_path, make_client, fake_api, bench_spool):
    fake_api.go_down()
    client = make_client()
    path = _write_bundle(tmp_path)
    result = bench.submit_bundle_file(client, bench_spool, path)
    assert not result.ok
    assert result.spooled
    assert bench_spool.count() == 1
    assert fake_api.bench_submissions == []


def test_submit_invalid_bundle_is_refused_without_network_call(tmp_path, make_client, fake_api, bench_spool):
    client = make_client()
    path = _write_bundle(tmp_path, model_size="oops")  # not a real field; additionalProperties: false
    result = bench.submit_bundle_file(client, bench_spool, path)
    assert not result.ok
    assert not result.spooled
    assert "validation" in result.message
    assert fake_api.bench_submissions == []
    assert bench_spool.count() == 0


def test_submit_denylisted_bundle_is_refused_without_network_call(tmp_path, make_client, fake_api, bench_spool):
    import json

    bundle = _minimal_bundle(nodes=[{"node_hash": "abc", "gpu_uuid": "leak"}])
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))

    client = make_client()
    result = bench.submit_bundle_file(client, bench_spool, path)
    assert not result.ok
    assert not result.spooled
    assert "denylisted" in result.message
    assert fake_api.bench_submissions == []


def test_submit_missing_file_is_refused_cleanly(tmp_path, make_client, fake_api, bench_spool):
    client = make_client()
    result = bench.submit_bundle_file(client, bench_spool, tmp_path / "nope.json")
    assert not result.ok
    assert not result.spooled
    assert fake_api.bench_submissions == []


def test_submit_opportunistically_drains_queued_bundle_first(tmp_path, make_client, fake_api, bench_spool):
    # Seed the spool as if an earlier submission had failed.
    bench_spool.enqueue({"bundle": _minimal_bundle()}, [])
    assert bench_spool.count() == 1

    client = make_client()
    path = _write_bundle(tmp_path, name="new-bundle.json")
    result = bench.submit_bundle_file(client, bench_spool, path)

    assert result.ok
    assert bench_spool.count() == 0  # queued one flushed, new one delivered
    assert len(fake_api.bench_submissions) == 2


# --- drain_bench_spool ---------------------------------------------------


def test_drain_bench_spool_flushes_all_pending(make_client, fake_api, bench_spool):
    bench_spool.enqueue({"bundle": _minimal_bundle(generated_at="2026-08-22T00:00:00Z")}, [])
    bench_spool.enqueue({"bundle": _minimal_bundle(generated_at="2026-08-22T01:00:00Z")}, [])
    client = make_client()

    result = bench.drain_bench_spool(client, bench_spool)

    assert result.drained == 2
    assert result.remaining == 0
    assert not result.stopped_early
    assert len(fake_api.bench_submissions) == 2


def test_drain_bench_spool_stops_early_on_continued_outage(make_client, fake_api, bench_spool):
    bench_spool.enqueue({"bundle": _minimal_bundle()}, [])
    bench_spool.enqueue({"bundle": _minimal_bundle()}, [])
    fake_api.go_down()
    client = make_client()

    result = bench.drain_bench_spool(client, bench_spool)

    assert result.drained == 0
    assert result.remaining == 2
    assert result.stopped_early


def test_drain_bench_spool_on_empty_queue_is_a_noop(make_client, fake_api, bench_spool):
    client = make_client()
    result = bench.drain_bench_spool(client, bench_spool)
    assert result == bench.BenchDrainResult(drained=0, remaining=0)


# --- vendored schema drift check ------------------------------------------
#
# The energy-bench repo is a sibling checkout the operator may or may not have
# on this box — it is never a dependency of this repo. Point
# $ENERGY_BENCH_SCHEMA_PATH at its schemas/bench_submission.schema.json to run
# this locally; the test SKIPS (not fails) when unset, so the public suite
# never depends on a second repo's presence.


def test_vendored_schema_matches_energy_bench_source():
    source = os.environ.get("ENERGY_BENCH_SCHEMA_PATH")
    if not source or not Path(source).exists():
        pytest.skip(
            "ENERGY_BENCH_SCHEMA_PATH not set or not found; "
            "drift check needs a local energy-bench checkout"
        )
    vendored = bench.SCHEMA_PATH.read_bytes()
    upstream = Path(source).read_bytes()
    assert hashlib.sha256(vendored).hexdigest() == hashlib.sha256(upstream).hexdigest(), (
        "vendored bench_submission.schema.json has drifted from the energy-bench "
        "source; re-copy it (and review bench.py's validator against the new shape)"
    )
