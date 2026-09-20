"""Which weights ran -- recorded on the row, never guessed.

A quantization NAME is not a set of weights (f3e6f01). These pin that the
Ollama path records the manifest digest actually pulled, the reference path
records the pinned repo/revision it verified the served file against, plain
`bench quick` on llama.cpp records None, and all three ride the bundle.
"""

from __future__ import annotations

import asyncio
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

from hmasync_controller.bench.bundle import _RUN_EXPORT_FIELDS, build_bundle
from hmasync_controller.bench.metrics.models import RunMetrics
from hmasync_controller.bench.quick import (
    QUICK_REFERENCE_MODELS,
    DetectedEngine,
    QuickModel,
    resolve_quick_model,
)
from hmasync_controller.bench.roster import ROSTER


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _adapter(*, digest):
    adapter = MagicMock()
    adapter.verify_model_pulled = AsyncMock(return_value="qwen3")
    if digest is not MagicMock:
        adapter.model_digest = AsyncMock(return_value=digest)
    return adapter


def _fake_run_metrics(**overrides) -> RunMetrics:
    fields = dict(
        run_id="quick_ollama_gsm8k_platinum_20260822_120000_ab12cd34",
        label="quick_ollama_gsm8k_platinum",
        model="Qwen/Qwen3.5-9B",
        quantization="Q4_K_M",
        target_host="localhost",
        joules_per_token=1.5,
        total_joules_gpu=500.0,
        kwh_delta=None,
        peak_gpu_w=300.0,
        mean_gpu_w=250.0,
        mean_tokens_per_second=20.0,
        run_duration_s=30.0,
        engine="ollama",
        engine_version="0.5.7",
        task="gsm8k_platinum",
        task_shape="decode",
        n_items=5,
        n_correct=4,
        accuracy=0.8,
    )
    fields.update(overrides)
    return RunMetrics(**fields)


def test_run_metrics_carries_the_three_provenance_fields():
    names = {f.name for f in fields(RunMetrics)}
    assert {"gguf_repo", "gguf_revision", "weights_digest"} <= names
    assert {"gguf_repo", "gguf_revision", "weights_digest"} <= set(_RUN_EXPORT_FIELDS)


def test_ollama_quick_records_the_digest_actually_pulled():
    adapter = _adapter(digest="sha256:abc123")
    engine = DetectedEngine(name="ollama", base_url="http://localhost:11434", adapter=adapter)
    model = _run(resolve_quick_model(engine))
    adapter.model_digest.assert_awaited_once_with(QUICK_REFERENCE_MODELS["ollama"]["tag"])
    assert model.record_weights_digest == "sha256:abc123"
    assert model.record_gguf_repo is None and model.record_gguf_revision is None


def test_ollama_digest_lookup_failure_records_none_not_a_guess():
    adapter = _adapter(digest=None)
    adapter.model_digest = AsyncMock(side_effect=RuntimeError("tags endpoint exploded"))
    engine = DetectedEngine(name="ollama", base_url="http://localhost:11434", adapter=adapter)
    model = _run(resolve_quick_model(engine))
    assert model.record_weights_digest is None


def test_ollama_adapter_without_digest_support_records_none():
    adapter = MagicMock()  # a bare mock: model_digest() returns a non-awaitable
    adapter.verify_model_pulled = AsyncMock(return_value=None)
    engine = DetectedEngine(name="ollama", base_url="http://localhost:11434", adapter=adapter)
    model = _run(resolve_quick_model(engine))
    assert model.record_weights_digest is None


def test_roster_model_with_a_different_digest_is_recorded_as_measured(caplog):
    entry = ROSTER[0]
    adapter = _adapter(digest="sha256:not-the-roster-digest")
    engine = DetectedEngine(name="ollama", base_url="http://localhost:11434", adapter=adapter)
    with caplog.at_level("WARNING"):
        model = _run(resolve_quick_model(engine, entry))
    assert model.record_weights_digest == "sha256:not-the-roster-digest"
    assert model.record_model == entry.hf_id
    assert any("roster pins" in r.getMessage() for r in caplog.records)


def test_llamacpp_quick_records_no_gguf_pin():
    adapter = MagicMock()
    client = AsyncMock()
    client.get_models = AsyncMock(return_value=["/models/whatever.gguf"])
    with patch("hmasync_controller.bench.quick.VLLMClient", return_value=client):
        engine = DetectedEngine(name="llama.cpp", base_url="http://localhost:8080", adapter=adapter)
        model = _run(resolve_quick_model(engine))
    assert model.record_gguf_repo is None
    assert model.record_gguf_revision is None
    assert model.record_weights_digest is None


def test_llamacpp_reference_pin_is_recorded_when_passed():
    adapter = MagicMock()
    client = AsyncMock()
    client.get_models = AsyncMock(return_value=["Qwen3.5-9B-Q4_K_M.gguf"])
    spec = QUICK_REFERENCE_MODELS["llama.cpp"]
    with patch("hmasync_controller.bench.quick.VLLMClient", return_value=client):
        engine = DetectedEngine(name="llama.cpp", base_url="http://localhost:8080", adapter=adapter)
        model = _run(
            resolve_quick_model(
                engine, gguf_repo=spec["gguf_repo"], gguf_revision=spec["revision"]
            )
        )
    assert model.record_gguf_repo == spec["gguf_repo"]
    assert model.record_gguf_revision == spec["revision"]


def test_bundle_carries_provenance_and_validates():
    from hmasync_controller.bench.submission import validate_bundle  # subset validator
    run = _fake_run_metrics(gguf_repo=None, gguf_revision=None, weights_digest="sha256:abc123")
    bundle = build_bundle([run], {"node_hash": "n" * 64}, suite="medium")
    exported = bundle["runs"][0]
    assert exported["weights_digest"] == "sha256:abc123"
    assert exported["gguf_repo"] is None
    assert exported["gguf_revision"] is None
    assert validate_bundle(bundle) == []


def test_reference_bundle_records_the_pinned_gguf():
    from hmasync_controller.bench.submission import validate_bundle
    spec = QUICK_REFERENCE_MODELS["llama.cpp"]
    run = _fake_run_metrics(
        engine="llama.cpp", gguf_repo=spec["gguf_repo"], gguf_revision=spec["revision"]
    )
    bundle = build_bundle([run], {"node_hash": "n" * 64}, suite="reference")
    assert bundle["runs"][0]["gguf_repo"] == spec["gguf_repo"]
    assert bundle["runs"][0]["gguf_revision"] == spec["revision"]
    assert validate_bundle(bundle) == []
