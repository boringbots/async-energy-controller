"""The completion text must not leave the box.

`VLLMClient.chat()` has always RETURNED the generated text -- the scorer needs
it -- and since 2026-09-01 `InferenceResult` also carries it, so a lab run can
tell a wrong answer from a missed extraction without re-spending the GPU time.

That convenience creates a hazard this file exists to close. A tuple element
has to be consumed deliberately; a dataclass field rides along inside any
`asdict()`. If someone later adds a blanket serialization to a log line, a
bundle, or the artifact writer, model output starts leaving the machine and
nothing fails. These tests are the tripwire: they assert the two boundaries
the text must never cross, so crossing one breaks a build instead of shipping
quietly.

Scope: the CONTROLLER is the package other people install and run. Its own
artifact writer and its submission bundle must stay text-free regardless of
what the lab does with the same dataclass.
"""

import json
from dataclasses import fields
from pathlib import Path

import pyarrow.parquet as pq

from hmasync_controller.bench.artifact import _write_items_parquet
from hmasync_controller.bench.metrics.models import InferenceResult, RunMetrics

TEXT_BEARING = ("completion_text", "reasoning_text")

SECRET = "PRIVATE-MODEL-OUTPUT-THAT-MUST-NOT-LEAVE-THE-BOX"


def _result() -> InferenceResult:
    return InferenceResult(
        request_id="req-0",
        prompt_tokens=10,
        completion_tokens=5,
        ttft_s=0.1,
        total_s=1.0,
        tokens_per_second=5.0,
        item_id="gsm8k:0",
        correct=True,
        completion_text=SECRET,
        reasoning_text=SECRET,
    )


def test_the_controllers_items_parquet_carries_no_generated_text(tmp_path):
    """The lab's writer (energy_bench.storage.artifact) stores the text on
    purpose. The controller's does not, and must not start by accident: a
    stranger running `bench quick` did not ask for their model's output to be
    written to disk."""
    out = tmp_path / "items.parquet"
    _write_items_parquet(out, [_result()])

    table = pq.read_table(out)
    for name in TEXT_BEARING:
        assert name not in table.column_names, (
            f"{name} reached the controller's items.parquet. If that is "
            "deliberate it needs a privacy decision, not a column."
        )
    # Belt and braces: no column's VALUES contain it either, in case a future
    # column is added under a different name.
    assert SECRET not in json.dumps(table.to_pydict(), default=str)


def test_run_metrics_has_no_text_field_at_all():
    """`write_run_artifact` does `json.dump(asdict(metrics))` -- a blanket
    serialization. That is safe only while RunMetrics carries no text, so
    assert it directly rather than trusting the writer."""
    names = {f.name for f in fields(RunMetrics)}
    assert not (names & set(TEXT_BEARING))
    assert not any("text" in n for n in names)


def test_the_submission_bundle_schema_has_no_content_field():
    """The bundle is what actually leaves the machine. Its `run` shape is
    run-level aggregates only -- `total_completion_tokens` is a COUNT. A
    content-bearing field appearing here would be model output going to a
    server, which is a different product decision from storing it locally."""
    schema_path = (
        Path(__file__).resolve().parent.parent
        / "hmasync_controller"
        / "schemas"
        / "bench_submission.schema.json"
    )
    raw = schema_path.read_text()
    for name in TEXT_BEARING:
        assert name not in raw, f"{name} appears in the submission schema"

    schema = json.loads(raw)
    run_props = schema["$defs"]["run"]["properties"]
    # Every run field must be a scalar measurement, never free text. `label`,
    # `model` and the version/name strings are identifiers the submitter
    # chooses; anything else string-shaped deserves a second look.
    allowed_strings = {
        f
        for f, spec in run_props.items()
        if "string" in json.dumps(spec.get("type", ""))
    }
    unexpected = {
        f for f in allowed_strings if any(t in f for t in ("text", "content", "answer"))
    }
    assert not unexpected, f"content-shaped fields in the bundle: {unexpected}"
