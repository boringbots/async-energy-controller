"""Scoped artifact writer for a `bench quick` run (US-MERGE-04).

Ported from energy-bench's `storage/artifact.py`, but deliberately NARROW:
that module resolves an NFS mount, writes `config.yaml`/`sweep_source.yaml`/
`run_meta.json`, and inserts into a DuckDB run index -- none of which
applies here. `bench quick` has no NFS mount, no sweep-chunk config, and no
DuckDB (GROUND TRUTH: the run index stays lab-side, in energy-bench, never
in this package). What DOES carry over unchanged is the two parquet
schemas -- `telemetry.parquet` and `items.parquet` -- and the run-id naming
convention, so a run measured here and a run measured by energy-bench are
shaped the same way on disk, even though nothing here ever reads a
telemetry.parquet back.

Each `bench quick` run gets its own folder,
`{BENCH_DATA_DIR}/{run_id}/`, containing:

    telemetry.parquet   every TelemetrySample collected across the run
    items.parquet        every InferenceResult (only when the task produced
                          any -- a free-text probe would have none, though
                          `bench quick` always runs a scored task today)
    metrics.json          the computed RunMetrics, `dataclasses.asdict` +
                          `json.dump` (RunMetrics is a plain dataclass here,
                          not pydantic -- no `.model_dump()`)

No config.yaml, no run_meta.json, no DuckDB insert.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from hmasync_controller.bench.metrics.models import InferenceResult, RunMetrics
from hmasync_controller.bench.sampler import TelemetrySample


class ArtifactWriteError(Exception):
    """Raised when writing a run's artifact folder fails."""

    def __init__(self, message: str, path: str | None = None):
        super().__init__(message)
        self.path = path


def generate_run_id(label: str) -> str:
    """Generate a unique run ID: `{label}_{YYYYMMDD_HHMMSS}_{8-char-uuid}`.

    Ported verbatim from energy-bench's `storage.artifact.generate_run_id`.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uuid_suffix = str(uuid4())[:8]
    return f"{label}_{timestamp}_{uuid_suffix}"


def _write_telemetry_parquet(telemetry_file: Path, samples: list[TelemetrySample]) -> None:
    try:
        data = {
            "ts": [s.ts for s in samples],
            "gpu_power_w": [s.gpu_power_w for s in samples],
            "gpu_util_pct": [s.gpu_util_pct for s in samples],
            "gpu_mem_used_mib": [s.gpu_mem_used_mib for s in samples],
            "gpu_temp_c": [s.gpu_temp_c for s in samples],
            "gpu_mem_util_pct": [s.gpu_mem_util_pct for s in samples],
            "gpu_energy_mj": [s.gpu_energy_mj for s in samples],
            "gpu_throttle_reasons": [s.gpu_throttle_reasons for s in samples],
            "gpu_sm_clock_mhz": [s.gpu_sm_clock_mhz for s in samples],
            "gpu_mem_clock_mhz": [s.gpu_mem_clock_mhz for s in samples],
            "gpu_fan_pct": [s.gpu_fan_pct for s in samples],
            "gpu_perf_state": [s.gpu_perf_state for s in samples],
            "cpu_rapl_uj": [s.cpu_rapl_uj for s in samples],
            "cpu_rapl_dram_uj": [s.cpu_rapl_dram_uj for s in samples],
        }
        schema = pa.schema(
            [
                ("ts", pa.float64()),
                ("gpu_power_w", pa.float64()),  # nullable
                ("gpu_util_pct", pa.float64()),  # nullable
                ("gpu_mem_used_mib", pa.float64()),  # nullable
                ("gpu_temp_c", pa.float64()),  # nullable
                ("gpu_mem_util_pct", pa.float64()),  # nullable
                ("gpu_energy_mj", pa.float64()),  # nullable
                ("gpu_throttle_reasons", pa.int64()),  # nullable
                ("gpu_sm_clock_mhz", pa.int64()),  # nullable
                ("gpu_mem_clock_mhz", pa.int64()),  # nullable
                ("gpu_fan_pct", pa.int64()),  # nullable
                ("gpu_perf_state", pa.int64()),  # nullable
                ("cpu_rapl_uj", pa.float64()),  # nullable, never set today (bench.sampler)
                ("cpu_rapl_dram_uj", pa.float64()),  # nullable, same
            ]
        )
        table = pa.table(data, schema=schema)
        pq.write_table(table, telemetry_file)
    except Exception as e:  # noqa: BLE001 - reraised as our own typed error
        raise ArtifactWriteError(
            f"Failed to write telemetry.parquet: {e}", str(telemetry_file)
        ) from e


def _write_items_parquet(items_file: Path, inference_results: list[InferenceResult]) -> None:
    try:
        data = {
            "item_id": [r.item_id for r in inference_results],
            "request_id": [r.request_id for r in inference_results],
            "prompt_tokens": [r.prompt_tokens for r in inference_results],
            "completion_tokens": [r.completion_tokens for r in inference_results],
            "correct": [
                float(r.correct) if r.correct is not None else None
                for r in inference_results
            ],
            "finish_reason": [r.finish_reason for r in inference_results],
            "t_start_s": [r.t_start_s for r in inference_results],
            "t_end_s": [r.t_end_s for r in inference_results],
            "total_s": [r.total_s for r in inference_results],
            "tokens_per_second": [r.tokens_per_second for r in inference_results],
        }
        schema = pa.schema(
            [
                ("item_id", pa.string()),  # nullable
                ("request_id", pa.string()),
                ("prompt_tokens", pa.int64()),
                ("completion_tokens", pa.int64()),
                # float64, not bool: a continuous-scored task (e.g.
                # longctx_summary's ROUGE-L F1) stores a value in [0, 1];
                # exact-match tasks store 1.0/0.0 for True/False.
                ("correct", pa.float64()),  # nullable
                ("finish_reason", pa.string()),  # nullable
                ("t_start_s", pa.float64()),  # nullable
                ("t_end_s", pa.float64()),  # nullable
                ("total_s", pa.float64()),
                ("tokens_per_second", pa.float64()),
            ]
        )
        table = pa.table(data, schema=schema)
        pq.write_table(table, items_file)
    except Exception as e:  # noqa: BLE001 - reraised as our own typed error
        raise ArtifactWriteError(f"Failed to write items.parquet: {e}", str(items_file)) from e


def write_run_artifact(
    data_dir: str | Path,
    run_id: str,
    telemetry_samples: list[TelemetrySample],
    inference_results: list[InferenceResult],
    metrics: RunMetrics,
) -> Path:
    """Write one run's artifact folder: `{data_dir}/{run_id}/`.

    Creates the folder (and `data_dir` itself) if missing. `items.parquet`
    is written only when `inference_results` is non-empty -- mirrors
    energy-bench's `write_artifact`, which skips it for a free-text probe
    with no scored items.

    Returns the run folder's path.

    Raises:
        ArtifactWriteError: On any I/O or serialization failure.
    """
    run_dir = Path(data_dir) / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ArtifactWriteError(f"Could not create {run_dir}: {e}", str(run_dir)) from e

    _write_telemetry_parquet(run_dir / "telemetry.parquet", telemetry_samples)
    if inference_results:
        _write_items_parquet(run_dir / "items.parquet", inference_results)

    metrics_file = run_dir / "metrics.json"
    try:
        with open(metrics_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2, default=str)
    except OSError as e:
        raise ArtifactWriteError(
            f"Failed to write metrics.json: {e}", str(metrics_file)
        ) from e

    return run_dir
