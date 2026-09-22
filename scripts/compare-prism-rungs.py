#!/usr/bin/env python3
"""Read the prism wave's rungs against each other -- the point of running it.

    python3 scripts/compare-prism-rungs.py
    python3 scripts/compare-prism-rungs.py --data-dir bench_data

Reads each rung's summary (written by `bench prism`) and the per-item
`items.parquet` beside it, and answers the wave's two questions:

REVERSAL  PTQ1_0 against PQ2_0 on one 27B checkpoint. On Ampere the narrower
          storage cost 1.20-1.57x MORE energy at indistinguishable accuracy.
          The ratio printed here is the same comparison on this machine.

FORMAT    F16 / PQ2_0 / Q2_0_g64 of one 4B checkpoint. These are three
CONTROL   storages of the SAME ternary weights, so at temperature 0 they
          should answer every item identically. That is checked per item --
          same `item_id`, same `correct`, same `completion_tokens` -- rather
          than by comparing two accuracy figures, because an identity claim
          checked through a confidence interval at 100 items is not checked
          at all.

Energy numbers here compare rungs measured on ONE machine back to back, which
is the only comparison Apple joules support: `energy_source='ioreport'` never
pools with an NVIDIA counter, and this script refuses to print a cross-source
ratio.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def load_summaries(data_dir: Path) -> dict[str, dict]:
    """Latest summary per rung key. A re-run supersedes its predecessor --
    the filename carries the timestamp, so the newest wins."""
    latest: dict[str, tuple[str, dict]] = {}
    for path in sorted((data_dir / "prism").glob("*.json")):
        try:
            summary = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping {path.name}: {e}")
            continue
        key = summary.get("rung", {}).get("key")
        if not key:
            continue
        if key not in latest or path.name > latest[key][0]:
            latest[key] = (path.name, summary)
    return {k: v[1] for k, v in latest.items()}


def load_items(data_dir: Path, run_id: str) -> dict[str, tuple[float | None, int]]:
    """`item_id -> (correct, completion_tokens)` for one measured task."""
    path = data_dir / run_id / "items.parquet"
    if not path.exists():
        return {}
    table = pq.read_table(path).to_pydict()
    out: dict[str, tuple[float | None, int]] = {}
    for item_id, correct, tokens in zip(
        table["item_id"], table["correct"], table["completion_tokens"]
    ):
        if item_id is not None:
            out[item_id] = (correct, tokens)
    return out


def _fmt(value, spec: str = ".3f") -> str:
    return "—" if value is None else format(value, spec)


def print_rows(summaries: dict[str, dict]) -> None:
    print("MEASURED RUNGS")
    print(
        f"  {'rung':<22} {'quant':<10} {'task':<16} {'acc':>6} {'J total':>10} "
        f"{'J/correct':>10} {'J/token':>8} {'tok':>8} {'source':>9}"
    )
    for key, summary in summaries.items():
        quant = summary["rung"]["quantization"]
        for task in summary["tasks"]:
            joules = task.get("total_joules_gpu_best") or task.get("total_joules_gpu")
            print(
                f"  {key:<22} {quant:<10} {str(task.get('task')):<16} "
                f"{_fmt(task.get('accuracy'))!s:>6} {_fmt(joules, '.0f')!s:>10} "
                f"{_fmt(task.get('joules_per_correct_answer'), '.1f')!s:>10} "
                f"{_fmt(task.get('joules_per_token'), '.3f')!s:>8} "
                f"{task.get('total_completion_tokens', 0):>8} "
                f"{str(task.get('energy_source')):>9}"
            )
    mechanisms = {s.get("thinking", {}).get("mechanism") for s in summaries.values()}
    if "unknown" in mechanisms:
        print(
            "\n  WARNING: at least one rung's chat template neither reads "
            "enable_thinking nor\n  hardcodes an empty <think> block. That rung "
            "may have been REASONING while its\n  row recorded thinking off. "
            "Check its summary's thinking.note before comparing."
        )


def _by_task(summary: dict) -> dict[str, dict]:
    return {t["task"]: t for t in summary["tasks"] if t.get("task")}


def _same_source(a: dict, b: dict) -> bool:
    return a.get("energy_source") == b.get("energy_source")


def compare_reversal(summaries: dict[str, dict]) -> None:
    narrow, wide = "bonsai2-27b-ptq1-0", "bonsai2-27b-pq2-0"
    if narrow not in summaries or wide not in summaries:
        print("\nREVERSAL: needs both 27B rungs; not run yet.")
        return
    print("\nREVERSAL -- PTQ1_0 (1.58 bpw, 5.95 GB) vs PQ2_0 (2.13 bpw, 7.21 GB)")
    print("  Ampere measured the NARROWER file costing 1.20-1.57x MORE energy.")
    print(f"  {'task':<16} {'J/correct':>18} {'J/token':>16} {'tokens':>16} {'acc':>14}")
    for task, narrow_run in _by_task(summaries[narrow]).items():
        wide_run = _by_task(summaries[wide]).get(task)
        if wide_run is None:
            continue
        if not _same_source(narrow_run, wide_run):
            print(f"  {task:<16}  energy sources differ -- not comparable")
            continue

        def ratio(field: str) -> str:
            a, b = narrow_run.get(field), wide_run.get(field)
            if not a or not b:
                return "—"
            return f"{a / b:.2f}x"

        acc_a = narrow_run.get("accuracy")
        acc_b = wide_run.get("accuracy")
        acc = "—" if acc_a is None or acc_b is None else f"{acc_a:.3f}/{acc_b:.3f}"
        print(
            f"  {task:<16} {ratio('joules_per_correct_answer'):>18} "
            f"{ratio('joules_per_token'):>16} {ratio('total_completion_tokens'):>16} "
            f"{acc:>14}"
        )
    print("  (>1.00x means the narrower storage cost more, as on Ampere.)")


def compare_format_control(summaries: dict[str, dict], data_dir: Path) -> None:
    rungs = ["bonsai-4b-f16", "bonsai-4b-pq2-0", "bonsai-4b-q2-0-g64"]
    present = [r for r in rungs if r in summaries]
    if len(present) < 2:
        print("\nFORMAT CONTROL: needs at least two 4B rungs; not run yet.")
        return
    print("\nFORMAT CONTROL -- one ternary checkpoint, three storages")
    baseline_key = present[0]
    tasks = set(_by_task(summaries[baseline_key]))

    for task in sorted(tasks):
        print(f"\n  {task}")
        baseline = _by_task(summaries[baseline_key])[task]
        base_items = load_items(data_dir, baseline["run_id"])
        for key in present:
            run = _by_task(summaries[key]).get(task)
            if run is None:
                continue
            items = load_items(data_dir, run["run_id"])
            shared = sorted(set(base_items) & set(items))
            if key == baseline_key or not shared:
                agreement = "baseline" if key == baseline_key else "no shared items"
            else:
                disagree = sum(1 for i in shared if base_items[i][0] != items[i][0])
                tok_diff = sum(1 for i in shared if base_items[i][1] != items[i][1])
                agreement = (
                    f"{len(shared) - disagree}/{len(shared)} same answer, "
                    f"{len(shared) - tok_diff}/{len(shared)} same token count"
                )
            joules = run.get("total_joules_gpu_best") or run.get("total_joules_gpu")
            base_joules = baseline.get("total_joules_gpu_best") or baseline.get(
                "total_joules_gpu"
            )
            base_quant = summaries[baseline_key]["rung"]["quantization"]
            if key == baseline_key:
                energy = f"baseline ({joules:.0f} J)" if joules else "baseline"
            elif joules and base_joules and _same_source(run, baseline):
                energy = f"{base_joules / joules:.2f}x cheaper than {base_quant}"
            else:
                energy = "—"
            print(
                f"    {summaries[key]['rung']['quantization']:<10} "
                f"acc {_fmt(run.get('accuracy'))}  {energy:<28} {agreement}"
            )
    print(
        "\n  Identical answers across storages is the claim. Any disagreement is\n"
        "  either a real difference in the weights or a run that did not hold\n"
        "  temperature 0 -- check the rung's summary before treating it as either."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="bench_data",
        help="BENCH_DATA_DIR -- where bench prism wrote artifacts and summaries.",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    summaries = load_summaries(data_dir)
    if not summaries:
        print(f"no prism summaries under {data_dir / 'prism'}; run the wave first.")
        return 1

    by_stage: dict[str, list[str]] = defaultdict(list)
    for key, summary in summaries.items():
        by_stage[summary["rung"]["stage"]].append(key)
    print(
        f"{len(summaries)} rung(s) measured: "
        + ", ".join(f"{stage} {len(keys)}" for stage, keys in sorted(by_stage.items()))
    )
    print()
    print_rows(summaries)
    compare_reversal(summaries)
    compare_format_control(summaries, data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
