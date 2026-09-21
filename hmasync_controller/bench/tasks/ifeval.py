"""IFEval — instruction-following with programmatic (non-LLM-judge) verification.

Power shape: DECODE-dominated, like gsm8k — prompts are short natural-language
instructions ("write a 300+ word summary... do not use any commas..."), and
correctness depends on the shape of the full generated response, so the model
must actually produce the requested length/format rather than a short answer.

Each row already IS a fully-formatted instruction: `row["prompt"]` is sent to
the model verbatim, with no added instruction preamble and no few-shot
exemplars (this is a zero-shot benchmark by design — there is no natural
"exemplar" for an arbitrary, one-off instruction; `n_shot` is accepted for
interface parity with other tasks but always ignored, same as gpqa_diamond).

Grading is "prompt-level strict accuracy" from the IFEval paper: an item
passes only if EVERY instruction in its `instruction_id_list` passes its
checker (see `ifeval_checks.py`) against the raw completion — no markdown
stripping or multi-variant "loose" retry, and no LLM judge.
"""

from hmasync_controller.bench.tasks.base import Task, TaskItem, fetch_parquet_rows, sample_indices
from hmasync_controller.bench.tasks.ifeval_checks import check_instruction

REPO = "google/IFEval"
DATA_FILE = "default/train/0000.parquet"
# google/IFEval ships only a raw .jsonl on `main`; the hub's auto-converted
# parquet mirror is on `refs/convert/parquet` (541 rows) — same pattern as
# mmlu_redux/gpqa_diamond/math500.
REVISION = "refs/convert/parquet"


class IFEvalTask(Task):
    """IFEval: 541 instruction-following prompts, programmatic strict-accuracy scoring."""

    name = "ifeval"
    shape = "decode"
    # Measured, not copied. energy-bench's reference wave caps by MEASURED
    # demand per task (8192 gsm8k/mmlu_redux, 16384 math500/gpqa_diamond) and
    # does not include ifeval at all, so there is no lab figure to match and
    # no comparability argument for borrowing 8192.
    #
    # Swept on qwen3.5:9b-q4_K_M, n=25, thinking off, cap opened to 8192 to
    # see what the task actually wants:
    #
    #     mean 268   median 225   p90 629   MAX 1323
    #     over 2048: 0      hit 8192: 0      accuracy 0.80
    #
    # Accuracy was identical at 1024, so the old cap was not costing answers
    # -- but 2 of 100 items across the roster finished on `length` at exactly
    # 1024, i.e. graded on a sentence the model never ended. 2048 is ~1.5x the
    # observed maximum: it clears that truncation and still bounds the task.
    #
    # Bounding matters more here than elsewhere: ifeval carries NO stop
    # sequence (`stop` is empty below, unlike gsm8k's "\nQuestion:"), so this
    # cap is the only thing standing between a rambling model and the context
    # window. An 8192 cap would be an 8x worst case bought for nothing.
    default_max_tokens = 2048
    description = "Instruction-following with programmatic verification (no LLM judge)"
    revision = REVISION

    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        # Zero-shot by design; n_shot accepted for interface parity, unused.
        rows = fetch_parquet_rows(REPO, DATA_FILE, REVISION)

        items: list[TaskItem] = []
        for idx in sample_indices(len(rows), n_items, seed):
            row = rows[idx]
            items.append(
                TaskItem(
                    item_id=f"ifeval:{idx}",
                    prompt=row["prompt"],
                    target=";".join(row["instruction_id_list"]),
                    metadata={
                        "key": row["key"],
                        "instruction_id_list": list(row["instruction_id_list"]),
                        "kwargs": [dict(kw) for kw in row["kwargs"]],
                    },
                )
            )
        return items

    def score(self, completion: str, item: TaskItem) -> bool:
        instruction_id_list = item.metadata["instruction_id_list"]
        kwargs_list = item.metadata["kwargs"]
        return all(
            check_instruction(instruction_id, completion, kwargs)
            for instruction_id, kwargs in zip(instruction_id_list, kwargs_list, strict=True)
        )
