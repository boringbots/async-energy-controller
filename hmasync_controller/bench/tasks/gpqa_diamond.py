"""GPQA Diamond — hard graduate-level science QA, gated dataset.

448 expert-written multiple-choice questions in biology, physics, and
chemistry ("Google-proof": PhDs outside their own subfield reach only ~34%
accuracy even with unrestricted web access). `gpqa_diamond` is the
highest-quality 198-question subset.

Same generative letter-answer protocol as `mmlu.py` (see that module's
docstring for the prefill-shape rationale) but zero-shot, matching the GPQA
paper's own evaluation setup: `n_shot` is accepted for interface parity with
other tasks but intentionally unused. Each item has one correct answer plus
three incorrect ones, shuffled into A-D deterministically from `seed` so the
same item always presents the same order across repeats.

`Idavidrein/gpqa` is a GATED dataset — the repo card asks users to accept its
terms on huggingface.co (to limit corpus leakage of the questions).
`hf_hub_download` picks up `HF_TOKEN` from the environment automatically;
without an accepted, token-bearing account every fetch 401s. `load()`
re-raises that as a `TaskLoadError` naming the fix instead of surfacing a raw
hub error. Both the raw CSV configs (`gpqa_diamond.csv`, per `configs:` in the
repo's README.md) and the hub's auto-converted parquet mirror
(`gpqa_diamond/train/0000.parquet` on `refs/convert/parquet`, same mechanism
`mmlu_redux` uses) exist and both 401 identically — gating applies repo-wide,
not per-file.

Column names (`Question`, `Correct Answer`, `Incorrect Answer 1/2/3`) are
taken from the reference implementations that already consume this dataset
(OpenAI simple-evals, EleutherAI lm-evaluation-harness).
"""

import random

from hmasync_controller.bench.tasks.base import (
    Task,
    TaskItem,
    TaskLoadError,
    extract_letter_answer,
    fetch_parquet_rows,
    sample_indices,
)

REPO = "Idavidrein/gpqa"
TEST_FILE = "gpqa_diamond/train/0000.parquet"
REVISION = "refs/convert/parquet"

LETTERS = ["A", "B", "C", "D"]

INSTRUCTION = (
    "The following is a multiple choice question (with answers) about "
    "graduate-level biology, physics, or chemistry. Respond with only the "
    "letter of the correct answer."
)


def _format_question(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question.strip()}"]
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    return "\n".join(lines)


def _is_gated_access_error(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)


def _load_test_rows() -> tuple[dict, ...]:
    try:
        return fetch_parquet_rows(REPO, TEST_FILE, REVISION)
    except TaskLoadError as e:
        if e.__cause__ is not None and _is_gated_access_error(e.__cause__):
            raise TaskLoadError(
                f"{REPO} is a gated dataset. Accept access at "
                f"https://huggingface.co/datasets/{REPO}, then set HF_TOKEN "
                "in your environment and retry."
            ) from e
        raise


class GPQADiamondTask(Task):
    """GPQA Diamond: 198 hard science questions, exact-match on the answer letter."""

    name = "gpqa_diamond"
    shape = "prefill"
    default_max_tokens = 512
    description = "Graduate-level science QA, zero-shot, letter-answer (gated dataset)"
    stop = ["\n\n", "Question:"]
    revision = REVISION

    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        # Zero-shot per the GPQA paper's own protocol; n_shot is accepted for
        # interface parity with other tasks but deliberately unused.
        test_rows = _load_test_rows()

        items: list[TaskItem] = []
        for idx in sample_indices(len(test_rows), n_items, seed):
            row = test_rows[idx]
            options = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            order = list(range(4))
            # Deterministic per-item shuffle: the same (seed, idx) always
            # yields the same option order, but every item shuffles
            # independently rather than sharing one global permutation.
            random.Random(f"{seed}:{idx}").shuffle(order)
            shuffled_choices = [options[i] for i in order]
            correct_letter = LETTERS[order.index(0)]
            body = _format_question(row["Question"], shuffled_choices)
            items.append(
                TaskItem(
                    item_id=f"gpqa_diamond:{idx}",
                    prompt=f"{INSTRUCTION}\n\n{body}\nAnswer:",
                    target=correct_letter,
                )
            )
        return items

    def score(self, completion: str, item: TaskItem) -> bool:
        # Shared extractor (tasks/base.py): explicit "Answer: X" marker first,
        # else the LAST standalone A-D. Last, not first, because the cap is now
        # large enough for a model to reason before answering.
        return extract_letter_answer(completion) == item.target
