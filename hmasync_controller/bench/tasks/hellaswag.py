"""HellaSwag — commonsense sentence-completion, generative letter-choice.

HellaSwag is saturated: modern instruction-tuned models solve it near ceiling,
so it carries almost no discriminating power between healthy configurations.
It earns its place in the suite anyway as a *canary*: a broken quantization,
a wrong chat template, or a truncated context reliably tanks HellaSwag even
when it barely dents a harder benchmark, so a collapsed score here is a fast
signal that something about the run is misconfigured rather than that the
model is weak. `is_canary = True` marks it so a downstream aggregate can keep
it out of graded comparisons while still surfacing it as a sanity check.

Same generative letter-answer protocol as `mmlu_redux.py` — 4-way multiple
choice, one letter emitted, exact-match scored. See that module's docstring
for why generative (not loglikelihood) scoring is used throughout this suite.
"""


from hmasync_controller.bench.tasks.base import (
    Task,
    TaskItem,
    extract_letter_answer,
    fetch_parquet_rows,
    sample_indices,
)

REPO = "Rowan/hellaswag"

# HellaSwag's `test` split ships with every `label` blank (labels withheld by
# the original authors) -- confirmed live, not assumed. `label` is on the
# `validation` split instead, which this task uses for scoring; `train`
# supplies few-shot exemplars, matching lm-eval-harness convention.
VALIDATION_FILE = "data/validation-00000-of-00001.parquet"
FEWSHOT_TRAIN_FILE = "data/train-00000-of-00001.parquet"

LETTERS = ["A", "B", "C", "D"]


def _format_item(ctx: str, endings: list[str]) -> str:
    lines = [f"Sentence: {ctx.strip()}"]
    lines += [f"{LETTERS[i]}. {e}" for i, e in enumerate(endings)]
    return "\n".join(lines)


class HellaSwagTask(Task):
    """HellaSwag: sentence-completion canary, exact-match on the answer letter."""

    name = "hellaswag"
    shape = "prefill"
    default_max_tokens = 512
    description = "Saturated commonsense-completion canary (is_canary=True), letter-answer protocol"
    stop = ["\n\n", "Sentence:"]
    is_canary = True

    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        rows = fetch_parquet_rows(REPO, VALIDATION_FILE)

        shot_block = ""
        if n_shot > 0:
            train_rows = fetch_parquet_rows(REPO, FEWSHOT_TRAIN_FILE)
            shots = train_rows[:n_shot]
            blocks = [
                _format_item(r["ctx"], list(r["endings"]))
                + f"\nAnswer: {LETTERS[int(r['label'])]}"
                for r in shots
            ]
            shot_block = "\n\n".join(blocks) + "\n\n"

        instruction = (
            "The following is the beginning of a sentence and four possible ways to "
            "complete it. Choose the most plausible ending. Respond with only the "
            "letter of the correct ending."
        )

        items: list[TaskItem] = []
        for idx in sample_indices(len(rows), n_items, seed):
            row = rows[idx]
            body = _format_item(row["ctx"], list(row["endings"]))
            items.append(
                TaskItem(
                    item_id=f"hellaswag:{idx}",
                    prompt=f"{instruction}\n\n{shot_block}{body}\nAnswer:",
                    target=LETTERS[int(row["label"])],
                    metadata={
                        "activity_label": row["activity_label"],
                        "split_type": row["split_type"],
                    },
                )
            )
        return items

    def score(self, completion: str, item: TaskItem) -> bool:
        # Shared extractor (tasks/base.py): explicit "Answer: X" marker first,
        # else the LAST standalone A-D. Last, not first, because the cap is now
        # large enough for a model to reason before answering.
        return extract_letter_answer(completion) == item.target
