"""MMLU — 4-way multiple choice across 57 subjects, scored on the letter.

Power shape: PREFILL-dominated. A long 5-shot prompt produces a single letter,
so nearly all the work is the compute-bound prefill pass — expected to draw a
short, high, spiky power profile, in contrast to GSM8K's decode plateau.

Note this is *generative* MMLU (the model emits "A".."D"), not the loglikelihood
scoring lm-eval uses by default. That is a deliberate choice: loglikelihood
scoring generates zero tokens, which makes joules/token undefined and
incomparable with generative tasks.
"""


from hmasync_controller.bench.tasks.base import (
    Task,
    TaskItem,
    extract_letter_answer,
    fetch_parquet_rows,
    sample_indices,
)

REPO = "cais/mmlu"
TEST_FILE = "all/test-00000-of-00001.parquet"
DEV_FILE = "all/dev-00000-of-00001.parquet"

LETTERS = ["A", "B", "C", "D"]


def _format_question(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question.strip()}"]
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    return "\n".join(lines)


class MMLUTask(Task):
    """MMLU: ~14k test questions, exact-match on the answer letter."""

    name = "mmlu"
    shape = "prefill"
    default_max_tokens = 512
    description = "57-subject multiple choice (long prompt, 1-token answer, prefill-heavy)"
    stop = ["\n\n", "Question:"]

    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        test_rows = fetch_parquet_rows(REPO, TEST_FILE)

        shot_block = ""
        if n_shot > 0:
            dev_rows = fetch_parquet_rows(REPO, DEV_FILE)
            shots = dev_rows[:n_shot]
            blocks = [
                _format_question(r["question"], list(r["choices"]))
                + f"\nAnswer: {LETTERS[r['answer']]}"
                for r in shots
            ]
            shot_block = "\n\n".join(blocks) + "\n\n"

        items: list[TaskItem] = []
        for idx in sample_indices(len(test_rows), n_items, seed):
            row = test_rows[idx]
            subject = str(row.get("subject", "general knowledge")).replace("_", " ")
            instruction = (
                f"The following are multiple choice questions (with answers) "
                f"about {subject}. Respond with only the letter of the correct answer."
            )
            body = _format_question(row["question"], list(row["choices"]))
            items.append(
                TaskItem(
                    item_id=f"mmlu:{idx}",
                    prompt=f"{instruction}\n\n{shot_block}{body}\nAnswer:",
                    target=LETTERS[row["answer"]],
                    metadata={"subject": row.get("subject", "")},
                )
            )
        return items

    def score(self, completion: str, item: TaskItem) -> bool:
        # Shared extractor (tasks/base.py): explicit "Answer: X" marker first,
        # else the LAST standalone A-D. Last, not first, because the cap is now
        # large enough for a model to reason before answering.
        return extract_letter_answer(completion) == item.target
