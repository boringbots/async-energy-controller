"""GSM8K — grade-school math word problems, scored by exact numeric match.

Power shape: DECODE-dominated. A short prompt produces a few hundred tokens of
chain-of-thought, so the GPU spends most of the run in the memory-bandwidth-bound
decode loop — expected to draw a lower, flatter power plateau than MMLU.
"""

from hmasync_controller.bench.tasks.base import (
    Task,
    TaskItem,
    TaskLoadError,
    extract_last_number,
    fetch_parquet_rows,
    numbers_equal,
    sample_indices,
)

REPO = "openai/gsm8k"
TEST_FILE = "main/test-00000-of-00001.parquet"
TRAIN_FILE = "main/train-00000-of-00001.parquet"

INSTRUCTION = (
    "Solve the math problem. Reason step by step, then give the final numeric "
    'answer on its own last line in the form "#### <answer>".'
)


def _gold_answer(answer_field: str) -> str:
    """GSM8K gold answers put the final value after a '####' marker."""
    if "####" not in answer_field:
        raise TaskLoadError(f"GSM8K row missing '####' marker: {answer_field[:80]!r}")
    return answer_field.split("####")[-1].strip().replace(",", "")


class GSM8KTask(Task):
    """GSM8K: 1319 test problems, exact-match on the final number."""

    name = "gsm8k"
    shape = "decode"
    default_max_tokens = 400
    description = "Grade-school math word problems (chain-of-thought, decode-heavy)"
    # Without these the model answers, then invents its own follow-up questions
    # until max_tokens — measured at ~4x the necessary tokens on some items.
    stop = ["Question:", "\nQuestion"]

    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        test_rows = fetch_parquet_rows(REPO, TEST_FILE)

        preamble = INSTRUCTION
        if n_shot > 0:
            train_rows = fetch_parquet_rows(REPO, TRAIN_FILE)
            # Few-shot exemplars come from train and are fixed across the whole
            # matrix, so every model sees the identical preamble.
            shots = train_rows[:n_shot]
            blocks = [
                f"Question: {r['question'].strip()}\nAnswer: {r['answer'].strip()}"
                for r in shots
            ]
            preamble = INSTRUCTION + "\n\n" + "\n\n".join(blocks)

        items: list[TaskItem] = []
        for idx in sample_indices(len(test_rows), n_items, seed):
            row = test_rows[idx]
            items.append(
                TaskItem(
                    item_id=f"gsm8k:{idx}",
                    prompt=f"{preamble}\n\nQuestion: {row['question'].strip()}\nAnswer:",
                    target=_gold_answer(row["answer"]),
                )
            )
        return items

    def score(self, completion: str, item: TaskItem) -> bool:
        # Read the FIRST '####' marker, not the last: if the model runs past its
        # answer into a hallucinated follow-up question, the last marker belongs
        # to that hallucination rather than the question we asked. Stop
        # sequences make this rare, but scoring stays defensive.
        if "####" in completion:
            first_answer_line = completion.split("####")[1].split("\n")[0]
            predicted = extract_last_number(first_answer_line)
            if predicted is not None:
                return numbers_equal(predicted, item.target)
        # Fallback: last number anywhere. Smaller models often ignore the format
        # instruction while still answering correctly.
        return numbers_equal(extract_last_number(completion), item.target)
