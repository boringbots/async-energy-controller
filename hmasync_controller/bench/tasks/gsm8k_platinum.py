"""GSM8K-Platinum — re-labeled/cleaned GSM8K test set, scored by exact numeric match.

Power shape: DECODE-dominated, same as `gsm8k` (short prompt, long chain-of-
thought). Test items come from the platinum-cleaned test split so scores are
not confounded by GSM8K's known label errors; few-shot exemplars stay on the
ORIGINAL gsm8k train split for continuity with the v1 task suite.
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

REPO = "madrylab/gsm8k-platinum"
TEST_FILE = "main/test-00000-of-00001.parquet"

# Few-shot exemplars deliberately stay on the ORIGINAL gsm8k train split (not
# platinum-cleaned) so every model sees the identical preamble it would have
# seen under the v1 gsm8k task — only the scored test items change.
FEWSHOT_REPO = "openai/gsm8k"
FEWSHOT_TRAIN_FILE = "main/train-00000-of-00001.parquet"

INSTRUCTION = (
    "Solve the math problem. Reason step by step, then give the final numeric "
    'answer on its own last line in the form "#### <answer>".'
)


def _gold_answer(answer_field: str) -> str:
    """GSM8K(-platinum) gold answers put the final value after a '####' marker."""
    if "####" not in answer_field:
        raise TaskLoadError(f"GSM8K-platinum row missing '####' marker: {answer_field[:80]!r}")
    return answer_field.split("####")[-1].strip().replace(",", "")


class GSM8KPlatinumTask(Task):
    """GSM8K-Platinum: cleaned/re-labeled GSM8K test set, exact-match on the final number."""

    name = "gsm8k_platinum"
    shape = "decode"
    default_max_tokens = 400
    description = "Cleaned/re-labeled GSM8K test set (chain-of-thought, decode-heavy)"
    # Without these the model answers, then invents its own follow-up questions
    # until max_tokens — measured at ~4x the necessary tokens on some items.
    stop = ["Question:", "\nQuestion"]

    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        test_rows = fetch_parquet_rows(REPO, TEST_FILE)

        preamble = INSTRUCTION
        if n_shot > 0:
            train_rows = fetch_parquet_rows(FEWSHOT_REPO, FEWSHOT_TRAIN_FILE)
            # Few-shot exemplars come from the original gsm8k train split and
            # are fixed across the whole matrix, so every model sees the
            # identical preamble.
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
                    item_id=f"gsm8k_platinum:{idx}",
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
