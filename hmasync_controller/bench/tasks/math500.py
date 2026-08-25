"""MATH-500 — 500 hard competition-math problems, scored by boxed-answer extraction.

Power shape: DECODE-dominated, like GSM8K — the model works through a
derivation before landing on \\boxed{<answer>} on the last line, so most
tokens are spent in the memory-bandwidth-bound decode loop rather than
prefill.

Scoring limitation: this is STRING equality after light LaTeX normalization
(plus a numeric fallback), not a computer-algebra-system (CAS) check. Two
answers that are mathematically equivalent but spelled differently (e.g.
"\\frac{1}{2}" vs "0.5", or "2\\sqrt{3}" vs "\\sqrt{12}") will NOT be counted
as a match. Treat math500 accuracy as a lower bound, not an exact figure.
"""

import re

from hmasync_controller.bench.tasks.base import (
    Task,
    TaskItem,
    extract_last_number,
    fetch_parquet_rows,
    numbers_equal,
    sample_indices,
)

# HuggingFaceH4/MATH-500 only ships a single test.jsonl on `main`; the hub's
# auto-converted parquet mirror on `refs/convert/parquet` has the same 500
# rows as one parquet file — same mechanism mmlu_redux/gpqa_diamond use.
REPO = "HuggingFaceH4/MATH-500"
TEST_FILE = "default/test/0000.parquet"
REVISION = "refs/convert/parquet"

# MATH-500 has no train split of its own (it's a curated subset of the
# original MATH test set). Few-shot exemplars instead come from the original
# Hendrycks MATH training data (per-subject parquet files); a single fixed
# subject (algebra) keeps the preamble small and identical across every item
# and model, mirroring gsm8k_platinum's two-repo (test vs. few-shot) shape.
FEWSHOT_REPO = "EleutherAI/hendrycks_math"
FEWSHOT_FILE = "algebra/train-00000-of-00001.parquet"

INSTRUCTION = (
    "Solve the problem. Reason step by step, then give the final answer on "
    "its own last line in the form \\boxed{<answer>}."
)

_WHITESPACE_RE = re.compile(r"\s+")


def extract_boxed_answers(text: str) -> list[str]:
    """Find every \\boxed{...} payload in `text`, in order.

    Uses brace matching (not a regex) so nested LaTeX like
    \\boxed{\\frac{1}{2}} is captured whole instead of truncating at the
    first inner '}'. A \\boxed{ with no matching '}' is ignored.
    """
    marker = "\\boxed{"
    results: list[str] = []
    search_from = 0
    while True:
        start = text.find(marker, search_from)
        if start == -1:
            break
        content_start = start + len(marker)
        depth = 1
        i = content_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            break  # unbalanced brace; nothing further to parse
        results.append(text[content_start : i - 1])
        search_from = i
    return results


def normalize_math_answer(raw: str) -> str:
    """Normalize a MATH-style answer for string comparison.

    Strips whitespace (including internal spacing, which LaTeX renders
    inconsistently), drops \\left/\\right and \\! (formatting-only commands),
    and trims a trailing ".0". Not a CAS — see module docstring.
    """
    s = raw.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "")
    s = _WHITESPACE_RE.sub("", s)
    if s.endswith(".0"):
        s = s[:-2]
    return s


class Math500Task(Task):
    """MATH-500: 500 competition-math problems, boxed-answer exact match."""

    name = "math500"
    shape = "decode"
    default_max_tokens = 1024
    description = "Competition math problems (chain-of-thought, decode-heavy)"
    # Without these the model answers, then starts inventing its own
    # follow-up problem in the few-shot Problem/Solution pattern.
    stop = ["Problem:", "\nProblem"]
    revision = REVISION

    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        test_rows = fetch_parquet_rows(REPO, TEST_FILE, REVISION)

        preamble = INSTRUCTION
        if n_shot > 0:
            train_rows = fetch_parquet_rows(FEWSHOT_REPO, FEWSHOT_FILE)
            shots = train_rows[:n_shot]
            blocks = [
                f"Problem: {r['problem'].strip()}\nSolution: {r['solution'].strip()}"
                for r in shots
            ]
            preamble = INSTRUCTION + "\n\n" + "\n\n".join(blocks)

        items: list[TaskItem] = []
        for idx in sample_indices(len(test_rows), n_items, seed):
            row = test_rows[idx]
            items.append(
                TaskItem(
                    item_id=f"math500:{idx}",
                    prompt=f"{preamble}\n\nProblem: {row['problem'].strip()}\nSolution:",
                    target=row["answer"].strip(),
                    metadata={"subject": row.get("subject"), "level": row.get("level")},
                )
            )
        return items

    def score(self, completion: str, item: TaskItem) -> bool:
        target_norm = normalize_math_answer(item.target)
        boxed = extract_boxed_answers(completion)
        if boxed:
            # Last box wins: a runaway model that starts a new problem inside
            # its own hallucinated Solution block would otherwise let that
            # box's answer overwrite the one we asked for.
            predicted_norm = normalize_math_answer(boxed[-1])
            if predicted_norm == target_norm:
                return True
            return numbers_equal(predicted_norm, target_norm)
        # No \boxed{} at all: smaller models often ignore the format
        # instruction while still landing on the right number.
        return numbers_equal(extract_last_number(completion), target_norm)
