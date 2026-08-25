"""Long-context summarization — long-prefill KV-cache stressor (instrument grade).

GovReport-derived long documents (ccdv/govreport-summarization) are bucketed
into a fixed set of token-count ranges covering 8K-32K tokens, giving a
controlled long-prefill workload distinct from every other task in this suite
(all of which sit well under 2K prompt tokens). Scored via ROUGE-L F1 against
the reference summary (`bench.tasks.rouge_l`, a numpy-free reimplementation
of `rouge-score`'s `rougeL` scorer — see that module), and returned as a
continuous value rather than thresholded to a pass/fail — the point of this
instrument is to detect *degraded but not garbage* output (e.g. a heavily
quantized or power-capped model producing a worse-but-still-on-topic
summary), which a binary cutoff would destroy. See `Task.score`'s docstring
for how a float score flows into an otherwise bool-shaped accuracy aggregate.

`is_canary = True`: ROUGE-L is a weak proxy for summary quality (it rewards
lexical overlap, not factual correctness), so this task is kept out of graded
aggregates until a better scorer is chosen — it is an instrument for the
prefill-length power-shape hypothesis, not a leaderboard axis.

Token counts used for bucketing are ESTIMATED (`len(text) // CHARS_PER_TOKEN`),
not measured by any model's real tokenizer — this suite runs the same items
against many models with different tokenizers, and picking one model's
tokenizer to bucket by would bias the buckets toward that model's vocabulary.
The estimate only needs to sort documents into a coarse bucket; the real
prompt-token count the engine returns per item is the ground truth for any
downstream energy-per-token analysis.
"""

from hmasync_controller.bench.tasks.base import (
    Task,
    TaskItem,
    fetch_parquet_rows,
    sample_indices,
)
from hmasync_controller.bench.tasks.rouge_l import rouge_l_fmeasure

REPO = "ccdv/govreport-summarization"

# Ships parquet directly on `main` (verified live via list_repo_files) —
# unlike gpqa_diamond/math500/mmlu_redux/ifeval, no refs/convert/parquet
# mirror is needed here.
TEST_FILE = "document/test-00000-of-00001.parquet"

# English-text rule of thumb (~4 chars/token). See module docstring for why
# this is an estimate rather than a real tokenizer count.
CHARS_PER_TOKEN = 4

MIN_TOKENS = 8_000
MAX_TOKENS = 32_000

# Fixed-width ranges covering MIN_TOKENS-MAX_TOKENS, recorded in item
# metadata so downstream analysis can group by prefill length.
TOKEN_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("8k-16k", 8_000, 16_000),
    ("16k-24k", 16_000, 24_000),
    ("24k-32k", 24_000, 32_000),
)


def _bucket_label(estimated_tokens: int) -> str:
    """Map an estimated token count in [MIN_TOKENS, MAX_TOKENS] to its bucket label."""
    for label, lo, hi in TOKEN_BUCKETS:
        if lo <= estimated_tokens <= hi:
            return label
    raise ValueError(f"{estimated_tokens} tokens is outside the bucketed range")


def _truncate_to_max_tokens(report: str) -> str:
    """Cap a report at MAX_TOKENS' worth of characters, deterministically."""
    max_chars = MAX_TOKENS * CHARS_PER_TOKEN
    return report if len(report) <= max_chars else report[:max_chars]


class LongctxSummaryTask(Task):
    """GovReport long-document summarization, ROUGE-L F1 scored (instrument grade)."""

    name = "longctx_summary"
    shape = "prefill"
    default_max_tokens = 1536
    description = (
        "Long-prefill KV-cache stressor (instrument grade, is_canary=True): "
        "GovReport summarization scored by ROUGE-L F1"
    )
    is_canary = True

    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        # Zero-shot only: an in-context example would itself be an
        # 8K-32K-token document, blowing well past any bucket boundary.
        # n_shot is accepted for interface parity and ignored, same as
        # gpqa_diamond/ifeval.
        rows = fetch_parquet_rows(REPO, TEST_FILE)

        eligible: list[tuple[int, str, int]] = []
        for idx, row in enumerate(rows):
            report = _truncate_to_max_tokens(row["report"])
            estimated_tokens = len(report) // CHARS_PER_TOKEN
            if estimated_tokens >= MIN_TOKENS:
                eligible.append((idx, report, estimated_tokens))

        items: list[TaskItem] = []
        for pos in sample_indices(len(eligible), n_items, seed):
            idx, report, estimated_tokens = eligible[pos]
            row = rows[idx]
            items.append(
                TaskItem(
                    item_id=f"longctx_summary:{idx}",
                    prompt=(
                        "Summarize the following government report in a few "
                        f"sentences.\n\nReport:\n{report}\n\nSummary:"
                    ),
                    target=row["summary"],
                    metadata={
                        "token_bucket": _bucket_label(estimated_tokens),
                        "estimated_prompt_tokens": estimated_tokens,
                    },
                )
            )
        return items

    def score(self, completion: str, item: TaskItem) -> float:
        return rouge_l_fmeasure(item.target, completion.strip())
