"""Benchmark task abstractions: item loading, prompt formatting, and scoring.

A Task turns a public benchmark dataset into a deterministic list of TaskItems.
Each item carries a fully-formatted prompt and a gold target, so the orchestrator
can send it through the same generative path regardless of which benchmark it
came from. That uniformity is deliberate: it keeps joules/token comparable across
tasks, which loglikelihood-style scoring (as used by lm-eval for MMLU) would
break by generating zero tokens.

Datasets are pulled as parquet straight from the HuggingFace hub via
huggingface_hub + pyarrow. The heavier `datasets` package is intentionally not a
dependency — it pulls pandas/dill/multiprocess for no benefit here.
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, ClassVar, Literal

import pyarrow.parquet as pq
from pydantic import BaseModel

# Power-shape classification. This is the hypothesis under test: prefill is
# compute-bound (short, high, spiky power draw) while decode is
# memory-bandwidth-bound (a longer, lower, flatter plateau).
PowerShape = Literal["prefill", "decode", "mixed"]


class TaskLoadError(Exception):
    """Raised when a benchmark dataset cannot be fetched or parsed."""


class TaskItem(BaseModel):
    """A single scored benchmark item, ready to send to the model."""

    item_id: str
    """Stable identifier: '<task>:<row index>'. Reproducible across runs."""

    prompt: str
    """Fully-formatted prompt text, including any few-shot preamble."""

    target: str
    """Normalized gold answer used by score()."""

    metadata: dict[str, Any] = {}
    """Task-specific extras (e.g. MMLU subject)."""


@lru_cache(maxsize=128)
def fetch_parquet_rows(
    repo_id: str, filename: str, revision: str = "main"
) -> tuple[dict[str, Any], ...]:
    """Download a dataset parquet from the HF hub and return its rows.

    Cached per (repo_id, filename, revision) for the process lifetime;
    huggingface_hub additionally caches the file on disk under HF_HOME, so
    repeat runs are offline after the first fetch. `revision` defaults to
    "main"; pass "refs/convert/parquet" for repos that only ship parquet via
    the hub's auto-conversion branch (e.g. mmlu_redux's per-subject files).

    Raises:
        TaskLoadError: If the file cannot be downloaded or read.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision
        )
        table = pq.read_table(path)
    except Exception as e:  # noqa: BLE001 - surface any hub/parquet failure uniformly
        raise TaskLoadError(f"Could not load {repo_id}/{filename}@{revision}: {e}") from e

    return tuple(table.to_pylist())


def sample_indices(n_total: int, n_items: int, seed: int) -> list[int]:
    """Deterministically choose which rows to benchmark.

    The same (n_total, n_items, seed) always yields the same items, so every
    model in a matrix sees an identical workload. Without this, cross-model
    energy comparisons would be confounded by different questions.
    """
    if n_items >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), n_items))


def extract_last_number(text: str) -> str | None:
    """Pull the final number out of a completion, ignoring commas and currency.

    Used for numeric-answer tasks where models often narrate before answering.
    """
    cleaned = text.replace(",", "").replace("$", "")
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    if not matches:
        return None
    return matches[-1].rstrip(".")


def numbers_equal(a: str | None, b: str | None) -> bool:
    """Compare two numeric strings tolerantly (2 == 2.0, 2.000001 != 2)."""
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return False


# "Answer: C", "the answer is (C)", "final answer - C", "**C**".
_MC_MARKED_ANSWER = re.compile(
    r"(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s*\**\(?([ABCD])\b",
    re.IGNORECASE,
)
_MC_BARE_LETTER = re.compile(r"\b([ABCD])\b")
# The instructed format: the completion opens with the letter, standing alone
# or followed by punctuation ("C", "C.", "C) because ...", "C, though B ...").
#
# The negative lookahead is what makes this safe. Requiring the next character
# to be punctuation or end-of-string separates an ANSWER from a letter that
# merely starts a sentence: "A is wrong, B is wrong, leaving C" opens with a
# letter that is being rejected, and is prose ("A" + space + word), so it falls
# through to the trailing-letter pass and scores C.
_MC_LEADING_LETTER = re.compile(r"^\(?([ABCD])(?![\w\s])")


def extract_letter_answer(completion: str) -> str | None:
    """Pull the chosen letter out of a multiple-choice completion.

    Three passes, in this order. The order is the whole design — every other
    ordering mis-scores one of the two shapes real models produce.

    1. **An explicit marker**, last match: "Answer: D", "the answer is (C)",
       "final answer - **B**". When a model says which letter it is choosing,
       that is the answer regardless of position, and the last such statement
       is its conclusion.
    2. **A leading standalone letter**, when the completion opens with one.
       The prompts instruct "respond with only the letter", so a compliant
       model answers "C" or "C." and may then add an aside — "C, though B is
       tempting" chooses C.
    3. **The last standalone letter**, otherwise. A model that reasons in
       prose and never marks its conclusion ("...so it must be B") ends on the
       answer.

    Why not simply first-match, which is what these tasks used before: it did
    not matter while generation was capped at 8 tokens and the whole
    completion was a bare "C." Once the cap is large enough for a model to
    reason first, first-match returns whichever option gets mentioned
    earliest, which in a reasoning trace is routinely one being rejected —
    "A is tempting but incorrect, so the answer is B" scores A. Pass 1 catches
    exactly that case, which is why the marker outranks the leading letter
    rather than the other way round.

    Returns None when no letter is found at all. That is a distinct outcome
    from a wrong answer — it means truncated or off-format — and callers
    should preserve the distinction rather than collapsing it to "incorrect".
    """
    text = completion.strip().upper()
    if not text:
        return None
    marked = _MC_MARKED_ANSWER.findall(text)
    if marked:
        return marked[-1].upper()
    leading = _MC_LEADING_LETTER.match(text)
    if leading:
        return leading.group(1)
    bare = _MC_BARE_LETTER.findall(text)
    return bare[-1] if bare else None


class Task(ABC):
    """Base class for a scored benchmark."""

    name: ClassVar[str]
    shape: ClassVar[PowerShape]
    default_max_tokens: ClassVar[int]
    description: ClassVar[str]

    stop: ClassVar[list[str]] = []
    """Sequences that halt generation.

    These are a *measurement integrity* control, not a formatting nicety.
    Few-shot prompts teach the model to continue the Question/Answer pattern, so
    without a stop it will answer correctly and then hallucinate further Q/A
    pairs until it hits max_tokens. That burns several times the necessary
    energy on discarded tokens, which would corrupt joules/item and penalise
    whichever model happens to ramble most.
    """

    revision: ClassVar[str | None] = None
    """Pinned HuggingFace dataset revision this task fetches from (e.g.
    "refs/convert/parquet"), for RunMetrics.dataset_revision. None when the
    task reads the default `main` branch and has no revision worth recording."""

    is_canary: ClassVar[bool] = False
    """True for a saturated/instrument-grade task kept for its sanity-check
    value rather than as a graded axis (e.g. hellaswag: near-ceiling on
    healthy configs, but reliably tanked by a broken one). Feeds
    RunMetrics.is_canary so downstream aggregates can exclude it."""

    @abstractmethod
    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        """Return exactly n_items deterministically-sampled, formatted items."""

    @abstractmethod
    def score(self, completion: str, item: TaskItem) -> bool | float:
        """Score a completion against the item's gold target.

        Most tasks are exact-match and return a plain bool. A task with a
        continuous quality metric (e.g. longctx_summary's ROUGE-L F1) may
        return a float in [0, 1] instead; callers that aggregate accuracy
        accept either uniformly (accuracy becomes the mean of whatever values
        score() returns)."""
