"""ROUGE-L F1, in pure Python — the scorer `bench.tasks.longctx_summary` uses.

Why this module exists instead of `rouge-score`: US-MERGE-01 pulled in
`rouge-score==0.1.2` for `longctx_summary`, and it drags `absl-py`, `nltk`,
`numpy` and `six` into the base install of a package whose whole point is a
single thin download (PRD operator decision 2: "Single download, no extra").
Five distributions for one LCS-based F-measure is a bad trade, so US-MERGE-06
replaced it with the ~30 lines below and dropped the dependency.

This is a deliberate REIMPLEMENTATION, not an approximation. It reproduces
`rouge_score.rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)` exactly:

- Tokenization matches `rouge_score.tokenize.tokenize(text, stemmer=None)`:
  lowercase, replace every non-`[a-z0-9]` run with a space, split on
  whitespace, drop anything that is not purely alphanumeric. (`six.ensure_str`
  in the reference is a no-op on `str`; the stemmer branch is unreachable at
  `use_stemmer=False`.)
- The score matches `_score_lcs`: precision = LCS / |prediction tokens|,
  recall = LCS / |target tokens|, F = 2PR/(P+R), and 0.0 when either side
  tokenizes to nothing.

The only intentional difference is memory: the reference keeps the whole
(m+1)x(n+1) DP table because it also needs to backtrack for `rougeLsum`;
only the final cell matters here, so two rolling rows suffice. The LCS length
— and therefore every returned score — is identical.

`tests/test_bench_rouge_l.py` pins that equivalence: golden values captured
from `rouge-score==0.1.2` itself, plus a differential test that re-runs the
comparison live whenever `rouge_score` happens to be importable.
"""

from __future__ import annotations

import re

_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9]+")
_SPACES_RE = re.compile(r"\s+")
_VALID_TOKEN_RE = re.compile(r"^[a-z0-9]+$")


def tokenize(text: str) -> list[str]:
    """Split `text` into ROUGE's alphanumeric lowercase tokens.

    Byte-for-byte the reference implementation's `use_stemmer=False` path —
    see the module docstring.
    """
    lowered = _NON_ALPHANUM_RE.sub(" ", text.lower())
    return [t for t in _SPACES_RE.split(lowered) if _VALID_TOKEN_RE.match(t)]


def lcs_length(target_tokens: list[str], prediction_tokens: list[str]) -> int:
    """Length of the longest common subsequence of two token lists.

    Standard O(m*n) DP over two rolling rows (the reference keeps the full
    table because `rougeLsum` backtracks through it; nothing here does).
    """
    if not target_tokens or not prediction_tokens:
        return 0
    previous = [0] * (len(prediction_tokens) + 1)
    for target_token in target_tokens:
        current = [0]
        for j, prediction_token in enumerate(prediction_tokens, start=1):
            if target_token == prediction_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l_fmeasure(target: str, prediction: str) -> float:
    """ROUGE-L F1 of `prediction` against reference `target`, in [0.0, 1.0].

    Returns 0.0 when either side has no tokens (an empty completion, or a
    reference summary of pure punctuation) — the reference implementation's
    `Score(precision=0, recall=0, fmeasure=0)` case, and the honest answer:
    there is no overlap to measure.
    """
    target_tokens = tokenize(target)
    prediction_tokens = tokenize(prediction)
    if not target_tokens or not prediction_tokens:
        return 0.0

    lcs = lcs_length(target_tokens, prediction_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(target_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
