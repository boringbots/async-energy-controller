"""`bench.tasks.rouge_l` — the numpy-free ROUGE-L that replaced `rouge-score`
(US-MERGE-06).

The point of this file is EQUIVALENCE, not "does an F-measure come out". The
dependency was dropped so the base install matches the agreed seven-package
set (`tests/test_base_install_audit.py`), which is only defensible if the
replacement scores identically — a silently different scorer would move every
`longctx_summary` number ever submitted from a community box.

Two layers:

1. `TestGoldenValues` — values captured from `rouge-score==0.1.2` itself on
   2026-08-25, hardcoded here. These run on a clean install where
   `rouge_score` is NOT importable, which is the whole environment under test.
2. `TestDifferentialAgainstRougeScore` — re-derives the comparison live
   whenever `rouge_score` happens to be importable (a dev box that still has
   it, from before the removal). Skipped, never failed, when it is absent.

Case selection is deliberate: the empty/punctuation-only sides, casing,
non-ASCII, and intra-word punctuation are exactly where a hand-rolled
tokenizer drifts from the reference.
"""

from __future__ import annotations

import random

import pytest

from hmasync_controller.bench.tasks.rouge_l import (
    lcs_length,
    rouge_l_fmeasure,
    tokenize,
)

# (target, prediction, rouge-score==0.1.2's rougeL fmeasure). Captured by
# running `RougeScorer(["rougeL"], use_stemmer=False).score(target, prediction)`
# under the dependency, 2026-08-25.
GOLDEN_CASES: list[tuple[str, str, float]] = [
    ("The quick brown fox jumps over the lazy dog",
     "The quick brown dog jumps on the log.", 0.5882352941176471),
    ("", "anything at all", 0.0),
    ("something", "", 0.0),
    ("!!! ??? ...", "hello world", 0.0),
    ("hello world", "!!! ??? ...", 0.0),
    ("Congress passed H.R. 1234 in 2021.",
     "The Congress passed HR 1234 during 2021", 0.5714285714285714),
    ("a b c d e f g", "g f e d c b a", 0.14285714285714285),
    ("repeat repeat repeat", "repeat", 0.5),
    ("MiXeD CaSe TeXt", "mixed case text", 1.0),
    ("emoji \U0001F600 test", "emoji test", 1.0),
    ("hyphen-ated words don't split", "hyphen ated words don t split", 1.0),
    ("The report covers fiscal year 2019 spending on defense programs.",
     "This summary describes 2019 defense spending in the fiscal year report.",
     0.28571428571428564),
    ("x", "x", 1.0),
    ("x", "y", 0.0),
    ("   ", "   ", 0.0),
    ("tabs\tand\nnewlines here", "tabs and newlines here", 1.0),
    ("123 456 789", "789 456 123", 0.3333333333333333),
    ("Ünïcödé accénts stay out",
     "unicode accents stay out", 0.36363636363636365),
]


def _random_pairs(n: int = 300) -> list[tuple[str, str]]:
    """Seeded token-salad pairs — same generator the removal was verified with."""
    rnd = random.Random(20260825)
    vocab = ["alpha", "beta", "gamma", "delta", "the", "of", "and",
             "report", "budget", "2020", "agency", "program"]
    pairs = []
    for _ in range(n):
        a = " ".join(rnd.choice(vocab) for _ in range(rnd.randint(0, 40)))
        b = " ".join(rnd.choice(vocab) for _ in range(rnd.randint(0, 40)))
        pairs.append((a, b))
    return pairs


class TestTokenize:
    def test_lowercases_and_strips_non_alphanumerics(self):
        assert tokenize("H.R. 1234, Fiscal-Year!") == ["h", "r", "1234", "fiscal", "year"]

    def test_punctuation_only_text_has_no_tokens(self):
        assert tokenize("!!! ??? ...") == []

    def test_non_ascii_letters_are_dropped_not_transliterated(self):
        # The reference regex is `[^a-z0-9]+` — accented letters are separators,
        # so "accénts" becomes two tokens rather than one folded one.
        assert tokenize("accénts") == ["acc", "nts"]

    def test_whitespace_kinds_are_equivalent(self):
        assert tokenize("tabs\tand\nnewlines") == ["tabs", "and", "newlines"]


class TestLcsLength:
    def test_identical_sequences(self):
        assert lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_subsequence_not_substring(self):
        assert lcs_length(["a", "b", "c", "d"], ["a", "c"]) == 2

    def test_reversed_sequence_keeps_one(self):
        assert lcs_length(["a", "b", "c"], ["c", "b", "a"]) == 1

    def test_empty_either_side_is_zero(self):
        assert lcs_length([], ["a"]) == 0
        assert lcs_length(["a"], []) == 0


class TestGoldenValues:
    """The equivalence contract, runnable without `rouge-score` installed."""

    @pytest.mark.parametrize("target,prediction,expected", GOLDEN_CASES)
    def test_matches_rouge_score_0_1_2(self, target, prediction, expected):
        assert rouge_l_fmeasure(target, prediction) == pytest.approx(expected, abs=1e-12)

    def test_score_is_bounded(self):
        for target, prediction, _ in GOLDEN_CASES:
            assert 0.0 <= rouge_l_fmeasure(target, prediction) <= 1.0

    def test_empty_prediction_scores_zero_not_an_error(self):
        # An engine that returns nothing must produce a scored 0.0 item, not a
        # crashed task run — `run_quick_task` scores every completion it gets.
        assert rouge_l_fmeasure("a real reference summary", "") == 0.0

    def test_no_third_party_import_in_the_module(self):
        """The reason this module exists: it must stay dependency-free."""
        import ast
        import inspect

        from hmasync_controller.bench.tasks import rouge_l

        tree = ast.parse(inspect.getsource(rouge_l))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= {"re", "__future__"}, imported

    def test_longctx_summary_scores_through_this_module(self):
        """The task's `score` is the caller this replacement exists for."""
        from hmasync_controller.bench.tasks.base import TaskItem
        from hmasync_controller.bench.tasks.longctx_summary import LongctxSummaryTask

        item = TaskItem(item_id="longctx_summary:0", prompt="p",
                        target="The quick brown fox jumps over the lazy dog")
        score = LongctxSummaryTask().score("  The quick brown dog jumps on the log.  ", item)
        assert score == pytest.approx(0.5882352941176471, abs=1e-12)
        assert isinstance(score, float)


class TestDifferentialAgainstRougeScore:
    """Live re-derivation, on any box that still has the dropped dependency."""

    def test_tokenizer_matches_the_reference(self):
        rouge_tokenize = pytest.importorskip("rouge_score.tokenize")
        for target, prediction, _ in GOLDEN_CASES:
            for text in (target, prediction):
                assert tokenize(text) == rouge_tokenize.tokenize(text, None)

    def test_scores_match_the_reference_on_every_case(self):
        rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        pairs = [(t, p) for t, p, _ in GOLDEN_CASES] + _random_pairs()
        for target, prediction in pairs:
            assert rouge_l_fmeasure(target, prediction) == (
                scorer.score(target, prediction)["rougeL"].fmeasure
            ), (target, prediction)
