"""Tests for benchmark task loading, formatting, and scoring (US-MERGE-01).

Dataset fetches hit the network, so these tests exercise the pure logic
(scoring, sampling determinism, registry) and stub the parquet rows.

Ported from energy-bench's tests/unit/test_tasks.py, minus every
HumanEvalPlus-specific test class: humaneval_plus stays lab-side (it executes
model-generated code in a sandbox) and is never registered in this package.
"""

import pytest

from hmasync_controller.bench.tasks import TASK_REGISTRY, UnknownTaskError, load_task
from hmasync_controller.bench.tasks import gpqa_diamond as gpqa_diamond_module
from hmasync_controller.bench.tasks import gsm8k_platinum as gsm8k_platinum_module
from hmasync_controller.bench.tasks import hellaswag as hellaswag_module
from hmasync_controller.bench.tasks import ifeval as ifeval_module
from hmasync_controller.bench.tasks import longctx_summary as longctx_summary_module
from hmasync_controller.bench.tasks import math500 as math500_module
from hmasync_controller.bench.tasks import mmlu_redux as mmlu_redux_module
from hmasync_controller.bench.tasks.base import (
    TaskItem,
    TaskLoadError,
    extract_last_number,
    extract_letter_answer,
    numbers_equal,
    sample_indices,
)
from hmasync_controller.bench.tasks.gpqa_diamond import GPQADiamondTask
from hmasync_controller.bench.tasks.gsm8k import GSM8KTask, _gold_answer
from hmasync_controller.bench.tasks.gsm8k_platinum import GSM8KPlatinumTask
from hmasync_controller.bench.tasks.gsm8k_platinum import _gold_answer as _gold_answer_platinum
from hmasync_controller.bench.tasks.hellaswag import HellaSwagTask
from hmasync_controller.bench.tasks.ifeval import IFEvalTask
from hmasync_controller.bench.tasks.longctx_summary import LongctxSummaryTask
from hmasync_controller.bench.tasks.math500 import (
    Math500Task,
    extract_boxed_answers,
    normalize_math_answer,
)
from hmasync_controller.bench.tasks.mmlu import MMLUTask
from hmasync_controller.bench.tasks.mmlu_redux import MMLUReduxTask


class TestRegistry:
    def test_load_known_tasks(self) -> None:
        assert isinstance(load_task("gsm8k"), GSM8KTask)
        assert isinstance(load_task("gsm8k_platinum"), GSM8KPlatinumTask)
        assert isinstance(load_task("mmlu"), MMLUTask)
        assert isinstance(load_task("mmlu_redux"), MMLUReduxTask)
        assert isinstance(load_task("gpqa_diamond"), GPQADiamondTask)
        assert isinstance(load_task("math500"), Math500Task)
        assert isinstance(load_task("ifeval"), IFEvalTask)
        assert isinstance(load_task("hellaswag"), HellaSwagTask)
        assert isinstance(load_task("longctx_summary"), LongctxSummaryTask)

    def test_unknown_task_raises_with_available_names(self) -> None:
        with pytest.raises(UnknownTaskError, match="gsm8k"):
            load_task("nonexistent")

    def test_registry_tasks_declare_a_power_shape(self) -> None:
        # The prefill/decode split is the hypothesis under test; every task has
        # to state where it sits or the comparison is unanchored.
        for name, cls in TASK_REGISTRY.items():
            assert cls.shape in ("prefill", "decode", "mixed"), name
            assert cls.default_max_tokens > 0, name

    def test_humaneval_plus_is_not_registered(self) -> None:
        # humaneval_plus stays lab-side (sandboxed code execution) — it must
        # not ship in the base controller registry.
        assert "humaneval_plus" not in TASK_REGISTRY

    def test_register_task_extends_the_registry(self) -> None:
        from hmasync_controller.bench.tasks import Task, register_task

        class _FakeTask(Task):
            name = "test_only_fake_task"
            shape = "decode"
            default_max_tokens = 8
            description = "fixture task for test_register_task_extends_the_registry"

            def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
                return []

            def score(self, completion: str, item: TaskItem) -> bool:
                return False

        try:
            assert register_task(_FakeTask) is _FakeTask
            assert TASK_REGISTRY["test_only_fake_task"] is _FakeTask
            assert isinstance(load_task("test_only_fake_task"), _FakeTask)
            with pytest.raises(ValueError, match="test_only_fake_task"):
                register_task(_FakeTask)
        finally:
            del TASK_REGISTRY["test_only_fake_task"]


class TestDatasetRevision:
    """Task.revision exposes the pinned HF revision. None where a task reads
    the default `main` branch."""

    def test_tasks_pinned_to_a_non_default_revision(self) -> None:
        assert GPQADiamondTask.revision == "refs/convert/parquet"
        assert Math500Task.revision == "refs/convert/parquet"
        assert MMLUReduxTask.revision == "refs/convert/parquet"
        assert IFEvalTask.revision == "refs/convert/parquet"

    def test_tasks_on_the_default_branch_have_no_revision(self) -> None:
        assert GSM8KTask.revision is None
        assert GSM8KPlatinumTask.revision is None
        assert MMLUTask.revision is None
        assert HellaSwagTask.revision is None
        assert LongctxSummaryTask.revision is None


class TestCanaryFlag:
    """Task.is_canary defaults False; set on saturated/instrument-grade tasks."""

    def test_hellaswag_is_a_canary(self) -> None:
        assert HellaSwagTask.is_canary is True

    def test_longctx_summary_is_a_canary(self) -> None:
        assert LongctxSummaryTask.is_canary is True

    def test_graded_tasks_are_not_canaries(self) -> None:
        for cls in (
            GSM8KTask,
            GSM8KPlatinumTask,
            MMLUTask,
            MMLUReduxTask,
            GPQADiamondTask,
            Math500Task,
            IFEvalTask,
        ):
            assert cls.is_canary is False, cls.name


class TestSampleIndices:
    def test_deterministic_for_same_seed(self) -> None:
        # Every model in a matrix must see an identical workload.
        assert sample_indices(1000, 10, 1234) == sample_indices(1000, 10, 1234)

    def test_different_seeds_differ(self) -> None:
        assert sample_indices(1000, 10, 1) != sample_indices(1000, 10, 2)

    def test_requesting_more_than_available_returns_all(self) -> None:
        assert sample_indices(5, 100, 1234) == [0, 1, 2, 3, 4]

    def test_returns_exactly_n_unique_in_range(self) -> None:
        got = sample_indices(1000, 10, 1234)
        assert len(got) == 10
        assert len(set(got)) == 10
        assert all(0 <= i < 1000 for i in got)


class TestNumberExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The answer is 72", "72"),
            ("#### 1,000", "1000"),
            ("costs $5.50 total", "5.50"),
            ("answer: -3", "-3"),
            ("ends with a period 72.", "72"),
            ("no digits here", None),
            ("", None),
        ],
    )
    def test_extract_last_number(self, text: str, expected: str | None) -> None:
        assert extract_last_number(text) == expected

    def test_extract_takes_last_not_first(self) -> None:
        assert extract_last_number("5 apples and 7 pears = 12") == "12"

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("72", "72", True),
            ("72.0", "72", True),
            ("72", "71", False),
            (None, "72", False),
            ("not a number", "72", False),
        ],
    )
    def test_numbers_equal(self, a: str | None, b: str | None, expected: bool) -> None:
        assert numbers_equal(a, b) is expected


class TestGSM8KScoring:
    def setup_method(self) -> None:
        self.task = GSM8KTask()
        self.item = TaskItem(item_id="gsm8k:0", prompt="p", target="72")

    def test_gold_answer_extraction(self) -> None:
        assert _gold_answer("Some reasoning.\n#### 1,234") == "1234"

    def test_marker_answer(self) -> None:
        assert self.task.score("Reasoning here.\n#### 72", self.item) is True

    def test_fallback_to_last_number_without_marker(self) -> None:
        assert self.task.score("So the answer is 72", self.item) is True

    def test_wrong_answer(self) -> None:
        assert self.task.score("#### 71", self.item) is False

    def test_no_number_is_incorrect(self) -> None:
        assert self.task.score("I cannot solve this", self.item) is False

    def test_reads_first_marker_not_last(self) -> None:
        # A model that answers, then hallucinates a follow-up question, must be
        # scored on the question we actually asked.
        runaway = "#### 72\n\nQuestion: A different problem?\nAnswer: 5\n#### 5"
        assert self.task.score(runaway, self.item) is True

    def test_declares_stop_sequences(self) -> None:
        # Absent these, runaway generation burns energy on discarded tokens.
        assert self.task.stop, "GSM8K must stop the model continuing the pattern"


class TestGSM8KPlatinumScoring:
    # Scoring logic is identical to gsm8k — mirror those tests exactly so any
    # future scoring change to one is caught if it drifts from the other.
    def setup_method(self) -> None:
        self.task = GSM8KPlatinumTask()
        self.item = TaskItem(item_id="gsm8k_platinum:0", prompt="p", target="72")

    def test_gold_answer_extraction(self) -> None:
        assert _gold_answer_platinum("Some reasoning.\n#### 1,234") == "1234"

    def test_marker_answer(self) -> None:
        assert self.task.score("Reasoning here.\n#### 72", self.item) is True

    def test_fallback_to_last_number_without_marker(self) -> None:
        assert self.task.score("So the answer is 72", self.item) is True

    def test_wrong_answer(self) -> None:
        assert self.task.score("#### 71", self.item) is False

    def test_no_number_is_incorrect(self) -> None:
        assert self.task.score("I cannot solve this", self.item) is False

    def test_reads_first_marker_not_last(self) -> None:
        # A model that answers, then hallucinates a follow-up question, must be
        # scored on the question we actually asked.
        runaway = "#### 72\n\nQuestion: A different problem?\nAnswer: 5\n#### 5"
        assert self.task.score(runaway, self.item) is True

    def test_declares_stop_sequences(self) -> None:
        assert self.task.stop, "gsm8k_platinum must stop the model continuing the pattern"


class TestGSM8KPlatinumLoad:
    """load() tests with mocked HF rows — no network access."""

    TEST_ROWS = tuple(
        {"question": f"Q{i}?", "answer": f"reasoning {i}\n#### {i}"} for i in range(20)
    )
    TRAIN_ROWS = tuple(
        {"question": f"shot-Q{i}?", "answer": f"shot-reasoning {i}\n#### {i}"} for i in range(10)
    )

    def _fake_fetch(self, calls: list[tuple[str, str]]):
        def fetch(repo_id: str, filename: str):
            calls.append((repo_id, filename))
            if repo_id == gsm8k_platinum_module.REPO:
                return self.TEST_ROWS
            return self.TRAIN_ROWS

        return fetch

    def test_test_rows_come_from_platinum_repo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(gsm8k_platinum_module, "fetch_parquet_rows", self._fake_fetch(calls))
        GSM8KPlatinumTask().load(n_items=3, n_shot=0, seed=1234)
        assert (
            gsm8k_platinum_module.REPO,
            gsm8k_platinum_module.TEST_FILE,
        ) in calls
        assert gsm8k_platinum_module.REPO == "madrylab/gsm8k-platinum"

    def test_few_shot_preamble_uses_original_gsm8k_train_repo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Continuity with v1: exemplars must NOT come from the platinum split.
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(gsm8k_platinum_module, "fetch_parquet_rows", self._fake_fetch(calls))
        items = GSM8KPlatinumTask().load(n_items=2, n_shot=2, seed=1234)
        assert (
            gsm8k_platinum_module.FEWSHOT_REPO,
            gsm8k_platinum_module.FEWSHOT_TRAIN_FILE,
        ) in calls
        assert gsm8k_platinum_module.FEWSHOT_REPO == "openai/gsm8k"
        assert "shot-Q" in items[0].prompt

    def test_load_deterministic_for_same_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            gsm8k_platinum_module, "fetch_parquet_rows", self._fake_fetch(calls=[])
        )
        first = GSM8KPlatinumTask().load(n_items=5, n_shot=1, seed=1234)
        second = GSM8KPlatinumTask().load(n_items=5, n_shot=1, seed=1234)
        assert [i.item_id for i in first] == [i.item_id for i in second]
        assert [i.prompt for i in first] == [i.prompt for i in second]

    def test_load_returns_exactly_n_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            gsm8k_platinum_module, "fetch_parquet_rows", self._fake_fetch(calls=[])
        )
        items = GSM8KPlatinumTask().load(n_items=5, n_shot=0, seed=1234)
        assert len(items) == 5
        assert all(item.item_id.startswith("gsm8k_platinum:") for item in items)


class TestMMLUScoring:
    def setup_method(self) -> None:
        self.task = MMLUTask()
        self.item = TaskItem(item_id="mmlu:0", prompt="p", target="C")

    @pytest.mark.parametrize("completion", ["C", "C.", "The answer is C", "c", " C "])
    def test_accepts_letter_forms(self, completion: str) -> None:
        assert self.task.score(completion, self.item) is True

    @pytest.mark.parametrize("completion", ["B", "", "none of these"])
    def test_rejects_wrong_or_absent(self, completion: str) -> None:
        assert self.task.score(completion, self.item) is False

    def test_takes_leading_letter_when_model_adds_an_aside(self) -> None:
        # The instructed format is "only the letter", so a completion that
        # OPENS with one has answered — a trailing mention is an aside, not a
        # revision.
        assert self.task.score("C, though B is tempting", self.item) is True

    def test_marker_beats_a_leading_distractor(self) -> None:
        # The case that made first-match untenable once the cap was raised:
        # the earliest letter is the one being rejected.
        assert self.task.score(
            "A is tempting but incorrect, so the answer is C", self.item
        ) is True

    def test_trailing_conclusion_without_a_marker(self) -> None:
        assert self.task.score("A is wrong, B is wrong, leaving C", self.item) is True


class TestExtractLetterAnswer:
    """The shared MC extractor (tasks/base.py), used by all four letter tasks.

    default_max_tokens is 512 (not a tiny answer-only cap), so completions
    can be whole reasoning traces rather than a single token — these pin both
    shapes: answer-first-then-aside, and reason-then-conclude.
    """

    @pytest.mark.parametrize(
        "completion,expected",
        [
            ("C", "C"),
            ("C.", "C"),
            (" c ", "C"),
            ("C) because the others fail", "C"),
            ("C, though B is tempting", "C"),
            ("The answer is C", "C"),
            ("A is tempting but incorrect, so the answer is B", "B"),
            ("Let me think. Option A says X, but wrong. Answer: D", "D"),
            ("We can rule out B. Final answer: **C**", "C"),
            ("Reasoning shows the third option, so it must be B", "B"),
            # Opens with a letter, but as the SUBJECT of a rejection rather
            # than as an answer — "A" + space + word is prose, so the leading
            # pass must decline it.
            ("A is wrong, B is wrong, leaving C", "C"),
            ("D is the only one that survives scrutiny", "D"),
        ],
    )
    def test_extracts(self, completion: str, expected: str) -> None:
        assert extract_letter_answer(completion) == expected

    @pytest.mark.parametrize(
        "completion", ["", "   ", "I need to analyse each option carefully and"]
    )
    def test_none_when_no_letter_present(self, completion: str) -> None:
        # Distinct from a wrong answer: this is truncated or off-format.
        assert extract_letter_answer(completion) is None


class TestMCTasksShareTheRaisedCap:
    def test_letter_tasks_allow_room_to_reason(self) -> None:
        """8 tokens scored a reasoning model 0% on every item.

        512 is measured to fit every context window in the roster.
        """
        for cls in (MMLUTask, MMLUReduxTask, GPQADiamondTask, HellaSwagTask):
            assert cls.default_max_tokens == 512, cls.__name__


class TestMMLUReduxScoring:
    # score() is byte-identical to mmlu's — mirror those tests exactly so any
    # future scoring change to one is caught if it drifts from the other.
    def setup_method(self) -> None:
        self.task = MMLUReduxTask()
        self.item = TaskItem(item_id="mmlu_redux:0", prompt="p", target="C")

    @pytest.mark.parametrize("completion", ["C", "C.", "The answer is C", "c", " C "])
    def test_accepts_letter_forms(self, completion: str) -> None:
        assert self.task.score(completion, self.item) is True

    @pytest.mark.parametrize("completion", ["B", "", "none of these"])
    def test_rejects_wrong_or_absent(self, completion: str) -> None:
        assert self.task.score(completion, self.item) is False

    def test_takes_first_letter_when_model_rambles(self) -> None:
        assert self.task.score("C, though B is tempting", self.item) is True


class TestMMLUReduxLoad:
    """load() tests with mocked HF rows — no network access."""

    SUBJECT_ROWS = {
        "abstract_algebra": [
            {
                "question": "Q-ok-0?",
                "choices": ["a", "b", "c", "d"],
                "answer": 0,
                "error_type": "ok",
            },
            {
                "question": "Q-bad-1?",
                "choices": ["a", "b", "c", "d"],
                "answer": 1,
                "error_type": "wrong_groundtruth",
            },
        ],
        "anatomy": [
            {
                "question": "Q-ok-2?",
                "choices": ["a", "b", "c", "d"],
                "answer": 2,
                "error_type": "ok",
            },
            {
                "question": "Q-ok-3?",
                "choices": ["a", "b", "c", "d"],
                "answer": 3,
                "error_type": "ok",
            },
        ],
    }
    DEV_ROWS = tuple(
        {"question": f"shot-Q{i}?", "choices": ["a", "b", "c", "d"], "answer": i % 4}
        for i in range(5)
    )

    def _fake_fetch(self, calls: list[tuple[str, str, str]]):
        def fetch(repo_id: str, filename: str, revision: str = "main"):
            calls.append((repo_id, filename, revision))
            if repo_id == mmlu_redux_module.FEWSHOT_REPO:
                return self.DEV_ROWS
            subject = filename.split("/")[0]
            return tuple(self.SUBJECT_ROWS.get(subject, []))

        return fetch

    def test_filters_to_ok_error_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(mmlu_redux_module, "fetch_parquet_rows", self._fake_fetch(calls))
        items = MMLUReduxTask().load(n_items=3, n_shot=0, seed=1234)
        assert len(items) == 3
        assert all("Q-bad" not in item.prompt for item in items)

    def test_subject_metadata_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(mmlu_redux_module, "fetch_parquet_rows", self._fake_fetch(calls))
        items = MMLUReduxTask().load(n_items=3, n_shot=0, seed=1234)
        subjects = {item.metadata["subject"] for item in items}
        assert subjects <= {"abstract_algebra", "anatomy"}
        assert all(item.metadata["error_type"] == "ok" for item in items)

    def test_test_rows_fetched_from_redux_repo_on_convert_revision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(mmlu_redux_module, "fetch_parquet_rows", self._fake_fetch(calls))
        MMLUReduxTask().load(n_items=3, n_shot=0, seed=1234)
        assert (
            mmlu_redux_module.REPO,
            "abstract_algebra/test/0000.parquet",
            mmlu_redux_module.REVISION,
        ) in calls
        assert mmlu_redux_module.REPO == "edinburgh-dawg/mmlu-redux-2.0"
        assert mmlu_redux_module.REVISION == "refs/convert/parquet"

    def test_few_shot_preamble_uses_cais_mmlu_dev_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(mmlu_redux_module, "fetch_parquet_rows", self._fake_fetch(calls))
        items = MMLUReduxTask().load(n_items=2, n_shot=2, seed=1234)
        assert (
            mmlu_redux_module.FEWSHOT_REPO,
            mmlu_redux_module.FEWSHOT_DEV_FILE,
            "main",
        ) in calls
        assert mmlu_redux_module.FEWSHOT_REPO == "cais/mmlu"
        assert "shot-Q" in items[0].prompt

    def test_load_deterministic_for_same_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mmlu_redux_module, "fetch_parquet_rows", self._fake_fetch(calls=[]))
        first = MMLUReduxTask().load(n_items=3, n_shot=0, seed=1234)
        second = MMLUReduxTask().load(n_items=3, n_shot=0, seed=1234)
        assert [i.item_id for i in first] == [i.item_id for i in second]
        assert [i.prompt for i in first] == [i.prompt for i in second]

    def test_load_returns_exactly_n_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mmlu_redux_module, "fetch_parquet_rows", self._fake_fetch(calls=[]))
        items = MMLUReduxTask().load(n_items=3, n_shot=0, seed=1234)
        assert len(items) == 3
        assert all(item.item_id.startswith("mmlu_redux:") for item in items)


class TestHellaSwagScoring:
    # score() is the same letter-extraction logic as mmlu_redux's/mmlu's —
    # mirror those tests exactly so a scoring change to one is caught if it
    # drifts from the others.
    def setup_method(self) -> None:
        self.task = HellaSwagTask()
        self.item = TaskItem(item_id="hellaswag:0", prompt="p", target="D")

    @pytest.mark.parametrize("completion", ["D", "D.", "The answer is D", "d", " D "])
    def test_accepts_letter_forms(self, completion: str) -> None:
        assert self.task.score(completion, self.item) is True

    @pytest.mark.parametrize("completion", ["B", "", "none of these"])
    def test_rejects_wrong_or_absent(self, completion: str) -> None:
        assert self.task.score(completion, self.item) is False

    def test_takes_first_letter_when_model_rambles(self) -> None:
        assert self.task.score("D, though B is tempting", self.item) is True


class TestHellaSwagLoad:
    """load() tests with mocked HF rows — no network access."""

    VALIDATION_ROWS = tuple(
        {
            "ctx": f"A person does thing {i}.",
            "endings": [f"ending-{i}-{j}" for j in range(4)],
            "label": str(i % 4),
            "activity_label": f"activity-{i}",
            "split_type": "indomain",
        }
        for i in range(5)
    )
    TRAIN_ROWS = tuple(
        {
            "ctx": f"shot-ctx-{i}",
            "endings": [f"shot-ending-{i}-{j}" for j in range(4)],
            "label": str(i % 4),
            "activity_label": f"shot-activity-{i}",
            "split_type": "indomain",
        }
        for i in range(3)
    )

    def _fake_fetch(self, calls: list[tuple[str, str, str]]):
        def fetch(repo_id: str, filename: str, revision: str = "main"):
            calls.append((repo_id, filename, revision))
            if filename == hellaswag_module.FEWSHOT_TRAIN_FILE:
                return self.TRAIN_ROWS
            return self.VALIDATION_ROWS

        return fetch

    def test_validation_split_fetched_on_default_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(hellaswag_module, "fetch_parquet_rows", self._fake_fetch(calls))
        HellaSwagTask().load(n_items=3, n_shot=0, seed=1234)
        assert (hellaswag_module.REPO, hellaswag_module.VALIDATION_FILE, "main") in calls
        assert hellaswag_module.REPO == "Rowan/hellaswag"

    def test_target_matches_label_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            hellaswag_module, "fetch_parquet_rows", self._fake_fetch(calls=[])
        )
        items = HellaSwagTask().load(n_items=5, n_shot=0, seed=1234)
        by_idx = {int(item.item_id.split(":")[1]): item for item in items}
        for idx, item in by_idx.items():
            assert item.target == hellaswag_module.LETTERS[idx % 4]

    def test_activity_metadata_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            hellaswag_module, "fetch_parquet_rows", self._fake_fetch(calls=[])
        )
        items = HellaSwagTask().load(n_items=3, n_shot=0, seed=1234)
        assert all(item.metadata["activity_label"].startswith("activity-") for item in items)
        assert all(item.metadata["split_type"] == "indomain" for item in items)

    def test_few_shot_preamble_uses_train_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(hellaswag_module, "fetch_parquet_rows", self._fake_fetch(calls))
        items = HellaSwagTask().load(n_items=2, n_shot=2, seed=1234)
        assert (hellaswag_module.REPO, hellaswag_module.FEWSHOT_TRAIN_FILE, "main") in calls
        assert "shot-ctx-0" in items[0].prompt

    def test_zero_shot_skips_train_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(hellaswag_module, "fetch_parquet_rows", self._fake_fetch(calls))
        HellaSwagTask().load(n_items=2, n_shot=0, seed=1234)
        assert hellaswag_module.FEWSHOT_TRAIN_FILE not in {f for _, f, _ in calls}

    def test_load_deterministic_for_same_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            hellaswag_module, "fetch_parquet_rows", self._fake_fetch(calls=[])
        )
        first = HellaSwagTask().load(n_items=3, n_shot=0, seed=1234)
        second = HellaSwagTask().load(n_items=3, n_shot=0, seed=1234)
        assert [i.item_id for i in first] == [i.item_id for i in second]
        assert [i.prompt for i in first] == [i.prompt for i in second]

    def test_load_returns_exactly_n_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            hellaswag_module, "fetch_parquet_rows", self._fake_fetch(calls=[])
        )
        items = HellaSwagTask().load(n_items=3, n_shot=0, seed=1234)
        assert len(items) == 3
        assert all(item.item_id.startswith("hellaswag:") for item in items)


class TestGPQADiamondScoring:
    # score() is the same letter-extraction logic as mmlu's — mirror those
    # tests exactly so a scoring change to one is caught if it drifts.
    def setup_method(self) -> None:
        self.task = GPQADiamondTask()
        self.item = TaskItem(item_id="gpqa_diamond:0", prompt="p", target="C")

    @pytest.mark.parametrize("completion", ["C", "C.", "The answer is C", "c", " C "])
    def test_accepts_letter_forms(self, completion: str) -> None:
        assert self.task.score(completion, self.item) is True

    @pytest.mark.parametrize("completion", ["B", "", "none of these"])
    def test_rejects_wrong_or_absent(self, completion: str) -> None:
        assert self.task.score(completion, self.item) is False

    def test_takes_first_letter_when_model_rambles(self) -> None:
        assert self.task.score("C, though B is tempting", self.item) is True


class _FakeHTTPResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeGatedRepoError(Exception):
    """Stand-in for huggingface_hub's GatedRepoError: carries a `.response`."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"{status_code} Client Error: gated repo")
        self.response = _FakeHTTPResponse(status_code)


class TestGPQADiamondLoad:
    """load() tests with mocked HF rows — no network access."""

    TEST_ROWS = tuple(
        {
            "Question": f"Q{i}?",
            "Correct Answer": f"Correct-{i}",
            "Incorrect Answer 1": f"Wrong1-{i}",
            "Incorrect Answer 2": f"Wrong2-{i}",
            "Incorrect Answer 3": f"Wrong3-{i}",
        }
        for i in range(20)
    )

    def _fake_fetch(self, calls: list[tuple[str, str, str]] | None = None):
        if calls is None:
            calls = []

        def fetch(repo_id: str, filename: str, revision: str = "main"):
            calls.append((repo_id, filename, revision))
            return self.TEST_ROWS

        return fetch

    def test_test_rows_fetched_from_gpqa_repo_on_convert_revision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(gpqa_diamond_module, "fetch_parquet_rows", self._fake_fetch(calls))
        GPQADiamondTask().load(n_items=3, n_shot=0, seed=1234)
        assert (
            gpqa_diamond_module.REPO,
            gpqa_diamond_module.TEST_FILE,
            gpqa_diamond_module.REVISION,
        ) in calls
        assert gpqa_diamond_module.REPO == "Idavidrein/gpqa"
        assert gpqa_diamond_module.REVISION == "refs/convert/parquet"

    def test_zero_shot_ignores_n_shot_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(gpqa_diamond_module, "fetch_parquet_rows", self._fake_fetch(calls))
        no_shot = GPQADiamondTask().load(n_items=3, n_shot=0, seed=1234)
        calls.clear()
        with_shot_arg = GPQADiamondTask().load(n_items=3, n_shot=5, seed=1234)
        # Exactly one fetch call either way: no separate few-shot file exists.
        assert len(calls) == 1
        assert [i.prompt for i in no_shot] == [i.prompt for i in with_shot_arg]

    def test_all_four_options_present_and_target_matches_correct_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpqa_diamond_module, "fetch_parquet_rows", self._fake_fetch())
        items = GPQADiamondTask().load(n_items=10, n_shot=0, seed=99)
        for item in items:
            idx = int(item.item_id.split(":")[1])
            row = self.TEST_ROWS[idx]
            for option in (
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ):
                assert option in item.prompt
            assert f"{item.target}. {row['Correct Answer']}" in item.prompt

    def test_option_order_deterministic_for_fixed_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpqa_diamond_module, "fetch_parquet_rows", self._fake_fetch())
        first = GPQADiamondTask().load(n_items=5, n_shot=0, seed=1234)
        second = GPQADiamondTask().load(n_items=5, n_shot=0, seed=1234)
        assert [i.prompt for i in first] == [i.prompt for i in second]
        assert [i.target for i in first] == [i.target for i in second]

    def test_different_seeds_can_shuffle_options_differently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpqa_diamond_module, "fetch_parquet_rows", self._fake_fetch())
        first = GPQADiamondTask().load(n_items=10, n_shot=0, seed=1)
        second = GPQADiamondTask().load(n_items=10, n_shot=0, seed=2)
        assert [i.prompt for i in first] != [i.prompt for i in second]

    def test_load_returns_exactly_n_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpqa_diamond_module, "fetch_parquet_rows", self._fake_fetch())
        items = GPQADiamondTask().load(n_items=5, n_shot=0, seed=1234)
        assert len(items) == 5
        assert all(item.item_id.startswith("gpqa_diamond:") for item in items)

    def test_gated_access_error_names_hf_token_and_accept_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_gated(*args: object, **kwargs: object) -> None:
            cause = _FakeGatedRepoError(403)
            raise TaskLoadError(f"Could not load {gpqa_diamond_module.REPO}: 403") from cause

        monkeypatch.setattr(gpqa_diamond_module, "fetch_parquet_rows", _raise_gated)
        with pytest.raises(TaskLoadError) as exc_info:
            GPQADiamondTask().load(n_items=1, n_shot=0, seed=1234)
        message = str(exc_info.value)
        assert "HF_TOKEN" in message
        assert "access" in message.lower()

    def test_non_gated_load_error_propagates_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = TaskLoadError("network blip, nothing to do with gating")

        def _raise(*args: object, **kwargs: object) -> None:
            raise original

        monkeypatch.setattr(gpqa_diamond_module, "fetch_parquet_rows", _raise)
        with pytest.raises(TaskLoadError) as exc_info:
            GPQADiamondTask().load(n_items=1, n_shot=0, seed=1234)
        assert exc_info.value is original


class TestMath500BoxedExtraction:
    def test_single_box(self) -> None:
        assert extract_boxed_answers("The answer is \\boxed{42}.") == ["42"]

    def test_nested_braces(self) -> None:
        assert extract_boxed_answers("\\boxed{\\frac{1}{2}}") == ["\\frac{1}{2}"]

    def test_deeply_nested_braces(self) -> None:
        text = "\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}"
        assert extract_boxed_answers(text) == ["\\left( 3, \\frac{\\pi}{2} \\right)"]

    def test_multiple_boxes_returns_all_in_order(self) -> None:
        text = "First \\boxed{1} then \\boxed{2} then \\boxed{3}"
        assert extract_boxed_answers(text) == ["1", "2", "3"]

    def test_no_box_returns_empty(self) -> None:
        assert extract_boxed_answers("no boxed answer here, just 42") == []

    def test_unbalanced_box_ignored(self) -> None:
        assert extract_boxed_answers("\\boxed{oops no close") == []


class TestMath500Normalization:
    def test_strips_whitespace(self) -> None:
        assert normalize_math_answer("  3 + 4  ") == "3+4"

    def test_strips_left_right(self) -> None:
        assert normalize_math_answer("\\left( 3, \\frac{\\pi}{2} \\right)") == "(3,\\frac{\\pi}{2})"

    def test_strips_thin_space_command(self) -> None:
        assert normalize_math_answer("3\\!.5") == "3.5"

    def test_strips_trailing_dot_zero(self) -> None:
        assert normalize_math_answer("72.0") == "72"

    def test_does_not_strip_non_trailing_dot_zero(self) -> None:
        assert normalize_math_answer("100") == "100"


class TestMath500Scoring:
    def setup_method(self) -> None:
        self.task = Math500Task()

    def test_boxed_exact_match(self) -> None:
        item = TaskItem(item_id="math500:0", prompt="p", target="72")
        assert self.task.score("Work it out.\n\\boxed{72}", item) is True

    def test_boxed_wrong_answer(self) -> None:
        item = TaskItem(item_id="math500:0", prompt="p", target="72")
        assert self.task.score("\\boxed{71}", item) is False

    def test_boxed_matches_after_latex_normalization(self) -> None:
        item = TaskItem(item_id="math500:0", prompt="p", target="\\left(3,\\frac{\\pi}{2}\\right)")
        completion = "So the point is \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}."
        assert self.task.score(completion, item) is True

    def test_boxed_numeric_fallback_tolerates_trailing_zero(self) -> None:
        item = TaskItem(item_id="math500:0", prompt="p", target="72")
        assert self.task.score("\\boxed{72.0}", item) is True

    def test_last_box_wins(self) -> None:
        # Unlike gsm8k's "first marker" rule, math500 takes the LAST \boxed{}
        # — a model that self-corrects commits to its final box.
        completion = "First attempt \\boxed{5}, but on reflection \\boxed{9}."
        item = TaskItem(item_id="math500:0", prompt="p", target="9")
        assert self.task.score(completion, item) is True
        item_wrong = TaskItem(item_id="math500:0", prompt="p", target="5")
        assert self.task.score(completion, item_wrong) is False

    def test_no_box_falls_back_to_last_number(self) -> None:
        item = TaskItem(item_id="math500:0", prompt="p", target="72")
        assert self.task.score("After computing, the answer is 72", item) is True

    def test_no_box_no_number_is_incorrect(self) -> None:
        item = TaskItem(item_id="math500:0", prompt="p", target="72")
        assert self.task.score("I cannot solve this", item) is False

    def test_declares_stop_sequences(self) -> None:
        assert self.task.stop, "math500 must stop the model continuing the pattern"


class TestMath500Load:
    """load() tests with mocked HF rows — no network access."""

    TEST_ROWS = tuple(
        {
            "problem": f"Solve P{i}.",
            "answer": f"{i}",
            "subject": "Algebra",
            "level": 3,
        }
        for i in range(20)
    )
    FEWSHOT_ROWS = tuple(
        {"problem": f"Shot-P{i}.", "solution": f"...\\boxed{{{i}}}"} for i in range(10)
    )

    def _fake_fetch(self, calls: list[tuple[str, ...]]):
        def fetch(repo_id: str, filename: str, revision: str = "main"):
            calls.append((repo_id, filename, revision))
            if repo_id == math500_module.FEWSHOT_REPO:
                return self.FEWSHOT_ROWS
            return self.TEST_ROWS

        return fetch

    def test_test_rows_fetched_from_math500_repo_on_convert_revision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, ...]] = []
        monkeypatch.setattr(math500_module, "fetch_parquet_rows", self._fake_fetch(calls))
        Math500Task().load(n_items=3, n_shot=0, seed=1234)
        assert (math500_module.REPO, math500_module.TEST_FILE, math500_module.REVISION) in calls
        assert math500_module.REPO == "HuggingFaceH4/MATH-500"
        assert math500_module.REVISION == "refs/convert/parquet"

    def test_few_shot_preamble_uses_hendrycks_math_train_repo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, ...]] = []
        monkeypatch.setattr(math500_module, "fetch_parquet_rows", self._fake_fetch(calls))
        items = Math500Task().load(n_items=2, n_shot=2, seed=1234)
        assert (math500_module.FEWSHOT_REPO, math500_module.FEWSHOT_FILE, "main") in calls
        assert math500_module.FEWSHOT_REPO == "EleutherAI/hendrycks_math"
        assert "Shot-P" in items[0].prompt

    def test_zero_shot_omits_fewshot_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, ...]] = []
        monkeypatch.setattr(math500_module, "fetch_parquet_rows", self._fake_fetch(calls))
        Math500Task().load(n_items=2, n_shot=0, seed=1234)
        assert len(calls) == 1

    def test_subject_and_level_metadata_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(math500_module, "fetch_parquet_rows", self._fake_fetch(calls=[]))
        items = Math500Task().load(n_items=3, n_shot=0, seed=1234)
        assert all(item.metadata["subject"] == "Algebra" for item in items)
        assert all(item.metadata["level"] == 3 for item in items)

    def test_load_deterministic_for_same_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(math500_module, "fetch_parquet_rows", self._fake_fetch(calls=[]))
        first = Math500Task().load(n_items=5, n_shot=1, seed=1234)
        second = Math500Task().load(n_items=5, n_shot=1, seed=1234)
        assert [i.item_id for i in first] == [i.item_id for i in second]
        assert [i.prompt for i in first] == [i.prompt for i in second]

    def test_load_returns_exactly_n_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(math500_module, "fetch_parquet_rows", self._fake_fetch(calls=[]))
        items = Math500Task().load(n_items=5, n_shot=0, seed=1234)
        assert len(items) == 5
        assert all(item.item_id.startswith("math500:") for item in items)


class TestIFEvalLoad:
    """load() tests with mocked HF rows — no network access."""

    TEST_ROWS = tuple(
        {
            "key": 1000 + i,
            "prompt": f"Write response number {i} without using any commas.",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [
                {
                    "num_highlights": None,
                    "relation": None,
                    "num_words": None,
                    "num_placeholders": None,
                    "prompt_to_repeat": None,
                    "num_bullets": None,
                    "section_spliter": None,
                    "num_sections": None,
                    "capital_relation": None,
                    "capital_frequency": None,
                    "keywords": None,
                    "num_paragraphs": None,
                    "language": None,
                    "let_relation": None,
                    "letter": None,
                    "let_frequency": None,
                    "end_phrase": None,
                    "forbidden_words": None,
                    "keyword": None,
                    "frequency": None,
                    "num_sentences": None,
                    "postscript_marker": None,
                    "first_word": None,
                    "nth_paragraph": None,
                }
            ],
        }
        for i in range(20)
    )

    def _fake_fetch(self, calls: list[tuple[str, str, str]]):
        def fetch(repo_id: str, filename: str, revision: str = "main"):
            calls.append((repo_id, filename, revision))
            return self.TEST_ROWS

        return fetch

    def test_rows_fetched_from_ifeval_repo_on_convert_revision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(ifeval_module, "fetch_parquet_rows", self._fake_fetch(calls))
        IFEvalTask().load(n_items=3, n_shot=0, seed=1234)
        assert (ifeval_module.REPO, ifeval_module.DATA_FILE, ifeval_module.REVISION) in calls
        assert ifeval_module.REPO == "google/IFEval"
        assert ifeval_module.REVISION == "refs/convert/parquet"

    def test_zero_shot_ignores_n_shot_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ifeval_module, "fetch_parquet_rows", self._fake_fetch(calls=[]))
        no_shot = IFEvalTask().load(n_items=3, n_shot=0, seed=1234)
        with_shot = IFEvalTask().load(n_items=3, n_shot=5, seed=1234)
        assert [i.prompt for i in no_shot] == [i.prompt for i in with_shot]

    def test_prompt_is_the_raw_dataset_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ifeval_module, "fetch_parquet_rows", self._fake_fetch(calls=[]))
        items = IFEvalTask().load(n_items=1, n_shot=0, seed=1234)
        idx = int(items[0].item_id.split(":")[1])
        assert items[0].prompt == self.TEST_ROWS[idx]["prompt"]

    def test_metadata_carries_instruction_id_list_and_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ifeval_module, "fetch_parquet_rows", self._fake_fetch(calls=[]))
        items = IFEvalTask().load(n_items=3, n_shot=0, seed=1234)
        for item in items:
            assert item.metadata["instruction_id_list"] == ["punctuation:no_comma"]
            assert len(item.metadata["kwargs"]) == 1
            assert "key" in item.metadata

    def test_load_deterministic_for_same_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ifeval_module, "fetch_parquet_rows", self._fake_fetch(calls=[]))
        first = IFEvalTask().load(n_items=5, n_shot=0, seed=1234)
        second = IFEvalTask().load(n_items=5, n_shot=0, seed=1234)
        assert [i.item_id for i in first] == [i.item_id for i in second]
        assert [i.prompt for i in first] == [i.prompt for i in second]

    def test_load_returns_exactly_n_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ifeval_module, "fetch_parquet_rows", self._fake_fetch(calls=[]))
        items = IFEvalTask().load(n_items=5, n_shot=0, seed=1234)
        assert len(items) == 5
        assert all(item.item_id.startswith("ifeval:") for item in items)


class TestIFEvalScoring:
    """score() requires EVERY instruction in the item's list to pass."""

    def _item(self, instruction_id_list: list[str], kwargs_list: list[dict]) -> TaskItem:
        return TaskItem(
            item_id="ifeval:0",
            prompt="p",
            target=";".join(instruction_id_list),
            metadata={"key": 0, "instruction_id_list": instruction_id_list, "kwargs": kwargs_list},
        )

    def setup_method(self) -> None:
        self.task = IFEvalTask()

    def test_known_pass_response_satisfies_every_instruction(self) -> None:
        item = self._item(
            instruction_id_list=["punctuation:no_comma", "detectable_format:title"],
            kwargs_list=[{}, {}],
        )
        completion = "<<My Title>> This response has no forbidden punctuation at all"
        assert self.task.score(completion, item) is True

    def test_known_fail_response_violates_one_instruction(self) -> None:
        item = self._item(
            instruction_id_list=["punctuation:no_comma", "detectable_format:title"],
            kwargs_list=[{}, {}],
        )
        # Has the title but also a comma, which no_comma forbids.
        completion = "<<My Title>> This response, unfortunately, has commas"
        assert self.task.score(completion, item) is False

    def test_all_instructions_must_pass_not_just_one(self) -> None:
        item = self._item(
            instruction_id_list=["punctuation:no_comma", "startend:quotation"],
            kwargs_list=[{}, {}],
        )
        # Satisfies no_comma but not the quotation wrapper.
        assert self.task.score("no commas here", item) is False
        # Satisfies both.
        assert self.task.score('"no commas here"', item) is True


class TestLongctxSummaryLoad:
    """longctx_summary buckets GovReport documents into fixed token-count
    ranges; reports over MAX_TOKENS are truncated, under MIN_TOKENS are
    excluded from sampling entirely."""

    # CHARS_PER_TOKEN=4: idx0 (400 chars ~= 100 tokens) is below MIN_TOKENS
    # and excluded; idx1 (40,000 chars ~= 10,000 tokens) lands in "8k-16k";
    # idx2 (72,000 chars ~= 18,000 tokens) lands in "16k-24k"; idx3
    # (200,000 chars) is truncated to MAX_TOKENS*CHARS_PER_TOKEN=128,000
    # chars, landing exactly on the "24k-32k" upper boundary.
    FIXTURE_ROWS = (
        {"report": "a" * 400, "summary": "too short to stress prefill"},
        {"report": "b" * 40_000, "summary": "mid-length summary"},
        {"report": "c" * 72_000, "summary": "long summary"},
        {"report": "d" * 200_000, "summary": "huge doc summary"},
    )

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            longctx_summary_module,
            "fetch_parquet_rows",
            lambda repo, filename: self.FIXTURE_ROWS,
        )

    def test_short_reports_are_excluded_from_sampling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch)
        items = LongctxSummaryTask().load(n_items=10, n_shot=0, seed=0)
        assert [item.item_id for item in items] == [
            "longctx_summary:1",
            "longctx_summary:2",
            "longctx_summary:3",
        ]

    def test_token_bucket_assigned_per_item(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch)
        items = {i.item_id: i for i in LongctxSummaryTask().load(n_items=10, n_shot=0, seed=0)}
        assert items["longctx_summary:1"].metadata["token_bucket"] == "8k-16k"
        assert items["longctx_summary:2"].metadata["token_bucket"] == "16k-24k"
        assert items["longctx_summary:3"].metadata["token_bucket"] == "24k-32k"

    def test_over_max_tokens_report_is_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch)
        items = {i.item_id: i for i in LongctxSummaryTask().load(n_items=10, n_shot=0, seed=0)}
        huge = items["longctx_summary:3"]
        assert huge.metadata["estimated_prompt_tokens"] == 32_000
        body = huge.prompt.split("Report:\n", 1)[1].rsplit("\n\nSummary:", 1)[0]
        assert len(body) == 32_000 * longctx_summary_module.CHARS_PER_TOKEN

    def test_target_is_the_reference_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch)
        items = {i.item_id: i for i in LongctxSummaryTask().load(n_items=10, n_shot=0, seed=0)}
        assert items["longctx_summary:1"].target == "mid-length summary"

    def test_n_shot_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Zero-shot only (a report-length exemplar would itself blow past a
        # bucket boundary) -- n_shot > 0 must not raise or change the result.
        self._patch(monkeypatch)
        items = LongctxSummaryTask().load(n_items=1, n_shot=5, seed=0)
        assert len(items) == 1

    def test_load_deterministic_for_same_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch)
        first = LongctxSummaryTask().load(n_items=2, n_shot=0, seed=42)
        second = LongctxSummaryTask().load(n_items=2, n_shot=0, seed=42)
        assert [i.item_id for i in first] == [i.item_id for i in second]

    def test_load_returns_exactly_n_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch)
        items = LongctxSummaryTask().load(n_items=2, n_shot=0, seed=0)
        assert len(items) == 2


class TestLongctxSummaryScoring:
    """score() returns a continuous ROUGE-L F1 float, not a bool."""

    def _item(self, target: str) -> TaskItem:
        return TaskItem(item_id="longctx_summary:0", prompt="p", target=target)

    def test_identical_text_scores_near_one(self) -> None:
        task = LongctxSummaryTask()
        item = self._item("the quick brown fox jumps over the lazy dog")
        score = task.score("the quick brown fox jumps over the lazy dog", item)
        assert score == pytest.approx(1.0)

    def test_unrelated_text_scores_near_zero(self) -> None:
        task = LongctxSummaryTask()
        item = self._item("the quick brown fox jumps over the lazy dog")
        score = task.score("completely different words entirely", item)
        assert score == pytest.approx(0.0, abs=0.05)

    def test_partial_overlap_scores_strictly_between(self) -> None:
        task = LongctxSummaryTask()
        item = self._item("alpha beta gamma delta epsilon")
        score = task.score("alpha beta gamma zeta eta", item)
        assert 0.0 < score < 1.0

    def test_score_is_a_float_not_a_bool(self) -> None:
        task = LongctxSummaryTask()
        item = self._item("alpha beta gamma")
        score = task.score("alpha beta gamma", item)
        assert isinstance(score, float)
        assert not isinstance(score, bool)

    def test_empty_completion_scores_zero_not_a_crash(self) -> None:
        task = LongctxSummaryTask()
        item = self._item("alpha beta gamma")
        assert task.score("", item) == 0.0
