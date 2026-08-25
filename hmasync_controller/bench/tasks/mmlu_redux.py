"""MMLU-Redux — human re-annotated MMLU test set, filtered to `error_type == "ok"`.

MMLU-Redux (Gema et al.) hand-audits each MMLU test question and tags it with
an `error_type`; `"ok"` rows are the ones the annotators found unmodified from
the original MMLU (no wrong ground truth, no bad question, etc.) — this task
is the cleaned drop-in. Same generative letter-answer protocol as `mmlu.py`;
see that module's docstring for the prefill-shape rationale this task shares.

`edinburgh-dawg/mmlu-redux-2.0` does not publish a single combined parquet the
way `cais/mmlu` does — its `main` branch only ships per-subject HF `datasets`
arrow files. The hub's auto-converted parquet mirror on `refs/convert/parquet`
does have one `<subject>/test/0000.parquet` per subject, with all 57 standard
MMLU subjects present (verified live), so the documented fallback repo
(`edinburgh-dawg/mmlu-redux`, whose auto-converted mirror only covers ~30
subjects) was not needed.

Few-shot exemplars come from `cais/mmlu`'s dev split, exactly as `mmlu.py`
uses it: the first `n_shot` rows regardless of subject, not subject-matched.
"""


from hmasync_controller.bench.tasks.base import (
    Task,
    TaskItem,
    extract_letter_answer,
    fetch_parquet_rows,
    sample_indices,
)

REPO = "edinburgh-dawg/mmlu-redux-2.0"
REVISION = "refs/convert/parquet"

FEWSHOT_REPO = "cais/mmlu"
FEWSHOT_DEV_FILE = "all/dev-00000-of-00001.parquet"

OK_ERROR_TYPE = "ok"

LETTERS = ["A", "B", "C", "D"]

# The 57 standard MMLU subjects, each a `<subject>/test/0000.parquet` file in
# REPO@REVISION. Hardcoded rather than discovered via list_repo_files at load
# time: it's a fixed taxonomy, and a live directory listing on every load()
# call would be one more network round-trip per run for no benefit.
SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def _format_question(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question.strip()}"]
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    return "\n".join(lines)


def _fetch_ok_rows() -> list[dict]:
    """Fetch every subject file and keep only `error_type == "ok"` rows.

    Row order is subject-major (SUBJECTS order, then file order within a
    subject) so `sample_indices` is deterministic given REPO/REVISION content.
    """
    rows: list[dict] = []
    for subject in SUBJECTS:
        subject_rows = fetch_parquet_rows(REPO, f"{subject}/test/0000.parquet", REVISION)
        for row in subject_rows:
            if row.get("error_type") != OK_ERROR_TYPE:
                continue
            rows.append({**row, "subject": subject})
    return rows


class MMLUReduxTask(Task):
    """MMLU-Redux: re-annotated MMLU, exact-match on the answer letter."""

    name = "mmlu_redux"
    shape = "prefill"
    default_max_tokens = 512
    description = (
        "Re-annotated MMLU (error_type == 'ok' rows only), same letter-answer protocol as mmlu"
    )
    stop = ["\n\n", "Question:"]
    revision = REVISION

    def load(self, n_items: int, n_shot: int, seed: int) -> list[TaskItem]:
        rows = _fetch_ok_rows()

        shot_block = ""
        if n_shot > 0:
            dev_rows = fetch_parquet_rows(FEWSHOT_REPO, FEWSHOT_DEV_FILE)
            shots = dev_rows[:n_shot]
            blocks = [
                _format_question(r["question"], list(r["choices"]))
                + f"\nAnswer: {LETTERS[r['answer']]}"
                for r in shots
            ]
            shot_block = "\n\n".join(blocks) + "\n\n"

        items: list[TaskItem] = []
        for idx in sample_indices(len(rows), n_items, seed):
            row = rows[idx]
            subject = str(row["subject"]).replace("_", " ")
            instruction = (
                f"The following are multiple choice questions (with answers) "
                f"about {subject}. Respond with only the letter of the correct answer."
            )
            body = _format_question(row["question"], list(row["choices"]))
            items.append(
                TaskItem(
                    item_id=f"mmlu_redux:{idx}",
                    prompt=f"{instruction}\n\n{shot_block}{body}\nAnswer:",
                    target=LETTERS[row["answer"]],
                    metadata={"subject": row["subject"], "error_type": row["error_type"]},
                )
            )
        return items

    def score(self, completion: str, item: TaskItem) -> bool:
        # Shared extractor (tasks/base.py): explicit "Answer: X" marker first,
        # else the LAST standalone A-D. Last, not first, because the cap is now
        # large enough for a model to reason before answering.
        return extract_letter_answer(completion) == item.target
