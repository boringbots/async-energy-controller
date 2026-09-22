"""The pinned roster and the multi-model tiers built on it."""

from __future__ import annotations

import pytest

from hmasync_controller import cli
from hmasync_controller.bench.roster import (
    REFERENCE_ENTRY,
    ROSTER,
    REFERENCE_TAG,
    TIER_MODEL_COUNTS,
    roster_for_tier,
    select_roster,
)
from hmasync_controller.bench.quick import QUICK_REFERENCE_MODELS


class TestRosterPins:
    def test_every_entry_pins_weights_not_just_a_quant_name(self):
        """A quantization NAME is not a set of weights. Without the digest a
        row says Q4_K_M and normalizes against whoever's build it happened
        to pull -- the exact silent-normalization failure energy-bench added
        gguf_repo/gguf_revision to prevent."""
        for e in ROSTER:
            # Bare hex, no "sha256:" prefix -- the exact shape `GET /api/tags`
            # reports, because that is what model_digest compares against.
            assert not e.digest.startswith("sha256:"), e.tag
            assert len(e.digest) == 64, e.tag
            assert all(c in "0123456789abcdef" for c in e.digest), e.tag
            assert e.hf_id.count("/") == 1, e.tag
            assert e.quantization
            assert e.size_gb > 0

    def test_tags_are_fully_qualified(self):
        """`qwen3.5:9b` and `qwen3.5:9b-q4_K_M` are different weights."""
        for e in ROSTER:
            assert ":" in e.tag, e.tag
            assert e.tag.split(":")[1] != "latest", e.tag

    def test_quick_reference_is_in_the_roster(self):
        """The tiers must be a superset of quick, not a second list that can
        drift from it. The reference is no longer the roster's HEAD: the
        three frontier MoEs above it do not fit the card quick is sized for."""
        assert REFERENCE_TAG == QUICK_REFERENCE_MODELS["ollama"]["tag"]
        assert REFERENCE_TAG in {e.tag for e in ROSTER}
        assert roster_for_tier("quick") == (REFERENCE_ENTRY,)
        assert REFERENCE_ENTRY.hf_id == "Qwen/Qwen3.5-9B"

    def test_roster_head_is_the_measured_frontier(self):
        """energy-bench's fleet frontier (2026-09-20): the three MoEs own it
        on gsm8k/math500/gpqa, Qwen2.5-7B Q4_K_M owns the cheap end."""
        tags = [e.tag for e in ROSTER]
        assert tags[:3] == ["qwen3-coder:30b-a3b-q4_K_M", "gemma4:26b-a4b-it-q4_K_M", "gpt-oss:20b"]
        assert "qwen2.5:7b-instruct-q4_K_M" in tags
        assert not any(t.startswith("mistral:") for t in tags)  # 0.45 on gsm8k, off the frontier

    def test_roster_is_ordered_largest_first(self):
        sizes = [e.size_gb for e in ROSTER]
        assert sizes == sorted(sizes, reverse=True)

    def test_no_duplicate_tags_or_digests(self):
        assert len({e.tag for e in ROSTER}) == len(ROSTER)
        assert len({e.digest for e in ROSTER}) == len(ROSTER)


class TestTierSelection:
    def test_tiers_take_a_prefix_largest_first(self):
        assert len(roster_for_tier("quick")) == 1
        assert len(roster_for_tier("medium")) == TIER_MODEL_COUNTS["medium"]
        assert roster_for_tier("medium")[0] == ROSTER[0]

    def test_budget_is_applied_before_the_count(self):
        """A 12 GB box still measures four models -- the four largest that
        fit -- rather than an empty tier because the head is three MoEs."""
        picked = roster_for_tier("medium", budget_gb=12.0)
        assert len(picked) == TIER_MODEL_COUNTS["medium"]
        assert all(e.size_gb <= 12.0 for e in picked)
        assert picked[0].tag == "qwen3.5:9b-q4_K_M"

    def test_full_measures_more_models_than_medium(self):
        """The tiers swapped jobs: medium took the thinking axis (it and the
        old full measured the same 4 models on the same 3 tasks), and full
        became breadth -- every model the box can serve."""
        assert len(roster_for_tier("full")) > len(roster_for_tier("medium"))
        # medium's roster stays a prefix of full's, so the two nest.
        assert roster_for_tier("medium") == roster_for_tier("full")[
            : len(roster_for_tier("medium"))
        ]

    def test_splits_into_measurable_and_needs_pulling(self):
        present, missing = select_roster(
            "medium", available_tags={ROSTER[0].tag, ROSTER[2].tag}
        )
        assert [e.tag for e in present] == [ROSTER[0].tag, ROSTER[2].tag]
        assert ROSTER[1] in missing

    def test_a_budget_drops_models_the_box_cannot_hold(self):
        """An 8 GB box is told 'too big for this machine', not 'go pull 6.6 GB
        you cannot run'."""
        present, missing = select_roster(
            "medium", available_tags={e.tag for e in ROSTER}, budget_gb=4.8
        )
        assert all(e.size_gb <= 4.8 for e in present + missing)
        assert ROSTER[0] not in present + missing
        # every model that fits is measured -- three here, not zero
        assert present == [e for e in ROSTER if e.size_gb <= 4.8]

    def test_nothing_pulled_yields_an_empty_present_list(self):
        present, missing = select_roster("medium", available_tags=set())
        assert present == []
        assert len(missing) == TIER_MODEL_COUNTS["medium"]


class TestTierCli:
    @pytest.mark.parametrize("tier", ["medium", "full"])
    def test_tier_subcommands_parse(self, tier):
        args = cli._parse_args(["bench", tier])
        assert args.bench_subcommand == tier
        assert args.timeout is None
        # No --thinking on the multi-model tiers: they pin it off.
        assert not hasattr(args, "thinking")

    def test_tier_backstops_are_sized_for_the_work(self):
        """With the thinking sweep gone, cost tracks breadth again and the
        backstops order the way the names imply."""
        assert cli.BENCH_FULL_TIMEOUT_S > cli.BENCH_MEDIUM_TIMEOUT_S
        assert cli.BENCH_MEDIUM_TIMEOUT_S > cli.BENCH_QUICK_TIMEOUT_S

    def test_the_schema_accepts_the_new_suites(self):
        import json
        from pathlib import Path

        schema = json.loads(
            (
                Path(cli.__file__).parent / "schemas" / "bench_submission.schema.json"
            ).read_text()
        )
        assert {"medium", "full"} <= set(schema["properties"]["suite"]["enum"])


class TestBudgetDetection:
    """The roster is ordered largest-first and its head is three MoEs at
    13.8-18.6 GB. Without a budget a small box is handed that head, told to
    pull ~37 GB it can never serve, and measures only the remainder."""

    def test_apple_silicon_budgets_below_the_metal_working_set(self, monkeypatch):
        import subprocess

        from hmasync_controller.bench import roster as r

        monkeypatch.setattr(r.platform if hasattr(r, "platform") else __import__("platform"),
                            "system", lambda: "Darwin", raising=False)

        class _Out:
            stdout = str(24 * 10**9)

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Out())
        with monkeypatch.context() as m:
            m.setattr("platform.system", lambda: "Darwin")
            m.setattr("platform.machine", lambda: "arm64")
            budget = r.detect_model_budget_gb()

        # Must leave room for the KV cache: Ollama reports a 17.8 GiB working
        # set on a 24 GB box, and the weights are not the only thing in it.
        assert budget is not None
        assert budget < 17.8, "a budget at or above the working set admits models that swap"
        assert budget > 13.8, "must still admit the MoE that measurably serves here"

    def test_an_unknown_budget_is_none_not_a_guess(self, monkeypatch):
        from hmasync_controller.bench import roster as r

        with monkeypatch.context() as m:
            m.setattr("platform.system", lambda: "Linux")
            m.setattr("platform.machine", lambda: "x86_64")
            m.setattr(
                "hmasync_controller.bench.sampler.select_gpu_sampler",
                lambda: (_ for _ in ()).throw(RuntimeError("no nvml")),
            )
            assert r.detect_model_budget_gb() is None

    def test_a_budget_backfills_rather_than_shrinking_the_tier(self):
        """Dropping two oversized models must not leave a 2-model medium --
        the tier count is the promise, the specific models are not."""
        all_tags = {e.tag for e in ROSTER}
        present, missing = select_roster("medium", available_tags=all_tags, budget_gb=16.8)
        assert len(present) == TIER_MODEL_COUNTS["medium"]
        assert all(e.size_gb <= 16.8 for e in present)


class TestTierShapeAfterTheSwap:
    """`medium` absorbed the old `full` (same 4 models, same 3 tasks, plus the
    thinking sweep). `full` became the breadth tier."""

    def test_full_measures_every_model_that_fits_not_a_prefix(self):
        from hmasync_controller.bench.roster import TIER_MODEL_COUNTS

        assert TIER_MODEL_COUNTS["full"] is None, "None = no prefix cap"
        budget = 16.8  # a 24 GB Apple Silicon box
        fits = [e for e in ROSTER if e.size_gb <= budget]
        assert len(roster_for_tier("full", budget_gb=budget)) == len(fits)
        # ... and that is strictly more than medium's prefix.
        assert len(roster_for_tier("full", budget_gb=budget)) > len(
            roster_for_tier("medium", budget_gb=budget)
        )

    def test_full_reaches_the_frontier_model_the_prefix_missed(self):
        """Qwen2.5-7B is on the lab's accuracy-vs-energy frontier (0.90 gsm8k
        at 304 J/correct) and sits 7th by size, so a 4-model prefix on a 24 GB
        box never saw it."""
        tags = {e.tag for e in roster_for_tier("full", budget_gb=16.8)}
        assert "qwen2.5:7b-instruct-q4_K_M" in tags
        assert "qwen2.5:7b-instruct-q4_K_M" not in {
            e.tag for e in roster_for_tier("medium", budget_gb=16.8)
        }

    def test_full_still_respects_the_budget(self):
        for e in roster_for_tier("full", budget_gb=16.8):
            assert e.size_gb <= 16.8

    def test_full_runs_the_lab_wave_tasks_at_lab_counts(self):
        from hmasync_controller.bench.quick import FULL_TASKS, QUICK_TASKS, tasks_for_tier

        assert tasks_for_tier("full") == FULL_TASKS
        assert dict(FULL_TASKS) == {
            "gsm8k_platinum": 100,
            "mmlu_redux": 100,
            "math500": 50,
        }
        # quick/medium keep the scaled-down slice
        assert tasks_for_tier("medium") == QUICK_TASKS
        assert tasks_for_tier("quick") == QUICK_TASKS

    def test_gpqa_is_excluded_from_full_on_purpose(self):
        """The wave's fourth task is a GATED HuggingFace dataset; including it
        would abort the breadth tier on any box without accepted terms."""
        from hmasync_controller.bench.quick import FULL_TASKS

        assert "gpqa_diamond" not in dict(FULL_TASKS)

    def test_ifeval_is_not_in_full(self):
        """It pairs with no lab row, which is what full exists to guarantee."""
        from hmasync_controller.bench.quick import FULL_TASKS

        assert "ifeval" not in dict(FULL_TASKS)


class TestNeitherTierSweepsThinking:
    """The axis is real but already measured: energy-bench has 48 paired
    configs across nine models (~9x energy, decisive on one task in four).
    Re-deriving it here would multiply every multi-model run by that factor."""

    def _axis_for(self, tier, monkeypatch, pin=None):
        seen = {}

        def fake(**kw):
            seen["axis"] = kw.get("thinking_axis")
            raise RuntimeError("stop here")

        monkeypatch.setattr(cli, "run_tier_suite", fake)
        monkeypatch.setattr(cli, "_merge_tier_results", lambda c: c)
        try:
            cli.run_bench_tier(cli.Settings(), tier, thinking=pin)
        except Exception:
            pass
        return seen["axis"]

    def test_both_tiers_pin_thinking_off(self, monkeypatch):
        assert self._axis_for("medium", monkeypatch) == (False,)
        assert self._axis_for("full", monkeypatch) == (False,)

    def test_no_pin_can_turn_a_tier_into_a_sweep(self, monkeypatch):
        for tier in ("medium", "full"):
            for pin in (None, True, False):
                assert self._axis_for(tier, monkeypatch, pin) == (False,), (tier, pin)

    def test_the_tiers_expose_no_thinking_flag(self):
        for tier in ("medium", "full"):
            with pytest.raises(SystemExit):
                cli._parse_args(["bench", tier, "--thinking"])

    def test_one_model_at_a_time_can_still_reach_the_axis(self):
        """`bench quick --thinking` is the affordable way to look."""
        assert cli._parse_args(["bench", "quick", "--thinking"]).thinking is True

    def test_medium_backstop_is_sized_for_a_single_pass(self):
        assert cli.BENCH_MEDIUM_TIMEOUT_S == 4 * 60 * 60.0


class TestDigestsArePinnedInTheRuntimesOwnVocabulary:
    """A pin that cannot match is not a pin.

    `ollama pull` prints the model LAYER digest and `ollama list` prints the
    MANIFEST digest; they are different numbers for one tag. The roster once
    pinned the layer while `model_digest` reads the manifest, so every model
    logged "resolved to X, roster pins Y -- recording what ran" and the check
    validated nothing while looking like it worked."""

    def test_digests_match_the_shape_model_digest_returns(self):
        # OllamaAdapter.model_digest returns entry["digest"] from /api/tags
        # verbatim, and Ollama reports that bare.
        for e in ROSTER:
            assert ":" not in e.digest, e.tag

    def test_the_known_manifest_digests_are_recorded(self):
        """Verified against registry.ollama.ai: sha256(manifest bytes) equals
        what a local daemon reports for the same tag."""
        expected = {
            "qwen3.5:9b-q4_K_M": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
            "gpt-oss:20b": "17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7",
        }
        by_tag = {e.tag: e.digest for e in ROSTER}
        for tag, digest in expected.items():
            assert by_tag[tag] == digest, tag
