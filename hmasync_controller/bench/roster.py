"""The pinned model roster the multi-model tiers measure.

`bench quick` answers "what does this hardware cost on a known workload" and
so pins exactly one model. The `medium` and `full` tiers answer a different
question -- "which model runs best on MY box, and at what settings" -- which
needs several models measured identically.

## Why a curated roster rather than "whatever is installed"

A quantization NAME is not a set of weights. energy-bench learned this the
expensive way (`grading/reference.py`, `docs/MODEL-PINS.md`): two community
Q4_K_M builds of one base model measure differently, so a row recording
`quantization: Q4_K_M` without saying WHOSE weights normalizes silently
against the wrong thing. Measuring whatever a box happens to have would make
every cross-submitter comparison exactly that mistake, at scale.

So each entry below pins the Ollama tag AND the manifest digest of the model
layer -- the digest IS the weights identity for an Ollama-served model, the
same role `gguf_repo` + `gguf_revision` play for llama.cpp. A pull that
resolves to a different digest is different weights and is recorded as such
rather than pooled.

`hf_id` is the FP16 HuggingFace repo id, kept as the join key even though the
model is served quantized -- the convention `RunMetrics.model` already uses
(see quick.py's `QUICK_REFERENCE_MODEL_HF_ID`), and what lets a row here join
to energy-bench's own rows for the same base model.

## Why these models

Every entry maps to a model energy-bench already measures on its fleet
(`docs/MODEL-PINS.md`), so a community row has something to be compared
against. The sizes span 1.9-6.6 GB deliberately: `select_roster` walks them
largest-first and skips what will not fit, so an 8 GB box still measures the
small end instead of failing outright.

Verified live 2026-09-20 against registry.ollama.ai: every tag resolves and
every digest below is that tag's current model layer.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ROSTER",
    "RosterEntry",
    "roster_for_tier",
    "select_roster",
]


@dataclass(frozen=True)
class RosterEntry:
    """One pinned model: how to serve it, and how to record what was served."""

    tag: str
    """The exact `ollama pull` tag. Never resolved loosely -- `qwen3.5:9b` and
    `qwen3.5:9b-q4_K_M` are different weights."""

    hf_id: str
    """FP16 HuggingFace repo id, the cross-repo join key for `RunMetrics.model`."""

    quantization: str
    """Recorded verbatim on the row. Meaningless without `digest`."""

    digest: str
    """Manifest digest of the model layer -- the weights identity. Two pulls
    of one tag that differ here are different weights."""

    size_gb: float
    """Model layer size. Used only to skip models a box cannot hold; the real
    footprint adds a KV cache that depends on the serving context."""

    @property
    def family(self) -> str:
        """Base model family, for grouping a size ladder in a report."""
        return self.hf_id.split("/")[-1].rsplit("-", 1)[0]


# Largest first: `select_roster` takes a prefix of what fits, and the biggest
# model a box can hold is the most interesting row it can contribute.
ROSTER: tuple[RosterEntry, ...] = (
    RosterEntry(
        tag="qwen3.5:9b-q4_K_M",
        hf_id="Qwen/Qwen3.5-9B",
        quantization="Q4_K_M",
        digest="sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c",
        size_gb=6.6,
    ),
    RosterEntry(
        tag="llama3.1:8b-instruct-q4_K_M",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        quantization="Q4_K_M",
        digest="sha256:667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29",
        size_gb=4.9,
    ),
    RosterEntry(
        tag="qwen2.5:7b-instruct-q4_K_M",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        quantization="Q4_K_M",
        digest="sha256:2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
        size_gb=4.7,
    ),
    RosterEntry(
        tag="mistral:7b-instruct-q4_K_M",
        hf_id="mistralai/Mistral-7B-Instruct-v0.2",
        quantization="Q4_K_M",
        digest="sha256:faf975975644f275f00075e7cf79cd207642412560640cd1930afbab95fea25c",
        size_gb=4.4,
    ),
    RosterEntry(
        tag="qwen3.5:4b-q4_K_M",
        hf_id="Qwen/Qwen3.5-4B",
        quantization="Q4_K_M",
        digest="sha256:81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490",
        size_gb=3.4,
    ),
    RosterEntry(
        tag="qwen3.5:2b-q4_K_M",
        hf_id="Qwen/Qwen3.5-2B",
        quantization="Q4_K_M",
        digest="sha256:7a3a8d55382135a773916fd7c35044b2a2a3a7b8dee788095d70f122e6d8f520",
        size_gb=1.9,
    ),
)

REFERENCE_TAG = ROSTER[0].tag
"""The one model `bench quick` measures -- the same tag quick.py pins, kept
as the roster's head so the tiers form a superset rather than a second list."""

TIER_MODEL_COUNTS: dict[str, int] = {
    "quick": 1,
    "medium": 4,
    "full": 4,
}
"""How many roster models each tier measures, largest-first.

`full` measures the same four as `medium` and spends its extra time on the
thinking axis instead of more models: energy-bench's F3 found thinking worth
~9x the energy and decisive on one task in four, which is a bigger effect
than a fifth model of the same size class.
"""


def roster_for_tier(tier: str) -> tuple[RosterEntry, ...]:
    """The roster prefix a tier measures, largest model first.

    Raises:
        KeyError: Unknown tier -- callers pass a validated CLI choice, so a
            miss here is a programming error rather than operator input.
    """
    return ROSTER[: TIER_MODEL_COUNTS[tier]]


def select_roster(
    tier: str,
    *,
    available_tags: set[str],
    budget_gb: float | None = None,
) -> tuple[list[RosterEntry], list[RosterEntry]]:
    """Split a tier's roster into (measurable now, not pulled).

    Never pulls. This package's standing rule is that model lifecycle belongs
    to the operator -- the caller names the exact `ollama pull` for each miss
    rather than fetching several GB on someone's behalf.

    `budget_gb` drops entries too large to serve before the pulled/missing
    split, so a small box is told "too big for this machine" rather than
    "go and pull 6.6 GB you cannot run".
    """
    wanted = roster_for_tier(tier)
    if budget_gb is not None:
        wanted = tuple(e for e in wanted if e.size_gb <= budget_gb)
    present = [e for e in wanted if e.tag in available_tags]
    missing = [e for e in wanted if e.tag not in available_tags]
    return present, missing
