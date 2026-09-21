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

The most-pulled models that sit on energy-bench's measured accuracy-vs-energy
frontier (fleet snapshots read 2026-09-20, stock power, best config per model):

    gsm8k_platinum   Qwen2.5-7B Q4_K_M 0.90 @ 304 J/correct; Qwen3-Coder-30B-A3B
                     0.97 @ 355; gpt-oss-20b 0.99 @ 472
    mmlu_redux       Qwen2.5-7B Q4_K_M 0.71 @ 17; Qwen3-Coder-30B-A3B 0.79 @ 29;
                     gemma-4-26B-A4B 0.84 @ 53; Qwen3.5-9B 0.86 @ 67
    gpqa_diamond     Qwen3-Coder-30B-A3B 0.60 @ 47 (nothing else is close)
    math500          Qwen3-Coder-30B-A3B 0.76 @ 1619; gpt-oss-20b 0.86 @ 2295

So the head of the roster is the three MoE models that own the frontier, then
the dense 7-9B class every box can hold, then the small end for 8 GB cards.
Llama-3.1-8B is here for popularity alone -- the lab has never measured it --
and Mistral-7B-v0.2 was dropped: 0.45 on gsm8k at ~1000 J/correct puts it far
off the frontier. LFM2.5-1.2B is on the frontier but has no Ollama tag.

`hf_id` join caveat: the lab's rows for the two AWQ-served MoEs are keyed by
the quantizer's repo (`stelterlab/...-AWQ`, `cyankiwi/...-AWQ-4bit`), and its
Ollama engine-wave rows by the bare tag (`gpt-oss:20b`). The FP16 ids below
are the honest base-model identity; joining to those lab rows needs an alias
on the lab side, not a lie here.

gpt-oss-20b ships MXFP4 natively (no Q4_K_M exists); its thinking knob is
`reasoning_effort`, and the thinking-off kwargs send `reasoning_effort: none`,
which has been verified on Qwen3.5 under Ollama but NOT yet on gpt-oss.

Verified live 2026-09-20 against registry.ollama.ai: every tag resolves and
every digest below is that tag's current model layer.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "REFERENCE_ENTRY",
    "detect_model_budget_gb",
    "REFERENCE_TAG",
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
        tag="qwen3-coder:30b-a3b-q4_K_M",
        hf_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        quantization="Q4_K_M",
        digest="sha256:1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a",
        size_gb=18.6,
    ),
    RosterEntry(
        tag="gemma4:26b-a4b-it-q4_K_M",
        hf_id="google/gemma-4-26B-A4B-it",
        quantization="Q4_K_M",
        digest="sha256:7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df",
        size_gb=18.0,
    ),
    RosterEntry(
        tag="gpt-oss:20b",
        hf_id="openai/gpt-oss-20b",
        quantization="mxfp4",
        digest="sha256:e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb",
        size_gb=13.8,
    ),
    RosterEntry(
        tag="qwen3.5:9b-q4_K_M",
        hf_id="Qwen/Qwen3.5-9B",
        quantization="Q4_K_M",
        digest="sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c",
        size_gb=6.6,
    ),
    RosterEntry(
        tag="qwen3:8b-q4_K_M",
        hf_id="Qwen/Qwen3-8B",
        quantization="Q4_K_M",
        digest="sha256:a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f",
        size_gb=5.2,
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
REFERENCE_TAG = "qwen3.5:9b-q4_K_M"
"""The one model `bench quick` measures -- the same tag quick.py pins. It sits
inside the roster (so the tiers are a superset of quick) but is no longer its
head: the roster is ordered largest-first and the three frontier MoEs above it
do not fit the card `bench quick` is sized for."""

REFERENCE_ENTRY: RosterEntry = next(e for e in ROSTER if e.tag == REFERENCE_TAG)


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


def detect_model_budget_gb() -> float | None:
    """How large a model this box can actually serve, in GB. None if unknown.

    Exists because the roster is ordered largest-first and its head is now
    three MoEs at 13.8-18.6 GB. Without a budget a small box is handed that
    head, told to `ollama pull` ~37 GB of weights it can never serve, and
    measures only whatever few of the tier happen to fit -- a partial tier
    AND bad advice.

    Two sources, because the honest number differs by platform:

    - **NVIDIA**: NVML's total VRAM, via the sampler's own `gpu_info()`, so
      this package keeps exactly one NVML path.
    - **Apple Silicon**: there is no VRAM -- memory is unified, which is why
      `gpu_mem_used_mib` is None on every Mac row. Metal's recommended
      working set bounds a model and tracks ~75% of system RAM (Ollama
      independently reports 17.8 GiB on this 24 GB M3, which is 74%). The
      budget is 65%, not 75%, because the working set has to hold the KV
      cache too: 75% of a 24 GB box is 19.3 GB, which would admit an 18.6 GB
      model with 0.7 GB left for everything else and send the machine to
      swap -- and a swapping run still reports a valid-looking number, just
      a much worse one. 65% gives 16.7 GB here, which admits the 13.8 GB MoE
      that measurably serves fine on this box and excludes the 18 GB pair
      that cannot.

    Returns None rather than guessing on anything else; an unknown budget
    selects the plain tier prefix, the behaviour before this existed.
    """
    import platform
    import subprocess

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return round(int(out.stdout.strip()) / 1e9 * 0.65, 1)
        except (subprocess.SubprocessError, OSError, ValueError):
            return None

    try:
        import asyncio

        from hmasync_controller.bench.sampler import select_gpu_sampler

        info = asyncio.run(select_gpu_sampler().gpu_info())
        total_mib = info.get("gpu_mem_total_mib")
        return round(float(total_mib) / 1024, 1) if total_mib else None
    except Exception:  # noqa: BLE001 - a budget is an optimization, never a gate
        return None


def roster_for_tier(tier: str, *, budget_gb: float | None = None) -> tuple[RosterEntry, ...]:
    """The roster a tier measures: the N largest models that fit `budget_gb`
    (every model when no budget is given), largest first. `quick` is always
    exactly the reference entry.

    Budget first, count second: the roster's head is three 14-19 GB MoEs, so
    taking a prefix and THEN dropping what does not fit would hand a 12 GB
    box an empty medium tier. The point of the tiers is "which model runs
    best on MY box", which needs the biggest models this box can hold.

    Raises:
        KeyError: Unknown tier -- callers pass a validated CLI choice, so a
            miss here is a programming error rather than operator input.
    """
    if tier == "quick":
        return (REFERENCE_ENTRY,)
    fitting = tuple(e for e in ROSTER if budget_gb is None or e.size_gb <= budget_gb)
    return fitting[: TIER_MODEL_COUNTS[tier]]


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

    `budget_gb` selects the tier's models BEFORE the pulled/missing split (see
    `roster_for_tier`), so a small box is told "too big for this machine"
    rather than "go and pull 18 GB you cannot run".
    """
    wanted = roster_for_tier(tier, budget_gb=budget_gb)
    present = [e for e in wanted if e.tag in available_tags]
    missing = [e for e in wanted if e.tag not in available_tags]
    return present, missing
