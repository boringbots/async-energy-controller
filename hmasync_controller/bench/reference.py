"""`bench reference` -- reproduce energy-bench's Efficiency Index anchor.

Every other mode in this package measures something about YOUR box. This one
measures the one configuration the lab normalizes everything else against, so
the number it produces can be put next to a lab row and mean something.

## The anchor

Mirrors `energy_bench.grading.reference.REFERENCE_CONFIG` field for field --
TEST-MATRIX.md section 1, the exact tuple rather than a paraphrase:

    Qwen3.5-9B, Q4_K_M GGUF, llama.cpp llama-server, batch 1, stock power,
    gsm8k_platinum, 100 items, seed 1234, 5-shot, thinking off

The lab's own `.114` node measures this at **83,348 J, 975 J/correct,
accuracy 0.81**, reproducing to 0.095% over six cold boots. That
reproducibility is the point: it is what lets absolute joules -- which never
travel between machines -- become comparable after normalization.

## Why this cannot be `bench quick` with different arguments

`bench quick` differs from the anchor in three ways at once, and every one of
them changes the number:

    quick                        anchor
    ------------------------     ------------------------
    25 items                     100 items
    Ollama (or llama.cpp)        llama.cpp specifically
    Ollama's own Q4_K_M build    lmstudio-community's GGUF

The third is the subtle one. A quantization NAME is not a set of weights:
`reference.py` pins `gguf_repo` + `gguf_revision` precisely because two
community Q4_K_M builds of one base model measure differently, and a row
recording only "Q4_K_M" normalizes silently against whichever build it
happened to serve. So this mode verifies the weights before it measures, and
refuses rather than produce a confidently-wrong anchor.

## What is deliberately NOT pinned here

`REFERENCE_CONFIG` pins no output cap. `docs/reference-wave.md` names that as
a real gap in the corpus -- 400/512/1024 caps were "never chosen and never
written down" -- and argues a reference point should run uncapped to EOS. That
change is the lab's to make, not this package's, so this mode uses the task's
own default cap and RECORDS it on the row (`max_tokens`), leaving a consumer
free to group by it. With thinking off the measured cost is ~241 tok/item, so
the cap is not binding in practice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hmasync_controller.bench.quick import (
    QUICK_REFERENCE_MODELS,
    QUICK_REFERENCE_MODEL_HF_ID,
    QUICK_REFERENCE_N_SHOT,
    QUICK_REFERENCE_SEED,
    ModelNotAvailableError,
    NoEngineDetectedError,
    QuickError,
    QuickSuiteResult,
    _run_bench_suite,
    detect_engine,
)
from hmasync_controller.bench.engines import DEFAULT_LLAMACPP_PORT, DEFAULT_OLLAMA_PORT
from hmasync_controller.bench.vllm_client import VLLMClient

logger = logging.getLogger(__name__)

__all__ = [
    "REFERENCE_ANCHOR",
    "REFERENCE_TASK",
    "REFERENCE_N_ITEMS",
    "ReferenceWeightsMismatchError",
    "run_reference_suite",
]

REFERENCE_TASK = "gsm8k_platinum"
REFERENCE_N_ITEMS = 100
"""100, not `bench quick`'s 25. The anchor's item count is part of the tuple
every submission normalizes against -- 25 items of the same task is a
different measurement, not a cheaper one."""


@dataclass(frozen=True)
class ReferenceAnchor:
    """The published figures this run is trying to reproduce."""

    joules_total: float = 83_348.0
    joules_per_correct: float = 975.0
    accuracy: float = 0.81
    reproducibility_pct: float = 0.095
    node: str = ".114 (RTX 3090)"


REFERENCE_ANCHOR = ReferenceAnchor()


class ReferenceWeightsMismatchError(QuickError):
    """llama-server is serving something other than the reference GGUF.

    Raised rather than measured: an anchor computed against the wrong weights
    is worse than no anchor, because it looks like a comparable number.
    """


def _served_model_matches(served: str) -> bool:
    """Whether llama-server's reported model id is the reference GGUF.

    Deliberately lenient about SHAPE and strict about IDENTITY: llama-server
    reports whatever alias it was started with -- a full path, a bare
    filename, or `-a` text -- so this matches on the GGUF's distinctive stem
    rather than demanding an exact string. `Qwen3.5-9B-Q4_K_M` appearing
    anywhere in the id is taken as the reference; a different quant or a
    different model does not contain it.
    """
    stem = QUICK_REFERENCE_MODELS["llama.cpp"]["gguf_file"].removesuffix(".gguf")
    return stem.lower() in served.lower()


def _how_to_serve_it() -> str:
    spec = QUICK_REFERENCE_MODELS["llama.cpp"]
    return (
        f"Serve the reference weights and re-run:\n"
        f"  huggingface-cli download {spec['gguf_repo']} {spec['gguf_file']} \\\n"
        f"    --revision {spec['revision']} --local-dir ./ref\n"
        f"  llama-server -m ./ref/{spec['gguf_file']} -c 4096 --port "
        f"{DEFAULT_LLAMACPP_PORT}\n"
        f"These exact weights are the anchor: a different community Q4_K_M "
        f"build of the same base model measures differently, which is why "
        f"the revision is pinned."
    )


async def run_reference_suite(
    *,
    host: str = "localhost",
    ollama_port: int = DEFAULT_OLLAMA_PORT,
    llamacpp_port: int = DEFAULT_LLAMACPP_PORT,
    target_host: str | None = None,
    restore_to_factory_default: bool = True,
    budget_s: float | None = None,
) -> QuickSuiteResult:
    """Measure the Efficiency Index anchor on this box.

    Verifies the engine and the weights before measuring anything, because
    every one of the anchor's fields is load-bearing.

    Raises:
        NoEngineDetectedError: No llama-server answered.
        ModelNotAvailableError: llama-server is up but serving nothing.
        ReferenceWeightsMismatchError: It is serving the wrong weights.
    """
    detected = await detect_engine("llamacpp", host, ollama_port, llamacpp_port)
    if detected is None:
        raise NoEngineDetectedError(
            f"bench reference needs llama.cpp -- the anchor is "
            f"'llama.cpp llama-server, batch 1', and an Ollama run of the same "
            f"model is a different engine serving different weights, so it "
            f"cannot reproduce the anchor. No llama-server answered on "
            f"{host}:{llamacpp_port}.\n\n" + _how_to_serve_it()
        )

    client = VLLMClient(host=host, port=llamacpp_port)
    client.base_url = detected.base_url
    served_models = await client.get_models()
    if not served_models:
        raise ModelNotAvailableError(
            "llama-server is healthy but GET /v1/models reports no loaded "
            "model.\n\n" + _how_to_serve_it()
        )
    served = served_models[0]
    if not _served_model_matches(served):
        raise ReferenceWeightsMismatchError(
            f"llama-server is serving '{served}', which is not the reference "
            f"GGUF. An Efficiency Index computed against other weights is not "
            f"comparable to anything, so this run is refused rather than "
            f"published.\n\n" + _how_to_serve_it()
        )

    logger.info("bench reference: reproducing energy-bench's Efficiency Index anchor")
    logger.info("  weights: %s", served)
    logger.info(
        "  anchor:  %s measured %.0f J, %.0f J/correct, accuracy %.2f (±%.3f%% over 6 boots)",
        REFERENCE_ANCHOR.node,
        REFERENCE_ANCHOR.joules_total,
        REFERENCE_ANCHOR.joules_per_correct,
        REFERENCE_ANCHOR.accuracy,
        REFERENCE_ANCHOR.reproducibility_pct,
    )
    logger.info(
        "  config:  %s x%d, seed %d, %d-shot, thinking off, stock power",
        REFERENCE_TASK,
        REFERENCE_N_ITEMS,
        QUICK_REFERENCE_SEED,
        QUICK_REFERENCE_N_SHOT,
    )

    return await _run_bench_suite(
        engine_choice="llamacpp",
        host=host,
        ollama_port=ollama_port,
        llamacpp_port=llamacpp_port,
        target_host=target_host,
        tasks=[(REFERENCE_TASK, REFERENCE_N_ITEMS)],
        # The anchor is stock power. The ladders are a separate, justified
        # axis (reference-wave.md §2) and would change the thing being
        # normalized against.
        max_sweep_points=0,
        label_prefix_stem="reference",
        restore_to_factory_default=restore_to_factory_default,
        budget_s=budget_s,
        # Pinned by the anchor: REFERENCE_CONFIG carries
        # thinking_mode="enable_thinking=false" precisely because Qwen3.5-9B
        # is reasoning-capable and thinking is the largest energy lever it
        # has, so leaving it unset left the anchor undefined where it mattered
        # most.
        thinking=False,
    )


def format_reference_comparison(result: QuickSuiteResult) -> str:
    """One line putting this box's anchor run beside the published one.

    Joules deliberately are NOT presented as a ratio against the lab's node:
    absolute joules do not travel between machines, which is the entire
    reason the Efficiency Index normalizes rather than compares. Accuracy
    does travel -- a correct answer is correct anywhere -- so that is the
    number worth calling out when it drifts.
    """
    if not result.runs:
        return "bench reference: nothing was measured."
    run = result.runs[0]
    lines = [
        f"  this box: {run.total_joules_gpu_best or float('nan'):.0f} J, "
        f"{run.joules_per_correct_answer or float('nan'):.0f} J/correct, "
        f"accuracy {run.accuracy:.2f}  [{run.energy_source}]",
        f"  {REFERENCE_ANCHOR.node}: {REFERENCE_ANCHOR.joules_total:.0f} J, "
        f"{REFERENCE_ANCHOR.joules_per_correct:.0f} J/correct, "
        f"accuracy {REFERENCE_ANCHOR.accuracy:.2f}  [counter]",
    ]
    if abs(run.accuracy - REFERENCE_ANCHOR.accuracy) > 0.10:
        lines.append(
            f"  NOTE: accuracy is {abs(run.accuracy - REFERENCE_ANCHOR.accuracy):.2f} "
            f"off the anchor. Accuracy is hardware-independent, so a gap this "
            f"size means the configuration differs, not the machine."
        )
    if run.energy_source != "counter":
        lines.append(
            f"  NOTE: energy_source is '{run.energy_source}', not NVML's "
            f"'counter'. Two vendors' models of their own silicon -- compare "
            f"the accuracy, not the joules."
        )
    return "\n".join(lines)
