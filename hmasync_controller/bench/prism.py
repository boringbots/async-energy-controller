"""`bench prism` -- what sub-4-bit weights cost in joules.

PrismML's Bonsai checkpoints are post-training quantizations of Qwen3-family
models down to ternary weights ({-1, 0, +1} with one FP16 scale per group,
~2.125 bits) or to one bit. They are not new pretrains, and every checkpoint
is Apache-2.0 over an Apache-2.0 base.

energy-bench's F10 established that quantization is an energy lever and not an
accuracy one -- but it stops at four bits, and it cannot separate "fewer bytes
per token" from "different weights", because an Int4 rung has both. The
published literature stops earlier still: nobody in the six reference studies
meters a sub-4-bit model on a consumer card, and PrismML's own energy claims
are phone-only.

## Why this is its own tier

Everything else here is Ollama. These weights are not on Ollama at all, and
the formats need a fork of llama.cpp (`PrismML-Eng/llama.cpp`) whose ternary
kernels upstream does not have. So `bench prism` is llama.cpp-only and, like
`bench reference`, attaches to a server the operator started.

One model per server is a hard consequence of attach mode: llama-server holds
exactly one GGUF for its lifetime and this package never launches or swaps an
engine. So a rung is one invocation, and a wave is the operator restarting the
server between them. That is more honest than it sounds -- every rung gets a
cold process, which is what the lab's own runner does deliberately.

## The format control is the point

`Ternary-Bonsai-8B-gguf` publishes ONE checkpoint in three storages: dense
F16, group-64 and group-128. A weight of {-1, 0, +1} times an FP16 group scale
is exactly representable in FP16, so the three files should return the SAME
answers. Accuracy is then held constant by construction rather than by a
confidence interval, and whatever energy separates them is bytes-per-token and
kernel, nothing else -- F10's claim with its confound removed.

If the three rungs disagree on answers, the premise is wrong and the rest is
uninterpretable. That is a result worth having either way, so this tier does
not average it away.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

from hmasync_controller.bench.engines import DEFAULT_LLAMACPP_PORT, DEFAULT_OLLAMA_PORT
from hmasync_controller.bench.quick import (
    FULL_TASKS,
    ModelNotAvailableError,
    NoEngineDetectedError,
    QuickError,
    QuickSuiteResult,
    _run_bench_suite,
    detect_engine,
)
from hmasync_controller.bench.vllm_client import VLLMClient

logger = logging.getLogger(__name__)

__all__ = [
    "PRISM_GGUFS",
    "PRISM_TASKS",
    "THINKING_AXIS_KWARG",
    "THINKING_AXIS_TEMPLATE",
    "THINKING_AXIS_UNKNOWN",
    "PrismGGUF",
    "PrismWeightsUnknownError",
    "match_served_gguf",
    "run_prism_suite",
    "thinking_axis_note",
]

PRISM_TASKS = FULL_TASKS
"""The same tasks and counts `bench full` runs, so a Bonsai row and an Ollama
row differ in the weights being measured and nothing else. gpqa_diamond is the
prism wave's fourth task upstream and is left out for the same reason it is
left out of `full`: `Idavidrein/gpqa` is gated and fails at load without
accepted terms and an HF_TOKEN."""


@dataclass(frozen=True)
class PrismGGUF:
    """One pinned Bonsai rung: how to identify it, and what to record."""

    filename: str
    """The GGUF's exact name. Also how a served model is recognised -- see
    `match_served_gguf`, and the warning about bare `*-Q2_0.gguf` below."""

    repo: str
    revision: str
    """HuggingFace provenance, pinned. A quantization NAME is not a set of
    weights; these two fields are what make a row reproducible."""

    hf_id: str
    """The join key recorded as `RunMetrics.model` -- the FP16 base this was
    quantized FROM, so a Bonsai row sits beside that base's own rows.

    For the 8B format control this is deliberately NOT Qwen3-8B: the model
    card's `base_model` is `Ternary-Bonsai-8B-unpacked`, so all three storages
    are the ternary model, stored differently. Recording Qwen3-8B there would
    claim a comparison the files do not support."""

    quantization: str
    """Recorded verbatim. The whole axis of this tier."""

    size_gb: float
    stage: str
    """Which question this rung answers -- see the module docstring."""


PRISM_GGUFS: tuple[PrismGGUF, ...] = (
    # Stage 1 -- the format control. One checkpoint, three storages.
    PrismGGUF(
        filename="Ternary-Bonsai-8B-F16.gguf",
        repo="prism-ml/Ternary-Bonsai-8B-gguf",
        revision="c2aefbeb4b24469cd11579c3384b990404c17a30",
        hf_id="prism-ml/Ternary-Bonsai-8B-unpacked",
        quantization="F16",
        size_gb=16.38,
        stage="1-format-control-dense",
    ),
    PrismGGUF(
        filename="Ternary-Bonsai-8B-Q2_0_g64.gguf",
        repo="prism-ml/Ternary-Bonsai-8B-gguf",
        revision="c2aefbeb4b24469cd11579c3384b990404c17a30",
        hf_id="prism-ml/Ternary-Bonsai-8B-unpacked",
        quantization="Q2_0_g64",
        size_gb=2.31,
        stage="1-format-control-group64",
    ),
    PrismGGUF(
        filename="Ternary-Bonsai-8B-PQ2_0.gguf",
        repo="prism-ml/Ternary-Bonsai-8B-gguf",
        revision="c2aefbeb4b24469cd11579c3384b990404c17a30",
        hf_id="prism-ml/Ternary-Bonsai-8B-unpacked",
        quantization="PQ2_0",
        size_gb=2.18,
        stage="1-format-control-group128",
    ),
    # Stage 2 -- the accuracy spine, on the SAME engine, so the comparison
    # does not import F7 (engines diverge in token count at temperature 0).
    PrismGGUF(
        filename="Qwen3-8B-Q8_0.gguf",
        repo="Qwen/Qwen3-8B-GGUF",
        revision="7c41481f57cb95916b40956ab2f0b139b296d974",
        hf_id="Qwen/Qwen3-8B",
        quantization="Q8_0",
        size_gb=8.71,
        stage="2-spine",
    ),
    PrismGGUF(
        filename="Qwen3-8B-Q4_K_M.gguf",
        repo="Qwen/Qwen3-8B-GGUF",
        revision="7c41481f57cb95916b40956ab2f0b139b296d974",
        hf_id="Qwen/Qwen3-8B",
        quantization="Q4_K_M",
        size_gb=5.03,
        stage="2-spine",
    ),
    # Stage 3 -- a 27B-class model on hardware that cannot otherwise hold one.
    PrismGGUF(
        filename="Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        repo="prism-ml/Ternary-Bonsai-2-27B-gguf",
        revision="6ed5e12bf84b7a63069882c91dd9e9218647d17b",
        hf_id="Qwen/Qwen3.8-27B",
        quantization="PTQ1_0",
        size_gb=5.95,
        stage="3-27b-ternary",
    ),
    PrismGGUF(
        filename="Ternary-Bonsai-2-27B-PQ2_0.gguf",
        repo="prism-ml/Ternary-Bonsai-2-27B-gguf",
        revision="6ed5e12bf84b7a63069882c91dd9e9218647d17b",
        hf_id="Qwen/Qwen3.8-27B",
        quantization="PQ2_0",
        size_gb=7.21,
        stage="3-27b-2bit",
    ),
    PrismGGUF(
        filename="Bonsai-27B-Q1_0.gguf",
        repo="prism-ml/Bonsai-27B-gguf",
        revision="f10afb355f104535e3e3e98cf7ab7795c72bd292",
        hf_id="Qwen/Qwen3.6-27B",
        quantization="Q1_0",
        size_gb=3.80,
        stage="3-27b-1bit",
    ),
    # The small end -- where ternary either holds up or does not.
    PrismGGUF(
        filename="Ternary-Bonsai-4B-PQ2_0.gguf",
        repo="prism-ml/Ternary-Bonsai-4B-gguf",
        revision="a3eb42bafe873f9686bc97486c43b72ef7d75ec8",
        hf_id="Qwen/Qwen3-4B",
        quantization="PQ2_0",
        size_gb=1.07,
        stage="4-small",
    ),
    PrismGGUF(
        filename="Ternary-Bonsai-1.7B-PQ2_0.gguf",
        repo="prism-ml/Ternary-Bonsai-1.7B-gguf",
        revision="983b5dec2ff16aab79990711ba0f828a499a7e6a",
        hf_id="Qwen/Qwen3-1.7B",
        quantization="PQ2_0",
        size_gb=0.46,
        stage="4-small",
    ),
)


class PrismWeightsUnknownError(QuickError):
    """llama-server is serving something this tier cannot identify.

    Refused rather than measured: the entire point is what a SPECIFIC storage
    of a SPECIFIC checkpoint costs, and a row that cannot name its weights
    answers nothing. The message lists what is recognised.
    """


def match_served_gguf(served: str) -> PrismGGUF | None:
    """Identify a served model id as one of the pinned rungs, or None.

    Matches on the GGUF's filename appearing in the id, because llama-server
    reports whatever `-m` was given -- a bare name or a path.

    A symlink or an `-a` alias therefore defeats this by design: the server
    reports the alias, and the real filename never reaches the wire. That is
    the right failure. Resolving it on this side would mean trusting a path
    that may not even exist here -- attach mode can point at another host --
    and inferring weights from a name we cannot see is exactly the guess this
    tier exists to refuse.

    Longest filename first, which is load-bearing rather than tidy:
    `Ternary-Bonsai-8B-Q2_0_g64.gguf` contains no shorter rung's name, but the
    model card ALSO publishes a bare `Ternary-Bonsai-8B-Q2_0.gguf` that is a
    different file. Matching the longer name first stops a g64 file being
    recorded as the rung it is not.
    """
    if not served:
        return None
    lowered = served.lower()
    for entry in sorted(PRISM_GGUFS, key=lambda e: -len(e.filename)):
        if entry.filename.lower() in lowered:
            return entry
    return None


def _how_to_serve_it() -> str:
    lines = [
        "bench prism needs llama-server from the PrismML fork of llama.cpp,",
        "serving one of the pinned Bonsai rungs. Upstream llama.cpp cannot read",
        "these formats -- the ternary kernels only exist in the fork:",
        "",
        "  git clone https://github.com/PrismML-Eng/llama.cpp",
        "  cd llama.cpp && git checkout 5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6",
        "  cmake -B build -DGGML_METAL=ON   # or -DGGML_CUDA=ON on NVIDIA",
        "  cmake --build build --config Release --target llama-server",
        "",
        "Then fetch a rung and serve it:",
        "",
    ]
    for e in sorted(PRISM_GGUFS, key=lambda e: e.size_gb):
        lines.append(
            f"  {e.size_gb:5.2f} GB  {e.filename:34} ({e.quantization}, {e.stage})"
        )
    lines += [
        "",
        "  hf download <repo> <file> --revision <rev> --local-dir ./prism-models",
        "  llama-server -m ./prism-models/<file> -c 4096 --port 8080",
        "",
        "One rung per server: llama-server holds one GGUF for its lifetime and",
        "this package never launches or swaps an engine. Restart it between",
        "rungs -- each then gets a cold process, which is what the lab does too.",
    ]
    return "\n".join(lines)


THINKING_AXIS_KWARG = "kwarg"
THINKING_AXIS_TEMPLATE = "template"
THINKING_AXIS_UNKNOWN = "unknown"

_THINK_OFF_PATTERN = re.compile(r"<think>(?:\\n|\s)*</think>")
"""An empty <think> block in the template's generation prompt -- thinking off
by construction. The alternation is not defensive vagueness: a GGUF stores the
template SOURCE, so the 4B's block is the two characters backslash-n repeated
(verified by reading the file's metadata, 2026-09-22), not the newlines it
renders to. Matching only the rendered form would report every such template
as unpinned and cry wolf on the one family that cannot be unpinned."""


def thinking_axis_note(chat_template: str | None) -> tuple[str, str]:
    """How thinking is actually held off for the served template.

    Returns `(mechanism, human note)` where mechanism is one of
    `kwarg` / `template` / `unknown`. This exists because `RunMetrics
    .thinking_mode` records what was SENT, and on a template with no
    `enable_thinking` variable what was sent reached nothing -- the row would
    carry a pin that did no work.

    Read out of the GGUFs themselves (2026-09-22), the two Bonsai families do
    not agree on how thinking is controlled. The 27B's template reads
    `enable_thinking` and with it UNSET emits a bare `<think>`, so the pin
    `run_prism_suite` sends is load-bearing: drop it and those rungs silently
    become thinking runs at many times the energy. The 4B's template has no
    `enable_thinking` at all and hardcodes an empty `<think>` block, so
    thinking is off whatever is sent and the pin is inert.

    Both therefore measure thinking-off, which is what the comparison needs. A
    third template that did neither would be thinking-ON while its row still
    read `enable_thinking=false`, and that is the case worth shouting about.
    """
    if not chat_template:
        return THINKING_AXIS_UNKNOWN, (
            "the server reported no chat template, so the thinking axis could "
            "not be confirmed; the row's thinking_mode records only what was sent"
        )
    if "enable_thinking" in chat_template:
        return THINKING_AXIS_KWARG, (
            "the template reads enable_thinking, so the pin this mode sends is "
            "what holds thinking off"
        )
    if _THINK_OFF_PATTERN.search(chat_template):
        return THINKING_AXIS_TEMPLATE, (
            "the template hardcodes an empty <think> block, so thinking is off "
            "by construction and the enable_thinking pin is inert here"
        )
    return THINKING_AXIS_UNKNOWN, (
        "the template neither reads enable_thinking nor hardcodes an empty "
        "<think> block -- this rung may be REASONING while its row records "
        "enable_thinking=false. Treat its energy as a different axis until "
        "the template is read by hand"
    )


async def _fetch_chat_template(base_url: str) -> str | None:
    """The template llama-server will render, or None if it cannot be read.

    Advisory only, and deliberately non-fatal: this tier identifies its rung
    from `GET /v1/models`, so nothing about the measurement depends on
    `/props`. It is read for the thinking axis alone and never stored on a
    row -- a server too old to serve `/props` should still produce a rung.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/props")
        if response.status_code != 200:
            return None
        template = response.json().get("chat_template")
    except (httpx.RequestError, ValueError):
        return None
    return template if isinstance(template, str) else None


async def run_prism_suite(
    *,
    host: str = "localhost",
    ollama_port: int = DEFAULT_OLLAMA_PORT,
    llamacpp_port: int = DEFAULT_LLAMACPP_PORT,
    target_host: str | None = None,
    restore_to_factory_default: bool = True,
    budget_s: float | None = None,
) -> QuickSuiteResult:
    """Measure whichever pinned Bonsai rung llama-server is currently serving.

    Raises:
        NoEngineDetectedError: No llama-server answered.
        ModelNotAvailableError: It is up but serving nothing.
        PrismWeightsUnknownError: It is serving weights this tier cannot name.
    """
    detected = await detect_engine("llamacpp", host, ollama_port, llamacpp_port)
    if detected is None:
        raise NoEngineDetectedError(
            f"no llama-server answered on {host}:{llamacpp_port}.\n\n" + _how_to_serve_it()
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
    entry = match_served_gguf(served)
    if entry is None:
        raise PrismWeightsUnknownError(
            f"llama-server is serving '{served}', which is not a pinned Bonsai "
            f"rung. This tier measures what a specific storage of a specific "
            f"checkpoint costs, so a row that cannot name its weights answers "
            f"nothing.\n\n"
            f"If that id looks like a symlink or an `-a` alias, THAT is the "
            f"problem: llama-server reports the path it was given, not the "
            f"file it resolves to, so an alias hides the one thing this tier "
            f"has to record. Serve the GGUF by its real name.\n\n"
            + _how_to_serve_it()
        )

    logger.info("bench prism: measuring sub-4-bit weights on llama.cpp")
    logger.info("  rung    : %s (%s, %.2f GB)", entry.filename, entry.quantization, entry.size_gb)
    logger.info("  stage   : %s", entry.stage)
    logger.info("  weights : %s @ %s", entry.repo, entry.revision[:12])
    logger.info("  recorded: model=%s quantization=%s", entry.hf_id, entry.quantization)

    # The row records the kwarg that was SENT; this says whether it did any
    # work. Advisory -- see `thinking_axis_note` for why the distinction
    # matters and `_fetch_chat_template` for why a failure here is not fatal.
    mechanism, note = thinking_axis_note(await _fetch_chat_template(detected.base_url))
    if mechanism == THINKING_AXIS_UNKNOWN:
        logger.warning("  thinking: %s", note)
    else:
        logger.info("  thinking: %s", note)

    return await _run_bench_suite(
        engine_choice="llamacpp",
        host=host,
        ollama_port=ollama_port,
        llamacpp_port=llamacpp_port,
        target_host=target_host,
        tasks=PRISM_TASKS,
        # Stock power, like every other attach-mode tier: the ladders are a
        # separate axis and would change the thing being compared.
        max_sweep_points=0,
        label_prefix_stem="prism",
        restore_to_factory_default=restore_to_factory_default,
        budget_s=budget_s,
        # Pinned off, matching `full`, so a Bonsai row and an Ollama row differ
        # in the weights and nothing else.
        thinking=False,
        entry=entry,
        gguf_repo=entry.repo,
        gguf_revision=entry.revision,
    )
