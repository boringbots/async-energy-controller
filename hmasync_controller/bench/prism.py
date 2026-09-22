"""`bench prism` -- Bonsai's sub-4-bit weights, measured on Apple Silicon.

The lab ran this wave on two NVIDIA cards (RTX 3090, RTX 2080) in September
2026 and it produced one result nobody has confirmed anywhere else:

    Below 2 bits per weight, the saving REVERSES. On one 27B checkpoint,
    PTQ1_0 against PQ2_0 -- same weights, 13% fewer bytes -- accuracy was
    indistinguishable but energy was 1.20-1.57x HIGHER. The dequantization
    costs more arithmetic than the memory traffic it saves.

Both measurements were Ampere. A Mac is the right second architecture to ask
it on, and not for novelty: the claim is a statement about the ratio between
arithmetic and memory traffic, and Apple Silicon sits at a very different
point on that ratio than a 3090 does. If the reversal is a property of the
format it should survive the move; if it is a property of one card's memory
system it should not.

## What this mode measures, and what it deliberately does not

Two stages, five rungs (see `PRISM_RUNGS`):

    reversal        27B ternary, PTQ1_0 vs PQ2_0 -- the open question above
    format-control  4B, the SAME checkpoint in three storages (F16, PQ2_0,
                    Q2_0_g64) -- storage is the only thing that differs, so
                    any energy gap between them is bytes-per-token and
                    kernel, nothing else

The 4B family, not the 8B one the lab used. Its F16 control rung is 8.05 GB
against the 8B's 16.38 GB, and on a 24 GB unified-memory Mac a 16.38 GB rung
plus a 16k window is close enough to the ceiling to swap. Swap inflates both
wall-clock and joules, macOS does not report it here, and a swapping rung
would silently become the most expensive one in a comparison whose entire
point is that nothing but storage differs. `prism-ml` publishes the same
three files for the 4B and 1.7B checkpoints, so the control is available at a
size that fits with room to spare.

`PRISM_TASKS` is 150 items per rung, not the lab's 700. That is enough for
the two questions asked here and not enough for a third:

  - the reversal is an ENERGY question at matched accuracy, and energy
    repeats to about 1% CV;
  - the format control is an IDENTITY question, checked per item rather than
    in aggregate -- three storages of one checkpoint at temperature 0 should
    return the same answer to the same item, and `items.parquet` records
    `correct` and `completion_tokens` per `item_id`, so the check is exact
    and needs no confidence interval (`scripts/compare-prism-rungs.py`);
  - it CANNOT resolve an accuracy difference. At 100-150 items the interval
    is +/-0.10 or worse. A Mac row from this mode is not evidence about
    whether Bonsai is more or less accurate than anything else.

## Why this mode never builds a submission bundle

The public submission contract (`schemas/bench_submission.schema.json`, owned
by the lab repo) enumerates `suite` as quick|calibrate|medium|full|reference,
and the leaderboard's config key is model|quant|engine|gpu|power|task|
thinking|cap -- it has no field for WHICH BUILD of llama.cpp served the row.
These rungs run on a fork that implements two private ggml types mainline
refuses; pooling them into "llama.cpp" would repeat, on the public
leaderboard, exactly the mistake the lab's own port interlock exists to
prevent. So `bench prism` writes run artifacts and a local summary, and
stops there. Publishing these rows is an operator decision that starts with
a schema change in the other repo, and this package does not pre-empt it.

Two further reasons the numbers stay local: Apple joules carry
`energy_source='ioreport'` and never pool with NVIDIA's `counter` (F5 found
22% between two identical 3090s -- across vendors it is worse), and the
thermal circuit-breaker is inert on Apple Silicon, so a throttled rung
completes looking healthy. Every comparison this wave makes is therefore
BETWEEN RUNGS ON ONE MAC, measured back to back, which is the only shape in
which those two facts do not matter.

## The engine, and the two interlocks

The fork is `github.com/PrismML-Eng/llama.cpp`. Its Metal backend implements
both private types -- `kernel_mul_mv_pq2_0_f32`, `kernel_mul_mv_ptq1_0_f32`,
the `mul_mm`/`mul_mm_id` templates and both dequantizers -- and its README
calls PQ2_0 "preferred on Metal, CUDA, HIP and CPU". Release
`prism-b10709-9a9394a` ships a prebuilt macos-arm64 `llama-server`, so a Mac
needs no toolchain. That release is 22 commits ahead of the lab's pinned
`5d80cff` and NOT ONE of them touches `ggml/` (they are speculative-decoding
work), so the quantization kernels are the same code that produced the lab's
rows. `engine_version` on every row records the build the server reports.

`llama-server` is attach-only here, as everywhere in this package -- the
wave's per-rung loop lives in `scripts/run-prism-wave-mac.sh`, which starts
one server per rung and calls this mode once against each. What this mode
does before measuring is verify, from `GET /props`, that the server is
serving the rung it was asked for:

  1. `model_path` -- the file itself, by name. Without this the F16 rung is
     the dangerous one: it is an ordinary GGUF that a mainline llama-server
     or an LM Studio instance would serve perfectly well, and the resulting
     row would be clean, plausible and wrong. (The lab hit the same hazard
     from the other direction and answered it with a dedicated port; this
     mode defaults to that same port, 8091, for the same reason.)
  2. `model_ftype` -- the loader's own words for the format it found, e.g.
     "PQ2_0 - 2.13 bpw (group 128)". This is the interlock that catches the
     LEGACY FILE TRAP: `prism-ml`'s 8B and 4B model cards recommend a bare
     `*-Q2_0.gguf`, which stores group-128 weights under the group-64 ftype
     id. The fork now loads that file and names it "... (group 128, legacy
     ftype)", so it no longer fails loudly -- it would just quietly occupy
     the Q2_0_g64 rung and make the format control compare a file against
     itself. `PRISM_LEGACY_FTYPE_MARKER` refuses it by name.
  3. `n_ctx` -- the window actually granted. llama.cpp's auto-fit shrinks the
     context rather than failing when a model does not fit, down to 4096. A
     control whose rungs differ in both storage AND window measures nothing,
     so the driver passes `--fit off` and this mode checks the result.

## The thinking axis is pinned two different ways, and only one is visible

Read out of the GGUFs themselves (ranged GET over the metadata block,
2026-09-22), the two families do not agree on how thinking is controlled:

    Ternary-Bonsai-2-27B  the template reads `enable_thinking`, and with it
                          UNSET emits a bare `<think>` -- i.e. thinking ON.
                          The kwarg this mode sends is load-bearing: drop it
                          and the 27B rungs silently become thinking runs at
                          many times the energy.
    Ternary-Bonsai-4B     the template has no `enable_thinking` at all. Its
                          generation prompt hardcodes `<think>\n\n</think>`,
                          so thinking is off no matter what is sent.

Both therefore measure thinking-off, which is what the comparison needs -- but
the row's `thinking_mode` field records the kwarg that was SENT, and on the 4B
that kwarg reads nothing. `thinking_axis_note` re-reads the served template
from `/props` and says which of the two mechanisms is actually holding the
axis, so the log does not let the label speak for the template.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from hmasync_controller.bench.engines import DEFAULT_LLAMACPP_PORT, DEFAULT_OLLAMA_PORT
from hmasync_controller.bench.quick import (
    ModelNotAvailableError,
    NoEngineDetectedError,
    QuickError,
    QuickSuiteResult,
    _run_bench_suite,
    detect_engine,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PRISM_CTX_SIZE",
    "PRISM_ENGINE_RELEASE",
    "PRISM_GUESSED_FTYPE_PREFIX",
    "PRISM_LEGACY_FILENAME_SUFFIX",
    "PRISM_LEGACY_FTYPE_MARKER",
    "PRISM_PORT",
    "PRISM_RUNGS",
    "PRISM_TASKS",
    "PrismContextMismatchError",
    "PrismFtypeMismatchError",
    "PrismRung",
    "PrismSuiteResult",
    "PrismWeightsMismatchError",
    "ServedModel",
    "THINKING_AXIS_KWARG",
    "THINKING_AXIS_TEMPLATE",
    "THINKING_AXIS_UNKNOWN",
    "describe_wave",
    "rung_by_key",
    "rung_keys",
    "run_prism_suite",
    "thinking_axis_note",
]


PRISM_PORT = 8091
"""Not 8080. Three of this wave's rungs are files another llama-server could
serve (F16 especially), so a shared port could attach a prism rung to a
mainline server and produce a plausible wrong row. A distinct port makes that
impossible rather than unlikely -- the same interlock, and the same number,
the lab's own wave uses."""

PRISM_CTX_SIZE = 16384
"""One window for every rung, matching the lab's wave. A ladder that changes
the window changes two variables at once; 16384 is also the largest per-task
cap in `PRISM_TASKS`, so no rung meets the window before it meets its cap."""

PRISM_ENGINE_RELEASE = "prism-b10709-9a9394a"
"""The fork release with a prebuilt macos-arm64 `llama-server`. 22 commits
ahead of the lab's pinned `5d80cff`, none of them under `ggml/` -- so the
ternary kernels are identical to the ones that produced the lab's rows, and
only the speculative-decoding code this wave never touches differs."""

PRISM_LAB_REF = "5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6"
"""What the lab's NVIDIA rows were measured on. Recorded here so a reader can
see which two builds the "identical kernels" claim above is about."""

PRISM_GUESSED_FTYPE_PREFIX = "(guessed) "
"""`llama_ftype_name` prepends this when the GGUF did not declare
`general.file_type` and the loader inferred it. Not fatal -- the file is
already pinned by repo, revision and name -- but it is worth a warning in a
run whose whole subject is the format."""

PRISM_LEGACY_FTYPE_MARKER = "legacy ftype"
"""Appears in the loader's name for the deprecated group-128-stored-as-id-42
files. See the module docstring: these load now, which is precisely why they
have to be refused by name."""

PRISM_LEGACY_FILENAME_SUFFIX = "-Q2_0.gguf"
"""No rung may name a bare `*-Q2_0.gguf`. `prism-ml`'s model cards still
recommend one; on these builds it is the legacy file. Asserted in tests so an
edit cannot reintroduce it quietly."""


@dataclass(frozen=True)
class PrismRung:
    """One measured configuration: which file, and how the row records it."""

    key: str
    """What `--rung` takes. Stable: it names the summary file and is what the
    comparison script joins on."""

    stage: str
    """`reversal` or `format-control` -- which question this rung serves."""

    model: str
    """`RunMetrics.model`. The lab's own identity string for the checkpoint
    (e.g. `prism-ml/Ternary-Bonsai-4B`), so a Mac row and a lab row for the
    same weights carry the same model name. Deliberately NOT the `-gguf`
    repo id: the three format-control rungs are one checkpoint and must share
    a model identity, differing only in `quantization`."""

    quantization: str
    """`RunMetrics.quantization`. The axis under test, so it is recorded
    rather than left None as the plain llama.cpp attach path does."""

    gguf_repo: str
    gguf_revision: str
    """Repo AND commit. A quantization name is not a set of weights, and a
    re-upload under one filename is exactly what this pins against."""

    gguf_file: str
    """Basename as published. Checked against `/props.model_path`."""

    ftype: str
    """What the loader must report for this file -- `llama_ftype_name`'s exact
    string, closing parenthesis included. The parenthesis matters: it is what
    separates "PQ2_0 - 2.13 bpw (group 128)" from the legacy file's
    "PQ2_0 - 2.13 bpw (group 128, legacy ftype)"."""

    size_gb: float
    """Published file size. Used only to tell an operator what to download and
    whether it will fit."""

    note: str
    """One line: what this rung is in the wave for."""


PRISM_RUNGS: tuple[PrismRung, ...] = (
    # --- Stage 1: the reversal test. The open question, so it runs first --
    # if a Mac only ever completes two rungs, these are the two worth having.
    PrismRung(
        key="bonsai2-27b-ptq1-0",
        stage="reversal",
        model="prism-ml/Ternary-Bonsai-2-27B",
        quantization="PTQ1_0",
        gguf_repo="prism-ml/Ternary-Bonsai-2-27B-gguf",
        gguf_revision="6ed5e12bf84b7a63069882c91dd9e9218647d17b",
        gguf_file="Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        ftype="PTQ1_0 - 1.75 bpw ternary (group 128)",
        size_gb=5.95,
        note="27B ternary, the narrower storage -- 1.20-1.57x MORE energy than "
        "PQ2_0 on Ampere at the same accuracy",
    ),
    PrismRung(
        key="bonsai2-27b-pq2-0",
        stage="reversal",
        model="prism-ml/Ternary-Bonsai-2-27B",
        quantization="PQ2_0",
        gguf_repo="prism-ml/Ternary-Bonsai-2-27B-gguf",
        gguf_revision="6ed5e12bf84b7a63069882c91dd9e9218647d17b",
        gguf_file="Ternary-Bonsai-2-27B-PQ2_0.gguf",
        ftype="PQ2_0 - 2.13 bpw (group 128)",
        size_gb=7.21,
        note="the same 27B one bit wider -- 1.21x the bytes, and the cheaper "
        "rung on Ampere",
    ),
    # --- Stage 2: the format control. One checkpoint, three storages. The
    # 4B family rather than the lab's 8B: its F16 rung fits a 24 GB Mac with
    # room for the window, and the 8B's 16.38 GB one does not.
    PrismRung(
        key="bonsai-4b-f16",
        stage="format-control",
        model="prism-ml/Ternary-Bonsai-4B",
        quantization="F16",
        gguf_repo="prism-ml/Ternary-Bonsai-4B-gguf",
        gguf_revision="a3eb42bafe873f9686bc97486c43b72ef7d75ec8",
        gguf_file="Ternary-Bonsai-4B-F16.gguf",
        ftype="F16",
        size_gb=8.05,
        note="ternary weights stored DENSE -- the re-quantization source, not "
        "the Qwen3-4B base",
    ),
    PrismRung(
        key="bonsai-4b-pq2-0",
        stage="format-control",
        model="prism-ml/Ternary-Bonsai-4B",
        quantization="PQ2_0",
        gguf_repo="prism-ml/Ternary-Bonsai-4B-gguf",
        gguf_revision="a3eb42bafe873f9686bc97486c43b72ef7d75ec8",
        gguf_file="Ternary-Bonsai-4B-PQ2_0.gguf",
        ftype="PQ2_0 - 2.13 bpw (group 128)",
        size_gb=1.07,
        note="the same weights packed at group 128 -- 7.5x smaller than F16",
    ),
    PrismRung(
        key="bonsai-4b-q2-0-g64",
        stage="format-control",
        model="prism-ml/Ternary-Bonsai-4B",
        quantization="Q2_0_g64",
        gguf_repo="prism-ml/Ternary-Bonsai-4B-gguf",
        gguf_revision="a3eb42bafe873f9686bc97486c43b72ef7d75ec8",
        gguf_file="Ternary-Bonsai-4B-Q2_0_g64.gguf",
        ftype="Q2_0",
        size_gb=1.14,
        note="the same weights at the official group 64 -- the only rung "
        "mainline llama.cpp could also serve",
    ),
)

PRISM_TASKS: list[tuple[str, int]] = [
    ("gsm8k_platinum", 50),
    ("mmlu_redux", 100),
]
"""(task, n_items) per rung. The lab's matrix is 700 items over four tasks and
took 9-13 hours on a 3090; the same thing on an M3 would run into days, and
`math500`/`gpqa_diamond` are the expensive half (16k caps, and Bonsai reasons
through them). These two tasks are one decode-heavy and one prefill-heavy
shape, which is what an energy comparison between storages needs. Fixed, not
configurable: rungs that differ in item count are not a control."""


class PrismWeightsMismatchError(QuickError):
    """llama-server is serving a different file than the rung asked for.

    Refused rather than measured. A row from the wrong file is not a failed
    measurement, it is a confident wrong one, and the F16 rung is a file any
    llama-server can serve.
    """


class PrismFtypeMismatchError(QuickError):
    """The loader reports a different format than the rung expects -- or the
    deprecated legacy packing wearing the right filename."""


class PrismContextMismatchError(QuickError):
    """The served window is not `PRISM_CTX_SIZE`; llama.cpp's auto-fit
    probably shrank it. Every rung must share one window."""


class UnknownPrismRungError(QuickError):
    """`--rung` named something that is not in `PRISM_RUNGS`."""


@dataclass(frozen=True)
class ServedModel:
    """What `GET /props` says is actually loaded, before anything is measured."""

    model_path: str | None
    ftype: str | None
    n_ctx: int | None
    build_info: str | None
    chat_template: str | None = None
    """The template the server will actually render. Read for the thinking
    axis alone (see `thinking_axis_note`); never stored on a row."""


@dataclass
class PrismSuiteResult:
    """One rung's measurement, plus the provenance that was verified first.

    Exposes `runs`/`task_runs` so the CLI's shared artifact writer treats it
    exactly like a `QuickSuiteResult`.
    """

    rung: PrismRung
    served: ServedModel
    suite: QuickSuiteResult

    @property
    def runs(self):  # noqa: ANN201 - mirrors QuickSuiteResult's own field
        return self.suite.runs

    @property
    def task_runs(self):  # noqa: ANN201 - mirrors QuickSuiteResult's own field
        return self.suite.task_runs


def rung_keys() -> tuple[str, ...]:
    return tuple(r.key for r in PRISM_RUNGS)


def rung_by_key(key: str) -> PrismRung:
    for rung in PRISM_RUNGS:
        if rung.key == key:
            return rung
    raise UnknownPrismRungError(
        f"unknown rung '{key}'. This wave's rungs are: {', '.join(rung_keys())}. "
        f"Run `async-energy-controller bench prism --list` for the plan."
    )


def describe_wave() -> str:
    """The wave's plan, as an operator needs to read it before starting."""
    lines = [
        "The prism wave -- Bonsai sub-4-bit weights on Apple Silicon.",
        "",
        f"Engine: PrismML's llama.cpp fork, release {PRISM_ENGINE_RELEASE} "
        f"(prebuilt macos-arm64).",
        f"Per rung: {', '.join(f'{t} x{n}' for t, n in PRISM_TASKS)}, "
        f"thinking off, {PRISM_CTX_SIZE}-token window, port {PRISM_PORT}.",
        "",
        f"{'rung':<22} {'stage':<15} {'quant':<10} {'GB':>5}  file",
    ]
    for rung in PRISM_RUNGS:
        lines.append(
            f"{rung.key:<22} {rung.stage:<15} {rung.quantization:<10} "
            f"{rung.size_gb:>5.2f}  {rung.gguf_file}"
        )
    total = sum(r.size_gb for r in PRISM_RUNGS)
    lines += [
        "",
        f"{total:.2f} GB of weights over {len(PRISM_RUNGS)} rungs. Download them "
        f"with `scripts/run-prism-wave-mac.sh fetch`, which pins every revision.",
        "",
        "Each rung is a separate server and a separate invocation; "
        "scripts/run-prism-wave-mac.sh run drives the loop.",
    ]
    return "\n".join(lines)


async def fetch_props(base_url: str) -> ServedModel:
    """Read what llama-server says it is serving.

    Raises `ModelNotAvailableError` rather than returning empties: every
    check this mode makes depends on this call, and a mode that cannot verify
    the weights must not measure them.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/props")
    except httpx.RequestError as e:
        raise ModelNotAvailableError(
            f"could not read GET {base_url}/props: {e}. bench prism verifies "
            f"the served weights before measuring and cannot proceed without it."
        ) from e
    if response.status_code != 200:
        raise ModelNotAvailableError(
            f"GET {base_url}/props returned HTTP {response.status_code}. bench "
            f"prism needs it to verify which GGUF is loaded."
        )
    data = response.json()
    settings = data.get("default_generation_settings") or {}
    n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
    return ServedModel(
        model_path=data.get("model_path"),
        ftype=data.get("model_ftype"),
        n_ctx=int(n_ctx) if isinstance(n_ctx, (int, float)) else None,
        build_info=data.get("build_info"),
        chat_template=data.get("chat_template"),
    )


def _how_to_serve(rung: PrismRung, port: int) -> str:
    return (
        f"Serve this rung and re-run:\n"
        f"  hf download {rung.gguf_repo} {rung.gguf_file} \\\n"
        f"    --revision {rung.gguf_revision} --local-dir ./models/prism\n"
        f"  llama-server -m ./models/prism/{rung.gguf_file} \\\n"
        f"    -c {PRISM_CTX_SIZE} --fit off -ngl 999 --port {port}\n"
        f"The llama-server must be PrismML's build ({PRISM_ENGINE_RELEASE}); "
        f"mainline llama.cpp does not implement this wave's ggml types.\n"
        f"`scripts/run-prism-wave-mac.sh` does all of this per rung."
    )


def verify_served(rung: PrismRung, served: ServedModel, *, port: int) -> None:
    """Refuse anything but this rung's file, format and window.

    Order matters: the file is checked first because a wrong file makes the
    other two answers meaningless, and its remedy is the one an operator can
    act on.
    """
    if not served.model_path:
        raise PrismWeightsMismatchError(
            "llama-server did not report a model_path, so the served weights "
            "cannot be verified. bench prism refuses to measure unverified "
            "weights -- a row from the wrong file reads as a real result.\n\n"
            + _how_to_serve(rung, port)
        )
    served_file = served.model_path.replace("\\", "/").rsplit("/", 1)[-1]
    if served_file != rung.gguf_file:
        raise PrismWeightsMismatchError(
            f"llama-server is serving '{served_file}', but rung '{rung.key}' is "
            f"'{rung.gguf_file}'. Refused rather than measured.\n\n"
            + _how_to_serve(rung, port)
        )

    if not served.ftype:
        raise PrismFtypeMismatchError(
            f"llama-server reported no model_ftype for {served_file}, so the "
            f"packing cannot be confirmed. This wave's whole subject is the "
            f"packing.\n\n" + _how_to_serve(rung, port)
        )
    ftype = served.ftype
    if ftype.startswith(PRISM_GUESSED_FTYPE_PREFIX):
        # The loader guessed because the file did not declare
        # general.file_type. The weights are still pinned by repo+revision, so
        # this is a warning rather than a refusal -- but in a format control it
        # belongs in the log.
        logger.warning(
            "bench prism: the loader GUESSED this file's format (%s). The file "
            "itself did not declare one.", ftype,
        )
        ftype = ftype[len(PRISM_GUESSED_FTYPE_PREFIX):]
    if PRISM_LEGACY_FTYPE_MARKER in ftype:
        raise PrismFtypeMismatchError(
            f"{served_file} loaded as '{served.ftype}' -- the DEPRECATED "
            f"group-128-stored-as-id-42 packing. It loads on this build, which "
            f"is why it has to be refused by name: left alone it would occupy "
            f"this rung and make the format control compare a file with "
            f"itself. Download {rung.gguf_file} from {rung.gguf_repo} at "
            f"revision {rung.gguf_revision}; do not use the bare "
            f"'*{PRISM_LEGACY_FILENAME_SUFFIX}' file the model card recommends."
        )
    if ftype != rung.ftype:
        raise PrismFtypeMismatchError(
            f"{served_file} loaded as '{served.ftype}', but rung '{rung.key}' "
            f"expects '{rung.ftype}'. The filename and the packing disagree, so "
            f"one of them is not what it claims.\n\n" + _how_to_serve(rung, port)
        )

    if served.n_ctx is not None and served.n_ctx != PRISM_CTX_SIZE:
        raise PrismContextMismatchError(
            f"llama-server granted a {served.n_ctx}-token window, not "
            f"{PRISM_CTX_SIZE}. llama.cpp's auto-fit shrinks the context rather "
            f"than failing when a model does not fit, and rungs that differ in "
            f"both storage and window measure nothing. Start it with `--fit off "
            f"-c {PRISM_CTX_SIZE}`; if it then refuses to load, that refusal is "
            f"the measurement -- record it and move to the next rung."
        )


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
    `kwarg` / `template` / `unknown`. This exists because
    `RunMetrics.thinking_mode` records what was SENT, and on a template with
    no `enable_thinking` variable what was sent reached nothing -- the row
    would carry a pin that did no work. Both Bonsai families end up
    thinking-off, by different routes; a third template that does neither
    would be thinking-ON while the row still claimed otherwise, and that is
    the case worth shouting about.
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


async def run_prism_suite(
    *,
    rung_key: str,
    host: str = "localhost",
    ollama_port: int = DEFAULT_OLLAMA_PORT,
    llamacpp_port: int = PRISM_PORT,
    target_host: str | None = None,
    budget_s: float | None = None,
) -> PrismSuiteResult:
    """Measure one rung against an already-running prism llama-server.

    Verifies the file, the packing and the window before measuring anything
    (see `verify_served`), then runs `PRISM_TASKS` at stock power with
    thinking pinned off.

    Raises:
        NoEngineDetectedError: No llama-server answered on `llamacpp_port`.
        ModelNotAvailableError: It answered but `/props` could not be read.
        PrismWeightsMismatchError / PrismFtypeMismatchError /
        PrismContextMismatchError: It is serving something other than this
            rung, in one of the three ways that matter.
        UnknownPrismRungError: `rung_key` is not in `PRISM_RUNGS`.
    """
    rung = rung_by_key(rung_key)

    detected = await detect_engine("llamacpp", host, ollama_port, llamacpp_port)
    if detected is None:
        raise NoEngineDetectedError(
            f"no llama-server answered on {host}:{llamacpp_port}. bench prism "
            f"never launches an engine itself.\n\n" + _how_to_serve(rung, llamacpp_port)
        )

    split = urlsplit(detected.base_url)
    served = await fetch_props(detected.base_url)
    verify_served(rung, served, port=split.port or llamacpp_port)

    logger.info("bench prism: rung %s (%s)", rung.key, rung.note)
    logger.info("  weights: %s @ %s", rung.gguf_repo, rung.gguf_revision[:12])
    logger.info("  served:  %s", served.model_path)
    logger.info("  format:  %s", served.ftype)
    logger.info("  window:  %s tokens", served.n_ctx)
    logger.info("  engine:  %s", served.build_info or "unreported")
    mechanism, note = thinking_axis_note(served.chat_template)
    if mechanism == THINKING_AXIS_UNKNOWN:
        logger.warning("  thinking: %s", note)
    else:
        logger.info("  thinking: off -- %s", note)

    suite = await _run_bench_suite(
        engine_choice="llamacpp",
        host=host,
        ollama_port=ollama_port,
        llamacpp_port=llamacpp_port,
        target_host=target_host,
        tasks=PRISM_TASKS,
        # No power sweep. Apple exposes no power-limit control at all, and the
        # comparison here is between storages at stock power.
        max_sweep_points=0,
        label_prefix_stem="prism",
        # Nothing to restore: the sweep never runs, and NVML's power limit does
        # not exist on the hardware this mode is for.
        restore_to_factory_default=False,
        budget_s=budget_s,
        # Pinned off, never left to the model -- matching the lab's own prism
        # configs (chat_template_kwargs: enable_thinking: false). Thinking is
        # the largest energy lever these models have, and an unpinned axis
        # would sit inside a storage comparison.
        thinking=False,
        model_id=rung.model,
        quantization=rung.quantization,
        gguf_repo=rung.gguf_repo,
        gguf_revision=rung.gguf_revision,
    )
    return PrismSuiteResult(rung=rung, served=served, suite=suite)


def format_rung_summary(result: PrismSuiteResult) -> str:
    """What this rung cost, per task -- the line an operator reads at 2am."""
    lines = [f"  rung {result.rung.key}  ({result.rung.quantization}, {result.rung.size_gb:.2f} GB)"]
    for run in result.runs:
        joules = run.total_joules_gpu_best or run.total_joules_gpu
        per_correct = run.joules_per_correct_answer
        lines.append(
            f"    {run.task or '?':<18} acc {run.accuracy if run.accuracy is not None else float('nan'):.3f}  "
            f"{joules:>9.0f} J  "
            f"{per_correct if per_correct is not None else float('nan'):>8.1f} J/correct  "
            f"{run.joules_per_token:>6.3f} J/token  [{run.energy_source}]"
        )
    if any(run.energy_source != "ioreport" for run in result.runs):
        lines.append(
            "    NOTE: energy_source is not 'ioreport' -- these joules are not "
            "from an Apple counter. Compare rungs measured the same way."
        )
    return "\n".join(lines)
