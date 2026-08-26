"""`bench quick`/`bench calibrate`: the ~25-minute Tier-C onboarding suite and
its ~3-5 minute slimmed sibling, in-process (US-MERGE-04, US-MERGE-05).

Ported from energy-bench's `quick.py` + the driving loop in its
`main.py::execute_quick`, collapsed into ONE module + one shared async
orchestrator (`_run_bench_suite`, driving the two public entry points
`run_quick_suite`/`run_calibrate_suite`) since this package has no separate
CLI-printing layer to split against -- `cli.py::run_bench_quick`/
`run_bench_calibrate` call straight into this module and map the
result/exception to an exit code, the same division of labor energy-bench's
`main.py`/`quick.py` had.

## Why Ollama/llama.cpp, not vLLM

A stranger's box running `pip install async-energy-controller` almost never
has vLLM; it might have Ollama or a bare `llama-server`. Both adapters
(`bench/engines.py`) are attach-mode only: HTTP against a server this
process does not own, launch, or stop.

## Reference configuration -- inlined, not imported from a `grading` module

energy-bench pins one exact configuration (Qwen/Qwen3.5-9B, Q4_K_M GGUF,
llama.cpp, stock power) in `grading/reference.py`'s `REFERENCE_CONFIG`, used
there for its Efficiency Index (`eb reindex-efficiency`, which needs a
DuckDB run index this package doesn't have -- GROUND TRUTH: `grading/`
stays lab-side entirely). This module needs exactly THREE of that tuple's
values -- the HF model id `resolve_quick_model` records for Ollama, and the
`n_shot`/`seed` defaults `run_quick_task` falls back to -- so they are
inlined below as their own constants (`QUICK_REFERENCE_MODEL_HF_ID`,
`QUICK_REFERENCE_N_SHOT`, `QUICK_REFERENCE_SEED`) rather than pulling in the
whole `grading` module for three scalars this package never uses for
anything else.

## Telemetry: local NVML only

energy-bench's Tier C also supports `--collector-host`, sampling from a
remote collector container over HTTP (`CollectorTelemetrySource`). No
collector container exists in this package's world -- `bench.sampler.
LocalNvmlSampler` (US-MERGE-02, already unified with `profiler.py`'s 1 Hz
sampler) is the only telemetry source `run_quick_suite` ever constructs.

## Scope deliberately left out, same as energy-bench's Tier C

No Home Assistant anywhere in this module: `RunMetrics.ambient_c_start` is
always `None`, and `measurement_tier` reads 'C' unconditionally, since
`compute_metrics()` derives it from wall-sample presence alone. CPU/RAPL
energy is also not read locally -- `LocalNvmlSampler` (bench.sampler) has no
local RAPL reader, so `rapl_max_energy_range_uj`/`_dram_...` always resolve
to `None` for a run measured here (see `run_quick_task`'s note on reading
those attributes defensively).

## Hardware safety: thermal reaction (US-MERGE-07)

`run_quick_task`'s item loop checks `bench.thermal.maybe_pause_for_thermal_throttle`
between items (never mid-request): a sustained `hw_thermal` NVML bit pauses
until it clears or raises `SustainedThermalThrottleError` past a timeout,
which this function lets propagate -- callers already treat "one task
raised" as "skip it, keep the rest of the suite" (see `_run_bench_suite`),
so an aborted task never silently reports a throttled-through result. Full
rationale, thresholds, and why `hw_thermal` specifically (not the broader
thermal mask): `bench/thermal.py`'s module docstring and
`HARDWARE-SAFETY.md`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from hmasync_controller.bench.artifact import generate_run_id
from hmasync_controller.bench.engines import (
    DEFAULT_LLAMACPP_PORT,
    DEFAULT_OLLAMA_PORT,
    AttachLlamaCppAdapter,
    OllamaAdapter,
    OllamaModelNotPulledError,
)
from hmasync_controller.bench.metrics import (
    MIN_SWEEP_POINTS,
    InferenceResult,
    MetricsComputeError,
    RunMetrics,
    compute_metrics,
)
from hmasync_controller.bench.sampler import LocalNvmlSampler, NvmlUnavailableError, TelemetrySample
from hmasync_controller.bench.tasks import load_task
from hmasync_controller.bench.thermal import (
    SustainedThermalThrottleError,
    maybe_pause_for_thermal_throttle,
)
from hmasync_controller.bench.vllm_client import VLLMClient, VLLMTimeoutError, VLLMUnavailableError

logger = logging.getLogger(__name__)

# Re-exported for callers that only want to `from ...quick import
# NvmlUnavailableError` (mirrors energy-bench's quick.py, which defines this
# type itself; here it is bench.sampler's -- see that module's docstring for
# why there is exactly one NVML implementation in this repo). Reused, not
# redefined -- `# noqa: F401`-equivalent intent, kept off the linter by the
# explicit `__all__` below.
__all__ = [
    "AllTasksFailedError",
    "CALIBRATE_MAX_SWEEP_POINTS",
    "CALIBRATE_TASKS",
    "CALIBRATE_TASK_N_ITEMS",
    "DetectedEngine",
    "ModelNotAvailableError",
    "NoEngineDetectedError",
    "NvmlUnavailableError",
    "PowerSweepPoint",
    "QUICK_REFERENCE_MODELS",
    "QUICK_REFERENCE_MODEL_HF_ID",
    "QUICK_REFERENCE_N_SHOT",
    "QUICK_REFERENCE_SEED",
    "QUICK_TASKS",
    "QuickError",
    "QuickModel",
    "QuickSuiteResult",
    "QuickTaskRun",
    "SustainedThermalThrottleError",
    "detect_engine",
    "resolve_quick_model",
    "run_calibrate_suite",
    "run_power_sweep",
    "run_quick_suite",
    "run_quick_task",
]

# Verified live 2026-08-22 by energy-bench's US-ENG-14 (HF API search + a
# direct repo fetch confirmed unsloth/Qwen3.5-9B-GGUF ships
# Qwen3.5-9B-Q4_K_M.gguf; ollama.com/library/qwen3.5/tags lists
# "qwen3.5:9b-q4_K_M" verbatim) -- copied verbatim, not re-verified here.
QUICK_REFERENCE_MODELS: dict[str, dict[str, str]] = {
    "ollama": {
        "tag": "qwen3.5:9b-q4_K_M",
        "quantization": "Q4_K_M",
    },
    "llama.cpp": {
        "gguf_repo": "unsloth/Qwen3.5-9B-GGUF",
        "gguf_file": "Qwen3.5-9B-Q4_K_M.gguf",
        "revision": "3885219b6810b007914f3a7950a8d1b469d598a5",
    },
}

# The HF repo id `resolve_quick_model` records for an Ollama run (`model`
# stays the FP16 HF join key even though the reference is served quantized,
# matching the established convention `RunMetrics.model` uses elsewhere) and
# the n_shot/seed `run_quick_task` falls back to. Mirrors energy-bench's
# `grading.reference.REFERENCE_CONFIG["model"]`/`["n_shot"]`/`["seed"]` --
# see module docstring for why this package inlines three scalars instead of
# importing that module.
QUICK_REFERENCE_MODEL_HF_ID = "Qwen/Qwen3.5-9B"
QUICK_REFERENCE_N_SHOT = 5
QUICK_REFERENCE_SEED = 1234

QUICK_TASKS: list[tuple[str, int]] = [
    ("gsm8k_platinum", 25),
    ("mmlu_redux", 50),
    ("ifeval", 25),
]
"""(task, n_items) pairs -- a scaled-down slice of the community core-9 chosen
to fit the ~25-minute time budget, same n counts as energy-bench's `eb
quick`."""

POWER_SWEEP_TASK = "gsm8k_platinum"
"""The mini power sweep reuses the baseline pass's own gsm8k_platinum run as
its stock (uncapped) point -- the flexibility metrics need >=4 points sharing
one (model, quantization, engine, target_host, task) group, and this avoids
measuring the same thing twice."""

POWER_SWEEP_CAPS_W: list[float] = [0.85, 0.75, 0.65]
"""The mini sweep's three capped points, as FRACTIONS of the card's own
stock power limit -- chosen to bracket published sweet spots (a 320 W 3090
at 225-250 W is 70-78%; a 575 W 5090 at 450 W is 78%).
`_derive_power_sweep_caps_w` turns these into concrete, clamped, distinct
watts for whatever card `bench quick` actually runs on."""

_CLI_ENGINE_NAMES: dict[str, str] = {"ollama": "ollama", "llamacpp": "llama.cpp"}
"""Maps a `--engine`-style CLI value (no dot, easier to type) to the
internal engine name this module uses everywhere else."""


class QuickError(Exception):
    """Base error for the `bench quick` pipeline."""


class NoEngineDetectedError(QuickError):
    """Neither Ollama nor llama-server answered on this box (or the
    explicitly requested engine didn't)."""


class ModelNotAvailableError(QuickError):
    """The model `bench quick` needs isn't available. The message names the
    exact operator action (e.g. `ollama pull ...`) -- this module never
    pulls a model itself (security standing rule)."""


class AllTasksFailedError(QuickError):
    """Every task in `QUICK_TASKS` failed -- nothing was measured, so there
    is nothing to build a bundle from."""


@dataclass
class DetectedEngine:
    """One reachable engine, ready to measure against."""

    name: str
    """Internal engine name: 'ollama' or 'llama.cpp'."""

    base_url: str
    adapter: OllamaAdapter | AttachLlamaCppAdapter


async def detect_engine(
    explicit: str | None,
    host: str = "localhost",
    ollama_port: int = DEFAULT_OLLAMA_PORT,
    llamacpp_port: int = DEFAULT_LLAMACPP_PORT,
) -> DetectedEngine | None:
    """Find a reachable engine on this box: Ollama first, then llama.cpp,
    or only the named one when `explicit` is given ('ollama' or 'llamacpp',
    'auto'/None tries both).

    Both adapters are constructed in **attach mode** (an explicit
    `base_url`) -- `bench quick` only ever measures a server it does not
    own; see module docstring.

    Returns None when nothing answered.
    """
    candidates = [explicit] if explicit and explicit != "auto" else ["ollama", "llamacpp"]
    for cli_name in candidates:
        registry_name = _CLI_ENGINE_NAMES.get(cli_name)
        if registry_name is None:
            continue
        if registry_name == "ollama":
            base_url = f"http://{host}:{ollama_port}"
            adapter: OllamaAdapter | AttachLlamaCppAdapter = OllamaAdapter(
                host=host, port=ollama_port, base_url=base_url
            )
        else:
            base_url = f"http://{host}:{llamacpp_port}"
            adapter = AttachLlamaCppAdapter(host=host, port=llamacpp_port, base_url=base_url)

        if await adapter.ready():
            return DetectedEngine(name=registry_name, base_url=base_url, adapter=adapter)

    return None


@dataclass
class QuickModel:
    """The model `bench quick` will measure, and what it took to confirm it."""

    name: str
    """The exact string sent as `"model"` in chat-completion requests."""

    note: str
    """Human-readable "what ran" description for logging and the bundle --
    never a plausible-looking guess: for llama.cpp this explicitly says
    "whatever was already loaded"."""

    record_model: str
    """The value to store in `RunMetrics.model`. Ollama: the pinned
    reference's HF repo id (`QUICK_REFERENCE_MODEL_HF_ID`). llama.cpp:
    whatever `name` is (the server's own identifier) -- there is no HF repo
    id to join to when the model is "closest available", not the pinned
    one, so recording the reference's id here would be a guess."""

    record_quantization: str | None
    """The value to store in `RunMetrics.quantization`. 'Q4_K_M' only for
    Ollama, where the pinned tag guarantees it; None for llama.cpp, where
    the actually-loaded GGUF's quant is unknown from an attach-mode HTTP
    surface alone."""


async def resolve_quick_model(engine: DetectedEngine) -> QuickModel:
    """Resolve + verify the model this run will measure. Never pulls.

    Raises:
        ModelNotAvailableError: The Ollama tag isn't pulled (names the exact
            `ollama pull ...` command), or llama-server reports no model
            loaded at all.
    """
    if engine.name == "ollama":
        tag = QUICK_REFERENCE_MODELS["ollama"]["tag"]
        try:
            family = await engine.adapter.verify_model_pulled(tag)
        except OllamaModelNotPulledError as e:
            raise ModelNotAvailableError(str(e)) from e
        note = f"ollama:{tag}" + (f" (family={family})" if family else "")
        return QuickModel(
            name=tag,
            note=note,
            record_model=QUICK_REFERENCE_MODEL_HF_ID,
            record_quantization=QUICK_REFERENCE_MODELS["ollama"]["quantization"],
        )

    # llama.cpp: attach-only, no swap-by-request API -- measure whatever is
    # already loaded (see module docstring).
    split = urlsplit(engine.base_url)
    client = VLLMClient(
        host=split.hostname or "localhost", port=split.port or DEFAULT_LLAMACPP_PORT
    )
    client.base_url = engine.base_url
    models = await client.get_models()
    if not models:
        raise ModelNotAvailableError(
            "llama-server is healthy but GET /v1/models reports no loaded "
            "model. bench quick never launches or swaps an attach-mode "
            "server -- start llama-server with a real GGUF (e.g. "
            f"{QUICK_REFERENCE_MODELS['llama.cpp']['gguf_repo']}/"
            f"{QUICK_REFERENCE_MODELS['llama.cpp']['gguf_file']}, or any "
            "model you already have) and re-run `bench quick`."
        )
    served = models[0]
    return QuickModel(
        name=served,
        note=f"llama.cpp: whatever was already loaded ({served})",
        record_model=served,
        record_quantization=None,
    )


@dataclass
class QuickTaskRun:
    """One measured task: everything needed to compute its `RunMetrics` and
    write its artifact, without reaching back into task internals."""

    task_name: str
    task_shape: str
    is_canary: bool
    dataset_revision: str | None
    n_items: int
    n_shot: int
    seed: int
    max_tokens: int
    power_limit_w: int | None
    inference_results: list[InferenceResult] = field(default_factory=list)
    telemetry_samples: list[TelemetrySample] = field(default_factory=list)
    streaming_used: bool = True
    rapl_max_energy_range_uj: float | None = None
    rapl_dram_max_energy_range_uj: float | None = None


async def run_quick_task(
    vllm_client: VLLMClient,
    telemetry: LocalNvmlSampler,
    model_name: str,
    task_name: str,
    n_items: int,
    *,
    power_limit_w: int | None = None,
    n_shot: int | None = None,
    seed: int | None = None,
) -> QuickTaskRun:
    """Load a task, sample telemetry across it, score every item.

    Streaming first, per-request fallback to non-streaming on failure --
    mirrors energy-bench's `orchestrator.runner._execute_probe` inference
    loop without the HA/power-limit/cooldown machinery that loop also
    carries, none of which applies to an attach-mode, HA-less run.

    `n_shot`/`seed` default to `QUICK_REFERENCE_N_SHOT`/`QUICK_REFERENCE_SEED`
    so a run against the exact reference model is directly comparable to it.
    """
    resolved_n_shot = n_shot if n_shot is not None else QUICK_REFERENCE_N_SHOT
    resolved_seed = seed if seed is not None else QUICK_REFERENCE_SEED

    task = load_task(task_name)
    max_tokens = task.default_max_tokens
    items = task.load(n_items=n_items, n_shot=resolved_n_shot, seed=resolved_seed)

    run_id = f"quick-{task_name}-{power_limit_w or 'stock'}-{int(time.time() * 1000)}"
    await telemetry.start(run_id)

    inference_results: list[InferenceResult] = []
    streaming_used = True
    try:
        for item in items:
            try:
                result, text = await vllm_client.chat(
                    prompt=item.prompt,
                    model=model_name,
                    max_tokens=max_tokens,
                    stop=task.stop,
                    temperature=0.0,
                    stream=True,
                )
            except (VLLMUnavailableError, VLLMTimeoutError):
                streaming_used = False
                result, text = await vllm_client.chat(
                    prompt=item.prompt,
                    model=model_name,
                    max_tokens=max_tokens,
                    stop=task.stop,
                    temperature=0.0,
                    stream=False,
                )
            result.item_id = item.item_id
            result.correct = task.score(text, item)
            inference_results.append(result)

            # Thermal reaction (US-MERGE-07): between items, never mid-request.
            # `current_samples` is read defensively -- a telemetry double that
            # doesn't define it (most test fakes) simply isn't monitored,
            # same convention as the rapl_*_range_uj getattr reads below.
            get_current_samples = getattr(telemetry, "current_samples", None)
            if get_current_samples is not None:
                await maybe_pause_for_thermal_throttle(get_current_samples)
    finally:
        telemetry_samples = await telemetry.stop()

    return QuickTaskRun(
        task_name=task.name,
        task_shape=task.shape,
        is_canary=task.is_canary,
        dataset_revision=task.revision,
        n_items=len(items),
        n_shot=resolved_n_shot,
        seed=resolved_seed,
        max_tokens=max_tokens,
        power_limit_w=power_limit_w,
        inference_results=inference_results,
        telemetry_samples=telemetry_samples,
        streaming_used=streaming_used,
        # `bench.sampler.LocalNvmlSampler` has no local RAPL reader (see
        # module docstring), so these never resolve to a real value today --
        # read via getattr rather than a direct attribute access so a
        # duck-typed test double (or a future telemetry source that DOES
        # carry RAPL) doesn't need to define attributes it has no use for.
        rapl_max_energy_range_uj=getattr(telemetry, "rapl_max_energy_range_uj", None),
        rapl_dram_max_energy_range_uj=getattr(telemetry, "rapl_dram_max_energy_range_uj", None),
    )


@dataclass
class PowerSweepPoint:
    requested_w: int
    confirmed_w: int | None
    run: QuickTaskRun | None


def _derive_power_sweep_caps_w(
    stock_w: int,
    min_w: int | None,
    max_w: int | None,
) -> list[int]:
    """Turn `POWER_SWEEP_CAPS_W`'s fraction ladder into concrete watts for
    THIS card: `round(fraction * stock_w)`, clamped into `[min_w, max_w]`
    when BOTH are known (never guess an unknown range). A candidate that
    collides with an earlier (higher-fraction) one after clamping is
    dropped -- distinct points are what the flexibility metrics need, so
    three duplicates are worse than two honest ones.

    A candidate that is not BELOW `stock_w` is dropped for the same reason:
    on a card whose supported floor is its own stock limit (`min_w ==
    stock_w`, common on OEM/laptop parts), every fraction clamps back up to
    stock, and measuring stock again under the name "capped point" is a
    fabricated sweep point, not a measurement. The result can therefore be
    shorter than `POWER_SWEEP_CAPS_W`, and on such a card it is EMPTY --
    `run_power_sweep` turns that into a stated skip."""
    caps: list[int] = []
    for fraction in POWER_SWEEP_CAPS_W:
        watts = round(stock_w * fraction)
        if min_w is not None and max_w is not None:
            watts = max(min_w, min(max_w, watts))
        if watts >= stock_w or watts in caps:
            continue
        caps.append(watts)
    return caps


async def run_power_sweep(
    vllm_client: VLLMClient,
    telemetry: LocalNvmlSampler,
    model_name: str,
    *,
    n_items: int,
    n_shot: int | None = None,
    seed: int | None = None,
    caps_w: list[int] | None = None,
    max_points: int | None = None,
) -> tuple[list[PowerSweepPoint], str | None]:
    """Run the mini power sweep's capped points (stock is the baseline pass's
    own `POWER_SWEEP_TASK` run -- not repeated here).

    `caps_w` bypasses derivation with explicit absolute watts (an escape
    hatch for a caller that wants fixed points). Otherwise, concrete watts
    are derived from the card's OWN stock power limit
    (`telemetry.get_power_limit_w()`) and `POWER_SWEEP_CAPS_W`'s fraction
    ladder via `_derive_power_sweep_caps_w`, clamped into the card's
    supported range when `telemetry.get_power_limit_constraints_w()` knows
    it. When the stock limit can't be read at all, the sweep is SKIPPED
    outright -- never falls back to a fixed wattage ladder that might not
    even fit this card.

    `max_points` (US-MERGE-05, `bench calibrate`) caps how many of the
    derived (or explicit `caps_w`) points are actually MEASURED, applied
    AFTER the short-ladder message above is computed -- a caller that only
    wants the first point still sees an accurate "this card's own range
    collapsed the ladder" reason when that is genuinely why, and never a
    false one manufactured by its own request to stop early. `None` (the
    default) measures every derived/explicit point, unchanged from before
    this parameter existed.

    Stops at the FIRST cap that fails to confirm: if the first derived cap
    fails (the common case -- SetPowerManagementLimit needs root), the whole
    sweep is skipped and the second return value names why. A LATER point
    failing after earlier ones succeeded keeps what already ran (partial
    sweep beats none) rather than discarding it.

    Returns (points, reason). `reason` is None only when the derived ladder
    was FULL and at least its first rung was measured. It is set with NO
    points when the sweep could not start at all, and set ALONGSIDE points
    when the card's own supported range collapsed the ladder to fewer rungs
    than `POWER_SWEEP_CAPS_W` has: a sweep that quietly comes back with two
    points where the flexibility metrics need `MIN_SWEEP_POINTS` is exactly
    the "withhold, but say why" case. Callers distinguish the two cases by
    whether `points` is empty.
    """
    ladder_reason: str | None = None
    if caps_w is not None:
        caps = caps_w
    else:
        # The card's FACTORY DEFAULT, falling back to whatever it is set to
        # now only if NVML cannot report the default. Deriving fractions from
        # the CURRENT limit means a leftover cap from a hard-killed previous
        # run becomes the new baseline, and each aborted run walks the ladder
        # further down someone's card. See
        # `GpuSampler.get_power_limit_default_w`.
        stock_w = await telemetry.get_power_limit_default_w()
        if stock_w is None:
            stock_w = await telemetry.get_power_limit_w()
        if stock_w is None:
            return [], (
                "could not read the card's stock power limit -- skipped the "
                "mini power sweep entirely rather than fall back to a fixed "
                "wattage ladder that might not fit this card."
            )
        min_w, max_w = await telemetry.get_power_limit_constraints_w()
        caps = _derive_power_sweep_caps_w(stock_w, min_w, max_w)
        range_str = (
            f"{min_w}-{max_w} W" if min_w is not None and max_w is not None else "unknown"
        )
        if not caps:
            return [], (
                f"this card's supported power range ({range_str}) leaves no "
                f"point below its {stock_w} W stock limit -- skipped the mini "
                "power sweep rather than re-measure stock and label it a "
                "capped point."
            )
        if len(caps) < len(POWER_SWEEP_CAPS_W):
            ladder_reason = (
                f"only {len(caps)} of {len(POWER_SWEEP_CAPS_W)} derived caps "
                f"are distinct and below stock on this {stock_w} W card "
                f"(supported range {range_str}) -- measured those; with the "
                f"stock point that is {len(caps) + 1} of the "
                f"{MIN_SWEEP_POINTS} points the flexibility metrics need, so "
                "they stay withheld for this card."
            )

    if max_points is not None:
        caps = caps[:max_points]

    points: list[PowerSweepPoint] = []

    for i, watts in enumerate(caps):
        confirmed = await telemetry.set_power_limit_w(watts)
        if confirmed is None:
            if i == 0:
                return [], (
                    "NVML SetPowerManagementLimit requires elevated "
                    "privileges (root/CAP_SYS_ADMIN on most drivers) -- "
                    "skipped the mini power sweep entirely. Re-run as root "
                    "to include it."
                )
            break  # partial sweep: keep what already ran

        run = await run_quick_task(
            vllm_client,
            telemetry,
            model_name,
            POWER_SWEEP_TASK,
            n_items,
            power_limit_w=confirmed,
            n_shot=n_shot,
            seed=seed,
        )
        points.append(PowerSweepPoint(requested_w=watts, confirmed_w=confirmed, run=run))

    return points, ladder_reason


@dataclass
class QuickSuiteResult:
    """Everything the CLI layer needs after a successful `run_quick_suite`:
    computed `RunMetrics` ready to write/bundle, paired 1:1 (same order) with
    the raw `QuickTaskRun` each came from -- the scoped artifact writer
    (`bench/artifact.py`) needs the raw telemetry/inference-result lists a
    `RunMetrics` itself doesn't carry."""

    engine_name: str
    engine_base_url: str
    model: QuickModel
    gpu_info: dict[str, object]
    engine_version: str | None
    power_sweep_skipped_reason: str | None
    runs: list[RunMetrics] = field(default_factory=list)
    task_runs: list[QuickTaskRun] = field(default_factory=list)


def _build_run_metrics(
    task_run: QuickTaskRun,
    model: QuickModel,
    engine_name: str,
    engine_version: str | None,
    gpu_info: dict[str, object],
    target_host: str,
    label_prefix: str,
) -> RunMetrics | None:
    """Build one `RunMetrics` from a measured `QuickTaskRun` via the same
    `compute_metrics()` every other caller uses. No Home Assistant anywhere
    here (see module docstring): `kwh_before`/`kwh_after`/`ambient_c_start`
    are always None, which is exactly what leaves `measurement_tier` at 'C'.

    Returns None (never raises) when `compute_metrics()` itself fails (e.g.
    zero completion tokens) -- the caller skips it and keeps going, the same
    "partial suite beats none" posture the rest of this module takes on a
    failed task.
    """
    power_suffix = f"_{task_run.power_limit_w}w" if task_run.power_limit_w is not None else ""
    label = f"{label_prefix}_{task_run.task_name}{power_suffix}"
    run_id = generate_run_id(label)
    try:
        return compute_metrics(
            run_id=run_id,
            label=label,
            model=model.record_model,
            quantization=model.record_quantization,
            target_host=target_host,
            samples=task_run.telemetry_samples,
            inference_results=task_run.inference_results,
            kwh_before=None,
            kwh_after=None,
            ambient_c_start=None,
            rapl_max_energy_range_uj=task_run.rapl_max_energy_range_uj,
            rapl_dram_max_energy_range_uj=task_run.rapl_dram_max_energy_range_uj,
            task=task_run.task_name,
            task_shape=task_run.task_shape,
            is_canary=task_run.is_canary,
            gpu_mem_total_mib=gpu_info.get("gpu_mem_total_mib"),
            engine=engine_name,
            engine_version=engine_version,
            driver_version=gpu_info.get("driver_version"),
            cuda_version=gpu_info.get("cuda_version"),
            gpu_name=gpu_info.get("gpu_name"),
            power_limit_w=task_run.power_limit_w,
            temperature=0.0,
            max_tokens=task_run.max_tokens,
            seed=task_run.seed,
            n_shot=task_run.n_shot,
            dataset_revision=task_run.dataset_revision,
            streaming_used=task_run.streaming_used,
        )
    except MetricsComputeError as e:
        logger.warning("bench quick: failed to compute metrics for %s: %s", label, e)
        return None


async def _run_bench_suite(
    *,
    engine_choice: str | None,
    host: str,
    ollama_port: int,
    llamacpp_port: int,
    target_host: str | None,
    tasks: list[tuple[str, int]],
    max_sweep_points: int | None,
    label_prefix_stem: str,
    restore_to_factory_default: bool = True,
) -> QuickSuiteResult:
    """Shared orchestration behind `run_quick_suite` and `run_calibrate_suite`
    (US-MERGE-05): detect an engine, verify (never pull) the reference
    model, measure `tasks`, attempt the mini power sweep (at most
    `max_sweep_points` capped points when given), and return everything the
    CLI layer needs to write artifacts and build a submission bundle. Only
    WHAT gets measured differs between the two callers -- the detection,
    restore-in-finally, and error-handling contract below is identical for
    both, so it lives here once.

    Mirrors energy-bench's `main.py::execute_quick` loop, minus what does
    not apply to this package: no collector fallback (there is no
    `--collector-host` here, see `bench.sampler`'s docstring), no DuckDB
    insert (GROUND TRUTH: the run index stays lab-side), no
    `apply_model_meta` (this package's `RunMetrics` carries no
    model_type/params_b/reasoning_mode_class fields to populate -- see
    `bench.metrics.models`'s docstring), and no console progress printing
    (this function logs at INFO/WARNING; the CLI layer decides how much of
    that a human sees).

    The original power limit is ALWAYS restored (`finally`), even if a task
    raises or the power sweep aborts mid-way -- an unsustainable lock only
    throttles, but leaving the card capped after this function returns would
    silently bias every run that comes after it.

    Raises:
        NoEngineDetectedError: Neither Ollama nor llama-server answered.
        ModelNotAvailableError: The reference model isn't pulled (Ollama) or
            nothing is loaded at all (llama.cpp).
        NvmlUnavailableError: No local NVML-backed GPU -- this package's one
            hard hardware requirement for a bench run.
        AllTasksFailedError: Every task in `tasks` failed -- nothing was
            measured, so there is nothing to bundle.
    """
    resolved_target_host = target_host or host

    logger.info(
        "bench %s: detecting an inference engine (Ollama, then llama.cpp)...", label_prefix_stem
    )
    detected = await detect_engine(engine_choice, host, ollama_port, llamacpp_port)
    if detected is None:
        which = (
            f"'{engine_choice}'"
            if engine_choice and engine_choice != "auto"
            else "Ollama or llama.cpp"
        )
        raise NoEngineDetectedError(
            f"no {which} server answered on {host} "
            f"(tried ollama:{ollama_port}, llamacpp:{llamacpp_port}). Start "
            f"one first -- bench {label_prefix_stem} never launches an engine itself."
        )
    logger.info("  engine: %s at %s", detected.name, detected.base_url)

    model = await resolve_quick_model(detected)
    logger.info("  model: %s", model.note)

    telemetry = LocalNvmlSampler()
    split = urlsplit(detected.base_url)
    vllm_client = VLLMClient(host=split.hostname or host, port=split.port or 80)
    vllm_client.base_url = detected.base_url

    # Propagates NvmlUnavailableError uncaught -- no local NVML-backed GPU
    # means there is nothing this function can measure at all.
    gpu_info = await telemetry.gpu_info()
    engine_version = await detected.adapter.version()
    # BOTH restore candidates, captured BEFORE the sweep touches anything --
    # after it runs, "what the card is set to" is just the last cap applied.
    #
    #   factory default : where "managed" puts the card back, so a cap left by
    #                     an interrupted earlier run heals here.
    #   observed        : where "preserve" puts it back, so a cap the operator
    #                     set themselves survives the benchmark.
    #
    # The ladder itself always derives from the factory default regardless of
    # policy (see `_run_power_sweep`) -- fractions of a leftover cap would walk
    # the card down further on every aborted run.
    factory_power_limit = await telemetry.get_power_limit_default_w()
    observed_power_limit = await telemetry.get_power_limit_w()
    original_power_limit = factory_power_limit
    if original_power_limit is None:
        original_power_limit = observed_power_limit

    task_runs: list[QuickTaskRun] = []
    sweep_points: list[PowerSweepPoint] = []
    skipped_reason: str | None = None

    try:
        for i, (task_name, n_items) in enumerate(tasks, start=1):
            logger.info("[%d/%d] %s (n=%d)...", i, len(tasks), task_name, n_items)
            try:
                task_run = await run_quick_task(
                    vllm_client, telemetry, model.name, task_name, n_items
                )
            except Exception as e:  # noqa: BLE001 - one failed task must not sink the suite
                logger.warning("  %s failed: %s", task_name, e)
                continue
            task_runs.append(task_run)
            n_correct = sum(1 for r in task_run.inference_results if r.correct)
            logger.info("  done: %d/%d correct", n_correct, task_run.n_items)

        gsm8k_baseline = next(
            (
                r
                for r in task_runs
                if r.task_name == POWER_SWEEP_TASK and r.power_limit_w is None
            ),
            None,
        )
        if gsm8k_baseline is not None:
            logger.info(
                "mini power sweep: stock + up to %d capped point(s) derived from "
                "this card's own power limit (needs NVML SetPowerManagementLimit "
                "-- usually root)...",
                max_sweep_points if max_sweep_points is not None else len(POWER_SWEEP_CAPS_W),
            )
            try:
                sweep_points, skipped_reason = await run_power_sweep(
                    vllm_client,
                    telemetry,
                    model.name,
                    n_items=gsm8k_baseline.n_items,
                    n_shot=gsm8k_baseline.n_shot,
                    seed=gsm8k_baseline.seed,
                    max_points=max_sweep_points,
                )
            except Exception as e:  # noqa: BLE001 - a sweep failure must not sink the suite
                skipped_reason = f"power sweep aborted: {e}"
            # A reason WITH points is a collapsed ladder, not a skipped
            # sweep: report what was measured, then why it is short.
            if skipped_reason and not sweep_points:
                logger.info("  skipped: %s", skipped_reason)
            else:
                logger.info("  %d capped point(s) measured.", len(sweep_points))
                if skipped_reason:
                    logger.info("  short ladder: %s", skipped_reason)
        else:
            skipped_reason = (
                "gsm8k_platinum baseline run failed -- the mini power sweep "
                "needs it as the stock (uncapped) point."
            )
            logger.info("mini power sweep: skipped (%s)", skipped_reason)
    finally:
        # `restore_to_factory_default` mirrors the controller's
        # POWER_CAP_POLICY (config.Settings): True = "managed", the card goes
        # back to the limit the driver shipped, so a cap left behind by an
        # interrupted run heals here. False = "preserve", a cap the operator
        # set themselves survives the benchmark -- at the cost that a leftover
        # one survives too, because the two are indistinguishable from here.
        # Both candidates were captured before the sweep ran; re-reading the
        # card here would return the last cap applied, not its start state.
        restore_target = original_power_limit
        if not restore_to_factory_default and observed_power_limit is not None:
            restore_target = observed_power_limit
        if restore_target is not None:
            await telemetry.set_power_limit_w(restore_target)
        telemetry.close()

    if not task_runs:
        raise AllTasksFailedError("every task failed -- nothing was measured.")

    label_prefix = f"{label_prefix_stem}_{detected.name.replace('.', '')}"
    all_task_runs = list(task_runs) + [p.run for p in sweep_points if p.run is not None]

    runs: list[RunMetrics] = []
    paired_task_runs: list[QuickTaskRun] = []
    for task_run in all_task_runs:
        run_metrics = _build_run_metrics(
            task_run,
            model,
            detected.name,
            engine_version,
            gpu_info,
            resolved_target_host,
            label_prefix,
        )
        if run_metrics is None:
            continue
        runs.append(run_metrics)
        paired_task_runs.append(task_run)

    if not runs:
        raise AllTasksFailedError(
            "every measured task's metrics failed to compute -- nothing was persisted."
        )

    return QuickSuiteResult(
        engine_name=detected.name,
        engine_base_url=detected.base_url,
        model=model,
        gpu_info=gpu_info,
        engine_version=engine_version,
        power_sweep_skipped_reason=skipped_reason,
        runs=runs,
        task_runs=paired_task_runs,
    )


async def run_quick_suite(
    *,
    engine_choice: str | None = None,
    host: str = "localhost",
    ollama_port: int = DEFAULT_OLLAMA_PORT,
    llamacpp_port: int = DEFAULT_LLAMACPP_PORT,
    target_host: str | None = None,
    restore_to_factory_default: bool = True,
) -> QuickSuiteResult:
    """Run the ~25-minute onboarding suite end to end, in-process: the full
    `QUICK_TASKS` set plus the mini power sweep's full derived ladder. The
    leaderboard-grade, publishable measurement -- see `_run_bench_suite` for
    the shared behavior contract (restore-in-finally, exceptions raised).
    """
    return await _run_bench_suite(
        engine_choice=engine_choice,
        host=host,
        ollama_port=ollama_port,
        llamacpp_port=llamacpp_port,
        target_host=target_host,
        tasks=QUICK_TASKS,
        max_sweep_points=None,
        label_prefix_stem="quick",
        restore_to_factory_default=restore_to_factory_default,
    )


CALIBRATE_TASK_N_ITEMS = 15
"""Item count for calibrate's one task -- the spec's '~15 items'."""

CALIBRATE_TASKS: list[tuple[str, int]] = [(POWER_SWEEP_TASK, CALIBRATE_TASK_N_ITEMS)]
"""`bench calibrate` measures exactly one decode task -- the same
`gsm8k_platinum` `run_quick_suite`'s own mini sweep uses as its stock
(uncapped) point, so this one task IS the sweep's baseline pass, not a
second measurement of the same thing."""

CALIBRATE_MAX_SWEEP_POINTS = 1
"""`bench calibrate` keeps only the FIRST derived capped point -- enough for
the predictor/cap-recommendation figures the spec asks for, without the
~25-minute cost of `run_quick_suite`'s full three-point ladder."""


async def run_calibrate_suite(
    *,
    engine_choice: str | None = None,
    host: str = "localhost",
    ollama_port: int = DEFAULT_OLLAMA_PORT,
    llamacpp_port: int = DEFAULT_LLAMACPP_PORT,
    target_host: str | None = None,
    restore_to_factory_default: bool = True,
) -> QuickSuiteResult:
    """Run the ~3-5 minute slimmed scheduling probe (US-MERGE-05): one
    decode task (`CALIBRATE_TASKS`, ~15 items) plus the mini power sweep's
    first capped point only -- not the full `QUICK_TASKS`/three-point ladder
    `run_quick_suite` measures. Same restore-in-finally guarantee and raised
    exceptions as `run_quick_suite`; see `_run_bench_suite` for the shared
    contract -- only WHAT gets measured differs.

    `bench calibrate` is NOT the leaderboard-grade path: it exists so a box
    that only needs the per-machine figures the predictor and cap
    recommendation want does not have to sit through the full suite.
    `bench quick` remains the comparable, publishable measurement -- the
    CLI layer stamps `suite: "calibrate"` on the bundle this produces so a
    server-side reader never pools it with `quick` results.
    """
    return await _run_bench_suite(
        engine_choice=engine_choice,
        host=host,
        ollama_port=ollama_port,
        llamacpp_port=llamacpp_port,
        target_host=target_host,
        tasks=CALIBRATE_TASKS,
        max_sweep_points=CALIBRATE_MAX_SWEEP_POINTS,
        label_prefix_stem="calibrate",
        restore_to_factory_default=restore_to_factory_default,
    )
