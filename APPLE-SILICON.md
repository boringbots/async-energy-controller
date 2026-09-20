# Benchmarking on Apple Silicon

`bench quick` runs on an M-series Mac. No sudo, no extra install step, no
graphics card.

```bash
pip install async-energy-controller
ollama serve &                      # or llama-server, if you prefer
ollama pull qwen3.5:9b-q4_K_M       # the fixed reference model, ~6 GB
async-energy-controller bench quick
```

That is the whole setup. The energy backend comes with the package on an
arm64 Mac and nowhere else, so there is still exactly one thing to install.

### How much memory you need

The reference model is fixed across every submission -- that is what makes
them comparable -- and it is a **Q4_K_M quantization of a 9B**, so budget
roughly 6 GB for weights plus whatever the KV cache grows to.

**16 GB or more is comfortable. On an 8 GB Mac, expect the machine to swap.**
That matters more here than it would elsewhere: swapping inflates both
wall-clock and energy, and the harness has no way to notice it, so the run
will look valid and simply read high. If you only have 8 GB, a number from
your machine is not wrong so much as unattributable -- it is measuring
macOS's memory pressure as much as the model.

The reference model is deliberately NOT swapped for a smaller one on
low-memory machines. A suite that measures a different model on different
hardware produces rows that cannot be compared, which defeats the point of
having a reference at all.

---

## What you get, and what you do not

Read this part before you compare a Mac number to anything.

| | Apple Silicon | NVIDIA |
|---|---|---|
| GPU energy | yes, IOReport counter | yes, NVML counter |
| CPU package energy | **yes** | no (RAPL is not read) |
| DRAM energy | **yes** | no |
| `energy_source` | `ioreport` | `counter` |
| GPU temperature | no | yes |
| GPU utilisation | no | yes |
| VRAM used | **there is none** | yes |
| Clocks, fan, perf state | no | yes |
| Thermal throttle detection | **no** | yes |
| Power-limit sweep | no | yes, with root |

A Mac actually reports *more* of the energy story than an NVIDIA box does
here — CPU and DRAM are measured, and on the NVIDIA path they are not. What
it reports less of is everything else.

### There is no VRAM, so no VRAM number is recorded

Memory is unified. `gpu_mem_used_mib` is `None` on every Mac run, and that is
deliberate: any figure put there would be a system-memory number wearing a
card-capacity label. Whether a model fits on your Mac is a question about
total RAM, which the machine fingerprint already records.

### Joules do not compare across vendors

`energy_source` reads `ioreport` on a Mac and `counter` on an NVIDIA box.
Those are two different vendors' models of their own silicon, and the project
already has a measured finding (F5) that **two identical RTX 3090s disagree
by 22% on the same work**. Across architectures and instruments it will be
worse, and nothing in this package can correct for it.

So:

- **Accuracy comparisons across machines are fine.** A correct answer is
  correct anywhere; accuracy is hardware-independent.
- **Joule comparisons across machines are not.** Compare Mac to Mac.
- Anything grouping by `energy_source` will keep the two apart on its own.

Upstream is also explicit that IOReport's energy values are believed to be
model-based estimates rather than readings off a shunt. NVML's are too, to a
degree. Neither is a wall meter.

### Thermal throttling is invisible here, and this is the one that bites

The safety circuit-breaker in `bench.thermal` trips on NVML's hardware
thermal-throttle bit. Apple exposes no unprivileged equivalent, so on a Mac:

- the circuit-breaker **never fires**,
- `thermal_throttle_pct` stays `None`,
- and a throttled run completes looking perfectly healthy.

On a fanless MacBook Air under a sustained benchmark, this is not
hypothetical. **If you care about the numbers, run on mains power, on a hard
surface, and prefer a short suite.** If a run's tokens/sec sags badly partway
through, suspect heat — the harness will not tell you, because inferring
throttling from a falling token rate is a guess, and this package does not
present guesses as measurements.

### Nothing touches your hardware

There is no power-limit control on Apple Silicon, so the mini power sweep is
skipped with a stated reason rather than attempted. `bench quick` on a Mac
only reads counters and sends HTTP requests to a server you started. See
[HARDWARE-SAFETY.md](HARDWARE-SAFETY.md).

---

## How it works

IOReport is the interface `powermetrics` itself reads. It is a private
Apple framework with no published documentation, which is part of why we do
not bind to it directly. We use it through
[`zeus-apple-silicon`](https://github.com/ml-energy/zeus-apple-silicon)
(Apache-2.0, from the ml-energy group), not by hand-rolling `ctypes` bindings
into a private Apple framework.

**Why not `powermetrics`?** It needs root. Asking someone to `sudo` a
freshly-installed Python package to measure their own laptop is a bad trade,
and it is a subprocess emitting text on its own cadence, so the sampler would
be parsing someone else's timing instead of controlling its own. IOReport is
counter-based — energy over a window is the difference of two counter reads —
which is the same shape as the NVML counter this project already prefers over
integrating a power trace.

The sampler runs overlapping windows **back to back**: the next opens
microseconds after the last closes, rather than around the 200 ms sleep, so
the only energy lost is the cost of two Python calls. GPU energy is the GPU
core plus its SRAM; unified DRAM is shared with the CPU and cannot be
attributed to the GPU, so it is reported on its own channel instead.

Implementation: [`hmasync_controller/bench/apple_sampler.py`](hmasync_controller/bench/apple_sampler.py).

---

## Which engine

Both supported engines run natively with Metal, and `bench quick` attaches
over HTTP — it never launches one for you.

- **Ollama** — `ollama serve`, then `ollama pull <model>`. Detected first.
- **llama.cpp** — `llama-server -m model.gguf`. Detected second.

MLX is not supported yet. `mlx_lm.server` speaks the OpenAI API, so the gap
is engine detection rather than anything deep, but nothing here has been
tested against it and it is not claimed.

Intel Macs are not supported. `is_apple_silicon()` gates on arm64, and an
Intel Mac falls through to the NVML path and fails there. Its power route
would be Intel RAPL, which this sampler does not implement.

---

## Troubleshooting

**`zeus-apple-silicon is not importable`** — the backend did not install.
`pip install zeus-apple-silicon`, then check you are on the Python you think
you are: `python -c "import zeus_apple_silicon"`.

**`AppleSiliconSampler needs an arm64 Mac`** — you are on an Intel Mac, or on
a Python running under Rosetta. Check `python -c "import platform;
print(platform.machine())"`; it must print `arm64`, not `x86_64`.

**`IOReport would not start on this Mac`** — no sudo is required, so this is
not a permissions problem. Most likely an unsupported chip or macOS build.
Upstream has tested M1 Max, M3 Pro, M4, M4 Pro and M5 Max.

**`no Ollama or llama.cpp server answered`** — start one first. `bench quick`
never launches an engine itself.

**Energy columns are null but the run succeeded** — a channel your chip does
not report is recorded as `None` rather than zero. Check `gpu_energy_channels`
in the run's GPU info: `gpu+sram` on newer silicon, `gpu` where the SRAM
channel is absent.

---

## Status: not yet verified on hardware

Every test in `tests/test_bench_apple_sampler.py` (30 of them) drives a stub
standing in for IOReport. They pin the parts that are ours — counter
arithmetic, window discipline, which channels are withheld, the platform gate
— and they cannot prove the numbers are right on a real Mac, because there is
no Apple Silicon in the lab this was written in.

**If you are the first to run it, these are the things to check:**

1. `bench quick` completes and writes a bundle.
2. `energy_source` reads `ioreport`.
3. GPU joules are in a plausible range — a few hundred to a few thousand J
   for the quick suite, not 0 and not 10⁹.
4. `total_joules_cpu` and `total_joules_cpu_dram` are populated and smaller
   than the GPU figure.
5. Sanity-check the total against `sudo powermetrics --samplers cpu_power,gpu_power`
   watched alongside a run. They will not agree exactly — different windows,
   different aggregation — but they should agree to within a few percent.
6. Run it twice. Energy should repeat to roughly 1%; if the second run is
   much cheaper, the first was probably heating the machine up.

Two upstream issues to know about, both recent and neither reproduced here:
a stalled CPU energy channel on a macOS 27 beta, and alternating
zero/inflated readings on M5 Max tied to IOReport's batch cadence. If GPU
power alternates between 0 W and an implausible spike, that is the second one
and it is not your machine.
