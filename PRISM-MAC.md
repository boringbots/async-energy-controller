# The prism wave on a Mac

Bonsai's sub-4-bit weights, metered on Apple Silicon. Five rungs, two
questions, one machine.

```bash
# in the venv this package is installed in
scripts/run-prism-wave-mac.sh list      # the plan, and what it will download
scripts/run-prism-wave-mac.sh fetch     # llama-server + 23 GB of weights
scripts/run-prism-wave-mac.sh run       # the wave, rung by rung
python3 scripts/compare-prism-rungs.py  # what it found
```

Everything below explains why the wave is shaped this way. Read the last
section before you compare a number from it to anything.

---

## The two questions

**Does the sub-2-bit reversal survive a change of architecture?** The lab
measured one 27B ternary checkpoint in two storages on an RTX 3090: PTQ1_0 at
5.95 GB against PQ2_0 at 7.21 GB. Accuracy was indistinguishable, and the
*narrower* file cost **1.20 to 1.57 times more energy** — unpacking it costs
more arithmetic than the memory traffic it saves. That was Ampere, twice. A
Mac balances arithmetic against memory bandwidth quite differently, so if the
reversal is a property of the format it should survive the move, and if it is
a property of one card's memory system it should not.

**Is a quantization really only a storage change?** `prism-ml` publishes one
4B ternary checkpoint as three files — F16 (8.05 GB), PQ2_0 (1.07 GB) and
Q2_0_g64 (1.14 GB). Ternary weights times an FP16 group scale are exactly
representable in FP16, so all three should answer every question identically,
and whatever energy separates them is bytes and kernels and nothing else.
That is a control the rest of the corpus cannot build: an ordinary Int4 rung
has both fewer bytes *and* different weights than its FP16 rung.

## What runs

| rung | stage | quant | GB | file |
|---|---|---|---|---|
| `bonsai2-27b-ptq1-0` | reversal | PTQ1_0 | 5.95 | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` |
| `bonsai2-27b-pq2-0` | reversal | PQ2_0 | 7.21 | `Ternary-Bonsai-2-27B-PQ2_0.gguf` |
| `bonsai-4b-f16` | format-control | F16 | 8.05 | `Ternary-Bonsai-4B-F16.gguf` |
| `bonsai-4b-pq2-0` | format-control | PQ2_0 | 1.07 | `Ternary-Bonsai-4B-PQ2_0.gguf` |
| `bonsai-4b-q2-0-g64` | format-control | Q2_0_g64 | 1.14 | `Ternary-Bonsai-4B-Q2_0_g64.gguf` |

Each rung runs `gsm8k_platinum` ×50 and `mmlu_redux` ×100 at 5-shot, seed
1234, temperature 0, thinking off, in a 16,384-token window, at stock power.
One decode-heavy shape and one prefill-heavy one, which is what an energy
comparison between storages needs.

The 4B family rather than the 8B one the lab used: the 8B's F16 control rung
is 16.38 GB, and on 24 GB of unified memory with a 16k window that is close
enough to the ceiling to swap. Swapping inflates wall-clock and joules, the
harness cannot see it, and it would land on exactly the rung that is supposed
to be the baseline.

**150 items settles two questions and not a third.** The reversal is an energy
question at matched accuracy, and energy repeats to about 1% here. The format
control is an identity question, and it is checked per item — same `item_id`,
same `correct`, same `completion_tokens` — rather than by comparing two
accuracy figures. What 150 items cannot do is resolve an accuracy difference:
that interval is ±0.10 at best. Nothing from this wave is evidence about
whether Bonsai is more or less accurate than any other model.

## Time and memory

Budget **6 to 8 hours** for the wave on an M3, plus the downloads. That is
scaled from the only Apple Silicon timings this project has — an M3 running
the quick suite at ~39 s/item on `gsm8k_platinum` and ~8 s/item on
`mmlu_redux` for a 9B Q4_K_M — not measured on these models. A rung's backstop
is 2 hours, tripled automatically on Apple Silicon, and `--timeout` raises it.

Weights: 23.4 GB on disk. Resident: the largest rung is the 8.05 GB F16
control, and the 27B ternary rungs sit near 7-9 GB with their window. **16 GB
is enough; 24 GB is comfortable.** On 8 GB the machine will swap and the run
will read high while looking valid.

**Run it plugged in, on a hard surface, and preferably not on a fanless Air.**
The thermal circuit-breaker trips on NVML's hardware-throttle bit, and Apple
exposes no unprivileged equivalent, so on a Mac it never fires:
`thermal_throttle_pct` stays null and a throttled rung finishes looking
perfectly healthy. The driver cools down for 3 minutes between rungs, which is
the only defence the wave has. If tokens/sec sags badly partway through a
rung, suspect heat — nothing in the harness will tell you.

## What stops a wrong number

Every rung is verified against `GET /props` before anything is measured, and
refused rather than measured if it does not match:

1. **The file**, by name. The F16 rung is the dangerous one — it is an
   ordinary GGUF that any llama-server, or LM Studio, would serve happily, and
   the resulting row would be clean, plausible and wrong. The wave also
   defaults to **port 8091, not 8080**, so it cannot attach to whatever else
   is already serving on this machine.
2. **The packing**, by the loader's own words (`PQ2_0 - 2.13 bpw (group
   128)`). This catches the legacy-file trap: `prism-ml`'s model cards
   recommend a bare `*-Q2_0.gguf`, which stores group-128 weights under the
   group-64 id. That file now *loads*, naming itself `(group 128, legacy
   ftype)` — so it would quietly occupy the Q2_0_g64 rung and make the format
   control compare a file with itself. `fetch` never downloads it and the
   check refuses it by name.
3. **The window**. llama.cpp's auto-fit shrinks the context rather than
   failing when a model does not fit, down to 4096. The driver passes
   `--fit off`; if a rung then refuses to load, that refusal is the
   measurement — record it and move on.

**Thinking is off on every rung, by two different mechanisms.** Reading the
GGUFs directly: the 27B template honours `enable_thinking` and thinks by
default, so the pin this mode sends is load-bearing there; the 4B template has
no such variable and hardcodes an empty `<think>` block, so the pin is inert
and the template holds the axis. Both end up off. The run's `thinking_mode`
field records what was *sent*, so each summary also records which mechanism
was actually holding it — and the comparison script shouts if a template does
neither.

## The engine

PrismML's fork of llama.cpp, release **`prism-b10709-9a9394a`**, prebuilt for
macos-arm64 (11.5 MB, no toolchain needed). `fetch` downloads it. Its Metal
backend implements both private ggml types — `kernel_mul_mv_pq2_0_f32`,
`kernel_mul_mv_ptq1_0_f32`, the matrix-multiply templates and both
dequantizers — and the fork's README calls PQ2_0 "preferred on Metal, CUDA,
HIP and CPU".

That release is 22 commits ahead of the `5d80cff` the lab's NVIDIA rows were
measured on, and none of those commits touches `ggml/` — they are
speculative-decoding work. The quantization kernels are the same code. Each
row records the build the server reports in `engine_version`.

Mainline llama.cpp cannot serve four of these five files at all.

## What comes out, and what it may be compared to

Per rung: an artifact folder per measured task under `BENCH_DATA_DIR`
(`telemetry.parquet`, `items.parquet`) and a summary JSON under
`BENCH_DATA_DIR/prism/`. `scripts/compare-prism-rungs.py` reads both and
prints the two comparisons.

**Nothing is submitted, by design.** The public submission contract's `suite`
field enumerates quick/calibrate/medium/full/reference, and the leaderboard's
config key is model|quant|engine|gpu|power|task|thinking|cap — it has no field
for *which build* of llama.cpp served a row. Publishing these rows would pool
a fork's numbers into "llama.cpp", which is the same mistake the wave's own
port interlock exists to prevent. `bench prism` writes locally and stops.

Two more limits on where these numbers travel:

- **Joules do not cross vendors.** A Mac row carries
  `energy_source='ioreport'`; an NVIDIA row carries `counter`. Two identical
  RTX 3090s already disagree by 22% on the same work — across vendors and
  instruments it is worse, and nothing here corrects for it. Every comparison
  this wave makes is between rungs on one machine, measured back to back,
  which is the only shape where that does not matter.
- **Accuracy does cross machines**, and is the right thing to compare against
  the lab's rows. A correct answer is correct anywhere.

## If something goes wrong

| what you see | what it means |
|---|---|
| `no llama-server answered on localhost:8091` | the rung's server died at load; read `prism-mac/logs/<rung>-server.log`. For a rung that does not fit, that *is* the result. |
| `is serving 'X', but rung 'Y' is 'Z'` | something else is on the port, or the driver was pointed at the wrong file |
| `the DEPRECATED group-128-stored-as-id-42 packing` | a bare `*-Q2_0.gguf` got downloaded by hand; use the pinned filenames |
| `granted a 4096-token window` | auto-fit shrank the context — the model did not fit; `--fit off` is already passed, so this rung needs a smaller model or a bigger Mac |
| `did not finish within Ns` | raise it: `async-energy-controller bench prism --rung K --timeout 21600` |
| the comparison script warns about a template | that rung may have been reasoning while its row says otherwise; read its summary's `thinking.note` before using its energy |
