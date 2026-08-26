# async-energy-controller

**Run your AI work when electricity is cheap.**

This is the on-box controller for [Async Energy](https://async.energy). It pulls
a nightly schedule from the optimizer, runs each of your jobs inside its assigned
window under a GPU energy profiler, and reports back what the run actually cost
in watt-hours.

It is deliberately small and deliberately dumb: it never optimizes, never
decides, and talks to exactly one host. All the intelligence — price curves,
energy prediction, placement — lives server-side. This package is the part that
runs on your hardware, which is exactly why it is the part you can read.

**The same install also carries a GPU energy benchmark suite.** `bench quick`
measures what your hardware costs per unit of useful work — joules per correct
answer and per token, across a set of standard tasks — against whichever
inference server you already run. Use it on its own to compare models,
quantizations and power caps on your box, with or without an Async Energy
account. Details under [Optimize + contribute](#optimize--contribute).

```
                        ┌─────────────────────────────┐
   api.async.energy ───▶│  schedule: run wf_a1b2 at   │
     (the optimizer)    │  02:00–04:00, deadline 07:00│
                        └──────────────┬──────────────┘
                                       │
                            ┌──────────▼──────────┐
                            │  this controller    │   ← you are here
                            └──────────┬──────────┘
                                       │  looks up wf_a1b2 in YOUR jobs.json
                            ┌──────────▼──────────┐
                            │  your job, profiled │
                            └──────────┬──────────┘
                                       │  duration · energy_wh · power profile
                        ┌──────────────▼──────────────┐
   api.async.energy ◀───│  run record (tomorrow's     │
                        │  plan gets better)          │
                        └─────────────────────────────┘
```

---

## What it will not do

Worth stating before you give anything a systemd unit:

- **It will never run a command the server sent it.** Scheduled work is
  identified by a workflow id. The controller looks that id up in *your* local
  `jobs.json` and runs what *you* wrote there. An id with no local entry is
  skipped with a warning. There is no code path that executes a server-supplied
  command — see [`executor.py`](hmasync_controller/executor.py).
- **It will not invent measurements.** If the box cannot measure energy, the
  field is `null` and every average that would have used it skips it. It is
  never backfilled with a plausible guess.
- **It will not phone anywhere else.** One host, the one you configure in
  `.env`. No telemetry vendors, no analytics.
- **It will not run without a GPU driver — it just measures less.** No NVML
  means no energy numbers; scheduling still works on duration alone.
- **It will not leave your GPU capped after a benchmark.** The suite lowers the
  board power limit while it measures, then restores the card's **factory
  default** — not whatever the limit happened to be when it started, which
  would silently re-apply a cap left behind by an earlier interrupted run. A
  cap only throttles and never damages, but a card left quietly throttled is
  its own kind of harm. Which limit counts as "as we found it" is yours to
  set — see [`POWER_CAP_POLICY`](#who-owns-your-gpus-power-limit--power_cap_policy)
  and [Hardware safety](HARDWARE-SAFETY.md).

---

## Install

Requires **Python 3.12+** and Linux.

```bash
git clone https://github.com/boringbots/async-energy-controller.git
cd async-energy-controller
python -m venv .venv && source .venv/bin/activate

pip install -e .   # installs and runs fine with no GPU driver — energy stays null
```

This puts `async-energy-controller` on your PATH (and `hm-async-controller` as a
permanent alias, so older installs and unit files keep working). The base
install above is also everything the benchmark suite needs — `bench quick` and
`bench calibrate` (below, under "Optimize + contribute") run with no extra
flag and no second package to install; there is no separate `energy-bench`
download.

## Configure

You need an Async Energy account — [sign up and register a workload
first](https://async.energy/quickstart/), it takes about five minutes and is all
`curl`.

Create a `.env` next to where you will run the controller:

```bash
cp .env.example .env
```

```ini
HM_ASYNC_API_URL=https://api.async.energy
HM_ASYNC_EMAIL=you@example.com
HM_ASYNC_PASSWORD=your-password
CONTROLLER_ID=workstation-1      # optional; defaults to the hostname
HM_ASYNC_JOB_CATALOG=jobs.json   # optional; where the job catalog lives
```

`CONTROLLER_ID` is half of the `(controller_id, run_id)` idempotency key the
server uses, so keep it stable for a given box.

Every knob works from this file. A real exported environment variable overrides
it, and a command-line flag overrides both.

## The job catalog — `jobs.json`

**This is the trust boundary.** It maps the workflow ids you registered with the
API to what this particular machine actually executes.

The short way — this creates the workflow *and* writes the entry, so the id is
never copy-pasted:

```bash
async-energy-controller register \
  --name fed-sentiment \
  --command 'docker exec my-container python job.py --years 1' \
  --recurrence daily --earliest-start 20:00 --deadline 'by 7am' \
  --nameplate-watts 250 --est-duration 700
```

Or start from the example and edit by hand:

```bash
cp jobs.example.json jobs.json
```

```json
{
  "2bac93f6-8d99-43ce-93ad-b64a8b0e603c": {
    "name": "nightly-embeddings",
    "framework": "ollama",
    "request": {
      "model": "nomic-embed-text",
      "prompt_file": "/data/queue/embed-batch.txt"
    }
  }
}
```

A missing catalog is a warning, not an error — the controller still pulls
schedules and drains its spool, it just skips workflows it has no local
definition for. Override the path with `--job-catalog PATH`, or
`HM_ASYNC_JOB_CATALOG` in `.env` or the environment.

**Comments are safe to keep.** The loader keeps only object-valued entries, so a
string or array value is dropped cleanly — `"_comment": ["...", "..."]` as a
multi-line header block works, and JSON gives you no other way to annotate a
file. `name` is likewise ignored by the executor and shown by `--check`.

### The three frameworks

| `framework` | What it does | `request` keys |
|---|---|---|
| `command` | Runs any shell command. This is how you schedule agent pipelines, robot/device charging, or anything scriptable. | `command`, `cwd`, `env`, `timeout` |
| `ollama` | Blocking call to a local Ollama server. | `model`, `prompt` or `prompt_file`, `options`, `timeout`, … |
| `openai` | Any OpenAI-compatible server — vLLM, LM Studio, llama.cpp. | `model`, `messages`/`prompt`/`prompt_file`, `base_url`, `timeout`, … |

`command` takes either form. Prefer the list once any argument contains a space
or a quote — it sidesteps shell quoting entirely, where the string form is
`shlex`-split first:

```json
"command": "python /opt/agents/nightly.py"
"command": ["docker", "exec", "my-container", "python", "job.py", "--years", "1"]
```

Adding a framework is one registered class in
[`adapters.py`](hmasync_controller/adapters.py) — it needs `run()`,
`fingerprint()`, and `preflight()`.

### Jobs are bounded by their window

A job that declares no `timeout` gets one from the time left in its placement
window. This matters more here than in a general job runner: an unbounded job
does not merely delay its successors, it runs out of the cheap window and into
peak pricing — the outcome the schedule existed to avoid — and you are billed for
it while the run record still shows a correctly-planned placement. Overrunning
produces a failed run the optimizer replans around.

Set `"timeout": 3600` in a request to override it. An explicit value is always
honored, including one that deliberately exceeds the window.

## Run it

```bash
async-energy-controller --check                # validate the whole setup, exit non-zero if wrong
async-energy-controller --poll-interval 30     # the run loop
```

| Flag | Default | Meaning |
|---|---|---|
| `--check` | off | Validate catalog, login, schedule, and the match between them, then exit. **Use this to verify a setup.** |
| `--once` | off | Run a single executor tick and exit. Good for cron. |
| `--poll-interval SECONDS` | `30` | Seconds between schedule polls in the run loop. |
| `--job-catalog PATH` | `$HM_ASYNC_JOB_CATALOG` or `jobs.json` | Where the local job catalog lives. |
| `--watch-catalog` | off | Re-read the catalog when it changes on disk instead of loading it once at start-up. |
| `--log-level LEVEL` | `INFO` | Standard Python logging levels. |

### Verifying a setup — `--check`

`--once` runs a tick, but its output cannot answer the question a smoke test is
asked: `outcomes=0` reads the same whether the catalog is empty, a workflow id is
mistyped, or it is simply 16:00 and the window opens at 20:00. `--check` resolves
every layer and names what is wrong:

```
catalog      /home/me/async-energy/jobs.json  (2 jobs)
auth         ok — me@example.com, controller_id=workstation
schedule     version 4, 2 placements, valid until 2026-08-12T20:10Z
matched      29812d27 stonks-fed-update      00:10–00:12  8.3 Wh
matched      45bd1534 stonks-fed-extend      00:12–00:23  48.6 Wh

ok — catalog, auth, and schedule all resolve.
```

The two set differences are the valuable part:

- **`unmatched`** — scheduled here, but absent from your catalog. The optimizer
  plans it and this box silently skips it. Always reported as a problem, and the
  exit code is non-zero.
- **`orphaned`** — in your catalog but not in this schedule. Usually benign (a
  weekly job, or one not yet planned), occasionally a mistyped or deleted id.
  Shown, never fatal.

The run loop reports the same counts on every tick, so a healthy idle daemon is
distinguishable from a misconfigured one at a glance:

```
tick: version=4 mode=normal reachable=True catalog=2 placements=2 pending=2 next=00:10 outcomes=0 drained=0
```

An empty catalog is re-warned periodically rather than only at startup — that one
line scrolls out of `journalctl -n 20` within minutes.

### Registering a workload — `register`

Creating a workflow otherwise means `curl` against `/api/v1/workflows` and then
copying a UUID into `jobs.json` by hand: the most error-prone step in the setup,
and the one where a typo fails silently. `register` does both halves at once, so
the catalog and the server cannot drift apart at creation time.

```bash
async-energy-controller register \
  --name stonks-fed-extend \
  --command 'docker exec stonks-dashboard python fed_sentiment.py extend --years 1' \
  --recurrence daily --earliest-start 20:00 --deadline 'by 7am' \
  --nameplate-watts 250 --est-duration 700
```

`--dry-run` prints the catalog entry it would write without calling the API or
touching the file. Existing entries — comment keys included — are preserved, the
write is atomic, and a catalog that exists but does not parse is never
overwritten.

Three things worth knowing:

- **`--deadline` / `--earliest-start` take human strings** — `by 7am`, `22:00`,
  `2026-08-20T09:00` — read in your account's timezone. They are validated when
  you register, so a typo is a clear error now rather than a job that quietly
  never fits. `--recurrence` is `none`, `daily`, or `weekly`.
- **They are not written into `jobs.json`.** The catalog's optional `deadline` is
  a local fallback hint that must be a tz-aware ISO datetime; `"by 7am"` parses to
  nothing there. The schedule carries the resolved window.
- **Your command is not uploaded.** What this box runs is described by your
  local `jobs.json` and stays there; `register` sends the scheduling constraints
  and nothing else. Pass `--share-request` if you would rather your account also
  hold a copy of the request payload.

### Registering workloads while it runs

By default the catalog is read once at start-up, so a workflow added afterwards
needs a restart before this box will run it. If something else is registering
workloads — a script, a pipeline, an agent — start with `--watch-catalog` and a
new `jobs.json` entry is picked up on the next lookup:

```bash
async-energy-controller --poll-interval 30 --watch-catalog
```

The trust boundary is unchanged: it still only ever runs what your local file
says. If the file goes missing or you save a syntax error mid-edit, it logs a
warning and **keeps the last good catalog** rather than silently stopping every
job. Emptying the file deliberately (`{}`) is honored — that is a real intention,
not a broken read.

### As a service

Two shapes ship, and the choice is usually decided by group membership.

**A user unit** — when the job needs *your* login user's environment. A job that
shells out to `docker exec` needs your `docker` group; putting a system account in
that group is a larger grant than it looks, since docker group membership is
effectively root on the host.

```bash
sudo loginctl enable-linger $USER          # <-- do not skip this
mkdir -p ~/.config/systemd/user
cp deploy/hm-async-controller.user.service ~/.config/systemd/user/
$EDITOR ~/.config/systemd/user/hm-async-controller.user.service   # check paths
systemctl --user daemon-reload
systemctl --user enable --now hm-async-controller
journalctl --user -u hm-async-controller -f
```

**`enable-linger` is not optional.** Without it systemd stops your user manager at
logout and takes the daemon with it. A job placed at 00:10 then never fires, and
nothing appears in any log to explain it — the controller was not running to say
anything. For a tool whose whole point is running work overnight while nobody is
watching, this is the first thing to rule out. Verify with
`loginctl show-user $USER --property=Linger`.

`--watch-catalog` pairs well with a user unit: edit `jobs.json`, or let `register`
write to it, and the new entry runs without a restart.

**A system unit** — right on a shared box, or when the work runs as a service
account anyway:

```bash
sudo cp deploy/hm-async-controller.service /etc/systemd/system/
sudoedit /etc/systemd/system/hm-async-controller.service   # check User= and paths
sudo systemctl daemon-reload
sudo systemctl enable --now hm-async-controller
journalctl -u hm-async-controller -f
```

Both unit files document their layout. Whichever you install, run `--check` once
afterwards before trusting it overnight.

---

## Or skip the daemon entirely

The daemon above is the right shape when **nothing else is awake** at 3am — it
is the thing holding the clock on a box you own. If your work is already driven
by something else (an Airflow DAG, a CI job, a cron entry, an agent assembling a
pipeline), a second scheduler is redundant and you may not even be able to
install a systemd unit.

For that case the same package is a library. Ask when to run, then stay in
charge of your own process:

```python
from hmasync_controller.sdk import AsyncEnergy

ae = AsyncEnergy(api_key="...")          # or HM_ASYNC_API_KEY in the environment

window = ae.next_window(est_duration_s=1800, deadline="by 7am")
print(window.start, window.grid_cost, "vs", window.now_cost, "now")

window.wait()                            # sleeps until the cheap hour
with ae.measure(fingerprint="nightly-embeddings") as run:
    run["work_units"] = do_the_work()    # your code, your process
    run["work_unit_kind"] = "tokens"
# the run reports itself on exit, energy included where the box can measure it
```

`next_window` is a single stateless call to `POST /api/v1/advise`. It registers
nothing and stores nothing — ask as often as you like. It answers against the
same price curve, optimizer, and cost model the nightly planner uses, so its
answer and a real scheduled placement agree by construction.

Points worth knowing:

- **No inbound trust boundary.** Nothing calls into your machine; your code
  calls out. The `jobs.json` mechanism exists because the daemon inverts that
  direction — here it simply does not apply.
- **`feasible: False` is an answer, not an error.** `window.reason` says what to
  change. Calling `.wait()` on an infeasible window raises rather than sleeping
  forever.
- **Reporting is best effort.** A failed telemetry push logs a warning; it never
  turns a job that succeeded into one that failed.
- **An exception inside `measure()` is recorded as a failed run and re-raised.**
  Your error handling is not swallowed to make a report tidy.
- **Use an API key, not your password.** Mint one in the dashboard. It is scoped
  to this caller and revocable without changing your account password.

What you give up versus the daemon: nothing retries your job if it fails, and
nothing runs it if your process is not alive. Pick the shape that matches who is
holding the clock.

## Optimize + contribute

Everything above runs on the price curve alone. There is a second, entirely
optional loop: contribute measured benchmark data, and the optimizer starts
giving *your* box better answers — a cold-start estimate for a workflow it has
never seen run here yet, and (if you opt into it separately) a power cap tuned
to your actual hardware instead of its nameplate rating.

It is off by default and stays off until you type `bench opt-in`:

```bash
async-energy-controller bench opt-in
```

That command prints the full consent text before it writes anything, and there
is no interactive `y/n` — safe to read on a box with no TTY before deciding
whether to run it for real. What it shares, verbatim:

```
Opting in shares, per benchmark submission:
  - a hardware fingerprint: GPU model name, VRAM (GB), driver version, CPU
    model, and RAM (GB)
  - software versions: this controller (the bench suite runs inside it)
  - benchmark metrics: energy (Wh), duration, throughput, and related numbers
    produced by the suite

It never shares prompts, commands, workflow definitions, or any workflow data.
It never shares GPU UUIDs, serial numbers, MAC addresses, hostnames, or Home
Assistant entity ids — this box is identified only by a salted local hash,
generated once and never transmitted in raw form.

Data license: submitted results feed Async Energy's shared routing-table and
cold-start-prediction aggregates. Opting out stops future submissions; it does
not withdraw data already submitted.
```

`bench opt-out` reverts it at any time — nothing further is sent, though data
already submitted is not withdrawn.

### Running the suite — `bench quick`

```bash
async-energy-controller bench quick
```

This runs the onboarding suite in-process — no separate `energy-bench` install,
no subprocess — against whichever inference server you already have running
(Ollama, then llama.cpp). Unmeasured here, but the suite's own name for it is
the ~25-minute tier: **Tier C, no smart plug required**, using GPU-reported
power instead of a wall meter. If opted in, the bundle it writes is submitted
automatically once the run finishes; if not, you get a bundle file on disk and
nothing leaves the box. `bench submit <bundle.json>` sends one by hand, e.g. a
bundle from an earlier run that was never opted in at the time.

A submission is validated and checked against the redaction denylist locally
before anything is sent, and refused rather than sent if either check fails. If
the API is unreachable when a submission is ready, it spools to disk and goes
out on the controller's normal reconnect cadence — the same offline-safe
pattern used for run reports.

### A faster option — `bench calibrate`

```bash
async-energy-controller bench calibrate
```

A ~3-5 minute slimmed probe: one decode task (~15 items) plus the mini power
sweep's first capped point, instead of `bench quick`'s full ~25-minute suite.
Enough to give the predictor and cap-recommendation the per-machine figures
they need, without the full run. Its bundle is stamped `suite: "calibrate"`
and never pooled with `bench quick` results — **`bench quick` remains the
leaderboard-grade, publishable measurement**; use `calibrate` when you only
need scheduling figures fast.

### What you get back

- **Cold-start priors.** `register --bench-gpu-class rtx4090
  --bench-model-size-class 7b --bench-quant int4` (all optional) tags a
  workflow with a hardware/model class so the optimizer can estimate its energy
  cost from the shared benchmark pool before this box has ever run it, instead
  of guessing from nameplate wattage alone.
- **A tuned power cap.** Set `APPLY_POWER_CAP=true` in `.env` (a separate
  opt-in from `bench opt-in` — one shares data, the other acts on a
  recommendation) and, once this node has a recommendation on file, the
  controller applies it via NVML around each scheduled GPU job and restores it
  afterward, logging what it did. No recommendation yet, or no permission to
  set one, is never an error — the job runs uncapped.

### Who owns your GPU's power limit — `POWER_CAP_POLICY`

Both the scheduler and the benchmark suite lower the board power limit and put
it back. What "back" means is your call:

```bash
POWER_CAP_POLICY=preserve   # default
POWER_CAP_POLICY=managed
```

**`preserve`** puts back the exact limit that was there. A cap you set
yourself survives every job and every benchmark. The cost: if a job is ever
killed outright — reboot, OOM-kill, `kill -9` — before it can restore, its cap
is left behind, and the next job reads that leftover cap as the value to put
back. Your card stays throttled until someone notices. The controller warns on
every job when it sees a limit below the card's factory default, naming both
numbers and the `nvidia-smi -pl` recovery, but it will not undo it for you.

**`managed`** puts back the card's factory default instead. A leftover cap
heals itself on the next job, and a benchmark leaves the card exactly as the
driver shipped it. The cost: a cap *you* set persistently gets reset to
factory, because from the controller's side a deliberate cap and a leftover
one look identical.

Pick `preserve` if you tune your own card — for noise, heat, or a shared power
budget. Pick `managed` if you don't, and would rather the controller keep the
card clean for benchmarking and optimization. The default is `preserve`
because silently undoing a setting you chose is a worse failure than leaving a
visible, warned-about one in place.

## How energy gets measured

The profiler samples the GPU at 1 Hz for the length of every run and records
**where the energy number came from**, so you always know how much to trust it:

| `energy_source` | Meaning |
|---|---|
| `counter` | Read from the GPU's cumulative energy counter via NVML. Exact. |
| `integrated` | No counter available; power samples integrated over the run. Good, but a spike between two samples is invisible. |
| `null` | The box could not measure it. Not estimated, not backfilled. |

Backends are tried in order: **NVML** → **`nvidia-smi`** → **null**. The null
profiler is always valid, so the controller runs anywhere. On Intel and AMD CPUs
the package RAPL counter is also sampled where readable (it is often
root-only on recent kernels — harmless when it is not).

Where a framework reports how much work it did — completion tokens, typically —
that is captured as `work_units` alongside the energy.

## When your internet drops

Run records spool to a local SQLite file and drain when the connection returns;
nothing is lost. If the API is unreachable the controller keeps following the
last schedule it holds, and past that schedule's `valid_until` it applies the
`fallback_policy` embedded in it. A failed job triggers a replan server-side so
its deadline can still be met.

---

## Layout

```
hmasync_controller/
├── cli.py         # entrypoint — wires client + spool + executor
├── config.py      # settings (every value empty-default) + controller id resolution
├── apiclient.py   # the four wire endpoints + auth; returns clean errors, never raises
├── executor.py    # the run loop; degrades explicitly when the API is down
├── adapters.py    # framework seam — command / ollama / openai
├── profiler.py    # telemetry seam — NVML → nvidia-smi → null; RAPL where readable
├── reporter.py    # report live / spool on outage / drain on reconnect
├── spool.py       # single-file SQLite store-and-forward, restart-durable
└── sdk.py         # the library face — next_window() + measure(), no daemon
```

`profiler.py`, `apiclient.py`, `adapters.py`, and `spool.py` import nothing from
each other, so they are usable on their own if you want to build something
different from either shape shipped here.

**The running daemon speaks exactly four endpoints**: `POST /runs`,
`POST /runs/{id}/samples`, `GET /schedule?after=<v>`, `POST /schedule/ack`. That
is the whole execution loop, and it is what the trust boundary rests on — none of
them can deliver a command to run. Schedules are versioned and immutable; a
replan publishes a new version rather than mutating one you already hold.

Two more routes exist and are reached only by an explicit call you make:
`POST /advise` (the SDK's `next_window`) and `POST /workflows` (`register`).
Neither runs in the loop.

## Development

```bash
pip install -e ".[test]"
python -m pytest tests/
```

The suite needs **no GPU, no network, and no account** — `httpx`, `subprocess`,
NVML, and the RAPL sysfs reads are all mocked.

## Troubleshooting

Run `async-energy-controller --check` first — it answers most of the table below
by name.

| Symptom | Cause / fix |
|---|---|
| The window passed and nothing ran | Run `--check`. Usually the controller was not running, or the workflow has no `jobs.json` entry (it shows as `unmatched`). |
| Nothing runs, and a user unit is installed | `loginctl enable-linger $USER`. Without it your user manager stops at logout and takes the daemon with it, silently. |
| `catalog=0` in the tick line | The catalog file was not found. Check `--job-catalog` / `HM_ASYNC_JOB_CATALOG`; `--check` prints the exact path it resolved. |
| A job runs but reports suspiciously little work | For `ollama`/`openai`, check `prompt_file` points at a file that exists — an unreadable one now fails the run rather than sending an empty prompt. |
| A job died with `command timed out` | It exceeded the time left in its placement window. Set an explicit `"timeout"` in its request if it genuinely needs longer, or widen the window. |
| `energy_wh` is null on every run | No NVIDIA driver on the box, so NVML has nothing to read. Scheduling still works on duration alone. |
| `cpu_rapl_uj` is null in traces | RAPL is root-readable-only on recent kernels. Harmless; GPU energy still measures. |
| Login fails at startup | Check `HM_ASYNC_API_URL`, and that the account confirmed its email. |
| Schedule says `degraded: true` | Server-side: prices were stale when the plan was made. The plan still respects every deadline. |
| A job shows `feasible: false` | It cannot fit between its earliest start and its deadline. Widen the window or split the work — it will not be run late. |

## Links

- **[async.energy](https://async.energy)** — what this is and why
- **[Quickstart](https://async.energy/quickstart/)** — account → workload → this controller
- **[How it works](https://async.energy/how-it-works/)** — the cost model and the optimizer
- **[Hardware safety](HARDWARE-SAFETY.md)** — every call that touches your GPU, its bounds, and its restore path
- **[Issues](https://github.com/boringbots/async-energy-controller/issues)** — bugs and questions

## License

MIT — see [LICENSE](LICENSE).

Powered by Hungry Machines Energy.
