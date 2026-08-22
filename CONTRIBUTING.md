# Contributing

Bug reports and adapter contributions are welcome. The controller is
intentionally small — that is a feature, and changes that grow it need a reason.

This repository is the canonical controller source. There is no other
controller tree to sync against; all controller work happens here.

## Ground rules

**The controller never optimizes.** It executes a schedule and reports what
happened. Anything that decides *when* or *whether* work runs belongs in the
optimizer, not here. A PR that adds scheduling logic to this package will be
asked to move it.

**Never fabricate a measurement.** If the hardware cannot report something, the
field is `null`. Do not substitute an estimate, a default, or a zero — every
average downstream filters nulls out on purpose, and a plausible-looking wrong
number is worse than a missing one.

**The job catalog is a trust boundary.** The controller runs what the local
`jobs.json` says and nothing else. Any change that would let a server response
influence *what command executes* is out of scope regardless of how it is
guarded.

**Everything is mocked in tests.** No test may require a GPU, a network, an
account, or a live API. `httpx`, `subprocess`, NVML, and the RAPL sysfs reads
are all mocked — keep it that way so the suite runs anywhere.

## Development setup

```bash
git clone https://github.com/boringbots/async-energy-controller.git
cd async-energy-controller
python -m venv.venv && source.venv/bin/activate
pip install -e ".[test]"

python -m pytest tests/                    # full suite
python -m pytest tests/test_adapters.py -v # one file, while iterating
```

## Adding a framework adapter

The most useful contribution. One class in `hmasync_controller/adapters.py`
implementing three methods:

| Method | Contract |
|---|---|
| `run(request) -> AdapterRunResult` | **Blocking** — the profiler wraps this call, so it must not return before the work is done. |
| `fingerprint(request) -> (hash, features)` | Two jobs that cost the same must hash the same. Exclude volatile inputs (seeds, timestamps, request ids) or prediction breaks. |
| `preflight(request=None) -> bool` | Is the framework reachable and the model loaded right now? |

Register it in `_REGISTRY`, add aliases to `_ALIASES` if the name has common
variants, and add tests covering the fingerprint stability (same logical request
→ same hash) and the failure path.

## Pull requests

- One concern per PR.
- Tests for anything behavioral; the suite must pass on 3.12 and 3.13.
- Match the surrounding style — the code is heavily commented with *why*, not
  *what*. Keep that.

For anything touching the wire contract with the optimizer, open an issue first
— that surface is shared with the server and changing it is a coordinated
release, not a merge.
