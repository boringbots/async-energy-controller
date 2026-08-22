"""
hm-async controller — the thin on-box client.

The controller runs on your AI machine. It executes the schedule the optimizer
API hands it, profiles each run's GPU energy, and reports run records upstream.
It never optimizes and it never decides — and it talks to exactly one host, the
one you configure.

  - config.py    — env-driven settings (boots clean with an empty .env)
  - apiclient.py — ApiClient: the four wire endpoints + auth, clean-error returns
  - spool.py     — Spool: a single-file SQLite store-and-forward buffer
  - reporter.py  — RunReporter: ties client + spool (report live, spool on outage,
                   drain on reconnect)
  - profiler.py  — the telemetry seam (NVML → nvidia-smi → null)
  - adapters.py  — the framework seam (command / ollama / openai)
  - executor.py  — the run loop; degrades explicitly when the API is down
  - sdk.py       — the library face: next_window() + measure(), no daemon

The profiler, framework adapters, and executor loop plug into these seams
without changing them.
"""

__all__ = ["__version__"]

__version__ = "0.3.0"
