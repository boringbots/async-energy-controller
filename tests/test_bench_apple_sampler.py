"""AppleSiliconSampler, exercised against a fake IOReport monitor.

NONE OF THIS RAN ON A MAC. There is no Apple Silicon in the lab, so every
test below drives `AppleSiliconSampler` with a stub standing in for
`zeus_apple_silicon.AppleEnergyMonitor`. That is enough to pin the parts that
are ours -- the counter arithmetic, the window discipline, which channels are
withheld, and the platform gate -- and not enough to prove the numbers are
right on real hardware. The one thing only a Mac can answer is whether
IOReport's channels report what upstream says they report; `APPLE-SILICON.md`
carries the checklist for that.

The arithmetic is worth pinning precisely because it is silently wrong when
it breaks: a `gpu_energy_mj` that is per-tick rather than cumulative still
produces a plausible-looking run, and `compute_counter_energy` reads its
first decrease as a counter reset and discards the energy for the whole run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from hmasync_controller.bench.apple_sampler import (
    AppleEnergyUnavailableError,
    AppleSiliconSampler,
    is_apple_silicon,
)
from hmasync_controller.bench.metrics.compute import (
    compute_counter_energy,
    compute_cpu_dram_energy,
    compute_cpu_energy,
)
from hmasync_controller.bench.metrics.compute import InferenceResult
from hmasync_controller.bench.sampler import GpuSampler, select_gpu_sampler


@dataclass
class FakeMetrics:
    """One `AppleEnergyMetrics`, with the field names upstream documents."""

    gpu_mj: float | None = 10.0
    gpu_sram_mj: float | None = 2.0
    cpu_total_mj: float | None = 50.0
    dram_mj: float | None = 5.0
    ane_mj: float | None = 0.0


class FakeMonitor:
    """Stand-in for `zeus_apple_silicon.AppleEnergyMonitor`.

    Records the label sequence so a test can prove windows are opened and
    closed in matched pairs -- a leaked window means IOReport keeps a
    subscription alive for a run that ended.
    """

    def __init__(self, metrics: list[FakeMetrics] | None = None) -> None:
        self.opened: list[str] = []
        self.closed: list[str] = []
        self._queue = list(metrics or [])
        self.live: set[str] = set()

    def begin_window(self, label: str) -> None:
        assert label not in self.live, f"window {label} opened twice"
        self.live.add(label)
        self.opened.append(label)

    def end_window(self, label: str) -> FakeMetrics:
        assert label in self.live, f"window {label} closed without being open"
        self.live.remove(label)
        self.closed.append(label)
        return self._queue.pop(0) if self._queue else FakeMetrics()


def _wired(sampler: AppleSiliconSampler, monitor: FakeMonitor) -> AppleSiliconSampler:
    """Attach a fake monitor, bypassing the platform gate."""
    sampler._monitor = monitor
    sampler._open_window()
    return sampler


class TestPlatformGate:
    def test_refuses_to_start_off_apple_silicon(self):
        """The gate is the hardware, not the import.

        On this Linux box `_ensure_monitor` must refuse before it ever
        reaches for `zeus_apple_silicon`, and say what box it actually found.
        """
        sampler = AppleSiliconSampler()
        with pytest.raises(AppleEnergyUnavailableError) as exc:
            sampler._ensure_monitor()
        assert "arm64 Mac" in str(exc.value)

    def test_is_apple_silicon_is_false_here(self):
        assert is_apple_silicon() is False

    def test_selector_picks_nvml_off_a_mac(self):
        """A non-Mac box must still get the NVML sampler and its error."""
        assert type(select_gpu_sampler()).__name__ == "LocalNvmlSampler"

    def test_selector_branches_on_hardware_not_on_the_backend_import(self, monkeypatch):
        """An arm64 Mac gets the Apple sampler even with no backend installed.

        Otherwise a Mac user missing `zeus-apple-silicon` is told they have
        "no local NVIDIA GPU" and goes looking for a graphics card.
        """
        monkeypatch.setattr(
            "hmasync_controller.bench.apple_sampler.is_apple_silicon", lambda: True
        )
        assert type(select_gpu_sampler()).__name__ == "AppleSiliconSampler"

    def test_satisfies_the_sampler_protocol(self):
        assert isinstance(AppleSiliconSampler(), GpuSampler)

    def test_energy_source_is_distinct_from_nvml(self):
        """The fence that stops a Mac row pooling with an NVIDIA one."""
        from hmasync_controller.bench.sampler import LocalNvmlSampler

        assert AppleSiliconSampler().energy_source == "ioreport"
        assert LocalNvmlSampler().energy_source == "counter"
        assert AppleSiliconSampler().energy_source != LocalNvmlSampler().energy_source


class TestEnergySourceReachesTheRun:
    """The attribute existing is not the same as the row carrying it.

    Caught 2026-09-19 after the first commit: `compute_metrics` hardcoded
    `"counter" if counter_joules is not None else "integrated"`, so a Mac run
    would have been stamped `counter` and pooled straight into the NVIDIA
    population. The sampler's `energy_source` was correct and reached
    nothing. Tests that only assert the attribute cannot see that, so these
    assert the END of the chain.
    """

    def _metrics(self, counter_source: str):
        import time

        from hmasync_controller.bench.metrics.compute import compute_metrics
        from hmasync_controller.bench.sampler import TelemetrySample

        t0 = time.time()
        samples = [
            TelemetrySample(
                ts=t0 + i * 0.2,
                gpu_power_w=50.0,
                gpu_util_pct=None,
                gpu_mem_used_mib=None,
                gpu_temp_c=None,
                gpu_energy_mj=1000.0 * (i + 1),
            )
            for i in range(5)
        ]
        results = [
            InferenceResult(
                request_id="r0",
                prompt_tokens=10,
                completion_tokens=20,
                ttft_s=0.1,
                total_s=1.0,
                tokens_per_second=20.0,
                t_start_s=t0,
                t_end_s=t0 + 0.8,
            )
        ]
        return compute_metrics(
            run_id="r",
            label="l",
            model="m",
            quantization=None,
            target_host="localhost",
            samples=samples,
            inference_results=results,
            kwh_before=None,
            kwh_after=None,
            ambient_c_start=None,
            counter_source=counter_source,
        )

    def test_ioreport_survives_into_the_row(self):
        assert self._metrics("ioreport").energy_source == "ioreport"

    def test_nvidia_callers_are_unchanged_by_default(self):
        assert self._metrics("counter").energy_source == "counter"

    def test_quick_passes_the_source_it_read_from_gpu_info(self):
        """The link in the middle of the chain.

        `gpu_info()` reports it and `compute_metrics` accepts it; this is the
        one line that actually joins them, and losing it would put every run
        back to a hardcoded 'counter' with no test failing.
        """
        import inspect

        from hmasync_controller.bench import quick

        src = inspect.getsource(quick)
        assert 'counter_source=gpu_info.get("energy_source"' in src

    def test_a_mac_shaped_sample_does_not_crash_the_health_rollup(self):
        """Every GPU channel but power is None on Apple Silicon.

        `compute_hardware_health` promised in its docstring to degrade rather
        than raise, and did not: temperature, VRAM and utilization went
        straight into `max()`/`sum()`. NVML always answers those three, so it
        took a Mac to find it -- `bench quick` completed the run and then
        died computing its metrics.
        """
        from hmasync_controller.bench.metrics.compute import compute_hardware_health

        monitor = FakeMonitor()
        sampler = _wired(AppleSiliconSampler(), monitor)
        samples = [sampler.sample() for _ in range(3)]

        health = compute_hardware_health(samples, gpu_mem_total_mib=None)
        for field in (
            "peak_gpu_temp_c",
            "mean_gpu_temp_c",
            "peak_gpu_mem_used_mib",
            "mean_gpu_mem_used_mib",
            "gpu_mem_used_pct_of_total",
            "mean_gpu_util_pct",
            "thermal_throttle_pct",
        ):
            assert health[field] is None, f"{field} should be withheld, got {health[field]}"


class TestCounterArithmetic:
    def test_gpu_energy_is_cumulative_not_per_tick(self):
        """`compute_counter_energy` diffs last-minus-first.

        A per-tick value would make the series non-monotonic the moment one
        tick is cheaper than the last, which that function reads as a counter
        reset and answers with None -- discarding the run's energy silently.
        """
        monitor = FakeMonitor([FakeMetrics(gpu_mj=10.0, gpu_sram_mj=2.0)] * 4)
        sampler = _wired(AppleSiliconSampler(), monitor)

        samples = [sampler.sample() for _ in range(4)]
        readings = [s.gpu_energy_mj for s in samples]

        assert readings == [12.0, 24.0, 36.0, 48.0]
        assert all(b >= a for a, b in zip(readings, readings[1:]))

    def test_a_cheaper_tick_never_moves_the_counter_backwards(self):
        """The failure mode above, with a genuinely varying workload."""
        monitor = FakeMonitor(
            [
                FakeMetrics(gpu_mj=100.0, gpu_sram_mj=0.0),
                FakeMetrics(gpu_mj=1.0, gpu_sram_mj=0.0),
                FakeMetrics(gpu_mj=50.0, gpu_sram_mj=0.0),
            ]
        )
        sampler = _wired(AppleSiliconSampler(), monitor)
        samples = [sampler.sample() for _ in range(3)]

        assert [s.gpu_energy_mj for s in samples] == [100.0, 101.0, 151.0]
        energy = compute_counter_energy(samples, 0.0)
        # last - first = 151 - 100 = 51 mJ = 0.051 J
        assert energy["total_joules_gpu_counter"] == pytest.approx(0.051)

    def test_gpu_energy_is_core_plus_sram(self):
        monitor = FakeMonitor([FakeMetrics(gpu_mj=7.0, gpu_sram_mj=3.0)])
        sampler = _wired(AppleSiliconSampler(), monitor)
        assert sampler.sample().gpu_energy_mj == 10.0

    def test_a_missing_sram_channel_is_not_read_as_zero(self):
        """Older silicon reports no SRAM channel.

        The GPU-core figure stands alone; `gpu_info()` then says which shape
        produced the number, so a reader can tell the two apart later instead
        of assuming every row means the same thing.
        """
        monitor = FakeMonitor([FakeMetrics(gpu_mj=7.0, gpu_sram_mj=None)])
        sampler = _wired(AppleSiliconSampler(), monitor)
        sample = sampler.sample()
        assert sample.gpu_energy_mj == 7.0
        assert sampler._sram_seen is False

    def test_a_missing_gpu_channel_withholds_rather_than_zeroes(self):
        monitor = FakeMonitor([FakeMetrics(gpu_mj=None, gpu_sram_mj=None)])
        sampler = _wired(AppleSiliconSampler(), monitor)
        sample = sampler.sample()
        assert sample.gpu_energy_mj is None
        assert sample.gpu_power_w is None

    def test_cpu_and_dram_energy_reach_the_metrics_layer(self):
        """An NVIDIA run leaves these empty; a Mac run fills them.

        The values are cumulative microjoules, so the RAPL wrap correction --
        which only fires on a NEGATIVE step -- never triggers, and no
        `max_energy_range_uj` is needed to get a total.
        """
        monitor = FakeMonitor([FakeMetrics(cpu_total_mj=1000.0, dram_mj=200.0)] * 3)
        sampler = _wired(AppleSiliconSampler(), monitor)
        samples = [sampler.sample() for _ in range(3)]

        assert [s.cpu_rapl_uj for s in samples] == [1e6, 2e6, 3e6]
        # 3e6 - 1e6 = 2e6 uJ = 2 J
        assert compute_cpu_energy(samples, None) == pytest.approx(2.0)
        assert compute_cpu_dram_energy(samples, None) == pytest.approx(0.4)

    def test_power_is_energy_over_the_windows_own_duration(self, monkeypatch):
        """Not over the nominal tick: a late wake-up must not inflate watts."""
        # Three reads, not two: _open_window (t0), sample's close (t1), then
        # sample re-opens the next window (t2).
        clock = iter([100.0, 100.5, 100.5])  # a 0.5 s window, not a 0.2 s one
        monkeypatch.setattr(
            "hmasync_controller.bench.apple_sampler.time.monotonic",
            lambda: next(clock),
        )
        monitor = FakeMonitor([FakeMetrics(gpu_mj=1000.0, gpu_sram_mj=0.0)])
        sampler = AppleSiliconSampler()
        sampler._monitor = monitor
        sampler._open_window()
        # 1000 mJ = 1 J over 0.5 s = 2 W (not the 5 W a 0.2 s assumption gives)
        assert sampler.sample().gpu_power_w == pytest.approx(2.0)


class TestWithheldChannels:
    @pytest.mark.parametrize(
        "field",
        [
            "gpu_util_pct",
            "gpu_mem_used_mib",
            "gpu_temp_c",
            "gpu_mem_util_pct",
            "gpu_throttle_reasons",
            "gpu_sm_clock_mhz",
            "gpu_mem_clock_mhz",
            "gpu_fan_pct",
            "gpu_perf_state",
        ],
    )
    def test_unavailable_channels_are_none_never_zero(self, field):
        """Rule 3. A zero here reads downstream as a measurement.

        `gpu_mem_used_mib` is the one worth naming: there is no VRAM on a
        unified-memory machine, so any number at all would be an invention.
        """
        monitor = FakeMonitor()
        sampler = _wired(AppleSiliconSampler(), monitor)
        assert getattr(sampler.sample(), field) is None

    def test_thermal_guard_finds_no_throttle_signal(self):
        """Documents the known gap rather than pretending it is covered.

        `consecutive_hw_thermal_seconds` treats a None mask as "not
        throttled", so the circuit-breaker cannot fire on Apple Silicon. On a
        fanless laptop that is a real limitation, stated in the module
        docstring and in APPLE-SILICON.md.
        """
        from hmasync_controller.bench.thermal import consecutive_hw_thermal_seconds

        monitor = FakeMonitor()
        sampler = _wired(AppleSiliconSampler(), monitor)
        samples = [sampler.sample() for _ in range(5)]
        assert consecutive_hw_thermal_seconds(samples) == 0.0


class TestWindowDiscipline:
    def test_windows_are_opened_and_closed_in_matched_pairs(self):
        monitor = FakeMonitor()
        sampler = _wired(AppleSiliconSampler(), monitor)
        for _ in range(3):
            sampler.sample()
        # One window stays open for the next tick; the rest are closed.
        assert len(monitor.opened) == len(monitor.closed) + 1
        assert monitor.closed == monitor.opened[:-1]

    def test_each_window_gets_a_distinct_label(self):
        """Upstream allows overlapping windows only with distinct labels."""
        monitor = FakeMonitor()
        sampler = _wired(AppleSiliconSampler(), monitor)
        for _ in range(4):
            sampler.sample()
        assert len(set(monitor.opened)) == len(monitor.opened)

    def test_stop_closes_the_window_the_last_tick_left_open(self):
        async def go():
            monitor = FakeMonitor()
            sampler = AppleSiliconSampler(sample_hz=50)
            sampler._monitor = monitor
            sampler._samples = []
            sampler._sampling = True
            sampler._open_window()
            sampler._task = asyncio.create_task(sampler._loop())
            await asyncio.sleep(0.1)
            await sampler.stop()
            return monitor

        monitor = asyncio.run(go())
        assert monitor.live == set(), "stop() leaked an open IOReport window"

    def test_start_resets_the_counters_between_runs(self):
        """Run two must not inherit run one's cumulative total."""

        async def go():
            monitor = FakeMonitor()
            sampler = AppleSiliconSampler(sample_hz=50)
            sampler._monitor = monitor

            async def one_run():
                sampler._samples = []
                sampler._cum_gpu_mj = 999.0  # residue from an earlier run
                await sampler.start()
                await asyncio.sleep(0.08)
                return await sampler.stop()

            first = await one_run()
            return first

        samples = asyncio.run(go())
        assert samples, "expected at least one tick"
        assert samples[0].gpu_energy_mj == 12.0, "start() did not zero the counter"


class TestGpuInfo:
    def test_reports_no_vram_and_names_the_instrument(self, monkeypatch):
        monkeypatch.setattr(
            "hmasync_controller.bench.apple_sampler.is_apple_silicon", lambda: True
        )
        monkeypatch.setattr(
            "hmasync_controller.bench.apple_sampler._chip_name", lambda: "Apple M3 Pro"
        )
        sampler = AppleSiliconSampler()
        sampler._monitor = FakeMonitor()
        info = asyncio.run(sampler.gpu_info())

        assert info["gpu_name"] == "Apple M3 Pro"
        assert info["gpu_mem_total_mib"] is None  # unified memory: no VRAM figure
        assert info["cuda_version"] is None
        assert info["energy_source"] == "ioreport"


class TestPowerLimits:
    def test_every_power_limit_call_declines(self):
        """Apple exposes no analogue, so the quick suite skips its sweep."""
        sampler = AppleSiliconSampler()
        assert asyncio.run(sampler.get_power_limit_w()) is None
        assert asyncio.run(sampler.get_power_limit_default_w()) is None
        assert asyncio.run(sampler.get_power_limit_constraints_w()) == (None, None)

    def test_setting_a_limit_fails_soft_rather_than_raising(self):
        """Same contract as NVML without privilege: None, never an exception."""
        assert asyncio.run(AppleSiliconSampler().set_power_limit_w(150)) is None


class TestPmsetReadings:
    THERM = (
        "Note: No thermal warning level has been recorded\n"
        "Note: No performance warning level has been recorded\n"
        "CPU_Scheduler_Limit \t= 100\n"
        "CPU_Available_CPUs \t= 8\n"
        "CPU_Speed_Limit \t= 62\n"
    )

    def test_cpu_speed_limit_is_parsed(self):
        from hmasync_controller.bench.apple_sampler import parse_cpu_speed_limit_pct

        assert parse_cpu_speed_limit_pct(self.THERM) == 62
        assert parse_cpu_speed_limit_pct("garbage") is None
        assert parse_cpu_speed_limit_pct(None) is None

    def test_power_source_is_parsed(self):
        from hmasync_controller.bench.apple_sampler import parse_power_source

        assert parse_power_source("Now drawing from 'AC Power'\n -InternalBattery-0\t100%; charged") == "ac"
        assert parse_power_source("Now drawing from 'Battery Power'\n -InternalBattery-0\t61%") == "battery"
        assert parse_power_source("") is None

    def test_the_reading_is_stamped_on_every_tick_and_polled_slowly(self):
        """A subprocess at 5 Hz would cost more than it tells; the sampler
        re-reads every THERM_POLL_S and repeats the last value between."""
        from hmasync_controller.bench import apple_sampler as mod

        monitor = FakeMonitor()
        sampler = _wired(AppleSiliconSampler(), monitor)
        calls = []

        def reader():
            calls.append(1)
            return 55

        sampler._therm_reader = reader
        first, second = sampler.sample(), sampler.sample()
        assert first.cpu_speed_limit_pct == 55 and second.cpu_speed_limit_pct == 55
        assert len(calls) == 1, "second tick inside THERM_POLL_S must reuse the reading"
        assert mod.THERM_POLL_S >= 1.0

    def test_a_throttled_mac_now_reaches_the_health_rollup(self):
        from hmasync_controller.bench.metrics.compute import compute_hardware_health

        sampler = _wired(AppleSiliconSampler(), FakeMonitor())
        sampler._therm_reader = lambda: 40
        samples = [sampler.sample() for _ in range(3)]
        assert compute_hardware_health(samples, gpu_mem_total_mib=None)["thermal_throttle_pct"] == 100.0

    def test_gpu_info_carries_the_power_source(self, monkeypatch):
        from hmasync_controller.bench import apple_sampler as mod

        monkeypatch.setattr(mod, "read_power_source", lambda: "battery")
        sampler = _wired(AppleSiliconSampler(), FakeMonitor())
        info = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(sampler.gpu_info())
        assert info["power_source"] == "battery"

