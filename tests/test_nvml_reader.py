"""Tests for hmasync_controller.nvml_reader (US-MERGE-02): the single
per-tick NVML register-read implementation shared by `profiler.NVMLProfiler`
and `bench.sampler.LocalNvmlSampler`. Mocked pynvml throughout — no live GPU.

Also the grep gate proving neither caller re-implements the read sequence:
`test_profiler_sample_loop_delegates_to_shared_reader` and
`test_bench_sampler_sample_delegates_to_shared_reader` inspect the actual
source of each caller's per-tick method and assert the raw `nvmlDeviceGet*`
calls appear nowhere but here.
"""

from __future__ import annotations

import inspect

from hmasync_controller.bench.sampler import LocalNvmlSampler
from hmasync_controller.nvml_reader import read_energy_counter_mj, read_nvml_channels
from hmasync_controller.profiler import NVMLProfiler

# The raw per-tick NVML calls that must live in nvml_reader.py alone.
_NVML_READ_CALLS = (
    "nvmlDeviceGetPowerUsage",
    "nvmlDeviceGetUtilizationRates",
    "nvmlDeviceGetMemoryInfo",
    "nvmlDeviceGetTemperature",
    "nvmlDeviceGetClockInfo",
    "nvmlDeviceGetFanSpeed",
    "nvmlDeviceGetPerformanceState",
    "nvmlDeviceGetCurrentClocksThrottleReasons",
    "nvmlDeviceGetCurrentClocksEventReasons",
    "nvmlDeviceGetTotalEnergyConsumption",
)


class _NvmlError(Exception):
    pass


class _Util:
    def __init__(self, gpu, memory):
        self.gpu = gpu
        self.memory = memory


class _Mem:
    def __init__(self, used):
        self.used = used


class FakeNvml:
    """A configurable stand-in for the pynvml module, scoped to exactly the
    channels `read_nvml_channels`/`read_energy_counter_mj` read."""

    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 0
    NVML_CLOCK_MEM = 1

    def __init__(self, *, disabled=()):
        self.disabled = set(disabled)

    def _guard(self, name):
        if name in self.disabled:
            raise _NvmlError(f"{name} disabled")

    def nvmlDeviceGetPowerUsage(self, h):
        self._guard("power")
        return 150_000

    def nvmlDeviceGetUtilizationRates(self, h):
        self._guard("util")
        return _Util(80, 40)

    def nvmlDeviceGetMemoryInfo(self, h):
        self._guard("memory")
        return _Mem(4 * 1024 * 1024 * 1024)  # 4 GiB in bytes

    def nvmlDeviceGetTemperature(self, h, kind):
        self._guard("temp")
        return 70

    def nvmlDeviceGetClockInfo(self, h, kind):
        self._guard("clocks")
        return 1900 if kind == self.NVML_CLOCK_SM else 9500

    def nvmlDeviceGetFanSpeed(self, h):
        self._guard("fan")
        return 60

    def nvmlDeviceGetPerformanceState(self, h):
        self._guard("perf")
        return 0

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, h):
        self._guard("throttle")
        return 0

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        self._guard("energy")
        return 5000.0


# ============================================================
# read_nvml_channels / read_energy_counter_mj
# ============================================================
def test_read_nvml_channels_maps_every_field():
    ch = read_nvml_channels(FakeNvml(), "h")
    assert ch == {
        "power_w": 150.0,
        "util_gpu": 80.0,
        "util_mem": 40.0,
        "mem_used_mib": 4096.0,
        "temp_c": 70.0,
        "sm_clock_mhz": 1900.0,
        "mem_clock_mhz": 9500.0,
        "fan_pct": 60.0,
        "perf_state": 0,
        "throttle_mask": 0,
    }


def test_read_nvml_channels_degrades_per_channel():
    ch = read_nvml_channels(FakeNvml(disabled={"power", "fan"}), "h")
    assert ch["power_w"] is None
    assert ch["fan_pct"] is None
    assert ch["util_gpu"] == 80.0  # unaffected channels intact
    assert ch["temp_c"] == 70.0


def test_read_nvml_channels_never_raises_on_missing_methods():
    class _Bare:
        pass

    ch = read_nvml_channels(_Bare(), "h")
    assert all(v is None for v in ch.values())


def test_read_nvml_channels_falls_back_to_event_reasons_alias():
    """A newer nvidia-ml-py that renamed the throttle getter entirely (the
    attribute is ABSENT, not just failing) still yields a throttle mask via
    the EventReasons alias."""

    class _RenamedNvml:
        def nvmlDeviceGetCurrentClocksEventReasons(self, h):
            return 0x8

    ch = read_nvml_channels(_RenamedNvml(), "h")
    assert ch["throttle_mask"] == 0x8


def test_read_energy_counter_mj():
    assert read_energy_counter_mj(FakeNvml(), "h") == 5000.0


def test_read_energy_counter_mj_none_on_failure():
    assert read_energy_counter_mj(FakeNvml(disabled={"energy"}), "h") is None


# ============================================================
# Grep gate: exactly one implementation of the per-tick read calls.
# ============================================================
def test_profiler_sample_loop_delegates_to_shared_reader():
    src = inspect.getsource(NVMLProfiler._read_gpu_sample) + inspect.getsource(
        NVMLProfiler._read_energy_counter_mj
    )
    assert "read_nvml_channels" in src
    assert "read_energy_counter_mj" in src
    for call in _NVML_READ_CALLS:
        assert call not in src, f"{call} re-implemented in NVMLProfiler's sample loop"


def test_bench_sampler_sample_delegates_to_shared_reader():
    src = inspect.getsource(LocalNvmlSampler.sample)
    assert "read_nvml_channels" in src
    assert "read_energy_counter_mj" in src
    for call in _NVML_READ_CALLS:
        assert call not in src, f"{call} re-implemented in LocalNvmlSampler.sample"
