"""Tests for hmasync_controller.bench.sampler (US-MERGE-02): LocalNvmlSampler,
ported from energy-bench's `quick.py`. Mocked pynvml throughout, via a
`FakeNvml` stand-in installed in `sys.modules` (same pattern as
tests/test_profiler.py) rather than `unittest.mock.MagicMock` -- so every
channel's value is explicit and asserted, not left to MagicMock's implicit
dunder defaults.

`LocalNvmlSampler`'s methods are async, and this suite has no
`pytest.mark.asyncio` precedent (pytest-asyncio is installed but
unconfigured) -- drive them with `asyncio.run(...)` inside plain sync tests.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from hmasync_controller.bench.sampler import (
    LocalNvmlSampler,
    NvmlUnavailableError,
    TelemetrySample,
    _format_cuda_version,
)


class _NvmlError(Exception):
    pass


class _Util:
    def __init__(self, gpu, memory):
        self.gpu = gpu
        self.memory = memory


class _Mem:
    def __init__(self, used, total):
        self.used = used
        self.total = total


class FakeNvml:
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 0
    NVML_CLOCK_MEM = 1

    def __init__(
        self,
        *,
        power_mw=200_000,
        util_gpu=75.0,
        util_mem=40.0,
        mem_used=8 * 1024 * 1024,
        mem_total=24 * 1024 * 1024,
        temp_c=65,
        sm_clock=1800,
        mem_clock=9500,
        fan_pct=55,
        perf_state=0,
        throttle_mask=0,
        energy_mj=12345.0,
        gpu_name="NVIDIA GeForce RTX 3090",
        driver_version="550.90.07",
        cuda_version=12040,
        power_limit_mw=300_000,
        power_limit_min_mw=100_000,
        power_limit_max_mw=350_000,
        init_fails=False,
        disabled=(),
    ):
        self.power_mw = power_mw
        self.util = _Util(util_gpu, util_mem)
        self.mem = _Mem(mem_used, mem_total)
        self.temp_c = temp_c
        self.sm_clock = sm_clock
        self.mem_clock = mem_clock
        self.fan_pct = fan_pct
        self.perf_state = perf_state
        self.throttle_mask = throttle_mask
        self.energy_mj = energy_mj
        self.gpu_name = gpu_name
        self.driver_version = driver_version
        self.cuda_version = cuda_version
        self.power_limit_mw = power_limit_mw
        self.power_limit_min_mw = power_limit_min_mw
        self.power_limit_max_mw = power_limit_max_mw
        self.init_fails = init_fails
        self.disabled = set(disabled)
        self.init_calls = 0
        self.shutdown_calls = 0
        self.set_power_limit_calls: list[int] = []

    def _guard(self, name):
        if name in self.disabled:
            raise _NvmlError(f"{name} disabled")

    def nvmlInit(self):
        if self.init_fails:
            raise _NvmlError("no NVIDIA GPU on this box")
        self.init_calls += 1

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"handle-{index}"

    def nvmlDeviceGetPowerUsage(self, h):
        self._guard("power")
        return self.power_mw

    def nvmlDeviceGetUtilizationRates(self, h):
        self._guard("util")
        return self.util

    def nvmlDeviceGetMemoryInfo(self, h):
        self._guard("memory")
        return self.mem

    def nvmlDeviceGetTemperature(self, h, kind):
        self._guard("temp")
        return self.temp_c

    def nvmlDeviceGetClockInfo(self, h, kind):
        self._guard("clocks")
        return self.sm_clock if kind == self.NVML_CLOCK_SM else self.mem_clock

    def nvmlDeviceGetFanSpeed(self, h):
        self._guard("fan")
        return self.fan_pct

    def nvmlDeviceGetPerformanceState(self, h):
        self._guard("perf")
        return self.perf_state

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, h):
        self._guard("throttle")
        return self.throttle_mask

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        self._guard("energy")
        return self.energy_mj

    def nvmlDeviceGetName(self, h):
        return self.gpu_name

    def nvmlSystemGetDriverVersion(self):
        return self.driver_version

    def nvmlSystemGetCudaDriverVersion(self):
        return self.cuda_version

    def nvmlDeviceGetPowerManagementLimit(self, h):
        return self.power_limit_mw

    def nvmlDeviceGetPowerManagementLimitConstraints(self, h):
        return self.power_limit_min_mw, self.power_limit_max_mw

    def nvmlDeviceSetPowerManagementLimit(self, h, limit_mw):
        self.set_power_limit_calls.append(limit_mw)
        self.power_limit_mw = limit_mw


def _install_fake_pynvml(monkeypatch, fake=None):
    fake = fake or FakeNvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    return fake


# ============================================================
def test_construction_performs_no_io(monkeypatch):
    fake = _install_fake_pynvml(monkeypatch)
    LocalNvmlSampler()
    assert fake.init_calls == 0


def test_ensure_handle_raises_when_pynvml_not_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    sampler = LocalNvmlSampler()
    with pytest.raises(NvmlUnavailableError):
        sampler._ensure_handle()


def test_ensure_handle_raises_when_nvml_init_fails(monkeypatch):
    _install_fake_pynvml(monkeypatch, FakeNvml(init_fails=True))
    sampler = LocalNvmlSampler()
    with pytest.raises(NvmlUnavailableError, match="no NVIDIA GPU"):
        sampler._ensure_handle()


def test_sample_reads_all_channels(monkeypatch):
    _install_fake_pynvml(monkeypatch)
    sampler = LocalNvmlSampler()
    sampler._ensure_handle()
    sample = sampler.sample()
    assert isinstance(sample, TelemetrySample)
    assert sample.gpu_power_w == pytest.approx(200.0)
    assert sample.gpu_util_pct == pytest.approx(75.0)
    assert sample.gpu_mem_used_mib == pytest.approx(8.0)
    assert sample.gpu_temp_c == pytest.approx(65.0)
    assert sample.gpu_mem_util_pct == pytest.approx(40.0)
    assert sample.gpu_energy_mj == pytest.approx(12345.0)
    assert sample.gpu_throttle_reasons == 0
    assert sample.gpu_sm_clock_mhz == 1800
    assert sample.gpu_mem_clock_mhz == 9500
    assert sample.gpu_fan_pct == 55
    assert sample.gpu_perf_state == 0


def test_sample_degrades_missing_channels_to_none_not_fabricated(monkeypatch):
    _install_fake_pynvml(monkeypatch, FakeNvml(disabled={"fan", "throttle", "energy"}))
    sampler = LocalNvmlSampler()
    sampler._ensure_handle()
    sample = sampler.sample()
    assert sample.gpu_fan_pct is None
    assert sample.gpu_throttle_reasons is None
    assert sample.gpu_energy_mj is None
    # Unaffected channels still read.
    assert sample.gpu_power_w == pytest.approx(200.0)
    assert sample.gpu_sm_clock_mhz == 1800


def test_start_stop_collects_samples(monkeypatch):
    _install_fake_pynvml(monkeypatch)
    sampler = LocalNvmlSampler(sample_hz=50)

    async def _run():
        await sampler.start("run-1")
        await asyncio.sleep(0.06)
        return await sampler.stop()

    samples = asyncio.run(_run())
    assert len(samples) >= 1
    assert all(isinstance(s, TelemetrySample) for s in samples)


def test_stop_returns_empty_after_drain(monkeypatch):
    _install_fake_pynvml(monkeypatch)
    sampler = LocalNvmlSampler(sample_hz=50)

    async def _run():
        await sampler.start("run-1")
        await asyncio.sleep(0.03)
        first = await sampler.stop()
        second = await sampler.stop()
        return first, second

    first, second = asyncio.run(_run())
    assert len(first) >= 1
    assert second == []


def test_gpu_info(monkeypatch):
    _install_fake_pynvml(monkeypatch)
    sampler = LocalNvmlSampler()
    info = asyncio.run(sampler.gpu_info())
    assert info["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert info["driver_version"] == "550.90.07"
    assert info["cuda_version"] == "12.4"
    assert info["gpu_mem_total_mib"] == pytest.approx(24.0)


def test_get_power_limit_w(monkeypatch):
    _install_fake_pynvml(monkeypatch)
    sampler = LocalNvmlSampler()
    assert asyncio.run(sampler.get_power_limit_w()) == 300


def test_get_power_limit_constraints_w(monkeypatch):
    _install_fake_pynvml(monkeypatch)
    sampler = LocalNvmlSampler()
    lo, hi = asyncio.run(sampler.get_power_limit_constraints_w())
    assert (lo, hi) == (100, 350)


def test_get_power_limit_constraints_w_none_on_failure(monkeypatch):
    class _Boom(FakeNvml):
        def nvmlDeviceGetPowerManagementLimitConstraints(self, h):
            raise _NvmlError("unsupported")

    _install_fake_pynvml(monkeypatch, _Boom())
    sampler = LocalNvmlSampler()
    assert asyncio.run(sampler.get_power_limit_constraints_w()) == (None, None)


def test_set_power_limit_w_success(monkeypatch):
    fake = _install_fake_pynvml(monkeypatch)
    sampler = LocalNvmlSampler()
    confirmed = asyncio.run(sampler.set_power_limit_w(280))
    assert confirmed == 280
    assert fake.set_power_limit_calls == [280_000]


def test_set_power_limit_w_failure_returns_none_never_raises(monkeypatch):
    """The mini power sweep's whole skip-gracefully contract depends on this
    returning None rather than raising -- SetPowerManagementLimit needs root
    on most drivers."""

    class _Boom(FakeNvml):
        def nvmlDeviceSetPowerManagementLimit(self, h, limit_mw):
            raise _NvmlError("Insufficient Permissions")

    _install_fake_pynvml(monkeypatch, _Boom())
    sampler = LocalNvmlSampler()
    assert asyncio.run(sampler.set_power_limit_w(280)) is None


def test_close_shuts_down_nvml_once_initialized(monkeypatch):
    fake = _install_fake_pynvml(monkeypatch)
    sampler = LocalNvmlSampler()
    sampler._ensure_handle()
    sampler.close()
    assert fake.shutdown_calls == 1


def test_close_is_a_noop_before_any_nvml_call(monkeypatch):
    fake = _install_fake_pynvml(monkeypatch)
    sampler = LocalNvmlSampler()
    sampler.close()
    assert fake.shutdown_calls == 0


def test_format_cuda_version():
    assert _format_cuda_version(12040) == "12.4"
    assert _format_cuda_version(11080) == "11.8"
