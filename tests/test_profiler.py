"""
Tests for the profiler seam.

Repo rule: no live GPU. pynvml, nvidia-smi (subprocess), and the RAPL sysfs reads
are mocked entirely, so these run on CI with no NVIDIA GPU. NVML is driven
through a `FakeNvml` module stand-in; smi through a patched `subprocess.run`; RAPL
through module-level path stand-ins.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from hmasync_controller import profiler as prof
from hmasync_controller.profiler import (
    CAP_CLOCKS,
    CAP_CPU_RAPL,
    CAP_ENERGY_COUNTER,
    CAP_MEMORY,
    CAP_POWER,
    CAP_TEMP,
    CAP_THROTTLE,
    CAP_UTIL,
    NullProfiler,
    NVMLProfiler,
    ProfilerUnavailable,
    RunTelemetry,
    SmiProfiler,
    get_profiler,
    read_rapl_energy_uj,
)


# ============================================================
# Fakes
# ============================================================
class _NvmlError(Exception):
    pass


class _Util:
    def __init__(self, gpu, memory):
        self.gpu = gpu
        self.memory = memory


class _Mem:
    def __init__(self, used, total=None):
        self.used = used
        self.total = total if total is not None else used


class FakeNvml:
    """A configurable stand-in for the pynvml module.

    Each getter can be disabled (raises NVMLError) to exercise per-channel null
    degradation and capability probing. The energy counter returns successive
    values from `energy_mj` so start()/stop() see a delta.
    """

    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVMLError = _NvmlError

    # Throttle-reason bit constants the decoder looks up by name.
    nvmlClocksThrottleReasonGpuIdle = 0x1
    nvmlClocksThrottleReasonSwPowerCap = 0x4
    nvmlClocksThrottleReasonHwSlowdown = 0x8
    nvmlClocksThrottleReasonSwThermalSlowdown = 0x20

    def __init__(
        self,
        *,
        power_mw=150_000,
        util_gpu=80,
        util_mem=40,
        mem_used=4 * 1024 * 1024 * 1024,  # 4 GiB in bytes
        mem_total=24 * 1024 * 1024 * 1024,  # 24 GiB in bytes
        temp_c=70,
        sm_clock=1900,
        energy_mj=None,  # list of successive counter reads (mJ)
        throttle_mask=0,
        disabled=(),  # channel names to make raise NVMLError
        gpu_name="NVIDIA GeForce RTX 4090",
        driver_version="550.90.07",
    ):
        self.power_mw = power_mw
        self.util = _Util(util_gpu, util_mem)
        self.mem = _Mem(mem_used, mem_total)
        self.temp_c = temp_c
        self.sm_clock = sm_clock
        self.energy_mj = list(energy_mj) if energy_mj is not None else None
        self._energy_idx = 0
        self.throttle_mask = throttle_mask
        self.disabled = set(disabled)
        self.init_calls = 0
        self.shutdown_calls = 0
        self.count = 1
        self.gpu_name = gpu_name
        self.driver_version = driver_version

    def _guard(self, name):
        if name in self.disabled:
            raise _NvmlError(f"{name} disabled")

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self):
        return self.count

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
        return self.sm_clock

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        self._guard("energy")
        if self.energy_mj is None:
            raise _NvmlError("no energy counter")
        val = self.energy_mj[min(self._energy_idx, len(self.energy_mj) - 1)]
        self._energy_idx += 1
        return val

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, h):
        self._guard("throttle")
        return self.throttle_mask

    def nvmlDeviceGetName(self, h):
        self._guard("name")
        return self.gpu_name

    def nvmlSystemGetDriverVersion(self):
        self._guard("driver_version")
        return self.driver_version


class _FakePath:
    def __init__(self, exists=False, text=""):
        self._exists = exists
        self._text = text

    def exists(self):
        return self._exists

    def read_text(self):
        return self._text


# ============================================================
# RAPL reader (Intel + AMD sysfs paths)
# ============================================================
def test_rapl_intel_path(monkeypatch):
    monkeypatch.setattr(prof, "_INTEL_RAPL", _FakePath(exists=True, text="123456\n"))
    monkeypatch.setattr(prof, "_AMD_RAPL", _FakePath(exists=False))
    assert read_rapl_energy_uj() == 123456.0


def test_rapl_amd_path(monkeypatch):
    monkeypatch.setattr(prof, "_INTEL_RAPL", _FakePath(exists=False))
    monkeypatch.setattr(prof, "_AMD_RAPL", _FakePath(exists=True, text="999\n"))
    assert read_rapl_energy_uj() == 999.0


def test_rapl_intel_preferred_over_amd(monkeypatch):
    monkeypatch.setattr(prof, "_INTEL_RAPL", _FakePath(exists=True, text="111"))
    monkeypatch.setattr(prof, "_AMD_RAPL", _FakePath(exists=True, text="222"))
    assert read_rapl_energy_uj() == 111.0


def test_rapl_absent_returns_none(monkeypatch):
    monkeypatch.setattr(prof, "_INTEL_RAPL", _FakePath(exists=False))
    monkeypatch.setattr(prof, "_AMD_RAPL", _FakePath(exists=False))
    assert read_rapl_energy_uj() is None


def test_rapl_unreadable_returns_none(monkeypatch):
    class _Boom(_FakePath):
        def read_text(self):
            raise OSError("permission denied")

    monkeypatch.setattr(prof, "_INTEL_RAPL", _Boom(exists=True))
    monkeypatch.setattr(prof, "_AMD_RAPL", _FakePath(exists=False))
    assert read_rapl_energy_uj() is None


# ============================================================
# Backend probe order (NVML → smi → null)
# ============================================================
class _StubNVML:
    def __init__(self, **kw):
        self.kw = kw


class _StubSmi:
    def __init__(self, **kw):
        self.kw = kw


def test_probe_prefers_nvml(monkeypatch):
    monkeypatch.setattr(prof, "_nvml_available", lambda: True)
    monkeypatch.setattr(prof, "_smi_available", lambda: True)
    monkeypatch.setattr(prof, "NVMLProfiler", _StubNVML)
    monkeypatch.setattr(prof, "SmiProfiler", _StubSmi)
    assert isinstance(get_profiler(), _StubNVML)


def test_probe_falls_to_smi_when_no_nvml(monkeypatch):
    monkeypatch.setattr(prof, "_nvml_available", lambda: False)
    monkeypatch.setattr(prof, "_smi_available", lambda: True)
    monkeypatch.setattr(prof, "SmiProfiler", _StubSmi)
    assert isinstance(get_profiler(), _StubSmi)


def test_probe_falls_to_null_when_nothing(monkeypatch):
    monkeypatch.setattr(prof, "_nvml_available", lambda: False)
    monkeypatch.setattr(prof, "_smi_available", lambda: False)
    assert isinstance(get_profiler(), NullProfiler)


def test_probe_falls_back_when_nvml_construct_fails(monkeypatch):
    """NVML probed available but construction raises ProfilerUnavailable → smi."""

    def _boom(**kw):
        raise ProfilerUnavailable("driver vanished")

    monkeypatch.setattr(prof, "_nvml_available", lambda: True)
    monkeypatch.setattr(prof, "_smi_available", lambda: True)
    monkeypatch.setattr(prof, "NVMLProfiler", _boom)
    monkeypatch.setattr(prof, "SmiProfiler", _StubSmi)
    assert isinstance(get_profiler(), _StubSmi)


def test_nvml_available_false_without_pynvml(monkeypatch):
    def _no_pynvml():
        raise ProfilerUnavailable("no pynvml")

    monkeypatch.setattr(prof, "_import_pynvml", _no_pynvml)
    assert prof._nvml_available() is False


def test_nvml_available_false_without_gpu(monkeypatch):
    fake = FakeNvml()
    fake.count = 0
    monkeypatch.setattr(prof, "_import_pynvml", lambda: fake)
    assert prof._nvml_available() is False
    assert fake.shutdown_calls >= 1  # probe cleans up


# ============================================================
# NVML channel reads + null degradation
# ============================================================
def test_nvml_reads_all_channels():
    p = NVMLProfiler(nvml=FakeNvml(throttle_mask=0), rapl_reader=lambda: 5000.0)
    p._ensure_nvml()
    s = p._read_gpu_sample()
    assert s["power_w"] == 150.0  # 150_000 mW / 1000
    assert s["util_gpu"] == 80.0
    assert s["util_mem"] == 40.0
    assert s["mem_used_mb"] == 4096.0  # 4 GiB → MiB
    assert s["temp_c"] == 70.0
    assert s["sm_clock_mhz"] == 1900.0
    assert s["throttle_reasons"] == "none"  # mask 0


def test_nvml_collect_one_adds_rapl_and_aware_ts():
    p = NVMLProfiler(nvml=FakeNvml(), rapl_reader=lambda: 8000.0)
    p._ensure_nvml()
    p._collect_one()
    s = p._samples[-1]
    assert s["cpu_rapl_uj"] == 8000.0
    assert s["ts"].tzinfo is not None  # no naive datetimes


def test_nvml_missing_channels_yield_null_not_fabricated():
    # Disable power, clocks, and throttle → those channels null, others intact.
    fake = FakeNvml(disabled={"power", "clocks", "throttle"})
    p = NVMLProfiler(nvml=fake, rapl_reader=lambda: None)
    p._ensure_nvml()
    s = p._read_gpu_sample()
    assert s["power_w"] is None
    assert s["sm_clock_mhz"] is None
    assert s["throttle_reasons"] is None
    # Non-disabled channels still read.
    assert s["temp_c"] == 70.0
    assert s["util_gpu"] == 80.0


def test_nvml_throttle_decode_thermal():
    # HW slowdown (0x8) | SW thermal (0x20) → both names, treated as throttled.
    fake = FakeNvml(throttle_mask=0x8 | 0x20)
    p = NVMLProfiler(nvml=fake)
    p._ensure_nvml()
    s = p._read_gpu_sample()
    assert "hw_slowdown" in s["throttle_reasons"]
    assert "sw_thermal" in s["throttle_reasons"]


def test_nvml_throttle_decode_idle_only():
    fake = FakeNvml(throttle_mask=0x1)  # GPU idle only
    p = NVMLProfiler(nvml=fake)
    p._ensure_nvml()
    s = p._read_gpu_sample()
    assert s["throttle_reasons"] == "gpuidle"  # rollup treats this as NOT throttled


# ============================================================
# NVML device_fingerprint (US-ONB-05: gpu_name/driver_version/vram_gb)
# ============================================================
def test_nvml_device_fingerprint_reads_name_driver_vram():
    fake = FakeNvml(gpu_name="NVIDIA GeForce RTX 4090", driver_version="550.90.07",
                     mem_total=24 * 1024 * 1024 * 1024)
    p = NVMLProfiler(nvml=fake)
    fp = p.device_fingerprint()
    assert fp["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert fp["driver_version"] == "550.90.07"
    assert fp["vram_gb"] == 24.0


def test_nvml_device_fingerprint_decodes_bytes():
    fake = FakeNvml(gpu_name=b"NVIDIA GeForce RTX 4090", driver_version=b"550.90.07")
    p = NVMLProfiler(nvml=fake)
    fp = p.device_fingerprint()
    assert fp["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert fp["driver_version"] == "550.90.07"


def test_nvml_device_fingerprint_never_includes_identity_fields():
    fake = FakeNvml()
    p = NVMLProfiler(nvml=fake)
    fp = p.device_fingerprint()
    assert set(fp) <= {"gpu_name", "driver_version", "vram_gb"}


def test_nvml_device_fingerprint_degrades_per_channel():
    fake = FakeNvml(disabled={"name", "driver_version", "memory"})
    p = NVMLProfiler(nvml=fake)
    fp = p.device_fingerprint()
    assert fp == {}


# ============================================================
# NVML capabilities
# ============================================================
def test_nvml_capabilities_full():
    fake = FakeNvml(energy_mj=[1000], throttle_mask=0)
    p = NVMLProfiler(nvml=fake, rapl_reader=lambda: 5000.0)
    caps = p.capabilities()
    assert caps == {
        CAP_POWER,
        CAP_UTIL,
        CAP_MEMORY,
        CAP_TEMP,
        CAP_CLOCKS,
        CAP_ENERGY_COUNTER,
        CAP_THROTTLE,
        CAP_CPU_RAPL,
    }


def test_nvml_capabilities_drop_missing():
    # No energy counter, no throttle, no RAPL → those capabilities absent.
    fake = FakeNvml(energy_mj=None, disabled={"throttle"})
    p = NVMLProfiler(nvml=fake, rapl_reader=lambda: None)
    caps = p.capabilities()
    assert CAP_ENERGY_COUNTER not in caps
    assert CAP_THROTTLE not in caps
    assert CAP_CPU_RAPL not in caps
    assert {CAP_POWER, CAP_UTIL, CAP_MEMORY, CAP_TEMP, CAP_CLOCKS} <= caps


# ============================================================
# Energy: counter preference vs integration
# ============================================================
def test_energy_counter_preferred():
    # start reads 3.6e6 mJ, stop reads 7.2e6 mJ → delta 3.6e6 mJ = 1 Wh.
    fake = FakeNvml(energy_mj=[3_600_000, 7_200_000])
    p = NVMLProfiler(
        nvml=fake, sample_interval_s=100.0, rapl_reader=lambda: None
    )
    p.start("run-1")
    telem = p.stop("run-1")
    assert telem.energy_source == "counter"
    assert telem.energy_wh == pytest.approx(1.0)


def test_energy_integrated_when_no_counter():
    # Two power samples 100 W, 10 s apart → 1000 J → 1000/3600 Wh.
    p = NVMLProfiler(nvml=FakeNvml(energy_mj=None))
    t0 = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)
    p._samples = [
        {"power_w": 100.0, "ts": t0},
        {"power_w": 100.0, "ts": t0 + timedelta(seconds=10)},
    ]
    energy_wh, source = p._compute_energy(None, None)
    assert source == "integrated"
    assert energy_wh == pytest.approx(1000.0 / 3600.0)


def test_energy_null_without_power_or_counter():
    p = NullProfiler(rapl_reader=lambda: None)
    p._samples = [{"power_w": None, "ts": datetime.now(timezone.utc)}]
    energy_wh, source = p._compute_energy(None, None)
    assert energy_wh is None
    assert source is None


def test_energy_counter_reset_falls_through_to_integration():
    # A decreasing counter (driver reload) must not yield a negative energy;
    # fall through to integration instead.
    p = NVMLProfiler(nvml=FakeNvml(energy_mj=None))
    t0 = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)
    p._samples = [
        {"power_w": 200.0, "ts": t0},
        {"power_w": 200.0, "ts": t0 + timedelta(seconds=3600)},
    ]
    energy_wh, source = p._compute_energy(9_000_000, 1_000_000)  # end < start
    assert source == "integrated"
    assert energy_wh == pytest.approx(200.0)  # 200 W for 1 h


# ============================================================
# Full start/stop lifecycle (real thread, tiny interval)
# ============================================================
def test_nvml_start_stop_collects_and_stops_cleanly():
    fake = FakeNvml(energy_mj=[0, 0])
    p = NVMLProfiler(nvml=fake, sample_interval_s=0.02, rapl_reader=lambda: 1000.0)
    p.start("run-x")
    time.sleep(0.07)  # allow a few ticks
    telem = p.stop("run-x")
    assert telem.run_id == "run-x"
    assert telem.duration_s >= 0.0
    assert len(telem.samples) >= 1
    # Sampler thread is fully stopped.
    assert p._thread is not None and not p._thread.is_alive()
    # Summary rollups populated from the trace.
    assert telem.avg_w == pytest.approx(150.0)
    assert telem.peak_w == pytest.approx(150.0)
    assert telem.gpu_mem_mb == pytest.approx(4096.0)
    # Every sample carries an aware ts + the RAPL channel.
    assert all(s["ts"].tzinfo is not None for s in telem.samples)
    assert all(s["cpu_rapl_uj"] == 1000.0 for s in telem.samples)


def test_stop_without_start_is_valid():
    p = NullProfiler(rapl_reader=lambda: None)
    telem = p.stop("never-started")
    assert isinstance(telem, RunTelemetry)
    assert telem.duration_s == 0.0
    assert telem.samples == []
    assert telem.energy_wh is None


# ============================================================
# Null backend
# ============================================================
def test_null_profiler_gpu_channels_null_rapl_present():
    p = NullProfiler(rapl_reader=lambda: 5000.0)
    p._collect_one()
    s = p._samples[-1]
    assert s["power_w"] is None
    assert s["util_gpu"] is None
    assert s["sm_clock_mhz"] is None
    assert s["cpu_rapl_uj"] == 5000.0  # RAPL independent of GPU backend
    assert s["ts"].tzinfo is not None


def test_null_profiler_capabilities_rapl_only():
    assert NullProfiler(rapl_reader=lambda: 5000.0).capabilities() == {CAP_CPU_RAPL}
    assert NullProfiler(rapl_reader=lambda: None).capabilities() == set()


def test_null_profiler_full_run_energy_null():
    p = NullProfiler(sample_interval_s=0.02, rapl_reader=lambda: 3000.0)
    p.start("nullrun")
    time.sleep(0.05)
    telem = p.stop("nullrun")
    assert telem.energy_wh is None
    assert telem.energy_source is None
    assert len(telem.samples) >= 1
    assert all(s["cpu_rapl_uj"] == 3000.0 for s in telem.samples)


# ============================================================
# nvidia-smi backend
# ============================================================
def test_smi_parse_csv():
    row = "150.50, 80, 40, 4096, 70, 1900"
    s = prof._parse_smi_csv(row)
    assert s["power_w"] == pytest.approx(150.5)
    assert s["util_gpu"] == 80.0
    assert s["util_mem"] == 40.0
    assert s["mem_used_mb"] == 4096.0
    assert s["temp_c"] == 70.0
    assert s["sm_clock_mhz"] == 1900.0
    assert s["throttle_reasons"] is None  # not in the basic query


def test_smi_parse_handles_not_supported():
    row = "[N/A], 55, [Not Supported], 2048, 60, 1500"
    s = prof._parse_smi_csv(row)
    assert s["power_w"] is None
    assert s["util_gpu"] == 55.0
    assert s["util_mem"] is None
    assert s["mem_used_mb"] == 2048.0


def test_smi_read_gpu_sample(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = "200.0, 90, 50, 8192, 75, 2100\n"

    monkeypatch.setattr(prof.subprocess, "run", lambda *a, **k: _Proc())
    p = SmiProfiler(smi_bin="/usr/bin/nvidia-smi", rapl_reader=lambda: None)
    s = p._read_gpu_sample()
    assert s["power_w"] == 200.0
    assert s["mem_used_mb"] == 8192.0


def test_smi_read_gpu_sample_failure_yields_empty(monkeypatch):
    class _Proc:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(prof.subprocess, "run", lambda *a, **k: _Proc())
    p = SmiProfiler(smi_bin="/usr/bin/nvidia-smi", rapl_reader=lambda: None)
    assert p._read_gpu_sample() == {}


def test_smi_subprocess_error_yields_none(monkeypatch):
    def _boom(*a, **k):
        raise OSError("no such binary")

    monkeypatch.setattr(prof.subprocess, "run", _boom)
    p = SmiProfiler(smi_bin="/usr/bin/nvidia-smi")
    assert p._run_smi() is None


def test_smi_energy_integrated_no_counter():
    # SmiProfiler has no cumulative counter → energy always integrates.
    p = SmiProfiler(smi_bin="/usr/bin/nvidia-smi")
    assert p._read_energy_counter_mj() is None
    t0 = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)
    p._samples = [
        {"power_w": 300.0, "ts": t0},
        {"power_w": 300.0, "ts": t0 + timedelta(seconds=3600)},
    ]
    energy_wh, source = p._compute_energy(None, None)
    assert source == "integrated"
    assert energy_wh == pytest.approx(300.0)


def test_smi_capabilities():
    p = SmiProfiler(smi_bin="/usr/bin/nvidia-smi", rapl_reader=lambda: 5000.0)
    caps = p.capabilities()
    assert {CAP_POWER, CAP_UTIL, CAP_MEMORY, CAP_TEMP, CAP_CLOCKS, CAP_CPU_RAPL} == caps
    assert CAP_ENERGY_COUNTER not in caps
    assert CAP_THROTTLE not in caps


# ============================================================
# RunTelemetry helper
# ============================================================
def test_run_record_fields_subset():
    telem = RunTelemetry(
        run_id="r",
        duration_s=12.5,
        energy_wh=3.0,
        energy_source="counter",
        avg_w=250.0,
        peak_w=300.0,
        gpu_mem_mb=4096.0,
    )
    fields = telem.to_record_fields()
    assert fields == {
        "duration_s": 12.5,
        "energy_wh": 3.0,
        "energy_source": "counter",
        "avg_w": 250.0,
        "peak_w": 300.0,
        "gpu_mem_mb": 4096.0,
    }
    # Leaves API-computed columns out entirely.
    assert "p95_w" not in fields
    assert "cpu_energy_wh" not in fields
    assert "power_profile" not in fields
