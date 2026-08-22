"""
Tests for hmasync_controller/fingerprint.py — the hardware-CLASS fingerprint
collected for bench_nodes registration (US-ONB-05).

No live GPU or /proc dependency: NVML is driven through a tiny fake module
(mirroring test_profiler.py's FakeNvml, kept local and minimal here since only
name/driver/memory are needed), and /proc reads go through tmp_path fixture
files passed directly to read_cpu_model/read_ram_gb rather than the real
/proc/cpuinfo /proc/meminfo.
"""

from __future__ import annotations

import hashlib

from hmasync_controller import bench, fingerprint
from hmasync_controller.profiler import NullProfiler, NVMLProfiler


# ============================================================
# Fakes
# ============================================================
class _Mem:
    def __init__(self, total):
        self.total = total


class FakeNvml:
    """Just enough of pynvml for device_fingerprint()."""

    def __init__(self, *, gpu_name="NVIDIA GeForce RTX 4090", driver_version="550.90.07",
                 vram_bytes=24 * 1024 * 1024 * 1024):
        self.gpu_name = gpu_name
        self.driver_version = driver_version
        self.mem = _Mem(vram_bytes)

    def nvmlInit(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"handle-{index}"

    def nvmlDeviceGetName(self, h):
        return self.gpu_name

    def nvmlSystemGetDriverVersion(self):
        return self.driver_version

    def nvmlDeviceGetMemoryInfo(self, h):
        return self.mem


# ============================================================
# read_cpu_model / read_ram_gb
# ============================================================
def test_read_cpu_model_parses_model_name_line(tmp_path):
    path = tmp_path / "cpuinfo"
    path.write_text("processor\t: 0\nmodel name\t: AMD Ryzen 9 7950X\ncache size\t: 1024 KB\n")
    assert fingerprint.read_cpu_model(path) == "AMD Ryzen 9 7950X"


def test_read_cpu_model_missing_file_is_none(tmp_path):
    assert fingerprint.read_cpu_model(tmp_path / "nope") is None


def test_read_cpu_model_no_matching_line_is_none(tmp_path):
    path = tmp_path / "cpuinfo"
    path.write_text("processor\t: 0\n")
    assert fingerprint.read_cpu_model(path) is None


def test_read_ram_gb_parses_memtotal_kb(tmp_path):
    path = tmp_path / "meminfo"
    # 32 GiB in kB.
    path.write_text("MemTotal:       33554432 kB\nMemFree:        1000 kB\n")
    assert fingerprint.read_ram_gb(path) == 32.0


def test_read_ram_gb_missing_file_is_none(tmp_path):
    assert fingerprint.read_ram_gb(tmp_path / "nope") is None


def test_read_ram_gb_no_matching_line_is_none(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemFree:        1000 kB\n")
    assert fingerprint.read_ram_gb(path) is None


# ============================================================
# read_gpu_fields — NVML-only, never via smi/Null
# ============================================================
def test_read_gpu_fields_via_nvml_profiler():
    profiler = NVMLProfiler(nvml=FakeNvml())
    fields = fingerprint.read_gpu_fields(profiler)
    assert fields["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert fields["driver_version"] == "550.90.07"
    assert fields["vram_gb"] == 24.0


def test_read_gpu_fields_empty_without_nvml_backend():
    assert fingerprint.read_gpu_fields(NullProfiler()) == {}


# ============================================================
# salt + node_hash
# ============================================================
def test_load_or_create_salt_creates_a_fresh_file(tmp_path):
    path = tmp_path / "salt"
    assert not path.exists()
    salt = fingerprint.load_or_create_salt(path)
    assert path.exists()
    assert salt == path.read_text().strip()
    assert len(salt) == 64  # secrets.token_hex(32) -> 64 hex chars


def test_load_or_create_salt_is_stable_across_calls(tmp_path):
    path = tmp_path / "salt"
    first = fingerprint.load_or_create_salt(path)
    second = fingerprint.load_or_create_salt(path)
    assert first == second


def test_load_or_create_salt_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "salt"
    salt = fingerprint.load_or_create_salt(path)
    assert path.exists()
    assert salt


def test_load_or_create_salt_reads_an_existing_operator_value(tmp_path):
    path = tmp_path / "salt"
    path.write_text("my-existing-salt\n")
    assert fingerprint.load_or_create_salt(path) == "my-existing-salt"


def test_compute_node_hash_is_deterministic():
    assert fingerprint.compute_node_hash("abc") == fingerprint.compute_node_hash("abc")


def test_compute_node_hash_differs_per_salt():
    assert fingerprint.compute_node_hash("abc") != fingerprint.compute_node_hash("xyz")


def test_compute_node_hash_never_equals_the_raw_salt():
    salt = "a-locally-generated-secret"
    assert fingerprint.compute_node_hash(salt) != salt


def test_compute_node_hash_matches_expected_sha256():
    salt = "test-salt"
    expected = hashlib.sha256(f"hmasync-node:{salt}".encode()).hexdigest()
    assert fingerprint.compute_node_hash(salt) == expected


# ============================================================
# collect_fingerprint — full payload + denylist
# ============================================================
def test_collect_fingerprint_always_has_node_hash(tmp_path):
    payload = fingerprint.collect_fingerprint(NullProfiler(), tmp_path / "salt")
    assert "node_hash" in payload and payload["node_hash"]


def test_collect_fingerprint_degrades_gracefully_with_no_gpu(tmp_path):
    payload = fingerprint.collect_fingerprint(NullProfiler(), tmp_path / "salt")
    assert "gpu_name" not in payload
    assert "driver_version" not in payload
    assert "vram_gb" not in payload


def test_collect_fingerprint_includes_gpu_fields_via_nvml(tmp_path):
    profiler = NVMLProfiler(nvml=FakeNvml())
    payload = fingerprint.collect_fingerprint(profiler, tmp_path / "salt")
    assert payload["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert payload["vram_gb"] == 24.0


def test_collect_fingerprint_includes_cpu_and_ram_from_proc(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("model name\t: AMD Ryzen 9 7950X\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       33554432 kB\n")
    monkeypatch.setattr(fingerprint, "PROC_CPUINFO", cpuinfo)
    monkeypatch.setattr(fingerprint, "PROC_MEMINFO", meminfo)

    payload = fingerprint.collect_fingerprint(NullProfiler(), tmp_path / "salt")

    assert payload["cpu_model"] == "AMD Ryzen 9 7950X"
    assert payload["ram_gb"] == 32.0


def test_collect_fingerprint_uses_same_node_hash_across_calls(tmp_path):
    salt_path = tmp_path / "salt"
    first = fingerprint.collect_fingerprint(NullProfiler(), salt_path)
    second = fingerprint.collect_fingerprint(NullProfiler(), salt_path)
    assert first["node_hash"] == second["node_hash"]


# The explicit denylist test the PRD's acceptance criteria calls for: the
# fingerprint payload never carries a GPU UUID, serial number, MAC address,
# hostname, or HA entity id, whatever this box's hardware looks like.
def test_collect_fingerprint_never_carries_denylisted_keys(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("model name\t: AMD Ryzen 9 7950X\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       33554432 kB\n")
    monkeypatch.setattr(fingerprint, "PROC_CPUINFO", cpuinfo)
    monkeypatch.setattr(fingerprint, "PROC_MEMINFO", meminfo)

    profiler = NVMLProfiler(nvml=FakeNvml())
    payload = fingerprint.collect_fingerprint(profiler, tmp_path / "salt")

    assert bench.denylisted_keys(payload) == []
    assert set(payload) == {"node_hash", "cpu_model", "ram_gb", "gpu_name", "driver_version", "vram_gb"}
