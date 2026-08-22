"""
Tests for the power-cap manager (US-ONB-06): apply/restore around a scheduled
GPU job, gated on config.APPLY_POWER_CAP + an NVML-backed profiler.

No live GPU, no live API: NVML is driven through a minimal FakeNvml stand-in
(same shape as test_profiler.py's), and the API client is a bare stand-in
exposing exactly the one method PowerCapManager calls, returning real
`ApiResult` values so the ok/data contract is exercised for real.
"""

from __future__ import annotations

from hmasync_controller.apiclient import ApiResult
from hmasync_controller.powercap import (
    PowerCapManager,
    STATUS_APPLIED,
    STATUS_NO_RECOMMENDATION,
    STATUS_NOTHING_TO_RESTORE,
    STATUS_RESTORED,
    STATUS_SKIPPED_ERROR,
    STATUS_SKIPPED_NO_GPU,
    STATUS_SKIPPED_NO_PERMISSION,
)
from hmasync_controller.profiler import NVMLProfiler, NullProfiler


class _NvmlError(Exception):
    pass


class FakeNvml:
    """Just enough of pynvml for the power-limit get/set path."""

    NVMLError_NoPermission = type("NVMLError_NoPermission", (_NvmlError,), {})

    def __init__(self, *, limit_mw=300_000, deny=False):
        self.limit_mw = limit_mw
        self.deny = deny
        self.set_calls: list[int] = []

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"handle-{index}"

    def nvmlDeviceGetPowerManagementLimit(self, h):
        return self.limit_mw

    def nvmlDeviceSetPowerManagementLimit(self, h, limit_mw):
        if self.deny:
            raise self.NVMLError_NoPermission("not privileged")
        self.set_calls.append(limit_mw)
        self.limit_mw = limit_mw


class FakeClient:
    def __init__(self, *, cap_w=None, ok=True, error="boom"):
        self.cap_w = cap_w
        self.ok = ok
        self.error = error
        self.calls = 0

    def get_recommended_cap(self, node_hash, *, tolerance_pct=5.0):
        self.calls += 1
        if not self.ok:
            return ApiResult(ok=False, error=self.error)
        return ApiResult(ok=True, status_code=200, data={"recommended_cap_w": self.cap_w})


def _clock(*times):
    it = iter(times)
    return lambda: next(it)


# ============================================================
# apply(): no GPU / no recommendation
# ============================================================
def test_apply_skips_when_no_gpu():
    mgr = PowerCapManager(client=FakeClient(cap_w=250.0), profiler=NullProfiler(), node_hash="n1")
    assert mgr.apply() == STATUS_SKIPPED_NO_GPU
    assert mgr.restore() == STATUS_NOTHING_TO_RESTORE


def test_apply_null_recommendation_is_no_recommendation():
    client = FakeClient(cap_w=None)
    profiler = NVMLProfiler(nvml=FakeNvml())
    mgr = PowerCapManager(client=client, profiler=profiler, node_hash="n1")
    assert mgr.apply() == STATUS_NO_RECOMMENDATION
    assert profiler._nvml.set_calls == []  # never touched the driver


def test_apply_fetch_failure_is_treated_as_no_recommendation():
    client = FakeClient(ok=False, error="unreachable")
    profiler = NVMLProfiler(nvml=FakeNvml())
    mgr = PowerCapManager(client=client, profiler=profiler, node_hash="n1")
    assert mgr.apply() == STATUS_NO_RECOMMENDATION


def test_apply_fetch_exception_is_treated_as_no_recommendation():
    class _BoomClient:
        def get_recommended_cap(self, node_hash, *, tolerance_pct=5.0):
            raise RuntimeError("boom")

    profiler = NVMLProfiler(nvml=FakeNvml())
    mgr = PowerCapManager(client=_BoomClient(), profiler=profiler, node_hash="n1")
    assert mgr.apply() == STATUS_NO_RECOMMENDATION


# ============================================================
# apply() + restore(): the happy path and the "never fails a job" paths
# ============================================================
def test_apply_sets_the_limit_and_restore_puts_the_prior_one_back():
    fake_nvml = FakeNvml(limit_mw=300_000)
    profiler = NVMLProfiler(nvml=fake_nvml)
    mgr = PowerCapManager(client=FakeClient(cap_w=250.0), profiler=profiler, node_hash="n1")

    assert mgr.apply() == STATUS_APPLIED
    assert fake_nvml.set_calls == [250_000]

    assert mgr.restore() == STATUS_RESTORED
    assert fake_nvml.set_calls == [250_000, 300_000]


def test_restore_without_a_prior_apply_is_a_noop():
    profiler = NVMLProfiler(nvml=FakeNvml())
    mgr = PowerCapManager(client=FakeClient(cap_w=250.0), profiler=profiler, node_hash="n1")
    assert mgr.restore() == STATUS_NOTHING_TO_RESTORE


def test_restore_runs_even_when_the_job_raises():
    """The AC's core guarantee: a crashing job still gets the cap taken off."""
    fake_nvml = FakeNvml(limit_mw=300_000)
    profiler = NVMLProfiler(nvml=fake_nvml)
    mgr = PowerCapManager(client=FakeClient(cap_w=250.0), profiler=profiler, node_hash="n1")

    assert mgr.apply() == STATUS_APPLIED
    try:
        raise RuntimeError("job blew up")
    except RuntimeError:
        pass
    finally:
        assert mgr.restore() == STATUS_RESTORED
    assert fake_nvml.set_calls == [250_000, 300_000]


def test_apply_no_permission_is_reported_and_never_raises():
    profiler = NVMLProfiler(nvml=FakeNvml(deny=True))
    mgr = PowerCapManager(client=FakeClient(cap_w=250.0), profiler=profiler, node_hash="n1")
    assert mgr.apply() == STATUS_SKIPPED_NO_PERMISSION
    # apply() never partially succeeded, so there's nothing queued to restore.
    assert mgr.restore() == STATUS_NOTHING_TO_RESTORE


def test_apply_when_current_limit_unreadable_skips_rather_than_apply_blind():
    class _UnreadableLimit(FakeNvml):
        def nvmlDeviceGetPowerManagementLimit(self, h):
            raise _NvmlError("unreadable")

    profiler = NVMLProfiler(nvml=_UnreadableLimit())
    mgr = PowerCapManager(client=FakeClient(cap_w=250.0), profiler=profiler, node_hash="n1")
    assert mgr.apply() == STATUS_SKIPPED_ERROR
    assert profiler._nvml.set_calls == []  # never applied a cap it couldn't undo


def test_restore_failure_never_raises():
    class _DenyOnSecondSet(FakeNvml):
        def nvmlDeviceSetPowerManagementLimit(self, h, limit_mw):
            if self.set_calls:
                raise _NvmlError("driver vanished")
            super().nvmlDeviceSetPowerManagementLimit(h, limit_mw)

    profiler = NVMLProfiler(nvml=_DenyOnSecondSet(limit_mw=300_000))
    mgr = PowerCapManager(client=FakeClient(cap_w=250.0), profiler=profiler, node_hash="n1")
    assert mgr.apply() == STATUS_APPLIED
    assert mgr.restore() == STATUS_SKIPPED_ERROR  # never raises


# ============================================================
# Caching (AC: "cache, refresh daily")
# ============================================================
def test_recommendation_is_cached_across_apply_calls():
    client = FakeClient(cap_w=250.0)
    profiler = NVMLProfiler(nvml=FakeNvml())
    mgr = PowerCapManager(
        client=client, profiler=profiler, node_hash="n1",
        cache_ttl_s=1000.0, clock=_clock(0.0, 10.0),
    )
    mgr.apply()
    mgr.restore()
    mgr.apply()
    assert client.calls == 1  # the second apply() reused the cached value


def test_recommendation_refetches_after_the_ttl_elapses():
    client = FakeClient(cap_w=250.0)
    profiler = NVMLProfiler(nvml=FakeNvml())
    mgr = PowerCapManager(
        client=client, profiler=profiler, node_hash="n1",
        cache_ttl_s=5.0, clock=_clock(0.0, 10.0),
    )
    mgr.apply()
    mgr.restore()
    mgr.apply()
    assert client.calls == 2  # ttl elapsed between the two fetches
