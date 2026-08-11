"""
ApiClient wire-contract tests.

Everything is exercised through httpx.MockTransport (the FakeApiServer in
conftest) — real request/response code, no live vendor. The two properties under
test are (1) the four wire endpoints speak the API's shapes, and (2) the
clean-error contract: no method ever raises, network/timeout/HTTP errors all come
back as ApiResult(ok=False).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from hmasync_controller.apiclient import ApiClient, ApiResult, server_run_id


def _run(controller_id="box-1", run_id="r-1", **extra) -> dict:
    rec = {"controller_id": controller_id, "run_id": run_id, "duration_s": 12.0}
    rec.update(extra)
    return rec


# --- auth -----------------------------------------------------------------

def test_login_stores_tokens(make_client, fake_api):
    client = make_client()
    result = client.login()
    assert result.ok
    assert result.status_code == 200
    assert client._access_token == "access-1"
    assert client._refresh_token == "refresh-1"


def test_login_without_credentials_fails_cleanly(make_client):
    client = make_client(email="", password="")
    result = client.login()
    assert not result.ok
    assert result.status_code is None
    assert "credentials" in result.error.lower()


def test_first_authed_call_logs_in_automatically(make_client, fake_api):
    client = make_client()
    # No explicit login() — push_run should auto-authenticate.
    result = client.push_run(_run())
    assert result.ok
    assert client._access_token == "access-1"


def test_refresh_on_401_then_retry(make_client, fake_api):
    client = make_client()
    assert client.push_run(_run(run_id="r-a")).ok  # logs in → access-1
    # Access token expires server-side; the next call 401s, client refreshes.
    fake_api.invalidate_access()
    result = client.push_run(_run(run_id="r-b"))
    assert result.ok
    # A refresh happened → the client now holds the reissued access token.
    assert client._access_token == "access-2"
    paths = [r.url.path for r in fake_api.requests]
    assert "/auth/refresh" in paths


def test_relogin_when_refresh_also_fails(make_client, fake_api):
    client = make_client()
    assert client.login().ok  # access-1 / refresh-1
    # Both the access token AND the stored refresh token are no longer valid.
    fake_api.invalidate_access()
    fake_api.valid_refresh.clear()
    result = client.push_run(_run())
    # Refresh fails, but a fresh full login recovers.
    assert result.ok
    paths = [r.url.path for r in fake_api.requests]
    assert "/auth/refresh" in paths
    assert paths.count("/auth/login") == 2


# --- push_run -------------------------------------------------------------

def test_push_run_success_returns_server_id(make_client, fake_api):
    client = make_client()
    result = client.push_run(_run())
    assert result.ok
    assert result.data["accepted"] == 1
    sid = server_run_id(result)
    assert sid is not None
    assert sid in fake_api.server_ids


def test_push_run_idempotent_duplicate_same_server_id(make_client):
    client = make_client()
    first = client.push_run(_run(run_id="dup"))
    second = client.push_run(_run(run_id="dup"))
    assert first.ok and second.ok
    assert second.data["duplicates"] == 1
    # Same (controller_id, run_id) → same server id on the re-push (spool-safe).
    assert server_run_id(first) == server_run_id(second)


def test_push_run_serializes_datetimes(make_client, fake_api):
    client = make_client()
    ts = datetime(2026, 7, 11, 3, 0, tzinfo=timezone.utc)
    result = client.push_run(_run(ts=ts, scheduled_start=ts))
    assert result.ok
    # The request body carried ISO strings, not a datetime the JSON encoder rejects.
    runs_req = [r for r in fake_api.requests if r.url.path == "/api/v1/runs"][0]
    body = runs_req.content.decode()
    assert "2026-07-11T03:00:00+00:00" in body


# --- push_samples ---------------------------------------------------------

def test_push_samples_success(make_client, fake_api):
    client = make_client()
    sid = server_run_id(client.push_run(_run()))
    samples = [
        {"ts": "2026-07-11T03:00:00+00:00", "power_w": 100.0},
        {"ts": "2026-07-11T03:00:01+00:00", "power_w": 110.0},
    ]
    result = client.push_samples(sid, samples)
    assert result.ok
    assert result.status_code == 201
    assert result.data["accepted"] == 2
    assert len(fake_api.samples[sid]) == 2


def test_push_samples_unknown_run_404(make_client):
    client = make_client()
    client.login()
    result = client.push_samples("00000000-0000-0000-0000-000000000000", [{"ts": "x"}])
    assert not result.ok
    assert result.status_code == 404


# --- pull_schedule --------------------------------------------------------

def test_pull_schedule_204_is_ok_with_none_data(make_client, fake_api):
    # No schedule published yet → 204, which is a SUCCESS with data=None.
    client = make_client()
    result = client.pull_schedule(after=-1)
    assert result.ok
    assert result.status_code == 204
    assert result.data is None


def test_pull_schedule_returns_newest(make_client, fake_api):
    fake_api.set_schedule(version=3)
    client = make_client()
    result = client.pull_schedule(after=1)
    assert result.ok
    assert result.status_code == 200
    assert result.data["version"] == 3


def test_pull_schedule_after_current_is_204(make_client, fake_api):
    fake_api.set_schedule(version=2)
    client = make_client()
    result = client.pull_schedule(after=2)
    assert result.ok
    assert result.status_code == 204
    assert result.data is None


# --- ack ------------------------------------------------------------------

def test_ack_started(make_client):
    client = make_client()
    result = client.ack(schedule_version=1, event="started", workflow_id="wf-1")
    assert result.ok
    assert result.data["event"] == "started"
    assert result.data["replanned"] is False


def test_ack_failed_flags_replan(make_client):
    client = make_client()
    at = datetime(2026, 7, 11, 4, 0, tzinfo=timezone.utc)
    result = client.ack(schedule_version=1, event="failed", workflow_id="wf-1", at=at)
    assert result.ok
    assert result.data["replanned"] is True


# --- create_workflow ------------------------------------------------------
#
# Not part of the execution loop — an operator action (`register`) that exists so
# the server's workflow id goes straight into jobs.json instead of being
# copy-pasted by hand, which is where a setup most often silently breaks.

def test_create_workflow_sends_name_and_framework(make_client, fake_api):
    client = make_client()
    result = client.create_workflow(name="nightly", framework="command")
    assert result.ok
    assert fake_api.created_workflows[0] == {"name": "nightly", "framework": "command"}


def test_create_workflow_omits_unset_fields(make_client, fake_api):
    """Let the server apply its own defaults rather than handing it nulls."""
    client = make_client()
    client.create_workflow(name="nightly", est_duration_s=700, deadline="by 7am")
    body = fake_api.created_workflows[0]
    assert body["est_duration_s"] == 700
    assert body["deadline"] == "by 7am"
    assert "recurrence" not in body and "nameplate_watts" not in body


def test_create_workflow_surfaces_a_validation_refusal(make_client, fake_api):
    client = make_client()
    fake_api.workflows_status = 422
    fake_api.workflows_response = {"detail": "deadline could not be parsed"}
    result = client.create_workflow(name="bad", deadline="whenever")
    assert not result.ok
    assert result.error == "deadline could not be parsed"


def test_create_workflow_is_authenticated(make_client, fake_api):
    client = make_client()
    client.create_workflow(name="nightly")
    request = [r for r in fake_api.requests if r.url.path == "/api/v1/workflows"][0]
    assert request.headers.get("Authorization", "").startswith("Bearer ")


def test_create_workflow_network_error_is_clean(make_client, fake_api):
    client = make_client()
    client.login()
    fake_api.go_down()
    result = client.create_workflow(name="nightly")
    assert not result.ok and result.transport_error is True


# --- clean-error contract -------------------------------------------------

def test_network_error_returns_clean_result_not_raise(make_client, fake_api):
    client = make_client()
    fake_api.go_down()
    result = client.push_run(_run())
    assert isinstance(result, ApiResult)
    assert not result.ok
    assert result.transport_error is True
    assert result.status_code is None


def test_pull_schedule_network_error_is_clean(make_client, fake_api):
    client = make_client()
    fake_api.go_down()
    result = client.pull_schedule()
    assert not result.ok
    assert result.transport_error is True


def test_timeout_returns_clean_result(make_client):
    def _raise_timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    client = make_client(http_client=httpx.Client(transport=httpx.MockTransport(_raise_timeout)))
    result = client.login()
    assert not result.ok
    assert result.transport_error is True
    assert "timed out" in result.error.lower()


def test_server_run_id_none_on_failed_result():
    assert server_run_id(ApiResult(ok=False)) is None
    assert server_run_id(ApiResult(ok=True, data={"runs": []})) is None
    assert server_run_id(ApiResult(ok=True, data={"no_runs": 1})) is None
