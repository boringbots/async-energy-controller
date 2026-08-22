"""
Shared test fixtures for the hm-async controller.

The load-bearing piece is `FakeApiServer`: an in-process stand-in for the
optimizer API built on `httpx.MockTransport`. Tests drive the real ApiClient
request/response code against it — no live vendor call (repo-wide
mock-all-vendors rule), yet the fake enforces the behaviours the controller
depends on: Bearer auth with token refresh, `(controller_id, run_id)`
idempotency on POST /runs, 204-vs-200 on GET /schedule, and a simulated outage
(`mode="down"`) that raises a transport error the way an unreachable host would.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict

import httpx
import pytest

from hmasync_controller.apiclient import ApiClient
from hmasync_controller.spool import Spool


def _json_response(status_code: int, payload) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


class FakeApiServer:
    """A scriptable fake of the hm-async optimizer API for MockTransport."""

    def __init__(self):
        self.mode = "up"  # "up" | "down" (down → transport error, i.e. unreachable)
        self.requests: list[httpx.Request] = []
        # Auth state.
        self._serial = 0
        self.valid_access: set[str] = set()
        self.valid_refresh: set[str] = set()
        # Run store: (controller_id, run_id) -> server id; server ids known.
        self.runs: dict[tuple, str] = {}
        self.server_ids: set[str] = set()
        self.samples: dict[str, list] = defaultdict(list)
        # Ordered log of (controller_id, run_id) as POST /runs handled them.
        self.run_push_order: list[tuple] = []
        # Newest schedule (or None → 204).
        self.schedule: dict | None = None
        # POST /workflows (the `register` command). Scriptable per-test because
        # what matters is how the CLI handles each shape of answer — an id under
        # any of several keys, a validation refusal, a success carrying no id.
        self.workflows_response: dict = {"id": "wf-created"}
        self.workflows_status: int = 201
        self.created_workflows: list[dict] = []
        # POST /api/v1/bench/submissions (US-ONB-04). Scriptable status/body so
        # tests can simulate a quarantined-but-accepted (2xx) response as well
        # as an outright refusal, the same way workflows_status/response works.
        self.bench_submissions: list[dict] = []
        self.bench_submission_status: int = 201
        self.bench_submission_response: dict = {"status": "accepted"}
        # POST /api/v1/bench/nodes (US-ONB-05) — same scriptable shape as the
        # two above, so a test can simulate an API that has not added this
        # route yet (404) as easily as a normal accept.
        self.registered_nodes: list[dict] = []
        self.register_node_status: int = 201
        self.register_node_response: dict = {"status": "ok"}
        # GET /api/v1/bench/nodes/{node_hash}/recommended-cap (US-ONB-06).
        # Scriptable status/body, same shape as the routes above; defaults to
        # the "no flexibility data yet" null case, which prd.json's GROUND
        # TRUTH names as the normal (not-error) response.
        self.recommended_cap_requests: list[tuple[str, dict]] = []
        self.recommended_cap_status: int = 200
        self.recommended_cap_response: dict = {"recommended_cap_w": None}

    # --- scripting helpers ------------------------------------------------

    def go_down(self) -> None:
        self.mode = "down"

    def go_up(self) -> None:
        self.mode = "up"

    def invalidate_access(self) -> None:
        """Expire every currently-valid access token (forces a 401 → refresh)."""
        self.valid_access.clear()

    def set_schedule(self, version: int, **fields) -> None:
        base = {
            "id": str(uuid.uuid4()),
            "version": version,
            "valid_until": "2026-07-12T00:00:00+00:00",
            "placements": [],
            "forecast_snapshot": [],
            "fallback_policy": "deadline_latest_start",
            "degraded": False,
        }
        base.update(fields)
        self.schedule = base

    def client(self) -> httpx.Client:
        """A real httpx.Client wired to this fake via MockTransport."""
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    # --- transport handler ------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.mode == "down":
            # A genuinely unreachable host surfaces as a transport error, not an
            # HTTP status — this is what the client/reporter must handle cleanly.
            raise httpx.ConnectError("simulated outage", request=request)

        path = request.url.path
        method = request.method
        body = self._body(request)

        if path == "/auth/login" and method == "POST":
            return self._issue_session()
        if path == "/auth/refresh" and method == "POST":
            if body.get("refresh_token") in self.valid_refresh:
                return self._issue_session()
            return _json_response(401, {"detail": "invalid refresh token"})

        # Everything below requires a valid Bearer access token.
        token = self._bearer(request)
        if token not in self.valid_access:
            return _json_response(401, {"detail": "invalid or expired token"})

        if path == "/api/v1/workflows" and method == "POST":
            self.created_workflows.append(body)
            return _json_response(self.workflows_status, self.workflows_response)
        if path == "/api/v1/bench/submissions" and method == "POST":
            self.bench_submissions.append(body)
            return _json_response(self.bench_submission_status, self.bench_submission_response)
        if path == "/api/v1/bench/nodes" and method == "POST":
            self.registered_nodes.append(body)
            return _json_response(self.register_node_status, self.register_node_response)
        if (
            path.startswith("/api/v1/bench/nodes/")
            and path.endswith("/recommended-cap")
            and method == "GET"
        ):
            node_hash = path.split("/")[5]
            self.recommended_cap_requests.append((node_hash, dict(request.url.params)))
            return _json_response(self.recommended_cap_status, self.recommended_cap_response)
        if path == "/api/v1/runs" and method == "POST":
            return self._handle_runs(body)
        if path.startswith("/api/v1/runs/") and path.endswith("/samples") and method == "POST":
            server_id = path.split("/")[4]
            return self._handle_samples(server_id, body)
        if path == "/api/v1/schedule" and method == "GET":
            after = int(request.url.params.get("after", "-1"))
            return self._handle_schedule(after)
        if path == "/api/v1/schedule/ack" and method == "POST":
            return _json_response(
                200,
                {
                    "status": "ok",
                    "event": body.get("event"),
                    "schedule_version": body.get("schedule_version"),
                    "workflow_id": body.get("workflow_id"),
                    "replanned": body.get("event") == "failed",
                },
            )
        return _json_response(404, {"detail": "not found"})

    # --- internals --------------------------------------------------------

    @staticmethod
    def _body(request: httpx.Request) -> dict:
        if not request.content:
            return {}
        try:
            return json.loads(request.content.decode())
        except (ValueError, UnicodeDecodeError):
            return {}

    @staticmethod
    def _bearer(request: httpx.Request) -> str | None:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[len("Bearer "):]
        return None

    def _issue_session(self) -> httpx.Response:
        self._serial += 1
        access = f"access-{self._serial}"
        refresh = f"refresh-{self._serial}"
        self.valid_access.add(access)
        self.valid_refresh.add(refresh)
        return _json_response(
            200,
            {
                "access_token": access,
                "refresh_token": refresh,
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {"id": "user-1", "email": "owner@example.com"},
            },
        )

    def _handle_runs(self, body) -> httpx.Response:
        records = body if isinstance(body, list) else [body]
        results = []
        accepted = 0
        duplicates = 0
        for rec in records:
            key = (rec.get("controller_id"), rec.get("run_id"))
            self.run_push_order.append(key)
            if key in self.runs:
                sid = self.runs[key]
                duplicates += 1
                status = "duplicate"
            else:
                sid = str(uuid.uuid4())
                self.runs[key] = sid
                self.server_ids.add(sid)
                accepted += 1
                status = "inserted"
            results.append(
                {
                    "run_id": rec.get("run_id"),
                    "controller_id": rec.get("controller_id"),
                    "id": sid,
                    "status": status,
                }
            )
        return _json_response(
            200, {"accepted": accepted, "duplicates": duplicates, "runs": results}
        )

    def _handle_samples(self, server_id: str, body) -> httpx.Response:
        if server_id not in self.server_ids:
            return _json_response(404, {"detail": "Run not found"})
        samples = body.get("samples", [])
        self.samples[server_id].extend(samples)
        return _json_response(201, {"accepted": len(samples)})

    def _handle_schedule(self, after: int) -> httpx.Response:
        if self.schedule is None:
            return httpx.Response(204)
        if (self.schedule.get("version") or 0) <= after:
            return httpx.Response(204)
        return _json_response(200, self.schedule)


@pytest.fixture
def fake_api() -> FakeApiServer:
    return FakeApiServer()


@pytest.fixture
def make_client(fake_api):
    """Factory: an ApiClient wired to the fake, with default owner credentials."""

    def _make(**overrides) -> ApiClient:
        kwargs = dict(
            base_url="https://api.hm-async.test",
            email="owner@example.com",
            password="s3cret",
            controller_id="box-1",
            http_client=fake_api.client(),
        )
        kwargs.update(overrides)
        return ApiClient(**kwargs)

    return _make


@pytest.fixture
def spool(tmp_path) -> Spool:
    s = Spool(str(tmp_path / "spool.db"))
    yield s
    s.close()
