"""
ApiClient — the controller's side of the wire contract.

The controller is a client of exactly four concerns on the optimizer API:

    POST /api/v1/runs                    push a RunRecord (or a batch on drain)
    POST /api/v1/runs/{id}/samples       push a run's telemetry trace (batched)
    GET  /api/v1/schedule?after=<v>      pull the newest schedule if newer
    POST /api/v1/schedule/ack            echo job started/finished/failed

plus owner authentication via the API's `/auth` proxies: the controller
authenticates as its owner with a JWT + refresh token.

Three routes outside that loop, first reached only by an explicit
operator/library call rather than the daemon's own schedule-execution logic:

    POST /api/v1/advise                  ask for a window without registering (sdk)
    POST /api/v1/workflows               register a workload (`register`)
    POST /api/v1/bench/submissions       submit a bench bundle (`bench submit`/`quick`;
                                          a queued one may later retry on the daemon's
                                          tick cadence via bench.drain_bench_spool)

**Clean-error contract (load-bearing):** every public method returns an
`ApiResult` and NEVER raises. Network failures, timeouts, non-2xx responses, and
malformed bodies all come back as `ApiResult(ok=False,...)`. The executor loop
 and the spool drain (reporter.py) branch on `.ok`; a raw
`httpx`/JSON exception must never escape into that loop.

**Auth refresh:** an authed call that comes back 401 triggers a single refresh
(then a fresh login if refresh fails) and one retry — so a short-lived access
token is renewed transparently without the caller knowing.

Transport is an injected `httpx.Client` (defaults to one this client owns), so
tests drive it with an `httpx.MockTransport` — real request/response code, no
live vendor call (repo-wide mock-all-vendors rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel

DEFAULT_TIMEOUT_S = 10.0


@dataclass
class ApiResult:
    """The outcome of one wire call — the clean-error return type.

    `ok` is the only thing callers must branch on. `status_code` is None for a
    transport-level failure (no HTTP response arrived). `data` is the parsed JSON
    body when there was one (None on 204 or an unparseable body). `error` is a
    human-readable reason on failure.
    """

    ok: bool
    status_code: int | None = None
    data: Any = None
    error: str | None = None
    # True when the failure was a transport-level problem (unreachable / timeout)
    # rather than an HTTP error response — the reporter treats this as "spool it".
    transport_error: bool = False


def _jsonable(value: Any) -> Any:
    """Recursively convert a payload into JSON-serializable primitives.

    The controller builds run/sample payloads that may carry tz-aware datetimes
    (the model guarantees no naive value) and pydantic models; both
    are normalized here so `httpx`'s JSON encoder and `sqlite` storage handle
    them uniformly. Datetimes become ISO-8601 strings the API parses straight
    back into `AwareDT`.
    """
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class ApiClient:
    """Authenticated client for the hm-async optimizer API."""

    def __init__(
        self,
        base_url: str,
        email: str = "",
        password: str = "",
        controller_id: str = "",
        *,
        api_key: str = "",
        timeout: float = DEFAULT_TIMEOUT_S,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.email = email
        self.password = password
        # An API key authenticates every request directly and never expires, so
        # it needs no login, no refresh, and no stored session. It takes
        # precedence over email/password when both are supplied. This is the
        # credential automation should hold: it is scoped to one caller and can
        # be revoked without changing a human's password.
        self.api_key = api_key
        self.controller_id = controller_id
        self.timeout = timeout
        # Own the client unless one is injected (tests inject a MockTransport
        # client). An owned client is closed by close()/the context manager.
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout)
        self._access_token: str | None = None
        self._refresh_token: str | None = None

    # --- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- low-level send ---------------------------------------------------

    def _send(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict | None = None,
        auth: bool = False,
    ) -> ApiResult:
        """Send one request and parse it into an ApiResult. Never raises."""
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {}
        if auth and self.api_key:
            headers["X-API-Key"] = self.api_key
        elif auth and self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        try:
            resp = self._http.request(
                method,
                url,
                json=_jsonable(json) if json is not None else None,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            return ApiResult(
                ok=False, error=f"request timed out: {exc}", transport_error=True
            )
        except httpx.RequestError as exc:
            return ApiResult(
                ok=False, error=f"network error: {exc}", transport_error=True
            )
        except Exception as exc:  # defensive: nothing escapes to the loop
            return ApiResult(ok=False, error=f"unexpected error: {exc}")
        return _parse_response(resp)

    # --- authentication ---------------------------------------------------

    def login(self) -> ApiResult:
        """Exchange email + password for an access + refresh token pair.

        Stores the tokens on success. Missing credentials return a clean
        `ApiResult(ok=False)` rather than attempting a doomed round-trip.
        """
        if not self.email or not self.password:
            return ApiResult(ok=False, error="No credentials configured")
        result = self._send(
            "POST",
            "/auth/login",
            json={"email": self.email, "password": self.password},
        )
        if result.ok:
            self._store_tokens(result.data)
        return result

    def _refresh(self) -> bool:
        """Exchange the refresh token for a fresh pair. Returns success."""
        if not self._refresh_token:
            return False
        result = self._send(
            "POST",
            "/auth/refresh",
            json={"refresh_token": self._refresh_token},
        )
        if result.ok:
            self._store_tokens(result.data)
            return self._access_token is not None
        return False

    def _store_tokens(self, data: Any) -> None:
        """Pull access/refresh tokens out of an auth response body.

        They usually arrive at the top level; some flows nest them under a
        `session` object. Accept either and keep any token we already hold if the
        body omits it (a refresh response always carries both).
        """
        if not isinstance(data, dict):
            return
        session = data.get("session") if isinstance(data.get("session"), dict) else data
        access = session.get("access_token")
        refresh = session.get("refresh_token")
        if access:
            self._access_token = access
        if refresh:
            self._refresh_token = refresh

    def _ensure_auth(self) -> ApiResult | None:
        """Ensure we hold an access token, logging in if needed.

        Returns None when authenticated; a failed-login ApiResult otherwise, so
        the caller can surface it without a request going out.
        """
        if self._access_token:
            return None
        result = self.login()
        return None if result.ok else result

    def _authed_request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict | None = None,
    ) -> ApiResult:
        """Send an authenticated request, renewing the token once on a 401."""
        # An API key has no session to establish or renew — a 401 means the key
        # is wrong or revoked, and retrying cannot help. Return it as-is so the
        # caller sees the real error instead of a confusing login failure.
        if self.api_key:
            return self._send(method, path, json=json, params=params, auth=True)

        pre = self._ensure_auth()
        if pre is not None:
            return pre

        result = self._send(method, path, json=json, params=params, auth=True)
        if result.status_code != 401:
            return result

        # Access token rejected: refresh (or re-login) once, then retry once.
        if self._refresh() or self.login().ok:
            return self._send(method, path, json=json, params=params, auth=True)
        return result

    # --- wire endpoints ---------------------------------------

    def advise(
        self,
        *,
        est_duration_s: float,
        deadline: str,
        earliest_start: str | None = None,
        nameplate_watts: float | None = None,
        framework: str = "command",
        fingerprint: str | None = None,
    ) -> ApiResult:
        """Ask when this work should run. Stateless — the server stores nothing.

        The one call a caller needs when it already has its own orchestrator and
        wants a time rather than a daemon.
        """
        body: dict[str, Any] = {
            "est_duration_s": est_duration_s,
            "deadline": deadline,
            "framework": framework,
        }
        if earliest_start is not None:
            body["earliest_start"] = earliest_start
        if nameplate_watts is not None:
            body["nameplate_watts"] = nameplate_watts
        if fingerprint is not None:
            body["fingerprint"] = fingerprint
        return self._authed_request("POST", "/api/v1/advise", json=body)

    def create_workflow(
        self,
        *,
        name: str,
        framework: str = "command",
        request: dict[str, Any] | None = None,
        est_duration_s: float | None = None,
        deadline: str | None = None,
        earliest_start: str | None = None,
        recurrence: str | None = None,
        nameplate_watts: float | None = None,
        enabled: bool | None = None,
    ) -> ApiResult:
        """POST /api/v1/workflows — register a workload and get its id back (201).

        NOT part of the execution loop. The four endpoints above are what the
        daemon speaks while running; this is an operator action invoked by hand
        (`register`), and keeping it here rather than in a shell script is what
        stops the returned UUID from being copy-pasted into `jobs.json` by a human.

        `deadline`/`earliest_start` are HUMAN strings ("by 7am", "22:00",
        "2026-07-20T09:00"), resolved by the API in the account's timezone — they
        are not the ISO datetimes the local catalog takes. `recurrence` is one of
        none/daily/weekly. An unparseable string is refused here, at registration,
        with a message naming the field, so a typo surfaces immediately.

        `request` defaults to omitted. What this box runs is described by the local
        job catalog, and the controller does not upload it unasked; callers who
        want the API to hold a copy pass it explicitly.
        """
        body: dict[str, Any] = {"name": name, "framework": framework}
        optional = {
            "request": request,
            "est_duration_s": est_duration_s,
            "deadline": deadline,
            "earliest_start": earliest_start,
            "recurrence": recurrence,
            "nameplate_watts": nameplate_watts,
            "enabled": enabled,
        }
        body.update({k: v for k, v in optional.items() if v is not None})
        return self._authed_request("POST", "/api/v1/workflows", json=body)

    def push_run(self, record: Any) -> ApiResult:
        """POST /api/v1/runs — push one run record (or a batch: pass a list).

        On success `data` is the API's `{accepted, duplicates, runs:[...]}` body;
        `server_run_id(result)` extracts the run's server UUID (the same value
        for a freshly-inserted and an idempotent-duplicate run), which the samples
        endpoint is addressed by.
        """
        return self._authed_request("POST", "/api/v1/runs", json=record)

    def push_samples(self, server_run_id: str, samples: list) -> ApiResult:
        """POST /api/v1/runs/{server_run_id}/samples — append a trace batch.

        `server_run_id` is the run's SERVER UUID from a prior push_run response,
        NOT the client run_id — they are two distinct ids.
        """
        return self._authed_request(
            "POST",
            f"/api/v1/runs/{server_run_id}/samples",
            json={"samples": list(samples)},
        )

    def pull_schedule(self, after: int = -1) -> ApiResult:
        """GET /api/v1/schedule?after=<version> — the newest schedule if newer.

        A 204 (no newer schedule) is a SUCCESS with `data=None` and
        `status_code=204` — the caller keeps running its current version. Only a
        200 carries a schedule body in `data`.
        """
        return self._authed_request(
            "GET", "/api/v1/schedule", params={"after": after}
        )

    def ack(
        self,
        schedule_version: int,
        event: str,
        workflow_id: str | None = None,
        at: datetime | None = None,
    ) -> ApiResult:
        """POST /api/v1/schedule/ack — echo a job's applied state.

        `event` is one of started/finished/failed. A `failed` event drives an
        event-driven re-plan on the API side.
        """
        body: dict[str, Any] = {
            "schedule_version": schedule_version,
            "event": event,
            "workflow_id": workflow_id,
        }
        if at is not None:
            body["at"] = at
        return self._authed_request("POST", "/api/v1/schedule/ack", json=body)

    def submit_bench_bundle(self, bundle: dict[str, Any]) -> ApiResult:
        """POST /api/v1/bench/submissions — submit one measured energy-bench bundle.

        Not part of the execution loop; invoked by `bench submit` / `bench
        quick`'s opt-in hand-off (cli.py / bench.py), never by the running
        daemon directly. A quarantined submission is still a 2xx (accepted,
        held for operator review) — quarantine is a server-side review state,
        not a client-visible failure, so no special-casing is needed beyond
        the normal 2xx/4xx split `_parse_response` already does.
        """
        return self._authed_request("POST", "/api/v1/bench/submissions", json=bundle)


def _parse_response(resp: httpx.Response) -> ApiResult:
    """Turn an httpx.Response into an ApiResult (204/2xx/4xx/5xx, bad JSON)."""
    if resp.status_code == 204:
        return ApiResult(ok=True, status_code=204, data=None)

    data: Any
    try:
        data = resp.json()
    except Exception:
        data = None

    if resp.status_code >= 400:
        detail = None
        if isinstance(data, dict):
            detail = data.get("detail") or data.get("error") or data.get("message")
        return ApiResult(
            ok=False,
            status_code=resp.status_code,
            data=data,
            error=detail or f"HTTP {resp.status_code}",
        )

    return ApiResult(ok=True, status_code=resp.status_code, data=data)


def server_run_id(result: ApiResult) -> str | None:
    """Extract the run's server UUID from a push_run ApiResult, or None.

    The push_run response is `{runs: [{run_id, controller_id, id, status},...]}`;
    for a single-run push the server id is `runs[0].id`. Works for both
    `status='inserted'` and `status='duplicate'` (idempotent re-push returns the
    same id), which is what makes a spool re-drain safe.
    """
    if not result.ok or not isinstance(result.data, dict):
        return None
    runs = result.data.get("runs")
    if isinstance(runs, list) and runs and isinstance(runs[0], dict):
        return runs[0].get("id")
    return None
