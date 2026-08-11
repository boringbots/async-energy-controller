"""
Tests for Settings (hmasync_controller/config.py).

The load-bearing one is `test_job_catalog_comes_from_env_file`: HM_ASYNC_JOB_CATALOG
used to be read straight off `os.environ`, which meant a line in `.env` — where the
quickstart puts every other knob — was silently dropped. pydantic-settings reads
`.env` into this object; it does NOT export to the process environment, so an
undeclared name had nowhere to land. The controller then authenticated, polled, and
executed nothing, presenting as "the optimizer isn't scheduling anything" rather
than as a config error.
"""

from __future__ import annotations

import os

from hmasync_controller.config import Settings, resolve_controller_id


def _write_env(tmp_path, body: str):
    path = tmp_path / ".env"
    path.write_text(body)
    return path


def test_job_catalog_is_a_declared_field():
    """Undeclared names are dropped by extra='ignore' — this must be declared."""
    assert "HM_ASYNC_JOB_CATALOG" in Settings.model_fields


def test_job_catalog_comes_from_env_file(tmp_path, monkeypatch):
    """A `.env` line sets it — the exact case that used to be a silent no-op."""
    monkeypatch.delenv("HM_ASYNC_JOB_CATALOG", raising=False)
    env = _write_env(tmp_path, "HM_ASYNC_JOB_CATALOG=/srv/jobs/nightly.json\n")

    settings = Settings(_env_file=str(env))

    assert settings.HM_ASYNC_JOB_CATALOG == "/srv/jobs/nightly.json"
    # And it is still absent from os.environ — proving the value arrived via the
    # settings object rather than by anything exporting it.
    assert "HM_ASYNC_JOB_CATALOG" not in os.environ


def test_real_environment_variable_beats_env_file(tmp_path, monkeypatch):
    """An exported shell variable still wins, as it did before."""
    env = _write_env(tmp_path, "HM_ASYNC_JOB_CATALOG=/from/dotenv.json\n")
    monkeypatch.setenv("HM_ASYNC_JOB_CATALOG", "/from/shell.json")

    assert Settings(_env_file=str(env)).HM_ASYNC_JOB_CATALOG == "/from/shell.json"


def test_job_catalog_defaults_to_jobs_json(monkeypatch):
    monkeypatch.delenv("HM_ASYNC_JOB_CATALOG", raising=False)
    assert Settings(_env_file=None).HM_ASYNC_JOB_CATALOG == "jobs.json"


def test_credentials_and_catalog_come_from_the_same_file(tmp_path, monkeypatch):
    """What made the original bug expensive: credentials in .env DID work.

    Login succeeded and the loop polled happily, so the failure never looked like
    a config problem. Both must now come from the one file.
    """
    monkeypatch.delenv("HM_ASYNC_JOB_CATALOG", raising=False)
    env = _write_env(tmp_path, (
        "HM_ASYNC_EMAIL=me@example.com\n"
        "HM_ASYNC_PASSWORD=s3cret\n"
        "HM_ASYNC_JOB_CATALOG=/srv/jobs.json\n"
    ))

    settings = Settings(_env_file=str(env))

    assert settings.HM_ASYNC_EMAIL == "me@example.com"
    assert settings.HM_ASYNC_JOB_CATALOG == "/srv/jobs.json"


def test_empty_env_still_constructs(monkeypatch):
    """The empty-environment policy: every value defaults, nothing blocks import."""
    for key in ("HM_ASYNC_API_URL", "HM_ASYNC_EMAIL", "HM_ASYNC_PASSWORD",
                "CONTROLLER_ID", "HM_ASYNC_JOB_CATALOG"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.HM_ASYNC_API_URL == ""
    assert settings.HM_ASYNC_JOB_CATALOG == "jobs.json"


def test_resolve_controller_id_prefers_configured():
    assert resolve_controller_id("box-42") == "box-42"


def test_resolve_controller_id_falls_back_to_hostname():
    assert resolve_controller_id("") != ""
