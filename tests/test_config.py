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

from hmasync_controller.config import (
    BENCH_CONSENT_TEXT,
    Settings,
    resolve_controller_id,
    set_bench_optin,
)


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


# ============================================================
# Bench opt-in settings + consent (US-ONB-02)
# ============================================================


def test_bench_settings_default_off(monkeypatch):
    for key in ("BENCH_OPTIN", "BENCH_BUNDLE_DIR", "ENERGY_BENCH_CMD"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.BENCH_OPTIN is False
    assert settings.BENCH_BUNDLE_DIR == "bench_bundles"
    assert settings.ENERGY_BENCH_CMD == "eb"


def test_bench_optin_comes_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCH_OPTIN", raising=False)
    env = _write_env(tmp_path, "BENCH_OPTIN=true\n")
    assert Settings(_env_file=str(env)).BENCH_OPTIN is True


def test_node_salt_path_default(monkeypatch):
    monkeypatch.delenv("NODE_SALT_PATH", raising=False)
    assert Settings(_env_file=None).NODE_SALT_PATH == "hmasync_node_salt"


def test_consent_text_states_what_is_shared_and_not():
    text = BENCH_CONSENT_TEXT.lower()
    # What a submission contains.
    for included in ("hardware fingerprint", "version", "benchmark metrics"):
        assert included in text
    # What it explicitly does not contain.
    for excluded in ("prompts", "commands", "workflow"):
        assert excluded in text
    # The denylisted identity fields (also enforced upstream in US-ONB-05).
    for identity_field in ("uuid", "serial", "mac address", "hostname", "entity id"):
        assert identity_field in text
    # The data-license line.
    assert "license" in text


def test_set_bench_optin_writes_a_fresh_env_file(tmp_path):
    env = tmp_path / ".env"
    set_bench_optin(True, env)
    assert "BENCH_OPTIN=true" in env.read_text().splitlines()


def test_set_bench_optin_updates_in_place_without_disturbing_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("HM_ASYNC_EMAIL=me@example.com\nBENCH_OPTIN=false\nCONTROLLER_ID=box-1\n")

    set_bench_optin(True, env)

    lines = env.read_text().splitlines()
    assert "BENCH_OPTIN=true" in lines
    assert "HM_ASYNC_EMAIL=me@example.com" in lines
    assert "CONTROLLER_ID=box-1" in lines
    assert lines.count("BENCH_OPTIN=false") == 0
    # Only ever one BENCH_OPTIN line.
    assert sum(1 for line in lines if line.startswith("BENCH_OPTIN=")) == 1


def test_set_bench_optin_preserves_comments(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# a header a human wrote\nHM_ASYNC_EMAIL=me@example.com\n")

    set_bench_optin(True, env)

    text = env.read_text()
    assert "# a header a human wrote" in text
    assert "BENCH_OPTIN=true" in text


def test_set_bench_optin_round_trips_through_settings(tmp_path):
    env = tmp_path / ".env"
    set_bench_optin(True, env)
    assert Settings(_env_file=str(env)).BENCH_OPTIN is True

    set_bench_optin(False, env)
    assert Settings(_env_file=str(env)).BENCH_OPTIN is False


def test_set_bench_optin_writes_atomically(tmp_path):
    env = tmp_path / ".env"
    set_bench_optin(True, env)
    assert not (tmp_path / ".env.tmp").exists()
