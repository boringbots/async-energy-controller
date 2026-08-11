# Security

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Report it to
**info@hungrymachines.io** with enough detail to reproduce. You will get an
acknowledgement, and a fix or an explanation of why it is not a vulnerability.

## What this software has access to

Worth being explicit, because you are installing a daemon that runs commands:

- **It executes what your local `jobs.json` says**, and only that. Scheduled
  work arrives as a workflow id; the controller resolves that id against your
  local catalog. An id with no local entry is skipped. The server cannot supply
  a command to run.
- **It runs as whatever user you give it.** The example systemd unit uses a
  dedicated `hmasync` user with `NoNewPrivileges=true` — use it, and give that
  user only the access your jobs actually need. A `command` adapter entry runs
  with that user's full permissions, so treat `jobs.json` as you would a
  crontab.
- **It stores credentials in `.env`** in plaintext, like most daemons. That file
  is gitignored here; keep it `chmod 600` and owned by the service user.
- **It talks to exactly one host**, the `HM_ASYNC_API_URL` you configure. There
  is no analytics, telemetry, or third-party reporting.
- **It spools run records to a local SQLite file** when the API is unreachable.
  That file contains job metadata and energy measurements — no credentials, no
  job output.

## Scope

In scope: anything that lets a remote party influence what this box executes,
escalate privileges, read credentials, or exfiltrate data.

Out of scope: the fact that a `command` adapter entry runs arbitrary shell — you
put it there, and that is the design.
