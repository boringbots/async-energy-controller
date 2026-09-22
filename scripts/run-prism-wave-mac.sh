#!/usr/bin/env bash
# The prism wave on an Apple Silicon Mac: Bonsai's sub-4-bit weights, one
# llama-server per rung, metered through `bench prism`.
#
#   scripts/run-prism-wave-mac.sh list           # the plan and its download size
#   scripts/run-prism-wave-mac.sh fetch          # binary + every GGUF, pinned
#   scripts/run-prism-wave-mac.sh run            # the whole wave, rung by rung
#   scripts/run-prism-wave-mac.sh run bonsai2-27b-ptq1-0 bonsai2-27b-pq2-0
#
# WHY A SHELL SCRIPT AND NOT A MODE THAT LOOPS. `bench prism` attaches to a
# server it did not start -- every engine adapter in this package is
# attach-only, deliberately, and llama.cpp has no swap-the-model request. One
# rung is one server process, so the loop belongs out here where processes
# live.
#
# FETCHING IS A SEPARATE VERB, never part of `run`. Pulling weights inside a
# measured window would put a download's energy in a model's row, and pulling
# anything at all is an explicit operator act (standing rule). `run` fails on
# a missing file rather than reaching for it.
#
# THE RUNG TABLE IS NOT DUPLICATED HERE. It is read out of
# `hmasync_controller.bench.prism` at run time, so this script cannot drift
# from the table the measurement itself verifies against.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
CONTROLLER="${CONTROLLER:-async-energy-controller}"
PRISM_DIR="${PRISM_DIR:-$PWD/prism-mac}"
PRISM_BIN="${PRISM_BIN:-$PRISM_DIR/bin/llama-server}"
PRISM_MODELS_DIR="${PRISM_MODELS_DIR:-$PRISM_DIR/models}"
PRISM_LOG_DIR="${PRISM_LOG_DIR:-$PRISM_DIR/logs}"
PORT="${PRISM_PORT:-8091}"
# Between rungs. The card, or here the SoC, ends a rung hot, and the next
# rung's first minutes would be measured on the previous rung's heat. Apple
# reports no thermal-throttle bit at all (APPLE-SILICON.md), so this is the
# only throttle defence the wave has.
COOLDOWN_S="${PRISM_COOLDOWN_S:-180}"

# Release with a prebuilt macos-arm64 llama-server. Pinned: a tag is the
# provenance of every row this wave produces. 22 commits ahead of the lab's
# 5d80cff, none of them under ggml/ -- same quantization kernels.
PRISM_RELEASE="${PRISM_RELEASE:-prism-b10709-9a9394a}"
PRISM_TARBALL="llama-${PRISM_RELEASE}-bin-macos-arm64.tar.gz"
PRISM_URL="https://github.com/PrismML-Eng/llama.cpp/releases/download/${PRISM_RELEASE}/${PRISM_TARBALL}"

CTX_SIZE="$("$PYTHON" -c 'from hmasync_controller.bench.prism import PRISM_CTX_SIZE; print(PRISM_CTX_SIZE)' 2>/dev/null || true)"
if [[ -z "$CTX_SIZE" ]]; then
  echo "cannot import hmasync_controller -- activate the venv this package is" >&2
  echo "installed in first (python3 -m venv .venv && . .venv/bin/activate &&" >&2
  echo "pip install -e .), or set PYTHON to that interpreter." >&2
  exit 2
fi

# rung_key|gguf_repo|revision|gguf_file|size_gb|stage, straight from the table
# the mode itself checks against.
rung_table() {
  "$PYTHON" - <<'PY'
from hmasync_controller.bench.prism import PRISM_RUNGS
for r in PRISM_RUNGS:
    print("|".join([r.key, r.gguf_repo, r.gguf_revision, r.gguf_file, f"{r.size_gb:.2f}", r.stage]))
PY
}

log() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

cmd_list() {
  "$CONTROLLER" bench prism --list
  echo
  echo "binary:  $PRISM_URL"
  echo "weights: $PRISM_MODELS_DIR"
  echo "logs:    $PRISM_LOG_DIR"
}

cmd_fetch() {
  mkdir -p "$PRISM_DIR/bin" "$PRISM_MODELS_DIR"

  if [[ -x "$PRISM_BIN" ]]; then
    log "have    $(basename "$PRISM_BIN")"
  else
    log "fetch   $PRISM_TARBALL"
    tmp="$(mktemp -d)"
    curl -fsSL "$PRISM_URL" -o "$tmp/$PRISM_TARBALL"
    tar -xzf "$tmp/$PRISM_TARBALL" -C "$tmp"
    # The release lays out build/bin/; take the whole bin directory, since
    # llama-server loads the shared ggml libraries beside it.
    # No -perm test: BSD find (what macOS ships) spells the symbolic form
    # differently from GNU, and the release tarball has exactly one
    # llama-server in it.
    found="$(find "$tmp" -name llama-server -type f | head -1)"
    [[ -n "$found" ]] || { echo "no llama-server in $PRISM_TARBALL" >&2; exit 1; }
    cp -R "$(dirname "$found")/." "$PRISM_DIR/bin/"
    rm -rf "$tmp"
    log "        installed to $PRISM_DIR/bin"
  fi

  command -v hf >/dev/null || {
    echo "the 'hf' CLI is not on PATH (pip install huggingface-hub[cli])" >&2
    exit 1
  }
  while IFS='|' read -r key repo rev file gb _stage; do
    if [[ -s "$PRISM_MODELS_DIR/$file" ]]; then
      log "have    $file"
      continue
    fi
    log "fetch   $file  (${gb} GB from $repo @ ${rev:0:12})"
    # --revision on every file: a quantization name is not a set of weights,
    # and a re-upload under one filename is what this pins against.
    hf download "$repo" "$file" --revision "$rev" --local-dir "$PRISM_MODELS_DIR"
  done < <(rung_table)

  log "done. next: $0 run"
}

SERVER_PID=""

port_is_free() {
  ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

start_server() {
  local file="$1" logfile="$2"
  # --fit off: llama.cpp's auto-fit would silently shrink the window instead
  # of failing, and rungs that differ in both storage and window measure
  # nothing. -ngl 999 puts every layer on Metal; on unified memory there is no
  # split worth measuring here, and a partial offload would be a second axis.
  #
  # NOT in a command substitution: that would make the server a child of a
  # subshell that then exits, so `wait` below would return immediately and the
  # next rung would start while this one still held the memory.
  "$PRISM_BIN" \
    -m "$PRISM_MODELS_DIR/$file" \
    -c "$CTX_SIZE" \
    --fit off \
    -ngl 999 \
    --host 127.0.0.1 \
    --port "$PORT" \
    >"$logfile" 2>&1 &
  SERVER_PID=$!
}

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  # A real wait: the next rung must not start until this model is out of
  # memory, or the two overlap and both readings are wrong.
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

wait_for_health() {
  for _ in $(seq 1 600); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1  # the server died -- for a rung that does not fit, that IS the result
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

cmd_run() {
  local wanted=("$@")
  mkdir -p "$PRISM_LOG_DIR"

  local ran=0 failed=0
  while IFS='|' read -r key repo rev file gb stage; do
    if ((${#wanted[@]})); then
      local match=0
      # `[[ ... ]] && match=1` alone would be the body's last command, and a
      # false condition then takes `set -e` with it -- which is the COMMON
      # case here, since most rungs do not match the first name given.
      for w in "${wanted[@]}"; do
        if [[ "$w" == "$key" ]]; then match=1; fi
      done
      ((match)) || continue
    fi

    if [[ ! -s "$PRISM_MODELS_DIR/$file" ]]; then
      log "SKIP    $key -- $file is not downloaded ($0 fetch)"
      failed=$((failed + 1))
      continue
    fi

    log "=== $key  ($stage, ${gb} GB) ==="
    if ! port_is_free; then
      log "ABORT   something is already serving on port ${PORT}."
      log "        This wave uses its own port so a rung can never attach to"
      log "        another server and produce a plausible wrong row. Stop that"
      log "        server, or set PRISM_PORT to a free one."
      return 1
    fi
    local serverlog="$PRISM_LOG_DIR/${key}-server.log"
    start_server "$file" "$serverlog"
    # Kill the server however this rung ends -- a leftover process holds the
    # whole SoC and would both starve and heat the next rung.
    trap stop_server EXIT

    if ! wait_for_health; then
      log "        server never became healthy -- see $serverlog"
      log "        (a rung that cannot fit is a RESULT: record it and move on)"
      stop_server
      trap - EXIT
      failed=$((failed + 1))
      continue
    fi

    set +e
    "$CONTROLLER" bench prism --rung "$key" --port "$PORT" \
      2>&1 | tee "$PRISM_LOG_DIR/${key}-bench.log"
    local rc=${PIPESTATUS[0]}
    set -e

    stop_server
    trap - EXIT

    if ((rc == 0)); then
      ran=$((ran + 1))
    else
      log "        rung exited $rc -- see $PRISM_LOG_DIR/${key}-bench.log"
      failed=$((failed + 1))
    fi

    log "        cooling down ${COOLDOWN_S}s"
    sleep "$COOLDOWN_S"
  done < <(rung_table)

  log "wave done: $ran measured, $failed not"
  echo
  echo "compare the rungs:  $PYTHON scripts/compare-prism-rungs.py"
  ((failed == 0))
}

case "${1:-}" in
  list)  cmd_list ;;
  fetch) cmd_fetch ;;
  run)   shift; cmd_run "$@" ;;
  *)
    echo "usage: $0 {list|fetch|run [rung ...]}" >&2
    exit 2
    ;;
esac
