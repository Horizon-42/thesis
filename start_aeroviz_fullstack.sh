#!/usr/bin/env bash
# Supervisor launcher for the AeroViz full stack (Python backend + Vite frontend).
#
# Each service runs as a background child. If ONE dies, the supervisor restarts
# just that one and leaves the other running — it no longer co-kills both (the
# old `while kill -0 A && kill -0 B` loop did, so a single backend crash, e.g. a
# casadi native abort, took the whole stack down). Crash output stays on this
# terminal, so a recurring abort is now visible instead of silently shutting
# everything off. Ctrl-C (or SIGTERM) stops both cleanly.
#
# Knobs (env): AEROVIZ_MAX_RAPID_RESTARTS (default 5), AEROVIZ_MIN_HEALTHY_S
# (default 8) — a service that keeps dying within MIN_HEALTHY_S of starting,
# MAX_RAPID_RESTARTS times in a row, is treated as a hard failure and the stack
# shuts down (so a boot-time misconfig doesn't spin forever).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AEROVIZ_APP_DIR="${AEROVIZ_APP_DIR:-$ROOT_DIR/aeroviz-4d}"
AEROVIZ_PUBLIC_DATA_DIR="$AEROVIZ_APP_DIR/public/data"
AEROVIZ_DIST_DATA_PATH="$AEROVIZ_APP_DIR/dist/data"
AEROVIZ_BACKEND_HOST="${AEROVIZ_BACKEND_HOST:-127.0.0.1}"
AEROVIZ_BACKEND_PORT="${AEROVIZ_BACKEND_PORT:-8765}"
VITE_HOST="${VITE_HOST:-127.0.0.1}"
VITE_PORT="${VITE_PORT:-5173}"
PYTHON_BIN="${PYTHON_BIN:-/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python}"
MAX_RAPID_RESTARTS="${AEROVIZ_MAX_RAPID_RESTARTS:-5}"
MIN_HEALTHY_S="${AEROVIZ_MIN_HEALTHY_S:-8}"

BACKEND_PID=""
VITE_PID=""
BACKEND_STARTED_AT=0
VITE_STARTED_AT=0
BACKEND_RAPID_FAILS=0
VITE_RAPID_FAILS=0
SHUTTING_DOWN=0

log() { echo "[supervisor] $*"; }

# Kill a process and its direct children (npm spawns node/vite/esbuild).
kill_tree() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
  (( SHUTTING_DOWN )) && return 0
  SHUTTING_DOWN=1
  log "shutting down…"
  kill_tree "$BACKEND_PID"
  kill_tree "$VITE_PID"
}
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT

start_backend() {
  "$PYTHON_BIN" "$ROOT_DIR/aeroviz_backend/http_server.py" \
    --host "$AEROVIZ_BACKEND_HOST" \
    --port "$AEROVIZ_BACKEND_PORT" &
  BACKEND_PID=$!
  BACKEND_STARTED_AT=$SECONDS
  log "backend started (pid $BACKEND_PID)"
}

start_frontend() {
  ( cd "$AEROVIZ_APP_DIR" && \
    VITE_AEROVIZ_BACKEND_URL="${VITE_AEROVIZ_BACKEND_URL:-http://${AEROVIZ_BACKEND_HOST}:${AEROVIZ_BACKEND_PORT}}" \
    exec npm run dev -- --host "$VITE_HOST" --port "$VITE_PORT" ) &
  VITE_PID=$!
  VITE_STARTED_AT=$SECONDS
  log "frontend started (pid $VITE_PID)"
}

# Decide whether to restart a service after it exited. Returns 0 (restart) or 1
# (give up). A death within MIN_HEALTHY_S of starting counts as a rapid failure;
# MAX_RAPID_RESTARTS of those in a row means a real boot problem, so we stop.
should_restart() {
  local name="$1" code="$2" started_at="$3" rapid_var="$4"
  local uptime=$(( SECONDS - started_at ))
  local rapid="${!rapid_var}"
  if (( uptime < MIN_HEALTHY_S )); then
    rapid=$(( rapid + 1 ))
  else
    rapid=0
  fi
  printf -v "$rapid_var" '%s' "$rapid"
  if (( rapid > MAX_RAPID_RESTARTS )); then
    log "$name died (exit $code) after ${uptime}s — ${rapid} rapid failures in a row, giving up"
    return 1
  fi
  log "$name died (exit $code) after ${uptime}s — restarting (rapid streak ${rapid}/${MAX_RAPID_RESTARTS})"
  return 0
}

if [[ ! -d "$AEROVIZ_PUBLIC_DATA_DIR" ]]; then
  echo "Missing AeroViz public data directory: $AEROVIZ_PUBLIC_DATA_DIR" >&2
  exit 1
fi

if [[ -d "$AEROVIZ_DIST_DATA_PATH" && ! -L "$AEROVIZ_DIST_DATA_PATH" ]]; then
  echo "Note: $AEROVIZ_DIST_DATA_PATH is an old copied build artifact." >&2
  echo "Run npm run build in $AEROVIZ_APP_DIR, or remove dist/data, to reclaim the duplicate storage." >&2
fi

start_backend
start_frontend

echo "AeroViz backend: http://${AEROVIZ_BACKEND_HOST}:${AEROVIZ_BACKEND_PORT}"
echo "Frontend: http://${VITE_HOST}:${VITE_PORT}"
echo "AeroViz data: $AEROVIZ_PUBLIC_DATA_DIR"

# Supervise: poll each child; restart whichever exited, keep the other running.
while true; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    wait "$BACKEND_PID" 2>/dev/null; code=$?
    (( SHUTTING_DOWN )) && break
    if should_restart "backend" "$code" "$BACKEND_STARTED_AT" BACKEND_RAPID_FAILS; then
      start_backend
    else
      break
    fi
  fi
  if ! kill -0 "$VITE_PID" 2>/dev/null; then
    wait "$VITE_PID" 2>/dev/null; code=$?
    (( SHUTTING_DOWN )) && break
    if should_restart "frontend" "$code" "$VITE_STARTED_AT" VITE_RAPID_FAILS; then
      start_frontend
    else
      break
    fi
  fi
  sleep 1
done
