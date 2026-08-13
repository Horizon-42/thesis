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
#
# Pass --replace to gracefully stop an earlier managed launcher for this same
# repository/backend-port/frontend-port tuple before starting. The identity is
# proven with a private runtime record, Linux process start time, and the open
# supervisor lock descriptor; the launcher never kills a process by name or by
# port alone.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AEROVIZ_APP_DIR="${AEROVIZ_APP_DIR:-$ROOT_DIR/aeroviz-4d}"
AEROVIZ_PUBLIC_DATA_DIR="$AEROVIZ_APP_DIR/public/data"
AEROVIZ_DIST_DATA_PATH="$AEROVIZ_APP_DIR/dist/data"
AEROVIZ_BACKEND_HOST="${AEROVIZ_BACKEND_HOST:-0.0.0.0}"
AEROVIZ_BACKEND_PORT="${AEROVIZ_BACKEND_PORT:-8765}"
VITE_HOST="${VITE_HOST:-0.0.0.0}"
VITE_PORT="${VITE_PORT:-5173}"
REPLACE_EXISTING=0
while (( $# )); do
  case "$1" in
    --replace)
      REPLACE_EXISTING=1
      ;;
    -h|--help)
      echo "Usage: $(basename "$0") [--replace]"
      echo "  --replace  Stop the previous managed AeroViz instance for these ports."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $(basename "$0") [--replace]" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! "$AEROVIZ_BACKEND_PORT" =~ ^[0-9]+$ ]] ||
   (( AEROVIZ_BACKEND_PORT < 1 || AEROVIZ_BACKEND_PORT > 65535 )); then
  echo "Invalid AEROVIZ_BACKEND_PORT: $AEROVIZ_BACKEND_PORT" >&2
  exit 2
fi
if [[ ! "$VITE_PORT" =~ ^[0-9]+$ ]] || (( VITE_PORT < 1 || VITE_PORT > 65535 )); then
  echo "Invalid VITE_PORT: $VITE_PORT" >&2
  exit 2
fi

command -v flock >/dev/null 2>&1 || {
  echo "Missing required command: flock" >&2
  exit 1
}
command -v setsid >/dev/null 2>&1 || {
  echo "Missing required command: setsid" >&2
  exit 1
}

RUNTIME_BASE="${AEROVIZ_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/tmp}/aeroviz-4d-${UID}}"
if [[ -L "$RUNTIME_BASE" ]]; then
  echo "Refusing symlinked AeroViz runtime directory: $RUNTIME_BASE" >&2
  exit 1
fi
mkdir -p -- "$RUNTIME_BASE" || exit 1
if [[ "$(stat -Lc '%u' "$RUNTIME_BASE" 2>/dev/null)" != "$UID" ]]; then
  echo "AeroViz runtime directory is not owned by uid $UID: $RUNTIME_BASE" >&2
  exit 1
fi
chmod 700 -- "$RUNTIME_BASE" || exit 1
ROOT_ID="$(stat -Lc '%d-%i' "$ROOT_DIR")"
INSTANCE_ID="${ROOT_ID}-${AEROVIZ_BACKEND_PORT}-${VITE_PORT}"
LOCK_PATH="$RUNTIME_BASE/$INSTANCE_ID.lock"
STATE_PATH="$RUNTIME_BASE/$INSTANCE_ID.state"
SUPERVISOR_LOCK_FD=""
SUPERVISOR_START_TICKS=""
STATE_OWNED=0

process_start_ticks() {
  local pid="$1" stat_line rest
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  IFS= read -r stat_line < "/proc/$pid/stat" || return 1
  rest="${stat_line##*) }"
  # After removing pid/comm, token 1 is field 3; starttime is field 22.
  set -- $rest
  [[ $# -ge 20 ]] || return 1
  printf '%s\n' "${20}"
}

read_previous_supervisor() {
  local line recorded_fd="" recorded_root=""
  PREVIOUS_PID=""
  PREVIOUS_START_TICKS=""
  [[ -f "$STATE_PATH" && ! -L "$STATE_PATH" ]] || return 1
  while IFS= read -r line; do
    case "$line" in
      pid=*) PREVIOUS_PID="${line#pid=}" ;;
      start_ticks=*) PREVIOUS_START_TICKS="${line#start_ticks=}" ;;
      lock_fd=*) recorded_fd="${line#lock_fd=}" ;;
      root=*) recorded_root="${line#root=}" ;;
    esac
  done < "$STATE_PATH"
  [[ "$PREVIOUS_PID" =~ ^[0-9]+$ ]] || return 1
  [[ "$PREVIOUS_START_TICKS" =~ ^[0-9]+$ ]] || return 1
  [[ "$recorded_fd" =~ ^[0-9]+$ ]] || return 1
  [[ "$recorded_root" == "$ROOT_DIR" ]] || return 1
  [[ "$(process_start_ticks "$PREVIOUS_PID" 2>/dev/null)" == "$PREVIOUS_START_TICKS" ]] || return 1
  [[ "$(readlink -f "/proc/$PREVIOUS_PID/fd/$recorded_fd" 2>/dev/null)" == "$LOCK_PATH" ]] || return 1
}

write_supervisor_state() {
  local state_tmp="$STATE_PATH.$$"
  SUPERVISOR_START_TICKS="$(process_start_ticks "$$")" || return 1
  umask 077
  printf 'pid=%s\nstart_ticks=%s\nlock_fd=%s\nroot=%s\n' \
    "$$" "$SUPERVISOR_START_TICKS" "$SUPERVISOR_LOCK_FD" "$ROOT_DIR" > "$state_tmp" || return 1
  mv -f -- "$state_tmp" "$STATE_PATH" || {
    rm -f -- "$state_tmp"
    return 1
  }
  STATE_OWNED=1
}

exec {SUPERVISOR_LOCK_FD}>"$LOCK_PATH" || exit 1
if ! flock -n "$SUPERVISOR_LOCK_FD"; then
  if (( ! REPLACE_EXISTING )); then
    echo "AeroViz is already running for backend $AEROVIZ_BACKEND_PORT and frontend $VITE_PORT." >&2
    echo "Use $0 --replace to stop that managed instance first." >&2
    exit 1
  fi
  # The first launcher writes this immediately after locking. Allow a very
  # small concurrent-start window before treating a missing record as unsafe.
  for _ in {1..20}; do
    [[ -f "$STATE_PATH" ]] && break
    sleep 0.05
  done
  if ! read_previous_supervisor; then
    echo "Refusing to replace: the active supervisor identity could not be validated." >&2
    exit 1
  fi
  echo "[supervisor] stopping previous managed instance (pid $PREVIOUS_PID)…"
  kill -TERM "$PREVIOUS_PID" 2>/dev/null || {
    echo "Failed to signal previous AeroViz supervisor pid $PREVIOUS_PID." >&2
    exit 1
  }
  if ! flock -w 15 "$SUPERVISOR_LOCK_FD"; then
    echo "Previous AeroViz supervisor did not shut down within 15 seconds; refusing to force-kill it." >&2
    exit 1
  fi
fi
write_supervisor_state || {
  echo "Failed to record the AeroViz supervisor identity." >&2
  exit 1
}

# Resolve the backend interpreter. PYTHON_BIN wins when set; otherwise ACTIVATE the
# thesis conda env via the shared helper and take its python. Activation — not a direct
# envs/<env>/bin/python exec — is deliberate: it runs the env's activate.d hooks (the
# LD_LIBRARY_PATH libstdc++ fix), which the backend children inherit; a direct exec
# bypasses them and re-opens the documented CXXABI clash the moment the backend imports
# torch/matplotlib. Resolution rules (casadi probe, AEROVIZ_CONDA_ENV pin): see the helper.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/activate_aeroviz_env.sh"
  aeroviz_activate_env || exit 1
  PYTHON_BIN="$(command -v python)"
fi
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

# Each service is started in a dedicated session/process group. This stops npm,
# Vite, and esbuild together without matching unrelated processes by name.
stop_service_group() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}

wait_for_service_group() {
  local pid="$1" deadline
  [[ -z "$pid" ]] && return 0
  deadline=$(( SECONDS + 5 ))
  while kill -0 -- "-$pid" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 0.1
  done
  if kill -0 -- "-$pid" 2>/dev/null; then
    log "service group $pid did not stop within 5 seconds — force-stopping owned group"
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  (( SHUTTING_DOWN )) && return 0
  SHUTTING_DOWN=1
  log "shutting down…"
  stop_service_group "$BACKEND_PID"
  stop_service_group "$VITE_PID"
  wait_for_service_group "$BACKEND_PID"
  wait_for_service_group "$VITE_PID"
  if (( STATE_OWNED )) &&
     [[ "$(process_start_ticks "$$" 2>/dev/null)" == "$SUPERVISOR_START_TICKS" ]]; then
    rm -f -- "$STATE_PATH"
    STATE_OWNED=0
  fi
}
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT

start_backend() {
  ( exec {SUPERVISOR_LOCK_FD}>&-
    exec setsid "$PYTHON_BIN" "$ROOT_DIR/aeroviz_backend/http_server.py" \
      --host "$AEROVIZ_BACKEND_HOST" \
      --port "$AEROVIZ_BACKEND_PORT"
  ) &
  BACKEND_PID=$!
  BACKEND_STARTED_AT=$SECONDS
  log "backend started (pid $BACKEND_PID)"
}

start_frontend() {
  ( cd "$AEROVIZ_APP_DIR" || exit 1
    export VITE_AEROVIZ_BACKEND_PORT="${VITE_AEROVIZ_BACKEND_PORT:-$AEROVIZ_BACKEND_PORT}"
    exec {SUPERVISOR_LOCK_FD}>&-
    exec setsid npm run dev -- --host "$VITE_HOST" --port "$VITE_PORT"
  ) &
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

echo "AeroViz backend listener: http://${AEROVIZ_BACKEND_HOST}:${AEROVIZ_BACKEND_PORT}"
if [[ "$VITE_HOST" == "0.0.0.0" || "$VITE_HOST" == "::" ]]; then
  echo "Frontend (this computer): http://127.0.0.1:${VITE_PORT}"
  if command -v hostname >/dev/null 2>&1; then
    for address in $(hostname -I 2>/dev/null); do
      if [[ "$address" == *.* && "$address" != 127.* ]]; then
        echo "Frontend (local network): http://${address}:${VITE_PORT}"
      fi
    done
  fi
else
  echo "Frontend: http://${VITE_HOST}:${VITE_PORT}"
fi
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
