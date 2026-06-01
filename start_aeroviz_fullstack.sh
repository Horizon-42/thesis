#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_SERVER_HOST="${SIM_SERVER_HOST:-127.0.0.1}"
SIM_SERVER_PORT="${SIM_SERVER_PORT:-8765}"
VITE_HOST="${VITE_HOST:-127.0.0.1}"
VITE_PORT="${VITE_PORT:-5173}"
PYTHON_BIN="${PYTHON_BIN:-/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python}"

cleanup() {
  if [[ -n "${SIM_SERVER_PID:-}" ]]; then
    kill "$SIM_SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "${VITE_PID:-}" ]]; then
    kill "$VITE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$PYTHON_BIN" "$ROOT_DIR/aerodynamic_model/simulation_server.py" \
  --host "$SIM_SERVER_HOST" \
  --port "$SIM_SERVER_PORT" &
SIM_SERVER_PID=$!

cd "$ROOT_DIR/aeroviz-4d"
VITE_PILOT_SERVER_URL="${VITE_PILOT_SERVER_URL:-http://${SIM_SERVER_HOST}:${SIM_SERVER_PORT}}" \
  npm run dev -- --host "$VITE_HOST" --port "$VITE_PORT" &
VITE_PID=$!

echo "Simulation server: http://${SIM_SERVER_HOST}:${SIM_SERVER_PORT}"
echo "Frontend: http://${VITE_HOST}:${VITE_PORT}"

while kill -0 "$SIM_SERVER_PID" 2>/dev/null && kill -0 "$VITE_PID" 2>/dev/null; do
  sleep 1
done
