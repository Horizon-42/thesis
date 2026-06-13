#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AEROVIZ_APP_DIR="${AEROVIZ_APP_DIR:-$ROOT_DIR/aeroviz-4d}"
AEROVIZ_PUBLIC_DATA_DIR="$AEROVIZ_APP_DIR/public/data"
AEROVIZ_DIST_DATA_PATH="$AEROVIZ_APP_DIR/dist/data"
AEROVIZ_BACKEND_HOST="${AEROVIZ_BACKEND_HOST:-127.0.0.1}"
AEROVIZ_BACKEND_PORT="${AEROVIZ_BACKEND_PORT:-8765}"
VITE_HOST="${VITE_HOST:-127.0.0.1}"
VITE_PORT="${VITE_PORT:-5173}"
PYTHON_BIN="${PYTHON_BIN:-/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python}"

cleanup() {
  if [[ -n "${AEROVIZ_BACKEND_PID:-}" ]]; then
    kill "$AEROVIZ_BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${VITE_PID:-}" ]]; then
    kill "$VITE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -d "$AEROVIZ_PUBLIC_DATA_DIR" ]]; then
  echo "Missing AeroViz public data directory: $AEROVIZ_PUBLIC_DATA_DIR" >&2
  exit 1
fi

if [[ -d "$AEROVIZ_DIST_DATA_PATH" && ! -L "$AEROVIZ_DIST_DATA_PATH" ]]; then
  echo "Note: $AEROVIZ_DIST_DATA_PATH is an old copied build artifact." >&2
  echo "Run npm run build in $AEROVIZ_APP_DIR, or remove dist/data, to reclaim the duplicate storage." >&2
fi

"$PYTHON_BIN" "$ROOT_DIR/aeroviz_backend/http_server.py" \
  --host "$AEROVIZ_BACKEND_HOST" \
  --port "$AEROVIZ_BACKEND_PORT" &
AEROVIZ_BACKEND_PID=$!

cd "$AEROVIZ_APP_DIR"
VITE_AEROVIZ_BACKEND_URL="${VITE_AEROVIZ_BACKEND_URL:-http://${AEROVIZ_BACKEND_HOST}:${AEROVIZ_BACKEND_PORT}}" \
  npm run dev -- --host "$VITE_HOST" --port "$VITE_PORT" &
VITE_PID=$!

echo "AeroViz backend: http://${AEROVIZ_BACKEND_HOST}:${AEROVIZ_BACKEND_PORT}"
echo "Frontend: http://${VITE_HOST}:${VITE_PORT}"
echo "AeroViz data: $AEROVIZ_PUBLIC_DATA_DIR"

while kill -0 "$AEROVIZ_BACKEND_PID" 2>/dev/null && kill -0 "$VITE_PID" 2>/dev/null; do
  sleep 1
done
