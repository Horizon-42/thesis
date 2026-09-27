#!/usr/bin/env bash
# Public demo of the running AeroViz stack through a Cloudflare quick tunnel.
#
#   ./start_demo_tunnel.sh            build dist/ (backend baked in as same-origin /api), serve, tunnel
#   ./start_demo_tunnel.sh --no-build reuse the existing demo build
#
# Needs the backend already up (./start_aeroviz_fullstack.sh, port 8765). The LAN dev server
# on 5173 is untouched. The public URL (https://<random>.trycloudflare.com) is printed by
# cloudflared and written to aeroviz-4d/.demo-tunnel-url; it changes every time this restarts.
# Quick tunnels carry no uptime guarantee. Ctrl-C stops the preview server and the tunnel.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/aeroviz-4d"
BACKEND_PORT="${AEROVIZ_BACKEND_PORT:-8765}"
URL_FILE="$APP_DIR/.demo-tunnel-url"

if ! curl -fsS -o /dev/null "http://127.0.0.1:$BACKEND_PORT/simulation/aircraft"; then
  echo "backend not answering on 127.0.0.1:$BACKEND_PORT — start ./start_aeroviz_fullstack.sh first" >&2
  exit 1
fi

cd "$APP_DIR"
if [[ "${1:-}" != "--no-build" ]]; then
  VITE_AEROVIZ_BACKEND_URL=/api npm run build
fi

npx vite preview --config vite.demo.config.ts &
PREVIEW_PID=$!
trap 'kill "$PREVIEW_PID" 2>/dev/null; rm -f "$URL_FILE"' EXIT

cloudflared tunnel --no-autoupdate --url http://127.0.0.1:4173 2>&1 | while IFS= read -r line; do
  echo "$line"
  if [[ "$line" =~ (https://[a-z0-9-]+\.trycloudflare\.com) ]]; then
    echo "${BASH_REMATCH[1]}" > "$URL_FILE"
    echo ">>> DEMO URL: ${BASH_REMATCH[1]}"
  fi
done
