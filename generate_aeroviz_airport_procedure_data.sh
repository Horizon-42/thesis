#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <ICAO>" >&2
  echo "Example: $0 KRDU" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ICAO=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')

python "$SCRIPT_DIR/aeroviz-4d/python/preprocess_procedures.py" \
  --cifp-root "$SCRIPT_DIR/data/CIFP/CIFP_260319" \
  --airport "$ICAO" \
  --include-all-rnav \
  --include-transitions \
  --charts-root "$SCRIPT_DIR/data/RNAV_CHARTS" \
  --output "$SCRIPT_DIR/aeroviz-4d/public/data/airports/$ICAO/procedures.geojson"
