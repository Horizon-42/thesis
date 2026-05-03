#!/usr/bin/env sh
set -eu

# Full AeroViz airport data preprocessing pipeline.
#
# Usage:
#   ./preprocess_aeroviz_airport.sh KRDU
#
# Current browser-facing data contract:
#   aeroviz-4d/public/data/airports/<ICAO>/airport.json
#   aeroviz-4d/public/data/airports/<ICAO>/runway.geojson
#   aeroviz-4d/public/data/airports/<ICAO>/procedures.geojson
#   aeroviz-4d/public/data/airports/<ICAO>/procedure-details/index.json
#   aeroviz-4d/public/data/airports/<ICAO>/procedure-details/*.json
#   aeroviz-4d/public/data/airports/<ICAO>/charts/index.json
#   aeroviz-4d/public/data/airports/<ICAO>/charts/<files referenced by index.json>
#   aeroviz-4d/public/data/airports/<ICAO>/obstacles.geojson
#   aeroviz-4d/public/data/airports/<ICAO>/waypoints.geojson
#   aeroviz-4d/public/data/airports/<ICAO>/trajectories.czml
#   aeroviz-4d/public/data/airports/<ICAO>/dsm/heightmap-terrain/**
#     Frontend path is still named dsm for compatibility; this script builds it
#     from OpenTopography DEM by default.
#
# Legacy/intermediate data intentionally ignored when deciding whether the
# current dataset is ready:
#   aeroviz-4d/public/data/airports/<ICAO>/charts/*.PDF duplicates
#   aeroviz-4d/public/data/airports/<ICAO>/trajectories.czml.backup
#   aeroviz-4d/public/data/airports/<ICAO>/dsm/3dtiles/**
#   aeroviz-4d/public/data/airports/<ICAO>/dsm/source/**
#   data/DSM/**
#
# DEM source policy:
#   - Current DEM source lives under data/opentopography/<ICAO>/dem/*.tif.
#   - If DEM GeoTIFFs already exist there, skip the OpenTopography downloader.
#   - If missing, call opentopography_downloader/download_opentopography_dem.py.
#   - Heightmap terrain is generated from TERRAIN_INPUT_DIR, which defaults to
#     that OpenTopography DEM directory.
#
# DSM source policy:
#   - DSM is not downloaded or derived by default.
#   - Set DOWNLOAD_DSM=1 only when you explicitly need cached USGS DSM GeoTIFFs
#     under data/usgs_lidar/<ICAO>/dsm. Existing DSM GeoTIFFs are reused.

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <ICAO-or-airport-code>" >&2
  echo "Example: $0 KRDU" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REQUESTED_AIRPORT=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')

# Keep operational settings here instead of exposing a wide CLI surface.
PYTHON_BIN=${PYTHON_BIN:-python}
NPM_BIN=${NPM_BIN:-npm}

AEROVIZ_ROOT=${AEROVIZ_ROOT:-"$SCRIPT_DIR/aeroviz-4d"}
PUBLIC_DATA_ROOT=${PUBLIC_DATA_ROOT:-"$AEROVIZ_ROOT/public/data"}
PUBLIC_AIRPORTS_ROOT=${PUBLIC_AIRPORTS_ROOT:-"$PUBLIC_DATA_ROOT/airports"}
COMMON_DATA_ROOT=${COMMON_DATA_ROOT:-"$PUBLIC_DATA_ROOT/common"}

AIRPORTS_CSV=${AIRPORTS_CSV:-"$COMMON_DATA_ROOT/airports.csv"}
RUNWAYS_CSV=${RUNWAYS_CSV:-"$COMMON_DATA_ROOT/runways.csv"}
CIFP_ROOT=${CIFP_ROOT:-"$SCRIPT_DIR/data/CIFP/CIFP_260319"}
DOF_ROOT=${DOF_ROOT:-"$SCRIPT_DIR/data/DOF/DOF_260412"}
RNAV_CHARTS_ROOT=${RNAV_CHARTS_ROOT:-"$SCRIPT_DIR/data/RNAV_CHARTS"}
OPENTOPOGRAPHY_ROOT=${OPENTOPOGRAPHY_ROOT:-"$SCRIPT_DIR/data/opentopography"}
USGS_LIDAR_ROOT=${USGS_LIDAR_ROOT:-"$SCRIPT_DIR/data/usgs_lidar"}
OPENSKY_OUTPUT_ROOT=${OPENSKY_OUTPUT_ROOT:-"$SCRIPT_DIR/opensky_data_query/outputs"}

OBSTACLE_RADIUS_KM=${OBSTACLE_RADIUS_KM:-20}
WAYPOINT_RADIUS_KM=${WAYPOINT_RADIUS_KM:-120}
WAYPOINT_MAX_COUNT=${WAYPOINT_MAX_COUNT:-60}
GENERATE_TRAJECTORIES=${GENERATE_TRAJECTORIES:-1}
DOWNLOAD_DSM=${DOWNLOAD_DSM:-0}

log() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf 'warning: %s\n' "$*" >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

run() {
  printf '+'
  printf ' %s' "$@"
  printf '\n'
  "$@"
}

require_file() {
  [ -f "$1" ] || die "required file not found: $1"
}

require_dir() {
  [ -d "$1" ] || die "required directory not found: $1"
}

has_pdf_files() {
  [ -d "$1" ] || return 1
  find "$1" -maxdepth 1 -type f \( -iname '*.pdf' -o -iname '*.PDF' \) -print -quit | grep -q .
}

has_geotiff_files() {
  [ -d "$1" ] || return 1
  find "$1" -maxdepth 1 -type f \( -iname '*.tif' -o -iname '*.tiff' \) -print -quit | grep -q .
}

latest_czml_input() {
  airport_tag=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  airport_output_dir="$OPENSKY_OUTPUT_ROOT/$airport_tag"
  [ -d "$airport_output_dir" ] || return 1
  latest=$(
    find "$airport_output_dir" -maxdepth 1 -type f -name "${airport_tag}_czml_input_*.json" \
      -print | sort | tail -n 1
  )
  [ -n "$latest" ] || return 1
  printf '%s\n' "$latest"
}

state_dof_file() {
  # FAA DOF files are named by numeric state/territory prefix.
  # Unknown countries return a non-zero status so the caller can skip obstacles
  # explicitly instead of accidentally using the wrong state file.
  country=$1
  region=$2

  case "$country" in
    US)
      state=${region#US-}
      case "$state" in
        AL) echo "$DOF_ROOT/01-AL.Dat" ;;
        AK) echo "$DOF_ROOT/02-AK.Dat" ;;
        AZ) echo "$DOF_ROOT/04-AZ.Dat" ;;
        AR) echo "$DOF_ROOT/05-AR.Dat" ;;
        CA) echo "$DOF_ROOT/06-CA.Dat" ;;
        CO) echo "$DOF_ROOT/08-CO.Dat" ;;
        CT) echo "$DOF_ROOT/09-CT.Dat" ;;
        DE) echo "$DOF_ROOT/10-DE.Dat" ;;
        DC) echo "$DOF_ROOT/11-DC.Dat" ;;
        FL) echo "$DOF_ROOT/12-FL.Dat" ;;
        GA) echo "$DOF_ROOT/13-GA.Dat" ;;
        HI) echo "$DOF_ROOT/15-HI.Dat" ;;
        ID) echo "$DOF_ROOT/16-ID.Dat" ;;
        IL) echo "$DOF_ROOT/17-IL.Dat" ;;
        IN) echo "$DOF_ROOT/18-IN.Dat" ;;
        IA) echo "$DOF_ROOT/19-IA.Dat" ;;
        KS) echo "$DOF_ROOT/20-KS.Dat" ;;
        KY) echo "$DOF_ROOT/21-KY.Dat" ;;
        LA) echo "$DOF_ROOT/22-LA.Dat" ;;
        ME) echo "$DOF_ROOT/23-ME.Dat" ;;
        MD) echo "$DOF_ROOT/24-MD.Dat" ;;
        MA) echo "$DOF_ROOT/25-MA.Dat" ;;
        MI) echo "$DOF_ROOT/26-MI.Dat" ;;
        MN) echo "$DOF_ROOT/27-MN.Dat" ;;
        MS) echo "$DOF_ROOT/28-MS.Dat" ;;
        MO) echo "$DOF_ROOT/29-MO.Dat" ;;
        MT) echo "$DOF_ROOT/30-MT.Dat" ;;
        NE) echo "$DOF_ROOT/31-NE.Dat" ;;
        NV) echo "$DOF_ROOT/32-NV.Dat" ;;
        NH) echo "$DOF_ROOT/33-NH.Dat" ;;
        NJ) echo "$DOF_ROOT/34-NJ.Dat" ;;
        NM) echo "$DOF_ROOT/35-NM.Dat" ;;
        NY) echo "$DOF_ROOT/36-NY.Dat" ;;
        NC) echo "$DOF_ROOT/37-NC.Dat" ;;
        ND) echo "$DOF_ROOT/38-ND.Dat" ;;
        OH) echo "$DOF_ROOT/39-OH.Dat" ;;
        OK) echo "$DOF_ROOT/40-OK.Dat" ;;
        OR) echo "$DOF_ROOT/41-OR.Dat" ;;
        PA) echo "$DOF_ROOT/42-PA.Dat" ;;
        RI) echo "$DOF_ROOT/44-RI.Dat" ;;
        SC) echo "$DOF_ROOT/45-SC.Dat" ;;
        SD) echo "$DOF_ROOT/46-SD.Dat" ;;
        TN) echo "$DOF_ROOT/47-TN.Dat" ;;
        TX) echo "$DOF_ROOT/48-TX.Dat" ;;
        UT) echo "$DOF_ROOT/49-UT.Dat" ;;
        VT) echo "$DOF_ROOT/50-VT.Dat" ;;
        VA) echo "$DOF_ROOT/51-VA.Dat" ;;
        WA) echo "$DOF_ROOT/53-WA.Dat" ;;
        WV) echo "$DOF_ROOT/54-WV.Dat" ;;
        WI) echo "$DOF_ROOT/55-WI.Dat" ;;
        WY) echo "$DOF_ROOT/56-WY.Dat" ;;
        PR) echo "$DOF_ROOT/PuertoRico.Dat" ;;
        *) return 1 ;;
      esac
      ;;
    CA)
      echo "$DOF_ROOT/Canada.Dat"
      ;;
    MX)
      echo "$DOF_ROOT/Mexico.Dat"
      ;;
    BS)
      echo "$DOF_ROOT/Bahamas.Dat"
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_airport_metadata() {
  # Print shell-safe assignments:
  #   ICAO=...
  #   AIRPORT_NAME=...
  #   ISO_COUNTRY=...
  #   ISO_REGION=...
  "$PYTHON_BIN" - "$AIRPORTS_CSV" "$REQUESTED_AIRPORT" <<'PY'
import csv
import shlex
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
wanted = sys.argv[2].strip().upper()

if not csv_path.exists():
    raise SystemExit(f"airport CSV not found: {csv_path}")

with csv_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        aliases = {
            (row.get("ident") or "").strip().upper(),
            (row.get("icao_code") or "").strip().upper(),
            (row.get("gps_code") or "").strip().upper(),
            (row.get("local_code") or "").strip().upper(),
            (row.get("iata_code") or "").strip().upper(),
        }
        if wanted not in aliases:
            continue

        icao = (row.get("icao_code") or row.get("ident") or wanted).strip().upper()
        values = {
            "ICAO": icao,
            "AIRPORT_NAME": (row.get("name") or icao).strip(),
            "ISO_COUNTRY": (row.get("iso_country") or "").strip().upper(),
            "ISO_REGION": (row.get("iso_region") or "").strip().upper(),
        }
        for key, value in values.items():
            print(f"{key}={shlex.quote(value)}")
        break
    else:
        raise SystemExit(f"airport {wanted} not found in {csv_path}")
PY
}

validate_json_file() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    json.load(handle)
PY
}

require_file "$AIRPORTS_CSV"
require_file "$RUNWAYS_CSV"
require_dir "$CIFP_ROOT"
require_file "$CIFP_ROOT/IN_CIFP.txt"
require_dir "$DOF_ROOT"
require_dir "$AEROVIZ_ROOT"
require_file "$SCRIPT_DIR/opentopography_downloader/download_opentopography_dem.py"
if [ "$DOWNLOAD_DSM" = "1" ]; then
  require_file "$SCRIPT_DIR/usgs_lidar_downloader/download_usgs_lidar.py"
fi

eval "$(resolve_airport_metadata)"

PUBLIC_AIRPORT_DIR="$PUBLIC_AIRPORTS_ROOT/$ICAO"
PUBLIC_DSM_HEIGHTMAP_DIR="$PUBLIC_AIRPORT_DIR/dsm/heightmap-terrain"
OPENTOPOGRAPHY_DEM_DIR="$OPENTOPOGRAPHY_ROOT/$ICAO/dem"
USGS_DSM_DIR="$USGS_LIDAR_ROOT/$ICAO/dsm"
TERRAIN_INPUT_DIR=${TERRAIN_INPUT_DIR:-"$OPENTOPOGRAPHY_DEM_DIR"}
case "$TERRAIN_INPUT_DIR" in
  /*) ;;
  *) TERRAIN_INPUT_DIR="$SCRIPT_DIR/$TERRAIN_INPUT_DIR" ;;
esac

log "Preprocessing $ICAO - $AIRPORT_NAME"
mkdir -p "$PUBLIC_AIRPORT_DIR"

log "1/10 Build airport.json, runway.geojson, and airports/index.json"
run "$PYTHON_BIN" "$AEROVIZ_ROOT/python/preprocess_airports.py" \
  --airport "$ICAO" \
  --airports-csv "$AIRPORTS_CSV" \
  --runways-csv "$RUNWAYS_CSV"

log "2/10 Ensure RNAV chart source PDFs exist"
mkdir -p "$RNAV_CHARTS_ROOT"
if has_pdf_files "$RNAV_CHARTS_ROOT/$ICAO"; then
  echo "RNAV source charts already exist: $RNAV_CHARTS_ROOT/$ICAO"
else
  run "$PYTHON_BIN" "$AEROVIZ_ROOT/python/download_faa_rnav_charts.py" \
    "$ICAO" \
    --output-root "$RNAV_CHARTS_ROOT"
fi

log "3/10 Generate procedures.geojson, procedure-details, and charts/index.json"
run "$PYTHON_BIN" "$AEROVIZ_ROOT/python/preprocess_procedures.py" \
  --cifp-root "$CIFP_ROOT" \
  --airport "$ICAO" \
  --include-all-rnav \
  --include-transitions \
  --charts-root "$RNAV_CHARTS_ROOT" \
  --output "$PUBLIC_AIRPORT_DIR/procedures.geojson"

log "4/10 Generate obstacles.geojson"
if DOF_INPUT=$(state_dof_file "$ISO_COUNTRY" "$ISO_REGION"); then
  if [ -f "$DOF_INPUT" ]; then
    run "$PYTHON_BIN" "$AEROVIZ_ROOT/python/preprocess_obstacles.py" \
      --input "$DOF_INPUT" \
      --airport-code "$ICAO" \
      --airport \
      --radius-km "$OBSTACLE_RADIUS_KM" \
      --output "$PUBLIC_AIRPORT_DIR/obstacles.geojson"
  else
    die "resolved DOF file does not exist: $DOF_INPUT"
  fi
else
  warn "no DOF mapping for country/region $ISO_COUNTRY/$ISO_REGION; skipping obstacles.geojson"
fi

log "5/10 Generate waypoints.geojson"
WAYPOINT_SOURCE_URL=${WAYPOINT_SOURCE_URL:-"https://opennav.com/waypoint/$ISO_COUNTRY"}
run "$PYTHON_BIN" "$AEROVIZ_ROOT/python/preprocess_waypoints.py" \
  --airport "$ICAO" \
  --airports-csv "$AIRPORTS_CSV" \
  --source-url "$WAYPOINT_SOURCE_URL" \
  --radius-km "$WAYPOINT_RADIUS_KM" \
  --max-waypoints "$WAYPOINT_MAX_COUNT" \
  --output "$PUBLIC_AIRPORT_DIR/waypoints.geojson"

log "6/10 Generate trajectories.czml if local CZML input exists"
if [ "$GENERATE_TRAJECTORIES" = "1" ]; then
  if CZML_INPUT=$(latest_czml_input "$ICAO"); then
    run "$PYTHON_BIN" "$AEROVIZ_ROOT/python/generate_czml.py" \
      --airport "$ICAO" \
      --input "$CZML_INPUT" \
      --output "$PUBLIC_AIRPORT_DIR/trajectories.czml"
  else
    warn "no CZML input found under $OPENSKY_OUTPUT_ROOT/$(printf '%s' "$ICAO" | tr '[:upper:]' '[:lower:]'); skipping trajectories.czml"
  fi
else
  warn "GENERATE_TRAJECTORIES=0; skipping trajectories.czml"
fi

log "7/10 Ensure OpenTopography DEM GeoTIFF source exists"
if has_geotiff_files "$OPENTOPOGRAPHY_DEM_DIR"; then
  echo "OpenTopography DEM GeoTIFFs already exist: $OPENTOPOGRAPHY_DEM_DIR"
else
  run "$PYTHON_BIN" "$SCRIPT_DIR/opentopography_downloader/download_opentopography_dem.py" \
    "$ICAO" \
    --out "$OPENTOPOGRAPHY_ROOT"
fi

has_geotiff_files "$OPENTOPOGRAPHY_DEM_DIR" || die "no DEM GeoTIFFs found after OpenTopography step: $OPENTOPOGRAPHY_DEM_DIR"

log "8/10 Optional USGS DSM GeoTIFF source"
if [ "$DOWNLOAD_DSM" = "1" ]; then
  if has_geotiff_files "$USGS_DSM_DIR"; then
    echo "USGS derived DSM GeoTIFFs already exist: $USGS_DSM_DIR"
  else
    run "$PYTHON_BIN" "$SCRIPT_DIR/usgs_lidar_downloader/download_usgs_lidar.py" \
      "$ICAO" \
      --out "$USGS_LIDAR_ROOT"
  fi

  has_geotiff_files "$USGS_DSM_DIR" || die "no DSM GeoTIFFs found after USGS step: $USGS_DSM_DIR"
else
  echo "DOWNLOAD_DSM=0; skipping USGS DSM download/derivation"
fi

log "9/10 Build frontend heightmap terrain from DEM"
has_geotiff_files "$TERRAIN_INPUT_DIR" || die "no terrain input GeoTIFFs found: $TERRAIN_INPUT_DIR"
echo "Terrain input GeoTIFFs: $TERRAIN_INPUT_DIR"
(
  cd "$AEROVIZ_ROOT"
  run "$NPM_BIN" run build:dsm-heightmap-terrain -- \
    --airport "$ICAO" \
    --input-dir "$TERRAIN_INPUT_DIR"
)

log "10/10 Validate current output contract"
require_file "$PUBLIC_AIRPORT_DIR/airport.json"
require_file "$PUBLIC_AIRPORT_DIR/runway.geojson"
require_file "$PUBLIC_AIRPORT_DIR/procedures.geojson"
require_file "$PUBLIC_AIRPORT_DIR/procedure-details/index.json"
require_file "$PUBLIC_AIRPORT_DIR/charts/index.json"
require_file "$PUBLIC_AIRPORT_DIR/obstacles.geojson"
require_file "$PUBLIC_AIRPORT_DIR/waypoints.geojson"
require_file "$PUBLIC_DSM_HEIGHTMAP_DIR/metadata.json"
has_geotiff_files "$OPENTOPOGRAPHY_DEM_DIR" || die "missing OpenTopography DEM GeoTIFFs: $OPENTOPOGRAPHY_DEM_DIR"

validate_json_file "$PUBLIC_AIRPORT_DIR/airport.json"
validate_json_file "$PUBLIC_AIRPORT_DIR/runway.geojson"
validate_json_file "$PUBLIC_AIRPORT_DIR/procedures.geojson"
validate_json_file "$PUBLIC_AIRPORT_DIR/procedure-details/index.json"
validate_json_file "$PUBLIC_AIRPORT_DIR/charts/index.json"
validate_json_file "$PUBLIC_AIRPORT_DIR/obstacles.geojson"
validate_json_file "$PUBLIC_AIRPORT_DIR/waypoints.geojson"
validate_json_file "$PUBLIC_DSM_HEIGHTMAP_DIR/metadata.json"

if [ "$GENERATE_TRAJECTORIES" = "1" ] && [ -f "$PUBLIC_AIRPORT_DIR/trajectories.czml" ]; then
  validate_json_file "$PUBLIC_AIRPORT_DIR/trajectories.czml"
fi

echo
echo "Done: current AeroViz data for $ICAO is under:"
echo "  $PUBLIC_AIRPORT_DIR"
