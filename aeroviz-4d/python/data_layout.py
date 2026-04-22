from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PYTHON_ROOT = Path(__file__).resolve().parent
AEROVIZ_ROOT = PYTHON_ROOT.parent
PUBLIC_DATA_DIR = AEROVIZ_ROOT / "public" / "data"
COMMON_DATA_DIR = PUBLIC_DATA_DIR / "common"
AIRPORTS_DATA_DIR = PUBLIC_DATA_DIR / "airports"
AIRPORTS_INDEX_PATH = AIRPORTS_DATA_DIR / "index.json"

AIRPORT_CODE_COLUMNS = ("ident", "gps_code", "icao_code", "local_code", "iata_code")


def normalize_airport_code(code: str) -> str:
    return code.strip().upper()


def common_data_path(file_name: str) -> Path:
    return COMMON_DATA_DIR / file_name


def airport_data_dir(airport_code: str) -> Path:
    return AIRPORTS_DATA_DIR / normalize_airport_code(airport_code)


def airport_data_path(airport_code: str, file_name: str) -> Path:
    return airport_data_dir(airport_code) / file_name


def airport_dsm_dir(airport_code: str) -> Path:
    return airport_data_dir(airport_code) / "dsm"


def airport_dsm_source_dir(airport_code: str) -> Path:
    return airport_dsm_dir(airport_code) / "source"


def airport_dsm_3dtiles_dir(airport_code: str) -> Path:
    return airport_dsm_dir(airport_code) / "3dtiles"


def airport_dsm_heightmap_dir(airport_code: str) -> Path:
    return airport_dsm_dir(airport_code) / "heightmap-terrain"


def resolve_common_csv(path: Path) -> Path:
    if path.exists():
        return path

    common_candidate = common_data_path(path.name)
    if common_candidate.exists():
      return common_candidate

    legacy_candidate = PUBLIC_DATA_DIR / path.name
    if legacy_candidate.exists():
      return legacy_candidate

    return path


def _airport_aliases(row: dict[str, str]) -> set[str]:
    aliases = set()
    for column in AIRPORT_CODE_COLUMNS:
        value = (row.get(column) or "").strip().upper()
        if value:
            aliases.add(value)
    return aliases


def find_airport_record(csv_path: Path, airport_code: str) -> dict[str, Any]:
    normalized_code = normalize_airport_code(airport_code)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if normalized_code not in _airport_aliases(row):
                continue

            lat_raw = row.get("latitude_deg")
            lon_raw = row.get("longitude_deg")
            if lat_raw in (None, "") or lon_raw in (None, ""):
                continue

            elevation_ft = row.get("elevation_ft")
            elevation_m = (
                float(elevation_ft) * 0.3048
                if elevation_ft not in (None, "")
                else 0.0
            )
            return {
                "code": normalize_airport_code(row.get("ident") or normalized_code),
                "name": (row.get("name") or normalized_code).strip(),
                "lat": float(lat_raw),
                "lon": float(lon_raw),
                "elevation_m": elevation_m,
            }

    raise ValueError(f"Airport {normalized_code} not found in {csv_path}")


def load_airports_index(index_path: Path = AIRPORTS_INDEX_PATH) -> dict[str, Any]:
    if not index_path.exists():
        return {"defaultAirport": None, "airports": []}

    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"defaultAirport": None, "airports": []}

    airports = data.get("airports")
    if not isinstance(airports, list):
        airports = []

    default_airport = data.get("defaultAirport")
    if default_airport is not None:
        default_airport = normalize_airport_code(str(default_airport))

    return {
        "defaultAirport": default_airport,
        "airports": airports,
    }


def upsert_airports_index(
    *,
    airport_code: str,
    airport_name: str,
    lat: float,
    lon: float,
    default_airport: str | None = None,
    index_path: Path = AIRPORTS_INDEX_PATH,
) -> dict[str, Any]:
    manifest = load_airports_index(index_path=index_path)
    airports_by_code = {
        normalize_airport_code(str(item.get("code", ""))): item
        for item in manifest["airports"]
        if isinstance(item, dict) and item.get("code")
    }

    normalized_code = normalize_airport_code(airport_code)
    airports_by_code[normalized_code] = {
        "code": normalized_code,
        "name": airport_name,
        "lat": lat,
        "lon": lon,
    }

    sorted_airports = sorted(airports_by_code.values(), key=lambda item: item["code"])
    next_default = normalize_airport_code(
        default_airport or manifest["defaultAirport"] or normalized_code
    )
    if next_default not in airports_by_code:
        next_default = normalized_code

    next_manifest = {
        "defaultAirport": next_default,
        "airports": sorted_airports,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(next_manifest, indent=2) + "\n", encoding="utf-8")
    return next_manifest
