"""Generate ``aeroviz-4d/src/generated/geoConstants.json`` from ``geokit.constants``.

The frontend (TypeScript) cannot import the Python ``geokit`` package, so it mirrors the
canonical constants through this generated JSON — keeping a single source of truth across
the language boundary. Run this whenever ``geokit.constants`` changes:

    python geokit/scripts/export_constants_json.py

A drift-guard test asserts the JSON matches ``geokit.constants`` so the two cannot
silently diverge.
"""

from __future__ import annotations

import json
from pathlib import Path

from geokit import constants as C

# geokit name -> frontend-facing name. EARTH_RADIUS_M is the spherical default used by
# the TS haversine/great-circle helpers (= the switchable SPHERE_RADIUS_M).
CONSTANTS = {
    "EARTH_RADIUS_M": C.SPHERE_RADIUS_M,
    "EARTH_RADIUS_MEAN_M": C.EARTH_RADIUS_MEAN_M,
    "WGS84_A": C.WGS84_A,
    "METERS_PER_NM": C.NM_M,
    "FEET_TO_METERS": C.FT_M,
    "METRES_PER_DEG_LAT": C.METRES_PER_DEG_LAT,
    "DEG2RAD": C.DEG2RAD,
    "RAD2DEG": C.RAD2DEG,
    # speed factors (multiply source unit -> m/s)
    "KNOTS_TO_MPS": C.KT_MS,
    "FT_MIN_TO_MPS": C.FT_MIN_MS,
}

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "aeroviz-4d" / "src" / "generated" / "geoConstants.json"
)


def build_payload() -> dict:
    return {
        "_comment": (
            "Generated from geokit.constants by geokit/scripts/export_constants_json.py. "
            "Do not edit by hand; run the script to regenerate."
        ),
        **CONSTANTS,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(build_payload(), indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
