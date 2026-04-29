"""
preprocess_procedures.py
========================
Extracts a display-oriented RNAV procedure path from FAA CIFP and writes
public/data/airports/<ICAO>/procedures.geojson for AeroViz-4D.

Default example:
  python aeroviz-4d/python/preprocess_procedures.py \
    --cifp-root data/CIFP/CIFP_260319 \
    --airport KRDU \
    --procedure-type SIAP \
    --procedure R05LY

This parser intentionally implements a small ARINC 424 subset:
- It uses IN_CIFP.txt as the airport/procedure index.
- It reads FAACIFP18 fixed-width procedure records for one airport/procedure.
- It resolves terminal fixes and runway thresholds from local coordinate records.
- It includes IF/TF legs in the route and reports unsupported/skipped legs.

The output is for research visualization only, not certified navigation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_layout import (
    airport_chart_path,
    airport_charts_dir,
    airport_data_path,
    airport_procedure_details_dir,
    airport_procedure_details_path,
    common_data_path,
    find_airport_record,
)

FEET_TO_METRES = 0.3048
NM_TO_METRES = 1852.0
DEFAULT_CIFP_ROOT = Path(__file__).parents[2] / "data" / "CIFP" / "CIFP_260319"
DEFAULT_RNAV_CHARTS_ROOT = Path(__file__).parents[2] / "data" / "RNAV_CHARTS"
DEFAULT_AIRPORT = "KRDU"
RUNWAY_ORDER = ["RW05L", "RW05R", "RW23L", "RW23R", "RW32"]
SUPPORTED_ROUTE_LEGS = {"IF", "TF"}
COORD_PAIR_RE = re.compile(r"([NS]\d{8,10})([EW]\d{9,11})")
LEG_TYPE_RE = re.compile(r"(?<![A-Z])(IF|TF|DF|CA|CF|HM|HF|RF|VI|VA|FA)(?![A-Z])")
CHART_PROCEDURE_RE = re.compile(r"R([YZ])?(\d{1,2})([LRC]?)")
SAFE_FILE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class FixRecord:
    ident: str
    lon: float
    lat: float
    altitude_ft: int | None
    source_line: int
    region_code: str | None = None
    source_kind: str = "airport-local-cifp"


@dataclass(frozen=True)
class ProcedureLeg:
    sequence: int
    branch: str
    fix_ident: str
    leg_type: str
    role: str
    altitude_ft: int | None
    source_line: int
    fix_region_code: str | None = None


@dataclass(frozen=True)
class RoutePoint:
    sequence: int
    fix_ident: str
    leg_type: str
    role: str
    lon: float
    lat: float
    altitude_ft: int | None
    geometry_altitude_ft: int
    time_seconds: float
    distance_from_start_m: float
    source_line: int


def decode_cifp_coordinate(token: str) -> float:
    """Decode compact CIFP DMS coordinates to decimal degrees."""
    value = token.strip().upper()
    if len(value) < 9:
        raise ValueError(f"Coordinate token is too short: {token!r}")

    hemisphere = value[0]
    if hemisphere not in {"N", "S", "E", "W"}:
        raise ValueError(f"Missing coordinate hemisphere: {token!r}")

    deg_width = 2 if hemisphere in {"N", "S"} else 3
    digits = value[1:]
    if len(digits) < deg_width + 4 or not digits.isdigit():
        raise ValueError(f"Invalid coordinate digits: {token!r}")

    degrees = int(digits[:deg_width])
    minutes = int(digits[deg_width : deg_width + 2])
    seconds_digits = digits[deg_width + 2 :]
    seconds_scale = 10 ** max(len(seconds_digits) - 2, 0)
    seconds = int(seconds_digits) / seconds_scale

    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if hemisphere in {"S", "W"}:
        decimal = -decimal
    return decimal


def decode_coordinate_pair(lat_token: str, lon_token: str) -> tuple[float, float]:
    """Return (lon, lat) from CIFP compact coordinate tokens."""
    return decode_cifp_coordinate(lon_token), decode_cifp_coordinate(lat_token)


def parse_signed_altitude_ft(text: str) -> int | None:
    """Parse the first ARINC-style five-digit altitude in a text slice."""
    match = re.search(r"([+\-V])\s*(\d{5})", text)
    if match:
        altitude = int(match.group(2))
        if match.group(1) == "-":
            altitude *= -1
        return altitude

    match = re.search(r"(?<!\d)(\d{5})(?!\d)", text)
    if not match:
        return None
    return int(match.group(1))


def parse_leg_altitude_ft(line: str) -> int | None:
    """Parse altitude constraints from the CIFP procedure leg altitude fields."""
    primary_altitude = parse_signed_altitude_ft(line[70:90])
    if primary_altitude is not None:
        return primary_altitude

    # Some CIFP legs encode altitude 2 immediately before the speed limit field,
    # e.g. "18000210" for 18,000 ft and 210 kt. Parse it by column, not regex.
    secondary_altitude = line[94:99].strip()
    if len(secondary_altitude) == 5 and secondary_altitude.isdigit():
        return int(secondary_altitude)

    return None


def procedure_exists(index_path: Path, airport: str, procedure_type: str, procedure: str) -> bool:
    """Return true when IN_CIFP.txt lists the requested airport/procedure."""
    with index_path.open(encoding="ascii", errors="replace") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 3:
                continue
            if (
                fields[0].upper() == airport.upper()
                and fields[1].upper() == procedure_type.upper()
                and fields[2].upper() == procedure.upper()
            ):
                return True
    return False


def discover_rnav_procedures(index_path: Path, airport: str, procedure_type: str) -> list[str]:
    """Return RNAV/RNP approach idents listed for one airport in IN_CIFP.txt."""
    procedures: list[str] = []
    seen: set[str] = set()
    with index_path.open(encoding="ascii", errors="replace") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 3:
                continue
            if fields[0].upper() != airport.upper() or fields[1].upper() != procedure_type.upper():
                continue

            procedure = fields[2].upper()
            if procedure_family(procedure) not in {"RNAV_GPS", "RNAV_RNP"}:
                continue
            if procedure in seen:
                continue
            procedures.append(procedure)
            seen.add(procedure)

    return sorted(procedures, key=procedure_sort_key)


def parse_procedure_list(raw_value: str) -> list[str]:
    """Parse a comma-separated CLI procedure list."""
    return [item.strip().upper() for item in raw_value.split(",") if item.strip()]


def runway_from_procedure_ident(procedure: str) -> str | None:
    """Infer runway ident from common KRDU-style CIFP approach idents."""
    ident = procedure.upper()
    match = re.match(r"^[RHIL]?(\d{2})([LRC]?)([YZ]?)$", ident)
    if not match:
        return None

    runway_number = match.group(1)
    runway_side = match.group(2)
    return f"RW{runway_number}{runway_side}"


def procedure_variant(procedure: str) -> str | None:
    """Return Y/Z-style procedure variant when encoded in the ident."""
    ident = procedure.upper()
    if ident[-1:] in {"Y", "Z"}:
        return ident[-1]
    return None


def procedure_family(procedure: str) -> str:
    """Classify procedure family from CIFP ident prefix."""
    ident = procedure.upper()
    if ident.startswith("RNV"):
        return "RNAV_GPS"
    if ident.startswith("R"):
        return "RNAV_GPS"
    if ident.startswith("H"):
        return "RNAV_RNP"
    if ident.startswith("I"):
        return "ILS"
    if ident.startswith("L"):
        return "LOC"
    return "UNKNOWN"


def final_branch_for_procedure(procedure: str) -> str:
    """Infer the main final-approach branch ident for a procedure family."""
    family = procedure_family(procedure)
    if family == "RNAV_RNP":
        return "H"
    if family == "LOC":
        return "L"
    if family == "ILS":
        return "I"
    return "R"


def branch_type_for(procedure: str, branch: str) -> str:
    """Classify branch display group."""
    if branch.upper() == final_branch_for_procedure(procedure):
        return "final"
    return "transition"


def branch_sort_key(procedure: str, branch: str) -> tuple[int, str]:
    """Put final branch first, transitions after."""
    return (0 if branch.upper() == final_branch_for_procedure(procedure) else 1, branch)


def parse_source_cycle(faacifp_path: Path) -> str | None:
    """Read the CIFP cycle number from the HDR01/HDR04 records when present."""
    with faacifp_path.open(encoding="ascii", errors="replace") as f:
        for line in f:
            match = re.search(r"VOLUME\s+(\d{4})", line)
            if match:
                return match.group(1)
            match = re.search(r"FAACIFP18\s+\d+[A-Z]\d+(\d{4})\s", line)
            if match:
                return match.group(1)
            if not line.startswith("HDR"):
                break
    return None


def parse_fix_ident(line: str) -> str:
    """Extract the terminal fix or runway ident from a CIFP coordinate record."""
    return line[13:19].strip().upper()


def parse_fix_region_code(line: str) -> str | None:
    """Extract the two-character ARINC region code associated with a fix."""
    region_code = line[19:21].strip().upper()
    return region_code or None


def parse_fix_altitude(line: str, coord_end: int) -> int | None:
    """Parse runway/fix elevation when it is encoded immediately after coordinates."""
    after_coords = line[coord_end : min(coord_end + 25, len(line))]
    return parse_signed_altitude_ft(after_coords)


def build_airport_fix_index(faacifp_path: Path, airport: str) -> dict[str, FixRecord]:
    """Collect airport-local CIFP coordinate records from SUSAP <airport> records."""
    fixes: dict[str, FixRecord] = {}
    airport_prefix = f"SUSAP {airport.upper()}"

    with faacifp_path.open(encoding="ascii", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n\r")
            if not line.startswith(airport_prefix):
                continue

            match = COORD_PAIR_RE.search(line)
            if not match:
                continue

            ident = parse_fix_ident(line)
            if not ident:
                continue

            lon, lat = decode_coordinate_pair(match.group(1), match.group(2))
            altitude_ft = parse_fix_altitude(line, match.end())
            fixes[ident] = FixRecord(
                ident=ident,
                lon=lon,
                lat=lat,
                altitude_ft=altitude_ft,
                source_line=line_number,
                region_code=parse_fix_region_code(line),
                source_kind="airport-local-cifp",
            )

    return fixes


def preferred_region_codes_by_fix(legs: list[ProcedureLeg]) -> dict[str, set[str]]:
    """Return region-code hints from procedure legs for non-local fix fallback."""
    regions: dict[str, set[str]] = {}
    for leg in legs:
        if not leg.fix_ident or leg.fix_region_code is None:
            continue
        regions.setdefault(leg.fix_ident, set()).add(leg.fix_region_code)
    return regions


def build_global_fix_fallback_index(
    faacifp_path: Path,
    airport: str,
    missing_idents: set[str],
    preferred_regions: dict[str, set[str]],
) -> dict[str, FixRecord]:
    """Resolve missing fixes from non-airport CIFP records using ident + region."""
    if not missing_idents:
        return {}

    airport_prefix = f"SUSAP {airport.upper()}"
    candidates: dict[str, tuple[int, FixRecord]] = {}

    with faacifp_path.open(encoding="ascii", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n\r")
            if line.startswith(airport_prefix):
                continue

            match = COORD_PAIR_RE.search(line)
            if not match:
                continue

            ident = parse_fix_ident(line)
            if ident not in missing_idents:
                continue

            region_code = parse_fix_region_code(line)
            expected_regions = preferred_regions.get(ident, set())
            if expected_regions and region_code not in expected_regions:
                continue

            lon, lat = decode_coordinate_pair(match.group(1), match.group(2))
            altitude_ft = parse_fix_altitude(line, match.end())
            score = 0 if region_code in expected_regions else 1
            existing = candidates.get(ident)
            if existing is not None and existing[0] <= score:
                continue

            candidates[ident] = (
                score,
                FixRecord(
                    ident=ident,
                    lon=lon,
                    lat=lat,
                    altitude_ft=altitude_ft,
                    source_line=line_number,
                    region_code=region_code,
                    source_kind="global-cifp-fallback",
                ),
            )

    return {ident: record for ident, (_, record) in candidates.items()}


def build_fix_index(
    faacifp_path: Path,
    airport: str,
    procedure_legs: list[ProcedureLeg] | None = None,
) -> dict[str, FixRecord]:
    """Collect airport-local fixes and fill missing procedure fixes from global CIFP."""
    fixes = build_airport_fix_index(faacifp_path, airport)
    if procedure_legs is None:
        return fixes

    required_idents = {leg.fix_ident for leg in procedure_legs if leg.fix_ident}
    missing_idents = required_idents - set(fixes)
    fixes.update(
        build_global_fix_fallback_index(
            faacifp_path=faacifp_path,
            airport=airport,
            missing_idents=missing_idents,
            preferred_regions=preferred_region_codes_by_fix(procedure_legs),
        )
    )
    return fixes


def parse_leg_type(line: str) -> str | None:
    """Extract the ARINC path terminator from the procedure-leg field."""
    match = LEG_TYPE_RE.search(line[40:70])
    return match.group(1) if match else None


def parse_leg_role(line: str, leg_type: str, fix_ident: str, sequence: int) -> str:
    """Map CIFP waypoint description hints to display roles."""
    waypoint_description = line[42].strip().upper() if len(line) > 42 else ""
    if fix_ident.startswith("RW"):
        return "MAPt"
    if leg_type in {"HM", "HF"}:
        return "MAHF"
    if waypoint_description == "F":
        return "FAF"
    if waypoint_description == "I" or leg_type == "IF":
        return "IF"
    if sequence <= 20:
        return "IF"
    return "Route"


def parse_procedure_legs(
    faacifp_path: Path,
    airport: str,
    procedure: str,
    branch: str,
) -> list[ProcedureLeg]:
    """Read matching fixed-width final/intermediate approach records."""
    legs: list[ProcedureLeg] = []
    airport_prefix = f"SUSAP {airport.upper()}"
    target_procedure = procedure.upper()
    target_branch = branch.upper()

    with faacifp_path.open(encoding="ascii", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n\r")
            if not line.startswith(airport_prefix):
                continue
            if len(line) < 70 or line[12] != "F":
                continue
            if line[13:19].strip().upper() != target_procedure:
                continue
            if line[19:26].strip().upper() != target_branch:
                continue

            sequence_text = line[26:29].strip()
            fix_ident = line[29:34].strip().upper()
            leg_type = parse_leg_type(line)
            if not sequence_text.isdigit() or leg_type is None:
                continue

            sequence = int(sequence_text)
            role = parse_leg_role(line, leg_type, fix_ident, sequence)
            altitude_ft = parse_leg_altitude_ft(line)
            legs.append(
                ProcedureLeg(
                    sequence=sequence,
                    branch=target_branch,
                    fix_ident=fix_ident,
                    leg_type=leg_type,
                    role=role,
                    altitude_ft=altitude_ft,
                    source_line=line_number,
                    fix_region_code=line[34:36].strip().upper() or None,
                )
            )

    return sorted(legs, key=lambda item: (item.sequence, item.source_line))


def parse_available_branches(
    faacifp_path: Path,
    airport: str,
    procedure: str,
) -> list[str]:
    """Return available branch idents for one procedure in CIFP order."""
    branches: list[str] = []
    seen: set[str] = set()
    airport_prefix = f"SUSAP {airport.upper()}"
    target_procedure = procedure.upper()

    with faacifp_path.open(encoding="ascii", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            if not line.startswith(airport_prefix):
                continue
            if len(line) < 29 or line[12] != "F":
                continue
            if line[13:19].strip().upper() != target_procedure:
                continue

            branch = line[19:26].strip().upper()
            if branch and branch not in seen:
                branches.append(branch)
                seen.add(branch)

    return sorted(branches, key=lambda item: branch_sort_key(target_procedure, item))


def distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    radius_m = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def build_route_points(
    legs: list[ProcedureLeg],
    fixes: dict[str, FixRecord],
    nominal_speed_kt: float,
) -> tuple[list[RoutePoint], list[str]]:
    """Resolve procedure legs into ordered 3D route points and warnings."""
    warnings: list[str] = []
    route_points: list[RoutePoint] = []
    elapsed_seconds = 0.0
    cumulative_distance_m = 0.0
    speed_mps = nominal_speed_kt * NM_TO_METRES / 3600.0
    previous_fix: FixRecord | None = None

    for leg in legs:
        if leg.leg_type not in SUPPORTED_ROUTE_LEGS:
            warnings.append(
                f"Skipped unsupported leg {leg.leg_type} at sequence {leg.sequence:03d}"
            )
            continue
        if not leg.fix_ident:
            warnings.append(f"Skipped sequence {leg.sequence:03d}: missing fix ident")
            continue

        fix = fixes.get(leg.fix_ident)
        if fix is None:
            warnings.append(
                f"Skipped sequence {leg.sequence:03d}: unresolved fix {leg.fix_ident}"
            )
            continue

        if previous_fix is not None:
            leg_distance_m = distance_m(previous_fix.lon, previous_fix.lat, fix.lon, fix.lat)
            cumulative_distance_m += leg_distance_m
            elapsed_seconds += leg_distance_m / speed_mps

        geometry_altitude_ft = leg.altitude_ft
        if leg.fix_ident.startswith("RW") and fix.altitude_ft is not None:
            geometry_altitude_ft = fix.altitude_ft
        elif geometry_altitude_ft is None:
            geometry_altitude_ft = fix.altitude_ft
        if geometry_altitude_ft is None:
            geometry_altitude_ft = 0
            warnings.append(
                f"Sequence {leg.sequence:03d} {leg.fix_ident}: altitude defaulted to 0 ft"
            )

        route_points.append(
            RoutePoint(
                sequence=leg.sequence,
                fix_ident=leg.fix_ident,
                leg_type=leg.leg_type,
                role=leg.role,
                lon=fix.lon,
                lat=fix.lat,
                altitude_ft=leg.altitude_ft,
                geometry_altitude_ft=geometry_altitude_ft,
                time_seconds=elapsed_seconds,
                distance_from_start_m=cumulative_distance_m,
                source_line=leg.source_line,
            )
        )
        previous_fix = fix

    if len(route_points) < 2:
        warnings.append("Fewer than two supported route points were resolved")

    return route_points, warnings


def procedure_display_name(procedure: str, runway: str | None) -> str:
    """Build a compact display name for the frontend label."""
    family = procedure_family(procedure)
    variant = procedure_variant(procedure)
    variant_suffix = f" {variant}" if variant else ""
    if runway and family == "RNAV_GPS":
        return f"RNAV(GPS){variant_suffix} {runway}"
    if runway and family == "RNAV_RNP":
        return f"RNAV(RNP){variant_suffix} {runway}"
    return procedure


def infer_runway(route_points: list[RoutePoint], procedure: str) -> str | None:
    """Infer runway from the first runway-like route point."""
    for point in route_points:
        if point.fix_ident.startswith("RW"):
            return point.fix_ident
    return runway_from_procedure_ident(procedure)


def build_leg_coverage(
    legs: list[ProcedureLeg],
    route_points: list[RoutePoint],
) -> dict[str, list[str]]:
    """Summarize parsed/rendered/skipped path terminators for UI warnings."""
    parsed = sorted({leg.leg_type for leg in legs})
    rendered = sorted({point.leg_type for point in route_points})
    skipped = sorted({leg.leg_type for leg in legs if leg.leg_type not in SUPPORTED_ROUTE_LEGS})
    simplified: list[str] = []
    return {
        "parsedLegTypes": parsed,
        "renderedLegTypes": rendered,
        "skippedLegTypes": skipped,
        "simplifiedLegTypes": simplified,
    }


def build_procedure_geojson(
    *,
    airport: str,
    procedure_type: str,
    procedure: str,
    branch: str,
    source_cycle: str | None,
    legs: list[ProcedureLeg],
    route_points: list[RoutePoint],
    warnings: list[str],
    nominal_speed_kt: float,
    tunnel_half_width_nm: float,
    tunnel_half_height_ft: float,
    sample_spacing_m: float,
) -> dict[str, Any]:
    """Build deterministic GeoJSON for route and fix rendering."""
    runway = infer_runway(route_points, procedure)
    procedure_name = procedure_display_name(procedure, runway)
    route_id = f"{airport.upper()}-{procedure.upper()}-{branch.upper()}"
    family = procedure_family(procedure)
    branch_type = branch_type_for(procedure, branch)
    variant = procedure_variant(procedure)
    leg_coverage = build_leg_coverage(legs, route_points)

    coordinates = [
        [
            round(point.lon, 8),
            round(point.lat, 8),
            round(point.geometry_altitude_ft * FEET_TO_METRES, 2),
        ]
        for point in route_points
    ]

    samples = [
        {
            "sequence": point.sequence,
            "fixIdent": point.fix_ident,
            "legType": point.leg_type,
            "role": point.role,
            "altitudeFt": point.altitude_ft,
            "geometryAltitudeFt": point.geometry_altitude_ft,
            "distanceFromStartM": round(point.distance_from_start_m, 1),
            "timeSeconds": round(point.time_seconds, 1),
            "sourceLine": point.source_line,
        }
        for point in route_points
    ]

    common_props = {
        "airport": airport.upper(),
        "procedureType": procedure_type.upper(),
        "procedureIdent": procedure.upper(),
        "procedureName": procedure_name,
        "branch": branch.upper(),
        "branchIdent": branch.upper(),
        "branchType": branch_type,
        "procedureFamily": family,
        "procedureVariant": variant,
        "runway": runway,
        "runwayIdent": runway,
        "source": "FAA-CIFP",
        "sourceCycle": source_cycle,
        "researchUseOnly": True,
    }

    features: list[dict[str, Any]] = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                **common_props,
                "featureType": "procedure-route",
                "routeId": route_id,
                "defaultVisible": branch_type == "final",
                "legCoverage": leg_coverage,
                "nominalSpeedKt": nominal_speed_kt,
                "tunnel": {
                    "lateralHalfWidthNm": tunnel_half_width_nm,
                    "verticalHalfHeightFt": tunnel_half_height_ft,
                    "sampleSpacingM": sample_spacing_m,
                    "mode": "visualApproximation",
                },
                "samples": samples,
                "warnings": warnings,
            },
        }
    ]

    for point, coordinates_item in zip(route_points, coordinates):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coordinates_item,
                },
                "properties": {
                    **common_props,
                    "featureType": "procedure-fix",
                    "routeId": route_id,
                    "name": point.fix_ident,
                    "sequence": point.sequence,
                    "legType": point.leg_type,
                    "role": point.role,
                    "altitudeFt": point.altitude_ft,
                    "geometryAltitudeFt": point.geometry_altitude_ft,
                    "distanceFromStartM": round(point.distance_from_start_m, 1),
                    "timeSeconds": round(point.time_seconds, 1),
                    "sourceLine": point.source_line,
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "metadata": {
            "airport": airport.upper(),
            "procedureType": procedure_type.upper(),
            "procedureIdent": procedure.upper(),
            "branch": branch.upper(),
            "branchType": branch_type,
            "procedureFamily": family,
            "procedureVariant": variant,
            "runwayIdent": runway,
            "sourceCycle": source_cycle,
            "researchUseOnly": True,
            "warnings": warnings,
        },
        "features": features,
    }


def generate_procedure_geojson(
    cifp_root: Path,
    airport: str,
    procedure_type: str,
    procedure: str,
    branch: str,
    nominal_speed_kt: float,
    tunnel_half_width_nm: float,
    tunnel_half_height_ft: float,
    sample_spacing_m: float,
) -> dict[str, Any]:
    """Build the legacy GeoJSON view from the canonical procedure-detail document."""
    documents = build_procedure_detail_documents(
        cifp_root=cifp_root,
        airport=airport,
        procedure_type=procedure_type,
        procedures=[procedure],
        nominal_speed_kt=nominal_speed_kt,
        tunnel_half_width_nm=tunnel_half_width_nm,
        tunnel_half_height_ft=tunnel_half_height_ft,
        sample_spacing_m=sample_spacing_m,
    )
    source_cycle = parse_source_cycle(cifp_root / "FAACIFP18")
    return procedure_detail_documents_to_geojson(
        airport=airport,
        procedure_type=procedure_type,
        source_cycle=source_cycle,
        documents=documents,
        include_transitions=False,
        branch=branch,
    )


def procedure_sort_key(procedure: str) -> tuple[int, str]:
    """Sort by runway first, then procedure ident."""
    runway = runway_from_procedure_ident(procedure)
    try:
        runway_index = RUNWAY_ORDER.index(runway or "")
    except ValueError:
        runway_index = len(RUNWAY_ORDER)
    return (runway_index, procedure)


def merge_feature_collections(
    *,
    airport: str,
    procedure_type: str,
    source_cycle: str | None,
    collections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge route/fix features for multiple procedure branches."""
    features: list[dict[str, Any]] = []
    procedure_families: list[str] = []
    procedure_idents: list[str] = []
    runway_idents: list[str] = []
    warnings: list[str] = []

    for collection in collections:
        features.extend(collection["features"])
        metadata = collection.get("metadata", {})
        family = metadata.get("procedureFamily")
        procedure_ident = metadata.get("procedureIdent")
        runway_ident = metadata.get("runwayIdent")
        if isinstance(family, str) and family not in procedure_families:
            procedure_families.append(family)
        if isinstance(procedure_ident, str) and procedure_ident not in procedure_idents:
            procedure_idents.append(procedure_ident)
        if isinstance(runway_ident, str) and runway_ident not in runway_idents:
            runway_idents.append(runway_ident)
        for warning in metadata.get("warnings", []):
            if isinstance(warning, str):
                warnings.append(f"{procedure_ident}: {warning}")

    procedure_idents.sort(key=procedure_sort_key)
    runway_idents.sort(key=lambda item: RUNWAY_ORDER.index(item) if item in RUNWAY_ORDER else 999)
    procedure_families.sort()

    return {
        "type": "FeatureCollection",
        "metadata": {
            "airport": airport.upper(),
            "procedureType": procedure_type.upper(),
            "procedureFamilies": procedure_families,
            "procedureIdents": procedure_idents,
            "runwayIdents": runway_idents,
            "sourceCycle": source_cycle,
            "generatedAt": None,
            "researchUseOnly": True,
            "warnings": warnings,
        },
        "features": features,
    }


def generate_procedures_geojson(
    cifp_root: Path,
    airport: str,
    procedure_type: str,
    procedures: list[str],
    include_transitions: bool,
    branch: str,
    nominal_speed_kt: float,
    tunnel_half_width_nm: float,
    tunnel_half_height_ft: float,
    sample_spacing_m: float,
) -> dict[str, Any]:
    """Generate the legacy GeoJSON projection from canonical detail documents."""
    source_cycle = parse_source_cycle(cifp_root / "FAACIFP18")
    documents = build_procedure_detail_documents(
        cifp_root=cifp_root,
        airport=airport,
        procedure_type=procedure_type,
        procedures=procedures,
        nominal_speed_kt=nominal_speed_kt,
        tunnel_half_width_nm=tunnel_half_width_nm,
        tunnel_half_height_ft=tunnel_half_height_ft,
        sample_spacing_m=sample_spacing_m,
    )
    return procedure_detail_documents_to_geojson(
        airport=airport,
        procedure_type=procedure_type,
        source_cycle=source_cycle,
        documents=documents,
        include_transitions=include_transitions,
        branch=branch,
    )


def runway_label(runway_ident: str | None) -> str:
    if not runway_ident:
        return "Unknown runway"
    normalized = runway_ident.upper()
    if normalized.startswith("RW"):
        return f"RWY {normalized[2:]}"
    return normalized


def procedure_chart_name(procedure: str, runway: str | None) -> str:
    family = procedure_family(procedure)
    variant = procedure_variant(procedure)
    variant_suffix = f" {variant}" if variant else ""
    if family == "RNAV_GPS":
        return f"RNAV(GPS){variant_suffix} {runway_label(runway)}"
    if family == "RNAV_RNP":
        return f"RNAV(RNP){variant_suffix} {runway_label(runway)}"
    return procedure_display_name(procedure, runway)


def normalize_fix_ref(fix_ident: str) -> str:
    return f"fix:{fix_ident.upper()}"


def normalize_branch_ref(branch_ident: str) -> str:
    return f"branch:{branch_ident.upper()}"


def approach_modes_for(procedure: str) -> list[str]:
    family = procedure_family(procedure)
    if family == "RNAV_GPS":
        return ["LPV", "LNAV/VNAV", "LNAV"]
    if family == "RNAV_RNP":
        return ["RNP AR"]
    return []


def path_construction_method(leg_type: str) -> str:
    return {
        "IF": "if_to_fix",
        "TF": "track_to_fix",
        "DF": "direct_to_fix",
        "CF": "course_to_fix",
        "CA": "course_to_altitude",
        "FA": "course_to_altitude",
        "HM": "hold_to_manual",
        "HF": "hold_to_fix",
        "RF": "radius_to_fix",
        "VI": "heading_to_intercept",
        "VA": "heading_to_altitude",
    }.get(leg_type.upper(), "procedure_leg")


def segment_type_for_leg(leg: ProcedureLeg, *, has_crossed_threshold: bool) -> str:
    if has_crossed_threshold:
        return "missed"
    if leg.fix_ident.startswith("RW") or leg.role.upper() in {"FAF", "MAPT"}:
        return "final"
    if leg.role.upper() == "IAF":
        return "initial"
    if leg.role.upper() == "IF":
        return "intermediate"
    return "route"


def preferred_geometry_altitude_ft(
    leg: ProcedureLeg,
    fix: FixRecord | None,
) -> int | None:
    if leg.fix_ident.startswith("RW") and fix is not None and fix.altitude_ft is not None:
        return fix.altitude_ft
    if leg.altitude_ft is not None:
        return leg.altitude_ft
    if fix is not None and fix.altitude_ft is not None:
        return fix.altitude_ft
    return None


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_fix_catalog(
    *,
    ordered_fix_idents: list[str],
    fix_records: dict[str, FixRecord],
    role_hints_by_fix: dict[str, list[str]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for fix_ident in ordered_fix_idents:
        fix = fix_records.get(fix_ident)
        role_hints = dedupe_preserve_order(role_hints_by_fix.get(fix_ident, []))
        if fix_ident.startswith("RW"):
            kind = "runway_threshold"
        elif "MAHF" in role_hints:
            kind = "missed_hold_fix"
        elif "FAF" in role_hints:
            kind = "final_approach_fix"
        elif "MAPT" in {item.upper() for item in role_hints}:
            kind = "missed_approach_point"
        elif "IAF" in role_hints:
            kind = "initial_approach_fix"
        else:
            kind = "named_fix"

        items.append(
            {
                "fixId": normalize_fix_ref(fix_ident),
                "ident": fix_ident,
                "kind": kind,
                "position": (
                    {
                        "lon": round(fix.lon, 8),
                        "lat": round(fix.lat, 8),
                    }
                    if fix is not None
                    else None
                ),
                "elevationFt": None if fix is None else fix.altitude_ft,
                "roleHints": role_hints,
                "sourceRefs": (
                    ["src:cifp-detail", "src:cifp-global-fix"]
                    if fix is not None and fix.source_kind == "global-cifp-fallback"
                    else ["src:cifp-detail"]
                ),
            }
        )
    return items


def merge_fix_ref_for_branch(
    branch_legs: list[ProcedureLeg],
    final_fix_idents: set[str],
    final_branch_ident: str,
    branch_ident: str,
) -> str | None:
    if branch_ident.upper() == final_branch_ident.upper():
        return None
    for leg in branch_legs:
        if leg.fix_ident in final_fix_idents:
            return normalize_fix_ref(leg.fix_ident)
    return None


def route_points_by_sequence(route_points: list[RoutePoint]) -> dict[int, RoutePoint]:
    return {point.sequence: point for point in route_points}


def build_branch_document(
    *,
    branch_ident: str,
    branch_order: int,
    final_branch_ident: str,
    branch_legs: list[ProcedureLeg],
    branch_route_points: list[RoutePoint],
    fix_records: dict[str, FixRecord],
    branch_warnings: list[str],
    final_fix_idents: set[str],
) -> dict[str, Any]:
    points_by_sequence = route_points_by_sequence(branch_route_points)
    has_crossed_threshold = False
    leg_documents: list[dict[str, Any]] = []

    for index, leg in enumerate(branch_legs):
        fix = fix_records.get(leg.fix_ident)
        geometry_altitude_ft = preferred_geometry_altitude_ft(leg, fix)
        previous_fix_ident = branch_legs[index - 1].fix_ident if index > 0 else None
        segment_type = segment_type_for_leg(leg, has_crossed_threshold=has_crossed_threshold)
        point = points_by_sequence.get(leg.sequence)
        quality_status = "exact" if fix is not None else "incomplete"
        if point is not None and geometry_altitude_ft is None:
            branch_warnings.append(
                f"Sequence {leg.sequence:03d} {leg.fix_ident}: altitude constraint missing in CIFP; profile may interpolate or use nearest available altitude"
            )

        leg_documents.append(
            {
                "legId": f"leg:{branch_ident.upper()}:{leg.sequence:03d}",
                "sequence": leg.sequence,
                "segmentType": segment_type,
                "path": {
                    "pathTerminator": leg.leg_type,
                    "constructionMethod": path_construction_method(leg.leg_type),
                    "startFixRef": None if previous_fix_ident is None else normalize_fix_ref(previous_fix_ident),
                    "endFixRef": normalize_fix_ref(leg.fix_ident),
                },
                "termination": {
                    "kind": "fix",
                    "fixRef": normalize_fix_ref(leg.fix_ident),
                },
                "constraints": {
                    "altitude": (
                        {
                            "qualifier": "at",
                            "valueFt": leg.altitude_ft,
                            "rawText": f"{leg.altitude_ft} ft",
                        }
                        if leg.altitude_ft is not None
                        else None
                    ),
                    "speedKt": None,
                    "geometryAltitudeFt": geometry_altitude_ft,
                },
                "roleAtEnd": leg.role,
                "sourceRefs": ["src:cifp-detail"],
                "quality": {
                    "status": quality_status,
                    "sourceLine": leg.source_line,
                    "renderedInPlanView": point is not None,
                },
            }
        )

        if leg.fix_ident.startswith("RW"):
            has_crossed_threshold = True

    branch_role = "final" if branch_ident.upper() == final_branch_ident.upper() else "transition"

    return {
        "branchId": normalize_branch_ref(branch_ident),
        "branchIdent": branch_ident.upper(),
        "branchRole": branch_role,
        "sequenceOrder": branch_order,
        "mergeFixRef": merge_fix_ref_for_branch(
            branch_legs,
            final_fix_idents,
            final_branch_ident=final_branch_ident,
            branch_ident=branch_ident,
        ),
        "continuesWithBranchId": (
            normalize_branch_ref(final_branch_ident)
            if branch_ident.upper() != final_branch_ident.upper()
            else None
        ),
        "defaultVisible": branch_ident.upper() == final_branch_ident.upper(),
        "warnings": branch_warnings,
        "legs": leg_documents,
    }


def build_vertical_profile_document(
    *,
    procedure: str,
    runway: str | None,
    final_branch_ident: str,
    branch_legs: list[ProcedureLeg],
    branch_route_points: list[RoutePoint],
    fix_records: dict[str, FixRecord],
) -> list[dict[str, Any]]:
    if not branch_legs or not branch_route_points:
        return []

    points_by_sequence = route_points_by_sequence(branch_route_points)
    constraint_samples: list[dict[str, Any]] = []
    warnings: list[str] = []

    for point in branch_route_points:
        leg = next((candidate for candidate in branch_legs if candidate.sequence == point.sequence), None)
        if leg is None:
            continue
        fix = fix_records.get(point.fix_ident)
        geometry_altitude_ft = preferred_geometry_altitude_ft(leg, fix)
        if geometry_altitude_ft is None:
            warnings.append(
                f"{point.fix_ident}: altitude unavailable in CIFP detail; frontend will interpolate for display"
            )

        constraint_samples.append(
            {
                "fixRef": normalize_fix_ref(point.fix_ident),
                "ident": point.fix_ident,
                "role": leg.role,
                "distanceFromStartM": round(point.distance_from_start_m, 1),
                "altitudeFt": leg.altitude_ft,
                "geometryAltitudeFt": geometry_altitude_ft,
                "sourceLine": point.source_line,
            }
        )

    if not constraint_samples:
        return []

    return [
        {
            "profileId": f"profile:{procedure.upper()}:{(runway or 'UNKNOWN').upper()}",
            "appliesToModes": approach_modes_for(procedure),
            "branchId": normalize_branch_ref(final_branch_ident),
            "fromFixRef": constraint_samples[0]["fixRef"],
            "toFixRef": constraint_samples[-1]["fixRef"],
            "basis": "cifp_leg_constraints",
            "glidepathAngleDeg": None,
            "thresholdCrossingHeightFt": None,
            "constraintSamples": constraint_samples,
            "warnings": dedupe_preserve_order(warnings),
        }
    ]


def build_validation_block(
    *,
    runway: str | None,
    branches: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    final_branch = next((branch for branch in branches if branch["branchRole"] == "final"), None)
    final_legs = [] if final_branch is None else final_branch["legs"]
    expected_if = next(
        (leg["path"]["endFixRef"] for leg in final_legs if leg.get("roleAtEnd") == "IF"),
        None,
    )
    expected_faf = next(
        (leg["path"]["endFixRef"] for leg in final_legs if leg.get("roleAtEnd") == "FAF"),
        None,
    )
    expected_mapt = next(
        (
            leg["path"]["endFixRef"]
            for leg in final_legs
            if str(leg["path"]["endFixRef"]).endswith((runway or "").upper())
        ),
        None,
    )
    threshold_seen = False
    expected_missed_fix = None
    for leg in final_legs:
        fix_ref = leg["path"]["endFixRef"]
        if runway and str(fix_ref).endswith(runway.upper()):
            threshold_seen = True
            continue
        if not threshold_seen or fix_ref == "fix:":
            continue
        if leg["path"]["pathTerminator"] in {"HM", "HF"}:
            expected_missed_fix = fix_ref
            break
        if expected_missed_fix is None:
            expected_missed_fix = fix_ref

    return {
        "expectedRunwayIdent": runway,
        "expectedIF": expected_if,
        "expectedFAF": expected_faf,
        "expectedMAPt": expected_mapt,
        "expectedMissedHoldFix": expected_missed_fix,
        "knownSimplifications": dedupe_preserve_order(warnings),
    }


def load_airport_details(airport: str) -> dict[str, Any]:
    return find_airport_record(common_data_path("airports.csv"), airport)


def build_procedure_detail_document(
    *,
    cifp_root: Path,
    airport: str,
    procedure_type: str,
    procedure: str,
    nominal_speed_kt: float,
    tunnel_half_width_nm: float,
    tunnel_half_height_ft: float,
    sample_spacing_m: float,
) -> dict[str, Any]:
    faacifp_path = cifp_root / "FAACIFP18"
    source_cycle = parse_source_cycle(faacifp_path)
    airport_details = load_airport_details(airport)
    available_branches = parse_available_branches(faacifp_path, airport, procedure)
    final_branch_ident = final_branch_for_procedure(procedure)
    if final_branch_ident not in available_branches:
        available_branches = [final_branch_ident, *available_branches]
    branch_legs_by_ident = {
        branch_ident.upper(): parse_procedure_legs(faacifp_path, airport, procedure, branch_ident)
        for branch_ident in available_branches
    }
    fix_records = build_fix_index(
        faacifp_path,
        airport,
        procedure_legs=[
            leg
            for branch_legs in branch_legs_by_ident.values()
            for leg in branch_legs
        ],
    )

    runway = runway_from_procedure_ident(procedure)
    ordered_fix_idents: list[str] = []
    role_hints_by_fix: dict[str, list[str]] = {}
    branch_documents: list[dict[str, Any]] = []
    branch_route_points_by_ident: dict[str, list[RoutePoint]] = {}
    warnings: list[str] = []

    for branch_order, branch_ident in enumerate(available_branches, start=1):
        branch_legs = branch_legs_by_ident[branch_ident.upper()]
        branch_route_points, branch_warnings = build_route_points(branch_legs, fix_records, nominal_speed_kt)
        branch_route_points_by_ident[branch_ident.upper()] = branch_route_points
        warnings.extend(f"[{branch_ident}] {warning}" for warning in branch_warnings)

        ordered_fix_idents.extend(leg.fix_ident for leg in branch_legs if leg.fix_ident)
        for leg in branch_legs:
            role_hints_by_fix.setdefault(leg.fix_ident, []).append(leg.role)

        branch_documents.append(
            build_branch_document(
                branch_ident=branch_ident,
                branch_order=branch_order,
                final_branch_ident=final_branch_ident,
                branch_legs=branch_legs,
                branch_route_points=branch_route_points,
                fix_records=fix_records,
                branch_warnings=branch_warnings,
                final_fix_idents={point.fix_ident for point in branch_route_points_by_ident.get(final_branch_ident, [])},
            )
        )

    ordered_fix_idents = dedupe_preserve_order(ordered_fix_idents)
    fix_catalog = build_fix_catalog(
        ordered_fix_idents=ordered_fix_idents,
        fix_records=fix_records,
        role_hints_by_fix=role_hints_by_fix,
    )

    threshold_fix = fix_records.get(runway or "")
    chart_name = procedure_chart_name(procedure, runway)
    final_branch_legs = branch_legs_by_ident.get(final_branch_ident.upper(), [])
    vertical_profiles = build_vertical_profile_document(
        procedure=procedure,
        runway=runway,
        final_branch_ident=final_branch_ident,
        branch_legs=final_branch_legs,
        branch_route_points=branch_route_points_by_ident.get(final_branch_ident, []),
        fix_records=fix_records,
    )

    research_warnings = dedupe_preserve_order(warnings)
    procedure_uid = f"{airport.upper()}-{procedure.upper()}-{(runway or 'UNKNOWN').upper()}"
    return {
        "schemaVersion": "1.0.0",
        "modelType": "rnav-procedure-runway",
        "procedureUid": procedure_uid,
        "provenance": {
            "assemblyMode": "cifp_primary_export",
            "researchUseOnly": True,
            "sources": [
                {
                    "sourceId": "src:cifp-index",
                    "kind": "FAA_CIFP_INDEX",
                    "cycle": source_cycle,
                    "path": str(cifp_root / "IN_CIFP.txt"),
                },
                {
                    "sourceId": "src:cifp-detail",
                    "kind": "FAA_CIFP",
                    "cycle": source_cycle,
                    "path": str(cifp_root / "FAACIFP18"),
                },
                {
                    "sourceId": "src:cifp-global-fix",
                    "kind": "FAA_CIFP_ENROUTE_OR_GLOBAL_FIX",
                    "cycle": source_cycle,
                    "path": str(cifp_root / "FAACIFP18"),
                },
            ],
            "warnings": research_warnings,
        },
        "airport": {
            "icao": airport_details["icao_code"],
            "faa": airport_details["faa_code"],
            "name": airport_details["name"],
        },
        "runway": {
            "ident": runway,
            "landingThresholdFixRef": None if runway is None else normalize_fix_ref(runway),
            "threshold": (
                {
                    "lon": round(threshold_fix.lon, 8),
                    "lat": round(threshold_fix.lat, 8),
                    "elevationFt": threshold_fix.altitude_ft,
                }
                if threshold_fix is not None
                else None
            ),
        },
        "procedure": {
            "procedureType": procedure_type.upper(),
            "procedureFamily": procedure_family(procedure),
            "procedureIdent": procedure.upper(),
            "chartName": chart_name,
            "variant": procedure_variant(procedure),
            "runwayIdent": runway,
            "baseBranchIdent": final_branch_ident,
            "approachModes": approach_modes_for(procedure),
        },
        "fixes": fix_catalog,
        "branches": branch_documents,
        "verticalProfiles": vertical_profiles,
        "validation": build_validation_block(
            runway=runway,
            branches=branch_documents,
            warnings=research_warnings,
        ),
        "displayHints": {
            "nominalSpeedKt": nominal_speed_kt,
            "defaultVisibleBranchIds": [
                branch["branchId"] for branch in branch_documents if branch["defaultVisible"]
            ],
            "tunnelDefaults": {
                "lateralHalfWidthNm": tunnel_half_width_nm,
                "verticalHalfHeightFt": tunnel_half_height_ft,
                "sampleSpacingM": sample_spacing_m,
                "mode": "visualApproximation",
            },
        },
    }


def build_procedure_detail_documents(
    *,
    cifp_root: Path,
    airport: str,
    procedure_type: str,
    procedures: list[str],
    nominal_speed_kt: float,
    tunnel_half_width_nm: float,
    tunnel_half_height_ft: float,
    sample_spacing_m: float,
) -> list[dict[str, Any]]:
    """Build canonical procedure documents used by every downstream view."""
    index_path = cifp_root / "IN_CIFP.txt"
    faacifp_path = cifp_root / "FAACIFP18"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing CIFP index: {index_path}")
    if not faacifp_path.exists():
        raise FileNotFoundError(f"Missing CIFP detail file: {faacifp_path}")

    documents: list[dict[str, Any]] = []
    for procedure in sorted({item.upper() for item in procedures}, key=procedure_sort_key):
        if not procedure_exists(index_path, airport, procedure_type, procedure):
            raise ValueError(f"{airport} {procedure_type} {procedure} not found in {index_path}")
        documents.append(
            build_procedure_detail_document(
                cifp_root=cifp_root,
                airport=airport,
                procedure_type=procedure_type,
                procedure=procedure,
                nominal_speed_kt=nominal_speed_kt,
                tunnel_half_width_nm=tunnel_half_width_nm,
                tunnel_half_height_ft=tunnel_half_height_ft,
                sample_spacing_m=sample_spacing_m,
            )
        )
    return documents


def detail_fix_index(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {fix["fixId"]: fix for fix in document.get("fixes", [])}


def detail_leg_is_renderable(leg: dict[str, Any], fix: dict[str, Any] | None) -> bool:
    if leg.get("quality", {}).get("renderedInPlanView") is not True:
        return False
    position = None if fix is None else fix.get("position")
    return (
        isinstance(position, dict)
        and isinstance(position.get("lon"), (int, float))
        and isinstance(position.get("lat"), (int, float))
    )


def detail_leg_geometry_altitude_ft(leg: dict[str, Any], fix: dict[str, Any] | None) -> int:
    constraints = leg.get("constraints", {})
    geometry_altitude_ft = constraints.get("geometryAltitudeFt")
    if isinstance(geometry_altitude_ft, (int, float)):
        return int(round(geometry_altitude_ft))

    altitude = constraints.get("altitude")
    if isinstance(altitude, dict) and isinstance(altitude.get("valueFt"), (int, float)):
        return int(round(altitude["valueFt"]))

    if fix is not None and isinstance(fix.get("elevationFt"), (int, float)):
        return int(round(fix["elevationFt"]))

    return 0


def detail_branch_leg_coverage(branch: dict[str, Any]) -> dict[str, list[str]]:
    parsed = sorted({leg["path"]["pathTerminator"] for leg in branch.get("legs", [])})
    rendered = sorted(
        {
            leg["path"]["pathTerminator"]
            for leg in branch.get("legs", [])
            if leg.get("quality", {}).get("renderedInPlanView") is True
        }
    )
    skipped = sorted(set(parsed) - set(rendered))
    return {
        "parsedLegTypes": parsed,
        "renderedLegTypes": rendered,
        "skippedLegTypes": skipped,
        "simplifiedLegTypes": skipped,
    }


def build_geojson_route_from_detail_branch(
    *,
    document: dict[str, Any],
    branch: dict[str, Any],
) -> dict[str, Any] | None:
    """Project one canonical detail branch into the legacy GeoJSON feature shape."""
    fix_by_id = detail_fix_index(document)
    procedure = document["procedure"]
    airport = document["airport"]["icao"]
    route_id = f"{airport}-{procedure['procedureIdent']}-{branch['branchIdent']}"
    defaults = document["displayHints"]["tunnelDefaults"]
    nominal_speed_kt = document["displayHints"]["nominalSpeedKt"]
    speed_mps = nominal_speed_kt * NM_TO_METRES / 3600.0
    warnings = list(branch.get("warnings", []))

    points: list[dict[str, Any]] = []
    cumulative_distance_m = 0.0
    elapsed_seconds = 0.0
    previous_point: dict[str, Any] | None = None

    for leg in branch.get("legs", []):
        fix_ref = leg["path"]["endFixRef"]
        fix = fix_by_id.get(fix_ref)
        if not detail_leg_is_renderable(leg, fix):
            continue

        position = fix["position"]
        geometry_altitude_ft = detail_leg_geometry_altitude_ft(leg, fix)
        if previous_point is not None:
            leg_distance_m = distance_m(
                previous_point["lon"],
                previous_point["lat"],
                position["lon"],
                position["lat"],
            )
            cumulative_distance_m += leg_distance_m
            elapsed_seconds += leg_distance_m / speed_mps

        altitude = leg.get("constraints", {}).get("altitude")
        altitude_ft = altitude.get("valueFt") if isinstance(altitude, dict) else None
        point = {
            "sequence": leg["sequence"],
            "fixIdent": fix["ident"],
            "legType": leg["path"]["pathTerminator"],
            "role": leg["roleAtEnd"],
            "lon": round(position["lon"], 8),
            "lat": round(position["lat"], 8),
            "altitudeFt": altitude_ft,
            "geometryAltitudeFt": geometry_altitude_ft,
            "distanceFromStartM": round(cumulative_distance_m, 1),
            "timeSeconds": round(elapsed_seconds, 1),
            "sourceLine": leg["quality"]["sourceLine"],
        }
        points.append(point)
        previous_point = point

    if len(points) < 2:
        warnings.append(f"Skipped branch {branch['branchIdent']}: fewer than two supported route points")
        return None

    coordinates = [
        [point["lon"], point["lat"], round(point["geometryAltitudeFt"] * FEET_TO_METRES, 2)]
        for point in points
    ]
    samples = [
        {
            "sequence": point["sequence"],
            "fixIdent": point["fixIdent"],
            "legType": point["legType"],
            "role": point["role"],
            "altitudeFt": point["altitudeFt"],
            "geometryAltitudeFt": point["geometryAltitudeFt"],
            "distanceFromStartM": point["distanceFromStartM"],
            "timeSeconds": point["timeSeconds"],
            "sourceLine": point["sourceLine"],
        }
        for point in points
    ]
    common_props = {
        "airport": airport,
        "procedureType": procedure["procedureType"],
        "procedureIdent": procedure["procedureIdent"],
        "procedureName": procedure["chartName"],
        "branch": branch["branchIdent"],
        "branchIdent": branch["branchIdent"],
        "branchType": branch["branchRole"],
        "procedureFamily": procedure["procedureFamily"],
        "procedureVariant": procedure["variant"],
        "runway": procedure["runwayIdent"],
        "runwayIdent": procedure["runwayIdent"],
        "routeId": route_id,
        "source": "procedure-details",
        "sourceCycle": document["provenance"]["sources"][0]["cycle"],
        "researchUseOnly": True,
    }

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coordinates},
            "properties": {
                **common_props,
                "featureType": "procedure-route",
                "defaultVisible": branch["defaultVisible"],
                "legCoverage": detail_branch_leg_coverage(branch),
                "nominalSpeedKt": nominal_speed_kt,
                "tunnel": defaults,
                "samples": samples,
                "warnings": warnings,
            },
        }
    ]

    for point, coordinates_item in zip(points, coordinates):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": coordinates_item},
                "properties": {
                    **common_props,
                    "featureType": "procedure-fix",
                    "name": point["fixIdent"],
                    "sequence": point["sequence"],
                    "legType": point["legType"],
                    "role": point["role"],
                    "altitudeFt": point["altitudeFt"],
                    "geometryAltitudeFt": point["geometryAltitudeFt"],
                    "distanceFromStartM": point["distanceFromStartM"],
                    "timeSeconds": point["timeSeconds"],
                    "sourceLine": point["sourceLine"],
                },
            }
        )

    return {
        "metadata": {
            "procedureFamily": procedure["procedureFamily"],
            "procedureIdent": procedure["procedureIdent"],
            "runwayIdent": procedure["runwayIdent"],
            "warnings": warnings,
        },
        "features": features,
    }


def procedure_detail_documents_to_geojson(
    *,
    airport: str,
    procedure_type: str,
    source_cycle: str | None,
    documents: list[dict[str, Any]],
    include_transitions: bool,
    branch: str,
) -> dict[str, Any]:
    """Project canonical procedure details into the legacy GeoJSON compatibility layer."""
    collections: list[dict[str, Any]] = []
    requested_branch = branch.upper() if branch else None
    warnings: list[str] = []

    for document in documents:
        for branch_document in document["branches"]:
            if not include_transitions:
                target_branch = requested_branch or document["procedure"]["baseBranchIdent"]
                if branch_document["branchIdent"].upper() != target_branch.upper():
                    continue
            collection = build_geojson_route_from_detail_branch(
                document=document,
                branch=branch_document,
            )
            if collection is None:
                warnings.append(
                    f"{document['procedure']['procedureIdent']}:{branch_document['branchIdent']} skipped in GeoJSON projection"
                )
                continue
            collections.append(collection)

    merged = merge_feature_collections(
        airport=airport,
        procedure_type=procedure_type,
        source_cycle=source_cycle,
        collections=collections,
    )
    merged["metadata"]["canonicalDataLayer"] = "procedure-details"
    merged["metadata"]["projection"] = "procedures.geojson"
    merged["metadata"]["warnings"].extend(warnings)
    return merged


def sanitize_public_chart_filename(file_name: str) -> str:
    original = Path(file_name).name
    suffix = Path(original).suffix.lower() or ".pdf"
    stem = original[: -len(Path(original).suffix)] if Path(original).suffix else original
    cleaned = SAFE_FILE_CHARS_RE.sub("-", stem).strip("-.")
    cleaned = re.sub(r"-{2,}", "-", cleaned) or "chart"
    return f"{cleaned}{suffix}"


def infer_chart_targets(file_name: str) -> tuple[str | None, str | None]:
    match = CHART_PROCEDURE_RE.search(file_name.upper())
    if not match:
        return None, None
    variant = match.group(1) or ""
    runway_number = match.group(2).zfill(2)
    runway_side = match.group(3)
    runway_ident = f"RW{runway_number}{runway_side}"
    procedure_ident = f"R{runway_number}{runway_side}{variant}" if variant else f"R{runway_number}{runway_side}"
    return procedure_ident, runway_ident


def build_procedure_details_index(
    *,
    airport: str,
    airport_name: str,
    source_cycle: str | None,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    runways: dict[str, dict[str, Any]] = {}
    for document in documents:
        procedure = document["procedure"]
        runway_ident = procedure["runwayIdent"] or "UNKNOWN"
        runway_entry = runways.setdefault(
            runway_ident,
            {
                "runwayIdent": runway_ident,
                "chartName": procedure["chartName"],
                "procedureUids": [],
                "procedures": [],
            },
        )
        runway_entry["procedureUids"].append(document["procedureUid"])
        runway_entry["procedures"].append(
            {
                "procedureUid": document["procedureUid"],
                "procedureIdent": procedure["procedureIdent"],
                "chartName": procedure["chartName"],
                "procedureFamily": procedure["procedureFamily"],
                "variant": procedure["variant"],
                "approachModes": procedure["approachModes"],
                "runwayIdent": runway_ident,
                "defaultBranchId": document["procedure"]["baseBranchIdent"],
            }
        )

    runway_entries = sorted(
        runways.values(),
        key=lambda item: RUNWAY_ORDER.index(item["runwayIdent"]) if item["runwayIdent"] in RUNWAY_ORDER else 999,
    )

    return {
        "airport": airport.upper(),
        "airportName": airport_name,
        "sourceCycle": source_cycle,
        "researchUseOnly": True,
        "runways": runway_entries,
    }


def publish_local_chart_manifest(
    *,
    airport: str,
    chart_root: Path,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    by_procedure_ident = {
        document["procedure"]["procedureIdent"]: document
        for document in documents
    }
    by_runway_ident = {
        document["procedure"]["runwayIdent"]: document
        for document in documents
        if document["procedure"]["runwayIdent"]
    }
    chart_dir = airport_charts_dir(airport)
    chart_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    source_dir = chart_root / airport.upper()
    if not source_dir.exists():
        return {"airport": airport.upper(), "researchUseOnly": True, "charts": []}

    used_names: set[str] = set()
    for source_path in sorted(source_dir.glob("*.pdf")):
        procedure_ident, runway_ident = infer_chart_targets(source_path.name)
        target_document = None
        if procedure_ident and procedure_ident in by_procedure_ident:
            target_document = by_procedure_ident[procedure_ident]
        elif runway_ident and runway_ident in by_runway_ident:
            target_document = by_runway_ident[runway_ident]

        safe_name = sanitize_public_chart_filename(source_path.name)
        if safe_name in used_names:
            safe_stem = Path(safe_name).stem
            safe_suffix = Path(safe_name).suffix
            counter = 2
            while f"{safe_stem}-{counter}{safe_suffix}" in used_names:
                counter += 1
            safe_name = f"{safe_stem}-{counter}{safe_suffix}"
        used_names.add(safe_name)

        target_path = airport_chart_path(airport, safe_name)
        shutil.copyfile(source_path, target_path)

        procedure_uid = None if target_document is None else target_document["procedureUid"]
        title = (
            source_path.stem
            if target_document is None
            else target_document["procedure"]["chartName"]
        )
        entries.append(
            {
                "chartId": f"chart:{airport.upper()}:{safe_name}",
                "procedureUid": procedure_uid,
                "procedureIdent": procedure_ident,
                "runwayIdent": runway_ident,
                "title": title,
                "originalFileName": source_path.name,
                "sourcePath": str(source_path),
                "url": f"/data/airports/{airport.upper()}/charts/{safe_name}",
            }
        )

    return {
        "airport": airport.upper(),
        "researchUseOnly": True,
        "charts": entries,
    }


def publish_procedure_details_assets(
    *,
    cifp_root: Path,
    airport: str,
    procedure_type: str,
    procedures: list[str],
    nominal_speed_kt: float,
    tunnel_half_width_nm: float,
    tunnel_half_height_ft: float,
    sample_spacing_m: float,
    chart_root: Path,
) -> dict[str, Any]:
    airport_details = load_airport_details(airport)
    source_cycle = parse_source_cycle(cifp_root / "FAACIFP18")
    documents = build_procedure_detail_documents(
        cifp_root=cifp_root,
        airport=airport,
        procedure_type=procedure_type,
        procedures=procedures,
        nominal_speed_kt=nominal_speed_kt,
        tunnel_half_width_nm=tunnel_half_width_nm,
        tunnel_half_height_ft=tunnel_half_height_ft,
        sample_spacing_m=sample_spacing_m,
    )
    procedure_dir = airport_procedure_details_dir(airport)
    procedure_dir.mkdir(parents=True, exist_ok=True)

    for document in documents:
        detail_path = airport_procedure_details_path(airport, f"{document['procedureUid']}.json")
        detail_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    index_manifest = build_procedure_details_index(
        airport=airport,
        airport_name=airport_details["name"],
        source_cycle=source_cycle,
        documents=documents,
    )
    index_path = airport_procedure_details_path(airport, "index.json")
    index_path.write_text(json.dumps(index_manifest, indent=2) + "\n", encoding="utf-8")

    chart_manifest = publish_local_chart_manifest(
        airport=airport,
        chart_root=chart_root,
        documents=documents,
    )
    chart_index_path = airport_chart_path(airport, "index.json")
    chart_index_path.write_text(json.dumps(chart_manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "procedureIndexPath": index_path,
        "chartIndexPath": chart_index_path,
        "procedureCount": len(documents),
        "chartCount": len(chart_manifest["charts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate procedures.geojson from FAA CIFP")
    parser.add_argument("--cifp-root", default=str(DEFAULT_CIFP_ROOT), help="CIFP cycle directory")
    parser.add_argument("--airport", default=DEFAULT_AIRPORT, help="Airport ICAO code")
    parser.add_argument("--procedure-type", default="SIAP", help="CIFP procedure type")
    parser.add_argument("--procedure", default="R05LY", help="Procedure ident, e.g. R05LY")
    parser.add_argument(
        "--procedures",
        help="Comma-separated procedure idents, e.g. R05LY,R05RY,R23LY,R23RY,R32",
    )
    parser.add_argument(
        "--include-all-rnav",
        action="store_true",
        help="Generate all RNAV/RNP approach procedures listed for the selected airport",
    )
    parser.add_argument(
        "--include-transitions",
        action="store_true",
        help="Generate all CIFP branches for each selected procedure",
    )
    parser.add_argument("--branch", default="R", help="Procedure branch/transition to render")
    parser.add_argument("--nominal-speed-kt", type=float, default=140.0, help="4D gate speed")
    parser.add_argument("--tunnel-half-width-nm", type=float, default=0.3, help="Tunnel half-width")
    parser.add_argument("--tunnel-half-height-ft", type=float, default=300.0, help="Tunnel half-height")
    parser.add_argument("--sample-spacing-m", type=float, default=250.0, help="Tunnel sample spacing")
    parser.add_argument(
        "--charts-root",
        default=str(DEFAULT_RNAV_CHARTS_ROOT),
        help="Local RNAV chart directory to publish into public/data/airports/<ICAO>/charts",
    )
    parser.add_argument(
        "--skip-procedure-details",
        action="store_true",
        help="Skip publishing browser-ready procedure detail JSON and chart manifests",
    )
    parser.add_argument("--output", default=None, help="Output GeoJSON path")
    args = parser.parse_args()

    index_path = Path(args.cifp_root) / "IN_CIFP.txt"
    if args.include_all_rnav:
        selected_procedures = discover_rnav_procedures(index_path, args.airport, args.procedure_type)
        if not selected_procedures:
            raise ValueError(
                f"No RNAV/RNP {args.procedure_type.upper()} procedures found for {args.airport.upper()} in {index_path}"
            )
    elif args.procedures:
        selected_procedures = parse_procedure_list(args.procedures)
    else:
        selected_procedures = [args.procedure.upper()]

    canonical_documents = build_procedure_detail_documents(
        cifp_root=Path(args.cifp_root),
        airport=args.airport,
        procedure_type=args.procedure_type,
        procedures=selected_procedures,
        nominal_speed_kt=args.nominal_speed_kt,
        tunnel_half_width_nm=args.tunnel_half_width_nm,
        tunnel_half_height_ft=args.tunnel_half_height_ft,
        sample_spacing_m=args.sample_spacing_m,
    )
    collection = procedure_detail_documents_to_geojson(
        airport=args.airport,
        procedure_type=args.procedure_type,
        source_cycle=parse_source_cycle(Path(args.cifp_root) / "FAACIFP18"),
        documents=canonical_documents,
        include_transitions=args.include_transitions,
        branch=args.branch,
    )

    output_path = Path(args.output) if args.output else airport_data_path(args.airport, "procedures.geojson")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(collection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    detail_publish_result = None
    if not args.skip_procedure_details:
        airport_details = load_airport_details(args.airport)
        procedure_dir = airport_procedure_details_dir(args.airport)
        procedure_dir.mkdir(parents=True, exist_ok=True)
        for document in canonical_documents:
            detail_path = airport_procedure_details_path(args.airport, f"{document['procedureUid']}.json")
            detail_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

        index_manifest = build_procedure_details_index(
            airport=args.airport,
            airport_name=airport_details["name"],
            source_cycle=parse_source_cycle(Path(args.cifp_root) / "FAACIFP18"),
            documents=canonical_documents,
        )
        index_path = airport_procedure_details_path(args.airport, "index.json")
        index_path.write_text(json.dumps(index_manifest, indent=2) + "\n", encoding="utf-8")

        chart_manifest = publish_local_chart_manifest(
            airport=args.airport,
            chart_root=Path(args.charts_root),
            documents=canonical_documents,
        )
        chart_index_path = airport_chart_path(args.airport, "index.json")
        chart_index_path.write_text(json.dumps(chart_manifest, indent=2) + "\n", encoding="utf-8")
        detail_publish_result = {
            "procedureIndexPath": index_path,
            "chartIndexPath": chart_index_path,
            "procedureCount": len(canonical_documents),
            "chartCount": len(chart_manifest["charts"]),
        }

    route_features = [
        feature
        for feature in collection["features"]
        if feature["properties"]["featureType"] == "procedure-route"
    ]
    point_count = len(collection["features"]) - len(route_features)
    warning_count = len(collection["metadata"]["warnings"])
    print("CIFP procedure preprocessing complete:")
    print(f"  Airport:           {args.airport.upper()}")
    print(f"  Procedures:        {', '.join(selected_procedures)}")
    print(f"  Branch mode:       {'all branches' if args.include_transitions else args.branch.upper()}")
    print(f"  Route features:    {len(route_features)}")
    print(f"  Fix points:        {point_count}")
    print(f"  Warnings:          {warning_count}")
    print(f"  Output:            {output_path}")
    if detail_publish_result is not None:
        print(f"  Detail index:      {detail_publish_result['procedureIndexPath']}")
        print(f"  Chart index:       {detail_publish_result['chartIndexPath']}")
        print(f"  Detail docs:       {detail_publish_result['procedureCount']}")
        print(f"  Published charts:  {detail_publish_result['chartCount']}")


if __name__ == "__main__":
    main()
