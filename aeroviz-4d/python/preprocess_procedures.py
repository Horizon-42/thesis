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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_layout import airport_data_path

FEET_TO_METRES = 0.3048
NM_TO_METRES = 1852.0
DEFAULT_CIFP_ROOT = Path(__file__).parents[2] / "data" / "CIFP" / "CIFP_260319"
DEFAULT_AIRPORT = "KRDU"
DEFAULT_RNAV_PROCEDURES = ["R05LY", "R05RY", "R23LY", "R23RY", "R32"]
RUNWAY_ORDER = ["RW05L", "RW05R", "RW23L", "RW23R", "RW32"]
SUPPORTED_ROUTE_LEGS = {"IF", "TF"}
COORD_PAIR_RE = re.compile(r"([NS]\d{8,10})([EW]\d{9,11})")
LEG_TYPE_RE = re.compile(r"(?<![A-Z])(IF|TF|DF|CA|CF|HM|HF|RF|VI|VA|FA)(?![A-Z])")


@dataclass(frozen=True)
class FixRecord:
    ident: str
    lon: float
    lat: float
    altitude_ft: int | None
    source_line: int


@dataclass(frozen=True)
class ProcedureLeg:
    sequence: int
    branch: str
    fix_ident: str
    leg_type: str
    role: str
    altitude_ft: int | None
    source_line: int


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
    match = re.search(r"([+\-V]?)\s*(\d{5})", text)
    if not match:
        return None
    altitude = int(match.group(2))
    if match.group(1) == "-":
        altitude *= -1
    return altitude


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


def parse_fix_altitude(line: str, coord_end: int) -> int | None:
    """Parse runway/fix elevation when it is encoded immediately after coordinates."""
    after_coords = line[coord_end : min(coord_end + 25, len(line))]
    return parse_signed_altitude_ft(after_coords)


def build_fix_index(faacifp_path: Path, airport: str) -> dict[str, FixRecord]:
    """Collect local CIFP coordinate records for one airport."""
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
            altitude_ft = parse_signed_altitude_ft(line[70:90])
            legs.append(
                ProcedureLeg(
                    sequence=sequence,
                    branch=target_branch,
                    fix_ident=fix_ident,
                    leg_type=leg_type,
                    role=role,
                    altitude_ft=altitude_ft,
                    source_line=line_number,
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
    """High-level extraction entrypoint used by the CLI and tests."""
    index_path = cifp_root / "IN_CIFP.txt"
    faacifp_path = cifp_root / "FAACIFP18"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing CIFP index: {index_path}")
    if not faacifp_path.exists():
        raise FileNotFoundError(f"Missing CIFP detail file: {faacifp_path}")
    if not procedure_exists(index_path, airport, procedure_type, procedure):
        raise ValueError(f"{airport} {procedure_type} {procedure} not found in {index_path}")

    fixes = build_fix_index(faacifp_path, airport)
    legs = parse_procedure_legs(faacifp_path, airport, procedure, branch)
    if not legs:
        raise ValueError(f"No CIFP procedure legs found for {airport} {procedure} branch {branch}")

    route_points, warnings = build_route_points(legs, fixes, nominal_speed_kt)
    source_cycle = parse_source_cycle(faacifp_path)
    return build_procedure_geojson(
        airport=airport,
        procedure_type=procedure_type,
        procedure=procedure,
        branch=branch,
        source_cycle=source_cycle,
        legs=legs,
        route_points=route_points,
        warnings=warnings,
        nominal_speed_kt=nominal_speed_kt,
        tunnel_half_width_nm=tunnel_half_width_nm,
        tunnel_half_height_ft=tunnel_half_height_ft,
        sample_spacing_m=sample_spacing_m,
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
    """Generate a merged collection for multiple procedures and branches."""
    faacifp_path = cifp_root / "FAACIFP18"
    source_cycle = parse_source_cycle(faacifp_path)
    collections: list[dict[str, Any]] = []

    for procedure in sorted({item.upper() for item in procedures}, key=procedure_sort_key):
        available_branches = parse_available_branches(faacifp_path, airport, procedure)
        if include_transitions:
            branches = available_branches
        else:
            requested_branch = branch.upper() if branch else final_branch_for_procedure(procedure)
            branches = [requested_branch]

        for branch_ident in branches:
            collection = generate_procedure_geojson(
                cifp_root=cifp_root,
                airport=airport,
                procedure_type=procedure_type,
                procedure=procedure,
                branch=branch_ident,
                nominal_speed_kt=nominal_speed_kt,
                tunnel_half_width_nm=tunnel_half_width_nm,
                tunnel_half_height_ft=tunnel_half_height_ft,
                sample_spacing_m=sample_spacing_m,
            )
            route_features = [
                feature
                for feature in collection["features"]
                if feature["properties"]["featureType"] == "procedure-route"
            ]
            has_valid_route = any(
                len(feature["geometry"]["coordinates"]) >= 2 for feature in route_features
            )
            if not has_valid_route:
                collection["features"] = []
                collection["metadata"]["warnings"].append(
                    f"Skipped branch {branch_ident}: fewer than two supported route points"
                )
            collections.append(collection)

    return merge_feature_collections(
        airport=airport,
        procedure_type=procedure_type,
        source_cycle=source_cycle,
        collections=collections,
    )


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
        help="Generate the default KRDU RNAV/GPS set: R05LY,R05RY,R23LY,R23RY,R32",
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
    parser.add_argument("--output", default=None, help="Output GeoJSON path")
    args = parser.parse_args()

    if args.include_all_rnav:
        selected_procedures = DEFAULT_RNAV_PROCEDURES
    elif args.procedures:
        selected_procedures = parse_procedure_list(args.procedures)
    else:
        selected_procedures = [args.procedure.upper()]

    collection = generate_procedures_geojson(
        cifp_root=Path(args.cifp_root),
        airport=args.airport,
        procedure_type=args.procedure_type,
        procedures=selected_procedures,
        include_transitions=args.include_transitions,
        branch=args.branch,
        nominal_speed_kt=args.nominal_speed_kt,
        tunnel_half_width_nm=args.tunnel_half_width_nm,
        tunnel_half_height_ft=args.tunnel_half_height_ft,
        sample_spacing_m=args.sample_spacing_m,
    )

    output_path = Path(args.output) if args.output else airport_data_path(args.airport, "procedures.geojson")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(collection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

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


if __name__ == "__main__":
    main()
