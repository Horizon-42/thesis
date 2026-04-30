"""
Cross-validate AeroViz CIFP parsing against third-party parsers.

This is an evaluation tool, not part of the production data export path. It
compares the extracted local parser with:
- arinc424: low-level ARINC 424 record decoder
- cifparse: FAA CIFP structured parser

Use it to decide whether a third-party package is complete enough to replace
the local fixed-width parser for procedure-details generation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cifp_parser import (
    FixRecord,
    ProcedureLeg,
    decode_cifp_coordinate,
    discover_rnav_procedures,
    final_branch_for_procedure,
    local_build_fix_index,
    local_parse_available_branches,
    local_parse_procedure_legs,
)

DEFAULT_CIFP_ROOT = Path(__file__).parents[2] / "data" / "CIFP" / "CIFP_260319"


@dataclass(frozen=True)
class LegSnapshot:
    procedure: str
    branch: str
    sequence: int
    fix_ident: str
    leg_type: str
    altitude_ft: int | None
    fix_region_code: str | None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.procedure, self.branch, self.sequence)


@dataclass(frozen=True)
class FixSnapshot:
    ident: str
    lon: float
    lat: float
    altitude_ft: int | None
    region_code: str | None


@dataclass(frozen=True)
class ParserSnapshot:
    name: str
    available: bool
    error: str | None
    legs: dict[tuple[str, str, int], LegSnapshot]
    fixes: dict[str, FixSnapshot]


def normalize_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.startswith(("+", "-")) and text[1:].isdigit():
        return int(text)
    if text.isdigit():
        return int(text)
    return None


def leg_from_local(procedure: str, leg: ProcedureLeg) -> LegSnapshot:
    return LegSnapshot(
        procedure=procedure,
        branch=normalize_branch(procedure, leg.branch),
        sequence=leg.sequence,
        fix_ident=leg.fix_ident,
        leg_type=leg.leg_type,
        altitude_ft=leg.altitude_ft,
        fix_region_code=leg.fix_region_code,
    )


def fix_from_local(record: FixRecord) -> FixSnapshot:
    return FixSnapshot(
        ident=record.ident,
        lon=record.lon,
        lat=record.lat,
        altitude_ft=record.altitude_ft,
        region_code=record.region_code,
    )


def collect_local_snapshot(faacifp_path: Path, airport: str, procedures: list[str]) -> ParserSnapshot:
    legs: dict[tuple[str, str, int], LegSnapshot] = {}
    raw_legs: list[ProcedureLeg] = []
    for procedure in procedures:
        for branch in local_parse_available_branches(faacifp_path, airport, procedure):
            for leg in local_parse_procedure_legs(faacifp_path, airport, procedure, branch):
                snapshot = leg_from_local(procedure, leg)
                legs[snapshot.key] = snapshot
                raw_legs.append(leg)

    required_fix_idents = {leg.fix_ident for leg in raw_legs if leg.fix_ident}
    all_fixes = local_build_fix_index(faacifp_path, airport, procedure_legs=raw_legs)
    fixes = {
        ident: fix_from_local(all_fixes[ident])
        for ident in sorted(required_fix_idents)
        if ident in all_fixes
    }
    return ParserSnapshot("local_fixed_width", True, None, legs, fixes)


def first_int_field(fields: dict[str, Any], names: list[str]) -> int | None:
    for name in names:
        value = normalize_int(fields.get(name))
        if value is not None:
            return value
    return None


def normalize_branch(procedure: str, branch: Any) -> str:
    normalized = str(branch or "").strip().upper()
    if not normalized:
        return final_branch_for_procedure(procedure)
    # FAA PF records combine route type + transition id in fixed-width cols
    # 19:26 (for example ACHWDR). arinc424/cifparse expose only CHWDR.
    if len(normalized) > 5 and normalized[0] in {"A", "S", "D"}:
        return normalized[1:]
    return normalized


def arinc_json(record: Any) -> dict[str, Any]:
    # arinc424.Record.json() prints by default; suppress that side effect.
    with contextlib.redirect_stdout(io.StringIO()):
        raw_json = record.json()
    return json.loads(raw_json)


def collect_arinc424_snapshot(faacifp_path: Path, airport: str, procedures: list[str]) -> ParserSnapshot:
    try:
        import arinc424
    except ImportError as error:
        return ParserSnapshot("arinc424", False, str(error), {}, {})

    procedure_set = {procedure.upper() for procedure in procedures}
    airport = airport.upper()
    legs: dict[tuple[str, str, int], LegSnapshot] = {}
    fixes: dict[str, FixSnapshot] = {}

    for raw_line in faacifp_path.read_text(encoding="ascii", errors="replace").splitlines():
        record = arinc424.Record()
        if not record.read(raw_line):
            continue
        fields = arinc_json(record)

        procedure = str(fields.get("SID/STAR/Approach Identifier", "")).strip().upper()
        if (
            fields.get("Airport Identifier", fields.get("Airport ICAO Identifier")) == airport
            and procedure in procedure_set
            and "Sequence Number" in fields
        ):
            sequence = normalize_int(fields.get("Sequence Number"))
            if sequence is not None:
                snapshot = LegSnapshot(
                    procedure=procedure,
                    branch=normalize_branch(procedure, fields.get("Transition Identifier")),
                    sequence=sequence,
                    fix_ident=str(fields.get("Fix Identifier", "")).strip().upper(),
                    leg_type=str(fields.get("Path and Termination", "")).strip().upper(),
                    altitude_ft=first_int_field(fields, ["Altitude", "Altitude (2)", "Transition Altitude"]),
                    fix_region_code=str(fields.get("ICAO Code (2)", "")).strip().upper() or None,
                )
                legs[snapshot.key] = snapshot

        waypoint_id = str(fields.get("Waypoint Identifier", "")).strip().upper()
        if waypoint_id:
            lat_token = fields.get("Waypoint Latitude")
            lon_token = fields.get("Waypoint Longitude")
            if lat_token and lon_token:
                fixes.setdefault(
                    waypoint_id,
                    FixSnapshot(
                        ident=waypoint_id,
                        lon=decode_cifp_coordinate(str(lon_token)),
                        lat=decode_cifp_coordinate(str(lat_token)),
                        altitude_ft=None,
                        region_code=str(fields.get("ICAO Code (2)", "")).strip().upper() or None,
                    ),
                )

        runway_id = str(fields.get("Runway Identifier", "")).strip().upper()
        if fields.get("Airport ICAO Identifier") == airport and runway_id:
            lat_token = fields.get("Runway Latitude")
            lon_token = fields.get("Runway Longitude")
            if lat_token and lon_token:
                fixes[runway_id] = FixSnapshot(
                    ident=runway_id,
                    lon=decode_cifp_coordinate(str(lon_token)),
                    lat=decode_cifp_coordinate(str(lat_token)),
                    altitude_ft=normalize_int(fields.get("Landing Threshold Elevation")),
                    region_code=str(fields.get("ICAO Code", "")).strip().upper() or None,
                )

    return ParserSnapshot("arinc424", True, None, legs, fixes)


def collect_cifparse_snapshot(faacifp_path: Path, airport: str, procedures: list[str]) -> ParserSnapshot:
    try:
        from cifparse import CIFP
    except ImportError as error:
        return ParserSnapshot("cifparse", False, str(error), {}, {})

    procedure_set = {procedure.upper() for procedure in procedures}
    airport = airport.upper()
    parser = CIFP(str(faacifp_path))

    legs: dict[tuple[str, str, int], LegSnapshot] = {}
    parser.parse_procedures()
    for item in parser.get_procedures():
        primary = item.to_dict().get("primary", {})
        if primary.get("fac_id") != airport or primary.get("procedure_id") not in procedure_set:
            continue
        sequence = normalize_int(primary.get("seq_no"))
        if sequence is None:
            continue
        snapshot = LegSnapshot(
            procedure=primary["procedure_id"],
            branch=normalize_branch(primary["procedure_id"], primary.get("transition_id")),
            sequence=sequence,
            fix_ident=str(primary.get("fix_id") or "").strip().upper(),
            leg_type=str(primary.get("path_term") or "").strip().upper(),
            altitude_ft=first_int_field(primary, ["alt_1", "alt_2", "trans_alt"]),
            fix_region_code=str(primary.get("fix_region") or "").strip().upper() or None,
        )
        legs[snapshot.key] = snapshot

    fixes: dict[str, FixSnapshot] = {}
    parser.parse_terminal_waypoints()
    for item in parser.get_terminal_waypoints():
        primary = item.to_dict().get("primary", {})
        if primary.get("environment_id") != airport:
            continue
        ident = str(primary.get("waypoint_id") or "").strip().upper()
        if not ident:
            continue
        fixes[ident] = FixSnapshot(
            ident=ident,
            lon=float(primary["lon"]),
            lat=float(primary["lat"]),
            altitude_ft=None,
            region_code=str(primary.get("waypoint_region") or "").strip().upper() or None,
        )

    parser.parse_runways()
    for item in parser.get_runways():
        primary = item.to_dict().get("primary", {})
        if primary.get("airport_id") != airport:
            continue
        ident = str(primary.get("runway_id") or "").strip().upper()
        if not ident:
            continue
        fixes[ident] = FixSnapshot(
            ident=ident,
            lon=float(primary["lon"]),
            lat=float(primary["lat"]),
            altitude_ft=normalize_int(primary.get("threshold_elevation")),
            region_code=str(primary.get("airport_region") or "").strip().upper() or None,
        )

    parser.parse_enroute_waypoints()
    for item in parser.get_enroute_waypoints():
        primary = item.to_dict().get("primary", {})
        ident = str(primary.get("waypoint_id") or "").strip().upper()
        if not ident or ident in fixes:
            continue
        fixes[ident] = FixSnapshot(
            ident=ident,
            lon=float(primary["lon"]),
            lat=float(primary["lat"]),
            altitude_ft=None,
            region_code=str(primary.get("waypoint_region") or "").strip().upper() or None,
        )

    return ParserSnapshot("cifparse", True, None, legs, fixes)


def distance_m(a: FixSnapshot, b: FixSnapshot) -> float:
    radius_m = 6_371_008.8
    phi1 = math.radians(a.lat)
    phi2 = math.radians(b.lat)
    d_phi = math.radians(b.lat - a.lat)
    d_lambda = math.radians(b.lon - a.lon)
    hav = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return radius_m * 2.0 * math.atan2(math.sqrt(hav), math.sqrt(1.0 - hav))


def compare_snapshots(baseline: ParserSnapshot, candidate: ParserSnapshot) -> dict[str, Any]:
    if not candidate.available:
        return {
            "available": False,
            "error": candidate.error,
            "legCount": 0,
            "fixCount": 0,
            "missingLegs": len(baseline.legs),
            "extraLegs": 0,
            "legFieldMismatches": [],
            "missingFixes": len(baseline.fixes),
            "fixCoordinateMismatches": [],
            "maxFixCoordinateDeltaM": None,
        }

    missing_leg_keys = sorted(set(baseline.legs) - set(candidate.legs))
    extra_leg_keys = sorted(set(candidate.legs) - set(baseline.legs))
    leg_field_mismatches: list[dict[str, Any]] = []
    for key in sorted(set(baseline.legs) & set(candidate.legs)):
        expected = baseline.legs[key]
        actual = candidate.legs[key]
        mismatched_fields = [
            field
            for field in ["fix_ident", "leg_type", "altitude_ft", "fix_region_code"]
            if getattr(expected, field) != getattr(actual, field)
        ]
        if mismatched_fields:
            leg_field_mismatches.append(
                {
                    "key": key,
                    "fields": mismatched_fields,
                    "expected": asdict(expected),
                    "actual": asdict(actual),
                }
            )

    missing_fix_ids = sorted(set(baseline.fixes) - set(candidate.fixes))
    fix_coordinate_mismatches: list[dict[str, Any]] = []
    max_delta = 0.0
    for ident in sorted(set(baseline.fixes) & set(candidate.fixes)):
        expected = baseline.fixes[ident]
        actual = candidate.fixes[ident]
        delta = distance_m(expected, actual)
        max_delta = max(max_delta, delta)
        altitude_mismatch = expected.altitude_ft != actual.altitude_ft and expected.altitude_ft is not None
        if delta > 0.5 or altitude_mismatch:
            fix_coordinate_mismatches.append(
                {
                    "ident": ident,
                    "deltaM": round(delta, 3),
                    "expected": asdict(expected),
                    "actual": asdict(actual),
                }
            )

    return {
        "available": True,
        "error": None,
        "legCount": len(candidate.legs),
        "fixCount": len(candidate.fixes),
        "missingLegs": len(missing_leg_keys),
        "extraLegs": len(extra_leg_keys),
        "legFieldMismatches": leg_field_mismatches[:20],
        "missingFixes": len(missing_fix_ids),
        "missingFixExamples": missing_fix_ids[:20],
        "fixCoordinateMismatches": fix_coordinate_mismatches[:20],
        "maxFixCoordinateDeltaM": round(max_delta, 3),
    }


def recommend_parser(comparisons: dict[str, dict[str, Any]]) -> str:
    cifparse = comparisons.get("cifparse", {})
    arinc424 = comparisons.get("arinc424", {})
    if cifparse.get("available") and not any(
        cifparse.get(name)
        for name in ["missingLegs", "extraLegs", "legFieldMismatches", "missingFixes", "fixCoordinateMismatches"]
    ):
        return (
            "cifparse is the best candidate for a future primary parser: it provides structured "
            "procedure, runway, terminal-waypoint, and enroute-waypoint objects with complete agreement "
            "for this validation set. Keep arinc424 as a line-level decoder for audit checks."
        )
    if arinc424.get("available") and not any(
        arinc424.get(name)
        for name in ["missingLegs", "extraLegs", "legFieldMismatches", "missingFixes", "fixCoordinateMismatches"]
    ):
        return (
            "arinc424 matches the validation set, but it is a low-level record decoder rather than a "
            "CIFP collection model. It is better as a cross-check than as the primary AeroViz parser."
        )
    return (
        "Keep the extracted local parser as the production parser for now. Use this report to close "
        "the listed mismatches before switching to a third-party parser."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validate CIFP parser package behavior.")
    parser.add_argument("--cifp-root", type=Path, default=DEFAULT_CIFP_ROOT)
    parser.add_argument("--airport", default="KRDU")
    parser.add_argument("--procedure-type", default="SIAP")
    parser.add_argument(
        "--procedures",
        help="Comma-separated procedure idents. Default: all RNAV/RNP procedures for the airport.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    faacifp_path = args.cifp_root / "FAACIFP18"
    index_path = args.cifp_root / "IN_CIFP.txt"
    procedures = (
        [item.strip().upper() for item in args.procedures.split(",") if item.strip()]
        if args.procedures
        else discover_rnav_procedures(index_path, args.airport, args.procedure_type)
    )

    baseline = collect_local_snapshot(faacifp_path, args.airport, procedures)
    candidates = [
        collect_arinc424_snapshot(faacifp_path, args.airport, procedures),
        collect_cifparse_snapshot(faacifp_path, args.airport, procedures),
    ]
    comparisons = {candidate.name: compare_snapshots(baseline, candidate) for candidate in candidates}
    report = {
        "airport": args.airport.upper(),
        "procedures": procedures,
        "baseline": {
            "name": baseline.name,
            "legCount": len(baseline.legs),
            "fixCount": len(baseline.fixes),
        },
        "comparisons": comparisons,
        "recommendation": recommend_parser(comparisons),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"CIFP parser validation for {report['airport']}")
    print(f"Procedures: {', '.join(procedures)}")
    print(f"Baseline: {len(baseline.legs)} legs, {len(baseline.fixes)} fixes")
    for name, comparison in comparisons.items():
        if not comparison["available"]:
            print(f"- {name}: unavailable ({comparison['error']})")
            continue
        print(
            f"- {name}: {comparison['legCount']} legs, {comparison['fixCount']} fixes, "
            f"missingLegs={comparison['missingLegs']}, extraLegs={comparison['extraLegs']}, "
            f"legMismatches={len(comparison['legFieldMismatches'])}, "
            f"missingFixes={comparison['missingFixes']}, "
            f"fixMismatches={len(comparison['fixCoordinateMismatches'])}, "
            f"maxFixDeltaM={comparison['maxFixCoordinateDeltaM']}"
        )
    print(f"Recommendation: {report['recommendation']}")


if __name__ == "__main__":
    main()
