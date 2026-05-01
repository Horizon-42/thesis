"""
cifp_parser.py
==============
Small FAA CIFP / ARINC 424 subset parser used by AeroViz procedure exports.

The module intentionally owns only source parsing concerns:
- IN_CIFP procedure discovery
- FAACIFP18 terminal procedure legs
- FAACIFP18 airport-local and global fix coordinate records
- compact ARINC coordinate and altitude field decoding

Higher-level visualization/export logic belongs in preprocess_procedures.py.
"""

from __future__ import annotations

from functools import lru_cache
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COORD_PAIR_RE = re.compile(r"([NS]\d{8,10})([EW]\d{9,11})")
LEG_TYPE_RE = re.compile(r"(?<![A-Z])(IF|TF|DF|CA|CF|HM|HF|RF|VI|VA|FA)(?![A-Z])")


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
    procedure_type: str | None = None
    transition_ident: str | None = None
    turn_direction: str | None = None
    arc_radius_nm: float | None = None
    center_fix_ident: str | None = None
    center_fix_region_code: str | None = None
    center_lat_deg: float | None = None
    center_lon_deg: float | None = None


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


def normalize_turn_direction(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT"}:
        return "LEFT"
    if text in {"R", "RIGHT"}:
        return "RIGHT"
    return None


def parse_rf_arc_radius_nm(line: str) -> float | None:
    text = line[56:61].strip()
    if len(text) != 5 or not text.isdigit():
        return None
    return int(text) / 100.0


def parse_rf_center_fix_ident(line: str) -> str | None:
    text = line[106:111].strip().upper()
    return text or None


def parse_rf_center_fix_region(line: str) -> str | None:
    text = line[112:114].strip().upper()
    return text or None


def rf_related_fix_idents(legs: list[ProcedureLeg]) -> set[str]:
    return {
        leg.center_fix_ident
        for leg in legs
        if leg.leg_type == "RF" and leg.center_fix_ident
    }


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


def procedure_sort_key(procedure: str) -> tuple[int, int, str]:
    """Stable runway-oriented sorting for procedure idents."""
    runway_ident = runway_from_procedure_ident(procedure)
    if runway_ident is None:
        return (99, 99, procedure)

    match = re.match(r"RW(\d{2})([LRC]?)", runway_ident)
    if not match:
        return (99, 99, procedure)

    runway_number = int(match.group(1))
    side_order = {"L": 0, "": 1, "C": 2, "R": 3}
    side = match.group(2)
    return (runway_number, side_order.get(side, 9), procedure)


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
    if len(line) > 71 and line[12] == "G":
        threshold_elevation = line[66:71].strip()
        if threshold_elevation.isdigit():
            return int(threshold_elevation)

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
        if leg.fix_ident and leg.fix_region_code is not None:
            regions.setdefault(leg.fix_ident, set()).add(leg.fix_region_code)
        if leg.center_fix_ident and leg.center_fix_region_code is not None:
            regions.setdefault(leg.center_fix_ident, set()).add(leg.center_fix_region_code)
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
    required_idents.update(rf_related_fix_idents(procedure_legs))
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
            is_rf = leg_type == "RF"
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
                    procedure_type=target_branch[0] if len(target_branch) > 5 else target_branch,
                    transition_ident=target_branch[1:] if len(target_branch) > 5 else None,
                    turn_direction=normalize_turn_direction(line[43]) if is_rf else None,
                    arc_radius_nm=parse_rf_arc_radius_nm(line) if is_rf else None,
                    center_fix_ident=parse_rf_center_fix_ident(line) if is_rf else None,
                    center_fix_region_code=parse_rf_center_fix_region(line) if is_rf else None,
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


# Keep the original fixed-width parser available for validation and fallback.
local_procedure_exists = procedure_exists
local_discover_rnav_procedures = discover_rnav_procedures
local_build_airport_fix_index = build_airport_fix_index
local_build_fix_index = build_fix_index
local_parse_procedure_legs = parse_procedure_legs
local_parse_available_branches = parse_available_branches


@dataclass(frozen=True)
class CifparseData:
    procedure_records: tuple[dict[str, Any], ...]
    terminal_waypoints: tuple[dict[str, Any], ...]
    runways: tuple[dict[str, Any], ...]
    enroute_waypoints: tuple[dict[str, Any], ...]
    header_line_count: int
    airport_fix_source_lines: dict[tuple[str, str], int]
    enroute_fix_source_lines: dict[str, int]


def cifparse_source_line(record_number: Any, header_line_count: int) -> int:
    record_number_int = int(record_number or 0)
    return record_number_int + header_line_count if record_number_int else 0


def count_header_lines(faacifp_path: Path) -> int:
    count = 0
    with faacifp_path.open(encoding="ascii", errors="replace") as f:
        for line in f:
            if not line.startswith("HDR"):
                break
            count += 1
    return count


def build_fix_source_line_maps(faacifp_path: Path) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    airport_fix_source_lines: dict[tuple[str, str], int] = {}
    enroute_fix_source_lines: dict[str, int] = {}
    with faacifp_path.open(encoding="ascii", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n\r")
            if line.startswith("SUSAP ") and len(line) > 19 and line[12] in {"C", "G"}:
                airport = line[6:10].strip().upper()
                ident = parse_fix_ident(line)
                if airport and ident:
                    airport_fix_source_lines.setdefault((airport, ident), line_number)
            elif line.startswith("SUSAEA") and len(line) > 19:
                ident = parse_fix_ident(line)
                if ident:
                    enroute_fix_source_lines.setdefault(ident, line_number)
    return airport_fix_source_lines, enroute_fix_source_lines


@lru_cache(maxsize=4)
def load_cifparse_data(faacifp_path_text: str) -> CifparseData:
    """Parse FAACIFP18 with cifparse once per path and cache normalized dicts."""
    faacifp_path = Path(faacifp_path_text)
    try:
        from cifparse import CIFP
    except ImportError as error:
        raise RuntimeError(
            "cifparse is required for production CIFP parsing. Install requirements.txt "
            "or use local_* parser functions for fixed-width fallback."
        ) from error

    parser = CIFP(str(faacifp_path))
    parser.parse_procedures()
    parser.parse_terminal_waypoints()
    parser.parse_runways()
    parser.parse_enroute_waypoints()
    airport_fix_source_lines, enroute_fix_source_lines = build_fix_source_line_maps(faacifp_path)
    return CifparseData(
        procedure_records=tuple(item.to_dict().get("primary", {}) for item in parser.get_procedures()),
        terminal_waypoints=tuple(item.to_dict().get("primary", {}) for item in parser.get_terminal_waypoints()),
        runways=tuple(item.to_dict().get("primary", {}) for item in parser.get_runways()),
        enroute_waypoints=tuple(item.to_dict().get("primary", {}) for item in parser.get_enroute_waypoints()),
        header_line_count=count_header_lines(faacifp_path),
        airport_fix_source_lines=airport_fix_source_lines,
        enroute_fix_source_lines=enroute_fix_source_lines,
    )


def cifparse_data(faacifp_path: Path) -> CifparseData:
    return load_cifparse_data(str(faacifp_path.resolve()))


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


def normalize_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def cifparse_branch(primary: dict[str, Any], procedure: str) -> str:
    """Return AeroViz's stable internal branch key."""
    transition_id = str(primary.get("transition_id") or "").strip().upper()
    if not transition_id:
        return final_branch_for_procedure(procedure)
    route_type = str(primary.get("procedure_type") or "").strip().upper()
    return f"{route_type}{transition_id}" if route_type else transition_id


def cifparse_transition_ident(primary: dict[str, Any]) -> str | None:
    transition_id = str(primary.get("transition_id") or "").strip().upper()
    return transition_id or None


def role_from_cifparse(primary: dict[str, Any], leg_type: str, fix_ident: str, sequence: int) -> str:
    desc_code = str(primary.get("desc_code") or "").upper()
    if fix_ident.startswith("RW"):
        return "MAPt"
    if leg_type in {"HM", "HF"}:
        return "MAHF"
    if "F" in desc_code:
        return "FAF"
    if "I" in desc_code or leg_type == "IF":
        return "IF"
    if sequence <= 20:
        return "IF"
    return "Route"


def procedure_exists(index_path: Path, airport: str, procedure_type: str, procedure: str) -> bool:
    """Return true when the CIFP index lists the requested airport/procedure."""
    return local_procedure_exists(index_path, airport, procedure_type, procedure)


def discover_rnav_procedures(index_path: Path, airport: str, procedure_type: str) -> list[str]:
    """Return RNAV/RNP approach idents listed for one airport in IN_CIFP.txt."""
    return local_discover_rnav_procedures(index_path, airport, procedure_type)


def parse_available_branches(
    faacifp_path: Path,
    airport: str,
    procedure: str,
) -> list[str]:
    """Return available branch idents for one procedure from cifparse records."""
    data = cifparse_data(faacifp_path)
    airport = airport.upper()
    procedure = procedure.upper()
    branches: list[str] = []
    seen: set[str] = set()
    for primary in data.procedure_records:
        if primary.get("fac_id") != airport or primary.get("procedure_id") != procedure:
            continue
        branch = cifparse_branch(primary, procedure)
        if branch and branch not in seen:
            branches.append(branch)
            seen.add(branch)
    return sorted(branches, key=lambda item: branch_sort_key(procedure, item))


def parse_procedure_legs(
    faacifp_path: Path,
    airport: str,
    procedure: str,
    branch: str,
) -> list[ProcedureLeg]:
    """Read matching final/intermediate approach records from cifparse objects."""
    data = cifparse_data(faacifp_path)
    airport = airport.upper()
    procedure = procedure.upper()
    branch = branch.upper()
    legs: list[ProcedureLeg] = []
    for primary in data.procedure_records:
        if primary.get("fac_id") != airport or primary.get("procedure_id") != procedure:
            continue
        if cifparse_branch(primary, procedure) != branch:
            continue

        sequence = normalize_int(primary.get("seq_no"))
        fix_ident = str(primary.get("fix_id") or "").strip().upper()
        leg_type = str(primary.get("path_term") or "").strip().upper()
        if sequence is None or not leg_type:
            continue

        altitude_ft = normalize_int(primary.get("alt_1"))
        if altitude_ft is None:
            altitude_ft = normalize_int(primary.get("alt_2"))
        if altitude_ft is None:
            altitude_ft = normalize_int(primary.get("trans_alt"))

        legs.append(
            ProcedureLeg(
                sequence=sequence,
                branch=branch,
                fix_ident=fix_ident,
                leg_type=leg_type,
                role=role_from_cifparse(primary, leg_type, fix_ident, sequence),
                altitude_ft=altitude_ft,
                source_line=cifparse_source_line(primary.get("record_number"), data.header_line_count),
                fix_region_code=str(primary.get("fix_region") or "").strip().upper() or None,
                procedure_type=str(primary.get("procedure_type") or "").strip().upper() or None,
                transition_ident=cifparse_transition_ident(primary),
                turn_direction=normalize_turn_direction(primary.get("turn_direction")),
                arc_radius_nm=normalize_float(primary.get("arc_radius")),
                center_fix_ident=str(primary.get("center_fix") or "").strip().upper() or None,
                center_fix_region_code=str(primary.get("center_fix_region") or "").strip().upper() or None,
            )
        )

    return sorted(legs, key=lambda item: (item.sequence, item.source_line))


def build_airport_fix_index(faacifp_path: Path, airport: str) -> dict[str, FixRecord]:
    """Collect airport-local fixes from cifparse terminal waypoints and runways."""
    data = cifparse_data(faacifp_path)
    airport = airport.upper()
    fixes: dict[str, FixRecord] = {}

    for primary in data.terminal_waypoints:
        if primary.get("environment_id") != airport:
            continue
        ident = str(primary.get("waypoint_id") or "").strip().upper()
        if not ident:
            continue
        fixes[ident] = FixRecord(
            ident=ident,
            lon=float(primary["lon"]),
            lat=float(primary["lat"]),
            altitude_ft=None,
            source_line=data.airport_fix_source_lines.get(
                (airport, ident),
                cifparse_source_line(primary.get("record_number"), data.header_line_count),
            ),
            region_code=str(primary.get("waypoint_region") or "").strip().upper() or None,
            source_kind="airport-local-cifp",
        )

    for primary in data.runways:
        if primary.get("airport_id") != airport:
            continue
        ident = str(primary.get("runway_id") or "").strip().upper()
        if not ident:
            continue
        fixes[ident] = FixRecord(
            ident=ident,
            lon=float(primary["lon"]),
            lat=float(primary["lat"]),
            altitude_ft=normalize_int(primary.get("threshold_elevation")),
            source_line=data.airport_fix_source_lines.get(
                (airport, ident),
                cifparse_source_line(primary.get("record_number"), data.header_line_count),
            ),
            region_code=str(primary.get("airport_region") or "").strip().upper() or None,
            source_kind="airport-local-cifp",
        )

    return fixes


def build_fix_index(
    faacifp_path: Path,
    airport: str,
    procedure_legs: list[ProcedureLeg] | None = None,
) -> dict[str, FixRecord]:
    """Collect airport-local fixes and fill missing procedure fixes from cifparse enroute records."""
    data = cifparse_data(faacifp_path)
    fixes = build_airport_fix_index(faacifp_path, airport)
    if procedure_legs is None:
        return fixes

    required_idents = {leg.fix_ident for leg in procedure_legs if leg.fix_ident}
    required_idents.update(rf_related_fix_idents(procedure_legs))
    missing_idents = required_idents - set(fixes)
    preferred_regions = preferred_region_codes_by_fix(procedure_legs)
    for primary in data.enroute_waypoints:
        ident = str(primary.get("waypoint_id") or "").strip().upper()
        if ident not in missing_idents or ident in fixes:
            continue
        region_code = str(primary.get("waypoint_region") or "").strip().upper() or None
        expected_regions = preferred_regions.get(ident, set())
        if expected_regions and region_code not in expected_regions:
            continue
        fixes[ident] = FixRecord(
            ident=ident,
            lon=float(primary["lon"]),
            lat=float(primary["lat"]),
            altitude_ft=None,
            source_line=data.enroute_fix_source_lines.get(
                ident,
                cifparse_source_line(primary.get("record_number"), data.header_line_count),
            ),
            region_code=region_code,
            source_kind="global-cifp-fallback",
        )

    return fixes
