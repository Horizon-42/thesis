from __future__ import annotations

from pathlib import Path

import pytest

from preprocess_procedures import (
    COORD_PAIR_RE,
    build_airport_fix_index,
    build_fix_index,
    decode_coordinate_pair,
    parse_procedure_legs,
)


CIFP_ROOT = Path(__file__).parents[3] / "data" / "CIFP" / "CIFP_260319"
FAACIFP_PATH = CIFP_ROOT / "FAACIFP18"


def test_airport_local_fix_records_are_parsed_from_faacifp18() -> None:
    """Airport-local fixes are SUSAP <airport> records in the FAA CIFP detail file."""
    lines = FAACIFP_PATH.read_text(encoding="ascii", errors="replace").splitlines()
    fixes = build_airport_fix_index(FAACIFP_PATH, "KRDU")

    assert fixes
    for ident, fix in fixes.items():
        source_line = lines[fix.source_line - 1]
        match = COORD_PAIR_RE.search(source_line)

        assert source_line.startswith("SUSAP KRDU")
        assert source_line[13:19].strip().upper() == ident
        assert match is not None
        lon, lat = decode_coordinate_pair(match.group(1), match.group(2))
        assert fix.lon == pytest.approx(lon)
        assert fix.lat == pytest.approx(lat)
        assert fix.source_kind == "airport-local-cifp"


def test_feeder_initial_fixes_have_no_transition_altitude_as_crossing() -> None:
    """Production (cifparse) path: KRDU R32 feeder IFs CONCA/SINNO carry no
    Altitude 1/2 constraint, so they must parse to ``altitude_ft is None``
    -- the 18000 ft Transition Altitude must NOT leak in as a crossing
    altitude.  NOSIC has a real "at or above 3400" constraint.
    Regression for docs/33-cifp-transition-altitude-misparse-postmortem.md.
    """
    by_branch_fix = {}
    for branch in ("R", "ACONCA", "ASINNO"):
        for leg in parse_procedure_legs(FAACIFP_PATH, "KRDU", "R32", branch):
            if leg.leg_type == "IF":
                by_branch_fix[(branch, leg.fix_ident)] = leg

    conca = by_branch_fix[("ACONCA", "CONCA")]
    sinno = by_branch_fix[("ASINNO", "SINNO")]
    nosic = by_branch_fix[("R", "NOSIC")]

    assert conca.altitude_ft is None
    assert sinno.altitude_ft is None
    assert nosic.altitude_ft == 3400
    assert nosic.altitude_qualifier == "atOrAbove"


def test_r32_missing_duham_is_resolved_from_matching_global_cifp_region() -> None:
    legs = parse_procedure_legs(FAACIFP_PATH, "KRDU", "R32", "R")
    fixes = build_fix_index(FAACIFP_PATH, "KRDU", procedure_legs=legs)

    duham = fixes["DUHAM"]

    assert duham.source_kind == "global-cifp-fallback"
    assert duham.region_code == "K7"
    assert duham.source_line == 33499
    assert duham.lat == pytest.approx(36.0516556)
    assert duham.lon == pytest.approx(-78.8345139)
