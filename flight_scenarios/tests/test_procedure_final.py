"""FAF read from the RNAV(GPS) procedure document (the cross-check against the backend parser lives in aeroviz_backend/tests/test_procedure_segments.py)."""

import json
from pathlib import Path

import pytest

from flight_scenarios.procedure_final import (
    FinalApproachFix,
    final_approach_fix,
    procedure_skeleton,
    rnav_gps_procedure_path,
)


def _write_document(root: Path, airport: str, runway: str, *, samples) -> Path:
    details = root / airport / "procedure-details"
    details.mkdir(parents=True)
    uid = f"{airport}-R{runway}Y-RW{runway}"
    (details / "index.json").write_text(json.dumps({"runways": [{
        "runwayIdent": f"RW{runway}",
        "procedures": [
            {"procedureUid": f"{airport}-H{runway}Z-RW{runway}", "procedureFamily": "RNAV_RNP"},
            {"procedureUid": uid, "procedureFamily": "RNAV_GPS"},
        ],
    }]}))
    (details / f"{uid}.json").write_text(json.dumps({
        "procedureUid": uid,
        "verticalProfiles": [{
            "glidepathAngleDeg": 3.0, "thresholdCrossingHeightFt": 57.4,
            "constraintSamples": samples,
        }],
    }))
    return details / f"{uid}.json"


def test_reads_the_faf_distance_from_the_vertical_profile(tmp_path):
    path = _write_document(tmp_path, "KXYZ", "09", samples=[
        {"role": "IF", "ident": "IFFIX", "distanceFromStartM": 0.0, "altitudeFt": 3000},
        {"role": "FAF", "ident": "FAFIX", "distanceFromStartM": 5561.3, "altitudeFt": 2200},
        {"role": "MAPt", "ident": "RW09", "distanceFromStartM": 15899.6, "altitudeFt": 424},
    ])
    assert rnav_gps_procedure_path("kxyz", "09", root=tmp_path) == path
    fix = final_approach_fix("KXYZ", "09", root=tmp_path)
    assert isinstance(fix, FinalApproachFix)
    assert fix.ident == "FAFIX"
    assert fix.distance_to_threshold_m == pytest.approx(15899.6 - 5561.3)
    assert fix.crossing_altitude_m == pytest.approx(2200 * 0.3048)


def test_missing_procedure_or_faf_raises_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        rnav_gps_procedure_path("KXYZ", "09", root=tmp_path)
    _write_document(tmp_path, "KXYZ", "27", samples=[
        {"role": "MAPt", "ident": "RW27", "distanceFromStartM": 10_000.0, "altitudeFt": 400},
    ])
    with pytest.raises(ValueError, match="RNAV\\(GPS\\)"):
        rnav_gps_procedure_path("KXYZ", "09", root=tmp_path)
    with pytest.raises(ValueError, match="FAF"):
        final_approach_fix("KXYZ", "27", root=tmp_path)


def _write_skeleton_document(root: Path, airport: str, runway: str) -> Path:
    """A final IF → FAF → MAPt (+ a missed leg that must be ignored) and one transition."""
    details = root / airport / "procedure-details"
    details.mkdir(parents=True)
    uid = f"{airport}-R{runway}Y-RW{runway}"
    (details / "index.json").write_text(json.dumps({"runways": [{
        "runwayIdent": f"RW{runway}",
        "procedures": [{"procedureUid": uid, "procedureFamily": "RNAV_GPS"}],
    }]}))

    def leg(seq, start, end, role, qualifier, value, rendered=True, raw=None):
        return {
            "sequence": seq, "path": {"startFixRef": start, "endFixRef": end},
            "roleAtEnd": role, "quality": {"renderedInPlanView": rendered},
            "constraints": {
                "altitude": {"qualifier": qualifier, "valueFt": value, "rawText": raw or f"{value} ft"},
                "speedKt": None,
            },
        }

    (details / f"{uid}.json").write_text(json.dumps({
        "procedureUid": uid,
        "runway": {"ident": f"RW{runway}", "landingThresholdFixRef": "fix:RWY",
                   "threshold": {"lat": 35.0, "lon": -78.0, "elevationFt": 400}},
        "fixes": [
            {"fixId": "fix:IAF", "ident": "IAF", "position": {"lat": 35.2, "lon": -78.3}},
            {"fixId": "fix:IFX", "ident": "IFX", "position": {"lat": 35.1, "lon": -78.15}},
            {"fixId": "fix:FAF", "ident": "FAF", "position": {"lat": 35.05, "lon": -78.08}},
            {"fixId": "fix:RWY", "ident": "RWY", "position": {"lat": 35.0, "lon": -78.0}},
            {"fixId": "fix:MAH", "ident": "MAH", "position": {"lat": 34.9, "lon": -77.9}},
        ],
        "branches": [
            {"branchId": "branch:R", "branchKey": "R", "branchRole": "final",
             "transitionIdent": None, "mergeFixRef": None, "legs": [
                 leg(10, None, "fix:IFX", "IF", "atOrAbove", 3000),
                 leg(20, "fix:IFX", "fix:FAF", "FAF", "block", 2200, raw="2000-2200 ft"),
                 leg(30, "fix:FAF", "fix:RWY", "MAPt", "at", 424),
                 leg(40, "fix:RWY", "fix:MAH", "MAHF", "atOrAbove", 2200, rendered=False),
             ]},
            {"branchId": "branch:AIAF", "branchKey": "AIAF", "branchRole": "transition",
             "transitionIdent": "IAF", "mergeFixRef": "fix:IFX", "legs": [
                 leg(10, None, "fix:IAF", "IF", "atOrAbove", 5000),
                 leg(20, "fix:IAF", "fix:IFX", "Route", "atOrAbove", 3000),
             ]},
        ],
        "verticalProfiles": [{"glidepathAngleDeg": 3.0, "thresholdCrossingHeightFt": 57.4,
                              "constraintSamples": []}],
    }))
    return details / f"{uid}.json"


def test_reads_the_skeleton_up_to_the_threshold_with_its_transitions(tmp_path):
    _write_skeleton_document(tmp_path, "KXYZ", "09")
    skeleton = procedure_skeleton("KXYZ", "09", root=tmp_path)
    assert [fix.ident for fix in skeleton.final.fixes] == ["IFX", "FAF", "RWY"]  # the missed leg is not a landing
    assert skeleton.faf.ident == "FAF"
    assert skeleton.final.fixes[0].altitude_min_m == pytest.approx(3000 * 0.3048)
    assert skeleton.final.fixes[0].altitude_max_m is None
    assert skeleton.faf.altitude_min_m == pytest.approx(2000 * 0.3048)   # a block reads both bounds
    assert skeleton.faf.altitude_max_m == pytest.approx(2200 * 0.3048)
    assert skeleton.final.fixes[-1].altitude_min_m == skeleton.final.fixes[-1].altitude_max_m
    assert skeleton.glidepath_angle_deg == 3.0
    assert skeleton.threshold_crossing_height_m == pytest.approx(57.4 * 0.3048)
    assert skeleton.threshold_elevation_m == pytest.approx(400 * 0.3048)
    assert len(skeleton.transitions) == 1
    transition = skeleton.transitions[0]
    assert transition.transition_ident == "IAF" and transition.merge_fix == "IFX"
    assert [fix.ident for fix in transition.fixes] == ["IAF", "IFX"]
    assert all(fix.speed_max_mps is None for fix in transition.fixes)


def test_a_threshold_the_cifp_codes_without_an_elevation_reads_as_unknown_not_sea_level(tmp_path):
    import json

    path = _write_skeleton_document(tmp_path, "KXYZ", "09")
    document = json.loads(path.read_text())
    document["runway"]["threshold"]["elevationFt"] = None  # KMSY RW20's two procedures
    path.write_text(json.dumps(document))
    assert procedure_skeleton("KXYZ", "09", root=tmp_path).threshold_elevation_m is None


def test_the_final_stops_at_the_mapt_even_when_the_document_names_no_threshold_fix(tmp_path):
    path = _write_skeleton_document(tmp_path, "KXYZ", "09")
    document = json.loads(path.read_text())
    del document["runway"]["landingThresholdFixRef"]
    path.write_text(json.dumps(document))
    procedure_skeleton.cache_clear()
    skeleton = procedure_skeleton("KXYZ", "09", root=tmp_path)
    assert [fix.ident for fix in skeleton.final.fixes] == ["IFX", "FAF", "RWY"]


def test_a_transition_that_does_not_end_on_its_merge_fix_is_refused(tmp_path):
    path = _write_skeleton_document(tmp_path, "KXYZ", "09")
    document = json.loads(path.read_text())
    document["branches"][1]["mergeFixRef"] = "fix:FAF"
    path.write_text(json.dumps(document))
    procedure_skeleton.cache_clear()
    with pytest.raises(ValueError, match="merge"):
        procedure_skeleton("KXYZ", "09", root=tmp_path)


def test_an_unknown_altitude_qualifier_is_refused(tmp_path):
    path = _write_skeleton_document(tmp_path, "KXYZ", "09")
    document = json.loads(path.read_text())
    document["branches"][0]["legs"][0]["constraints"]["altitude"]["qualifier"] = "recommended"
    path.write_text(json.dumps(document))
    procedure_skeleton.cache_clear()
    with pytest.raises(ValueError, match="qualifier"):
        procedure_skeleton("KXYZ", "09", root=tmp_path)
