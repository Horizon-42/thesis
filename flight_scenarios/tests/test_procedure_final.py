"""FAF read from the RNAV(GPS) procedure document (the cross-check against the backend parser lives in aeroviz_backend/tests/test_procedure_segments.py)."""

import json
from pathlib import Path

import pytest

from flight_scenarios.procedure_final import (
    FinalApproachFix,
    final_approach_fix,
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
