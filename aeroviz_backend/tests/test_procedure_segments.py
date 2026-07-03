"""ProcedureConstraint -> constraints.SegmentSpec translation."""

import math

import numpy as np
import pytest

from aeroviz_backend import paths  # noqa: F401  (puts the optimization dir on sys.path)
from aeroviz_backend.procedure_constraint import ProcedureConstraint
from aeroviz_backend.procedure_segments import build_constraint_segments

import approach_constraints as ac
from geokit import DEG2RAD, FT_M, WGS84_A

TARGET_LAT, TARGET_LON, TARGET_ALT = 35.0, -78.0, 120.0


def _lat_n_metres_north(n_m: float) -> float:
    """Latitude that is ``n_m`` metres north of the target."""
    return TARGET_LAT + n_m / (DEG2RAD * WGS84_A)


def _payload():
    # IF (11 km north) -> FAF (5 km north) -> threshold (at target); cumulative dist from the IF.
    return {
        "procedureUid": "P", "airportIcao": "KRDU", "runwayIdent": "05L", "branchId": "B",
        "approachCourseDeg": 180.0, "nominalSpeedKt": 180.0,
        "glidepath": {"angleDeg": 3.0, "tchFt": 55.0},
        "waypoints": [
            {"ident": "SCHOO", "role": "IF", "legType": "IF",
             "latDeg": _lat_n_metres_north(11000.0), "lonDeg": TARGET_LON,
             "altitude": {"kind": "atOrAbove", "minFtMsl": 3000.0}, "distanceFromStartM": 0.0},
            {"ident": "WEPAS", "role": "FAF", "legType": "TF",
             "latDeg": _lat_n_metres_north(5000.0), "lonDeg": TARGET_LON,
             "altitude": {"kind": "at", "minFtMsl": 1253.0, "maxFtMsl": 1253.0},
             "distanceFromStartM": 6000.0},
            {"ident": "RW05L", "role": "MAPt", "legType": "TF",
             "latDeg": TARGET_LAT, "lonDeg": TARGET_LON,
             "altitude": None, "distanceFromStartM": 11000.0},
        ],
    }


def _build():
    pc = ProcedureConstraint.from_payload(_payload())
    return build_constraint_segments(pc, TARGET_LAT, TARGET_LON, TARGET_ALT)


def test_segment_kinds():
    segments = _build()
    assert [s.kind for s in segments] == [ac.SegmentKind.INTERMEDIATE, ac.SegmentKind.FINAL_LPV]


def test_intermediate_box_and_floor():
    segments = _build()
    inter = segments[0]
    assert inter.halfwidth_m == pytest.approx(1852.0)            # 1 x RNP
    assert inter.start_ident == "SCHOO" and inter.end_ident == "WEPAS"
    # floor = the FAF's at-or-above altitude (1253 ft -> metres), held over the leg's coded length
    assert len(inter.step_downs) == 1
    assert inter.step_downs[0].min_alt_m == pytest.approx(1253.0 * FT_M)
    assert inter.step_downs[0].s_from_start_m == pytest.approx(6000.0)
    assert inter.max_descent_deg == pytest.approx(3.0)


def test_final_lpv_geometry():
    segments = _build()
    final = segments[1]
    assert final.kind is ac.SegmentKind.FINAL_LPV
    lpv = final.lpv
    assert np.allclose(lpv.ltp_ne, [0.0, 0.0])
    # LPV course width = max(350 ft, tan(1.5deg)*dGARP); dGARP = (9023+1000) ft -> floor wins
    assert lpv.course_width_m == pytest.approx(350.0 * FT_M, rel=1e-3)
    assert lpv.gpa_deg == pytest.approx(3.0)
    # glidepath passes through the target altitude at the threshold: tdze + tch == target_alt
    assert lpv.tdze_m + lpv.tch_m == pytest.approx(TARGET_ALT)
    # GARP/FPAP are past the LTP on the inbound axis (north side here -> negative n)
    assert lpv.garp_ne[0] < 0.0 and lpv.fpap_ne[0] < 0.0


def test_threshold_waypoint_maps_to_origin():
    pc = ProcedureConstraint.from_payload(_payload())
    frame = ac.TargetFrame(TARGET_LAT, TARGET_LON)
    assert np.allclose(frame.to_ne(pc.waypoints[-1].lat_deg, pc.waypoints[-1].lon_deg), [0.0, 0.0])


def _lon_e_metres_east(e_m: float) -> float:
    return TARGET_LON + e_m / (DEG2RAD * WGS84_A * math.cos(math.radians(TARGET_LAT)))


def test_warns_on_excessive_pfaf_intercept():
    # IF offset far east so the IF->FAF track meets the (due-south) final course at ~45°.
    payload = _payload()
    payload["waypoints"][0]["lonDeg"] = _lon_e_metres_east(6000.0)
    pc = ProcedureConstraint.from_payload(payload)
    with pytest.warns(UserWarning, match="intercept angle"):
        build_constraint_segments(pc, TARGET_LAT, TARGET_LON, TARGET_ALT)
