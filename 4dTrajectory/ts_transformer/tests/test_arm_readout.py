"""A prediction directory read as an arm: the endpoint against the assigned threshold (`inference/arm_readout.py`)."""

from __future__ import annotations

import math

import pytest

from geokit import METRES_PER_DEG_LAT
from ts_transformer.inference.arm_readout import endpoint_geometry, print_table


def _state(t: float, lat: float, lon: float, **extra: float) -> dict:
    return {"t": t, "lat": lat, "lon": lon, "alt": 300.0, **extra}


def test_the_endpoint_is_read_on_the_assigned_course_right_of_it_positive(capsys):
    # A northbound runway (math-ENU psi = 90°) at the origin; the prediction ends 100 m east of its threshold,
    # i.e. 100 m RIGHT of the inbound course, and 50 m before it.
    target = {"lat": 0.0, "lon": 0.0, "psi": math.pi / 2, "alt": 0.0}
    anchor = {"lat": -0.1, "lon": 0.0, "V": 70.0, "gamma": 0.0, "psi": math.pi / 2}
    end_lat = -50.0 / METRES_PER_DEG_LAT
    end_lon = 100.0 / METRES_PER_DEG_LAT
    states = {"predicted_states": [_state(0.0, -0.1, 0.0), _state(2.0, -0.1 + 140.0 / METRES_PER_DEG_LAT, 0.0),
                                   _state(60.0, end_lat, end_lon)]}
    geometry = endpoint_geometry({"target_state": target, "initial_state": anchor}, states,
                                 {"arr_airport": "KXXX", "runway": "36"})
    assert geometry["endpoint_cross_track_m"] == pytest.approx(100.0, abs=0.5)
    assert geometry["endpoint_along_track_m"] == pytest.approx(-50.0, abs=0.5)
    assert geometry["endpoint_height_m"] == pytest.approx(300.0)
    assert geometry["first_step_offset_m"] == pytest.approx(0.0, abs=0.5)     # it flew the anchor's own velocity
    assert geometry["closer_to_sibling"] is None                              # no parallel runway listed
    print_table("t", ["arm", "ADE"], [["A", "1"]])
    assert capsys.readouterr().out == "\n### t\n\n| arm | ADE |\n|---|---:|\n| A | 1 |\n"
