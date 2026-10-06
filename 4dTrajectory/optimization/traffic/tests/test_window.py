"""The window check and its rows on a synthetic final: one recorded aircraft 2 NM ahead on the same runway."""

import math

import casadi as ca
import numpy as np
import pytest
from geokit import METRES_PER_DEG_LAT, NM_M, metres_per_deg_lon

from aerodynamic_model.common import GeodeticState
from traffic import rules
from traffic.check import FlownTrack, Window, check, check_times
from traffic.frame import TargetFrame
from traffic.check import Conflict
from traffic.rows import (
    HORIZONTAL, LANDING_AFTER, VERTICAL, VERTICAL_M, Branch, RowSpec, branch_for, extra_rows, row_spec,
)
from traffic.runways import Runway, established, relative
from traffic.scene import RecordedFlight

LAT0, LON0, T0 = 35.0, -78.0, 1_800_000_000.0
SPEED = 70.0
TARGET = GeodeticState(LAT0, LON0, 120.0, SPEED, 0.0, math.radians(-3.0), 60_000.0)
RUNWAY = Runway("09", 0.0, 0.0, 90.0, 10_000.0)          # course 090, the threshold at the target, FAF 10 km


def _west(metres):
    """The longitude ``metres`` west of the threshold."""
    return LON0 - metres / metres_per_deg_lon(LAT0)


def _final(t, start_before_m, start_t=0.0):
    """Positions on the 09 final at ``SPEED``, ``start_before_m`` before the threshold at ``start_t``."""
    before = start_before_m - SPEED * (np.asarray(t, float) - start_t)
    return np.full_like(before, LAT0), _west(before), 120.0 + np.tan(math.radians(3.0)) * before


def _window(recorded, categories=(None,)):
    return Window(
        flight_key="OWN", t0_utc_s=T0, frame=TargetFrame(TARGET), runways={"09": RUNWAY},
        rules=rules.separation({"09": {"lat": LAT0, "lon": LON0, "course_deg": 90.0}}, SPEED),
        runway="09", category=None, recorded=tuple(recorded), categories=tuple(categories),
        uncategorised_types=(),
    )


def _flown(start_before_m):
    t = np.arange(0.0, start_before_m / SPEED + 1e-9, 0.5)
    lat, lon, alt = _final(t, start_before_m)
    return FlownTrack(t, lat, lon, alt, np.full_like(t, 90.0))


def _ahead(start_before_m):
    t = np.arange(0.0, start_before_m / SPEED + 1e-9, 2.0)
    lat, lon, alt = _final(t, start_before_m)
    return RecordedFlight("AHEAD", "09", None, T0 + t, lat, lon, alt)


def test_relative_and_established_on_the_final():
    before, right, off = relative(RUNWAY, 100.0, -1000.0, 75.0)      # 1 km before, 100 m left, 15° off
    assert float(before) == pytest.approx(1000.0) and float(right) == pytest.approx(-100.0)
    assert float(off) == pytest.approx(-15.0)
    assert established(RUNWAY, before, right, off)
    assert not established(RUNWAY, before, right, -25.0)              # 25° off the course
    assert not established(RUNWAY, 11_000.0, 0.0, 0.0)                # outside the FAF
    assert established(RUNWAY, -0.05, 0.0, 0.0)                        # over the threshold (a replay's end)
    assert not established(RUNWAY, -4000.0, 0.0, 0.0)                  # past the cone's apex at the GARP
    assert not established(Runway("09", 0, 0, 90.0, None), before, right, off)   # no coded FAF


def test_the_frame_is_the_targets_and_states_its_east_error():
    frame = TargetFrame(TARGET)
    n, e = frame.to_ne(LAT0 + 1000.0 / METRES_PER_DEG_LAT, LON0)
    assert float(e) == pytest.approx(0.0) and float(n) == pytest.approx(1000.0, rel=0.01)
    lat = LAT0 + 0.2
    assert frame.east_scale_error([lat]) == pytest.approx(abs(math.cos(math.radians(LAT0)) / math.cos(math.radians(lat)) - 1),
                                                          rel=0.01)


def test_check_finds_the_in_trail_loss_the_follower_answers_for():
    window = _window([_ahead(8000.0 - 2.0 * NM_M)])         # 2 NM ahead at t = 0, same speed
    found = check(window, _flown(8000.0), reading=rules.VISUAL, step_s=1.0)
    in_trail = [c for c in found.conflicts if c.kind == rules.IN_TRAIL]
    assert in_trail and all(c.responsible for c in in_trail)
    assert in_trail[0].required_m == pytest.approx(3.0 * NM_M)       # categories unknown: the radar minimum
    assert in_trail[0].distance_m == pytest.approx(2.0 * NM_M, rel=1e-3)
    assert found.background == 0
    # the leader's landing (its last sample) is a check instant (the wake rule's)
    assert window.recorded[0].end_utc_s - T0 in set(check_times(window, 8000.0 / SPEED, 1.0))


def test_a_horizontal_row_measures_the_distance_to_the_record_at_the_nodes_symbolic_time():
    window = _window([_ahead(8000.0 - 2.0 * NM_M)])
    spec = RowSpec(HORIZONTAL, 20.0, 0, 3.0 * NM_M)
    node_times = np.array([10.0, 20.0, 30.0, 200.0])                   # the last is outside the row window
    rows = extra_rows([spec], window, node_times, row_window_s=60.0)
    nodes = [ca.SX.sym(f"x{k}", 6) for k in range(4)]
    times = [ca.SX.sym(f"t{k}") for k in range(4)]
    out = rows(nodes, times)
    assert len(out) == 3 and all(lb == pytest.approx((3.0 * NM_M) ** 2) for _e, lb, _u in out)
    # node 1 at the follower's own position on the final, at t = 25 s: the gap is the 2 NM spacing
    n, e = window.frame.to_ne(*_final(25.0, 8000.0)[:2])
    gap2 = ca.Function("g", [nodes[1], times[1]], [out[1][0]])([float(n), float(e), 0, 0, 0, 0], 25.0)
    assert math.sqrt(float(gap2)) == pytest.approx(2.0 * NM_M, rel=1e-3)


def test_the_wake_at_the_threshold_asks_for_a_later_landing_on_the_approach_clock():
    leader = _ahead(8000.0 - 2.0 * NM_M)
    leader_landed = leader.end_utc_s - T0
    conflict = Conflict(leader_landed, 0, rules.AT_THRESHOLD, 5.0 * NM_M, 2.0 * NM_M, 0.0, True, True)
    spec = row_spec(conflict, _window([leader]), margin=0.01, branch=None)
    assert spec.family == LANDING_AFTER
    assert spec.bound == pytest.approx(leader_landed + 5.0 * NM_M * 1.01 / SPEED)
    # two parallels: the leader's threshold 1 NM further along its course lands it 1 NM "later" on the clock
    targets = {"09": {"lat": LAT0, "lon": LON0, "course_deg": 90.0},
               "09B": {"lat": LAT0 + 300.0 / METRES_PER_DEG_LAT, "lon": LON0 + NM_M / metres_per_deg_lon(LAT0),
                       "course_deg": 90.0}}
    window = _window([RecordedFlight("AHEAD", "09B", None, leader.t_utc_s, leader.lat_deg, leader.lon_deg,
                                     leader.alt_m)])
    window = Window(**{**window.__dict__, "rules": rules.separation(targets, SPEED)})
    a = window.rules.along_nm
    spec = row_spec(conflict, window, margin=0.01, branch=None)
    assert a["09B"] - a["09"] == pytest.approx(1.0, rel=0.01)
    assert spec.bound == pytest.approx(leader_landed + ((a["09"] - a["09B"]) * NM_M + 5.0 * NM_M * 1.01) / SPEED)


def test_one_branch_per_aircraft():
    either = dict(kind=rules.RADAR_OR_VERTICAL, required_m=5556.0, responsible=True)
    assert branch_for([Conflict(1.0, 0, distance_m=1000.0, vertical_m=200.0, above=False, **either)]) == Branch(VERTICAL, -1.0)
    assert branch_for([Conflict(1.0, 0, distance_m=5000.0, vertical_m=10.0, above=True, **either)]) == Branch(HORIZONTAL)
    in_trail = Conflict(2.0, 0, rules.IN_TRAIL, 5556.0, 1000.0, 200.0, True, True)
    assert branch_for([Conflict(1.0, 0, distance_m=1000.0, vertical_m=200.0, above=True, **either), in_trail]) == Branch(HORIZONTAL)
    with pytest.raises(ValueError):
        branch_for([Conflict(1.0, 0, rules.AT_THRESHOLD, 9000.0, 1000.0, 0.0, True, True)])


def test_a_vertical_row_and_the_presence_weight_after_a_landing():
    window = _window([_ahead(8000.0 - 2.0 * NM_M)])
    landed = window.recorded[0].end_utc_s - T0
    node_times = np.array([landed - 5.0, landed + 30.0])
    specs = [RowSpec(VERTICAL, landed - 5.0, 0, VERTICAL_M, -1.0), RowSpec(HORIZONTAL, landed + 20.0, 0, 3.0 * NM_M)]
    out = extra_rows(specs, window, node_times, row_window_s=10.0)([ca.SX.sym(f"x{k}", 6) for k in range(2)],
                                                                   [ca.SX.sym(f"t{k}") for k in range(2)])
    assert len(out) == 2
    # the vertical row at node 0, the leader in the air: sign·(h − h_leader); 300 m below it is +300
    _lat, _lon, h_leader = _final(landed - 5.0, 8000.0 - 2.0 * NM_M)
    nodes = [ca.SX.sym("a", 6), ca.SX.sym("b", 6)]
    times = [ca.SX.sym("ta"), ca.SX.sym("tb")]
    vertical, horizontal = extra_rows(specs, window, node_times, row_window_s=10.0)(nodes, times)
    v = ca.Function("v", [nodes[0], times[0]], [vertical[0]])([0, 0, float(h_leader) - 300.0, 0, 0, 0], landed - 5.0)
    assert float(v) == pytest.approx(300.0, abs=1.0) and vertical[1] == pytest.approx(VERTICAL_M)
    # the horizontal row at node 1, 30 s after the leader landed: relaxed by the full weight, at any position
    g = ca.Function("g", [nodes[1], times[1]], [horizontal[0]])
    on_its_threshold = g([0, 0, 0, 0, 0, 0], landed + 30.0)
    assert float(on_its_threshold) >= horizontal[1] - 1e-6
    assert float(g([0, 0, 0, 0, 0, 0], landed - 30.0)) < horizontal[1]      # in the air: not relaxed
    # the vertical row is relaxed fully too: 30 s after the landing, any height passes (σ = −1: "below")
    late = ca.Function("late", [nodes[0], times[0]], [vertical[0]])
    assert float(late([0, 0, float(h_leader) + 2000.0, 0, 0, 0], landed + 30.0)) >= vertical[1]


def test_the_replay_track_is_compass():
    sample = lambda psi: type("S", (), dict(t=0.0, lat=LAT0, lon=LON0, alt=100.0, psi=psi))()   # noqa: E731
    flown = FlownTrack.from_samples([sample(math.pi / 2), sample(0.0), sample(-math.pi / 2)])
    assert flown.track_deg.tolist() == pytest.approx([0.0, 90.0, 180.0])
