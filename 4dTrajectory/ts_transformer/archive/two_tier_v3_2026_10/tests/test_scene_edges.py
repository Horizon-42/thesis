"""The edge features a scene prior reads between aircraft (`inference/scene_edges`, multi-aircraft design §2.5)."""

import math

import numpy as np
import pytest

from ts_transformer.inference.runway_schedule import CWT_ON_APPROACH_NM, DEPENDENT, Separation
from ts_transformer.inference.scene_edges import EDGE_FEATURES, SceneRows, scene_edge_blocks, scene_edges

K = {name: k for k, name in enumerate(EDGE_FEATURES)}
#: Runway "R" landing east (thresholds at the clock's origin), a dependent parallel "P", and "X" of another direction.
SEPARATION = Separation(same_nm=3.0, speed_mps=70.0, relations={frozenset(("R", "P")): DEPENDENT},
                        spacing_nm={frozenset(("R", "P")): 0.6}, right_nm={("R", "P"): -0.6, ("P", "R"): 0.6},
                        diagonal_nm={frozenset(("R", "P")): 1.5}, wake_nm=CWT_ON_APPROACH_NM,
                        along_nm={"R": 0.0, "P": 0.0, "X": 0.0})
GLIDE = math.tan(math.radians(3.0))


def rows(*aircraft, steps: int = 2) -> SceneRows:
    """Each aircraft: dict(time, e, n, height, speed, track, climb, runway, category). ``time`` is the last step's row
    time (earlier rows 2 s apart) or a list over the steps (NaN: absent); ``e``, ``n``, ``height`` the last step's
    position, earlier rows flown back from it at ``speed``, ``track`` and ``climb`` — or lists over the steps. The clock
    position is ``e`` (runways R and P land east from the origin)."""
    def times(a):
        return np.array(a["time"] if isinstance(a["time"], list) else [a["time"] - 2.0 * (steps - 1 - t)
                                                                        for t in range(steps)])

    time = np.stack([times(a) for a in aircraft])

    def column(name, rate):
        out = np.full(time.shape, np.nan)
        for k, a in enumerate(aircraft):
            last = np.flatnonzero(~np.isnan(time[k]))[-1]
            values = (np.array(a[name], dtype=float) if isinstance(a[name], list)
                      else a[name] - rate(a) * (time[k, last] - time[k]))
            out[k] = np.where(np.isnan(time[k]), np.nan, values)
        return out

    def speed_along(axis):
        return lambda a: a["speed"] * axis(math.radians(a["track"]))

    e, n = column("e", speed_along(math.sin)), column("n", speed_along(math.cos))
    runway = [[a["runway"] if not np.isnan(time[k, t]) else None for t in range(steps)] for k, a in enumerate(aircraft)]
    said = np.array([[r is not None for r in steps_] for steps_ in runway], dtype=bool)
    return SceneRows(time, e, n, column("height", lambda a: a["climb"]), np.where(said, e, np.nan), runway,
                     [a["category"] for a in aircraft])


def _aircraft(e, *, speed=70.0, height=600.0, n=0.0, time=0.0, runway="R", category="F", track=90.0):
    return {"time": time, "e": e, "n": n, "height": height, "speed": speed, "track": track,
            "climb": -speed * GLIDE if track == 90.0 else 0.0, "runway": runway, "category": category}


def test_the_design_example_both_ways():
    """§2.5: i 6 km behind j on one final, 75 against 70 m/s, j 300 m lower — read at the second row, where both have
    moved since their first."""
    edges = scene_edges(rows(_aircraft(-12_000.0, speed=75.0, height=900.0), _aircraft(-6_000.0)), SEPARATION)[1]
    i_j, j_i = edges[0, 1], edges[1, 0]
    six, cpa = math.asinh(6_000.0 / 5_556.0), math.asinh(5_400.0 / 5_556.0)
    vertical = (-300.0 + (-70.0 * GLIDE + 75.0 * GLIDE) * 120.0) / 1_000.0
    expected = {"self": 0.0, "front": six, "left": 0.0, "height": -0.3, "clock_ahead": six, "clock_incomparable": 0.0,
                "same": 1.0, "closing": 0.05, "cpa_time": 1.0, "cpa_horizontal": cpa, "cpa_vertical": vertical,
                "motion_unknown": 0.0, "required": math.asinh(1.0)}
    for name, value in expected.items():
        assert i_j[K[name]] == pytest.approx(value, abs=1e-5), name
    for name, value in {**expected, "front": -six, "height": 0.3, "clock_ahead": -six, "cpa_vertical": -vertical}.items():
        assert j_i[K[name]] == pytest.approx(value, abs=1e-5), name
    assert edges[0, 0].tolist() == [1.0] + [0.0] * (len(EDGE_FEATURES) - 1)


def test_the_required_distance_follows_the_one_ahead_and_its_wake():
    """A heavy (B) ahead of an F: 5 NM (TBL 5-5-2), whichever way the edge looks; an F ahead of a B: 3 NM."""
    heavy_ahead = scene_edges(rows(_aircraft(-12_000.0), _aircraft(-6_000.0, category="B")), SEPARATION)[1]
    assert heavy_ahead[0, 1, K["required"]] == heavy_ahead[1, 0, K["required"]] == pytest.approx(math.asinh(5.0 / 3.0))
    heavy_behind = scene_edges(rows(_aircraft(-12_000.0, category="B"), _aircraft(-6_000.0)), SEPARATION)[1]
    assert heavy_behind[0, 1, K["required"]] == pytest.approx(math.asinh(1.0))


def test_runways_of_other_directions_or_none_yet_are_not_on_one_clock():
    other = scene_edges(rows(_aircraft(-12_000.0), _aircraft(-6_000.0, runway="X")), SEPARATION)[1]
    assert other[0, 1, K["unrelated"]] == other[0, 1, K["clock_incomparable"]] == 1.0
    assert other[0, 1, K["clock_ahead"]] == other[0, 1, K["required"]] == 0.0
    none_yet = scene_edges(rows(_aircraft(-12_000.0), _aircraft(-6_000.0, runway=None)), SEPARATION)[1]
    assert none_yet[0, 1, K["clock_incomparable"]] == 1.0
    assert all(none_yet[0, 1, K[relation]] == 0.0 for relation in ("same", "single", "dependent", "independent",
                                                                   "unrelated"))
    # a dependent parallel is on the clock, its required distance the diagonal's stagger along the course
    parallel = scene_edges(rows(_aircraft(-12_000.0), _aircraft(-6_000.0, runway="P", n=1_111.0)), SEPARATION)[1]
    assert parallel[0, 1, K["dependent"]] == 1.0 and parallel[0, 1, K["clock_incomparable"]] == 0.0
    assert parallel[0, 1, K["required"]] == pytest.approx(math.asinh(math.sqrt(1.5 ** 2 - 0.6 ** 2) * 1_852.0 / 5_556.0))


def test_a_neighbour_is_read_at_the_same_instant_and_a_first_row_has_no_motion():
    """Every row is on a step: i's at 0, 2 and 4 s, j's at 2 and 4 s, both flying east at 70 m/s, 6 km apart. Step 2: j at
    its row there, nothing carried, both motions known (no closing). Step 1: j's first row has no motion — the pair's
    motion is unknown. Step 0: j is absent."""
    i = _aircraft([-12_000.0, -11_860.0, -11_720.0], time=[0.0, 2.0, 4.0])
    j = _aircraft([0.0, -5_860.0, -5_720.0], time=[np.nan, 2.0, 4.0])
    edges = scene_edges(rows(i, j, steps=3), SEPARATION)
    ahead = math.asinh(6_000.0 / 5_556.0)
    assert edges[2, 0, 1, K["front"]] == pytest.approx(ahead, abs=1e-6)
    assert edges[2, 0, 1, K["clock_ahead"]] == pytest.approx(ahead, abs=1e-6)
    assert edges[2, 0, 1, K["motion_unknown"]] == 0.0 and edges[2, 0, 1, K["closing"]] == pytest.approx(0.0, abs=1e-6)
    assert edges[1, 0, 1, K["front"]] == pytest.approx(ahead, abs=1e-6)
    assert edges[1, 0, 1, K["motion_unknown"]] == 1.0
    assert all(edges[1, 0, 1, K[name]] == 0.0 for name in ("closing", "cpa_time", "cpa_horizontal", "cpa_vertical"))
    assert not edges[0, 0, 1].any() and not edges[0, 1].any()


def test_a_scene_whose_aircraft_are_at_different_times_at_a_step_is_refused():
    """A step is one time (rows on each flight's own first sample, the artefacts before 2026-10-02, are not)."""
    i = _aircraft([-12_000.0, -11_860.0, -11_720.0], time=[0.0, 2.0, 4.0])
    j = _aircraft([0.0, -5_860.0, -5_720.0], time=[np.nan, 2.9, 4.9])
    with pytest.raises(ValueError, match="different times"):
        scene_edges(rows(i, j, steps=3), SEPARATION)


def test_the_frame_is_the_direction_of_the_displacement_since_the_row_before():
    """i flies east, then its last displacement is north: j, 1 km east of i, is on i's right, not ahead — the future
    (i's next row) never enters, and neither does a fitted track."""
    i = _aircraft([-2_000.0, -1_860.0, -1_860.0], n=[0.0, 0.0, 140.0], time=2.0, runway=None)
    j = _aircraft([-860.0, -860.0, -860.0], n=[140.0, 140.0, 140.0], time=2.0, runway=None)
    edge = scene_edges(rows(i, j, steps=3), SEPARATION)[2, 0, 1]
    assert edge[K["front"]] == pytest.approx(0.0, abs=1e-6)
    assert edge[K["left"]] == pytest.approx(-math.asinh(1_000.0 / 5_556.0), abs=1e-6)
    # an aircraft that did not move since its row before has no frame; its motion (none) is known
    still = _aircraft([-1_860.0, -1_860.0], n=[140.0, 140.0], time=2.0, runway=None)
    edge = scene_edges(rows(still, _aircraft([-860.0, -860.0], n=[140.0, 140.0], time=2.0, runway=None)),
                       SEPARATION)[1, 0, 1]
    assert edge[K["front"]] == edge[K["left"]] == edge[K["motion_unknown"]] == 0.0
    assert edge[K["cpa_horizontal"]] == pytest.approx(math.asinh(1_000.0 / 5_556.0), abs=1e-6)


def test_a_non_finite_position_a_broken_run_or_a_runway_without_a_clock_is_refused():
    broken = _aircraft(-6_000.0)
    broken["height"] = float("nan")
    with pytest.raises(ValueError, match="non-finite height_m"):
        rows(_aircraft(-12_000.0), broken)
    with pytest.raises(ValueError, match="consecutive steps"):
        rows(_aircraft([-12_000.0, 0.0, -11_720.0], time=[0.0, np.nan, 4.0]), steps=3)
    scene = rows(_aircraft(-12_000.0))
    with pytest.raises(ValueError, match="on the approach clock exactly"):
        SceneRows(scene.time_s, scene.e_m, scene.n_m, scene.height_m, np.full((1, 2), np.nan), scene.runway,
                  scene.category)


def test_an_absent_aircraft_has_no_edges_and_a_crossing_pair_closes_to_its_closest_point():
    absent = _aircraft([-6_140.0, -6_000.0, 0.0], time=[0.0, 2.0, np.nan])
    edges = scene_edges(rows(_aircraft(-12_000.0, time=4.0), absent, steps=3), SEPARATION)
    assert not edges[2, 0, 1].any() and not edges[2, 1].any() and edges[2, 0, 0, K["self"]] == 1.0
    # j 4 km north of i's track, 4 km ahead, flying south at 70 m/s across it; i east at 70 m/s: closest at 57.1 s
    crossing = scene_edges(rows(_aircraft(-12_000.0), _aircraft(-8_000.0, n=4_000.0, track=180.0, runway=None)),
                           SEPARATION)[1, 0, 1]
    assert crossing[K["cpa_time"]] == pytest.approx(4_000.0 / 70.0 / 120.0)
    assert crossing[K["cpa_horizontal"]] == pytest.approx(0.0, abs=1e-6)
    assert crossing[K["left"]] == pytest.approx(math.asinh(4_000.0 / 5_556.0))


def test_scenes_stacked_on_one_axis_read_each_scene_as_alone():
    """`scene_edge_blocks` (the loop stacks the scenes of one airport in one call): each scene's block is its features
    alone, value for value (no pair across two scenes is read)."""
    first = rows(_aircraft(-4_000.0), _aircraft(-9_000.0, category="C"), _aircraft(-6_000.0, runway="P", n=-1_100.0),
                 steps=4)
    # the second scene at other times: each scene's steps are its own
    second = rows(_aircraft(-12_000.0, runway="X", track=270.0, time=600.0), _aircraft(-3_000.0, speed=65.0, time=600.0),
                  steps=4)
    stacked = SceneRows(np.concatenate([first.time_s, second.time_s]), np.concatenate([first.e_m, second.e_m]),
                        np.concatenate([first.n_m, second.n_m]), np.concatenate([first.height_m, second.height_m]),
                        np.concatenate([first.along_m, second.along_m]), [*first.runway, *second.runway],
                        [*first.category, *second.category])
    one, two = scene_edge_blocks(stacked, SEPARATION, [3, 2])
    assert np.array_equal(one, scene_edges(first, SEPARATION)) and np.array_equal(two, scene_edges(second, SEPARATION))
    assert one[..., 1:].any() and two[..., 1:].any()
    with pytest.raises(ValueError, match="stacked"):
        scene_edge_blocks(stacked, SEPARATION, [3, 3])
