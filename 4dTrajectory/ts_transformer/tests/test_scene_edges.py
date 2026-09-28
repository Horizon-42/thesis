"""The edge features a scene prior reads between aircraft (`inference/scene_edges`, multi-aircraft design §2.5)."""

import math

import numpy as np
import pytest

from ts_transformer.inference.runway_schedule import CWT_ON_APPROACH_NM, DEPENDENT, Separation
from ts_transformer.inference.scene_edges import EDGE_FEATURES, SceneRows, scene_edges

K = {name: k for k, name in enumerate(EDGE_FEATURES)}
#: Runway "R" landing east (thresholds at the clock's origin), a dependent parallel "P", and "X" of another direction.
SEPARATION = Separation(same_nm=3.0, speed_mps=70.0, relations={frozenset(("R", "P")): DEPENDENT},
                        spacing_nm={frozenset(("R", "P")): 0.6}, right_nm={("R", "P"): -0.6, ("P", "R"): 0.6},
                        diagonal_nm={frozenset(("R", "P")): 1.5}, wake_nm=CWT_ON_APPROACH_NM,
                        along_nm={"R": 0.0, "P": 0.0, "X": 0.0})
GLIDE = math.tan(math.radians(3.0))


def rows(*aircraft, steps: int = 1) -> SceneRows:
    """Each aircraft: dict(time, e, n, height, speed, track, climb, along, runway, category) of scalars (every step
    alike unless a value is a list over the steps)."""
    def column(name):
        return np.array([[a[name][t] if isinstance(a[name], list) else a[name] for t in range(steps)] for a in aircraft],
                        dtype=float)

    time = column("time")
    runway = [[a["runway"] if not np.isnan(time[k, t]) else None for t in range(steps)] for k, a in enumerate(aircraft)]
    said = np.array([[r is not None for r in steps_] for steps_ in runway], dtype=bool)
    rate = np.array([[a["speed"] * math.cos(math.radians(a["track"] - 90.0))] * steps for a in aircraft], dtype=float)
    return SceneRows(time, column("e"), column("n"), column("height"), column("speed"), column("track"),
                     column("climb"), np.where(said, column("along"), np.nan), np.where(said, rate, np.nan), runway,
                     [a["category"] for a in aircraft])


def _aircraft(e, *, speed=70.0, height=600.0, n=0.0, time=0.0, runway="R", category="F", track=90.0):
    return {"time": time, "e": e, "n": n, "height": height, "speed": speed, "track": track,
            "climb": -speed * GLIDE if track == 90.0 else 0.0, "along": e, "runway": runway, "category": category}


def test_the_design_example_both_ways():
    """§2.5: i 6 km behind j on one final, 75 against 70 m/s, j 300 m lower."""
    edges = scene_edges(rows(_aircraft(-12_000.0, speed=75.0, height=900.0), _aircraft(-6_000.0)), SEPARATION)[0]
    i_j, j_i = edges[0, 1], edges[1, 0]
    six, cpa = math.asinh(6_000.0 / 5_556.0), math.asinh(5_400.0 / 5_556.0)
    vertical = (-300.0 + (-70.0 * GLIDE + 75.0 * GLIDE) * 120.0) / 1_000.0
    expected = {"self": 0.0, "front": six, "left": 0.0, "height": -0.3, "clock_ahead": six, "clock_incomparable": 0.0,
                "same": 1.0, "closing": 0.05, "cpa_time": 1.0, "cpa_horizontal": cpa, "cpa_vertical": vertical,
                "required": math.asinh(1.0)}
    for name, value in expected.items():
        assert i_j[K[name]] == pytest.approx(value, abs=1e-5), name
    for name, value in {**expected, "front": -six, "height": 0.3, "clock_ahead": -six, "cpa_vertical": -vertical}.items():
        assert j_i[K[name]] == pytest.approx(value, abs=1e-5), name
    assert edges[0, 0].tolist() == [1.0] + [0.0] * (len(EDGE_FEATURES) - 1)


def test_the_required_distance_follows_the_one_ahead_and_its_wake():
    """A heavy (B) ahead of an F: 5 NM (TBL 5-5-2), whichever way the edge looks; an F ahead of a B: 3 NM."""
    heavy_ahead = scene_edges(rows(_aircraft(-12_000.0), _aircraft(-6_000.0, category="B")), SEPARATION)[0]
    assert heavy_ahead[0, 1, K["required"]] == heavy_ahead[1, 0, K["required"]] == pytest.approx(math.asinh(5.0 / 3.0))
    heavy_behind = scene_edges(rows(_aircraft(-12_000.0, category="B"), _aircraft(-6_000.0)), SEPARATION)[0]
    assert heavy_behind[0, 1, K["required"]] == pytest.approx(math.asinh(1.0))


def test_runways_of_other_directions_or_none_yet_are_not_on_one_clock():
    other = scene_edges(rows(_aircraft(-12_000.0), _aircraft(-6_000.0, runway="X")), SEPARATION)[0]
    assert other[0, 1, K["unrelated"]] == other[0, 1, K["clock_incomparable"]] == 1.0
    assert other[0, 1, K["clock_ahead"]] == other[0, 1, K["required"]] == 0.0
    none_yet = scene_edges(rows(_aircraft(-12_000.0), _aircraft(-6_000.0, runway=None)), SEPARATION)[0]
    assert none_yet[0, 1, K["clock_incomparable"]] == 1.0
    assert all(none_yet[0, 1, K[relation]] == 0.0 for relation in ("same", "single", "dependent", "independent",
                                                                   "unrelated"))
    # a dependent parallel is on the clock, its required distance the diagonal's stagger along the course
    parallel = scene_edges(rows(_aircraft(-12_000.0), _aircraft(-6_000.0, runway="P", n=1_111.0)), SEPARATION)[0]
    assert parallel[0, 1, K["dependent"]] == 1.0 and parallel[0, 1, K["clock_incomparable"]] == 0.0
    assert parallel[0, 1, K["required"]] == pytest.approx(math.asinh(math.sqrt(1.5 ** 2 - 0.6 ** 2) * 1_852.0 / 5_556.0))


def test_a_neighbour_is_read_from_its_last_row_before_the_aircrafts_instant_and_carried_on_to_it():
    """Rows hang within half a step of their steps: i's at −0.9 s and 1.1 s, j's at 0.9 s and 2.9 s. Step 1: j's row
    (2.9 s) is after i's (1.1 s) — j is read at step 0 (0.9 s) carried 0.2 s on. Step 0: j's row (0.9 s) is after i's
    (−0.9 s) with no earlier one: carried 1.8 s back."""
    i = _aircraft([-12_000.0, -11_860.0], time=[-0.9, 1.1])
    j = _aircraft([-6_000.0, -5_860.0], time=[0.9, 2.9])
    edges = scene_edges(rows(i, j, steps=2), SEPARATION)
    at_step_1 = (-6_000.0 + 70.0 * 0.2) - (-11_860.0)
    assert edges[1, 0, 1, K["front"]] == pytest.approx(math.asinh(at_step_1 / 5_556.0), abs=1e-6)
    at_step_0 = (-6_000.0 - 70.0 * 1.8) - (-12_000.0)
    assert edges[0, 0, 1, K["front"]] == pytest.approx(math.asinh(at_step_0 / 5_556.0), abs=1e-6)
    # the clock is carried the same way: j is 0.2 s × 70 m/s further along at step 1
    assert edges[1, 0, 1, K["clock_ahead"]] == pytest.approx(math.asinh(at_step_1 / 5_556.0), abs=1e-6)


def test_a_non_finite_state_or_a_runway_without_a_clock_is_refused():
    broken = _aircraft(-6_000.0)
    broken["speed"] = float("nan")
    with pytest.raises(ValueError, match="non-finite ground_speed_mps"):
        rows(_aircraft(-12_000.0), broken)
    scene = rows(_aircraft(-12_000.0))
    with pytest.raises(ValueError, match="on the approach clock exactly"):
        SceneRows(scene.time_s, scene.e_m, scene.n_m, scene.height_m, scene.ground_speed_mps, scene.track_deg,
                  scene.vertical_rate_mps, np.full((1, 1), np.nan), scene.along_rate_mps, scene.runway, scene.category)


def test_an_absent_aircraft_has_no_edges_and_a_crossing_pair_closes_to_its_closest_point():
    absent = _aircraft(-6_000.0, time=[0.0, np.nan])
    edges = scene_edges(rows(_aircraft(-12_000.0), absent, steps=2), SEPARATION)
    assert not edges[1, 0, 1].any() and not edges[1, 1].any() and edges[1, 0, 0, K["self"]] == 1.0
    # j 4 km north of i's track, 4 km ahead, flying south at 70 m/s across it; i east at 70 m/s: closest at 57.1 s
    crossing = scene_edges(rows(_aircraft(-12_000.0), _aircraft(-8_000.0, n=4_000.0, track=180.0, runway=None)),
                           SEPARATION)[0, 0, 1]
    assert crossing[K["cpa_time"]] == pytest.approx(4_000.0 / 70.0 / 120.0)
    assert crossing[K["cpa_horizontal"]] == pytest.approx(0.0, abs=1e-9)
    assert crossing[K["left"]] == pytest.approx(math.asinh(4_000.0 / 5_556.0))
