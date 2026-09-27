"""The multi-aircraft design's §2.5 figure of the approach clock (`experiments/approach_clock_figure`): the file in the
docs is what the runner draws, and the rule it draws — position = the threshold's `along_nm` − ``before_threshold_m``
— has the conventions of the code it cites."""

import math

import pytest
from geokit import METRES_PER_DEG_LAT, NM_M, metres_per_deg_lon

from ts_transformer.experiments import approach_clock_figure as figure
from ts_transformer.inference.runway_schedule import Separation, parallel_relations
from ts_transformer.instructions.airport import RunwayCandidate, relative_to_runway

COURSE_DEG = 225.0     # a compass course, near KRDU's 23s (225.03°)


def unit(course_deg: float) -> tuple[float, float]:
    """(east, north) of a compass course."""
    return math.sin(math.radians(course_deg)), math.cos(math.radians(course_deg))


def test_the_figure_in_the_docs_is_what_the_runner_draws():
    """Change the example or the drawing, then run `python run_ts.py approach_clock_figure` and commit the SVG with it."""
    assert figure.FIGURE.read_text(encoding="utf-8") == figure.render()


def test_j_is_ahead_by_the_difference_of_the_distances_plus_the_stagger():
    assert figure.ahead_m() == pytest.approx(8_000.0 - 6_000.0 + figure.STAGGER_M)


def test_along_nm_is_a_thresholds_position_along_the_landing_direction_from_the_first_runway_by_name():
    """23R's threshold placed `STAGGER_M` back along the course (and 1,067 m to its right) reads −`STAGGER_M` from
    23L's, the origin — the figure's `ALONG_M`. 23R is listed first, so the origin is 23L by NAME, not by order."""
    lat0, lon0 = 35.88, -78.78
    ue, un = unit(COURSE_DEG)
    back, right = -figure.STAGGER_M, 1_067.0
    east, north = back * ue + right * un, back * un - right * ue
    targets = {"23R": {"lat": lat0 + north / METRES_PER_DEG_LAT, "lon": lon0 + east / metres_per_deg_lon(lat0),
                       "course_deg": COURSE_DEG},
               "23L": {"lat": lat0, "lon": lon0, "course_deg": COURSE_DEG}}
    along_nm = parallel_relations(targets, ())[4]
    assert {runway: along_nm[runway] * NM_M for runway in along_nm} == pytest.approx(figure.ALONG_M, abs=1e-6)


def test_before_threshold_is_how_far_the_aircraft_still_is_before_it_along_the_course():
    candidate = RunwayCandidate("23L", 1_000.0, -2_000.0, COURSE_DEG, 130.0, 3_000.0)
    ue, un = unit(COURSE_DEG)
    # 6,000 m back along the course from the threshold, 300 m to one side: 6,000 m before it
    e = candidate.threshold_e_m - 6_000.0 * ue + 300.0 * un
    n = candidate.threshold_n_m - 6_000.0 * un - 300.0 * ue
    assert float(relative_to_runway(e, n, COURSE_DEG, 500.0, candidate).before_threshold_m) == pytest.approx(6_000.0)


def test_ahead_is_the_approach_clocks_order():
    """The figure's "ahead" is the approach clock of `runway_schedule` (T3's one definition): both aircraft flown on at
    one speed, j crosses its threshold ahead of i on the clock by exactly the figure's distance."""
    speed_mps = 70.0
    separation = Separation(same_nm=3.0, speed_mps=speed_mps,
                            along_nm={runway: along / NM_M for runway, along in figure.ALONG_M.items()})
    clock = {name: separation.approach_time_s(runway, before / speed_mps) for name, runway, before in figure.AIRCRAFT}
    assert (clock["i"] - clock["j"]) * speed_mps == pytest.approx(figure.ahead_m())
