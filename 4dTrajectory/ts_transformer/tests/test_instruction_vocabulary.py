"""The instruction vocabulary's parts: the spec, the words, the grammar, the candidate runways, the envelopes,
the piecewise fit, the measurements and the artefact (design: docs/two_tier/two_tier_design.md §3, §4)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from aerodynamic_model.common import GeodeticState
from ts_transformer.data.channels import channels_from_states

from ts_transformer.instructions import envelope, measure
from ts_transformer.instructions.airport import AirportGeometry, airport_geometry, relative_to_runway
from ts_transformer.instructions.artefact import (
    CANDIDATES_SCHEMA, SENTENCES_SCHEMA, keep_spec, load_candidates, load_day_split, load_sentences, load_signals,
    load_spec, write_candidates, write_sentences, write_signals, write_spec,
)
from ts_transformer.instructions.labeller.read import admit, read_flight
from ts_transformer.instructions.piecewise import fit_pieces, moving_average
from ts_transformer.instructions.signals import FlightSignals, signals_from_series
from ts_transformer.instructions.spec import SPEC_SCHEMA, VocabularySpec
from ts_transformer.instructions.words import (
    ANGLE_LEVEL, COLUMNS, Words, compass_from_math_rad, math_rad_from_compass, wrap180,
)
from ts_transformer.data.day_split import SealedDay
from ts_transformer.tests.support import (
    fixture_days, fly_legs, instruction_airport, instruction_flight, instruction_spec, landing_on,
)


@pytest.fixture
def geometry() -> AirportGeometry:
    return instruction_airport()


spec = instruction_spec


# ---- spec
def test_the_sha_is_stable_and_moves_with_any_field():
    a, b = spec(), spec()
    assert a.sha256 == b.sha256
    assert spec(level_band_m=25.0).sha256 == a.sha256
    assert spec(altitude_segment_steps_m=(60.0, 120.0, 450.0)).sha256 == a.sha256
    assert spec(heading_tolerance_deg=5.0).sha256 != a.sha256


def test_from_dict_refuses_a_missing_or_an_extra_key_and_another_reading_rule():
    data = spec().to_dict()
    with pytest.raises(ValueError, match="missing"):
        VocabularySpec.from_dict({k: v for k, v in data.items() if k != "step_s"})
    with pytest.raises(ValueError, match="unexpected"):
        VocabularySpec.from_dict({**data, "stray": 1})
    with pytest.raises(ValueError, match="reading rule"):
        VocabularySpec.from_dict({**data, "reading_rule": "instruction-v0"})


@pytest.mark.parametrize("changes, message", [
    ({"heading_tolerance_deg": 2.0}, "half a heading step"),
    ({"altitude_segment_steps_m": (80.0, 120.0, 450.0)}, "whole number of 80 m steps"),
    ({"altitude_segment_tops_m": (1260.0, 1200.0, 5400.0)}, "whole number"),
    ({"altitude_segment_steps_m": (60.0, 120.0)}, "one step per segment top"),
    ({"go_around_along_m": (3000.0, -10000.0)}, "increasing pair"),
    ({"heading_step_deg": 7.0}, "does not divide 360"),
    ({"heading_lead_s": 3.0}, "not a whole number of steps"),
    ({"heading_lead_s": -2.0}, "not a whole number of steps"),
    ({"descent_angle_edges_deg": [-0.5, 3.0, 2.6, 3.7, 10.0]}, "edges must increase"),
    ({"descent_angle_centres_deg": [0.8, 2.1, 3.0, 11.0]}, "inside its class"),
])
def test_the_spec_refuses_inconsistent_values(changes, message):
    with pytest.raises(ValueError, match=message):
        spec(**changes)


# ---- words
def test_the_columns_are_five_and_the_class_counts_are_the_designs():
    """§3.1: runway, heading, altitude, angle, speed; 72 headings, 40 levels + "no level-off", level + 4 descents +
    climb, 47 speeds + "unspecified" (each column's "unchanged" apart)."""
    assert COLUMNS == ("runway", "heading", "altitude", "angle", "speed")
    assert Words(spec()).class_counts() == {"heading": 72, "altitude": 41, "angle": 6, "speed": 48}


@pytest.mark.parametrize("course", [90.0, 122.3, 242.7, 302.3, 358.0, 2.0])
def test_heading_classes_are_relative_to_the_course_both_ways(course):
    """D8: class k is the track course + 5k°; class 0 is the course itself (KSTL's 2.3°-off-grid courses included),
    class 36 the opposite direction, 18 and 54 the base legs; the conversion wraps 0°/360° both ways."""
    words = Words(spec())
    assert words.heading_class(course, course) == 0
    assert words.heading_track_deg(0, course) == pytest.approx(course % 360.0)
    assert words.heading_track_deg(36, course) == pytest.approx((course + 180.0) % 360.0)
    assert words.heading_track_deg(18, course) == pytest.approx((course + 90.0) % 360.0)
    assert words.heading_track_deg(54, course) == pytest.approx((course - 90.0) % 360.0)
    for k in range(words.n_heading):
        track = words.heading_track_deg(k, course)
        assert 0.0 <= track < 360.0 and words.heading_class(track, course) == k
        assert words.heading_class(track + 2.4, course) == k and words.heading_class(track - 2.4 + 360.0, course) == k


def test_relative_headings_lie_in_the_half_open_circle():
    words = Words(spec())
    assert words.heading_relative_deg(0) == 0.0 and words.heading_relative_deg(36) == 180.0
    assert words.heading_relative_deg(37) == -175.0 and words.heading_relative_deg(71) == -5.0
    assert words.heading_index(-5.0) == 71 and words.heading_index(-180.0) == 36 and words.heading_index(357.6) == 0


def test_altitude_words_are_the_40_levels_of_three_segments():
    """D22: 60 m steps 0–1,260 m, 120 m to 2,700 m, 450 m to 5,400 m; ε is half the larger gap to a level's neighbours
    + 10 m — half the segment's step, except at a segment's top level, which takes the next segment's half step."""
    words = Words(spec())
    levels = words.altitude_levels
    assert len(levels) == 40 == words.n_altitude_levels
    assert levels.tolist() == [*np.arange(0.0, 1261.0, 60.0), *np.arange(1380.0, 2701.0, 120.0),
                               *np.arange(3150.0, 5401.0, 450.0)]
    assert [words.altitude_tolerance_m(i) for i in (0, 20, 21, 22, 32, 33, 34, 39)] == [
        40.0, 40.0, 70.0, 70.0, 70.0, 235.0, 235.0, 235.0]
    assert words.altitude_tolerance_m(words.altitude_no_level_off) == 40.0
    assert words.altitude_m(words.altitude_index(914.0)) == 900.0
    assert words.altitude_m(words.altitude_index(630.0)) == 600.0          # a tie goes to the lower level
    assert words.altitude_m(words.altitude_index(1319.0)) == 1260.0
    assert words.altitude_m(words.altitude_index(1321.0)) == 1380.0
    assert words.altitude_m(words.altitude_index(5600.0)) == 5400.0
    assert words.altitude_m(words.altitude_no_level_off) is None
    for outside in (5626.0, -31.0):
        with pytest.raises(ValueError):
            words.altitude_index(outside)


def test_angle_classes_follow_the_edges():
    words = Words(spec())
    assert words.angle_index(-0.2) == 1           # nearly flat inside a descent keeps a descent class
    assert words.angle_index(3.0) == 3
    assert words.angle_index(10.0) == 4
    assert words.angle_index(-2.0) == words.angle_climb
    assert words.angle_bounds(3) == (2.6, 3.7)
    assert words.angle_bounds(ANGLE_LEVEL) == (0.0, 0.0)
    with pytest.raises(ValueError):
        words.angle_index(12.0)
    with pytest.raises(ValueError):
        words.angle_index(-20.0)


def test_speed_words_and_unspecified():
    words = Words(spec())
    assert words.speed_mps(words.speed_index(101.0)) == 100.0
    assert words.speed_mps(words.speed_unspecified) is None
    with pytest.raises(ValueError):
        words.speed_index(10.0)


def test_compass_and_math_headings_round_trip():
    for track in (0.0, 45.0, 90.0, 225.0, 315.0):
        assert compass_from_math_rad(math_rad_from_compass(track)) == pytest.approx(track % 360.0)
    assert math.degrees(float(math_rad_from_compass(315.0))) % 360.0 == pytest.approx(135.0)
    assert float(wrap180(190.0)) == pytest.approx(-170.0)


# ---- airport
def test_the_candidates_are_the_published_geometry_with_the_configured_length():
    ends = [SimpleNamespace(ident="23R", lat=35.8937988, lon=-78.7779999, course_deg=225.3),
            SimpleNamespace(ident="05L", lat=35.8753, lon=-78.7970, course_deg=45.3)]
    geometry = airport_geometry("KRDU", {"23R": {"lat": 35.8937988, "lon": -78.7779999, "elevation_msl_m": 124.7,
                                                 "course_deg": 225.3}}, ends)
    (candidate,) = geometry.candidates
    assert candidate.ident == "23R"
    assert candidate.course_deg == pytest.approx(225.3)
    assert candidate.length_m == pytest.approx(3048.0)
    assert (candidate.threshold_e_m, candidate.threshold_n_m) == pytest.approx((843.0, 1683.0), abs=15.0)
    assert [end.ident for end in geometry.runway_ends] == ["05L", "23R"]      # every runway end, not only candidates
    assert AirportGeometry.from_dict(geometry.to_dict()) == geometry
    with pytest.raises(KeyError):
        geometry.candidate_index("05L")
    with pytest.raises(KeyError, match="not runway ends the harvest builds"):
        airport_geometry("KRDU", {"23R": {"lat": 35.8937988, "lon": -78.7779999, "elevation_msl_m": 124.7,
                                          "course_deg": 225.3}}, ends[1:])


def test_relative_position_signs(geometry):
    candidate = geometry.candidates[0]                 # threshold at the origin, course 090
    rel = relative_to_runway(np.array([-3000.0, -3000.0, 500.0]), np.array([0.0, -200.0, 0.0]),
                             np.array([90.0, 80.0, 90.0]), np.array([300.0, 300.0, 100.0]), candidate)
    assert rel.before_threshold_m.tolist() == pytest.approx([3000.0, 3000.0, -500.0])
    assert rel.right_of_course_m.tolist() == pytest.approx([0.0, 200.0, 0.0])    # south of an eastbound course is right
    assert rel.track_minus_course_deg.tolist() == pytest.approx([0.0, -10.0, 0.0])
    assert rel.height_above_threshold_m.tolist() == pytest.approx([200.0, 200.0, 0.0])


def test_the_frame_reads_latlon_back_into_its_own_metres(geometry):
    frame = geometry.frame
    points = [(0.0, 0.0), (12345.0, -6789.0), (-40000.0, 25000.0)]
    latlon = [frame.latlon_from_horizontal(e, n) for e, n in points]
    e, n = frame.horizontal_from_latlon(np.array([p[0] for p in latlon]), np.array([p[1] for p in latlon]))
    assert np.allclose(e, [p[0] for p in points], atol=1e-6) and np.allclose(n, [p[1] for p in points], atol=1e-6)


def _series(geometry, target_psi: float):
    """A turn from compass 350 through north to 010 (math psi 100° → 80°), 2 s apart, 70 m/s
    along a 3° climb, as the ts data plane's FlightSeries carries it."""
    frame = geometry.frame
    samples, e, n = [], -20000.0, -500.0
    for row in range(11):
        compass = 350.0 + 2.0 * row
        lat, lon = frame.latlon_from_horizontal(e, n)
        samples.append((2.0 * row, GeodeticState(latitude=lat, longitude=lon, altitude=900.0 + row * 7.0, V=70.0,
                                                 psi=float(math_rad_from_compass(compass)), gamma=math.radians(3.0), m=6.0e4)))
        e += 2.0 * 70.0 * math.cos(math.radians(3.0)) * math.sin(math.radians(compass))
        n += 2.0 * 70.0 * math.cos(math.radians(3.0)) * math.cos(math.radians(compass))
    times, values = channels_from_states(samples, frame)
    lat0, lon0 = frame.latlon_from_horizontal(0.0, 0.0)
    target = GeodeticState(latitude=lat0, longitude=lon0, altitude=100.0, V=70.0, psi=target_psi, gamma=0.0, m=6.0e4)
    # A GLF4 is flown as its substitute CRJ9 (aircraft/performance_index.json).
    scenario = SimpleNamespace(source={"runway": "09", "resolved_typecode": "GLF4", "entry_time_utc": "2026-06-01T11:58:00.5Z",
                                       "landing_time_utc": "2026-06-01T12:00:00Z"}, target=target,
                               initial=samples[0][1], aircraft=SimpleNamespace(code="CRJ9"))
    return SimpleNamespace(airport="KXXX", dataset_id="KXXX:turn", scenario=scenario, times=times, values=values,
                           frame=frame), samples


def test_a_flight_without_dynamics_keeps_its_signals_and_names_no_type(geometry):
    # `all-flights` keeps a flight whose type has no dynamics (or no identity): no aircraft,
    # unknown mass. Its signals are kinematics only, and its type is None -- never an A320.
    series, _ = _series(geometry, float(math_rad_from_compass(90.0)))
    series.scenario.aircraft = None
    series.scenario.initial = replace(series.scenario.initial, m=math.nan)
    series.scenario.source["resolved_typecode"] = None
    flight = signals_from_series(series, geometry)
    assert flight.typecode is None
    assert flight.ground_speed_mps.tolist() == pytest.approx([70.0 * math.cos(math.radians(3.0))] * 11)


def test_signals_from_a_series_are_an_unwrapped_compass_track_in_the_airport_frame(geometry):
    series, samples = _series(geometry, float(math_rad_from_compass(90.0)))
    flight = signals_from_series(series, geometry)
    assert flight.runway == "09" and flight.typecode == "GLF4"     # its own type, not the flown one
    # its absolute clock: row 0 at the arrival slice's entry, and the landing that names its day
    assert (flight.entry_time_utc, flight.landing_time_utc) == ("2026-06-01T11:58:00.5Z", "2026-06-01T12:00:00Z")
    assert flight.track_deg.tolist() == pytest.approx([350.0 + 2.0 * row for row in range(11)], abs=1e-6)
    assert flight.ground_speed_mps.tolist() == pytest.approx([70.0 * math.cos(math.radians(3.0))] * 11)
    assert flight.vertical_rate_mps.tolist() == pytest.approx([70.0 * math.sin(math.radians(3.0))] * 11)
    assert flight.altitude_m.tolist() == pytest.approx([900.0 + row * 7.0 for row in range(11)])
    assert (flight.e_m[0], flight.n_m[0]) == pytest.approx((-20000.0, -500.0), abs=1e-6)
    # the flight's own target must be the candidate's threshold and course
    series, _ = _series(geometry, float(math_rad_from_compass(92.0)))
    with pytest.raises(ValueError, match="not candidate 09's threshold and course"):
        signals_from_series(series, geometry)


# ---- envelopes
def test_the_vertical_tube_stops_at_the_target():
    low, high = envelope.vertical_tube(np.array([0.0, 1000.0, 20000.0]), 1200.0, 900.0, (2.6, 3.7), 25.0)
    assert low[0] == pytest.approx(1175.0) and high[0] == pytest.approx(1225.0)
    assert high[1] == pytest.approx(1200.0 - 1000.0 * math.tan(math.radians(2.6)) + 25.0)
    assert (low[2], high[2]) == pytest.approx((875.0, 925.0))
    low, high = envelope.vertical_tube(np.array([0.0, 10000.0]), 900.0, 1500.0, (-15.0, -0.5), 25.0)
    assert (low[1], high[1]) == pytest.approx((900.0 + 10000.0 * math.tan(math.radians(0.5)) - 25.0, 1525.0))
    # the class's direction clamps: a climb class whose target sits just below the start holds
    # the tube at the target instead of opening it upward
    low, high = envelope.vertical_tube(np.array([0.0, 10000.0]), 910.0, 900.0, (-15.0, -0.5), 25.0)
    assert (low[1], high[1]) == pytest.approx((875.0, 925.0))


def test_the_corridor_widens_with_distance():
    width = envelope.corridor_half_width_m(np.array([0.0, 10000.0, -50.0]), 20.0, 0.45)
    assert width.tolist() == pytest.approx([20.0, 20.0 + 10000.0 * math.tan(math.radians(0.45)), 20.0])
    inside = envelope.corridor(np.array([30.0, 30.0]), np.array([0.0, 3.0]), np.array([5000.0, 5000.0]), 20.0, 0.45, 2.0)
    assert inside.tolist() == [True, False]


def test_a_heading_word_is_judged_from_its_row_plus_the_lead_to_the_next_words():
    """§3.3: a word said at row r says where the track is a lead later — judged from r + lead to the next word's row +
    lead, never at or past the end; a word the lead carries past it has no rows."""
    assert envelope.heading_word_rows([0, 10, 25, 29], 2, 30) == [(2, 12), (12, 27), (27, 30), (31, 31)]
    track = np.concatenate((np.full(12, 90.0), np.full(10, 95.0), np.full(8, 101.0)))
    judged = envelope.heading_words_inside(track, [(0, 90.0), (10, 95.0), (20, 100.0)], 2, 30, 4.5)
    assert [(j["rows"], j["inside"]) for j in judged] == [(10, 10), (10, 10), (8, 8)]
    # the same words over a track that turned 6° past its word from row 25 on: those rows are outside
    late = track.copy()
    late[25:] = 106.0
    assert [j["inside"] for j in envelope.heading_words_inside(late, [(0, 90.0), (10, 95.0), (20, 100.0)], 2, 30, 4.5)] \
        == [10, 10, 3]
    # the circle: 358° against a word of 005 is 7° off, 002° is 3° off
    assert envelope.heading_words_inside(np.array([358.0, 2.0]), [(0, 5.0)], 0, 2, 4.5)[0]["inside"] == 1


def test_speed_transitions():
    assert envelope.speed_transition_ok(np.array([120.0, 110.0, 101.0]), 100.0, 5.0)
    assert not envelope.speed_transition_ok(np.array([120.0, 100.0, 112.0]), 100.0, 5.0)


# ---- piecewise
def test_fit_pieces_finds_the_breaks_within_the_tolerance():
    x = np.arange(0.0, 300.0)
    y = np.where(x < 100, 1000.0, np.where(x < 200, 1000.0 - (x - 100) * 5.0, 500.0))
    pieces = fit_pieces(x, y + np.sin(x) * 2.0, tolerance=5.0)
    assert [p.start for p in pieces] == [0] + [p.stop for p in pieces[:-1]]
    assert pieces[-1].stop == 300
    assert 3 <= len(pieces) <= 5
    for piece in pieces:
        rows = slice(piece.start, piece.stop)
        fitted = piece.slope * x[rows] + piece.intercept
        assert np.max(np.abs((y + np.sin(x) * 2.0)[rows] - fitted)) <= 5.0 + 1e-9 or piece.rows <= 2


def test_moving_average_keeps_length_and_pads_with_the_ends():
    values = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    smoothed = moving_average(values, 3)
    assert smoothed.tolist() == pytest.approx([0.0, 10 / 3, 10 / 3, 10 / 3, 0.0])
    assert moving_average(values, 1).tolist() == values.tolist()


# ---- measurements
def test_descent_classes_recover_separated_clusters():
    rng = np.random.default_rng(0)
    angles = np.concatenate([rng.normal(c, 0.05, 400) for c in (0.8, 2.0, 3.0, 4.5)])
    fit = measure.fit_descent_classes(angles, np.full(len(angles), 5000.0), 4)
    assert fit["centres_deg"] == pytest.approx([0.8, 2.0, 3.0, 4.5], abs=0.05)
    assert fit["edges_deg"][0] == measure.DESCENT_FLOOR_DEG and fit["edges_deg"][-1] == measure.DESCENT_CEILING_DEG


def test_the_corridor_contains_every_bins_p99():
    rng = np.random.default_rng(1)
    distance = rng.uniform(0.0, 29000.0, 200000)
    offset = np.abs(rng.normal(0.0, 5.0 + distance * 0.002))
    fit = measure.fit_corridor(offset, distance)
    for item in fit["bins"]:
        middle = (item["from_m"] + item["to_m"]) / 2
        assert fit["half_width_m"] + fit["slope"] * middle >= item["p99_m"] - 1e-9


def test_rounding_goes_outward_and_keeps_exact_steps():
    assert measure.round_up(4.1, 0.5) == 4.5 and measure.round_up(4.5, 0.5) == 4.5
    assert measure.round_down(6.9, 1.0) == 6.0 and measure.round_down(7.0, 1.0) == 7.0
    assert measure.round_up(1.62, 0.1) == 1.7 and measure.round_up(0.44, 0.05) == 0.45


def test_the_measurements_read_only_the_admitted_rows(geometry):
    one = spec()
    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0))), (10, 0.0, 70.0, 0.0)]
    flight = admit(instruction_flight(*fly_legs(legs, 270.0, 1110.0, 1400.0, 0.0)), geometry, measure.provisional_spec())
    assert flight.cut_at_crossing and flight.relative.before_threshold_m.min() >= 0.0
    arrays = measure.measure_flight(flight, measure.provisional_spec())
    assert {f"heading_wander_deg_band{h:g}" for h in measure.FREE_HOLD_HALF_RANGES_DEG} <= set(arrays)
    assert len(arrays["turn_mean_rate_deg_s"]) == 2                   # onto the base and onto the final
    assert arrays["move_angle_deg"] == pytest.approx([3.0], abs=0.05)
    first = measure.aligned_final_row(flight, 2.0)
    finals, rows = measure.measure_final(flight, one, 2.0, {5.0: 4.5})
    assert len(finals["aligned_offset_m"]) == flight.signals.n_rows - first
    # each 90° turn at 6° a row says 15 words on the 5° grid (the grid skips a cell every 30°)
    assert rows["5"][0] == 30.0


def test_every_fitted_angle_comes_with_rounder_candidates_and_the_fit_each_leaves():
    """D15: the fitted descent centres / inner edges and the climb centre, each rounded to 0.5°, 0.25° and 0.1°, with the
    end-of-piece height error each leaves; the fitted row is `fit_descent_classes`'s own error."""
    rng = np.random.default_rng(0)
    angle = np.concatenate((rng.normal(3.0, 0.2, 300), rng.normal(1.0, 0.2, 300), rng.normal(-1.5, 0.3, 40)))
    length = rng.uniform(500.0, 3000.0, len(angle))
    fit = measure.fit_descent_classes(angle, length, 4)
    rows = measure.rounding_candidates(angle, length, fit["centres_deg"], fit["edges_deg"], 1.47)
    assert set(rows) == {"fitted", "0.5", "0.25", "0.1"}
    assert rows["fitted"]["descent_end_height_error_m"] == pytest.approx(fit["end_height_error_m"])
    assert rows["0.5"]["climb_centre_deg"] == 1.5 and rows["0.1"]["climb_centre_deg"] == 1.5
    assert all(c * 4 == round(c * 4) for c in rows["0.25"]["descent_centres_deg"])
    assert rows["0.5"]["descent_edges_deg"][0] == measure.DESCENT_FLOOR_DEG
    climbs = measure.climb_distribution(angle, length)
    assert climbs["pieces"] == 40 and 1.0 < climbs["length_weighted_deg"]["p50"] < 2.0


def test_the_level_rounding_error_is_read_under_the_grid_and_a_uniform_30_m_grid():
    words = Words(spec())
    reading = measure.level_rounding(np.array([630.0, 905.0, 1320.0, 3375.0]), words)
    assert reading["levels"] == 4 and reading["grid"]["max"] == pytest.approx(225.0)
    assert reading["uniform_30_m"]["max"] == pytest.approx(15.0)


# ---- artefact
def _signals(identifier: str, rows: int, split: str = "train") -> FlightSignals:
    t = np.arange(rows) * 2.0
    return FlightSignals(identifier, "KXXX", "09", "A320", "2026-06-01T11:00:00Z", landing_on(split), t, -t * 70.0,
                         np.zeros(rows), np.full(rows, 500.0), np.full(rows, 90.0), np.full(rows, 70.0), np.zeros(rows))


def test_the_artefact_round_trips_and_refuses_overwrites_and_other_specs(tmp_path, geometry):
    flights = {"train": [_signals("KXXX:a", 5), _signals("KXXX:b", 7)], "val": [_signals("KXXX:c", 4, "val")]}
    write_signals(tmp_path, flights, {"note": "test"}, fixture_days())
    back = load_signals(tmp_path, "train")
    assert [f.dataset_id for f in back] == ["KXXX:a", "KXXX:b"]
    assert back[1].e_m.tolist() == flights["train"][1].e_m.tolist()
    assert (back[1].entry_time_utc, back[1].landing_time_utc) == ("2026-06-01T11:00:00Z", landing_on("train"))
    assert load_day_split(tmp_path) == fixture_days()
    write_candidates(tmp_path, {"KXXX": geometry})
    assert load_candidates(tmp_path)["KXXX"] == geometry
    # a candidates file without this schema's name (the runway ends came with it) is refused by name
    (tmp_path / "unnamed").mkdir()
    (tmp_path / "unnamed" / "candidates.json").write_text(json.dumps({"airports": {"KXXX": geometry.to_dict()}}),
                                                          encoding="utf-8")
    with pytest.raises(ValueError, match=f"is not a {CANDIDATES_SCHEMA} file"):
        load_candidates(tmp_path / "unnamed")
    one = spec()
    source = {"labeller_code_sha256": "c" * 64, "git": {"head": "test", "dirty": False}}
    write_spec(tmp_path, one, {"n": 1}, source)
    assert load_spec(tmp_path) == one
    with pytest.raises(FileExistsError):
        write_spec(tmp_path, one, {"n": 1}, source)
    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    reading = read_flight(instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0)), geometry, one)
    reading.go_around_rows, reading.runway_again_rows = [5], [9]          # the go-around columns, round-tripped
    write_sentences(tmp_path, "train", one, [reading, reading], [1, 0], "c" * 64)
    with pytest.raises(ValueError, match="read with spec"):
        load_sentences(tmp_path, "train", spec(heading_tolerance_deg=5.0))
    sentences = load_sentences(tmp_path, "train", one)
    assert sentences["words"].tolist() == reading.words.tolist() * 2 and sentences["signal_index"].tolist() == [1, 0]
    assert sentences["words"].shape[1] == 5 and len(sentences["instruction_row"]) == 2 * len(reading.instructions)
    assert sentences["go_around_offsets"].tolist() == [0, 1, 2] and sentences["runway_again_row"].tolist() == [9, 9]
    assert "join_row" not in sentences and str(sentences["labeller_code_sha256"]) == "c" * 64
    # a sentence file of another schema (an old artefact) is refused by its name
    with np.load(tmp_path / "sentences_train.npz") as arrays:
        old = {name: arrays[name] for name in arrays.files}
    (tmp_path / "v6").mkdir()
    np.savez_compressed(tmp_path / "v6" / "sentences_train.npz", **{**old, "schema": np.array("ts-instruction-sentences-v2")})
    with pytest.raises(ValueError, match=f"is not a {SENTENCES_SCHEMA} file"):
        load_sentences(tmp_path / "v6", "train", one)
    # a spec file of another schema is refused by name
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "spec.json").write_text(json.dumps({"schema": f"{SPEC_SCHEMA}-other", "sha256": one.sha256,
                                                            "spec": one.to_dict()}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"is not a {SPEC_SCHEMA} file"):
        load_spec(tmp_path / "old")
    assert SPEC_SCHEMA == "ts-instruction-spec-v5" and one.reading_rule == "instruction-v4"


def test_a_spec_is_kept_for_new_rows_byte_for_byte(tmp_path):
    """`keep_spec`: new rows under the same vocabulary keep the spec (the instruction_spec runner's ``--spec-from``)."""
    config = {"dt_s": 2.0, "seq_len": 60}
    for name in ("old", "new", "other_rows", "foreign"):
        (tmp_path / name).mkdir()
        write_signals(tmp_path / name, {"train": [_signals("KXXX:a", 5)]},
                      {"config": {**config, "dt_s": 1.0} if name == "other_rows" else config}, fixture_days())
    one = spec()
    source = {"labeller_code_sha256": "c" * 64, "git": {"head": "measured", "dirty": False}}
    write_spec(tmp_path / "old", one, {"n": 1}, source)
    assert keep_spec(tmp_path / "old", tmp_path / "new", {"spec_from": "old", "git": {"head": "kept"}}) == one
    for name in ("spec.json", "measurements.json"):
        assert (tmp_path / "new" / name).read_bytes() == (tmp_path / "old" / name).read_bytes()
    assert load_spec(tmp_path / "new") == one
    record = json.loads((tmp_path / "new" / "spec_from.json").read_text(encoding="utf-8"))
    assert (record["spec_from"], record["spec_sha256"], record["git"]["head"]) == ("old", one.sha256, "kept")
    with pytest.raises(FileExistsError):
        keep_spec(tmp_path / "old", tmp_path / "new", {})
    # signals built under another configuration are refused
    with pytest.raises(ValueError, match="built under"):
        keep_spec(tmp_path / "old", tmp_path / "other_rows", {})
    assert not (tmp_path / "other_rows" / "spec.json").exists()


def test_the_artefact_holds_development_days_only_and_each_flight_on_its_own_split(tmp_path):
    # a flight landing on a sealed test day is refused on the way in, by name
    with pytest.raises(SealedDay, match="sealed test day"):
        write_signals(tmp_path / "a", {"train": [_signals("KXXX:a", 5, "test")]}, {"note": "test"}, fixture_days())
    # so is a flight filed under a split its day is not dealt to
    with pytest.raises(ValueError, match="lands on a val day, not a train day"):
        write_signals(tmp_path / "b", {"train": [_signals("KXXX:a", 5, "val")]}, {"note": "test"}, fixture_days())
    # and on the way out: a file edited to hold a test-day flight is refused when read
    (tmp_path / "c").mkdir()
    write_signals(tmp_path / "c", {"train": [_signals("KXXX:a", 5)]}, {"note": "test"}, fixture_days())
    record = json.loads((tmp_path / "c" / "signals.json").read_text(encoding="utf-8"))
    record["splits"]["train"]["flights"][0]["landing_time_utc"] = landing_on("test")
    (tmp_path / "c" / "signals.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SealedDay):
        load_signals(tmp_path / "c", "train")
    with pytest.raises(ValueError, match="holds the splits"):
        load_signals(tmp_path / "c", "test")
    # every split is checked before the first file is written: nothing is left half-written
    (tmp_path / "d").mkdir()
    with pytest.raises(ValueError, match="holds the splits"):
        write_signals(tmp_path / "d", {"train": [_signals("KXXX:a", 5)], "test": []}, {"note": "test"}, fixture_days())
    with pytest.raises(ValueError, match="not a train day"):
        write_signals(tmp_path / "d", {"val": [_signals("KXXX:v", 5, "val")], "train": [_signals("KXXX:a", 5, "val")]},
                      {"note": "test"}, fixture_days())
    assert not any((tmp_path / "d").iterdir())


# ---- the landing, as the harvest and the evaluator judge it
def test_a_landing_is_a_low_crossing_near_the_centreline():
    from ts_transformer.instructions.airport import RunwayRelative
    from ts_transformer.instructions.labeller.read import landing_passages

    def relative(right, height, before=(600.0, 100.0, -50.0, -400.0)):
        return RunwayRelative(before_threshold_m=np.array(before), right_of_course_m=np.full(4, right),
                              track_minus_course_deg=np.zeros(4),
                              height_above_threshold_m=np.array([height + 30.0, height + 5.0, height, height - 10.0]))
    one = spec()
    assert landing_passages(relative(20.0, 20.0), 1000.0, one) == [2]
    assert landing_passages(relative(20.0, 300.0), 1000.0, one) == []          # over the airfield, too high
    assert landing_passages(relative(20.0, -150.0), 1000.0, one) == []         # as far below: the harvest's |height|
    assert landing_passages(relative(1500.0, 20.0), 1000.0, one) == []         # abeam, too far off the line
    assert landing_passages(relative(150.0, 20.0), 106.7, one) == []           # nearer the parallel runway
    # a row exactly on the plane is the first one past it, as the harvest brackets (ahead < 0 <= past)
    assert landing_passages(relative(20.0, 20.0, before=(600.0, 100.0, 0.0, -400.0)), 1000.0, one) == [2]


def test_the_landing_constants_are_the_harvest_s_and_the_parallel_limit_mirrors_its_rule():
    from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
    from final_approach.assign import LandingScreen
    from trajectory_data_process.harvest.airports import load_airport
    from trajectory_data_process.harvest.threshold_event import MAX_PARALLEL_COURSE_DELTA_DEG, _runway_bracket_cross_limit
    from ts_transformer.instructions.airport import landing_cross_limit_m

    one = spec()
    assert one.landing_cross_limit_m == LandingScreen().threshold_radius_m
    assert one.landing_max_height_m == LandingScreen().max_crossing_height_m
    assert one.parallel_course_delta_deg == MAX_PARALLEL_COURSE_DELTA_DEG

    def targets(runways):
        return {r.ident: {"lat": r.lat, "lon": r.lon, "elevation_msl_m": 0.0, "course_deg": r.course_deg}
                for r in runways}
    for code in ("KMSY", "KRDU", "KSJC", "KSMF", "KSTL"):
        runways = tuple(load_airport(code, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways)
        geometry = airport_geometry(code, targets(runways), runways)
        for runway in runways:
            mine = landing_cross_limit_m(geometry, geometry.candidate_index(runway.ident), one.landing_cross_limit_m,
                                         one.parallel_course_delta_deg)
            theirs = _runway_bracket_cross_limit(runway, runways, fallback_m=LandingScreen().threshold_radius_m)
            assert mine == pytest.approx(theirs, abs=1.0), (code, runway.ident)
    # a runway whose parallel partner is not a candidate (no published target) is still capped by it
    runways = tuple(load_airport("KSJC", config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways)
    alone = airport_geometry("KSJC", targets(r for r in runways if r.ident == "30L"), runways)
    assert landing_cross_limit_m(alone, 0, one.landing_cross_limit_m, one.parallel_course_delta_deg) == pytest.approx(
        _runway_bracket_cross_limit(next(r for r in runways if r.ident == "30L"), runways, fallback_m=1000.0), abs=1.0)
