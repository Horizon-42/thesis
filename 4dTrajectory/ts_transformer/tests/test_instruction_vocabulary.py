"""The instruction vocabulary's parts: the spec, the words, the candidate runways, the envelopes,
the piecewise fit, the measurements and the artefact (design: docs/2026-09-23_instruction_vocabulary_design.zh.md)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from aerodynamic_model.common import GeodeticState
from ts_transformer.data.channels import channels_from_states

from ts_transformer.instructions import envelope, measure
from ts_transformer.instructions.airport import AirportGeometry, airport_geometry, relative_to_runway
from ts_transformer.instructions.artefact import (
    labeller_source_sha256, load_candidates, load_sentences, load_signals, load_spec, require_current_labeller,
    write_candidates, write_sentences, write_signals, write_spec,
)
from ts_transformer.instructions.labeller.read import admit, read_flight
from ts_transformer.instructions.piecewise import fit_pieces, moving_average
from ts_transformer.instructions.signals import FlightSignals, signals_from_series
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import (
    ANGLE_LEVEL, Words, compass_from_math_rad, math_rad_from_compass, wrap180,
)
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec


@pytest.fixture
def geometry() -> AirportGeometry:
    return instruction_airport()


spec = instruction_spec


# ---- spec
def test_the_sha_is_stable_and_moves_with_any_field():
    a, b = spec(), spec()
    assert a.sha256 == b.sha256
    assert spec(altitude_step_m=30.0).sha256 == a.sha256
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
    ({"heading_split_part_deg": 145.0}, "split part plus its lead"),
    ({"altitude_tolerance_m": 10.0}, "half an altitude step"),
    ({"heading_step_deg": 7.0}, "does not divide 360"),
    ({"descent_angle_edges_deg": [-0.5, 3.0, 2.6, 3.7, 10.0]}, "edges must increase"),
    ({"descent_angle_centres_deg": [0.8, 2.1, 3.0, 11.0]}, "inside its class"),
])
def test_the_spec_refuses_inconsistent_values(changes, message):
    with pytest.raises(ValueError, match=message):
        spec(**changes)


# ---- words
def test_heading_words_wrap_the_circle():
    words = Words(spec())
    assert words.n_heading == 72
    assert words.heading_index(357.6) == 0
    assert words.heading_index(-2.4) == 0
    assert words.heading_deg(words.heading_index(222.4)) == 220.0


def test_altitude_words_refuse_out_of_range_and_keep_land_apart():
    words = Words(spec())
    assert words.altitude_m(words.altitude_index(914.0)) == 900.0
    assert words.altitude_m(words.altitude_land) is None
    with pytest.raises(ValueError):
        words.altitude_index(6000.0)
    with pytest.raises(ValueError):
        words.altitude_index(-40.0)


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
    geometry = airport_geometry("KRDU", {"23R": {"lat": 35.8937988, "lon": -78.7779999, "elevation_msl_m": 124.7,
                                                 "course_deg": 225.3}})
    (candidate,) = geometry.candidates
    assert candidate.ident == "23R"
    assert candidate.course_deg == pytest.approx(225.3)
    assert candidate.length_m == pytest.approx(3048.0)
    assert (candidate.threshold_e_m, candidate.threshold_n_m) == pytest.approx((843.0, 1683.0), abs=15.0)
    assert AirportGeometry.from_dict(geometry.to_dict()) == geometry
    with pytest.raises(KeyError):
        geometry.candidate_index("05L")


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
    scenario = SimpleNamespace(source={"runway": "09"}, target=target, initial=samples[0][1],
                               aircraft=SimpleNamespace(code="A320"))
    return SimpleNamespace(airport="KXXX", dataset_id="KXXX:turn", scenario=scenario, times=times, values=values,
                           frame=frame), samples


def test_signals_from_a_series_are_an_unwrapped_compass_track_in_the_airport_frame(geometry):
    series, samples = _series(geometry, float(math_rad_from_compass(90.0)))
    flight = signals_from_series(series, geometry)
    assert flight.runway == "09" and flight.typecode == "A320"
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


@pytest.mark.parametrize("heading, offset, before, converges", [
    (0.0, 800.0, 12000.0, True),       # a base 92° off an off-grid course (092), from its right: within 90° + 4.5°
    (180.0, -800.0, 12000.0, True),    # the mirror base from its left, 88° off
    (270.0, 800.0, 12000.0, False),    # flying away from the course
    (90.0, -60.0, 4000.0, True),       # aligned, 60 m out where the corridor is 51 m: a track within 4.5° meets the line
    (90.0, -2000.0, 12000.0, False),   # parallel 2 km out: even 4.5° toward the line meets it beyond the threshold
    (90.0, 800.0, 12000.0, True),      # 2° toward the line from its right... within the tolerance
    (100.0, 800.0, 12000.0, False),    # 8° away from the line: no track within the tolerance comes back
])
def test_a_heading_converges_when_a_track_within_its_tolerance_meets_the_final(heading, offset, before, converges):
    assert envelope.heading_converges(heading, 92.0, offset, before, 4.5, 20.0, 0.45) is converges


def test_the_corridor_widens_with_distance():
    width = envelope.corridor_half_width_m(np.array([0.0, 10000.0, -50.0]), 20.0, 0.45)
    assert width.tolist() == pytest.approx([20.0, 20.0 + 10000.0 * math.tan(math.radians(0.45)), 20.0])
    inside = envelope.corridor(np.array([30.0, 30.0]), np.array([0.0, 3.0]), np.array([5000.0, 5000.0]), 20.0, 0.45, 2.0)
    assert inside.tolist() == [True, False]


def test_turn_and_speed_progress():
    assert envelope.turn_progress_ok(np.array([0.0, 20.0, 50.0, 88.0, 90.0]), 90.0, 4.5)
    assert not envelope.turn_progress_ok(np.array([0.0, 30.0, 20.0, 90.0]), 90.0, 4.5)
    assert not envelope.turn_progress_ok(np.array([0.0, 50.0, 100.0]), 90.0, 4.5)
    assert envelope.turn_bank_ok(90.0, 12.0, 30.0, 6.0, 33.0, 10.0)
    assert not envelope.turn_bank_ok(90.0, 4.0, 30.0, 6.0, 33.0, 10.0)      # a slow large turn
    assert envelope.turn_bank_ok(-5.0, 1.0, 3.0, 6.0, 33.0, 10.0)           # a small change of track: no lower bound
    assert not envelope.turn_bank_ok(-5.0, 12.0, 35.0, 6.0, 33.0, 10.0)     # but never beyond the highest bank
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
    assert len(arrays["turn_mean_bank_deg"]) == 2                     # onto the base and onto the final
    assert arrays["move_angle_deg"] == pytest.approx([3.0], abs=0.05)
    first = measure.aligned_final_row(flight, 2.0)
    finals, rows = measure.measure_final(flight, one, 2.0, {5.0: 4.5})
    assert len(finals["aligned_offset_m"]) == flight.signals.n_rows - first
    assert rows["5"][0] == 2.0                                        # the downwind and the base


# ---- artefact
def _signals(identifier: str, rows: int) -> FlightSignals:
    t = np.arange(rows) * 2.0
    return FlightSignals(identifier, "KXXX", "09", "A320", t, -t * 70.0, np.zeros(rows), np.full(rows, 500.0),
                         np.full(rows, 90.0), np.full(rows, 70.0), np.zeros(rows))


def test_the_artefact_round_trips_and_refuses_overwrites_and_other_specs(tmp_path, geometry):
    flights = {"train": [_signals("KXXX:a", 5), _signals("KXXX:b", 7)], "val": [_signals("KXXX:c", 4)]}
    write_signals(tmp_path, flights, {"note": "test"})
    back = load_signals(tmp_path, "train")
    assert [f.dataset_id for f in back] == ["KXXX:a", "KXXX:b"]
    assert back[1].e_m.tolist() == flights["train"][1].e_m.tolist()
    write_candidates(tmp_path, {"KXXX": geometry})
    assert load_candidates(tmp_path)["KXXX"] == geometry
    one = spec()
    source = {"labeller_source_sha256": labeller_source_sha256(), "git": {"head": "test", "dirty": False}}
    write_spec(tmp_path, one, {"n": 1}, source)
    assert load_spec(tmp_path) == one
    require_current_labeller(tmp_path)
    with pytest.raises(FileExistsError):
        write_spec(tmp_path, one, {"n": 1}, source)
    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    reading = read_flight(instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0)), geometry, one)
    write_sentences(tmp_path, "train", one, [reading], [1])
    with pytest.raises(ValueError, match="read with spec"):
        load_sentences(tmp_path, "train", spec(heading_tolerance_deg=5.0))
    sentences = load_sentences(tmp_path, "train", one)
    assert sentences["words"].tolist() == reading.words.tolist() and sentences["signal_index"].tolist() == [1]
    assert len(sentences["instruction_row"]) == len(reading.instructions)
    # a spec file of another schema is refused by name
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "spec.json").write_text(json.dumps({"schema": "ts-instruction-spec-v1", "sha256": one.sha256,
                                                            "spec": one.to_dict()}), encoding="utf-8")
    with pytest.raises(ValueError, match="is not a ts-instruction-spec-v2 file"):
        load_spec(tmp_path / "old")
    # a spec measured by another labeller: labelling refuses, and so does the sentence file
    other = tmp_path / "other"
    other.mkdir()
    write_spec(other, one, {"n": 1}, {**source, "labeller_source_sha256": "0" * 64})
    with pytest.raises(ValueError, match="measured by labeller 000000000000"):
        require_current_labeller(other)
    (other / "sentences_train.npz").write_bytes((tmp_path / "sentences_train.npz").read_bytes())
    with pytest.raises(ValueError, match="not the one that measured the spec"):
        load_sentences(other, "train", one)
