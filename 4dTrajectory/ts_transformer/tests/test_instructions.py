"""The instruction words (two-tier v3 stage B, §5.2.1): bins, segmentation, the plateau-after-the-
manoeuvre rule, the intercept, the fixed-position sentence, the hashed vocabulary artefact — the
mechanics are pinned on synthetic and hand-built tracks, never a number from data."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ts_transformer.config import CONTROL_RECIPE_SIMPLE_V3, PREDICTION_CONTROL, TSConfig, recipe_settings
from ts_transformer.data.dataset import build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.manoeuvre import instructions as ins
from ts_transformer.tests.support import AIRPORT, RUNWAY


def _config() -> TSConfig:
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(prediction_output=PREDICTION_CONTROL, control_horizon_s=20.0, n_segments=2, control_imitation_loss_weight=0.0,
                         final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=8, d_model=16, n_heads=4, d_ff=32,
                         e_layers=1, dropout=0.0, device="cpu", epochs=1, patience=1))
    return TSConfig(**settings)


@pytest.fixture(scope="module")
def flights():
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=4, seed=5), _config(), airport=AIRPORT)
    return series


def test_the_bins_are_the_vocabulary_and_round_trip_through_the_artefact(tmp_path):
    v = ins.Vocabulary()
    assert v.words == {"heading": 36, "altitude": 11, "speed": 21, "intercept": 3}
    assert v.heading_bin(0.0) == 0 and v.heading_bin(-4.9) == 0 and v.heading_bin(5.1) == 1 and v.heading_bin(180.0) == 18
    assert v.heading_bin(-90.0) == 27 and v.heading_bin(365.0) == 1 and v.heading_centre_deg(27) == pytest.approx(-90.0)
    assert v.altitude_bin(0.0) == (0, False) and v.altitude_bin(1100.0 * ins.FT) == (1, False)
    assert v.altitude_bin(-30.0) == (0, False) and v.altitude_bin(9000.0 * ins.FT) == (9, False) and v.altitude_bin(11000.0 * ins.FT) == (10, True)
    assert v.speed_bin(120.0 * ins.KT) == (0, False) and v.speed_bin(174.0 * ins.KT) == (5, False) and v.speed_bin(300.0 * ins.KT) == (18, False)
    assert v.speed_bin(330.0 * ins.KT) == (20, True)
    assert v.speed_bin(100.0 * ins.KT) == (0, True) and v.speed_centre_mps(5) == pytest.approx(170.0 * ins.KT)
    assert v.intercept_bin(-25.0) == 0 and v.intercept_bin(40.0) == 1 and v.intercept_bin(70.0) == 2
    path = ins.write_vocabulary(tmp_path, v, cohort_identity={"name": "x"}, counts={}, source={})
    loaded, payload = ins.load_vocabulary(path)
    assert loaded == v and payload["sha256"] == v.sha256 and payload["words"] == v.words
    with pytest.raises(FileExistsError):
        ins.write_vocabulary(tmp_path, v, cohort_identity={"name": "x"}, counts={}, source={})
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["spec"]["heading_bin_deg"] = 15.0
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="sha"):
        ins.load_vocabulary(tmp_path / "bad.json")
    with pytest.raises(ValueError, match="divide"):
        ins.Vocabulary(heading_bin_deg=7.0)
    with pytest.raises(ValueError, match="divide"):
        ins.Vocabulary(speed_max_mps=325.0 * ins.KT)
    assert ins.FT == 0.3048 and ins.KT == pytest.approx(1852.0 / 3600.0)                 # geokit's exact definitions
    assert ins.Vocabulary(token_step_s=5.0).sha256 != v.sha256          # τ is part of the identity


def test_runs_and_smoothing():
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    assert ins.runs_where(mask, 2) == [(1, 3), (4, 8)] and ins.runs_where(mask, 3) == [(4, 8)] and ins.runs_where(mask, 1)[-1] == (9, 10)
    signal = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    assert ins.smooth(signal, 1).tolist() == signal.tolist()
    smoothed = ins.smooth(signal, 3)
    assert smoothed.shape == signal.shape and smoothed[2] == pytest.approx(10.0 / 3) and smoothed[0] == pytest.approx(0.0)


def test_a_manoeuvre_that_lands_on_the_word_in_force_is_absorbed_not_an_instruction():
    """Two deceleration runs whose plateaus fall in one bin read as ONE speed instruction: the
    second run continues the first (a split deceleration is not two calls) and is recorded as
    absorbed, with what it changed."""
    v = ins.Vocabulary()
    times = np.arange(60, dtype=float)
    speed = np.concatenate((np.full(10, 90.0), np.full(20, 83.0), np.full(30, 81.0)))       # 175 → 161 → 157 kt
    words, absorbed = ins._manoeuvre_words("speed", times, speed, [(8, 12), (28, 32)], lambda x: (v.speed_bin(x)[0], x, False))
    assert v.speed_bin(83.0)[0] == v.speed_bin(81.0)[0] != v.speed_bin(90.0)[0]          # one bin (161 and 157 kt)
    assert [(w.word, w.issued_s) for w in words] == [(v.speed_bin(90.0)[0], 0.0), (v.speed_bin(83.0)[0], 8.0)]
    assert words[1].target == 83.0 and words[1].settled_s == 12.0                      # the FIRST plateau's target and settle
    assert absorbed == [ins.Absorbed("speed", 28.0, 32.0, words[1].word, 81.0 - 83.0)]
    assert ins.min_rows(10.0, 2.0) == 6 and ins.min_rows(4.0, 2.0) == 3                  # k rows span (k − 1) · dt ≥ the seconds
    with pytest.raises(ValueError, match="plateau needs rows"):
        ins._plateau_value(speed, 5, 5)


def _track(item, t: np.ndarray, relative_deg: np.ndarray, height_m: np.ndarray, speed_mps: np.ndarray):
    """A hand-built series from profiles against time: the ground track integrated from the
    relative course, starting 20 km before the threshold and 3 km right of the course (so the
    established rule is off unless the track captures the course)."""
    from dataclasses import replace
    course = float(item.scenario.target.psi)
    psi = course + np.radians(relative_deg)
    ve, vn = speed_mps * np.cos(psi), speed_mps * np.sin(psi)
    dt = np.diff(t, prepend=t[0])
    e = item.target_chart[0] - 20_000.0 * math.cos(course) + 3_000.0 * math.sin(course) + np.cumsum(ve * dt)
    n = item.target_chart[1] - 20_000.0 * math.sin(course) - 3_000.0 * math.cos(course) + np.cumsum(vn * dt)
    u = item.target_chart[2] + height_m
    values = np.column_stack((e, n, u, ve, vn, np.gradient(u, t)))
    return replace(item, times=t, values=values, supervision_times=t, supervision_values=values,
                   supervision_weights=np.full(values.shape, 1.0 / values.shape[1]))


def test_a_turn_is_read_by_its_sweep_and_its_declared_seconds(flights):
    """A 12° correction over 8 s is a heading instruction (issued at the turn's start, target the
    new course); a 6° wobble over 4 s is not (its sweep is under one bin) — the thresholds bind in
    seconds and degrees, not in rows."""
    v = ins.Vocabulary()
    item = flights[0]
    t = np.arange(0.0, 120.0, 2.0)
    level, speed = np.full(len(t), 800.0), np.full(len(t), 80.0)
    twelve = np.interp(t, [0.0, 40.0, 48.0, 120.0], [0.0, 0.0, 12.0, 12.0])
    reading = ins.read_instructions(_track(item, t, twelve, level, speed), v)
    headings = [i for i in reading.instructions if i.kind == "heading"]
    assert [i.word for i in headings] == [0, v.heading_bin(12.0)] and abs(headings[1].issued_s - 40.0) <= 4.0
    assert headings[1].settled_s is not None and abs(headings[1].settled_s - 48.0) <= 6.0 and headings[1].target == pytest.approx(12.0, abs=1.0)
    six = np.interp(t, [0.0, 40.0, 44.0, 120.0], [0.0, 0.0, 6.0, 6.0])
    reading = ins.read_instructions(_track(item, t, six, level, speed), v)
    assert [i.kind for i in reading.instructions] == ["heading", "altitude", "speed"] and not reading.absorbed


def test_a_level_off_is_the_descent_s_target_and_the_last_descent_targets_the_end(flights):
    """5 000 ft, descend to a 3 000 ft level-off held 40 s, descend again to the end: three altitude
    words — the start's, "descend to 3 000" issued at the first descent's start and settled at the
    level-off, and the final descent's word (the end value, never settled)."""
    v = ins.Vocabulary()
    item = flights[0]
    t = np.arange(0.0, 300.0, 2.0)
    height = np.interp(t, [0.0, 20.0, 142.0, 182.0, 298.0], [5000 * ins.FT, 5000 * ins.FT, 3000 * ins.FT, 3000 * ins.FT, 1100 * ins.FT])
    reading = ins.read_instructions(_track(item, t, np.zeros(len(t)), height, np.full(len(t), 80.0)), v)
    altitudes = [i for i in reading.instructions if i.kind == "altitude"]
    assert [i.word for i in altitudes] == [5, 3, 1] and altitudes[0].issued_s == 0.0
    assert abs(altitudes[1].issued_s - 20.0) <= 6.0 and abs(altitudes[1].settled_s - 142.0) <= 8.0
    assert altitudes[1].target == pytest.approx(3000 * ins.FT, abs=30.0) and not altitudes[1].clamped
    assert abs(altitudes[2].issued_s - 182.0) <= 6.0 and altitudes[2].settled_s is None


def test_a_clamped_target_reaches_the_instruction_and_an_orbit_is_absorbed(flights):
    """A record starting at 11 000 ft carries a clamped altitude word; a 360° orbit back onto the
    same heading is not a heading instruction but is recorded as absorbed with its sweep."""
    v = ins.Vocabulary()
    item = flights[0]
    t = np.arange(0.0, 200.0, 2.0)
    high = np.full(len(t), 11_000 * ins.FT)
    reading = ins.read_instructions(_track(item, t, np.zeros(len(t)), high, np.full(len(t), 80.0)), v)
    altitude = next(i for i in reading.instructions if i.kind == "altitude")
    assert altitude.word == v.altitude_words - 1 and altitude.clamped
    orbit = np.interp(t, [0.0, 30.0, 150.0, 200.0], [0.0, 0.0, 360.0, 360.0])
    reading = ins.read_instructions(_track(item, t, orbit, np.full(len(t), 800.0), np.full(len(t), 80.0)), v)
    assert [i.word for i in reading.instructions if i.kind == "heading"] == [0]
    assert len(reading.absorbed) == 1 and reading.absorbed[0].kind == "heading" and abs(reading.absorbed[0].change) >= 180.0
    assert (reading.words[:, 0] == 0).all() and reading.to_dict()["absorbed"][0]["kind"] == "heading"


def test_the_course_frame_refuses_a_row_with_no_course(flights):
    from dataclasses import replace
    item = flights[0]
    values = np.array(item.values, dtype=np.float64)
    values[3, 3:5] = 0.0
    with pytest.raises(ValueError, match="m/s"):
        ins.course_frame(replace(item, values=values))


def test_the_established_rule_is_in_the_spec_but_not_settable():
    v = ins.Vocabulary()
    assert v.to_dict()["established_cross_track_m"] == ins.ESTABLISHED_CROSS_TRACK_M
    with pytest.raises(ValueError, match="package's one"):
        ins.Vocabulary(established_cross_track_m=1000.0)
    assert ins.Vocabulary(turn_min_sweep_deg=15.0).sha256 != v.sha256


def test_the_synthetic_arrival_reads_as_one_intercept_turn_a_descent_to_the_threshold_and_a_deceleration(flights):
    """The synthetic track flies straight to the FAF, turns onto the course and descends on the
    glidepath while slowing: the heading words are the entry course then the course (bin 0) with
    an intercept, the last altitude word is the threshold's bin 0, the speed words end at the
    threshold speed's bin, every position carries the three words in force."""
    v = ins.Vocabulary()
    for item in flights:
        reading = ins.read_instructions(item, v)
        headings = [i for i in reading.instructions if i.kind == "heading"]
        assert headings[0].issued_s == item.times[0] and headings[0].word == v.heading_bin(headings[0].target)
        intercept = [i for i in reading.instructions if i.kind == "intercept"]
        if abs(headings[0].target) >= 15.0:        # a corner the 1°/s rule sees over the 10 s smoothing window
            assert len(headings) == 2 and headings[-1].word == 0                          # settles on the course
            assert len(intercept) == 1 and intercept[0].target == pytest.approx(abs(headings[0].target), abs=1.0)
            assert intercept[0].word == v.intercept_bin(intercept[0].target) and intercept[0].issued_s == headings[1].issued_s
        else:                                       # too small a corner to be a turn: one word, the entry course's bin
            assert len(headings) == 1 and not intercept and abs(headings[0].target) < 15.0
        altitudes = [i for i in reading.instructions if i.kind == "altitude"]
        assert altitudes[-1].word == 0 and altitudes[-1].settled_s is None                # descending to the end
        assert all(a.kind in ins.MANDATORY_KINDS for a in reading.absorbed)
        speeds = [i for i in reading.instructions if i.kind == "speed"]
        assert speeds[-1].word == v.speed_bin(speeds[-1].target)[0]                     # the last word is the final speed's bin
        assert abs(speeds[-1].target - float(ins.course_frame(item)["ground_speed_mps"][-1])) < 10.0
        assert reading.positions_s[0] == item.times[0] and np.all(np.diff(reading.positions_s) == v.token_step_s)
        assert reading.words.shape == (len(reading.positions_s), 4) and (reading.words[:, :3] >= 0).all()
        # the words in force: the intercept is absent before its turn, present from it on
        if intercept:
            first_intercept = np.searchsorted(reading.positions_s, intercept[0].issued_s)
            assert (reading.words[:first_intercept, 3] == ins.NO_INTERCEPT).all() and (reading.words[first_intercept:, 3] == intercept[0].word).all()
        assert reading.to_dict()["instructions"][0]["kind"] in ins.INSTRUCTION_KINDS
        assert not reading.established_from_start


def test_a_track_that_is_already_on_the_course_and_level_carries_only_its_starting_words(flights):
    """A hand-built series: straight along the course at constant height and speed — one word of
    each kind issued at t = 0, no intercept (nothing to capture), no manoeuvre."""
    from dataclasses import replace
    item = flights[0]
    n = 40
    t = np.arange(n) * 2.0
    course = float(item.scenario.target.psi)
    speed = 150.0 * ins.KT
    to_go = 20_000.0 - speed * t
    e = item.target_chart[0] - to_go * math.cos(course)
    nn = item.target_chart[1] - to_go * math.sin(course)
    u = item.target_chart[2] + 1500.0 * ins.FT
    values = np.column_stack((e, nn, np.full(n, u), np.full(n, speed * math.cos(course)), np.full(n, speed * math.sin(course)), np.zeros(n)))
    straight = replace(item, times=t, values=values, supervision_times=t, supervision_values=values,
                       supervision_weights=np.full(values.shape, 1.0 / values.shape[1]))
    reading = ins.read_instructions(straight, ins.Vocabulary())
    assert [i.kind for i in reading.instructions] == ["heading", "altitude", "speed"] and all(i.issued_s == 0.0 for i in reading.instructions)
    words = {i.kind: i.word for i in reading.instructions}
    assert words == {"heading": 0, "altitude": 2, "speed": 3} and reading.established_from_start
    assert (reading.words[:, 3] == ins.NO_INTERCEPT).all() and len(reading.positions_s) == 8      # 78 s at τ = 10 s


def test_the_words_in_force_at_a_time_and_the_executor_s_view_of_them():
    """`words_at` reads the sentence, not the issue times (an instruction issued at 25 s is in
    force from the 30 s position); the last words stay in force past the last position; the
    conditioning carries the bin centres (cos / sin heading, fractions of the ceilings, a graded
    intercept flag) and the segment positions are Δ / τ of them."""
    v = ins.Vocabulary()
    positions = np.array([0.0, 10.0, 20.0, 30.0])
    words = np.array([[18, 3, 4, ins.NO_INTERCEPT], [18, 3, 4, ins.NO_INTERCEPT], [18, 3, 4, ins.NO_INTERCEPT], [27, 3, 4, 1]])
    reading = ins.Reading("d", "f", (), positions, words, False, 35.0)
    assert reading.words_at(np.array([0.0, 9.9, 27.0, 30.0, 95.0])).tolist() == [words[0].tolist(), words[0].tolist(), words[2].tolist(), words[3].tolist(), words[3].tolist()]
    with pytest.raises(ValueError, match="before the record's start"):
        reading.words_at(np.array([-1.0]))
    assert ins.segment_positions_s(40.0, 20.0, 10.0).tolist() == [40.0, 50.0]
    assert ins.segment_positions_s(0.0, 10.0, 10.0).tolist() == [0.0]
    with pytest.raises(ValueError, match="whole number"):
        ins.segment_positions_s(0.0, 25.0, 10.0)
    cond = v.conditioning(reading.words_at(ins.segment_positions_s(20.0, 20.0, 10.0)))
    assert cond.shape == (2, v.CONDITIONING_WIDTH) and cond.dtype == np.float32
    assert cond[0, 0] == pytest.approx(-1.0) and abs(cond[0, 1]) < 1e-6 and cond[0, 4] == 0.0          # downwind, no intercept
    assert cond[1, 0] == pytest.approx(0.0, abs=1e-6) and cond[1, 1] == pytest.approx(-1.0) and cond[1, 4] == pytest.approx(2 / 3)  # −90° base, bin 1
    assert cond[0, 2] == pytest.approx(3 * v.altitude_bin_m / v.altitude_max_m) and cond[0, 3] == pytest.approx(v.speed_centre_mps(4) / v.speed_max_mps)
    with pytest.raises(ValueError, match="words are"):
        v.conditioning(np.zeros((2, 3)))


def test_the_sentence_holds_the_latest_word_of_each_kind_and_refuses_a_kind_with_no_word_at_the_start():
    words = [ins.Instruction("heading", 18, 180.0, 0.0, 0.0), ins.Instruction("altitude", 3, 900.0, 0.0, 0.0),
             ins.Instruction("speed", 4, 80.0, 0.0, 0.0), ins.Instruction("heading", 27, -90.0, 25.0, 40.0),
             ins.Instruction("heading", 0, 0.0, 61.0, 90.0), ins.Instruction("intercept", 1, 40.0, 61.0, 90.0)]
    positions, table = ins.sentence(words, 0.0, 100.0, 10.0)
    assert positions.tolist() == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert table[:, 0].tolist() == [18, 18, 18, 27, 27, 27, 27, 0, 0, 0, 0]              # issued at 25 s → in force from the 30 s position
    assert table[:, 3].tolist() == [-1] * 7 + [1] * 4 and (table[:, 1] == 3).all() and (table[:, 2] == 4).all()
    with pytest.raises(ValueError, match="no word in force"):
        ins.sentence(words[:2], 0.0, 100.0, 10.0)
