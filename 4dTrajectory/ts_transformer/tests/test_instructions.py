"""The instruction words (two-tier v3 stage B, §5.2.1): bins, segmentation, the plateau-after-the-
manoeuvre rule, the event sequence, the hashed vocabulary artefact — the
mechanics are pinned on synthetic and hand-built tracks, never a number from data."""

from __future__ import annotations

import json
from dataclasses import replace
import math

import numpy as np
import pytest

from ts_transformer.config import CONTROL_RECIPE_SIMPLE_V3, PREDICTION_CONTROL, TSConfig, recipe_settings
from ts_transformer.data.dataset import build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.manoeuvre import instructions as ins
from ts_transformer.tests.support import AIRPORT, OTHER_RUNWAY_WORD, RUNWAY, RUNWAY_WORD

#: The runway word's classes for these fixtures (D62): every synthetic flight lands on RUNWAY,
#: and a second ident is carried so the word is not degenerate — `read_instructions` must pick
#: the flight's own class out of a set, never the only one there is.
#: The speed kind's minimum change is a FRACTION of the speed IN FORCE, so `_manoeuvre_words`
#: takes a function of it. These fixtures hold around 90 m/s; the rule itself is the
#: vocabulary's, imported rather than restated, so a change to it fails here instead of being
#: copied into the assertion.
def SPEED_MIN_CHANGE(in_force: float = 90.0) -> float:
    return ins.Vocabulary().min_change("speed", in_force)
RUNWAYS = ins.RunwayVocabulary.from_idents([RUNWAY_WORD, OTHER_RUNWAY_WORD])


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
    # the SPEC counts four kinds; the runway's classes are the cohort's (see word_counts below)
    assert v.words == {"heading": 72, "vertical": 6, "speed": 16, "duration": 151, "terminal": 3}
    assert v.heading_bin(0.0) == 0 and v.heading_bin(-2.4) == 0 and v.heading_bin(2.6) == 1 and v.heading_bin(180.0) == 36
    assert v.heading_bin(-90.0) == 54 and v.heading_bin(365.0) == 1 and v.heading_centre_deg(54) == pytest.approx(-90.0)
    # the vertical word is a flight path angle, descent POSITIVE, and word 0 is the go-around climb
    assert v.vertical_centre_deg(0) == -3.0 and v.vertical_bin(-3.0) == (0, False)
    assert v.vertical_bin(0.0) == (1, False) and v.vertical_bin(3.05) == (4, False)
    assert v.vertical_bin(-5.0) == (0, True) and v.vertical_bin(9.0) == (5, True)    # past the ends, counted
    # the speed centres are fitted, not a uniform grid: a word is the NEAREST centre
    assert v.speed_bin(44.0) == (0, False) and v.speed_bin(75.0) == (4, False) and v.speed_bin(157.0) == (15, False)
    assert v.speed_bin(30.0) == (0, True) and v.speed_bin(200.0) == (15, True)
    assert v.speed_centre_mps(4) == pytest.approx(74.0)
    # the tolerances a word carries: the level mode absolute, everything else a fraction
    assert v.vertical_tolerance_deg(1) == pytest.approx(0.1)
    assert v.vertical_tolerance_deg(4) == pytest.approx(3.1 * 0.07)
    assert v.speed_tolerance(4) == pytest.approx(74.0 * 0.03)
    path = ins.write_vocabulary(tmp_path, v, runway_vocabulary=RUNWAYS, cohort_identity={"name": "x"}, counts={}, source={})
    loaded, runways, payload = ins.load_vocabulary(path)
    assert loaded == v and payload["sha256"] == v.sha256 and payload["words"] == v.words
    # the runway classes ride BESIDE the spec: in it they would give every airport its own sha
    assert runways == RUNWAYS and payload["runway_idents"] == list(RUNWAYS.idents)
    assert "runway" not in payload["spec"] and ins.Vocabulary().sha256 == v.sha256
    with pytest.raises(FileExistsError):
        ins.write_vocabulary(tmp_path, v, runway_vocabulary=RUNWAYS, cohort_identity={"name": "x"}, counts={}, source={})
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["spec"]["heading_bin_deg"] = 15.0
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="sha"):
        ins.load_vocabulary(tmp_path / "bad.json")
    assert ins.word_counts(v, RUNWAYS) == {**v.words, "runway": 2}                   # the runway count is the COHORT's
    assert tuple(ins.word_counts(v, RUNWAYS)) == ins.INSTRUCTION_KINDS
    with pytest.raises(ValueError, match="divide"):
        ins.Vocabulary(heading_bin_deg=7.0)
    with pytest.raises(ValueError, match="sorted"):
        ins.Vocabulary(speed_centres_mps=(80.0, 60.0))
    with pytest.raises(ValueError, match="level mode"):
        ins.Vocabulary(vertical_modes_deg=(1.0, 2.0, 3.0))
    assert ins.FT == 0.3048 and ins.KT == pytest.approx(1852.0 / 3600.0)                 # geokit's exact definitions
    assert ins.Vocabulary(token_step_s=5.0).sha256 != v.sha256          # τ is part of the identity
    assert ins.Vocabulary(duration_bin_s=1.0).sha256 != v.sha256        # so is the duration's bin (D71)


def test_plateaus_and_smoothing():
    """A plateau opens where `rows` rows fit in the band and holds while the signal stays within the
    tolerance of the value it opened at; a ramp between two levels is no plateau; a slow drift
    ends one plateau and opens the next."""
    signal = np.concatenate((np.full(10, 80.0), np.linspace(80.0, 70.0, 6), np.full(12, 70.0), [66.0, 62.0]))
    assert ins.plateaus(signal, 2.5, 5) == [(0, 12), (15, 28)]                          # the edges sit inside the band
    # a slow transit crossing the band is not a plateau: 1 unit per row over 5 rows drifts 4 > 1.25
    assert ins.plateaus(np.concatenate((np.full(6, 50.0), np.arange(50.0, 30.0, -1.0), np.full(6, 30.0))), 2.5, 5) == [(0, 9), (25, 32)]
    assert ins.plateaus(np.linspace(0.0, 100.0, 30), 2.5, 5) == []                      # never flat
    assert ins.plateaus(np.linspace(0.0, 10.0, 41), 2.5, 5) == [(0, 13), (13, 26), (26, 39)]   # a drift: plateaus back to back
    assert ins.min_rows(20.0, 2.0) == 11 and ins.min_rows(4.0, 2.0) == 3               # k rows span (k − 1) · dt ≥ the seconds
    signal = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    assert ins.smooth(signal, 1).tolist() == signal.tolist()
    smoothed = ins.smooth(signal, 3)
    assert smoothed.shape == signal.shape and smoothed[2] == pytest.approx(10.0 / 3) and smoothed[0] == pytest.approx(0.0)


def test_a_plateau_that_reads_as_the_word_in_force_is_absorbed_not_an_instruction():
    """Three plateaus, the last two in one bin: two speed instructions (the start's, then
    "reduce to 161 kt" issued where the first plateau ends), the third plateau absorbed with
    what it changed; a signal that never settles is one manoeuvre to its end."""
    v = ins.Vocabulary()
    times = np.arange(60, dtype=float)
    speed = np.concatenate((np.full(10, 93.0), np.full(20, 84.0), np.full(30, 89.0)))       # two centres, then the same one again
    flats = ins.plateaus(speed, v.speed_tolerance_mps, 6)
    assert flats == [(0, 10), (10, 30), (30, 60)]
    to_word = lambda x: (v.speed_bin(x)[0], x, False)                                  # noqa: E731
    words, absorbed = ins._manoeuvre_words("speed", times, speed, flats, 6, v.speed_tolerance_mps, SPEED_MIN_CHANGE, v.hold_min_s, to_word)
    assert v.speed_bin(89.0)[0] == v.speed_bin(84.0)[0] != v.speed_bin(93.0)[0]          # the same centre twice
    assert [(w.word, w.issued_s, w.settled_s) for w in words] == [(v.speed_bin(93.0)[0], 0.0, 0.0), (v.speed_bin(84.0)[0], 10.0, 10.0)]
    assert absorbed == [ins.Absorbed("speed", 30.0, 30.0, words[1].word, 89.0 - 84.0, ins.ABSORBED_SAME_WORD)]
    ramp, dropped = ins._manoeuvre_words("speed", times, np.linspace(100.0, 70.0, 60), [], 6, v.speed_tolerance_mps, SPEED_MIN_CHANGE, v.hold_min_s, to_word)
    assert [(w.word, w.issued_s, w.settled_s) for w in ramp] == [(v.speed_bin(70.0)[0], 0.0, None)] and not dropped
    # a tail shorter than a plateau is unreadable: recorded, never a word
    tail = np.concatenate((np.full(56, 90.0), [85.0, 80.0, 75.0, 70.0]))
    words, absorbed = ins._manoeuvre_words("speed", times, tail, ins.plateaus(tail, v.speed_tolerance_mps, 6), 6, v.speed_tolerance_mps, SPEED_MIN_CHANGE, v.hold_min_s, to_word)
    assert [w.word for w in words] == [v.speed_bin(90.0)[0]] and absorbed == [ins.Absorbed("speed", 56.0, 59.0, words[0].word, 70.0 - 90.0, ins.ABSORBED_SHORT_TAIL)]
    # an intermediate level that was held is where the next instruction was given: 98 → 95 (absorbed, held 20 rows)
    # → 91 issues "91" where the aircraft left 95, not where it left 98 (v6 did that and put words 100–200 s early)
    staircase = np.concatenate((np.full(20, 98.0), np.full(20, 95.4), np.full(20, 91.0)))
    assert abs(91.0 - 98.0) >= v.min_change("speed", 98.0) > abs(95.4 - 98.0) and abs(91.0 - 95.4) > v.speed_tolerance_mps
    words, absorbed = ins._manoeuvre_words("speed", times, staircase, ins.plateaus(staircase, v.speed_tolerance_mps, 6), 6,
                                           v.speed_tolerance_mps, SPEED_MIN_CHANGE, v.hold_min_s, to_word)
    assert [(w.word, w.issued_s, w.settled_s) for w in words] == [(v.speed_bin(98.0)[0], 0.0, 0.0), (v.speed_bin(91.0)[0], 40.0, 40.0)]
    assert [(a.reason, a.start_s, a.end_s) for a in absorbed] == [(ins.ABSORBED_SMALL_CHANGE, 20.0, 20.0)]
    # a long tail settling back inside the word in force's own centre: absorbed "same word"
    tail_back = np.concatenate((np.full(40, 84.0), np.full(20, 89.0)))
    words, absorbed = ins._manoeuvre_words("speed", times, tail_back, ins.plateaus(tail_back, v.speed_tolerance_mps, 6), 6,
                                           v.speed_tolerance_mps, SPEED_MIN_CHANGE, v.hold_min_s, to_word)
    assert [w.word for w in words] == [v.speed_bin(84.0)[0]] and v.speed_bin(89.0)[0] == words[0].word
    assert [(a.reason, a.end_s) for a in absorbed] == [(ins.ABSORBED_SAME_WORD, 40.0)]
    long_tail = np.concatenate((np.full(50, 90.0), np.linspace(90.0, 70.0, 10)))
    words, absorbed = ins._manoeuvre_words("speed", times, long_tail, ins.plateaus(long_tail, v.speed_tolerance_mps, 6), 6, v.speed_tolerance_mps, SPEED_MIN_CHANGE, v.hold_min_s, to_word)
    assert [(w.word, w.settled_s) for w in words] == [(v.speed_bin(90.0)[0], 0.0), (v.speed_bin(70.0)[0], None)] and not absorbed
    # a wobble under one bin is absorbed even across a bin edge (a 2.6 m/s change is under the 3 % minimum)
    wobble = np.concatenate((np.full(20, 88.2), np.full(20, 90.8), np.full(20, 88.2)))          # 172 / 180 kt: bins 5 and 6
    assert v.speed_bin(88.2)[0] != v.speed_bin(90.8)[0] and 2.6 < v.min_change("speed", 88.2)
    words, absorbed = ins._manoeuvre_words("speed", times, wobble, ins.plateaus(wobble, v.speed_tolerance_mps, 6), 6, v.speed_tolerance_mps, SPEED_MIN_CHANGE, v.hold_min_s, to_word)
    # out: the excursion is a transient under the minimum; the return reads as the word in force again
    assert [w.word for w in words] == [v.speed_bin(88.2)[0]]
    assert [a.reason for a in absorbed] == [ins.ABSORBED_SMALL_CHANGE, ins.ABSORBED_SAME_WORD]
    assert absorbed[0].start_s == 20.0 and absorbed[0].end_s == 20.0
    # the SAME 4 m/s step, held past hold_min_s: a level the aircraft flew is an instruction whatever its size
    step = np.concatenate((np.full(20, 88.2), np.full(41, 90.8)))
    words, absorbed = ins._manoeuvre_words("speed", np.arange(61, dtype=float), step, ins.plateaus(step, v.speed_tolerance_mps, 6), 6,
                                           v.speed_tolerance_mps, SPEED_MIN_CHANGE, v.hold_min_s, to_word)
    assert [w.word for w in words] == [v.speed_bin(88.2)[0], v.speed_bin(90.8)[0]] and not absorbed
    assert 2.6 < v.min_change("speed", 88.2) and v.hold_min_s == 40.0                     # under the minimum, but held 40 s
    # one steady stretch split by a brief excursion is ONE level, however long the second half is
    # held: its median moved 1.4 m/s, under the tolerance, even though it crossed a bin edge
    edge = np.concatenate((np.full(20, 75.9), np.full(5, 84.0), np.full(42, 77.1)))
    clock = np.arange(len(edge), dtype=float)
    flats = ins.plateaus(edge, v.speed_tolerance_mps, 6)
    assert flats == [(0, 20), (25, 67)] and v.speed_bin(75.9)[0] != v.speed_bin(77.1)[0]
    assert abs(75.2 - 73.8) < v.speed_tolerance_mps and clock[66] - clock[25] >= v.hold_min_s    # held past the hold, still one level
    words, absorbed = ins._manoeuvre_words("speed", clock, edge, flats, 6, v.speed_tolerance_mps,
                                           SPEED_MIN_CHANGE, v.hold_min_s, to_word)
    assert [w.word for w in words] == [v.speed_bin(75.9)[0]] and [a.reason for a in absorbed] == [ins.ABSORBED_SMALL_CHANGE]
    # The speed's minimum change is a FRACTION now, so it has no fixed value to compare against a
    # tolerance at construction — the reader computes it from the track. The heading's is absolute
    # and still carries the check: two plateaus closer than the tolerance are one plateau.
    with pytest.raises(ValueError, match="at least the kind's tolerance"):
        ins.Vocabulary(heading_min_change_deg=0.5)
    with pytest.raises(ValueError, match="at least plateau_min_s"):
        ins.Vocabulary(hold_min_s=10.0)
    assert ins.departure_row(np.array([5.0, 5.0, 5.4, 6.0, 6.8, 8.0]), 0, 5, 5.0, 2.0) == 4          # 6.8 is the first row past ±1


def test_the_runway_vocabulary_is_the_cohort_s_thresholds_and_refuses_a_stranger():
    """The runway word's classes are one airport's thresholds, sorted and unique. An ident the
    cohort never carried is a DIFFERENT airport, so `index` raises — there is no fallback class
    a stranger could be quietly filed under."""
    v = ins.RunwayVocabulary.from_idents(["KRDU:23R", "KRDU:05L", "KRDU:05L"])
    assert v.idents == ("KRDU:05L", "KRDU:23R") and len(v) == 2
    assert v.index("KRDU:05L") == 0 and v.index("KRDU:23R") == 1 and v.ident(1) == "KRDU:23R"
    with pytest.raises(ValueError, match="not in this vocabulary"):
        v.index("KRDU:18")
    with pytest.raises(ValueError, match="stored sorted"):
        ins.RunwayVocabulary(("KRDU:23R", "KRDU:05L"))
    with pytest.raises(ValueError, match="unique"):
        ins.RunwayVocabulary(("KRDU:05L", "KRDU:05L"))
    with pytest.raises(ValueError, match="at least one"):
        ins.RunwayVocabulary(())


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


def test_a_heading_change_is_read_when_it_lands_on_a_new_plateau_whatever_its_rate(flights):
    """A 12° change over 40 s (0.3°/s — no rate rule would see it) is a heading instruction issued
    where the old plateau ends, settled where the new one begins; a 3° wobble stays inside the
    tolerance and is nothing."""
    v = ins.Vocabulary()
    item = flights[0]
    t = np.arange(0.0, 160.0, 2.0)
    level, speed = np.full(len(t), 800.0), np.full(len(t), 80.0)
    slow = np.interp(t, [0.0, 40.0, 60.0, 160.0], [0.0, 0.0, 12.0, 12.0])                # 0.6°/s: the band's lag is ~7 s
    reading = ins.read_instructions(_track(item, t, slow, level, speed), v, RUNWAYS)
    headings = [i for i in reading.instructions if i.kind == "heading"]
    assert [i.word for i in headings] == [0, v.heading_bin(12.0)] and 40.0 <= headings[1].issued_s <= 40.0 + 10.0
    assert headings[1].settled_s is not None and 60.0 - 10.0 <= headings[1].settled_s <= 60.0 + 2.0 and headings[1].target == pytest.approx(12.0, abs=1.0)
    fast = np.interp(t, [0.0, 40.0, 44.0, 160.0], [0.0, 0.0, 12.0, 12.0])                # 3°/s: within a smoothing window
    reading = ins.read_instructions(_track(item, t, fast, level, speed), v, RUNWAYS)
    headings = [i for i in reading.instructions if i.kind == "heading"]
    assert [i.word for i in headings] == [0, v.heading_bin(12.0)] and abs(headings[1].issued_s - 40.0) <= 4.0
    assert abs(headings[1].settled_s - 44.0) <= 6.0
    three = np.interp(t, [0.0, 40.0, 44.0, 160.0], [0.0, 0.0, 1.5, 1.5])   # under the 2.5° minimum at 5° a bin
    reading = ins.read_instructions(_track(item, t, three, level, speed), v, RUNWAYS)
    assert [i.kind for i in reading.instructions] == ["heading", "vertical", "speed", "runway"] and not reading.absorbed


def test_a_level_off_is_the_descent_s_target_and_the_last_descent_targets_the_end(flights):
    """5 000 ft, descend to a 3 000 ft level-off held 40 s, descend again to the end. The vertical
    word is an ANGLE, so this profile is four segments — level, descend, level, descend — and the
    breakpoints land within a sample or two of the truth. The 3.57° descents read as the 3.1°
    mode, the nearest the vocabulary has."""
    v = ins.Vocabulary()
    item = flights[0]
    t = np.arange(0.0, 300.0, 2.0)
    height = np.interp(t, [0.0, 20.0, 142.0, 182.0, 298.0], [5000 * ins.FT, 5000 * ins.FT, 3000 * ins.FT, 3000 * ins.FT, 1100 * ins.FT])
    reading = ins.read_instructions(_track(item, t, np.zeros(len(t)), height, np.full(len(t), 80.0)), v, RUNWAYS)
    verticals = [i for i in reading.instructions if i.kind == "vertical"]
    level, glidepath = v.vertical_modes_deg.index(0.0), v.vertical_modes_deg.index(3.1)
    assert [i.word for i in verticals] == [level, glidepath, level, glidepath]
    assert verticals[0].issued_s == 0.0
    assert 20.0 - 4.0 <= verticals[1].issued_s <= 20.0 + 4.0        # the descent starts where it starts
    assert 142.0 - 4.0 <= verticals[2].issued_s <= 142.0 + 4.0      # and the level-off where it does
    assert 182.0 - 4.0 <= verticals[3].issued_s <= 182.0 + 4.0
    assert verticals[1].target == pytest.approx(3.57, abs=0.2) and not verticals[1].clamped
    # the four-second sliver the fit spends at a breakpoint is absorbed, never an instruction —
    # as a SHORT SEGMENT, not a "short tail": that name means the last plateau runs out before the
    # record ends, and this sliver is in the middle of the profile
    sliver = [a for a in reading.absorbed if a.kind == "vertical"]
    assert [a.reason for a in sliver] == [ins.ABSORBED_SHORT_SEGMENT]
    # `change` is what the manoeuvre MOVED the angle by, as for every other kind — not the angle
    assert abs(sliver[0].change) < 90.0


def test_a_clamped_target_reaches_the_instruction_and_an_orbit_is_absorbed(flights):
    """A record starting at 11 000 ft carries a clamped altitude word; a 360° orbit back onto the
    same heading is not a heading instruction but is recorded as absorbed with its sweep."""
    v = ins.Vocabulary()
    item = flights[0]
    t = np.arange(0.0, 200.0, 2.0)
    # A descent far steeper than the steepest mode reads AS that mode and is counted outside.
    steep = np.linspace(3000.0, 0.0, len(t))
    reading = ins.read_instructions(_track(item, t, np.zeros(len(t)), steep, np.full(len(t), 80.0)), v, RUNWAYS)
    vertical = next(i for i in reading.instructions if i.kind == "vertical")
    assert vertical.word == v.vertical_words - 1 and vertical.clamped and vertical.target > 4.4
    orbit = np.interp(t, [0.0, 30.0, 150.0, 200.0], [0.0, 0.0, 360.0, 360.0])
    reading = ins.read_instructions(_track(item, t, orbit, np.full(len(t), 800.0), np.full(len(t), 80.0)), v, RUNWAYS)
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


def test_a_pause_inside_a_descent_and_a_drift_with_no_plateau_produce_no_word(flights):
    """A 4 s pause at 1000 m inside a 3 m/s descent is not a level-off (the opening window's
    fitted slope says the aircraft is still descending); a course drifting 0.3°/s the whole
    record has no plateau at all and reads as one manoeuvre to its end."""
    v = ins.Vocabulary()
    item = flights[0]
    t = np.arange(0.0, 300.0, 2.0)
    speed = np.full(len(t), 75.0)
    height = np.interp(t, [0.0, 40.0, 206.0, 210.0, 300.0], [1500.0, 1500.0, 1000.0, 1000.0, 730.0])
    reading = ins.read_instructions(_track(item, t, np.zeros(len(t)), height, speed), v, RUNWAYS)
    verticals = [i for i in reading.instructions if i.kind == "vertical"]
    # level, descend, level, descend — and the last descent runs to the end
    assert [v.vertical_centre_deg(i.word) for i in verticals][0] == 0.0 and verticals[-1].word != v.vertical_modes_deg.index(0.0)
    drift = t * 0.3
    reading = ins.read_instructions(_track(item, t, drift, np.full(len(t), 800.0), speed), v, RUNWAYS)
    headings = [i for i in reading.instructions if i.kind == "heading"]
    assert len(headings) == 1 and headings[0].settled_s is None and headings[0].word == v.heading_bin(float(drift[-1]))


def test_refusals_at_the_boundary(flights):
    from dataclasses import replace
    item = flights[0]
    values = np.array(item.values, dtype=np.float64)
    values[5, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ins.course_frame(replace(item, values=values))
    with pytest.raises(ValueError, match="has no manoeuvre to read"):
        ins.read_instructions(replace(item, times=item.times[:1], values=item.values[:1]), ins.Vocabulary(), RUNWAYS)
    with pytest.raises(ValueError, match="at least 3 rows"):
        ins.plateaus(np.zeros(10), 1.0, 2)
    with pytest.raises(ValueError, match="at most plateau_min_s"):
        ins.Vocabulary(token_step_s=30.0)


def _capture(item, t: np.ndarray, relative_deg: np.ndarray, height_m: np.ndarray, speed_mps: np.ndarray):
    """Like `_track`, but ending ON the course at the threshold: the path is integrated backwards
    from the threshold, so the last rows are established whatever the course before."""
    from dataclasses import replace
    course = float(item.scenario.target.psi)
    psi = course + np.radians(relative_deg)
    ve, vn = speed_mps * np.cos(psi), speed_mps * np.sin(psi)
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2]))
    e = item.target_chart[0] - 500.0 * math.cos(course) - (np.cumsum((ve * dt)[::-1])[::-1] - ve * dt)
    n = item.target_chart[1] - 500.0 * math.sin(course) - (np.cumsum((vn * dt)[::-1])[::-1] - vn * dt)
    u = item.target_chart[2] + height_m
    values = np.column_stack((e, n, u, ve, vn, np.gradient(u, t)))
    return replace(item, times=t, values=values, supervision_times=t, supervision_values=values,
                   supervision_weights=np.full(values.shape, 1.0 / values.shape[1]))


def test_the_synthetic_arrival_reads_as_one_capture_turn_a_descent_to_the_threshold_and_a_deceleration(flights):
    """The synthetic track flies straight to the FAF, turns onto the course and descends on the
    glidepath while slowing: the heading words are the entry course then the course (bin 0) with
    a capture turn, the last altitude word is the threshold's bin 0, the speed words end at the
    threshold speed's bin, every position carries the three words in force."""
    v = ins.Vocabulary()
    for item in flights:
        reading = ins.read_instructions(item, v, RUNWAYS)
        headings = [i for i in reading.instructions if i.kind == "heading"]
        assert headings[0].issued_s == item.times[0] and headings[0].word == v.heading_bin(headings[0].target)
        if headings[0].word != 0:                   # the entry leg reads as another word: the capture is a turn onto the course
            assert len(headings) == 2 and headings[-1].word == 0                          # settles on the course
        else:                                       # the entry leg already reads as the course: one word, no capture to speak of
            assert len(headings) == 1 and abs(headings[0].target) < v.heading_bin_deg / 2
        verticals = [i for i in reading.instructions if i.kind == "vertical"]
        assert v.vertical_centre_deg(verticals[-1].word) > 0.0                            # descending to the end
        assert all(a.kind in ins.MANDATORY_KINDS for a in reading.absorbed)
        speeds = [i for i in reading.instructions if i.kind == "speed"]
        assert speeds[-1].word == v.speed_bin(speeds[-1].target)[0]
        frame = ins.course_frame(item)
        # the last word targets the plateau it settled on, or the final value when it never settled
        at = float(frame["ground_speed_mps"][-1]) if speeds[-1].settled_s is None else float(np.interp(speeds[-1].settled_s, frame["t"], frame["ground_speed_mps"]))
        assert abs(speeds[-1].target - at) < 2 * v.speed_tolerance_mps + 1.0
        # D52: an EVENT sequence — the first event is the record's start and the rest are the
        # moments something changed, so the gaps are irregular and never zero
        assert reading.event_times_s[0] == item.times[0] and (np.diff(reading.event_times_s) > 0).all()
        # every issue time is an event, plus ONE more: the landing (2026-09-21), which is an event
        # because the flight ended rather than because a word moved
        assert set(reading.event_times_s.tolist()) == {i.issued_s for i in reading.instructions} | {item.times[-1]}
        assert reading.event_times_s[-1] == item.times[-1]
        assert reading.words[-1, 5] == ins.TERMINAL_LANDED and (reading.words[:-1, 5] == ins.TERMINAL_CONTINUE).all()
        # SIX kinds since D73: heading / altitude / speed / runway / duration / terminal
        assert reading.words.shape == (len(reading.event_times_s), 6) and (reading.words[:, :3] >= 0).all()
        # the runway is said at the first position and stands for the whole sentence
        assert reading.runway == RUNWAY_WORD and (reading.words[:, 3] == RUNWAYS.index(RUNWAY_WORD)).all()
        runways = [i for i in reading.instructions if i.kind == "runway"]
        assert len(runways) == 1 and runways[0].issued_s == item.times[0]
        # a threshold has a NAME, not a number, so there is no continuous value it was binned from.
        # It is NOT NaN: `write_json_atomic` allows NaN, which Python reads back and strict JSON
        # readers (jq, JSON.parse) refuse — and the sentences files are read by both.
        assert runways[0].target == ins.NO_TARGET and not math.isnan(runways[0].target)
        assert reading.to_dict()["instructions"][0]["kind"] in ins.INSTRUCTION_KINDS
        assert not reading.established_from_start


def test_a_track_that_is_already_on_the_course_and_level_carries_only_its_starting_words(flights):
    """A hand-built series: straight along the course at constant height and speed — one word of
    each kind issued at t = 0, no manoeuvre."""
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
    reading = ins.read_instructions(straight, ins.Vocabulary(), RUNWAYS)
    assert [i.kind for i in reading.instructions] == ["heading", "vertical", "speed", "runway"] and all(i.issued_s == 0.0 for i in reading.instructions)
    words = {i.kind: i.word for i in reading.instructions}
    level = ins.Vocabulary().vertical_modes_deg.index(0.0)
    assert words == {"heading": 0, "vertical": level, "speed": ins.Vocabulary().speed_bin(150.0 * ins.KT)[0],
                     "runway": RUNWAYS.index(RUNWAY_WORD)} and reading.established_from_start
    # one event for the words said at t = 0, and one for the landing: nothing changes in between,
    # and the landing row is what says how long the whole thing lasted
    assert reading.event_times_s.tolist() == [0.0, t[-1]]
    assert reading.words[-1, 5] == ins.TERMINAL_LANDED
    assert ins.Vocabulary().duration_centre_s(int(reading.words[-1, 4])) == t[-1]


def test_the_words_in_force_at_a_time_and_the_executor_s_view_of_them():
    """`words_at` reads the sentence, not the issue times (an instruction issued at 25 s is in
    force from the 30 s position); the last words stay in force past the last position; the
    conditioning carries the bin centres (cos / sin heading, fractions of the ceilings, a graded
    ) and the segment positions are Δ / τ of them."""
    v = ins.Vocabulary()
    positions = np.array([0.0, 10.0, 20.0, 30.0])
    rwy = RUNWAYS.index(RUNWAY_WORD)
    C, L = ins.TERMINAL_CONTINUE, ins.TERMINAL_LANDED
    words = np.array([[36, 3, 4, rwy, 0, C], [36, 3, 4, rwy, 5, C],     # 36 = 180° at 5° a bin
                      [36, 3, 4, rwy, 5, C], [54, 3, 4, rwy, 5, L]])    # 54 = −90°
    reading = ins.Reading("d", "f", (), positions, words, RUNWAY, False, 35.0)
    assert reading.words_at(np.array([0.0, 9.9, 27.0, 30.0, 95.0])).tolist() == [words[0].tolist(), words[0].tolist(), words[2].tolist(), words[3].tolist(), words[3].tolist()]
    with pytest.raises(ValueError, match="before the first event"):
        reading.words_at(np.array([-1.0]))
    assert ins.segment_positions_s(40.0, 20.0, 10.0).tolist() == [40.0, 50.0]
    assert ins.segment_positions_s(0.0, 10.0, 10.0).tolist() == [0.0]
    with pytest.raises(ValueError, match="whole number"):
        ins.segment_positions_s(0.0, 25.0, 10.0)
    cond = v.conditioning(reading.words_at(ins.segment_positions_s(20.0, 20.0, 10.0)))
    assert cond.shape == (2, v.CONDITIONING_WIDTH) and cond.dtype == np.float32
    assert cond[0, 0] == pytest.approx(-1.0) and abs(cond[0, 1]) < 1e-6 and cond.shape[-1] == 4          # downwind, no intercept
    assert cond[1, 0] == pytest.approx(0.0, abs=1e-6) and cond[1, 1] == pytest.approx(-1.0)   # (2 / 3)  # −90° base, bin 1
    modes = np.asarray(v.vertical_modes_deg)
    assert cond[0, 2] == pytest.approx(v.vertical_centre_deg(3) / abs(modes).max())
    assert cond[0, 3] == pytest.approx(v.speed_centre_mps(4) / max(v.speed_centres_mps))
    # the RUNWAY column is read in (the words are [E, 6]) but never conditioned on: the executor
    # already works in that runway's frame, so a second runway word changes no feature (D62)
    other = words.copy()
    other[:, 3] = 1
    assert np.array_equal(v.conditioning(other), v.conditioning(words)) and v.CONDITIONED_KINDS == ins.MANDATORY_KINDS   # ("intercept",)
    with pytest.raises(ValueError, match="words are"):
        v.conditioning(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="words are"):
        v.conditioning(np.zeros((2, 7)))                                 # seven columns is the PRE-D73 sentence


def test_the_sentence_is_one_row_per_moment_something_changed_and_carries_the_gap_to_the_one_before():
    """D52: an EVENT sequence, not an even grid.

    Three moments here — 0 s (four kinds at once), 26 s (a turn), 62 s (a turn and the capture) —
    plus the LANDING at 140 s, so four rows, never eleven. The times are on the ADS-B row grid,
    where the 2 s duration bin is exact (off-grid, the sum below is the rounding, not the span). Each row holds the latest word of every
    kind, plus the gap to the row before it as a word (D71, 2 s a bin).
    """
    said = [ins.Instruction("heading", 36, 180.0, 0.0, 0.0), ins.Instruction("vertical", 3, 2.4, 0.0, 0.0),
            ins.Instruction("speed", 4, 80.0, 0.0, 0.0), ins.Instruction("runway", 1, ins.NO_TARGET, 0.0, 0.0),
            ins.Instruction("heading", 27, -90.0, 26.0, 40.0),
            ins.Instruction("heading", 0, 0.0, 62.0, 90.0)]
    v = ins.Vocabulary()
    times, table, clamped = ins.sentence(said, v, ends_s=140.0)
    assert times.tolist() == [0.0, 26.0, 62.0, 140.0] and table.shape == (4, 6) and clamped == 0
    assert table[:, 0].tolist() == [36, 27, 0, 0]                    # the heading of each moment
    assert (table[:, 1] == 3).all() and (table[:, 2] == 4).all()     # vertical and speed hold
    assert (table[:, 3] == 1).all()                                  # the runway holds over the whole sentence
    assert table[:, 4].tolist() == [0, v.duration_bin(26.0)[0], v.duration_bin(36.0)[0], v.duration_bin(78.0)[0]]
    # D72: the terminal word says where the sentence stops — and LANDED is the landing's own row,
    # not the last change (2026-09-21): on real tracks the last change is a median 136 s before the
    # record ends, so without this row nothing in the sentence says when the flight lands.
    assert table[:, 5].tolist() == [ins.TERMINAL_CONTINUE] * 3 + [ins.TERMINAL_LANDED]
    # the duration words sum to the flight's own span, which is what makes the landing time sayable
    assert sum(v.duration_centre_s(int(w)) for w in table[:, 4]) == 140.0
    with pytest.raises(ValueError, match="no word in force"):
        ins.sentence(said[:2], v, ends_s=140.0)                      # no runway word: refused, not -1


def test_a_sentence_whose_last_change_IS_the_landing_carries_no_extra_row():
    """The landing row exists because the record outlives the last change. When it does not — the
    words move at the very last sample — the last change IS the landing and a second row there
    would repeat it."""
    v = ins.Vocabulary()
    said = [ins.Instruction("heading", 0, 0.0, 0.0, 0.0), ins.Instruction("vertical", 0, 0.0, 0.0, 0.0),
            ins.Instruction("speed", 4, 80.0, 0.0, 0.0), ins.Instruction("runway", 1, ins.NO_TARGET, 0.0, 0.0),
            ins.Instruction("speed", 5, 85.0, 60.0, 60.0)]
    times, table, _clamped = ins.sentence(said, v, ends_s=60.0)
    assert times.tolist() == [0.0, 60.0]
    assert table[:, 5].tolist() == [ins.TERMINAL_CONTINUE, ins.TERMINAL_LANDED]


def test_a_gap_longer_than_the_duration_ceiling_clamps_and_is_counted():
    v = ins.Vocabulary()
    far = v.duration_max_s + 60.0
    said = [ins.Instruction("heading", 0, 0.0, 0.0, 0.0), ins.Instruction("vertical", 0, 0.0, 0.0, 0.0),
            ins.Instruction("speed", 4, 80.0, 0.0, 0.0), ins.Instruction("runway", 1, ins.NO_TARGET, 0.0, 0.0),
            ins.Instruction("speed", 5, 85.0, far, far)]
    _times, table, clamped = ins.sentence(said, v, ends_s=far)
    assert clamped == 1 and table[1, 4] == v.duration_words - 1


def test_a_sentence_without_a_runway_word_is_refused_rather_than_indexing_an_embedding_at_minus_one():
    """Every kind must be in force from the first event; the runway most of all.

    Its column feeds `nn.Embedding`, where −1 silently reads the LAST runway instead of raising —
    a wrong answer with no symptom. The agent that propagated D62 flagged this and left it; the
    the guard covers every kind.
    """
    v = ins.Vocabulary()
    heading = ins.Instruction("heading", 0, 0.0, 0.0, 0.0, False)
    altitude = ins.Instruction("vertical", 0, 0.0, 0.0, 0.0, False)
    speed = ins.Instruction("speed", 5, 0.0, 0.0, 0.0, False)
    runway = ins.Instruction("runway", 2, float("nan"), 0.0, 0.0, False)
    times, words, _clamped = ins.sentence([heading, altitude, speed, runway], v, ends_s=0.0)
    assert words.shape == (len(times), len(ins.INSTRUCTION_KINDS))
    assert (words[:, ins.INSTRUCTION_KINDS.index("runway")] == 2).all()
    with pytest.raises(ValueError, match="runway"):
        ins.sentence([heading, altitude, speed], v, ends_s=0.0)


def _profile_series(t, height, speed=80.0):
    """times, heights, a constant ground speed — the three arrays the vertical reader takes."""
    return np.asarray(t, float), np.asarray(height, float), np.full(len(t), float(speed))


def test_the_vertical_instructions_tile_the_track_and_preserve_the_height_they_describe():
    """The central claim of the vertical reader: a RATE describes the shape of the whole curve, so
    its segments must cover the track with no gap and no overlap — and folding two of them must
    not invent or lose height. Neither was covered before (review, 2026-09-21)."""
    v = ins.Vocabulary()
    t = np.arange(0.0, 600.0, 2.0)
    height = np.concatenate((np.full(50, 2000.0), 2000.0 - np.arange(150) * 8.0, np.full(100, 800.0)))
    out, _absorbed, rms, _merged = ins._vertical_instructions(*_profile_series(t, height), v)
    spans = [(i.issued_s, i.settled_s) for i in out]
    assert spans[0][0] == t[0] and spans[-1][1] == t[-1]
    for (_a, end), (start, _b) in zip(spans, spans[1:]):
        assert end == start, spans                       # no gap, no overlap
    # the words' own profile is within a few metres of the track it describes
    assert rms < 60.0, rms
    # every angle the words name is one the vocabulary can say
    assert all(i.word in range(v.vertical_words) for i in out)


def test_the_breakpoints_are_the_OPTIMAL_cut_not_a_greedy_one():
    """`_breakpoints` is a dynamic program because the best k-segment fit is not the best
    (k−1)-segment fit plus one more. Checked against brute force on a small case."""
    from itertools import combinations
    rng = np.random.default_rng(3)
    x = np.arange(12, dtype=float)
    y = np.concatenate((np.zeros(4), np.arange(1, 5, dtype=float), np.full(4, 4.0))) + rng.normal(0, 0.05, 12)
    cost = ins._segment_costs(x, y)
    got = ins._breakpoints(x, y, 3)
    best = min(combinations(range(1, 11), 2), key=lambda c: cost[0][c[0]] + cost[c[0]][c[1]] + cost[c[1]][11])
    assert got == [0, best[0], best[1], 11], (got, best)


def test_the_segment_ceiling_is_reported_rather_than_silently_binding():
    """More phases than `vertical_segments` can express: the extra ones are lost — which is
    allowed — but the loss has to be VISIBLE, and the fit residual is what shows it."""
    v = ins.Vocabulary()
    t = np.arange(0.0, 700.0, 2.0)
    steps = []
    height = 3000.0
    for k in range(7):                                   # level / descend / level / descend / …
        rate = 0.0 if k % 2 == 0 else 6.0
        steps.append(height - np.arange(50) * rate)
        height = steps[-1][-1]
    height_m = np.concatenate(steps)[: len(t)]
    _out, _absorbed, rms, _merged = ins._vertical_instructions(*_profile_series(t, height_m), v)
    roomy = ins.Vocabulary(vertical_segments=9)
    _o2, _a2, rms_roomy, _m2 = ins._vertical_instructions(*_profile_series(t, height_m), roomy)
    assert rms > rms_roomy, (rms, rms_roomy)             # the ceiling costs height error …
    assert rms > 20.0                                    # … and enough of it to see in the summary


def test_a_half_circle_turn_is_split_so_the_short_way_is_the_way_the_aircraft_went():
    """A heading word names a direction, and a direction cannot say which way round.

    Here the aircraft reverses course by turning through the LEFT (the unwrapped course falls by
    180°). Unsplit, the sentence would hold two words a half circle apart, and anything flying it
    has to guess: `wrap_deg(180)` is −180 on every implementation, so it would turn right and
    mirror the whole track. Split, each leg is under `turn_split_deg` and the short way IS the
    flown way — and the intermediate word is issued when the TURN STARTS, not when the aircraft
    arrives, or nothing would send it there.
    """
    v = ins.Vocabulary()
    t = np.arange(0.0, 400.0, 2.0)
    course = np.clip(np.interp(t, [0.0, 100.0, 280.0, 400.0], [0.0, 0.0, -180.0, -180.0]), -180.0, 0.0)
    headings = [ins.Instruction("heading", v.heading_bin(0.0), 0.0, 0.0, 0.0),
                ins.Instruction("heading", v.heading_bin(-180.0), -180.0, 100.0, 280.0)]
    split = ins._split_long_turns(headings, t, course, v)
    assert len(split) == 3, [i.word for i in split]
    via = split[1]
    assert via.issued_s == 100.0                                  # said when the turn starts
    assert 100.0 < via.settled_s <= 280.0                         # settled when the aircraft gets there
    assert split[2].issued_s == via.settled_s                     # and the far word only then
    # every leg is now inside the split angle, so the short way is the flown way
    centres = [v.heading_centre_deg(i.word) for i in split]
    steps = [abs(ins.wrap_deg(b - a)) for a, b in zip(centres, centres[1:])]
    assert max(steps) < v.turn_split_deg
    assert all(ins.wrap_deg(b - a) < 0 for a, b in zip(centres, centres[1:])), "the split must keep the LEFT turn left"


def test_a_turn_inside_the_split_angle_is_left_alone():
    v = ins.Vocabulary()
    t = np.arange(0.0, 200.0, 2.0)
    course = np.interp(t, [0.0, 60.0, 140.0, 200.0], [0.0, 0.0, 90.0, 90.0])
    headings = [ins.Instruction("heading", v.heading_bin(0.0), 0.0, 0.0, 0.0),
                ins.Instruction("heading", v.heading_bin(90.0), 90.0, 60.0, 140.0)]
    assert ins._split_long_turns(headings, t, course, v) == headings


def _on_runway(flights, ident: str):
    """The first fixture flight, relabelled onto another threshold.

    Only the SOURCE's `runway` moves: the word is read from the manifest's ident, not from the
    geometry, so this is exactly the input `read_instructions` looks at.
    """
    item = flights[0]
    scenario = replace(item.scenario, source={**item.scenario.source, "runway": ident})
    return replace(item, scenario=scenario)


def test_the_runway_word_is_the_FLIGHT_S_own_class_not_a_constant(flights):
    """The fixtures' own runway sorts FIRST, so every other assertion compares the word to 0.

    A labeller that wrote a constant 0 — or another flight's runway — would pass the rest of this
    file. Pin a flight whose runway is NOT index 0 (review finding, 2026-09-20).
    """
    v = ins.Vocabulary()
    other = OTHER_RUNWAY_WORD
    assert other != RUNWAY_WORD and RUNWAYS.index(other) != 0, "the fixture must exercise a non-zero class"
    item = _on_runway(flights, other.split(":", 1)[1])
    reading = ins.read_instructions(item, v, RUNWAYS)
    assert ins.flight_runway(item) == other
    assert reading.runway == other
    assert (reading.words[:, ins.INSTRUCTION_KINDS.index("runway")] == RUNWAYS.index(other)).all()


def test_a_runway_the_vocabulary_was_not_built_on_raises_rather_than_falling_back(flights):
    v = ins.Vocabulary()
    item = _on_runway(flights, "99Z")
    with pytest.raises(ValueError, match="99Z"):
        ins.read_instructions(item, v, RUNWAYS)


def test_the_runway_word_is_qualified_by_airport_so_a_shared_ident_is_two_words(flights):
    """KSJC and KSTL both have 12L/12R/30L/30R, KSTL and KMSY both have 11/29 — six idents over
    twelve runways whose approach courses point different ways. Bare, one embedding would have to
    stand for both and the word could not say which runway it meant (the pooled cohort: 16 classes, not 22).
    """
    here = flights[0]
    elsewhere = replace(here, scenario=replace(here.scenario, source={**here.scenario.source, "arr_airport": "KSJC"}))
    assert ins.flight_runway(here) == RUNWAY_WORD
    assert ins.flight_runway(elsewhere) == f"KSJC:{RUNWAY}"
    assert len(ins.RunwayVocabulary.from_idents(ins.flight_runway(item) for item in (here, elsewhere))) == 2
