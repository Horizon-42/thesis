"""The instruction labeller on synthetic flights whose sentence is known (vocabulary design §3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ts_transformer.instructions.airport import RunwayRelative
from ts_transformer.instructions.labeller.lateral import read_lateral
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.labeller.sentence import assemble
from ts_transformer.instructions.labeller.speed import read_speed
from ts_transformer.instructions.labeller.vertical import read_vertical, tube_checks
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, ANGLE_LEVEL, APPROACH, APPROACH_CLEARED, APPROACH_NOT_CLEARED, HEADING, RUNWAY, SPEED,
    UNCHANGED, Words,
)
from ts_transformer.tests.support import (
    INSTRUCTION_STEP_S, fixture_days, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec,
)

def columns(reading, column):
    grid = reading.words
    return [(int(row), int(grid[row, column])) for row in np.nonzero(grid[:, column] != UNCHANGED)[0]]


def test_downwind_base_final_reads_as_words_turn_by_turn_a_clearance_and_a_landing():
    """§10.1: the downwind, the turn onto the base read 5° at a time (4 s early), the base, and the clearance where the
    turn onto the final begins — the base heading already reaches the final, so the capture turn is the executor's."""
    one, words = spec(), Words(spec())
    # downwind west, left turn onto a base south, left turn onto the final east (course 090)
    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    reading = read_flight(instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0)), instruction_airport(), one, words)
    headings = [(row, words.heading_deg(value)) for row, value in columns(reading, HEADING)]
    values = [value for _, value in headings]
    assert values[0] == 270.0 and 180.0 in values and values == sorted(values, reverse=True)
    # the base turn begins at row 60; the 6 s smoothing and the 2-row lead put its first word at row 59
    assert headings[1][0] == 59
    assert 94 <= reading.join_row <= 96 and all(row < reading.join_row for row, _ in headings)
    assert columns(reading, APPROACH) == [(0, APPROACH_NOT_CLEARED), (reading.join_row, APPROACH_CLEARED)]
    assert columns(reading, RUNWAY) == [(0, 0)]
    altitude = columns(reading, ALTITUDE)
    assert words.altitude_m(altitude[0][1]) == 1110.0 and altitude[-1][1] == words.altitude_land
    angle = columns(reading, ANGLE)
    assert angle[0] == (0, ANGLE_LEVEL) and words.angle_bounds(angle[-1][1]) == (2.6, 3.7)
    assert reading.capture_row > reading.join_row
    capture = reading.checks["capture_turn"]
    assert capture["start_row"] == reading.join_row and capture["progress_ok"] and capture["rate_ok"]
    assert all(h["inside"] == h["rows"] for h in reading.checks["heading"])     # the observed track, by construction
    assert all(check["contained"] for check in reading.checks["vertical"])
    # two 90° turns; the smoothed track turns slower than the onset rate at their very ends
    assert reading.checks["capture_before_threshold_m"] > 0.0 and reading.checks["turning_deg"] == pytest.approx(180.0, abs=5.0)
    assert columns(reading, SPEED)[-1][1] == words.speed_unspecified


def test_a_flight_is_cut_at_its_landing():
    legs = [(100, 0.0, 70.0, -3.0), (20, 0.0, 60.0, 0.0)]
    e, n, altitude, track, speed = fly_legs(legs, 90.0, 700.0, 1000.0, 0.0)     # the last 1 km is over the runway
    reading = read_flight(instruction_flight(e, n, altitude, track, speed), instruction_airport(), spec())
    assert reading.cut_at_crossing
    assert len(reading.words) == int(np.nonzero(e >= 0.0)[0][0])


def _relative(n_rows: int, capture: int, course: float, track: np.ndarray, offset_before: float) -> RunwayRelative:
    right = np.where(np.arange(n_rows) >= capture, 0.0, offset_before)
    return RunwayRelative(before_threshold_m=np.linspace(30000.0, 500.0, n_rows), right_of_course_m=right,
                          track_minus_course_deg=((track - course + 180.0) % 360.0) - 180.0,
                          height_above_threshold_m=np.full(n_rows, 500.0))


def _per_step(track, capture, course, offset, **changes):
    one = spec(**changes)
    words = Words(one)
    return read_lateral(track, np.full(len(track), 90.0), _relative(len(track), capture, course, track, offset),
                        course, one, words), words


# a slow continuous turn from north onto an eastbound final (course 090), 500 m left of it: 3° a row for 30 rows
CONTINUOUS = np.concatenate((np.full(30, 0.0), np.arange(1, 31) * 3.0, np.full(30, 90.0)))


def test_per_step_merges_the_rows_into_a_word_per_grid_cell_and_runs_to_the_capture():
    """§10.1: the words are the run-length reading of every row's nearest grid heading (here with no lead) — no hold,
    no split, no inserted intercept. The turn onto the course began at row 29, where 000 does not reach the final:
    the clearance waits until 090 is in force (row 60), and the rows after it say no word."""
    reading, words = _per_step(CONTINUOUS, 70, 90.0, -500.0, heading_lead_s=0.0)
    heading = [(i.row, words.heading_deg(i.value), i.kind) for i in reading.instructions if i.column == HEADING]
    cells = np.round(CONTINUOUS[:70] / 5.0) * 5.0
    expected = [(0, cells[0])] + [(r, cells[r]) for r in range(1, 70) if cells[r] != cells[r - 1]]
    assert [(row, deg) for row, deg, _ in heading] == expected
    assert heading[-1][1] == 90.0 and {kind for _, _, kind in heading[1:]} == {"per-step"}
    cleared = [i.row for i in reading.instructions if i.column == APPROACH and i.value == APPROACH_CLEARED]
    assert cleared == [reading.join_row] == [60]


def test_per_step_lead_labels_each_row_with_the_track_that_many_seconds_later():
    base, _ = _per_step(CONTINUOUS, 70, 90.0, -500.0, heading_lead_s=0.0)
    led, words = _per_step(CONTINUOUS, 70, 90.0, -500.0, heading_lead_s=4.0)
    rows = lambda reading: [(i.row, i.value) for i in reading.instructions if i.column == HEADING and i.row > 0]  # noqa: E731
    assert rows(led) == [(row - 2, value) for row, value in rows(base)]


def test_per_step_says_a_word_each_time_a_wander_crosses_a_cell_edge():
    """§10.1 (user, 2026-09-24): the boundary words are kept — a wander of ±2° about 092.4 crosses the 092.5 edge, and
    each crossing is a word (their average is the track the words describe)."""
    track = np.concatenate((92.4 + 2.0 * np.sin(np.arange(70) / 3.0), np.full(10, 90.0)))
    reading, words = _per_step(track, 70, 90.0, -200.0, heading_lead_s=0.0)
    values = [words.heading_deg(i.value) for i in reading.instructions if i.column == HEADING]
    assert len(values) > 5 and set(values) == {90.0, 95.0}


# a westbound downwind north of an eastbound final (course 090), then one continuous left turn of 180° onto it,
# 3° a row, captured where the turn ends
DOWNWIND_TO_FINAL = np.concatenate((np.full(30, 270.0), 270.0 - np.arange(1, 61) * 3.0))


def test_the_clearance_waits_for_a_heading_word_that_reaches_the_final():
    """§10.1: the capture turn — one continuous left turn of 180° from a downwind — begins at row 29, where the
    downwind word does not reach the final; the words run on until 180 is in force (185 is 95° off the course, beyond
    90° + the tolerance), and the clearance is said there."""
    reading, words = _per_step(DOWNWIND_TO_FINAL, 89, 90.0, -100.0, heading_lead_s=0.0)
    said = [(i.row, round(words.heading_deg(i.value))) for i in reading.instructions if i.column == HEADING]
    assert said[0] == (0, 270) and said[-1][1] == 180 and reading.join_row == said[-1][0] + 1
    assert all(row < reading.join_row for row, _ in said)
    cleared = [i.row for i in reading.instructions if i.column == APPROACH and i.value == APPROACH_CLEARED]
    assert cleared == [reading.join_row] and reading.capture_turn["start_row"] == reading.join_row
    assert reading.turning_deg == pytest.approx(180.0)


def test_level_descend_level_descend_reads_targets_angles_and_land():
    one, words = spec(), Words(spec())
    distance = np.arange(400) * 150.0
    altitude = np.concatenate((np.full(100, 1500.0), 1500.0 - np.arange(1, 81) * 150.0 * np.tan(np.radians(2.1)),
                               np.full(120, 1500.0 - 80 * 150.0 * np.tan(np.radians(2.1)))))
    altitude = np.concatenate((altitude, altitude[-1] - np.arange(1, 101) * 150.0 * np.tan(np.radians(3.0))))
    reading = read_vertical(distance, altitude, one, words)
    targets = [(i.row, words.altitude_m(i.value)) for i in reading.instructions if i.column == ALTITUDE]
    checks = tube_checks(reading.instructions, distance, altitude, one, words)
    assert targets[0] == (0, 1500.0)
    assert targets[1][1] == pytest.approx(round((1500.0 - 80 * 150.0 * np.tan(np.radians(2.1))) / 30.0) * 30.0)
    assert targets[2][1] is None
    assert 97 <= targets[1][0] <= 101 and 297 <= targets[2][0] <= 301
    angles = [words.angle_deg(i.value) for i in reading.instructions if i.column == ANGLE]
    assert angles == [0.0, 2.1, 3.0]
    assert len(checks) == 3 and all(move["contained"] for move in checks)


def test_the_speed_is_unspecified_after_the_clearance_unless_a_hold_ends_far_enough_out():
    one, words = spec(), Words(spec())
    time = np.arange(200) * INSTRUCTION_STEP_S
    speed = np.concatenate((np.full(50, 120.0), 120.0 - np.arange(1, 31) * 1.0, np.full(60, 90.0),
                            90.0 - np.arange(1, 31) * 0.5, np.full(30, 75.0)))
    far = np.linspace(40000.0, 300.0, 200)
    near = np.linspace(9000.0, 300.0, 200)
    kept = read_speed(time, speed, far, 60, one, words)
    assert [words.speed_mps(i.value) for i in kept.instructions] == [120.0, 90.0, None]
    # the 90 m/s hold ends 9.3 km or more out; the deceleration's first rows still fit the hold's
    # straight piece within the 1.5 m/s fit tolerance
    assert 140 <= kept.unspecified_row <= 145
    # no hold ends far enough out: unspecified from the clearance (row 60) — and since the
    # deceleration under way there began less than a minimum hold before it, from where it began
    dropped = read_speed(time, speed, near, 60, one, words)
    assert [words.speed_mps(i.value) for i in dropped.instructions] == [120.0, None]
    assert 50 <= dropped.unspecified_row <= 53


def test_the_assembly_drops_repeats_and_refuses_conflicts_and_an_empty_step_zero():
    one, words = spec(), Words(spec())
    altitude = np.full(10, 1000.0)
    base = [Instruction(RUNWAY, 0, 0, "i"), Instruction(APPROACH, 0, 0, "i"), Instruction(HEADING, 18, 0, "i"),
            Instruction(ALTITUDE, 33, 0, "i"), Instruction(ANGLE, ANGLE_LEVEL, 0, "i"), Instruction(SPEED, 16, 0, "i")]
    grid, kept = assemble(10, [*base, Instruction(HEADING, 18, 4, "repeat")], altitude, one, words)
    assert grid[4, HEADING] == UNCHANGED and len(kept) == 6
    with pytest.raises(Refused, match="two instructions in one step"):
        assemble(10, [*base, Instruction(HEADING, 20, 0, "clash")], altitude, one, words)
    with pytest.raises(Refused, match="step 0 incomplete"):
        assemble(10, base[:-1], altitude, one, words)
    with pytest.raises(Refused, match="altitude and angle incompatible"):
        assemble(10, [*base, Instruction(ALTITUDE, 20, 3, "descend with a level angle")], altitude, one, words)


def test_a_flight_on_the_final_from_row_0_is_cleared_at_row_0():
    one, words = spec(), Words(spec())
    track = np.full(40, 91.0)
    reading = read_lateral(track, np.full(40, 80.0), _relative(40, 0, 90.0, track, 0.0), 90.0, one, words)
    assert reading.capture_row == 0 and reading.join_row == 0 and reading.capture_turn is None
    assert [(i.column, i.row, i.value) for i in reading.instructions] == [
        (HEADING, 0, words.heading_index(91.0)), (APPROACH, 0, APPROACH_CLEARED)]


def test_a_flight_entering_mid_turn_is_told_the_heading_a_lead_ahead_at_row_0():
    one, words = spec(), Words(spec())
    # turning from 000 at 3°/row onto 090, which converges at 30° on a final of 060 from its left
    track = np.concatenate((np.arange(0, 30) * 3.0, np.full(40, 90.0), 90.0 - np.arange(1, 16) * 2.0, np.full(20, 60.0)))
    reading = read_lateral(track, np.full(len(track), 90.0), _relative(len(track), 85, 60.0, track, -600.0), 60.0, one, words)
    heading = [(i.row, words.heading_deg(i.value)) for i in reading.instructions if i.column == HEADING]
    assert heading[0] == (0, 5.0) and heading[-1][1] == 90.0         # row 0 is told the track 4 s (2 rows) on: 006
    assert reading.join_row == 69                                    # the turn from 090 onto the course begins there


@pytest.mark.parametrize("track, offset", [
    # from the right of an eastbound course of 092 (south of it): 272 → a right turn onto a base
    # of 000 (92° off the course) → a right turn onto the final
    (np.concatenate((np.full(30, 272.0), 272.0 + np.arange(1, 16) * 88.0 / 15, np.full(30, 360.0),
                     360.0 + np.arange(1, 16) * 92.0 / 15, np.full(20, 452.0))), 800.0),
    # the mirror from its left: a left turn onto a base of 180 (88° off), a left turn onto the final
    (np.concatenate((np.full(30, 272.0), 272.0 - np.arange(1, 16) * 92.0 / 15, np.full(30, 180.0),
                     180.0 - np.arange(1, 16) * 88.0 / 15, np.full(20, 92.0))), -800.0),
])
def test_perpendicular_bases_on_both_sides_of_an_off_grid_course_are_cleared_where_the_capture_turn_begins(track, offset):
    """A base 88–92° off the course reaches the final from either side: the clearance comes at the capture turn's onset
    (its last straight row, 74), the lead having said the turn's first 5° there."""
    one, words = spec(), Words(spec())
    reading = read_lateral(track, np.full(len(track), 90.0), _relative(len(track), 90, 92.0, track, offset), 92.0, one, words)
    heading = [words.heading_deg(i.value) for i in reading.instructions if i.column == HEADING]
    assert heading[0] in (270.0, 275.0) and heading[-1] in (5.0, 175.0)
    assert reading.join_row == 74


def test_an_aligned_heading_just_outside_the_corridor_drifts_in_on_one_word():
    one, words = spec(), Words(spec())
    # 090 against a course of 092, 100 m left of the line where the corridor is about 84 m wide
    track = np.concatenate((np.full(60, 90.0), np.array([91.0, 92.0]), np.full(20, 92.0)))
    reading = read_lateral(track, np.full(len(track), 80.0), _relative(len(track), 62, 92.0, track, -100.0), 92.0, one, words)
    # already within the course tolerance from row 0: no turn onto the course, cleared at entry
    assert reading.capture_row == 62 and reading.join_row == 0
    assert [words.heading_deg(i.value) for i in reading.instructions if i.column == HEADING] == [90.0]


def test_a_step_between_two_levels_is_issued_at_the_first_levels_end_with_its_own_direction():
    one, words = spec(), Words(spec())
    distance = np.arange(60) * 200.0
    altitude = np.concatenate((np.full(30, 900.0), np.full(30, 940.0)))
    reading = read_vertical(distance, altitude, one, words)
    assert [(i.column, i.row, i.value) for i in reading.instructions] == [
        (ALTITUDE, 0, words.altitude_index(900.0)), (ANGLE, 0, ANGLE_LEVEL),
        (ALTITUDE, 29, words.altitude_index(930.0)), (ANGLE, 29, words.angle_climb)]
    grid, _ = assemble(60, [Instruction(RUNWAY, 0, 0, "i"), Instruction(APPROACH, 0, 0, "i"),
                            Instruction(HEADING, 18, 0, "i"), Instruction(SPEED, 16, 0, "i"), *reading.instructions],
                       altitude, one, words)
    assert grid[29, ANGLE] == words.angle_climb


def test_a_flight_climbing_at_the_end_is_refused():
    one, words = spec(), Words(spec())
    altitude = np.concatenate((np.full(30, 900.0), 900.0 + np.arange(1, 31) * 3.0))
    with pytest.raises(Refused, match="climbing at the end"):
        read_vertical(np.arange(60) * 200.0, altitude, one, words)


def test_a_second_word_in_a_cell_is_refused_even_after_a_repeat_is_dropped():
    one, words = spec(), Words(spec())
    base = [Instruction(RUNWAY, 0, 0, "i"), Instruction(APPROACH, 0, 0, "i"), Instruction(HEADING, 18, 0, "i"),
            Instruction(ALTITUDE, 33, 0, "i"), Instruction(ANGLE, ANGLE_LEVEL, 0, "i"), Instruction(SPEED, 16, 0, "i")]
    with pytest.raises(Refused, match="two instructions in one step"):
        assemble(10, [*base, Instruction(HEADING, 18, 4, "repeat"), Instruction(HEADING, 30, 4, "other")],
                 np.full(10, 1000.0), one, words)


@pytest.mark.parametrize("altitude, refused", [(150.0, True), (600.0, False)])
def test_a_low_pass_the_flight_comes_back_from_is_refused_and_an_overflight_is_no_landing(altitude, refused):
    # eastbound over the threshold of 09 (elevation 100 m), a right turn back through the south,
    # westbound to end 3 km short: at 50 m above the threshold that is a landing as the harvest
    # judges one, and the flight coming back from it is refused; at 500 m it is an overflight
    legs = [(40, 0.0, 70.0, 0.0), (30, 6.0, 70.0, 0.0), (60, 0.0, 70.0, 0.0)]
    radius = 70.0 * INSTRUCTION_STEP_S / np.radians(6.0)
    flight = instruction_flight(*fly_legs(legs, 90.0, altitude, -3000.0, -2.0 * radius))
    if refused:
        with pytest.raises(Refused, match="threshold passed before the landing"):
            read_flight(flight, instruction_airport(), spec())
    else:
        with pytest.raises(Refused) as caught:
            read_flight(flight, instruction_airport(), spec())
        assert "threshold passed before the landing" not in str(caught.value)


def test_the_labels_runner_maps_each_sentence_to_its_signals_row(tmp_path, monkeypatch):
    from ts_transformer.experiments import instruction_labels
    from ts_transformer.instructions.artefact import (
        labeller_source_sha256, load_sentences, write_candidates, write_signals, write_spec,
    )

    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    good = fly_legs(legs, 270.0, 1110.0, -400.0, 0.0)
    slow = (*good[:4], np.full(len(good[0]), 10.0))                     # impossible ground speed
    flights = {"train": [instruction_flight(*good, dataset_id="KXXX:a"), instruction_flight(*slow, dataset_id="KXXX:b"),
                         instruction_flight(*good, dataset_id="KXXX:c")],
               "select": [instruction_flight(*good, dataset_id="KXXX:e", split="select")],
               "val": [instruction_flight(*good, dataset_id="KXXX:d", split="val")]}
    write_signals(tmp_path, flights, {"note": "test"}, fixture_days())
    write_candidates(tmp_path, {"KXXX": instruction_airport()})
    write_spec(tmp_path, spec(), {"n": 1}, {"labeller_source_sha256": labeller_source_sha256(),
                                            "git": {"head": "test", "dirty": False}})
    monkeypatch.setattr(instruction_labels, "CHUNK", 2)
    assert instruction_labels.main(["--dir", str(tmp_path), "--workers", "1"]) == 0
    train = load_sentences(tmp_path, "train", spec())
    assert train["signal_index"].tolist() == [0, 2]
    labels = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert [r["dataset_id"] for r in labels["train"]["labelled"]] == ["KXXX:a", "KXXX:c"]
    assert [(r["dataset_id"], r["reason"]) for r in labels["train"]["refused"]] == [("KXXX:b", "impossible ground speed")]
    assert load_sentences(tmp_path, "val", spec())["signal_index"].tolist() == [0]
    assert [r["dataset_id"] for r in labels["select"]["labelled"]] == ["KXXX:e"]


