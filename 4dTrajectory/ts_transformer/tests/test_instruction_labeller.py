"""The instruction labeller on synthetic flights whose sentence is known (design docs/two_tier/design/vocabulary.md §4)."""

from __future__ import annotations

import numpy as np
import pytest

from ts_transformer.instructions import grammar
from ts_transformer.instructions.airport import RunwayRelative, relative_to_runway
from ts_transformer.instructions.labeller.go_around import go_arounds, held_level, low_passes
from ts_transformer.instructions.labeller.interval import first_interval_row, in_force, interval_rows, on_interval
from ts_transformer.instructions.labeller.lateral import Approach, per_step_words, read_lateral
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.labeller.sentence import assemble
from ts_transformer.instructions.labeller.speed import read_speed
from ts_transformer.instructions.readout import go_around_in_force
from ts_transformer.instructions.labeller.vertical import LEVEL, MOVE, read_vertical, tube_checks, vertical_pieces
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, ANGLE_LEVEL, HEADING, RUNWAY, RUNWAY_GO_AROUND, SPEED, UNCHANGED, Words,
)
from ts_transformer.tests.support import (
    INSTRUCTION_STEP_S, PARALLEL_SPACING_M, fly_legs, instruction_airport, instruction_flight,
    instruction_spec as spec, parallel_airport, raised_airport,
)


def columns(reading, column):
    grid = reading.words
    return [(int(row), int(grid[row, column])) for row in np.nonzero(grid[:, column] != UNCHANGED)[0]]


#: A downwind west of runway 09 (course 090), a left turn onto a base south, a left turn onto the final east, a 3° descent.
DOWNWIND_BASE_FINAL = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
                       (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
GLIDE = 70.0 * np.tan(np.radians(3.0))
#: Level eastbound on the final of 09 at 900 m, a 3° descent to 247 m (147 m above the threshold, 960 m before it), a
#: climb straight ahead to 887 m, a level-off, a left 180° turn, a westbound downwind, a left 180° turn back onto the
#: final, a level stretch and a 3° descent to 400 m before the threshold: one go-around and a second approach.
GO_AROUND_LEGS = [(30, 0, 70, 0), (89, 0, 70, -GLIDE), (40, 0, 70, 8.0), (10, 0, 70, 0), (30, -6, 70, 0),
                  (170, 0, 70, 0), (30, -6, 70, 0), (20, 0, 70, 0), (104, 0, 70, -GLIDE)]


def test_downwind_base_final_reads_heading_words_relative_to_the_course_to_the_end():
    """§4.3 (D2, D8): the downwind is class 36 (the course's opposite), the turns onto the base and onto the final are
    read 5° at a time (4 s early) down to class 0, the course — to the last row; no clearance, the runway at row 0 only;
    the last descent is "no level-off" and the speed is "unspecified" from the capture row (D4)."""
    one, words = spec(), Words(spec())
    reading = read_flight(instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0)),
                          instruction_airport(), one, words)
    headings = [(row, words.heading_relative_deg(value)) for row, value in columns(reading, HEADING)]
    values = [value for _, value in headings]
    assert headings[0] == (0, 180.0) and values[-1] == 0.0 and -90.0 not in values and 90.0 in values
    assert values == sorted(values, reverse=True)                     # two left turns: 180 → 90 → 0
    # the base turn begins at row 60; the 6 s smoothing and the 2-row lead put its first word at row 59
    assert headings[1][0] == 59
    assert headings[-1][0] > reading.capture_row - 5                  # the turn onto the final is the model's words
    assert columns(reading, RUNWAY) == [(0, 0)] and reading.go_around_rows == []
    altitude = columns(reading, ALTITUDE)
    assert words.altitude_level_m(altitude[0][1]) == 1020.0 and altitude[-1][1] == words.altitude_no_level_off  # 1,080 m MSL
    angle = columns(reading, ANGLE)
    assert angle[0] == (0, ANGLE_LEVEL) and words.angle_bounds(angle[-1][1]) == (2.6, 3.7)
    assert all(h["inside"] == h["rows"] for h in reading.checks["heading"])     # the observed track, by construction
    assert sum(h["rows"] for h in reading.checks["heading"]) == len(reading.words) - 2   # every row from the lead on
    assert all(check["contained"] for check in reading.checks["vertical"])
    assert reading.checks["turning_deg"] == pytest.approx(180.0, abs=5.0)
    speed = columns(reading, SPEED)
    assert speed[-1] == (reading.unspecified_row, words.speed_unspecified)
    # "unspecified" at the capture row, or where a slowing begun less than a minimum hold before it began
    assert reading.capture_row - 10 <= reading.unspecified_row <= reading.capture_row


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


def _lateral(track, relative, course, one, words):
    """`read_lateral` on one approach to one runway of course ``course``."""
    return read_lateral(track, [relative], [course], [Approach(0, len(track), 0)], np.zeros(len(track), dtype=np.int64),
                        one, words)


def _per_step(track, capture, course, offset, **changes):
    one = spec(**changes)
    words = Words(one)
    return _lateral(track, _relative(len(track), capture, course, track, offset), course, one, words), words


# a slow continuous turn from north onto an eastbound final (course 090), 500 m left of it: 3° a row for 30 rows
CONTINUOUS = np.concatenate((np.full(30, 0.0), np.arange(1, 31) * 3.0, np.full(30, 90.0)))


def test_per_step_merges_the_rows_into_a_word_per_grid_cell_relative_to_the_course_to_the_end():
    """§3.3: the words are the run-length reading of every row's nearest grid value of track − course (here with no
    lead) — no hold, no split, no inserted intercept, and on to the last row."""
    reading, words = _per_step(CONTINUOUS, 70, 90.0, -500.0, heading_lead_s=0.0)
    heading = [(i.row, words.heading_relative_deg(i.value), i.kind) for i in reading.instructions if i.column == HEADING]
    cells = np.round((CONTINUOUS - 90.0) / 5.0) * 5.0
    expected = [(0, cells[0])] + [(r, cells[r]) for r in range(1, len(cells)) if cells[r] != cells[r - 1]]
    assert [(row, deg) for row, deg, _ in heading] == expected
    assert heading[-1][1] == 0.0 and {kind for _, _, kind in heading[1:]} == {"per-step"}
    assert all(i.info["target_deg"] == pytest.approx((90.0 + i.info["relative_deg"]) % 360.0)
               for i in reading.instructions)


def test_per_step_lead_labels_each_row_with_the_track_that_many_seconds_later():
    base, _ = _per_step(CONTINUOUS, 70, 90.0, -500.0, heading_lead_s=0.0)
    led, words = _per_step(CONTINUOUS, 70, 90.0, -500.0, heading_lead_s=4.0)
    rows = lambda reading: [(i.row, i.value) for i in reading.instructions if i.column == HEADING and i.row > 0]  # noqa: E731
    assert rows(led) == [(row - 2, value) for row, value in rows(base)]


def test_an_off_grid_course_is_held_by_class_0():
    """§3.3 reason 1 (KSTL's 122.3°): a final flown on the course reads as class 0 — no absolute 5° word would hold it."""
    track = np.full(60, 122.3)
    reading, words = _per_step(track, 0, 122.3, 0.0)
    assert [(i.row, i.value) for i in reading.instructions] == [(0, 0)]


def test_per_step_says_a_word_each_time_a_wander_crosses_a_cell_edge():
    """The boundary words are kept — a wander of ±2° about 2.4° right of the course crosses the 2.5° edge, and each
    crossing is a word (their average is the track the words describe)."""
    track = np.concatenate((92.4 + 2.0 * np.sin(np.arange(70) / 3.0), np.full(10, 90.0)))
    reading, words = _per_step(track, 70, 90.0, -200.0, heading_lead_s=0.0)
    values = [words.heading_relative_deg(i.value) for i in reading.instructions if i.column == HEADING]
    assert len(values) > 5 and set(values) == {0.0, 5.0}


def test_a_flight_on_the_final_from_row_0_is_captured_at_row_0():
    one, words = spec(), Words(spec())
    track = np.full(40, 91.0)
    reading = _lateral(track, _relative(40, 0, 90.0, track, 0.0), 90.0, one, words)
    assert reading.capture_rows == [0] and reading.turning_deg == 0.0
    assert [(i.column, i.row, i.value) for i in reading.instructions] == [(HEADING, 0, 0)]


def test_a_word_in_force_keeps_its_track_when_the_runway_changes_course():
    """§3.3, §5.4: the executor keeps a word's track when R changes; a new word is said only where the track leaves it,
    relative to the new course."""
    track = np.concatenate((np.full(20, 90.0), np.full(20, 100.0)))
    course = np.concatenate((np.full(10, 90.0), np.full(30, 80.0)))     # R changes course at row 10
    assert per_step_words(track, course, 5.0, 0) == [(0, 0.0), (20, 20.0)]


def test_an_approach_that_ends_at_a_go_around_off_the_corridor_has_no_capture_row():
    """D26: each approach its own capture row, to its own end; the landing approach must end on the final."""
    one, words = spec(), Words(spec())
    track = np.full(60, 90.0)
    relative = _relative(60, 40, 90.0, track, -500.0)                  # off the final before row 40
    approaches = [Approach(0, 20, 0), Approach(30, 60, 0)]
    reading = read_lateral(track, [relative], [90.0], approaches, np.zeros(60, dtype=np.int64), one, words)
    assert reading.capture_rows == [None, 40]
    with pytest.raises(Refused, match="not on the final at the end"):
        read_lateral(track, [relative], [90.0], [Approach(0, 30, 0)], np.zeros(60, dtype=np.int64), one, words)


def test_level_descend_level_descend_reads_targets_angles_and_no_level_off():
    one, words = spec(), Words(spec())
    distance = np.arange(400) * 150.0
    altitude = np.concatenate((np.full(100, 1500.0), 1500.0 - np.arange(1, 81) * 150.0 * np.tan(np.radians(2.1)),
                               np.full(120, 1500.0 - 80 * 150.0 * np.tan(np.radians(2.1)))))
    altitude = np.concatenate((altitude, altitude[-1] - np.arange(1, 101) * 150.0 * np.tan(np.radians(3.0))))
    reading = read_vertical(distance, altitude, one, words, [])
    targets = [(i.row, words.altitude_level_m(i.value)) for i in reading.instructions if i.column == ALTITUDE]
    checks = tube_checks(reading.instructions, distance, altitude, one, words)
    assert targets[0] == (0, 1500.0)                            # segment 2: 1,380 + 120 k
    assert targets[1][1] == 1080.0                              # the level at 1,060 m, nearest 60 m level
    assert targets[2][1] is None
    assert 97 <= targets[1][0] <= 101 and 297 <= targets[2][0] <= 301
    angles = [words.angle_deg(i.value) for i in reading.instructions if i.column == ANGLE]
    assert angles == [0.0, 2.1, 3.0]
    assert len(checks) == 3 and all(move["contained"] for move in checks)
    assert [round(c["tube_width_end_m"]) for c in checks][:2] == [140, 80]  # ε 70 m in segment 2, 40 m in segment 1


def test_a_level_between_two_grid_levels_is_found_and_named_by_the_nearest():
    """§4.4: the level test is the piece's own median ± 25 m, not the grid's band — a level at 630 m, 30 m from both
    neighbours, is a level (said as the lower, at a tie)."""
    one, words = spec(), Words(spec())
    distance = np.arange(80) * 150.0
    altitude = np.concatenate((np.full(40, 630.0), 630.0 - np.arange(1, 41) * 150.0 * np.tan(np.radians(3.0))))
    pieces = vertical_pieces(distance, altitude, one, words)
    assert pieces[0].kind == LEVEL and words.altitude_level_m(pieces[0].target_index) == 600.0 and pieces[0].stop >= 39
    reading = read_vertical(distance, altitude, one, words, [])
    assert [(i.column, i.row, i.value) for i in reading.instructions][:2] == [
        (ALTITUDE, 0, words.altitude_index(600.0)), (ANGLE, 0, ANGLE_LEVEL)]


def test_a_slow_descent_inside_one_450_m_step_is_not_a_level():
    """A 0.5° descent from 3,350 m to 3,150 m stays inside the band of the 3,150 m level (ε 235 m) all the way: it is
    still a move — the level test reads the piece, not the grid."""
    one, words = spec(), Words(spec())
    distance = np.arange(150) * 150.0
    altitude = 3350.0 - distance * np.tan(np.radians(0.5))
    pieces = vertical_pieces(distance, altitude, one, words)
    assert {piece.kind for piece in pieces} == {MOVE}


def test_a_move_between_two_heights_of_one_level_says_nothing():
    """§3.4: levels at 1,240 m and 1,280 m both round to 1,260 m — the move between them cannot be said."""
    one, words = spec(), Words(spec())
    distance = np.arange(126) * 150.0
    altitude = np.concatenate((np.full(60, 1280.0), 1280.0 - np.arange(1, 7) * 40.0 / 6.0, np.full(60, 1240.0)))
    reading = read_vertical(distance, altitude, one, words, [])
    assert [(i.column, i.row, i.value) for i in reading.instructions] == [
        (ALTITUDE, 0, words.altitude_index(1260.0)), (ANGLE, 0, ANGLE_LEVEL)]


def test_a_step_between_two_levels_is_issued_at_the_first_levels_end_with_its_own_direction():
    one, words = spec(), Words(spec())
    distance = np.arange(60) * 400.0
    altitude = np.concatenate((np.full(30, 900.0), np.full(30, 970.0)))
    reading = read_vertical(distance, altitude, one, words, [])
    assert [(i.column, i.row, i.value) for i in reading.instructions] == [
        (ALTITUDE, 0, words.altitude_index(900.0)), (ANGLE, 0, ANGLE_LEVEL),
        (ALTITUDE, 29, words.altitude_index(960.0)), (ANGLE, 29, words.angle_climb)]
    grid, _ = assemble(60, [Instruction(RUNWAY, 0, 0, "i"), Instruction(HEADING, 18, 0, "i"),
                            Instruction(SPEED, 16, 0, "i"), *reading.instructions], altitude, words, [90.0])
    assert grid[29, ANGLE] == words.angle_climb


def test_a_flight_climbing_at_the_end_is_refused():
    one, words = spec(), Words(spec())
    altitude = np.concatenate((np.full(30, 900.0), 900.0 + np.arange(1, 31) * 3.0))
    with pytest.raises(Refused, match="climbing at the end"):
        read_vertical(np.arange(60) * 200.0, altitude, one, words, [])


def test_the_speed_is_unspecified_from_the_capture_row_unless_a_hold_ends_far_enough_out():
    """§4.5 (D4): "unspecified" from the capture row (here 60), unless a hold of 30 s ending at or after it ends 9,260 m
    or more out (then from that hold's end), and from where a slowing under way at that row began, if less than a
    minimum hold before it."""
    one, words = spec(), Words(spec())
    time = np.arange(200) * INSTRUCTION_STEP_S
    speed = np.concatenate((np.full(50, 120.0), 120.0 - np.arange(1, 31) * 1.0, np.full(60, 90.0),
                            90.0 - np.arange(1, 31) * 0.5, np.full(30, 75.0)))
    far = np.linspace(40000.0, 300.0, 200)
    near = np.linspace(9000.0, 300.0, 200)
    kept = read_speed(time, speed, far, 60, one, words)
    assert [words.speed_mps(i.value) for i in kept.instructions] == [120.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0, None]
    # the 90 m/s hold ends 9.3 km or more out; the deceleration's first rows still fit the hold's
    # straight piece within the 1.5 m/s fit tolerance
    assert 140 <= kept.unspecified_row <= 145
    dropped = read_speed(time, speed, near, 60, one, words)
    assert [words.speed_mps(i.value) for i in dropped.instructions] == [120.0, None]
    assert 50 <= dropped.unspecified_row <= 53
    # a capture row inside the 90 m/s hold, no hold ending far out: "unspecified" from the capture row itself
    assert read_speed(time, speed, near, 100, one, words).unspecified_row == 100


def test_unspecified_past_the_approach_s_last_row_is_refused_by_name():
    """A32: a hold that runs to the approach's last row and ends far enough out would put "unspecified" after its last
    row; it is refused by name — and an instruction outside the sentence's rows is refused by name by the assembly —
    never an `IndexError`."""
    from ts_transformer.instructions.labeller.sentence import assemble

    one, words = spec(), Words(spec())
    speed = np.full(200, 90.0)
    time = np.arange(len(speed)) * INSTRUCTION_STEP_S
    with pytest.raises(Refused, match="unspecified past the approach: row 200 of an approach of 200 rows"):
        read_speed(time, speed, np.linspace(40000.0, 20000.0, len(speed)), 60, one, words)
    first = [Instruction(column, 0, 0, "initial") for column in range(5)]
    with pytest.raises(Refused, match="instruction outside the sentence: speed at row 3 of 3"):
        assemble(3, [*first, Instruction(4, 1, 3, "unspecified")], np.full(3, 900.0), words, [90.0])


# ---- speed words in steps (D43)
def _speeds(reading, words):
    return [(i.row, words.speed_mps(i.value)) for i in reading.instructions]


def test_a_deceleration_of_40_mps_from_hold_to_hold_says_its_eight_steps_where_the_speed_comes_nearer_to_each():
    """§4.5 rule 3 (D43): each grid value from the word in force to the target, in order, at the first row of the run
    where the smoothed speed is nearer to it than to the value before; the last is the next hold's value."""
    one, words = spec(), Words(spec())
    speed = np.concatenate((np.full(40, 130.0), 130.0 - np.arange(1, 41) * 1.0, np.full(40, 90.0)))
    time = np.arange(len(speed)) * INSTRUCTION_STEP_S
    reading = read_speed(time, speed, np.full(len(speed), 40000.0), None, one, words)
    said = _speeds(reading, words)
    assert [v for _, v in said] == [130.0, 125.0, 120.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0]
    (run,) = [piece for piece in reading.pieces if piece.kind == "transition"]
    for (row, value), (_, before) in zip(said[1:], said[:-1]):
        nearer = [r for r in range(run.start, len(speed)) if abs(speed[r] - value) < abs(speed[r] - before)]
        assert row == nearer[0]


def test_a_run_that_turns_without_a_hold_steps_to_the_grid_value_nearest_the_turn_and_noise_says_nothing_back():
    one, words = spec(), Words(spec())
    down = 100.0 - np.arange(1, 15) * 1.0                              # to 86 m/s, then back up without a hold
    noise = np.where(np.arange(30) % 2, 0.6, -0.6)                     # inside the fit tolerance
    speed = np.concatenate((np.full(30, 100.0), down, down[::-1][1:], [100.0], np.full(30, 100.0) + noise))
    time = np.arange(len(speed)) * INSTRUCTION_STEP_S
    said = _speeds(read_speed(time, speed, np.full(len(speed), 40000.0), None, one, words), words)
    assert [v for _, v in said] == [100.0, 95.0, 90.0, 85.0, 90.0, 95.0, 100.0]
    wobbly = 110.0 - np.arange(60) * 0.5 + np.where(np.arange(60) % 2, 1.2, -1.2)    # one deceleration with noise
    speed = np.concatenate((np.full(30, 110.0), wobbly, np.full(30, 80.0)))
    said = _speeds(read_speed(np.arange(len(speed)) * INSTRUCTION_STEP_S, speed, np.full(len(speed), 40000.0), None,
                              one, words), words)
    values = [v for _, v in said]
    assert values == sorted(values, reverse=True) and values[-1] == 80.0 and len(set(values)) == len(values)


def _first_step(words):
    return [Instruction(RUNWAY, 0, 0, "i"), Instruction(HEADING, 18, 0, "i"), Instruction(ALTITUDE, 15, 0, "i"),
            Instruction(ANGLE, ANGLE_LEVEL, 0, "i"), Instruction(SPEED, 16, 0, "i")]


def test_the_assembly_drops_repeats_and_refuses_conflicts_and_an_empty_first_step():
    words = Words(spec())
    altitude = np.full(10, 900.0)
    base = _first_step(words)
    grid, kept = assemble(10, [*base, Instruction(HEADING, 18, 4, "repeat")], altitude, words, [90.0])
    assert grid[4, HEADING] == UNCHANGED and len(kept) == 5 and grid.shape == (10, 5)
    with pytest.raises(Refused, match="two instructions in one step"):
        assemble(10, [*base, Instruction(HEADING, 20, 0, "clash")], altitude, words, [90.0])
    with pytest.raises(Refused, match="two instructions in one step"):
        assemble(10, [*base, Instruction(HEADING, 18, 4, "repeat"), Instruction(HEADING, 30, 4, "other")], altitude,
                 words, [90.0])
    with pytest.raises(Refused, match="first step incomplete"):
        assemble(10, base[:-1], altitude, words, [90.0])
    with pytest.raises(Refused, match=r"first step incomplete: no word for \['runway'\]"):
        assemble(10, base[1:], altitude, words, [90.0])
    with pytest.raises(Refused, match=r"first step incomplete: no word for \['heading'\]"):
        assemble(10, [base[0], *base[2:], Instruction(HEADING, 18, 4, "later")], altitude, words, [90.0])
    with pytest.raises(Refused, match="altitude and angle incompatible"):
        assemble(10, [*base, Instruction(ALTITUDE, 5, 3, "descend with a level angle")], altitude, words, [90.0])


def test_the_runway_said_again_after_a_go_around_is_kept():
    words = Words(spec())
    instructions = [*_first_step(words), Instruction(RUNWAY, RUNWAY_GO_AROUND, 3, "go-around"),
                    Instruction(RUNWAY, 0, 6, "runway again")]
    grid, kept = assemble(10, instructions, np.full(10, 900.0), words, [90.0])
    assert [(int(r), int(grid[r, RUNWAY])) for r in np.nonzero(grid[:, RUNWAY] != UNCHANGED)[0]] == [
        (0, 0), (3, RUNWAY_GO_AROUND), (6, 0)]


# ---- the grammar (§3.2, §3.7)
def _state(**changes):
    return grammar.InForce(**{"runway": 0, "go_around": False, "heading": 0, "altitude": 15, "angle": ANGLE_LEVEL,
                              "speed": 16, **changes})


def _step(**said):
    row = [UNCHANGED] * 5
    for name, value in said.items():
        row[{"runway": RUNWAY, "heading": HEADING, "altitude": ALTITUDE, "angle": ANGLE, "speed": SPEED}[name]] = value
    return row


@pytest.mark.parametrize("in_force, step, allowed", [
    # the first predicted step: a candidate and every column (rule 1)
    (None, [0, 0, 15, ANGLE_LEVEL, 16], True),
    (None, [RUNWAY_GO_AROUND, 0, 15, ANGLE_LEVEL, 16], False),
    (None, [UNCHANGED, 0, 15, ANGLE_LEVEL, 16], False),
    (None, [5, 0, 15, ANGLE_LEVEL, 16], False),                    # not a candidate of 2
    # G false: another candidate (no runway lock, D12), never the one in force, a go-around
    ({}, {"runway": 1}, True),
    ({}, {"runway": 0}, False),
    ({}, {"runway": RUNWAY_GO_AROUND}, True),
    # G true: any candidate ends it, R included; no second go-around
    ({"go_around": True}, {"runway": 0}, True),
    ({"go_around": True}, {"runway": 1}, True),
    ({"go_around": True}, {"runway": RUNWAY_GO_AROUND}, False),
])
def test_every_row_of_the_runway_table(in_force, step, allowed):
    words = Words(spec())
    state = None if in_force is None else _state(**in_force)
    row = step if isinstance(step, list) else _step(**step)
    assert grammar.step_allowed(state, row, 900.0, words, 2) is allowed


def test_the_runway_word_ends_the_go_around_and_keeps_r_through_it():
    words = Words(spec())
    after = grammar.apply(_state(runway=1), _step(runway=RUNWAY_GO_AROUND), 900.0, words, 2)
    assert (after.runway, after.go_around) == (1, True)
    assert grammar.apply(after, _step(runway=0), 900.0, words, 2) == _state(runway=0)


def test_rule_5_no_level_off_waits_for_the_runway_word():
    """D14: "no level-off" is not said while G is true; the runway word, first in its row, ends G for it."""
    words = Words(spec())
    descending = {"altitude": words.altitude_no_level_off, "angle": 3}
    assert not grammar.step_allowed(_state(go_around=True), _step(**descending), 900.0, words, 1)
    assert grammar.step_allowed(_state(go_around=True), _step(runway=0, **descending), 900.0, words, 1)
    assert grammar.step_allowed(_state(), _step(**descending), 900.0, words, 1)


def test_rule_6_a_go_around_on_a_final_descent_says_a_level_above_and_the_climb():
    """D27: "go-around" while "no level-off" is in force also says a level more than its band above the aircraft, and
    rule 3 makes that row climb; with a level in force "go-around" alone is a sentence."""
    words = Words(spec())
    final = _state(altitude=words.altitude_no_level_off, angle=3)
    above, near = words.altitude_index(1200.0), words.altitude_index(900.0)
    go = {"runway": RUNWAY_GO_AROUND}
    assert grammar.step_allowed(final, _step(**go, altitude=above, angle=words.angle_climb), 900.0, words, 1)
    with pytest.raises(grammar.Ungrammatical, match="go-around without a level above"):
        grammar.apply(final, _step(**go), 900.0, words, 1)
    with pytest.raises(grammar.Ungrammatical, match="go-around without a level above"):
        grammar.apply(final, _step(**go, altitude=near, angle=words.angle_climb), 900.0, words, 1)   # inside its band
    with pytest.raises(grammar.Ungrammatical, match="altitude and angle incompatible"):
        grammar.apply(final, _step(**go, altitude=above), 900.0, words, 1)                     # still descending
    assert grammar.step_allowed(_state(), _step(**go), 900.0, words, 1)


def test_rules_3_and_4_the_level_and_the_angle_agree():
    words = Words(spec())
    lower = words.altitude_index(600.0)
    assert not grammar.step_allowed(_state(), _step(altitude=lower), 900.0, words, 1)             # level angle
    assert grammar.step_allowed(_state(), _step(altitude=lower, angle=2), 900.0, words, 1)
    assert not grammar.step_allowed(_state(), _step(altitude=lower, angle=words.angle_climb), 900.0, words, 1)
    # within the level's band no direction is needed: 900 ± 40 m
    assert grammar.step_allowed(_state(), _step(altitude=words.altitude_index(900.0)), 935.0, words, 1)
    assert not grammar.step_allowed(_state(), _step(altitude=words.altitude_no_level_off), 900.0, words, 1)  # rule 4


def test_each_broken_rule_says_what_it_found():
    """The labeller counts refusals by reason; the detail says which of the rules that share a reason broke."""
    words = Words(spec())
    for in_force, step, found in (
            (None, _step(runway=0), "first step incomplete: no word for ['heading', 'altitude', 'angle', 'speed']"),
            (None, [5, 0, 15, ANGLE_LEVEL, 16], "runway word not permitted: the first step says 5, not a candidate"),
            (_state(), _step(runway=5), "runway word not permitted: 5 is not a candidate of 2"),
            (_state(), _step(runway=0), "runway word not permitted: runway 0 is already in force"),
            (_state(go_around=True), _step(runway=RUNWAY_GO_AROUND),
             "runway word not permitted: go-around while a go-around is in force")):
        with pytest.raises(grammar.Ungrammatical) as refused:
            grammar.apply(in_force, step, 900.0, words, 2)
        assert str(refused.value) == found
    top = words.altitude_no_level_off
    for outside in (-2, top + 1):
        with pytest.raises(ValueError, match=f"altitude class {outside} outside 0..{top}"):
            grammar.apply(_state(), _step(altitude=outside), 900.0, words, 2)
        with pytest.raises(ValueError, match=f"altitude class {outside} outside 0..{top}"):
            grammar.column_mask([_state()], np.array([[UNCHANGED, UNCHANGED, outside]]), ANGLE, np.array([900.0]), [2],
                                words)


def test_a_height_that_is_not_a_number_compares_with_nothing():
    """A height that is not a number makes every comparison with it false, as `apply` always read it: no direction is
    needed (rule 3), and a level said with "go-around" is not "not above" it (rule 6)."""
    words = Words(spec())
    nan = float("nan")
    assert grammar.step_allowed(_state(), _step(altitude=words.altitude_index(600.0)), nan, words, 1)
    final = _state(altitude=words.altitude_no_level_off, angle=3)
    go = _step(runway=RUNWAY_GO_AROUND, altitude=words.altitude_index(1200.0), angle=words.angle_climb)
    assert grammar.step_allowed(final, go, nan, words, 1)
    with pytest.raises(grammar.Ungrammatical, match="go-around without a level above"):
        grammar.apply(final, _step(runway=RUNWAY_GO_AROUND), nan, words, 1)


def _small_spec():
    """A vocabulary small enough to say every row: 3 heading classes, 3 levels (0, 600, 1,200 m), 2 descent classes,
    3 speeds."""
    return spec(heading_step_deg=120.0, heading_tolerance_deg=62.0, altitude_segment_steps_m=(600.0,),
                altitude_segment_tops_m=(1200.0,), descent_angle_edges_deg=(-0.5, 2.5, 10.0),
                descent_angle_centres_deg=(1.5, 3.0), speed_min_mps=20.0, speed_max_mps=30.0)


def _completions_pass(in_force, prefix, column, height, words, candidates, permitted):
    """D62's definition: for each word of ``column``, whether some words of the later columns, each among ``permitted``
    (`column_words` order), make the row pass `apply` — every completion tried."""
    import itertools

    later = range(column + 1, 5)
    choices = [[v for v, ok in zip(grammar.column_words(c, words, candidates), permitted[c]) if ok] for c in later]
    out = []
    for word in grammar.column_words(column, words, candidates):
        out.append(any(grammar.step_allowed(in_force, [*prefix, word, *rest], height, words, candidates)
                       for rest in itertools.product(*choices)))
    return out


def _all_permitted(words, candidates, column, rng=None):
    sizes = {c: len(grammar.column_words(c, words, candidates)) for c in range(column + 1, 5)}
    if rng is None:
        return {c: np.ones(n, bool) for c, n in sizes.items()}
    return {c: rng.random(n) < 0.5 for c, n in sizes.items()}


def test_the_column_mask_is_every_completion_through_apply_on_every_row_of_a_small_vocabulary():
    """D62: on a small vocabulary, every state in force (and none: the first step), every word of the earlier columns,
    three heights, every column: the mask is the definition — some completion of the later columns passes `apply` —
    with every later word permitted and with random permitted words."""
    import itertools

    words, candidates = Words(_small_spec()), 2
    rng = np.random.default_rng(7)
    states = [None] + [grammar.InForce(r, g, 0, a, angle, 0) for r in range(candidates) for g in (False, True)
                       for a in range(words.altitude_no_level_off + 1) for angle in range(words.n_descent + 2)]
    checked = 0
    for column in range(5):
        prefixes = list(itertools.product(*(grammar.column_words(c, words, candidates) for c in range(column))))
        for state in states:
            for height in (100.0, 620.0, 1250.0):
                for draw in (None, rng):
                    permitted = _all_permitted(words, candidates, column, draw)
                    mask = grammar.column_mask([state] * len(prefixes), np.array(prefixes).reshape(len(prefixes), column),
                                               column, np.full(len(prefixes), height), [candidates] * len(prefixes),
                                               words, {c: np.tile(p, (len(prefixes), 1)) for c, p in permitted.items()})
                    for k, prefix in enumerate(prefixes):
                        expected = _completions_pass(state, list(prefix), column, height, words, candidates, permitted)
                        assert mask[k].tolist() == expected, (column, state, height, prefix)
                        checked += 1
    assert checked > 20_000


def test_the_column_mask_is_every_completion_through_apply_on_random_rows_of_the_chosen_vocabulary():
    """D62 on the spec of D59 (`instruction_spec`'s grid and classes): random states, earlier words and heights, a batch
    of aircraft with different numbers of candidates; with every later word permitted where the completions are few
    enough to try them all, and with a few random permitted words in every column."""
    words = Words(spec())
    rng = np.random.default_rng(11)
    for column in range(5):
        for few in (False, True):
            if column < 2 and not few:
                continue                                       # 72 × 42 × 6 × 48 completions: too many to try
            batch = 6
            candidates = [int(rng.integers(1, 4)) for _ in range(batch)]
            in_force = [None if rng.random() < 0.2 else grammar.InForce(
                int(rng.integers(0, n)), bool(rng.random() < 0.3), int(rng.integers(0, words.n_heading)),
                int(rng.integers(0, words.altitude_no_level_off + 1)), int(rng.integers(0, words.n_descent + 2)),
                int(rng.integers(0, words.speed_unspecified + 1))) for n in candidates]
            prefix = np.array([[int(rng.choice(grammar.column_words(c, words, n))) for c in range(column)]
                               for n in candidates]).reshape(batch, column)
            height = rng.uniform(0.0, 3000.0, batch)
            most = max(candidates)
            permitted = {c: (rng.random((batch, len(grammar.column_words(c, words, most)))) < 4.0 / len(
                grammar.column_words(c, words, most))) if few else np.ones((batch, len(grammar.column_words(c, words, most))),
                                                                          bool) for c in range(column + 1, 5)}
            mask = grammar.column_mask(in_force, prefix, column, height, candidates, words, permitted)
            assert mask.shape == (batch, len(grammar.column_words(column, words, most)))
            for j in range(batch):
                mine = {c: p[j][: len(grammar.column_words(c, words, candidates[j]))] for c, p in permitted.items()}
                expected = _completions_pass(in_force[j], prefix[j].tolist(), column, height[j], words, candidates[j],
                                             mine)
                own = len(grammar.column_words(column, words, candidates[j]))
                assert mask[j, :own].tolist() == expected, (column, few, j)
                assert not mask[j, own:].any()                 # a candidate the airport does not have


def test_a_level_below_is_permitted_under_a_level_angle_and_then_only_a_descent_class():
    """D62: with the level angle in force, a level below the aircraft is permitted (a descent class in the angle column
    makes the row grammatical); said, the angle column permits only the descent classes. When the caller permits no
    descent class, that level is not permitted."""
    words = Words(spec())
    lower = words.altitude_index(600.0)
    state = _state()                                              # level 900 m, the level angle
    altitude = grammar.column_mask([state], np.array([[UNCHANGED, UNCHANGED]]), ALTITUDE, np.array([900.0]), [1],
                                   words)[0]
    assert altitude[lower + 1]                                    # position 0 is "unchanged"
    angle = grammar.column_mask([state], np.array([[UNCHANGED, UNCHANGED, lower]]), ANGLE, np.array([900.0]), [1],
                                words)[0]
    descents = {c + 1 for c in range(1, words.n_descent + 1)}
    assert {k for k, ok in enumerate(angle) if ok} == descents
    no_descent = np.ones(words.n_descent + 3, bool)
    no_descent[[c + 1 for c in range(1, words.n_descent + 1)]] = False
    blocked = grammar.column_mask([state], np.array([[UNCHANGED, UNCHANGED]]), ALTITUDE, np.array([900.0]), [1], words,
                                  {ANGLE: no_descent[None, :]})[0]
    assert not blocked[lower + 1] and blocked[words.altitude_index(900.0) + 1]


# ---- the altitude words above the airport elevation (D58)
def test_two_airports_of_different_elevation_say_one_word_for_one_height_above_it():
    """D58: the same flight 128 m higher at an airport 128 m higher (E = 188 m, the highest threshold of the training
    airports) reads the same sentence — a go-around included, so grammar rules 3 and 6 are read on the height above E;
    read at the MSL heights instead, the same words break them."""
    from ts_transformer.instructions.labeller.sentence import check_grammar

    one, words = spec(), Words(spec())
    low = instruction_airport()
    high = raised_airport(low, 188.0)
    rise = high.elevation_m - low.elevation_m
    e, n, altitude, track, speed = fly_legs(GO_AROUND_LEGS, 90.0, 900.0, -400.0, 0.0)
    first = read_flight(instruction_flight(e, n, altitude, track, speed), low, one, words)
    second = read_flight(instruction_flight(e, n, altitude + rise, track, speed), high, one, words)
    assert second.go_around_rows == first.go_around_rows and len(first.go_around_rows) == 1
    assert np.array_equal(second.words, first.words)
    assert second.held_height_m == pytest.approx(first.held_height_m, abs=1e-6)
    assert words.altitude_level_m(int(second.words[0, ALTITUDE])) == 840.0          # 1,028 m MSL at E = 188 m
    with pytest.raises(Refused, match="altitude and angle incompatible"):
        check_grammar(second.words, second.held_height_m + high.elevation_m, words, 1)


#: `GO_AROUND_LEGS` with the climb in two steps: 240 m, a level of 24 s, then 400 m — the first level after the
#: go-around is 240 m above its lowest point, between 150 m and 150 m + E at an airport of E = 188 m.
GO_AROUND_STEP_LEGS = [*GO_AROUND_LEGS[:2], (20, 0, 70, 6.0), (12, 0, 70, 0), (18, 0, 70, 400.0 / 36.0),
                       *GO_AROUND_LEGS[4:]]


def test_the_runway_word_after_a_go_around_reads_the_climb_above_the_airport_elevation():
    """D19, D58: the runway word that ends a go-around is said at the first level at least 150 m above the go-around's
    lowest point, both heights above E — at E = 188 m the level 240 m up is it (read on MSL against heights above E,
    it would not be: 240 < 150 + 188)."""
    one, words = spec(), Words(spec())
    high = raised_airport(instruction_airport(), 188.0)
    e, n, altitude, track, speed = fly_legs(GO_AROUND_STEP_LEGS, 90.0, 1028.0, -400.0, 0.0)
    reading = read_flight(instruction_flight(e, n, altitude, track, speed), high, one, words)
    (go,), (again,) = reading.go_around_rows, reading.runway_again_rows
    assert go < again <= go + 25                                     # the 24 s level after the first 240 m


def test_rule_6_reads_the_height_it_is_given():
    """§3.7 rule 6 with D58: a go-around while "no level-off" is in force says a level more than its ε above the
    aircraft — 300 m against 200 m above E passes; the same rows read 188 m higher (MSL at E = 188 m) do not."""
    from ts_transformer.instructions.labeller.sentence import check_grammar

    words = Words(spec())
    grid = np.full((2, 5), UNCHANGED, dtype=np.int64)
    grid[0] = [0, 0, words.altitude_no_level_off, 3, words.speed_unspecified]
    grid[1, RUNWAY], grid[1, ALTITUDE], grid[1, ANGLE] = RUNWAY_GO_AROUND, words.altitude_index(300.0), words.angle_climb
    check_grammar(grid, np.array([250.0, 200.0]), words, 1)
    with pytest.raises(Refused, match="go-around without a level above"):
        check_grammar(grid, np.array([250.0, 200.0]) + 188.0, words, 1)


# ---- go-arounds (§4.6, D18, D19)
def test_a_go_around_and_a_second_approach():
    """D18, D26: "go-around" in the row of the climb's words — its level and "climb" — the first row of the climb after
    the low pass; the descent to it is the first approach's last, "no level-off", and the first approach (on the final
    from row 0) is "unspecified" from its capture row, so both are in force at the go-around row; D19: the landed runway
    said again at the first level-off after the climb, before the final's "no level-off" (rule 5 holds)."""
    one, words = spec(), Words(spec())
    e, n, altitude, track, speed = fly_legs(GO_AROUND_LEGS, 90.0, 900.0, -400.0, 0.0)
    reading = read_flight(instruction_flight(e, n, altitude, track, speed), instruction_airport(), one, words)
    (go,), (again,) = reading.go_around_rows, reading.runway_again_rows
    assert abs(go - 119) <= 2                                        # the climb starts at row 119, 960 m out
    assert columns(reading, RUNWAY) == [(0, 0), (go, RUNWAY_GO_AROUND), (again, 0)]
    assert words.altitude_level_m(int(reading.words[go, ALTITUDE])) == 840.0 and reading.words[go, ANGLE] == words.angle_climb
    # 900 m MSL: 840 m above E
    held = in_force(reading.words)
    assert held[go - 1, ALTITUDE] == words.altitude_no_level_off and held[go - 1, SPEED] == words.speed_unspecified
    assert 158 <= again <= 164                                       # the level-off after the climb (row 159)
    final = [row for row, value in columns(reading, ALTITUDE) if value == words.altitude_no_level_off]
    assert len(final) == 2 and final[0] < go and final[1] > again    # each approach's own
    assert reading.checks["go_arounds"][0]["drop_m"] >= 150.0 and reading.checks["go_arounds"][0]["climb_m"] >= 150.0
    first, second = reading.approaches
    assert (first.first, first.end, first.capture_row, first.unspecified_row) == (0, go, 0, 0)
    assert second.first == again and second.end == len(reading.words) and second.capture_row > again
    assert reading.capture_row == second.capture_row                 # the landing approach's
    # the second approach's speed reading starts at the go-around row: the speed held is said there
    assert reading.words[go, SPEED] == words.speed_index(70.0)
    assert go_around_in_force(reading, words) == [{"no_level_off": True, "unspecified": True, "captured": True,
                                                    "other_runway": False, "past_threshold": False}]


def test_a_go_around_on_one_runway_and_a_landing_on_another():
    """D26, §4.2: the first row says the runway of the first low pass (09L, the parallel), the runway word that ends the
    go-around the runway the flight landed on (09); the heading words read each one's course."""
    one, words = spec(), Words(spec())
    legs = [*GO_AROUND_LEGS[:6], (40, -4.5, 70, 0), *GO_AROUND_LEGS[7:]]   # a wider second turn: 891 m further south
    e, n, altitude, track, speed = fly_legs(legs, 90.0, 900.0, -400.0, 0.0)
    assert n[100] == pytest.approx(PARALLEL_SPACING_M, abs=1.0)       # the first final: 09L's centreline
    airport = parallel_airport()
    reading = read_flight(instruction_flight(e, n, altitude, track, speed), airport, one, words)
    (go,), (again,) = reading.go_around_rows, reading.runway_again_rows
    left, landed = [candidate.ident for candidate in airport.candidates].index("09L"), 0
    assert columns(reading, RUNWAY) == [(0, left), (go, RUNWAY_GO_AROUND), (again, landed)]
    assert [(a.runway_index, a.capture_row is not None) for a in reading.approaches] == [(left, True), (landed, True)]
    assert reading.runway_index == landed and reading.checks["go_arounds"][0]["candidate"] == left
    assert go_around_in_force(reading, words)[0]["other_runway"]


def test_a_go_around_without_a_climb_word_is_refused():
    """D26: the go-around row is the row of the climb word; a low point with no climb after it is refused."""
    one, words = spec(), Words(spec())
    distance = np.arange(80) * 150.0
    altitude = np.concatenate((np.full(40, 900.0), 900.0 - np.arange(1, 41) * 150.0 * np.tan(np.radians(3.0))))
    with pytest.raises(Refused, match="go-around without a climb word"):
        read_vertical(distance, altitude, one, words, [79])


LOW_OVER_THE_RUNWAY = [(30, 0, 70, 0), (102, 0, 70, -GLIDE), (40, 0, 70, 8.0), (10, 0, 70, 0), (30, -6, 70, 0),
                      (162, 0, 70, 0), (30, -6, 70, 0), (20, 0, 70, 0), (99, 0, 70, -GLIDE)]


@pytest.mark.parametrize("short", [True, False])
def test_a_low_pass_over_the_runway_followed_by_a_second_approach_is_read_not_refused(short):
    """§4.6 item 5: a go-around that crosses the threshold low (a landing as the harvest judges one) and comes back is
    read with the go-around rule. A real series ends short of its landing (the data plane's cut): then it holds the low
    crossing only; a series that crosses again is cut at that second crossing, the one it never comes back from."""
    e, n, altitude, track, speed = fly_legs(LOW_OVER_THE_RUNWAY, 90.0, 900.0, 1000.0, 0.0)
    crossings = np.nonzero((e[:-1] < 0.0) & (e[1:] >= 0.0))[0] + 1
    assert len(crossings) == 2 and altitude[crossings[0]] - 100.0 < 100.0   # the go-around crosses under 100 m
    if short:                                                                 # ended 10 rows before the landing
        e, n, altitude, track, speed = (a[: crossings[1] - 10] for a in (e, n, altitude, track, speed))
    reading = read_flight(instruction_flight(e, n, altitude, track, speed), instruction_airport(), spec())
    if short:
        assert not reading.cut_at_crossing and len(reading.words) == len(e)
    else:
        assert reading.cut_at_crossing and len(reading.words) == crossings[1]
    assert len(reading.go_around_rows) == 1 and abs(reading.go_around_rows[0] - crossings[0]) <= 2


def test_the_runway_word_after_a_go_around_waits_for_the_level_off_after_the_climb():
    """D19, D26: a missed approach that first flies level at its low point (early, to the missed approach point): the
    go-around row is where the climb is said, after that level, and the runway is said again at the level held after
    the climb, not at the low one. The descent ends at the low level, so it is no final descent ("no level-off")."""
    # down 15 m under the low level and back within 4 s, so the lowest row comes before the level piece begins
    legs = [(30, 0, 70, 0), (78, 0, 70, -GLIDE), (2, 0, 70, 4.0), (15, 0, 70, 0), (40, 0, 70, 8.0), (10, 0, 70, 0),
            (30, -6, 70, 0), (170, 0, 70, 0), (30, -6, 70, 0), (20, 0, 70, 0), (104, 0, 70, -GLIDE)]
    words = Words(spec())
    reading = read_flight(instruction_flight(*fly_legs(legs, 90.0, 900.0, -400.0, 0.0)), instruction_airport(), spec(),
                          words)
    (go,), (again,) = reading.go_around_rows, reading.runway_again_rows
    assert 124 <= go <= 127 and again >= go + 38                     # the climb starts at row 125, lasts 40 rows
    assert in_force(reading.words)[go - 1, ALTITUDE] != words.altitude_no_level_off


def test_a_go_around_needs_a_held_level_before_and_after():
    """A dip on the final and a shallow level-off are not a go-around: no level held 150 m above the low point after it."""
    legs = [(30, 0, 70, 0), (89, 0, 70, -GLIDE), (10, 0, 70, 4.0), (30, -6, 70, 0), (170, 0, 70, 0), (30, -6, 70, 0),
            (200, 0, 70, -GLIDE * 0.15)]
    e, n, altitude, track, speed = fly_legs(legs, 90.0, 900.0, -400.0, 0.0)
    relatives = [relative_to_runway(e, n, track, altitude, candidate) for candidate in instruction_airport().candidates]
    time = np.arange(len(e)) * INSTRUCTION_STEP_S
    assert low_passes(relatives, spec())                              # there is a low pass ...
    assert go_arounds(time, altitude, relatives, spec()) == []        # ... but no go-around


def test_held_level_is_the_highest_height_held_for_the_hold():
    time = np.arange(30) * 2.0
    altitude = np.concatenate((np.full(10, 500.0), np.full(10, 700.0), np.full(10, 300.0)))
    # 700 m is held 18 s, under the 20 s hold; 500 m is held 20 s (its 10 rows and the next)
    assert held_level(time, altitude, 0, 30, 20.0) == 500.0
    assert held_level(time, altitude, 0, 30, 18.0) == 700.0
    assert held_level(time, altitude, 0, 5, 20.0) == -np.inf


def test_a_touch_and_go_is_refused():
    """A low pass past the threshold under 15 m is on the runway: a landing the vocabulary cannot say."""
    legs = [(30, 0, 70, 0), (110, 0, 70, -GLIDE), (12, 0, 70, -1.0), (40, 0, 70, 8.0), (10, 0, 70, 0),
            (30, -6, 70, 0), (200, 0, 70, 0), (30, -6, 70, 0), (20, 0, 70, 0), (104, 0, 70, -GLIDE)]
    e, n, altitude, track, speed = fly_legs(legs, 90.0, 900.0, -400.0, 0.0)
    low = int(np.argmin(altitude[:200]))
    assert e[low] > 0.0 and altitude[low] - 100.0 < 15.0
    with pytest.raises(Refused, match="touch-and-go"):
        read_flight(instruction_flight(e, n, altitude, track, speed), instruction_airport(), spec())


@pytest.mark.parametrize("altitude, refused", [(150.0, True), (600.0, False)])
def test_a_low_pass_the_flight_never_comes_back_from_is_refused_and_an_overflight_is_no_landing(altitude, refused):
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


# ---- the row interval (§4.8, D11, D25)
def test_the_row_interval_divides_the_observation():
    assert [interval_rows(seconds, 2.0) for seconds in (2.0, 4.0, 8.0)] == [1, 2, 4]
    for seconds in (6.0, 3.0, 0.0, 16.0 + 2.0):
        with pytest.raises(ValueError):
            interval_rows(seconds, 2.0)
    assert interval_rows(16.0, 2.0) == 8


def test_the_first_interval_row_is_on_a_utc_multiple():
    assert first_interval_row("2026-06-01T11:00:00Z", 4.0, 2.0) == 0
    assert first_interval_row("2026-06-01T11:00:02Z", 4.0, 2.0) == 1
    assert first_interval_row("2026-06-01T11:00:02Z", 8.0, 2.0) == 3
    with pytest.raises(ValueError):
        first_interval_row("2026-06-01T11:00:01Z", 4.0, 2.0)


def test_at_the_data_step_the_sentence_comes_back():
    words = Words(spec())
    flight = instruction_flight(*fly_legs(GO_AROUND_LEGS, 90.0, 900.0, -400.0, 0.0))
    reading = read_flight(flight, instruction_airport(), spec(), words)
    assert np.array_equal(on_interval(reading.words, 0, 2.0, 2.0, reading.held_height_m, words, [90.0]), reading.words)


def _grid(rows):
    grid = np.full((len(rows), 5), UNCHANGED, dtype=np.int16)
    for row, said in enumerate(rows):
        for column, value in said.items():
            grid[row, column] = value
    return grid


#: a hand-built sentence on 2 s rows, level at 900 m: two heading words inside one 4 s interval, a speed change, a
#: descent, a go-around and the runway again inside one 8 s interval
HAND = _grid([
    {RUNWAY: 0, HEADING: 0, ALTITUDE: 15, ANGLE: ANGLE_LEVEL, SPEED: 16},
    {HEADING: 1}, {HEADING: 2}, {}, {SPEED: 14}, {}, {ALTITUDE: 10, ANGLE: 2}, {},
    {RUNWAY: RUNWAY_GO_AROUND}, {}, {RUNWAY: 0}, {}, {HEADING: 0}, {}, {}, {},
])


def test_four_seconds_keeps_the_last_word_of_each_interval():
    words = Words(spec())
    out = on_interval(HAND, 0, 4.0, 2.0, np.full(16, 900.0), words, [90.0])
    assert out.shape == (8, 5)
    expected = _grid([
        {RUNWAY: 0, HEADING: 0, ALTITUDE: 15, ANGLE: ANGLE_LEVEL, SPEED: 16},
        {HEADING: 2},                  # rows 1, 2: the last of the two words
        {SPEED: 14},                   # row 4
        {ALTITUDE: 10, ANGLE: 2},      # row 6
        {RUNWAY: RUNWAY_GO_AROUND},    # row 8
        {RUNWAY: 0},                   # row 10
        {HEADING: 0},                  # row 12
        {},
    ])
    assert np.array_equal(out, expected)
    assert np.array_equal(in_force(out)[-1], in_force(HAND)[-1])


def test_eight_seconds_puts_each_word_on_the_nearest_row_and_cancels_a_go_around_inside_one_row():
    """D45: at 8 s a Δ row takes the words from 2 s before it to 2 s after it; a word 4 s after a Δ row goes to the next."""
    words = Words(spec())
    out = on_interval(HAND, 0, 8.0, 2.0, np.full(16, 900.0), words, [90.0])
    assert np.array_equal(out, _grid([
        {RUNWAY: 0, HEADING: 1, ALTITUDE: 15, ANGLE: ANGLE_LEVEL, SPEED: 16},   # rows 0, 1
        {HEADING: 2, SPEED: 14},                                   # rows 2–5
        {RUNWAY: RUNWAY_GO_AROUND, ALTITUDE: 10, ANGLE: 2},        # rows 6–9: the go-around is in force there
        {RUNWAY: 0, HEADING: 0},                                   # rows 10–13
    ]))
    # from row 1 (a later UTC multiple): the go-around (row 8) and the runway again (row 10) go to one Δ row (rows 7–10)
    # and cancel; the first row says every word in force
    shifted = on_interval(HAND, 1, 8.0, 2.0, np.full(16, 900.0), words, [90.0])
    assert (shifted[0] != UNCHANGED).all() and np.array_equal(in_force(shifted), in_force(HAND)[[2, 6, 10, 14]])
    assert (shifted[:, RUNWAY] == RUNWAY_GO_AROUND).sum() == 0


def test_a_word_goes_to_the_nearest_interval_row_and_a_tie_to_the_later():
    """D45 at 4 and 8 s: a word 2 s after a Δ row goes to that row, 2 s before a Δ row to that row, a word exactly between
    two to the later; the words keep their order and the words of one 2 s row stay in one Δ row."""
    words = Words(spec())
    first = {RUNWAY: 0, HEADING: 0, ALTITUDE: 15, ANGLE: ANGLE_LEVEL, SPEED: 16}
    sentence = _grid([first, {SPEED: 15}, {}, {HEADING: 1}, {}, {SPEED: 14}, {HEADING: 2, SPEED: 13}, {}, {},
                      {ALTITUDE: 10, ANGLE: 2}, {}, {}, {}])
    out = on_interval(sentence, 0, 8.0, 2.0, np.full(13, 900.0), words, [90.0])
    assert np.array_equal(out, _grid([
        {**first, SPEED: 15},                       # row 1: 2 s after row 0
        {HEADING: 1, SPEED: 14},                    # row 3: 2 s before row 4; row 5: 2 s after it
        {HEADING: 2, SPEED: 13, ALTITUDE: 10, ANGLE: 2},   # row 6: a tie, the later; row 9: 2 s after row 8
        {},
    ]))
    out = on_interval(sentence, 0, 4.0, 2.0, np.full(13, 900.0), words, [90.0])
    # ties go to the later row: row 1 to row 2; rows 5 and 6 to row 6, where the last word stays
    assert [int(v) for v in out[:, SPEED]] == [16, 15, UNCHANGED, 13, UNCHANGED, UNCHANGED, UNCHANGED]
    assert int(out[2, HEADING]) == 1 and int(out[3, HEADING]) == 2 and int(out[5, ANGLE]) == 2


def test_six_seconds_is_refused_and_a_go_around_at_the_first_row_too():
    words = Words(spec())
    with pytest.raises(ValueError, match="16 s observation"):
        on_interval(HAND, 0, 6.0, 2.0, np.full(16, 900.0), words, [90.0])
    with pytest.raises(Refused, match="go-around at the first row"):
        on_interval(HAND, 8, 4.0, 2.0, np.full(16, 900.0), words, [90.0])


def test_the_grammar_is_read_on_the_projected_rows():
    """A lower level said with no descent class is refused on the projected rows, as on the labeller's."""
    words = Words(spec())
    split = HAND.copy()
    split[6, ANGLE], split[7, ANGLE] = UNCHANGED, 2
    with pytest.raises(Refused):
        on_interval(split, 0, 2.0, 2.0, np.full(16, 900.0), words, [90.0])


def test_a_heading_word_moved_across_a_runway_word_is_said_in_the_frame_where_it_is_heard():
    """D46: a heading word said under one course before a runway word and heard with it at one Δ row is the class nearest
    its absolute track under the new course: the same class after a parallel runway, the class of its track after a
    runway 6° off. No sentence is refused; at the step, it is the same word."""
    words = Words(spec())
    sentence = _grid([
        {RUNWAY: 0, HEADING: 0, ALTITUDE: 15, ANGLE: ANGLE_LEVEL, SPEED: 16},
        {}, {}, {RUNWAY: RUNWAY_GO_AROUND}, {}, {}, {HEADING: 2}, {RUNWAY: 1}, {}])
    assert np.array_equal(on_interval(sentence, 0, 2.0, 2.0, np.full(9, 900.0), words, [90.0, 96.0]), sentence)
    for courses, heard in (([90.0, 90.005], 2), ([90.0, 96.0], 1)):          # 100° under 96°: +4°, class 1
        out = on_interval(sentence, 0, 8.0, 2.0, np.full(9, 900.0), words, courses)
        assert int(out[2, RUNWAY]) == 1 and int(out[2, HEADING]) == heard     # rows 6, 7 → Δ row 8


def test_a_heading_word_of_the_class_in_force_under_another_course_is_said_on_the_interval():
    """D46, Claude's reading: a class says a track only with its course. A Δ row whose new heading word, in the frame where
    it is heard, is the class in force under another course says it (its track differs); a change of runway alone says
    no heading word."""
    words = Words(spec())
    sentence = _grid([
        {RUNWAY: 0, HEADING: 1, ALTITUDE: 15, ANGLE: ANGLE_LEVEL, SPEED: 16},
        {}, {}, {RUNWAY: RUNWAY_GO_AROUND}, {}, {}, {HEADING: 3}, {RUNWAY: 1}, {}, {}, {}, {},
        {RUNWAY: RUNWAY_GO_AROUND}, {}, {}, {RUNWAY: 0}, {}])
    out = on_interval(sentence, 0, 8.0, 2.0, np.full(17, 900.0), words, [90.0, 100.0])
    assert int(out[2, HEADING]) == 1 and int(out[2, RUNWAY]) == 1       # 105° under 100°: class 1, as in force
    assert int(out[3, RUNWAY]) == RUNWAY_GO_AROUND and int(out[4, RUNWAY]) == 0 and (out[3:, HEADING] == UNCHANGED).all()


TWO_GO_AROUNDS = [*GO_AROUND_LEGS[:8], (89, 0, 70, -GLIDE), (40, 0, 70, 8.0), (10, 0, 70, 0), (30, -6, 70, 0),
                  (170, 0, 70, 0), (30, -6, 70, 0), (20, 0, 70, 0), (104, 0, 70, -GLIDE)]


def test_two_go_arounds_are_three_approaches_each_with_its_own_words():
    """D26 with two go-arounds: each climb is its own go-around row, each approach ends there with "no level-off" in
    force, and the speed reading of each later approach starts at the go-around row before it. The first low pass is
    440 m past the threshold: the capture corridor ends at the threshold, so that approach has no capture row and no
    "unspecified" (the letter of §4.5; whether a pass past the threshold should keep its capture is the user's to say,
    the readout counts them)."""
    one, words = spec(), Words(spec())
    reading = read_flight(instruction_flight(*fly_legs(TWO_GO_AROUNDS, 90.0, 900.0, -400.0, 0.0)),
                          instruction_airport(), one, words)
    first, second = reading.go_around_rows
    assert abs(first - 119) <= 2 and second > reading.runway_again_rows[0] > first
    assert [(a.first, a.end) for a in reading.approaches] == [
        (0, first), (reading.runway_again_rows[0], second), (reading.runway_again_rows[1], len(reading.words))]
    assert go_around_in_force(reading, words) == [
        {"no_level_off": True, "unspecified": False, "captured": False, "other_runway": False, "past_threshold": True},
        {"no_level_off": True, "unspecified": True, "captured": True, "other_runway": False, "past_threshold": False}]
    held = in_force(reading.words)
    for row in reading.go_around_rows:                               # 70 m/s: said, or still in force from row 0
        assert held[row, SPEED] == words.speed_index(70.0) and reading.words[row, ANGLE] == words.angle_climb
    assert reading.approaches[0].capture_row is None
    assert all(a.first <= a.capture_row < a.end for a in reading.approaches[1:])


def test_a_heading_word_of_the_class_in_force_after_a_runway_change_is_said():
    """D48 (§4.3): under one course a new word is always another class; after R changes course the track can need the
    class in force again, and the sentence says it, because its track is another."""
    one, words = spec(), Words(spec())
    track = np.concatenate((np.full(20, 90.0), np.full(20, 93.0)))
    relatives = [_relative(40, 0, course, track, 0.0) for course in (90.0, 92.0)]
    runway_rows = np.where(np.arange(40) < 10, 0, 1)
    lateral = read_lateral(track, relatives, [90.0, 92.0], [Approach(0, 40, 1)], runway_rows, one, words)
    assert [(i.row, i.value) for i in lateral.instructions][0] == (0, 0)
    (row, value), = [(i.row, i.value) for i in lateral.instructions[1:]]
    assert value == 0 and 10 < row < 20 and lateral.instructions[1].info["target_deg"] == 92.0


def test_the_assembly_says_the_class_in_force_again_under_another_course_and_drops_it_under_the_same():
    """D48: a heading word is its absolute track. After a runway word to a course 6° off the class in force is said
    again; to a parallel runway (the same course) it says nothing new and is dropped."""
    words = Words(spec())
    base = _first_step(words)
    again = [Instruction(RUNWAY, RUNWAY_GO_AROUND, 3, "go-around"), Instruction(RUNWAY, 1, 6, "runway again"),
             Instruction(HEADING, 18, 7, "per-step")]
    altitude = np.full(10, 900.0)
    grid, kept = assemble(10, [*base, *again], altitude, words, [90.0, 96.0])
    assert grid[7, HEADING] == 18 and len(kept) == 8
    grid, kept = assemble(10, [*base, *again], altitude, words, [90.0, 90.0])
    assert grid[7, HEADING] == UNCHANGED and len(kept) == 7


def test_a_step_up_from_a_level_is_a_go_around_climb():
    """`vertical.climb_after`: a level followed straight by a higher level (a step) is a climb said at the first
    level's last row."""
    from ts_transformer.instructions.labeller.vertical import VerticalPiece, climb_after

    one = spec()
    groups = [[VerticalPiece(0, 20, MOVE, angle_deg=3.0)], [VerticalPiece(20, 40, LEVEL, target_index=5)],
              [VerticalPiece(40, 60, LEVEL, target_index=8)]]
    assert climb_after(groups, 30, 60, one) == 2
    with pytest.raises(Refused, match="go-around without a climb word"):
        climb_after(groups[:2], 30, 40, one)


def test_the_figures_runner_draws_a_labelled_flight(tmp_path, monkeypatch):
    from ts_transformer.experiments import instruction_figures
    from ts_transformer.tests.support import labelled_instruction_artefact

    labelled_instruction_artefact(tmp_path / "artefact", split="val")
    monkeypatch.setattr(instruction_figures, "require_conforming_labeller", lambda directory: None)
    assert instruction_figures.main(["--dir", str(tmp_path / "artefact"), "--count", "2"]) == 0


# ---- the readout of the labeller's runner (outline D85)
def test_the_labellers_readout_gives_only_counts_for_the_val_days():
    """A37, D85: `readout.json`, `readout.md` and the printed text (the same text) show every number of train and select
    and, of val, only the flights labelled and refused — no refusal reason, no stratum, airport or word count."""
    import json

    from ts_transformer.experiments.instruction_labels import render, shown
    from ts_transformer.instructions.readout import class_usage, flight_record, summarise

    one, words = spec(), Words(spec())
    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    reading = read_flight(instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0)), instruction_airport(), one,
                          words)
    refused = {"dataset_id": "KXXX:r", "airport": "KXXX", "status": "refused", "reason": "a val-only reason",
               "detail": ""}
    numbers = {**summarise([flight_record(reading, words)], [refused]), "class_usage": class_usage(reading.words, words)}
    summary = {split: numbers for split in ("train", "select", "val")}
    out = shown(summary)
    assert out["train"] == numbers and out["select"] == numbers
    assert set(out["val"]) == {"labelled", "refused", "readings"} and (out["val"]["labelled"], out["val"]["refused"]) \
        == (1, 1)
    text = render(out, one)
    assert "| val | 1 | 1 |" in text and "## train" in text and "## select" in text and "## val" not in text
    assert text.count("a val-only reason") == 1                            # train + select's reasons, not val's
    assert "D85" in text and json.loads(json.dumps(out))["val"]["readings"].startswith("not shown")
