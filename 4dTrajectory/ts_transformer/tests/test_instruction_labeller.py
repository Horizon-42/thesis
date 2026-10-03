"""The instruction labeller on synthetic flights whose sentence is known (design docs/two_tier/two_tier_design.md §4)."""

from __future__ import annotations

import numpy as np
import pytest

from ts_transformer.instructions import grammar
from ts_transformer.instructions.airport import RunwayRelative, relative_to_runway
from ts_transformer.instructions.labeller.go_around import go_arounds, held_level, low_passes
from ts_transformer.instructions.labeller.interval import first_interval_row, in_force, interval_rows, on_interval
from ts_transformer.instructions.labeller.lateral import read_lateral
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.labeller.sentence import assemble
from ts_transformer.instructions.labeller.speed import read_speed
from ts_transformer.instructions.labeller.vertical import LEVEL, MOVE, read_vertical, tube_checks, vertical_pieces
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, ANGLE_LEVEL, HEADING, RUNWAY, RUNWAY_GO_AROUND, SPEED, UNCHANGED, Words,
)
from ts_transformer.tests.support import (
    INSTRUCTION_STEP_S, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec,
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
    assert words.altitude_m(altitude[0][1]) == 1080.0 and altitude[-1][1] == words.altitude_no_level_off
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


def _per_step(track, capture, course, offset, **changes):
    one = spec(**changes)
    words = Words(one)
    return read_lateral(track, _relative(len(track), capture, course, track, offset), course, one, words), words


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
    reading = read_lateral(track, _relative(40, 0, 90.0, track, 0.0), 90.0, one, words)
    assert reading.capture_row == 0 and reading.turning_deg == 0.0
    assert [(i.column, i.row, i.value) for i in reading.instructions] == [(HEADING, 0, 0)]


def test_level_descend_level_descend_reads_targets_angles_and_no_level_off():
    one, words = spec(), Words(spec())
    distance = np.arange(400) * 150.0
    altitude = np.concatenate((np.full(100, 1500.0), 1500.0 - np.arange(1, 81) * 150.0 * np.tan(np.radians(2.1)),
                               np.full(120, 1500.0 - 80 * 150.0 * np.tan(np.radians(2.1)))))
    altitude = np.concatenate((altitude, altitude[-1] - np.arange(1, 101) * 150.0 * np.tan(np.radians(3.0))))
    reading = read_vertical(distance, altitude, one, words)
    targets = [(i.row, words.altitude_m(i.value)) for i in reading.instructions if i.column == ALTITUDE]
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
    assert pieces[0].kind == LEVEL and words.altitude_m(pieces[0].target_index) == 600.0 and pieces[0].stop >= 39
    reading = read_vertical(distance, altitude, one, words)
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
    reading = read_vertical(distance, altitude, one, words)
    assert [(i.column, i.row, i.value) for i in reading.instructions] == [
        (ALTITUDE, 0, words.altitude_index(1260.0)), (ANGLE, 0, ANGLE_LEVEL)]


def test_a_step_between_two_levels_is_issued_at_the_first_levels_end_with_its_own_direction():
    one, words = spec(), Words(spec())
    distance = np.arange(60) * 400.0
    altitude = np.concatenate((np.full(30, 900.0), np.full(30, 970.0)))
    reading = read_vertical(distance, altitude, one, words)
    assert [(i.column, i.row, i.value) for i in reading.instructions] == [
        (ALTITUDE, 0, words.altitude_index(900.0)), (ANGLE, 0, ANGLE_LEVEL),
        (ALTITUDE, 29, words.altitude_index(960.0)), (ANGLE, 29, words.angle_climb)]
    grid, _ = assemble(60, [Instruction(RUNWAY, 0, 0, "i"), Instruction(HEADING, 18, 0, "i"),
                            Instruction(SPEED, 16, 0, "i"), *reading.instructions], altitude, words, 1)
    assert grid[29, ANGLE] == words.angle_climb


def test_a_flight_climbing_at_the_end_is_refused():
    one, words = spec(), Words(spec())
    altitude = np.concatenate((np.full(30, 900.0), 900.0 + np.arange(1, 31) * 3.0))
    with pytest.raises(Refused, match="climbing at the end"):
        read_vertical(np.arange(60) * 200.0, altitude, one, words)


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
    assert [words.speed_mps(i.value) for i in kept.instructions] == [120.0, 90.0, None]
    # the 90 m/s hold ends 9.3 km or more out; the deceleration's first rows still fit the hold's
    # straight piece within the 1.5 m/s fit tolerance
    assert 140 <= kept.unspecified_row <= 145
    dropped = read_speed(time, speed, near, 60, one, words)
    assert [words.speed_mps(i.value) for i in dropped.instructions] == [120.0, None]
    assert 50 <= dropped.unspecified_row <= 53
    # a capture row inside the 90 m/s hold, no hold ending far out: "unspecified" from the capture row itself
    assert read_speed(time, speed, near, 100, one, words).unspecified_row == 100


def _first_step(words):
    return [Instruction(RUNWAY, 0, 0, "i"), Instruction(HEADING, 18, 0, "i"), Instruction(ALTITUDE, 15, 0, "i"),
            Instruction(ANGLE, ANGLE_LEVEL, 0, "i"), Instruction(SPEED, 16, 0, "i")]


def test_the_assembly_drops_repeats_and_refuses_conflicts_and_an_empty_first_step():
    words = Words(spec())
    altitude = np.full(10, 900.0)
    base = _first_step(words)
    grid, kept = assemble(10, [*base, Instruction(HEADING, 18, 4, "repeat")], altitude, words, 1)
    assert grid[4, HEADING] == UNCHANGED and len(kept) == 5 and grid.shape == (10, 5)
    with pytest.raises(Refused, match="two instructions in one step"):
        assemble(10, [*base, Instruction(HEADING, 20, 0, "clash")], altitude, words, 1)
    with pytest.raises(Refused, match="two instructions in one step"):
        assemble(10, [*base, Instruction(HEADING, 18, 4, "repeat"), Instruction(HEADING, 30, 4, "other")], altitude,
                 words, 1)
    with pytest.raises(Refused, match="first step incomplete"):
        assemble(10, base[:-1], altitude, words, 1)
    with pytest.raises(Refused, match="altitude and angle incompatible"):
        assemble(10, [*base, Instruction(ALTITUDE, 5, 3, "descend with a level angle")], altitude, words, 1)


def test_the_runway_said_again_after_a_go_around_is_kept():
    words = Words(spec())
    instructions = [*_first_step(words), Instruction(RUNWAY, RUNWAY_GO_AROUND, 3, "go-around"),
                    Instruction(RUNWAY, 0, 6, "runway again")]
    grid, kept = assemble(10, instructions, np.full(10, 900.0), words, 1)
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
    # a "no level-off" said before the go-around stays in force through it (the go-around's climb replaces it)
    assert grammar.step_allowed(_state(go_around=True, altitude=words.altitude_no_level_off, angle=3),
                                _step(heading=5), 900.0, words, 1)


def test_rules_3_and_4_the_level_and_the_angle_agree():
    words = Words(spec())
    lower = words.altitude_index(600.0)
    assert not grammar.step_allowed(_state(), _step(altitude=lower), 900.0, words, 1)             # level angle
    assert grammar.step_allowed(_state(), _step(altitude=lower, angle=2), 900.0, words, 1)
    assert not grammar.step_allowed(_state(), _step(altitude=lower, angle=words.angle_climb), 900.0, words, 1)
    # within the level's band no direction is needed: 900 ± 40 m
    assert grammar.step_allowed(_state(), _step(altitude=words.altitude_index(900.0)), 935.0, words, 1)
    assert not grammar.step_allowed(_state(), _step(altitude=words.altitude_no_level_off), 900.0, words, 1)  # rule 4


def test_the_runway_mask_follows_the_table():
    assert grammar.runway_words_allowed(None, 3).tolist() == [True, True, True, False]
    assert grammar.runway_words_allowed(_state(runway=1), 3).tolist() == [True, False, True, True]
    assert grammar.runway_words_allowed(_state(runway=1, go_around=True), 3).tolist() == [True, True, True, False]


# ---- go-arounds (§4.6, D18, D19)
def test_a_go_around_and_a_second_approach():
    """D18: "go-around" at the lowest row of the low pass; the climb read as altitude and angle words; D19: the landed
    runway said again at the first level-off after the climb, before the final's "no level-off" (rule 5 holds)."""
    one, words = spec(), Words(spec())
    e, n, altitude, track, speed = fly_legs(GO_AROUND_LEGS, 90.0, 900.0, -400.0, 0.0)
    reading = read_flight(instruction_flight(e, n, altitude, track, speed), instruction_airport(), one, words)
    (go,), (again,) = reading.go_around_rows, reading.runway_again_rows
    assert abs(go - 119) <= 2                                        # the lowest row, 960 m before the threshold
    assert columns(reading, RUNWAY) == [(0, 0), (go, RUNWAY_GO_AROUND), (again, 0)]
    altitude_words = columns(reading, ALTITUDE)
    climb = [(row, value) for row, value in altitude_words if go - 3 <= row <= go + 3]
    assert climb and words.altitude_m(climb[0][1]) == 900.0 and (climb[0][0], words.angle_climb) in columns(reading, ANGLE)
    assert 158 <= again <= 164                                       # the level-off after the climb (row 159)
    final = [row for row, value in altitude_words if value == words.altitude_no_level_off]
    assert final and final[0] > again
    assert reading.checks["go_arounds"][0]["drop_m"] >= 150.0 and reading.checks["go_arounds"][0]["climb_m"] >= 150.0
    assert reading.capture_row > again                               # the second approach's capture


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
    """D19: a missed approach that first flies level at its low point (early, to the missed approach point) does not end
    the go-around there: the runway is said again at the level held after the climb."""
    legs = [(30, 0, 70, 0), (89, 0, 70, -GLIDE), (15, 0, 70, 0), (40, 0, 70, 8.0), (10, 0, 70, 0), (30, -6, 70, 0),
            (185, 0, 70, 0), (30, -6, 70, 0), (20, 0, 70, 0), (104, 0, 70, -GLIDE)]
    reading = read_flight(instruction_flight(*fly_legs(legs, 90.0, 900.0, -400.0, 0.0)), instruction_airport(), spec())
    (go,), (again,) = reading.go_around_rows, reading.runway_again_rows
    assert 119 <= go <= 136 and again >= 172                         # the climb starts at row 134, levels at 174


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
    assert np.array_equal(on_interval(reading.words, 0, 2.0, 2.0, reading.held_altitude_m, words, 1), reading.words)


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
    out = on_interval(HAND, 0, 4.0, 2.0, np.full(16, 900.0), words, 1)
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


def test_eight_seconds_keeps_the_words_in_force_and_cancels_a_go_around_inside_one_interval():
    words = Words(spec())
    out = on_interval(HAND, 0, 8.0, 2.0, np.full(16, 900.0), words, 1)
    assert np.array_equal(out, _grid([
        {RUNWAY: 0, HEADING: 0, ALTITUDE: 15, ANGLE: ANGLE_LEVEL, SPEED: 16},
        {HEADING: 2, SPEED: 14},                                   # row 4
        {RUNWAY: RUNWAY_GO_AROUND, ALTITUDE: 10, ANGLE: 2},        # row 8: the go-around is in force there
        {RUNWAY: 0, HEADING: 0},                                   # row 12
    ]))
    # from row 3 (a later UTC multiple): the go-around (row 8) and the runway again (row 10) fall between rows 7 and
    # 11 — one interval — and cancel; the first row says every word in force
    shifted = on_interval(HAND, 3, 8.0, 2.0, np.full(16, 900.0), words, 1)
    assert (shifted[0] != UNCHANGED).all() and np.array_equal(in_force(shifted), in_force(HAND)[[3, 7, 11, 15]])
    assert (shifted[:, RUNWAY] == RUNWAY_GO_AROUND).sum() == 0


def test_six_seconds_is_refused_and_a_go_around_at_the_first_row_too():
    words = Words(spec())
    with pytest.raises(ValueError, match="16 s observation"):
        on_interval(HAND, 0, 6.0, 2.0, np.full(16, 900.0), words, 1)
    with pytest.raises(Refused, match="go-around at the first row"):
        on_interval(HAND, 8, 4.0, 2.0, np.full(16, 900.0), words, 1)


def test_the_grammar_is_read_on_the_projected_rows():
    """A lower level said with no descent class is refused on the projected rows, as on the labeller's."""
    words = Words(spec())
    split = HAND.copy()
    split[6, ANGLE], split[7, ANGLE] = UNCHANGED, 2
    with pytest.raises(Refused):
        on_interval(split, 0, 2.0, 2.0, np.full(16, 900.0), words, 1)
