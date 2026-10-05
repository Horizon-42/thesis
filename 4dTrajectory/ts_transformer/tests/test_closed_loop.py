"""The closed-loop reading (vocabulary §4.9, D32): the comparison, the corrections, and a flight read and flown again."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.closed_loop import Corrector, ObservedPath
from ts_transformer.instructions.artefact import ClosedLoopSentence, SentenceRows, Withheld, closed_loop_sentences
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.speed import approach_speed_ias_mps
from ts_transformer.instructions.labeller.interval import in_force, last_heard_row, on_interval
from ts_transformer.instructions.labeller.read import read_flight, smooth, truncated
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, ANGLE_LEVEL, HEADING, RUNWAY, RUNWAY_GO_AROUND, SPEED, UNCHANGED, Words,
)
from ts_transformer.tests.support import (
    closed_loop_artefact, executor_inputs, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec,
)

CPU = torch.device("cpu")


def _sentence(*, grid, correction, states, on_interval, lateral_m, vertical_m, uncorrectable, observed_row, matched_row,
              first_row=0, start=0, timed_out=False, outcome="landed"):
    """A closed-loop sentence built by hand: its rows apart from what is withheld (D82), the withheld fields of the
    labelled flight a synthetic one's."""
    return ClosedLoopSentence(
        rows=SentenceRows(first_row=first_row, start=start, grid=grid, correction=correction, states=states,
                          on_interval=on_interval),
        withheld=Withheld(runway="09", runway_index=0, landing_time_utc="2026-07-01T12:00:00+00:00", capture_row=0,
                          go_around_rows=np.zeros(0, dtype=np.int64), stratum="straight-in", outcome=outcome,
                          timed_out=timed_out, lateral_m=lateral_m, vertical_m=vertical_m, uncorrectable=uncorrectable,
                          observed_row=observed_row, matched_row=matched_row))


def _changed(sentence, **changes):
    """``sentence`` with ``changes``, each to its rows or to what is withheld (D82), by its name."""
    rows = {name: value for name, value in changes.items() if name in {f.name for f in fields(SentenceRows)}}
    held = {name: value for name, value in changes.items() if name not in rows}
    return ClosedLoopSentence(rows=replace(sentence.rows, **rows), withheld=replace(sentence.withheld, **held))


def _params():
    from ts_transformer.experiments.executor_spec import ROLL_RATE_DEG_S

    return ExecutorParams(cycle_s=1.0, bank_rate_deg_s=ROLL_RATE_DEG_S, path_time_constant_s=2.0, path_rate_factor=2.0,
                          timeout_factor=1.5, start_rule="trailing-fit-8s")


# ---- the comparison
def _loop_path():
    """East along n = 0 from e = 0 to 1,000 m, north 500 m, west to e = 500, south across the first leg to n = −500:
    points 50 m apart, height falling 1 m a point."""
    legs = [((0.0, 0.0), (1000.0, 0.0)), ((1000.0, 0.0), (1000.0, 500.0)), ((1000.0, 500.0), (500.0, 500.0)),
            ((500.0, 500.0), (500.0, -500.0))]
    points = [np.array(a) + (np.array(b) - np.array(a)) * t
              for a, b in legs for t in np.linspace(0.0, 1.0, int(math.dist(a, b) / 50.0), endpoint=False)]
    points.append(np.array([500.0, -500.0]))
    e, n = np.array(points).T
    return e, n, 1000.0 - np.arange(len(e))


def test_the_matched_point_moves_forward_and_a_path_that_crosses_itself_does_not_jump():
    """§4.9: the matched point is searched forward from the row before's; e_y right of the observed track positive,
    e_h the flown height above the observed one."""
    e, n, height = _loop_path()
    path = ObservedPath(e, n, height, 0)
    lateral, vertical, *_ = path.match(500.0, 10.0, 990.0)        # over the crossing, on the first leg: 10 m left of east
    assert lateral == pytest.approx(-10.0) and vertical == pytest.approx(990.0 - (1000.0 - 10.0))
    path.match(1010.0, 250.0, 0.0)                              # the north leg, 10 m right of it
    path.match(750.0, 490.0, 0.0)                               # the west leg
    lateral, *_ = path.match(505.0, 5.0, 0.0)                 # over the crossing again: now on the south leg
    assert lateral == pytest.approx(-5.0)                       # 5 m east of southbound is left of it
    assert path.segment > len(e) - 25


def test_a_flown_position_behind_the_matched_point_keeps_it():
    """The matched point never moves back; behind it, e_y is the distance from the path's segments (D83), not from the
    segment's line extended back: at the vertex the matched point stays at, the smaller of the two segments around it
    (A37) — here the one before, from e = 450 to 500."""
    e, n, height = _loop_path()
    path = ObservedPath(e, n, height, 10)
    lateral, *_ = path.match(100.0, -30.0, 0.0)               # behind segment 10 (e = 500 to 550), 30 m right of east
    assert lateral == pytest.approx(math.hypot(350.0, 30.0)) and path.segment == 10    # segment 9 starts at e = 450


def test_e_y_at_the_outside_of_a_turn_is_the_distance_from_the_path():
    """D83: at a vertex of the observed path, a flown position on the outside of the turn is as far from the path as it
    is from the vertex — not from the line of the segment it is matched on — and on its side: right of a left turn."""
    e = np.array([0.0, 100.0, 200.0, 200.0, 200.0])            # east 200 m, then north: a left turn at (200, 0)
    n = np.array([0.0, 0.0, 0.0, 100.0, 200.0])
    path = ObservedPath(e, n, np.zeros(5), 1)
    lateral, *_ = path.match(230.0, -40.0, 0.0)                # beyond the vertex, south-east of it: outside the turn
    assert lateral == pytest.approx(math.hypot(30.0, 40.0)) and lateral > 0.0
    inside = ObservedPath(e, n, np.zeros(5), 1).match(150.0, 20.0, 0.0)     # left of the first leg, along it: as before
    assert inside.lateral_m == pytest.approx(-20.0)


def test_e_y_at_a_vertex_behind_the_matched_segment_is_the_smaller_distance_on_its_side():
    """A37, D83: matched on the segment after a vertex (it never moves back), a position the vertex is the nearest point
    of segment i is measured against segment i − 1 too: the smaller distance, on the nearer segment's side; where the
    vertex is the nearest point of both (the outside of the turn), on the side of the sum of their normals."""
    e = np.array([0.0, 100.0, 200.0, 200.0, 200.0])            # east 200 m, then north: a left turn at (200, 0)
    n = np.array([0.0, 0.0, 0.0, 100.0, 200.0])
    # matched on the northbound segment (row 2 on): a position 5 m south of the eastbound leg, 10 m before the vertex
    south = ObservedPath(e, n, np.zeros(5), 2).match(190.0, -5.0, 0.0)
    assert south.lateral_m == pytest.approx(5.0)                # right of the eastbound leg, 5 m (not 11.2 m left)
    back = ObservedPath(e, n, np.zeros(5), 2).match(150.0, -3.0, 0.0)          # 50 m before the vertex, 3 m south
    assert back.lateral_m == pytest.approx(3.0)
    outside = ObservedPath(e, n, np.zeros(5), 2).match(205.0, -5.0, 0.0)       # south-east of the vertex
    assert outside.lateral_m == pytest.approx(math.hypot(5.0, 5.0))           # as far as the vertex, right of the turn
    for path in (ObservedPath(e, n, np.zeros(5), 2),):
        path.match(190.0, -5.0, 0.0)
        assert path.segment == 2                                 # the matched point stays (it never moves back)
    # a left turn of 135° (more than 90°: there segment 2's own side is the wrong one on the outside of the turn)
    c, k = math.cos(math.radians(135.0)), math.sin(math.radians(135.0))
    sharp_e, sharp_n = np.array([0.0, 100.0, 200.0, 200.0 + 100.0 * c, 200.0 + 200.0 * c]), np.array([0, 0, 0, 100 * k, 200 * k])
    sharp = ObservedPath(sharp_e, sharp_n, np.zeros(5), 2).match(201.0, -3.0, 0.0)
    assert sharp.lateral_m == pytest.approx(math.hypot(1.0, 3.0))          # right of the path, as far as the vertex
    # the matched point at the far end of its segment (the next segment rounded farther from the shared vertex, so the
    # search stopped there; `match`'s branch for t = 1): the same vertex rule, with the segment ahead
    ahead = ObservedPath(sharp_e, sharp_n, np.zeros(5), 1)
    assert ahead._at_vertex(2, 1, 2, 201.0, -3.0, math.hypot(1.0, 3.0)) == pytest.approx(math.hypot(1.0, 3.0))
    # past segment 1's end but beside segment 2: segment 2's distance and side (its own nearest point is inside it)
    beside = (5.0 * 100.0 * k - 20.0 * 100.0 * c) / 100.0                 # right of segment 2, 17.7 m
    assert ahead._at_vertex(2, 1, 2, 205.0, 20.0, math.hypot(5.0, 20.0)) == pytest.approx(beside) and beside > 0.0
    # through `match`: a sharp turn where the search stops at segment 1's end (segment 2 rounds farther from the shared
    # vertex): the vertex rule gives the outside's side (segment 1's alone would give the other)
    found_e = np.array([-49.9353376713203, 64.6364306847856, 179.2081990408915, 75.26840705546014, -28.671384929971197])
    found_n = np.array([97658.23678408828] * 3 + [97706.4364702519, 97754.6361564155])
    stops = ObservedPath(found_e, found_n, np.zeros(5), 1)
    found = stops.match(202.25326014026516, 97696.83302642421, 0.0)
    assert stops.segment == 1 and found.lateral_m == pytest.approx(44.9527, abs=1e-4)
    # a reversal (the normals cancel): the matched segment's side
    back_e, back_n = np.array([0.0, 100.0, 200.0, 100.0, 0.0]), np.zeros(5)
    for north, side in ((5.0, 1.0), (-5.0, -1.0)):           # north is right of the westbound segment 2
        assert ObservedPath(back_e, back_n, np.zeros(5), 2).match(205.0, north, 0.0).lateral_m == pytest.approx(
            side * math.hypot(5.0, 5.0))


# ---- the corrections
def _grid(rows):
    grid = np.full((len(rows), 5), UNCHANGED, dtype=np.int16)
    for row, said in enumerate(rows):
        for column, value in said.items():
            grid[row, column] = value
    return grid


def _first(words, altitude_m=900.0, angle=ANGLE_LEVEL):
    return {RUNWAY: 0, HEADING: 0, ALTITUDE: words.altitude_index(altitude_m), ANGLE: angle,
            SPEED: words.speed_index(70.0)}


def test_a_heading_correction_is_one_class_toward_the_path_and_ends_under_half_the_tolerance_or_at_a_sign_change():
    words = Words(spec())
    left, right = words.n_heading - 1, 1                        # classes −5° and +5°
    corrector = Corrector(_grid([_first(words), *[{}] * 9]), 0, 0, 1, words, [90.0])
    corrector.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)

    def heading(k, lateral):
        said, added = corrector.row(k, lateral, 0.0, 900.0, holding=False, past_end=False)
        return int(said[HEADING]), bool(added[HEADING])

    assert heading(1, 25.0) == (UNCHANGED, False)                # inside Y = 30 m
    assert heading(2, 40.0) == (left, True)                      # right of the path: one class to the left
    assert heading(3, 20.0) == (UNCHANGED, False)                # still over Y / 2: in force
    assert heading(4, 10.0) == (0, True)                         # under Y / 2: the observed word again
    assert heading(5, -45.0) == (right, True)
    assert heading(6, 2.0) == (0, True)                          # the sign changed, within Y: the observed word
    assert heading(7, 35.0) == (left, True)
    assert heading(8, -40.0) == (right, True)                    # an overshoot beyond Y: the opposite at once (D53)
    assert heading(9, -20.0) == (UNCHANGED, False)               # in force


def test_a_new_observed_heading_word_ends_a_correction():
    words = Words(spec())
    corrector = Corrector(_grid([_first(words), {}, {HEADING: 2}, {}]), 0, 0, 1, words, [90.0])
    corrector.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    assert int(corrector.row(1, 50.0, 0.0, 900.0, holding=False, past_end=False)[0][HEADING]) == words.n_heading - 1
    said, added = corrector.row(2, 50.0, 0.0, 900.0, holding=False, past_end=False)            # the observed word, not a correction
    assert int(said[HEADING]) == 2 and not added[HEADING]
    said, added = corrector.row(3, 50.0, 0.0, 900.0, holding=False, past_end=False)            # a correction from the new word
    assert int(said[HEADING]) == 1 and added[HEADING]


def test_an_angle_correction_on_a_final_descent_and_none_on_a_level_a_climb_or_beyond_the_classes():
    words = Words(spec())
    final = {ALTITUDE: words.altitude_no_level_off, ANGLE: 3}
    corrector = Corrector(_grid([_first(words), final, *[{}] * 8]), 0, 0, 1, words, [90.0])
    corrector.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    corrector.row(1, 0.0, 0.0, 900.0, holding=False, past_end=False)

    def angle(k, vertical):
        said, added = corrector.row(k, 0.0, vertical, 800.0, holding=False, past_end=False)
        return int(said[ANGLE]), bool(added[ANGLE])

    assert angle(2, 20.0) == (4, True)                           # too high: the next steeper class
    assert angle(3, 10.0) == (UNCHANGED, False)
    assert angle(4, 5.0) == (3, True)                            # under H / 2: the observed class again
    assert angle(5, -20.0) == (2, True)                          # too low: the next shallower
    assert angle(6, 3.0) == (3, True)
    assert angle(7, 20.0) == (4, True)
    assert angle(8, -20.0) == (2, True)                          # an overshoot beyond H: the shallower class at once (D53)
    level = Corrector(_grid([_first(words), *[{}] * 3]), 0, 0, 1, words, [90.0])
    level.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    assert int(level.row(1, 0.0, 60.0, 960.0, holding=False, past_end=False)[0][ANGLE]) == UNCHANGED            # a level hold: none
    climb = Corrector(_grid([_first(words, 1200.0, words.angle_climb), *[{}] * 3]), 0, 0, 1, words, [90.0])
    climb.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    assert int(climb.row(1, 0.0, -60.0, 900.0, holding=False, past_end=False)[0][ANGLE]) == UNCHANGED           # a climb: none
    steepest = Corrector(_grid([_first(words, 600.0, words.n_descent), *[{}] * 3]), 0, 0, 1, words, [90.0])
    steepest.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    assert int(steepest.row(1, 0.0, 60.0, 900.0, holding=False, past_end=False)[0][ANGLE]) == UNCHANGED        # no class steeper than descent 4
    shallowest = Corrector(_grid([_first(words, 600.0, 1), *[{}] * 3]), 0, 0, 1, words, [90.0])
    shallowest.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    assert int(shallowest.row(1, 0.0, -60.0, 900.0, holding=False, past_end=False)[0][ANGLE]) == UNCHANGED     # none shallower than descent 1
    assert int(shallowest.row(2, 0.0, 60.0, 900.0, holding=False, past_end=False)[0][ANGLE]) == 2
    said, added = shallowest.row(3, 0.0, -60.0, 900.0, holding=False, past_end=False)   # an overshoot with no class beyond
    assert int(said[ANGLE]) == 1 and added[ANGLE]                                         # descent 1: the observed class


def test_the_vertical_tolerance_in_force_is_h_final_only_while_no_level_off_is_in_force():
    """D66: an e_h between H_final (5 m) and H (15 m) starts an angle correction in a final descent and not before it; the
    tolerance changes where "no level-off" is said and where a go-around row says a level (rule 6), and again at the
    next final descent; at H_final = H no correction starts anywhere."""
    def read(final_vertical_m):
        words = Words(spec(closed_loop_final_vertical_m=final_vertical_m))
        final = {ALTITUDE: words.altitude_no_level_off, ANGLE: 3}
        rows = [(_first(words, 600.0, 2), 0.0, 900.0), ({}, 10.0, 850.0),          # a descent to a level: H
                (final, 10.0, 800.0), ({}, 10.0, 700.0), ({}, 2.0, 600.0),           # the final descent: H_final
                ({RUNWAY: RUNWAY_GO_AROUND, ALTITUDE: words.altitude_index(900.0), ANGLE: words.angle_climb}, 10.0, 300.0),
                ({}, 10.0, 500.0), ({RUNWAY: 0}, 10.0, 700.0),                       # a climb: no correction
                ({ALTITUDE: words.altitude_index(600.0), ANGLE: 2}, 10.0, 900.0), ({}, 10.0, 850.0),   # H again
                (final, 10.0, 700.0), ({}, 10.0, 650.0)]                             # H_final again
        grid = _grid([said for said, _, _ in rows])
        corrector = Corrector(grid, 0, 0, 1, words, [90.0])
        said = [corrector.row(k, 0.0, e_h, height, holding=False, past_end=False) for k, (_, e_h, height) in enumerate(rows)]
        return words, grid, said

    words, grid, said = read(5.0)
    assert closed_loop.vertical_tolerance_m(words, in_force(grid)[:, ALTITUDE]).tolist() == [
        15.0, 15.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0, 15.0, 15.0, 5.0, 5.0]
    assert [k for k, (_, added) in enumerate(said) if added[ANGLE]] == [3, 4, 11]
    assert [int(said[k][0][ANGLE]) for k in (3, 4, 11)] == [4, 3, 4]    # one class steeper, the observed class again
    _, _, before = read(15.0)
    assert not any(added[ANGLE] for _, added in before)


def test_a_new_observed_altitude_word_ends_an_angle_correction():
    words = Words(spec())
    corrector = Corrector(_grid([_first(words, 600.0, 2), {}, {ALTITUDE: words.altitude_index(480.0)}]), 0, 0, 1, words, [90.0])
    corrector.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    assert int(corrector.row(1, 0.0, 30.0, 800.0, holding=False, past_end=False)[0][ANGLE]) == 3
    said, _ = corrector.row(2, 0.0, 30.0, 800.0, holding=False, past_end=False)
    assert int(said[ANGLE]) == 2 and int(said[ALTITUDE]) == words.altitude_index(480.0)


# ---- a flight
def _turn(degrees: float, speed_mps: float, per_row_deg: float = 4.5):
    side = math.copysign(1.0, degrees)
    steady = int(round((abs(degrees) - 4.0 * per_row_deg) / per_row_deg))
    return [(2, side * per_row_deg / 3.0, speed_mps, 0.0), (2, side * per_row_deg * 2.0 / 3.0, speed_mps, 0.0),
            (steady, side * per_row_deg, speed_mps, 0.0),
            (2, side * per_row_deg * 2.0 / 3.0, speed_mps, 0.0), (2, side * per_row_deg / 3.0, speed_mps, 0.0)]


#: A downwind, a base and a final with a 3° descent (`test_autopilot`'s): flown on its words alone on the sentence's own
#: clock, the turn onto the final ends 300-odd metres right of the line and class 0 keeps it there.
DOWNWIND_BASE_FINAL = [(60, 0.0, 100.0, 0.0), *_turn(-90.0, 100.0), (20, 0.0, 90.0, 0.0), *_turn(-90.0, 85.0),
                       (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]


def _batch(interval_s=2.0, legs=DOWNWIND_BASE_FINAL, track_deg=270.0, altitude_m=1110.0, speeds=None, cut=0,
           final_vertical_m=15.0):
    """A synthetic flight's batch, its executor inputs at the first predicted step and the words; ``speeds``: the speed
    words its open-loop reading says instead of the observed ones, ``{2 s row: m/s}`` (none: the observed words);
    ``cut``: the observed flight and its reading end that many 2 s rows early (the slice stops short of the threshold);
    ``final_vertical_m``: H_final (D66; H = 15 m, the reading before D66)."""
    one, geometry = spec(closed_loop_final_vertical_m=final_vertical_m), instruction_airport()
    words = Words(one)
    signals = instruction_flight(*fly_legs(legs, track_deg, altitude_m, -400.0, 0.0))
    reading = read_flight(signals, geometry, one, words)
    if speeds is not None or cut:
        grid = reading.words.copy()
        if speeds is not None:
            grid[:, SPEED] = UNCHANGED
            for row, speed_mps in speeds.items():
                grid[row, SPEED] = words.speed_index(speed_mps)
        rows = len(grid) - cut
        reading = replace(reading, words=grid[:rows], held_height_m=reading.held_height_m[:rows])
    sentence = replay.sentence_on_interval(reading, signals, interval_s, geometry, words)
    observed = replay.from_row(signals, sentence.first_row)
    batch = replay.Batch(indices=[0], signals=[observed], observed=[signals], series=[None], readings=[reading],
                         sentences=[sentence], row_interval_s=interval_s, geometries=[geometry],
                         approach_ias_mps=[approach_speed_ias_mps("A320", 62000.0)], groups=[replay.OWN], drawn={})
    start = closed_loop.start_row(interval_s) * int(round(interval_s / one.step_s))
    return batch, executor_inputs(signals, geometry, sentence.first_row + start), words


@pytest.mark.parametrize("interval_s", [2.0, 4.0])
def test_a_flight_whose_words_leave_an_offset_is_brought_back_by_heading_corrections(interval_s):
    """D32: the turn onto the final leaves the flown path right of the observed one; heading words one class toward the
    path bring it back, and it ends within the tolerance. The rows before the first predicted step stay observed."""
    batch, inputs, words = _batch(interval_s)
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    assert isinstance(sentence, ClosedLoopSentence)
    start, every = sentence.rows.start, int(round(interval_s / 2.0))
    assert start == 16 / interval_s and (sentence.rows.grid[0] != UNCHANGED).all() and not sentence.rows.correction[0].any()
    observed = batch.signals[0]
    rows = np.arange(start * every)                                    # the observed 2 s rows (D51)
    assert np.array_equal(sentence.rows.states[:start * every, :3], np.column_stack([observed.e_m[rows], observed.n_m[rows],
                                                                               observed.altitude_m[rows]]))
    corrected = sentence.rows.correction[:, HEADING]
    assert corrected.any()
    in_force = closed_loop.in_force(sentence.rows.grid)[:, HEADING]
    reading = batch.readings[0].words[batch.sentences[0].first_row:]
    observed_held = closed_loop.in_force(reading)[sentence.withheld.observed_row, HEADING]
    steps = (in_force.astype(int) - observed_held.astype(int)) % words.n_heading
    assert set(steps.tolist()) <= {0, 1, words.n_heading - 1}  # at most one class from the observed word
    # inside a turn every row says a new observed word, which ends a correction (§4.9): the offset the turns leave is
    # taken out on the straight legs after them, and the final is flown within the tolerance
    assert np.abs(sentence.withheld.lateral_m).max() > words.spec.closed_loop_lateral_m
    assert np.abs(sentence.withheld.lateral_m[-len(sentence.withheld.lateral_m) // 4:]).max() < words.spec.closed_loop_lateral_m


def _replayed(batch, inputs, sentence, params, words, monkeypatch):
    """``sentence`` flown again as the replay flies it (`replay_batch`, `replay.fly_sentences`)."""
    stored = {7: sentence}
    batch.indices[0] = 7
    moved, missing = closed_loop.replay_batch(batch, stored, words)
    monkeypatch.setattr(replay.Batch, "inputs", lambda self, rule, device: inputs)
    return stored, moved, missing, replay.fly_sentences(moved, params, words, device=CPU)


@pytest.mark.parametrize("interval_s", [2.0, 4.0, 8.0])
def test_a_closed_loop_sentence_flown_again_gives_its_states_on_every_2_s_row(monkeypatch, interval_s):
    """D51: the states are on the data's 2 s rows from the sentence's first row to its last said row, its Δ rows marked;
    the replay flies a closed-loop sentence on its own rows through every 2 s row of it, between the Δ rows too."""
    batch, inputs, words = _batch(interval_s)
    params = _params()
    (sentence,) = closed_loop.read(batch, inputs, params, words, device=CPU)
    every = int(round(interval_s / 2.0))
    assert len(sentence.rows.states) == (sentence.rows.start + len(sentence.rows.grid) - 1) * every + 1
    assert np.array_equal(np.flatnonzero(sentence.rows.on_interval), np.arange(sentence.rows.start + len(sentence.rows.grid)) * every)
    _, _, _, flown = _replayed(batch, inputs, sentence, params, words, monkeypatch)
    track = flown_track(flown.states[0].numpy(), batch.geometries[0])
    at = np.arange(len(sentence.rows.flown_states)) * 2                    # the cycle starting each 2 s row
    again = np.column_stack([track["e"][at], track["n"][at], track["height"][at]])
    assert np.abs(again - sentence.rows.flown_states[:, :3]).max() < 1e-9


@pytest.mark.parametrize("interval_s", [4.0, 8.0])
def test_the_marked_rows_are_the_said_rows_and_the_rows_between_are_flown(interval_s):
    """D51: at Δ = 4 and 8 s the marked 2 s rows hold the states the reading compared (the said rows' e_y is read from
    them), and the rows between hold the executor's states there: in the turns, off the chord of their two Δ rows."""
    batch, inputs, words = _batch(interval_s)
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    every, start = int(round(interval_s / 2.0)), sentence.rows.start
    signals = batch.signals[0]
    span = len(batch.readings[0].words) - batch.sentences[0].first_row
    smoothed = smooth(truncated(signals, span), words.spec)
    path = ObservedPath(signals.e_m[:span], signals.n_m[:span], smoothed.altitude_m, start * every)
    marked = sentence.rows.states[sentence.rows.on_interval][start:]
    assert np.allclose([path.match(*row[:3]).lateral_m for row in marked], sentence.withheld.lateral_m, atol=1e-9)
    flown = sentence.rows.flown_states
    assert len(flown) == (len(sentence.rows.grid) - 1) * every + 1
    middle = flown[every // 2::every][: len(sentence.rows.grid) - 1, :2]          # the 2 s row halfway between two Δ rows
    chord = 0.5 * (flown[:-every:every, :2] + flown[every::every, :2])
    assert np.abs(middle - chord).max() > 1.0


def test_a_go_around_at_the_first_predicted_step_and_a_short_sentence_are_refused():
    """The go-around in force in the 2 s words the first predicted step says (those before Δ/2 after its observed time; a tie waits, D45, A14)."""
    batch, inputs, words = _batch(4.0)
    reading, sentence = batch.readings[0], batch.sentences[0]
    first = sentence.first_row + closed_loop.start_row(4.0) * 2

    def gone(row):
        grid = reading.words.copy()
        grid[row, RUNWAY] = RUNWAY_GO_AROUND
        return replace(reading, words=grid)

    short = replay.Sentence(grid=sentence.grid[:5], instructions=[], first_row=sentence.first_row)
    assert closed_loop.refusal(sentence, gone(first + 1), 4.0, 2.0) is None        # a tie: heard at the second row
    for case, reason in (((gone(first), sentence), "go-around at the first predicted step"),
                         ((reading, short), "too short")):
        batch.readings[0], batch.sentences[0] = case
        (result,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
        assert isinstance(result, Refused) and result.reason.startswith(reason)


def test_each_outcome_is_stored_by_name_and_a_file_of_the_former_format_is_refused(tmp_path):
    """D74: the reading's outcome of each sentence — a landing and every kind of failure of the judge (§5.8) — is written
    by its name and read back with its sentence; a closed-loop file of the former format (no outcome) is refused by name."""
    from ts_transformer.autopilot.judge import OUTCOMES
    from ts_transformer.instructions.artefact import (
        CLOSED_LOOP_SCHEMA, closed_loop_path, load_closed_loop, write_closed_loop,
    )

    batch, inputs, words = _batch(4.0)
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    assert sentence.withheld.outcome in OUTCOMES
    directory = closed_loop_artefact(tmp_path / "artefact", batch, words.spec, copies=len(OUTCOMES))
    path = closed_loop_path(directory, "train", 4.0)
    write_closed_loop(path, words.spec, executor_params_sha256="x", row_interval_s=4.0, start_row=sentence.rows.start,
                      sentences={k: _changed(sentence, outcome=name) for k, name in enumerate(OUTCOMES)})
    stored = closed_loop_sentences(directory, "train", 4.0, words.spec)
    assert [stored[k].withheld.outcome for k in range(len(OUTCOMES))] == list(OUTCOMES)
    assert set(OUTCOMES) >= {"landed", "unstable_at_minimums", "crossed_too_high", "crossed_off_runway",
                             "crossed_other_runway", "ground_contact", "timeout", "dynamics_failure"}
    with np.load(path) as data:
        former = {name: data[name] for name in data.files}
    former["schema"] = np.array("ts-instruction-closed-loop-v7")             # the same fields, the former format
    np.savez_compressed(tmp_path / "former.npz", **former)
    with pytest.raises(ValueError, match=f"is not a {CLOSED_LOOP_SCHEMA} file"):
        load_closed_loop(tmp_path / "former.npz", words.spec)


def test_the_closed_loop_file_round_trips_and_a_replay_flies_its_stored_states(tmp_path, monkeypatch):
    """§4.9 "Artefact": the words, corrections, states and errors written and read back by signal index; the replay's
    batch flies each from its first predicted step (its sentence and observed flight moved there) and the replay's check
    finds the stored states again."""
    from ts_transformer.experiments.executor_replay import closed_loop_columns
    from ts_transformer.instructions.artefact import closed_loop_path, write_closed_loop

    batch, inputs, words = _batch(4.0)
    params = _params()
    (sentence,) = closed_loop.read(batch, inputs, params, words, device=CPU)
    directory = closed_loop_artefact(tmp_path / "artefact", batch, words.spec, copies=8)
    path = closed_loop_path(directory, "train", 4.0)
    assert sentence.rows.first_row == batch.sentences[0].first_row
    write = dict(executor_params_sha256="x", row_interval_s=4.0, start_row=sentence.rows.start)
    write_closed_loop(path, words.spec, **write, sentences={7: sentence})
    with pytest.raises(ValueError, match="nothing to write"):
        write_closed_loop(tmp_path / "empty.npz", words.spec, **write, sentences={})
    with pytest.raises(FileExistsError):
        write_closed_loop(path, words.spec, **write, sentences={7: sentence})
    with pytest.raises(ValueError, match="Δ rows marked"):
        write_closed_loop(tmp_path / "unmarked.npz", words.spec, **write,
                          sentences={7: _changed(sentence, on_interval=np.roll(sentence.rows.on_interval, 1))})
    # D61, D82, vocabulary §6 item 3: the reader gives back every field written — the words, the marks, all the states —
    # and apart from them what is withheld: the reading's own, the labelled ones joined from the sentence file
    stored = closed_loop_sentences(directory, "train", 4.0, words.spec)
    assert list(stored) == [7] and [f.name for f in fields(stored[7])] == ["rows", "withheld"]
    for part in ("rows", "withheld"):
        for name in (f.name for f in fields(getattr(sentence, part))):
            assert np.array_equal(getattr(getattr(stored[7], part), name), getattr(getattr(sentence, part), name),
                                  equal_nan=name == "vertical_m"), (part, name)
    assert (stored[7].withheld.runway, stored[7].withheld.stratum) == ("09", "vectored")
    from ts_transformer.instructions.artefact import closed_loop_indices

    assert closed_loop_indices(directory, "train", 4.0, words.spec) == {7}
    misnamed = closed_loop_path(directory, "train", 8.0)                    # a 4 s file under the 8 s name
    misnamed.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="holds sentences at 4 s, not 8 s"):
        closed_loop_indices(directory, "train", 8.0, words.spec)
    with pytest.raises(ValueError, match="holds sentences at 4 s, not 8 s"):
        closed_loop_sentences(directory, "train", 8.0, words.spec)
    with pytest.raises(ValueError, match="holds sentences at 4 s, not 2 s"):
        closed_loop_sentences(directory, "train", 2.0, words.spec, path=path)        # a part file of another Δ
    with pytest.raises(ValueError, match="is not a ts-instruction-closed-loop-v8 file read with spec"):
        closed_loop_indices(directory, "train", 4.0, spec(closed_loop_final_vertical_m=5.0))  # another vocabulary
    assert np.array_equal(stored[7].rows.flown_states,
                          sentence.rows.states[int(np.flatnonzero(sentence.rows.on_interval)[sentence.rows.start]):])
    _, moved, missing, flown = _replayed(batch, inputs, sentence, params, words, monkeypatch)
    assert missing == 0 and moved.sentences[0].first_row == batch.sentences[0].first_row + sentence.rows.start * 2
    assert moved.signals[0].e_m[0] == batch.signals[0].e_m[sentence.rows.start * 2]
    verdicts = replay.judge_batch(moved, flown, words)
    assert verdicts[0].outcome == sentence.withheld.outcome                                  # D74: the replay's is the stored
    (row,) = closed_loop_columns(stored, 2.0)(moved, flown, verdicts)
    assert row["largest_lateral_m"] == pytest.approx(float(np.abs(sentence.withheld.lateral_m).max()))
    assert row["uncorrected_lateral_m"] == closed_loop.uncorrected_m(sentence.withheld.lateral_m, sentence.withheld.uncorrectable[:, 0])
    assert row["uncorrected_vertical_m"] == closed_loop.uncorrected_m(sentence.withheld.vertical_m, sentence.withheld.uncorrectable[:, 1])
    assert row["uncorrected_lateral_m"] != row["uncorrected_vertical_m"]
    assert sum(row["correction_words"].values()) == int(sentence.rows.correction.sum())
    stored[7] = _changed(sentence, outcome="ground_contact" if sentence.withheld.outcome != "ground_contact" else "landed")
    with pytest.raises(ValueError, match=f"flown again to {sentence.withheld.outcome}, stored {stored[7].withheld.outcome}"):
        closed_loop_columns(stored, 2.0)(moved, flown, verdicts)
    stored[7] = _changed(sentence, states=sentence.rows.states + [1.0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="from its closed-loop states"):
        closed_loop_columns(stored, 2.0)(moved, flown, verdicts)


@pytest.mark.parametrize("interval_s", [2.0, 8.0])
def test_a_sentence_that_reaches_its_time_limit_stores_the_timeout_the_replay_gives(monkeypatch, interval_s):
    """D74 at the time limit: a flight cut short of the runway is done in the cycle that reaches its time limit, stored
    as `timeout`, and flown again it is judged the same."""
    batch, inputs, words = _batch(interval_s, cut=120)
    params = _params()
    (sentence,) = closed_loop.read(batch, inputs, params, words, device=CPU)
    assert sentence.withheld.timed_out and sentence.withheld.outcome == "timeout"
    _, moved, _, flown = _replayed(batch, inputs, sentence, params, words, monkeypatch)
    assert replay.judge_batch(moved, flown, words)[0].outcome == sentence.withheld.outcome


def test_the_runner_counts_the_flights_without_a_sentence_by_reason():
    """§4.9: a flight the replay does not fly (no dynamics, …), one the row interval refuses and one the closed loop
    refuses give no training sentence, each counted by its reason."""
    from ts_transformer.experiments.instruction_closed_loop import merge_tallies, summarise, tally

    batch, inputs, words = _batch()
    batch = replace(batch, refused_seen=Counter({"too short": 1}))
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    lateness = closed_loop.heading_lateness_rows(sentence, batch.readings[0].words[batch.sentences[0].first_row:]) * 2.0
    counted = tally(batch, [sentence, Refused("go-around at the first predicted step")], words)
    numbers = summarise(counted, {"no aircraft dynamics": 3})
    # a split read in parts: each part's tally in order gives the numbers of the whole
    whole = replay.subset(batch, [0, 0, 0])
    first, second = replay.subset(batch, [0, 0]), replace(replay.subset(batch, [0]), refused_seen=Counter())
    gone = Refused("go-around at the first predicted step")
    assert summarise(merge_tallies([tally(first, [sentence, gone], words), tally(second, [sentence], words)]), {}) \
        == summarise(tally(whole, [sentence, gone, sentence], words), {})
    assert numbers["sentences"] == 1
    assert numbers["outcomes"] == {sentence.withheld.outcome: 1}                                             # D74
    assert numbers["without_a_sentence"] == {
        "not flown": {"no aircraft dynamics": 3}, "refused on the row interval": {"too short": 1},
        "refused by the closed loop": {"go-around at the first predicted step": 1}}
    assert numbers["correction_words"]["heading"] == int(sentence.rows.correction[:, HEADING].sum()) > 0
    assert numbers["heading_word_lateness_s"]["n"] == len(lateness) > 0
    assert numbers["heading_word_lateness_s"]["mean"] == pytest.approx(float(lateness.mean()))
    lateral = numbers["outside_the_tolerance"]["lateral"]
    assert lateral["outside"] > 0 and lateral["without_a_correction_toward_the_path"] == 0        # D50: the rule holds
    assert lateral["correctable_rows"] == int((~sentence.withheld.uncorrectable[:, 0]).sum())


def _outside(batch, sentence, words):
    return closed_loop.outside_rows(sentence, batch.readings[0].words, batch.sentences[0].first_row, words,
                                    [c.course_deg for c in batch.geometries[0].candidates])


def test_the_rule_of_d50_reads_the_direction_of_the_correction_in_force():
    """D50 (vocabulary §12.2): on a correctable row with |e_y| > Y a heading correction toward the path is in force after it, read
    from the stored sentence and its open-loop reading. Taken away (the observed words alone), the rule breaks on
    exactly the rows that were outside; turned away from the path, too — a word that only cancels a correction, or one
    on the wrong side, answers nothing."""
    batch, inputs, words = _batch()
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    correctable, outside, breaks = _outside(batch, sentence, words)["lateral"]
    assert outside.any() and not breaks.any()
    assert np.array_equal(outside, ~sentence.withheld.uncorrectable[:, 0] & (np.abs(sentence.withheld.lateral_m)
                                                                    > words.spec.closed_loop_lateral_m))
    observed = batch.readings[0].words[batch.sentences[0].first_row:]
    for heading, name in ((in_force(observed)[sentence.withheld.observed_row, HEADING], "the observed words"),
                          ((2 * in_force(observed)[sentence.withheld.observed_row, HEADING].astype(int)
                            - in_force(sentence.rows.grid)[:, HEADING].astype(int)) % words.n_heading, "away")):
        grid = sentence.rows.grid.copy()
        grid[:, HEADING] = heading
        assert np.array_equal(_outside(batch, _changed(sentence, grid=grid), words)["lateral"][2], outside), name


def test_the_readings_of_d34_read_the_vertical_tolerance_in_force_at_each_row():
    """D66, D50: a row is outside vertically when |e_h| exceeds the tolerance in force there — H_final in the final
    descent, H before it — and the rule of D50 holds on the rows the corrector read."""
    words = Words(spec(closed_loop_final_vertical_m=5.0))
    final = {ALTITUDE: words.altitude_no_level_off, ANGLE: 3}
    observed = _grid([_first(words, 600.0, 2), {}, final, {}, {}])
    corrector = Corrector(observed, 0, 0, 1, words, [90.0])
    errors = [0.0, 10.0, 10.0, 10.0, 10.0]
    rows, uncorrectable = [], []
    for k, (e_h, height) in enumerate(zip(errors, (900.0, 850.0, 800.0, 750.0, 700.0))):
        rows.append(corrector.row(k, 0.0, e_h, height, holding=False, past_end=False))
        uncorrectable.append(corrector.uncorrectable.copy())
    n = len(rows)
    sentence = _sentence(
        first_row=0, grid=np.array([r[0] for r in rows]), correction=np.array([r[1] for r in rows]),
        states=np.zeros((n, 6)), on_interval=np.ones(n, bool), lateral_m=np.zeros(n), vertical_m=np.array(errors),
        uncorrectable=np.array(uncorrectable), observed_row=np.arange(n), matched_row=np.arange(n, dtype=float),
        start=0, timed_out=False, outcome="landed")
    correctable, outside, breaks = closed_loop.outside_rows(sentence, observed, 0, words, [90.0])["vertical"]
    assert correctable.tolist() == [False, True, False, True, True]     # the first row and a new word: none
    assert outside.tolist() == [False, False, False, True, True] and not breaks.any()


def test_a_closed_loop_flight_with_h_final_differs_only_in_its_final_descent():
    """D66 on a whole flight: read at H_final = 5 m and at H_final = H, the sentence and the flown states are the same on
    every row before the first row where "no level-off" is in force; after it, angle corrections that H leaves out."""
    readings = {}
    for final_vertical_m in (15.0, 5.0):
        batch, inputs, words = _batch(final_vertical_m=final_vertical_m)
        (readings[final_vertical_m],) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    before, after = readings[15.0], readings[5.0]
    final = in_force(before.rows.grid)[:, ALTITUDE] == words.altitude_no_level_off
    first = int(np.argmax(final))
    assert final.any() and np.array_equal(in_force(after.rows.grid)[:first + 1, ALTITUDE], in_force(before.rows.grid)[:first + 1, ALTITUDE])
    assert np.array_equal(before.rows.grid[:first + 1], after.rows.grid[:first + 1])
    assert np.array_equal(before.rows.correction[:first + 1], after.rows.correction[:first + 1])
    every = int(round(2.0 / words.spec.step_s))
    assert np.array_equal(before.rows.states[:(before.rows.start + first) * every + 1], after.rows.states[:(after.rows.start + first) * every + 1])
    assert after.rows.correction[first:, ANGLE].sum() > before.rows.correction[first:, ANGLE].sum()


def test_the_rule_of_d50_reads_a_word_moved_across_a_runway_word_in_its_new_frame():
    """D46 + D50: an observed heading word said under runway 0 (course 90°, class +5° → 95°) and heard after the runway
    word of runway 1 (course 92°) is said as class +5° there (97°): 2° right of its observed track, but no correction —
    left of the path it answers nothing, and the rule counts the row."""
    words = Words(spec())
    observed = np.full((6, 5), UNCHANGED, dtype=np.int16)
    observed[0, :] = [0, 0, words.altitude_index(900.0), ANGLE_LEVEL, words.speed_index(70.0)]
    observed[1, RUNWAY], observed[2, HEADING], observed[3, RUNWAY] = RUNWAY_GO_AROUND, 1, 1
    grid = np.full((5, 5), UNCHANGED, dtype=np.int16)
    grid[0] = observed[0]
    grid[1, RUNWAY], grid[3, RUNWAY], grid[3, HEADING] = RUNWAY_GO_AROUND, 1, 1
    n = len(grid)
    sentence = _sentence(
        first_row=0, grid=grid, correction=np.zeros((n, 5), bool), states=np.zeros((n, 6)), on_interval=np.ones(n, bool),
        lateral_m=np.array([0.0, 0.0, 0.0, 0.0, -60.0]), vertical_m=np.zeros(n),
        uncorrectable=np.array([[True, True], *[[False, True]] * (n - 1)]),
        observed_row=np.array([0, 1, 2, 3, 3]), matched_row=np.zeros(n), start=0, timed_out=False, outcome="landed")
    _, outside, breaks = closed_loop.outside_rows(sentence, observed, 0, words, [90.0, 92.0])["lateral"]
    assert outside.tolist() == [False, False, False, False, True] and breaks.tolist() == outside.tolist()
    corrected = grid.copy()
    corrected[4, HEADING] = 2                                       # one class right of 97°: toward the path
    _, _, breaks = closed_loop.outside_rows(_changed(sentence, grid=corrected), observed, 0, words, [90.0, 92.0])["lateral"]
    assert not breaks.any()


def test_an_overshoot_takes_the_opposite_correction_in_its_row_and_the_rule_of_d50_holds():
    """D53: the flown aircraft crosses the path in one row, 60 m right then 60 m left of it: the overshoot row says the
    opposite correction at once, and the rule of D50 holds on every row. A word that only cancels a correction would
    answer nothing there, and the rule counts it."""
    words = Words(spec())
    corrector = Corrector(_grid([_first(words, 900.0, ANGLE_LEVEL), *[{}] * 5]), 0, 0, 1, words, [90.0])
    rows = [corrector.row(s, e_y, 0.0, 900.0, holding=False, past_end=False)
            for s, e_y in enumerate((0.0, 60.0, -60.0, -60.0, 0.0))]
    grid = np.array([r[0] for r in rows])
    n = len(grid)
    sentence = _sentence(
        first_row=0, grid=grid, correction=np.array([r[1] for r in rows]), states=np.zeros((n, 6)), on_interval=np.ones(n, bool),
        lateral_m=np.array([0.0, 60.0, -60.0, -60.0, 0.0]), vertical_m=np.zeros(n),
        uncorrectable=np.array([[True, True], *[[False, True]] * (n - 1)]), observed_row=np.zeros(n, dtype=int),
        matched_row=np.zeros(n), start=0, timed_out=False, outcome="landed")
    observed = _grid([_first(words, 900.0, ANGLE_LEVEL), *[{}] * 5])
    _, outside, breaks = closed_loop.outside_rows(sentence, observed, 0, words, [90.0])["lateral"]
    assert outside.tolist() == [False, True, True, True, False] and not breaks.any()
    assert [int(r[0][HEADING]) for r in rows[1:3]] == [words.n_heading - 1, 1] and rows[2][1][HEADING]
    cancelled = grid.copy()
    cancelled[2, HEADING] = 0                                    # the observed word: the correction only cancelled
    cancelled[3, HEADING] = 1
    _, _, breaks = closed_loop.outside_rows(_changed(sentence, grid=cancelled), observed, 0, words, [90.0])["lateral"]
    assert breaks.tolist() == [False, False, True, False, False]



def test_no_angle_correction_while_the_executor_holds_the_level_reached_by_a_descent():
    """§4.9 vertical item 1: a level reached by a descent says no angle word, the descent class stays in force; once the
    executor holds the level (its level-off begun) no correction starts, and one in force ends — while it still
    descends toward the level, corrections go on."""
    words = Words(spec())
    corrector = Corrector(_grid([_first(words, 1080.0, 2), *[{}] * 5]), 0, 0, 1, words, [90.0])
    corrector.row(0, 0.0, 0.0, 1300.0, holding=False, past_end=False)
    assert int(corrector.row(1, 0.0, 30.0, 1110.0, holding=False, past_end=False)[0][ANGLE]) == 3   # 30 m above: still descending
    said, added = corrector.row(2, 0.0, 30.0, 1081.5, holding=True, past_end=False)   # the same error, the level held: ended
    assert int(said[ANGLE]) == 2 and added[ANGLE]
    assert int(corrector.row(3, 0.0, 30.0, 1080.0, holding=True, past_end=False)[0][ANGLE]) == UNCHANGED   # and none starts


def test_a_repeated_observed_position_is_no_segment():
    """A row at the position of the one before (here the last) gives no segment of zero length to measure against."""
    path = ObservedPath(np.array([0.0, 100.0, 200.0, 200.0]), np.zeros(4), np.full(4, 500.0), 0)
    lateral, vertical, *_ = path.match(150.0, -10.0, 510.0)
    assert lateral == pytest.approx(10.0) and vertical == pytest.approx(10.0)
    match = path.match(250.0, -10.0, 510.0)                          # past the end: the last segment, not a point
    assert match.lateral_m == pytest.approx(10.0) and match.past_end


def _reference(tmp_path, monkeypatch, results):
    """A closed-loop reference written from ``results`` (`_reference_results` stubbed: no harvest to draw from)."""
    batch, inputs, words = _batch()
    params = _params()
    (sentence,) = closed_loop.read(batch, inputs, params, words, device=CPU)
    target = tmp_path / closed_loop.CLOSED_LOOP_DIRECTORY / closed_loop.CONFORMANCE
    target.parent.mkdir(parents=True)
    monkeypatch.setattr(closed_loop, "_reference_results",
                        lambda instructions, p, w, intervals, device: {2.0: (["KXXX:test"], [sentence])})
    closed_loop.write_reference(tmp_path, params, words, [2.0], git={"dirty": True}, target=target)
    monkeypatch.setattr(closed_loop, "_reference_results",
                        lambda instructions, p, w, intervals, device: {2.0: results(sentence)})
    return closed_loop.check(tmp_path, params, words)


def test_the_conformance_check_passes_the_same_reading_and_finds_every_change(tmp_path, monkeypatch):
    assert _reference(tmp_path / "same", monkeypatch, lambda s: (["KXXX:test"], [s])).passed
    moved = _reference(tmp_path / "moved", monkeypatch, lambda s: (["KXXX:test"], [
        _changed(s, states=s.rows.states + [0.0, 0.0, 1e-3, 0.0, 0.0, 0.0])]))
    assert not moved.passed and moved.largest_state_difference_m == pytest.approx(1e-3)
    lost = _reference(tmp_path / "nan", monkeypatch, lambda s: (["KXXX:test"], [
        _changed(s, vertical_m=np.where(np.arange(len(s.withheld.vertical_m)) == 3, np.nan, s.withheld.vertical_m))]))
    assert not lost.passed
    shorter = _reference(tmp_path / "shape", monkeypatch, lambda s: (["KXXX:test"], [
        _changed(s, grid=s.rows.grid[:-1], correction=s.rows.correction[:-1], states=s.rows.states[:-1], lateral_m=s.withheld.lateral_m[:-1],
                vertical_m=s.withheld.vertical_m[:-1])]))
    assert not shorter.passed
    other = _reference(tmp_path / "other", monkeypatch, lambda s: (["KXXX:other"], [s]))
    assert not other.passed and "other flights" in str(other.mismatches)
    flags = _reference(tmp_path / "flags", monkeypatch, lambda s: (["KXXX:test"], [
        _changed(s, uncorrectable=~s.withheld.uncorrectable)]))
    assert not flags.passed and "the uncorrectable differ" in str(flags.mismatches)
    judged = _reference(tmp_path / "outcome", monkeypatch, lambda s: (["KXXX:test"], [
        _changed(s, outcome="timeout" if s.withheld.outcome != "timeout" else "landed")]))
    assert not judged.passed and "outcome" in str(judged.mismatches)                              # D74
    refused = _reference(tmp_path / "refused", monkeypatch, lambda s: (["KXXX:test"], [Refused("too short")]))
    assert not refused.passed


def test_reading_closed_loop_sentences_runs_the_checks_and_refuses_a_difference_by_name(tmp_path, monkeypatch):
    """D69, D73: `require_conforming_closed_loop` opens the executor spec (the labeller's and the executor's checks) and
    reads the closed-loop reference again, every time: alike, it returns what `open_executor` returns and writes nothing;
    a reading off the reference is refused by name."""
    _reference(tmp_path, monkeypatch, lambda s: (["KXXX:test"], [s]))
    batch, inputs, words = _batch()
    monkeypatch.setattr(replay, "CHECKED", {})
    monkeypatch.setattr(replay, "open_executor",
                        lambda executor, instructions: (_params(), {"sha256": "e", "checks": {"executor": "x"}}, words))
    before = sorted(p.name for p in (tmp_path / closed_loop.CLOSED_LOOP_DIRECTORY / closed_loop.CONFORMANCE).iterdir())
    params, record, opened_words = closed_loop.require_conforming_closed_loop(tmp_path, tmp_path / "executor")
    assert params == _params() and opened_words is words and record["sha256"] == "e"
    assert record["checks"]["executor"] == "x" and record["checks"]["closed_loop"]["flights"] == 1
    assert sorted(p.name for p in (tmp_path / closed_loop.CLOSED_LOOP_DIRECTORY / closed_loop.CONFORMANCE).iterdir()) \
        == before
    monkeypatch.setattr(closed_loop, "_reference_results", lambda instructions, p, w, intervals, device: {
        2.0: (["KXXX:test"], [_changed(s, states=s.rows.states + 1e-3) for s in [closed_loop.read(batch, inputs, _params(),
                                                                                             words, device=CPU)[0]]])})
    closed_loop.require_conforming_closed_loop(tmp_path, tmp_path / "executor")     # checked once in this process
    replay.CHECKED.clear()                                                          # a new process checks again
    with pytest.raises(ValueError, match=f"reads {tmp_path.name}'s closed-loop reference otherwise: 2 s"):
        closed_loop.require_conforming_closed_loop(tmp_path, tmp_path / "executor")


def test_the_replay_refuses_a_sentence_of_another_first_row(monkeypatch):
    batch, inputs, words = _batch()
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    stored = {0: _changed(sentence, first_row=3)}
    with pytest.raises(ValueError, match="starts at 2 s row 3"):
        closed_loop.replay_batch(batch, stored, words)


def test_read_chunked_puts_each_result_in_its_place(monkeypatch):
    """Refusals stay in place, and a sentence read in a chunk is the one read alone."""
    batch, inputs, words = _batch()
    params = _params()
    (alone,) = closed_loop.read(batch, inputs, params, words, device=CPU)
    short = replay.Sentence(grid=batch.sentences[0].grid[:9], instructions=[], first_row=0)
    batch = replay.Batch(indices=[0, 1, 2], signals=batch.signals * 3, observed=batch.observed * 3, series=[None] * 3,
                         readings=batch.readings * 3,
                         sentences=[short, batch.sentences[0], batch.sentences[0]], row_interval_s=2.0,
                         geometries=batch.geometries * 3,
                         approach_ias_mps=batch.approach_ias_mps * 3, groups=batch.groups * 3, drawn={})
    monkeypatch.setattr(closed_loop, "start_inputs", lambda part, params, step_s, device: closed_loop._rows(
        closed_loop.FlightInputs(*(torch.cat([getattr(inputs, n)] * len(part.sentences)) for n in (
            "initial_state", "aero_params", "frame_params", "max_thrust_n"))), list(range(len(part.sentences)))))
    for chunk in (1, 3):                    # 3: the refused flight ahead of the flown ones in one reading (D74's index)
        results = closed_loop.read_chunked(batch, params, words, chunk=chunk, device=CPU)
        assert isinstance(results[0], Refused)
        for result in results[1:]:
            assert np.array_equal(result.rows.grid, alone.rows.grid) and np.array_equal(result.rows.states, alone.rows.states)
            assert result.withheld.outcome == alone.withheld.outcome


def test_the_rows_without_a_correction_are_marked_for_the_ablation():
    """D34: the rows where §4.9 makes no correction — the first predicted step, a new observed word, a level hold, a climb,
    no class beyond — are marked per column, and each flight's largest error on them is read."""
    words = Words(spec())
    corrector = Corrector(_grid([_first(words, 600.0, words.n_descent), {HEADING: 2}, {}, {}]), 0, 0, 1, words, [90.0])
    corrector.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    assert corrector.uncorrectable.tolist() == [True, True]                  # the first predicted step
    corrector.row(1, 40.0, 30.0, 900.0, holding=False, past_end=False)
    assert corrector.uncorrectable.tolist() == [True, True]                  # a new heading word; no class steeper
    corrector.row(2, 40.0, -30.0, 900.0, holding=False, past_end=False)
    assert corrector.uncorrectable.tolist() == [False, False]                # both corrected
    corrector.row(3, 40.0, -30.0, 600.0, holding=True, past_end=False)
    assert corrector.uncorrectable.tolist() == [False, True]                 # the level held
    climb = Corrector(_grid([_first(words, 1200.0, words.angle_climb), {}]), 0, 0, 1, words, [90.0])
    climb.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    climb.row(1, 0.0, -30.0, 900.0, holding=False, past_end=False)
    assert climb.uncorrectable.tolist() == [False, True]                     # a climb
    shallow = Corrector(_grid([_first(words, 600.0, 1), {}, {}]), 0, 0, 1, words, [90.0])
    shallow.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    shallow.row(1, 0.0, -30.0, 900.0, holding=False, past_end=False)
    assert shallow.uncorrectable.tolist() == [False, True]                   # descent 1, too low: none shallower
    shallow.row(2, 0.0, 5.0, 900.0, holding=False, past_end=False)
    assert shallow.uncorrectable.tolist() == [False, False]                  # within H: nothing to correct
    level = Corrector(_grid([_first(words, 600.0, 2), {ALTITUDE: words.altitude_index(480.0)}]), 0, 0, 1, words, [90.0])
    level.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    level.row(1, 0.0, 30.0, 900.0, holding=False, past_end=False)
    assert level.uncorrectable.tolist() == [False, True]                     # a new observed altitude word
    assert closed_loop.uncorrected_m(np.array([5.0, -40.0, 9.0]), np.array([True, False, True])) == 9.0
    batch, inputs, words = _batch()
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    assert sentence.withheld.uncorrectable.shape == (len(sentence.rows.grid), 2) and sentence.withheld.uncorrectable[0].all()


# ---- the observed words at the place (D42)
def test_the_matched_point_has_the_observed_time_of_its_place():
    """D42: the observed time of the matched point is when the observed aircraft was there — between two rows, in
    proportion; a position the observed aircraft held for a row is reached at the earlier row and left at the later;
    at or past the end, the last row (a position held there too)."""
    path = ObservedPath(np.array([0.0, 100.0, 100.0, 200.0, 200.0]), np.zeros(5), np.full(5, 500.0), 0)
    assert path.match(50.0, 0.0, 500.0).row == pytest.approx(0.5)
    assert path.match(150.0, 3.0, 500.0).row == pytest.approx(2.5)
    assert path.match(260.0, 0.0, 500.0).row == pytest.approx(4.0)
    # D45 "less than Δ/2 after": at Δ = 2 s a word 0.8 s after the matched point is reached, 1 s after (a tie) waits; at
    # Δ = 8 s, 3.8 s after is reached, 4 s after waits
    assert [last_heard_row(row, every) for row, every in ((2.6, 1), (2.5, 1), (7.1, 4), (7.0, 4))] == [3, 2, 9, 8]


def test_a_flight_on_schedule_hears_in_closed_loop_the_words_the_interval_grid_says():
    """A14: a flown aircraft exactly where the observed one was at each Δ row hears the words in force that the Δ grid
    says at that row — every word, a tie on the later row in both — at Δ = 2, 4 and 8 s."""
    words = Words(spec())
    rows = [_first(words), *[{HEADING: k % words.n_heading} if k % 3 else {} for k in range(1, 48)]]
    grid = _grid(rows)
    for every in (1, 2, 4):
        expected = in_force(on_interval(grid, 0, 2.0 * every, 2.0, np.full(len(rows), 900.0), words, [90.0]))
        corrector = Corrector(grid, 0, 0, every, words, [90.0])
        said = [corrector.row(float(k * every), 0.0, 0.0, 900.0, holding=False, past_end=False)[0]
                for k in range(len(expected))]
        assert np.array_equal(in_force(np.array(said)), expected)


def test_past_the_end_of_the_observed_path_its_last_segment_goes_on_without_a_height():
    """D44: an observed slice stops short of the threshold. Past its end the path goes on along its last segment's line,
    e_y measured against it; there is no observed height (e_h NaN)."""
    e = np.arange(5) * 100.0
    path = ObservedPath(e, np.zeros(5), 500.0 - e * 0.05, 0)
    inside = path.match(350.0, 4.0, 480.0)
    assert not inside.past_end and inside.vertical_m == pytest.approx(480.0 - (500.0 - 350.0 * 0.05))
    match = path.match(700.0, 4.0, 500.0 - 700.0 * 0.05)
    assert match.past_end and math.isnan(match.vertical_m) and match.lateral_m == pytest.approx(-4.0)
    assert match.row == 4.0 and match.along_m == pytest.approx(700.0)


def test_past_the_end_of_the_observed_path_no_correction_starts_and_one_in_force_ends():
    """D44: past the end the reading says no correction, lateral or vertical; one in force ends (the observed word again),
    and the rows count as rows without correction (D34)."""
    words = Words(spec())
    final = {ALTITUDE: words.altitude_no_level_off, ANGLE: 3}
    corrector = Corrector(_grid([_first(words), final, *[{}] * 6]), 0, 0, 1, words, [90.0])
    corrector.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    corrector.row(1, 0.0, 0.0, 900.0, holding=False, past_end=False)
    said, added = corrector.row(2, 50.0, 20.0, 800.0, holding=False, past_end=False)
    assert int(said[HEADING]) == words.n_heading - 1 and int(said[ANGLE]) == 4 and added[[HEADING, ANGLE]].all()
    said, added = corrector.row(3, 50.0, math.nan, 790.0, holding=False, past_end=True)    # ended: the observed words
    assert int(said[HEADING]) == 0 and int(said[ANGLE]) == 3 and corrector.uncorrectable.all()
    said, _ = corrector.row(4, 80.0, math.nan, 780.0, holding=False, past_end=True)        # and none starts
    assert (said == UNCHANGED).all() and corrector.uncorrectable.all()


def test_a_flight_whose_observed_path_ends_early_gets_no_correction_past_its_end():
    """D44 on a flight: the observed slice ends 60 s (4.5 km) before the threshold; the flown aircraft flies on to the
    threshold on its words, its rows there say no correction, have no e_h and count as rows without correction."""
    batch, inputs, words = _batch(cut=30)
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    assert isinstance(sentence, ClosedLoopSentence)
    past = np.isnan(sentence.withheld.vertical_m)
    assert past.sum() > 20 and past[int(np.argmax(past)):].all()          # from the end of the path to the threshold
    observed = closed_loop.in_force(batch.readings[0].words[batch.sentences[0].first_row:])[sentence.withheld.observed_row]
    held = closed_loop.in_force(sentence.rows.grid)
    assert np.array_equal(held[past][:, [HEADING, ANGLE]], observed[past][:, [HEADING, ANGLE]])  # the observed words
    assert sentence.rows.correction[past].sum() <= 2 and sentence.withheld.uncorrectable[past].all()  # at most a correction ended
    assert np.isfinite(sentence.withheld.lateral_m).all()
    assert closed_loop.largest_m(sentence.withheld.vertical_m) == float(np.abs(sentence.withheld.vertical_m[~past]).max())


def test_the_observed_words_wait_while_the_flown_aircraft_is_behind():
    """D42: a row whose matched point has not reached the next observed row says no observed word — a correction goes
    on — and the word comes at the row whose matched point reaches it."""
    words = Words(spec())
    corrector = Corrector(_grid([_first(words), {HEADING: 2}, {}, {}]), 0, 0, 1, words, [90.0])
    corrector.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    said, _ = corrector.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)                 # behind: still at row 0
    assert (said == UNCHANGED).all() and corrector.observed_row == 0
    said, added = corrector.row(0, 50.0, 0.0, 900.0, holding=False, past_end=False)            # behind, and 50 m right: corrected
    assert int(said[HEADING]) == words.n_heading - 1 and added[HEADING]
    said, added = corrector.row(1, 50.0, 0.0, 900.0, holding=False, past_end=False)            # the word, where it was heard
    assert int(said[HEADING]) == 2 and not added[HEADING] and corrector.observed_row == 1
    corrector.row(3, 0.0, 0.0, 900.0, holding=False, past_end=False)
    said, _ = corrector.row(2, 0.0, 0.0, 900.0, holding=False, past_end=False)                 # a matched point back: no row twice
    assert (said == UNCHANGED).all() and corrector.observed_row == 3


def test_a_flown_aircraft_ahead_passes_two_observed_rows_in_one_and_hears_the_last_word_of_each_column():
    words = Words(spec())
    corrector = Corrector(_grid([_first(words), {HEADING: 2, SPEED: words.speed_index(80.0)}, {HEADING: 4}, {}]), 0, 0,
                          1, words, [90.0])
    corrector.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)
    said, added = corrector.row(2, 50.0, 0.0, 900.0, holding=False, past_end=False)
    assert int(said[HEADING]) == 4 and int(said[SPEED]) == words.speed_index(80.0) and not added.any()
    assert corrector.observed_row == 2 and corrector.uncorrectable[0]          # a new observed heading word: no correction
    said, _ = corrector.row(2, 0.0, 0.0, 900.0, holding=False, past_end=False)                 # nothing new
    assert (said == UNCHANGED).all()


def test_the_first_predicted_step_says_every_column_in_force_where_the_matched_point_is():
    words = Words(spec())
    corrector = Corrector(_grid([_first(words), {HEADING: 2}, {}, {HEADING: 4}]), 0, 1, 1, words, [90.0])
    said, added = corrector.row(0, 0.0, 0.0, 900.0, holding=False, past_end=False)             # never before the first predicted step
    assert int(said[HEADING]) == 2 and (said != UNCHANGED).all() and not added.any() and corrector.observed_row == 1


def test_a_heading_word_passed_with_a_runway_word_is_said_in_the_frame_where_it_is_heard():
    """D46: a heading word is relative to the course of the runway in force when it is heard. Passed with a later runway
    word, it is the class nearest its absolute track under the new course: the same class after a parallel runway, the
    class of its track after a runway 6° off; no sentence is refused. Said after the runway word, it is heard as said."""
    words = Words(spec())
    final = {RUNWAY: 0, HEADING: 0, ALTITUDE: words.altitude_no_level_off, ANGLE: 3, SPEED: words.speed_index(70.0)}
    go = {RUNWAY: RUNWAY_GO_AROUND, ALTITUDE: words.altitude_index(900.0), ANGLE: words.angle_climb}

    def heard(rows, courses):
        corrector = Corrector(_grid([final, go, *rows]), 0, 0, 1, words, courses)
        corrector.row(0, 0.0, 0.0, 300.0, holding=False, past_end=False)
        corrector.row(1, 0.0, 0.0, 300.0, holding=False, past_end=False)
        said, added = corrector.row(3, 0.0, 0.0, 900.0, holding=False, past_end=False)
        assert int(said[RUNWAY]) == 1 and not added.any()
        return int(said[HEADING]), corrector.target_deg

    before = [{HEADING: 2}, {RUNWAY: 1}]                         # said under runway 0 (90°), heard under runway 1
    assert heard(before, [90.0, 90.005]) == (2, pytest.approx(100.005))
    assert heard(before, [90.0, 96.0]) == (1, pytest.approx(101.0))     # 100° under 96°: +4°, class 1 (+5°)
    assert heard([{RUNWAY: 1}, {HEADING: 2}], [90.0, 96.0]) == (2, pytest.approx(106.0))


def test_a_change_of_runway_alone_says_no_heading_word_and_a_correction_is_in_the_frame_where_it_is_heard():
    """§3.3: the executor keeps its absolute track across a change of runway, so the reading says no heading word for it;
    a correction that starts afterwards is one class from the observed word's track under the new course (D46)."""
    words = Words(spec())
    final = {RUNWAY: 0, HEADING: 2, ALTITUDE: words.altitude_no_level_off, ANGLE: 3, SPEED: words.speed_index(70.0)}
    go = {RUNWAY: RUNWAY_GO_AROUND, ALTITUDE: words.altitude_index(900.0), ANGLE: words.angle_climb}
    corrector = Corrector(_grid([final, go, {RUNWAY: 1}, {}, {}]), 0, 0, 1, words, [90.0, 96.0])
    corrector.row(0, 0.0, 0.0, 300.0, holding=False, past_end=False)
    corrector.row(1, 0.0, 0.0, 300.0, holding=False, past_end=False)
    said, _ = corrector.row(2, 0.0, 0.0, 900.0, holding=False, past_end=False)
    assert int(said[RUNWAY]) == 1 and int(said[HEADING]) == UNCHANGED and corrector.target_deg == pytest.approx(100.0)
    said, added = corrector.row(3, 40.0, 0.0, 900.0, holding=False, past_end=False)     # right of the path
    assert int(said[HEADING]) == 0 and added[HEADING]          # 100° under 96°: class 1, one class left: class 0 (96°)


def test_the_observed_heading_words_of_a_turn_are_said_at_the_nearest_row_on_average():
    """D45: the lateness of each observed heading word said (the matched point's observed time at the row that says it
    minus the word's 2 s time) is spread over ±Δ/2 and near zero on average at Δ = 2, 4 and 8 s; said at the first row
    past its place (A10) it was Δ/2 late on average. A slow turn: a word every 10 s, the flown aircraft 3 % slower."""
    words = Words(spec())
    first_row = 3                                     # the sentence's first Δ row: 2 s row 3 of the reading
    rows = [_first(words), {}, {}, {},
            *[{HEADING: (k // 5) % words.n_heading} if k % 5 == 0 else {} for k in range(1, 200)]]
    grid = _grid(rows)
    for every in (1, 2, 4):
        corrector = Corrector(grid, first_row, 0, every, words, [90.0])
        lateness, said_rows, s = [], [], 0
        while corrector.next < len(rows) - first_row - 1:
            said, added = corrector.row(s * every * 0.97, 0.0, 0.0, 900.0, holding=False, past_end=False)
            said_rows.append((said, added, corrector.observed_row, corrector.matched_row))
            if s and said[HEADING] != UNCHANGED:      # the word's 2 s row, counted from the sentence's first row
                lateness.append(corrector.matched_row - (corrector.heading_row[corrector.observed_row] - first_row))
            s += 1
        lateness = np.array(lateness) * 2.0
        assert len(lateness) >= 35 and np.abs(lateness).max() <= every + 1e-9          # within ±Δ/2
        assert abs(lateness.mean()) < 0.5
        # the runner's readout reads the same from the sentence and the reading from its first row
        n = len(said_rows)
        sentence = _sentence(
            first_row=0, grid=np.array([r[0] for r in said_rows]), correction=np.array([r[1] for r in said_rows]),
            states=np.zeros((n, 6)), on_interval=np.ones(n, bool), lateral_m=np.zeros(n), vertical_m=np.zeros(n), uncorrectable=np.ones((n, 2), bool),
            observed_row=np.array([r[2] for r in said_rows]), matched_row=np.array([r[3] for r in said_rows]), start=0,
            timed_out=False, outcome="landed")
        assert np.allclose(closed_loop.heading_lateness_rows(sentence, grid[first_row:]) * 2.0, lateness)


def test_a_flown_aircraft_behind_hears_the_turn_where_the_observed_one_did_and_flies_past_the_observed_time():
    """D42: told 70 m/s from the downwind (the observed aircraft flies 100 m/s there), the flown aircraft falls behind.
    It hears the turn onto the base where the observed aircraft heard it, later than the observed time, flies the final
    within Y of the path, and is not cut at the end of the open-loop sentence (§4.9 item 6): it lands later."""
    batch, inputs, words = _batch(speeds={0: 70.0})
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    assert isinstance(sentence, ClosedLoopSentence)
    open_grid, start = batch.sentences[0].grid, sentence.rows.start
    turn = start + int(np.argmax(open_grid[start + 1:, HEADING] != UNCHANGED)) + 1      # the first turn word
    heard = int(np.argmax(sentence.withheld.observed_row >= turn))
    assert sentence.withheld.observed_row[heard] == turn and int(sentence.rows.grid[heard, HEADING]) == int(open_grid[turn, HEADING])
    assert not sentence.rows.correction[heard, HEADING]
    assert start + heard > turn + 5                                     # behind: more than 10 s after the observed time
    observed = batch.signals[0]
    flown = sentence.rows.states[start + heard, :2]
    assert math.dist(flown, (observed.e_m[turn], observed.n_m[turn])) < 250.0         # within one row's flight
    assert len(sentence.rows.grid) > len(open_grid) - start and not sentence.withheld.timed_out
    (short,) = closed_loop.read(batch, inputs, replace(_params(), timeout_factor=1.0), words, device=CPU)
    assert short.withheld.timed_out and len(short.rows.grid) == len(open_grid) - start         # the limit: the observed time
    assert np.abs(sentence.withheld.lateral_m[-len(sentence.withheld.lateral_m) // 4:]).max() < words.spec.closed_loop_lateral_m
    assert -2.0 * 75.0 < sentence.rows.states[-1, 0] <= 0.0               # its last row starts one row before the threshold


def test_a_flown_aircraft_ahead_hears_the_go_around_before_the_threshold_where_the_observed_one_went_around(
        monkeypatch):
    """D42: told 90 m/s on the final (the observed aircraft flies 70 m/s), the flown aircraft is ahead of the observed
    one; said at the observed time, the go-around would come after it crossed the threshold (vocabulary §9.7, KSTL). Said at the
    place, it comes where the observed aircraft went around, before the threshold, and the sentence flies it again as a
    go-around."""
    from ts_transformer.tests.test_instruction_labeller import GO_AROUND_LEGS

    probe, _, _ = _batch(legs=GO_AROUND_LEGS, track_deg=90.0, altitude_m=900.0)
    (go,) = np.flatnonzero(probe.sentences[0].grid[:, RUNWAY] == RUNWAY_GO_AROUND)
    batch, inputs, words = _batch(legs=GO_AROUND_LEGS, track_deg=90.0, altitude_m=900.0,
                                  speeds={0: 90.0, int(go): 70.0})
    params = _params()
    (sentence,) = closed_loop.read(batch, inputs, params, words, device=CPU)
    assert isinstance(sentence, ClosedLoopSentence)
    start = sentence.rows.start
    (heard,) = np.flatnonzero(sentence.rows.grid[:, RUNWAY] == RUNWAY_GO_AROUND)
    assert sentence.withheld.observed_row[heard] >= go > sentence.withheld.observed_row[heard - 1]     # the first row that reaches it
    assert start + heard < go - 5                                     # ahead: more than 10 s before the observed time
    observed = batch.signals[0]
    flown = sentence.rows.states[start + heard, :2]
    assert flown[0] < 0.0 and math.dist(flown, (observed.e_m[go], observed.n_m[go])) < 250.0   # before the threshold
    assert sentence.rows.states[go, 0] > 0.0          # at the observed time it is past the threshold (e = 0) already
    _, moved, _, replayed = _replayed(batch, inputs, sentence, params, words, monkeypatch)
    assert moved.sentences[0].go_arounds == 1 and replayed.modes["go_around"][0].any()


def test_the_replay_reads_the_speed_words_and_how_far_along_the_path_the_flown_aircraft_is(monkeypatch):
    """Vocabulary §9.8, A11: each flown sentence's speed words other than "unspecified", and the largest distance along the observed
    path from the observed aircraft of the same time before "unspecified" — told 70 m/s on the 100 m/s downwind (the
    open loop, the time clock), the flown aircraft falls far behind."""
    from ts_transformer.experiments.executor_replay import along_columns

    params = _params()
    read = {}
    for name, speeds, interval_s in (("observed", None, 2.0), ("slow", {0: 70.0}, 2.0), ("slow_4", {0: 70.0}, 4.0)):
        batch, inputs, words = _batch(interval_s, speeds=speeds)
        monkeypatch.setattr(replay.Batch, "inputs", lambda self, rule, device, inputs=executor_inputs(
            batch.observed[0], batch.geometries[0], batch.sentences[0].first_row): inputs)
        flown = replay.fly_sentences(batch, params, words, device=CPU)
        (read[name],) = along_columns(words)(batch, flown, replay.judge_batch(batch, flown, words))
        grid = batch.sentences[0].grid
        held = closed_loop.in_force(grid)[:, SPEED] == words.speed_unspecified
        before = int(np.argmax(held)) if held.any() else len(grid)
        assert read[name]["speed_words"] == int((grid[:before, SPEED] != UNCHANGED).sum())     # before "unspecified"
    assert read["slow"]["speed_words"] == read["slow_4"]["speed_words"] == 1
    assert read["slow"]["largest_along_m"] > 1000.0 > read["observed"]["largest_along_m"]
    assert read["slow_4"]["largest_along_m"] == pytest.approx(read["slow"]["largest_along_m"], rel=0.1)


def test_the_runner_reads_the_same_files_and_summary_with_any_number_of_workers(tmp_path, monkeypatch):
    """`instruction_closed_loop --workers N` reads each split (drawn once, every row interval in turn) in its own process,
    and ``--train-parts P`` the train split in P parts of its draw: the files and the summary are those of one process
    reading each split whole. Synthetic flights have no harvest: their draw (in parts, as `replay.part_of` cuts it) and
    executor inputs stand in."""
    import json as json_module

    from ts_transformer.autopilot.flights import FlightInputs
    from ts_transformer.experiments import instruction_closed_loop as runner
    from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
    from ts_transformer.tests.test_instruction_conformance import _artefact as labelled_artefact

    clean = {"head": "h", "dirty": False}

    def draw(directory, split, one, words, *, per_airport, seed, groups, part=(0, 1), go_around_per_airport=0):
        stored, signals, geometries = load_sentences(directory, split, one), load_signals(directory, split), \
            load_candidates(directory)
        indices = replay.part_of([int(i) for i in stored["signal_index"]], part)
        readings = [read_flight(signals[i], geometries[signals[i].airport], one, words) for i in indices]
        n = len(indices)
        return replay.Drawn(indices=indices, signals=[signals[i] for i in indices], series=[None] * n,
                            groups=[replay.OWN] * n, geometries=geometries,
                            excluded_seen=Counter({"no aircraft dynamics": n}), description={"split": split, "seed": seed, "per_airport": per_airport, "groups": list(groups),
                                         "threshold_crossing_heights_m": {}, "pool": 2 * n, "read": 2 * n, "flights": n,
                                         "excluded": {"no aircraft dynamics": n}, "by_group": {"own": n}}), readings

    def inputs(batch, params, step_s, *, device):
        start = closed_loop.start_row(batch.row_interval_s) * int(round(batch.row_interval_s / step_s))
        parts = [executor_inputs(o, g, sentence.first_row + start)
                 for o, g, sentence in zip(batch.observed, batch.geometries, batch.sentences)]
        return FlightInputs(*(torch.cat([getattr(p, name) for p in parts])
                              for name in ("initial_state", "aero_params", "frame_params", "max_thrust_n")))

    words = Words(spec())
    monkeypatch.setattr(replay, "open_executor",
                        lambda executor, instructions: (_params(), {"sha256": "e", "checks": {}}, words))
    monkeypatch.setattr(replay, "draw_readings", draw)
    monkeypatch.setattr(replay, "flight_approach_ias_mps", lambda series, group: approach_speed_ias_mps("A320", 62000.0))
    monkeypatch.setattr(closed_loop, "start_inputs", inputs)
    monkeypatch.setattr(runner, "git_state", lambda: clean)
    monkeypatch.setattr(runner, "POOL_OPTIONS", {"start_method": "fork"})
    read = {}
    runs = ((1, 1), (2, 1), (1, 2), (2, 3))                                   # (workers, train parts)
    for workers, parts in runs:
        directory = labelled_artefact(tmp_path / f"w{workers}p{parts}", monkeypatch, straight_in=True)
        assert runner.main(["--instructions", str(directory), "--executor", str(tmp_path), "--row-interval-s", "2", "8",
                            "--workers", str(workers), "--train-parts", str(parts)]) == 0
        assert not (directory / "closed_loop" / runner.PARTS).exists()
        files = sorted(p.name for p in (directory / "closed_loop").glob("*.npz"))
        summary = json_module.loads((directory / "closed_loop" / "summary.json").read_text(encoding="utf-8"))
        summary.pop("written_utc")
        arrays = {}
        for name in files:
            with np.load(directory / "closed_loop" / name) as data:
                arrays[name] = {key: data[key] for key in data.files}
        read[workers, parts] = (files, summary, arrays)
    whole = read[1, 1]
    assert whole[0] == ["select_2s.npz", "select_8s.npz", "train_2s.npz", "train_8s.npz", "val_2s.npz", "val_8s.npz"]
    assert list(whole[1]["splits"]) == ["train", "select", "val"]
    train = whole[1]["splits"]["train"]
    assert train["drawn"]["flights"] == 3 and train["intervals"]["2"]["sentences"] == 3
    assert train["drawn"]["excluded"] == {"no aircraft dynamics": 3} and train["drawn"]["pool"] == 6
    # D85: of val, the flights drawn and the sentences written, no reading
    val = whole[1]["splits"]["val"]
    assert set(val["drawn"]) == {"flights", "readings"} and val["drawn"]["flights"] >= 1
    assert all(set(one) == {"sentences", "readings"} and one["sentences"] == val["drawn"]["flights"]
               for one in val["intervals"].values())
    for run in runs[1:]:
        files, summary, arrays = read[run]
        assert files == whole[0] and json_module.dumps(summary) == json_module.dumps(whole[1]), run   # text too
        for name in files:
            assert list(arrays[name]) == list(whole[2][name]), (run, name)
            for key, value in whole[2][name].items():
                assert np.array_equal(value, arrays[name][key], equal_nan=value.dtype.kind == "f"), (run, name, key)
    for bad in (["--workers", "0"], ["--train-parts", "0"]):
        with pytest.raises(SystemExit):
            runner.main(["--instructions", str(tmp_path / "w1p1"), "--executor", str(tmp_path), "--row-interval-s", "2",
                         *bad])


def test_a_split_is_drawn_in_parts_of_its_one_permutation_and_their_descriptions_add_up(tmp_path, monkeypatch):
    """`replay.part_of`: the parts in turn are the whole order, sizes differing by at most one; a part that does not
    exist and a part with a cap per airport are refused. `draw_flights` in parts: their flights in turn are the whole
    draw's, and `merge_descriptions` gives the whole draw's description as its text — a stand-in drawn first, exclusions
    tied — and refuses parts of different draws, a field it does not know and exclusions not the description's."""
    import json as json_module
    from types import SimpleNamespace

    order = list(range(10, 27))
    for n in (1, 2, 3, 5, 17):
        parts = [replay.part_of(order, (k, n)) for k in range(n)]
        assert [i for p in parts for i in p] == order and max(map(len, parts)) - min(map(len, parts)) <= 1
    for bad in ((3, 3), (-1, 2)):
        with pytest.raises(ValueError, match="does not exist"):
            replay.part_of(order, bad)
    with pytest.raises(ValueError, match="only with every flight"):
        replay.draw_flights(tmp_path, "train", order, per_airport=5, seed=1, part=(0, 2))
    flights = 40
    # each flight's group by its place in the drawing order: a stand-in first though own flights are the most, and two
    # reasons not drawn that end tied (3 each) — "no aircraft dynamics" met first, yet fewer of it in the first half
    drawing = [int(i) for i in np.random.default_rng(7).permutation(list(range(flights)))]
    by_place = {0: replay.STAND_IN, 1: "no aircraft dynamics", 2: "no identified type", 3: "no identified type",
                20: "no aircraft dynamics", 21: "no aircraft dynamics", 22: "no identified type"}
    group = {i: by_place.get(place, replay.OWN) for place, i in enumerate(drawing)}
    monkeypatch.setattr(replay, "load_candidates", lambda directory: {"KXXX": SimpleNamespace(candidates=[])})
    monkeypatch.setattr(replay, "load_signals", lambda directory, split: [SimpleNamespace(airport="KXXX", i=i)
                                                                        for i in range(flights)])
    monkeypatch.setattr(replay, "rebuild_series", lambda directory, signals: list(signals))
    monkeypatch.setattr(replay, "group_of", lambda series: group[series.i])
    groups = (replay.OWN, replay.STAND_IN)
    whole = replay.draw_flights(tmp_path, "train", list(range(flights)), per_airport=0, seed=7, groups=groups)
    assert list(whole.description["excluded"]) == ["no aircraft dynamics", "no identified type"]       # a tie
    assert list(whole.description["by_group"]) == [replay.STAND_IN, replay.OWN]
    for n in (2, 3, 7):
        parts = [replay.draw_flights(tmp_path, "train", list(range(flights)), per_airport=0, seed=7, groups=groups,
                                     part=(k, n)) for k in range(n)]
        assert [i for p in parts for i in p.indices] == whole.indices
        assert [g for p in parts for g in p.groups] == whole.groups
        merged = replay.merge_descriptions([(p.description, p.excluded_seen) for p in parts])
        assert json_module.dumps(merged) == json_module.dumps(whole.description), n
    first = whole.description, whole.excluded_seen
    with pytest.raises(ValueError, match="not of one draw"):
        replay.merge_descriptions([first, ({**first[0], "seed": 2}, first[1])])
    with pytest.raises(ValueError, match="not of one draw"):
        replay.merge_descriptions([first, ({**first[0], "new": 1}, first[1])])
    with pytest.raises(ValueError, match="not its description's"):
        replay.merge_descriptions([first, (first[0], Counter({"no identified type": 1}))])


def test_a_stored_track_is_in_0_to_360_degrees_and_a_remainder_of_360_is_0():
    """D82, A37: a track a hair below 0° has a remainder of exactly 360° in floats; every stored row writes it as 0°
    (`flights.compass_track`: the observed rows, and the flown rows of `start.state_rows`)."""
    from ts_transformer.autopilot.flights import compass_track
    from ts_transformer.autopilot.frame import Kinematics
    from ts_transformer.autopilot.start import state_rows

    assert np.remainder(-1e-14, 360.0) == 360.0                                  # the case itself
    assert compass_track(np.array([-1e-14, 0.0, 359.5, 360.0, 720.5])).tolist() == [0.0, 0.0, 359.5, 0.0, 0.5]
    one = torch.tensor([1.0], dtype=torch.float64)
    state = Kinematics(*(one if f.name != "track_deg" else torch.tensor([360.0], dtype=torch.float64)
                         for f in fields(Kinematics)))
    assert state_rows(state)[0, 3] == 0.0


@pytest.mark.parametrize("interval_s", [2.0, 4.0, 8.0])
def test_no_start_state_and_no_stored_observed_row_reads_a_sample_after_the_first_predicted_step(interval_s):
    """D77: the start state and the observed rows a closed-loop sentence stores take their velocity from the rows at or
    before the first predicted step, by each formal start rule — moving every sample after it changes neither; the
    rows before are what moves them (the window, cut at the flight's row 0: row 0 takes the line through rows 0 and 1)."""
    from ts_transformer.autopilot.flights import observed_rows, start_state
    from ts_transformer.autopilot.params import FORMAL_START_RULES

    batch, _, words = _batch(interval_s)
    signals, geometry = batch.observed[0], batch.geometries[0]
    every = int(round(interval_s / 2.0))
    first = batch.sentences[0].first_row + closed_loop.start_row(interval_s) * every
    later = replace(signals, **{name: np.where(np.arange(signals.n_rows) > first, getattr(signals, name) + shift,
                                               getattr(signals, name))
                                for name, shift in (("e_m", 300.0), ("n_m", -200.0), ("altitude_m", 50.0),
                                                    ("track_deg", 20.0), ("ground_speed_mps", 9.0),
                                                    ("vertical_rate_mps", 3.0))})
    rows = batch.sentences[0].first_row + np.arange(closed_loop.start_row(interval_s) * every)
    for rule in FORMAL_START_RULES:
        params = replace(_params(), start_rule=rule)
        assert np.array_equal(start_state(signals, first, rule, geometry, 62000.0),
                              start_state(later, first, rule, geometry, 62000.0)), rule
        assert np.array_equal(observed_rows(signals, rows, rule, geometry), observed_rows(later, rows, rule, geometry))
        read = {}
        for name, flight in (("observed", signals), ("later", later)):
            moved = replace(batch, observed=[flight])
            (read[name],) = closed_loop.read(moved, executor_inputs(flight, geometry, first, rule=rule), params, words,
                                             device=CPU)
        start = closed_loop.start_row(interval_s) * every
        assert np.array_equal(read["observed"].rows.states[:start], read["later"].rows.states[:start]), rule
        assert np.array_equal(read["observed"].rows.states[:start], observed_rows(signals, rows, rule, geometry))
        tracks = read["observed"].rows.states[:, 3]
        assert ((0.0 <= tracks) & (tracks < 360.0)).all()                                       # D82
    # A37: the formal rules read the positions and heights only — the data plane's track, ground speed and vertical rate
    # changed on every row (before the first predicted step too) change no start state and no stored observed row
    channels = replace(signals, track_deg=signals.track_deg + 37.0, ground_speed_mps=signals.ground_speed_mps * 1.3,
                       vertical_rate_mps=signals.vertical_rate_mps - 4.0)
    for rule in FORMAL_START_RULES:
        assert np.array_equal(start_state(signals, first, rule, geometry, 62000.0),
                              start_state(channels, first, rule, geometry, 62000.0)), rule
        assert np.array_equal(observed_rows(signals, rows, rule, geometry), observed_rows(channels, rows, rule, geometry))
    assert not np.array_equal(start_state(signals, first, "centred-fit-15s", geometry, 62000.0),
                              start_state(channels, first, "centred-fit-15s", geometry, 62000.0))   # the data plane's own
    earlier = replace(signals, e_m=np.where(np.arange(signals.n_rows) == first - 1, signals.e_m + 300.0, signals.e_m))
    assert not np.array_equal(start_state(signals, first, "displacement-2s", geometry, 62000.0),
                              start_state(earlier, first, "displacement-2s", geometry, 62000.0))
    zero, one = (start_state(signals, row, "trailing-fit-15s", geometry, 62000.0) for row in (0, 1))
    assert np.array_equal(zero[3:6], one[3:6])                    # row 0 alone in its window: rows 0 and 1, as row 1


def test_a_references_go_around_flights_are_drawn_after_the_others_among_those_not_drawn(monkeypatch):
    """A37 (§7.2 #2, #3): the references' draw takes up to ``go_around_per_airport`` more flights whose sentence has a
    go-around, among those the first draw did not take, after them, at most that many (`draw_flights` ``at_most``);
    refused in parts (each part would draw them)."""
    from types import SimpleNamespace

    #: six labelled flights (signal indices 0–5), with a go-around in the sentences of 1, 2 and 4
    sentences = {"signal_index": np.arange(6), "go_around_offsets": np.array([0, 0, 1, 2, 2, 3, 3]),
                 "words": np.zeros((6, 5), dtype=np.int16), "offsets": np.arange(7), "runway_index": np.zeros(6)}
    calls = []

    def draw_flights(directory, split, keys, *, per_airport, seed, groups, part=(0, 1), at_most=False):
        calls.append((list(keys), per_airport, at_most))
        taken = list(keys)[:per_airport]
        return replay.Drawn(indices=taken, signals=[SimpleNamespace(airport="KXXX", dataset_id=f"KXXX:{i}") for i in taken],
                            series=[None] * len(taken), groups=[replay.OWN] * len(taken), geometries={"KXXX": None},
                            description={"flights": len(taken)}, excluded_seen=Counter())

    monkeypatch.setattr(replay, "load_sentences", lambda directory, split, spec_: sentences)
    monkeypatch.setattr(replay, "draw_flights", draw_flights)
    monkeypatch.setattr(replay, "read_flight", lambda flight, geometry, spec_, words: SimpleNamespace(
        words=np.zeros((1, 5), dtype=np.int16), runway_index=0))
    drawn, readings = replay.draw_readings(None, "train", None, None, per_airport=2, seed=1, go_around_per_airport=2)
    assert calls == [([0, 1, 2, 3, 4, 5], 2, False), ([2, 4], 2, True)]      # 1 was drawn already
    assert drawn.indices == [0, 1, 2, 4] and len(readings) == 4
    assert drawn.description["go_around_flights"] == {"flights": 2}
    with pytest.raises(ValueError, match="not drawn in parts"):
        replay.draw_readings(None, "train", None, None, per_airport=2, seed=1, go_around_per_airport=2, part=(0, 2))


def test_the_readers_two_parts_have_their_names():
    """D82, A37: what a model may read (the rows) and what it must not (the withheld fields), by name."""
    assert [f.name for f in fields(ClosedLoopSentence)] == ["rows", "withheld"]
    assert [f.name for f in fields(SentenceRows)] == ["first_row", "start", "grid", "correction", "states", "on_interval"]
    assert [f.name for f in fields(Withheld)] == [
        "runway", "runway_index", "landing_time_utc", "capture_row", "go_around_rows", "stratum", "outcome", "timed_out",
        "lateral_m", "vertical_m", "uncorrectable", "observed_row", "matched_row"]


@pytest.mark.parametrize("runner", ["executor_replay", "closed_loop_start_check", "final_descent_tolerance"])
def test_a_readout_that_serves_a_choice_reads_train_or_select_only(runner, capsys):
    """A37, D85: the val days are read once, in the stage's validation readout; a readout runner refuses them."""
    import importlib

    from ts_transformer.instructions.artefact import READ_SPLITS, SEALED_READINGS

    assert READ_SPLITS == ("train", "select") and SEALED_READINGS == ("val",)
    module = importlib.import_module(f"ts_transformer.experiments.{runner}")
    with pytest.raises(SystemExit) as exited:
        module.main(["--instructions", "i", "--executor", "e", "--split", "val", "--row-interval-s", "4",
                     "--out", "o"])
    assert exited.value.code == 2 and "invalid choice: 'val'" in capsys.readouterr().err
