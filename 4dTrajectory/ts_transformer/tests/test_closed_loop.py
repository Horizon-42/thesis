"""The closed-loop reading (vocabulary §4.9, D32): the comparison, the corrections, and a flight read and flown again."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.closed_loop import ClosedLoopSentence, Corrector, ObservedPath
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
    executor_inputs, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec,
)

CPU = torch.device("cpu")


def _params():
    from ts_transformer.experiments.executor_spec import ROLL_RATE_DEG_S

    return ExecutorParams(cycle_s=1.0, bank_rate_deg_s=ROLL_RATE_DEG_S, path_time_constant_s=2.0, path_rate_factor=2.0,
                          timeout_factor=1.5)


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
    e, n, height = _loop_path()
    path = ObservedPath(e, n, height, 10)
    lateral, *_ = path.match(100.0, -30.0, 0.0)               # behind segment 10 (e = 500), 30 m right of east
    assert lateral == pytest.approx(30.0) and path.segment == 10


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


def _batch(interval_s=2.0, legs=DOWNWIND_BASE_FINAL, track_deg=270.0, altitude_m=1110.0, speeds=None, cut=0):
    """A synthetic flight's batch, its executor inputs at the first predicted step and the words; ``speeds``: the speed
    words its open-loop reading says instead of the observed ones, ``{2 s row: m/s}`` (none: the observed words);
    ``cut``: the observed flight and its reading end that many 2 s rows early (the slice stops short of the threshold)."""
    one, geometry = spec(), instruction_airport()
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
    batch = replay.Batch(indices=[0], signals=[observed], series=[None], readings=[reading], sentences=[sentence],
                         row_interval_s=interval_s, geometries=[geometry], vertical_paths=[()],
                         approach_ias_mps=[approach_speed_ias_mps("A320", 62000.0)], groups=[replay.OWN], drawn={})
    start = closed_loop.start_row(interval_s) * int(round(interval_s / one.step_s))
    return batch, executor_inputs(observed, geometry, start), words


@pytest.mark.parametrize("interval_s", [2.0, 4.0])
def test_a_flight_whose_words_leave_an_offset_is_brought_back_by_heading_corrections(interval_s):
    """D32: the turn onto the final leaves the flown path right of the observed one; heading words one class toward the
    path bring it back, and it ends within the tolerance. The rows before the first predicted step stay observed."""
    batch, inputs, words = _batch(interval_s)
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    assert isinstance(sentence, ClosedLoopSentence)
    start, every = sentence.start, int(round(interval_s / 2.0))
    assert start == 16 / interval_s and (sentence.grid[0] != UNCHANGED).all() and not sentence.correction[0].any()
    observed = batch.signals[0]
    rows = np.arange(start * every)                                    # the observed 2 s rows (D51)
    assert np.array_equal(sentence.states[:start * every, :3], np.column_stack([observed.e_m[rows], observed.n_m[rows],
                                                                               observed.altitude_m[rows]]))
    corrected = sentence.correction[:, HEADING]
    assert corrected.any()
    in_force = closed_loop.in_force(sentence.grid)[:, HEADING]
    reading = batch.readings[0].words[batch.sentences[0].first_row:]
    observed_held = closed_loop.in_force(reading)[sentence.observed_row, HEADING]
    steps = (in_force.astype(int) - observed_held.astype(int)) % words.n_heading
    assert set(steps.tolist()) <= {0, 1, words.n_heading - 1}  # at most one class from the observed word
    # inside a turn every row says a new observed word, which ends a correction (§4.9): the offset the turns leave is
    # taken out on the straight legs after them, and the final is flown within the tolerance
    assert np.abs(sentence.lateral_m).max() > words.spec.closed_loop_lateral_m
    assert np.abs(sentence.lateral_m[-len(sentence.lateral_m) // 4:]).max() < words.spec.closed_loop_lateral_m


def _flown(sentence):
    """A closed-loop sentence's flown 2 s rows, from its first predicted step (D51: `Stored.states`)."""
    return sentence.states[int(np.flatnonzero(sentence.on_interval)[sentence.start]):]


def _replayed(batch, inputs, sentence, params, words, monkeypatch):
    """``sentence`` flown again as the replay flies it (`replay_batch`, `replay.fly_sentences` on the time clock)."""
    stored = {7: closed_loop.Stored(grid=sentence.grid, correction=sentence.correction,
                                    first_row=batch.sentences[0].first_row, states=_flown(sentence),
                                    lateral_m=sentence.lateral_m, vertical_m=sentence.vertical_m,
                                    uncorrectable=sentence.uncorrectable, observed_row=sentence.observed_row,
                                    matched_row=sentence.matched_row)}
    batch.indices[0] = 7
    moved, missing = closed_loop.replay_batch(batch, stored, words)
    monkeypatch.setattr(replay.Batch, "inputs", lambda self, device: inputs)
    return stored, moved, missing, replay.fly_sentences(moved, params, words, device=CPU)


@pytest.mark.parametrize("interval_s", [2.0, 4.0, 8.0])
def test_a_closed_loop_sentence_flown_again_gives_its_states_on_every_2_s_row(monkeypatch, interval_s):
    """D51: the states are on the data's 2 s rows from the sentence's first row to its last said row, its Δ rows marked;
    the replay flies a closed-loop sentence on the time clock (the runner sets it, whatever the spec's) through every 2 s
    row of it, between the Δ rows too."""
    batch, inputs, words = _batch(interval_s)
    params = _params()
    (sentence,) = closed_loop.read(batch, inputs, params, words, device=CPU)
    every = int(round(interval_s / 2.0))
    assert len(sentence.states) == (sentence.start + len(sentence.grid) - 1) * every + 1
    assert np.array_equal(np.flatnonzero(sentence.on_interval), np.arange(sentence.start + len(sentence.grid)) * every)
    _, _, _, flown = _replayed(batch, inputs, sentence, params, words, monkeypatch)
    track = flown_track(flown.states[0].numpy(), batch.geometries[0])
    at = np.arange(len(_flown(sentence))) * 2                    # the cycle starting each 2 s row
    again = np.column_stack([track["e"][at], track["n"][at], track["height"][at]])
    assert np.abs(again - _flown(sentence)[:, :3]).max() < 1e-9


@pytest.mark.parametrize("interval_s", [4.0, 8.0])
def test_the_marked_rows_are_the_said_rows_and_the_rows_between_are_flown(interval_s):
    """D51: at Δ = 4 and 8 s the marked 2 s rows hold the states the reading compared (the said rows' e_y is read from
    them), and the rows between hold the executor's states there: in the turns, off the chord of their two Δ rows."""
    batch, inputs, words = _batch(interval_s)
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    every, start = int(round(interval_s / 2.0)), sentence.start
    signals = batch.signals[0]
    span = len(batch.readings[0].words) - batch.sentences[0].first_row
    smoothed = smooth(truncated(signals, span), words.spec)
    path = ObservedPath(signals.e_m[:span], signals.n_m[:span], smoothed.altitude_m, start * every)
    marked = sentence.states[sentence.on_interval][start:]
    assert np.allclose([path.match(*row[:3]).lateral_m for row in marked], sentence.lateral_m, atol=1e-9)
    flown = _flown(sentence)
    assert len(flown) == (len(sentence.grid) - 1) * every + 1
    middle = flown[every // 2::every][: len(sentence.grid) - 1, :2]          # the 2 s row halfway between two Δ rows
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


def test_the_closed_loop_file_round_trips_and_a_replay_flies_its_stored_states(tmp_path, monkeypatch):
    """§4.9 "Artefact": the words, corrections, states and errors written and read back by signal index; the replay's
    batch flies each from its first predicted step (its sentence and observed flight moved there) and the replay's check
    finds the stored states again."""
    from ts_transformer.experiments.executor_replay import closed_loop_columns
    from ts_transformer.instructions.artefact import load_closed_loop, write_closed_loop

    batch, inputs, words = _batch(4.0)
    params = _params()
    (sentence,) = closed_loop.read(batch, inputs, params, words, device=CPU)
    path = tmp_path / "train_4s.npz"
    write_closed_loop(path, words.spec, executor_params_sha256="x", row_interval_s=4.0, start_row=sentence.start,
                      signal_index=[7], first_row=[batch.sentences[0].first_row], grids=[sentence.grid],
                      corrections=[sentence.correction], states=[sentence.states],
                      on_interval=[sentence.on_interval], lateral_m=[sentence.lateral_m],
                      vertical_m=[sentence.vertical_m], uncorrectable=[sentence.uncorrectable],
                      observed_row=[sentence.observed_row], matched_row=[sentence.matched_row],
                      timed_out=[sentence.timed_out])
    with pytest.raises(ValueError, match="nothing to write"):
        write_closed_loop(tmp_path / "empty.npz", words.spec, executor_params_sha256="x", row_interval_s=4.0,
                          start_row=sentence.start, signal_index=[], first_row=[], grids=[], corrections=[], states=[],
                          on_interval=[], lateral_m=[], vertical_m=[], uncorrectable=[], observed_row=[], matched_row=[],
                          timed_out=[])
    with pytest.raises(FileExistsError):
        write_closed_loop(path, words.spec, executor_params_sha256="x", row_interval_s=4.0, start_row=sentence.start,
                          signal_index=[7], first_row=[batch.sentences[0].first_row], grids=[sentence.grid],
                          corrections=[sentence.correction], states=[sentence.states],
                          on_interval=[sentence.on_interval], lateral_m=[sentence.lateral_m],
                          vertical_m=[sentence.vertical_m], uncorrectable=[sentence.uncorrectable],
                      observed_row=[sentence.observed_row], matched_row=[sentence.matched_row],
                      timed_out=[sentence.timed_out])
    stored = closed_loop.stored_sentences(load_closed_loop(path, words.spec))
    assert list(stored) == [7] and np.array_equal(stored[7].grid, sentence.grid)
    assert np.array_equal(stored[7].states, _flown(sentence))
    with pytest.raises(ValueError, match="Δ rows marked"):
        write_closed_loop(tmp_path / "unmarked.npz", words.spec, executor_params_sha256="x", row_interval_s=4.0,
                          start_row=sentence.start, signal_index=[7], first_row=[batch.sentences[0].first_row],
                          grids=[sentence.grid], corrections=[sentence.correction], states=[sentence.states],
                          on_interval=[np.roll(sentence.on_interval, 1)], lateral_m=[sentence.lateral_m],
                          vertical_m=[sentence.vertical_m], uncorrectable=[sentence.uncorrectable],
                          observed_row=[sentence.observed_row], matched_row=[sentence.matched_row],
                          timed_out=[sentence.timed_out])
    assert np.array_equal(stored[7].uncorrectable, sentence.uncorrectable)
    _, moved, missing, flown = _replayed(batch, inputs, sentence, params, words, monkeypatch)
    assert missing == 0 and moved.sentences[0].first_row == batch.sentences[0].first_row + sentence.start * 2
    assert moved.signals[0].e_m[0] == batch.signals[0].e_m[sentence.start * 2]
    (row,) = closed_loop_columns(stored, 2.0)(moved, flown, [None])
    assert row["largest_lateral_m"] == pytest.approx(float(np.abs(sentence.lateral_m).max()))
    assert row["uncorrected_lateral_m"] == closed_loop.uncorrected_m(sentence.lateral_m, sentence.uncorrectable[:, 0])
    assert row["uncorrected_vertical_m"] == closed_loop.uncorrected_m(sentence.vertical_m, sentence.uncorrectable[:, 1])
    assert row["uncorrected_lateral_m"] != row["uncorrected_vertical_m"]
    assert sum(row["correction_words"].values()) == int(sentence.correction.sum())
    stored[7] = closed_loop.Stored(grid=sentence.grid, correction=sentence.correction, first_row=0,
                                   states=_flown(sentence) + [1.0, 0, 0, 0, 0, 0],
                                   lateral_m=sentence.lateral_m, vertical_m=sentence.vertical_m,
                                   uncorrectable=sentence.uncorrectable, observed_row=sentence.observed_row,
                                   matched_row=sentence.matched_row)
    with pytest.raises(ValueError, match="from its closed-loop states"):
        closed_loop_columns(stored, 2.0)(moved, flown, [None])


def test_the_runner_counts_the_flights_without_a_sentence_by_reason():
    """§4.9: a flight the replay does not fly (no dynamics, …), one the row interval refuses and one the closed loop
    refuses give no training sentence, each counted by its reason."""
    from ts_transformer.experiments.instruction_closed_loop import summarise

    batch, inputs, words = _batch()
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    lateness = closed_loop.heading_lateness_rows(sentence, batch.readings[0].words[batch.sentences[0].first_row:]) * 2.0
    numbers = summarise([sentence, Refused("go-around at the first predicted step")],
                        {"no aircraft dynamics": 3}, {"too short": 1}, [lateness], [_outside(batch, sentence, words)])
    assert numbers["sentences"] == 1
    assert numbers["without_a_sentence"] == {
        "not flown": {"no aircraft dynamics": 3}, "refused on the row interval": {"too short": 1},
        "refused by the closed loop": {"go-around at the first predicted step": 1}}
    assert numbers["correction_words"]["heading"] == int(sentence.correction[:, HEADING].sum()) > 0
    assert numbers["heading_word_lateness_s"]["n"] == len(lateness) > 0
    assert numbers["heading_word_lateness_s"]["mean"] == pytest.approx(float(lateness.mean()))
    lateral = numbers["outside_the_tolerance"]["lateral"]
    assert lateral["outside"] > 0 and lateral["without_a_correction_toward_the_path"] == 0        # D50: the rule holds
    assert lateral["correctable_rows"] == int((~sentence.uncorrectable[:, 0]).sum())


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
    assert np.array_equal(outside, ~sentence.uncorrectable[:, 0] & (np.abs(sentence.lateral_m)
                                                                    > words.spec.closed_loop_lateral_m))
    observed = batch.readings[0].words[batch.sentences[0].first_row:]
    for heading, name in ((in_force(observed)[sentence.observed_row, HEADING], "the observed words"),
                          ((2 * in_force(observed)[sentence.observed_row, HEADING].astype(int)
                            - in_force(sentence.grid)[:, HEADING].astype(int)) % words.n_heading, "away")):
        grid = sentence.grid.copy()
        grid[:, HEADING] = heading
        assert np.array_equal(_outside(batch, replace(sentence, grid=grid), words)["lateral"][2], outside), name


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
    sentence = ClosedLoopSentence(
        grid=grid, correction=np.zeros((n, 5), bool), states=np.zeros((n, 6)), on_interval=np.ones(n, bool),
        lateral_m=np.array([0.0, 0.0, 0.0, 0.0, -60.0]), vertical_m=np.zeros(n),
        uncorrectable=np.array([[True, True], *[[False, True]] * (n - 1)]),
        observed_row=np.array([0, 1, 2, 3, 3]), matched_row=np.zeros(n), start=0, timed_out=False)
    _, outside, breaks = closed_loop.outside_rows(sentence, observed, 0, words, [90.0, 92.0])["lateral"]
    assert outside.tolist() == [False, False, False, False, True] and breaks.tolist() == outside.tolist()
    corrected = grid.copy()
    corrected[4, HEADING] = 2                                       # one class right of 97°: toward the path
    _, _, breaks = closed_loop.outside_rows(replace(sentence, grid=corrected), observed, 0, words, [90.0, 92.0])["lateral"]
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
    sentence = ClosedLoopSentence(
        grid=grid, correction=np.array([r[1] for r in rows]), states=np.zeros((n, 6)), on_interval=np.ones(n, bool),
        lateral_m=np.array([0.0, 60.0, -60.0, -60.0, 0.0]), vertical_m=np.zeros(n),
        uncorrectable=np.array([[True, True], *[[False, True]] * (n - 1)]), observed_row=np.zeros(n, dtype=int),
        matched_row=np.zeros(n), start=0, timed_out=False)
    observed = _grid([_first(words, 900.0, ANGLE_LEVEL), *[{}] * 5])
    _, outside, breaks = closed_loop.outside_rows(sentence, observed, 0, words, [90.0])["lateral"]
    assert outside.tolist() == [False, True, True, True, False] and not breaks.any()
    assert [int(r[0][HEADING]) for r in rows[1:3]] == [words.n_heading - 1, 1] and rows[2][1][HEADING]
    cancelled = grid.copy()
    cancelled[2, HEADING] = 0                                    # the observed word: the correction only cancelled
    cancelled[3, HEADING] = 1
    _, _, breaks = closed_loop.outside_rows(replace(sentence, grid=cancelled), observed, 0, words, [90.0])["lateral"]
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
    closed_loop.write_reference(tmp_path, params, words, [2.0], git={"dirty": False}, target=target)
    monkeypatch.setattr(closed_loop, "_reference_results",
                        lambda instructions, p, w, intervals, device: {2.0: results(sentence)})
    return closed_loop.check(tmp_path, params, words, git={"dirty": False})


def test_the_conformance_check_passes_the_same_reading_and_finds_every_change(tmp_path, monkeypatch):
    from dataclasses import replace

    assert _reference(tmp_path / "same", monkeypatch, lambda s: (["KXXX:test"], [s])).passed
    moved = _reference(tmp_path / "moved", monkeypatch, lambda s: (["KXXX:test"], [
        replace(s, states=s.states + [0.0, 0.0, 1e-3, 0.0, 0.0, 0.0])]))
    assert not moved.passed and moved.largest_state_difference_m == pytest.approx(1e-3)
    lost = _reference(tmp_path / "nan", monkeypatch, lambda s: (["KXXX:test"], [
        replace(s, vertical_m=np.where(np.arange(len(s.vertical_m)) == 3, np.nan, s.vertical_m))]))
    assert not lost.passed
    shorter = _reference(tmp_path / "shape", monkeypatch, lambda s: (["KXXX:test"], [
        replace(s, grid=s.grid[:-1], correction=s.correction[:-1], states=s.states[:-1], lateral_m=s.lateral_m[:-1],
                vertical_m=s.vertical_m[:-1])]))
    assert not shorter.passed
    other = _reference(tmp_path / "other", monkeypatch, lambda s: (["KXXX:other"], [s]))
    assert not other.passed and "other flights" in str(other.mismatches)
    flags = _reference(tmp_path / "flags", monkeypatch, lambda s: (["KXXX:test"], [
        replace(s, uncorrectable=~s.uncorrectable)]))
    assert not flags.passed and "the uncorrectable differ" in str(flags.mismatches)
    refused = _reference(tmp_path / "refused", monkeypatch, lambda s: (["KXXX:test"], [Refused("too short")]))
    assert not refused.passed


def test_the_replay_refuses_a_sentence_of_another_first_row(monkeypatch):
    batch, inputs, words = _batch()
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    stored = {0: closed_loop.Stored(grid=sentence.grid, correction=sentence.correction, first_row=3,
                                    states=_flown(sentence), lateral_m=sentence.lateral_m,
                                    vertical_m=sentence.vertical_m, uncorrectable=sentence.uncorrectable,
                                    observed_row=sentence.observed_row, matched_row=sentence.matched_row)}
    with pytest.raises(ValueError, match="starts at 2 s row 3"):
        closed_loop.replay_batch(batch, stored, words)


def test_read_chunked_puts_each_result_in_its_place(monkeypatch):
    """Refusals stay in place, and a sentence read in a chunk is the one read alone."""
    batch, inputs, words = _batch()
    params = _params()
    (alone,) = closed_loop.read(batch, inputs, params, words, device=CPU)
    short = replay.Sentence(grid=batch.sentences[0].grid[:9], instructions=[], first_row=0)
    batch = replay.Batch(indices=[0, 1, 2], signals=batch.signals * 3, series=[None] * 3, readings=batch.readings * 3,
                         sentences=[short, batch.sentences[0], batch.sentences[0]], row_interval_s=2.0,
                         geometries=batch.geometries * 3, vertical_paths=[()] * 3,
                         approach_ias_mps=batch.approach_ias_mps * 3, groups=batch.groups * 3, drawn={})
    monkeypatch.setattr(closed_loop, "start_inputs", lambda part, step_s, device: closed_loop._rows(
        closed_loop.FlightInputs(*(torch.cat([getattr(inputs, n)] * len(part.sentences)) for n in (
            "initial_state", "aero_params", "frame_params", "max_thrust_n"))), list(range(len(part.sentences)))))
    results = closed_loop.read_chunked(batch, params, words, chunk=1, device=CPU)
    assert isinstance(results[0], Refused)
    for result in results[1:]:
        assert np.array_equal(result.grid, alone.grid) and np.array_equal(result.states, alone.states)


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
    assert sentence.uncorrectable.shape == (len(sentence.grid), 2) and sentence.uncorrectable[0].all()


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
    past = np.isnan(sentence.vertical_m)
    assert past.sum() > 20 and past[int(np.argmax(past)):].all()          # from the end of the path to the threshold
    observed = closed_loop.in_force(batch.readings[0].words[batch.sentences[0].first_row:])[sentence.observed_row]
    held = closed_loop.in_force(sentence.grid)
    assert np.array_equal(held[past][:, [HEADING, ANGLE]], observed[past][:, [HEADING, ANGLE]])  # the observed words
    assert sentence.correction[past].sum() <= 2 and sentence.uncorrectable[past].all()  # at most a correction ended
    assert np.isfinite(sentence.lateral_m).all()
    assert closed_loop.largest_m(sentence.vertical_m) == float(np.abs(sentence.vertical_m[~past]).max())


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
        sentence = ClosedLoopSentence(
            grid=np.array([r[0] for r in said_rows]), correction=np.array([r[1] for r in said_rows]),
            states=np.zeros((n, 6)), on_interval=np.ones(n, bool), lateral_m=np.zeros(n), vertical_m=np.zeros(n), uncorrectable=np.ones((n, 2), bool),
            observed_row=np.array([r[2] for r in said_rows]), matched_row=np.array([r[3] for r in said_rows]), start=0,
            timed_out=False)
        assert np.allclose(closed_loop.heading_lateness_rows(sentence, grid[first_row:]) * 2.0, lateness)


def test_a_flown_aircraft_behind_hears_the_turn_where_the_observed_one_did_and_flies_past_the_observed_time():
    """D42: told 70 m/s from the downwind (the observed aircraft flies 100 m/s there), the flown aircraft falls behind.
    It hears the turn onto the base where the observed aircraft heard it, later than the observed time, flies the final
    within Y of the path, and is not cut at the end of the open-loop sentence (§4.9 item 6): it lands later."""
    batch, inputs, words = _batch(speeds={0: 70.0})
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    assert isinstance(sentence, ClosedLoopSentence)
    open_grid, start = batch.sentences[0].grid, sentence.start
    turn = start + int(np.argmax(open_grid[start + 1:, HEADING] != UNCHANGED)) + 1      # the first turn word
    heard = int(np.argmax(sentence.observed_row >= turn))
    assert sentence.observed_row[heard] == turn and int(sentence.grid[heard, HEADING]) == int(open_grid[turn, HEADING])
    assert not sentence.correction[heard, HEADING]
    assert start + heard > turn + 5                                     # behind: more than 10 s after the observed time
    observed = batch.signals[0]
    flown = sentence.states[start + heard, :2]
    assert math.dist(flown, (observed.e_m[turn], observed.n_m[turn])) < 250.0         # within one row's flight
    assert len(sentence.grid) > len(open_grid) - start and not sentence.timed_out
    (short,) = closed_loop.read(batch, inputs, replace(_params(), timeout_factor=1.0), words, device=CPU)
    assert short.timed_out and len(short.grid) == len(open_grid) - start         # the limit: the observed time
    assert np.abs(sentence.lateral_m[-len(sentence.lateral_m) // 4:]).max() < words.spec.closed_loop_lateral_m
    assert -2.0 * 75.0 < sentence.states[-1, 0] <= 0.0               # its last row starts one row before the threshold


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
    start = sentence.start
    (heard,) = np.flatnonzero(sentence.grid[:, RUNWAY] == RUNWAY_GO_AROUND)
    assert sentence.observed_row[heard] >= go > sentence.observed_row[heard - 1]     # the first row that reaches it
    assert start + heard < go - 5                                     # ahead: more than 10 s before the observed time
    observed = batch.signals[0]
    flown = sentence.states[start + heard, :2]
    assert flown[0] < 0.0 and math.dist(flown, (observed.e_m[go], observed.n_m[go])) < 250.0   # before the threshold
    assert sentence.states[go, 0] > 0.0          # at the observed time it is past the threshold (e = 0) already
    _, moved, _, replayed = _replayed(batch, inputs, sentence, params, words, monkeypatch)
    assert moved.sentences[0].go_arounds == 1 and replayed.modes["go_around"][0].any()


def test_the_replay_reads_the_speed_words_and_how_far_along_the_path_the_flown_aircraft_is(monkeypatch):
    """Vocabulary §9.8, A11: each flown sentence's speed words other than "unspecified", and the largest distance along the observed
    path from the observed aircraft of the same time before "unspecified" — told 70 m/s on the 100 m/s downwind (the
    open loop, the time clock), the flown aircraft falls far behind."""
    from ts_transformer.experiments.executor_replay import along_columns
    from ts_transformer.tests.test_autopilot import vertical_paths

    params = _params()
    read = {}
    for name, speeds, interval_s in (("observed", None, 2.0), ("slow", {0: 70.0}, 2.0), ("slow_4", {0: 70.0}, 4.0)):
        batch, inputs, words = _batch(interval_s, speeds=speeds)
        batch.vertical_paths[0] = vertical_paths(batch.geometries[0])
        monkeypatch.setattr(replay.Batch, "inputs", lambda self, device, inputs=executor_inputs(
            batch.signals[0], batch.geometries[0]): inputs)
        flown = replay.fly_sentences(batch, params, words, device=CPU)
        (read[name],) = along_columns(words)(batch, flown, replay.judge_batch(batch, flown, words))
        grid = batch.sentences[0].grid
        held = closed_loop.in_force(grid)[:, SPEED] == words.speed_unspecified
        before = int(np.argmax(held)) if held.any() else len(grid)
        assert read[name]["speed_words"] == int((grid[:before, SPEED] != UNCHANGED).sum())     # before "unspecified"
    assert read["slow"]["speed_words"] == read["slow_4"]["speed_words"] == 1
    assert read["slow"]["largest_along_m"] > 1000.0 > read["observed"]["largest_along_m"]
    assert read["slow_4"]["largest_along_m"] == pytest.approx(read["slow"]["largest_along_m"], rel=0.1)
