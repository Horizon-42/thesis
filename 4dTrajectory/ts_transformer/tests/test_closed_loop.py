"""The closed-loop reading (design §4.9, D32): the comparison, the corrections, and a flight read and flown again."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.closed_loop import ClosedLoopSentence, Corrector, ObservedPath
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.speed import approach_speed_ias_mps
from ts_transformer.instructions.labeller.read import read_flight
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
                          timeout_factor=1.5, word_clock="time", decision_cone_share=1.0,
                          decision_glidepath_tolerance_m=60.0)


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
    lateral, vertical = path.match(500.0, 10.0, 990.0)        # over the crossing, on the first leg: 10 m left of east
    assert lateral == pytest.approx(-10.0) and vertical == pytest.approx(990.0 - (1000.0 - 10.0))
    path.match(1010.0, 250.0, 0.0)                              # the north leg, 10 m right of it
    path.match(750.0, 490.0, 0.0)                               # the west leg
    lateral, _ = path.match(505.0, 5.0, 0.0)                    # over the crossing again: now on the south leg
    assert lateral == pytest.approx(-5.0)                       # 5 m east of southbound is left of it
    assert path.segment > len(e) - 25


def test_a_flown_position_behind_the_matched_point_keeps_it():
    e, n, height = _loop_path()
    path = ObservedPath(e, n, height, 10)
    lateral, _ = path.match(100.0, -30.0, 0.0)                  # behind segment 10 (e = 500), 30 m right of east
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
    corrector = Corrector(_grid([_first(words), *[{}] * 9]), 0, words, 1)
    corrector.row(0, 0.0, 0.0, 900.0)

    def heading(k, lateral):
        said, added = corrector.row(k, lateral, 0.0, 900.0)
        return int(said[HEADING]), bool(added[HEADING])

    assert heading(1, 25.0) == (UNCHANGED, False)                # inside Y = 30 m
    assert heading(2, 40.0) == (left, True)                      # right of the path: one class to the left
    assert heading(3, 20.0) == (UNCHANGED, False)                # still over Y / 2: in force
    assert heading(4, 10.0) == (0, True)                         # under Y / 2: the observed word again
    assert heading(5, -45.0) == (right, True)
    assert heading(6, 2.0) == (0, True)                          # the sign changed
    assert heading(7, 35.0) == (left, True)


def test_a_new_observed_heading_word_ends_a_correction():
    words = Words(spec())
    corrector = Corrector(_grid([_first(words), {}, {HEADING: 2}, {}]), 0, words, 1)
    corrector.row(0, 0.0, 0.0, 900.0)
    assert int(corrector.row(1, 50.0, 0.0, 900.0)[0][HEADING]) == words.n_heading - 1
    said, added = corrector.row(2, 50.0, 0.0, 900.0)            # the observed word, not a correction
    assert int(said[HEADING]) == 2 and not added[HEADING]
    said, added = corrector.row(3, 50.0, 0.0, 900.0)            # a correction from the new word
    assert int(said[HEADING]) == 1 and added[HEADING]


def test_an_angle_correction_on_a_final_descent_and_none_on_a_level_a_climb_or_beyond_the_classes():
    words = Words(spec())
    final = {ALTITUDE: words.altitude_no_level_off, ANGLE: 3}
    corrector = Corrector(_grid([_first(words), final, *[{}] * 8]), 0, words, 1)
    corrector.row(0, 0.0, 0.0, 900.0)
    corrector.row(1, 0.0, 0.0, 900.0)

    def angle(k, vertical):
        said, added = corrector.row(k, 0.0, vertical, 800.0)
        return int(said[ANGLE]), bool(added[ANGLE])

    assert angle(2, 20.0) == (4, True)                           # too high: the next steeper class
    assert angle(3, 10.0) == (UNCHANGED, False)
    assert angle(4, 5.0) == (3, True)                            # under H / 2: the observed class again
    assert angle(5, -20.0) == (2, True)                          # too low: the next shallower
    assert angle(6, 3.0) == (3, True)
    level = Corrector(_grid([_first(words), *[{}] * 3]), 0, words, 1)
    level.row(0, 0.0, 0.0, 900.0)
    assert int(level.row(1, 0.0, 60.0, 960.0)[0][ANGLE]) == UNCHANGED            # a level hold: none
    climb = Corrector(_grid([_first(words, 1200.0, words.angle_climb), *[{}] * 3]), 0, words, 1)
    climb.row(0, 0.0, 0.0, 900.0)
    assert int(climb.row(1, 0.0, -60.0, 900.0)[0][ANGLE]) == UNCHANGED           # a climb: none
    steepest = Corrector(_grid([_first(words, 600.0, words.n_descent), *[{}] * 3]), 0, words, 1)
    steepest.row(0, 0.0, 0.0, 900.0)
    assert int(steepest.row(1, 0.0, 60.0, 900.0)[0][ANGLE]) == UNCHANGED        # no class steeper than descent 4
    shallowest = Corrector(_grid([_first(words, 600.0, 1), *[{}] * 3]), 0, words, 1)
    shallowest.row(0, 0.0, 0.0, 900.0)
    assert int(shallowest.row(1, 0.0, -60.0, 900.0)[0][ANGLE]) == UNCHANGED     # none shallower than descent 1


def test_a_new_observed_altitude_word_ends_an_angle_correction():
    words = Words(spec())
    corrector = Corrector(_grid([_first(words, 600.0, 2), {}, {ALTITUDE: words.altitude_index(480.0)}]), 0, words, 1)
    corrector.row(0, 0.0, 0.0, 900.0)
    assert int(corrector.row(1, 0.0, 30.0, 800.0)[0][ANGLE]) == 3
    said, _ = corrector.row(2, 0.0, 30.0, 800.0)
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


def _batch(interval_s=2.0):
    one, geometry = spec(), instruction_airport()
    words = Words(one)
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    reading = read_flight(signals, geometry, one, words)
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
    rows = np.arange(start) * every
    assert np.array_equal(sentence.states[:start, :3], np.column_stack([observed.e_m[rows], observed.n_m[rows],
                                                                       observed.altitude_m[rows]]))
    corrected = sentence.correction[:, HEADING]
    assert corrected.any()
    held = np.maximum.accumulate(np.where(sentence.grid[:, HEADING] != UNCHANGED, np.arange(len(sentence.grid)), 0))
    in_force = sentence.grid[held, HEADING]
    open_held = closed_loop.in_force(batch.sentences[0].grid)[start:start + len(sentence.grid), HEADING]
    steps = (in_force.astype(int) - open_held.astype(int)) % words.n_heading
    assert set(steps.tolist()) <= {0, 1, words.n_heading - 1}  # at most one class from the observed word
    # inside a turn every row says a new observed word, which ends a correction (§4.9): the offset the turns leave is
    # taken out on the straight legs after them, and the final is flown within the tolerance
    assert np.abs(sentence.lateral_m).max() > 200.0
    assert np.abs(sentence.lateral_m[-len(sentence.lateral_m) // 4:]).max() < words.spec.closed_loop_lateral_m


def _replayed(batch, inputs, sentence, params, words, monkeypatch):
    """``sentence`` flown again as the replay flies it (`replay_batch`, `replay.fly_sentences` on the time clock)."""
    stored = {7: closed_loop.Stored(grid=sentence.grid, correction=sentence.correction,
                                    first_row=batch.sentences[0].first_row, states=sentence.states[sentence.start:],
                                    lateral_m=sentence.lateral_m, vertical_m=sentence.vertical_m)}
    batch.indices[0] = 7
    moved, missing = closed_loop.replay_batch(batch, stored, words)
    monkeypatch.setattr(replay.Batch, "inputs", lambda self, device: inputs)
    return stored, moved, missing, replay.fly_sentences(moved, params, words, device=CPU)


def test_a_closed_loop_sentence_flown_again_gives_its_states(monkeypatch):
    """The replay flies a closed-loop sentence on the time clock (the runner sets it, whatever the spec's)."""
    batch, inputs, words = _batch()
    params = _params()
    (sentence,) = closed_loop.read(batch, inputs, params, words, device=CPU)
    rows = len(sentence.grid)
    _, _, _, flown = _replayed(batch, inputs, sentence, params, words, monkeypatch)
    track = flown_track(flown.states[0].numpy(), batch.geometries[0])
    at = np.arange(rows) * 2                                     # the cycle starting each row
    again = np.column_stack([track["e"][at], track["n"][at], track["height"][at]])
    assert np.abs(again - sentence.states[sentence.start:, :3]).max() < 1e-9


def test_a_go_around_at_the_first_predicted_step_and_a_short_sentence_are_refused():
    batch, inputs, words = _batch()
    grid = batch.sentences[0].grid.copy()
    grid[3, RUNWAY] = RUNWAY_GO_AROUND
    gone = replay.Sentence(grid=grid, instructions=[], first_row=0)
    short = replay.Sentence(grid=grid[:9], instructions=[], first_row=0)
    for sentence, reason in ((gone, "go-around at the first predicted step"), (short, "too short")):
        batch.sentences[0] = sentence
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
                      corrections=[sentence.correction], states=[sentence.states], lateral_m=[sentence.lateral_m],
                      vertical_m=[sentence.vertical_m], ended=[sentence.ended])
    with pytest.raises(ValueError, match="nothing to write"):
        write_closed_loop(tmp_path / "empty.npz", words.spec, executor_params_sha256="x", row_interval_s=4.0,
                          start_row=sentence.start, signal_index=[], first_row=[], grids=[], corrections=[], states=[],
                          lateral_m=[], vertical_m=[], ended=[])
    with pytest.raises(FileExistsError):
        write_closed_loop(path, words.spec, executor_params_sha256="x", row_interval_s=4.0, start_row=sentence.start,
                          signal_index=[7], first_row=[batch.sentences[0].first_row], grids=[sentence.grid],
                          corrections=[sentence.correction], states=[sentence.states], lateral_m=[sentence.lateral_m],
                          vertical_m=[sentence.vertical_m], ended=[sentence.ended])
    stored = closed_loop.stored_sentences(load_closed_loop(path, words.spec))
    assert list(stored) == [7] and np.array_equal(stored[7].grid, sentence.grid)
    assert np.array_equal(stored[7].states, sentence.states[sentence.start:])
    _, moved, missing, flown = _replayed(batch, inputs, sentence, params, words, monkeypatch)
    assert missing == 0 and moved.sentences[0].first_row == batch.sentences[0].first_row + sentence.start * 2
    assert moved.signals[0].e_m[0] == batch.signals[0].e_m[sentence.start * 2]
    (row,) = closed_loop_columns(stored)(moved, flown, [None])
    assert row["largest_lateral_m"] == pytest.approx(float(np.abs(sentence.lateral_m).max()))
    assert sum(row["correction_words"].values()) == int(sentence.correction.sum())
    stored[7] = closed_loop.Stored(grid=sentence.grid, correction=sentence.correction, first_row=0,
                                   states=sentence.states[sentence.start:] + [1.0, 0, 0, 0, 0, 0],
                                   lateral_m=sentence.lateral_m, vertical_m=sentence.vertical_m)
    with pytest.raises(ValueError, match="from its closed-loop states"):
        closed_loop_columns(stored)(moved, flown, [None])


def test_the_runner_counts_the_flights_without_a_sentence_by_reason():
    """§4.9: a flight the replay does not fly (no dynamics, …), one the row interval refuses and one the closed loop
    refuses give no training sentence, each counted by its reason."""
    from ts_transformer.experiments.instruction_closed_loop import summarise

    batch, inputs, words = _batch()
    (sentence,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    numbers = summarise([sentence, Refused("go-around at the first predicted step")],
                        {"no aircraft dynamics": 3}, {"heading word across a runway change": 1})
    assert numbers["sentences"] == 1
    assert numbers["without_a_sentence"] == {
        "not flown": {"no aircraft dynamics": 3}, "refused on the row interval": {"heading word across a runway change": 1},
        "refused by the closed loop": {"go-around at the first predicted step": 1}}
    assert numbers["correction_words"]["heading"] == int(sentence.correction[:, HEADING].sum()) > 0


def test_no_angle_correction_while_the_level_reached_by_a_descent_is_held():
    """§4.9 vertical item 1: a level reached by a descent says no angle word, the descent class stays in force; while the
    flown height is within the level's band the level is held — no correction starts, and one in force ends."""
    words = Words(spec())
    corrector = Corrector(_grid([_first(words, 1080.0, 2), *[{}] * 5]), 0, words, 1)
    corrector.row(0, 0.0, 0.0, 1300.0)
    assert int(corrector.row(1, 0.0, 30.0, 1250.0)[0][ANGLE]) == 3        # still descending to 1,080 m: corrected
    said, added = corrector.row(2, 0.0, -18.5, 1081.5)                   # level at 1,080 m: the correction ends
    assert int(said[ANGLE]) == 2 and added[ANGLE]
    assert int(corrector.row(3, 0.0, -18.5, 1080.0)[0][ANGLE]) == UNCHANGED   # and none starts


def test_a_repeated_observed_position_is_no_segment():
    e = np.array([0.0, 100.0, 100.0, 200.0, 300.0])
    path = ObservedPath(e, np.zeros(5), np.full(5, 500.0), 2)
    lateral, vertical = path.match(150.0, -10.0, 510.0)
    assert lateral == pytest.approx(10.0) and vertical == pytest.approx(10.0) and math.isfinite(lateral)
