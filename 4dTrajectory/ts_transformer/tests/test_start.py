"""The start of a closed loop (vocabulary §6 item 5, D67, §12.1 A26): `autopilot/start.py`."""

from __future__ import annotations

import math
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import closed_loop, replay, start as start_module
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import read_state
from ts_transformer.autopilot.judge import TIMEOUT, outcome_of
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.speed import approach_speed_ias_mps
from ts_transformer.autopilot.start import NO_MOVE, GoAroundBeyondMost, Loop, Move, RowRefused, start, start_moved
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.instructions.artefact import (
    CLOSED_LOOP_DIRECTORY, ClosedLoopSentence, closed_loop_path, closed_loop_sentences, load_candidates,
    load_signals, write_closed_loop,
)
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, HEADING, RUNWAY, RUNWAY_GO_AROUND, SPEED, UNCHANGED, Words,
)
from ts_transformer.tests.support import executor_inputs, labelled_instruction_artefact

CPU = torch.device("cpu")
A320_IAS = approach_speed_ias_mps("A320", 62000.0)


def _params():
    from ts_transformer.experiments.executor_spec import ROLL_RATE_DEG_S

    return ExecutorParams(cycle_s=1.0, bank_rate_deg_s=ROLL_RATE_DEG_S, path_time_constant_s=2.0, path_rate_factor=2.0,
                          timeout_factor=1.5, start_rule="trailing-fit-8s")


def _executor(directory, artefact, params, vocabulary_sha256, *, reference=True):
    """An executor spec at ``directory`` measured against ``vocabulary_sha256``; with ``reference``, its reference
    tracks written with it (the artefact's flight from row 0, `conformance.draw_reference_batch` standing in for the
    draw: no harvest), so opening it flies them again (D73)."""
    from ts_transformer.autopilot import conformance
    from ts_transformer.autopilot import spec as executor_spec

    executor_spec.write_spec(directory, params, vocabulary_sha256, {}, {"git": {"head": "x", "dirty": False}})
    if reference:
        conformance.write_reference(directory, artefact, batch=conformance.draw_reference_batch(artefact, None, None))
    return directory


def _artefact(tmp_path, monkeypatch, interval_s):
    """A tmp artefact of one labelled flight (no harvest behind it: its rebuild, aircraft and approach speed stand in) with
    its labeller reference, its closed-loop sentence read at ``interval_s``, written into the artefact and read back as
    the reader gives it, the batch it was read from and the executor spec that flew it, with its reference tracks."""
    directory = tmp_path / "artefact"
    spec = labelled_instruction_artefact(directory)
    words = Words(spec)
    (signals,) = load_signals(directory, "train")
    geometry = load_candidates(directory)[signals.airport]
    reading = read_flight(signals, geometry, spec, words)
    sentence = replay.sentence_on_interval(reading, signals, interval_s, geometry, words)
    every = int(round(interval_s / spec.step_s))
    anchor = sentence.first_row + start_module.start_row(interval_s) * every
    inputs = executor_inputs(signals, geometry, anchor)
    batch = replay.Batch(indices=[0], signals=[replay.from_row(signals, sentence.first_row)], observed=[signals],
                         series=[None],
                         readings=[reading], sentences=[sentence], row_interval_s=interval_s, geometries=[geometry],
                         approach_ias_mps=[A320_IAS], groups=[replay.OWN], drawn={})
    (read,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    assert isinstance(read, ClosedLoopSentence)
    (directory / CLOSED_LOOP_DIRECTORY).mkdir()
    path = closed_loop_path(directory, "train", interval_s)
    write_closed_loop(path, spec, executor_params_sha256=params_sha256(_params()), row_interval_s=interval_s,
                      start_row=start_module.start_row(interval_s), sentences={0: read})
    (stored,) = closed_loop_sentences(directory, "train", interval_s, spec).values()
    monkeypatch.setattr(start_module, "rebuild_series", lambda d, flights: [None] * len(flights))
    monkeypatch.setattr(replay, "group_of", lambda series: replay.OWN)
    monkeypatch.setattr(replay, "flight_approach_ias_mps", lambda series, group: A320_IAS)
    monkeypatch.setattr(start_module, "flight_inputs", lambda series, flights, anchors, airports, rule, device:
                        executor_inputs(signals, geometry, anchors[0], rule=rule))
    monkeypatch.setattr(replay.Batch, "inputs", lambda self, rule, device: inputs)
    from ts_transformer.autopilot import conformance
    from ts_transformer.instructions import conformance as labeller

    labeller.write_reference(directory, [signals], [labeller.labelled_record(reading)], [reading],
                             git={"head": "x", "dirty": False})
    flown = replay.sentence_on_interval(reading, signals, spec.step_s, geometry, words)
    reference = replay.Batch(indices=[0], signals=[signals], observed=[signals], series=[None], readings=[reading],
                             sentences=[flown],
                             row_interval_s=spec.step_s, geometries=[geometry], approach_ias_mps=[A320_IAS],
                             groups=[replay.OWN], drawn={"split": "train"})
    monkeypatch.setattr(conformance, "draw_reference_batch", lambda instructions, words, draw: reference)
    _executor(tmp_path / "executor", directory, _params(), spec.sha256)
    return directory, words, batch, stored, anchor


@pytest.mark.parametrize("interval_s", [2.0, 4.0, 8.0])
def test_a_stored_sentence_said_through_the_start_gives_its_states_and_its_outcome(tmp_path, monkeypatch, interval_s):
    """D67: the words of a closed-loop sentence, said row by row through the start, give back its stored states on the 2 s
    rows from the first predicted step and the outcome its replay is judged to; the time limit is the closed-loop
    reading's, and whether the flight timed out is the judge's outcome (D90)."""
    directory, words, batch, stored, anchor = _artefact(tmp_path, monkeypatch, interval_s)
    params = _params()
    loop, order = start(directory, "train", interval_s, {0: stored}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    assert order == [0]
    limit = (len(batch.readings[0].words) - anchor) * words.spec.step_s * params.timeout_factor   # §5.8, written out
    assert float(loop.executor.time_limit_s[0]) == limit
    flown = [loop.rows()[0]]
    for k, row in enumerate(stored.rows.grid):
        rows, done = loop.step(row[None, :])
        if k < len(stored.rows.grid) - 1:
            assert not done[0]
            flown += list(rows[0])
    assert done[0]                                     # done in the row the sentence ends with
    every = int(round(interval_s / words.spec.step_s))
    expected = stored.rows.states[stored.rows.start * every:]
    assert len(flown) == len(expected)
    assert np.abs(np.array(flown)[:, :3] - expected[:, :3]).max() <= STATE_BOUND_M     # the conformance tolerance
    assert np.array_equal(np.array(flown), expected)                                   # and here, the same code: exact
    moved, _ = closed_loop.replay_batch(batch, {0: stored}, words)
    replayed = outcome_of(replay.fly_sentences(moved, params, words, device=CPU), 0, batch.geometries[0], words.spec)
    assert loop.outcome(0).outcome == replayed.outcome
    assert loop.outcome(0).crossing == replayed.crossing
    assert stored.withheld.outcome == replayed.outcome           # D74: the reading's outcome, stored, is the replay's and the start's
    assert (loop.outcome(0).outcome == TIMEOUT) == stored.withheld.timed_out


def test_the_loops_executor_is_public_and_the_loop_has_no_timed_out(tmp_path, monkeypatch):
    """D90, §6 item 5: a caller reads from `Loop.executor` each flight's end cycle (``done_cycle``), the flown record the
    judge reads (``flown()``) and the aero parameters (``inputs.aero_params``); the loop has no ``timed_out()`` (why a
    flight ended is the judge's outcome)."""
    directory, words, batch, stored, anchor = _artefact(tmp_path, monkeypatch, 4.0)
    loop, _ = start(directory, "train", 4.0, {0: stored}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    flown = []
    for row in stored.rows.grid:
        rows, done = loop.step(row[None, :])
        flown += list(rows[0])
    assert done[0]
    end = int(loop.executor.done_cycle[0])
    assert loop.executor.count - 4.0 / _params().cycle_s <= end < loop.executor.count     # in the last Δ row flown
    record = loop.executor.flown()                     # the record holds the 2 s rows the loop gave
    for k, row in enumerate(flown):
        at = record.states[:, (k + 1) * loop.row_cycles]
        assert np.array_equal(start_module.state_rows(read_state(at, loop.executor.charts))[0], row)
    assert outcome_of(record, 0, batch.geometries[0], words.spec).outcome == stored.withheld.outcome
    expected = executor_inputs(batch.observed[0], batch.geometries[0], anchor, rule=_params().start_rule)
    assert torch.equal(loop.executor.inputs.aero_params, expected.aero_params)
    assert not hasattr(loop, "timed_out")


def test_a_go_around_beyond_the_most_given_is_refused_by_name(tmp_path, monkeypatch):
    """D67: each flight may say the most go-arounds the loop was started with; one more refuses the row by name before
    anything is flown, and the time limit grows by 900 s for each one heard."""
    directory, words, _, stored, _ = _artefact(tmp_path, monkeypatch, 2.0)
    climb = np.full((1, 5), UNCHANGED, dtype=np.int64)
    climb[0, [RUNWAY, ALTITUDE, ANGLE]] = RUNWAY_GO_AROUND, words.altitude_index(1500.0), words.angle_climb
    again = np.full((1, 5), UNCHANGED, dtype=np.int64)
    again[0, RUNWAY] = 0
    for most in (0, 1):
        loop, _ = start(directory, "train", 2.0, {0: stored}, tmp_path / "executor", most_go_arounds=most, device=CPU)
        loop.step(stored.rows.grid[:1])
        before = float(loop.executor.time_limit_s[0])
        if most == 0:
            with pytest.raises(GoAroundBeyondMost, match="beyond the most 0"):
                loop.step(climb)
            assert loop.steps == 1 and loop.executor.count == 2 and loop.go_arounds[0] == 0      # nothing flown
            continue
        loop.step(climb)
        assert float(loop.executor.time_limit_s[0]) == before + 900.0 and loop.go_arounds[0] == 1
        loop.step(again)
        with pytest.raises(GoAroundBeyondMost, match="beyond the most 1"):
            loop.step(climb)
        assert loop.go_arounds[0] == 1
    # a halted flight's words are not heard either
    loop, _ = start(directory, "train", 2.0, {0: stored}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    loop.step(stored.rows.grid[:1])
    loop.halt(np.array([True]))
    loop.step(climb)
    assert loop.go_arounds[0] == 0
    # a done flight's words are not heard: its go-around is no go-around
    loop, _ = start(directory, "train", 2.0, {0: stored}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    with pytest.raises(ValueError, match="not done"):
        loop.outcome(0)
    for row in stored.rows.grid:
        loop.step(row[None, :])
    ended = loop.outcome(0)
    loop.step(climb)
    assert loop.go_arounds[0] == 0 and loop.outcome(0) == ended


def test_a_row_the_grammar_refuses_is_refused_by_name_with_nothing_changed(tmp_path, monkeypatch):
    """D80: before anything changes, `Loop.step` checks each flying flight's row with the grammar at the height above E of
    the executor's state: "go-around" while G is true, a word outside its column, and "no level-off" while G is true are
    refused by name, and the loop is as it was — the words in force, the go-arounds counted, the time limit, the cycles."""
    directory, words, _, stored, _ = _artefact(tmp_path, monkeypatch, 2.0)
    climb = np.full((1, 5), UNCHANGED, dtype=np.int64)
    climb[0, [RUNWAY, ALTITUDE, ANGLE]] = RUNWAY_GO_AROUND, words.altitude_index(1500.0), words.angle_climb
    loop, _ = start(directory, "train", 2.0, {0: stored}, tmp_path / "executor", most_go_arounds=2, device=CPU)
    loop.step(stored.rows.grid[:1])
    loop.step(climb)                                                   # G true now
    outside = np.full((1, 5), UNCHANGED, dtype=np.int64)
    outside[0, HEADING] = words.n_heading                              # no heading word
    no_level_off = np.full((1, 5), UNCHANGED, dtype=np.int64)
    no_level_off[0, ALTITUDE] = words.altitude_no_level_off
    for row, reason in ((climb, "runway word not permitted"), (outside, "word outside its column"),
                        (no_level_off, "no level-off")):
        before = (loop.steps, loop.executor.count, loop.go_arounds.copy(), list(loop.grammar),
                  float(loop.executor.time_limit_s[0]), len(loop.spoken.grid), loop.spoken.value.clone())
        with pytest.raises(RowRefused, match="flight 0 at row 2") as refused:
            loop.step(row)
        assert refused.value.reason.startswith(reason), refused.value.reason
        after = (loop.steps, loop.executor.count, loop.go_arounds.copy(), list(loop.grammar),
                 float(loop.executor.time_limit_s[0]), len(loop.spoken.grid), loop.spoken.value.clone())
        assert after[:2] == before[:2] and np.array_equal(after[2], before[2]) and after[3:6] == before[3:6]
        assert torch.equal(after[6], before[6])                         # the words said: unchanged too
    # a halted flight's words are not read by the grammar ("go-around" while G is true passes) ...
    loop.halt(np.array([True]))
    loop.step(climb)
    # ... but their values are (D80): a word outside its column, or a runway that is no candidate, refused by name
    runway = np.full((1, 5), UNCHANGED, dtype=np.int64)
    runway[0, RUNWAY] = len(loop.geometries[0].candidates)
    for row in (outside, runway):
        before = (loop.steps, loop.executor.count, len(loop.spoken.grid))
        with pytest.raises(RowRefused, match="flight 0 at row 3") as refused:
            loop.step(row)
        assert refused.value.reason == "word outside its column"
        assert (loop.steps, loop.executor.count, len(loop.spoken.grid)) == before


# ---- the interface for the post-training (D97 (2), (3)): a copy of chosen flights, a flight alone and in a batch. The
# bound (the user, 2026-10-05, on A38's measurement: a flight alone against in chunks of 2048 differed by <= 7.9e-10 m,
# chunks of 1000 against 2048 not at all): words, done and outcome exact, states within STATE_BOUND_M.
#: The Δ rows of the schedules below: B's go-around held from row 3 to the runway again at row 8, C's two turns (150°
#: right, then 150° more: past 180°, said word by word), D halted at row 9.
GO_AROUND_ROW, AGAIN_ROW, TURN_ROWS, HALT_ROW = 3, 8, (2, 4), 9


def _schedules(tmp_path, monkeypatch, interval_s=2.0):
    """One stored flight said four ways from its first predicted step — A its stored sentence, B the same with a
    go-around (speed unspecified) held for five rows, C the same with two heading turns of 150° each, D as A and halted
    at `HALT_ROW` — and a loop of any of them: ``(loop_of(ways), row_of(way, k))``."""
    directory, words, batch, stored, _ = _artefact(tmp_path, monkeypatch, interval_s)
    one, _ = start(directory, "train", interval_s, {0: stored}, tmp_path / "executor", most_go_arounds=1, device=CPU)
    inputs, limit, geometry = one.executor.inputs, float(one.executor.time_limit_s[0]), batch.geometries[0]
    grid = stored.rows.grid.astype(np.int64)
    unchanged = np.full(5, UNCHANGED, dtype=np.int64)
    climb, again = unchanged.copy(), unchanged.copy()
    climb[[RUNWAY, ALTITUDE, ANGLE, SPEED]] = (RUNWAY_GO_AROUND, words.altitude_index(1500.0), words.angle_climb,
                                               words.speed_unspecified)
    again[RUNWAY] = 0
    step = words.spec.heading_step_deg
    turned = {row: (int(grid[0, HEADING]) + int(round(150.0 * (n + 1) / step))) % words.n_heading
              for n, row in enumerate(TURN_ROWS)}

    def row_of(way: str, k: int) -> np.ndarray:
        if way == "B" and GO_AROUND_ROW <= k < AGAIN_ROW:
            return climb.copy() if k == GO_AROUND_ROW else unchanged.copy()
        if way == "B" and k == AGAIN_ROW:
            return again.copy()
        row = grid[k].copy() if k < len(grid) else unchanged.copy()
        if way == "C" and k in turned:
            row[HEADING] = turned[k]
        return row

    def loop_of(ways: str) -> Loop:
        many = FlightInputs(**{f.name: getattr(inputs, f.name).expand(len(ways), *getattr(inputs, f.name).shape[1:])
                               .clone() for f in fields(FlightInputs)})
        return Loop(many, [geometry] * len(ways), [A320_IAS] * len(ways), [limit] * len(ways), _params(), words,
                    interval_s=interval_s, most_go_arounds=1, device=CPU)

    return loop_of, row_of


def _fly(loop: Loop, ways: str, row_of, k: int, *, until_done: bool = False, steps: int = 0
         ) -> tuple[list[np.ndarray], int]:
    """``loop`` flown from its step ``k`` on, each flight's words its way's (way D halted at `HALT_ROW`): ``steps``
    steps, or until every flight is done or halted; the 2 s rows of each step and the next step."""
    flown = []
    executor = loop.executor
    while (until_done and not bool((executor.done | executor.halted).all())) or len(flown) < steps:
        assert k < 5000, "never done"
        if k == HALT_ROW:
            loop.halt(np.array([way == "D" for way in ways]))
        rows, _ = loop.step(np.stack([row_of(way, k) for way in ways]))
        flown.append(rows)
        k += 1
    return flown, k


def _same_flight(a_rows, a_loop, a, b_rows, b_loop, b):
    """Flight ``a`` of one loop and ``b`` of another flew alike: states within STATE_BOUND_M; done, halted, end cycle,
    time limit, go-arounds and outcome exact."""
    for x, y in zip(a_rows, b_rows):
        assert np.abs(x[a] - y[b]).max() <= STATE_BOUND_M
    one, other = a_loop.executor, b_loop.executor
    for name in ("done", "halted", "done_cycle", "time_limit_s"):
        assert getattr(one, name)[a].item() == getattr(other, name)[b].item(), name
    assert a_loop.go_arounds[a] == b_loop.go_arounds[b]
    assert bool(one.done[a]) != bool(one.halted[a])             # each flight ends, or is held
    if bool(one.done[a]):
        assert a_loop.outcome(a) == b_loop.outcome(b)


def _same_record(original: Loop, i: int, copy: Loop, j: int) -> None:
    """The flown record of flight ``i`` and of its copy ``j`` (`Executor.flown`), over the cycles both flew: every field,
    states and commands within STATE_BOUND_M, the rest exact."""
    one, other = original.executor.flown(), copy.executor.flown()
    assert int(one.done_cycle[i]) == int(other.done_cycle[j])
    for name in ("states", "commands", "wanted", "runway", "sentence_s", "limits", "modes"):
        x, y = getattr(one, name), getattr(other, name)
        pairs = [(x[k], y[k]) for k in x] if isinstance(x, dict) else [(x, y)]
        for a, b in pairs:
            cycles = min(a.shape[1], b.shape[1])
            a, b = a[i, :cycles], b[j, :cycles]
            if a.dtype.is_floating_point:
                assert float(torch.nan_to_num(a - b).abs().max()) <= STATE_BOUND_M, name
                assert torch.equal(a.isnan(), b.isnan()), name
            else:
                assert torch.equal(a, b), name


def test_a_copy_of_chosen_flights_flies_as_its_originals(tmp_path, monkeypatch):
    """D97 (2): copied at two Δ rows — in B's go-around (G and speed "unspecified" in force) and C's turn past 180°,
    every flight flying; and when the first flight is done, D halted (its record, its go-around, its hold come with it) —
    each flight taken, one twice, and flown on with the same words, each copy flies what its original flies, records it
    alike and ends with its outcome; copying changes nothing of the loop copied."""
    loop_of, row_of = _schedules(tmp_path, monkeypatch)
    ways, taken = "ABCD", [2, 0, 3, 1, 1]
    copied_ways = "".join(ways[i] for i in taken)
    loop = loop_of(ways)
    rows, k = _fly(loop, ways, row_of, 0, steps=5)
    assert loop.go_arounds[1] == 1 and not bool(loop.executor.done.any())
    copies = []
    executor = loop.executor
    while not bool((executor.done | executor.halted).all()):
        if k == 5 or (bool(executor.done.any()) and len(copies) == 1):
            before = (loop.rows().copy(), executor.count, loop.spoken.steps, loop.go_arounds.copy(),
                      executor.time_limit_s.clone(), executor.done.clone(), executor.halted.clone())
            copy = loop.copy(taken)
            copies.append((k, copy))
            assert np.array_equal(before[0], loop.rows()) and before[1:3] == (executor.count, loop.spoken.steps)
            assert np.array_equal(before[3], loop.go_arounds) and torch.equal(before[4], executor.time_limit_s)
            assert torch.equal(before[5], executor.done) and torch.equal(before[6], executor.halted)
            for name in ("done", "done_cycle", "halted", "time_limit_s"):
                assert torch.equal(getattr(copy.executor, name), getattr(executor, name)[taken]), name
        flown, k = _fly(loop, ways, row_of, k, steps=1)
        rows += flown
    assert len(copies) == 2 and copies[1][0] > HALT_ROW and bool(copies[1][1].executor.halted[2])
    for at, copy in copies:
        copied, _ = _fly(copy, copied_ways, row_of, at, until_done=True)
        for j, i in enumerate(taken):
            _same_flight(rows[at:], loop, i, copied, copy, j)
            _same_record(loop, i, copy, j)
    for wrong in ([], [4], [-1], [True, False, True], [0.9]):
        with pytest.raises(ValueError, match="a copy takes one or more"):
            loop.copy(wrong)


def test_a_flights_states_do_not_depend_on_the_other_flights_of_its_loop(tmp_path, monkeypatch):
    """D97 (3), on the CPU: each flight flown alone and in a batch with the others flies alike."""
    loop_of, row_of = _schedules(tmp_path, monkeypatch)
    together = loop_of("ABCD")
    rows, _ = _fly(together, "ABCD", row_of, 0, until_done=True)
    for i, way in enumerate("ABCD"):
        alone = loop_of(way)
        own, _ = _fly(alone, way, row_of, 0, until_done=True)
        assert len(own) <= len(rows)
        _same_flight(own, alone, 0, rows, together, i)


# ---- moved starts (D97 (4)): the user's readings (2026-10-05) — positions and heights stretched about the first predicted
# step, the time limit the flight's own
def _moved_artefact(tmp_path, monkeypatch, interval_s=4.0):
    """`_artefact` with the start's physical context built from the flights it is given (the moved ones), not from the
    stored synthetic flight."""
    directory, words, batch, stored, anchor = _artefact(tmp_path, monkeypatch, interval_s)
    monkeypatch.setattr(start_module, "flight_inputs", lambda series, flights, anchors, airports, rule, device:
                        executor_inputs(flights[0], batch.geometries[0], anchors[0], rule=rule))
    return directory, words, batch, stored, anchor


def _started(directory, stored, tmp_path, interval_s, move=None):
    if move is None:
        loop, _ = start(directory, "train", interval_s, {0: stored}, tmp_path / "executor", most_go_arounds=0, device=CPU)
        return loop, None
    loop, _, observed = start_moved(directory, "train", interval_s, {0: stored}, tmp_path / "executor", {0: move},
                                    most_go_arounds=0, device=CPU)
    return loop, observed[0]


def test_a_move_of_zero_is_the_start_without_a_move_bit_for_bit(tmp_path, monkeypatch):
    """D97 (4): `NO_MOVE` gives the start, its observed rows the stored sentence's, and the same flight, bit for bit."""
    from ts_transformer.autopilot.flights import START_RULES, start_state, start_velocity
    from ts_transformer.autopilot.start import moved_signals
    from ts_transformer.instructions.signals import ROW_FIELDS
    from ts_transformer.tests.support import START_RULE

    directory, words, batch, stored, anchor = _moved_artefact(tmp_path, monkeypatch)
    signals, geometry = batch.observed[0], batch.geometries[0]
    for row in (anchor, 60, 120, 200, 221):            # the cut and the identity, for every start rule (at 221 a
        # stretch by 1.0 would change heights' last bits: the move skips a part that is zero)
        same = moved_signals(signals, row, NO_MOVE)
        assert same.n_rows == row + 1
        for name in ROW_FIELDS:
            assert np.array_equal(getattr(same, name), getattr(signals, name)[: row + 1]), name
        for rule in START_RULES:
            assert np.array_equal(start_velocity(same, [row], rule, geometry), start_velocity(signals, [row], rule, geometry))
    plain, _ = _started(directory, stored, tmp_path, 4.0)
    zero, observed = _started(directory, stored, tmp_path, 4.0, NO_MOVE)
    mass = float(plain.executor.inputs.initial_state[0, 6])
    assert np.array_equal(zero.executor.inputs.initial_state[0].numpy(),
                          start_state(signals, anchor, START_RULE, geometry, mass))      # the unmoved, uncut flight's
    for f in fields(FlightInputs):
        assert torch.equal(getattr(plain.executor.inputs, f.name), getattr(zero.executor.inputs, f.name)), f.name
    assert torch.equal(plain.executor.time_limit_s, zero.executor.time_limit_s)
    every = int(round(4.0 / words.spec.step_s))
    assert np.array_equal(observed, stored.rows.states[: stored.rows.start * every])
    for row in stored.rows.grid:
        a, _ = plain.step(row[None, :])
        b, _ = zero.step(row[None, :])
        assert np.array_equal(a, b)


def test_the_moved_observed_rows_give_the_moved_start_state_by_the_start_rule(tmp_path, monkeypatch):
    """D97 (4): a turn about the airport reference, a change of height and of speed move the observed rows to the first
    predicted step (`moved_signals`), and the start is the start rule's state and rows on them — exactly; so the start
    lies as far from the reference on a bearing turned by δ, raised by Δh, its speed scaled by 1 + κ with the path angle
    kept and its track turned — up to the airport frame's ground scale, which the start rule reads at each row's
    latitude and height (D87: within 1e-3 here) — and the time limit stays the flight's own."""
    from ts_transformer.autopilot.flights import observed_rows, start_state
    from ts_transformer.autopilot.start import moved_signals
    from ts_transformer.tests.support import START_RULE

    directory, words, batch, stored, anchor = _moved_artefact(tmp_path, monkeypatch)
    frame, geometry = batch.geometries[0].frame, batch.geometries[0]
    move = Move(turn_deg=10.0, height_m=100.0, speed_scale=1.04)
    plain, _ = _started(directory, stored, tmp_path, 4.0)
    moved, observed = _started(directory, stored, tmp_path, 4.0, move)
    every = int(round(4.0 / words.spec.step_s))
    shifted = moved_signals(batch.observed[0], anchor, move)
    mass = float(plain.executor.inputs.initial_state[0, 6])
    assert np.array_equal(moved.executor.inputs.initial_state[0].numpy(),
                          start_state(shifted, anchor, START_RULE, geometry, mass))
    assert np.array_equal(observed, observed_rows(shifted, stored.rows.first_row + np.arange(stored.rows.start * every),
                                                  START_RULE, geometry))
    (lat0, lon0, h0, v0, psi0, gamma0, _), (lat1, lon1, h1, v1, psi1, gamma1, _) = (
        loop.executor.inputs.initial_state[0].tolist() for loop in (plain, moved))
    (e0, n0), (e1, n1) = (frame.horizontal_from_latlon(lat, lon) for lat, lon in ((lat0, lon0), (lat1, lon1)))
    assert math.hypot(e1, n1) == pytest.approx(math.hypot(e0, n0), abs=1e-6)
    bearing = lambda e, n: math.degrees(math.atan2(e, n))  # noqa: E731 — compass
    assert (bearing(e1, n1) - bearing(e0, n0)) % 360.0 == pytest.approx(10.0, abs=1e-9)
    assert h1 == pytest.approx(h0 + 100.0, abs=1e-9)
    assert v1 == pytest.approx(v0 * 1.04, rel=1e-3) and gamma1 == pytest.approx(gamma0, abs=1e-3)
    assert math.remainder(psi0 - psi1 - math.radians(10.0), 2 * math.pi) == pytest.approx(0.0, abs=1e-3)
    assert torch.equal(plain.executor.time_limit_s, moved.executor.time_limit_s)
    before = stored.rows.states[: stored.rows.start * every]          # the unmoved observed rows
    assert np.allclose(observed[:, 2] - observed[-1, 2], 1.04 * (before[:, 2] - before[-1, 2]), atol=1e-6)
    assert np.allclose(np.mod(observed[:, 3] - before[:, 3] + 180.0, 360.0) - 180.0, 10.0, atol=0.1)
    assert np.allclose(observed[:, 4], 1.04 * before[:, 4], rtol=1e-3)


def test_a_move_stretches_heights_with_positions_and_keeps_the_path_angle_for_every_start_rule(tmp_path, monkeypatch):
    """D97 (4), the user's reading: the speed change stretches positions and heights about the first predicted step, so
    the start rule reads the speed scaled and the path angle kept — in a descent (row 120) and in a turn (row 200), for
    every start rule (the centred fit reads the data plane's track, ground speed and vertical rate, turned and scaled
    alike) — up to the frame's ground scale at the moved rows (D87; a 30° turn here: within 0.5 %)."""
    from ts_transformer.autopilot.flights import START_RULES, start_state
    from ts_transformer.autopilot.start import moved_signals

    _, _, batch, _, _ = _moved_artefact(tmp_path, monkeypatch)
    signals, geometry = batch.observed[0], batch.geometries[0]
    move = Move(turn_deg=30.0, height_m=200.0, speed_scale=1.05)
    for row in (120, 200):
        moved = moved_signals(signals, row, move)
        h0, h1 = signals.altitude_m[: row + 1], moved.altitude_m
        assert np.allclose(h1 - h1[row], 1.05 * (h0 - h0[row]), atol=1e-9) and h1[row] == pytest.approx(h0[row] + 200.0)
        assert np.allclose(moved.track_deg, signals.track_deg[: row + 1] + 30.0)          # unwrapped, turned
        across = moved_signals(signals, row, Move(turn_deg=120.0)).track_deg       # past 360°: still unwrapped
        assert np.allclose(across, signals.track_deg[: row + 1] + 120.0) and across.max() > 360.0
        assert np.allclose(moved.ground_speed_mps, 1.05 * signals.ground_speed_mps[: row + 1])
        for rule in START_RULES:
            _, _, _, v0, psi0, gamma0, _ = start_state(signals, row, rule, geometry, 62000.0)
            _, _, _, v1, psi1, gamma1, _ = start_state(moved, row, rule, geometry, 62000.0)
            assert v1 == pytest.approx(1.05 * v0, rel=5e-3), (row, rule)
            assert gamma1 == pytest.approx(gamma0, abs=5e-4), (row, rule)
            assert math.degrees(math.remainder(psi0 - psi1, 2 * math.pi)) == pytest.approx(30.0, abs=0.2), (row, rule)


def test_a_moved_start_refuses_a_sentence_not_its_flights_and_a_wrong_move(tmp_path, monkeypatch):
    """D97 (4): a sentence whose observed rows are not its flight's is refused with a move too (that the check reads the
    flight unmoved: the moved start of a valid sentence starts, above), and a move is one per flight, finite, its speed
    scale positive."""
    directory, words, _, stored, _ = _moved_artefact(tmp_path, monkeypatch)
    states = stored.rows.states.copy()
    states[0, 0] += 1.0
    other = replace(stored, rows=replace(stored.rows, states=states))
    with pytest.raises(ValueError, match="its observed rows differ"):
        start_moved(directory, "train", 4.0, {0: other}, tmp_path / "executor", {0: Move(turn_deg=5.0)},
                    most_go_arounds=0, device=CPU)
    with pytest.raises(ValueError, match="a move for each flight"):
        start_moved(directory, "train", 4.0, {0: stored}, tmp_path / "executor", {}, most_go_arounds=0, device=CPU)
    for wrong in (dict(turn_deg=math.nan), dict(height_m=math.inf), dict(speed_scale=0.0), dict(speed_scale=-1.0)):
        with pytest.raises(ValueError, match="a move is finite"):
            Move(**wrong)


def _same_start(a, b, grid) -> None:
    """Two starts (`start_moved`'s ``(loop, order, observed)``) give the same order, observed rows, physical context and
    time limits, and fly ``grid`` (the stored sentence's rows) to the same states, bit for bit."""
    (loop_a, order_a, observed_a), (loop_b, order_b, observed_b) = a, b
    assert order_a == order_b and observed_a.keys() == observed_b.keys()
    for i in observed_a:
        assert np.array_equal(observed_a[i], observed_b[i])
    for f in fields(FlightInputs):
        assert torch.equal(getattr(loop_a.executor.inputs, f.name), getattr(loop_b.executor.inputs, f.name)), f.name
    assert torch.equal(loop_a.executor.time_limit_s, loop_b.executor.time_limit_s)
    for row in grid:
        rows_a, done_a = loop_a.step(row[None, :])
        rows_b, done_b = loop_b.step(row[None, :])
        assert np.array_equal(rows_a, rows_b) and np.array_equal(done_a, done_b)


def test_an_opened_start_gives_start_moveds_loop_order_and_observed_rows_bit_for_bit(tmp_path, monkeypatch):
    """A44 (D138): `Start.moved` gives what the one-call form `start_moved` gives — no move and a moved start, and a
    flight started again on the series it kept — bit for bit."""
    from ts_transformer.autopilot.start import Start

    from types import SimpleNamespace

    directory, words, batch, stored, _ = _moved_artefact(tmp_path, monkeypatch)
    # each flight's series a marker of its flight, and the start's context refused for a series not its flight's: a
    # series kept under another key than the flight's place in the split is caught
    monkeypatch.setattr(start_module, "rebuild_series",
                        lambda d, flights: [SimpleNamespace(dataset_id=f.dataset_id) for f in flights])

    def inputs(series, flights, anchors, airports, rule, device):
        assert [s.dataset_id for s in series] == [f.dataset_id for f in flights]
        return executor_inputs(flights[0], batch.geometries[0], anchors[0], rule=rule)

    monkeypatch.setattr(start_module, "flight_inputs", inputs)
    opened = Start(directory, "train", 4.0, tmp_path / "executor")
    for move in (NO_MOVE, Move(turn_deg=10.0, speed_scale=1.04), NO_MOVE):     # a raise makes its words ungrammatical
        _same_start(opened.moved({0: stored}, {0: move}, most_go_arounds=0, device=CPU),
                    start_moved(directory, "train", 4.0, {0: stored}, tmp_path / "executor", {0: move},
                                most_go_arounds=0, device=CPU),
                    stored.rows.grid)
    with pytest.raises(ValueError, match="a move for each flight"):
        opened.moved({0: stored}, {}, most_go_arounds=0, device=CPU)


def test_an_opened_starts_second_call_reads_no_file(tmp_path, monkeypatch):
    """A44: a `Start` reads the artefact, the spec and the closed-loop file when it is opened and rebuilds a flight the
    first time it is started; a second call opens no file and rebuilds nothing (the readers counted)."""
    import builtins
    import io

    from ts_transformer.autopilot.start import Start

    directory, _, _, stored, _ = _moved_artefact(tmp_path, monkeypatch)
    rebuilt = []
    rebuild = start_module.rebuild_series
    monkeypatch.setattr(start_module, "rebuild_series", lambda d, flights: rebuilt.append(len(flights)) or rebuild(
        d, flights))
    opened = Start(directory, "train", 4.0, tmp_path / "executor")
    first = opened.moved({0: stored}, {0: NO_MOVE}, most_go_arounds=0, device=CPU)
    assert rebuilt == [1]
    files = []
    real_open = builtins.open

    def counted(path, *args, **kwargs):
        files.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counted)
    monkeypatch.setattr(io, "open", counted)
    again = opened.moved({0: stored}, {0: Move(turn_deg=5.0)}, most_go_arounds=0, device=CPU)
    monkeypatch.setattr(builtins, "open", real_open)
    monkeypatch.setattr(io, "open", real_open)
    assert files == [] and rebuilt == [1]
    assert again[1] == first[1]


def _forked_start(opened, stored, out) -> None:
    """In a forked process: one thread (a fork after the parent's CPU threads ran can hang them), the flight started on
    the parent's opened `Start` and flown; its observed rows and flown states sent back."""
    torch.set_num_threads(1)
    loop, order, observed = opened.moved({0: stored}, {0: NO_MOVE}, most_go_arounds=0, device=CPU)
    out.send((order, observed[0], [loop.step(row[None, :])[0] for row in stored.rows.grid]))
    out.close()


def test_a_start_opened_before_a_fork_serves_the_forked_process(tmp_path, monkeypatch):
    """A44: a `Start` opened in the parent serves a forked child — the child starts and flies the flight as the parent's
    one-call start does, bit for bit."""
    import multiprocessing

    from ts_transformer.autopilot.start import Start

    directory, _, _, stored, _ = _moved_artefact(tmp_path, monkeypatch)
    opened = Start(directory, "train", 4.0, tmp_path / "executor")
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_forked_start, args=(opened, stored, send), daemon=True)
    child.start()
    send.close()
    try:
        assert receive.poll(120), "the forked start sent nothing in 120 s"
        order, observed, flown = receive.recv()
        child.join(30)
        assert child.exitcode == 0
    finally:                                           # a hung child never holds the test (or its worker) open
        child.kill()
        child.join()
    loop, here, rows = start_moved(directory, "train", 4.0, {0: stored}, tmp_path / "executor", {0: NO_MOVE},
                                   most_go_arounds=0, device=CPU)
    assert order == here and np.array_equal(observed, rows[0])
    for row, states in zip(stored.rows.grid, flown, strict=True):
        assert np.array_equal(loop.step(row[None, :])[0], states)


def test_the_start_refuses_sentences_not_read_under_its_artefact_split_and_executor(tmp_path, monkeypatch):
    """The sentences are tied to what they were read under: the vocabulary, the executor parameters of the artefact's
    closed-loop file, the row interval, and the flight (its observed rows are its stored signals: a sentence of another
    split or another flight is refused)."""
    directory, words, _, stored, _ = _artefact(tmp_path, monkeypatch, 4.0)
    with pytest.raises(ValueError, match="no sentence"):
        start(directory, "train", 4.0, {}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    late = ClosedLoopSentence(rows=replace(stored.rows, start=stored.rows.start + 1), withheld=stored.withheld)
    with pytest.raises(ValueError, match="does not start at row 4 of 4 s"):
        start(directory, "train", 4.0, {0: late}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    with pytest.raises(FileNotFoundError):             # no closed-loop sentence of the split read at 2 s
        start(directory, "train", 2.0, {0: stored}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    other_params = _executor(tmp_path / "other_params", directory, replace(_params(), timeout_factor=2.0),
                             words.spec.sha256)
    with pytest.raises(ValueError, match="flown by executor parameters"):
        start(directory, "train", 4.0, {0: stored}, other_params, most_go_arounds=0, device=CPU)
    moved = ClosedLoopSentence(rows=replace(stored.rows, states=stored.rows.states + np.array([1.0, 0, 0, 0, 0, 0])),
                               withheld=stored.withheld)                              # another flight's rows
    with pytest.raises(ValueError, match="its observed rows differ"):
        start(directory, "train", 4.0, {0: moved}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    # D71, D73: the start opens the spec itself — one measured against another vocabulary, or one whose reference tracks
    # the code in the process does not fly within the bounds, is refused by name
    other_vocabulary = _executor(tmp_path / "other_vocabulary", directory, _params(), "0" * 64, reference=False)
    with pytest.raises(ValueError, match="the executor spec was measured against vocabulary 000000000000"):
        start(directory, "train", 4.0, {0: stored}, other_vocabulary, most_go_arounds=0, device=CPU)
    off = _executor(tmp_path / "off", directory, _params(), words.spec.sha256)
    with np.load(off / "conformance" / "reference.npz") as data:
        arrays = {name: data[name] for name in data.files}
    arrays["states"][:, -1, 2] += 1.0                                  # its last height a metre off what is flown
    with (off / "conformance" / "reference.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    with pytest.raises(ValueError, match="the executor flies off's reference tracks otherwise: batch: KXXX:a"):
        start(directory, "train", 4.0, {0: stored}, off, most_go_arounds=0, device=CPU)
    with pytest.raises(ValueError, match="negative"):
        Loop(executor_inputs(load_signals(directory, "train")[0], load_candidates(directory)["KXXX"]),
             [load_candidates(directory)["KXXX"]], [A320_IAS], [100.0], _params(), words, interval_s=4.0,
             most_go_arounds=-1, device=CPU)


@pytest.mark.parametrize("interval_s", [2.0, 8.0])
def test_the_start_check_runner_says_stored_sentences_through_the_start(tmp_path, monkeypatch, interval_s):
    """R53 (§12.2 item 7): the artefact's stored sentences said through the start give back their states, are done at
    their last row and time out as stored; a flight off its stored states fails the check."""
    from ts_transformer.experiments import closed_loop_start_check as runner

    directory, words, _, _, _ = _artefact(tmp_path, monkeypatch, interval_s)
    (row,) = runner.check_interval(directory, "train", interval_s, tmp_path / "executor", words, per_airport=0,
                                   seed=1337, chunk=8, device=CPU)
    assert row["same_rows"] and row["position_m"] == 0.0 and row["other_columns"] == 0.0 and runner.passes(row)
    assert not runner.passes({**row, "position_m": 2e-6}) and not runner.passes({**row, "done_at_last_row": False})
    assert row["outcome_as_stored"] and not runner.passes({**row, "outcome_as_stored": False})          # D74


def test_the_start_check_samples_each_airport_alike():
    from ts_transformer.experiments.closed_loop_start_check import sample

    airports = ["A"] * 5 + ["B"] * 3
    chosen = sample(airports, 2, 1337)
    assert len(chosen) == 4 and sorted(airports[k] for k in chosen) == ["A", "A", "B", "B"] and chosen == sorted(chosen)
    assert sample(airports, 0, 1337) == list(range(8)) and sample(airports, 2, 1337) == chosen
    with pytest.raises(ValueError, match="fewer than 4"):
        sample(airports, 4, 1337)
    with pytest.raises(ValueError, match="negative"):
        sample(airports, -1, 1337)


def test_the_start_check_runner_exits_1_on_a_failed_flight_and_never_overwrites(tmp_path, monkeypatch):
    from ts_transformer.experiments import closed_loop_start_check as runner

    monkeypatch.setattr(closed_loop, "require_conforming_closed_loop",
                        lambda instructions, executor: (_params(), {"sha256": "e", "checks": {}}, None))
    good = {"dataset_id": "KXXX:a", "same_rows": True, "position_m": 0.0, "other_columns": 0.0,
            "done_at_last_row": True, "timed_out_as_stored": True, "outcome_as_stored": True}
    argv = ["--instructions", str(tmp_path), "--executor", str(tmp_path), "--split", "train", "--row-interval-s", "2"]
    for name, rows, code in (("ok", [good], 0), ("bad", [good, {**good, "dataset_id": "KXXX:b", "same_rows": False,
                                                                     "position_m": None, "other_columns": None}], 1)):
        monkeypatch.setattr(runner, "check_interval", lambda *a, rows=rows, **k: rows)
        assert runner.main([*argv, "--out", str(tmp_path / name)]) == code
    with pytest.raises(SystemExit):
        runner.main([*argv, "--out", str(tmp_path / "ok")])
