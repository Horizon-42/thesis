"""The start of a closed loop (vocabulary §6 item 5, D67, §12.1 A26): `autopilot/start.py`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import closed_loop, replay, start as start_module
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.judge import outcome_of
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.speed import approach_speed_ias_mps
from ts_transformer.autopilot.start import GoAroundBeyondMost, Loop, start
from ts_transformer.instructions.artefact import ClosedLoopSentence, load_candidates, load_signals
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import ALTITUDE, ANGLE, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words
from ts_transformer.tests.support import executor_inputs, labelled_instruction_artefact

CPU = torch.device("cpu")
A320_IAS = approach_speed_ias_mps("A320", 62000.0)


def _params():
    from ts_transformer.experiments.executor_spec import ROLL_RATE_DEG_S

    return ExecutorParams(cycle_s=1.0, bank_rate_deg_s=ROLL_RATE_DEG_S, path_time_constant_s=2.0, path_rate_factor=2.0,
                          timeout_factor=1.5)


def _artefact(tmp_path, monkeypatch, interval_s):
    """A tmp artefact of one labelled flight (no harvest behind it: its rebuild, aircraft and approach speed stand in),
    its closed-loop sentence read at ``interval_s`` and the batch it was read from."""
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
    batch = replay.Batch(indices=[0], signals=[replay.from_row(signals, sentence.first_row)], series=[None],
                         readings=[reading], sentences=[sentence], row_interval_s=interval_s, geometries=[geometry],
                         approach_ias_mps=[A320_IAS], groups=[replay.OWN], drawn={})
    (stored,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    assert isinstance(stored, ClosedLoopSentence)
    monkeypatch.setattr(start_module, "rebuild_series", lambda d, flights: [None] * len(flights))
    monkeypatch.setattr(replay, "group_of", lambda series: replay.OWN)
    monkeypatch.setattr(replay, "flight_approach_ias_mps", lambda series, group: A320_IAS)
    monkeypatch.setattr(start_module, "flight_inputs",
                        lambda series, anchors, device: executor_inputs(signals, geometry, anchors[0]))
    monkeypatch.setattr(replay.Batch, "inputs", lambda self, device: inputs)
    return directory, words, batch, stored, anchor


@pytest.mark.parametrize("interval_s", [2.0, 4.0, 8.0])
def test_a_stored_sentence_said_through_the_start_gives_its_states_and_its_outcome(tmp_path, monkeypatch, interval_s):
    """D67: the words of a closed-loop sentence, said row by row through the start, give back its stored states on the 2 s
    rows from the first predicted step and the outcome its replay is judged to; the time limit is the closed-loop
    reading's."""
    directory, words, batch, stored, anchor = _artefact(tmp_path, monkeypatch, interval_s)
    params = _params()
    loop, order = start(directory, "train", interval_s, {0: stored}, params, words, most_go_arounds=0, device=CPU)
    assert order == [0]
    limit = replay.remaining_observed_s(batch.readings[0], anchor, words.spec.step_s) * params.timeout_factor
    assert float(loop.executor.time_limit_s[0]) == limit
    flown = [loop.rows()[0]]
    for k, row in enumerate(stored.grid):
        rows, done = loop.step(row[None, :])
        if k < len(stored.grid) - 1:
            assert not done[0]
            flown += list(rows[0])
    assert done[0]                                     # done in the row the sentence ends with
    every = int(round(interval_s / words.spec.step_s))
    expected = stored.states[stored.start * every:]
    assert len(flown) == len(expected)
    assert np.abs(np.array(flown)[:, :3] - expected[:, :3]).max() <= STATE_BOUND_M
    moved, _ = closed_loop.replay_batch(batch, {0: stored}, words)
    replayed = outcome_of(replay.fly_sentences(moved, params, words, device=CPU), 0, batch.geometries[0], words.spec)
    assert loop.outcome(0).outcome == replayed.outcome
    assert loop.outcome(0).crossing == replayed.crossing
    assert bool(loop.timed_out()[0]) == stored.timed_out


def test_a_go_around_beyond_the_most_given_is_refused_by_name(tmp_path, monkeypatch):
    """D67: each flight may say the most go-arounds the loop was started with; one more refuses the row by name before
    anything is flown, and the time limit grows by 900 s for each one heard."""
    directory, words, _, stored, _ = _artefact(tmp_path, monkeypatch, 2.0)
    params = _params()
    climb = np.full((1, 5), UNCHANGED, dtype=np.int64)
    climb[0, [RUNWAY, ALTITUDE, ANGLE]] = RUNWAY_GO_AROUND, words.altitude_index(1500.0), words.angle_climb
    again = np.full((1, 5), UNCHANGED, dtype=np.int64)
    again[0, RUNWAY] = 0
    for most in (0, 1):
        loop, _ = start(directory, "train", 2.0, {0: stored}, params, words, most_go_arounds=most, device=CPU)
        loop.step(stored.grid[:1])
        before = float(loop.executor.time_limit_s[0])
        if most == 0:
            with pytest.raises(GoAroundBeyondMost, match="beyond the most 0"):
                loop.step(climb)
            assert loop.steps == 1 and loop.executor.count == 2       # nothing flown
            continue
        loop.step(climb)
        assert float(loop.executor.time_limit_s[0]) == before + 900.0
        loop.step(again)
        with pytest.raises(GoAroundBeyondMost, match="beyond the most 1"):
            loop.step(climb)


def test_the_start_refuses_another_vocabulary_or_another_row_interval(tmp_path, monkeypatch):
    directory, words, _, stored, _ = _artefact(tmp_path, monkeypatch, 4.0)
    with pytest.raises(ValueError, match="does not start at row 8 of 2 s"):
        start(directory, "train", 2.0, {0: stored}, _params(), words, most_go_arounds=0, device=CPU)
    other = Words(type(words.spec).from_dict({**words.spec.to_dict(), "closed_loop_lateral_m": 20.0}))
    with pytest.raises(ValueError, match="another vocabulary"):
        start(directory, "train", 4.0, {0: stored}, _params(), other, most_go_arounds=0, device=CPU)
    with pytest.raises(ValueError, match="negative"):
        Loop(executor_inputs(load_signals(directory, "train")[0], load_candidates(directory)["KXXX"]),
             [load_candidates(directory)["KXXX"]], [A320_IAS], [100.0], _params(), words, interval_s=4.0,
             most_go_arounds=-1, device=CPU)
