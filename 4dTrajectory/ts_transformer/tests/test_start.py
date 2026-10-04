"""The start of a closed loop (vocabulary §6 item 5, D67, §12.1 A26): `autopilot/start.py`."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import closed_loop, replay, start as start_module
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.judge import outcome_of
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.speed import approach_speed_ias_mps
from ts_transformer.autopilot.start import GoAroundBeyondMost, Loop, start
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.instructions.artefact import (
    CLOSED_LOOP_DIRECTORY, ClosedLoopSentence, closed_loop_path, closed_loop_sentences, load_candidates, load_closed_loop,
    load_signals, write_closed_loop,
)
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import ALTITUDE, ANGLE, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words
from ts_transformer.tests.support import executor_inputs, labelled_instruction_artefact

CPU = torch.device("cpu")
A320_IAS = approach_speed_ias_mps("A320", 62000.0)


def _params():
    from ts_transformer.experiments.executor_spec import ROLL_RATE_DEG_S

    return ExecutorParams(cycle_s=1.0, bank_rate_deg_s=ROLL_RATE_DEG_S, path_time_constant_s=2.0, path_rate_factor=2.0,
                          timeout_factor=1.5)


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
    batch = replay.Batch(indices=[0], signals=[replay.from_row(signals, sentence.first_row)], series=[None],
                         readings=[reading], sentences=[sentence], row_interval_s=interval_s, geometries=[geometry],
                         approach_ias_mps=[A320_IAS], groups=[replay.OWN], drawn={})
    (read,) = closed_loop.read(batch, inputs, _params(), words, device=CPU)
    assert isinstance(read, ClosedLoopSentence)
    (directory / CLOSED_LOOP_DIRECTORY).mkdir()
    path = closed_loop_path(directory, "train", interval_s)
    write_closed_loop(path, spec, executor_params_sha256=params_sha256(_params()), row_interval_s=interval_s,
                      start_row=start_module.start_row(interval_s), sentences={0: read})
    (stored,) = closed_loop_sentences(load_closed_loop(path, spec)).values()
    monkeypatch.setattr(start_module, "rebuild_series", lambda d, flights: [None] * len(flights))
    monkeypatch.setattr(replay, "group_of", lambda series: replay.OWN)
    monkeypatch.setattr(replay, "flight_approach_ias_mps", lambda series, group: A320_IAS)
    monkeypatch.setattr(start_module, "flight_inputs",
                        lambda series, anchors, device: executor_inputs(signals, geometry, anchors[0]))
    monkeypatch.setattr(replay.Batch, "inputs", lambda self, device: inputs)
    from ts_transformer.autopilot import conformance
    from ts_transformer.instructions import conformance as labeller

    labeller.write_reference(directory, [signals], [labeller.labelled_record(reading)], [reading],
                             git={"head": "x", "dirty": False})
    flown = replay.sentence_on_interval(reading, signals, spec.step_s, geometry, words)
    reference = replay.Batch(indices=[0], signals=[signals], series=[None], readings=[reading], sentences=[flown],
                             row_interval_s=spec.step_s, geometries=[geometry], approach_ias_mps=[A320_IAS],
                             groups=[replay.OWN], drawn={"split": "train"})
    monkeypatch.setattr(conformance, "draw_reference_batch", lambda instructions, words, draw: reference)
    _executor(tmp_path / "executor", directory, _params(), spec.sha256)
    return directory, words, batch, stored, anchor


@pytest.mark.parametrize("interval_s", [2.0, 4.0, 8.0])
def test_a_stored_sentence_said_through_the_start_gives_its_states_and_its_outcome(tmp_path, monkeypatch, interval_s):
    """D67: the words of a closed-loop sentence, said row by row through the start, give back its stored states on the 2 s
    rows from the first predicted step and the outcome its replay is judged to; the time limit is the closed-loop
    reading's."""
    directory, words, batch, stored, anchor = _artefact(tmp_path, monkeypatch, interval_s)
    params = _params()
    loop, order = start(directory, "train", interval_s, {0: stored}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    assert order == [0]
    limit = (len(batch.readings[0].words) - anchor) * words.spec.step_s * params.timeout_factor   # §5.8, written out
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
    assert np.abs(np.array(flown)[:, :3] - expected[:, :3]).max() <= STATE_BOUND_M     # the conformance tolerance
    assert np.array_equal(np.array(flown), expected)                                   # and here, the same code: exact
    moved, _ = closed_loop.replay_batch(batch, {0: stored}, words)
    replayed = outcome_of(replay.fly_sentences(moved, params, words, device=CPU), 0, batch.geometries[0], words.spec)
    assert loop.outcome(0).outcome == replayed.outcome
    assert loop.outcome(0).crossing == replayed.crossing
    assert bool(loop.timed_out()[0]) == stored.timed_out


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
        loop.step(stored.grid[:1])
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
    loop.step(stored.grid[:1])
    loop.halt(np.array([True]))
    loop.step(climb)
    assert loop.go_arounds[0] == 0
    # a done flight's words are not heard: its go-around is no go-around
    loop, _ = start(directory, "train", 2.0, {0: stored}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    with pytest.raises(ValueError, match="not done"):
        loop.outcome(0)
    for row in stored.grid:
        loop.step(row[None, :])
    ended = loop.outcome(0)
    loop.step(climb)
    assert loop.go_arounds[0] == 0 and loop.outcome(0) == ended


def test_the_start_refuses_sentences_not_read_under_its_artefact_split_and_executor(tmp_path, monkeypatch):
    """The sentences are tied to what they were read under: the vocabulary, the executor parameters of the artefact's
    closed-loop file, the row interval, and the flight (its observed rows are its stored signals: a sentence of another
    split or another flight is refused)."""
    directory, words, _, stored, _ = _artefact(tmp_path, monkeypatch, 4.0)
    with pytest.raises(ValueError, match="no sentence"):
        start(directory, "train", 4.0, {}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    with pytest.raises(ValueError, match="does not start at row 8 of 2 s"):
        start(directory, "train", 2.0, {0: stored}, tmp_path / "executor", most_go_arounds=0, device=CPU)
    other_params = _executor(tmp_path / "other_params", directory, replace(_params(), timeout_factor=2.0),
                             words.spec.sha256)
    with pytest.raises(ValueError, match="flown by executor parameters"):
        start(directory, "train", 4.0, {0: stored}, other_params, most_go_arounds=0, device=CPU)
    moved = replace(stored, states=stored.states + np.array([1.0, 0, 0, 0, 0, 0]))     # another flight's rows
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
            "done_at_last_row": True, "timed_out_as_stored": True}
    argv = ["--instructions", str(tmp_path), "--executor", str(tmp_path), "--split", "train", "--row-interval-s", "2"]
    for name, rows, code in (("ok", [good], 0), ("bad", [good, {**good, "dataset_id": "KXXX:b", "same_rows": False,
                                                                     "position_m": None, "other_columns": None}], 1)):
        monkeypatch.setattr(runner, "check_interval", lambda *a, rows=rows, **k: rows)
        assert runner.main([*argv, "--out", str(tmp_path / name)]) == code
    with pytest.raises(SystemExit):
        runner.main([*argv, "--out", str(tmp_path / "ok")])
