"""The start of a closed loop (vocabulary §6 item 5, D67, §12.1 A26): `autopilot/start.py`."""

from __future__ import annotations

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
from ts_transformer.autopilot.start import GoAroundBeyondMost, Loop, RowRefused, start
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
