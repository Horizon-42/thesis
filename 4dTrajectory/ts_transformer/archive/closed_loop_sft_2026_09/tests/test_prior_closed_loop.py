"""Closed-loop supervised fine-tuning (prior design §9.2, `experiments.prior_closed_loop`): branches copied by `take`,
the chain kept closest to the observed flight, the targets re-read against the chain's words, one pass that counts only
the asked columns."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot.executor import Executor
from ts_transformer.autopilot.frame import read_state
from ts_transformer.autopilot.judge import OUTCOMES
from ts_transformer.autopilot.sentence import Spoken
from ts_transformer.experiments.prior_closed_loop import ChainConfig, choose, fly_chains
from ts_transformer.experiments.prior_free_generation import speak_and_fly
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, APPROACH, APPROACH_CLEARED, APPROACH_NOT_CLEARED, HEADING, RUNWAY, SPEED, UNCHANGED, Words,
)
from ts_transformer.prior import data as prior_data
from ts_transformer.prior.data import Split, chain_record, column_classes
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.relabel import relabel
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.prior.train import FineTuneConfig, FineTuner, column_nll, to_batch
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec as spec
from ts_transformer.tests.test_autopilot import DOWNWIND_BASE_FINAL, F64, _params, _physics

CPU = torch.device("cpu")


# ---- the targets
def _base(words: Words) -> np.ndarray:
    return np.array([0, APPROACH_NOT_CLEARED, 10, words.altitude_index(900.0), 0, 5], dtype=np.int64)


def _sentence(rows: int, base: np.ndarray, changes: dict[tuple[int, int], int]) -> np.ndarray:
    grid = np.full((rows, 6), UNCHANGED, dtype=np.int64)
    grid[0] = base
    for (row, column), word in changes.items():
        grid[row, column] = word
    return grid


def test_the_chain_is_asked_for_the_label_s_words_around_the_rows_the_label_says_them():
    one = spec()
    words = Words(one)
    base = _base(words)
    label = _sentence(N_LOOK + 20, base, {(N_LOOK + 4, HEADING): 14, (N_LOOK + 15, HEADING): 18,
                                          (N_LOOK + 6, SPEED): 7})
    steps = 25                                               # the chain outlives the sentence by 5 steps
    said = np.full((steps, 6), UNCHANGED, dtype=np.int64)
    said[0] = [*base[:5], 9]                                 # its own speed at the first step
    said[1, HEADING] = 14                                    # 3 rows before the label: answered early
    no = np.zeros(steps, dtype=bool)
    out = relabel(label, said, no, no, np.full(steps, 900.0), one, words, window=5)
    assert (out.classes[0] == base + 1).all()                # the first step: the label's words in force
    heading, speed = out.classes[:, HEADING], out.classes[:, SPEED]
    assert (heading[1:15] == 0).all()                        # the early word is not taken back
    assert (heading[15:21] == 18 + 1).all() and out.asked[15:21, HEADING].all()   # unanswered: from u to u + 5
    assert not out.asked[21:, HEADING].any()                 # missed by more than 5 rows: not asked (past the end too)
    assert (speed[1:6] == 0).all()                           # its first-step speed is never corrected
    assert (speed[6:12] == 7 + 1).all() and not out.asked[12:, SPEED].any()
    assert out.counts["said on time"] == 12 and out.counts["missed"] == 4 + 13 and out.counts["answered"] == 0
    # the speed answered with its own value, inside the window: nothing more asked
    said[8, SPEED] = 8
    answered = relabel(label, said, no, no, np.full(steps, 900.0), one, words, window=5)
    assert (answered.classes[6:9, SPEED] == 7 + 1).all() and (answered.classes[9:, SPEED] == 0).all()
    assert answered.asked[:, SPEED].all() and answered.counts["answered"] == steps - 9
    # the heading answered 3 rows early with a word of its own: not taken back, nothing asked
    said[12, HEADING] = 16
    own = relabel(label, said, no, no, np.full(steps, 900.0), one, words, window=5)
    assert (own.classes[1:, HEADING] == 0).all() and own.asked[:, HEADING].all()
    for targets in (out, answered, own):
        assert (targets.classes[~targets.asked] == 0).all()         # a column left out reads as "unchanged"


def test_a_clearance_is_asked_for_until_it_is_given_and_a_word_the_executor_ignores_is_withheld():
    one = spec()
    words = Words(one)
    base = _base(words)
    label = _sentence(N_LOOK + 30, base, {(N_LOOK + 2, APPROACH): APPROACH_CLEARED})
    said = np.full((30, 6), UNCHANGED, dtype=np.int64)
    said[0] = base
    no = np.zeros(30, dtype=bool)
    out = relabel(label, said, no, no, np.full(30, 900.0), one, words, window=5)
    assert (out.classes[2:, APPROACH] == APPROACH_CLEARED + 1).all() and out.asked.all()
    assert out.counts["said on time"] == 6 and out.counts["clearance late"] == 30 - 8
    said[12, APPROACH] = APPROACH_CLEARED
    assert (relabel(label, said, no, no, np.full(30, 900.0), one, words, window=5).classes[13:, APPROACH] == 0).all()
    # cleared at the handover: asked for until the chain gives it, however early the label gave it
    handover = _sentence(N_LOOK + 30, np.array([0, APPROACH_CLEARED, *base[2:]]), {})
    said[12, APPROACH] = UNCHANGED
    said[6, APPROACH] = APPROACH_CLEARED
    early = relabel(handover, said, no, no, np.full(30, 900.0), one, words, window=5)
    assert (early.classes[1:7, APPROACH] == APPROACH_CLEARED + 1).all() and (early.classes[7:, APPROACH] == 0).all()
    assert early.counts["clearance at the handover"] == 6
    # the label changes the runway, takes the clearance back and turns: an executor that has cleared or captured
    # would not act on the first two, one that has captured not on the third
    label = _sentence(N_LOOK + 10, np.array([0, APPROACH_CLEARED, *base[2:]]),
                      {(N_LOOK + 3, RUNWAY): 1, (N_LOOK + 3, APPROACH): APPROACH_NOT_CLEARED, (N_LOOK + 3, HEADING): 12})
    said = np.full((10, 6), UNCHANGED, dtype=np.int64)
    said[0] = [0, APPROACH_CLEARED, *base[2:]]
    cleared, captured = np.ones(10, dtype=bool), np.arange(10) >= 5
    out = relabel(label, said, cleared, captured, np.full(10, 900.0), one, words, window=5)
    assert (out.classes[1:, RUNWAY] == 0).all() and (out.classes[1:, APPROACH] == 0).all()
    assert (out.classes[3:5, HEADING] == 12 + 1).all() and (out.classes[5:, HEADING] == 0).all()
    assert out.counts["runway locked"] == 7 and out.counts["not cleared after a clearance"] == 7
    assert out.counts["heading after the capture"] == 5 and out.asked.all()


def test_a_step_that_breaks_a_compatibility_rule_is_not_asked_in_that_rule_s_columns():
    one = spec()
    words = Words(one)
    base = _base(words)
    low = words.altitude_index(300.0)
    label = _sentence(N_LOOK + 10, base, {(N_LOOK + 2, ALTITUDE): low})       # 300 m, level (the label was low)
    said = np.full((10, 6), UNCHANGED, dtype=np.int64)
    said[0] = base
    no = np.zeros(10, dtype=bool)
    high = relabel(label, said, no, no, np.full(10, 900.0), one, words, window=5)
    assert not high.asked[2:8, [ALTITUDE, ANGLE]].any() and high.asked[:2].all() and high.asked[:, HEADING].all()
    assert not high.asked[8:, ALTITUDE].any() and high.asked[8:, ANGLE].all()     # missed; the label's angle: kept
    assert (high.classes[2:, [ALTITUDE, ANGLE]] == 0).all() and high.counts["broke the vertical rule"] == 6
    there = relabel(label, said, no, no, np.full(10, 300.0), one, words, window=5)
    assert there.asked[:8].all() and (there.classes[2:8, ALTITUDE] == low + 1).all()
    assert there.counts["broke the vertical rule"] == 0 and not there.asked[8:, ALTITUDE].any()


def test_the_round_kept_is_the_earliest_within_the_tie_of_the_best():
    assert choose([0.80, 0.90, 0.895, 0.91]) == 1
    assert choose([0.80, 0.85, 0.90]) == 2
    assert choose([0.90, 0.80]) == 0


# ---- branches
def _stepped(inputs, runways, charts, approach, params, words, limit, rows, swaps=()):
    """Fly per-flight step rows (``rows[b][step]``), taking ``[1, 0]`` at each step in ``swaps``."""
    executor = Executor(inputs, runways, charts, approach, params, words, time_limit_s=limit)
    spoken = Spoken(len(rows), words, device=CPU)
    order = list(range(len(rows)))
    for step in range(executor.cycles // executor.step_rows + 1):
        if step in swaps:
            executor.take(torch.tensor([1, 0]))
            spoken.take(torch.tensor([1, 0]))
            order.reverse()
        spoken.say(np.stack([rows[b][step] for b in order]))
        for _ in range(executor.step_rows):
            if executor.count < executor.cycles:
                executor.cycle(spoken.at(torch.full((len(rows),), step * words.spec.step_s, dtype=F64)),
                               torch.full((len(rows),), executor.count * params.cycle_s, dtype=F64))
    return executor


def _two(inputs, runways, charts, approach):
    index = torch.tensor([0, 0])
    return inputs.take(index), runways.take(index), charts.take(index), approach[index]


def test_a_taken_branch_flies_on_as_the_flight_it_was_taken_from():
    """Two flights told different words, swapped (`Executor.take`, `Spoken.take`) three times — before the turn, in it
    and on the final — end state for state as the two flown straight through, in the swapped order."""
    one, geometry = spec(), instruction_airport()
    words, params = Words(one), _params()
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    grid = read_flight(signals, geometry, one, words).words
    other = grid.copy()
    other[3, HEADING] = (grid[0, HEADING] + 3) % words.n_heading
    other[3, SPEED] = max(grid[0, SPEED] - 4, 0)
    steps = int(len(grid) * params.timeout_factor) + 2
    rest = np.full((steps - len(grid), 6), UNCHANGED, dtype=np.int64)
    rows = [np.concatenate((grid, rest)), np.concatenate((other, rest))]
    physics = _two(*_physics(signals, geometry))
    limit = torch.full((2,), len(grid) * one.step_s * params.timeout_factor, dtype=F64)
    straight = _stepped(*physics, params, words, limit, rows).flown()
    swapped = _stepped(*physics, params, words, limit, rows, swaps=(10, 60, 110)).flown()
    back = torch.tensor([1, 0])
    assert not torch.equal(straight.states[0], straight.states[1])          # the two flights differ
    assert torch.equal(swapped.states, straight.states[back])               # three swaps: the order ends reversed
    assert torch.equal(swapped.commands, straight.commands[back])
    assert torch.equal(swapped.done_cycle, straight.done_cycle[back])
    for name in straight.modes:
        assert torch.equal(swapped.modes[name], straight.modes[name][back]), name


def test_take_names_every_per_flight_field():
    """A per-flight tensor added to the executor, a law or the speaker without being named in its `PER_FLIGHT` would
    stay behind when a branch is taken."""
    from ts_transformer.prior.generate import Speaker

    one, geometry = spec(), instruction_airport()
    words, params = Words(one), _params()
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    inputs, runways, charts, approach = _physics(signals, geometry)
    index = torch.tensor([0, 0, 0])
    executor = Executor(inputs.take(index), runways.take(index), charts.take(index), approach[index], params, words,
                        time_limit_s=torch.full((3,), 60.0, dtype=F64))
    spoken = Spoken(3, words, device=CPU)
    spoken.say(np.stack([read_flight(signals, geometry, one, words).words[0]] * 3))
    executor.cycle(spoken.at(torch.zeros(3, dtype=F64)), torch.zeros(3, dtype=F64))
    model = _model(words)
    speaker = Speaker(model, [signals] * 3, [geometry] * 3, None, words, max_rows=40,
                      generator=torch.Generator().manual_seed(0))
    for holder in (executor, executor.lateral, executor.vertical, executor.speed, speaker):
        per_flight = {name for name, value in vars(holder).items()
                      if (isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim and len(value) == 3)
                      or (isinstance(value, list) and len(value) == 3)}
        assert per_flight <= set(type(holder).PER_FLIGHT), (type(holder).__name__, per_flight - set(holder.PER_FLIGHT))
    # the executor's records — a list, or a dict of lists, of per-cycle [3, …] rows — are its HISTORIES; its context
    # tables are its CONTEXT
    def per_cycle(value):
        rows = [v for row in value.values() for v in row] if isinstance(value, dict) else value
        return isinstance(value, (list, dict)) and bool(rows) and all(
            isinstance(v, torch.Tensor) and len(v) == 3 for v in rows)
    assert {name for name, value in vars(executor).items() if per_cycle(value)} == set(Executor.HISTORIES)
    assert {name for name, value in vars(executor).items() if hasattr(value, "take")
            and not isinstance(value, torch.Tensor)} - {"lateral", "vertical", "speed"} == set(Executor.CONTEXT)
    before = {name: np.array(getattr(speaker, name)) for name in ("e", "n", "h", "value", "said_row")}
    speaker.take(np.array([2, 0]))
    for name, value in before.items():
        assert np.array_equal(getattr(speaker, name), value[[2, 0]]), name
    assert speaker.features.shape[0] == 2 and len(speaker.geometries) == 2


def _model(words, slots=1, variant="no-context"):
    torch.manual_seed(0)
    return Prior(PriorConfig(classes=column_classes(words, slots), airports=("KXXX",), candidate_slots=slots,
                             variant=variant, d_model=32, layers=2, heads=4, feedforward=64, dropout=0.0),
                 torch.as_tensor(prior_data.candidate_table({"KXXX": instruction_airport()}, ("KXXX",), slots))).eval()


def _flight():
    one, geometry = spec(), instruction_airport()
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    start = replace(signals, **{name: getattr(signals, name)[N_LOOK:] for name in
                                ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps",
                                 "vertical_rate_mps")})
    return one, geometry, signals, _physics(start, geometry)


def test_a_chain_is_flown_by_its_own_words_and_one_branch_is_free_generation():
    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words, params = Words(one), _params()
    model = _model(words)
    observed = np.column_stack((signals.e_m, signals.n_m, signals.altitude_m))
    chains = fly_chains(model, [signals], [geometry], inputs, runways, charts, approach, [80.0], [observed], words,
                        params, None, ChainConfig(branches=3, segment_steps=3),
                        generator=torch.Generator().manual_seed(5))
    said, position = chains.said[0], chains.positions[0]
    assert (said[0] != UNCHANGED).all() and position.shape == (N_LOOK + len(said), 3)
    assert chains.outcomes[0] in OUTCOMES and len(chains.kept_m[0]) == -(-len(said) // 3)   # while it flew
    assert np.allclose(position[: N_LOOK + 1], observed[: N_LOOK + 1])
    # the kept branches, flown again from the start by their own words alone, pass through the chain's rows
    executor = Executor(inputs, runways, charts, approach, params, words,
                        time_limit_s=torch.tensor([80.0], dtype=F64))
    spoken = Spoken(1, words, device=CPU)
    for step, row in enumerate(said):
        if step:
            now = read_state(executor.state, charts)
            assert float(now.e_m[0]) == pytest.approx(position[N_LOOK + step, 0], abs=1e-6)
            assert float(now.height_m[0]) == pytest.approx(position[N_LOOK + step, 2], abs=1e-6)
        spoken.say(row[None])
        for _ in range(executor.step_rows):
            if executor.count < executor.cycles and not bool(executor.done.all()):
                executor.cycle(spoken.at(torch.tensor([step * one.step_s], dtype=F64)),
                               torch.tensor([executor.count * params.cycle_s], dtype=F64))
    # one branch is free generation, word for word
    one_branch = fly_chains(model, [signals], [geometry], inputs, runways, charts, approach, [80.0], [observed], words,
                            params, None, ChainConfig(branches=1, segment_steps=3),
                            generator=torch.Generator().manual_seed(5))
    _, free, _, _ = speak_and_fly(model, [signals], [geometry], inputs, runways, charts, approach, [80.0], words, params,
                                  None, generator=torch.Generator().manual_seed(5), temperature=1.0)
    assert np.array_equal(one_branch.said[0], free[0, : len(one_branch.said[0])])


def test_a_chain_s_training_rows_are_the_rows_the_prior_read_on_it():
    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words, params = Words(one), _params()
    model = _model(words)
    _, said, _, speaker = speak_and_fly(model, [signals], [geometry], inputs, runways, charts, approach, [60.0], words,
                                        params, None, generator=torch.Generator().manual_seed(2), temperature=1.0)
    said = said[0]
    rows = N_LOOK + len(said)
    classes = np.ones((len(said), 6), dtype=np.int64)
    asked = np.ones((len(said), 6), dtype=bool)
    asked[3, ALTITUDE] = False
    flight = chain_record(signals, speaker.e[0, :rows], speaker.n[0, :rows], speaker.h[0, :rows], said, classes, asked,
                          geometry, None, 0, 0, one.step_s)
    assert np.allclose(flight.features, speaker.features[0, 0, :rows].numpy(), atol=1e-6)
    assert np.allclose(flight.relative, speaker.relative[0, 0, :rows].numpy(), atol=1e-6)
    assert np.array_equal(flight.in_force, speaker.in_force[0, 0, :rows].numpy())
    assert np.allclose(flight.since, speaker.since[0, 0, :rows].numpy(), atol=1e-6)
    assert (flight.targets[:N_LOOK] == 0).all() and (flight.targets[N_LOOK:] == 1).all()
    assert not flight.asked[:N_LOOK].any() and not flight.asked[N_LOOK + 3, ALTITUDE]
    assert flight.asked[N_LOOK:].sum() == 6 * len(said) - 1


def test_one_pass_counts_only_the_asked_columns():
    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words, params = Words(one), _params()
    _, said, _, speaker = speak_and_fly(_model(words), [signals], [geometry], inputs, runways, charts, approach, [40.0],
                                        words, params, None, generator=torch.Generator().manual_seed(2),
                                        temperature=1.0)
    said = said[0]
    rows = N_LOOK + len(said)
    asked = np.ones((len(said), 6), dtype=bool)
    asked[:, SPEED] = False

    def flight(speed_class: int):
        classes = np.ones((len(said), 6), dtype=np.int64)
        classes[:, SPEED] = speed_class
        return chain_record(signals, speaker.e[0, :rows], speaker.n[0, :rows], speaker.h[0, :rows], said, classes,
                            asked, geometry, None, 0, 0, one.step_s)

    table = prior_data.candidate_table({"KXXX": geometry}, ("KXXX",), 1)
    splits = [Split([flight(c)], ("KXXX",), table, (("09",),), ((90.0,),), column_classes(words, 1), "no-context")
              for c in (0, 3)]
    model = _model(words)
    batches = [to_batch(s, [0], CPU) for s in splits]
    nll = [column_nll(model(b["features"], b["relative"], b["static"], b["in_force"], b["since"], b["airport"],
                            b["present"], b["edges"], b["targets"]), b["targets"], b["present"], b["asked"])
           for b in batches]
    assert float(nll[0][SPEED].detach()) == 0.0 and torch.allclose(nll[0][:SPEED], nll[1][:SPEED])
    states = []
    for s in splits:
        model = _model(words)
        FineTuner(model, FineTuneConfig(warmup_steps=1), CPU, seed=0).one_pass(s)
        states.append(model.state_dict())
    for name, value in states[0].items():
        assert torch.equal(value, states[1][name]), name


def test_rows_encoded_one_at_a_time_are_the_rows_encoded_together():
    """`Prior.extend` (a speaker's row by row, each layer's keys and values kept) gives what `encode` gives, absent
    rows included."""
    from ts_transformer.prior.model import self_edges

    words = Words(spec())
    torch.manual_seed(3)
    model = _model(words, slots=2, variant="full")
    batch, rows = 3, N_LOOK + 6
    features = torch.randn(batch, 1, rows, len(prior_data.STEP_FEATURES))
    relative = torch.randn(batch, 1, rows, 2, len(prior_data.VARIANTS["full"].relative_features))
    in_force = torch.randint(0, 3, (batch, 1, rows, 6))
    since = torch.rand(batch, 1, rows, 6)
    airport, static = torch.zeros(batch, dtype=torch.long), torch.zeros(batch, 1, 0)
    present = torch.ones(batch, 1, rows, dtype=torch.bool)
    present[1, 0, 3] = False
    whole, tokens, valid = model.encode(features, relative, static, in_force, since, airport, present,
                                        self_edges(batch, 1, rows, CPU))
    past, pieces = model.no_past(batch, rows), []
    for low, high in ((0, N_LOOK + 1), *((t, t + 1) for t in range(N_LOOK + 1, rows))):
        h, piece_tokens, _, past = model.extend(features[:, :, low:high], relative[:, :, low:high], static,
                                                in_force[:, :, low:high], since[:, :, low:high], airport,
                                                present[:, :, low:high], self_edges(batch, 1, high - low, CPU), past)
        pieces.append(h)
        assert torch.allclose(piece_tokens, tokens[:, :, low:high])
    assert torch.allclose(torch.cat(pieces, dim=2), whole, atol=1e-5)
    taken = [p.take(torch.tensor([2, 0])) for p in past]
    assert torch.equal(taken[0].keys, past[0].keys[[2, 0]]) and taken[0].present.shape == (2, rows)
    with pytest.raises(ValueError, match="the past holds"):
        model.extend(features[:, :, :1], relative[:, :, :1], static, in_force[:, :, :1], since[:, :, :1], airport,
                     present[:, :, :1], self_edges(batch, 1, 1, CPU), past)


def test_the_cached_rows_of_taken_branches_are_the_rows_they_read():
    """A closed loop of four copies of a flight, re-formed mid-flight (`ClosedLoop.take`): every copy's cached keys are
    the keys a fresh encode of the rows it read gives, and its words and states are its source's."""
    from ts_transformer.experiments.prior_free_generation import ClosedLoop
    from ts_transformer.prior.model import self_edges

    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words, params = Words(one), _params()
    model = _model(words)
    four = torch.tensor([0, 0, 0, 0])
    loop = ClosedLoop(model, [signals] * 4, [geometry] * 4, inputs.take(four), runways.take(four), charts.take(four),
                      approach[four], [60.0] * 4, words, params, None, generator=torch.Generator().manual_seed(4),
                      temperature=1.0)
    for _ in range(7):
        loop.step()
    before = {"said": loop.spoken.sentences(), "state": loop.executor.state.clone(), "e": loop.speaker.e.copy()}
    index = np.array([3, 3, 0, 1])
    loop.take(index)
    assert np.array_equal(loop.spoken.sentences(), before["said"][index])
    assert torch.equal(loop.executor.state, before["state"][index]) and np.array_equal(loop.speaker.e, before["e"][index])
    for _ in range(5):
        loop.step()
    speaker, rows = loop.speaker, loop.speaker.encoded
    fresh = model.extend(speaker.features[:, :, :rows], speaker.relative[:, :, :rows], speaker.static,
                         speaker.in_force[:, :, :rows], speaker.since[:, :, :rows], speaker.airport,
                         torch.ones((4, 1, rows), dtype=torch.bool), self_edges(4, 1, rows, CPU),
                         model.no_past(4, rows))[3]
    for kept, again in zip(speaker.past, fresh):
        assert kept.rows == again.rows == rows
        assert torch.allclose(kept.keys[:, :, :rows], again.keys, atol=1e-5)
        assert torch.allclose(kept.values[:, :, :rows], again.values, atol=1e-5)


def test_a_round_s_chains_become_flights_a_summary_and_a_file(tmp_path):
    """`chain_flights`, `chain_summary` and `write_chains` on two chains of one flight: the flights are the rows the
    chain read with the relabelled targets, the summary is JSON, the file reads back."""
    import json

    from ts_transformer.autopilot import replay
    from ts_transformer.experiments.prior_closed_loop import chain_flights, chain_summary, write_chains

    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words, params = Words(one), _params()
    reading = read_flight(signals, geometry, one, words)
    observed = np.column_stack((signals.e_m, signals.n_m, signals.altitude_m))[: len(reading.words)]
    two = torch.tensor([0, 0])
    chains = fly_chains(_model(words), [signals] * 2, [geometry] * 2, inputs.take(two), runways.take(two),
                        charts.take(two), approach[two], [60.0] * 2, [observed] * 2, words, params, None,
                        ChainConfig(branches=2, segment_steps=4), generator=torch.Generator().manual_seed(6))
    batch = replay.Batch(signals=[signals] * 2, series=[], readings=[reading] * 2, geometries=[geometry] * 2,
                         crossing_heights=[], approach_ias_mps=[], groups=[], drawn={})
    flights, targets, counts = chain_flights(batch, chains, ("KXXX",), words, None, 5)
    for flight, chain, target in zip(flights, chains.said, targets):
        assert flight.rows == N_LOOK + len(chain) and np.array_equal(flight.targets[N_LOOK:], target.classes)
    assert counts["column_steps_after_the_first"] == sum(6 * (len(s) - 1) for s in chains.said)
    summary = chain_summary(chains, counts, 4)
    json.dumps(summary)
    assert summary["flights"] == 2 and 0.0 <= summary["segments_parted"] <= 1.0
    write_chains(tmp_path / "chains.npz", batch, chains, targets)
    stored = np.load(tmp_path / "chains.npz")
    assert stored["step_offsets"][-1] == len(stored["said"]) == sum(len(s) for s in chains.said)
    assert np.array_equal(stored["classes"], np.concatenate([t.classes for t in targets]))
