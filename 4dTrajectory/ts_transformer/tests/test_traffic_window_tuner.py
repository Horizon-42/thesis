"""The multi-aircraft post-training in windows (`experiments/traffic_window_tuner`, multi-aircraft design §6.6 step 7,
the 7.6 plan): a window's words scored whole give back the distributions they were sampled from — two commanded aircraft
in the air together, replayed ones, a traffic attention that reads the others, more candidate slots than the airport has,
a window scored with others of another pre-roll or on its own — an aircraft whose flight ends inside a step keeps the
words it said there, and each aircraft's rows are rebuilt with the landing context each was encoded with. On the
scene-data fixture."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.instructions.words import Words
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.test_traffic_window import STEP_S, _airport, _keys, _physics_of, _pool

CPU = torch.device("cpu")


def _reading_model(spec, variant="no-context", slots=2):
    """A traffic prior whose traffic attention reads the others (its output layer off zero), with ``slots`` candidate
    slots (the fixture's airport has one candidate: a pooled model has as many as its largest airport)."""
    from ts_transformer.inference.scene_edges import EDGE_FEATURES
    from ts_transformer.prior.model import with_traffic
    from ts_transformer.tests.test_prior_speaker import _model

    torch.manual_seed(1)
    model = with_traffic(_model(Words(spec), slots=slots, variant=variant), EDGE_FEATURES).eval()
    for layer in model.layers:
        torch.nn.init.normal_(layer.traffic.out.weight, std=0.2)
    return model


def _loop(model, airport, signals, spec, commanded, limits, landings=None, seed=4):
    from ts_transformer.experiments.traffic_window import WindowLoop, window_of
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params

    windows = [window_of(airport, 0.0, keys, [limit] * len(keys), STEP_S) for keys, limit in zip(commanded, limits)]
    flights = [signals[k] for window in windows for k in window.commanded]
    geometry = airport.flights.geometry
    per_aircraft = [limit for keys, limit in zip(commanded, limits) for _ in keys]
    return WindowLoop(model, windows, flights, [geometry] * len(flights), *_physics_of(flights, geometry), per_aircraft,
                      Words(spec), _params(), landings, generator=torch.Generator().manual_seed(seed), temperature=1.0,
                      procedure_masks=ProcedureMasks.none())


def _flights(loop, landings, spec, records):
    from ts_transformer.experiments.traffic_window_tuner import window_flight

    return [window_flight(r, loop.windows[r.window], loop.flights[i], loop.geometries[i], landings, 0, 0, len(r.grid),
                          spec.step_s) for i, r in enumerate(records)]


def _run_spied(loop, model, monkeypatch):
    """The loop run to its end, each round's draw recorded: ``(now, row, the six columns' logits, chosen)``."""
    from ts_transformer.prior.window_speaker import WindowSpeaker

    calls, sampled = [], []
    logits, sample = model.logits, WindowSpeaker._sample

    def spy_logits(*args):
        out = logits(*args)
        calls.append([logit[:, 0, 0].detach().clone() for logit in out])
        return out

    def spy_sample(speaker, h, tokens, valid, now, opening, row, runway_locked, forced):
        first = len(calls)
        chosen = sample(speaker, h, tokens, valid, now, opening, row, runway_locked, forced)
        sampled.append((now.copy(), row.copy(), calls[first: first + 6], chosen.copy()))
        return chosen

    model.logits = spy_logits
    monkeypatch.setattr(WindowSpeaker, "_sample", spy_sample)
    try:
        while loop.running:
            loop.step()
    finally:
        del model.logits
    return sampled


def _same(sampled, got, index=None):
    """Every word sampled (of the aircraft in ``index``, the batch's place → the scored place; all by default) scored as
    its distribution was; the (aircraft, row) pairs compared."""
    compared = set()
    for now, row, columns, _ in sampled:
        for i in np.flatnonzero(now):
            if index is not None and int(i) not in index:
                continue
            j = int(i) if index is None else index[int(i)]
            step = int(row[i])
            for c in range(6):
                want, have = columns[c][c][i], got[c][j, 0, step]
                finite = torch.isfinite(want)
                assert torch.equal(finite, torch.isfinite(have)), (i, step, c)
                assert torch.allclose(have[finite], want[finite].to(have.dtype), atol=1e-4), (i, step, c)
            compared.add((int(i), step))
    return compared


def _in_four_blocks(monkeypatch):
    """Every encoding in four blocks of steps (`prior.model.step_blocks` asked for the pairs that give four): the blocks
    each encoding ran in, recorded."""
    from ts_transformer.prior import model as prior_model

    whole, ran = prior_model.step_blocks, []

    def four(batch, aircraft, steps, pairs):
        blocks = whole(batch, aircraft, steps, batch * aircraft * aircraft * -(-steps // 4))
        ran.append(len(blocks))
        return blocks

    monkeypatch.setattr(prior_model, "step_blocks", four)
    return ran


def test_a_window_scored_whole_gives_back_what_each_of_its_words_was_sampled_from(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_window import window_places
    from ts_transformer.experiments.traffic_window_tuner import window_layout, window_logits
    from ts_transformer.instructions.artefact import load_signals

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _reading_model(spec)
    # f3 and f4 (100 steps apart) commanded long enough to be in the air together, f1, f2 and f5 replayed around them;
    # f0 alone at the start of the segment, needing no pre-roll: the batch's is f3's window's
    loop = _loop(model, airport, signals, spec, [_keys(3, 4), _keys(0)], [240.0, 40.0])
    assert loop.windows[0].others and window_places(loop.windows[1:], STEP_S).pre < loop.pre
    sampled = _run_spied(loop, model, monkeypatch)
    rows = loop.speaker.rows
    assert loop.start[0] + rows[0] > loop.start[1] + N_LOOK + 10          # f3 still flying when f4 speaks
    records = loop.records()
    flights = _flights(loop, None, spec, records)
    with torch.no_grad():
        got = window_logits(model, window_layout(model, loop.windows, flights, records, spec.step_s, CPU))
        alone = window_logits(model, window_layout(model, loop.windows[1:], flights[2:], records[2:], spec.step_s, CPU))
    compared = _same(sampled, got)
    assert {i for i, _ in compared} == {0, 1, 2} and len(compared) > 150
    # f0's window scored on its own — another pre-roll — gives back the same
    assert len(_same(sampled, alone, index={2: 0})) > 10
    # in blocks of steps (two windows, replayed aircraft, absent steps, each aircraft at its own rows): the same
    ran = _in_four_blocks(monkeypatch)
    with torch.no_grad():
        blocked = window_logits(model, window_layout(model, loop.windows, flights, records, spec.step_s, CPU))
    assert ran == [4]
    for x, y in zip(blocked, got):
        assert torch.equal(torch.isinf(x), torch.isinf(y))
        torch.testing.assert_close(x[torch.isfinite(y)], y[torch.isfinite(y)], rtol=0, atol=1e-5)
    # the others did change the words: the same layout read with the traffic attention at zero
    for layer in model.layers:
        torch.nn.init.zeros_(layer.traffic.out.weight)
    with torch.no_grad():
        deaf = window_logits(model, window_layout(model, loop.windows, flights, records, spec.step_s, CPU))
    finite = torch.isfinite(got[3][:2])
    assert float((got[3][:2][finite] - deaf[3][:2][finite]).abs().max()) > 1e-2


def test_an_aircraft_whose_flight_ends_inside_a_step_keeps_the_words_it_said_there(tmp_path, monkeypatch):
    """Own ends inside a step (not a time limit) — f3 at its sixth step, f4 at its first — forced through the loop's
    crossing check: each one's last row holds the words it said there, as the results count them, and scores back."""
    from ts_transformer.experiments import traffic_window
    from ts_transformer.experiments.traffic_window_tuner import window_layout, window_logits
    from ts_transformer.instructions.artefact import load_signals

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _reading_model(spec)
    loop = _loop(model, airport, signals, spec, [_keys(3, 4)], [240.0])
    ends = {0: 5, 1: 0}                                        # aircraft → the step its flight ends in
    crossed, outcome_of = traffic_window.WindowLoop._crossed, traffic_window.outcome_of
    asked: dict[str, int] = {}

    def forced_crossed(self, index, own):
        out = crossed(self, index, own)
        for p, i in enumerate(index.tolist()):
            asked[i] = int(own[p])                               # each aircraft's own step
            if i in ends and own[p] == ends[i]:
                out[p] = True
        return out

    def forced_outcome(flown, place, geometry, runway, words_spec):
        for i, end in ends.items():
            if asked.get(i) == end and place == int(loop.place[i]) and not loop.left[i]:
                return SimpleNamespace(outcome="ground_contact", end_row=end * int(round(STEP_S / flown.cycle_s)) + 1,
                                       crossing=None)
        return outcome_of(flown, place, geometry, runway, words_spec)

    monkeypatch.setattr(traffic_window.WindowLoop, "_crossed", forced_crossed)
    monkeypatch.setattr(traffic_window, "outcome_of", forced_outcome)
    sampled = _run_spied(loop, model, monkeypatch)
    records, results = loop.records(), loop.results()
    for i, end in ends.items():
        assert results[i].own == "ground_contact" and len(records[i].grid) == end + 1
        last = [chosen[i] for now, row, _, chosen in sampled if now[i] and row[i] == N_LOOK + end]
        assert last and np.array_equal(records[i].grid[-1], np.where(last[0] > 0, last[0] - 1, -1))
        assert np.array_equal(records[i].grid[: results[i].counted], results[i].said[: results[i].counted])
    flights = _flights(loop, None, spec, records)
    with torch.no_grad():
        got = window_logits(model, window_layout(model, loop.windows, flights, records, spec.step_s, CPU))
    assert {(i, N_LOOK + end) for i, end in ends.items()} <= _same(sampled, got)


def test_each_row_is_rebuilt_with_the_landing_context_it_was_encoded_with(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_window_tuner import window_flight
    from ts_transformer.instructions.artefact import load_signals

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _reading_model(spec, variant="full")
    landings = {"KXXX": _pool(airport)}
    loop = _loop(model, airport, signals, spec, [_keys(0, 1)], [400.0], landings)          # 200 s apart
    speaker = loop.speaker
    # a landing in the loop before f1 is in the scene (it reads it from its row 0), then one after it is (a second
    # before the last row encoded, as a landing found at a step comes after its row), and the loop runs on
    for _ in range(3):
        loop.step()
    early = loop.window_time_s(0, speaker.steps_encoded - 1) - 1.0
    loop.landing_s[0] = early
    loop._landed(0, 0)
    while speaker.step < int(loop.start[1]) + N_LOOK + 6:
        loop.step()
    encoded = speaker.steps_encoded
    landing_s = loop.window_time_s(0, encoded - 1) - 1.0
    loop.landing_s[0] = landing_s
    loop._landed(0, 0)
    for _ in range(4):
        loop.step()
    records = loop.records()
    switch = encoded - int(loop.start[1])
    assert not records[0].context_changes
    assert records[1].context_changes == [(0, early, "09"), (switch, landing_s, "09")]
    flights = _flights(loop, landings, spec, records)
    for i, flight in enumerate(flights):
        rows = int(speaker.rows[i])
        assert flight.rows == rows
        assert torch.equal(torch.as_tensor(flight.features), speaker.features[i, :rows]), i
        assert torch.equal(torch.as_tensor(flight.relative), speaker.relative[i, :rows, : flight.relative.shape[1]]), i
        assert torch.equal(torch.as_tensor(flight.in_force, dtype=torch.long), speaker.in_force[i, :rows]), i
        assert torch.allclose(torch.as_tensor(flight.since), speaker.since[i, :rows]), i
    # read with both landings from row 0, f1's last row encoded before the second would read it: the switch row matters
    whole = window_flight(dataclasses.replace(records[1], context_changes=[(0, early, "09"), (0, landing_s, "09")]),
                          loop.windows[0], loop.flights[1], loop.geometries[1], landings, 0, 0, len(records[1].grid),
                          spec.step_s)
    before = switch - 1
    width = whole.relative.shape[1]
    assert not torch.equal(torch.as_tensor(whole.relative[before]), speaker.relative[1, before, :width])
    assert torch.equal(torch.as_tensor(whole.relative[switch:]), speaker.relative[1, switch: flights[1].rows, :width])


# ---- 7.6.2: a round's window sentences

def test_window_sentences_are_the_readout_s_rows_with_what_each_aircraft_read_and_the_masks_it_said_under(tmp_path,
                                                                                                         monkeypatch):
    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params
    from ts_transformer.tests.test_traffic_speaking import _batch
    from ts_transformer.tests.test_traffic_window import LIMIT_S, _as_drawn, _patch_runner_physics

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0, 3_600.0], (0, 1, 2))
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    commanded = [("KXXX:f0", "KXXX:f1"), ("KXXX:f2",)]
    keys = [k for keys in commanded for k in keys]
    windows = [window_of(airport, 0.0, c, [LIMIT_S] * len(c), STEP_S) for c in commanded]
    drawn = _as_drawn(windows, [range(0, 2), range(2, 3)], _batch(airport, signals, spec, keys), [LIMIT_S] * 3)
    _patch_runner_physics(monkeypatch, signals, airport.flights.geometry, keys)
    every = {"KXXX": _pool(airport)}
    model = _reading_model(spec)

    def spoken(read):
        return read(model, drawn, [0, 1], "scene", Words(spec), _params(), None, every, 2,
                    generator=torch.Generator().manual_seed(4), temperature=1.0, procedure_masks=ProcedureMasks.none())

    got = spoken(runner.window_sentences)
    assert got.rows == spoken(runner.model_rows)
    assert [r.key for r in got.records] == [row["dataset_id"] for row in got.rows]
    masked = next(iter(got.allowed[0]))
    for row, record, allowed in zip(got.rows, got.records, got.allowed):
        assert len(record.grid) >= row["counted"] and len(record.e) == N_LOOK + len(record.grid)
        assert all(len(bits) == row["counted"] for bits in allowed.values()) and masked in allowed


def _advantage_row(window, key, sample, reward, lost=False, probed=False, forced=None):
    return {"window": window, "dataset_id": key, "sample": sample, "reward": reward, "starts_in_a_loss": lost,
            "probed": probed, "forced": forced, "given": False}


def test_an_aircraft_s_advantage_is_against_its_own_samples_and_it_trains_only_on_a_contrast():
    from ts_transformer.experiments.traffic_window_tuner import window_advantages

    row = _advantage_row
    rows = [row(0, "a", 0, 1.0), row(0, "b", 0, 1.0), row(0, "c", 0, 1.0, lost=True),     # window 0, sample 0
            row(0, "a", 1, 0.0), row(0, "b", 1, 1.0), row(0, "c", 1, 0.0),                # window 0, sample 1
            row(1, "a", 0, 0.0), row(1, "a", 1, 1.0)]                                     # "a" again, in another window
    advantages, gains, trained = window_advantages(rows, 2)
    assert advantages.tolist() == [0.5, 0.0, 0.5, -0.5, 0.0, -0.5, -0.5, 0.5]
    assert not gains.any()                                                                # no probe
    # b has no contrast; c's first sample started in a loss it answers for: neither is trained on
    assert trained.tolist() == [0, 3, 6, 7]
    # two draws' rows mixed up (window 1's "a" twice as sample 1) would group 2K rows: refused
    with pytest.raises(ValueError, match="not 0 … 1 once each"):
        window_advantages(rows + [row(1, "a", 1, 0.0)], 2)


def test_probes_are_weighed_among_themselves_and_their_go_around_against_the_unprobed_samples():
    """Multi-aircraft design §6.6 step 8 item 10 (readout 2026-10-02 §3-§4): an aircraft's unprobed samples, its probes
    that said no go-around and its probes that said one are three groups, each against its own mean (a go-around's
    sentence never against one without); a probe's go-around word is learned by how much better its sentence did than
    the unprobed samples' mean, and its sentence is trained on for that alone where its group has no contrast."""
    from ts_transformer.experiments.traffic_window_tuner import window_advantages

    row = _advantage_row
    rows = [row(0, "a", 0, 1.0), row(0, "a", 1, 0.0),                                    # a: unprobed 1, 0
            row(0, "a", 2, 0.9, probed=True, forced=5), row(0, "a", 3, 0.1, probed=True, forced=7),
            row(0, "b", 0, 0.0), row(0, "b", 1, 0.0),                                    # b: unprobed 0, 0
            row(0, "b", 2, 0.6, probed=True, forced=4), row(0, "b", 3, 0.0, probed=True),  # one fired, one did not
            row(0, "c", 0, 1.0), row(0, "c", 1, 1.0),                                    # c: no contrast anywhere
            row(0, "c", 2, 0.5, probed=True, forced=3), row(0, "c", 3, 0.5, probed=True, forced=3),
            row(0, "f", 0, 1.0), row(0, "f", 1, 1.0),                                    # f: no probe fired
            row(0, "f", 2, 1.0, probed=True), row(0, "f", 3, 0.0, probed=True)]
    advantages, gains, trained = window_advantages(rows, 4)
    assert advantages == pytest.approx([0.5, -0.5, 0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.5, -0.5])
    assert gains == pytest.approx([0.0, 0.0, 0.4, -0.4, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
    # a: both groups differ; b: its go-around (alone in its group) did better than the unprobed samples; c: nothing
    # differs and no go-around did better; f: its probes that fired no go-around differ among themselves
    assert trained.tolist() == [0, 1, 2, 3, 6, 14, 15]
    # a go-around that beat the unprobed samples is trained for its word even where the probes tie
    tie = [row(0, "d", 0, 0.0), row(0, "d", 1, 0.0), row(0, "d", 2, 0.5, probed=True, forced=2),
           row(0, "d", 3, 0.5, probed=True, forced=6)]
    advantages, gains, trained = window_advantages(tie, 4)
    assert not advantages.any() and gains.tolist() == [0.0, 0.0, 0.5, 0.5] and trained.tolist() == [2, 3]
    # ... but not where the aircraft started in a loss it answers for
    _, _, trained = window_advantages([dict(r, starts_in_a_loss=r["sample"] == 0) for r in tie], 4)
    assert trained.tolist() == []
    # a probe's go-around with no unprobed sample to weigh it against: refused
    with pytest.raises(ValueError, match="no unprobed sample"):
        window_advantages([row(0, "e", 0, 0.0, probed=True, forced=1), row(0, "e", 1, 1.0, probed=True)], 2)


# ---- 7.6.3: the window tuner

def _one_commanded_round(tmp_path, monkeypatch):
    """Windows of one commanded aircraft each (f3 twice, with replayed ones around it; f7 twice), spoken by a reading
    model, as the window tuner and the scene tuner read them: ``(model, spec, window split, scene split, advantages,
    masks, data)``."""
    from ts_transformer.experiments.traffic_scene_data import build_split
    from ts_transformer.experiments.traffic_speaking import scene_of
    from ts_transformer.experiments.traffic_tuner import SceneSplit
    from ts_transformer.experiments.traffic_window_tuner import WindowSplit, window_flight
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.instructions.words import UNCHANGED
    from ts_transformer.prior import data as prior_data
    from ts_transformer.prior.data import Split, chain_record, column_classes

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _reading_model(spec)
    loop = _loop(model, airport, signals, spec, [_keys(3), _keys(3), _keys(7), _keys(7)], [60.0] * 4)
    while loop.running:
        loop.step()
    records, results = loop.records(), loop.results()
    counted = [r.counted for r in results]
    assert min(counted) > 5
    geometry = airport.flights.geometry
    words = Words(spec)
    table = Split([], ("KXXX",), prior_data.candidate_table({"KXXX": geometry}, ("KXXX",), 2), (("09",),), ((90.0,),),
                  column_classes(words, 2), "no-context")
    allowed = [{c: masks[i, : counted[i]].copy() for c, masks in loop.speaker.allowed.items()} for i in range(4)]
    advantages = np.array([0.5, -0.5, 0.25, -0.25])
    window_split = WindowSplit(
        table, list(loop.windows),
        [[window_flight(r, loop.windows[i], loop.flights[i], geometry, None, 0, 0, counted[i], spec.step_s)]
         for i, r in enumerate(records)],
        [[r] for r in records], [[0]] * 4, [advantages[i: i + 1] for i in range(4)], [[a] for a in allowed],
        [[-1]] * 4, [np.zeros(1)] * 4)
    scene_flights, positions = [], []
    for i, r in enumerate(records):
        said = r.grid[: counted[i]]
        rows = N_LOOK + counted[i]
        flight = chain_record(loop.flights[i], r.e[:rows], r.n[:rows], r.h[:rows], said,
                              np.where(said != UNCHANGED, said + 1, 0), np.ones(said.shape, dtype=bool), geometry, None,
                              0, 0, spec.step_s)
        scene_flights.append(flight)
        positions.append(np.column_stack((r.e[:rows], r.n[:rows], r.h[:rows])))
    scenes = [scene_of(airport, r.key, 60.0, spec.step_s) for r in records]
    assert all(scene.others == window.others for scene, window in zip(scenes, loop.windows))
    scene_split = SceneSplit(dataclasses.replace(table, flights=scene_flights), scenes, positions)
    data, _ = build_split(tmp_path / "artefact", "train", spec, ("KXXX",), None, 2_048)
    return model, spec, window_split, scene_split, advantages, allowed, data


def _gradients(tuner):
    """The tuner's optimiser steps, each step's gradients recorded."""
    seen, step = [], tuner.optimiser.step
    tuner.optimiser.step = lambda: (seen.append({name: p.grad.clone() for name, p in tuner.model.named_parameters()
                                                 if p.grad is not None}), step())[1]
    return seen


@pytest.mark.parametrize("tokens, passes", [(16_384, 1), (80, 2)])
def test_with_one_commanded_aircraft_a_window_trains_as_its_scene_does(tmp_path, monkeypatch, tokens, passes):
    """One update of every sentence, and — a batch a sentence (R32's cut: a sentence's rows to its counted steps) — four
    updates a pass over two passes, in the same order of batches: the same records and every update's gradients."""
    import copy

    from ts_transformer.experiments.traffic_tuner import SceneRewardTuner
    from ts_transformer.experiments.traffic_window_tuner import WindowRewardTuner
    from ts_transformer.prior.train import RewardConfig
    from ts_transformer.tests.test_prior_speaker import _model

    model, spec, window_split, scene_split, advantages, allowed, data = _one_commanded_round(tmp_path, monkeypatch)
    reference = _model(Words(spec), slots=2)
    records, gradients = [], []
    for kind in ("scene", "window"):
        tuner = (SceneRewardTuner if kind == "scene" else WindowRewardTuner)(
            copy.deepcopy(model), reference, RewardConfig(tokens_per_batch=tokens), CPU, seed=0,
            traffic_learning_rate=3e-4, step_s=spec.step_s)
        seen = _gradients(tuner)
        if kind == "scene":
            distance = tuner.distance(scene_split, allowed)
            records.append(tuner.one_pass(scene_split, advantages, data, allowed, slots=2, passes=passes))
        else:
            assert tuner.window_distance(window_split) == pytest.approx(distance, rel=1e-5)
            records.append(tuner.window_pass(window_split, data, slots=2, passes=passes))
        gradients.append(seen)
    updates = 1 if tokens > 1_000 else 4 * passes
    assert records[0]["batches"] == records[1]["batches"] == len(gradients[0]) == len(gradients[1]) == updates
    for name in ("reward_mean", "kl_mean", "data_mean"):         # (the reward term's mean sits near 0: float rounding)
        assert records[1][name] == pytest.approx(records[0][name], rel=1e-5, abs=1e-6), name
    assert np.allclose(records[1]["kl_trace"], records[0]["kl_trace"], rtol=1e-5, atol=1e-9)
    assert records[0]["sentences"] == records[1]["sentences"] == 4
    assert max(float(g.abs().max()) for name, g in gradients[0][0].items() if ".traffic." in name) > 0.0
    for scene, window in zip(*gradients):
        assert scene.keys() == window.keys()
        for name, g in scene.items():
            assert torch.allclose(window[name], g, rtol=1e-4, atol=1e-7), name


def test_window_samples_scored_in_parts_have_the_gradient_of_one_piece(tmp_path, monkeypatch):
    import copy

    from ts_transformer.experiments import traffic_window_tuner
    from ts_transformer.experiments.traffic_window_tuner import WindowRewardTuner
    from ts_transformer.prior.train import RewardConfig
    from ts_transformer.tests.test_prior_speaker import _model

    model, spec, window_split, _, _, _, data = _one_commanded_round(tmp_path, monkeypatch)
    gradients = []
    for budget in (10 ** 9, 1, "in blocks"):
        if budget == "in blocks":                  # one part of the four samples, each encoding in four blocks of steps
            ran = _in_four_blocks(monkeypatch)
            budget = 10 ** 9
        monkeypatch.setattr(traffic_window_tuner, "SCORE_BUDGET", budget)
        tuner = WindowRewardTuner(copy.deepcopy(model), _model(Words(spec), slots=2), RewardConfig(), CPU, seed=0,
                                  traffic_learning_rate=3e-4, step_s=spec.step_s)
        assert len(tuner._window_parts(window_split, [0, 1, 2, 3])) == (1 if budget > 1 else 4)
        seen = _gradients(tuner)
        tuner.window_pass(window_split, data, slots=2)
        gradients.append(seen[0])
    assert ran and set(ran) == {4}
    for name, g in gradients[0].items():
        assert torch.allclose(gradients[1][name], g, rtol=1e-4, atol=1e-7), name
        assert torch.allclose(gradients[2][name], g, rtol=1e-4, atol=1e-7), name


def test_an_aircraft_not_trained_on_is_read_by_the_others_and_adds_no_word(tmp_path, monkeypatch):
    """Two commanded aircraft in the air together, only the first trained on: its window sample counts one sentence,
    and the second's words move the first's gradient only through what the first reads."""
    import copy

    from ts_transformer.experiments.traffic_window_tuner import WindowRewardTuner, WindowSplit, window_flight
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior import data as prior_data
    from ts_transformer.prior.data import Split, column_classes
    from ts_transformer.prior.train import RewardConfig
    from ts_transformer.experiments.traffic_scene_data import build_split
    from ts_transformer.tests.test_prior_speaker import _model

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _reading_model(spec)
    loop = _loop(model, airport, signals, spec, [_keys(3, 4)], [240.0])
    while loop.running:
        loop.step()
    records, results = loop.records(), loop.results()
    geometry = airport.flights.geometry
    table = Split([], ("KXXX",), prior_data.candidate_table({"KXXX": geometry}, ("KXXX",), 2), (("09",),), ((90.0,),),
                  column_classes(Words(spec), 2), "no-context")
    data, _ = build_split(tmp_path / "artefact", "train", spec, ("KXXX",), None, 2_048)

    def split(trained):
        counted = [results[i].counted if i in trained else 0 for i in range(2)]
        return WindowSplit(table, list(loop.windows),
                           [[window_flight(r, loop.windows[0], loop.flights[i], geometry, None, 0, 0, counted[i],
                                           spec.step_s) for i, r in enumerate(records)]],
                           [records], [list(trained)], [np.full(len(trained), 0.5)],
                           [[{c: m[i, : counted[i]].copy() for c, m in loop.speaker.allowed.items()} for i in trained]],
                           [[-1] * len(trained)], [np.zeros(len(trained))])

    one = split([0])
    assert one.sentences == 1
    tuner = WindowRewardTuner(copy.deepcopy(model), _model(Words(spec), slots=2), RewardConfig(), CPU, seed=0,
                              traffic_learning_rate=3e-4, step_s=spec.step_s)
    record = tuner.window_pass(one, data, slots=2)
    assert record["sentences"] == 1 and record["batches"] == 1 and np.isfinite(record["reward_mean"])
    assert split([0, 1]).sentences == 2
    with pytest.raises(ValueError, match="one or more trained"):
        WindowSplit(table, list(loop.windows), one.flights, one.records, [[]], [np.zeros(0)], [[]], [[]],
                    [np.zeros(0)])



def _two_commanded(tmp_path, monkeypatch, samples=2):
    """Two commanded aircraft in the air together (f3, f4), ``samples`` window samples, the tuner's inputs:
    ``(model, spec, loop, records, results, table, data)``."""
    from ts_transformer.experiments.traffic_scene_data import build_split
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior import data as prior_data
    from ts_transformer.prior.data import Split, column_classes

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _reading_model(spec)
    loop = _loop(model, airport, signals, spec, [_keys(3, 4)] * samples, [240.0] * samples)
    while loop.running:
        loop.step()
    geometry = airport.flights.geometry
    table = Split([], ("KXXX",), prior_data.candidate_table({"KXXX": geometry}, ("KXXX",), 2), (("09",),), ((90.0,),),
                  column_classes(Words(spec), 2), "no-context")
    data, _ = build_split(tmp_path / "artefact", "train", spec, ("KXXX",), None, 2_048)
    return model, spec, loop, loop.records(), loop.results(), table, data


def _two_split(loop, records, results, table, spec, trained):
    """The loop's window samples trained on the aircraft at ``trained`` (places in each sample), advantage 0.5 each."""
    from ts_transformer.experiments.traffic_window_tuner import WindowSplit, window_flight

    windows, flights, read, places, gains, masks = [], [], [], [], [], []
    for w, window in enumerate(loop.windows):
        members = [i for i, r in enumerate(records) if r.window == w]
        counted = [results[i].counted if m in trained else 0 for m, i in enumerate(members)]
        windows.append(window)
        flights.append([window_flight(records[i], window, loop.flights[i], loop.geometries[i], None, 0, 0, c,
                                      spec.step_s) for i, c in zip(members, counted)])
        read.append([records[i] for i in members])
        places.append(list(trained))
        gains.append(np.full(len(trained), 0.5))
        masks.append([{c: m[members[k], : counted[k]].copy() for c, m in loop.speaker.allowed.items()}
                      for k in trained])
    return WindowSplit(table, windows, flights, read, places, gains, masks, [[-1] * len(trained)] * len(windows),
                       [np.zeros(len(trained))] * len(windows))


def test_a_window_s_trained_sentences_add_as_sentences_do(tmp_path, monkeypatch):
    """Two commanded aircraft in the air together, the gradient clip off: trained on both, one update's gradient is the
    mean of the two trained alone (each sentence its mean over its counted steps, averaged over the batch's sentences;
    the untrained one only read), the data term alike in each."""
    import copy

    from ts_transformer.experiments.traffic_window_tuner import WindowRewardTuner
    from ts_transformer.prior.train import RewardConfig
    from ts_transformer.tests.test_prior_speaker import _model

    model, spec, loop, records, results, table, data = _two_commanded(tmp_path, monkeypatch, samples=1)
    gradients = {}
    for trained in ((0, 1), (0,), (1,)):
        tuner = WindowRewardTuner(copy.deepcopy(model), _model(Words(spec), slots=2), RewardConfig(clip_norm=1e12), CPU,
                                  seed=0, traffic_learning_rate=3e-4, step_s=spec.step_s)
        seen = _gradients(tuner)
        record = tuner.window_pass(_two_split(loop, records, results, table, spec, trained), data, slots=2)
        assert record["sentences"] == len(trained) and len(seen) == 1
        gradients[trained] = seen[0]
    for name, both in gradients[(0, 1)].items():
        assert torch.allclose(2.0 * both, gradients[(0,)][name] + gradients[(1,)][name], rtol=1e-4, atol=1e-7), name


def test_a_later_aircraft_trained_is_found_in_its_part_however_the_samples_are_parted(tmp_path, monkeypatch):
    import copy

    from ts_transformer.experiments import traffic_window_tuner
    from ts_transformer.experiments.traffic_window_tuner import WindowRewardTuner
    from ts_transformer.prior.train import RewardConfig
    from ts_transformer.tests.test_prior_speaker import _model

    model, spec, loop, records, results, table, data = _two_commanded(tmp_path, monkeypatch, samples=2)
    split = _two_split(loop, records, results, table, spec, (1,))
    gradients = []
    for budget in (10 ** 9, 1):
        monkeypatch.setattr(traffic_window_tuner, "SCORE_BUDGET", budget)
        tuner = WindowRewardTuner(copy.deepcopy(model), _model(Words(spec), slots=2), RewardConfig(clip_norm=1e12), CPU,
                                  seed=0, traffic_learning_rate=3e-4, step_s=spec.step_s)
        assert len(tuner._window_parts(split, [0, 1])) == (1 if budget > 1 else 2)
        seen = _gradients(tuner)
        tuner.window_pass(split, data, slots=2)
        gradients.append(seen[0])
    for name, g in gradients[0].items():
        assert torch.allclose(gradients[1][name], g, rtol=1e-4, atol=1e-7), name


def test_window_samples_are_batched_as_sentences_are_by_their_counted_rows():
    from types import SimpleNamespace

    from ts_transformer.experiments.traffic_window_tuner import update_batches
    from ts_transformer.prior.data import batches

    def flight(counted, extra):
        """A trained sentence of ``counted`` steps flying ``extra`` rows on after them (read, not trained)."""
        asked = np.zeros((N_LOOK + counted + extra, 6), dtype=bool)
        asked[N_LOOK: N_LOOK + counted] = True
        return SimpleNamespace(asked=asked, rows=len(asked))

    counted = [30, 12, 25, 40, 18, 33, 7]
    split = SimpleNamespace(flights=[[flight(c, 200)] for c in counted], trained=[[0]] * len(counted))
    sentences = [SimpleNamespace(rows=N_LOOK + c) for c in counted]
    for rng in (None, 3):
        want = batches(sentences, 100, None if rng is None else np.random.default_rng(rng))
        got = update_batches(split, 100, None if rng is None else np.random.default_rng(rng))
        assert got == list(want)


def test_a_probes_go_around_is_left_out_of_the_ratio_and_learned_where_its_probe_gain_is_positive(tmp_path, monkeypatch):
    """Multi-aircraft design §6.6 step 8 item 10: the approach word a probe said for an aircraft is not asked (it did not
    sample it) and is learned by a cross-entropy of the imitation weight times its probe gain — only where that is above
    0 — never its advantage (`window_advantages`)."""
    import copy

    import torch

    from ts_transformer.experiments.traffic_window_tuner import WindowRewardTuner, WindowSplit
    from ts_transformer.instructions.words import APPROACH, APPROACH_GO_AROUND
    from ts_transformer.prior.train import RewardConfig
    from ts_transformer.tests.test_prior_speaker import _model

    model, spec, split, _, advantages, allowed, _ = _one_commanded_round(tmp_path, monkeypatch)
    step = 3
    rebuilt, masks = [], []
    for s in range(4):
        flight = split.flights[s][0]
        asked = flight.asked.copy()
        asked[N_LOOK + step, APPROACH] = False
        rebuilt.append([dataclasses.replace(flight, asked=asked)])
        # a probe says its go-around only where the masks allow it there
        allows = {c: m.copy() for c, m in split.allowed[s][0].items()}
        allows[APPROACH][step] = np.packbits(np.ones(model.config.classes[APPROACH], dtype=bool), bitorder="little")
        masks.append([allows])
    probe_gains = [-a for a in split.advantages]                     # of the other sign: the gain is what is read
    probed = WindowSplit(split.table, split.windows, rebuilt, split.records, split.trained, split.advantages, masks,
                         [[step]] * 4, probe_gains)
    with pytest.raises(ValueError, match="a probe's go-around"):
        WindowSplit(split.table, split.windows, split.flights, split.records, split.trained, split.advantages,
                    split.allowed, [[step]] * 4, probe_gains)        # its word still asked
    tuner = WindowRewardTuner(copy.deepcopy(model), _model(Words(spec), slots=2), RewardConfig(), CPU, seed=0,
                              traffic_learning_rate=3e-4, step_s=spec.step_s, imitation_weight=2.0)
    batch, logits, _, _, _, scored, imitation = tuner._window_scored(probed, [0, 1, 2, 3], None)
    assert scored.tolist() == np.concatenate(split.advantages).tolist()
    assert not batch["asked"][:, 0, N_LOOK + step, APPROACH].any()
    log_p = torch.log_softmax(logits[APPROACH][:, 0, N_LOOK + step], dim=-1)[:, APPROACH_GO_AROUND + 1]
    gain = torch.as_tensor(np.concatenate(probe_gains), dtype=log_p.dtype)
    assert torch.isfinite(log_p).all()
    expected = torch.where(gain > 0, -2.0 * gain * log_p, torch.zeros_like(log_p))
    assert torch.allclose(imitation, expected) and (imitation[gain <= 0] == 0).all()
    _, _, _, _, _, _, none = tuner._window_scored(split, [0, 1, 2, 3], None)
    assert none is None


def test_window_flight_leaves_a_probes_word_unasked_and_a_pass_learns_it(tmp_path, monkeypatch):
    """`window_flight(forced=…)` clears the probe's approach cell only (the step stays counted), and a pass over a
    sentence a probe did better in adds the term."""
    import copy

    from ts_transformer.experiments.traffic_scene_data import build_split
    from ts_transformer.experiments.traffic_window_reward import probed_at
    from ts_transformer.experiments.traffic_window_tuner import (
        WindowRewardTuner, WindowSplit, counted_rows, window_flight,
    )
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.instructions.words import APPROACH
    from ts_transformer.prior import data as prior_data
    from ts_transformer.prior.data import Split, column_classes
    from ts_transformer.prior.train import RewardConfig
    from ts_transformer.tests.test_prior_speaker import _model

    assert probed_at({"forced": 4}, 10) == 4 and probed_at({"forced": 12}, 10) is None
    assert probed_at({"forced": None}, 10) is None
    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _reading_model(spec)
    loop = _loop(model, airport, signals, spec, [_keys(3), _keys(3)], [60.0] * 2)
    while loop.running:
        loop.step()
    records, counted = loop.records(), [r.counted for r in loop.results()]
    geometry, step = airport.flights.geometry, 3
    flights = [window_flight(r, loop.windows[i], loop.flights[i], geometry, None, 0, 0, counted[i], spec.step_s, step)
               for i, r in enumerate(records)]
    plain = window_flight(records[0], loop.windows[0], loop.flights[0], geometry, None, 0, 0, counted[0], spec.step_s)
    assert not flights[0].asked[N_LOOK + step, APPROACH] and flights[0].asked[N_LOOK + step].any()
    assert counted_rows(flights[0]) == counted_rows(plain)
    assert ((flights[0].asked != plain.asked).sum()) == 1
    masks = []
    for i in range(2):
        allows = {c: m[i, : counted[i]].copy() for c, m in loop.speaker.allowed.items()}
        allows[APPROACH][step] = np.packbits(np.ones(model.config.classes[APPROACH], dtype=bool), bitorder="little")
        masks.append([allows])
    table = Split([], ("KXXX",), prior_data.candidate_table({"KXXX": geometry}, ("KXXX",), 2), (("09",),), ((90.0,),),
                  column_classes(Words(spec), 2), "no-context")
    split = WindowSplit(table, list(loop.windows), [[f] for f in flights], [[r] for r in records], [[0]] * 2,
                        [np.array([0.5]), np.array([-0.5])], masks, [[step]] * 2, [np.array([0.5]), np.array([-0.5])])
    data, _ = build_split(tmp_path / "artefact", "train", spec, ("KXXX",), None, 2_048)
    tuner = WindowRewardTuner(copy.deepcopy(model), _model(Words(spec), slots=2), RewardConfig(), CPU, seed=0,
                              traffic_learning_rate=3e-4, step_s=spec.step_s)
    assert tuner.window_pass(split, data, slots=2)["imitation_mean"] > 0.0

