"""Stage C, C10: the post-training campaign (post-training §2, §8 C10) — on A26's one-flight synthetic artefact (the
fixture of `test_post_window_loop`); every write root under tmp."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.experiments import post_train
from ts_transformer.experiments import training_export as export
from ts_transformer.experiments.post_train import (
    INPUT_PATHS, KINDS, Context, Settings, Speakers, batches, done_rounds, draw_round, open_campaign, run_campaign,
    speak_round, start_model, train_pass, update_pairs, window_record,
)
from ts_transformer.autopilot.start import Start, start_moved
from ts_transformer.instructions.artefact import closed_loop_sentences
from ts_transformer.post.scene import INSERTED, LEADER_MOVED, MOVED_START, NO_START_MOVE, REAL
from ts_transformer.prior.checkpoint import claim_validation_read
from ts_transformer.prior.source import ArtefactSource
from ts_transformer.tests.test_post_branches import _ahead, _round
from ts_transformer.tests.test_post_window_loop import CPU, DELTA, setup  # noqa: F401

PER_KIND = {REAL: 1, INSERTED: 1, LEADER_MOVED: 1, MOVED_START: 1}


def _settings(**changed):
    values = dict(rounds=2, per_kind=PER_KIND, batch_windows=2, continuations=2, seed=1337, prior_lr=1e-4,
                  traffic_lr=1e-3, weight_decay=0.0, update_groups=1, data_sentences=1, select_per_airport=1,
                  traffic_hidden=16, traffic_heads=4, start=None)
    return Settings(**{**values, **changed})


def _inputs(settings, root, **paths):
    """A campaign's inputs, its paths under ``root``'s linked data trees (``paths``: one changed; never read here)."""
    values = {"prior": "4dTrajectory/outputs/POOLED/prior/base/run",
              "instructions": "4dTrajectory/outputs/POOLED/instruction_language/v",
              "executor": "4dTrajectory/outputs/POOLED/executor/v", "windows": "4dTrajectory/outputs/POOLED/post/windows",
              "procedure_root": "aeroviz-4d/public/data/airports", **paths}
    return {**{key: str(root / path) for key, path in values.items()}, "settings": asdict(settings), "smoke": False}


def _context(s):
    """The campaign's context on the fixture's artefact: its train split read as the select days too (the artefact has
    a train split only), with the split's opened `Start`."""
    code = s["geometry"].code
    train = {"windows": s["windows"], "sentences": closed_loop_sentences(s["directory"], "train", DELTA,
                                                                         s["words"].spec),
             "flights": s["flights"], "signals": {}, "faults": {code: {}}}
    from ts_transformer.instructions.artefact import load_signals

    train["signals"] = {x.dataset_id: x for x in load_signals(s["directory"], "train")}
    train["start"] = Start(s["directory"], "train", DELTA, s["directory"].parent / "executor")
    data = ArtefactSource(s["directory"], DELTA, "full", {code: s["roster"]}, "all").sentences("train", code)
    return Context(s["directory"], s["directory"].parent / "executor", s["reference"], CPU, s["words"], DELTA, s["base"],
                   {"base": "the fixture's"}, s["geometries"], {code: s["roster"]}, s["finals"],
                   {"train": train, "select": train}, data)


def test_the_context_holds_the_base_on_its_device(monkeypatch, tmp_path):
    """`open_context` puts the base on the context's device in eval mode, for every caller (the loss's pull reads it
    beside the model: a base left on the CPU failed C8's first profile on the GPU). The inputs stubbed; "meta" stands in
    for the GPU."""
    base = torch.nn.Linear(2, 2).train()
    base.config = SimpleNamespace(variant="full")
    prior = SimpleNamespace(checkpoint=SimpleNamespace(model=base, identity={}), interval_s=4.0, geometries={},
                            landings={}, selection="landed")
    monkeypatch.setattr(post_train, "open_prior", lambda *a, **k: prior)
    monkeypatch.setattr(post_train, "load_spec", lambda instructions: None)
    monkeypatch.setattr(post_train, "Words", lambda spec: None)
    monkeypatch.setattr(post_train, "ArtefactSource", lambda *a: None)
    context = post_train.open_context(tmp_path, tmp_path, tmp_path, tmp_path, torch.device("meta"), tmp_path,
                                      formal=False, data=False, splits=())
    assert next(context.base.parameters()).device.type == "meta" and not context.base.training


def test_the_settings_take_a_count_of_every_kind_and_positive_sizes():
    assert set(_settings().per_kind) == set(KINDS)
    with pytest.raises(ValueError, match="each kind"):
        _settings(per_kind={REAL: 1})
    with pytest.raises(ValueError, match="not all zero"):
        _settings(per_kind=dict.fromkeys(KINDS, 0))
    with pytest.raises(ValueError, match="positive"):
        _settings(update_groups=0)


def test_a_batch_commands_each_flight_once_in_signal_order():
    windows = [SimpleNamespace(signal_index=i, signal_indices=(i,), span_s=0.0)                  # stage C: one aircraft
               for i in (3, 3, 1, 2, 3, 1)]
    out = batches(windows, 2)
    assert sorted(p for b in out for p in b) == list(range(6))
    for b in out:
        flights = [windows[p].signal_index for p in b]
        assert len(b) <= 2 and flights == sorted(set(flights))
    assert out == [[2, 0], [3, 1], [5, 4]]


def test_the_draw_counts_each_kind_and_its_shortfall(setup):
    """One flight: a real window and its window B; no flight for A to insert, no aircraft ahead for D — a shortfall,
    recorded. The same seed draws the same windows."""
    context = _context(setup)
    windows, counted = draw_round(context, PER_KIND, np.random.default_rng([1, 0]))
    assert [w.kind for w in windows] == [REAL, MOVED_START]
    assert counted["drawn"] == {REAL: 1, INSERTED: 0, LEADER_MOVED: 0, MOVED_START: 1}
    assert counted["shortfall"] == {INSERTED: 1, LEADER_MOVED: 1} and counted["left_out_inside_loss"] == {}
    assert windows[1].start_move != NO_START_MOVE and windows[0].start_move == NO_START_MOVE
    again, _ = draw_round(context, PER_KIND, np.random.default_rng([1, 0]))
    assert [asdict(w.start_move) for w in again] == [asdict(w.start_move) for w in windows]
    assert batches(windows, 2) == [[0], [1]]                     # the real window and its B command the same flight


def test_the_pass_reads_the_groups_file_by_file_a_few_at_a_time(setup, tmp_path):
    s = setup
    context = _context(s)
    settings = _settings(update_groups=2)
    model, optimizer = start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))
    directory = tmp_path / "round"
    directory.mkdir()
    torch.save([rewarded] * 3, directory / "groups_0.pt")          # 3 groups: an update of 2, then of 1
    torch.save([], directory / "groups_1.pt")                      # a batch without an informative group
    torch.save([rewarded], directory / "groups_10.pt")             # read after groups_1 (by number, not by name)
    updates = list(update_pairs(directory, context.data, settings, np.random.default_rng(0), CPU, part_width=0))
    sizes = [[piece.rows.asked.shape[0] for piece in pieces] for pieces, _ in updates]
    assert sizes == [[3, 3], [3], [3]]                             # a piece a group, 3 sentences each
    before = [p.detach().clone() for p in model.parameters()]
    passed = train_pass(model, context, optimizer, directory, settings, np.random.default_rng(0), part_width=0)
    assert passed["updates"] == 3 and np.isfinite(passed["loss"]) and not model.training
    assert any(not torch.equal(a, p) for a, p in zip(before, model.parameters()))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert train_pass(model, context, optimizer, empty, settings, np.random.default_rng(0), part_width=0) == {"updates": 0}


def test_the_pass_shuffles_each_file_s_groups_by_the_round_s_numbers(tmp_path, monkeypatch):
    """D130: each file's groups in an order drawn by the round's numbers, the files still in the order spoken; the same
    numbers give the same order."""
    directory = tmp_path / "round"
    directory.mkdir()
    torch.save(list(range(8)), directory / "groups_0.pt")
    torch.save(list(range(10, 15)), directory / "groups_1.pt")
    monkeypatch.setattr(post_train, "samples", lambda groups, device, part_width: list(groups))
    monkeypatch.setattr(post_train, "collate", lambda sentences, device: None)
    settings = _settings(update_groups=3)

    def updates(seed):
        return [[g for piece in pieces for g in piece]           # each piece one group (`samples` returns it)
                for pieces, _ in update_pairs(directory, list(range(4)), settings, np.random.default_rng(seed), CPU,
                                              part_width=0)]

    chunks = updates([1337, 0, 1])
    first = [g for chunk in chunks for g in chunk]
    assert sorted(first[:8]) == list(range(8)) and sorted(first[8:]) == list(range(10, 15))
    assert first != sorted(first) and updates([1337, 0, 1]) == chunks and updates([1337, 1, 1]) != chunks
    # the file is shuffled before it is cut, so an update mixes groups spoken more than one update apart
    assert any(max(c) - min(c) >= settings.update_groups for c in chunks)


def test_speakers_give_the_groups_and_the_record_of_speaking_here(setup, tmp_path, monkeypatch):
    """Two worker processes speak a round's two batches (two short windows of the one flight, `short_round`'s, drawn by
    the workers too: the draw is patched before the fork): the same groups files, byte for byte, and the same record as
    speaking here; the round's model file is gone after; a worker refuses a batch whose windows are not the campaign's;
    one worker is no speakers."""
    s = setup
    context = _context(s)
    settings = _settings(continuations=2)
    model, _ = start_model(context, settings)
    windows = [_ahead(s["windows"][0])] * 2
    monkeypatch.setattr(post_train, "draw_round", lambda context, per_kind, rng: (windows, {}))
    assert len(batches(windows, settings.batch_windows)) == 2
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    record = speak_round(model, context, windows, settings, 0, here)
    speakers = Speakers(context, settings, 2, CPU)
    try:
        assert speak_round(model, context, windows, settings, 0, there, speakers) == record
    finally:
        speakers.close()
    assert sorted(p.name for p in there.iterdir()) == ["groups_0.pt", "groups_1.pt"]
    assert all((here / name).read_bytes() == (there / name).read_bytes() for name in ("groups_0.pt", "groups_1.pt"))
    torch.save(model.state_dict(), tmp_path / "state.pt")
    wrong = [{**window_record(windows[0]), "row0_s": windows[0].row0_s + 4.0}]
    with pytest.raises(ValueError, match="other windows"):
        post_train._speak(0, str(tmp_path / "state.pt"), str(tmp_path), 0, [0], wrong)
    with pytest.raises(ValueError, match="2 or more"):
        Speakers(context, settings, 1, CPU)


def _digest(model) -> float:
    return float(sum(p.detach().double().sum() for p in model.parameters()))


def test_each_speaker_speaks_with_the_round_s_model_and_a_dead_one_fails_the_round(setup, tmp_path, monkeypatch):
    """Through the pool: in two rounds with two different models, every batch is spoken with that round's model (the
    worker reloads it each round) and the campaign's windows; a worker that dies fails the round with
    `BrokenProcessPool`, never hangs it. (The workers' one thread, `_initialise_speaker`, is not tested here: the suite
    runs with one OpenMP thread, where a fork never hangs.)"""
    from concurrent.futures.process import BrokenProcessPool

    s = setup
    context = _context(s)
    settings = _settings(continuations=2)
    windows = [_ahead(s["windows"][0])] * 2
    monkeypatch.setattr(post_train, "draw_round", lambda context, per_kind, rng: (windows, {}))

    def spoken(model, context, windows, places, settings, round_, directory, k):
        if round_ == 9:
            os._exit(1)
        return {"round": round_, "k": k, "digest": _digest(model), "windows": [window_record(windows[i]) for i in places]}

    monkeypatch.setattr(post_train, "speak_batch", spoken)
    speakers = Speakers(context, settings, 2, CPU)
    places = batches(windows, settings.batch_windows)
    try:
        for round_, shift in ((0, 0.01), (1, -0.02)):
            model, _ = start_model(context, settings)
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(shift)
            parts = speakers.speak(model, windows, places, round_, tmp_path)
            assert [(x["round"], x["k"]) for x in parts] == [(round_, 0), (round_, 1)]
            assert all(abs(x["digest"] - _digest(model)) < 1e-6 for x in parts)
            assert [x["windows"] for x in parts] == [[window_record(windows[i]) for i in p] for p in places]
            assert not (tmp_path / "speaking_model.pt").exists()
        with pytest.raises(BrokenProcessPool):
            speakers.speak(model, windows, places, 9, tmp_path)
        assert not (tmp_path / "speaking_model.pt").exists()
    finally:
        speakers.close()


class _OneCall:
    """A split's start that starts every batch by the one-call form (`start_moved`): the reference of C13's `Start`."""

    def __init__(self, directory, executor):
        self.directory, self.executor = directory, executor

    def moved(self, sentences, moves, *, most_go_arounds, device):
        return start_moved(self.directory, "train", DELTA, sentences, self.executor, moves,
                           most_go_arounds=most_go_arounds, device=device)


def _same_ends(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x.index, x.kind, x.outcome, x.reward, x.go_arounds, x.speed_mask_rows, x.faulty_steps,
                x.loss_reads_fault) == (y.index, y.kind, y.outcome, y.reward, y.go_arounds, y.speed_mask_rows,
                                        y.faulty_steps, y.loss_reads_fault)
        assert np.array_equal(x.words, y.words) and np.array_equal(x.states, y.states)


def test_a_round_started_through_start_is_the_round_started_by_the_one_call_form(setup, tmp_path, monkeypatch):
    """C13: the windows started through the split's opened `Start` speak the round the one-call `start_moved` speaks —
    its record and its groups' bytes (the words, the rewards), and the readout's ends (the words, the rewards, the
    states) — a short window (its own flight inserted 8 s ahead, `_ahead`), the second pass on the series the start
    kept; and window B's moved start is the one-call form's, started twice."""
    from ts_transformer.experiments.post_train import B_HEIGHT_M, B_SPEED_SCALE, B_TURN_DEG
    from ts_transformer.post.scene import moved_start_window

    s = setup
    context = _context(s)
    reference = replace(context, splits={"train": {**context.splits["train"],
                                                   "start": _OneCall(s["directory"], context.executor)}})
    settings = _settings(continuations=2)
    model, _ = start_model(context, settings)
    real = s["windows"][0]
    moved = moved_start_window(real, np.random.default_rng(3), turn_deg=B_TURN_DEG, height_m=B_HEIGHT_M,
                               speed_scale=B_SPEED_SCALE)
    windows = [_ahead(real)]                       # window B carries no other aircraft: its start compared below
    here, there = tmp_path / "start", tmp_path / "one_call"
    here.mkdir()
    there.mkdir()
    assert (speak_round(model, context, windows, settings, 0, here)
            == speak_round(model, reference, windows, settings, 0, there))
    assert sorted(p.name for p in here.iterdir()) == sorted(p.name for p in there.iterdir())
    assert all((here / p.name).read_bytes() == p.read_bytes() for p in there.iterdir())
    _same_ends(post_train.read_batch(model, context, windows, [0], settings, "train"),
               post_train.read_batch(model, reference, windows, [0], settings, "train"))
    assert moved.start_move != NO_START_MOVE
    for _ in range(2):                                 # the second start on the series the first kept
        a, b = (c.start_loop("train", [moved])([moved.signal_index]) for c in (context, reference))
        assert a[1] == b[1] and all(np.array_equal(a[2][i], b[2][i]) for i in a[2])
        assert all(torch.equal(getattr(a[0].executor.inputs, f), getattr(b[0].executor.inputs, f))
                   for f in ("initial_state", "aero_params", "frame_params", "max_thrust_n"))
        assert torch.equal(a[0].executor.time_limit_s, b[0].executor.time_limit_s)


def test_the_selection_readout_through_two_workers_is_the_one_process_readout(setup, tmp_path, monkeypatch):
    """C13: the readout's batches read by two speaking workers give the ends of the one-process readout, window for
    window (the words, the rewards, the states), and the same sums; a worker refuses readout windows not its own."""
    s = setup
    context = _context(s)
    settings = _settings()
    model, _ = start_model(context, settings)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)                                   # not the model at the start: the reading's model reaches them
    (real,) = post_train.selection_windows(context, settings)
    select = [_ahead(real)] * 2           # one short window twice (`_ahead`): two batches, each window its own numbers
    monkeypatch.setattr(post_train, "selection_windows", lambda context, settings, split="select": select)
    places = batches(select, settings.batch_windows)
    assert places == [[0], [1]]
    here = [post_train.read_batch(model, context, select, p, settings, "select") for p in places]
    speakers = Speakers(context, settings, 2, CPU)
    try:
        there = speakers.read(model, select, places, "select")
        for a, b in zip(here, there, strict=True):
            _same_ends(a, b)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(-0.03)                              # a second reading's model: a worker keeping the first's fails
        again = [post_train.read_batch(model, context, select, p, settings, "select") for p in places]
        with pytest.raises(AssertionError):                # the second model says otherwise: the check can fail
            _same_ends(here[0], again[0])
        for a, b in zip(again, speakers.read(model, select, places, "select"), strict=True):
            _same_ends(a, b)
        assert (post_train.selection_readout(model, context, select, settings, speakers=speakers)
                == post_train.selection_readout(model, context, select, settings))
        wrong = [{**window_record(select[0]), "row0_s": real.row0_s + 4.0}]
        with pytest.raises(ValueError, match="other readout windows"):
            speakers.pool.submit(post_train._read, 99, str(tmp_path / "x.pt"), "select", [0], wrong, 0).result()
    finally:
        speakers.close()


def test_a_draw_reaches_each_window_s_numbers_here_and_through_the_workers(setup, tmp_path, monkeypatch):
    """P48 (D161): the ceiling readout's draw d reads each window with `readout_numbers(seed, place, d)`, in one process
    and through the workers (forked after the numbers are recorded to a file, so they record too)."""
    s = setup
    context = _context(s)
    settings = _settings()
    model, _ = start_model(context, settings)
    (real,) = post_train.selection_windows(context, settings)
    select = [_ahead(real)] * 2
    monkeypatch.setattr(post_train, "selection_windows", lambda context, settings, split="select": select)
    log = tmp_path / "numbers.txt"
    numbers = post_train.readout_numbers

    def recorded(seed, place, draw=0):
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{seed} {place} {draw}\n")
        return numbers(seed, place, draw)

    monkeypatch.setattr(post_train, "readout_numbers", recorded)
    places = batches(select, settings.batch_windows)
    post_train.read_batch(model, context, select, places[0], settings, "select", 2)
    speakers = Speakers(context, settings, 2, CPU)
    try:
        speakers.read(model, select, places, "select", 3)
    finally:
        speakers.close()
    assert sorted(log.read_text().splitlines()) == ["1337 0 2", "1337 0 3", "1337 1 3"]


def test_the_memory_sampler_sees_a_peak_freed_before_the_end_and_raises_its_failure(monkeypatch):
    """O15: memory taken and freed inside the measured block counts in its peak (the sampling thread sees it), and a
    failure of the thread is raised, not left as a smaller peak."""
    import threading
    import time

    with post_train._PeakSampler(post_train._own_memory()["swapped"]) as sampled:
        start = sampled.peak
        block = np.ones(400 << 20, dtype=np.uint8)                       # 400 MiB, written: resident
        time.sleep(3 * post_train.PEAK_SAMPLE_S)
        del block
        time.sleep(post_train.PEAK_SAMPLE_S)
    assert sampled.peak - start >= 380 << 20
    own = post_train._own_memory

    def failing():
        if threading.current_thread() is not threading.main_thread():
            raise OSError("smaps_rollup unreadable")
        return own()

    monkeypatch.setattr(post_train, "_own_memory", failing)
    with pytest.raises(RuntimeError, match="the memory sampler failed"):
        with post_train._PeakSampler(0):
            time.sleep(3 * post_train.PEAK_SAMPLE_S)


def test_the_pass_memory_measures_one_group_alone_and_takes_the_largest_peak(monkeypatch):
    """O15 with the update in pieces (post-training §2 item 5): the memory rule asks the profile for one group as well as
    for an update's groups, so the widest and the densest group alone are measured, and the pass's peak is the largest
    of every update measured (here the densest group's)."""
    from pathlib import Path

    from ts_transformer.experiments import post_profile

    asked = []

    def measure(model, context, directory, settings, counts, *, part_width):
        asked.append(list(counts))
        assert part_width == 0
        return {"groups_written": 9, "updates": {"1": {"gpu_peak_reserved_gib": 1.0},
                                                 "1_densest": {"gpu_peak_reserved_gib": 3.0},
                                                 "4": {"gpu_peak_reserved_gib": 2.0}}}

    monkeypatch.setattr(post_profile, "pass_memory", measure)
    monkeypatch.setattr(post_train, "start_model", lambda context, settings: (None, None))
    monkeypatch.setattr(post_train, "_gpu_used", lambda: 5 << 30)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 1 << 30)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    out = post_train.pass_memory_of(SimpleNamespace(device=torch.device("cuda")), SimpleNamespace(update_groups=4),
                                    Path("groups"))
    assert asked == [[1, 4]]
    assert out == {"peak": (5 << 30) - (1 << 30) + (3 << 30), "now": 5 << 30, "groups": 9}


def test_n_workers_are_refused_by_name_where_one_workers_measured_memory_does_not_fit(setup, monkeypatch):
    """O15: one worker speaks the first batch of the round to come and its memory is measured (here on the CPU: the host
    only); N workers fit when N times its peak is at most what is available plus what it holds already, and the pass
    (on a GPU) when its growth and what the other workers hold fit in the free memory; refused by name otherwise."""
    gib = 1 << 30
    held = {"reader_model": gib // 4, "series": gib // 2, "round_flights": 100}
    measured = {"host": {"peak": 2 * gib, "now": 1 * gib}, "gpu": {"peak": 1 * gib, "now": gib // 4}, "held": held}
    passed = {"peak": 3 * gib, "now": gib // 2, "groups": 4}
    # host: 3 × (2 + 0.5 series) = 7.5 ≤ 6.5 + 1; GPU: 3 × (1 + 0.25 model) = 3.75 ≤ 3.75 + 0.25, the pass
    # 2.5 + 2 × (0.25 + 0.25) + 0.25 = 3.75 ≤ 3.75
    assert post_train.workers_fit(3, measured, passed, {"host": int(6.5 * gib), "gpu": int(3.75 * gib)},
                                  held_now=True) == []
    short = post_train.workers_fit(3, measured, passed, {"host": 6 * gib, "gpu": int(3.25 * gib)}, held_now=True)
    assert [line.split(":")[0] for line in short] == ["host", "gpu", "gpu"]
    assert "3 workers speaking need 7.5 GiB" in short[0] and "the pass beside 3 workers needs 3.8 GiB more" in short[2]
    assert [line.split(":")[0] for line in post_train.workers_fit(3, measured, passed, {"host": int(6.5 * gib),
                                                                                      "gpu": int(3.7 * gib)},
                                                                  held_now=True)] == ["gpu"]
    # nothing held yet (a campaign's start from its profile): host 3 × 2.5 = 7.5 ≤ 7.5; GPU speaking 3.75 ≤ 3.75, the
    # pass its whole peak 3 beside 3 workers holding 0.25 + 0.25 each: 4.5 > 3.75
    assert [line.split(":")[0] for line in post_train.workers_fit(3, measured, passed, {"host": int(7.5 * gib),
                                                                                      "gpu": int(3.75 * gib)},
                                                                  held_now=False)] == ["gpu"]
    assert post_train.workers_fit(3, measured, passed, {"host": int(7.5 * gib), "gpu": int(4.5 * gib)},
                                  held_now=False) == []
    assert [line.split(":")[0] for line in post_train.workers_fit(3, measured, passed, {"host": int(7.4 * gib),
                                                                                      "gpu": int(4.5 * gib)},
                                                                  held_now=False)] == ["host"]
    on_cpu = {**measured, "gpu": None}                     # the readout's model on the host: 3 × 2.75 = 8.25 ≤ 7.25 + 1
    assert post_train.workers_fit(3, on_cpu, None, {"host": int(7.25 * gib), "gpu": None}, held_now=True) == []
    assert post_train.workers_fit(3, on_cpu, None, {"host": 7 * gib, "gpu": None}, held_now=True)
    s = setup
    context = _context(s)
    short_round(monkeypatch, s)                            # before the fork: the workers draw the short round too
    speakers = Speakers(context, _settings(continuations=2), 2, CPU)
    try:
        settings = _settings(continuations=2)
        measure = post_train.require_workers_fit(speakers, context, settings, 0, start_model(context, settings)[0])
        assert measure["pass"] is None and measure["measured"]["gpu"] is None and measure["measured"]["windows"] >= 1
        assert measure["measured"]["host"]["peak"] >= measure["measured"]["host"]["now"] > 0
        held = measure["measured"]["held"]
        assert held["round_flights"] == 1 and held["series"] > 0 and held["reader_model"] == sum(
            p.numel() * p.element_size() for p in (*start_model(context, settings)[0].parameters(),
                                                   *start_model(context, settings)[0].buffers()))
        monkeypatch.setattr(post_train, "available_memory", lambda device: {"host": 0, "gpu": None})
        with pytest.raises(SystemExit, match=r"do not fit \(O15\): host: 2 workers speaking need"):
            post_train.require_workers_fit(speakers, context, settings, 0, start_model(context, settings)[0])
    finally:
        speakers.close()


def short_round(monkeypatch, s):
    """The round's windows replaced by one short window (its own flight inserted 8 s ahead: lost at its first row flown,
    one branch point), so a round flies only a few rows; the draw has its own test, the real window's whole flight
    `test_post_branches`."""
    window = _ahead(s["windows"][0])
    monkeypatch.setattr(post_train, "draw_round", lambda context, per_kind, rng: ([window], {"drawn": {INSERTED: 1}}))
    return window


def test_a_campaign_round_end_to_end(setup, tmp_path, monkeypatch):
    """One round through every step but the draw (`short_round`): the two passes, the pass, the selection readout (the
    real window of the select days), the record and the checkpoint with its identity."""
    s = setup
    short_round(monkeypatch, s)
    settings = _settings(rounds=1, continuations=2)
    out = tmp_path / "campaign"
    open_campaign(out, {"settings": asdict(settings)}, {"head": "x", "dirty": False}, {})
    released = []
    real = Start.release
    monkeypatch.setattr(Start, "release", lambda self: (released.append(self.split), real(self))[1])
    run_campaign(out, settings, _context(s))
    assert done_rounds(out) == 1
    assert released == ["train"]                           # C13: the round's kept series released at its start
    record = json.loads((out / "round_0" / "round.json").read_text())
    assert [w["kind"] for w in record["windows"]] == [INSERTED] and record["speaking"]["spoken_again"] == 1
    assert record["speaking"]["groups"] == 1 and record["speaking"]["outcomes"] == {"lost_separation": 1}
    assert record["groups_bytes"] > 0 and not list((out / "round_0").glob("groups_*.pt"))     # deleted after the round
    assert record["speaking"]["windows"] == 1 and sum(record["speaking"]["outcomes"].values()) == 1
    readout = record["selection_readout"][s["geometry"].code]
    assert readout["windows"] == 1 and sum(readout["outcomes"].values()) == 1
    identity = torch.load(out / "round_0" / "checkpoint.pt", weights_only=False)["identity"]
    assert identity["schema"] == post_train.POST_CHECKPOINT_SCHEMA and identity["base"] == {"base": "the fixture's"}
    assert identity["rounds"] == [record["windows"]]


def test_a_resumed_campaign_is_the_campaign_run_through(setup, tmp_path, monkeypatch):
    """The resume's mechanics, with the speaking and the readout replaced (each round writes the same informative
    groups, so every round updates the model, its data term with dropout): a round broken off is moved aside and run
    again from the last checkpoint, and the weights after the last round are those of the campaign run through, bit
    for bit; other settings are refused, and a directory that holds no campaign."""
    s = setup
    context = _context(s)
    settings = _settings(rounds=2, per_kind={**dict.fromkeys(KINDS, 0), REAL: 1}, update_groups=1)
    model, _ = start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))

    def speak(model, context, windows, settings, round_, directory, speakers, *, stage):
        torch.save([rewarded, rewarded], directory / "groups_0.pt")
        return {"windows": len(windows)}

    monkeypatch.setattr(post_train, "speak_round", speak)
    monkeypatch.setattr(post_train, "selection_readout", lambda *a, **k: {"KXXX": {"reward_mean": 0.0}})
    inputs = _inputs(settings, tmp_path)
    through = tmp_path / "through"
    open_campaign(through, inputs, {"head": "x", "dirty": False}, {})
    run_campaign(through, settings, context)
    record = json.loads((through / "round_1" / "round.json").read_text())
    assert record["pass"]["updates"] == 2 and done_rounds(through) == 2
    state = torch.load(through / "round_1" / "checkpoint.pt", weights_only=False)
    broken = tmp_path / "broken"
    open_campaign(broken, inputs, {"head": "x", "dirty": False}, {})
    passes = []
    real = post_train.train_pass

    def killed_in_round_1(*a, **k):
        passes.append(1)
        if len(passes) == 2:
            raise RuntimeError("killed")
        return real(*a, **k)

    monkeypatch.setattr(post_train, "train_pass", killed_in_round_1)
    with pytest.raises(RuntimeError, match="killed"):
        run_campaign(broken, settings, context)
    assert done_rounds(broken) == 1 and (broken / "round_1").exists()
    monkeypatch.setattr(post_train, "train_pass", real)
    reopened = open_campaign(broken, inputs, {"head": "y", "dirty": False}, {})
    assert len(reopened["aborted"]) == 1 and (broken / reopened["aborted"][0]).is_dir()
    assert reopened["resumed"][0]["git"] == {"head": "y", "dirty": False}
    run_campaign(broken, settings, context)
    resumed = torch.load(broken / "round_1" / "checkpoint.pt", weights_only=False)
    assert all(torch.equal(resumed["model"][k], state["model"][k]) for k in state["model"])
    assert any(not torch.equal(state["model"][k], v) for k, v in model.state_dict().items())   # the rounds moved it
    with pytest.raises(SystemExit, match="other inputs"):
        open_campaign(broken, _inputs(replace(settings, seed=1), tmp_path), {"head": "y", "dirty": False}, {})
    (tmp_path / "stray").mkdir()
    with pytest.raises(SystemExit, match="holds no campaign"):
        open_campaign(tmp_path / "stray", inputs, {"head": "x", "dirty": False}, {})


def _same(a, b) -> bool:
    """Equal nested states (a checkpoint: the model, the optimizer, the identity), every tensor bit for bit."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_a_campaign_raised_to_more_rounds_is_the_campaign_of_that_count_from_its_start(setup, tmp_path, monkeypatch):
    """D157: a campaign of 2 rounds raised to 3 by a resume runs round 2 only, rounds 0 and 1 untouched, and its round 2
    is the round 2 of a campaign of 3 rounds from its start: the checkpoint (the model, the optimizer and the identity,
    bit for bit), ``round.json`` (but its commit and time) and the selection readout. The speaking writes the same
    informative groups each round and the readout reads the weights (the resume's test); the record keeps its paths,
    raises its count and adds the resume's entry."""
    s = setup
    context = _context(s)
    three = _settings(rounds=3, per_kind={**dict.fromkeys(KINDS, 0), REAL: 1}, update_groups=1)
    two = replace(three, rounds=2)
    model, _ = start_model(context, three)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))

    def speak(model, context, windows, settings, round_, directory, speakers, *, stage):
        torch.save([rewarded, rewarded], directory / "groups_0.pt")
        return {"windows": len(windows), "weights": _digest(model)}

    monkeypatch.setattr(post_train, "speak_round", speak)
    monkeypatch.setattr(post_train, "selection_readout", lambda model, *a, **k: {"KXXX": {"reward_mean": _digest(model)}})
    through, raised = tmp_path / "through", tmp_path / "raised"
    open_campaign(through, _inputs(three, tmp_path), {"head": "x", "dirty": False}, {})
    run_campaign(through, three, context)
    open_campaign(raised, _inputs(two, tmp_path), {"head": "x", "dirty": False}, {})
    run_campaign(raised, two, context)
    kept = {path: path.read_bytes() for path in sorted(raised.glob("round_[01]/*"))}
    passes = []
    real = post_train.train_pass
    monkeypatch.setattr(post_train, "train_pass", lambda *a, **k: (passes.append(1), real(*a, **k))[1])
    record = open_campaign(raised, _inputs(three, tmp_path), {"head": "y", "dirty": False}, {"stub": True})
    run_campaign(raised, three, context)
    assert passes == [1] and done_rounds(raised) == 3                                      # round 2 only
    assert {path: path.read_bytes() for path in sorted(raised.glob("round_[01]/*"))} == kept
    assert _same(*(torch.load(c / "round_2" / "checkpoint.pt", weights_only=False) for c in (through, raised)))
    ends = [{k: v for k, v in json.loads((c / "round_2" / "round.json").read_text()).items()
             if k not in ("git", "finished_utc")} for c in (through, raised)]
    assert ends[0] == ends[1] and ends[0]["pass"]["updates"] == 2
    before = json.loads((through / "round_1" / "round.json").read_text())["selection_readout"]
    assert ends[0]["selection_readout"] != before                         # round 2 moved the weights the readout reads
    stored = json.loads((raised / "campaign.json").read_text())
    assert stored == record and stored["inputs"] == _inputs(three, tmp_path)
    (entry,) = stored["resumed"]
    assert entry["rounds"] == {"before": 2, "after": 3} and entry["git"] == {"head": "y", "dirty": False}
    assert entry["checks"] == {"stub": True} and entry["inputs"] == {k: _inputs(three, tmp_path)[k] for k in INPUT_PATHS}


def test_a_campaign_starts_from_a_round_of_another_campaign_with_a_new_optimizer(setup, tmp_path, monkeypatch):
    """The user, 2026-10-07: a campaign started from round 0 of another campaign starts from that checkpoint's weights
    (`round_model(None)`), with a new optimizer (its state after one round counts that round's updates only); a start
    whose bytes or identity are not the recorded ones is refused by name; a resume with another start is refused."""
    from ts_transformer.io_utils import file_sha256

    s = setup
    context = _context(s)
    settings = _settings(rounds=1, per_kind={**dict.fromkeys(KINDS, 0), REAL: 1}, update_groups=1)
    model, _ = start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))
    monkeypatch.setattr(post_train, "speak_round", lambda model, context, windows, settings, round_, directory, speakers, *, stage: (
        torch.save([rewarded, rewarded], directory / "groups_0.pt"), {"windows": 1, "weights": _digest(model)})[1])
    monkeypatch.setattr(post_train, "selection_readout", lambda model, *a, **k: {"KXXX": {"reward_mean": _digest(model)}})
    source, out = tmp_path / "source", tmp_path / "from_round_0"
    open_campaign(source, _inputs(settings, tmp_path), {"head": "x", "dirty": False}, {})
    run_campaign(source, settings, context)
    held = torch.load(source / "round_0" / "checkpoint.pt", weights_only=False)
    start = {"campaign": str(source), "round": 0, "checkpoint_sha256": file_sha256(source / "round_0" / "checkpoint.pt")}
    started = replace(settings, start=start, seed=2024)                     # D162: another seed than the source's
    first = post_train.round_model(context, started, out, None).state_dict()
    assert all(torch.equal(first[k], held["model"][k]) for k in held["model"])
    assert any(not torch.equal(first[k], v) for k, v in model.state_dict().items())          # not the base's start
    open_campaign(out, _inputs(started, tmp_path), {"head": "x", "dirty": False}, {})
    run_campaign(out, started, context)
    after = torch.load(out / "round_0" / "checkpoint.pt", weights_only=False)
    spoken = json.loads((out / "round_0" / "round.json").read_text())
    updates = spoken["pass"]["updates"]
    probe = post_train.start_model(context, settings)[0]
    probe.load_state_dict(held["model"])
    assert spoken["speaking"]["weights"] == _digest(probe)                       # round 0 spoke with the start's weights
    steps = {float(v["step"]) for v in after["optimizer"]["state"].values()}
    assert steps == {float(updates)} and {float(v["step"]) for v in held["optimizer"]["state"].values()} == {2.0}
    with pytest.raises(SystemExit, match="other inputs"):
        open_campaign(out, _inputs(settings, tmp_path), {"head": "x", "dirty": False}, {})
    with pytest.raises(ValueError, match="not the checkpoint the campaign started from"):
        post_train.campaign_start(context, replace(started, start={**start, "checkpoint_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="another base, masks, traffic shape or round"):
        post_train.campaign_start(context, replace(started, traffic_hidden=32))
    with pytest.raises(ValueError, match=r"takes another seed.*\(D162\)"):
        post_train.campaign_start(context, replace(started, seed=settings.seed))
    with pytest.raises(ValueError, match="a start is its campaign, round and checkpoint_sha256"):
        replace(settings, start={"campaign": str(source)})


def test_a_start_is_named_by_both_options_and_a_formal_campaign_takes_no_smoke_start(tmp_path, monkeypatch, capsys):
    """`post_train --start-campaign --start-round`: refused by name before anything is opened when one is missing, the
    round is not done or the source is a smoke campaign under a formal one."""
    source = tmp_path / "source"
    (source / "round_0").mkdir(parents=True)
    (source / "round_0" / "checkpoint.pt").write_bytes(b"weights")
    (source / "campaign.json").write_text(json.dumps({"schema": post_train.CAMPAIGN_SCHEMA, "inputs": {"smoke": True}}))
    argv = ["--prior", str(tmp_path / "p"), "--instructions", str(tmp_path / "i"), "--executor", str(tmp_path / "e"),
            "--windows", str(tmp_path / "w"), "--out", str(tmp_path / "campaign"), "--rounds", "1",
            "--batch-windows", "1", "--seed", "1", "--prior-lr", "1e-4", "--traffic-lr", "1e-3", "--weight-decay", "0",
            "--update-groups", "1", "--data-sentences", "1", "--select-per-airport", "1", "--traffic-hidden", "16",
            "--traffic-heads", "4"]
    argv += [x for kind in KINDS for x in (f"--windows-{kind.lower()}", "1")]
    monkeypatch.setattr(post_train, "git_state", lambda: (_ for _ in ()).throw(RuntimeError("past the options")))
    for more, says in ((["--start-campaign", str(source)], "go together"),
                       (["--start-campaign", str(source), "--start-round", "0"], "smoke campaign"),
                       (["--start-campaign", str(source), "--start-round", "1", "--smoke"], "not 1")):
        with pytest.raises(SystemExit):
            post_train.main(argv + more)
        assert says in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="past the options"):                  # a smoke may start from a smoke
        post_train.main(argv + ["--start-campaign", str(source), "--start-round", "0", "--smoke"])
    assert not (tmp_path / "campaign").exists()


def test_a_resume_may_raise_the_rounds_and_change_nothing_else(tmp_path):
    """D157: fewer rounds, or more rounds with another setting or input changed, are refused by name, and the record is
    left as it was; P47: once the val read is claimed (spent or not), the rounds are not raised, a resume of the same
    count still opens."""
    settings, git = _settings(rounds=3), {"head": "x", "dirty": False}
    out = tmp_path / "campaign"
    open_campaign(out, _inputs(settings, tmp_path), git, {})
    written = (out / "campaign.json").read_text()
    with pytest.raises(SystemExit, match="not lower them to 2"):
        open_campaign(out, _inputs(replace(settings, rounds=2), tmp_path), git, {})
    for changed in (_inputs(replace(settings, rounds=4, seed=1), tmp_path),
                    _inputs(replace(settings, rounds=4, update_groups=2), tmp_path),
                    {**_inputs(replace(settings, rounds=4), tmp_path), "smoke": True},
                    _inputs(replace(settings, rounds=4), tmp_path, executor="4dTrajectory/outputs/POOLED/executor/w")):
        with pytest.raises(SystemExit, match="other inputs or settings"):
            open_campaign(out, changed, git, {})
    assert (out / "campaign.json").read_text() == written
    claim_validation_read(out, post_train.CLAIM_READER, tmp_path / "validation", {"round": 0, "device": "cpu"})
    with pytest.raises(SystemExit, match=r"val read is claimed .*not raised after it \(P47\)"):
        open_campaign(out, _inputs(replace(settings, rounds=4), tmp_path), git, {})
    assert (out / "campaign.json").read_text() == written
    assert open_campaign(out, _inputs(settings, tmp_path), git, {})["resumed"][0]["rounds"] == {"before": 3, "after": 3}


def test_a_campaign_recorded_in_a_worktree_resumes_from_another_checkout(tmp_path, monkeypatch):
    """D157: the recorded paths are compared as this checkout reads them (`this_checkout`, its own test in
    `test_training_export`): a campaign recorded under a worktree's linked data trees, the worktree deleted since,
    resumes from the main checkout; the record keeps its paths and the resume's entry gives the paths it read; a path
    that reads as another is refused."""
    main = tmp_path / "thesis"
    for tree in export.LINKED_TREES:
        (main / tree).mkdir(parents=True)
    monkeypatch.setattr(export, "this_checkout", partial(export.this_checkout, root=main))
    gone, git = main / ".claude" / "worktrees" / "v4-post", {"head": "x", "dirty": False}
    settings = _settings(rounds=2)
    out = main / "4dTrajectory" / "outputs" / "POOLED" / "post" / "campaign"
    open_campaign(out, _inputs(settings, gone), git, {})
    record = open_campaign(out, _inputs(replace(settings, rounds=3), main), git, {})
    assert record["inputs"] == _inputs(replace(settings, rounds=3), gone)
    assert record["resumed"][0]["inputs"] == {key: _inputs(settings, main)[key] for key in INPUT_PATHS}
    assert open_campaign(out, _inputs(replace(settings, rounds=3), gone), git, {})["resumed"][1]["inputs"] \
        == record["resumed"][0]["inputs"]                                   # read from the deleted worktree's name too
    with pytest.raises(SystemExit, match="other inputs"):
        open_campaign(out, _inputs(replace(settings, rounds=3), main, windows="4dTrajectory/outputs/POOLED/post/other"),
                      git, {})


def test_a_formal_campaign_needs_a_clean_tree(tmp_path, monkeypatch):
    """Refused before anything is opened: a tree with changes (unless a smoke). Its intent is not checked here (the
    user, 2026-10-07): a clean tree goes on to the checks."""
    argv = ["--prior", str(tmp_path / "p"), "--instructions", str(tmp_path / "i"), "--executor", str(tmp_path / "e"),
            "--windows", str(tmp_path / "w"), "--procedure-root", str(tmp_path / "cifp"),
            "--out", str(tmp_path / "no_such_campaign_20991231"), "--rounds", "1", "--batch-windows", "1",
            "--seed", "1", "--prior-lr", "1e-4", "--traffic-lr", "1e-3", "--weight-decay", "0", "--update-groups", "1",
            "--data-sentences", "1", "--select-per-airport", "1", "--traffic-hidden", "16", "--traffic-heads", "4"]
    argv += [x for kind in KINDS for x in (f"--windows-{kind.lower()}", "1")]
    monkeypatch.setattr(post_train, "git_state", lambda: {"head": "x", "dirty": True})
    with pytest.raises(SystemExit):
        post_train.main(argv)
    monkeypatch.setattr(post_train, "git_state", lambda: {"head": "x", "dirty": False})
    monkeypatch.setattr(post_train, "require_conforming_closed_loop",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("checked")))
    with pytest.raises(RuntimeError, match="checked"):                 # no intent asked for: on to the checks
        post_train.main(argv)
    assert not (tmp_path / "no_such_campaign_20991231").exists()


def test_a_formal_context_refuses_a_fold_or_a_smoke_prior(monkeypatch, tmp_path):
    """D132: a formal campaign and its validation readout (``formal``) take stage B's formal base only; a fold or a smoke
    prior is refused by name; a smoke run does not check."""
    def opened(held_out, sample):
        base = torch.nn.Linear(2, 2)
        base.config = SimpleNamespace(variant="full")
        return SimpleNamespace(directory=tmp_path / "run", config={"run": {"held_out": held_out}},
                               checkpoint=SimpleNamespace(model=base, identity={}, run={"sample": sample}),
                               interval_s=4.0, geometries={}, landings={}, selection="landed")

    monkeypatch.setattr(post_train, "load_spec", lambda instructions: None)
    monkeypatch.setattr(post_train, "Words", lambda spec: None)
    monkeypatch.setattr(post_train, "ArtefactSource", lambda *a: None)

    def context(prior, formal):
        monkeypatch.setattr(post_train, "open_prior", lambda *a, **k: prior)
        return post_train.open_context(tmp_path, tmp_path, tmp_path, tmp_path, CPU, tmp_path, formal=formal,
                                       data=False, splits=())

    with pytest.raises(SystemExit, match="is a fold"):
        context(opened("KSJC", None), True)
    with pytest.raises(SystemExit, match="is a smoke prior"):
        context(opened(None, {"per_airport_and_split": 200, "seed": 1337}), True)     # prior_train's record
    assert context(opened(None, None), True).base_identity == {}
    assert context(opened("KSJC", {"per_airport_and_split": 200, "seed": 1337}), False).base_identity == {}


def test_a_formal_campaign_checks_its_base_and_a_smoke_does_not(tmp_path, monkeypatch):
    """`post_train` opens the context with ``formal`` true for a formal campaign, false for a smoke (D132)."""
    asked = []
    monkeypatch.setattr(post_train, "git_state", lambda: {"head": "x", "dirty": False})
    monkeypatch.setattr(post_train, "require_conforming_closed_loop", lambda *a: (None, {"checks": {}}, None))
    monkeypatch.setattr(post_train, "checked_edges", lambda path: None)

    def stop(*a, formal, **k):
        asked.append(formal)
        raise RuntimeError("opened")

    monkeypatch.setattr(post_train, "open_context", stop)
    argv = ["--prior", str(tmp_path / "p"), "--instructions", str(tmp_path / "i"), "--executor", str(tmp_path / "e"),
            "--windows", str(tmp_path / "w"), "--procedure-root", str(tmp_path / "cifp"),
            "--out", str(tmp_path / "campaign_20991231"), "--rounds", "1", "--batch-windows", "1",
            "--seed", "1", "--prior-lr", "1e-4", "--traffic-lr", "1e-3", "--weight-decay", "0", "--update-groups", "1",
            "--data-sentences", "1", "--select-per-airport", "1", "--traffic-hidden", "16", "--traffic-heads", "4"]
    argv += [x for kind in KINDS for x in (f"--windows-{kind.lower()}", "1")]
    for extra in ([], ["--smoke"]):
        with pytest.raises(RuntimeError, match="opened"):
            post_train.main(argv + extra)
    assert asked == [True, False] and not (tmp_path / "campaign_20991231").exists()


def test_a_stage_gives_the_campaigns_round_its_parts_and_stage_cs_is_the_campaign_as_before(setup, tmp_path, monkeypatch):
    """Post-training §9 item 11 (multi-aircraft control D149): `run_campaign` is the round's skeleton; a stage gives it
    its parts — the campaign's start, the selection set, the draw, the speaking (its batch speaker), the readout (its
    batch reader), the checkpoint's identity — in that order; stage C's stage (`STAGE_C`, the default) runs the campaign it ran: a campaign run through a
    stage that wraps `STAGE_C`'s parts leaves the same checkpoint, bit for bit."""
    from dataclasses import fields as dataclass_fields

    s = setup
    short_round(monkeypatch, s)
    settings = _settings(rounds=1, continuations=2)
    called = []

    def wrapped(name, part):
        def call(*args, **kwargs):
            called.append(name)
            return part(*args, **kwargs)
        return call

    spy = post_train.Stage(**{f.name: wrapped(f.name, getattr(post_train.STAGE_C, f.name))
                              if callable(getattr(post_train.STAGE_C, f.name)) else getattr(post_train.STAGE_C, f.name)
                              for f in dataclass_fields(post_train.Stage)})
    states = []
    for name, stage in (("plain", post_train.STAGE_C), ("spied", spy)):
        out = tmp_path / name
        open_campaign(out, {"settings": asdict(settings)}, {"head": "x", "dirty": False}, {})
        run_campaign(out, settings, _context(s), stage=stage)
        states.append(torch.load(out / "round_0" / "checkpoint.pt", weights_only=False))
    # the speaking and the readout run the stage's own batch speaker and reader (the campaign hands them the stage)
    assert called == ["start", "selection", "draw", "speak", "batches", "speak_batch", "readout", "batches",
                      "read_batch", "record", "identity"]
    assert states[0]["model"].keys() == states[1]["model"].keys()
    assert all(torch.equal(states[0]["model"][k], states[1]["model"][k]) for k in states[0]["model"])
    assert states[0]["identity"] == states[1]["identity"]
    assert post_train.run_campaign.__defaults__[-1] is post_train.STAGE_C


def test_the_speaking_workers_run_the_stages_parts(setup, tmp_path):
    """Post-training §9 item 11 (multi-aircraft control MC4): workers given a stage run its parts of a batch — its model
    at the start, its draw, its record of a window and its batch speaker; of a readout, its selection and its batch
    reader; of the measure (O15), its model, draw and batch speaker — not stage C's; one process speaks and reads with
    the same stage's parts; workers of another stage are refused by the speaking, the readout and the campaign; stage C's
    stage is the default."""
    from dataclasses import replace as dc_replace

    s = setup
    context = _context(s)
    settings = _settings(continuations=2)
    windows = [_ahead(s["windows"][0])] * 2                     # window A: stage C's draw of these settings has none
    marks, here_pid = tmp_path / "marks", os.getpid()
    marks.mkdir()

    def started(context, settings):
        (marks / f"started_{os.getpid()}").write_text("stage")
        return post_train.STAGE_C.start_model(context, settings)

    def spoken(model, context, windows, places, settings, round_, directory, k):
        (directory / f"spoken_{k}").write_text("stage")
        if directory.name == "measured":                          # the measure: a real batch, whose series it weighs
            assert [windows[i].kind for i in places] == [INSERTED]  # of the stage's draw
            (marks / "measured_with").write_text(repr(sum(float(p.detach().double().sum()) for p in model.parameters())))
            return post_train.speak_batch(model, context, windows, places, settings, round_, directory, k)
        return {"k": k, "windows": len(places), "groups": 0, "informative_groups": 0, "spoken_again": 0, "differed": [],
                "reward_sum": 0.0, "faulty_steps": 0, "losses_reading_fault": 0, "outcomes": {}}

    class Read(Exception):
        pass

    def read(model, context, windows, places, settings, split, draw=0):
        if os.getpid() == here_pid:
            raise Read(split)
        return [("read", split, len(places))]

    stage = dc_replace(post_train.STAGE_C, start_model=started, draw=lambda context, settings, round_: (windows, {}),
                       record=lambda window: {**window_record(window), "mine": True}, speak_batch=spoken,
                       read_batch=read, selection=lambda context, settings, split: windows)
    model, _ = start_model(context, settings)
    places = batches(windows, settings.batch_windows)
    here = tmp_path / "here"
    here.mkdir()
    assert post_train.speak_round(model, context, windows, settings, 0, here, stage=stage)["batches"] == 2
    assert sorted(p.name for p in here.glob("spoken_*")) == ["spoken_0", "spoken_1"]          # one process: the stage's
    with pytest.raises(Read, match="select"):                                                # its reader too
        post_train.selection_readout(model, context, windows, settings, stage=stage)
    speakers = Speakers(context, settings, 2, CPU, stage=stage)
    measured = tmp_path / "measured"
    measured.mkdir()
    try:
        assert [(x["k"], x["windows"]) for x in speakers.speak(model, windows, places, 0, tmp_path)] == [(0, 1), (1, 1)]
        assert list(marks.glob("started_*"))                     # the speaking workers' model: the stage's
        assert speakers.read(model, windows, places, "select") == [[("read", "select", 1)]] * 2
        given, _ = start_model(context, settings)               # the round's model, not the stage's start model
        with torch.no_grad():
            next(given.parameters()).add_(0.5)
        assert speakers.measure(0, measured, given)["windows"] == 1
        assert float((marks / "measured_with").read_text()) == sum(float(p.detach().double().sum()) for p in given.parameters())
        assert not (measured / "measuring_model.pt").exists()
        with pytest.raises(ValueError, match="another stage"):
            post_train.speak_round(model, context, windows, settings, 0, tmp_path, speakers)
        with pytest.raises(ValueError, match="another stage"):
            post_train.selection_readout(model, context, windows, settings, speakers=speakers)
        with pytest.raises(ValueError, match="another stage"):
            run_campaign(tmp_path / "campaign", settings, context, speakers)
    finally:
        speakers.close()
    assert sorted(p.name for p in tmp_path.glob("spoken_*")) == ["spoken_0", "spoken_1"]
    assert (measured / "spoken_0").exists()                       # the measure spoke the stage's batch
    assert len(list(marks.glob("started_*"))) >= 1 and not (marks / f"started_{os.getpid()}").exists()
    import inspect

    defaults = inspect.signature(Speakers.__init__).parameters
    assert defaults["stage"].default is post_train.STAGE_C and defaults["gpu_budget"].default is None


def test_a_stage_made_from_stage_cs_runs_its_own_batches_through_the_campaign(setup, tmp_path, monkeypatch):
    """Post-training §9 item 11 (the review of MC4, round 2): a stage that keeps stage C's speaking and readout and gives
    its own batch speaker runs that batch speaker through `run_campaign` in one process and with workers alike (the
    campaign hands the stage to its speaking and readout), with the same speaking record."""
    from dataclasses import replace as dc_replace

    s = setup
    short_round(monkeypatch, s)
    settings = _settings(rounds=1, continuations=2)

    def mine(model, context, windows, places, settings, round_, directory, k):
        (directory / f"mine_{k}").write_text("stage")
        return post_train.speak_batch(model, context, windows, places, settings, round_, directory, k)

    stage = dc_replace(post_train.STAGE_C, speak_batch=mine)
    records = []
    for name, workers in (("here", 0), ("there", 2)):
        out = tmp_path / name
        open_campaign(out, {"settings": asdict(settings)}, {"head": "x", "dirty": False}, {})
        context = _context(s)
        speakers = Speakers(context, settings, workers, CPU, stage=stage) if workers else None
        try:
            run_campaign(out, settings, context, speakers, stage=stage)
        finally:
            if speakers is not None:
                speakers.close()
        assert (out / "round_0" / "mine_0").exists(), name
        records.append(json.loads((out / "round_0" / "round.json").read_text())["speaking"])
    assert records[0] == records[1]


def test_a_round_opens_as_a_start_through_one_function(setup, tmp_path, monkeypatch):
    """Post-training §9 item 12 (D162, multi-aircraft control D164): `round_start` opens a campaign's round as a start —
    its checkpoint's model in eval mode and its identity — and refuses by name: other bytes than the recorded ones, a
    round not done, a smoke source under a formal campaign, a checkpoint of another base or of another round, the
    source's seed; `start_of` names the start as the setting records it."""
    import shutil

    from ts_transformer.io_utils import file_sha256

    s = setup
    short_round(monkeypatch, s)
    settings = _settings(rounds=2, continuations=2)
    out = tmp_path / "campaign"
    open_campaign(out, {"settings": asdict(settings), "smoke": False}, {"head": "x", "dirty": False}, {})
    context = _context(s)
    run_campaign(out, settings, context)
    start = post_train.start_of(out, 0, formal=True)
    assert start == {"campaign": str(out), "round": 0, "checkpoint_sha256": file_sha256(out / "round_0" / "checkpoint.pt")}
    formal = replace(context, formal=True)
    model, identity = post_train.round_start(formal, start, seed=2024)
    state = torch.load(out / "round_0" / "checkpoint.pt", weights_only=False)
    assert identity == state["identity"] and not model.training
    assert all(torch.equal(value, state["model"][name]) for name, value in model.state_dict().items())
    with pytest.raises(ValueError, match="not the checkpoint the campaign started from"):
        post_train.round_start(formal, {**start, "checkpoint_sha256": "0" * 64}, seed=2024)
    with pytest.raises(ValueError, match="holds the checkpoints of rounds 0–1, not 2"):
        post_train.start_of(out, 2, formal=True)
    with pytest.raises(ValueError, match=r"takes another seed.*\(D162\)"):
        post_train.round_start(formal, start, seed=settings.seed)
    with pytest.raises(ValueError, match="another base, masks, traffic shape or round"):
        post_train.round_start(replace(formal, base_identity={"base": "another"}), start, seed=2024)
    later = tmp_path / "later"                                     # round 0's place holding round 1's checkpoint
    shutil.copytree(out, later)
    shutil.copy(later / "round_1" / "checkpoint.pt", later / "round_0" / "checkpoint.pt")
    with pytest.raises(ValueError, match="another base, masks, traffic shape or round"):
        post_train.round_start(formal, post_train.start_of(later, 0, formal=True), seed=2024)
    smoke = tmp_path / "smoke"
    open_campaign(smoke, {"settings": asdict(settings), "smoke": True}, {"head": "x", "dirty": True}, {})
    run_campaign(smoke, settings, context)
    with pytest.raises(ValueError, match="is a smoke campaign: a formal campaign does not start from it"):
        post_train.start_of(smoke, 0, formal=True)
    smoke_start = post_train.start_of(smoke, 0, formal=False)
    with pytest.raises(ValueError, match="is a smoke campaign"):
        post_train.round_start(formal, smoke_start, seed=2024)
    post_train.round_start(context, smoke_start, seed=2024)              # a smoke campaign's context opens it


def test_a_batch_commands_each_flight_once_with_windows_of_several_aircraft():
    """Multi-aircraft control D146: windows that share any commanded flight (an anchor or a later aircraft) go to
    different batches; stage C's windows of one aircraft as before."""
    window = lambda indices: SimpleNamespace(signal_index=indices[0], signal_indices=tuple(indices),  # noqa: E731
                                             span_s=300.0)
    windows = [window((1, 4)), window((2,)), window((4, 6)), window((3, 5)), window((5,))]
    out = batches(windows, 10)
    for batch in out:
        held = [i for p in batch for i in windows[p].signal_indices]
        assert len(held) == len(set(held))
        assert [windows[p].signal_index for p in batch] == sorted(windows[p].signal_index for p in batch)
    assert out == [[0, 1, 3], [2, 4]]
