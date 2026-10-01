"""Multi-aircraft design §6.6 step 7.7: rewinding one aircraft before a loss of separation (`traffic_window_rewind`)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.tests.test_traffic_window import (
    STEP_S, _airport, _as_drawn, _patch_runner_physics, _traffic_model,
)


def _flown_windows(tmp_path, monkeypatch, seed=0):
    """`_busy_windows`' flights through the runner's own path: ``(drawn, words, params, every landing, model, masks,
    originals by window)``."""
    import torch

    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments import traffic_window_rewind as rewind
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.masks import PROCEDURE_ALTITUDES, ProcedureMasks
    from ts_transformer.prior.scene import Landings
    from ts_transformer.tests.test_autopilot import _params
    from ts_transformer.tests.test_prior_procedure import _final
    from ts_transformer.tests.test_traffic_speaking import _batch

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0, 80.0, 200.0, 230.0, 400.0, 600.0, 610.0],
                                 tuple(range(8)))
    airport = airports["KXXX"]
    geometry = airport.flights.geometry
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    masks = ProcedureMasks((PROCEDURE_ALTITUDES,), {"KXXX": tuple(_final(candidate=c, crossing_m=1_200.0,
                                                                         faf_d_m=40_000.0)
                                                                  for c in geometry.candidates)})
    groups = [("KXXX:f0", "KXXX:f1", "KXXX:f2"), ("KXXX:f3", "KXXX:f4"), ("KXXX:f5",), ("KXXX:f6", "KXXX:f7")]
    limits = [100.0, 160.0, 60.0, 130.0]
    keys = [k for g in groups for k in g]
    windows = [window_of(airport, 0.0, g, [limit] * len(g), STEP_S) for g, limit in zip(groups, limits)]
    members, at = [], 0
    for g in groups:
        members.append(range(at, at + len(g)))
        at += len(g)
    drawn = _as_drawn(windows, members, _batch(airport, signals, spec, keys),
                      [limit for g, limit in zip(groups, limits) for _ in g])
    _patch_runner_physics(monkeypatch, signals, geometry, keys)
    times = np.sort([f.presence.landing_s for f in airport.flights.flights.values()])
    every = {"KXXX": Landings(times, {c.ident: times if c.ident == "09" else np.zeros(0) for c in geometry.candidates})}
    words, params, model = Words(spec), _params(), _traffic_model(spec)
    chunk = list(range(len(windows)))
    flown = runner.fly_windows(model, drawn, chunk, "scene", words, params, None, 1,
                               generator=torch.Generator().manual_seed(seed), temperature=1.0, procedure_masks=masks)
    sentences = runner.flown_sentences(flown, drawn, "scene", words, every, 1)
    originals = {o.window: o for o in rewind.flown_originals(flown, sentences, chunk)}
    flown.loop.close()
    return drawn, words, params, every, model, masks, originals


def _fly_branches(branches, events, originals, drawn, words, params, every, model, masks, seed):
    """The branches flown in one loop batch, as the runner reads them: ``[(branch, (keys, said, rows, run))]``."""
    import torch

    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments import traffic_window_rewind as rewind

    windows = [events[b.event].window for b in branches]
    given = [g for b in branches for g in rewind.given_of(b, originals[events[b.event].window])]
    flown = runner.fly_windows(model, drawn, windows, "scene", words, params, None, 1,
                               generator=torch.Generator().manual_seed(seed), temperature=1.0, procedure_masks=masks,
                               given=given)
    sentences = runner.flown_sentences(flown, drawn, "scene", words, every, 1)
    out = [(b, (again.keys, again.said, again.rows, flown.loop.runs[k]))
           for k, (again, b) in enumerate(zip(rewind.flown_originals(flown, sentences, windows), branches))]
    flown.loop.close()
    return out


def test_a_rewound_window_s_control_replays_it_and_its_branches_keep_everything_before_their_step(tmp_path,
                                                                                                  monkeypatch):
    """The whole path on the fixture's windows: events found, every control replays its window to the last field (in
    a batch with the branches, from another stream), each branch speaks again only from its step and is read against
    its event."""
    from ts_transformer.experiments import traffic_window_rewind as rewind

    drawn, words, params, every, model, masks, originals = _flown_windows(tmp_path, monkeypatch)
    events = []
    for w in sorted(originals):
        events += rewind.events_of(originals[w], len(events))
    assert events, "the fixture should hold a loss of separation past a first step"
    branches, skipped = [], Counter()
    for event in events:
        got, why = rewind.branches_of(event, originals[event.window], (10.0, 30.0), 2, STEP_S)
        branches += got
        skipped.update(why)
    flown = _fly_branches(branches, events, originals, drawn, words, params, every, model, masks, seed=11)
    controls = 0
    for branch, (keys, said, rows, run) in flown:
        event, original = events[branch.event], originals[events[branch.event].window]
        if branch.speaker is None:
            rewind.check_control(event, original, keys, said, rows, run)
            controls += 1
            continue
        got = rewind.branch_result(branch, event, original, keys, said, rows, run, STEP_S)
        i = original.keys.index(branch.speaker)
        assert (said[i][: branch.step] == original.said[i][: branch.step]).all()
        assert got["rescued"] == (got["pair_cleared"] and got["landed"] and not got["new_losses"])
        assert set(got["changed"]) == set(COLUMNS)
    assert controls == len(events)


def test_a_control_that_differs_stops_the_run(tmp_path, monkeypatch):
    import dataclasses as dc

    from ts_transformer.experiments import traffic_window_rewind as rewind

    drawn, words, params, every, model, masks, originals = _flown_windows(tmp_path, monkeypatch)
    events = []
    for w in sorted(originals):
        events += rewind.events_of(originals[w], len(events))
    event = events[0]
    original = originals[event.window]
    control = rewind.Branch(event.number, None, None, "control", -1, 0)
    (_, (keys, said, rows, run)), = _fly_branches([control], events, originals, drawn, words, params, every, model,
                                                  masks, seed=3)
    rewind.check_control(event, original, keys, said, rows, run)
    moved = [dict(r) for r in rows]
    moved[0]["landing_s"] = None if moved[0]["landing_s"] is not None else 1.0
    with pytest.raises(ValueError, match="the control did not replay"):
        rewind.check_control(event, original, keys, said, moved, run)
    other = [s.copy() for s in said]
    other[0][0, 0] = other[0][0, 0] + 1
    with pytest.raises(ValueError, match="words"):
        rewind.check_control(event, original, keys, other, rows, run)
    with pytest.raises(ValueError, match="the judge's books"):
        rewind.check_control(event, dc.replace(original, ended={}), keys, said, rows, run)


def _original(keys, said_lengths, ended, first_s=None, starts=None):
    from ts_transformer.experiments.traffic_window_rewind import Original

    rows = [{"starts_in_a_loss": bool(starts and k in starts), "airport": "KXXX"} for k in keys]
    return Original(0, list(keys), [np.zeros((n, 6), dtype=np.int64) for n in said_lengths],
                    list(first_s or [0.0] * len(keys)), rows, ended, [], [])


def test_events_one_a_loss_with_their_speakers_roles():
    from ts_transformer.experiments import traffic_window_rewind as rewind

    loss = {"kind": "in_trail", "relation": "same"}
    # a ended for b (both commanded, b not ended): answered + partner
    o = _original(["a", "b"], [10, 10], {"a": {"t_s": 40.0, "with": "b", **loss}})
    (e,) = rewind.events_of(o, 0)
    assert e.speakers == (("a", rewind.ANSWERED), ("b", rewind.PARTNER)) and e.t_s == 40.0 and e.pair == ("a", "b")
    # ended for each other at one step: one event, both
    o = _original(["a", "b"], [10, 10], {"a": {"t_s": 40.0, "with": "b", **loss},
                                         "b": {"t_s": 40.0, "with": "a", **loss}})
    (e,) = rewind.events_of(o, 7)
    assert e.number == 7 and e.speakers == (("a", rewind.BOTH), ("b", rewind.BOTH))
    # with a replayed aircraft: the answered one only; one that started in its loss is no event
    o = _original(["a", "c"], [10, 10], {"a": {"t_s": 40.0, "with": "r", **loss},
                                         "c": {"t_s": 0.0, "with": "r", **loss}}, starts={"c"})
    (e,) = rewind.events_of(o, 0)
    assert e.speakers == (("a", rewind.ANSWERED),) and e.pair == ("a", "r")


def test_branches_start_at_their_offsets_and_skip_what_cannot_speak_again():
    from ts_transformer.experiments import traffic_window_rewind as rewind

    loss = {"kind": "in_trail", "relation": "same"}
    # a: first state at 0 s, ended at 40 s (step 20, 21 lines said); b: first at 20 s, silent past its 3 lines
    o = _original(["a", "b"], [21, 3], {"a": {"t_s": 40.0, "with": "b", **loss}}, first_s=[0.0, 20.0])
    (event,) = rewind.events_of(o, 0)
    branches, skipped = rewind.branches_of(event, o, (10.0, 30.0, 60.0), 2, 2.0)
    steps = {(b.speaker, b.offset): b.step for b in branches if b.speaker}
    assert steps == {("a", "10"): 15, ("a", "30"): 5, ("a", "start"): 0, ("b", "start"): 0}
    assert Counter(b.speaker for b in branches) == Counter({"a": 6, "b": 2, None: 1}) and branches[-1].speaker is None
    assert skipped == Counter({("answered", "60", rewind.BEFORE_FIRST_STEP): 1,
                               ("partner", "10", rewind.SILENT_THEN): 1,            # own step 5 of its 3 lines
                               ("partner", "30", rewind.BEFORE_FIRST_STEP): 1,
                               ("partner", "60", rewind.BEFORE_FIRST_STEP): 1})
    given = rewind.given_of(branches[0], o)
    assert [g.until for g in given] == [15, 3]


def test_a_pair_lost_again_from_a_step_on_counts_a_wake_shortfall_at_a_landing():
    from types import SimpleNamespace

    from ts_transformer.experiments.traffic_window_rewind import _pair_again

    run = SimpleNamespace(episodes=[{"pair": ["b", "a"], "last_s": 30.0}], at_threshold=[])
    assert _pair_again(run, ("a", "b"), 30.0) and not _pair_again(run, ("a", "b"), 32.0)
    assert not _pair_again(run, ("a", "c"), 0.0)
    run = SimpleNamespace(episodes=[], at_threshold=[{"leader": "b", "follower": "a", "t_s": 50.0}])
    assert _pair_again(run, ("a", "b"), 40.0) and not _pair_again(run, ("a", "b"), 52.0)


def test_words_changed_between_a_step_and_the_loss_past_a_sentence_s_end_unchanged():
    from ts_transformer.experiments.traffic_window_rewind import _changed

    original = np.full((5, 6), UNCHANGED, dtype=np.int64)
    again = np.full((3, 6), UNCHANGED, dtype=np.int64)
    original[2, 2] = 4                                         # heading said at own step 2
    again[1, 5] = 2                                            # speed said at own step 1
    got = _changed(original, again, 1, 4)
    assert got == {"runway": 0, "approach": 0, "heading": 1, "altitude": 0, "angle": 0, "speed": 1}


def test_cells_are_tallied_and_split_by_the_speaker_s_stratum():
    from ts_transformer.experiments import traffic_window_rewind as rewind

    def branch(key, offset, rescued):
        return {"speaker": key, "offset": offset, "rescued": rescued, "pair_cleared": rescued, "landed": rescued,
                "new_losses": [], "changed": {c: int(rescued) for c in COLUMNS}}
    records = [{"airport": "KXXX", "kind": "in_trail", "relation": "same", "commanded": 2, "augmented": None,
                "speakers": [["a", rewind.ANSWERED], ["b", rewind.PARTNER]],
                "strata": {"a": "vectored", "b": "straight-in"},
                "branches": [branch("a", "10", r) for r in (True, False, False, True)]
                + [branch("b", "10", False) for _ in range(4)]}]
    got = rewind.readout(records, Counter({("partner", "30", rewind.BEFORE_FIRST_STEP): 1}), ["10", "30", "start"])
    a = got["pooled"][rewind.ANSWERED]["10"]
    assert (a["cells"], a["branches"], a["rescued_any"], a["rescued_mean"]) == (1, 4, 1.0, 0.5)
    assert a["tallies"] == {"0": 0, "1-2": 1, "3-5": 0, "6+": 0}
    assert a["changed_rescued"]["heading"] == 1.0 and a["changed_not_rescued"]["heading"] == 0.0
    b = got["pooled"][rewind.PARTNER]["10"]
    assert b["rescued_any"] == 0.0 and b["tallies"]["0"] == 1 and b["changed_rescued"] is None
    assert set(got["groups"]["stratum"]) == {"vectored", "straight-in"}
    assert set(got["groups"]["stratum"]["vectored"]) == {rewind.ANSWERED}
    assert got["skipped"] == [{"role": "partner", "offset": "30", "why": rewind.BEFORE_FIRST_STEP, "count": 1}]
    assert "augmented_kind" not in got["groups"]
