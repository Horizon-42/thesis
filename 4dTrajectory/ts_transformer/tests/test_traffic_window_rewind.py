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
    given = [g for b in branches for g in rewind.given_of(b, originals[events[b.event].window], STEP_S)]
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
            assert rewind.check_control(event, original, keys, said, rows, run) <= rewind.ROUNDOFF
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
    assert rewind.check_control(event, original, keys, said, rows, run) <= rewind.ROUNDOFF
    nudged = [dict(r) for r in rows]
    nudged[0]["flown_s"] += 1e-9                                       # round-off: replayed
    rewind.check_control(event, original, keys, said, nudged, run)
    moved = [dict(r) for r in rows]
    moved[0]["landing_s"] = None if moved[0]["landing_s"] is not None else 1.0
    with pytest.raises(ValueError, match="the control did not replay"):
        rewind.check_control(event, original, keys, said, moved, run)
    other = [s.copy() for s in said]
    other[0][0, 0] = other[0][0, 0] + 1
    with pytest.raises(ValueError, match="words"):
        rewind.check_control(event, original, keys, other, rows, run)
    with pytest.raises(ValueError, match="the judge's ends"):
        rewind.check_control(event, dc.replace(original, ended={}), keys, said, rows, run)


def _original(keys, said_lengths, ended, first_s=None, starts=None, episodes=(), step_s=2.0):
    """An original window: each aircraft's words (``said_lengths`` lines), first state's time and row — an ended one
    counted to its end — the judge's ends and episodes."""
    from ts_transformer.experiments.traffic_window_rewind import Original

    first_s = list(first_s or [0.0] * len(keys))
    rows = [{"starts_in_a_loss": bool(starts and k in starts), "airport": "KXXX", "end": ended.get(k),
             "counted": n if k not in ended else int(round((ended[k]["t_s"] - f) / step_s))}
            for k, n, f in zip(keys, said_lengths, first_s)]
    return Original(0, list(keys), [np.zeros((n, 6), dtype=np.int64) for n in said_lengths], first_s, rows, ended,
                    list(episodes), [])


def _episode(a, b, first_s, last_s):
    return {"pair": sorted([a, b]), "first_s": first_s, "last_s": last_s}


def test_events_one_an_episode_from_its_first_step_with_their_speakers_roles():
    from ts_transformer.experiments import traffic_window_rewind as rewind

    loss = {"kind": "in_trail", "relation": "same"}
    # a ended for b at 40 s in an episode from 36 s (b answered nothing): answered + partner, t_L the episode's start
    o = _original(["a", "b"], [30, 30], {"a": {"t_s": 40.0, "with": "b", **loss}},
                  episodes=[_episode("a", "b", 36.0, 44.0)])
    (e,) = rewind.events_of(o, 0)
    assert e.speakers == (("a", rewind.ANSWERED), ("b", rewind.PARTNER)) and e.t_s == 36.0 and e.pair == ("a", "b")
    # one episode ending both — at one step or one after the other — is one event, both
    for later in (40.0, 42.0):
        o = _original(["a", "b"], [30, 30], {"a": {"t_s": 40.0, "with": "b", **loss},
                                             "b": {"t_s": later, "with": "a", **loss}},
                      episodes=[_episode("a", "b", 38.0, 44.0)])
        (e,) = rewind.events_of(o, 7)
        assert e.number == 7 and e.t_s == 38.0 and e.speakers == (("a", rewind.BOTH), ("b", rewind.BOTH))
    # with a replayed aircraft: the answered one only; one that started in its loss is no event; a wake shortfall at a
    # landing has no episode: its own time
    o = _original(["a", "c", "d"], [30, 30, 30],
                  {"a": {"t_s": 40.0, "with": "r", **loss}, "c": {"t_s": 0.0, "with": "r", **loss},
                   "d": {"t_s": 50.0, "with": "a", "kind": "at_threshold", "relation": "same"}},
                  starts={"c"}, episodes=[_episode("a", "r", 40.0, 40.0), _episode("c", "r", 0.0, 0.0)])
    first, second = rewind.events_of(o, 0)
    assert first.speakers == (("a", rewind.ANSWERED),) and first.pair == ("a", "r")
    assert second.pair == ("a", "d") and second.t_s == 50.0 and second.kind == "at_threshold"
    assert second.speakers == (("d", rewind.ANSWERED), ("a", rewind.PARTNER))


def test_branches_start_at_their_offsets_and_skip_what_cannot_speak_again():
    from ts_transformer.experiments import traffic_window_rewind as rewind

    loss = {"kind": "in_trail", "relation": "same"}
    # a: first state at 0 s, ended at 40 s for b in an episode from 40 s; b: first at 20 s, ended at 24 s (it spoke 3
    # lines, its words run on silent); c: first at 40 s, the loss's own step
    o = _original(["a", "b"], [30, 30], {"a": {"t_s": 40.0, "with": "b", **loss},
                                         "b": {"t_s": 24.0, "with": "r", **loss}},
                  first_s=[0.0, 20.0], episodes=[_episode("a", "b", 40.0, 40.0), _episode("b", "r", 24.0, 24.0)])
    (event,) = [e for e in rewind.events_of(o, 0) if e.pair == ("a", "b")]
    assert rewind.spoken_steps(o, 1, 2.0) == 3 and rewind.spoken_steps(o, 0, 2.0) == 21
    branches, skipped = rewind.branches_of(event, o, (10.0, 30.0, 60.0), 2, 2.0)
    steps = {(b.speaker, b.offset): b.step for b in branches if b.speaker}
    assert steps == {("a", "10"): 15, ("a", "30"): 5, ("a", "start"): 0, ("b", "start"): 0}
    assert Counter(b.speaker for b in branches) == Counter({"a": 6, "b": 2, None: 1}) and branches[-1].speaker is None
    assert skipped == Counter({("answered", "60", rewind.BEFORE_FIRST_STEP): 1,
                               ("partner", "10", rewind.SILENT_THEN): 1,            # own step 5 of its 3 spoken
                               ("partner", "30", rewind.BEFORE_FIRST_STEP): 1,
                               ("partner", "60", rewind.BEFORE_FIRST_STEP): 1})
    # the speaking aircraft given to its step, the other as far as it spoke
    assert [g.until for g in rewind.given_of(branches[0], o, 2.0)] == [15, 3]
    # an aircraft whose first state is the loss's step cannot avoid it from its start
    o = _original(["a", "c"], [30, 30], {"a": {"t_s": 40.0, "with": "c", **loss}}, first_s=[0.0, 40.0],
                  episodes=[_episode("a", "c", 40.0, 40.0)])
    (event,) = rewind.events_of(o, 0)
    _, skipped = rewind.branches_of(event, o, (), 2, 2.0)
    assert skipped == Counter({("partner", "start", rewind.AT_THE_LOSS): 1})


def test_a_wake_shortfall_off_the_grid_is_read_at_the_step_after_it():
    """A follower ended at a landing 1.5 s after its step 19 (t 38 s) was judged — and had spoken — at step 20: it
    spoke 21 lines, and the loss's step is 20; offsets count from there."""
    from ts_transformer.experiments import traffic_window_rewind as rewind

    end = {"t_s": 38.5, "with": "r", "kind": "at_threshold", "relation": "same"}
    o = _original(["a"], [30], {"a": end})
    assert rewind.own_step_at(o, 0, 38.5, 2.0) == 20 and rewind.own_step_at(o, 0, 40.0, 2.0) == 20
    assert rewind.spoken_steps(o, 0, 2.0) == 21
    (event,) = rewind.events_of(o, 0)
    assert event.t_s == 38.5
    branches, skipped = rewind.branches_of(event, o, (2.0, 10.0), 1, 2.0)
    assert {b.offset: b.step for b in branches if b.speaker} == {"2": 19, "10": 15, "start": 0} and not skipped


def test_a_branch_is_rescued_only_with_no_new_loss_an_aircraft_ended_with_another_one_counting():
    from types import SimpleNamespace

    from ts_transformer.experiments import traffic_window_rewind as rewind

    loss = {"kind": "in_trail", "relation": "same"}
    o = _original(["a", "b"], [30, 30], {"a": {"t_s": 40.0, "with": "b", **loss}},
                  episodes=[_episode("a", "b", 40.0, 40.0)])
    (event,) = rewind.events_of(o, 0)
    branch = rewind.Branch(event.number, "b", rewind.PARTNER, "start", 0, 0)
    rows = [{"reward": 0.0, "outcome": "lost_separation", "landed_here": False},
            {"reward": 0.83, "outcome": "landed", "landed_here": True}]          # b landed after a go-around
    said = [np.zeros((30, 6), dtype=np.int64)] * 2
    # b lands, the pair is clear — but a is now ended with a replayed aircraft: not rescued
    run = SimpleNamespace(ended={"a": {"t_s": 60.0, "with": "r", **loss}}, episodes=[_episode("a", "r", 60.0, 60.0)],
                          at_threshold=[])
    got = rewind.branch_result(branch, event, o, ["a", "b"], said, rows, run, 2.0)
    assert (got["pair_cleared"], got["landed"], got["new_losses"], got["rescued"]) == (True, True, ["a"], False)
    run = SimpleNamespace(ended={}, episodes=[], at_threshold=[])
    assert rewind.branch_result(branch, event, o, ["a", "b"], said, rows, run, 2.0)["rescued"]


def test_floats_alike_to_round_off_and_anything_else_differing():
    from ts_transformer.experiments.traffic_window_rewind import float_difference

    assert float_difference({"a": [1.0, "x", None]}, {"a": [1.0 + 1e-9, "x", None]}) == pytest.approx(1e-9)
    assert float_difference(float("nan"), float("nan")) == 0.0
    assert float_difference({"a": 1.0}, {"b": 1.0}) == float("inf")
    assert float_difference([1.0], [1.0, 2.0]) == float("inf")
    assert float_difference("landed", "timeout") == float("inf") and float_difference(None, 1.0) == float("inf")
    assert float_difference(True, 1.0) == float("inf")
    assert float_difference(float("inf"), float("inf")) == 0.0


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
