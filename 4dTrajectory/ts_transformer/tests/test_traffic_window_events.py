"""Multi-aircraft design §6.6 step 8 item 11: hard events from window rewind runs as windows to train on and read
(`experiments/traffic_window_events`, R37's event windows)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ts_transformer.tests.test_traffic_window import STEP_S
from ts_transformer.tests.test_traffic_window_rewind import _flown_windows


def _branches(key, rescued, offset="start"):
    return [{"speaker": key, "offset": offset, "rescued": r} for r in rescued]


def test_an_event_is_hard_when_its_answered_aircraft_rescued_none_of_its_start_branches():
    from ts_transformer.experiments.traffic_window_events import hard

    record = {"speakers": [["a", "answered"], ["b", "partner"]],
              "branches": _branches("a", [False] * 3) + _branches("b", [True] * 3) + _branches("a", [True], "120")}
    assert hard(record, 3) == "a"                       # the partner's and other offsets' rescues do not count
    assert hard({**record, "branches": _branches("a", [False, True, False])}, 3) is None
    assert hard({**record, "branches": _branches("a", [False] * 2)}, 3) is None          # not all its branches
    assert hard({"speakers": [["a", "both"], ["b", "both"]], "branches": _branches("a", [False] * 3)}, 3) is None


def test_a_round_picks_so_many_hard_events_an_airport_without_replacement():
    from ts_transformer.experiments.traffic_window_events import EventPool, EventScene, pick

    def pool(airports):
        return EventPool(None, [EventScene(n, n, a, "x", ()) for n, a in enumerate(airports)], {})

    pools = [pool(["KAAA"] * 5 + ["KBBB"]), pool(["KAAA"] * 3 + ["KCCC"] * 2)]
    picked = pick(pools, 2, np.random.default_rng(7))
    by_airport = {}
    for p, s in picked:
        by_airport.setdefault(pools[p].scenes[s].airport, []).append((p, s))
    assert {a: len(v) for a, v in by_airport.items()} == {"KAAA": 2, "KBBB": 1, "KCCC": 2}   # KBBB: its one
    assert len(set(picked)) == len(picked)
    assert pick(pools, 2, np.random.default_rng(7)) == picked                                # seeded
    assert len(pick(pools, 100, np.random.default_rng(1))) == 11


def _run(tmp_path, originals, events, *, schema=None, split="train", executor="spec", roles=("answered",)):
    """A rewind run's files holding ``events`` (`events.jsonl` records) over ``originals``' words."""
    from ts_transformer.experiments import traffic_window_rewind as rewind

    run = tmp_path / "rewind"
    run.mkdir(exist_ok=True)
    rewind.write_original_words(run / "original_words.npz", [originals[w] for w in sorted(originals)], STEP_S)
    with (run / "events.jsonl").open("w", encoding="utf-8") as stream:
        for record in events:
            stream.write(json.dumps(record) + "\n")
    (run / "window_rewind.json").write_text(json.dumps({
        "schema": schema or rewind.SCHEMA, "split": split, "instructions": str(tmp_path / "artefact"),
        "executor": {"sha256": executor}, "roles": list(roles), "branches": 3, "windows_per_airport": 4, "seed": 1,
        "augment_seed": None, "drawn": {"KXXX": 4}, "augmenting": None,
        "files": {"events": "events.jsonl", "original_words": "original_words.npz"}}), encoding="utf-8")
    return run


def _pool(tmp_path, monkeypatch, run, drawn, **more):
    from ts_transformer.experiments import traffic_window_events as events

    monkeypatch.setattr(events, "draw_windows", lambda *a, **k: SimpleNamespace(counts=more.get("counts", {"KXXX": 4})))
    monkeypatch.setattr(events, "drawn_windows", lambda draw, *a: drawn)
    spec = SimpleNamespace(step_s=STEP_S)
    return events.event_pool(run, more.get("split", "train"), instructions=tmp_path / "artefact", spec=spec,
                             words=None, airports={}, params=None, executor_sha256=more.get("executor", "spec"),
                             most={}, max_rows=0, windows_alt=None)


def test_a_rewind_run_s_hard_events_become_scenes_the_others_given_their_words(tmp_path, monkeypatch):
    from ts_transformer.experiments import traffic_window_rewind as rewind

    drawn, _, _, _, _, _, originals = _flown_windows(tmp_path, monkeypatch)
    keys = originals[0].keys                                         # f0, f1, f2
    hard = {"event": 0, "window": 0, "airport": "KXXX", "commanded": 3, "augmented": None,
            "speakers": [[keys[1], "answered"], [keys[0], "partner"]], "branches": _branches(keys[1], [False] * 3)}
    soft = {"event": 1, "window": 1, "airport": "KXXX", "commanded": 2, "augmented": None,
            "speakers": [[originals[1].keys[0], "answered"]],
            "branches": _branches(originals[1].keys[0], [False, True, False])}
    run = _run(tmp_path, originals, [hard, soft])
    pool = _pool(tmp_path, monkeypatch, run, drawn)
    (scene,) = pool.scenes
    assert (scene.event, scene.window, scene.answered) == (0, 0, keys[1])
    assert len(pool.drawn.windows) == 1 and pool.drawn.windows[0].commanded == drawn.windows[0].commanded
    assert scene.given[1] is None                                    # the answered aircraft speaks
    for i in (0, 2):
        assert scene.given[i].until == rewind.spoken_steps(originals[0], i, STEP_S)
        assert (scene.given[i].said == originals[0].said[i]).all()
    assert pool.record["events"] == 2 and pool.record["hard"] == {"KXXX": 1}
    # refused: another schema, split, executor, roles without the answered aircraft; a window not coming back alike
    for changes, why in (({"schema": "ts-traffic-window-rewind-v2"}, "schema"), ({"split": "select"}, "train"),
                         ({"executor": "other"}, "executor"), ({"roles": ("partner",)}, "answered")):
        bad = _run(tmp_path, originals, [hard], **changes)
        with pytest.raises(ValueError, match=why):
            _pool(tmp_path, monkeypatch, bad, drawn)
    import dataclasses
    swapped = dataclasses.replace(drawn, windows=[dataclasses.replace(drawn.windows[0],
                                                                      commanded=tuple(reversed(keys)))]
                                  + list(drawn.windows[1:]))
    with pytest.raises(ValueError, match="came back otherwise"):
        _pool(tmp_path, monkeypatch, _run(tmp_path, originals, [hard]), swapped)
    with pytest.raises(ValueError, match="came back otherwise"):              # the record's own window, otherwise
        _pool(tmp_path, monkeypatch, _run(tmp_path, originals, [{**hard, "commanded": 2}]), drawn)
    with pytest.raises(ValueError, match="does not come back"):               # another draw
        _pool(tmp_path, monkeypatch, _run(tmp_path, originals, [hard]), drawn, counts={"KXXX": 3})
    with pytest.raises(ValueError, match="none of its 1 events is hard"):
        _pool(tmp_path, monkeypatch, _run(tmp_path, originals, [soft]), drawn)


def test_an_event_window_flies_the_others_on_their_words_and_trains_only_the_answered_one(tmp_path, monkeypatch):
    """The scene flown as R37 speaks it (`loop_given`: each window K times): the others' rows marked ``given`` and
    their words the original pass's as far as they spoke; the answered aircraft's spoken; only it trained."""
    import torch

    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments import traffic_window_rewind as rewind
    from ts_transformer.experiments.traffic_window_reward import EVENT, event_round, loop_given
    from ts_transformer.experiments.traffic_window_tuner import window_advantages

    drawn, words, params, every, model, masks, originals = _flown_windows(tmp_path, monkeypatch)
    keys = originals[0].keys
    hard = {"event": 0, "window": 0, "airport": "KXXX", "commanded": 3, "augmented": None,
            "speakers": [[keys[1], "answered"]], "branches": _branches(keys[1], [False] * 3)}
    pool = _pool(tmp_path, monkeypatch, _run(tmp_path, originals, [hard]), drawn)
    round_ = event_round([pool], [(0, 0)])
    assert round_.kinds == [EVENT] and list(round_.drawn.windows[0].commanded) == keys
    samples = 2
    given = loop_given(round_, [0], samples)
    assert len(given) == samples * len(keys) and given[1] is None and given[1 + len(keys)] is None
    got = runner.window_sentences(model, round_.drawn, [0], "scene", words, params, None, every, samples,
                                  generator=torch.Generator().manual_seed(3), temperature=1.0, procedure_masks=masks,
                                  given=given)
    for row, record in zip(got.rows, got.records):
        i = keys.index(row["dataset_id"])
        assert row["given"] == (i != 1)
        if row["given"]:
            spoken = rewind.spoken_steps(originals[0], i, STEP_S)
            assert (record.grid[:spoken] == originals[0].said[i][:spoken]).all()
    for row in got.rows:                                          # a contrast for every aircraft, none answerable
        row["reward"], row["starts_in_a_loss"] = float(row["sample"]), False
    advantages, _, trained = window_advantages(got.rows, samples)
    assert {got.rows[k]["dataset_id"] for k in trained} == {keys[1]}
    assert not advantages[[k for k, r in enumerate(got.rows) if r["given"]]].any()


def test_given_in_some_samples_only_is_refused():
    from ts_transformer.experiments.traffic_window_tuner import window_advantages

    rows = [{"window": 0, "dataset_id": "a", "sample": s, "reward": float(s), "starts_in_a_loss": False,
             "probed": False, "forced": None, "given": s == 0} for s in range(2)]
    with pytest.raises(ValueError, match="some samples only"):
        window_advantages(rows, 2)


def test_a_round_holds_a_kind_and_given_lines_per_window():
    from ts_transformer.experiments.traffic_window_reward import WindowRound, side_of

    drawn = SimpleNamespace(windows=["w0", "w1"], members=[range(0, 2), range(2, 3)])
    WindowRound(drawn, ["real", "event"], [None, (None,)])
    with pytest.raises(ValueError, match="per window"):
        WindowRound(drawn, ["real"], [None, None])
    with pytest.raises(ValueError, match="1 given lines for 2 aircraft"):
        WindowRound(drawn, ["real", "event"], [(None,), None])
    assert [side_of(k) for k in ("real", "event", "A", "B", "C")] == ["real", "events"] + ["augmented"] * 3


def test_the_select_hard_events_are_read_over_the_answered_aircraft():
    from ts_transformer.experiments.traffic_window_reward import event_readout

    def row(given, reward, outcome, landed, go_around=None):
        return {"given": given, "reward": reward, "outcome": outcome, "landed_here": landed, "go_around": go_around}

    spoken = SimpleNamespace(rows=[row(True, 1.0, "landed", True), row(False, 0.0, "lost_separation", False),
                                   row(False, 0.62, "landed", True, {"reward": 0.62}), row(True, 0.0, "timeout", False)])
    out = event_readout(spoken)
    assert out["sentences"] == 2 and out["reward"] == pytest.approx(0.31)
    assert (out["landed_here"], out["lost_separation"], out["said_a_go_around"]) == (0.5, 0.5, 0.5)
    assert out["outcomes"] == {"landed": 0.5, "lost_separation": 0.5} and len(out["aircraft"]) == 2
