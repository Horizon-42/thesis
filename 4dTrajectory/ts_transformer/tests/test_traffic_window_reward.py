"""Multi-aircraft M4's second pass (`experiments/traffic_window_reward`, the 7.6 plan): a round's windows selected and
joined, a window round spoken by the speaking processes' plan, the samples with an aircraft trained on as the tuner reads
them, and the select readout in the shape the round choice reads. On the scene-data fixture."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.instructions.words import Words
from ts_transformer.tests.test_traffic_window import LIMIT_S, STEP_S, _airport, _as_drawn, _patch_runner_physics, _pool
from ts_transformer.tests.test_traffic_window_tuner import _reading_model


def _round(tmp_path, monkeypatch):
    """Two windows — f0 and f1 together, f2 an hour later alone — as a window round (the fixture's physics)."""
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.experiments.traffic_window_reward import WindowRound
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.tests.test_traffic_speaking import _batch

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0, 3_600.0], (0, 1, 2))
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    commanded = [("KXXX:f0", "KXXX:f1"), ("KXXX:f2",)]
    keys = [k for keys in commanded for k in keys]
    windows = [window_of(airport, 0.0, c, [LIMIT_S] * len(c), STEP_S) for c in commanded]
    drawn = _as_drawn(windows, [range(0, 2), range(2, 3)], _batch(airport, signals, spec, keys), [LIMIT_S] * 3)
    _patch_runner_physics(monkeypatch, signals, airport.flights.geometry, keys)
    return WindowRound(drawn, ["real", "real"]), airport, spec


def _speaking(spec, airport):
    from ts_transformer.experiments.traffic_window_reward import WindowSpeaking
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params

    return WindowSpeaking(Words(spec), _params(), None, ProcedureMasks.none(), 100_000, {"KXXX": _pool(airport)})


def test_windows_selected_and_joined_keep_their_aircraft_and_an_airport_short_is_refused(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_window_reward import drawn_join, drawn_subset, first_windows_per_airport

    round_, _, _ = _round(tmp_path, monkeypatch)
    drawn = round_.drawn
    flipped = drawn_join([drawn_subset(drawn, [1]), drawn_subset(drawn, [0])])
    assert [w.commanded for w in flipped.windows] == [drawn.windows[1].commanded, drawn.windows[0].commanded]
    assert [list(r) for r in flipped.members] == [[0], [1, 2]]
    assert [s.dataset_id for s in flipped.batch.signals] == ["KXXX:f2", "KXXX:f0", "KXXX:f1"]
    assert flipped.limits == [LIMIT_S] * 3 and flipped.moves == [None] * 3
    assert first_windows_per_airport(drawn, 2) == [0, 1]
    with pytest.raises(ValueError, match=r"3 windows an airport wanted: \{'KXXX': 1\} short"):
        first_windows_per_airport(drawn, 3)


def test_a_window_round_is_spoken_by_its_plan_and_its_trained_samples_are_what_the_tuner_reads(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_window_reward import window_split
    from ts_transformer.experiments.traffic_window_tuner import window_advantages
    from ts_transformer.prior import data as prior_data
    from ts_transformer.prior.data import Split, column_classes

    round_, airport, spec = _round(tmp_path, monkeypatch)
    speaking = _speaking(spec, airport)
    model = _reading_model(spec)
    plan = speaking.plan(round_, 2)
    assert sorted(w for chunk in plan for w in chunk) == [0, 1]
    parts = {n: speaking.speak(model, round_, chunk, n, 2, seed=5, source="train") for n, chunk in enumerate(plan)}
    spoken = speaking.assemble(round_, 2, plan, parts)
    assert len(spoken.rows) == 2 * 3 and all(r["kind"] == "real" for r in spoken.rows)
    assert speaking.fingerprint(round_) == speaking.fingerprint(round_)
    # a batch speaks the same whatever else is spoken
    again = speaking.speak(model, round_, plan[0], 0, 2, seed=5, source="train")
    assert again.rows == parts[0].rows
    rows = spoken.rows
    for k, row in enumerate(rows):                              # a contrast for f0 and f2, none for f1
        row["reward"] = float(row["dataset_id"] != "KXXX:f1" and row["sample"] == 0)
        row["starts_in_a_loss"] = False
    advantages, trained = window_advantages(rows)
    geometry = airport.flights.geometry
    table = Split([], ("KXXX",), prior_data.candidate_table({"KXXX": geometry}, ("KXXX",), 2), (("09",),), ((90.0,),),
                  column_classes(Words(spec), 2), "no-context")
    split, kinds = window_split(round_, spoken, advantages, trained, table, None, spec.step_s)
    assert len(split.windows) == 4 and kinds == ["real"] * 4           # both windows, both samples
    assert split.sentences == len(trained) == 4
    for window, flights, places, gains in zip(split.windows, split.flights, split.trained, split.advantages):
        assert len(flights) == len(window.commanded)
        for m, flight in enumerate(flights):
            if m in places:
                assert flight.asked.any()
            else:
                assert not flight.asked.any()                # f1: read by f0, never trained on
        assert set(np.abs(gains).tolist()) == {0.5}


def test_the_select_readout_is_in_the_shape_the_round_choice_reads(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_reward import guarded_choice, ordering_failures
    from ts_transformer.experiments.traffic_window_reward import side_readout
    from ts_transformer.instructions.words import COLUMNS

    round_, airport, spec = _round(tmp_path, monkeypatch)
    speaking = _speaking(spec, airport)
    model = _reading_model(spec)
    plan = speaking.plan(round_, 2)
    spoken = speaking.assemble(round_, 2, plan, {n: speaking.speak(model, round_, chunk, n, 2, seed=5, source="select")
                                                 for n, chunk in enumerate(plan)})
    readout = {side: side_readout(spoken, round_, spec.step_s, real=side == "real") for side in ("real", "augmented")}
    real = readout["real"]
    assert set(real["free_generation"]["all"]["words_after_first_per_flight"]) == set(COLUMNS)
    assert 0.0 <= real["separation"]["lost_separation"] <= 1.0 and real["reward_by_kind"].keys() == {"real"}
    assert set(real["ordering"]) >= {"time_ratio", "gap_ratio"} and len(real["aircraft"]) == 6
    history = [{"round": n, **{side: {k: v for k, v in readout[side].items() if k != "aircraft"}
                               for side in ("real", "augmented")}} for n in (0, 1)]
    for row in history:                                        # (the fixture's aircraft land after their time limit)
        row["real"]["ordering"] = {"time_ratio": 1.0, "gap_ratio": 1.0}
        row["real"]["free_generation"]["all"]["landed_on_observed_runway"] = 1.0
    assert ordering_failures(history[0]) == []
    labelled = {side: {name: 1.0 for name in COLUMNS} for side in ("real", "augmented")}
    rewards = [{(r["window"], r["dataset_id"], r["sample"]): r["reward"] for r in spoken.rows}] * 2
    kept, excluded = guarded_choice(history, labelled, rewards)
    assert kept == 0 and excluded == {}                        # the same readout twice: no round beats round 0
