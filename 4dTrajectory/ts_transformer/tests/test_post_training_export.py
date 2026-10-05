"""Stage C, C11: the Training export of the windows (post-training §8 C11, outline §6) — on A26's one-flight synthetic
artefact (the fixture of `test_post_window_loop`); every write root under tmp."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from ts_transformer.experiments import post_train
from ts_transformer.experiments import post_training_export as export
from ts_transformer.experiments.post_train import KINDS, done_rounds, open_campaign, round_model, run_campaign
from ts_transformer.post import training_files as files
from ts_transformer.post.scene import INSERTED, INSERTED_SUFFIX, REAL
from ts_transformer.tests import test_start
from ts_transformer.tests.test_post_branches import _ahead, _round
from ts_transformer.tests.test_post_train import _context, _settings
from ts_transformer.tests.test_post_window_loop import CPU, DELTA, setup, window_setup  # noqa: F401

#: The window set the frontend's readers and the backend's test read (`stage_c_fixture`).
FIXTURE_SET = "fixture-windows"


def head_of(s):
    """The commanded flight's head as stage A's export gives it (`split_flights`), built here by stage A's functions:
    a synthetic flight has no harvest series, and its observed attitude is the no-airframe one (as stage B's test)."""
    from ts_transformer.autopilot import closed_loop, replay
    from ts_transformer.experiments import training_export as stage_a
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.instructions.labeller.read import read_flight

    words, geometry, stored = s["words"], s["geometry"], s["stored"]
    (signals,) = load_signals(s["directory"], "train")
    reading = read_flight(signals, geometry, words.spec, words)
    attitude = {"headingDeg": signals.track_deg,
                "pathAngleDeg": np.degrees(np.arctan2(signals.vertical_rate_mps, signals.ground_speed_mps)),
                "bankRightDeg": None, "attackDeg": None}
    head = stage_a.flight_head(signals, signals.dataset_id, "train", "vectored", "without go-around", "own", reading,
                               geometry, attitude, words)
    batch, missing = closed_loop.replay_batch(s["batch"], {0: stored}, words)
    assert not missing
    flown, (verdict,) = replay.fly_batch(batch, test_start._params(), words, device=CPU)
    replayed = stage_a.replay_payload(flown, 0, verdict, batch, stored,
                                      batch.inputs(test_start._params().start_rule, CPU).aero_params[0].numpy(),
                                      words.spec, words)
    head["closedLoop"]["4"] = stage_a.closed_loop_payload(stored, replayed, DELTA, geometry, words)
    return head


def _hae(s):
    return {c.ident: -33.0 for c in s["geometry"].candidates}


def test_a_window_s_sentence_its_end_and_its_traffic(setup):
    """The window lost at its first row flown (its own flight inserted 8 s ahead) and the real one, flown by the start
    model: the loss with its other aircraft, the judged outcome and crossing, the track and attitude on the 2 s rows
    from the first predicted step, and the traffic over the window."""
    s = setup
    context = _context(s)
    ahead, real = _ahead(s["windows"][0]), s["windows"][0]
    model = round_model(context, _settings(), s["directory"], None)
    (lost, judged), observed = export.fly_round(model, context, "train", [ahead, real], 2, 1337)
    assert np.array_equal(observed[1], s["stored"].rows.states[: len(observed[1])])     # no move: the stored rows
    assert lost["outcome"] == "lost_separation" and lost["crossing"] is None
    assert lost["end"]["loss"]["other"] == real.commanded.key + INSERTED_SUFFIX and lost["end"]["reward"] == 0.0
    assert lost["end"]["loss"]["step"] == s["stored"].rows.start + 1
    assert judged["outcome"] != "lost_separation" and judged["end"]["loss"] is None
    for said in (lost, judged):
        track = said["track"]
        assert track["rows"] == len(track["eM"]) == len(track["heightMslM"]) == len(said["attitude"]["headingDeg"])
        assert said["rows"] == len(said["words"]) and said["flownFromRow"] == said["startRow"] * 2
    # the lost window's track: the rows flown to the end of the row it ended in, its states there
    assert lost["track"]["rows"] == len(lost["words"]) * 2 + 1
    spy = []
    payload = export.sentence_payload
    try:
        export.sentence_payload = lambda loop, b, result, *a: (spy.append(result.states), payload(loop, b, result, *a))[1]
        export.fly_round(model, context, "train", [ahead], 2, 1337)
    finally:
        export.sentence_payload = payload
    flown_rows = spy[0][lost["flownFromRow"]:]
    assert len(flown_rows) == lost["track"]["rows"]
    assert np.allclose(lost["track"]["heightMslM"], flown_rows[:, 2], atol=0.051)
    (inserted,) = export.traffic_payload(ahead, ahead.commanded.end_s, _hae(s))
    assert inserted["role"] == "inserted" and inserted["shiftS"] == -8.0
    # clipped to the window's span from its row 0; the inserted record ends 8 s before the commanded one's
    assert inserted["tS"][0] == 0.0 and inserted["tS"][-1] == pytest.approx(ahead.commanded.end_s - 8.0 - ahead.row0_s)
    assert np.allclose(np.diff(inserted["tS"]), 2.0)
    assert len(inserted["tS"]) == len(inserted["latDeg"]) == len(inserted["heightMslM"])
    assert export.traffic_payload(real, real.commanded.end_s, _hae(s)) == []          # one flight: nobody else
    window = export.window_payload(ahead, [export.START], [lost], observed[0], _hae(s), s["words"].spec.step_s)
    assert window["kind"] == INSERTED and window["rounds"][0]["round"] == export.START and window["movedStart"] is None
    assert window["firstStepS"] == pytest.approx(ahead.first_step_s - ahead.row0_s)


def test_the_rounds_stand_side_by_side_and_a_foreign_checkpoint_is_refused(setup, tmp_path, monkeypatch):
    s = setup
    context = _context(s)
    settings = _settings(rounds=1, per_kind={**dict.fromkeys(KINDS, 0), REAL: 1})
    model, _ = post_train.start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))
    monkeypatch.setattr(post_train, "speak_round", lambda model, context, windows, settings, round_, directory: (
        torch.save([rewarded], directory / "groups_0.pt"), {"windows": 1})[1])
    monkeypatch.setattr(post_train, "selection_readout", lambda *a, **k: {})
    campaign = tmp_path / "campaign"
    open_campaign(campaign, {"settings": asdict(settings)}, {"head": "x", "dirty": False}, {})
    run_campaign(campaign, settings, context)
    assert done_rounds(campaign) == 1
    ahead = _ahead(s["windows"][0])
    monkeypatch.setattr(export, "chosen_windows", lambda *a, **k: ([ahead], {"pool": 1, "real": 1, "leftOut": {}}))
    monkeypatch.setattr(export, "split_flights", lambda *a, **k: ([head_of(s)], s["geometry"]))
    flights, windows, drawn = export.build_airport(
        context, campaign, settings, s["geometry"].code, [export.START, 0], split="train", per_airport=1,
        kinds=[REAL], seed=1337, params=test_start._params(), hae_minus_msl_m=_hae(s))
    (window,) = windows
    assert [r["round"] for r in window["rounds"]] == [export.START, 0] and flights[0]["haeMinusMslM"] == -33.0
    state = torch.load(campaign / "round_0" / "checkpoint.pt", weights_only=False)
    state["identity"]["seed"] = 7
    torch.save(state, campaign / "round_0" / "checkpoint.pt")
    with pytest.raises(ValueError, match="another base"):
        round_model(context, settings, campaign, 0)


def test_the_draw_of_each_airport_s_windows(setup):
    """Real windows drawn once with the seed; the kinds built from each, a kind it admits none of left out and counted
    (one flight: no A, no D)."""
    s = setup
    context = _context(s)
    windows, drawn = export.chosen_windows(context, "train", s["geometry"].code, 5, list(KINDS), 1337)
    assert [w.kind for w in windows] == [REAL, "B"] and drawn["real"] == drawn["pool"] == 1
    assert drawn["leftOut"] == {INSERTED: 1, "D": 1}
    again, _ = export.chosen_windows(context, "train", s["geometry"].code, 5, list(KINDS), 1337)
    assert [asdict(w.start_move) for w in again] == [asdict(w.start_move) for w in windows]


def test_the_runner_writes_a_set_beside_the_other_indexes_and_refuses_what_it_cannot_trust(setup, tmp_path,
                                                                                            monkeypatch):
    s = setup
    context = _context(s)
    settings = _settings(rounds=1)
    campaign, root = tmp_path / "campaign", tmp_path / "airports"
    inputs = {"prior": "p", "instructions": str(s["directory"]), "executor": str(tmp_path / "executor"),
              "windows": str(tmp_path / "census"), "procedure_root": "c", "settings": asdict(settings), "smoke": True}
    open_campaign(campaign, inputs, {"head": "x", "dirty": False}, {})
    ahead = _ahead(s["windows"][0])
    monkeypatch.setattr(export, "require_conforming_closed_loop",
                        lambda *a: (test_start._params(), {"sha256": "fixture", "checks": {}}, s["words"]))
    monkeypatch.setattr(export, "checked_edges", lambda reference: None)
    monkeypatch.setattr(export, "open_context", lambda *a, **k: context)
    monkeypatch.setattr(export, "chosen_windows", lambda *a, **k: ([ahead], {"pool": 1, "real": 1, "leftOut": {}}))
    monkeypatch.setattr(export, "split_flights", lambda *a, **k: ([head_of(s)], s["geometry"]))
    monkeypatch.setattr(export, "candidate_hae_minus_msl_m", lambda ends, geometry: _hae(s))
    monkeypatch.setattr(export, "repo_relative", lambda path: f"fixture/{path.name}")
    monkeypatch.setattr(export, "git_state", lambda: {"head": "fixture", "dirty": False})
    argv = ["--campaign", str(campaign), "--rounds", "start", "--split", "train", "--set-id", "one", "--root", str(root),
            "--per-airport", "1"]
    with pytest.raises(SystemExit):                                       # a smoke campaign gives only a smoke set
        export.main(argv)
    assert export.main([*argv, "--smoke"]) == 0
    code = s["geometry"].code
    training = root / code / "training"
    entry, sample = files.listed_set(training, code, "one")
    assert sample["schema"] == files.SAMPLE_SCHEMA and entry["windows"] == 1 and entry["flights"] == 1
    assert sample["model"]["rounds"] == [export.START] and sample["windows"][0]["traffic"][0]["role"] == "inserted"
    assert json.loads((training / files.INDEX_FILE).read_text())["schema"] == files.INDEX_SCHEMA
    with pytest.raises(ValueError, match="never overwritten"):
        export.main([*argv, "--smoke"])
    with pytest.raises(SystemExit):                                       # the val days: never from here
        export.main([*argv[:6], "val", *argv[7:], "--smoke"])


def stage_c_fixture(tmp_path, monkeypatch, *, texts=False):
    """A set of three windows of one synthetic flight (A26's artefact, `window_setup`) written by the export itself
    (`main`, into a tmp root): the real window, the window with its own flight inserted 8 s ahead (lost at its first row
    flown) and a window B (its start moved), each flown by the start model; the head as stage A's export gives it
    (`head_of`). The live roots are replaced as in the runner's test; the paths and times written are fixed names.
    Returns the window setup, the set's index and sample as read back, each window's unrounded states (in the set's
    order) and, with ``texts``, the files' texts."""
    from ts_transformer.post.scene import moved_start_window

    s = window_setup(tmp_path, monkeypatch)
    context = _context(s)
    real = s["windows"][0]
    windows = [real, _ahead(real), moved_start_window(real, np.random.default_rng(1), turn_deg=15.0, height_m=300.0,
                                                       speed_scale=0.1)]
    campaign, root = tmp_path / "campaign", tmp_path / "airports"
    inputs = {"prior": "p", "instructions": str(s["directory"]), "executor": str(tmp_path / "executor"),
              "windows": str(tmp_path / "census"), "procedure_root": "c", "settings": asdict(_settings(rounds=1)),
              "smoke": True}
    open_campaign(campaign, inputs, {"head": "fixture", "dirty": False}, {})
    states = []
    payload = export.sentence_payload
    with monkeypatch.context() as patch:
        patch.setattr(export, "sentence_payload", lambda loop, b, result, *a: (
            states.append(result.states), payload(loop, b, result, *a))[1])
        patch.setattr(export, "require_conforming_closed_loop",
                      lambda *a: (test_start._params(), {"sha256": "fixture", "checks": {}}, s["words"]))
        patch.setattr(export, "checked_edges", lambda reference: None)
        patch.setattr(export, "open_context", lambda *a, **k: context)
        patch.setattr(export, "chosen_windows", lambda *a, **k: (windows, {"pool": 1, "real": 1, "leftOut": {}}))
        patch.setattr(export, "split_flights", lambda *a, **k: ([head_of(s)], s["geometry"]))
        patch.setattr(export, "candidate_hae_minus_msl_m", lambda ends, geometry: _hae(s))
        patch.setattr(export, "repo_relative", lambda path: f"fixture/{path.name}")
        patch.setattr(export, "git_state", lambda: {"head": "fixture", "dirty": False})
        patch.setattr(files, "utc_now", lambda: "fixture")
        assert export.main(["--campaign", str(campaign), "--rounds", "start", "--split", "train", "--set-id",
                            FIXTURE_SET, "--root", str(root), "--per-airport", "1", "--smoke"]) == 0
    training = root / s["geometry"].code / "training"
    entry, sample = files.listed_set(training, s["geometry"].code, FIXTURE_SET)
    index = json.loads((training / files.INDEX_FILE).read_text(encoding="utf-8"))
    # the batches fly the windows by flight: put the states back in the set's order (each window its own batch here)
    out = {"setup": s, "root": root, "index": index, "sample": sample, "states": states}
    if texts:
        out["texts"] = {name: (training / name).read_text(encoding="utf-8")
                        for name in (files.INDEX_FILE, f"{FIXTURE_SET}/{files.SAMPLE_FILE}")}
    return out
