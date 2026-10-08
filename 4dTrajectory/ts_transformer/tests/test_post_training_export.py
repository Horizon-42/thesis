"""Stage C, C11: the Training export of the windows (post-training §8 C11, outline §6) — on A26's one-flight synthetic
artefact (the fixture of `test_post_window_loop`); every write root under tmp."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ts_transformer.experiments import post_train
from ts_transformer.experiments import post_training_export as export
from ts_transformer.experiments.post_train import KINDS, done_rounds, open_campaign, round_model, run_campaign
from ts_transformer.instructions import training_files as stage_a_files
from ts_transformer.post import training_files as files
from ts_transformer.instructions.words import HEADING, UNCHANGED
from ts_transformer.post.scene import INSERTED, INSERTED_SUFFIX, REAL
from ts_transformer.tests import test_start
from ts_transformer.tests.test_post_branches import _ahead, _round
from ts_transformer.tests.test_post_train import _context, _settings
from ts_transformer.tests.test_post_window_loop import CPU, DELTA, setup, window_setup  # noqa: F401

#: The window set the frontend's readers and the backend's test read (`stage_c_fixture`).
FIXTURE_SET = "fixture-windows"
#: Where the frontend's readers find it (written by `test_the_frontend_fixtures_are_what_the_export_writes`).
FIXTURES = Path(__file__).resolve().parents[3] / "aeroviz-4d" / "src" / "data" / "__tests__" / "fixtures" / "stage_c"


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
    ([lost], [judged]), (lost_end, judged_end), observed = export.fly_round(model, context, "train", [ahead, real],
                                                                            export.stage_c_flying(2, lambda place, member: export.member_numbers(1337, place, member)))
    assert np.array_equal(observed[1], s["stored"].rows.states[: len(observed[1])])     # no move: the stored rows
    assert lost["outcome"] == "lost_separation" and lost["crossing"] is None and lost["reward"] == 0.0
    # the window's loss: its two aircraft, the commanded one answering for it, and it costs W (D30's reward 0)
    (loss,) = lost_end["losses"]
    assert loss["aircraft"] == [real.commanded.key, real.commanded.key + INSERTED_SUFFIX]
    assert loss["answering"] == [real.commanded.key] and loss["costsW"] is True
    assert loss["step"] == s["stored"].rows.start + 1
    assert judged["outcome"] != "lost_separation" and judged_end["losses"] == []
    assert lost["silentFromRow"] is None and judged["silentFromRow"] is None   # stage C: its window ends at its loss
    for said in (lost, judged):
        track = said["track"]
        assert track["rows"] == len(track["eM"]) == len(track["heightMslM"]) == len(said["attitude"]["headingDeg"])
        assert said["rows"] == len(said["words"]) and said["flownFromRow"] == said["startRow"] * 2
        # the envelopes of its words (D135), each judged from the flown row where it was heard: its said row × Δ / 2 s
        bands = said["envelopes"]["heading"]
        heard = [t * 2 for t, row in enumerate(said["words"]) if row[HEADING] != UNCHANGED]
        assert bands and [band["row"] for band in bands] == heard[: len(bands)]
    # the lost window's track: the rows flown to the end of the row it ended in, its states there
    assert lost["track"]["rows"] == len(lost["words"]) * 2 + 1
    spy = []
    payload = export.sentence_payload
    try:
        export.sentence_payload = lambda loop, b, result, *a: (spy.append(result.states), payload(loop, b, result, *a))[1]
        export.fly_round(model, context, "train", [ahead], export.stage_c_flying(2, lambda place, member: export.member_numbers(1337, place, member)))
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
    window = export.window_payload(ahead, [export.START], [[lost]], [lost_end], observed[0], _hae(s),
                                   s["words"].spec.step_s)
    # the window format of stages C and D (frontend §5.7): stage C's one commanded aircraft, at the window's row 0
    (aircraft,) = window["commanded"]
    assert window["kind"] == INSERTED and window["c"] is None and window["rounds"] == [{"round": export.START, **lost_end}]
    assert aircraft["datasetId"] == ahead.commanded.key and aircraft["joinS"] == 0.0 and aircraft["shiftS"] is None
    assert aircraft["rounds"] == [{"round": export.START, **lost}] and aircraft["movedStart"] is None
    assert aircraft["firstStepS"] == pytest.approx(ahead.first_step_s - ahead.row0_s)


def test_the_rounds_stand_side_by_side_and_a_foreign_checkpoint_is_refused(setup, tmp_path, monkeypatch):
    s = setup
    context = _context(s)
    settings = _settings(rounds=1, per_kind={**dict.fromkeys(KINDS, 0), REAL: 1})
    model, _ = post_train.start_model(context, settings)
    (group,) = _round(s, model, [_ahead(s["windows"][0])]).groups
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))
    monkeypatch.setattr(post_train, "speak_round", lambda model, context, windows, settings, round_, directory, speakers, *, stage: (
        torch.save([rewarded], directory / "groups_0.pt"), {"windows": 1})[1])
    monkeypatch.setattr(post_train, "selection_readout", lambda *a, **k: {})
    campaign = tmp_path / "campaign"
    open_campaign(campaign, {"settings": asdict(settings)}, {"head": "x", "dirty": False}, {})
    run_campaign(campaign, settings, context)
    assert done_rounds(campaign) == 1
    ahead = _ahead(s["windows"][0])
    monkeypatch.setattr(export, "split_flights", lambda *a, **k: ([head_of(s)], s["geometry"]))
    flights, windows = export.window_set(
        context, "train", [ahead], [export.START, 0],
        lambda r: export.STAGE_C.model_of(context, settings, campaign, r), export.STAGE_C.flying(
            context, settings, export.STAGE_C.drawn_numbers(settings, 1337)),
        test_start._params(), _hae(s))
    (window,) = windows
    assert [r["round"] for r in window["rounds"]] == [export.START, 0] and flights[0]["haeMinusMslM"] == -33.0
    assert [r["round"] for r in window["commanded"][0]["rounds"]] == [export.START, 0]
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
    entry, sample = files.FILES.listed_set(training, code, "one")
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
        patch.setattr(stage_a_files, "utc_now", lambda: "fixture")   # the one writer's
        assert export.main(["--campaign", str(campaign), "--rounds", "start", "--split", "train", "--set-id",
                            FIXTURE_SET, "--root", str(root), "--per-airport", "1",
                            "--smoke"]) == 0
    training = root / s["geometry"].code / "training"
    entry, sample = files.FILES.listed_set(training, s["geometry"].code, FIXTURE_SET)
    index = json.loads((training / files.INDEX_FILE).read_text(encoding="utf-8"))
    # the batches fly the windows by flight: put the states back in the set's order (each window its own batch here)
    out = {"setup": s, "root": root, "index": index, "sample": sample, "states": states}
    if texts:
        out["texts"] = {name: (training / name).read_text(encoding="utf-8")
                        for name in (files.INDEX_FILE, f"{FIXTURE_SET}/{files.SAMPLE_FILE}")}
    return out


def listed_export(tmp_path, monkeypatch, listed, *more):
    """Stage C's export (`main`) of the synthetic flight's selection window (`_context`: its select days are its train
    days; one window) from the window list ``listed`` (`post.window_lists.WindowList`, written to a file), with the
    arguments ``more``; the stand-ins of `stage_c_fixture`. Returns the set's sample, the window setup, the context, the
    campaign's settings, and the model at its start."""
    from ts_transformer.post import window_lists

    s = window_setup(tmp_path, monkeypatch)
    context = _context(s)
    # a select seed that is not the draw's default, and the flight's window second among two selection windows (its
    # own flight inserted ahead first), so that a listed window flies with its place's readout numbers or the test fails
    settings = _settings(rounds=1, select_seed=7)
    real = s["windows"][0]
    selection = [_ahead(real), real]
    campaign, root, path = tmp_path / "campaign", tmp_path / "airports", tmp_path / "list.json"
    with monkeypatch.context() as patch:
        patch.setattr(window_lists, "utc_now", lambda: "fixture")              # the list's sha256 is the fixture's
        window_lists.write_window_list(path, listed(s))
    open_campaign(campaign, {"prior": "p", "instructions": str(s["directory"]), "executor": str(tmp_path / "executor"),
                             "windows": str(tmp_path / "census"), "procedure_root": "c", "settings": asdict(settings),
                             "smoke": True}, {"head": "fixture", "dirty": False}, {})
    with monkeypatch.context() as patch:
        patch.setattr(export, "require_conforming_closed_loop",
                      lambda *a: (test_start._params(), {"sha256": "fixture", "checks": {}}, s["words"]))
        patch.setattr(export, "checked_edges", lambda reference: None)
        patch.setattr(export, "open_context", lambda *a, **k: context)
        patch.setattr(export, "selection_windows", lambda *a: selection)
        patch.setattr(export, "split_flights", lambda *a, **k: ([head_of(s)], s["geometry"]))
        patch.setattr(export, "candidate_hae_minus_msl_m", lambda ends, geometry: _hae(s))
        patch.setattr(export, "repo_relative", lambda path_: f"fixture/{path_.name}")
        patch.setattr(export, "git_state", lambda: {"head": "fixture", "dirty": False})
        patch.setattr(stage_a_files, "utc_now", lambda: "fixture")
        export.main(["--campaign", str(campaign), "--rounds", "start", "--split", "select", "--set-id", "listed",
                     "--root", str(root), "--windows", str(path),
                            "--smoke", *more])
    training = root / s["geometry"].code / "training"
    _, sample = files.FILES.listed_set(training, s["geometry"].code, "listed")
    texts = {name: (training / name).read_text(encoding="utf-8") for name in (files.INDEX_FILE, f"listed/{files.SAMPLE_FILE}")}
    return sample, s, context, settings, texts, selection


def _the_list(s, **changed):
    """A window list naming the synthetic flight's selection window at place 1 (its identity; `listed_export`'s
    selection), ``changed`` its fields."""
    from ts_transformer.post.window_lists import ListedWindow, WindowList

    (window,) = s["windows"]
    item = ListedWindow(1, s["geometry"].code, {"flight": window.commanded.key, "row0_s": window.row0_s,
                                                "kind": window.kind}, {"outcome": "lost_separation"})
    values = {"stage": "C", "split": "select", "selection": {"select_seed": 7, "per_airport": 1},
              "chose": "the select windows that lost separation in the fixture's read", "readouts": (), "windows": (item,)}
    return WindowList(**{**values, **changed})


def test_a_listed_set_flies_each_window_as_its_selection_readout_did(tmp_path, monkeypatch):
    """D176 (3): a set of a window list holds the list's windows, each flown with its selection readout's numbers — the
    sentence the readout's batch says for it (`post_train.read_batch`, the same model and window alone in its batch,
    so post-training §6.4's boundary does not enter) —, and its cohort is the list's: path, sha256, sentence, counts and
    select seed."""
    from ts_transformer.io_utils import file_sha256

    sample, s, context, settings, texts, selection = listed_export(tmp_path, monkeypatch, _the_list)
    (window,) = sample["windows"]
    said = window["commanded"][0]["rounds"][0]
    (read,) = post_train.read_batch(round_model(context, settings, tmp_path / "campaign", None), context, selection,
                                    [1], settings, "select")
    assert said["words"] == read.words.astype(int).tolist() and said["outcome"] == read.outcome
    assert sample["cohort"] == {"form": "listed", "split": "select", "list": "fixture/list.json",
                                "sha256": file_sha256(tmp_path / "list.json"),
                                "chose": "the select windows that lost separation in the fixture's read",
                                "listCount": 1, "selectSeed": 7, "windows": 1}
    # written again, the frontend's fixture of a listed set (``AEROVIZ_WRITE_FIXTURES=1``)
    listed_fixtures = FIXTURES.parent / "stage_c_listed"
    if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
        for name, text in texts.items():
            (listed_fixtures / name).parent.mkdir(parents=True, exist_ok=True)
            (listed_fixtures / name).write_text(text, encoding="utf-8")
    for name, text in texts.items():
        assert (listed_fixtures / name).read_text(encoding="utf-8") == text, (
            f"{listed_fixtures / name} is not what the export writes now: AEROVIZ_WRITE_FIXTURES=1 writes it again")


@pytest.mark.parametrize("listed, more, says", [
    (lambda s: _the_list(s, stage="D", selection={"select_seed": 7, "per_airport": 1, "spans_s": [300.0]},
                         windows=(replace(_the_list(s).windows[0], identity={
                             "flight": s["windows"][0].commanded.key, "row0_s": s["windows"][0].row0_s,
                             "span_s": 300.0}),)), (), "of stage D"),
    (lambda s: _the_list(s, selection={"select_seed": 1337, "per_airport": 1}), (), "indexes the selection"),
    (lambda s: _the_list(s, windows=(replace(_the_list(s).windows[0], place=3),)), (), "selection windows"),
    (lambda s: _the_list(s, windows=(replace(_the_list(s).windows[0], place=0),)), (), "is not the campaign's"),
    (lambda s: _the_list(s, windows=(replace(_the_list(s).windows[0], airport="KYYY"),)), (), "is not the campaign's"),
    (lambda s: _the_list(s, windows=(replace(_the_list(s).windows[0], identity={
        **_the_list(s).windows[0].identity, "row0_s": 0.0}),)), (), "is not the campaign's"),
    (_the_list, ("--split", "train"), "the set is of train windows"),
    (_the_list, ("--per-airport", "2"), "refused with --per-airport"),
    (_the_list, ("--seed", "2", "--kinds", "real"), "refused with --seed, --kinds"),
    (_the_list, ("--airports", "KXXX"), "refused with --airports"),
])
def test_a_window_list_is_refused_by_name(tmp_path, monkeypatch, capsys, listed, more, says):
    """D176 (3): a list of another stage or selection, a window not at its place (none there, another window there,
    another airport, another identity), and a list given with the draw's arguments or --airports (a listed set's
    airports are the list's), each refused by name before anything is written."""
    with pytest.raises(SystemExit):
        listed_export(tmp_path, monkeypatch, listed, *more)
    assert says in capsys.readouterr().err


def test_a_drawn_set_says_its_cohort_is_drawn(tmp_path, monkeypatch):
    """Frontend §5.7 (D176): a drawn set's cohort is the form ``drawn`` with the draw's fields, as before."""
    cohort = stage_c_fixture(tmp_path, monkeypatch)["sample"]["cohort"]
    assert cohort["form"] == "drawn" and (cohort["perAirport"], cohort["seed"], cohort["kinds"]) == (1, 1337, ["real"])


def test_the_frontend_fixtures_are_what_the_export_writes(tmp_path, monkeypatch):
    """The fixture the frontend's readers are tested on is the export's own output today, the files its writers write (the
    sample by the export, the index by `TrainingFiles.write_set`); a change of the export moves it, and this test says so
    until it is written again (``AEROVIZ_WRITE_FIXTURES=1``)."""
    texts = stage_c_fixture(tmp_path, monkeypatch, texts=True)["texts"]
    if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
        for name, text in texts.items():
            (FIXTURES / name).parent.mkdir(parents=True, exist_ok=True)
            (FIXTURES / name).write_text(text, encoding="utf-8")
    for name, text in texts.items():
        assert (FIXTURES / name).read_text(encoding="utf-8") == text, (
            f"{FIXTURES / name} is not what the export writes now: AEROVIZ_WRITE_FIXTURES=1 writes it again")


def test_a_loss_two_commanded_aircraft_answer_is_written_once_with_both(setup):
    """`round_end_payload`: two commanded aircraft that answer the same loss (one step, one pair) give one loss, both
    answering, reading a fault when either one's other aircraft read one; another step of the pair is another loss."""
    from types import SimpleNamespace

    s = setup
    (window,) = s["windows"]
    loss = SimpleNamespace(kind="radar_or_vertical", relation="same", required_m=5556.0, distance_m=3000.0,
                           vertical_m=50.0, wake_known=True)

    def end(other, step, fault):
        return SimpleNamespace(loss=loss, loss_step=step, other=other, loss_reads_fault=fault, faulty_steps=0)

    written = export.round_end_payload([end("b", 9, False), end("a", 9, True), end("a", 12, False)], ["a", "b", "c"],
                                       window)
    first, second = written["losses"]
    assert first["aircraft"] == ["a", "b"] and first["answering"] == ["a", "b"] and first["readsFault"] is True
    assert second["step"] == 12 and second["answering"] == ["c"]


#: The set of §8 F3's fixture: stage C's export on a synthetic window of two commanded aircraft.
TWO_SET = "fixture-windows-two"


def stage_c_two_fixture(tmp_path, monkeypatch):
    """Stage C's export (`main`) on a synthetic window of two commanded aircraft (post-training §9
    items 1, 3, 8; `test_post_window_multi._multi`'s window: the anchor and a copy of it under its own key joining 8
    steps behind it, on the same path), flown by the start model under stage C's loop and rule of who answers — the
    anchor answers its loss with the copy and is silent from there, flying on; the files as `stage_c_fixture` writes
    them (the copy's head is the anchor's under the copy's key). Returns the texts of the set's index and sample."""
    from dataclasses import fields

    from ts_transformer.tests.test_post_window_multi import _multi

    s = window_setup(tmp_path, monkeypatch)
    context = _context(s)
    window, loop_of = _multi(s, joins=(0, 8))
    parts = loop_of.parts
    keys = [record.key for record in window.commanded_all]
    (signal,) = context.splits["train"]["signals"].values()
    data = {**context.splits["train"], "windows": [window], "sentences": parts["sentences"], "flights": parts["flights"],
            "signals": dict.fromkeys(keys, signal)}

    class TwoContext(type(context)):
        def start_loop(self, split, windows):
            return parts["start"]

    two = TwoContext(**{**{f.name: getattr(context, f.name) for f in fields(context)},
                        "splits": {"train": data, "select": data}, "rosters": parts["rosters"]})
    head = head_of(s)
    campaign, root = tmp_path / "campaign", tmp_path / "airports_two"
    inputs = {"prior": "p", "instructions": str(s["directory"]), "executor": str(tmp_path / "executor"),
              "windows": str(tmp_path / "census"), "procedure_root": "c", "settings": asdict(_settings(rounds=1)),
              "smoke": True}
    open_campaign(campaign, inputs, {"head": "fixture", "dirty": False}, {})
    with monkeypatch.context() as patch:
        patch.setattr(export, "require_conforming_closed_loop",
                      lambda *a: (test_start._params(), {"sha256": "fixture", "checks": {}}, s["words"]))
        patch.setattr(export, "checked_edges", lambda reference: None)
        patch.setattr(export, "open_context", lambda *a, **k: two)
        patch.setattr(export, "chosen_windows", lambda *a, **k: ([window], {"pool": 1, "real": 1, "leftOut": {}}))
        patch.setattr(export, "split_flights", lambda *a, **k: ([{**head, "datasetId": key} for key in keys],
                                                                 s["geometry"]))
        patch.setattr(export, "candidate_hae_minus_msl_m", lambda ends, geometry: _hae(s))
        patch.setattr(export, "repo_relative", lambda path: f"fixture/{path.name}")
        patch.setattr(export, "git_state", lambda: {"head": "fixture", "dirty": False})
        patch.setattr(stage_a_files, "utc_now", lambda: "fixture")
        assert export.main(["--campaign", str(campaign), "--rounds", "start", "--split", "train", "--set-id",
                            TWO_SET, "--root", str(root), "--per-airport", "1",
                            "--smoke"]) == 0
    training = root / s["geometry"].code / "training"
    return {name: (training / name).read_text(encoding="utf-8")
            for name in (files.INDEX_FILE, f"{TWO_SET}/{files.SAMPLE_FILE}")}


def test_a_window_of_two_commanded_aircraft_is_written_with_both_its_silence_and_its_loss_by_pair(tmp_path, monkeypatch):
    """The two-aircraft window as the set holds it (frontend §5.7): both commanded aircraft in the order they join
    (the copy 8 steps, 32 s, after the window's row 0), each with its own sentence; the anchor answers its loss with
    the copy, so it is silent from that loss's row and flies on (outcome lost separation, reward 0), and the loss is
    written once, the anchor answering; the copy is no traffic; written again it is the frontend's fixture
    (`AEROVIZ_WRITE_FIXTURES=1`)."""
    texts = stage_c_two_fixture(tmp_path, monkeypatch)
    sample = json.loads(texts[f"{TWO_SET}/{files.SAMPLE_FILE}"])
    (window,) = sample["windows"]
    anchor, copy = window["commanded"]
    assert [anchor["joinS"], copy["joinS"]] == [0.0, 8 * DELTA] and copy["firstStepS"] - anchor["firstStepS"] == 8 * DELTA
    assert copy["datasetId"] not in {t["key"] for t in window["traffic"]}
    (said,) = anchor["rounds"]
    (end,) = window["rounds"]
    answered = [loss for loss in end["losses"] if anchor["datasetId"] in loss["answering"]]
    assert len(answered) == 1 and set(answered[0]["aircraft"]) == {anchor["datasetId"], copy["datasetId"]}
    assert said["reward"] == 0.0 and said["silentFromRow"] is not None and said["silentFromRow"] < said["rows"]


START_FIXTURE = "start_from_round.json"


def test_the_frontend_fixture_of_a_start_from_a_round_is_what_post_train_writes(tmp_path, monkeypatch):
    """The setting of a start from another campaign's round (`Settings.start`, D162) as `post_train.start_of` writes it
    — the value the export copies into a set's ``model.settings.start``: the frontend's reader is tested on it (frontend
    §4.3), so a change of its keys moves this fixture (``AEROVIZ_WRITE_FIXTURES=1`` writes it again). The source is a
    smoke campaign with round 0 done (a checkpoint of fixed bytes: `start_of` only names and hashes it)."""
    source = tmp_path / "post_source_fixture"
    open_campaign(source, {"settings": asdict(_settings(rounds=1)), "smoke": True}, {"head": "fixture", "dirty": False}, {})
    (source / "round_0").mkdir()
    (source / "round_0" / "checkpoint.pt").write_bytes(b"the fixture's round 0")
    monkeypatch.setattr(post_train, "repo_relative", lambda path: f"fixture/{path.name}")
    text = json.dumps({"start": post_train.start_of(source, 0, formal=False)}, indent=2) + "\n"
    if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
        (FIXTURES / START_FIXTURE).write_text(text, encoding="utf-8")
    assert (FIXTURES / START_FIXTURE).read_text(encoding="utf-8") == text, (
        f"{FIXTURES / START_FIXTURE} is not what post_train.start_of writes now: AEROVIZ_WRITE_FIXTURES=1 writes it again")
