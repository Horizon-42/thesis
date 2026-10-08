"""F4, the Training export of stage D (`experiments/multi_training_export.py`; frontend §5.7, §8 F4): on A26's synthetic
flight, a window of two commanded aircraft (`test_post_window_multi._multi`'s: the anchor and a copy of it joining 8 steps
behind) flown by a model of stage D's shape under stage D's loop (its rule of who answers, D145, and its token part,
D152) and readout numbers, written by the shared export (`post_training_export.export_sets`) into stage D's index:
each loss its commanded aircraft answer for, costing W; every other loss holding a commanded aircraft at the states they
flew, answered by none and costing nothing; no loss twice. Written again, it is the frontend's stage D fixture
(``AEROVIZ_WRITE_FIXTURES=1``)."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, fields, replace
from types import SimpleNamespace

import pytest

from ts_transformer.experiments import multi_training_export as multi_export
from ts_transformer.experiments import post_train
from ts_transformer.experiments import post_training_export as export
from ts_transformer.experiments.post_train import open_campaign
from ts_transformer.instructions import training_files as stage_a_files
from ts_transformer.multi.tokens import TOKENS_SCHEMA, TokenPart
from ts_transformer.post import training_files as files
from ts_transformer.post.traffic_attention import add_token_part
from ts_transformer.tests import test_start
from ts_transformer.tests.test_post_train import _context, _settings
from ts_transformer.tests.test_post_training_export import FIXTURES, _hae, head_of
from ts_transformer.tests.test_post_window_loop import _with_module, window_setup

#: The set the frontend's stage D view is tested on.
STAGE_D_SET = "fixture-windows-two"
#: Where its files go (beside stage C's fixtures).
STAGE_D_FIXTURES = FIXTURES.parent / "stage_d"


def two_window(tmp_path, monkeypatch, joins=(0, 8)):
    """A26's flight's window of two commanded aircraft (`test_post_window_multi._multi`, the copy ``joins[1]`` steps
    behind), its context (which starts the window's loop) and a model of stage D's shape (its token part at zero)."""
    from ts_transformer.tests.test_post_window_multi import _multi

    s = window_setup(tmp_path, monkeypatch)
    context = _context(s)
    window, loop_of = _multi(s, joins=joins)
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
    model = _with_module(s["base"])
    add_token_part(model, TokenPart.width, TOKENS_SCHEMA)
    model.eval()                                     # the part added in training mode: the speaker reads eval (D107)
    return s, two, window, model, keys


def stage_d_set(tmp_path, monkeypatch):
    """The synthetic window of two flown and written by stage D's export (module docstring): the texts of the index
    and the sample, keyed by their names under the airport's ``training/``."""
    s, two, window, model, keys = two_window(tmp_path, monkeypatch)
    # stage D's campaign: its start is a stage C round (D164), as `post_train.start_of` writes it
    source = tmp_path / "post_source_fixture"
    open_campaign(source, {"settings": asdict(_settings(rounds=1)), "smoke": True}, {"head": "fixture", "dirty": False}, {})
    (source / "round_0").mkdir()
    (source / "round_0" / "checkpoint.pt").write_bytes(b"the fixture's round 0")
    with monkeypatch.context() as patch:
        patch.setattr(post_train, "repo_relative", lambda path: f"fixture/{path.name}")
        start = post_train.start_of(source, 0, formal=False)
    campaign = tmp_path / "multi_campaign"
    campaign.mkdir()
    (campaign / "campaign.json").write_text(json.dumps({
        "schema": multi_export.MULTI_CAMPAIGN_SCHEMA, "git": {"head": "fixture", "dirty": False},
        "inputs": {"prior": "p", "instructions": str(s["directory"]), "executor": str(tmp_path / "executor"),
                   "windows": str(tmp_path / "census"), "procedure_root": "c", "smoke": True,
                   "settings": {"start": start, "seed": 2027}}}))
    # stage D's settings as its flying reads them: the window's span alone, a batch of its two aircraft
    settings = SimpleNamespace(spans_s=[window.span_s], batch_rows=[2])
    stage = replace(multi_export.STAGE_D, settings_of=lambda record: settings,
                    windows=lambda *a: ([window], {"pool": 1, "real": 1, "leftOut": {}}),
                    model_of=lambda context_, settings_, campaign_, r: model)
    root = tmp_path / "airports"
    with monkeypatch.context() as patch:
        patch.setattr(export, "require_conforming_closed_loop",
                      lambda *a: (test_start._params(), {"sha256": "fixture", "checks": {}}, s["words"]))
        patch.setattr(export, "checked_edges", lambda reference: None)
        patch.setattr(export, "open_context", lambda *a, **k: two)
        patch.setattr(export, "split_flights", lambda *a, **k: ([{**head_of(s), "datasetId": key} for key in keys],
                                                                 s["geometry"]))
        patch.setattr(export, "candidate_hae_minus_msl_m", lambda ends, geometry: _hae(s))
        patch.setattr(export, "repo_relative", lambda path: f"fixture/{path.name}")
        patch.setattr(export, "git_state", lambda: {"head": "fixture", "dirty": False})
        patch.setattr(stage_a_files, "utc_now", lambda: "fixture")
        parser = argparse.ArgumentParser()
        export.add_arguments(parser, stage)
        args = parser.parse_args(["--campaign", str(campaign), "--rounds", "start", "--split", "train", "--set-id",
                                  STAGE_D_SET, "--root", str(root), "--per-airport", "1", "--smoke"])
        assert export.export_sets(parser, args, stage) == 0
    training = root / s["geometry"].code / "training"
    texts = {name: (training / name).read_text(encoding="utf-8")
             for name in (files.MULTI_INDEX_FILE, f"{STAGE_D_SET}/{files.SAMPLE_FILE}")}
    return texts


def test_stage_d_s_set_holds_the_answered_losses_and_every_other_one_dashed(tmp_path, monkeypatch):
    """The set in stage D's index (module docstring): both aircraft in the order they join; the losses the loop's
    aircraft answer cost W; every other loss of a commanded aircraft is answered by none and costs nothing; no (step,
    pair) twice; the source names no speed readout (none of stage D, D136); the index entry is stage D's."""
    texts = stage_d_set(tmp_path, monkeypatch)
    sample = json.loads(texts[f"{STAGE_D_SET}/{files.SAMPLE_FILE}"])
    index = json.loads(texts[files.MULTI_INDEX_FILE])
    (entry,) = index["sets"]
    assert entry["title"].startswith("Stage D · windows") and sample["source"]["speed"] is None
    assert sample["formats"]["campaign"] == multi_export.MULTI_CAMPAIGN_SCHEMA
    (window,) = sample["windows"]
    assert [a["joinS"] for a in window["commanded"]] == [0.0, 32.0]
    (end,) = window["rounds"]
    answered = [loss for loss in end["losses"] if loss["answering"]]
    unanswered = [loss for loss in end["losses"] if not loss["answering"]]
    assert answered and all(loss["costsW"] for loss in answered)
    assert not any(loss["costsW"] for loss in unanswered)
    # the losses between the two while the anchor is silent continue the one it answered: not written again
    assert [loss["step"] for loss in end["losses"]] == [loss["step"] for loss in answered] == [8, 13]
    ids = {a["datasetId"] for a in window["commanded"]}
    assert all(set(loss["aircraft"]) & ids for loss in end["losses"])            # each holds a commanded aircraft
    # an aircraft answers for at most one loss, as the reader requires (D144)
    for aircraft in window["commanded"]:
        assert sum(aircraft["datasetId"] in loss["answering"] for loss in end["losses"]) <= 1
    if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
        for name, text in texts.items():
            (STAGE_D_FIXTURES / name).parent.mkdir(parents=True, exist_ok=True)
            (STAGE_D_FIXTURES / name).write_text(text, encoding="utf-8")
    for name, text in texts.items():
        assert (STAGE_D_FIXTURES / name).read_text(encoding="utf-8") == text, (
            f"{STAGE_D_FIXTURES / name} is not what stage D's export writes now: AEROVIZ_WRITE_FIXTURES=1 writes it again")


def test_a_loss_is_written_once_at_the_step_it_starts(monkeypatch):
    """`round_end`: a pair's loss over successive steps that no aircraft answered is one loss, written at its first
    step; a run holding a step the loop's aircraft answered is that answer's — not written again, even from a step
    before the answer (the responsible aircraft still observed, D145); the same pair again after a gap is another."""
    from ts_transformer.multi.census import JudgedStep, StepLoss

    loss = SimpleNamespace(kind="radar_or_vertical", relation="same", required_m=5556.0, distance_m=3000.0,
                           vertical_m=50.0, wake_known=True)
    by_step = {step: [StepLoss("commanded_commanded", ("a", "b"), loss)] for step in (4, 5, 6, 7, 9, 10)}
    for step in (3, 4):
        by_step.setdefault(step, []).insert(0, StepLoss("recorded_only", ("a", "r"), loss))
    by_step[5].append(StepLoss("recorded_only", ("b", "r"), loss))
    steps = [JudgedStep(step, None, 2, tuple(items)) for step, items in sorted(by_step.items())]
    monkeypatch.setattr(multi_export, "flown_positions", lambda *a: None)
    monkeypatch.setattr(multi_export, "judged_steps", lambda *a: iter(steps))
    monkeypatch.setattr(multi_export, "round_end_payload", lambda ends, keys, window: {"losses": [
        {"step": 5, "aircraft": ["a", "b"], "answering": ["a"]}], "faultySteps": 0})
    flown = SimpleNamespace(records=[SimpleNamespace(key="a"), SimpleNamespace(key="b")], members=[[0, 1]],
                            fault_readings=lambda w: {4: frozenset({"r"})}, speaking=SimpleNamespace(start=0))
    window = SimpleNamespace(step_s=lambda step: 10.0 * step, row0_s=0.0, scene=SimpleNamespace(
        geometry=SimpleNamespace(code="KXXX")))
    context = SimpleNamespace(separations={"KXXX": None}, finals={"KXXX": None}, words=SimpleNamespace(
        spec=SimpleNamespace(step_s=2.0)))
    ends = [SimpleNamespace(states=None, words=None, crossing=None)] * 2
    written = multi_export.round_end(context, flown, 0, window, ends)["losses"]
    assert [(x["step"], sorted(x["aircraft"]), x["answering"]) for x in written] == [
        (3, ["a", "r"], []), (5, ["a", "b"], ["a"]), (5, ["b", "r"], []), (9, ["a", "b"], [])]
    assert written[0]["readsFault"] is False and written[2]["costsW"] is False
    # "r" reads a faulty point at step 4: after the onset of (a, r) at 3, within the 2 Δ before (b, r) at 5
    assert [x["readsFault"] for x in written if not x["answering"]] == [False, True, False]


def test_a_silent_aircraft_the_executor_finished_ends_where_the_judge_ended_it(tmp_path, monkeypatch):
    """`post_training_export.sentence_payload`: an aircraft that answered a loss, went silent (D144) and was flown on by
    the executor to an end of its own (its window not ended at the loss) is written to the judge's outcome, end and
    last state — not to the end of a row past the executor's end — with its reward the loop's 0 and its silent row; its
    timed-out flag is the written outcome's. The copy joins 4 steps behind; stage D's rule answers the first loss only."""
    from ts_transformer.experiments.multi_train import loop_options, readout_numbers
    from ts_transformer.experiments.training_flights import last_state_cycle

    s, two, window, model, keys = two_window(tmp_path, monkeypatch, joins=(0, 4))
    options = loop_options(two)
    real, spent = options["answering"], []

    def once(*args):
        found = real(*args)
        if found and not spent:
            spent.append(True)
            return found
        return frozenset() if spent else found

    flown_of = []

    def end(flown, w, window_, ends):
        flown_of.append(flown)
        return multi_export.round_end(two, flown, w, window_, ends)

    monkeypatch.setattr(export, "split_flights", lambda *a, **k: ([], s["geometry"]))
    flying = export.Flying(batches=lambda windows: [[0]], numbers=lambda place, member: readout_numbers(1337, place, member),
                           options={**options, "answering": once}, end=end)
    ([anchor, _],), (end_,), _ = export.fly_round(model, two, "train", [window], flying)
    (flown,) = flown_of
    assert spent and not flown.speaking.ended[0]                      # silent, then finished by the executor
    judged = flown.speaking.loop.outcome(0)
    assert anchor["silentFromRow"] is not None and anchor["reward"] == 0.0
    assert (anchor["outcome"], anchor["endCycle"]) == (judged.outcome, int(judged.end_row))
    assert anchor["track"]["lastCycle"] == last_state_cycle(judged.outcome, judged.end_row)
    assert anchor["timedOut"] == (anchor["outcome"] == "timeout")
    assert any(keys[0] in loss["answering"] for loss in end_["losses"])


def test_stage_d_s_positions_and_listed_parts_are_the_readout_s(tmp_path, monkeypatch):
    """The positions `round_end` hands the census (`multi.census.judged_steps`) are those `multi_train.window_losses_of`
    builds for the readout, on the same flown window (a MIRROR pinned here); stage D's listed parts (post-training
    D176 (3)) are the readout's numbers (`multi_train.readout_numbers` of the campaign's seed) and a window list's
    field names (`post.window_lists`)."""
    import numpy as np

    from ts_transformer.experiments import multi_train
    from ts_transformer.post.window_lists import IDENTITY_FIELDS, SELECTION_FIELDS

    s, two, window, model, keys = two_window(tmp_path, monkeypatch)
    seen: dict[str, list] = {}
    flown_positions = multi_export.flown_positions

    def capture(name):
        def read(window_, flown, words):
            seen.setdefault(name, []).append(flown)
            return flown_positions(window_, flown, words)
        return read

    monkeypatch.setattr(multi_export, "flown_positions", capture("export"))
    monkeypatch.setattr(multi_train, "flown_positions", capture("readout"))
    settings = SimpleNamespace(spans_s=[window.span_s], batch_rows=[2], seed=2027, select_per_airport=1)
    flying = multi_export.stage_d_flying(two, settings, multi_export.STAGE_D.readout_numbers(settings))
    flown_ends = []
    end = flying.end
    flying = replace(flying, end=lambda flown, w, window_, ends: (flown_ends.append((flown, list(ends))),
                                                                   end(flown, w, window_, ends))[1])
    export.fly_round(model, two, "train", [window], flying)
    ((flown, ends),) = flown_ends
    multi_train.window_losses_of(window, ends, flown.speaking.start, two)
    (ours,), (theirs,) = seen["export"], seen["readout"]
    assert len(ours) == len(theirs) == 2
    for a, b in zip(ours, theirs):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
    # with landed aircraft (a crossing each): each one's landed runway read, in a window of several, on both sides
    class Captured(Exception):
        pass

    def stop(name):
        def read(window_, flown_, words):
            seen.setdefault(name, []).append(flown_)
            raise Captured
        return read

    landed = [replace(e, crossing={"runway_index": k}) for k, e in enumerate(ends)]
    monkeypatch.setattr(multi_export, "flown_positions", stop("export landed"))
    monkeypatch.setattr(multi_train, "flown_positions", stop("readout landed"))
    with pytest.raises(Captured):
        multi_export.round_end(two, flown, 0, window, landed)
    with pytest.raises(Captured):
        multi_train.window_losses_of(window, landed, flown.speaking.start, two)
    (ours,), (theirs,) = seen["export landed"], seen["readout landed"]
    assert [a[3] for a in ours] == [b[3] for b in theirs] == [0, 1]
    ours_numbers = multi_export.STAGE_D.readout_numbers(settings)(3, 1).random(4)
    assert ours_numbers.tolist() == multi_train.readout_numbers(2027, 3, 1).random(4).tolist()
    assert tuple(multi_export.STAGE_D.identity(window)) == IDENTITY_FIELDS["D"]
    assert tuple(multi_export.STAGE_D.selection_fields(SimpleNamespace(seed=2027, select_per_airport=1,
                                                                       spans_s=[300.0]))) == SELECTION_FIELDS["D"]
    assert tuple(export.STAGE_C.identity(window)) == IDENTITY_FIELDS["C"]
    assert tuple(export.STAGE_C.selection_fields(SimpleNamespace(select_seed=7, select_per_airport=1))) == \
        SELECTION_FIELDS["C"]


def test_stage_d_s_export_refuses_a_campaign_of_another_stage(tmp_path, capsys):
    """`export_sets` refuses a campaign record that is not stage D's, by name, before anything is opened."""
    campaign = tmp_path / "c"
    campaign.mkdir()
    (campaign / "campaign.json").write_text(json.dumps({"schema": post_train.CAMPAIGN_SCHEMA, "inputs": {"smoke": True}}))
    parser = argparse.ArgumentParser()
    export.add_arguments(parser, multi_export.STAGE_D)
    args = parser.parse_args(["--campaign", str(campaign), "--rounds", "start", "--split", "select", "--set-id", "x",
                              "--smoke"])
    with pytest.raises(SystemExit):
        export.export_sets(parser, args, multi_export.STAGE_D)
    assert f"not {multi_export.MULTI_CAMPAIGN_SCHEMA}" in capsys.readouterr().err
