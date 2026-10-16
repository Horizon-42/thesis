"""Stage D (multi-aircraft control D211): the diagnostic readout — on the stage D campaign of one round of
`test_multi_compare` — and the fields of a loss of a commanded aircraft after the anchor, on a synthetic window of a
two-runway airport. Every write root under tmp."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.experiments import multi_diagnose as diagnose
from ts_transformer.experiments import multi_train, post_diagnose, post_train
from ts_transformer.experiments.multi_train import MULTI_CLAIM_READER, MultiSettings, stage_d
from ts_transformer.experiments.post_window_loop import LANDED, LOST_SEPARATION
from ts_transformer.instructions.labeller.interval import OBSERVATION_S
from ts_transformer.instructions.words import COLUMNS, RUNWAY, UNCHANGED
from ts_transformer.multi.joint import joint_modules
from ts_transformer.multi.windows import REAL_KIND
from ts_transformer.post.scene import REAL, Joined, Scene, Window
from ts_transformer.post.traffic_attention import traffic_modules
from ts_transformer.tests.test_multi_compare import GIT, campaign  # noqa: F401
from ts_transformer.tests.test_multi_train import _multi_settings
from ts_transformer.tests.test_post_branches import _ahead
from ts_transformer.tests.test_post_train import _context
from ts_transformer.tests.test_post_diagnose import DELTA, FINALS, GEOMETRY, SEPARATION, STEP, _landings, _record
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401

INTENT = {"G0": "on below start in lost separation with the interval excluding 0: the rounds changed it"}


@pytest.fixture
def world(setup, campaign, monkeypatch, tmp_path):  # noqa: F811
    """The campaign of `test_multi_compare` in the world of `_world`."""
    out, context, settings, opened = campaign
    return _world(setup, out, context, settings, opened, monkeypatch, tmp_path)


@pytest.fixture
def world_v2(setup, monkeypatch, tmp_path):  # noqa: F811
    """A stage D campaign of version 2 (D212) of one round from the base (`test_multi_joint`'s), in the world of
    `world`."""
    s = setup
    context = _context(s)
    window = replace(_ahead(s["windows"][0]), span_s=1.0)
    settings = replace(_multi_settings(rounds=1, start=None, traffic_hidden=16, traffic_heads=4), joint_heads=4)
    out = tmp_path / "multi_v2"
    inputs = {"prior": str(tmp_path / "prior"), "instructions": str(s["directory"]),
              "executor": str(tmp_path / "executor"), "windows": str(tmp_path / "census"),
              "procedure_root": str(tmp_path / "cifp"), "settings": asdict(settings), "smoke": False}
    with monkeypatch.context() as patch:
        patch.setattr(multi_train, "draw_windows", lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
        post_train.open_campaign(out, inputs, GIT, {}, schema=settings.campaign_schema, reader=MULTI_CLAIM_READER,
                                 settings_type=MultiSettings)
        post_train.run_campaign(out, settings, context, stage=stage_d())
    return _world(s, out, context, settings, [], monkeypatch, tmp_path)


def _world(setup, out, context, settings, opened, monkeypatch, tmp_path):  # noqa: F811
    """The campaign at ``out`` with this runner's checks stubbed, its context ``context``, its roster landings the
    setup's (the runner reads them itself, D175), the intent written."""
    monkeypatch.setattr(diagnose, "git_state", lambda: GIT)
    monkeypatch.setattr(diagnose, "require_conforming_closed_loop", lambda *a: (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(diagnose, "checked_edges", lambda path: None)
    monkeypatch.setattr(diagnose, "open_context", lambda *a, **k: (opened.append((k["splits"], k["data"], k["formal"])),
                                                                   context)[1])
    rosters = {setup["geometry"].code: setup["roster"]}
    monkeypatch.setattr(diagnose, "airport_landings", lambda geometries, days: rosters)
    intent = tmp_path / "intent.json"
    intent.write_text(json.dumps(INTENT))
    return SimpleNamespace(out=out, context=context, settings=settings, opened=opened, intent=intent)


def _argv(w, out, *more):
    return ["--campaign", str(w.out), "--round", "0", "--intent", str(w.intent), "--out", str(out), "--device", "cpu",
            *more]


def test_a_round_is_read_on_off_and_from_its_start_and_the_outputs_are_sealed(world, tmp_path, monkeypatch):
    """The select days opened formal without data; the read on is the round's readout (checked); then off and the
    campaign's start; a line a window and read, its lost aircraft with their fields; the summary by airport and span
    with its three paired differences; the failures; the directory sealed."""
    w = world
    out = tmp_path / "diagnose"
    models, zeroed = [], []
    read, off = diagnose.selection_read, diagnose.traffic_off
    monkeypatch.setattr(diagnose, "selection_read", lambda model, *a, **k: (models.append(model), read(model, *a, **k))[1])
    monkeypatch.setattr(diagnose, "traffic_off", lambda model: (zeroed.append((len(models), model)), off(model))[1])
    assert diagnose.main(_argv(w, out)) == 0
    assert w.opened == [(("select",), False, True)]
    # off zeroes the round's model after its read on and reads that model; the start is another model
    assert len(models) == 3 and zeroed == [(1, models[0])]
    assert models[1] is models[0] and models[2] is not models[0]
    windows = stage_d().selection(w.context, w.settings, "select")
    lines = [json.loads(x) for x in (out / "per_window.jsonl").read_text().splitlines()]
    assert [x["read"] for x in lines] == [r for r in diagnose.READS for _ in windows]
    assert [x["place"] for x in lines] == list(range(len(windows))) * 3
    for line in lines:
        assert [item["member"] for item in line["lost"]] == [m for m, o in enumerate(line["outcomes"])
                                                            if o == LOST_SEPARATION]
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["reads"]) == set(diagnose.READS)
    assert set(summary["differences"]) == {"on-off", "on-start", "off-start"}
    held = json.loads((w.out / "round_0" / "round.json").read_text())["selection_readout"]
    on = summary["reads"]["on"]
    assert on["all"]["aircraft"] == sum(airport["all"]["all"]["aircraft"] for airport in held.values())
    assert set(on["spans"]) <= {"1", "2", "1.0", "2.0"} and set(on["airports"]) == set(held)
    # the fixture's round made no update: its start (`Stage.start`, a round of stage C) reads as the round does
    assert summary["reads"]["start"] == on
    assert [x["outcomes"] for x in lines if x["read"] == "start"] == [x["outcomes"] for x in lines if x["read"] == "on"]
    landed = summary["differences"]["on-start"]["all"]["landed"]
    assert landed["interval"][0] <= landed["difference"] <= landed["interval"][1]
    failures = json.loads((out / "failures.json").read_text())
    assert set(failures) == {"schema", *diagnose.READS} and "spans" in failures["on"]
    config = json.loads((out / "config.json").read_text())
    assert config["schema"] == diagnose.MULTI_DIAGNOSE_SCHEMA and config["reads"] == list(diagnose.READS)
    assert config["start"] == json.loads((w.out / "campaign.json").read_text())["inputs"]["settings"]["start"]
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert {line.split("  ")[1] for line in sums} == {"intent.json", "config.json", "per_window.jsonl", "summary.json",
                                                      "failures.json", "log.jsonl"}
    assert all(not os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for p in (out, *out.iterdir()))


def test_a_read_on_that_is_not_the_round_s_readout_stops_the_read(world, tmp_path):
    w = world
    path = w.out / "round_0" / "round.json"
    held = json.loads(path.read_text())
    code = next(iter(held["selection_readout"]))
    held["selection_readout"][code]["all"]["all"]["reward_sum"] += 1.0
    path.write_text(json.dumps(held))
    with pytest.raises(SystemExit, match="not the round's selection readout"):
        diagnose.main(_argv(w, tmp_path / "diagnose"))
    assert not (tmp_path / "diagnose" / "summary.json").exists()


def test_the_options_are_refused_by_name_before_anything_is_read(world, tmp_path, capsys):
    w = world
    (tmp_path / "taken").mkdir()
    (tmp_path / "list.json").write_text("[1]")
    for argv, message in (([*_argv(w, tmp_path / "d"), "--round", "1"], "holds the checkpoints of rounds 0–0"),
                          (_argv(w, tmp_path / "taken"), "writes into a new directory"),
                          (["--campaign", str(w.out), "--round", "0", "--intent", str(tmp_path / "list.json"),
                            "--out", str(tmp_path / "d"), "--device", "cpu"], "the decision rules are a JSON object")):
        with pytest.raises(SystemExit):
            diagnose.main(argv)
        assert message in capsys.readouterr().err
    assert w.opened == [] and not (tmp_path / "d").exists()


def test_a_round_of_version_2_is_read_with_its_joint_modules_off_too(world_v2, tmp_path, monkeypatch):
    """D216: a version 2 round is read on, with its joint modules off, with its traffic modules off too, and from its
    start; the read on is the round's readout (checked); the summary holds the difference on − joint_off besides
    version 1's three."""
    w = world_v2
    out = tmp_path / "diagnose"
    models, zeroed = [], []

    def zero(modules):
        return all(not m.out.weight.any() and not m.out.bias.any() for m in modules)

    read, joint, traffic = diagnose.selection_read, diagnose.joint_off, diagnose.traffic_off
    monkeypatch.setattr(diagnose, "selection_read", lambda model, *a, **k: (
        models.append((model, zero(joint_modules(model)), zero(traffic_modules(model)))), read(model, *a, **k))[1])
    monkeypatch.setattr(diagnose, "joint_off", lambda model: (zeroed.append(("joint", len(models), model)),
                                                              joint(model))[1])
    monkeypatch.setattr(diagnose, "traffic_off", lambda model: (zeroed.append(("traffic", len(models), model)),
                                                                traffic(model))[1])
    assert diagnose.main(_argv(w, out)) == 0
    assert [m for m, *_ in models[:3]] == [models[0][0]] * 3 and models[3][0] is not models[0][0]
    # the joint modules off after the read on, the traffic modules too after the read joint_off: off reads no other
    # aircraft at all (the fixture's round made no update, so the zero checks below hold either way)
    assert zeroed == [("joint", 1, models[0][0]), ("traffic", 2, models[0][0])]
    assert models[1][1] and models[2][1:] == (True, True) and models[3][1]
    lines = [json.loads(x) for x in (out / "per_window.jsonl").read_text().splitlines()]
    windows = stage_d().selection(w.context, w.settings, "select")
    assert [x["read"] for x in lines] == [r for r in diagnose.READS_V2 for _ in windows]
    summary = json.loads((out / "summary.json").read_text())
    assert list(summary["reads"]) == list(diagnose.READS_V2)
    assert set(summary["differences"]) == {"on-off", "on-start", "off-start", "on-joint_off"}
    assert json.loads((out / "config.json").read_text())["reads"] == list(diagnose.READS_V2)
    assert set(json.loads((out / "failures.json").read_text())) == {"schema", *diagnose.READS_V2}


def test_joint_off_zeroes_the_joint_modules_output_layers_only(world_v2):
    w = world_v2
    model = multi_train.round_model(w.context, w.settings, w.out, 0)
    for module in joint_modules(model):
        torch.nn.init.ones_(module.out.weight)
        torch.nn.init.ones_(module.out.bias)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert diagnose.joint_off(model) is model
    after = model.state_dict()
    for name, value in after.items():
        if ".beside.out." in name:
            assert not value.any(), name
        else:
            assert torch.equal(before[name], value), name


# ---- the fields of a commanded aircraft after the anchor, on the two-runway airport of `test_post_diagnose`
def _joined_window(*others, join_steps=3):
    """The anchor 40 km out and a commanded aircraft joining ``join_steps`` Δ later 30 km out on runway 09's final,
    with the recorded ``others``."""
    anchor = _record("anchor", -40_000.0, 100.0)
    late = _record("late", -30_000.0, 100.0)
    late = replace(late, first_step_s=late.first_step_s + join_steps * DELTA, start_s=late.start_s + join_steps * DELTA)
    return Window(REAL, anchor, 0, Scene(GEOMETRY, "select", (anchor, late, *others), DELTA),
                  joined=(Joined(late, 1),), span_s=600.0)


def _end(window, member, other, own_loss_step, runway=0):
    """The end of ``window``'s commanded aircraft ``member``: its record's states from its own row 0 (2 s rows), the
    words of runway ``runway`` at each of its Δ rows, lost against ``other`` at its own Δ row ``own_loss_step``."""
    record = window.commanded_all[member]
    every = int(round(DELTA / STEP))
    rows = own_loss_step * every + 1
    r0 = int(round((record.first_step_s - OBSERVATION_S - record.start_s) / STEP))
    states = np.stack([record.e_m[r0:r0 + rows], record.n_m[r0:r0 + rows], record.height_m[r0:r0 + rows]], axis=1)
    words = np.full((own_loss_step, len(COLUMNS)), UNCHANGED)
    words[:, RUNWAY] = runway
    join = int(window.join_steps()[member])
    return SimpleNamespace(index=member, outcome=LOST_SEPARATION, other=other, loss_step=own_loss_step + join,
                           words=words, states=states)


def test_a_joined_aircraft_s_fields_are_read_on_its_own_rows_and_a_commanded_other_is_named_so():
    ahead = _record("ahead", -20_000.0, 100.0)
    window = _joined_window(ahead)
    assert list(window.join_steps()) == [0, 3]
    first = int(round(OBSERVATION_S / DELTA))
    end = _end(window, 1, ahead.key, first + 5)
    fields = post_diagnose.loss_fields(window, 1, end, SEPARATION, FINALS, _landings(ahead), STEP, None)
    at = end.states[-1]
    threshold = GEOMETRY.candidates[0]
    assert fields["distance_km"] == pytest.approx(np.hypot(at[0] - threshold.threshold_e_m,
                                                           at[1] - threshold.threshold_n_m) / 1000.0)
    assert fields["after_first_step_s"] == pytest.approx(5 * DELTA)            # its own first step, not the anchor's
    assert fields["other_class"] == post_diagnose.LEADER and "start_conflict" in fields
    against_anchor = post_diagnose.loss_fields(window, 1, _end(window, 1, window.commanded.key, first + 5), SEPARATION,
                                               FINALS, _landings(ahead), STEP, None)
    assert against_anchor["other_class"] == post_diagnose.COMMANDED and "start_conflict" not in against_anchor


def test_the_lost_lines_name_each_lost_aircraft_by_its_place_in_the_window(monkeypatch):
    """A window whose anchor landed and whose joined aircraft lost separation: one line, member 1, its key, the loss on
    the window's steps, its fields."""
    ahead = _record("ahead", -20_000.0, 100.0)
    window = _joined_window(ahead)
    first = int(round(OBSERVATION_S / DELTA))
    lost = _end(window, 1, ahead.key, first + 5)
    ends = [SimpleNamespace(outcome=LANDED), lost]
    context = SimpleNamespace(separations={"KXXX": SEPARATION}, finals={"KXXX": FINALS},
                              words=SimpleNamespace(spec=SimpleNamespace(step_s=STEP)))
    monkeypatch.setattr(diagnose, "window_landings", lambda window, roster: roster)
    (line,) = diagnose.lost_lines(window, ends, context, {"KXXX": _landings(ahead)})
    assert (line["member"], line["aircraft"], line["other"]) == (1, window.commanded_all[1].key, ahead.key)
    assert line["loss_step"] == first + 5 + 3 and line["fields"]["other_class"] == post_diagnose.LEADER


def test_the_paired_difference_counts_aircraft_in_their_windows_and_the_failures_by_span():
    windows = [SimpleNamespace(scene=SimpleNamespace(geometry=SimpleNamespace(code=code)), span_s=span)
               for code, span in (("KA", 300.0), ("KA", 600.0), ("KB", 600.0))]
    E = diagnose.End
    a = [[E(LANDED, 1.0), E(LANDED, 1.0)], [E(LANDED, 1.0)], [E(LOST_SEPARATION, 0.0)]]
    b = [[E(LANDED, 1.0), E(LOST_SEPARATION, 0.0)], [E(LOST_SEPARATION, 0.0)], [E(LOST_SEPARATION, 0.0)]]
    paired = diagnose.paired(windows, a, b, np.random.default_rng(0))
    assert paired["all"]["landed"]["difference"] == pytest.approx(2 / 4)
    assert paired["airports"]["KB"]["landed"]["difference"] == 0.0
    assert paired["spans"]["600"]["lost_separation"]["difference"] == pytest.approx(-1 / 2)
    reading = diagnose.reading(windows, a)
    assert reading["all"]["aircraft"] == 4 and reading["all"]["landed"] == 0.75 and reading["spans"]["300"]["windows"] == 1
    lines = [{"airport": "KA", "span": "300", "lost": [{"fields": {"other_class": post_diagnose.COMMANDED}}]},
             {"airport": "KB", "span": "600", "lost": [{"fields": {"other_class": post_diagnose.LEADER,
                                                                   "start_conflict": True}}]}]
    counted = diagnose.counted_failures(lines)
    assert counted["all"]["lost"] == 2 and counted["all"]["other_class"][post_diagnose.COMMANDED]["count"] == 1
    assert counted["spans"]["600"]["start_conflict"]["true"]["share"] == 1.0
    assert counted["spans"]["300"]["other_class"][post_diagnose.COMMANDED]["share"] == 1.0
