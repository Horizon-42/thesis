"""Stage D: a round's per-window read (`multi_window_read`, post-training D176 (2)) and the window list that reads it
(`window_list --multi`) — on the stage D campaign of one round of `test_multi_compare`. Every write root under tmp."""

from __future__ import annotations

import argparse
import json
import os
import stat

import pytest

from ts_transformer.experiments import multi_train
from ts_transformer.experiments import multi_window_read as reader
from ts_transformer.experiments import window_list as lister
from ts_transformer.experiments.multi_train import stage_d
from ts_transformer.experiments.multi_training_export import STAGE_D
from ts_transformer.experiments.post_training_export import listed_windows
from ts_transformer.experiments.post_window_loop import LANDED, LOST_SEPARATION
from ts_transformer.post.window_lists import identity_of, read_window_list
from ts_transformer.tests.test_multi_compare import GIT, campaign  # noqa: F401
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401


@pytest.fixture
def read_world(campaign, monkeypatch):  # noqa: F811
    """The campaign of `test_multi_compare` with this runner's checks stubbed and its context the fixture's."""
    out, context, settings, opened = campaign
    monkeypatch.setattr(reader, "git_state", lambda: GIT)
    monkeypatch.setattr(reader, "require_conforming_closed_loop", lambda *a: (None, {"checks": {"stub": True}}, None))
    monkeypatch.setattr(reader, "checked_edges", lambda path: None)
    monkeypatch.setattr(reader, "open_context", lambda *a, **k: (opened.append((k["splits"], k["data"], k["formal"])),
                                                                 context)[1])
    return out, context, settings, opened


def test_a_read_is_the_rounds_readout_a_line_a_window_and_sealed(read_world, tmp_path):
    """The select days opened formal without data; the round's model read on the campaign's select windows by its own
    readout; a line a window in place order, with its identity (`identity_of`), its aircraft's outcomes and rewards,
    whose sums are the round's readout; the selection as the Training export gives it; the directory sealed; a second
    read to the same directory refused."""
    out, context, settings, opened = read_world
    read = tmp_path / "read"
    assert reader.main(["--campaign", str(out), "--round", "0", "--out", str(read), "--device", "cpu"]) == 0
    assert opened == [(("select",), False, True)]
    config = json.loads((read / reader.CONFIG).read_text())
    lines = [json.loads(line) for line in (read / reader.LINES).read_text().splitlines()]
    windows = stage_d().selection(context, settings, "select")
    assert config["schema"] == reader.WINDOW_READ_SCHEMA and config["round"] == 0 and config["windows"] == len(windows)
    assert config["selection"] == STAGE_D.selection_fields(settings) and config["checks"] == {"stub": True}
    assert config["inputs"] == json.loads((out / "campaign.json").read_text())["inputs"] and config["smoke"] is False
    assert [line["place"] for line in lines] == list(range(len(windows)))
    assert [line["identity"] for line in lines] == [json.loads(json.dumps(identity_of("D", w))) for w in windows]
    held = json.loads((out / "round_0" / "round.json").read_text())["selection_readout"]
    for code, airport in held.items():
        mine = [line for line in lines if line["airport"] == code]
        whole = airport["all"]["all"]
        assert sum(len(line["outcomes"]) for line in mine) == whole["aircraft"]
        assert sum(line["outcomes"].count(LANDED) for line in mine) == whole["outcomes"].get(LANDED, 0)
        assert sum(sum(line["rewards"]) for line in mine) == pytest.approx(whole["reward_sum"])
    assert all(len(line["aircraft"]) == len(line["outcomes"]) == len(line["rewards"]) for line in lines)
    assert sorted(p.name for p in read.iterdir()) == ["SHA256SUMS", reader.CONFIG, reader.LINES]
    assert all(not os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for p in (read, *read.iterdir()))
    with pytest.raises(SystemExit):
        reader.main(["--campaign", str(out), "--round", "0", "--out", str(read), "--device", "cpu"])
    # the list written from it is one the Training export takes: its selection and each window's identity at its place
    given = sorted({outcome for line in lines for outcome in line["outcomes"]})
    assert lister.main(["--multi", str(read), "--outcomes", *given, "--out", str(tmp_path / "list")]) == 0
    stage = stage_d()
    taken = listed_windows(argparse.ArgumentParser(), STAGE_D, read_window_list(tmp_path / "list" / "list.json"),
                           "select", stage.selection(context, settings, "select"), STAGE_D.selection_fields(settings))
    assert [place for place, _ in taken] == list(range(len(lines)))


def test_a_read_stops_unless_this_code_reads_the_round_as_its_readout(read_world, tmp_path, monkeypatch):
    """The behaviour check: a round whose recorded readout this code does not give again stops the read, nothing
    written; a round not done, a formal read of a smoke campaign and a formal read from a tree with changes are refused
    before anything opens; with ``--smoke`` the read runs from that tree and says so."""
    out, _, _, opened = read_world
    path = out / "round_0" / "round.json"
    original = path.read_text()
    held = json.loads(original)
    (airport,) = held["selection_readout"].values()
    airport["all"]["all"]["reward_sum"] += 1.0
    path.write_text(json.dumps(held))
    with pytest.raises(SystemExit, match="round 0: this code's readout of its model on the cpu is not the round's"):
        reader.main(["--campaign", str(out), "--round", "0", "--out", str(tmp_path / "read"), "--device", "cpu"])
    assert not (tmp_path / "read").exists()
    opened.clear()
    with pytest.raises(SystemExit):
        reader.main(["--campaign", str(out), "--round", "1", "--out", str(tmp_path / "read"), "--device", "cpu"])
    record = json.loads((out / "campaign.json").read_text())
    record["inputs"]["smoke"] = True
    (out / "campaign.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit):
        reader.main(["--campaign", str(out), "--round", "0", "--out", str(tmp_path / "read"), "--device", "cpu"])
    assert opened == [] and not (tmp_path / "read").exists()
    record["inputs"]["smoke"] = False
    (out / "campaign.json").write_text(json.dumps(record))
    path.write_text(original)
    monkeypatch.setattr(reader, "git_state", lambda: {**GIT, "dirty": True})
    with pytest.raises(SystemExit):                                       # a tree with changes: a formal read refused
        reader.main(["--campaign", str(out), "--round", "0", "--out", str(tmp_path / "read"), "--device", "cpu"])
    assert opened == [] and not (tmp_path / "read").exists()
    assert reader.main(["--campaign", str(out), "--round", "0", "--out", str(tmp_path / "read"), "--device", "cpu",
                        "--smoke"]) == 0
    assert json.loads((tmp_path / "read" / reader.CONFIG).read_text())["smoke"] is True


SELECTION = {"select_seed": 1337, "per_airport": 50, "spans_s": [300.0, 600.0]}


def _multi(root, name, outcomes, selection=SELECTION, smoke=False):
    """A hand-made stage D per-window read of windows with the commanded aircraft's ``outcomes``."""
    directory = root / name
    directory.mkdir()
    (directory / reader.CONFIG).write_text(json.dumps({
        "schema": reader.WINDOW_READ_SCHEMA, "smoke": smoke, "round": 4, "selection": selection,
        "windows": len(outcomes)}))
    lines = [{"place": k, "airport": f"K{a}", "identity": {"flight": f"K{a}:F{k}", "row0_s": 1000.0 * k,
                                                             "span_s": 300.0},
              "outcomes": list(o)} for k, (a, o) in enumerate(zip("AABB", outcomes))]
    (directory / reader.LINES).write_text("".join(json.dumps(line) + "\n" for line in lines))
    return directory


def test_a_stage_d_list_chooses_a_window_by_any_of_its_aircraft_and_names_its_selection(tmp_path):
    """``--multi``: a window is chosen when an outcome of any of its commanded aircraft is one of ``--outcomes``; the
    list is of stage D, its selection the read's, each window's information its aircraft's outcomes; a stage C read
    beside it, a read of other selection windows and a smoke read are refused by name."""
    lost, landed = LOST_SEPARATION, LANDED
    read = _multi(tmp_path, "read", [(landed, landed), (landed, lost), ("timeout",), (landed,)])
    assert lister.main(["--multi", str(read), "--outcomes", lost, "timeout", "--out", str(tmp_path / "list")]) == 0
    listed = read_window_list(tmp_path / "list" / "list.json")
    assert listed.stage == "D" and listed.selection == SELECTION and [w.place for w in listed.windows] == [1, 2]
    assert [w.airport for w in listed.windows] == ["KA", "KB"]
    assert listed.windows[0].identity == {"flight": "KA:F1", "row0_s": 1000.0, "span_s": 300.0}
    assert listed.windows[0].info["outcomes"] == {"read:round_4": [landed, lost]}
    assert listed.readouts[0]["path"].endswith(reader.LINES)
    other = _multi(tmp_path, "other", [(lost,)] * 4, selection={**SELECTION, "spans_s": [300.0]})
    smoke = _multi(tmp_path, "smoke", [(lost,)] * 4, smoke=True)
    stage_c = tmp_path / "diagnose"
    stage_c.mkdir()
    (stage_c / "config.json").write_text(json.dumps({
        "schema": lister.DIAGNOSE_SCHEMA, "smoke": False,
        "windows": [{"flight": f"KA:F{k}", "row0_s": 0.0, "kind": "real"} for k in range(4)],
        "settings": {"select_seed": 1337, "select_per_airport": 50}}))
    (stage_c / "per_window.jsonl").write_text("".join(json.dumps({"read": "on", "place": k, "outcome": lost}) + "\n"
                                                      for k in range(4)))
    for more, message in ((["--multi", str(other)], "other selection windows"),
                          (["--diagnose", str(stage_c), "on"], "is a read of stage D, diagnose:on of stage C"),
                          (["--multi", str(smoke)], "smoke readout")):
        with pytest.raises(SystemExit, match=message):
            lister.main(["--multi", str(read), *more, "--outcomes", lost, "--out", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists()


def test_the_readouts_lines_are_given_only_when_asked_and_leave_the_readout_as_it_was(read_world):
    """`selection_readout` with ``lines``: the same readout as without, and a line a window in place order."""
    _, context, settings, _ = read_world
    stage = stage_d()
    windows = stage.selection(context, settings, "select")
    model, _ = stage.start(context, settings)
    lines: list = []
    with_lines = multi_train.selection_readout(model, context, windows, settings, None, stage=stage, lines=lines)
    assert with_lines == multi_train.selection_readout(model, context, windows, settings, None, stage=stage)
    assert [line["place"] for line in lines] == list(range(len(windows)))
