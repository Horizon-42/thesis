"""Stage C, C25 (post-training D176): window lists — the format `ts-window-list-v1` and the runner `window_list` on
hand-made D175 and D161 readouts. Every write root under tmp."""

from __future__ import annotations

import json
import os
import stat

import pytest

from ts_transformer.experiments import window_list as runner
from ts_transformer.experiments.post_ceiling import CEILING_SCHEMA
from ts_transformer.experiments.post_diagnose import DIAGNOSE_SCHEMA
from ts_transformer.experiments.post_window_loop import LOST_SEPARATION
from ts_transformer.post.reward import LANDED
from ts_transformer.post.window_lists import (
    WINDOW_LIST_SCHEMA, ListedWindow, WindowList, read_window_list, write_window_list,
)

WINDOWS = [{"flight": f"K{a}:F{k}", "row0_s": 1000.0 * k, "kind": "real", "moved": [], "start_move": {}}
           for k, a in enumerate("AABB")]


def _list(stage="C", split="select", **changed):
    identity = ({"flight": "KA:F0", "row0_s": 0.0, "kind": "real"} if stage == "C"
                else {"flight": "KA:F0", "row0_s": 0.0, "span_s": 600.0})
    selection = {"select_seed": 1337, "per_airport": 200} if stage == "C" else \
        {"select_seed": 1337, "per_airport": 50, "spans_s": [600.0]}
    values = dict(stage=stage, split=split, selection=selection, chose="the windows a rule chose",
                  readouts=({"path": "a/b.jsonl", "sha256": "0" * 64},),
                  windows=(ListedWindow(3, "KA", identity, {"note": 1}),))
    return WindowList(**{**values, **changed})


@pytest.mark.parametrize("stage", ["C", "D"])
def test_a_list_is_written_and_read_again(tmp_path, stage):
    listed = _list(stage)
    payload = write_window_list(tmp_path / "list.json", listed)
    assert payload["schema"] == WINDOW_LIST_SCHEMA and payload["count"] == 1 and payload["by_airport"] == {"KA": 1}
    assert read_window_list(tmp_path / "list.json") == listed
    with pytest.raises(FileExistsError):
        write_window_list(tmp_path / "list.json", listed)


def test_a_list_is_refused_by_name_for_val_another_schema_or_a_window_without_place_or_identity(tmp_path):
    with pytest.raises(ValueError, match="select windows only"):
        _list(split="val")
    write_window_list(tmp_path / "list.json", _list())
    payload = json.loads((tmp_path / "list.json").read_text())
    for name, change, message in (("val", {"split": "val"}, "select windows only"),
                                  ("schema", {"schema": "ts-other-v1"}, "not a ts-window-list-v1"),
                                  ("place", {"windows": [{"airport": "KA", "identity": {}, "info": {}}]}, "no place"),
                                  ("identity", {"windows": [{"place": 1, "airport": "KA", "info": {}}]}, "no identity")):
        (tmp_path / f"{name}.json").write_text(json.dumps({**payload, **change}))
        with pytest.raises(ValueError, match=message):
            read_window_list(tmp_path / f"{name}.json")
    with pytest.raises(ValueError, match="identity is"):              # a stage D identity in a stage C list
        _list(windows=(ListedWindow(0, "KA", {"flight": "KA:F0", "row0_s": 0.0, "span_s": 1.0}),))


def _diagnose(root, name, on, off, windows=WINDOWS, seed=1337):
    directory = root / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({
        "schema": DIAGNOSE_SCHEMA, "smoke": False, "windows": windows,
        "settings": {"select_seed": seed, "select_per_airport": 200}}))
    lines = []
    for read, outcomes in (("on", on), ("off", off)):
        for place, outcome in enumerate(outcomes):
            line = {"read": read, "place": place, "outcome": outcome}
            if outcome == LOST_SEPARATION:
                line["fields"] = {"other_class": "leader"}
            lines.append(json.dumps(line))
    (directory / "per_window.jsonl").write_text("\n".join(lines) + "\n")
    return directory


def _ceiling(root, name, start):
    """A D161 readout of campaign ``campaign`` (its select seed in the campaign's settings only) with the start's
    draws."""
    campaign = root / "campaign"
    campaign.mkdir(exist_ok=True)
    (campaign / "campaign.json").write_text(json.dumps({"inputs": {"settings": {"select_seed": 1337,
                                                                                "select_per_airport": 200}}}))
    directory = root / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({
        "schema": CEILING_SCHEMA, "smoke": False, "campaign": str(campaign), "models": ["start"],
        "windows": WINDOWS, "settings": {"select_per_airport": 200}}))
    (directory / "model_start.json").write_text(json.dumps({"outcomes": [[o, LANDED] for o in start]}))
    return directory


def test_the_rule_chooses_by_any_read_or_every_read_and_the_list_is_sealed(tmp_path):
    """Three reads of four windows (two diagnose reads and a ceiling's draw 0): any read losing separation chooses
    places 0, 1 and 3; every read, place 0 only. The list names its reads by path and sha256; the directory is sealed."""
    lost, landed = LOST_SEPARATION, LANDED
    diagnose = _diagnose(tmp_path, "diagnose", on=[lost, lost, landed, landed], off=[lost, landed, landed, landed])
    ceiling = _ceiling(tmp_path, "ceiling", start=[lost, landed, landed, lost])
    reads = ["--diagnose", str(diagnose), "on", "--diagnose", str(diagnose), "off", "--ceiling", str(ceiling), "start",
             "--outcomes", lost]
    assert runner.main([*reads, "--out", str(tmp_path / "any")]) == 0
    assert runner.main([*reads, "--all", "--out", str(tmp_path / "every")]) == 0
    anyone, every = read_window_list(tmp_path / "any" / "list.json"), read_window_list(tmp_path / "every" / "list.json")
    assert [w.place for w in anyone.windows] == [0, 1, 3] and [w.place for w in every.windows] == [0]
    assert anyone.selection == {"select_seed": 1337, "per_airport": 200} and anyone.stage == "C"
    assert [w.airport for w in anyone.windows] == ["KA", "KA", "KB"]
    assert anyone.windows[2].identity == {"flight": "KB:F3", "row0_s": 3000.0, "kind": "real"}
    assert anyone.windows[0].info["outcomes"] == {"diagnose:on": lost, "diagnose:off": lost, "ceiling:start": lost}
    assert anyone.windows[0].info["fields"] == {"diagnose:on": {"other_class": "leader"},
                                                "diagnose:off": {"other_class": "leader"}}
    assert [r["path"].rsplit("/", 1)[1] for r in anyone.readouts] == ["per_window.jsonl", "per_window.jsonl",
                                                                      "model_start.json"]
    assert "any" in anyone.chose and "every one" in every.chose
    out = tmp_path / "any"
    assert (out / "SHA256SUMS").read_text().split()[1] == "list.json"
    assert all(not os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for p in (out, *out.iterdir()))


def test_reads_of_other_selection_windows_or_unknown_reads_are_refused_by_name(tmp_path):
    lost = LOST_SEPARATION
    diagnose = _diagnose(tmp_path, "diagnose", on=[lost] * 4, off=[lost] * 4)
    other = _diagnose(tmp_path, "other", on=[lost] * 4, off=[lost] * 4,
                      windows=[{**w, "row0_s": w["row0_s"] + 4.0} for w in WINDOWS])
    seeded = _diagnose(tmp_path, "seeded", on=[lost] * 4, off=[lost] * 4, seed=2028)
    ceiling = _ceiling(tmp_path, "ceiling", start=[lost] * 4)
    for more, message in ((["--diagnose", str(other), "on"], "other selection windows"),
                          (["--diagnose", str(seeded), "on"], "other selection windows"),
                          (["--diagnose", str(diagnose), "both"], "not 'both'"),
                          (["--ceiling", str(ceiling), "8"], "not '8'"),
                          (["--outcomes", "lost-separation"], "no read gives it")):
        with pytest.raises(SystemExit, match=message):
            runner.main(["--diagnose", str(diagnose), "on", "--outcomes", lost, *more, "--out", str(tmp_path / "x")])
    config = json.loads((diagnose / "config.json").read_text())
    (diagnose / "config.json").write_text(json.dumps({**config, "smoke": True}))
    with pytest.raises(SystemExit, match="smoke readout"):
        runner.main(["--diagnose", str(diagnose), "on", "--outcomes", lost, "--out", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists()
