"""Stage D, MC1: the windows of several commanded aircraft and their census (multi-aircraft control D146, §3.1, §11
MC1; `multi/windows.py`, `multi/census.py`) — on the synthetic flights of stage C's scene tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import pytest

from ts_transformer.instructions.artefact import load_candidates

from ts_transformer.multi.census import summary, window_count

from ts_transformer.multi.separation import PAIRS, classify

from ts_transformer.multi.windows import COMPRESSED, REAL_KIND, Anchors, Drawn, compressed, left_out, opening_loss

from ts_transformer.post.runways import airport_separation

from ts_transformer.post.scene import airport_scenes, real_windows

from ts_transformer.tests.post_support import categories, finals, scene_artefact, straight_in, train_noon

from ts_transformer.tests.support import INSTRUCTION_STEP_S

DELTA = 4.0

def test_the_census_runner_writes_each_split_span_and_kind_and_never_under_the_outputs(tmp_path, monkeypatch):
    """MC1 (multi-aircraft control §11): the census of train and select, for each span and kind, on the records and on
    the closed-loop states (the baseline), written to a new directory outside `4dTrajectory/outputs/`; a sample is
    stated; the closed loop's checks run first (a stand-in here: the synthetic artefact has no executor spec)."""
    import json

    from ts_transformer.experiments import multi_windows
    from ts_transformer.instructions.words import Words
    from ts_transformer.post import scene as post_scene
    from ts_transformer.repo_layout import REPO_ROOT

    flights = {"train": [straight_in("KXXX:a", train_noon(0.0)), straight_in("KXXX:b", train_noon(120.0)),
                         straight_in("KXXX:c", train_noon(242.0)), straight_in("KXXX:far", train_noon(7_200.0))],
               "select": [straight_in("KXXX:s", train_noon(60.0, split="select"))]}
    spec = scene_artefact(tmp_path / "art", flights, interval_s=DELTA)
    checked = []
    monkeypatch.setattr(multi_windows, "require_conforming_closed_loop",
                        lambda instructions, executor: checked.append(executor) or (None, {"stand-in": True},
                                                                                      Words(spec)))
    monkeypatch.setattr(multi_windows, "airport_scenes",
                        lambda *a, **k: post_scene.airport_scenes(*a, **k, category_of=categories))
    monkeypatch.setattr(multi_windows, "airport_finals", lambda geometry, root: finals(geometry))
    out = tmp_path / "census"
    args = ["--instructions", str(tmp_path / "art"), "--executor", str(tmp_path / "executor"), "--interval-s", "4"]
    assert multi_windows.main(args + ["--out", str(out), "--spans-min", "0", "5"]) == 0
    assert checked == [tmp_path / "executor"]
    record = json.loads((out / "census.json").read_text())
    assert record["schema"] == multi_windows.CENSUS_SCHEMA and set(record["splits"]) == {"train", "select"}
    train = record["splits"]["train"]
    assert train["anchors"] == 4 and set(train["spans"]) == {"0min", "5min"}
    assert set(train["spans"]["0min"]) == {"real"}                         # L = 0: no compressed window
    five = train["spans"]["5min"]
    assert set(five) == {"real", "compressed_0.6", "compressed_0.8"}
    assert five["real"]["windows"] == 4 and five["compressed_0.6"]["windows"] == 2     # the windows with a later aircraft
    assert five["real"]["records"]["KXXX"]["commanded"]["max"] == 3.0
    assert set(five["real"]["baseline"]["all"]["losses"]) == set(PAIRS)
    assert five["real"]["baseline"]["all"]["steps_judged"] > 0
    assert record["sample"] is None and record["checks"] == {"stand-in": True}
    sampled = tmp_path / "sampled"
    multi_windows.main(args + ["--out", str(sampled), "--splits", "train", "--sample", "2", "--spans-min", "0"])
    again = json.loads((sampled / "census.json").read_text())
    assert again["splits"]["train"]["anchors"] == 2 and again["sample"] == 2 and again["sample_seed"] == 1337
    with pytest.raises(SystemExit):
        multi_windows.main(args + ["--out", str(out)])                       # an existing directory
    forbidden = REPO_ROOT / "4dTrajectory" / "outputs" / "multi_windows_test_never_written"
    with pytest.raises(SystemExit):
        multi_windows.main(args + ["--out", str(forbidden)])
    assert not forbidden.exists()
    with pytest.raises(SystemExit):
        multi_windows.main(args + ["--out", str(tmp_path / "v"), "--splits", "val"])
