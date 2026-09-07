"""The campaign runner's dry run is a DRY run.

`run_ts_frame_ablation.py` is the one arm-campaign driver (its predecessor
`run_ts_control_arms.py` was archived on 2026-09-07). A dry run exists to answer two
questions before anything is spent: does every arm's config construct, and what will be
run? Neither needs a file — and one aimed at `4dTrajectory/outputs`, a read-only symlink in
a development worktree, used to leave a campaign directory behind in the shared tree.
"""

from __future__ import annotations

import json

import pytest

import run_ts_frame_ablation as runner

ARMS = {
    "base": {"prediction_output": "state", "model": "itransformer",
             "horizon_mode": "full", "device": "cpu"},
    "arms": [
        {"key": "A_threshold_enu", "label": "threshold ENU", "overrides": {}},
        {"key": "B_airport_enu", "label": "airport ENU",
         "overrides": {"coordinate_frame": "airport-enu"}},
    ],
}


def _declaration(tmp_path):
    path = tmp_path / "arms.json"
    path.write_text(json.dumps(ARMS), encoding="utf-8")
    return path


def test_a_dry_run_creates_nothing_under_the_campaign_path(tmp_path, monkeypatch, capsys):
    campaign = tmp_path / "campaign"
    monkeypatch.setattr(runner, "HARVEST_ROOT", tmp_path / "harvest")
    for name in ("manifest.json", "lateral_pass_eligibility.json"):
        path = tmp_path / "harvest" / "KRDU" / "arrivals" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    assert runner.main([
        "--arms", str(_declaration(tmp_path)), "--campaign", str(campaign),
        "--airport", "KRDU", "--dry-run",
    ]) == 0
    assert not campaign.exists()
    # ...and it still says what it would run, per arm.
    printed = capsys.readouterr().out
    assert "A_threshold_enu: train" in printed and "B_airport_enu: train" in printed


def test_a_dry_run_still_constructs_every_arm_config(tmp_path, monkeypatch):
    """Finding an unrunnable arm is most of what a dry run is for, so validation stays."""
    campaign = tmp_path / "campaign"
    monkeypatch.setattr(runner, "HARVEST_ROOT", tmp_path / "harvest")
    for name in ("manifest.json", "lateral_pass_eligibility.json"):
        path = tmp_path / "harvest" / "KRDU" / "arrivals" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    declaration = tmp_path / "broken.json"
    declaration.write_text(json.dumps({
        **ARMS,
        "arms": [{"key": "impossible", "overrides": {"seq_len": 0}}],
    }), encoding="utf-8")

    with pytest.raises(ValueError):
        runner.main([
            "--arms", str(declaration), "--campaign", str(campaign),
            "--airport", "KRDU", "--dry-run",
        ])
    assert not campaign.exists()


def test_a_real_run_writes_the_config_it_will_train_from(tmp_path):
    """The file is what the training subprocess reads, so a non-dry run must write it."""
    destination = tmp_path / "arm" / "config.json"
    path, config, declared = runner.arm_config(
        ARMS["base"], {"coordinate_frame": "airport-enu"}, destination
    )
    assert path == destination and destination.is_file()
    assert json.loads(destination.read_text())["coordinate_frame"] == "airport-enu"
    assert config.coordinate_frame == "airport-enu"
    # The settings come back rather than being read off disk: `arm_steps` needs them on a
    # dry run too, when there is no file.
    assert declared["coordinate_frame"] == "airport-enu"
    _path, _config, dry = runner.arm_config(
        ARMS["base"], {}, tmp_path / "other" / "config.json", write=False
    )
    assert not (tmp_path / "other").exists()
    assert dry["prediction_output"] == "state"
