"""The campaign runner's dry run is a DRY run.

`experiments/frame_ablation.py` is the one arm-campaign driver (its predecessor
`run_ts_control_arms.py` was archived on 2026-09-07 (`archive/control_arms_runner_2026_08/`)). A dry run exists to answer two
questions before anything is spent: does every arm's config construct, and what will be
run? Neither needs a file — and one aimed at `4dTrajectory/outputs`, a read-only symlink in
a development worktree, used to leave a campaign directory behind in the shared tree.
"""

from __future__ import annotations

import json

import pytest

import ts_transformer.experiments.frame_ablation as runner

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
    config, declared = runner.arm_config(ARMS["base"], {"coordinate_frame": "airport-enu"})
    assert config.coordinate_frame == "airport-enu"
    # The settings come back rather than being read off disk: `arm_steps` needs them on a
    # dry run too, when there is no file.
    assert declared["coordinate_frame"] == "airport-enu"
    assert not destination.exists()
    assert runner.write_arm_config(destination, declared) == destination
    assert json.loads(destination.read_text())["coordinate_frame"] == "airport-enu"


def test_a_stored_arm_is_resumed_only_when_complete_and_unchanged(tmp_path):
    """Review C-8. Resume is `history.json` exists AND the trained config agrees with the
    arm's overrides today; a checkpoint without a history (a tail crash) and a changed arm
    are both refused by name, and nothing is deleted."""
    _config, declared = runner.arm_config(ARMS["base"], {"coordinate_frame": "airport-enu"})
    arm = tmp_path / "A"
    assert runner.stale_arm_error("A", arm, declared) is None          # never trained: runs
    arm.mkdir()
    (arm / "checkpoint.pt").write_bytes(b"x")
    error = runner.stale_arm_error("A", arm, declared)
    assert error and "without history.json" in error and "aborted" in error
    # The trained config carries fields the arm never declared (the CLI's seed, say);
    # only the DECLARED fields are compared.
    (arm / "history.json").write_text(json.dumps({"config": {**declared, "seed": 4711}}))
    assert runner.stale_arm_error("A", arm, declared) is None
    changed = {**declared, "coordinate_frame": "enu"}
    error = runner.stale_arm_error("A", arm, changed)
    assert error and "coordinate_frame" in error and "new arm" in error
    assert (arm / "checkpoint.pt").exists() and (arm / "history.json").exists()


def test_a_field_added_after_an_arm_trained_reads_as_the_default_it_flew(tmp_path):
    """Review of 2026-09-14 (M1, blocker): `control_thrust_parameterization` is pinned by
    every named recipe, and no history.json written before it carries it. Read as None, the
    field refused the resume of every recipe arm on disk — 4 of 4 in b1_quantile_20260907.
    It reads as `TSConfig.from_dict` reads it: the default, which is what those arms flew. A
    missing REQUIRED field stays a difference, because `from_dict` refuses it."""
    # What `main` builds for a declaration that names `"base_recipe": "simple-v3"`.
    base = {**runner.recipe_settings("simple-v3", keep_name=True), "device": "cpu"}
    _config, declared = runner.arm_config(base, {})
    assert declared["control_thrust_parameterization"] == "thrust-fraction"
    # Every field a recipe pins that no stored history.json carries (N4 added the second).
    added_later = ("control_thrust_parameterization", "control_condition_features")
    assert declared["control_condition_features"] == "raw"
    stored = {key: value for key, value in declared.items() if key not in added_later}
    arm = tmp_path / "B1"
    arm.mkdir()
    (arm / "history.json").write_text(json.dumps({"config": stored}))
    assert runner.stale_arm_error("B1", arm, declared) is None
    # ...but a stored run that flew the OTHER law is a different arm.
    (arm / "history.json").write_text(json.dumps(
        {"config": {**stored, "control_thrust_parameterization": "specific-force"}}))
    assert "control_thrust_parameterization" in runner.stale_arm_error("B1", arm, declared)
    # ...and a REQUIRED field that is missing is not defaulted.
    required = {key: value for key, value in declared.items() if key != "control_dynamics_model"}
    (arm / "history.json").write_text(json.dumps({"config": required}))
    assert "control_dynamics_model" in runner.stale_arm_error("B1", arm, declared)


def test_the_train_step_is_done_when_history_json_exists(tmp_path):
    """`train` writes checkpoint.pt first and history.json last, so the step's artifact is
    the history — a checkpoint alone is a crash, not a finished arm."""
    _config, declared = runner.arm_config(ARMS["base"], {})
    steps = runner.arm_steps(
        "A", tmp_path / "A" / "config.json", declared, airport="KRDU",
        campaign=tmp_path, split="val", device="cpu", seed=1, split_seed=1, formal=False,
    )
    assert steps[0][2] == tmp_path / "A" / "history.json"


def test_a_train_only_declaration_plans_no_predict_step(tmp_path, monkeypatch, capsys):
    """`"predict": false` (two-tier L1): the campaign's own runners read the checkpoints, so
    an arm is its train step alone — and a predict-only arm has nothing left to do there."""
    _config, declared = runner.arm_config(ARMS["base"], {})
    steps = runner.arm_steps(
        "A", tmp_path / "A" / "config.json", declared, airport="KRDU",
        campaign=tmp_path, split="val", device="cpu", seed=1, split_seed=1, formal=False, predict=False,
    )
    assert [label for label, _command, _artifact in steps] == ["A: train"]
    _harvest(tmp_path, monkeypatch)
    declaration = tmp_path / "train_only.json"
    declaration.write_text(json.dumps({**ARMS, "predict": False}), encoding="utf-8")
    assert runner.main(["--arms", str(declaration), "--campaign", str(tmp_path / "campaign"),
                        "--airport", "KRDU", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "A_threshold_enu: train" in printed and "predict" not in printed.split("steps pending")[1]
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps({"predict": False, "base": ARMS["base"], "arms": [
        {"key": "A", "label": "trains", "overrides": {}},
        {"key": "B", "label": "reads", "checkpoint": "elsewhere/checkpoint.pt"},
    ]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        runner.main(["--arms", str(mixed), "--campaign", str(tmp_path / "campaign2"), "--airport", "KRDU", "--dry-run"])


def test_a_development_cohort_is_handed_to_every_train_step_and_must_exist(tmp_path, monkeypatch, capsys):
    """`"development_cohort"` (two-tier L1): the explicit train roster the train CLI demands when
    the random-anchor future contract leaves a train flight without an anchor — data written
    before the campaign, so a missing file is refused at plan time, dry run included."""
    _config, declared = runner.arm_config(ARMS["base"], {})
    cohort = tmp_path / "development_cohort.json"
    steps = runner.arm_steps(
        "A", tmp_path / "A" / "config.json", declared, airport="KRDU",
        campaign=tmp_path, split="val", device="cpu", seed=1, split_seed=1, formal=False,
        development_cohort=cohort,
    )
    command = steps[0][1]
    assert command[command.index("--development-cohort") + 1] == str(cohort)
    _harvest(tmp_path, monkeypatch)
    relative = cohort.relative_to(runner.REPO_ROOT) if cohort.is_relative_to(runner.REPO_ROOT) else cohort
    declaration = tmp_path / "cohort_arms.json"
    declaration.write_text(json.dumps({**ARMS, "development_cohort": str(relative).replace("KRDU", "{airport}")}),
                           encoding="utf-8")
    with pytest.raises(SystemExit):        # not written yet
        runner.main(["--arms", str(declaration), "--campaign", str(tmp_path / "campaign"), "--airport", "KRDU", "--dry-run"])
    cohort.write_text("{}", encoding="utf-8")
    assert runner.main(["--arms", str(declaration), "--campaign", str(tmp_path / "campaign"), "--airport", "KRDU",
                        "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert f"development cohort: {cohort}" in printed and printed.count("--development-cohort") == 2


def _harvest(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HARVEST_ROOT", tmp_path / "harvest")
    for name in ("manifest.json", "lateral_pass_eligibility.json"):
        path = tmp_path / "harvest" / "KRDU" / "arrivals" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def _dependent_arms(tmp_path, campaign, *, external: bool = False):
    """A training arm followed by a predict-only arm that reads ITS checkpoint (the L1.c
    shape) — or, with ``external``, one that reads a checkpoint nobody in the file writes."""
    target = (tmp_path / "elsewhere" / "checkpoint.pt") if external else (campaign / "A_train" / "checkpoint.pt")
    arms = {
        "base": dict(ARMS["base"]),
        "arms": [
            {"key": "A_train", "label": "trains", "overrides": {}},
            {"key": "A_train_hooked", "label": "reads A_train", "checkpoint": str(target.relative_to(runner.REPO_ROOT))
             if target.is_relative_to(runner.REPO_ROOT) else str(target), "predict_args": []},
        ],
    }
    path = tmp_path / "dependent_arms.json"
    path.write_text(json.dumps(arms), encoding="utf-8")
    return path


def test_a_predict_only_arm_may_read_a_checkpoint_the_campaign_itself_will_write(tmp_path, monkeypatch, capsys):
    """L1.c: the hook arms read the penalty arms' checkpoints, which cannot exist at plan
    time. The plan must still build, and say which arm produces the file."""
    campaign = tmp_path / "campaign"
    _harvest(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    assert runner.main([
        "--arms", str(_dependent_arms(tmp_path, campaign)), "--campaign", str(campaign),
        "--airport", "KRDU", "--dry-run",
    ]) == 0
    printed = capsys.readouterr().out
    assert "produced by arm A_train" in printed and "A_train_hooked: predict" in printed


def test_a_predict_only_arm_reading_a_missing_external_checkpoint_still_fails_at_plan_time(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    _harvest(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="A_train_hooked: checkpoint .* does not exist"):
        runner.main([
            "--arms", str(_dependent_arms(tmp_path, campaign, external=True)), "--campaign", str(campaign),
            "--airport", "KRDU", "--dry-run",
        ])


def test_a_deferred_checkpoint_that_never_appears_is_refused_by_the_producing_arm_name(tmp_path, monkeypatch):
    """If the producing arm's train step runs but writes no checkpoint, the dependent
    predict step refuses by name instead of handing the CLI a missing file."""
    campaign = tmp_path / "campaign"
    _harvest(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "MINIMUM_FREE_BYTES", 0)
    ran: list[list[str]] = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kw: ran.append(command))  # trains nothing
    with pytest.raises(FileNotFoundError, match="arm A_train did not train"):
        runner.main([
            "--arms", str(_dependent_arms(tmp_path, campaign)), "--campaign", str(campaign),
            "--airport", "KRDU", "--informal",
        ])
    assert any("train" in c for c in ran[0]), "the producing arm's train step ran first"
