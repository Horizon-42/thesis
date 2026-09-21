"""The stage B′ queue (`experiments/two_tier_bprime_queue.py`): the declaration's groups and their
naming rule, the plan's steps in order (baselines, train, truth-instruction readings, gate B1,
the gated priors), the seed-line check, and the runner's skip / gate / stop behaviour — on a
declaration written into tmp, with commands that touch nothing."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from ts_transformer.experiments import two_tier_bprime_queue as q
from ts_transformer.manoeuvre import instructions as ins


def _declaration(tmp_path: Path, vocabulary: Path) -> dict:
    cohort = tmp_path / "development_cohort.json"
    cohort.write_text("{}", encoding="utf-8")
    return {
        "predict": False,
        "development_cohort": str(cohort),
        "base_recipe": "simple-v3",
        "base": {
            "control_recipe_name": "custom", "control_thrust_parameterization": "specific-force+path-angle",
            "plan_conditioning": "instruction", "instruction_vocabulary": str(vocabulary), "anchor_floor_index": 0,
            "control_imitation_loss_weight": 0.0, "control_heading_rate_loss_weight": 0.0, "control_bank_tv_loss_weight": 0.0,
            "final_time_loss_weight": 0.0, "state_endpoint_loss_weight": 0.0, "random_train_anchor": True,
            "random_train_anchor_sampling": "remaining-path-uniform", "random_train_anchor_l1_share": 0.0,
            "checkpoint_selection_metric": "fixed-anchor-common-grid-ade", "lr_plateau_metric": "objective",
            "epochs": 1, "patience": 1, "split_seed": 1337, "seq_len": 30, "control_horizon_s": 20.0, "n_segments": 2,
            "random_train_anchor_min_future_s": 20.0,
        },
        "stage_b": {
            "seed_line_from": str(tmp_path / "grid_gate.json"),
            "baselines": {"L60_D20": {"what": "the no-token executor", "checkpoint": str(tmp_path / "L60_D20_s{seed}" / "checkpoint.pt")}},
            "configurations": {"I20": {"what": "the instruction executor", "baseline": "L60_D20"}},
            "prior": {"epochs": 3, "patience": 2, "d_model": 32},
        },
        "arms": [
            {"key": "I20_s1337", "configuration": "I20", "label": "seed 1337", "overrides": {"seed": 1337}},
            {"key": "I20_s2024", "configuration": "I20", "label": "seed 2024", "overrides": {"seed": 2024}},
        ],
    }


@pytest.fixture
def declared(tmp_path):
    vocabulary = ins.write_vocabulary(tmp_path, ins.Vocabulary(), runway_vocabulary=ins.RunwayVocabulary.from_idents(["KRDU:05L", "KRDU:23R"]),
                                      cohort_identity={}, counts={}, source={})
    declaration = _declaration(tmp_path, vocabulary)
    path = tmp_path / "arms.json"
    path.write_text(json.dumps(declaration), encoding="utf-8")
    return declaration, path, tmp_path / "campaign"


def test_the_groups_follow_the_naming_rule_and_need_two_seeds(declared):
    declaration, _path, _campaign = declared
    assert q.groups_of(declaration) == {"I20": {1337: "I20_s1337", 2024: "I20_s2024"}}
    wrong = {**declaration, "arms": [{**declaration["arms"][0], "key": "I20_s2024"}, declaration["arms"][1]]}
    with pytest.raises(ValueError, match="not named"):
        q.groups_of(wrong)
    with pytest.raises(ValueError, match="names configuration"):
        q.groups_of({**declaration, "arms": [{**declaration["arms"][0], "configuration": "X"}, declaration["arms"][1]]})
    with pytest.raises(ValueError, match="exactly two"):
        q.groups_of({**declaration, "arms": declaration["arms"][:1]})
    with pytest.raises(ValueError, match="not declared under stage_b"):
        q.groups_of({**declaration, "stage_b": {**declaration["stage_b"], "configurations": {}}})
    with pytest.raises(ValueError, match="steps this queue does not build"):
        q.groups_of({**declaration, "stage_b": {**declaration["stage_b"], "closed_loop": {}}})


def test_the_plan_runs_the_prior_first_then_the_baselines_then_each_group(declared):
    declaration, path, campaign = declared
    plan = q.plan_of(declaration, declaration_path=path, campaign=campaign, airport="KRDU", device="cpu", groups=None)
    assert list(plan) == ["prior", "baselines", "I20"]                 # the second layer first (user, 2026-09-20)
    priors = plan["prior"]
    assert [step.label for step in priors] == ["instruction prior (seed 1337)", "instruction prior (seed 2024)"]
    prior = priors[0]
    assert prior.gate is None                                          # ungated: it needs no trained executor
    assert prior.artefact == campaign / "priors" / "s1337" / "prior.pt"
    assert prior.command[2] == "instruction_prior"
    # the --executor is only the door to the split, so it is the DECLARED BASELINE's stage A checkpoint
    assert prior.command[prior.command.index("--executor") + 1].endswith("L60_D20_s1337/checkpoint.pt")
    assert prior.command[prior.command.index("--seed") + 1] == "1337"
    assert prior.command[prior.command.index("--epochs") + 1] == "3" and prior.command[prior.command.index("--d-model") + 1] == "32"
    assert "--n-layers" not in prior.command                           # only the declared settings are passed
    labels = [step.label for step in plan["baselines"]]
    assert labels == ["baseline L60_D20 seed 1337: lockstep none", "baseline L60_D20 seed 1337: failure modes",
                      "baseline L60_D20 seed 2024: lockstep none", "baseline L60_D20 seed 2024: failure modes"]
    first = plan["baselines"][0]
    assert first.command[2:4] == ["manoeuvre_lockstep", "--executor"] and "--protocol" in first.command and "none" in first.command
    assert "--cohort" in first.command and first.artefact == campaign / "baseline" / "L60_D20_s1337" / "L-1" / "manoeuvre_lockstep.json"
    group = plan["I20"]
    assert [step.label for step in group] == [
        "I20: train I20_s1337, I20_s2024",
        "I20_s1337: lockstep truth-instruction", "I20_s1337: failure modes truth-instruction",
        "I20_s2024: lockstep truth-instruction", "I20_s2024: failure modes truth-instruction",
        "I20: gate b1 (truth-instruction vs none)",
    ]
    train = group[0]
    assert train.artefact is None and train.command[2] == "frame_ablation" and train.command[-2:] == ["I20_s1337", "I20_s2024"]
    reading = group[1]
    assert reading.command[reading.command.index("--protocol") + 1] == "truth-instruction" and "--write-records" in reading.command
    assert reading.command[reading.command.index("--executor") + 1] == str(campaign / "I20_s1337" / "checkpoint.pt")
    gate = group[5]
    assert gate.artefact == campaign / "gate" / "b1_I20" / "relative_gate.json" and "--seed-line-from" in gate.command
    assert gate.command[gate.command.index("--baseline") + 1] == f"1337={campaign / 'baseline' / 'L60_D20_s1337' / 'L-1'}"
    assert gate is group[-1]                                           # the gate is the group's last step now
    with pytest.raises(ValueError, match="does not have"):
        q.plan_of(declaration, declaration_path=path, campaign=campaign, airport="KRDU", device="cpu", groups=["X"])


def test_the_seed_line_check_names_what_is_missing(tmp_path):
    path = tmp_path / "grid_gate.json"
    assert q.seed_line_problem(path) == "missing"
    path.write_text(json.dumps({"verdict": None}), encoding="utf-8")
    assert q.seed_line_problem(path).startswith("no verdict")
    path.write_text(json.dumps({"verdict": {"seed_line": {m: {"p75": 0.1} for m in ("established_all", "established_vectored")}
                                            | {"vectored_ade_mean_m": {"p50": 1.0}}}}), encoding="utf-8")
    assert q.seed_line_problem(path) == "the seed line lacks p75 of vectored_ade_mean_m"
    path.write_text(json.dumps({"verdict": {"seed_line": {m: {"p75": 0.1} for m in ("established_all", "established_vectored", "vectored_ade_mean_m")}}}), encoding="utf-8")
    assert q.seed_line_problem(path) is None


def test_the_runner_skips_done_steps_holds_gated_steps_and_stops_on_a_failure(tmp_path, capsys):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    done = campaign / "done.json"
    done.write_text("{}", encoding="utf-8")
    verdict = campaign / "gate" / "b1_I20" / "relative_gate.json"
    verdict.parent.mkdir(parents=True)
    verdict.write_text(json.dumps({"verdicts": {"b1": {"pass": False}}}), encoding="utf-8")
    ok = [sys.executable, "-c", "pass"]
    plan = {"I20": [q.Step("already done", ok, done), q.Step("runs", ok, campaign / "never.json"),
                    q.Step("gated prior", ok, campaign / "prior.pt", gate=(verdict, "b1"))]}
    assert q._run(plan, campaign) == 0
    out = capsys.readouterr().out
    assert "skip already done" in out and "RUN runs" in out and "skip gated prior: gate b1 of b1_I20 did not pass" in out and "CELL I20 complete" in out
    verdict.write_text(json.dumps({"verdicts": {"b1": {"pass": True}}}), encoding="utf-8")
    plan = {"I20": [q.Step("gated prior", [sys.executable, "-c", "raise SystemExit(7)"], campaign / "prior.pt", gate=(verdict, "b1"))]}
    assert q._run(plan, campaign) == 7
    assert "STOP: gated prior failed rc=7" in capsys.readouterr().out
    plan = {"I20": [q.Step("gated prior", ok, campaign / "prior.pt", gate=(campaign / "gate" / "absent" / "relative_gate.json", "b1"))]}
    assert q._run(plan, campaign) == 4
    assert "has no verdict" in capsys.readouterr().out


def test_the_dry_run_prints_the_plan_and_refuses_a_missing_seed_line_or_vocabulary(declared, capsys):
    declaration, path, campaign = declared
    with pytest.raises(SystemExit):
        q.main(["--arms", str(path), "--campaign", str(campaign), "--airport", "krdu", "--dry-run"])
    assert "seed_line_from: missing" in capsys.readouterr().err
    Path(declaration["stage_b"]["seed_line_from"]).write_text(json.dumps({"verdict": {"seed_line": {
        m: {"p75": 0.1} for m in ("established_all", "established_vectored", "vectored_ade_mean_m")}}}), encoding="utf-8")
    assert q.main(["--arms", str(path), "--campaign", str(campaign), "--airport", "krdu", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("CELL prior") and "CELL baselines" in out and "CELL I20" in out and "todo I20: train" in out
    assert out.index("CELL prior") < out.index("CELL baselines") < out.index("CELL I20")
    assert "[needs gate" not in out                                    # nothing is gated on b1 any more
    assert not campaign.exists()                                                     # a dry run writes nothing
