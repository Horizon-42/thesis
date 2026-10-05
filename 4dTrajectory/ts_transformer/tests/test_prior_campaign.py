"""B5's campaign and its choices (prior §5, §12 B5): the rules of §5 on synthetic scores, a fold read only when it is
complete and its own, the plan of the 31 runs, a whole campaign flown with a stand-in for the runners (the choices made
by `prior_select` itself), a resume after a kill, and the refusals of another commit or other inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ts_transformer.experiments import prior_campaign as campaign_module
from ts_transformer.experiments import prior_select as select_module
from ts_transformer.experiments.prior_select import arm_name, choose_configuration, choose_variant

SEEDS = (1337, 2024)
AIRPORTS = ("KAAA", "KBBB", "KCCC", "KDDD", "KEEE")
#: The parameters of each configuration's stand-in (B smaller than A, C larger, D as A).
PARAMETERS = {"A": 2_000_000, "B": 1_000_000, "C": 4_000_000, "D": 2_000_000}


def scores(**given):
    return {arm: {"score": score, "parameters": PARAMETERS[arm[0]]} for arm, score in given.items()}


def test_the_fewest_parameters_within_twice_the_seed_scale_of_the_best():
    # seed scale 0.02: the best is C (1.50); within 1.54: A (1.53), C — A has the fewer parameters
    base = dict(A_full_s1337=1.53, A_full_s2024=1.55, B_full_s1337=1.60, C_full_s1337=1.50, D_full_s1337=1.545)
    choice = choose_configuration(scores(**base), SEEDS)
    assert choice["seed_scale"] == pytest.approx(0.02) and choice["within"] == ["A", "C"] and choice["chosen"] == "A"
    # a smaller seed scale leaves C alone
    choice = choose_configuration(scores(**{**base, "A_full_s2024": 1.535}), SEEDS)
    assert choice["within"] == ["C"] and choice["chosen"] == "C"
    # B within as well: the fewest parameters
    assert choose_configuration(scores(**{**base, "B_full_s1337": 1.52}), SEEDS)["chosen"] == "B"


def test_constants_only_when_better_than_full_by_more_than_twice_the_seed_scale():
    base = dict(A_full_s1337=1.50, A_full_s2024=1.52, B_full_s1337=1.48)        # seed scale 0.02
    assert choose_variant(scores(**base, B_constants_s1337=1.45), "B", SEEDS)["chosen"] == "full"   # by 0.03
    assert choose_variant(scores(**base, B_constants_s1337=1.43), "B", SEEDS)["chosen"] == "constants"  # by 0.05


def write_fold(run: Path, airport: str, loss: float, parameters: int, *, held_out: str | None = None,
               identity: str = "artefact") -> None:
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"run": {"airports": list(AIRPORTS), "held_out": held_out or airport},
                                                 "sample": None, "parameters": parameters, "identity": identity}))
    (run / "held_out.json").write_text(json.dumps({"airport": held_out or airport, "loss_per_step": loss}))


def test_a_fold_is_read_only_complete_and_as_the_fold_it_is_named(tmp_path):
    for k, airport in enumerate(AIRPORTS):
        write_fold(tmp_path / "A_full_s1337" / airport, airport, 1.0 + k, 2_000_000)
    assert select_module.arm_score(tmp_path, "A_full_s1337", AIRPORTS)["score"] == pytest.approx(3.0)
    (tmp_path / "A_full_s1337" / "KEEE" / "held_out.json").unlink()
    with pytest.raises(ValueError, match="not complete"):
        select_module.arm_score(tmp_path, "A_full_s1337", AIRPORTS)
    write_fold(tmp_path / "B_full_s1337" / "KAAA", "KAAA", 1.0, 1, held_out="KBBB")
    with pytest.raises(ValueError, match="a fold of KBBB, not KAAA"):
        select_module.arm_score(tmp_path, "B_full_s1337", ["KAAA"])
    with pytest.raises(ValueError, match="a formal run in a smoke campaign"):
        select_module.arm_score(tmp_path, "A_full_s1337", AIRPORTS[:4], smoke=True)


def record(tmp_path, smoke=None):
    return {"instructions": str(tmp_path / "artefact"), "executor": str(tmp_path / "executor"), "row_interval_s": 4.0,
            "airports": list(AIRPORTS), "seeds": list(SEEDS), "smoke": smoke}


def test_the_plan_holds_31_training_runs_and_the_steps_after_a_choice_only_once_it_is_made(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    steps = campaign_module.plan(campaign, record(tmp_path), "cuda")
    assert [s.runner for s in steps].count("prior_train") == 25 and steps[-1].name == "choice_configuration"
    assert steps[0].name == "A_full_s1337/KAAA" and steps[1].name == "A_full_s1337/KAAA/free_generation"
    assert "--held-out" in steps[0].argv and "landed" in steps[0].argv
    (campaign / "choice_configuration.json").write_text(json.dumps({"chosen": "C"}))
    steps = campaign_module.plan(campaign, record(tmp_path), "cuda")
    assert steps[-1].name == "choice_variant" and steps[-3].name == "C_constants_s1337/KEEE"
    assert "--d-model" in steps[-3].argv and "256" in steps[-3].argv
    (campaign / "choice_variant.json").write_text(json.dumps({"chosen": "full"}))
    steps = campaign_module.plan(campaign, record(tmp_path), "cuda")
    assert [s.runner for s in steps].count("prior_train") == 31
    assert [s.name for s in steps[-3:]] == ["base/run", "base/validation", "base/free_generation"]
    assert "--held-out" not in steps[-3].argv and "val" in steps[-1].argv


class StandIn:
    """The runners, as the campaign sees them: each writes its step's files; `prior_select` is the real one. A fold's
    loss: configuration B best, by more than twice the seed scale; `constants` better by less."""

    LOSS = {"A_full_s1337": 1.50, "A_full_s2024": 1.51, "B_full_s1337": 1.40, "C_full_s1337": 1.45,
            "D_full_s1337": 1.50, "B_constants_s1337": 1.39}

    def __init__(self, fail_at: str | None = None):
        self.ran, self.fail_at = [], fail_at

    def __call__(self, step, log):
        self.ran.append(step.name)
        log.write_text("stand-in")
        if step.name == self.fail_at:
            step.out.mkdir(parents=True)           # a kill: the directory without its last file
            return 1
        if step.runner == "prior_select":
            return select_module.main(list(step.argv))
        if step.runner == "prior_train" and step.name.startswith("base/"):
            step.out.mkdir(parents=True)
            step.done.write_text("checkpoint")
        elif step.runner == "prior_train":
            arm, airport = step.name.split("/")
            write_fold(step.out, airport, self.LOSS[arm], PARAMETERS[arm[0]])
        else:
            step.out.mkdir(parents=True)
            step.done.write_text("{}")
        return 0


def test_a_campaign_runs_every_step_once_makes_its_choices_and_resumes_after_a_kill(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_module, "load_candidates", lambda instructions: dict.fromkeys(AIRPORTS))
    campaign = tmp_path / "campaign"
    git = {"head": "a" * 40, "dirty": False}
    rec = campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 4.0, git)
    killed = StandIn(fail_at="B_full_s1337/KCCC")
    with pytest.raises(SystemExit, match="B_full_s1337/KCCC failed"):
        campaign_module.run_campaign(campaign, rec, "cpu", killed, lambda line: None)
    rec = campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 4.0, git)
    runner = StandIn()
    campaign_module.run_campaign(campaign, rec, "cpu", runner, lambda line: None)
    assert runner.ran[0] == "B_full_s1337/KCCC"                         # the killed step again, first
    assert not set(runner.ran) & set(killed.ran[:-1])                  # nothing done is run again
    stored = json.loads((campaign / "campaign.json").read_text())
    assert [a["step"] for a in stored["aborted"]] == ["B_full_s1337/KCCC"]
    assert (campaign / "B_full_s1337" / Path(stored["aborted"][0]["moved_to"]).name).exists()
    configuration = json.loads((campaign / "choice_configuration.json").read_text())
    variant = json.loads((campaign / "choice_variant.json").read_text())
    assert configuration["chosen"] == "B" and variant["chosen"] == "full"
    assert (campaign / "base" / "free_generation" / "readout.json").exists()
    trained = [name for name in killed.ran + runner.ran if "/" in name and name.count("/") == 1
               and not name.endswith(("validation", "free_generation"))]
    assert len(trained) == 31 + 1                                      # 31 runs, the killed one twice


def test_a_resume_with_another_commit_or_other_inputs_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_module, "load_candidates", lambda instructions: dict.fromkeys(AIRPORTS))
    campaign = tmp_path / "campaign"
    git = {"head": "a" * 40, "dirty": False}
    campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 4.0, git)
    with pytest.raises(SystemExit, match="one commit"):
        campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 4.0,
                                      {"head": "b" * 40, "dirty": False})
    with pytest.raises(SystemExit, match="row_interval_s"):
        campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 2.0, git)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "file").write_text("x")
    with pytest.raises(SystemExit, match="is no campaign"):
        campaign_module.open_campaign(tmp_path / "other", tmp_path / "artefact", tmp_path / "executor", 4.0, git)


def test_a_smoke_campaign_tells_every_runner_it_is_a_smoke_and_a_formal_one_none(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "choice_configuration.json").write_text(json.dumps({"chosen": "A"}))
    (campaign / "choice_variant.json").write_text(json.dumps({"chosen": "full"}))
    formal = campaign_module.plan(campaign, record(tmp_path), "cuda")
    smoke = campaign_module.plan(campaign, record(tmp_path, {"sample": 20, "flights": 5}), "cuda")
    assert not any("--sample" in s.argv or "--smoke" in s.argv for s in formal)
    for step in smoke:
        if step.runner == "prior_train":
            assert step.argv[step.argv.index("--sample") + 1] == "20"
        elif step.runner in ("prior_free_generation", "prior_validation"):
            assert "--smoke" in step.argv
    assert all(s.argv[s.argv.index("--per-airport") + 1] == str(campaign_module.SMOKE_FLIGHTS)
               for s in smoke if s.runner == "prior_free_generation")
