"""B5's campaign and its choices (prior §5, §12 B5): the rules of §5 on synthetic scores, a fold read only when it is
complete and its own, the plan of the 31 runs, a whole campaign flown with a stand-in for the runners (the choices made
by `prior_select` itself), a resume after a kill, the refusal of other inputs, and the behaviour check before each step
in place of a commit (D108)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ts_transformer.experiments import prior_campaign as campaign_module
from ts_transformer.experiments import prior_select as select_module
from ts_transformer.experiments.prior_select import arm_name, choose_configuration, choose_variant

SEEDS = (1337, 2024)
AIRPORTS = ("KAAA", "KBBB", "KCCC", "KDDD", "KEEE")
#: The parameters of each configuration's stand-in (B smaller than A, C larger, D as A).
PARAMETERS = {"A": 2_000_000, "B": 1_000_000, "C": 4_000_000, "D": 2_000_000}


def same(instructions):
    """The stand-in of the behaviour check (`prior_behaviour`): the code behaves as at the start, its settings those of
    the code this process loaded."""
    return {"train_loss": ["0x1p+0"], "words": [[0]], "settings": campaign_module.settings()}


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
    # A and D have one shape: of the two, the lower score (the user, 2026-10-05)
    tie = dict(A_full_s1337=1.53, A_full_s2024=1.55, B_full_s1337=1.60, C_full_s1337=1.70, D_full_s1337=1.50)
    assert choose_configuration(scores(**tie), SEEDS)["chosen"] == "D"
    assert choose_configuration(scores(**{**tie, "D_full_s1337": 1.54}), SEEDS)["chosen"] == "A"


def test_constants_only_when_better_than_full_by_more_than_twice_the_seed_scale():
    base = dict(A_full_s1337=1.50, A_full_s2024=1.52, B_full_s1337=1.48)        # seed scale 0.02
    assert choose_variant(scores(**base, B_constants_s1337=1.45), "B", SEEDS)["chosen"] == "full"   # by 0.03
    assert choose_variant(scores(**base, B_constants_s1337=1.43), "B", SEEDS)["chosen"] == "constants"  # by 0.05


def test_the_choices_at_their_edges():
    """§5 at its edges, on scores a float holds exactly: a score at exactly twice the seed scale of the best is within;
    a seed scale of 0 leaves the best alone; `constants` better by exactly twice the seed scale is not chosen, and with a
    seed scale of 0 it is when lower at all."""
    edge = dict(A_full_s1337=1.5, A_full_s2024=1.75, B_full_s1337=1.0, C_full_s1337=2.0, D_full_s1337=1.75)
    choice = choose_configuration(scores(**edge), SEEDS)                 # scale 0.25: A at exactly 1.0 + 0.5
    assert choice["seed_scale"] == 0.25 and choice["within"] == ["A", "B"] and choice["chosen"] == "B"
    choice = choose_configuration(scores(**{**edge, "B_full_s1337": 1.75, "A_full_s2024": 1.5}), SEEDS)  # scale 0
    assert choice["seed_scale"] == 0.0 and choice["within"] == ["A"] and choice["chosen"] == "A"
    base = dict(A_full_s1337=1.5, A_full_s2024=1.625)                    # scale 0.125
    assert choose_variant(scores(**base, A_constants_s1337=1.25), "A", SEEDS)["chosen"] == "full"   # by exactly 0.25
    zero = dict(A_full_s1337=1.5, A_full_s2024=1.5)
    assert choose_variant(scores(**zero, A_constants_s1337=1.375), "A", SEEDS)["chosen"] == "constants"


def write_fold(run: Path, airport: str, loss: float, configuration: str = "A", variant: str = "full",
               seed: int = SEEDS[0], *, held_out: str | None = None, rule: str = "landed", **changed) -> None:
    """A fold's files as `prior_train` writes them, of its arm (``changed``: values of its shape or training)."""
    values = {**select_module.configuration_values(configuration), **changed}
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "run": {"airports": list(AIRPORTS), "held_out": held_out or airport}, "sample": None,
        "parameters": PARAMETERS[configuration],
        "model_config": {**{k: values[k] for k in select_module.SHAPE_FIELDS}, "variant": variant},
        "train_config": {**{k: values[k] for k in select_module.TRAIN_FIELDS}, "seed": seed},
        "identity": {"artefact": "fixture", "selection": {"rule": rule}}}))
    (run / "held_out.json").write_text(json.dumps({"airport": held_out or airport, "loss_per_step": loss}))


def test_a_fold_is_read_only_complete_and_as_the_fold_of_its_arm(tmp_path):
    for k, airport in enumerate(AIRPORTS):
        write_fold(tmp_path / "A_full_s1337" / airport, airport, 1.0 + k)
    assert select_module.arm_score(tmp_path, "A", "full", 1337, AIRPORTS)["score"] == pytest.approx(3.0)
    (tmp_path / "A_full_s1337" / "KEEE" / "held_out.json").unlink()
    with pytest.raises(ValueError, match="not complete"):
        select_module.arm_score(tmp_path, "A", "full", 1337, AIRPORTS)
    with pytest.raises(ValueError, match="a formal run in a smoke campaign"):
        select_module.arm_score(tmp_path, "A", "full", 1337, AIRPORTS[:4], smoke=True)
    write_fold(tmp_path / "B_full_s1337" / "KAAA", "KAAA", 1.0, "B", held_out="KBBB")
    with pytest.raises(ValueError, match="a fold of KBBB, not KAAA"):
        select_module.arm_score(tmp_path, "B", "full", 1337, ["KAAA"])
    # a fold of another arm: another variant, seed, shape or training value, or the selection `all`
    for name, kwargs in {"variant": {"variant": "constants"}, "seed": {"seed": 2024}, "shape": {"d_model": 128},
                         "rope base": {"rope_base": 500.0}, "weight decay": {"weight_decay": 0.01},
                         "patience": {"patience": 5}, "epochs": {"max_epochs": 40}, "warm-up": {"warmup_steps": 100},
                         "clip": {"clip_norm": 0.5}, "batch": {"tokens_per_batch": 8_192},
                         "selection": {"rule": "all"}}.items():
        run = tmp_path / name / "D_full_s1337" / "KAAA"
        write_fold(run, "KAAA", 1.0, "D", **kwargs)
        with pytest.raises(ValueError, match="not a fold of D_full_s1337"):
            select_module.arm_score(tmp_path / name, "D", "full", 1337, ["KAAA"])
    # a fold whose held-out loss is not a number is refused by name, not left out of the choice
    write_fold(tmp_path / "nan" / "A_full_s1337" / "KAAA", "KAAA", float("nan"))
    with pytest.raises(ValueError, match="not a finite score"):
        select_module.arm_score(tmp_path / "nan", "A", "full", 1337, ["KAAA"])


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

    def __call__(self, step, log, started):
        self.ran.append(step.name)
        started(4242)
        log.write_text("stand-in")
        if step.name == self.fail_at:
            step.out.mkdir(parents=True)           # a kill: the directory without its last file
            return 1
        if step.runner == "prior_select":
            return select_module.main(list(step.argv))
        if step.runner == "prior_train" and step.name.startswith("base/"):
            step.out.mkdir(parents=True)
            (step.out / "checkpoint.pt").write_text("checkpoint")
            step.done.write_text("{}")
        elif step.runner == "prior_train":
            arm, airport = step.name.split("/")
            configuration, variant, seed = arm.split("_")
            write_fold(step.out, airport, self.LOSS[arm], configuration, variant, int(seed[1:]))
        else:
            step.out.mkdir(parents=True)
            step.done.write_text("{}")
        return 0


def test_a_campaign_runs_every_step_once_makes_its_choices_and_resumes_after_a_kill(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_module, "load_candidates", lambda instructions: dict.fromkeys(AIRPORTS))
    campaign = tmp_path / "campaign"
    git = {"head": "a" * 40, "dirty": False}
    rec = campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 4.0, git,
                                        behaviour=same)
    killed = StandIn(fail_at="B_full_s1337/KCCC")
    with pytest.raises(SystemExit, match="B_full_s1337/KCCC failed"):
        campaign_module.run_campaign(campaign, rec, "cpu", killed, lambda line: None, lambda: git, behaviour=same)
    # resumed from another commit whose code behaves the same: the commit is information (D108)
    later = {"head": "c" * 40, "dirty": False}
    rec = campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 4.0, later,
                                        behaviour=same)
    runner = StandIn()
    campaign_module.run_campaign(campaign, rec, "cpu", runner, lambda line: None, lambda: later, behaviour=same)
    assert json.loads((campaign / "campaign.json").read_text())["running"] is None
    assert runner.ran[0] == "B_full_s1337/KCCC"                         # the killed step again, first
    assert not set(runner.ran) & set(killed.ran[:-1])                  # nothing done is run again
    stored = json.loads((campaign / "campaign.json").read_text())
    assert stored["schema"] == campaign_module.CAMPAIGN_SCHEMA and stored["behaviour"] == same(None)
    assert [s["git"]["head"][0] for s in stored["steps"]] == ["a"] * len(killed.ran) + ["c"] * len(runner.ran)
    assert [a["step"] for a in stored["aborted"]] == ["B_full_s1337/KCCC"]
    assert (campaign / "B_full_s1337" / Path(stored["aborted"][0]["moved_to"]).name).exists()
    configuration = json.loads((campaign / "choice_configuration.json").read_text())
    variant = json.loads((campaign / "choice_variant.json").read_text())
    assert configuration["chosen"] == "B" and variant["chosen"] == "full"
    assert (campaign / "base" / "free_generation" / "readout.json").exists()
    trained = [name for name in killed.ran + runner.ran if "/" in name and name.count("/") == 1
               and not name.endswith(("validation", "free_generation"))]
    assert len(trained) == 31 + 1                                      # 31 runs, the killed one twice


def test_a_resume_with_other_inputs_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_module, "load_candidates", lambda instructions: dict.fromkeys(AIRPORTS))
    campaign = tmp_path / "campaign"
    git = {"head": "a" * 40, "dirty": False}
    campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 4.0, git, behaviour=same)
    with pytest.raises(SystemExit, match="row_interval_s"):
        campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 2.0, git, behaviour=same)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "file").write_text("x")
    with pytest.raises(SystemExit, match="is no campaign"):
        campaign_module.open_campaign(tmp_path / "other", tmp_path / "artefact", tmp_path / "executor", 4.0, git,
                                      behaviour=same)


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


def test_a_campaign_stops_when_its_code_behaves_otherwise_or_its_tree_has_changes(tmp_path, monkeypatch):
    """D108: before each step the behaviour check on fixed inputs, compared with the campaign's start; a difference
    stops it by name, as uncommitted changes do (a formal campaign runs on a clean checkout). A resume while its step
    runs is refused."""
    monkeypatch.setattr(campaign_module, "load_candidates", lambda instructions: dict.fromkeys(AIRPORTS))
    campaign = tmp_path / "campaign"
    git = {"head": "a" * 40, "dirty": False}
    rec = campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 4.0, git,
                                        behaviour=same)
    runner = StandIn()
    with pytest.raises(SystemExit, match=r"behaves otherwise.*\['words'\]"):
        campaign_module.run_campaign(campaign, rec, "cpu", runner, lambda line: None, lambda: git,
                                     behaviour=lambda i: {**same(i), "words": [[1]]})
    with pytest.raises(SystemExit, match="uncommitted changes"):
        campaign_module.run_campaign(campaign, rec, "cpu", runner, lambda line: None,
                                     lambda: {"head": "a" * 40, "dirty": True}, behaviour=same)
    # the code on the disk sets another temperature (the behaviour check's process gives the settings, D108): stopped
    disk = {**same(None), "settings": {**campaign_module.settings(), "temperature": 0.5}}
    with pytest.raises(SystemExit, match=r"behaves otherwise.*\['settings'\]"):
        campaign_module.run_campaign(campaign, rec, "cpu", runner, lambda line: None, lambda: git,
                                     behaviour=lambda i: disk)
    with monkeypatch.context() as patch:            # this process plans with other settings than the disk's: refused
        patch.setattr(campaign_module, "FREE_GENERATION", {**campaign_module.FREE_GENERATION, "samples": 3})
        with pytest.raises(SystemExit, match=r"the campaign's start record sets \['free_generation'\] otherwise than "
                                             r"this campaign's process"):
            campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 4.0, git,
                                          behaviour=same)
        with pytest.raises(SystemExit, match="free_generation"):
            campaign_module.open_campaign(tmp_path / "new", tmp_path / "artefact", tmp_path / "executor", 4.0, git,
                                          behaviour=lambda i: {**same(i), "settings": disk["settings"]})
    assert runner.ran == []                                            # nothing ran
    stored = json.loads((campaign / "campaign.json").read_text())
    stored["running"] = {"step": "A_full_s1337/KAAA", "pid": 4242, "utc": "x"}
    (campaign / "campaign.json").write_text(json.dumps(stored))
    monkeypatch.setattr(campaign_module, "alive_step", lambda running: True)
    with pytest.raises(SystemExit, match="still runs as PID 4242"):
        campaign_module.open_campaign(campaign, tmp_path / "artefact", tmp_path / "executor", 4.0, git)


def test_the_behaviour_check_gives_one_answer_and_sees_a_change_of_the_training_or_the_speaker(monkeypatch):
    """D108: `prior_behaviour` on fixed inputs gives the same answer twice; a change of the training's loss or of the
    speaker's draw changes it."""
    from ts_transformer.experiments import prior_behaviour
    from ts_transformer.instructions.words import Words
    from ts_transformer.prior import speaker as speaker_module
    from ts_transformer.prior import train as train_module
    from ts_transformer.tests.support import fixture_days, instruction_spec
    from ts_transformer.tests.test_prior_speaker import finals

    words, days = Words(instruction_spec()), fixture_days()
    answer = prior_behaviour.behaviour(words, finals(), days)
    assert answer == prior_behaviour.behaviour(words, finals(), days)
    assert {"train_config", "model_config", "first_step_runway", "constants_loss", "heard", "inputs",
            "speaking_loop"} <= set(answer)
    assert [len(w) for w in answer["speaking_loop"]["words"]] == [3, 5, 7]       # each flight to the row it ended in
    assert len(answer["train_loss"]) == 2 and len(answer["words"]) == prior_behaviour.SAID
    with monkeypatch.context() as patch:
        nll = train_module.step_nll
        patch.setattr(train_module, "step_nll", lambda logits, rows: nll(logits, rows) * 1.0001)
        changed = prior_behaviour.behaviour(words, finals(), days)
        assert changed["train_loss"] != answer["train_loss"]
    with monkeypatch.context() as patch:
        patch.setattr(speaker_module, "draw", lambda probabilities, numbers: probabilities.argmax(-1, keepdim=True))
        changed = prior_behaviour.behaviour(words, finals(), days)
        assert changed["train_loss"] == answer["train_loss"] and changed["words"] != answer["words"]


def test_the_behaviour_check_refuses_a_batch_mode_of_the_masks_that_says_otherwise(monkeypatch):
    """B14: `prior_behaviour` speaks its fixed inputs with the procedure masks in both modes; a batch mode that says
    otherwise (its edge 100 m up, its own check of the fixed rows skipped) is refused by name."""
    from ts_transformer.experiments import prior_behaviour
    from ts_transformer.instructions.words import Words
    from ts_transformer.prior import procedure
    from ts_transformer.tests.support import fixture_days, instruction_spec
    from ts_transformer.tests.test_prior_speaker import finals

    edge = procedure._Gathered.edge_m
    monkeypatch.setattr(procedure, "require_same_masks", lambda finals, words: None)
    monkeypatch.setattr(procedure._Gathered, "edge_m", lambda self, e, n: edge(self, e, n) + 100.0)
    with pytest.raises(ValueError, match="says otherwise with the procedure masks' batch mode"):
        prior_behaviour.behaviour(Words(instruction_spec()), finals(), fixture_days())


def test_the_behaviour_check_covers_the_sentence_rows_the_selection_the_choices_and_the_settings(monkeypatch):
    """B12 (D108): the answer changes when the training sentences' inputs (`inputs.sentence_rows`: the landings counted
    at another UTC), the selection (`selection.left_out`) or a rule of `prior_select` (the seed scale) changes; its
    settings are the campaign's as a second process loads them."""
    import os
    import subprocess
    import sys

    from ts_transformer.experiments import prior_behaviour
    from ts_transformer.instructions.words import Words
    from ts_transformer.prior import inputs as inputs_module
    from ts_transformer.prior import selection as selection_module
    from ts_transformer.repo_layout import REPO_ROOT
    from ts_transformer.tests.support import fixture_days, instruction_spec
    from ts_transformer.tests.test_prior_speaker import finals

    words, days = Words(instruction_spec()), fixture_days()
    answer = prior_behaviour.behaviour(words, finals(), days)
    assert {"settings", "selection", "select_rules", "sentence_rows"} <= set(answer)
    assert answer["selection"]["left_out"]["landed"]["landed"] == {"False": None, "True": "fault"}
    assert answer["selection"]["kept"]["landed"]["landed"] == {"False": True, "True": False}
    assert answer["selection"]["side"]["all"]["crossed_too_high"] == {"False": "inside", "True": "inside"}
    assert answer["select_rules"]["tie"]["chosen"] == "D" and answer["select_rules"]["zero"]["within"] == ["B"]
    assert answer["select_rules"]["edge"]["within"] == ["A", "B"]
    assert answer["select_rules"]["variant_edge"]["chosen"] == "full"
    assert answer["select_rules"]["variant_zero"]["chosen"] == "constants"
    for patch_it, key in ((lambda patch: patch.setattr(inputs_module, "utc_s",
                                                       lambda text, real=inputs_module.utc_s: real(text) + 600.0),
                           "sentence_rows"),
                          (lambda patch: patch.setattr(selection_module, "LANDING", "crossed_too_high"), "selection"),
                          (lambda patch: patch.setattr(select_module, "seed_scale", lambda scores, seeds: 0.0),
                           "select_rules")):
        with monkeypatch.context() as patch:
            patch_it(patch)
            changed = prior_behaviour.behaviour(words, finals(), days)
        assert changed[key] != answer[key] and changed["train_loss"] == answer["train_loss"], key
    from ts_transformer.experiments import prior_behaviour as behaviour_module
    from ts_transformer.experiments import prior_free_generation as generation_module
    from ts_transformer.prior.landings import LandingIndex
    from ts_transformer.prior.loop import LoopRows

    # B13: each new part of the answer moves when what it holds changes
    counts = LandingIndex.counts_before
    rows_init = LoopRows.__init__
    assert any(any(v != 0 for v in row) for row in answer["inputs"]["landings"])
    for patch_it, keys in (
            # the landings counted 30 s off the rows' UTC (the fixed sentence's own and same-second landings lie
            # between two of its rows)
            (lambda patch: patch.setattr(LandingIndex, "counts_before", lambda self, times, *, without:
                                         counts(self, np.asarray(times) - 30.0, without=without)),
             ("sentence_rows",)),
            # every aircraft of the loop given the first one's landings (D105)
            (lambda patch: patch.setattr(LoopRows, "__init__", lambda self, geometries, landings, *given, **named:
                                         rows_init(self, geometries, [landings[0]] * len(landings), *given, **named)),
             ("speaking_loop",)),
            (lambda patch: patch.setattr(behaviour_module, "kept", lambda rule, outcome, faulty: True), ("selection",)),
            (lambda patch: patch.setattr(behaviour_module, "side", lambda rule, outcome, faulty: "inside"),
             ("selection",)),
            (lambda patch: patch.setattr(campaign_module, "FREE_GENERATION",
                                         {**campaign_module.FREE_GENERATION, "per_airport": 100}),
             ("campaign_plan", "settings")),
            (lambda patch: patch.setattr(generation_module, "draw",
                                         lambda sentences, flights, airports, per_airport, seed: sorted(sentences)[:4]),
             ("free_generation_draw",))):
        with monkeypatch.context() as patch:
            patch_it(patch)
            changed = prior_behaviour.behaviour(words, finals(), days)
        assert all(changed[key] != answer[key] for key in keys), keys
    loaded = subprocess.run([sys.executable, "-c", "import json; from ts_transformer.experiments.prior_campaign import "
                             "settings; print(json.dumps(settings()))"], cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}, capture_output=True, text=True,
                            check=True)
    assert json.loads(loaded.stdout) == answer["settings"] == campaign_module.settings()


def test_a_resume_marks_spent_the_claim_of_a_val_step_killed_after_its_readout(tmp_path, monkeypatch):
    """D128 along the campaign's path: a val step killed between its readout and its claim's mark is done (its
    readout.json is there) and never runs again; the campaign — its loop, on the steps of its own plan — marks its claim
    spent; a smoke campaign reads no val."""
    campaign = tmp_path / "campaign"
    rec = {**record(tmp_path), "smoke": None}
    for step, chosen in (("configuration", "A"), ("variant", "full")):
        campaign.mkdir(exist_ok=True)
        (campaign / f"choice_{step}.json").write_text(json.dumps({"chosen": chosen}))
    steps = campaign_module.plan(campaign, rec, "cpu")
    val = [step for step in steps if step.runner in campaign_module.VAL_READERS and step.name.startswith("base/")]
    assert [step.runner for step in val] == list(campaign_module.VAL_READERS)
    for step in steps:                                                  # every step done: a resume runs none
        step.done.parent.mkdir(parents=True, exist_ok=True)
        if not step.done.exists():                                      # the choices' files are their own
            step.done.write_text("{}")
    run = campaign / "base" / "run"
    for step in val:
        (run / f"val_read_{step.runner}.json").write_text(json.dumps({"reader": step.runner, "out": str(step.out),
                                                                      "utc": "x", "options": {}}))
    campaign_module.settle_val_steps(campaign, {**rec, "smoke": {"sample": 5}}, steps)
    assert not any("spent_utc" in json.loads(path.read_text()) for path in run.glob("val_read_*.json"))
    git = {"head": "a" * 40, "dirty": False}
    campaign_module.run_campaign(campaign, {**rec, "behaviour": same(None), "steps": [], "aborted": [], "running": None},
                                 "cpu", StandIn(), lambda line: None, lambda: git, behaviour=same)
    spent = [json.loads((run / f"val_read_{step.runner}.json").read_text()) for step in val]
    assert len(spent) == 2 and all("spent_utc" in claim for claim in spent)
