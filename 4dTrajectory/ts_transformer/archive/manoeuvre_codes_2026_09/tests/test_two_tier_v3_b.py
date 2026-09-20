"""Two-tier plan v3, stage B (`docs/2026-09-18_two_tier_plan_v3.zh.md` §5.2 / §6.2): the arm
declaration pins the numbered decisions — the executor is the grid's L60_D20 (D30), the token
span S ∈ {20, 60} s in three configurations (D31 / D38), the learned K16 codebook and the command
vocabulary (D35 / D36), ONE cohort for every arm (D33: the grid's L60_D60 cohort), the readings
from L-1 (D41) — and every run and reading variant has its intent."""

from __future__ import annotations

import json

import pytest

from ts_transformer.config import (
    MANOEUVRE_TOKENIZER_COMMAND_VOCABULARY, MANOEUVRE_TOKENIZER_LEARNED, PLAN_CONDITIONING_MANOEUVRE_CODE,
    RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, default_anchor, token_hold, token_span_s, token_step_s,
)
from ts_transformer.experiments.support import TS_DIR, arm_config, declaration_base

DECLARATION = TS_DIR / "docs" / "experiments" / "two_tier_v3_b_arms.json"
INTENTS = TS_DIR / "docs" / "experiments" / "intents.json"
CAMPAIGN = "two_tier_v3_b_20260919"
#: configuration → (horizon Δ, token span S, token step, hold, N1, the random-anchor future floor)
CONFIGURATIONS = {"S20": (20.0, 20.0, 20.0, 1, 2, 20.0), "S60h60": (60.0, 60.0, 20.0, 3, 6, 60.0), "S60held": (20.0, 60.0, 20.0, 3, 2, 20.0)}
TOKENIZERS = ("K16", "cv")
SEEDS = (1337, 2024)
READINGS = ("lockstep-C", "lockstep-A", "lockstep-A-truth")


def _declaration() -> dict:
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


def test_the_stage_b_declaration_pins_the_decisions():
    declaration = _declaration()
    base = declaration_base(declaration)
    arms = declaration["arms"]
    assert declaration["predict"] is False
    # D33: one cohort for every arm — the grid's L60_D60 cohort (records ≥ 120 s), not a new file
    assert declaration["development_cohort"].endswith("two_tier_v3_grid_20260918/cohorts/L60_D60/development_cohort.json")
    assert "{airport}" in declaration["development_cohort"] and not any("development_cohort" in arm for arm in arms)
    # file order = queue order (§8.2): K16 first — S20, S60-h60, S60-held — then the command vocabulary, the seeds together
    expected = [f"{c}_{t}_s{s}" for t in TOKENIZERS for c in CONFIGURATIONS for s in SEEDS]
    assert [arm["key"] for arm in arms] == expected and len(arms) == 12
    stage_b = declaration["stage_b"]
    assert set(stage_b["configurations"]) == set(CONFIGURATIONS) and stage_b["seed_line_from"].endswith("gate/after_L120_D120/grid_gate.json")
    assert "{arm}" in stage_b["codebook_dir"]
    # the baselines the queue flies itself (never another campaign's payload): the stage A checkpoints, read no-token
    assert set(stage_b["baselines"]) == {"L60_D20", "L60_D60_exec20"}
    for name, spec in stage_b["baselines"].items():
        assert "{seed}" in spec["checkpoint"] and f"two_tier_v3_grid_20260918/{name.split('_exec')[0]}_s{{seed}}/checkpoint.pt" in spec["checkpoint"]
    assert stage_b["baselines"]["L60_D20"]["execute_s"] == 0 and stage_b["baselines"]["L60_D60_exec20"]["execute_s"] == 20.0
    for arm in arms:
        configuration, tokenizer, seed_text = arm["key"].split("_")
        assert arm["configuration"] == configuration
        horizon, span, step, hold, n1, floor = CONFIGURATIONS[configuration]
        config, settings = arm_config(base, arm["overrides"])
        assert config.seed == int(seed_text[1:]) and config.split_seed == 1337                          # D1 / D9
        assert config.plan_conditioning == PLAN_CONDITIONING_MANOEUVRE_CODE                             # D32
        assert config.seq_len * config.dt_s == 60.0 and default_anchor(config) == 29                    # D30: L60, first prediction at L-1
        assert config.control_horizon_s == horizon and config.n_segments == n1                          # D31 / D43
        assert (token_span_s(config), token_step_s(config), token_hold(config)) == (span, step, hold)  # D31 / D38
        assert config.random_train_anchor_min_future_s == floor                                         # B-dev1 (S60-held: the horizon's, Q3)
        assert config.random_train_anchor and config.random_train_anchor_l1_share == 0.0 and config.anchor_floor_index == 0   # D30
        assert config.random_train_anchor_sampling == RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM
        assert config.control_thrust_parameterization == "specific-force+path-angle" and config.control_dynamics_model == "first-order-lag"
        assert config.epochs == config.patience == 180 and config.learning_rate == pytest.approx(3e-5) and config.batch_size == 512   # D34
        assert config.checkpoint_selection_metric == "fixed-anchor-common-grid-ade" and not config.manoeuvre_codebook               # D34: joint
        if tokenizer == "K16":
            assert config.manoeuvre_tokenizer == MANOEUVRE_TOKENIZER_LEARNED and config.manoeuvre_fsq_levels == (4, 4)               # D35
        else:
            assert config.manoeuvre_tokenizer == MANOEUVRE_TOKENIZER_COMMAND_VOCABULARY and config.manoeuvre_fsq_levels == ()        # D36
        # the arm restates its configuration's numbers (the file is the record)
        declared = stage_b["configurations"][configuration]
        for field in ("control_horizon_s", "n_segments", "random_train_anchor_min_future_s", "manoeuvre_token_s", "manoeuvre_token_step_s"):
            assert settings.get(field, 0.0) == declared.get(field, 0.0), (arm["key"], field)
        assert settings.get("manoeuvre_token_s", 0.0) != horizon and settings.get("manoeuvre_token_step_s", 0.0) != horizon   # 0 spells the horizon
        assert declared["baseline"] in stage_b["baselines"]
    # the two configurations on the 20 s executor are judged against B0, S60-h60 against the re-flown A3-a
    assert stage_b["configurations"]["S20"]["baseline"] == stage_b["configurations"]["S60held"]["baseline"] == "L60_D20"
    assert stage_b["configurations"]["S60h60"]["baseline"] == "L60_D60_exec20"


def test_every_stage_b_run_and_reading_has_its_intent():
    intents = json.loads(INTENTS.read_text(encoding="utf-8"))["campaigns"]
    campaign = intents[CAMPAIGN]
    keys = [arm["key"] for arm in _declaration()["arms"]]
    assert set(campaign["runs"]) == set(keys)
    assert set(campaign["variants"]) == {f"{key}@{reading}" for key in keys for reading in READINGS}
    assert campaign["title"] and campaign["intent"] and "two_tier_v3_b_arms.json" in campaign["design"]
    # B0's baseline re-read is a reading variant of the grid's L60_D20 arms
    grid = intents["two_tier_v3_grid_20260918"]["variants"]
    assert all(f"L60_D20_s{seed}@lockstep-none-b" in grid and f"L60_D60_s{seed}@lockstep-none-exec20-b" in grid for seed in SEEDS)


def test_the_b_queue_groups_the_arms_by_configuration_and_vocabulary_and_gates_the_prior_steps(tmp_path, capsys):
    from ts_transformer.experiments import two_tier_b_queue as queue

    declaration = _declaration()
    groups = queue.groups_of(declaration)
    assert list(groups) == ["S20_K16", "S60h60_K16", "S60held_K16", "S20_cv", "S60h60_cv", "S60held_cv"]
    assert groups["S20_K16"] == {1337: "S20_K16_s1337", 2024: "S20_K16_s2024"}
    broken = _declaration()
    broken["arms"][1]["overrides"]["seed"] = 1337
    with pytest.raises(ValueError, match="not named"):                # the key says s2024, the config 1337
        queue.groups_of(broken)
    broken["arms"][1]["key"] = "S20_K16_s1337"
    with pytest.raises(ValueError, match="same group and seed"):
        queue.groups_of(broken)
    broken = _declaration()
    broken["arms"][0]["configuration"] = "S60held"
    with pytest.raises(ValueError, match="names configuration"):
        queue.groups_of(broken)
    # the baseline checkpoints, the cohort and the seed line are other campaigns' artefacts: here, tmp
    for spec in declaration["stage_b"]["baselines"].values():
        spec["checkpoint"] = str(tmp_path / "ckpt" / spec["checkpoint"].split("/")[-2] / "checkpoint.pt")
    declaration["development_cohort"] = str(tmp_path / "cohort.json")
    declaration["stage_b"]["seed_line_from"] = str(tmp_path / "grid_gate.json")
    declaration["stage_b"]["codebook_dir"] = str(tmp_path / "codebooks" / "{arm}")
    campaign = tmp_path / "campaign"
    arms_path = tmp_path / "arms.json"
    # step 0: the needed baselines × seed flown by this code on the declared cohort, with records, then their failure modes
    zero = queue.baseline_steps(declaration=declaration, campaign=campaign, airport="KRDU", device="cpu", seeds=list(SEEDS),
                                names=["L60_D20", "L60_D60_exec20"])
    assert [step.label for step in zero] == [f"baseline {name} seed {seed}: {what}" for name in ("L60_D20", "L60_D60_exec20") for seed in SEEDS
                                             for what in ("lockstep none", "failure modes")]
    first = zero[0].command
    assert first[first.index("--executor") + 1] == str(tmp_path / "ckpt" / "L60_D20_s1337" / "checkpoint.pt") and "--execute-s" not in first
    assert first[first.index("--cohort") + 1] == str(tmp_path / "cohort.json") and "--write-records" in first and first[first.index("--protocol") + 1] == "none"
    assert zero[0].artefact == campaign / "baseline" / "L60_D20_s1337" / "L-1" / "manoeuvre_lockstep.json"
    exec20 = zero[4].command
    assert exec20[exec20.index("--execute-s") + 1] == "20" and zero[4].artefact == campaign / "baseline" / "L60_D60_exec20_s1337" / "L-1" / "manoeuvre_lockstep.json"
    assert zero[1].artefact == campaign / "failure_modes" / "L60_D20_s1337_none" / "failure_modes.json" and all(step.gate is None for step in zero)
    steps = queue.group_steps("S60held_K16", groups["S60held_K16"], declaration=declaration, declaration_path=arms_path,
                              campaign=campaign, airport="KRDU", device="cpu")
    labels = [step.label for step in steps]
    assert labels[0] == "S60held_K16: train S60held_K16_s1337, S60held_K16_s2024" and steps[0].artefact is None
    assert steps[0].command[steps[0].command.index("--only") + 1:] == ["S60held_K16_s1337", "S60held_K16_s2024"]
    assert labels[1:5] == ["S60held_K16_s1337: codebook", "S60held_K16_s1337: lockstep C", "S60held_K16_s1337: failure modes C",
                           "S60held_K16_s1337: code atlas"]
    assert labels[9] == "S60held_K16: gate b1 (C vs none)" and steps[9].gate is None
    assert all(step.gate == (steps[9].artefact, "b1") for step in steps[10:]) and labels[-1] == "S60held_K16: gate b (A vs none)"
    assert [step.label.split(": ")[1] for step in steps[10:15]] == ["prior (seed 1337)", "lockstep A", "lockstep A-truth", "failure modes A", "prior (seed 2024)"]
    c = steps[2].command
    assert "--write-records" in c and c[c.index("--protocol") + 1] == "C" and c[c.index("--codebook") + 1] == str(tmp_path / "codebooks" / "S60held_K16_s1337")
    assert steps[2].artefact == campaign / "lockstep" / "S60held_K16_s1337" / "C" / "manoeuvre_lockstep.json"
    gate = steps[9].command
    assert f"1337={campaign / 'baseline' / 'L60_D20_s1337' / 'L-1'}" in gate and f"2024={campaign / 'lockstep' / 'S60held_K16_s2024' / 'C'}" in gate
    h60 = queue.group_steps("S60h60_K16", groups["S60h60_K16"], declaration=declaration, declaration_path=arms_path, campaign=campaign,
                            airport="KRDU", device="cpu")
    assert f"2024={campaign / 'baseline' / 'L60_D60_exec20_s2024' / 'L-1'}" in h60[9].command
    declaration["stage_b"]["configurations"]["S20"]["baseline"] = "nowhere"
    with pytest.raises(ValueError, match="stage_b.baselines does not declare"):
        queue.group_steps("S20_K16", groups["S20_K16"], declaration=declaration, declaration_path=arms_path, campaign=campaign, airport="KRDU", device="cpu")
    declaration["stage_b"]["configurations"]["S20"]["baseline"] = "L60_D20"
    assert gate[gate.index("--seed-line-from") + 1] == str(tmp_path / "grid_gate.json") and steps[9].artefact == campaign / "gate" / "b1_S60held_K16" / "relative_gate.json"
    prior = steps[10].command
    assert prior[prior.index("--seed") + 1] == "1337" and steps[10].artefact == campaign / "priors" / "S60held_K16_s1337" / "prior.pt"
    a_truth = next(step for step in steps if step.label.endswith("lockstep A-truth")).command
    assert "--write-records" not in a_truth and a_truth[a_truth.index("--prior") + 1] == str(campaign / "priors" / "S60held_K16_s1337" / "prior.pt")
    # a dry run refuses a missing checkpoint / cohort and a pending seed line by name, then lists the steps, the baselines first
    arms_path.write_text(json.dumps(declaration), encoding="utf-8")
    argv = ["--arms", str(arms_path), "--campaign", str(campaign), "--airport", "KRDU", "--groups", "S20_K16", "--dry-run"]
    with pytest.raises(SystemExit):
        queue.main(argv)
    assert "the baselines need are missing" in capsys.readouterr().err
    (tmp_path / "cohort.json").write_text("{}", encoding="utf-8")
    for name in ("L60_D20", "L60_D60"):
        for seed in SEEDS:
            (tmp_path / "ckpt" / f"{name}_s{seed}").mkdir(parents=True)
            (tmp_path / "ckpt" / f"{name}_s{seed}" / "checkpoint.pt").write_bytes(b"")
    (tmp_path / "grid_gate.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):                                       # a pending seed line is refused up front
        queue.main(argv)
    assert "no verdict" in capsys.readouterr().err
    (tmp_path / "grid_gate.json").write_text(json.dumps({"verdict": {"seed_line": {}}}), encoding="utf-8")
    with pytest.raises(SystemExit):                                       # a seed line without the gate's metrics too
        queue.main(argv)
    assert "lacks p75" in capsys.readouterr().err
    line = {m: {"p50": 0.0, "p75": 0.1} for m in ("established_all", "established_vectored", "vectored_ade_mean_m")}
    (tmp_path / "grid_gate.json").write_text(json.dumps({"verdict": {"seed_line": line}}), encoding="utf-8")
    assert queue.main(argv) == 0
    printed = capsys.readouterr().out
    assert printed.count("GROUP ") == 2 and printed.index("GROUP baselines") < printed.index("GROUP S20_K16")
    # only the baseline the selected group needs is flown in step 0
    assert "todo baseline L60_D20 seed 2024: failure modes" in printed and "L60_D60_exec20" not in printed
    assert "[if gate b1 passes]" in printed and "todo S20_K16_s1337: codebook" in printed
    assert queue.seed_line_problem(tmp_path / "nowhere.json") == "missing"
    assert not (campaign / queue.PID_FILE).exists()
    with pytest.raises(SystemExit):
        queue.main(argv[:-3] + ["--groups", "S99_K16", "--dry-run"])
    # the chain: a failed gate B1 skips the prior steps, a passed one runs them; a written artefact is skipped
    gate_json = campaign / "gate" / "b1_x" / "relative_gate.json"
    gate_json.parent.mkdir(parents=True)
    gate_json.write_text(json.dumps({"verdicts": {"b1": {"pass": False}}}), encoding="utf-8")
    ran = tmp_path / "ran"
    plan = {"x": [queue.Step("x: gate b1 (C vs none)", ["true"], gate_json),
                  queue.Step("x: prior", ["touch", str(ran)], ran, gate=(gate_json, "b1"))]}
    assert queue._run(plan, campaign) == 0 and not ran.exists()
    gate_json.write_text(json.dumps({"verdicts": {"b1": {"pass": True}}}), encoding="utf-8")
    assert queue._run(plan, campaign) == 0 and ran.exists()
    printed = capsys.readouterr().out
    assert "GATE B1 FAIL for x" in printed and "skip x: gate b1" in printed and printed.count("GROUP x complete") == 2
    assert queue._run({"y": [queue.Step("y: boom", ["false"], None)]}, campaign) != 0
    assert "STOP: y: boom failed" in capsys.readouterr().out
