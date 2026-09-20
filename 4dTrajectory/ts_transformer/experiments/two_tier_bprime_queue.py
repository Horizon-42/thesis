"""Stage B′ queue (two-tier v3 §5.2.5, D42): the instruction-vocabulary campaign, one step at a time.

    python run_ts.py two_tier_bprime_queue --arms docs/experiments/two_tier_v3_bprime_arms.json \\
        --campaign 4dTrajectory/outputs/KRDU/experiments/two_tier_v3_bprime_20260920 --airport KRDU [--groups I20] [--dry-run]

The declaration (`two_tier_v3_bprime_arms.json`) names ONE development cohort (D33 / D47: the grid's
L60_D60 cohort), the instruction executor's base (the grid's L60_D20 with `plan_conditioning =
instruction` and the vocabulary artefact B0′ wrote), a ``stage_b`` block — the seed line's source
(the stage A grid gate, D39), the baselines (a stage A checkpoint per seed, flown by THIS queue on
the B cohort), the configurations and the prior's settings — and the arms, ``<configuration>_s<seed>``.

The steps, serial, each skipped when its last artefact exists (a crash is cheap: rerun the same
command), and each STOPPING the queue when it fails (D11):

    step 0    baselines: every baseline the selected groups are judged against × seed — the
              no-token executor flown ``protocol none`` from L−1 on the B cohort with records
              (`manoeuvre_lockstep`), and its failure-mode table
    B1′       per group: train the two arms (`frame_ablation --only`), fly each ``protocol
              truth-instruction`` with records (the truth's words by flown position — the upper
              bound), its failure modes, then gate B1 against the baseline (D57)
    prior     per seed, only past gate B1: `instruction_prior` on the group's vocabulary and that
              seed's executor (its readings are stage B3′'s first line)

B2′ (the executor's closed-loop fine-tuning, D49), B3′'s decoding protocols (D56) and B4′ (τ = 5 s)
are not built: the queue refuses a declaration that names them. Every line is stamped; ``CELL
<group> complete`` is what a watcher waits for; a PID file refuses a second queue on the campaign.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from ts_transformer.experiments.instruction_prior import PRIOR_FILE
from ts_transformer.experiments.support import REPO_ROOT, arm_config, declaration_base
from ts_transformer.experiments.two_tier_grid_queue import live_pid
from ts_transformer.manoeuvre.gates import GRID_SEED_LINE_QUANTILE, RELATIVE_METRICS
from ts_transformer.manoeuvre.lockstep import PROTOCOL_NONE, PROTOCOL_TRUTH_INSTRUCTION
from ts_transformer.repo_layout import TS_SCRIPT as RUN_TS

MINIMUM_FREE_BYTES = 3 * 1024**3
PID_FILE = "two_tier_bprime_queue.pid"
LOCKSTEP_DIR = "lockstep"
#: The artefact each runner writes last — MUST match `manoeuvre_lockstep`, `executor_failure_modes` and
#: `executor_relative_gate` (literals there; the prior's is imported). A wrong name is loud: the runner
#: refuses an existing --out.
LOCKSTEP_JSON, FAILURE_MODES_JSON, GATE_JSON = "manoeuvre_lockstep.json", "failure_modes.json", "relative_gate.json"
BASELINE_DIR = "baseline"
#: What a declaration may ask this queue for; anything else in ``stage_b`` is a step not built.
STAGE_B_KEYS = {"seed_line_from", "baselines", "configurations", "prior"}
PRIOR_SETTINGS = ("epochs", "patience", "batch_size", "learning_rate", "d_model", "n_layers", "n_heads", "d_ff", "dropout")


class Step:
    """One queue step: ``label``, ``command``, the ``artefact`` whose existence skips it (None: the
    runner resumes by itself), and ``gate`` — a ``(relative_gate.json, row)`` this step needs to have
    PASSED, read at run time."""

    def __init__(self, label: str, command: list[str], artefact: Path | None, gate: tuple[Path, str] | None = None):
        self.label, self.command, self.artefact, self.gate = label, command, artefact, gate


def groups_of(declaration: dict[str, Any]) -> dict[str, dict[int, str]]:
    """``{group: {seed: arm key}}`` in declaration order; a group is one configuration and needs
    exactly the two seeds every gate is judged on."""
    unknown = set(declaration["stage_b"]) - STAGE_B_KEYS
    if unknown:
        raise ValueError(f"stage_b names {sorted(unknown)}: steps this queue does not build (only {sorted(STAGE_B_KEYS)})")
    base = declaration_base(declaration)
    groups: dict[str, dict[int, str]] = {}
    for arm in declaration["arms"]:
        config, _settings = arm_config(base, arm.get("overrides", {}))      # an unrunnable arm fails here, dry run included
        group, _sep, seed_text = arm["key"].rpartition("_s")
        if not seed_text.isdigit() or int(seed_text) != config.seed:
            raise ValueError(f"arm {arm['key']} is not named <configuration>_s<seed> for its seed {config.seed}")
        if arm.get("configuration") != group:
            raise ValueError(f"arm {arm['key']} names configuration {arm.get('configuration')!r}, not its key's {group!r}")
        if group not in declaration["stage_b"]["configurations"]:
            raise ValueError(f"arm {arm['key']}: configuration {group!r} is not declared under stage_b.configurations")
        if config.seed in groups.setdefault(group, {}):
            raise ValueError(f"arms {groups[group][config.seed]} and {arm['key']} are the same group and seed")
        groups[group][config.seed] = arm["key"]
    for group, by_seed in groups.items():
        if len(by_seed) != 2:
            raise ValueError(f"group {group} has seeds {sorted(by_seed)}; every gate is judged on exactly two")
    return groups


def _seeded(template: str, *, airport: str, seed: int) -> Path:
    return REPO_ROOT / template.format(airport=airport, seed=seed)


def baseline_reading(campaign: Path, name: str, seed: int) -> Path:
    return campaign / BASELINE_DIR / f"{name}_s{seed}" / "L-1"


def baseline_steps(*, declaration: dict[str, Any], campaign: Path, airport: str, device: str, seeds: list[int],
                   names: list[str]) -> list[Step]:
    """Step 0: the baselines ``names`` × seed, flown by THIS code on the declaration's cohort — the
    no-token reading from L−1 with records, and its failure-mode table."""
    py = sys.executable
    cohort = REPO_ROOT / declaration["development_cohort"].format(airport=airport)
    steps: list[Step] = []
    for name in names:
        spec = declaration["stage_b"]["baselines"][name]
        for seed in seeds:
            out = baseline_reading(campaign, name, seed)
            steps.append(Step(
                f"baseline {name} seed {seed}: lockstep {PROTOCOL_NONE}",
                [py, str(RUN_TS), "manoeuvre_lockstep", "--executor", str(_seeded(spec["checkpoint"], airport=airport, seed=seed)),
                 "--protocol", PROTOCOL_NONE, "--cohort", str(cohort), "--write-records", "--device", device, "--out", str(out)],
                out / LOCKSTEP_JSON,
            ))
            failure = campaign / "failure_modes" / f"{name}_s{seed}_{PROTOCOL_NONE}"
            steps.append(Step(f"baseline {name} seed {seed}: failure modes",
                              [py, str(RUN_TS), "executor_failure_modes", "--lockstep", str(out), "--out", str(failure)],
                              failure / FAILURE_MODES_JSON))
    return steps


def group_steps(group: str, arms: dict[int, str], *, declaration: dict[str, Any], declaration_path: Path, campaign: Path,
                airport: str, device: str) -> list[Step]:
    """The steps of one group (module docstring); the prior steps carry the gate B1 verdict they need."""
    py = sys.executable
    stage_b = declaration["stage_b"]
    configuration = stage_b["configurations"][group]
    if configuration["baseline"] not in stage_b["baselines"]:
        raise ValueError(f"configuration {group} names baseline {configuration['baseline']!r}, "
                         f"which stage_b.baselines does not declare ({sorted(stage_b['baselines'])})")
    seed_line = REPO_ROOT / stage_b["seed_line_from"].format(airport=airport)
    cohort = REPO_ROOT / declaration["development_cohort"].format(airport=airport)
    vocabulary = REPO_ROOT / declaration_base(declaration)["instruction_vocabulary"]
    keys = [arms[seed] for seed in sorted(arms)]
    steps: list[Step] = [Step(
        f"{group}: train {', '.join(keys)}",
        [py, str(RUN_TS), "frame_ablation", "--arms", str(declaration_path), "--campaign", str(campaign), "--airport", airport,
         "--device", device, "--only", *keys],
        None,
    )]

    def checkpoint(arm: str) -> Path:
        return campaign / arm / "checkpoint.pt"

    def reading(arm: str, protocol: str) -> Path:
        return campaign / LOCKSTEP_DIR / arm / protocol

    for arm in keys:
        out = reading(arm, PROTOCOL_TRUTH_INSTRUCTION)
        steps.append(Step(f"{arm}: lockstep {PROTOCOL_TRUTH_INSTRUCTION}",
                          [py, str(RUN_TS), "manoeuvre_lockstep", "--executor", str(checkpoint(arm)), "--protocol", PROTOCOL_TRUTH_INSTRUCTION,
                           "--cohort", str(cohort), "--write-records", "--device", device, "--out", str(out)],
                          out / LOCKSTEP_JSON))
        failure = campaign / "failure_modes" / f"{arm}_{PROTOCOL_TRUTH_INSTRUCTION}"
        steps.append(Step(f"{arm}: failure modes {PROTOCOL_TRUTH_INSTRUCTION}",
                          [py, str(RUN_TS), "executor_failure_modes", "--lockstep", str(out), "--out", str(failure)],
                          failure / FAILURE_MODES_JSON))
    gate_out = campaign / "gate" / f"b1_{group}"
    b1 = Step(f"{group}: gate b1 ({PROTOCOL_TRUTH_INSTRUCTION} vs {PROTOCOL_NONE})",
              [py, str(RUN_TS), "executor_relative_gate",
               "--baseline", *[f"{seed}={baseline_reading(campaign, configuration['baseline'], seed)}" for seed in sorted(arms)],
               "--candidate", *[f"{seed}={reading(arms[seed], PROTOCOL_TRUTH_INSTRUCTION)}" for seed in sorted(arms)],
               "--seed-line-from", str(seed_line), "--out", str(gate_out)],
              gate_out / GATE_JSON)
    steps.append(b1)
    needs_b1 = (b1.artefact, "b1")
    prior_settings = stage_b["prior"]
    for seed in sorted(arms):
        arm = arms[seed]
        prior = campaign / "priors" / arm
        steps.append(Step(f"{arm}: instruction prior (seed {seed})",
                          [py, str(RUN_TS), "instruction_prior", "--vocabulary", str(vocabulary), "--executor", str(checkpoint(arm)),
                           "--cohort", str(cohort), "--seed", str(seed), "--device", device, "--out", str(prior),
                           *[flag for name in PRIOR_SETTINGS if name in prior_settings
                             for flag in (f"--{name.replace('_', '-')}", f"{prior_settings[name]}")]],
                          prior / PRIOR_FILE, gate=needs_b1))
    return steps


def seed_line_problem(path: Path) -> str | None:
    """Why `executor_relative_gate --seed-line-from` could not read this grid verdict, or None: missing,
    pending (no verdict), or a seed line without one of the gate's metrics at the quantile it reads."""
    if not path.is_file():
        return "missing"
    verdict = json.loads(path.read_text(encoding="utf-8"))["verdict"]
    if verdict is None:
        return "no verdict (pending readings): no seed line to read"
    line = verdict["seed_line"]
    absent = [m for m in RELATIVE_METRICS if f"p{GRID_SEED_LINE_QUANTILE}" not in line[m]]
    if absent:
        return f"the seed line lacks p{GRID_SEED_LINE_QUANTILE} of {', '.join(absent)}"
    return None


def gate_passed(verdict_path: Path, row: str) -> bool:
    return bool(json.loads(verdict_path.read_text(encoding="utf-8"))["verdicts"][row]["pass"])


def say(line: str) -> None:
    print(f"[{datetime.now():%F %T}] {line}", flush=True)


def plan_of(declaration: dict[str, Any], *, declaration_path: Path, campaign: Path, airport: str, device: str,
            groups: list[str] | None) -> dict[str, list[Step]]:
    """The whole queue as ``{cell: steps}``: the baselines first, then every selected group."""
    all_groups = groups_of(declaration)
    if groups:
        unknown = [g for g in groups if g not in all_groups]
        if unknown:
            raise ValueError(f"--groups names groups the declaration does not have: {', '.join(unknown)}")
        all_groups = {g: arms for g, arms in all_groups.items() if g in set(groups)}
    seeds = sorted({seed for arms in all_groups.values() for seed in arms})
    names = sorted({declaration["stage_b"]["configurations"][g]["baseline"] for g in all_groups})
    plan = {"baselines": baseline_steps(declaration=declaration, campaign=campaign, airport=airport, device=device, seeds=seeds, names=names)}
    for group, arms in all_groups.items():
        plan[group] = group_steps(group, arms, declaration=declaration, declaration_path=declaration_path, campaign=campaign,
                                  airport=airport, device=device)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--airport", required=True)
    parser.add_argument("--groups", nargs="+", default=None, help="a subset of the configurations, in the declaration's order (the baselines always run)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    declaration_path = args.arms if args.arms.is_absolute() else REPO_ROOT / args.arms
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    airport = args.airport.upper()
    try:
        plan = plan_of(declaration, declaration_path=declaration_path, campaign=campaign, airport=airport, device=args.device, groups=args.groups)
    except ValueError as exc:
        parser.error(str(exc))
    problem = seed_line_problem(REPO_ROOT / declaration["stage_b"]["seed_line_from"].format(airport=airport))
    if problem is not None:
        parser.error(f"stage_b.seed_line_from: {problem}")
    vocabulary = REPO_ROOT / declaration_base(declaration)["instruction_vocabulary"]
    if not vocabulary.is_file():
        parser.error(f"the vocabulary artefact {vocabulary} is not there: run instruction_vocabulary (B0′) first")
    if args.dry_run:
        for cell, steps in plan.items():
            print(f"CELL {cell}")
            for step in steps:
                done = step.artefact is not None and step.artefact.exists()
                gated = f" [needs gate {step.gate[1]} of {step.gate[0].parent.name}]" if step.gate else ""
                print(f"  {'done ' if done else 'todo '}{step.label}{gated}\n      {' '.join(step.command)}")
        return 0
    campaign.mkdir(parents=True, exist_ok=True)
    pid_file = campaign / PID_FILE
    previous = live_pid(pid_file)
    if previous is not None:
        parser.error(f"a queue is still running on this campaign (pid {previous}, {pid_file})")
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        return _run(plan, campaign)
    finally:
        pid_file.unlink()


def _run(plan: dict[str, list[Step]], campaign: Path) -> int:
    say(f"queue start: {len(plan)} cells, campaign {campaign}, pid {os.getpid()}")
    for cell, steps in plan.items():
        free = shutil.disk_usage(campaign).free
        say(f"CELL {cell} begin · free {free / 1024**3:.1f} GiB")
        if free < MINIMUM_FREE_BYTES:
            say(f"STOP: under {MINIMUM_FREE_BYTES / 1024**3:.0f} GiB free")
            return 3
        for step in steps:
            if step.artefact is not None and step.artefact.exists():
                say(f"skip {step.label} ({step.artefact.name} exists)")
                continue
            if step.gate is not None:
                verdict, row = step.gate
                if not verdict.is_file():
                    say(f"STOP: {step.label} needs gate {row} of {verdict.parent.name}, which has no verdict")
                    return 4
                if not gate_passed(verdict, row):
                    say(f"skip {step.label}: gate {row} of {verdict.parent.name} did not pass")
                    continue
            say(f"RUN {step.label}")
            completed = subprocess.run(step.command, cwd=REPO_ROOT)
            if completed.returncode != 0:
                say(f"STOP: {step.label} failed rc={completed.returncode}")
                return completed.returncode
        say(f"CELL {cell} complete")
    say("queue done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
