"""Two-tier v3 stage B's queue (§8.2): fly the baselines, then train one configuration × vocabulary group at a time, read it under the truth-token protocol, judge gate B1, and past it train the prior and read the prior-token protocols.

    python run_ts.py two_tier_b_queue --arms 4dTrajectory/ts_transformer/docs/experiments/two_tier_v3_b_arms.json \\
        --campaign 4dTrajectory/outputs/KRDU/experiments/two_tier_v3_b_20260919 --airport KRDU \\
        [--groups S20_K16 …] [--device auto] [--dry-run]

Every input a stage B gate compares is produced HERE, under one code version: the queue never points at
another campaign's reading (a payload written by other code is refused by the gate by schema, never read
around). Step 0, ``baselines`` (`baseline_steps`): for every entry of the declaration's ``stage_b.baselines``
and every seed, `manoeuvre_lockstep --protocol none --cohort <the declaration's development cohort>
--write-records` from L-1 — B0's re-read of the stage A executor L60_D20 (D33), and the A3-a reading of
L60_D60 flown 20 s at a time (``execute_s``) for S60-h60 — into ``<campaign>/baseline/<name>_s<seed>/L-1/``,
then `executor_failure_modes` into ``<campaign>/failure_modes/<name>_s<seed>_none/``.

One group = the declaration's two seeds of one ``<configuration>_<tokenizer>`` (`groups_of`: the arm keys
are ``<configuration>_<tokenizer>_s<seed>``, both seeds required — every gate is judged on two). Per
group, in the declaration's order (§8.2: K16 first — S20, S60-h60, S60-held — then the command
vocabulary):

1. `frame_ablation --only <its two arms>` (an arm whose ``history.json`` exists is skipped by the runner itself);
2. per arm: `manoeuvre_codebook` → the declaration's ``stage_b.codebook_dir`` (``{arm}`` substituted);
   `manoeuvre_lockstep --protocol C --write-records` → ``<campaign>/lockstep/<arm>/C/`` (the truth-token
   reading, from L-1, D41; records for the failure-mode table); `executor_failure_modes` →
   ``<campaign>/failure_modes/<arm>_C/``; `manoeuvre_code_atlas` → ``<campaign>/atlas/<arm>/``
   (the end-point spread over codes: does the executor read z at all);
3. `executor_relative_gate` — the two C readings against the configuration's baseline
   (``stage_b.configurations[<configuration>].baseline`` names an entry of ``stage_b.baselines``), the seed
   line from ``stage_b.seed_line_from`` (D39: stage A's grid verdict — a number source, not a reading) →
   ``<campaign>/gate/b1_<group>/``;
4. ONLY when that verdict's ``b1`` row passes (read at run time, never planned as certain): per arm
   `manoeuvre_prior` (the arm's seed) → ``<campaign>/priors/<arm>/``; `manoeuvre_lockstep --protocol A
   --write-records` and ``--protocol A-truth`` → ``<campaign>/lockstep/<arm>/{A,A-truth}/``;
   `executor_failure_modes` on A → ``failure_modes/<arm>_A/``; then the relative gate on the two A readings
   → ``<campaign>/gate/b_<group>/`` (row B). A group that fails gate B1 logs ``GATE B1 FAIL`` and skips these.

A step whose artefact exists is skipped; the first failed step stops the chain with its exit code (D42: no
automatic rerun; a step that died after creating its output directory leaves one the runners refuse to
overwrite — move it aside as ``<dir>.aborted-<UTC>``, never delete, and rerun the same command). Every
line is timestamped; ``GROUP <group> complete`` is what a watcher matches. The free disk is checked before
every group (≥ 3 GB). The PID is written to ``<campaign>/two_tier_b_queue.pid`` for the run's duration and
removed at exit; a second queue refuses to start while that PID is alive. The baseline checkpoints, the
cohort file and the seed line's verdict are checked before anything runs. Run it detached from the runs
worktree at the committed SHA.
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

from ts_transformer.experiments.manoeuvre_prior import PRIOR_FILE
from ts_transformer.experiments.support import REPO_ROOT, RUN_TS, arm_config, declaration_base
from ts_transformer.experiments.two_tier_grid_queue import live_pid
from ts_transformer.manoeuvre.gates import GRID_SEED_LINE_QUANTILE, RELATIVE_METRICS
from ts_transformer.manoeuvre.tokenizer import CODEBOOK_MANIFEST

MINIMUM_FREE_BYTES = 3 * 1024**3
PID_FILE = "two_tier_b_queue.pid"
LOCKSTEP_DIR = "lockstep"
#: The artefact each runner writes last — MUST match `manoeuvre_lockstep`, `executor_failure_modes`,
#: `manoeuvre_code_atlas` and `executor_relative_gate` (literals there; the codebook's and the prior's
#: are imported). A wrong name is loud: the runner refuses an existing --out.
LOCKSTEP_JSON, FAILURE_MODES_JSON, ATLAS_JSON, GATE_JSON = "manoeuvre_lockstep.json", "failure_modes.json", "manoeuvre_code_atlas.json", "relative_gate.json"
BASELINE_DIR = "baseline"
BASELINE_GROUP = "baselines"
PROTOCOL_C, PROTOCOL_A, PROTOCOL_A_TRUTH, PROTOCOL_NONE = "C", "A", "A-truth", "none"


class Step:
    """One queue step: ``label``, ``command``, the ``artefact`` whose existence skips it (None: the
    runner resumes by itself), and ``gate`` — a ``(relative_gate.json, row)`` this step needs to have
    PASSED, read at run time (the prior steps past gate B1)."""

    def __init__(self, label: str, command: list[str], artefact: Path | None, gate: tuple[Path, str] | None = None):
        self.label, self.command, self.artefact, self.gate = label, command, artefact, gate


def groups_of(declaration: dict[str, Any]) -> dict[str, dict[int, str]]:
    """``{group: {seed: arm key}}`` in declaration order; a group is one configuration × vocabulary
    and needs exactly the two seeds every gate is judged on."""
    base = declaration_base(declaration)
    groups: dict[str, dict[int, str]] = {}
    for arm in declaration["arms"]:
        config, _settings = arm_config(base, arm.get("overrides", {}))      # mirrors frame_ablation's read of an arm
        group, _sep, seed_text = arm["key"].rpartition("_s")
        if not seed_text.isdigit() or int(seed_text) != config.seed:
            raise ValueError(f"arm {arm['key']} is not named <configuration>_<tokenizer>_s<seed> for its seed {config.seed}")
        if arm.get("configuration") != group.split("_")[0]:
            raise ValueError(f"arm {arm['key']} names configuration {arm.get('configuration')!r}, not its key's {group.split('_')[0]!r}")
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
    """Step 0: the baselines ``names`` (the ones the selected groups are judged against) × seed, flown
    by THIS code on the declaration's cohort — the no-token reading from L-1 (``execute_s`` > 0: only
    that much of each forecast, A3-a) with records, and its failure-mode table."""
    py = sys.executable
    cohort = REPO_ROOT / declaration["development_cohort"].format(airport=airport)
    steps: list[Step] = []
    for name in names:
        spec = declaration["stage_b"]["baselines"][name]
        for seed in seeds:
            out = baseline_reading(campaign, name, seed)
            steps.append(Step(
                f"baseline {name} seed {seed}: lockstep none",
                [py, str(RUN_TS), "manoeuvre_lockstep", "--executor", str(_seeded(spec["checkpoint"], airport=airport, seed=seed)),
                 "--protocol", PROTOCOL_NONE, "--cohort", str(cohort),
                 *(["--execute-s", f"{spec['execute_s']:g}"] if spec["execute_s"] else []),
                 "--write-records", "--device", device, "--out", str(out)],
                out / LOCKSTEP_JSON,
            ))
            failure = campaign / "failure_modes" / f"{name}_s{seed}_{PROTOCOL_NONE}"
            steps.append(Step(f"baseline {name} seed {seed}: failure modes",
                              [py, str(RUN_TS), "executor_failure_modes", "--lockstep", str(out), "--out", str(failure)],
                              failure / FAILURE_MODES_JSON))
    return steps


def group_steps(group: str, arms: dict[int, str], *, declaration: dict[str, Any], declaration_path: Path, campaign: Path,
                airport: str, device: str) -> list[Step]:
    """The steps of one group (see the module docstring); the prior steps carry the gate B1 verdict they need."""
    py = sys.executable
    stage_b = declaration["stage_b"]
    configuration = stage_b["configurations"][group.split("_")[0]]
    if configuration["baseline"] not in stage_b["baselines"]:
        raise ValueError(f"configuration {group.split('_')[0]} names baseline {configuration['baseline']!r}, "
                         f"which stage_b.baselines does not declare ({sorted(stage_b['baselines'])})")
    seed_line = REPO_ROOT / stage_b["seed_line_from"].format(airport=airport)
    keys = [arms[seed] for seed in sorted(arms)]
    steps: list[Step] = [Step(
        f"{group}: train {', '.join(keys)}",
        [py, str(RUN_TS), "frame_ablation", "--arms", str(declaration_path), "--campaign", str(campaign), "--airport", airport,
         "--device", device, "--only", *keys],
        None,
    )]

    def codebook(arm: str) -> Path:
        return REPO_ROOT / stage_b["codebook_dir"].format(airport=airport, arm=arm)

    def checkpoint(arm: str) -> Path:
        return campaign / arm / "checkpoint.pt"

    def reading(arm: str, protocol: str) -> Path:
        return campaign / LOCKSTEP_DIR / arm / protocol

    def lockstep(arm: str, protocol: str, *, prior: Path | None, records: bool) -> Step:
        command = [py, str(RUN_TS), "manoeuvre_lockstep", "--executor", str(checkpoint(arm)), "--codebook", str(codebook(arm)),
                   "--protocol", protocol, *(["--prior", str(prior)] if prior is not None else []),
                   *(["--write-records"] if records else []), "--device", device, "--out", str(reading(arm, protocol))]
        return Step(f"{arm}: lockstep {protocol}", command, reading(arm, protocol) / LOCKSTEP_JSON)

    def failure_modes(arm: str, protocol: str) -> Step:
        out = campaign / "failure_modes" / f"{arm}_{protocol}"
        return Step(f"{arm}: failure modes {protocol}",
                    [py, str(RUN_TS), "executor_failure_modes", "--lockstep", str(reading(arm, protocol)), "--out", str(out)],
                    out / FAILURE_MODES_JSON)

    def gate(row: str, protocol: str) -> Step:
        out = campaign / "gate" / f"{row}_{group}"
        command = [py, str(RUN_TS), "executor_relative_gate",
                   "--baseline", *[f"{seed}={baseline_reading(campaign, configuration['baseline'], seed)}" for seed in sorted(arms)],
                   "--candidate", *[f"{seed}={reading(arms[seed], protocol)}" for seed in sorted(arms)],
                   "--seed-line-from", str(seed_line), "--out", str(out)]
        return Step(f"{group}: gate {row} ({protocol} vs none)", command, out / GATE_JSON)

    for arm in keys:
        steps.append(Step(f"{arm}: codebook", [py, str(RUN_TS), "manoeuvre_codebook", "--checkpoint", str(checkpoint(arm)),
                                               "--out", str(codebook(arm))], codebook(arm) / CODEBOOK_MANIFEST))
        steps.append(lockstep(arm, PROTOCOL_C, prior=None, records=True))
        steps.append(failure_modes(arm, PROTOCOL_C))
        atlas = campaign / "atlas" / arm
        steps.append(Step(f"{arm}: code atlas", [py, str(RUN_TS), "manoeuvre_code_atlas", "--executor", str(checkpoint(arm)),
                                                 "--codebook", str(codebook(arm)), "--device", device, "--out", str(atlas)],
                          atlas / ATLAS_JSON))
    b1 = gate("b1", PROTOCOL_C)
    steps.append(b1)
    needs_b1 = (b1.artefact, "b1")
    for seed in sorted(arms):
        arm = arms[seed]
        prior = campaign / "priors" / arm
        steps.append(Step(f"{arm}: prior (seed {seed})",
                          [py, str(RUN_TS), "manoeuvre_prior", "--codebook", str(codebook(arm)), "--executor", str(checkpoint(arm)),
                           "--seed", str(seed), "--device", device, "--out", str(prior)],
                          prior / PRIOR_FILE, gate=needs_b1))
        for protocol, records in ((PROTOCOL_A, True), (PROTOCOL_A_TRUTH, False)):
            step = lockstep(arm, protocol, prior=prior / PRIOR_FILE, records=records)
            step.gate = needs_b1
            steps.append(step)
        step = failure_modes(arm, PROTOCOL_A)
        step.gate = needs_b1
        steps.append(step)
    step = gate("b", PROTOCOL_A)
    step.gate = needs_b1
    steps.append(step)
    return steps


def seed_line_problem(path: Path) -> str | None:
    """Why `executor_relative_gate --seed-line-from` could not read this grid verdict, or None: missing,
    pending (no verdict), or a seed line without one of the gate's metrics at the quantile it reads."""
    if not path.is_file():
        return "missing"
    verdict = json.loads(path.read_text(encoding="utf-8")).get("verdict")
    if verdict is None:
        return "no verdict (pending readings): no seed line to read"
    line = verdict.get("seed_line", {})
    absent = [m for m in RELATIVE_METRICS if f"p{GRID_SEED_LINE_QUANTILE}" not in line.get(m, {})]
    if absent:
        return f"the seed line lacks p{GRID_SEED_LINE_QUANTILE} of {', '.join(absent)}"
    return None


def gate_passed(verdict_path: Path, row: str) -> bool:
    return bool(json.loads(verdict_path.read_text(encoding="utf-8"))["verdicts"][row]["pass"])


def say(line: str) -> None:
    print(f"[{datetime.now():%F %T}] {line}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--airport", required=True)
    parser.add_argument("--groups", nargs="+", default=None, help="a subset of the arm groups, in the declaration's order (the baselines always run)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    declaration_path = args.arms if args.arms.is_absolute() else REPO_ROOT / args.arms
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    airport = args.airport.upper()
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    try:
        groups = groups_of(declaration)
    except ValueError as exc:
        parser.error(str(exc))
    if args.groups:
        unknown = [group for group in args.groups if group not in groups]
        if unknown:
            parser.error(f"--groups names groups the declaration does not have: {', '.join(unknown)}")
        groups = {group: arms for group, arms in groups.items() if group in set(args.groups)}
    seeds = sorted({seed for arms in groups.values() for seed in arms})
    # what step 0 and the gates need from outside this campaign: the baseline checkpoints, the cohort
    # and stage A's seed line (a verdict, not a reading) — checked now, not hours in
    stage_b = declaration["stage_b"]
    missing = [path for path in (
        REPO_ROOT / declaration["development_cohort"].format(airport=airport),
        *(_seeded(spec["checkpoint"], airport=airport, seed=seed) for spec in stage_b["baselines"].values() for seed in seeds),
    ) if not path.is_file()]
    if missing:
        parser.error(f"{len(missing)} file(s) the baselines need are missing (first: {missing[0]})")
    seed_line = REPO_ROOT / stage_b["seed_line_from"].format(airport=airport)
    problem = seed_line_problem(seed_line)
    if problem:
        parser.error(f"{seed_line}: {problem}")
    baseline_names = sorted({stage_b["configurations"][group.split("_")[0]]["baseline"] for group in groups})
    try:
        plan = {BASELINE_GROUP: baseline_steps(declaration=declaration, campaign=campaign, airport=airport, device=args.device, seeds=seeds,
                                               names=baseline_names)}
        plan.update({group: group_steps(group, arms, declaration=declaration, declaration_path=declaration_path, campaign=campaign,
                                        airport=airport, device=args.device) for group, arms in groups.items()})
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        for group, steps in plan.items():
            print(f"GROUP {group}")
            for step in steps:
                done = step.artefact is not None and step.artefact.exists()
                gated = "" if step.gate is None else f" [if gate {step.gate[1]} passes]"
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
    say(f"queue start: {len(plan)} groups, campaign {campaign}, pid {os.getpid()}")
    for group, steps in plan.items():
        free = shutil.disk_usage(campaign).free
        say(f"GROUP {group} begin · free {free / 1024**3:.1f} GiB")
        if free < MINIMUM_FREE_BYTES:
            say(f"STOP: under {MINIMUM_FREE_BYTES / 1024**3:.0f} GiB free")
            return 3
        failed_gate: str | None = None
        for step in steps:
            if step.gate is not None:
                verdict, row = step.gate
                if failed_gate == row:
                    continue
                if not gate_passed(verdict, row):
                    say(f"GATE {row.upper()} FAIL for {group} ({verdict}): the prior steps are skipped")
                    failed_gate = row
                    continue
            if step.artefact is not None and step.artefact.exists():
                say(f"skip {step.label} ({step.artefact.name} exists)")
                continue
            say(f"RUN {step.label}")
            completed = subprocess.run(step.command, cwd=REPO_ROOT)
            if completed.returncode != 0:
                say(f"STOP: {step.label} failed rc={completed.returncode}")
                return completed.returncode
            if step.gate is None and step.artefact is not None and step.artefact.name == GATE_JSON:
                say(f"GATE B1 {'PASS' if gate_passed(step.artefact, 'b1') else 'FAIL'} for {group}")
        say(f"GROUP {group} complete")
    say("queue done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
