"""B5: the cross-validation and the base as one campaign (prior design §5, §12 B5; D39–D41, D75) — the 31 training runs
from one commit on a clean checkout, one at a time on the GPU, each fold's readouts, the choices between the steps, and
the base with its one validation readout.

THE STEPS, in order (each a runner of the prior, run as its own process, its output in its own directory):

1. Variant `full`, configurations A, B, C, D (`CONFIGURATION_FLAGS`), the first seed, each on the 5 folds (leave one
   airport out): `prior_train --held-out <airport>`, then the fold's free generation at its held-out airport
   (`prior_free_generation --split select`, `FREE_GENERATION` flights × samples).
2. Configuration A, the second seed, on the 5 folds, each with its free generation.
   → `prior_select --step configuration`.
3. Variant `constants`, the chosen configuration, the first seed, on the 5 folds, each with its free generation.
   → `prior_select --step variant`.
4. The base: the chosen configuration and variant on all five airports (`prior_train`, no held-out airport), then its
   one validation readout: `prior_validation` and `prior_free_generation --split val`.

Every training run reads the sentences of the selection `landed` (D75). No criterion is applied to the readouts: the
user reads them (D7).

ONE COMMIT. The campaign records the commit it started from (``campaign.json``); it runs from a clean tree, and a
resume from another commit or with other inputs is refused by name (the design: one campaign from one commit).

SMOKE (``--smoke N``): every step at a small size, to check the chain on the formal artefact before the campaign —
each training run on a random sample of N sentences of each airport and split (`prior_train --sample`), each free
generation of `SMOKE_FLIGHTS` flights, every runner told it is a smoke; from a tree with changes too. Its numbers are
no result. The record says so, and `prior_select` reads a smoke campaign's folds only.

RESUMABLE. A step is done when its own last file is there (`Step.done`); a run is never repeated. A step's directory
without that file is a step a crash or a kill left (outline: E8): it is moved aside as ``<dir>.aborted-<UTC>``, the move
recorded in ``campaign.json``, and the step runs again. Each step's output goes to ``logs/<step>.log``.

    python run_ts.py prior_campaign --instructions 4dTrajectory/outputs/POOLED/instruction_language/<A34's artefact> \\
        --executor 4dTrajectory/outputs/POOLED/executor/<its spec> --row-interval-s 4 \\
        --out 4dTrajectory/outputs/POOLED/prior/<campaign id>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

from ts_transformer.experiments.prior_select import CONFIGURATIONS, arm_name
from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of ``campaign.json``.
CAMPAIGN_SCHEMA = "ts-prior-campaign-v1"
#: The two seeds of §5 (the second: step 2's seed scale). The values are Claude's (§0.3).
SEEDS = (1337, 2024)
#: §5's configurations, as `prior_train`'s flags that differ from configuration A (its defaults, D40; heads 32 wide).
CONFIGURATION_FLAGS = {
    "A": (),
    "B": ("--d-model", "128", "--heads", "4", "--feedforward", "512"),
    "C": ("--d-model", "256", "--heads", "8", "--feedforward", "1024"),
    "D": ("--dropout", "0.2", "--weight-decay", "0.05"),
}
#: The free generation of a fold at its held-out airport (§5: 200 flights × 2) and of the base on the val days.
FREE_GENERATION = {"per_airport": 200, "samples": 2, "seed": 1337}
#: A smoke campaign's free generation: flights of each airport.
SMOKE_FLIGHTS = 5
SELECTION = "landed"


class Step(NamedTuple):
    name: str                   # its directory's (or file's) name in the campaign, and its log's
    runner: str                 # `run_ts.py`'s runner
    argv: tuple[str, ...]
    out: Path                   # what it writes: a directory, or a file of the campaign
    done: Path                  # its last file: the step is done when this exists


def _choice(campaign: Path, step: str) -> str | None:
    path = campaign / f"choice_{step}.json"
    return json.loads(path.read_text(encoding="utf-8"))["chosen"] if path.exists() else None


def smoke_flags(record: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    """A smoke campaign's flags of `prior_train` and of the readouts, and its flights of a free generation."""
    if record["smoke"] is None:
        return (), (), FREE_GENERATION["per_airport"]
    return ("--sample", str(record["smoke"]["sample"])), ("--smoke",), SMOKE_FLIGHTS


def fold_steps(campaign: Path, record: dict[str, Any], configuration: str, variant: str, seed: int, device: str
               ) -> list[Step]:
    """One arm's 5 folds, each its training run then its free generation at its held-out airport."""
    arm = arm_name(configuration, variant, seed)
    common = ("--instructions", record["instructions"], "--executor", record["executor"])
    sample, smoke, flights = smoke_flags(record)
    out = []
    for airport in record["airports"]:
        run = campaign / arm / airport
        out.append(Step(f"{arm}/{airport}", "prior_train",
                        (*common, "--row-interval-s", f"{record['row_interval_s']:g}", "--variant", variant,
                         "--selection", SELECTION, "--held-out", airport, "--seed", str(seed), "--device", device,
                         *CONFIGURATION_FLAGS[configuration], *sample, "--out", str(run)),
                        run, run / "held_out.json"))
        generation = run / "free_generation"
        out.append(Step(f"{arm}/{airport}/free_generation", "prior_free_generation",
                        ("--prior", str(run), *common, "--split", "select", "--airports", airport,
                         "--per-airport", str(flights), "--samples", str(FREE_GENERATION["samples"]),
                         "--seed", str(FREE_GENERATION["seed"]), "--device", device, *smoke, "--out", str(generation)),
                        generation, generation / "readout.json"))
    return out


def plan(campaign: Path, record: dict[str, Any], device: str) -> list[Step]:
    """Every step the campaign holds so far (module docstring), in order: the steps after a choice only once it is
    made."""
    first, second = record["seeds"]
    steps = [step for c in CONFIGURATIONS for step in fold_steps(campaign, record, c, "full", first, device)]
    steps += fold_steps(campaign, record, "A", "full", second, device)
    choice = campaign / "choice_configuration.json"
    steps.append(Step("choice_configuration", "prior_select", ("--campaign", str(campaign), "--step", "configuration"),
                      choice, choice))
    configuration = _choice(campaign, "configuration")
    if configuration is None:
        return steps
    steps += fold_steps(campaign, record, configuration, "constants", first, device)
    choice = campaign / "choice_variant.json"
    steps.append(Step("choice_variant", "prior_select", ("--campaign", str(campaign), "--step", "variant"),
                      choice, choice))
    variant = _choice(campaign, "variant")
    if variant is None:
        return steps
    common = ("--instructions", record["instructions"], "--executor", record["executor"])
    sample, smoke, flights = smoke_flags(record)
    base = campaign / "base"
    steps.append(Step("base/run", "prior_train",
                      (*common, "--row-interval-s", f"{record['row_interval_s']:g}", "--variant", variant,
                       "--selection", SELECTION, "--seed", str(first), "--device", device,
                       *CONFIGURATION_FLAGS[configuration], *sample, "--out", str(base / "run")),
                      base / "run", base / "run" / "checkpoint.pt"))
    steps.append(Step("base/validation", "prior_validation",
                      ("--prior", str(base / "run"), *common, "--device", device, *smoke,
                       "--out", str(base / "validation")),
                      base / "validation", base / "validation" / "readout.json"))
    steps.append(Step("base/free_generation", "prior_free_generation",
                      ("--prior", str(base / "run"), *common, "--split", "val",
                       "--per-airport", str(flights), "--samples", str(FREE_GENERATION["samples"]),
                       "--seed", str(FREE_GENERATION["seed"]), "--device", device, *smoke,
                       "--out", str(base / "free_generation")),
                      base / "free_generation", base / "free_generation" / "readout.json"))
    return steps


def run_step(step: Step, log: Path) -> int:
    """``python run_ts.py <runner> <argv>`` from the repository, its output into ``log``; its exit code."""
    with open(log, "w", encoding="utf-8") as stream:
        return subprocess.run([sys.executable, str(REPO_ROOT / "run_ts.py"), step.runner, *step.argv], cwd=REPO_ROOT,
                              stdout=stream, stderr=subprocess.STDOUT, check=False).returncode


def open_campaign(campaign: Path, instructions: Path, executor: Path, interval_s: float, git: dict[str, Any],
                  smoke: int | None = None) -> dict[str, Any]:
    """The campaign's record: written at its start, or read again on a resume and refused unless it is of the same
    inputs, the same smoke and the same commit."""
    record = {"instructions": str(instructions), "executor": str(executor), "row_interval_s": interval_s,
              "smoke": None if smoke is None else {"sample": smoke, "flights": SMOKE_FLIGHTS}}
    path = campaign / "campaign.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["schema"] != CAMPAIGN_SCHEMA:
            raise SystemExit(f"{campaign} is a {stored['schema']} campaign, not {CAMPAIGN_SCHEMA}")
        changed = sorted(k for k, v in record.items() if stored[k] != v)
        if changed:
            raise SystemExit(f"{campaign}: started with other {changed}; a campaign is resumed with its own inputs")
        if stored["git"]["head"] != git["head"]:
            raise SystemExit(f"{campaign}: started at {stored['git']['head'][:12]}, this tree is at "
                             f"{git['head'][:12]}; a campaign runs from one commit")
        return stored
    if campaign.exists() and any(campaign.iterdir()):
        raise SystemExit(f"{campaign} exists and is no campaign")
    campaign.mkdir(parents=True, exist_ok=True)
    record = {"schema": CAMPAIGN_SCHEMA, "written_utc": utc_now(), **record, "git": git,
              "airports": sorted(load_candidates(instructions)), "seeds": list(SEEDS), "selection": SELECTION,
              "configurations": {name: list(flags) for name, flags in CONFIGURATION_FLAGS.items()},
              "free_generation": FREE_GENERATION, "aborted": []}
    write_json_atomic(path, record)
    return record


def run_campaign(campaign: Path, record: dict[str, Any], device: str, runner: Callable[[Step, Path], int],
                 log: Callable[[str], None]) -> None:
    """Every step not yet done, in order, one at a time (module docstring); the plan read again after each step (a
    choice adds the steps after it). Stops at the first step that fails, naming its log."""
    (campaign / "logs").mkdir(exist_ok=True)
    while True:
        steps = plan(campaign, record, device)
        pending = [step for step in steps if not step.done.exists()]
        if not pending:
            break
        step = pending[0]
        if step.out.exists():                      # left by a crash or a kill: moved aside, recorded, run again
            aside = step.out.with_name(f"{step.out.name}.aborted-{utc_now().replace(':', '')}")
            step.out.rename(aside)
            record["aborted"].append({"step": step.name, "moved_to": str(aside), "utc": utc_now()})
            write_json_atomic(campaign / "campaign.json", record)
            log(f"{step.name}: an unfinished output moved aside to {aside.name}")
        started = time.perf_counter()
        log(f"{step.name}: {step.runner} ({len(steps) - len(pending) + 1} of {len(steps)} planned so far)")
        logfile = campaign / "logs" / f"{step.name.replace('/', '__')}.log"
        code = runner(step, logfile)
        if code != 0 or not step.done.exists():
            raise SystemExit(f"{step.name} failed (exit {code}); its log: {logfile}")
        log(f"{step.name}: done in {time.perf_counter() - started:.0f} s")
    log(f"{campaign}: every step done")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--row-interval-s", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the campaign's directory (new, or one to resume)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", type=int, default=None, metavar="N",
                        help="SMOKE: every step at a small size, N sentences of each airport and split (no result)")
    args = parser.parse_args(argv)
    instructions, executor, campaign = (path if path.is_absolute() else REPO_ROOT / path
                                        for path in (args.instructions, args.executor, args.out))
    git = git_state()
    if git["dirty"] and args.smoke is None:
        parser.error("the tree has uncommitted changes; a campaign runs from one commit on a clean checkout")
    record = open_campaign(campaign, instructions, executor, args.row_interval_s, git, args.smoke)
    run_campaign(campaign, record, args.device, run_step, lambda line: print(f"{utc_now()} {line}", flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
