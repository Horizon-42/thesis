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

ONE BEHAVIOUR (D108). The campaign runs on a clean checkout. At its start it runs the behaviour check of the prior's
code on fixed inputs (`prior_behaviour`, its own process: two steps of training on a fixed synthetic set and some rows
said with fixed numbers, on the CPU with one thread; the campaign's settings, `settings`, as the code on the disk sets
them) and records the answer (``campaign.json``), refused unless its settings are those this process plans with;
before each step it runs the check again, and an answer that differs, bit for bit, stops the campaign by name. The commit of each step is
recorded as information and never compared (results of different code are comparable once the code is shown to
behave the same on fixed inputs, never by an equal commit). A resume with other inputs is refused by name.

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
import fcntl
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

from ts_transformer.experiments.prior_free_generation import TEMPERATURE
from ts_transformer.experiments.prior_select import CONFIGURATIONS, SELECTION, arm_name
from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.speaker import MOST_GO_AROUNDS
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of ``campaign.json``. v2 (B10, D108): the answer of the behaviour check and each step's commit (no commit
#: of the campaign compared). v3 (B12, D108): the settings in the behaviour check's answer (the code on the disk's).
CAMPAIGN_SCHEMA = "ts-prior-campaign-v3"
#: The two seeds of §5 (the second: step 2's seed scale). The values are Claude's (§0.3).
SEEDS = (1337, 2024)
#: §5's configurations (`prior_select.CONFIGURATIONS`) as `prior_train`'s flags of the same names.
CONFIGURATION_FLAGS = {name: tuple(item for key, value in values.items()
                                   for item in (f"--{key.replace('_', '-')}", str(value)))
                       for name, values in CONFIGURATIONS.items()}
#: The free generation of a fold at its held-out airport (§5: 200 flights × 2) and of the base on the val days.
FREE_GENERATION = {"per_airport": 200, "samples": 2, "seed": 1337, "chunk": 400}
#: A smoke campaign's free generation: flights of each airport.
SMOKE_FLIGHTS = 5


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
                         "--seed", str(FREE_GENERATION["seed"]), "--chunk", str(FREE_GENERATION["chunk"]),
                         "--device", device, *smoke, "--out", str(generation)),
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
                      base / "run", base / "run" / "procedure_masks.json"))       # `prior_train`'s last file
    # the base's one validation readout reads the val days; a smoke never does: it reads the select days (D85)
    split = "select" if smoke else "val"
    steps.append(Step("base/validation", "prior_validation",
                      ("--prior", str(base / "run"), *common, "--split", split, "--device", device, *smoke,
                       "--out", str(base / "validation")),
                      base / "validation", base / "validation" / "readout.json"))
    steps.append(Step("base/free_generation", "prior_free_generation",
                      ("--prior", str(base / "run"), *common, "--split", split,
                       "--per-airport", str(flights), "--samples", str(FREE_GENERATION["samples"]),
                       "--seed", str(FREE_GENERATION["seed"]), "--chunk", str(FREE_GENERATION["chunk"]),
                       "--device", device, *smoke, "--out", str(base / "free_generation")),
                      base / "free_generation", base / "free_generation" / "readout.json"))
    return steps


def run_step(step: Step, log: Path, started: Callable[[int], None]) -> int:
    """``python run_ts.py <runner> <argv>`` from the repository, its output into ``log``; its exit code. ``started``
    is told the child's PID; the child is stopped if this process is stopped (it never runs on alone)."""
    with open(log, "w", encoding="utf-8") as stream:
        child = subprocess.Popen([sys.executable, str(REPO_ROOT / "run_ts.py"), step.runner, *step.argv], cwd=REPO_ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT)
        started(child.pid)
        try:
            return child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()


def alive_step(running: dict[str, Any] | None) -> bool:
    """Whether the child a campaign recorded as running is a live `run_ts.py` process (a killed campaign's child)."""
    if running is None:
        return False
    cmdline = Path(f"/proc/{running['pid']}/cmdline")
    return cmdline.exists() and b"run_ts.py" in cmdline.read_bytes()


def run_behaviour(instructions: str) -> dict[str, Any]:
    """The behaviour check of the prior's code on the disk now (`prior_behaviour`, its own process: not the code this
    process loaded): its answer; a check that fails stops the campaign with its error."""
    done = subprocess.run([sys.executable, str(REPO_ROOT / "run_ts.py"), "prior_behaviour", "--instructions",
                           instructions], cwd=REPO_ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"the behaviour check failed (exit {done.returncode}):\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def settings() -> dict[str, Any]:
    """What the campaign's code decides of its steps beside the prior's code: the seeds, the selection, each
    configuration's flags, the free generation of a fold and of a smoke, free generation's temperature and bound of
    go-arounds (D68). The behaviour check's process gives them as the code on the disk sets them (D108)."""
    return {"seeds": list(SEEDS), "selection": SELECTION,
            "configurations": {name: list(flags) for name, flags in CONFIGURATION_FLAGS.items()},
            "free_generation": FREE_GENERATION, "smoke_flights": SMOKE_FLIGHTS, "temperature": TEMPERATURE,
            "most_go_arounds": MOST_GO_AROUNDS}


def require_settings(campaign: Path, answer: dict[str, Any]) -> None:
    """The settings of the code on the disk (the behaviour check's ``answer``) are those this process plans with."""
    disk, here = answer["settings"], settings()
    if disk != here:
        differ = sorted(key for key in set(disk) | set(here)
                        if key not in disk or key not in here or disk[key] != here[key])
        raise SystemExit(f"{campaign}: the code on the disk sets {differ} otherwise than this campaign's process; stopped")


def open_campaign(campaign: Path, instructions: Path, executor: Path, interval_s: float, git: dict[str, Any],
                  smoke: int | None = None, behaviour: Callable[[str], dict[str, Any]] = run_behaviour
                  ) -> dict[str, Any]:
    """The campaign's record: written at its start with the answer of the behaviour check (``behaviour``), or read again
    on a resume and refused unless it is of the same inputs and the same smoke, and no step of it still runs."""
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
        require_settings(campaign, stored["behaviour"])
        if alive_step(stored["running"]):
            raise SystemExit(f"{campaign}: its step {stored['running']['step']} still runs as PID "
                             f"{stored['running']['pid']}; stop it first")
        return stored
    if campaign.exists() and any(campaign.iterdir()):
        raise SystemExit(f"{campaign} exists and is no campaign")
    campaign.mkdir(parents=True, exist_ok=True)
    answer = behaviour(str(instructions))
    require_settings(campaign, answer)
    record = {"schema": CAMPAIGN_SCHEMA, "written_utc": utc_now(), **record, "git": git,
              "airports": sorted(load_candidates(instructions)), "seeds": list(SEEDS), "behaviour": answer,
              "steps": [], "aborted": [], "running": None}
    write_json_atomic(path, record)
    return record


def run_campaign(campaign: Path, record: dict[str, Any], device: str,
                 runner: Callable[[Step, Path, Callable[[int], None]], int], log: Callable[[str], None],
                 tree: Callable[[], dict[str, Any]], behaviour: Callable[[str], dict[str, Any]] = run_behaviour) -> None:
    """Every step not yet done, in order, one at a time (module docstring); the plan read again after each step (a
    choice adds the steps after it). Before each step the tree (``tree``: `git_state`) must be clean unless a smoke,
    and the behaviour check (``behaviour``) must give the campaign's answer (D108); the step's commit is recorded.
    Stops at the first step that fails, naming its log."""
    (campaign / "logs").mkdir(exist_ok=True)

    def save() -> None:
        write_json_atomic(campaign / "campaign.json", record)

    while True:
        steps = plan(campaign, record, device)
        pending = [step for step in steps if not step.done.exists()]
        if not pending:
            break
        step = pending[0]
        now = tree()
        if now["dirty"] and record["smoke"] is None:
            raise SystemExit(f"the tree has uncommitted changes before {step.name}; a campaign runs on a clean checkout")
        answer = behaviour(record["instructions"])
        if answer != record["behaviour"]:
            stored = record["behaviour"]
            differ = sorted(key for key in set(answer) | set(stored)
                            if key not in answer or key not in stored or answer[key] != stored[key])
            raise SystemExit(f"before {step.name}: the prior's code behaves otherwise than at the campaign's start "
                             f"({differ} differ on the fixed inputs of `prior_behaviour`, D108); stopped")
        record["steps"].append({"step": step.name, "git": now, "utc": utc_now()})      # information, never compared
        if step.out.exists():                      # left by a crash or a kill: moved aside, recorded, run again
            aside = step.out.with_name(f"{step.out.name}.aborted-{utc_now().replace(':', '')}")
            step.out.rename(aside)
            record["aborted"].append({"step": step.name, "moved_to": str(aside), "utc": utc_now()})
            save()
            log(f"{step.name}: an unfinished output moved aside to {aside.name}")
        started = time.perf_counter()
        log(f"{step.name}: {step.runner} ({len(steps) - len(pending) + 1} of {len(steps)} planned so far)")
        logfile = campaign / "logs" / f"{step.name.replace('/', '__')}.log"

        def running(pid: int, name: str = step.name) -> None:
            record["running"] = {"step": name, "pid": pid, "utc": utc_now()}
            save()

        code = runner(step, logfile, running)
        record["running"] = None
        save()
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
    # one process for a campaign: a second launch on the directory is refused while this one holds the lock
    campaign.parent.mkdir(parents=True, exist_ok=True)
    lock = open(campaign.parent / f".{campaign.name}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"{campaign} is run by another process")
    # a stop (SIGTERM) unwinds through `run_step`, which stops the step's child
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(f"stopped (signal {signum})"))
    record = open_campaign(campaign, instructions, executor, args.row_interval_s, git, args.smoke)
    run_campaign(campaign, record, args.device, run_step, lambda line: print(f"{utc_now()} {line}", flush=True),
                 git_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
