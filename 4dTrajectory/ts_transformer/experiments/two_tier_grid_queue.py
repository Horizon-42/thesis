"""Two-tier v3 stage A1's queue (§8): train the (L, Δ) grid one cell at a time and read each cell as it lands.

    python run_ts.py two_tier_grid_queue --arms 4dTrajectory/ts_transformer/docs/experiments/two_tier_v3_grid_arms.json \\
        --campaign 4dTrajectory/outputs/KRDU/experiments/two_tier_v3_grid_20260918 --airport KRDU \\
        [--readings L-1 12km 8km 6km] [--cells L30_D20 …] [--device auto] [--dry-run]

One cell = the declaration's arms of one (lookback, segment) (`executor_grid_gate.grid_cells`, named
`gates.cell_name`: its two seeds). Per cell, in the declaration's order (Δ ascending, then L
ascending): `frame_ablation --only <its arms>` (an arm whose ``history.json`` exists is skipped by
the runner itself), then for every arm and reading `manoeuvre_lockstep --protocol none` into
``<campaign>/lockstep/<arm>/<reading>/`` (``L-1`` = the first prediction at the fixed anchor; ``<X>km`` =
``--anchor-remaining-km X``; a reading whose artefact exists is skipped), then `executor_grid_gate`
into ``<campaign>/gate/after_<cell>/`` — the running table. No records are written here (40 arms ×
4 readings would be ~6 GB); the winner's readings are re-flown with ``--write-records`` after
M-A1. Every line is timestamped; ``CELL <cell> complete`` is what a watcher matches; the first
failed step stops the chain (D11: no automatic rerun) with its exit code — a step that died after
creating its output directory leaves one those runners refuse to overwrite: move it aside
(``<dir>.aborted-<UTC>``, never delete) and rerun the same command; the free disk is checked before
every cell (§8: ≥ 3 GB). The PID is written to ``<campaign>/two_tier_grid_queue.pid`` for the run's
duration and removed at exit; a second queue refuses to start while that PID is alive. Run it
detached from the runs worktree at the committed SHA.
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

from ts_transformer.experiments.executor_grid_gate import READING_L1, grid_cells, reading_path
from ts_transformer.experiments.support import REPO_ROOT, RUN_TS
from ts_transformer.manoeuvre.gates import cell_name

MINIMUM_FREE_BYTES = 3 * 1024**3
PID_FILE = "two_tier_grid_queue.pid"
DEFAULT_READINGS = (READING_L1, "12km", "8km", "6km")


def cells_of(declaration: dict[str, Any]) -> dict[str, list[str]]:
    """``{cell name: [arm keys, seed order]}`` in declaration order — the gate's own cells
    (`grid_cells`, read off each arm's config), so ``--cells`` names what the gate table prints."""
    return {cell_name(*cell): [by_seed[seed] for seed in sorted(by_seed)] for cell, by_seed in grid_cells(declaration).items()}


def live_pid(path: Path) -> int | None:
    """The PID a previous queue left in ``path``, if that process is still alive."""
    if not path.is_file():
        return None
    pid = int(path.read_text(encoding="utf-8").strip() or 0)
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return None
    except PermissionError:
        return pid
    return pid


def reading_flags(reading: str) -> list[str]:
    if reading == READING_L1:
        return []
    if reading.endswith("km"):
        return ["--anchor-remaining-km", reading[:-2]]
    if reading.startswith("row") and reading[3:].isdigit():
        return ["--first-prediction-row", reading[3:]]
    raise ValueError(f"a reading is {READING_L1!r}, '<X>km' or 'row<N>', got {reading!r}")


def cell_steps(cell: str, arms: list[str], *, declaration: Path, campaign: Path, airport: str, readings: list[str],
               device: str) -> list[tuple[str, list[str], Path | None]]:
    """``(label, command, artefact whose existence skips the step)`` for one cell."""
    py = sys.executable
    steps: list[tuple[str, list[str], Path | None]] = [(
        f"{cell}: train {', '.join(arms)}",
        [py, str(RUN_TS), "frame_ablation", "--arms", str(declaration), "--campaign", str(campaign), "--airport", airport,
         "--device", device, "--only", *arms],
        None,      # frame_ablation resumes by itself, arm by arm
    )]
    for arm in arms:
        for reading in readings:
            artefact = reading_path(campaign, arm, reading)
            steps.append((
                f"{arm}: lockstep none, {reading}",
                [py, str(RUN_TS), "manoeuvre_lockstep", "--executor", str(campaign / arm / "checkpoint.pt"), "--protocol", "none",
                 *reading_flags(reading), "--device", device, "--out", str(artefact.parent)],
                artefact,
            ))
    gate = campaign / "gate" / f"after_{cell}"
    steps.append((f"{cell}: grid gate", [py, str(RUN_TS), "executor_grid_gate", "--campaign", str(campaign), "--arms", str(declaration),
                                          "--reading", READING_L1, "--out", str(gate)], gate / "grid_gate.json"))
    return steps


def say(line: str) -> None:
    print(f"[{datetime.now():%F %T}] {line}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--airport", required=True)
    parser.add_argument("--readings", nargs="+", default=list(DEFAULT_READINGS))
    parser.add_argument("--cells", nargs="+", default=None, help="a subset of the cells, in the declaration's order")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    declaration = args.arms if args.arms.is_absolute() else REPO_ROOT / args.arms
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    cells = cells_of(json.loads(declaration.read_text(encoding="utf-8")))
    if args.cells:
        unknown = [cell for cell in args.cells if cell not in cells]
        if unknown:
            parser.error(f"--cells names cells the declaration does not have: {', '.join(unknown)}")
        cells = {cell: arms for cell, arms in cells.items() if cell in set(args.cells)}
    plan = {cell: cell_steps(cell, arms, declaration=declaration, campaign=campaign, airport=args.airport.upper(),
                             readings=args.readings, device=args.device) for cell, arms in cells.items()}
    if args.dry_run:
        for cell, steps in plan.items():
            print(f"CELL {cell}")
            for label, command, artefact in steps:
                done = artefact is not None and artefact.exists()
                print(f"  {'done ' if done else 'todo '}{label}\n      {' '.join(command)}")
        return 0
    campaign.mkdir(parents=True, exist_ok=True)
    pid_file = campaign / PID_FILE
    previous = live_pid(pid_file)
    if previous is not None:
        parser.error(f"a queue is still running on this campaign (pid {previous}, {pid_file})")
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        return _run(plan, campaign, args.readings)
    finally:
        pid_file.unlink()


def _run(plan: dict[str, list[tuple[str, list[str], Path | None]]], campaign: Path, readings: list[str]) -> int:
    say(f"queue start: {len(plan)} cells, readings {', '.join(readings)}, campaign {campaign}, pid {os.getpid()}")
    for cell, steps in plan.items():
        free = shutil.disk_usage(campaign).free
        say(f"CELL {cell} begin · free {free / 1024**3:.1f} GiB")
        if free < MINIMUM_FREE_BYTES:
            say(f"STOP: under {MINIMUM_FREE_BYTES / 1024**3:.0f} GiB free")
            return 3
        for label, command, artefact in steps:
            if artefact is not None and artefact.exists():
                say(f"skip {label} ({artefact.name} exists)")
                continue
            say(f"RUN {label}")
            completed = subprocess.run(command, cwd=REPO_ROOT)
            if completed.returncode != 0:
                say(f"STOP: {label} failed rc={completed.returncode}")
                return completed.returncode
        say(f"CELL {cell} complete")
    say("queue done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
