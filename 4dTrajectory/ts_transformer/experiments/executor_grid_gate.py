"""Two-tier v3 A1 (§5.1, §3.3): the (lookback, segment) grid's table and the cell that wins, off the lockstep artefacts.

    python run_ts.py executor_grid_gate --campaign <dir> \\
        --arms 4dTrajectory/ts_transformer/docs/experiments/two_tier_v3_grid_arms.json --out <dir> [--reading L-1]

Reads ``<campaign>/lockstep/<arm>/<reading>/manoeuvre_lockstep.json`` for every training arm of
the declaration (`two_tier_grid_queue` writes them; ``L-1`` is §3.1's first reading, ``12km`` /
``8km`` / ``6km`` the second), groups the arms by cell (the arm's ``development_cohort``
directory names it) and seed, and writes ``grid_gate.json`` + ``grid_gate.txt`` under ``--out``
(refused if it exists): every cell's numbers per seed, and — once every cell has both seeds —
`manoeuvre/gates.gate_grid`'s verdict (the seed line read off the grid, the eligible cells, the
leaders, the winner). Cells still missing a reading are listed as pending and the verdict is
absent, so the queue can call this after every cell for a running table. The plan's verdict is
the ``L-1`` reading's; a bin reading's verdict is information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ts_transformer.experiments.support import REPO_ROOT, arm_config, declaration_base
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre.gates import GRID_SEED_LINE_QUANTILE, cell_name, cell_reading, gate_grid

GRID_GATE_SCHEMA = "ts-executor-grid-gate-v1"
#: Where the queue writes every arm's readings: <campaign>/LOCKSTEP_DIR/<arm>/<reading>/.
LOCKSTEP_DIR = "lockstep"
READING_L1 = "L-1"


def grid_cells(declaration: dict[str, Any]) -> dict[tuple[float, float], dict[int, str]]:
    """``{(lookback_s, segment_s): {seed: arm key}}`` for the declaration's training arms."""
    base = declaration_base(declaration)
    cells: dict[tuple[float, float], dict[int, str]] = {}
    for arm in declaration["arms"]:
        if "checkpoint" in arm:
            continue
        config, _settings = arm_config(base, arm.get("overrides", {}))
        cell = (config.seq_len * config.dt_s, config.control_horizon_s)
        if config.seed in cells.setdefault(cell, {}):
            raise ValueError(f"arms {cells[cell][config.seed]} and {arm['key']} are the same cell and seed")
        cells[cell][config.seed] = arm["key"]
    return cells


def reading_path(campaign: Path, arm: str, reading: str) -> Path:
    return campaign / LOCKSTEP_DIR / arm / reading / "manoeuvre_lockstep.json"


def render(payload: dict[str, Any]) -> str:
    lines = [f"executor grid gate · reading {payload['reading']} · {payload['campaign']}", ""]
    lines.append(f"{'cell':<10}{'seed':>6}{'n':>6}{'estab all':>11}{'vectored':>10}{'straight':>10}{'flyable':>9}{'vec ADE':>9}{'FDE p50':>9}")
    for cell, seeds in payload["table"].items():
        for seed, row in seeds.items():
            if row is None:
                lines.append(f"{cell:<10}{seed:>6}{'pending':>6}")
                continue
            lines.append(f"{cell:<10}{seed:>6}{row['n']:>6}{row['established_all']:>11.3f}{row['established_vectored']:>10.3f}"
                         f"{row['established_straight']:>10.3f}{row['fully_flyable']:>9.3f}{row['vectored_ade_mean_m']:>9.0f}{row['fde_p50_m']:>9.0f}")
    verdict = payload["verdict"]
    lines.append("")
    if verdict is None:
        lines.append(f"verdict: pending ({len(payload['pending'])} readings missing: {', '.join(payload['pending'][:6])}"
                     + (" …" if len(payload["pending"]) > 6 else "") + ")")
    else:
        line, q = verdict["seed_line"], f"p{GRID_SEED_LINE_QUANTILE}"
        lines.append(f"seed line ({verdict['seed_line_rule']}): established all {line['established_all'][q]:.3f} "
                     f"(p50 {line['established_all']['p50']:.3f}), vectored {line['established_vectored'][q]:.3f}, "
                     f"vectored ADE {line['vectored_ade_mean_m'][q]:.0f} m")
        lines.append(f"eligible (fully flyable ≥ {verdict['flyable_floor']:g} on both seeds): {len(verdict['eligible'])} cells"
                     + (f"; not: {', '.join(verdict['ineligible'])}" if verdict["ineligible"] else ""))
        lines.append(f"leaders on all flights: {', '.join(verdict['leaders']['established_all'])}")
        lines.append(f"leaders on the vectored group: {', '.join(verdict['leaders']['established_vectored'])}")
        lines.append(f"winners: {', '.join(verdict['winners'])} → selected {verdict['selected']['cell'] if verdict['selected'] else '—'}"
                     f" ({'decisive' if verdict['decisive'] else 'NOT decisive'})" + (f"; {verdict['note']}" if verdict["note"] else ""))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True, help="the grid's arm declaration")
    parser.add_argument("--reading", default=READING_L1, help="the reading directory under lockstep/<arm>/ (L-1, 12km, 8km, 6km)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    declaration = args.arms if args.arms.is_absolute() else REPO_ROOT / args.arms
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a verdict is never overwritten")
    cells = grid_cells(json.loads(declaration.read_text(encoding="utf-8")))
    payloads: dict[tuple[float, float], dict[int, dict[str, Any]]] = {}
    table: dict[str, dict[int, dict[str, Any] | None]] = {}
    sources: dict[str, dict[str, str]] = {}
    pending: list[str] = []
    for cell, by_seed in cells.items():
        table[cell_name(*cell)] = {}
        for seed, arm in by_seed.items():
            path = reading_path(campaign, arm, args.reading)
            if not path.is_file():
                pending.append(f"{arm}/{args.reading}")
                table[cell_name(*cell)][seed] = None
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            payloads.setdefault(cell, {})[seed] = payload
            sources[arm] = {"path": str(path), "sha256": file_sha256(path)}
            table[cell_name(*cell)][seed] = cell_reading(payload)
    verdict = gate_grid(payloads) if not pending else None
    result = {
        "schema": GRID_GATE_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign), "arms": str(declaration),
        "reading": args.reading, "sources": sources, "pending": pending, "table": table, "verdict": verdict,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "grid_gate.json", result)
    text = render(result)
    (out / "grid_gate.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
