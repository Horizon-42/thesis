"""Judge a candidate closed-loop reading against its no-token baseline on the same flights (two-tier v3 §3.3 rows A3 and B, gate B1; D45).

    python run_ts.py executor_relative_gate --baseline 1337=<dir> 2024=<dir> --candidate 1337=<dir> 2024=<dir> \\
        --seed-line-from <grid_gate.json> --out <dir> [--flyable-floor 0.95]
    python run_ts.py executor_relative_gate … --seed-line 0.078 0.184 150 --out <dir>

Each <dir> holds a ``manoeuvre_lockstep.json``. Per seed the two payloads are intersected on
their flight keys and the strata recomputed over the common flights (`stratum_table`), so both
sides are read over ONE cohort; the flights on each side and in common are stated, and the two
sides must start their closed loops by the same rule (`first_prediction.rule`). Both are
protocol-none readings (the intent-code protocols are archived, 2026-09-20) and both are payloads
of THIS code's schema (an older schema is refused by name — a stage's queue flies every input it
compares). The seed line is the grid gate's (its verdict's p75 lines, D39) or three numbers given
explicitly, and the output names its source. The verdicts are §3.3's rows A3 and B and the
upper-bound row B1 (both established shares not worse on both seeds, one metric beyond the line on
both seeds, fully flyable — row B without the ADE's not-worse clause; `gates.gate_relative`); which
row applies is the caller's question, every row is written.
Writes ``relative_gate.json`` / ``relative_gate.txt`` under ``--out`` (refused if it exists).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ts_transformer.experiments.manoeuvre_lockstep import LOCKSTEP_SCHEMA, stratum_table
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.manoeuvre.gates import GRID_FLYABLE_FLOOR, GRID_SEED_LINE_QUANTILE, RELATIVE_METRICS, cell_reading, gate_relative

RELATIVE_GATE_SCHEMA = "ts-executor-relative-gate-v1"


def _side(specs: list[str], parser: argparse.ArgumentParser) -> dict[int, tuple[Path, dict[str, Any]]]:
    """``{seed: (dir, payload)}`` from ``SEED=DIR`` pairs."""
    side: dict[int, tuple[Path, dict[str, Any]]] = {}
    for spec in specs:
        seed_text, _, dir_text = spec.partition("=")
        if not seed_text.isdigit() or not dir_text:
            parser.error(f"a side is given as SEED=DIR pairs, got {spec!r}")
        path = Path(dir_text)
        path = path if path.is_absolute() else REPO_ROOT / path
        payload = json.loads((path / "manoeuvre_lockstep.json").read_text(encoding="utf-8"))
        if payload.get("schema") != LOCKSTEP_SCHEMA:
            # the gate reads what this code writes, nothing older: a stage's queue flies every input it compares
            parser.error(f"{path}: lockstep schema {payload.get('schema')!r} is not {LOCKSTEP_SCHEMA!r}; fly the reading again with this code")
        side[int(seed_text)] = (path, payload)
    return side


def common_reading(payload: dict[str, Any], keys: list[str]) -> dict[str, float]:
    """`cell_reading` of the payload's rows restricted to ``keys`` (the strata recomputed)."""
    rows = {key: payload["rows"][key] for key in keys}
    return cell_reading({**payload, "flights": len(keys), "strata": stratum_table(rows)})


def render(result: dict[str, Any]) -> str:
    lines = [f"executor relative gate · candidate {result['candidate_protocol']} vs baseline none · seed line from {result['seed_line_source']}", ""]
    for seed in result["seeds"]:
        s = str(seed)
        b, c = result["baseline"][s], result["candidate"][s]
        lines.append(f"seed {seed}: flights baseline {result['flights'][s]['baseline']} · candidate {result['flights'][s]['candidate']} · "
                     f"common {result['flights'][s]['common']} · executed s {b['executed_s']:g} / {b['segment_s']:g} vs {c['executed_s']:g} / {c['segment_s']:g}")
        lines.append(f"  {'metric':<24}{'baseline':>10}{'candidate':>11}{'improve':>9}{'line':>8}  not worse  beyond")
        for m in RELATIVE_METRICS:
            lines.append(f"  {m:<24}{b[m]:>10.3f}{c[m]:>11.3f}{result['improvement'][m][s]:>9.3f}{result['seed_line'][m]:>8.3f}"
                         f"  {str(result['not_worse'][m][s]):<9}  {result['beyond'][m][s]}")
        lines.append(f"  {'fully_flyable':<24}{b['fully_flyable']:>10.3f}{c['fully_flyable']:>11.3f}{'':>9}{result['flyable_floor']:>8.2f}")
    v = result["verdicts"]
    lines.append("")
    lines.append(f"row A3: {'PASS' if v['a3']['pass'] else 'FAIL'} (on {', '.join(v['a3']['on']) or 'no primary'}) — {v['a3']['rule']}")
    lines.append(f"row B:  {'PASS' if v['b']['pass'] else 'FAIL'} (beyond on {', '.join(v['b']['beyond_on']) or 'nothing'}; not worse {v['b']['not_worse']}) — {v['b']['rule']}")
    lines.append(f"gate B1: {'PASS' if v['b1']['pass'] else 'FAIL'} (beyond on {', '.join(v['b1']['beyond_on']) or 'nothing'}) — {v['b1']['rule']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--baseline", nargs=2, required=True, metavar="SEED=DIR", help="the protocol-none reading, one directory per seed")
    parser.add_argument("--candidate", nargs=2, required=True, metavar="SEED=DIR", help="the reading judged against it, one directory per seed")
    line = parser.add_mutually_exclusive_group(required=True)
    line.add_argument("--seed-line-from", type=Path, default=None, help="a grid_gate.json whose verdict's p75 lines are the seed line")
    line.add_argument("--seed-line", type=float, nargs=3, default=None, metavar=RELATIVE_METRICS)
    parser.add_argument("--flyable-floor", type=float, default=GRID_FLYABLE_FLOOR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; the gate never overwrites a verdict")
    baseline, candidate = _side(args.baseline, parser), _side(args.candidate, parser)
    if sorted(baseline) != sorted(candidate) or len(baseline) != 2:
        parser.error(f"two seeds on both sides, the same two: baseline {sorted(baseline)}, candidate {sorted(candidate)}")
    for seed, (_path, payload) in baseline.items():
        if payload["protocol"] != "none":
            parser.error(f"the baseline is a protocol-none reading; seed {seed}'s is {payload['protocol']!r}")
    protocols = {payload["protocol"] for _path, payload in candidate.values()}
    if len(protocols) != 1:
        parser.error(f"the candidate's two seeds are one protocol; got {sorted(protocols)}")
    if args.seed_line_from is not None:
        source = args.seed_line_from if args.seed_line_from.is_absolute() else REPO_ROOT / args.seed_line_from
        verdict = json.loads(source.read_text(encoding="utf-8"))["verdict"]
        if verdict is None:
            parser.error(f"{source} has no verdict yet (pending readings): no seed line to read")
        seed_line = {m: verdict["seed_line"][m][f"p{GRID_SEED_LINE_QUANTILE}"] for m in RELATIVE_METRICS}
        seed_line_source = str(source)
    else:
        seed_line = dict(zip(RELATIVE_METRICS, args.seed_line, strict=True))
        seed_line_source = "given on the command line"
    rules = {payload["first_prediction"]["rule"] for _path, payload in (*baseline.values(), *candidate.values())}
    if len(rules) != 1:
        parser.error(f"the four readings start their closed loops by different rules ({sorted(rules)}); compare like with like")
    readings_b: dict[int, dict[str, float]] = {}
    readings_c: dict[int, dict[str, float]] = {}
    flights: dict[str, dict[str, int]] = {}
    for seed in sorted(baseline):
        (_pb, pb), (_pc, pc) = baseline[seed], candidate[seed]
        common = sorted(set(pb["rows"]) & set(pc["rows"]))
        if not common:
            parser.error(f"seed {seed}: the two readings share no flight")
        flights[str(seed)] = {"baseline": len(pb["rows"]), "candidate": len(pc["rows"]), "common": len(common)}
        readings_b[seed], readings_c[seed] = common_reading(pb, common), common_reading(pc, common)
    gate = gate_relative(readings_b, readings_c, seed_line=seed_line, flyable_floor=args.flyable_floor)

    def provenance(side: dict[int, tuple[Path, dict[str, Any]]]) -> dict[str, dict[str, str]]:
        return {str(seed): {"dir": str(path), "executor": payload["executor"], "executor_sha256": payload["executor_sha256"]}
                for seed, (path, payload) in side.items()}

    # JSON wants string keys: the per-seed maps (and the per-metric maps of per-seed values) are re-keyed
    result = {
        "schema": RELATIVE_GATE_SCHEMA, "written_utc": utc_now(),
        "baseline_sources": provenance(baseline), "candidate_sources": provenance(candidate),
        "candidate_protocol": next(iter(protocols)), "first_prediction": next(iter(rules)),
        "flights": flights, "seed_line_source": seed_line_source,
        "gate": gate["gate"], "seeds": gate["seeds"], "seed_line": gate["seed_line"], "flyable_floor": gate["flyable_floor"],
        "baseline": {str(s): r for s, r in gate["baseline"].items()}, "candidate": {str(s): r for s, r in gate["candidate"].items()},
        **{key: {m: {str(s): v for s, v in by_seed.items()} for m, by_seed in gate[key].items()} for key in ("improvement", "not_worse", "beyond")},
        "fully_flyable_ok": gate["fully_flyable_ok"], "verdicts": gate["verdicts"],
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "relative_gate.json", result)
    table = render(result)
    (out / "relative_gate.txt").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
