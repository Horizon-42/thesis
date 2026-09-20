"""Judge the pre-registered gates (plan §3.3) over written artefacts — two seeds each, never a
hand-typed number.

    python run_ts.py manoeuvre_gates --gate X --lockstep 1337=<dir>/manoeuvre_lockstep.json \\
        --lockstep 2024=<dir>/manoeuvre_lockstep.json --guidance-established 0.879 --out <dir>
    python run_ts.py manoeuvre_gates --gate E --lockstep 1337=… --lockstep 2024=… --out <dir>
    python run_ts.py manoeuvre_gates --gate P-open-loop --lockstep 1337=<A-truth …> --lockstep 2024=… --out <dir>
    python run_ts.py manoeuvre_gates --gate P-discrete-vs-continuous --lockstep 1337=<A discrete> --lockstep 2024=… \\
        --continuous 1337=<A continuous> --continuous 2024=… --out <dir>
    python run_ts.py manoeuvre_gates --gate S --candidate 60:1337=<A …> --candidate 60:2024=… --candidate 30:1337=… … --out <dir>
    python run_ts.py manoeuvre_gates --gate T --readout <dir>/manoeuvre_readout.json --out <dir>

Writes ``manoeuvre_gate_<gate>.json`` under ``--out`` (refused if the file exists) and prints it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre import gates

GATES = ("T", "X", "P-open-loop", "P-discrete-vs-continuous", "E", "S")


def _by_seed(parser: argparse.ArgumentParser, entries: list[str] | None, flag: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for entry in entries or ():
        seed, separator, raw = entry.partition("=")
        if not separator or not seed.strip().isdigit():
            parser.error(f"{flag} takes SEED=PATH, got {entry!r}")
        path = Path(raw.strip())
        out[int(seed)] = _load(path if path.is_absolute() else REPO_ROOT / path)
    return out


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_source"] = {"path": str(path), "sha256": file_sha256(path)}
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--gate", required=True, choices=GATES)
    parser.add_argument("--lockstep", action="append", help="SEED=manoeuvre_lockstep.json (the gate's protocol)")
    parser.add_argument("--continuous", action="append", help="SEED=manoeuvre_lockstep.json of the continuous prior (P)")
    parser.add_argument("--candidate", action="append", help="SEGMENT:SEED=manoeuvre_lockstep.json (S)")
    parser.add_argument("--readout", type=Path, help="manoeuvre_readout.json (T)")
    parser.add_argument("--guidance-established", type=float, help="the rule guidance's established share along the truth segments (X)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    target = out / f"manoeuvre_gate_{args.gate}.json"
    if target.exists():
        parser.error(f"{target} exists; a verdict is never overwritten")
    if args.gate == "T":
        if args.readout is None:
            parser.error("gate T reads --readout")
        verdict = gates.gate_t(_load(args.readout if args.readout.is_absolute() else REPO_ROOT / args.readout))
    elif args.gate == "X":
        if args.guidance_established is None:
            parser.error("gate X needs --guidance-established (measured by the guidance run on this cohort)")
        verdict = gates.gate_x(_by_seed(parser, args.lockstep, "--lockstep"), guidance_established=args.guidance_established)
    elif args.gate == "E":
        verdict = gates.gate_e(_by_seed(parser, args.lockstep, "--lockstep"))
    elif args.gate == "P-open-loop":
        verdict = gates.gate_p_open_loop(_by_seed(parser, args.lockstep, "--lockstep"))
    elif args.gate == "P-discrete-vs-continuous":
        verdict = gates.gate_p_discrete_vs_continuous(_by_seed(parser, args.lockstep, "--lockstep"), _by_seed(parser, args.continuous, "--continuous"))
    else:
        candidates: dict[float, dict[int, dict]] = {}
        for entry in args.candidate or ():
            head, separator, raw = entry.partition("=")
            segment, colon, seed = head.partition(":")
            if not separator or not colon or not seed.isdigit():
                parser.error(f"--candidate takes SEGMENT:SEED=PATH, got {entry!r}")
            path = Path(raw.strip())
            candidates.setdefault(float(segment), {})[int(seed)] = _load(path if path.is_absolute() else REPO_ROOT / path)
        verdict = gates.gate_s(candidates)
    payload = {"schema": "ts-manoeuvre-gate-v1", "written_utc": utc_now(), "gate": args.gate, "verdict": verdict,
               "inputs": {name: getattr(args, name) for name in ("lockstep", "continuous", "candidate", "guidance_established")
                          if getattr(args, name) is not None} | ({"readout": str(args.readout)} if args.readout else {})}
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, payload)
    print(json.dumps(verdict, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
