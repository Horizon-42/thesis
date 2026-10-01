"""The executor checked by what it flies (executor design §12.3, the user 2026-10-01; `autopilot.conformance`): fly a
spec's reference tracks again with the code on disk in every way the executor flies, compare them within the bounds,
and write the passed record `replay.open_executor` asks for — or, with ``--write-reference``, write the spec's reference
first (only from a clean checkout, with the code that measured the spec).

    python run_ts.py executor_conformance --executor 4dTrajectory/outputs/POOLED/executor/<spec> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> [--write-reference]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ts_transformer.autopilot import conformance
from ts_transformer.repo_layout import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact it flies")
    parser.add_argument("--write-reference", action="store_true",
                        help="fly and write the spec's reference first (a new conformance directory)")
    args = parser.parse_args(argv)
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    if args.write_reference:
        print(f"reference written → {conformance.write_reference(executor, instructions)}", flush=True)
    checked = conformance.check(executor, instructions)
    differences = checked.differences
    for mode, difference in differences.items():
        summary = difference.summary()
        print(f"{mode}: {summary['flights']} flights, states {summary['horizontal_m']:.3g} m / {summary['vertical_m']:.3g} m "
              f"apart, other floats {summary['other']:.3g}; {summary['mismatched_flights']} beyond the bounds", flush=True)
        for name, problems in difference.mismatches.items():
            print(f"  {name}: " + "; ".join(problems[:6]) + (" …" if len(problems) > 6 else ""))
    if not all(d.passed for d in differences.values()):
        print("NOT conforming: no passed record written")
        return 1
    print(f"conforming → {conformance.write_passed(executor, checked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
