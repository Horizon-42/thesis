"""The executor checked by what it flies (executor design §12.3, the user 2026-10-01; D73; `autopilot.conformance`): fly
a spec's reference tracks again with the code on disk in every way the executor flies and compare them within the
bounds. Information only — every process that opens the spec runs the same check itself, before its work; nothing is
written (the reference is written with the spec, `executor_spec`).

    python run_ts.py executor_conformance --executor 4dTrajectory/outputs/POOLED/executor/<spec> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact>
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
    args = parser.parse_args(argv)
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    checked = conformance.check(executor, instructions)
    for mode, difference in checked.differences.items():
        summary = difference.summary()
        print(f"{mode}: {summary['flights']} flights, states {summary['horizontal_m']:.3g} m / {summary['vertical_m']:.3g} m "
              f"apart, other floats {summary['other']:.3g}; {summary['mismatched_flights']} beyond the bounds", flush=True)
        for name, problems in difference.mismatches.items():
            print(f"  {name}: " + "; ".join(problems[:6]) + (" …" if len(problems) > 6 else ""))
    print("conforming" if checked.passed else "NOT conforming")
    return 0 if checked.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
