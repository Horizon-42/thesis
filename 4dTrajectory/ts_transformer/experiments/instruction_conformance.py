"""The labeller checked by what it reads (vocabulary §7.2 #2, D21, D73; `instructions.conformance`): read an artefact's
reference flights again with the code on disk and require the same sentences and refusals. Information only — every process
that uses the labeller on the artefact runs the same check itself, before its work; nothing is written.

    python run_ts.py instruction_conformance --dir 4dTrajectory/outputs/POOLED/instruction_language/<artefact>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ts_transformer.instructions import conformance
from ts_transformer.repo_layout import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--dir", type=Path, required=True, help="the instruction artefact (with its conformance/)")
    args = parser.parse_args(argv)
    directory = args.dir if args.dir.is_absolute() else REPO_ROOT / args.dir
    checked = conformance.check(directory)
    print(f"{checked.flights} reference flights read again; {len(checked.mismatches)} read otherwise", flush=True)
    for name, problems in list(checked.mismatches.items())[:20]:
        print(f"  {name}: " + "; ".join(problems[:6]))
    print("conforming" if checked.passed else "NOT conforming")
    return 0 if checked.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
