"""The labeller checked by what it reads (design §9.2 #2, D21; `instructions.conformance`): read an artefact's reference
flights again with the code on disk, require the same sentences and refusals, and write the passed record the labelling
and replay runners ask for (only from a clean checkout).

    python run_ts.py instruction_conformance --dir 4dTrajectory/outputs/POOLED/instruction_language/<artefact>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ts_transformer.instructions import conformance
from ts_transformer.repo_layout import REPO_ROOT, git_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--dir", type=Path, required=True, help="the instruction artefact (with its conformance/)")
    args = parser.parse_args(argv)
    directory = args.dir if args.dir.is_absolute() else REPO_ROOT / args.dir
    checked = conformance.check(directory, git=git_state())
    print(f"{checked.flights} reference flights read again by labeller code {checked.code_sha256[:12]}; "
          f"{len(checked.mismatches)} read otherwise", flush=True)
    for name, problems in list(checked.mismatches.items())[:20]:
        print(f"  {name}: " + "; ".join(problems[:6]))
    if not checked.passed:
        print("NOT conforming: no passed record written")
        return 1
    print(f"conforming → {conformance.write_passed(directory, checked, git=git_state())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
