"""Executor E7: write the executor's spec (vocabulary §5, §12.1 A6; executor design §9–§10).

The executor takes no information beyond the vocabulary (the user's rule, 2026-09-24) and the procedure standards:
the roll rate p is the standards' 5°/s (`ROLL_RATE_DEG_S`); the executor makes no turn of its own (D2, D27), so no
time constant of one; the turn rates, the bank limit, the speed word's rate (a_max, D43), "unspecified"'s pace and the
level bands are the vocabulary's,
read at run time; a
word takes effect when it is said. The judge's decision-altitude check has no parameter (vocabulary §5.8, D38: the
evaluation module's vertical bound and the FAS cone); each candidate's
published vertical path and decision altitude (the artefact's `candidates.json`, D61) are recorded in
``measurements.json`` as the spec is written. Nothing is measured from data. The design's fixed choices are module
constants below. Writes ``spec.json`` + ``measurements.json`` into
``--dir`` (never over an existing file), from a clean tree only: the spec records the commit it was measured at; then, in
the same run, the spec's reference tracks, flown by that code (`autopilot.conformance`) — every process that opens the
spec flies them again first and is refused by name off them (executor design §12.3, D73).

    python run_ts.py executor_spec \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> \\
        --dir 4dTrajectory/outputs/POOLED/executor/<name>
"""

from __future__ import annotations

import argparse
import os
import platform
from dataclasses import asdict
from pathlib import Path

from ts_transformer.autopilot import conformance
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.speed import speed_change_mps2
from ts_transformer.autopilot.spec import params_sha256, write_spec
from ts_transformer.instructions.artefact import load_candidates, load_spec
from ts_transformer.instructions.conformance import require_conforming_labeller
from ts_transformer.instructions.words import Words
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: Δt, the control period (the user's choice, design §14 item 5).
CYCLE_S = 1.0
#: τ_γ: the smallest §5.1 admits (2 Δt) — the path angle follows its reference as fast as the loop allows.
PATH_TIME_CONSTANT_S = 2.0 * CYCLE_S
#: γ̇_max as a multiple of §5.1's lower bound (method A).
PATH_RATE_FACTOR = 2.0
#: A flight is given this many times its own sentence's time before it times out (§8.3).
TIMEOUT_FACTOR = 1.5
#: p, deg/s: the roll rate the procedure standards allow (user 2026-09-27, the heading-lead ablation: it reads the same as
#: the bank limit over the lead, 8°/s, which is faster than every source). FAA Order 8260.3G (2024) App. E, Sec. 4, ¶6.a:
#: "Roll-in rates of up to five degrees per second and bank angles of 25 degrees may be used"; ICAO Doc 8168 Vol II
#: Part II, Sec 4, Ch 1, 1.3.9.1: "5 seconds for establishment of bank" (25°, reading: 5°/s). Quotes and files:
#: repo `docs/literature/roll_rate_and_turn_response/` §3.1–§3.2.
ROLL_RATE_DEG_S = 5.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact the executor flies")
    parser.add_argument("--dir", type=Path, required=True, help="the new executor spec directory")
    args = parser.parse_args(argv)
    # normalised (".." resolved, links kept): a worktree's data trees are links to the main tree's
    instructions = Path(os.path.normpath(args.instructions if args.instructions.is_absolute()
                                         else REPO_ROOT / args.instructions))
    directory = args.dir if args.dir.is_absolute() else REPO_ROOT / args.dir
    if directory.exists():
        parser.error(f"{directory} exists; an executor spec is never overwritten")
    # named as the repository names it (a worktree's data trees are links to the main tree's)
    if not instructions.is_relative_to(REPO_ROOT):
        parser.error(f"{instructions} is outside this repository ({REPO_ROOT}); name the artefact inside it")
    artefact_name = instructions.relative_to(REPO_ROOT).as_posix()
    git = git_state()
    if git["dirty"]:
        parser.error("the tree has uncommitted changes; an executor spec is measured at a commit")
    labeller = require_conforming_labeller(instructions)
    spec = load_spec(instructions)

    params = ExecutorParams(cycle_s=CYCLE_S, bank_rate_deg_s=ROLL_RATE_DEG_S,
                            path_time_constant_s=PATH_TIME_CONSTANT_S, path_rate_factor=PATH_RATE_FACTOR,
                            timeout_factor=TIMEOUT_FACTOR)
    params.check(spec, spec.step_s)
    print(f"the standards' roll rate p {ROLL_RATE_DEG_S:g}°/s", flush=True)

    measurements = {
        "from_the_standards": {
            "bank_rate_deg_s": {"value": ROLL_RATE_DEG_S,
                                "rule": "the procedure standards' roll rate: FAA Order 8260.3G App. E Sec. 4 ¶6.a (up to "
                                        "5°/s), ICAO Doc 8168 Vol II Part II Sec 4 Ch 1 1.3.9.1 (bank established in 5 s)"},
        },
        "from_the_vocabulary": {"turn_rate_max_deg_s": spec.turn_rate_max_deg_s,
                                "turn_bank_max_deg": spec.turn_bank_max_deg, "heading_lead_s": spec.heading_lead_s,
                                "speed_word_accel_mps2": spec.speed_accel_max_mps2,
                                "unspecified_pace_mps2": speed_change_mps2(spec),
                                "narrowest_level_band_m": float(Words(spec).altitude_tolerances.min()),
                                "lined_up_deg": spec.lined_up_deg, "landing_max_height_m": spec.landing_max_height_m},
        "from_the_runway": {
            "rule": "each candidate's published threshold crossing height, glidepath angle and decision altitude (the "
                    "judge's decision-altitude check), from the artefact's candidates.json (D61); recorded here as the "
                    "spec was written",
            "published": {code: {candidate.ident: asdict(candidate.vertical_path) for candidate in geometry.candidates}
                          for code, geometry in sorted(load_candidates(instructions).items())},
        },
        "fixed": {"cycle_s": CYCLE_S, "path_time_constant_s": PATH_TIME_CONSTANT_S, "path_rate_factor": PATH_RATE_FACTOR,
                  "timeout_factor": TIMEOUT_FACTOR},
    }
    source = {"python": platform.python_version(), "instructions": artefact_name, "git": git,
              "labeller_check": {"flights": labeller.flights, "read_otherwise": len(labeller.mismatches)}}
    write_spec(directory, params, spec.sha256, measurements, source)
    print(f"executor spec {params_sha256(params)[:12]} → {directory}")
    for name, value in asdict(params).items():
        print(f"  {name:26s} {value}")
    # its reference tracks, flown by the code that measured it in this run (D73), and this code flies them alike in
    # every way it flies
    print(f"reference tracks → {conformance.write_reference(directory, instructions)}", flush=True)
    checked = conformance.check(directory, instructions)
    if not checked.passed:
        raise SystemExit(f"the code that measured the spec does not fly its own reference alike in every way: "
                         f"{checked.summary()}")
    print(f"flown alike in every way: {checked.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
