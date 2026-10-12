"""Executor E7: write the executor's spec (vocabulary §5, §12.1 A6; executor design §9–§10).

The executor takes no information beyond the vocabulary (the user's rule, 2026-09-24), the dynamics and values decided
or assumed (executor design §9, draft 5): the roll rate p 15°/s (`ROLL_RATE_DEG_S`, D2), the actuators' first-order time
constants (assumed, not identified from data, §9.2); the words' leads are the vocabulary's, read at run time; the limits
are physical (the load band, the stall, the thrust box, the bank the wing can turn at). The judge's decision-altitude check has no parameter (vocabulary §5.8, D38: the
evaluation module's vertical bound and the FAS cone); each candidate's
published vertical path and decision altitude (the artefact's `candidates.json`, D61) are recorded in
``measurements.json`` as the spec is written. Nothing is measured from data. The design's fixed choices are module
constants below. Writes ``spec.json`` + ``measurements.json`` into
``--dir`` (never over an existing file), from a clean tree only: the spec records the commit it was measured at; then, in
the same run, the spec's reference tracks, flown by that code (`autopilot.conformance`) — every process that opens the
spec flies them again first and is refused by name off them (executor design §12.3, D73).

The start rule (D77: how the start state reads the observed rows before the row it starts at) is the user's choice from
A33's measurement, given with ``--start-rule``; only the rules of `params.FORMAL_START_RULES` are written (the data
plane's centred fit, which reads up to 7.5 s after the row, is A33's comparison only).

    python run_ts.py executor_spec \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> \\
        --dir 4dTrajectory/outputs/POOLED/executor/<name> --start-rule <rule>
"""

from __future__ import annotations

import argparse
import os
import platform
from dataclasses import asdict
from pathlib import Path

from ts_transformer.autopilot import conformance
from ts_transformer.autopilot.params import FORMAL_START_RULES, ExecutorParams
from ts_transformer.autopilot.spec import params_sha256, write_spec
from ts_transformer.instructions.artefact import load_candidates, load_spec
from ts_transformer.instructions.conformance import require_conforming_labeller
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: Δt, the control period (the user's choice, design §14 item 5).
CYCLE_S = 1.0
#: τ_x, τ_φ, τ_n, s: the actuators' first-order time constants — ASSUMED (executor design §9.2: the control path's
#: `TSConfig` defaults; the repository holds no measured one, ADS-B carries no attitude; the acceptance runs over the
#: literature's 1–3.5 s for the bank, V3).
THRUST_TIME_CONSTANT_S, BANK_TIME_CONSTANT_S, LOAD_TIME_CONSTANT_S = 1.5, 2.0, 0.8
#: A flight is given this many times its own sentence's time before it times out (§8.3).
TIMEOUT_FACTOR = 1.5
#: p, deg/s: the largest rate the ACTUAL bank is rolled at (executor design §4.4, D2: the user's decision of 2026-10-11, from
#: the roll-rate sweep on the lag plant — 5°/s on top of the bank's lag left flights > 300 m off their path; the
#: procedure standards' 5°/s, FAA Order 8260.3G App. E Sec. 4 ¶6.a, is a design value for protected areas, repo
#: `docs/literature/roll_rate_and_turn_response/` §3.1–§3.2).
ROLL_RATE_DEG_S = 15.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact the executor flies")
    parser.add_argument("--dir", type=Path, required=True, help="the new executor spec directory")
    parser.add_argument("--start-rule", required=True, choices=FORMAL_START_RULES,
                        help="how the start state reads the observed rows before it (D77; the user's choice, A33)")
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
                            thrust_time_constant_s=THRUST_TIME_CONSTANT_S, bank_time_constant_s=BANK_TIME_CONSTANT_S,
                            load_time_constant_s=LOAD_TIME_CONSTANT_S, timeout_factor=TIMEOUT_FACTOR,
                            start_rule=args.start_rule)
    params.check(spec, spec.step_s)
    print(f"roll rate p {ROLL_RATE_DEG_S:g}°/s; actuator time constants (assumed) {params.time_constants_s} s",
          flush=True)

    measurements = {
        "decided": {
            "bank_rate_deg_s": {"value": ROLL_RATE_DEG_S,
                                "rule": "the user's decision D2 (executor design §4.4, 2026-10-11): the largest rate the "
                                        "actual bank is rolled at, from the roll-rate sweep on the lag plant"},
        },
        "assumed": {
            "time_constants_s": {"value": list(params.time_constants_s),
                                 "rule": "thrust, bank, load factor: the control path's TSConfig defaults, not "
                                         "identified from data (executor design §9.2); accepted over the literature's "
                                         "1–3.5 s for the bank (V3)"},
        },
        "from_the_vocabulary": {"heading_lead_s": spec.heading_lead_s, "angle_lead_s": spec.angle_lead_s,
                                "speed_lead_s": spec.speed_lead_s, "lined_up_deg": spec.lined_up_deg},
        "from_the_runway": {
            "rule": "each candidate's published threshold crossing height, glidepath angle and decision altitude (the "
                    "judge's decision-altitude check), from the artefact's candidates.json (D61); recorded here as the "
                    "spec was written",
            "published": {code: {candidate.ident: asdict(candidate.vertical_path) for candidate in geometry.candidates}
                          for code, geometry in sorted(load_candidates(instructions).items())},
        },
        "fixed": {"cycle_s": CYCLE_S, "timeout_factor": TIMEOUT_FACTOR},
        "chosen": {"start_rule": {"value": args.start_rule,
                                  "rule": "the user's choice from the measurement of A33 (D77): how the start state "
                                          "takes its velocity from the observed rows at or before its row"}},
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
