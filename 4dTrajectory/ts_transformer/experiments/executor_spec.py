"""Executor E7: write the executor's spec (executor design §9–§10).

The executor takes no information beyond the vocabulary (the user's rule, 2026-09-24) and the procedure standards:
method A (`autopilot/derive.py`) sets τ_ψ = the heading lead; the roll rate p is the standards' 5°/s (`ROLL_RATE_DEG_S`);
the turn rates, the bank limit, the speed changes' pace and the altitude tolerance are the vocabulary's, read at run time; a word takes effect
when it is said; the landing crosses the pointed runway at its published threshold crossing height and never
descends under its published glidepath's lower edge (read at replay, `runway_data.published_vertical_paths`;
every candidate's is recorded in ``measurements.json`` as the spec is written). Nothing is measured from data (the measurements that set these before are
archived: `archive/executor_vocabulary_only_2026_09/`). The design's fixed choices are module constants below. Writes ``spec.json`` + ``measurements.json`` into
``--dir`` (never over an existing file), from a clean tree only: the spec records the commit it was
measured at and the executor's source hash, and a replay refuses a spec written by other code.

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

from ts_transformer.autopilot import derive
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.runway_data import published_vertical_paths
from ts_transformer.autopilot.sentence import CLOCKS
from ts_transformer.autopilot.speed import speed_change_mps2
from ts_transformer.autopilot.spec import executor_source_sha256, params_sha256, write_spec
from ts_transformer.instructions.artefact import (
    labeller_source_sha256, load_candidates, load_spec, require_current_labeller,
)
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
    parser.add_argument("--word-clock", choices=CLOCKS, required=True,
                        help="the clock a replay says a truth sentence's words on (§11)")
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
    require_current_labeller(instructions)
    executor, labeller = executor_source_sha256(), labeller_source_sha256()
    spec = load_spec(instructions)

    tau, roll_rate = derive.heading_time_constant_s(spec, CYCLE_S), ROLL_RATE_DEG_S
    floor = derive.stopping_roll_rate_deg_s(spec, tau)
    if roll_rate < floor:
        parser.error(f"the standards' roll rate {roll_rate:g}°/s is under {floor:.2f}°/s: with this vocabulary's bank limit "
                     "and lead the executor's own turns would outrun their stopping rate (`derive.stopping_roll_rate_deg_s`)")
    params = ExecutorParams(cycle_s=CYCLE_S, heading_time_constant_s=tau, bank_rate_deg_s=roll_rate,
                            path_time_constant_s=PATH_TIME_CONSTANT_S, path_rate_factor=PATH_RATE_FACTOR,
                            timeout_factor=TIMEOUT_FACTOR, word_clock=args.word_clock)
    params.check(spec)
    print(f"method A (the vocabulary): τ_ψ {tau:g} s; the standards' roll rate p {roll_rate:g}°/s", flush=True)

    if executor_source_sha256() != executor or labeller_source_sha256() != labeller:
        raise SystemExit("the executor's or the labeller's code changed while the spec was measured; measure again")
    measurements = {
        "method_a": {
            "heading_time_constant_s": {"value": tau, "rule": "heading_lead_s, the time a heading word gives to "
                                                              "arrive — τ_ψ eases out the executor's own turns only; "
                                                              "heading words arrive a lead after they are heard"},
        },
        "from_the_standards": {
            "bank_rate_deg_s": {"value": roll_rate, "rule": "the procedure standards' roll rate: FAA Order 8260.3G App. E "
                                                            "Sec. 4 ¶6.a (up to 5°/s), ICAO Doc 8168 Vol II Part II Sec 4 "
                                                            "Ch 1 1.3.9.1 (bank established in 5 s)",
                                "floor_deg_s": floor, "floor_rule": "tan(turn_bank_max_deg) / (2 τ_ψ): the executor's own "
                                                                    "turns stop within the bank limit"},
        },
        "from_the_vocabulary": {"turn_rate_max_deg_s": spec.turn_rate_max_deg_s,
                                "turn_rate_min_deg_s": spec.turn_rate_min_deg_s,
                                "turn_bank_max_deg": spec.turn_bank_max_deg, "heading_lead_s": spec.heading_lead_s,
                                "speed_change_mps2": speed_change_mps2(spec),
                                "altitude_tolerance_m": spec.altitude_tolerance_m,
                                "landing_max_height_m": spec.landing_max_height_m},
        "from_the_runway": {
            "rule": "each candidate's published threshold crossing height (the landing's crossing point) and glidepath "
                    "angle (its lower edge not descended under before the threshold), read at replay "
                    "(runway_data.published_vertical_paths); recorded here as the spec was written",
            "published": {code: {candidate.ident: asdict(path)
                                 for candidate, path in zip(geometry.candidates, published_vertical_paths(geometry))}
                          for code, geometry in sorted(load_candidates(instructions).items())},
        },
        "fixed": {"cycle_s": CYCLE_S, "path_time_constant_s": PATH_TIME_CONSTANT_S, "path_rate_factor": PATH_RATE_FACTOR,
                  "timeout_factor": TIMEOUT_FACTOR},
    }
    source = {"executor_source_sha256": executor, "python": platform.python_version(), "labeller_source_sha256": labeller,
              "instructions": artefact_name, "git": git}
    write_spec(directory, params, spec.sha256, measurements, source)
    print(f"executor spec {params_sha256(params)[:12]} → {directory}")
    for name, value in asdict(params).items():
        print(f"  {name:26s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
