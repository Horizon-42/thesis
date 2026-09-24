"""Executor E7: measure the executor's parameters and write its spec (executor design §9–§10).

Two sources (the executor takes no information beyond the vocabulary, the user's rule of 2026-09-24; what the
vocabulary does not settle yet is still measured, the plan's second stage):

1. method A (`autopilot/derive.py`), from the vocabulary: τ_ψ = the heading lead, p = the bank limit over the lead.
   The turn rates and the bank limit are the vocabulary's, read at run time; a word takes effect when it is said;
2. data (`autopilot/measure.py`): a_dec, a_acc, a_unspec and the landing aim, on every labelled TRAIN flight, each
   re-read and checked against its stored sentence.

The design's fixed choices are module constants below. Writes ``spec.json`` + ``measurements.json`` into
``--dir`` (never over an existing file), from a clean tree only: the spec records the commit it was
measured at and the executor's source hash, and a replay refuses a spec measured by other code.

    python run_ts.py executor_spec \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> \\
        --dir 4dTrajectory/outputs/POOLED/executor/<name>
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from ts_transformer.autopilot import derive, measure
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import CLOCKS
from ts_transformer.autopilot.spec import executor_source_sha256, params_sha256, write_spec
from ts_transformer.instructions.artefact import (
    labeller_source_sha256, load_candidates, load_sentences, load_signals, load_spec, require_current_labeller,
)
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.repo_layout import REPO_ROOT, git_state

CHUNK = 1000
#: Δt, the control period (the user's choice, design §14 item 5).
CYCLE_S = 1.0
#: τ_γ: the smallest §5.1 admits (2 Δt) — the path angle follows its reference as fast as the loop allows.
PATH_TIME_CONSTANT_S = 2.0 * CYCLE_S
#: γ̇_max as a multiple of §5.1's lower bound (method A).
PATH_RATE_FACTOR = 2.0
#: A flight is given this many times its own sentence's time before it times out (§8.3).
TIMEOUT_FACTOR = 1.5


def data_parameters(directory: Path, spec: VocabularySpec, workers: int) -> tuple[dict[str, Any], dict[str, list[float]]]:
    signals = load_signals(directory, "train")
    sentences = load_sentences(directory, "train", spec)
    flights, stored = [], []
    for k, index in enumerate(sentences["signal_index"]):
        flights.append(signals[int(index)])
        stored.append((sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]],
                       int(sentences["runway_index"][k])))
    geometry_data = {code: geometry.to_dict() for code, geometry in load_candidates(directory).items()}
    pooled: dict[str, list[float]] = {name: [] for name in measure.MEASUREMENTS}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(measure.measure_chunk, flights[i: i + CHUNK], stored[i: i + CHUNK], spec.to_dict(),
                               geometry_data)
                   for i in range(0, len(flights), CHUNK)]
        for future in futures:
            for name, values in future.result().items():
                pooled[name] += values
    measured = measure.measured_values(pooled, spec)
    measured["labelled_train_flights"] = len(flights)
    return measured, pooled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact the executor flies")
    parser.add_argument("--dir", type=Path, required=True, help="the new executor spec directory")
    parser.add_argument("--word-clock", choices=CLOCKS, required=True,
                        help="the clock a replay says a truth sentence's words on (§11)")
    parser.add_argument("--workers", type=int, default=8)
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
    started = time.perf_counter()
    spec = load_spec(instructions)

    measured, _ = data_parameters(instructions, spec, args.workers)
    values = measured["values"]
    print(f"data parameters from {measured['labelled_train_flights']} train flights, "
          f"{time.perf_counter() - started:.0f}s: {values}", flush=True)

    tau, roll_rate = derive.heading_time_constant_s(spec, CYCLE_S), derive.roll_rate_deg_s(spec)
    params = ExecutorParams(cycle_s=CYCLE_S, heading_time_constant_s=tau, bank_rate_deg_s=roll_rate,
                            path_time_constant_s=PATH_TIME_CONSTANT_S, path_rate_factor=PATH_RATE_FACTOR,
                            decel_mps2=values["decel_mps2"], accel_mps2=values["accel_mps2"],
                            unspecified_decel_mps2=values["unspecified_decel_mps2"],
                            land_aim_height_m=values["land_aim_height_m"], land_window_low_m=values["land_window_low_m"],
                            land_window_high_m=values["land_window_high_m"], timeout_factor=TIMEOUT_FACTOR,
                            word_clock=args.word_clock)
    params.check(spec)
    print(f"method A (the vocabulary): τ_ψ {tau:g} s, p {roll_rate:g}°/s", flush=True)

    if executor_source_sha256() != executor or labeller_source_sha256() != labeller:
        raise SystemExit("the executor's or the labeller's code changed while the spec was measured; measure again")
    measurements = {
        "data": measured,
        "method_a": {
            "heading_time_constant_s": {"value": tau, "rule": "heading_lead_s, the time a heading word gives to "
                                                              "arrive — τ_ψ eases out the executor's own turns only; "
                                                              "heading words arrive a lead after they are heard"},
            "bank_rate_deg_s": {"value": roll_rate, "rule": "turn_bank_max_deg / heading_lead_s: the vocabulary's "
                                                            "bank limit reached within one lead"},
        },
        "from_the_vocabulary": {"turn_rate_max_deg_s": spec.turn_rate_max_deg_s,
                                "turn_rate_min_deg_s": spec.turn_rate_min_deg_s,
                                "turn_bank_max_deg": spec.turn_bank_max_deg, "heading_lead_s": spec.heading_lead_s,
                                "word_delay_s": 0.0},
        "fixed": {"cycle_s": CYCLE_S, "path_time_constant_s": PATH_TIME_CONSTANT_S, "path_rate_factor": PATH_RATE_FACTOR,
                  "timeout_factor": TIMEOUT_FACTOR},
        "elapsed_s": time.perf_counter() - started,
    }
    source = {"executor_source_sha256": executor, "labeller_source_sha256": labeller,
              "instructions": artefact_name, "git": git}
    write_spec(directory, params, spec.sha256, measurements, source)
    print(f"executor spec {params_sha256(params)[:12]} → {directory}")
    for name, value in asdict(params).items():
        print(f"  {name:26s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
