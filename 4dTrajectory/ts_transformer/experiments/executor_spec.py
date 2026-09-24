"""Executor E7: measure the executor's parameters and write its spec (executor design §9–§10).

Three sources, in order, each feeding the next:

1. data (`autopilot/measure.py`): r_turn, φ_cap, a_dec, a_acc, a_unspec and the landing aim, on every
   labelled TRAIN flight, each re-read and checked against its stored sentence;
2. method A (`autopilot/derive.py`): τ_ψ = the heading lead (the executor's own turns; heading words have their own
   law), p the least bank rate at which its own largest turn stays within the heading tolerance and a typical turn
   said word by word is flown inside every word's envelope;
3. method B (`autopilot/observe.py`): the vertical and speed word delays — a seeded train sample (`replay.draw`)
   flown with no delay, each flown track re-read by the labeller through the observation operator; a delay is the
   median of how much earlier the re-read places a word than the executor received it, floored at 0 (the executor
   cannot act on a word before it is said; `observe.delays_from_leads`). Heading words have none: each says where
   the track is a lead later.

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
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import derive, measure, observe, replay
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import CLOCKS, DELAY_GROUPS, Delays
from ts_transformer.autopilot.spec import executor_source_sha256, params_sha256, write_spec
from ts_transformer.instructions.artefact import (
    labeller_source_sha256, load_candidates, load_sentences, load_signals, load_spec, require_current_labeller,
)
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import COLUMNS, Words
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
#: Method B: a re-read word matches a received one of the same column and value within this many seconds.
LEAD_WINDOW_S = 30.0
PERCENTILES = (5, 25, 50, 75, 95)


def _percentiles(values: list[float]) -> dict[str, float]:
    return {f"p{q}": float(np.percentile(values, q)) for q in PERCENTILES} | {"n": len(values)}


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


def method_b(batch: replay.Batch, params: ExecutorParams, words: Words, device: torch.device) -> dict[str, Any]:
    spec = words.spec
    flown, verdicts = replay.fly_batch(batch, params, words, device=device)
    leads: dict[int, list[float]] = {column: [] for column in observe.MEASURED_COLUMNS}
    received: Counter = Counter()
    refused: Counter = Counter()
    for j, verdict in enumerate(verdicts):
        states = flown.states[j, : verdict.end_row + 1].cpu().numpy()
        try:
            pairs, counts = observe.flight_leads(states, flown.sentence_s[j].cpu().numpy(), params.cycle_s,
                                                 batch.series[j], batch.geometries[j], batch.readings[j], spec, words,
                                                 LEAD_WINDOW_S)
        except Refused as refusal:
            refused[refusal.reason] += 1
            continue
        for column, lead in pairs:
            leads[column].append(lead)
        received.update(counts)
    delays = observe.delays_from_leads(leads)
    by_group = {name: [lead for column in columns for lead in leads[column]] for name, columns in DELAY_GROUPS.items()}
    return {
        "delays": delays,
        "record": {
            "drawn": batch.drawn, "flown_with": {"delays": "0 s on every column"},
            "flights": replay.summary(verdicts),
            "flown_track_refused_by_the_labeller": dict(refused.most_common()),
            "words_received_and_matched": {COLUMNS[column]: {"received": received[column], "matched": len(leads[column])}
                                           for column in observe.MEASURED_COLUMNS},
            "lead_s_by_column": {COLUMNS[column]: _percentiles(values) for column, values in leads.items() if values},
            "lead_s_by_delay": {name: _percentiles(values) for name, values in by_group.items()},
            "rule": "delay = max(0, median lead of the group's columns); lead = received − re-read, seconds; "
                    f"matched within {LEAD_WINDOW_S:g} s on the same column and value; heading words are not measured "
                    "(each says the track a lead later: no delay)",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact the executor flies")
    parser.add_argument("--dir", type=Path, required=True, help="the new executor spec directory")
    parser.add_argument("--word-clock", choices=CLOCKS, required=True,
                        help="the clock a replay says a truth sentence's words on (§11); method B measures on it")
    parser.add_argument("--method-b-per-airport", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
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
    words = Words(spec)
    device = torch.device(args.device)

    measured, _ = data_parameters(instructions, spec, args.workers)
    values = measured["values"]
    print(f"data parameters from {measured['labelled_train_flights']} train flights, "
          f"{time.perf_counter() - started:.0f}s: {values}", flush=True)

    tau = derive.heading_time_constant_s(spec, CYCLE_S)
    provisional = ExecutorParams(cycle_s=CYCLE_S, turn_rate_deg_s=values["turn_rate_deg_s"],
                                 bank_cap_deg=values["bank_cap_deg"], heading_time_constant_s=tau,
                                 bank_rate_deg_s=derive.ROLL_RATE_STEP_DEG_S, path_time_constant_s=PATH_TIME_CONSTANT_S,
                                 path_rate_factor=PATH_RATE_FACTOR, decel_mps2=values["decel_mps2"],
                                 accel_mps2=values["accel_mps2"], unspecified_decel_mps2=values["unspecified_decel_mps2"],
                                 land_aim_height_m=values["land_aim_height_m"],
                                 land_window_low_m=values["land_window_low_m"],
                                 land_window_high_m=values["land_window_high_m"], delays=Delays(0.0, 0.0),
                                 timeout_factor=TIMEOUT_FACTOR, word_clock=args.word_clock)
    roll_rate, roll_checks = derive.roll_rate_deg_s(provisional, spec)
    undelayed = replace(provisional, bank_rate_deg_s=roll_rate)
    undelayed.check(spec)
    print(f"method A: τ_ψ {tau:g} s, p {roll_rate:g}°/s (by speed: a {derive.largest_own_turn_deg(spec):g}° own turn's "
          f"overshoot {roll_checks['overshoot_deg']}, a {derive.FOLLOW_TURN_DEG:g}° worded turn's excess past its "
          f"envelopes {roll_checks['follow_excess_deg']})", flush=True)

    batch = replay.draw(instructions, "train", spec, words, per_airport=args.method_b_per_airport, seed=args.seed)
    b = method_b(batch, undelayed, words, device)
    params = replace(undelayed, delays=b["delays"])
    params.check(spec)
    print(f"method B on {batch.drawn['flights']} flights: {b['delays']}, {time.perf_counter() - started:.0f}s", flush=True)

    if executor_source_sha256() != executor or labeller_source_sha256() != labeller:
        raise SystemExit("the executor's or the labeller's code changed while the spec was measured; measure again")
    measurements = {
        "data": measured,
        "method_a": {
            "heading_time_constant_s": {"value": tau, "rule": "chosen: heading_lead_s, the time a heading word gives "
                                                              "to arrive — τ_ψ eases out the executor's own turns only; "
                                                              "heading words arrive a lead after they are heard"},
            "bank_rate_deg_s": {"value": roll_rate, "by_speed_mps": roll_checks,
                                "rule": f"the least p on a {derive.ROLL_RATE_STEP_DEG_S:g}°/s grid at which, at every "
                                        f"speed, the executor's own {derive.largest_own_turn_deg(spec):g}° turn passes "
                                        "its target by at most heading_tolerance_deg and a "
                                        f"{derive.FOLLOW_TURN_DEG:g}° turn at r_turn (or the bank cap's rate), said "
                                        "word by word (read through a moving-average approximation of the velocity "
                                        "fit), is flown inside every word's envelope as the judge reads it"},
        },
        "method_b": b["record"],
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
