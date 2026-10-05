"""Vocabulary A33 (D77): the start rules measured — for the user's choice (D55). NO CRITERION IS READ (D7).

One readout reads one instruction artefact, built as the formal one, with its executor spec: every labelled TRAIN flight
(on its own dynamics or a stand-in's, `replay.draw_readings`; train only, D85) read in closed loop at one row interval
(`instruction_closed_loop.read_interval`, in memory: nothing is written into the artefact) once for each start rule of
`START_RULES` — the spec's executor parameters with that rule in place of its own (``centred-fit-15s``, the data plane's
centred fit, is no rule of a formal spec: it is read here for the comparison only).

For each rule: the closed-loop reading's numbers (`instruction_closed_loop.summarise`: the readings of D34 — 1 the
correction words, 2 the errors left where no correction is made, 3 the rows outside the tolerances — and 4 the outcomes,
the judge's of what the reading flew, D74), the outcomes by airport, and the start track at the first predicted step
(`turns`): on the flights that turn there — more than `TURN_DEG` between the observed direction of the `LOOK_S` before
it and that of the `LOOK_S` after it — how far the rule's start track (`flights.start_velocity`, the executor's own) is
from the observed direction of the `LOOK_S` after it. That direction is a readout of the future, never an input; each
direction is the displacement between two 2 s rows in metres on the ground (`flights.ground_scale`). Every flight of the
batch is counted whatever its closed-loop result (the batch is the same for every rule); one whose labelled rows (they
end before the landing) end within `LOOK_S` of its first predicted step is counted apart.

``--workers N --parts P``: train in P parts — consecutive blocks of the one seeded permutation it is drawn in
(`replay.part_of`) — each drawn once and read at every rule in its own process, up to N at once
(`instruction_closed_loop.POOL_OPTIONS`); the parts' numbers are put together in their order
(`instruction_closed_loop.merge_tallies`), so the readout is the same whatever N and P.

    python run_ts.py start_rules --instructions <scratch artefact> --executor <its executor spec> --row-interval-s 4 \\
        --workers 4 --parts 8 --out <new dir>
"""

from __future__ import annotations

import argparse
import resource
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.flights import ground_scale, start_velocity
from ts_transformer.autopilot.params import START_RULES
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.experiments import instruction_closed_loop
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

READOUT_SCHEMA = "ts-start-rules-readout-v1"
SPLIT = "train"
SEED = instruction_closed_loop.SEED
#: A flight turns at its first predicted step when the observed directions of the `LOOK_S` before it and after it differ
#: by more than `TURN_DEG` (vocabulary §12.1 A33).
LOOK_S = 8.0
TURN_DEG = 5.0
PERCENTILES = (50, 75, 90, 95, 99)


def directions_deg(signals: Any, first: np.ndarray, last: np.ndarray, geometry: Any) -> np.ndarray:
    """The compass direction (°) of the observed displacement from each row ``first`` to the row ``last``, in metres on
    the ground at ``first``."""
    to_east, to_north = ground_scale(signals, first, geometry)
    east = (signals.e_m[last] - signals.e_m[first]) * to_east
    north = (signals.n_m[last] - signals.n_m[first]) * to_north
    return np.degrees(np.arctan2(east, north))


def wrapped_deg(difference: np.ndarray) -> np.ndarray:
    """|difference| of two compass directions, in [0°, 180°]."""
    return np.abs((np.asarray(difference) + 180.0) % 360.0 - 180.0)


def turns(batch: replay.Batch, rule: str, step_s: float) -> dict[str, Any]:
    """Each flight of ``batch`` at its first predicted step (`closed_loop.first_predicted_rows`, where the closed loop
    starts the executor; module docstring): the turn there (°) and the start track's distance (°) from the observed
    direction of the `LOOK_S` after it, by the start rule ``rule`` — both NaN when its labelled rows (which end before
    the landing) end within `LOOK_S`."""
    look = int(round(LOOK_S / step_s))
    turn, after_error, airports = [], [], []
    for signals, reading, row, geometry in zip(batch.observed, batch.readings,
                                               closed_loop.first_predicted_rows(batch, step_s), batch.geometries):
        airports.append(signals.airport)
        if row + look >= len(reading.words):          # past the labelled rows: the landing cut (`closed_loop.read`)
            turn.append(np.nan)
            after_error.append(np.nan)
            continue
        before, after = directions_deg(signals, np.array([row - look, row]), np.array([row, row + look]), geometry)
        velocity = start_velocity(signals, [row], rule, geometry)[0]
        track = np.degrees(np.arctan2(velocity[0], velocity[1]))
        turn.append(float(wrapped_deg(after - before)))
        after_error.append(float(wrapped_deg(track - after)))
    return {"turn_deg": np.array(turn), "start_from_after_deg": np.array(after_error), "airports": airports}


def turn_summary(rows: dict[str, Any]) -> dict[str, Any]:
    """The start track against the observed direction after the first predicted step, on the turning flights (pooled
    and by airport), with the flights counted."""
    turn, error = rows["turn_deg"], rows["start_from_after_deg"]
    airports = np.array(rows["airports"])
    known = ~np.isnan(turn)
    turning = known & (turn > TURN_DEG)

    def percentiles(values: np.ndarray) -> dict[str, float] | None:
        if not len(values):
            return None
        return {f"p{q}": float(np.percentile(values, q)) for q in PERCENTILES} | {"mean": float(values.mean()),
                                                                                "n": len(values)}

    return {"flights": len(turn), "labelled_rows_end_within_look": int((~known).sum()), "turning": int(turning.sum()),
            "turning_share": float(turning.sum() / known.sum()) if known.any() else None,
            "start_from_after_deg": percentiles(error[turning]),
            "by_airport": {airport: percentiles(error[turning & (airports == airport)])
                           for airport in sorted(set(airports.tolist()))},
            "all_flights_start_from_after_deg": percentiles(error[known])}


def read_part(instructions: Path, executor: Path, part: tuple[int, int], interval_s: float, rules: tuple[str, ...], *,
              chunk: int, device: str) -> tuple[int, tuple[dict[str, Any], Counter], dict[str, Any], float]:
    """Part ``part`` = ``(k, n)`` of the train split: drawn once and read in closed loop at ``interval_s`` by each rule
    of ``rules``. Returns ``k``, the draw's description and exclusions, each rule's tally, outcomes by airport and turn
    rows, and the process's peak memory (GB)."""
    params, _, words = replay.open_executor(executor, instructions)
    drawn, readings = replay.draw_readings(instructions, SPLIT, words.spec, words, per_airport=0, seed=SEED,
                                           groups=(replay.OWN, replay.STAND_IN), part=part)
    by_rule = {}
    for rule in rules:
        batch, results, tally = instruction_closed_loop.read_interval(
            drawn, readings, interval_s, replace(params, start_rule=rule), words, chunk=chunk, device=torch.device(device))
        outcomes = Counter((batch.observed[j].airport, r.withheld.outcome) for j, r in enumerate(results)
                           if isinstance(r, ClosedLoopSentence))
        by_rule[rule] = {"tally": tally, "outcomes": outcomes, "turns": turns(batch, rule, words.spec.step_s)}
        del batch, results
    return part[0], (drawn.description, drawn.excluded_seen), by_rule, \
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def merge(parts: list[dict[str, Any]], excluded: dict[str, int]) -> dict[str, Any]:
    """One rule's numbers from its parts' in order (module docstring)."""
    tally = instruction_closed_loop.merge_tallies([p["tally"] for p in parts])
    outcomes = sum((p["outcomes"] for p in parts), Counter())
    airports = sorted({airport for airport, _ in outcomes})
    by_airport = {}
    for airport in airports:
        counts = {outcome: n for (a, outcome), n in outcomes.most_common() if a == airport}
        by_airport[airport] = {"outcomes": counts, "landed": counts.get("landed", 0) / sum(counts.values())}
    rows = {name: np.concatenate([p["turns"][name] for p in parts]) for name in ("turn_deg", "start_from_after_deg")}
    rows["airports"] = [a for p in parts for a in p["turns"]["airports"]]
    return {"closed_loop": instruction_closed_loop.summarise(tally, excluded), "by_airport": by_airport,
            "start_track": turn_summary(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--row-interval-s", type=float, required=True, help="the chosen row interval (D11: 4 s)")
    parser.add_argument("--rules", nargs="+", choices=tuple(START_RULES), default=list(START_RULES))
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parts", type=int, default=1, help="train read in this many parts (module docstring)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.parts < 1:
        parser.error("--workers and --parts must be at least 1")
    if len(set(args.rules)) != len(args.rules):
        parser.error(f"--rules {args.rules} names a rule twice")
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    params, record, words = replay.open_executor(executor, instructions)
    interval_rows(args.row_interval_s, words.spec.step_s)                   # refused unless it divides 16 s
    rules = tuple(args.rules)
    started = time.perf_counter()
    done: dict[int, tuple[tuple[dict[str, Any], Counter], dict[str, Any]]] = {}

    def finished(k: int, drawn: tuple[dict[str, Any], Counter], by_rule: dict[str, Any], peak_gb: float) -> None:
        done[k] = (drawn, by_rule)
        print(f"part {k + 1} of {args.parts}: {drawn[0]['flights']} flights, peak {peak_gb:.1f} GB, "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    tasks = [(k, args.parts) for k in range(args.parts)]
    if args.workers == 1:
        for part in tasks:
            finished(*read_part(instructions, executor, part, args.row_interval_s, rules, chunk=args.chunk,
                                device=args.device))
    else:
        options = dict(instruction_closed_loop.POOL_OPTIONS)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context(options.pop("start_method")),
                                 **options) as pool:
            futures = {pool.submit(read_part, instructions, executor, part, args.row_interval_s, rules,
                                   chunk=args.chunk, device=args.device): part for part in tasks}
            for future in as_completed(futures):
                try:
                    finished(*future.result())
                except BaseException as error:                    # stop the parts not yet started, name the part
                    pool.shutdown(wait=False, cancel_futures=True)
                    k, n = futures[future]
                    raise SystemExit(f"part {k + 1} of {n} failed: {error!r}") from error
    drawn = replay.merge_descriptions([done[k][0] for k in range(args.parts)])
    readings = {rule: merge([done[k][1][rule] for k in range(args.parts)], drawn["excluded"]) for rule in rules}
    out.mkdir(parents=True)
    write_json_atomic(out / "readout.json", {
        "schema": READOUT_SCHEMA, "written_utc": utc_now(), "git": git, "instructions": str(instructions),
        "executor": str(executor), "executor_spec_sha256": record["sha256"], "checks": record["checks"],
        "spec_start_rule": params.start_rule, "vocabulary_spec_sha256": words.spec.sha256, "split": SPLIT,
        "seed": SEED, "parts": args.parts, "row_interval_s": args.row_interval_s, "drawn": drawn,
        "look_s": LOOK_S, "turn_deg": TURN_DEG,
        "rules": {rule: {"executor_params_sha256": params_sha256(replace(params, start_rule=rule)), **readings[rule]}
                  for rule in rules},
        "elapsed_s": time.perf_counter() - started})
    for rule, one in readings.items():
        loop, track = one["closed_loop"], one["start_track"]["start_from_after_deg"]
        median = "—" if track is None else f"{track['p50']:.1f}°"
        print(f"{rule}: {loop['sentences']} sentences, corrections {loop['correction_words']}, outcomes "
              f"{loop['outcomes']}; turning {one['start_track']['turning']}, start track from the direction after "
              f"p50 {median}", flush=True)
    print(f"→ {out / 'readout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
