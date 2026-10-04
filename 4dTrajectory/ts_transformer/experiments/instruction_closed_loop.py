"""The closed-loop reading (vocabulary §4.9, D32, §12.1 A9; `autopilot.closed_loop`): every labelled flight of every split
flown on its open-loop words by an executor spec, with the correction words its flown path needs, at each row interval
of the ablation — written into the instruction artefact as ``closed_loop/`` (never over an existing one), with its
reference sample, written in the same run; ``--check`` runs the checks every reader runs (`require_conforming_closed_loop`:
the labeller's, the executor's and the closed loop's, D73) and writes nothing.

Who: the replay's flights (`replay.group_of`): on their own dynamics or on a stand-in's; a flight with no identified type,
no aircraft dynamics or no published approach speed gives no training sentence. Each split's flights are counted by
reason in ``closed_loop/summary.json`` — not flown, refused on the row interval, refused by the closed loop — beside the
correction words for each column, the flights with any correction, the largest |e_y| and |e_h| per flight (percentiles),
the flights done at their time limit, and the lateness of the observed heading words (A12: the matched point's observed
time at the row that says a word minus the word's 2 s time, s; information). Written from a clean tree only (the artefact records the
commit).

``--workers N`` reads the SPLITS (`read_split`: the split drawn once, then read at every row interval in turn) in up to N
processes, the largest split (train) first; a split is the same reading in a worker as in one process (read in chunks of
``--chunk``), so the files and the summary are the same whatever N. A worker reads one split and ends (its memory goes
back), so the peak printed is the split's; a split that fails stops the splits not yet started, and its name is in the
error. (A worker for each split and row interval was tried first: it drew the train split three times over, 12 min each,
and three train draws at once ran the host out of memory.)

    python run_ts.py instruction_closed_loop --row-interval-s 2 4 8 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> \\
        --executor 4dTrajectory/outputs/POOLED/executor/<name>
    python run_ts.py instruction_closed_loop --check --instructions <artefact> --executor <name>
"""

from __future__ import annotations

import argparse
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.instructions.artefact import (
    CLOSED_LOOP_DIRECTORY, CLOSED_LOOP_SCHEMA, SPLITS, ClosedLoopSentence, write_closed_loop,
)
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

SEED = 1337
#: How the workers start (`--workers`): spawned, as the other runners' pools, one split each (tests fork, so their
#: stand-ins carry over; a forked pool takes no task limit).
POOL_OPTIONS = {"start_method": "spawn", "max_tasks_per_child": 1}


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 95, 99)} | {"max": float(max(values)),
                                                                                   "n": len(values)}


def summarise(results: list[ClosedLoopSentence | Any], excluded: dict[str, int],
              refused_on_interval: dict[str, int], heading_lateness_s: list[np.ndarray],
              outside: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """One split's numbers at one row interval (module docstring); ``heading_lateness_s`` and ``outside`` each sentence's
    (`closed_loop.heading_lateness_rows`, in seconds; `closed_loop.outside_rows`)."""
    read = [r for r in results if isinstance(r, ClosedLoopSentence)]
    counts = {name: np.zeros(3, dtype=np.int64) for name in ("lateral", "vertical")}
    for rows_of in outside:
        for name, rows in rows_of.items():
            counts[name] += rows.sum(axis=1)
    lateness = np.concatenate([np.zeros(0), *heading_lateness_s])
    refused = Counter(r.reason for r in results if not isinstance(r, ClosedLoopSentence))
    corrections = np.array([r.correction.sum(axis=0) for r in read]).reshape(-1, len(COLUMNS))
    return {"sentences": len(read),
            "without_a_sentence": {"not flown": excluded, "refused on the row interval": refused_on_interval,
                                   "refused by the closed loop": dict(refused.most_common())},
            "correction_words": dict(zip(COLUMNS, corrections.sum(axis=0).tolist())),
            "flights_with_a_correction": {column: int((corrections[:, c] > 0).sum())
                                          for c, column in enumerate(COLUMNS)},
            "largest_lateral_m": _percentiles([closed_loop.largest_m(r.lateral_m) for r in read]),
            "largest_vertical_m": _percentiles([closed_loop.largest_m(r.vertical_m) for r in read]),
            # D34: what is left where §4.9 makes no correction
            "uncorrected_lateral_m": _percentiles([closed_loop.uncorrected_m(r.lateral_m, r.uncorrectable[:, 0])
                                                   for r in read]),
            "uncorrected_vertical_m": _percentiles([closed_loop.uncorrected_m(r.vertical_m, r.uncorrectable[:, 1])
                                                    for r in read]),
            "last_row_lateral_m": _percentiles([float(abs(r.lateral_m[-1])) for r in read]),
            # D34 reading 3 and the rule of D50: the correctable rows, those outside the tolerance, and of those the
            # rows after which no correction toward the path is in force
            "outside_the_tolerance": {name: {"correctable_rows": int(c[0]), "outside": int(c[1]),
                                             "share": float(c[1] / c[0]) if c[0] else None,
                                             "without_a_correction_toward_the_path": int(c[2])}
                                      for name, c in counts.items()},
            "timed_out": sum(1 for r in read if r.timed_out),
            # D44: the rows past the end of the observed path (no observed height there)
            "rows_past_the_end": int(sum(np.isnan(r.vertical_m).sum() for r in read)),
            "heading_word_lateness_s": None if not len(lateness) else {
                "mean": float(lateness.mean()), "n": len(lateness),
                **{f"p{q}": float(np.percentile(lateness, q)) for q in (5, 25, 50, 75, 95)}}}


def read_interval(drawn: replay.Drawn, readings: list[Any], interval_s: float, params: Any, words: Any, *, chunk: int,
                  device: torch.device) -> tuple[replay.Batch, list[ClosedLoopSentence | Any], dict[str, Any]]:
    """One split's drawn flights read in closed loop at one row interval: the batch, every flight's result and the split's
    numbers (`summarise`)."""
    batch = replay.batch_of(drawn, list(range(len(readings))), readings, interval_s, words)
    results = closed_loop.read_chunked(batch, params, words, chunk=chunk, device=device)
    kept = [(j, r) for j, r in enumerate(results) if isinstance(r, ClosedLoopSentence)]
    lateness = [closed_loop.heading_lateness_rows(r, batch.readings[j].words[batch.sentences[j].first_row:])
                * words.spec.step_s for j, r in kept]
    outside = [closed_loop.outside_rows(r, batch.readings[j].words, batch.sentences[j].first_row, words,
                                        [c.course_deg for c in batch.geometries[j].candidates]) for j, r in kept]
    return batch, results, summarise(results, drawn.description["excluded"], batch.drawn["refused_on_interval"],
                                     lateness, outside)


def read_split(instructions: Path, executor: Path, split: str, intervals_s: list[float], staging: Path, *, chunk: int,
               device: str) -> tuple[str, dict[str, Any], dict[float, dict[str, Any]], float]:
    """One split (module docstring): its labelled flights drawn once and read in closed loop at each of ``intervals_s``
    in turn, each interval's sentences written into ``staging`` before the next is read (none when no flight keeps one:
    the numbers say why). Returns the split, the draw's description, the numbers by interval and the process's peak
    memory (GB, Linux's ``ru_maxrss`` in KiB)."""
    params, _, words = replay.open_executor(executor, instructions)
    drawn, readings = replay.draw_readings(instructions, split, words.spec, words, per_airport=0, seed=SEED,
                                           groups=(replay.OWN, replay.STAND_IN))
    numbers = {}
    for interval_s in intervals_s:
        batch, results, numbers[interval_s] = read_interval(drawn, readings, interval_s, params, words, chunk=chunk,
                                                            device=torch.device(device))
        kept = [(j, r) for j, r in enumerate(results) if isinstance(r, ClosedLoopSentence)]
        if kept:
            write_closed_loop(staging / f"{split}_{interval_s:g}s.npz", words.spec,
                              executor_params_sha256=params_sha256(params), row_interval_s=interval_s,
                              start_row=closed_loop.start_row(interval_s),
                              sentences={batch.indices[j]: r for j, r in kept})
        del batch, results, kept
    return split, drawn.description, numbers, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--row-interval-s", type=float, nargs="+", default=None,
                        help="the row intervals of the ablation (vocabulary §4.8, D25: 2, 4, 8 s)")
    parser.add_argument("--check", action="store_true", help="run the readers' checks against the artefact; write nothing")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1, help="splits read in up to this many processes (module docstring)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    if args.check:
        closed_loop.require_conforming_closed_loop(instructions, executor)        # refused by name on a difference
        print("conforming")
        return 0
    git = git_state()
    if git["dirty"]:
        parser.error("the tree has uncommitted changes; the closed-loop reading is written at a commit (it records it)")
    params, record, words = replay.open_executor(executor, instructions)
    if args.row_interval_s is None:
        parser.error("--row-interval-s is required to write the closed-loop reading")
    for interval in args.row_interval_s:
        interval_rows(interval, words.spec.step_s)                      # refused unless it divides 16 s
    if len(set(args.row_interval_s)) != len(args.row_interval_s):
        parser.error(f"--row-interval-s {args.row_interval_s} names a row interval twice")
    target = instructions / CLOSED_LOOP_DIRECTORY
    if target.exists():
        parser.error(f"{target} exists; the closed-loop reading is never overwritten")
    staging = instructions / f".{CLOSED_LOOP_DIRECTORY}.writing-{os.getpid()}"
    staging.mkdir()
    started = time.perf_counter()
    summary: dict[str, Any] = {"schema": CLOSED_LOOP_SCHEMA, "written_utc": utc_now(), "git": git,
                               "executor": executor.relative_to(REPO_ROOT).as_posix()
                               if executor.is_relative_to(REPO_ROOT) else str(executor),
                               "executor_spec_sha256": record["sha256"], "executor_params_sha256": params_sha256(params),
                               "checks": record["checks"], "row_intervals_s": args.row_interval_s, "splits": {}}
    done: dict[str, tuple[dict[str, Any], dict[float, dict[str, Any]]]] = {}

    def finished(split: str, drawn: dict[str, Any], numbers: dict[float, dict[str, Any]], peak_gb: float) -> None:
        done[split] = (drawn, numbers)
        for interval, one in numbers.items():
            print(f"{split} {interval:g} s: {one['sentences']} sentences, corrections {one['correction_words']}, "
                  f"without a sentence {one['without_a_sentence']}", flush=True)
        print(f"{split}: peak {peak_gb:.1f} GB, {time.perf_counter() - started:.0f}s", flush=True)

    intervals = list(args.row_interval_s)
    if args.workers == 1:
        for split in SPLITS:                                       # SPLITS: train, the largest, first
            finished(*read_split(instructions, executor, split, intervals, staging, chunk=args.chunk,
                                 device=args.device))
    else:
        options = dict(POOL_OPTIONS)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context(options.pop("start_method")),
                                 **options) as pool:
            futures = {pool.submit(read_split, instructions, executor, split, intervals, staging, chunk=args.chunk,
                                   device=args.device): split for split in SPLITS}
            for future in as_completed(futures):
                try:
                    finished(*future.result())
                except BaseException as error:                    # stop the splits not yet started, name the split
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(f"the split {futures[future]} failed: {error!r}") from error
    for split in SPLITS:
        drawn, numbers = done[split]
        summary["splits"][split] = {"drawn": drawn, "intervals": {
            f"{interval:g}": numbers[interval] for interval in intervals}}
    write_json_atomic(staging / "summary.json", summary)
    closed_loop.write_reference(instructions, params, words, args.row_interval_s, git=git,
                                target=staging / closed_loop.CONFORMANCE)
    staging.rename(target)
    checked = closed_loop.check(instructions, params, words)
    if not checked.passed:
        raise SystemExit(f"the code that wrote the reference reads it otherwise: {checked.mismatches}")
    print(f"→ {target} (its reference read again: {checked.flights} flights, largest state difference "
          f"{checked.largest_state_difference_m:.2g} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
