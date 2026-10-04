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

``--workers N`` reads the splits in up to N processes (`read_part`: drawn once, then read at every row interval in turn),
the largest split (train) first; ``--train-parts P`` cuts train into P parts — consecutive blocks of the one seeded
permutation it is drawn in (`replay.part_of`) — each drawn and read by itself in its own process: A25's train reading took
1,837 s of a 31 min closed loop in one process, and a part draws and reads only its flights. A part is the same reading in
a worker as in one process (read in chunks of ``--chunk``), its sentences are put together in the parts' order
(`join_parts`), its draw's description and its numbers too (`replay.merge_descriptions`, `merge_tallies`: each sentence's
numbers in order, so the percentiles are those of the split read whole), so the files and the summary are the same whatever
N and P. A worker reads one part and ends (its memory goes back), so the peak printed is the part's; a part that fails
stops those not yet started, and its name is in the error. (A worker for each split and row interval was tried first: it
drew the train split three times over, 12 min each, and three train draws at once ran the host out of memory.)

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.instructions.artefact import (
    CLOSED_LOOP_DIRECTORY, CLOSED_LOOP_SCHEMA, SPLITS, ClosedLoopSentence, closed_loop_sentences, load_closed_loop,
    write_closed_loop,
)
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

SEED = 1337
#: How the workers start (`--workers`): spawned, as the other runners' pools, one split each (tests fork, so their
#: stand-ins carry over; a forked pool takes no task limit).
POOL_OPTIONS = {"start_method": "spawn", "max_tasks_per_child": 1}
#: The staging directory's folder for the parts of a split read in parts (`join_parts` empties and removes it).
PARTS = "parts"


@dataclass
class Tally:
    """What one split's numbers at one row interval are made of (`summarise`), sentence by sentence in the reading's
    order: the parts of a split drawn in parts are put together in their order (`merge_tallies`) and give the numbers of
    the split read whole."""

    corrections: np.ndarray                 # [sentences, 5] the correction words of each sentence per column
    largest_lateral_m: np.ndarray           # [sentences] each sentence's largest |e_y| (`closed_loop.largest_m`)
    largest_vertical_m: np.ndarray
    uncorrected_lateral_m: np.ndarray       # [sentences] on the rows without a correction (D34)
    uncorrected_vertical_m: np.ndarray
    last_row_lateral_m: np.ndarray
    timed_out: np.ndarray                   # [sentences] bool
    outcomes: list[str]                     # D74
    rows_past_the_end: int                  # D44
    refused: Counter                        # the flights the closed loop refused, by reason
    refused_on_interval: Counter            # the flights the row interval refused, by reason in the order first met
    lateness_s: np.ndarray                  # the observed heading words' lateness (A12), sentence after sentence
    outside: dict[str, np.ndarray]          # by column the three counts of `closed_loop.outside_rows`


def tally(batch: replay.Batch, results: list[ClosedLoopSentence | Any], words: Any) -> Tally:
    """A row interval's reading of ``batch`` (`read_interval`) as its `Tally`."""
    kept = [(j, r) for j, r in enumerate(results) if isinstance(r, ClosedLoopSentence)]
    read = [r for _, r in kept]
    outside = {name: np.zeros(3, dtype=np.int64) for name in ("lateral", "vertical")}
    for j, r in kept:
        rows_of = closed_loop.outside_rows(r, batch.readings[j].words, batch.sentences[j].first_row, words,
                                           [c.course_deg for c in batch.geometries[j].candidates])
        for name, rows in rows_of.items():
            outside[name] += rows.sum(axis=1)
    lateness = [closed_loop.heading_lateness_rows(r, batch.readings[j].words[batch.sentences[j].first_row:])
                * words.spec.step_s for j, r in kept]
    return Tally(
        corrections=np.array([r.correction.sum(axis=0) for r in read], dtype=np.int64).reshape(-1, len(COLUMNS)),
        largest_lateral_m=np.array([closed_loop.largest_m(r.lateral_m) for r in read]),
        largest_vertical_m=np.array([closed_loop.largest_m(r.vertical_m) for r in read]),
        uncorrected_lateral_m=np.array([closed_loop.uncorrected_m(r.lateral_m, r.uncorrectable[:, 0]) for r in read]),
        uncorrected_vertical_m=np.array([closed_loop.uncorrected_m(r.vertical_m, r.uncorrectable[:, 1]) for r in read]),
        last_row_lateral_m=np.array([float(abs(r.lateral_m[-1])) for r in read]),
        timed_out=np.array([r.timed_out for r in read], dtype=bool), outcomes=[r.outcome for r in read],
        rows_past_the_end=int(sum(np.isnan(r.vertical_m).sum() for r in read)),
        refused=Counter(r.reason for r in results if not isinstance(r, ClosedLoopSentence)),
        refused_on_interval=batch.refused_seen,
        lateness_s=np.concatenate([np.zeros(0), *lateness]), outside=outside)


def merge_tallies(parts: list[Tally]) -> Tally:
    """The `Tally` of a split read whole, from those of its parts in order."""
    joined = {name: np.concatenate([getattr(p, name) for p in parts]) for name in (
        "corrections", "largest_lateral_m", "largest_vertical_m", "uncorrected_lateral_m", "uncorrected_vertical_m",
        "last_row_lateral_m", "timed_out", "lateness_s")}
    return Tally(**joined, outcomes=[o for p in parts for o in p.outcomes],
                 rows_past_the_end=sum(p.rows_past_the_end for p in parts),
                 refused=sum((p.refused for p in parts), Counter()),
                 refused_on_interval=sum((p.refused_on_interval for p in parts), Counter()),
                 outside={name: sum(p.outside[name] for p in parts) for name in parts[0].outside})


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    return {f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 95, 99)} | {"max": float(values.max()),
                                                                                   "n": len(values)}


def summarise(counted: Tally, excluded: dict[str, int]) -> dict[str, Any]:
    """One split's numbers at one row interval (module docstring), from its `Tally` and the flights its draw did not fly
    (``excluded``, by reason)."""
    corrections = counted.corrections
    lateness = counted.lateness_s
    return {"sentences": len(corrections),
            "without_a_sentence": {"not flown": excluded,
                                   "refused on the row interval": dict(counted.refused_on_interval.most_common()),
                                   "refused by the closed loop": dict(counted.refused.most_common())},
            "correction_words": dict(zip(COLUMNS, corrections.sum(axis=0).tolist())),
            "flights_with_a_correction": {column: int((corrections[:, c] > 0).sum())
                                          for c, column in enumerate(COLUMNS)},
            "largest_lateral_m": _percentiles(counted.largest_lateral_m),
            "largest_vertical_m": _percentiles(counted.largest_vertical_m),
            # D34: what is left where §4.9 makes no correction
            "uncorrected_lateral_m": _percentiles(counted.uncorrected_lateral_m),
            "uncorrected_vertical_m": _percentiles(counted.uncorrected_vertical_m),
            "last_row_lateral_m": _percentiles(counted.last_row_lateral_m),
            # D34 reading 3 and the rule of D50: the correctable rows, those outside the tolerance, and of those the
            # rows after which no correction toward the path is in force
            "outside_the_tolerance": {name: {"correctable_rows": int(c[0]), "outside": int(c[1]),
                                             "share": float(c[1] / c[0]) if c[0] else None,
                                             "without_a_correction_toward_the_path": int(c[2])}
                                      for name, c in counted.outside.items()},
            "timed_out": int(counted.timed_out.sum()),
            # D74: the sentences by the judge's outcome of what the reading flew (for readouts and selection)
            "outcomes": dict(Counter(counted.outcomes).most_common()),
            # D44: the rows past the end of the observed path (no observed height there)
            "rows_past_the_end": counted.rows_past_the_end,
            "heading_word_lateness_s": None if not len(lateness) else {
                "mean": float(lateness.mean()), "n": len(lateness),
                **{f"p{q}": float(np.percentile(lateness, q)) for q in (5, 25, 50, 75, 95)}}}


def read_interval(drawn: replay.Drawn, readings: list[Any], interval_s: float, params: Any, words: Any, *, chunk: int,
                  device: torch.device) -> tuple[replay.Batch, list[ClosedLoopSentence | Any], Tally]:
    """One split's drawn flights (or a part of them) read in closed loop at one row interval: the batch, every flight's
    result and its `Tally`."""
    batch = replay.batch_of(drawn, list(range(len(readings))), readings, interval_s, words)
    results = closed_loop.read_chunked(batch, params, words, chunk=chunk, device=device)
    return batch, results, tally(batch, results, words)


def part_path(staging: Path, split: str, interval_s: float, part: tuple[int, int]) -> Path:
    """Where a part's closed-loop sentences wait to be put together (`read_part`); a split read whole goes straight
    to its file."""
    k, n = part
    if n == 1:
        return staging / f"{split}_{interval_s:g}s.npz"
    return staging / PARTS / f"{split}_{interval_s:g}s.part{k}.npz"


def read_part(instructions: Path, executor: Path, split: str, part: tuple[int, int], intervals_s: list[float],
              staging: Path, *, chunk: int, device: str
              ) -> tuple[str, int, tuple[dict[str, Any], Counter], dict[float, Tally], float]:
    """Part ``part`` = ``(k, n)`` of a split (module docstring; ``(0, 1)`` the whole split): its labelled flights drawn
    once (`replay.draw_readings`) and read in closed loop at each of ``intervals_s`` in turn, each interval's sentences
    written (`part_path`) before the next is read (none when no flight keeps one: the tally says why). Returns the split,
    ``k``, the draw's description and exclusions in the order first met (`replay.merge_descriptions`), the tallies by
    interval and the process's peak memory (GB, Linux's ``ru_maxrss`` in KiB)."""
    params, _, words = replay.open_executor(executor, instructions)
    drawn, readings = replay.draw_readings(instructions, split, words.spec, words, per_airport=0, seed=SEED,
                                           groups=(replay.OWN, replay.STAND_IN), part=part)
    tallies = {}
    for interval_s in intervals_s:
        batch, results, tallies[interval_s] = read_interval(drawn, readings, interval_s, params, words, chunk=chunk,
                                                            device=torch.device(device))
        kept = [(j, r) for j, r in enumerate(results) if isinstance(r, ClosedLoopSentence)]
        if kept:
            write_closed_loop(part_path(staging, split, interval_s, part), words.spec,
                              executor_params_sha256=params_sha256(params), row_interval_s=interval_s,
                              start_row=closed_loop.start_row(interval_s),
                              sentences={batch.indices[j]: r for j, r in kept})
        del batch, results, kept
    return (split, part[0], (drawn.description, drawn.excluded_seen), tallies,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)


def join_parts(staging: Path, split: str, tallies: list[dict[float, Tally]], params: Any, words: Any) -> None:
    """A split read in parts (``tallies``: each part's, in order): each interval's part files put together in order into
    the split's file, as the split read whole writes it, and taken away. Refused unless a part has a file exactly when
    its tally counts a sentence, and the split's file holds every sentence the tallies count."""
    for interval_s in tallies[0]:
        sentences: dict[int, ClosedLoopSentence] = {}
        written = []
        for k, counted in enumerate(tallies):
            path = part_path(staging, split, interval_s, (k, len(tallies)))
            if path.exists() != bool(counted[interval_s].outcomes):
                raise ValueError(f"{path.name}: a part's file is there exactly when its tally counts a sentence")
            if path.exists():
                sentences.update(closed_loop_sentences(load_closed_loop(path, words.spec)))
                written.append(path)
        if len(sentences) != sum(len(counted[interval_s].outcomes) for counted in tallies):
            raise ValueError(f"{split} {interval_s:g} s: the parts' files do not hold the sentences their tallies count")
        if sentences:
            write_closed_loop(staging / f"{split}_{interval_s:g}s.npz", words.spec,
                              executor_params_sha256=params_sha256(params), row_interval_s=interval_s,
                              start_row=closed_loop.start_row(interval_s), sentences=sentences)
        del sentences
        for path in written:
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--row-interval-s", type=float, nargs="+", default=None,
                        help="the row intervals of the ablation (vocabulary §4.8, D25: 2, 4, 8 s)")
    parser.add_argument("--check", action="store_true", help="run the readers' checks against the artefact; write nothing")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1,
                        help="splits, or parts of the train split, read in up to this many processes (module docstring)")
    parser.add_argument("--train-parts", type=int, default=1,
                        help="the train split read in this many parts, each drawn and read by itself (module docstring)")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.train_parts < 1:
        parser.error("--workers and --train-parts must be at least 1")
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
    parts = {split: args.train_parts if split == "train" else 1 for split in SPLITS}
    tasks = [(split, (k, parts[split])) for split in SPLITS for k in range(parts[split])]   # train, the largest, first
    done: dict[str, dict[int, tuple[tuple[dict[str, Any], Counter], dict[float, Tally]]]] = {split: {} for split in SPLITS}
    intervals = list(args.row_interval_s)
    if any(n > 1 for n in parts.values()):
        (staging / PARTS).mkdir()

    def finished(split: str, k: int, drawn: tuple[dict[str, Any], Counter], tallies: dict[float, Tally],
                 peak_gb: float) -> None:
        done[split][k] = (drawn, tallies)
        name = split if parts[split] == 1 else f"{split} part {k + 1} of {parts[split]}"
        print(f"{name}: {drawn[0]['flights']} flights, peak {peak_gb:.1f} GB, {time.perf_counter() - started:.0f}s",
              flush=True)

    if args.workers == 1:
        for split, part in tasks:
            finished(*read_part(instructions, executor, split, part, intervals, staging, chunk=args.chunk,
                                device=args.device))
    else:
        options = dict(POOL_OPTIONS)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context(options.pop("start_method")),
                                 **options) as pool:
            futures = {pool.submit(read_part, instructions, executor, split, part, intervals, staging, chunk=args.chunk,
                                   device=args.device): (split, part) for split, part in tasks}
            for future in as_completed(futures):
                try:
                    finished(*future.result())
                except BaseException as error:                    # stop the parts not yet started, name the part
                    pool.shutdown(wait=False, cancel_futures=True)
                    split, (k, n) = futures[future]
                    raise SystemExit(f"the split {split} (part {k + 1} of {n}) failed: {error!r}") from error
    for split in SPLITS:
        if parts[split] > 1:
            join_parts(staging, split, [done[split][k][1] for k in range(parts[split])], params, words)
        drawn = replay.merge_descriptions([done[split][k][0] for k in range(parts[split])])
        numbers = {interval: summarise(merge_tallies([done[split][k][1][interval] for k in range(parts[split])]),
                                       drawn["excluded"]) for interval in intervals}
        for interval, one in numbers.items():
            print(f"{split} {interval:g} s: {one['sentences']} sentences, corrections {one['correction_words']}, "
                  f"without a sentence {one['without_a_sentence']}, outcomes {one['outcomes']}", flush=True)
        summary["splits"][split] = {"drawn": drawn, "intervals": {
            f"{interval:g}": numbers[interval] for interval in intervals}}
    if any(n > 1 for n in parts.values()):
        (staging / PARTS).rmdir()
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
