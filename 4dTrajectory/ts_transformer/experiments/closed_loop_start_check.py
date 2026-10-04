"""Vocabulary A26 on a real artefact (§12.2 item 7, D67): stored closed-loop sentences said through the start of a closed
loop (`autopilot.start`) must give back their stored states — the check that a speaker's loop starts and flies where the
closed-loop reading did, with the flights rebuilt from the harvest, their aircraft and approach speeds by the replay's
rule and their time limits from the sentence file (what `tests/test_start.py` stands in for).

For each ``--row-interval-s``: the artefact's closed-loop sentences of ``--split`` (`artefact.closed_loop_sentences`),
``--per-airport`` of each airport in a permutation seeded by ``--seed`` (0: every one; an airport with fewer is refused),
started ``--chunk`` flights together (`start.start`, the most go-arounds the most any of them says) and said row by row —
each flight's own rows, then "unchanged" once its sentence has ended; a flight's 2 s rows are kept while it flies, so one
done early has fewer rows than stored. Per flight: the largest difference of the position and height (columns 0–2) and
of the other columns (the track wrapped to ±180°) between the flown 2 s rows and the stored ones; whether it is done
exactly at its sentence's last row; and whether it timed out as stored. Refused (exit 1) unless every flight is within the executor conformance's
bound (`STATE_BOUND_M`; other columns `ROUNDOFF`), done at its last row and timed out as stored. Writes ``check.json``.

    python run_ts.py closed_loop_start_check --instructions <artefact> --executor <spec> --split train \\
        --row-interval-s 2 4 8 --per-airport 50 --out <new dir>
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import closed_loop
from ts_transformer.autopilot.conformance import ROUNDOFF, STATE_BOUND_M
from ts_transformer.autopilot.start import start
from ts_transformer.instructions.artefact import (
    SPLITS, closed_loop_path, closed_loop_sentences, load_closed_loop, load_signals,
)
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.instructions.words import RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, wrap180
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: v2 (A29, D73): the checks the run ran before its work, as information.
CHECK_SCHEMA = "ts-closed-loop-start-check-v2"


def sample(airports: list[str], per_airport: int, seed: int) -> list[int]:
    """The places (in ``airports``' order) of the first ``per_airport`` of each airport in a permutation seeded by
    ``seed`` (0: every one), sorted; refused for an airport with fewer."""
    if per_airport < 0:
        raise ValueError(f"--per-airport {per_airport} is negative")
    if not per_airport:
        return list(range(len(airports)))
    short = {a: n for a, n in Counter(airports).items() if n < per_airport}
    if short:
        raise ValueError(f"airports with fewer than {per_airport} closed-loop sentences: {short}")
    left = {airport: per_airport for airport in set(airports)}
    chosen = []
    for k in np.random.default_rng(seed).permutation(len(airports)).tolist():
        if left[airports[k]]:
            left[airports[k]] -= 1
            chosen.append(k)
    return sorted(chosen)


def check_interval(instructions: Path, split: str, interval_s: float, executor: Path, words: Any, *, per_airport: int,
                   seed: int, chunk: int, device: torch.device) -> list[dict[str, Any]]:
    """One row interval's flights said through the start (module docstring), ``chunk`` at a time: one row each."""
    stored = closed_loop_sentences(load_closed_loop(closed_loop_path(instructions, split, interval_s), words.spec))
    signals = load_signals(instructions, split)
    keys = sorted(stored)
    chosen = [keys[k] for k in sample([signals[i].airport for i in keys], per_airport, seed)]
    out = []
    for first in range(0, len(chosen), chunk):
        out += _check_chunk(instructions, split, interval_s, {i: stored[i] for i in chosen[first: first + chunk]},
                            signals, executor, device=device)
    return out


def _check_chunk(instructions: Path, split: str, interval_s: float, chosen: dict[int, Any], signals: list[Any],
                 executor: Path, *, device: torch.device) -> list[dict[str, Any]]:
    most = max(int((s.grid[:, RUNWAY] == RUNWAY_GO_AROUND).sum()) for s in chosen.values())
    loop, order = start(instructions, split, interval_s, chosen, executor, most_go_arounds=most, device=device)
    every = interval_rows(interval_s, loop.words.spec.step_s)
    flown: list[list[np.ndarray]] = [[row] for row in loop.rows()]
    done_at = np.full(len(order), -1)
    for k in range(max(len(chosen[i].grid) for i in order)):
        said = np.full((len(order), 5), UNCHANGED, dtype=np.int64)
        for f, i in enumerate(order):
            if k < len(chosen[i].grid):
                said[f] = chosen[i].grid[k]
        rows, done = loop.step(said)
        for f, i in enumerate(order):
            if k < len(chosen[i].grid) - 1 and not done[f]:       # kept while it flies: an early end shows
                flown[f] += list(rows[f])
        done_at = np.where(done & (done_at < 0), k, done_at)
    timed_out = loop.timed_out()
    out = []
    for f, i in enumerate(order):
        sentence = chosen[i]
        expected = sentence.states[sentence.start * every:]
        got = np.array(flown[f])
        same_shape = got.shape == expected.shape
        apart = got - expected if same_shape else None
        if same_shape:
            apart[:, 3] = wrap180(apart[:, 3])                     # the track, compass degrees
        out.append({"index": i, "dataset_id": signals[i].dataset_id, "airport": signals[i].airport,
                    "rows": len(sentence.grid), "same_rows": same_shape,
                    "position_m": float(np.abs(apart[:, :3]).max()) if same_shape else None,
                    "other_columns": float(np.abs(apart[:, 3:]).max()) if same_shape else None,
                    "done_at_last_row": bool(done_at[f] == len(sentence.grid) - 1),
                    "timed_out_as_stored": bool(timed_out[f]) == sentence.timed_out})
    return out


def passes(row: dict[str, Any]) -> bool:
    return (row["same_rows"] and row["position_m"] <= STATE_BOUND_M and row["other_columns"] <= ROUNDOFF
            and row["done_at_last_row"] and row["timed_out_as_stored"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--row-interval-s", type=float, nargs="+", required=True)
    parser.add_argument("--per-airport", type=int, default=50, help="0: every closed-loop sentence of the split")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=2048, help="flights started together")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a check is never overwritten")
    _, record, words = closed_loop.require_conforming_closed_loop(instructions, executor)
    started = time.perf_counter()
    intervals = {}
    for interval in args.row_interval_s:
        rows = check_interval(instructions, args.split, interval, executor, words, per_airport=args.per_airport,
                              seed=args.seed, chunk=args.chunk, device=torch.device(args.device))
        failed = [r["dataset_id"] for r in rows if not passes(r)]
        intervals[f"{interval:g}"] = {"flights": len(rows), "failed": failed,
                                      "largest_position_m": max(r["position_m"] or 0.0 for r in rows),
                                      "largest_other_columns": max(r["other_columns"] or 0.0 for r in rows),
                                      "rows": rows}
        print(f"Δ {interval:g} s: {len(rows)} flights, {len(failed)} failed, largest position difference "
              f"{intervals[f'{interval:g}']['largest_position_m']:.3g} m, {time.perf_counter() - started:.0f}s", flush=True)
    out.mkdir(parents=True)
    write_json_atomic(out / "check.json", {
        "schema": CHECK_SCHEMA, "written_utc": utc_now(), "git": git_state(), "instructions": str(instructions),
        "executor": str(executor), "executor_spec_sha256": record["sha256"], "checks": record["checks"],
        "split": args.split,
        "per_airport": args.per_airport, "seed": args.seed, "bounds": {"position_m": STATE_BOUND_M,
                                                                       "other_columns": ROUNDOFF},
        "intervals": intervals})
    failed = sum(len(v["failed"]) for v in intervals.values())
    print(f"→ {out / 'check.json'}: {'PASSED' if not failed else f'{failed} FAILED'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
