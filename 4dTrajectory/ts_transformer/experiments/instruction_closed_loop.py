"""The closed-loop reading (design §4.9, D32, §14.2 A9; `autopilot.closed_loop`): every labelled flight of every split
flown on its open-loop words by an executor spec, with the correction words its flown path needs, at each row interval
of the ablation — written into the instruction artefact as ``closed_loop/`` (never over an existing one), with its
reference sample and the passed record of the code that wrote it; ``--check`` reads that reference again with the code
on disk and writes its passed record.

Who: the replay's flights (`replay.group_of`): on their own dynamics or on a stand-in's; a flight with no identified type,
no aircraft dynamics or no published approach speed gives no training sentence. Each split's flights are counted by
reason in ``closed_loop/summary.json`` — not flown, refused on the row interval, refused by the closed loop — beside the
correction words for each column, the flights with any correction, the largest |e_y| and |e_h| per flight (percentiles),
the flights done at their time limit, and the lateness of the observed heading words (A12: the matched point's observed
time at the row that says a word minus the word's 2 s time, s; information). Written from a clean tree only (the artefact records the
commit).

    python run_ts.py instruction_closed_loop --row-interval-s 2 4 8 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> \\
        --executor 4dTrajectory/outputs/POOLED/executor/<name>
    python run_ts.py instruction_closed_loop --check --instructions <artefact> --executor <name>
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.closed_loop import ClosedLoopSentence
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.instructions.artefact import CLOSED_LOOP_DIRECTORY, CLOSED_LOOP_SCHEMA, SPLITS, write_closed_loop
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

SEED = 1337


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 95, 99)} | {"max": float(max(values)),
                                                                                   "n": len(values)}


def summarise(results: list[ClosedLoopSentence | Any], excluded: dict[str, int],
              refused_on_interval: dict[str, int], heading_lateness_s: list[np.ndarray]) -> dict[str, Any]:
    """One split's numbers at one row interval (module docstring); ``heading_lateness_s`` each sentence's
    (`closed_loop.heading_lateness_rows`, in seconds)."""
    read = [r for r in results if isinstance(r, ClosedLoopSentence)]
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
            "timed_out": sum(1 for r in read if r.timed_out),
            # D44: the rows past the end of the observed path (no observed height there)
            "rows_past_the_end": int(sum(np.isnan(r.vertical_m).sum() for r in read)),
            "heading_word_lateness_s": None if not len(lateness) else {
                "mean": float(lateness.mean()), "n": len(lateness),
                **{f"p{q}": float(np.percentile(lateness, q)) for q in (5, 25, 50, 75, 95)}}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--row-interval-s", type=float, nargs="+", default=None,
                        help="the row intervals of the ablation (design §4.8, D25: 2, 4, 8 s)")
    parser.add_argument("--check", action="store_true", help="read the reference again and write the passed record")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    git = git_state()
    if git["dirty"]:
        parser.error("the tree has uncommitted changes; the closed-loop reading is written and checked at a commit")
    params, record, words = replay.open_executor(executor, instructions)
    if args.check:
        checked = closed_loop.check(instructions, params, words, git=git)
        print(f"{checked.flights} reference flights read again by code {checked.code_sha256[:12]}; largest state "
              f"difference {checked.largest_state_difference_m:.3g} m; {len(checked.mismatches)} read otherwise")
        for name, problems in list(checked.mismatches.items())[:20]:
            print(f"  {name}: " + "; ".join(problems[:6]))
        if not checked.passed:
            print("NOT conforming: no passed record written")
            return 1
        print(f"conforming → {closed_loop.write_passed(instructions, checked)}")
        return 0
    if args.row_interval_s is None:
        parser.error("--row-interval-s is required to write the closed-loop reading")
    for interval in args.row_interval_s:
        interval_rows(interval, words.spec.step_s)                      # refused unless it divides 16 s
    target = instructions / CLOSED_LOOP_DIRECTORY
    if target.exists():
        parser.error(f"{target} exists; the closed-loop reading is never overwritten")
    staging = instructions / f".{CLOSED_LOOP_DIRECTORY}.writing-{os.getpid()}"
    staging.mkdir()
    device, started = torch.device(args.device), time.perf_counter()
    summary: dict[str, Any] = {"schema": CLOSED_LOOP_SCHEMA, "written_utc": utc_now(), "git": git,
                               "executor": executor.relative_to(REPO_ROOT).as_posix()
                               if executor.is_relative_to(REPO_ROOT) else str(executor),
                               "executor_spec_sha256": record["sha256"], "executor_params_sha256": params_sha256(params),
                               "code_sha256": closed_loop.closed_loop_code_sha256(),
                               "row_intervals_s": args.row_interval_s, "splits": {}}
    for split in SPLITS:
        drawn, readings = replay.draw_readings(instructions, split, words.spec, words, per_airport=0, seed=SEED,
                                               groups=(replay.OWN, replay.STAND_IN))
        summary["splits"][split] = {"drawn": drawn.description, "intervals": {}}
        for interval in args.row_interval_s:
            batch = replay.batch_of(drawn, list(range(len(readings))), readings, interval, words)
            results = closed_loop.read_chunked(batch, params, words, chunk=args.chunk, device=device)
            kept = [(j, r) for j, r in enumerate(results) if isinstance(r, ClosedLoopSentence)]
            lateness = [closed_loop.heading_lateness_rows(r, batch.readings[j].words[batch.sentences[j].first_row:])
                        * words.spec.step_s for j, r in kept]
            if not kept:                                       # nothing to write: the summary says why
                summary["splits"][split]["intervals"][f"{interval:g}"] = summarise(
                    results, drawn.description["excluded"], batch.drawn["refused_on_interval"], lateness)
                continue
            write_closed_loop(staging / f"{split}_{interval:g}s.npz", words.spec,
                              executor_params_sha256=params_sha256(params), row_interval_s=interval,
                              start_row=closed_loop.start_row(interval),
                              signal_index=[batch.indices[j] for j, _ in kept],
                              first_row=[batch.sentences[j].first_row for j, _ in kept],
                              grids=[r.grid for _, r in kept], corrections=[r.correction for _, r in kept],
                              states=[r.states for _, r in kept], lateral_m=[r.lateral_m for _, r in kept],
                              vertical_m=[r.vertical_m for _, r in kept],
                              uncorrectable=[r.uncorrectable for _, r in kept],
                              observed_row=[r.observed_row for _, r in kept],
                              matched_row=[r.matched_row for _, r in kept],
                              timed_out=[r.timed_out for _, r in kept])
            numbers = summarise(results, drawn.description["excluded"], batch.drawn["refused_on_interval"], lateness)
            summary["splits"][split]["intervals"][f"{interval:g}"] = numbers
            print(f"{split} {interval:g} s: {numbers['sentences']} sentences, corrections "
                  f"{numbers['correction_words']}, without a sentence {numbers['without_a_sentence']}, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    write_json_atomic(staging / "summary.json", summary)
    closed_loop.write_reference(instructions, params, words, args.row_interval_s, git=git,
                                target=staging / closed_loop.CONFORMANCE)
    staging.rename(target)
    checked = closed_loop.check(instructions, params, words, git=git)
    if not checked.passed:
        raise SystemExit(f"the code that wrote the reference reads it otherwise: {checked.mismatches}")
    print(f"→ {target}; passed → {closed_loop.write_passed(instructions, checked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
