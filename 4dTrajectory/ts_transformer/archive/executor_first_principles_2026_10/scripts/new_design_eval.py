#!/usr/bin/env python
"""The first-principles executor (design draft 5, branch two-tier2) on the same 750 train flights as `new_vs_current`.

Imports the PACKAGES OF THE two-tier2 WORKTREE (its labeller with the angle and speed leads, its executor), never the main
checkout's. Nothing in the repository is edited, nothing is written outside --out.

The v12 artefact's spec is instruction-v6 and the v17 executor spec is v10: the new code refuses both by name. So, in
memory only:
- the vocabulary spec = v12's spec.json values + `angle_lead_s` / `speed_lead_s` of the arm, as instruction-v7;
- the executor parameters = cycle 1 s, p 15 deg/s (D2), time constants 1.5 / 2.0 / 0.8 s (assumed), the v17 spec's
  timeout factor and start rule;
- the flights = `replay.draw_flights` on the stored sentences' signal indices (the same seeded sample as `replay.draw`),
  each READ AGAIN by the labeller under the arm's spec (`read_flight`), read in closed loop at the row interval
  (`closed_loop.read_chunked`), its closed-loop sentence flown again and graded by the evaluation module
  (`executor_replay.fly_airport`, the formal replay's path).
The official baseline is not flown here: `new_vs_current` (Δ 2 s) and `new_vs_current_4s` hold it for these flights.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import archive_paths   # noqa: E402  (README §2)

WORKTREE = archive_paths.REPO   # the repository this archive sits in (on the Mac: the two-tier2 worktree)
sys.path[:0] = [str(WORKTREE / "4dTrajectory"), str(WORKTREE)]
DATA = archive_paths.POOLED
INSTRUCTIONS = DATA / "instruction_language/v12_20261005"
EXECUTOR = DATA / "executor/v17_20261005"
HARVEST = archive_paths.HARVEST

import numpy as np   # noqa: E402
import torch         # noqa: E402

import ts_transformer   # noqa: E402
assert str(WORKTREE) in ts_transformer.__file__, ts_transformer.__file__
from ts_transformer.autopilot import closed_loop, replay                                  # noqa: E402
from ts_transformer.autopilot import flights as flights_module                            # noqa: E402
from ts_transformer.autopilot.params import ExecutorParams                                # noqa: E402
from ts_transformer.experiments import executor_replay as er                              # noqa: E402
from ts_transformer.instructions.labeller.read import read_flight                         # noqa: E402
from ts_transformer.instructions.labeller.records import Refused                          # noqa: E402
from ts_transformer.instructions.spec import READING_RULE, VocabularySpec                 # noqa: E402
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words                   # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                              # noqa: E402

# (name, heading lead, angle lead, speed lead): one column changed at a time (the user, 2026-10-12)
ARMS = [("A: leads 4/0/0 (the labelling before the leads)", 4.0, 0.0, 0.0),
        ("B1: angle lead 4 s", 4.0, 4.0, 0.0), ("B2: angle lead 6 s", 4.0, 6.0, 0.0),
        ("C1: speed lead 4 s", 4.0, 0.0, 4.0), ("C2: speed lead 6 s", 4.0, 0.0, 6.0)]


def spec_of(heading: float, angle: float, speed: float) -> VocabularySpec:
    data = json.loads((INSTRUCTIONS / "spec.json").read_text())
    values = data["spec"] if "spec" in data else data
    values = {**values, "reading_rule": READING_RULE, "heading_lead_s": heading, "angle_lead_s": angle,
              "speed_lead_s": speed}
    return VocabularySpec.from_dict(values)


def params_of() -> ExecutorParams:
    data = json.loads((EXECUTOR / "spec.json").read_text())
    old = data["params"] if "params" in data else data
    return ExecutorParams(cycle_s=1.0, bank_rate_deg_s=15.0, thrust_time_constant_s=1.5, bank_time_constant_s=2.0,
                          load_time_constant_s=0.8, timeout_factor=float(old["timeout_factor"]),
                          start_rule=str(old["start_rule"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-airport", type=int, default=150)
    ap.add_argument("--row-interval-s", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    flights_module.arrival_manifest_path = lambda airport: arrival_manifest_path(airport, HARVEST)
    cpu, started = torch.device("cpu"), time.perf_counter()
    params = params_of()
    with np.load(INSTRUCTIONS / "sentences_train.npz") as data:
        labelled = [int(i) for i in data["signal_index"]]
    drawn = replay.draw_flights(INSTRUCTIONS, "train", labelled, per_airport=args.per_airport, seed=args.seed,
                                groups=(replay.OWN, replay.STAND_IN))
    print(f"{len(drawn.signals)} flights drawn, {time.perf_counter() - started:.0f}s", flush=True)
    results = []
    for index, (name, heading, angle, speed) in enumerate(ARMS):
        if args.only is not None and index not in args.only:
            continue
        spec = spec_of(heading, angle, speed)
        words = Words(spec)
        keep, readings, refused = [], [], Counter()
        for k, flight in enumerate(drawn.signals):
            try:
                readings.append(read_flight(flight, drawn.geometries[flight.airport], spec, words))
                keep.append(k)
            except Refused as refusal:
                refused[refusal.reason] += 1
        for interval in args.row_interval_s:
            batch = replay.batch_of(drawn, keep, readings, interval, words)
            sentences, closed_refused = {}, Counter()
            for i, r in zip(batch.indices, closed_loop.read_chunked(batch, params, words, chunk=args.chunk, device=cpu)):
                if isinstance(r, Refused):
                    closed_refused[r.reason] += 1
                else:
                    sentences[i] = r
            replayed, _ = closed_loop.replay_batch(batch, sentences, words)
            closed = er.closed_loop_columns(sentences, spec.step_s)
            along = er.along_columns(words)

            def per_flight(part, flown, verdicts):
                return [{**a, **c} for a, c in zip(along(part, flown, verdicts), closed(part, flown, verdicts))]
            by_airport = defaultdict(list)
            for j, f in enumerate(replayed.signals):
                by_airport[f.airport].append(j)
            rows, reports = [], []
            label = f"arm{index}_d{interval:g}"
            for airport, members in sorted(by_airport.items()):
                r, graded = er.fly_airport(replayed, members, params, words, chunk=args.chunk, device=cpu,
                                           records=args.out / label / "records" / airport, split="train",
                                           checkpoint=label, extra_summary={"arm": name, "row_interval_s": interval},
                                           per_flight=per_flight)
                rows += r
                reports += graded["trajectories"]
            outcomes = Counter(r["outcome"] for r in rows)
            passed = lambda field: sum(t.get(field) == "pass" for t in reports)
            corrections = Counter()
            for s in sentences.values():
                for column, n in zip(COLUMNS, s.rows.correction.sum(axis=0).tolist()):
                    corrections[column] += n
            n = max(len(sentences), 1)
            lateral = [float(np.max(np.abs(s.withheld.lateral_m))) for s in sentences.values()]
            res = {"arm": name, "leads_s": [heading, angle, speed], "row_interval_s": interval,
                   "drawn": len(drawn.signals), "labeller_refused": dict(refused), "closed_loop_refused": dict(closed_refused),
                   "flights": len(rows), "landed": outcomes["landed"], "outcomes": dict(outcomes),
                   "eval_pass": sum(t["verdict"] == "pass" for t in reports), "graded": len(reports),
                   "lateral_pass": passed("lateral_result"), "vertical_pass": passed("vertical_result"),
                   "speed_pass": passed("speed_result"),
                   "corrections_per_sentence": {k: v / n for k, v in corrections.items()},
                   "said_words_per_sentence": float(sum(int((s.rows.grid != UNCHANGED).sum())
                                                        for s in sentences.values()) / n),
                   "left_the_path_gt300m": int(sum(x > 300 for x in lateral))}
            results.append(res)
            (args.out / "results.json").write_text(json.dumps(results, indent=1))
            print(f"  [{index}] {name:42s} Δ {interval:g}: refused {sum(refused.values())}+{sum(closed_refused.values())}, "
                  f"landed {res['landed']}/{res['flights']}, eval {res['eval_pass']}/{res['graded']} (lat "
                  f"{res['lateral_pass']}, vert {res['vertical_pass']}, speed {res['speed_pass']}), corr/sent "
                  f"{ {k: round(v, 1) for k, v in res['corrections_per_sentence'].items() if v} }, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    print(f"→ {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
