#!/usr/bin/env python
"""V12 on the Mac (the user, 2026-10-12): the three leads, one column at a time, on 1000 train flights (200 per airport,
the replay's seeded sample over the stored sentences' flights: a superset of `new_design_eval`'s 750).

Base = heading 4 s, angle 0, speed 0 (the labelling before D7/D8). Each arm changes ONE lead to 0/2/4/6/8 s (the base's
own value is the base arm). Per arm and row interval (2, 4 s): the labeller's reading under that spec (in memory, as
`new_design_eval`), the closed loop, the replay graded by the evaluation module (per-flight verdicts kept for paired
comparison), and the V12 runner's own readouts (`experiments/lead_ablation.py`: the column's envelopes on the observed
tracks, corrections per sentence, the error around the column's observed words at −1..+4 rows).
Executor = the two-tier2 worktree's (design draft 5), parameters as `new_design_eval.params_of`. Nothing in the
repository is edited; nothing is written outside --out; each cell's evaluation records are deleted once graded
(the per-flight verdicts stay in results.json).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import new_design_eval as nde   # noqa: E402  (puts the worktree on sys.path)

import numpy as np   # noqa: E402
import torch         # noqa: E402

from ts_transformer.autopilot import closed_loop, replay                                  # noqa: E402
from ts_transformer.autopilot import flights as flights_module                            # noqa: E402
from ts_transformer.experiments import executor_replay as er                              # noqa: E402
from ts_transformer.experiments import lead_ablation as la                                # noqa: E402
from ts_transformer.instructions.artefact import ClosedLoopSentence                       # noqa: E402
from ts_transformer.instructions.words import COLUMNS, Words                              # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                               # noqa: E402

BASE = {"heading": 4.0, "angle": 0.0, "speed": 0.0}
VALUES = (0.0, 2.0, 4.0, 6.0, 8.0)


def arms() -> list[tuple[str, dict[str, float], str | None]]:
    out = [("base", dict(BASE), None)]
    for column in ("heading", "angle", "speed"):
        for value in VALUES:
            if value != BASE[column]:
                out.append((f"{column}{value:g}", {**BASE, column: value}, column))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-airport", type=int, default=200)
    ap.add_argument("--row-interval-s", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--only", nargs="*", default=None, help="arm names to run (default: all)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert args.out.is_absolute() and not args.out.exists()
    args.out.mkdir(parents=True)
    flights_module.arrival_manifest_path = lambda airport: arrival_manifest_path(airport, nde.HARVEST)
    cpu, started = torch.device("cpu"), time.perf_counter()
    params = nde.params_of()
    with np.load(nde.INSTRUCTIONS / "sentences_train.npz") as data:
        labelled = [int(i) for i in data["signal_index"]]
    drawn = replay.draw_flights(nde.INSTRUCTIONS, "train", labelled, per_airport=args.per_airport, seed=1337,
                                groups=(replay.OWN, replay.STAND_IN))
    print(f"{len(drawn.signals)} flights drawn {drawn.description}, {time.perf_counter() - started:.0f}s", flush=True)
    results = []
    for name, leads, column in arms():
        if args.only is not None and name not in args.only:
            continue
        spec = nde.spec_of(leads["heading"], leads["angle"], leads["speed"])
        words = Words(spec)
        keep, readings, refused = la.read_again(drawn, spec, words)
        envelopes = {c: la.envelopes(readings, c) for c in ("heading", "angle", "speed")}
        for interval in args.row_interval_s:
            batch = replay.batch_of(drawn, keep, readings, interval, words)
            raw = closed_loop.read_chunked(batch, params, words, chunk=args.chunk, device=cpu)
            sentences = {i: r for i, r in zip(batch.indices, raw) if isinstance(r, ClosedLoopSentence)}
            errors = {c: {o: [] for o in la.OFFSETS} for c in ("heading", "angle", "speed")}
            for j, r in enumerate(raw):
                if isinstance(r, ClosedLoopSentence):
                    for c in errors:
                        for o, v in la.word_errors(r, batch.observed[j].ground_speed_mps, c).items():
                            errors[c][o] += v
            replayed, _ = closed_loop.replay_batch(batch, sentences, words)
            by_airport = defaultdict(list)
            for j, f in enumerate(replayed.signals):
                by_airport[f.airport].append(j)
            reports = []
            label = f"{name}_d{interval:g}"
            for airport, members in sorted(by_airport.items()):
                _, graded = er.fly_airport(replayed, members, params, words, chunk=args.chunk, device=cpu,
                                           records=args.out / "records" / label / airport, split="train",
                                           checkpoint=label, extra_summary={"arm": name, "row_interval_s": interval},
                                           per_flight=lambda part, flown, verdicts: [{} for _ in verdicts])
                reports += graded["trajectories"]
            corrections = Counter()
            for s in sentences.values():
                for c, n in zip(COLUMNS, s.rows.correction.sum(axis=0).tolist()):
                    corrections[c] += n
            n = max(len(sentences), 1)
            res = {"arm": name, "column": column, "leads_s": leads, "row_interval_s": interval,
                   "drawn": len(drawn.signals), "labeller_refused": dict(refused), "graded": len(reports),
                   "eval_pass": sum(t["verdict"] == "pass" for t in reports),
                   "fails": dict(Counter(g for t in reports if t["verdict"] == "fail" for g in t["violations"])),
                   "landed": sum(s.withheld.outcome == "landed" for s in sentences.values()),
                   "left_the_path_gt300m": int(sum(float(np.max(np.abs(s.withheld.lateral_m))) > 300
                                                   for s in sentences.values())),
                   "corrections_per_sentence": {k: v / n for k, v in corrections.items()},
                   "envelopes": envelopes,
                   "after_words": {c: la.error_table(errors[c], la.tolerance_of(spec, c)) for c in errors},
                   "verdicts": {t["flight_key"]: {"verdict": t["verdict"], "violations": t["violations"]}
                                for t in reports}}
            results.append(res)
            shutil.rmtree(args.out / "records" / label)   # ~170 MB a cell; the verdicts are kept above (disk 99 % full)
            (args.out / "results.json").write_text(json.dumps(results))
            print(f"  {label:12s} eval {res['eval_pass']}/{res['graded']} fails {res['fails']} landed {res['landed']} "
                  f">300m {res['left_the_path_gt300m']} corr {({k: round(v, 1) for k, v in res['corrections_per_sentence'].items() if v})}, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    print(f"→ {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
