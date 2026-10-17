#!/usr/bin/env python
"""Why does the point-mass baseline fail 77 of 500 flights, and how well does an executor follow the words it hears?

Part 1 (no flying): the baseline's failing flights by gate, against what the OBSERVED flight scores in the same
evaluation (observed_verdicts_500.json), and against the gate's limit.
Part 2: HEADING-FOLLOWING ERROR of three executors on the same flights. A heading word said at row i says the track the
aircraft has L = 4 s later (row i + 2). At each 2 s row j the track the words imply is the latest heading word said at
row <= j − 2 (the executor's own target). The error is track_j − implied_j. The observed aircraft has no such error by
construction (the labeller read the words off its track) apart from the 5-degree grid; the spec's own tolerance is 4.5°.
Executors: baseline = point-mass plant + the artefact's sentences; old executor on the lag plant (words regenerated);
new executor (lag_executor.py) on the lag plant (words regenerated).
Only words, states and the flown track are read; nothing is flown beyond the three regenerations.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import archive_paths   # noqa: E402  (README §2)
sys.path.insert(0, str(HERE))
import compare_plants as cp   # noqa: E402
import lag_executor as le     # noqa: E402
from ts_transformer.autopilot import closed_loop, replay                                  # noqa: E402
from ts_transformer.autopilot import executor as executor_module                          # noqa: E402
from ts_transformer.autopilot import flights as flights_module                            # noqa: E402
from ts_transformer.autopilot.plant import Plant                                          # noqa: E402
from ts_transformer.autopilot.start import start_row                                      # noqa: E402
from ts_transformer.instructions.artefact import closed_loop_sentences                    # noqa: E402
from ts_transformer.instructions.labeller.records import Refused                          # noqa: E402
from ts_transformer.instructions.words import HEADING                                     # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                              # noqa: E402

ROW_S, LEAD_ROWS = 2.0, 2     # heading_lead_s = 4 s = 2 rows of 2 s
TOL_DEG = 4.5
TRACK = 3                     # STATE_COLUMNS index of track_deg


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def heading_error(sentence, geometry, words) -> np.ndarray:
    """track − implied track at every flown row from the first predicted step + L; NaN before the first word's deadline."""
    start = start_row(2.0)
    states = sentence.rows.states
    heard = {ins.row: ins.info["target_deg"] for ins in replay.instructions_of(sentence.rows.grid, geometry, words)
             if ins.column == HEADING}
    implied = np.full(len(states), np.nan)
    current = np.nan
    for j in range(len(states)):
        i = j - start - LEAD_ROWS                     # the row whose word is due at state row j
        if i in heard:
            current = heard[i]
        implied[j] = current
    flown = np.arange(len(states)) >= start + LEAD_ROWS
    err = wrap180(states[:, TRACK] - implied)
    err[~flown] = np.nan
    return err


def summarize(label, errors, strata):
    out = {}
    for group in ("all", "straight-in", "vectored"):
        e = np.concatenate([x for x, s in zip(errors, strata) if (group == "all" or s == group) and x.size] or [np.empty(0)])
        e = np.abs(e[np.isfinite(e)])
        if e.size:
            out[group] = {"rows": int(e.size), "p50": float(np.percentile(e, 50)), "p90": float(np.percentile(e, 90)),
                          "p99": float(np.percentile(e, 99)), "within_4p5": float((e <= TOL_DEG).mean()),
                          "within_2p5": float((e <= 2.5).mean())}
    print(f"{label:34s}", "  ".join(f"{g}: p50 {v['p50']:.1f} p90 {v['p90']:.1f} p99 {v['p99']:.1f} in-4.5° {v['within_4p5']*100:.1f}%"
                                   for g, v in out.items()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-airport", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run", type=Path, required=True, help="the 100-per-airport compare_plants run (baseline records)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    flights_module.arrival_manifest_path = lambda airport: arrival_manifest_path(airport, cp.HARVEST)
    cpu = torch.device("cpu")
    started = time.perf_counter()

    # ---------------- part 1: the baseline's failures against the observed flights
    rows = {r["flight_key"]: r for r in json.loads((args.run / "rows.json").read_text())["point-mass"]}
    observed = json.loads((archive_paths.RESULTS / "observed_verdicts_500.json").read_text())
    ev = {}
    for a in ["KMSY", "KRDU", "KSJC", "KSMF", "KSTL"]:
        rep = json.loads((args.run / "point-mass/records" / a / "evaluation_report.json").read_text())
        ev.update({t["flight_key"]: t for t in rep["trajectories"]})
    fails = [k for k, t in ev.items() if t["verdict"] == "fail"]
    part1 = {"failed": len(fails)}
    pair = Counter((observed[k]["verdict"], ev[k]["verdict"]) for k in ev)
    part1["observed_vs_replay_verdict"] = {f"{a}->{b}": n for (a, b), n in pair.items()}
    gates = {}
    for gate, field, limit in (("lateral", "cross_track_m", None), ("vertical", "vertical_m", None), ("speed", "speed_margin_ms", None)):
        mine = [k for k in fails if gate in ev[k]["violations"]]
        both = [k for k in mine if gate in (observed[k]["violations"] or [])]
        gates[gate] = {"replay_fails": len(mine), "observed_also_fails": len(both),
                       "observed_passes_this_gate": sum(observed[k][f"{gate}_result"] == "pass" for k in mine),
                       "observed_indeterminate": sum(observed[k][f"{gate}_result"] == "indeterminate" for k in mine)}
    # how far past the limit
    over = []
    for k in fails:
        if "lateral" in ev[k]["violations"]:
            over.append(abs(ev[k]["cross_track_m"]) - ev[k]["bounds"]["lateral_m"])
    gates["lateral"]["miss_beyond_limit_m"] = {q: float(np.percentile(over, q)) for q in (10, 50, 90, 100)}
    sp = [ev[k]["bounds"]["speed_lower_ms"] - ev[k]["crossing_speed_ms"] for k in fails if "speed" in ev[k]["violations"]]
    gates["speed"]["below_lower_bound_ms"] = {q: float(np.percentile(sp, q)) for q in (10, 50, 90, 100)}
    part1["gates"] = gates
    # the observed flight's own lateral offset at the threshold, for all 500 and for the replay-failing ones
    obs_x = [abs(observed[k]["cross_track_m"]) for k in ev if observed[k].get("cross_track_m") is not None]
    rep_x = [abs(ev[k]["cross_track_m"]) for k in ev if ev[k].get("cross_track_m") is not None]
    part1["cross_track_at_threshold_m"] = {"observed": {q: float(np.percentile(obs_x, q)) for q in (50, 90, 95)},
                                           "replay": {q: float(np.percentile(rep_x, q)) for q in (50, 90, 95)}}
    print(json.dumps(part1, indent=1))

    # ---------------- part 2: heading-following error
    params, record, words = replay.open_spec(cp.EXECUTOR, cp.INSTRUCTIONS)
    drawn = replay.draw(cp.INSTRUCTIONS, "train", words.spec, words, per_airport=args.per_airport, seed=args.seed,
                        groups=(replay.OWN, replay.STAND_IN), row_interval_s=2.0)
    stored = closed_loop_sentences(cp.INSTRUCTIONS, "train", 2.0, words.spec)
    geom = {i: drawn.geometries[j] for j, i in enumerate(drawn.indices)}
    strata = {i: rows[drawn.series[j].scenario.source["flight_key"]]["stratum"] for j, i in enumerate(drawn.indices)}
    arms = {}

    def regenerate():
        out = {}
        for i, r in zip(drawn.indices, closed_loop.read_chunked(drawn, params, words, chunk=250, device=cpu)):
            if not isinstance(r, Refused):
                out[i] = r
        return out

    arms["baseline: point-mass plant, artefact sentences"] = {i: stored[i] for i in drawn.indices if i in stored}
    orig_exec, orig_plant = executor_module.Executor, Plant
    t, b, l = cc_taus = (1.5, 2.0, 0.8)
    from compare_compensation import make_plant                     # noqa: E402
    executor_module.Plant = make_plant((t, b, l), (t, b, l), (False, False, False), 32.0)
    arms["old executor, lag plant (words regenerated)"] = regenerate()
    executor_module.Plant = orig_plant
    le.install((t, b, l), (t, b, l))
    arms["new executor, lag plant (words regenerated)"] = regenerate()
    le.uninstall(orig_exec, orig_plant)
    print(f"regenerated, {time.perf_counter() - started:.0f}s")

    part2 = {}
    for label, sentences in arms.items():
        errs = [heading_error(s, geom[i], words) for i, s in sentences.items()]
        part2[label] = summarize(label, errs, [strata[i] for i in sentences])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diagnosis.json").write_text(json.dumps({"part1": part1, "heading_following": part2}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
