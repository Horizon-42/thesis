#!/usr/bin/env python
"""Diagnosis of `new_design_eval` (the two-tier2 executor, 750 train flights): why more lateral failures at Δ 2 s than
the official executor, and why the angle lead adds vertical failures.

Lateral: one change of the lateral law undone at a time, patched into the worktree's modules in this process only:
  floor2   the time left floored at 2 Δt (the old law), not Δt
  cap32    the bank no more than the vocabulary's 32° (the old cap), besides φ_phys
  rmax     the track rate no more than the vocabulary's turn-rate limit (the old cap)
  p5, p30  the roll rate p 5 or 30 deg/s, not 15
Each arm flies arm A's labelling (leads 4/0/0) at --row-interval-s. Vertical: the per-flight evaluation records of
`new_design_eval` arms A and B1 read side by side (no flying).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
import archive_paths   # noqa: E402  (README §2)
sys.path.insert(0, str(HERE))
import new_design_eval as nde   # noqa: E402  (puts the worktree on sys.path)

import numpy as np   # noqa: E402
import torch         # noqa: E402

from ts_transformer.autopilot import closed_loop, inverse, lateral, replay                 # noqa: E402
from ts_transformer.autopilot import flights as flights_module                             # noqa: E402
from ts_transformer.experiments import executor_replay as er                               # noqa: E402
from ts_transformer.instructions.labeller.read import read_flight                          # noqa: E402
from ts_transformer.instructions.labeller.records import Refused                           # noqa: E402
from ts_transformer.instructions.words import Words                                        # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                               # noqa: E402

ORIGINAL_WORD_RATE, ORIGINAL_BANK_LIMIT = lateral.word_rate, inverse.bank_limit_rad


def patch(arm: str, spec) -> dict:
    """Patch the modules for ``arm``; the params' changes returned."""
    lateral.word_rate, inverse.bank_limit_rad = ORIGINAL_WORD_RATE, ORIGINAL_BANK_LIMIT
    if arm == "floor2":
        lateral.word_rate = lambda error, to_go, params: error / to_go.clamp(min=2.0 * params.cycle_s)
    elif arm == "cap32":
        lateral.word_rate = ORIGINAL_WORD_RATE
        inverse.bank_limit_rad = lambda state, ceiling: ORIGINAL_BANK_LIMIT(state, ceiling).clamp(
            max=math.radians(spec.turn_bank_max_deg))
    elif arm == "rmax":
        lateral.word_rate = lambda error, to_go, params: (error / to_go.clamp(min=params.cycle_s)).clamp(
            -spec.turn_rate_max_deg_s, spec.turn_rate_max_deg_s)
    # noise1..3: a change of no physical meaning (1e-4 s of a time constant), the closed loop's own spread
    return {"p5": {"bank_rate_deg_s": 5.0}, "p30": {"bank_rate_deg_s": 30.0}, "base": {},
            "noise1": {"thrust_time_constant_s": 1.5001}, "noise2": {"bank_time_constant_s": 2.0001},
            "noise3": {"load_time_constant_s": 0.8001}}.get(arm, {})


def fly(drawn, keep, readings, interval, params, words, out: Path, label: str, chunk: int):
    cpu = torch.device("cpu")
    batch = replay.batch_of(drawn, keep, readings, interval, words)
    sentences = {i: r for i, r in zip(batch.indices, closed_loop.read_chunked(batch, params, words, chunk=chunk,
                                                                              device=cpu))
                 if not isinstance(r, Refused)}
    replayed, _ = closed_loop.replay_batch(batch, sentences, words)
    by_airport: dict[str, list[int]] = {}
    for j, f in enumerate(replayed.signals):
        by_airport.setdefault(f.airport, []).append(j)
    reports = []
    for airport, members in sorted(by_airport.items()):
        _, graded = er.fly_airport(replayed, members, params, words, chunk=chunk, device=cpu,
                                   records=out / label / "records" / airport, split="train", checkpoint=label,
                                   extra_summary={"arm": label},
                                   per_flight=lambda part, flown, verdicts: [{} for _ in verdicts])
        reports += graded["trajectories"]
    return reports


def describe(reports) -> dict:
    fails = Counter(g for t in reports if t["verdict"] == "fail" for g in t["violations"])
    xt = np.array([abs(t["cross_track_m"]) for t in reports if t.get("cross_track_m") is not None])
    return {"pass": sum(t["verdict"] == "pass" for t in reports), "fails": dict(fails),
            "cross_track_p50": float(np.percentile(xt, 50)), "cross_track_p90": float(np.percentile(xt, 90)),
            "cross_track_22.86_to_30": int(((xt > 22.86) & (xt <= 30.0)).sum()), "cross_track_over_30": int((xt > 30).sum())}


def vertical_side_by_side(base: Path) -> dict:
    out = {}
    for d in ("d2", "d4"):
        rows = {}
        for idx in (0, 1):
            reps = [t for f in sorted(glob.glob(str(base / d / f"arm{idx}_{d}" / "records" / "*" /
                                                    "evaluation_report.json")))
                    for t in json.load(open(f))["trajectories"]]
            rows[idx] = {t["flight_key"]: t for t in reps}
        cell = {}
        for idx in (0, 1):
            fails = [t for t in rows[idx].values() if "vertical" in t["violations"]]
            cell[f"arm{idx}"] = {"vertical_fails": len(fails),
                                 "high": sum(t["vertical_m"] > 0 for t in fails), "low": sum(t["vertical_m"] < 0 for t in fails),
                                 "vertical_m_p10_p50_p90_all": [float(np.percentile([t["vertical_m"] for t in rows[idx].values()
                                                                                     if t.get("vertical_m") is not None], q))
                                                                for q in (10, 50, 90)]}
        only_b1 = [k for k, t in rows[1].items() if "vertical" in t["violations"] and "vertical" not in rows[0][k]["violations"]]
        cell["fail_in_B1_not_A"] = [{"flight": k, "B1": round(rows[1][k]["vertical_m"], 1), "A": round(rows[0][k]["vertical_m"], 1)}
                                    for k in only_b1]
        out[d] = cell
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-airport", type=int, default=150)
    ap.add_argument("--row-interval-s", type=float, default=2.0)
    ap.add_argument("--arms", nargs="+", default=["floor2", "cap32", "rmax", "p5", "p30"])
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    result = {"vertical": vertical_side_by_side(archive_paths.RESULTS / "new_design_eval")}
    (args.out / "results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["vertical"], indent=1)[:3000], flush=True)
    flights_module.arrival_manifest_path = lambda airport: arrival_manifest_path(airport, nde.HARVEST)
    with np.load(nde.INSTRUCTIONS / "sentences_train.npz") as data:
        labelled = [int(i) for i in data["signal_index"]]
    drawn = replay.draw_flights(nde.INSTRUCTIONS, "train", labelled, per_airport=args.per_airport, seed=1337,
                                groups=(replay.OWN, replay.STAND_IN))
    spec = nde.spec_of(4.0, 0.0, 0.0)
    words = Words(spec)
    keep, readings = [], []
    for k, flight in enumerate(drawn.signals):
        try:
            readings.append(read_flight(flight, drawn.geometries[flight.airport], spec, words))
            keep.append(k)
        except Refused:
            pass
    result["lateral"] = {}
    for arm in args.arms:
        params = replace(nde.params_of(), **patch(arm, spec))
        reports = fly(drawn, keep, readings, args.row_interval_s, params, words, args.out, arm, args.chunk)
        result["lateral"][arm] = describe(reports)
        (args.out / "results.json").write_text(json.dumps(result, indent=1))
        print(f"  {arm:7s} {result['lateral'][arm]}, {time.perf_counter() - started:.0f}s", flush=True)
    patch("none", spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
