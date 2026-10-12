#!/usr/bin/env python
"""Two checks of `new_design_eval` (the two-tier2 executor, the same 750 train flights).

--check roll: p = 5 and p = 15 deg/s at the row intervals given, labelling A (leads 4/0/0): the evaluation pass and its
  gates, and from the closed-loop sentences the flights that left their observed path by more than 300 m (largest
  |e_y|), the largest |e_y| p50 / p90 and the correction words per sentence.
--check vertical: labellings A (angle lead 0) and B1 (angle lead 4 s) at Δ 2 s, the final descent ("no level-off" in
  force) of every flight: its angle corrections (how many, when the last one is said, where the steeper ones arrive)
  and e_h (flown − observed height at the matched point) at the end of the observed path; per flight for the ones that
  fail the vertical gate in B1 but not in A (`new_design_eval` records).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
import archive_paths   # noqa: E402  (README §2)
sys.path.insert(0, str(HERE))
import diagnose_new_design as dnd   # noqa: E402
import new_design_eval as nde       # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

from ts_transformer.autopilot import closed_loop, replay                                   # noqa: E402
from ts_transformer.autopilot import flights as flights_module                             # noqa: E402
from ts_transformer.instructions.labeller.interval import in_force                         # noqa: E402
from ts_transformer.instructions.labeller.read import read_flight                          # noqa: E402
from ts_transformer.instructions.labeller.records import Refused                           # noqa: E402
from ts_transformer.instructions.words import ALTITUDE, ANGLE, COLUMNS, UNCHANGED, Words   # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                               # noqa: E402


def setup(per_airport: int):
    flights_module.arrival_manifest_path = lambda airport: arrival_manifest_path(airport, nde.HARVEST)
    with np.load(nde.INSTRUCTIONS / "sentences_train.npz") as data:
        labelled = [int(i) for i in data["signal_index"]]
    return replay.draw_flights(nde.INSTRUCTIONS, "train", labelled, per_airport=per_airport, seed=1337,
                               groups=(replay.OWN, replay.STAND_IN))


def read_all(drawn, spec):
    words = Words(spec)
    keep, readings = [], []
    for k, flight in enumerate(drawn.signals):
        try:
            readings.append(read_flight(flight, drawn.geometries[flight.airport], spec, words))
            keep.append(k)
        except Refused:
            pass
    return words, keep, readings


def closed(drawn, keep, readings, interval, params, words, chunk):
    batch = replay.batch_of(drawn, keep, readings, interval, words)
    results = closed_loop.read_chunked(batch, params, words, chunk=chunk, device=torch.device("cpu"))
    return batch, {i: r for i, r in zip(batch.indices, results) if not isinstance(r, Refused)}


def sentence_stats(sentences) -> dict:
    lateral = np.array([float(np.max(np.abs(s.withheld.lateral_m))) for s in sentences.values()])
    corrections = Counter()
    for s in sentences.values():
        for column, n in zip(COLUMNS, s.rows.correction.sum(axis=0).tolist()):
            corrections[column] += n
    n = len(sentences)
    return {"sentences": n, "left_the_path_gt300m": int((lateral > 300).sum()),
            "largest_lateral_p50": float(np.percentile(lateral, 50)), "largest_lateral_p90": float(np.percentile(lateral, 90)),
            "corrections_per_sentence": {k: v / n for k, v in corrections.items() if v}}


def check_roll(args, drawn) -> dict:
    spec = nde.spec_of(4.0, 0.0, 0.0)
    words, keep, readings = read_all(drawn, spec)
    out = {}
    for interval in args.row_interval_s:
        for p in (5.0, 15.0):
            label = f"p{p:g}_d{interval:g}"
            params = replace(nde.params_of(), bank_rate_deg_s=p)
            dnd.patch("none", spec)
            reports = dnd.fly(drawn, keep, readings, interval, params, words, args.out, label, args.chunk)
            _, sentences = closed(drawn, keep, readings, interval, params, words, args.chunk)
            out[label] = {**dnd.describe(reports), **sentence_stats(sentences),
                          "landed": sum(s.withheld.outcome == "landed" for s in sentences.values())}
            print(f"  {label}: {out[label]}", flush=True)
    return out


def final_descent(sentence, words) -> dict:
    rows, withheld = sentence.rows, sentence.withheld
    held = in_force(rows.grid)
    final = held[:, ALTITUDE] == words.altitude_no_level_off
    if not final.any():
        return {}
    first = int(np.argmax(final))
    angle = rows.grid[:, ANGLE]
    corr = [m for m in range(first, len(angle)) if rows.correction[m, ANGLE] and angle[m] != UNCHANGED]
    steeper = [m for m in corr if angle[m] > held[m - 1, ANGLE]]
    vm = withheld.vertical_m
    finite = np.flatnonzero(np.isfinite(vm))
    last = int(finite[-1]) if len(finite) else None
    return {"first_row": first, "rows": int(final.sum()), "corrections": len(corr), "steeper": len(steeper),
            "last_correction_row_from_end": None if not corr or last is None else last - corr[-1],
            "e_h_last": None if last is None else float(vm[last]),
            "e_h_max_in_final": float(np.nanmax(vm[first:])) if np.isfinite(vm[first:]).any() else None,
            "trace": [(m, int(angle[m]) if angle[m] != UNCHANGED else None, bool(rows.correction[m, ANGLE]),
                       None if not np.isfinite(vm[m]) else round(float(vm[m]), 1)) for m in range(first, len(angle))]}


def check_vertical(args, drawn) -> dict:
    keys_b1_only = []
    rows = {}
    for idx in (0, 1):
        reps = [t for f in glob.glob(str(archive_paths.RESULTS / "new_design_eval/d2" / f"arm{idx}_d2" / "records/*/evaluation_report.json"))
                for t in json.load(open(f))["trajectories"]]
        rows[idx] = {t["flight_key"]: t for t in reps}
    keys_b1_only = [k for k, t in rows[1].items() if "vertical" in t["violations"] and "vertical" not in rows[0][k]["violations"]]
    out = {"flights": keys_b1_only}
    params = nde.params_of()
    for label, angle_lead in (("A", 0.0), ("B1", 4.0)):
        spec = nde.spec_of(4.0, angle_lead, 0.0)
        words, keep, readings = read_all(drawn, spec)
        batch, sentences = closed(drawn, keep, readings, 2.0, params, words, args.chunk)
        key_of = {batch.indices[j]: s.scenario.source["flight_key"] for j, s in enumerate(batch.series)}
        finals = {key_of[i]: final_descent(s, words) for i, s in sentences.items()}
        finals = {k: v for k, v in finals.items() if v}
        e_last = np.array([v["e_h_last"] for v in finals.values() if v["e_h_last"] is not None])
        out[label] = {"final_descents": len(finals),
                      "corrections_per_final": float(np.mean([v["corrections"] for v in finals.values()])),
                      "steeper_per_final": float(np.mean([v["steeper"] for v in finals.values()])),
                      "e_h_last_p50_p90": [float(np.percentile(e_last, 50)), float(np.percentile(e_last, 90))],
                      "e_h_last_over_10": int((e_last > 10).sum()), "e_h_last_over_15": int((e_last > 15).sum()),
                      "per_flight": {k: {kk: vv for kk, vv in finals[k].items()} for k in keys_b1_only if k in finals}}
        print(f"  {label}: { {k: v for k, v in out[label].items() if k != 'per_flight'} }", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=("roll", "vertical"), required=True)
    ap.add_argument("--per-airport", type=int, default=150)
    ap.add_argument("--row-interval-s", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    drawn = setup(args.per_airport)
    result = check_roll(args, drawn) if args.check == "roll" else check_vertical(args, drawn)
    (args.out / "results.json").write_text(json.dumps(result, indent=1))
    print(f"→ {args.out / 'results.json'}, {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
