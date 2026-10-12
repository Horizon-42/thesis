#!/usr/bin/env python
"""Corrector thresholds that depend on the PROGRESS of the approach: official for the first 2/3, tight for the last 1/3.

Arms (same flights, same seed):
  OFFICIAL              original executor, point-mass plant (no lag), the artefact's thresholds
  official + new thr.   the same executor and plant, final-phase thresholds
  NEW exec + official thr.   `lag_executor` on the lag plant, the artefact's thresholds
  NEW exec + new thr.   `lag_executor` on the lag plant, final-phase thresholds

FINAL-PHASE THRESHOLDS. The final phase is the one the corrector already names (D66): "no level-off" in force, the
final descent told by the words. There
  vertical tolerance H_final = 7.5 m   (the artefact: 10 m; elsewhere H = 15 m unchanged)
  lateral threshold          = 15 m    (the artefact: 30 m for every phase; elsewhere 30 m unchanged)
and the release at half of each, as the corrector already does (7.5 m and 3.75 m).

THE 4 s WORDS FROM THE 2 s WORDS (the artefact's own rule, `labeller.interval.on_utc_grid`: one reading gives every
Δ, nothing is read again): after the closed loop at Δ = 2 s, each flight's words (the open-loop observed rows before the
first predicted step, then the closed loop's said rows with their corrections) are put on the 4 s UTC grid — each word
on the nearest 4 s row, the last of several in a row — and the 4 s sentence is flown from its first predicted step by the
same executor and graded. The corrections were decided against the 2 s flight; the 4 s sentence is not read again.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_compensation as cc   # noqa: E402
import compare_plants as cp         # noqa: E402
import lag_executor as le           # noqa: E402
from ts_transformer.autopilot import closed_loop, replay                                  # noqa: E402
from ts_transformer.autopilot import executor as executor_module                          # noqa: E402
from ts_transformer.autopilot import flights as flights_module                            # noqa: E402
from ts_transformer.autopilot.plant import Plant                                          # noqa: E402
from ts_transformer.autopilot.spec import params_sha256                                   # noqa: E402
from ts_transformer.autopilot.start import start_row                                      # noqa: E402
from ts_transformer.experiments import executor_replay as er                              # noqa: E402
from ts_transformer.instructions.artefact import closed_loop_path                         # noqa: E402
from ts_transformer.instructions.labeller.interval import in_force, on_utc_grid           # noqa: E402
from ts_transformer.instructions.labeller.records import Refused                          # noqa: E402
from ts_transformer.instructions.words import (                                           # noqa: E402
    ALTITUDE, COLUMNS, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words,
)
from ts_transformer.repo_layout import arrival_manifest_path                              # noqa: E402

ORIGINAL_EXECUTOR, ORIGINAL_PLANT = executor_module.Executor, Plant
ORIGINAL_CORRECTOR = closed_loop.Corrector


_PHASE = {"tight": False}
TIGHT_LATERAL_M, TIGHT_FINAL_VERTICAL_M, TIGHT_FROM = 15.0, 7.5, 2.0 / 3.0
_ORIGINAL_TOLERANCE = closed_loop.vertical_tolerance_m


def _phase_tolerance(words, altitude):
    """`vertical_tolerance_m` with H_final replaced by the tight value while the corrector is in its last third."""
    spec = words.spec
    final = TIGHT_FINAL_VERTICAL_M if _PHASE["tight"] else spec.closed_loop_final_vertical_m
    return np.where(np.asarray(altitude) == words.altitude_no_level_off, final, spec.closed_loop_vertical_m)


class ProgressCorrector(ORIGINAL_CORRECTOR):
    """`Corrector` with the tight thresholds from ``TIGHT_FROM`` of the approach on. Progress is the matched point's
    share of the observed rows between the first predicted step and the end of the observed sentence (the labeller
    reads the whole observed path; the executor never sees this number). ``ENABLED`` False: the artefact's thresholds."""

    ENABLED = False

    def row(self, matched_row, lateral_m, vertical_m, height_m, *, holding, past_end):
        tight = False
        if self.ENABLED and self.said is not None:
            last = len(self.observed) - 1
            progress = (float(matched_row) - self.start) / max(last - self.start, 1)
            tight = progress >= TIGHT_FROM
        _PHASE["tight"] = tight
        self.lateral_m = TIGHT_LATERAL_M if tight else self.words.spec.closed_loop_lateral_m
        self.phase_tight_rows = getattr(self, "phase_tight_rows", 0) + int(tight)
        return super().row(matched_row, lateral_m, vertical_m, height_m, holding=holding, past_end=past_end)


def arms() -> list[dict]:
    t, b, l = cc.DEFAULT_TAUS
    base = {"true": (t, b, l), "assumed": (t, b, l)}
    return [
        {**base, "name": "OFFICIAL: original executor, point-mass plant (no lag), official thresholds everywhere",
         "pm": True, "split": False},
        {**base, "name": "  original executor, point-mass plant, first 2/3 official + last 1/3 tight (15 m / 7.5 m)",
         "pm": True, "split": True},
        {**base, "name": "  NEW executor, lag plant, official thresholds everywhere", "pm": False, "split": False},
        {**base, "name": "NEWEST: NEW executor, lag plant, first 2/3 official + last 1/3 tight (15 m / 7.5 m)",
         "pm": False, "split": True},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-airport", type=int, default=150)
    ap.add_argument("--row-interval-s", type=float, default=2.0)
    ap.add_argument("--merge-to-s", type=float, default=None, help="then put the words on this coarser grid and fly again")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--p-new", type=float, default=None, help="roll rate limit deg/s for the NEW executor arms")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    flights_module.arrival_manifest_path = lambda airport: arrival_manifest_path(airport, cp.HARVEST)
    cpu = torch.device("cpu")
    started = time.perf_counter()
    params, record, base_words = replay.open_spec(cp.EXECUTOR, cp.INSTRUCTIONS)
    spec = base_words.spec
    with np.load(closed_loop_path(cp.INSTRUCTIONS, "train", 2.0)) as data:
        assert str(data["executor_params_sha256"]) == params_sha256(params)
    kw = dict(per_airport=args.per_airport, seed=args.seed, groups=(replay.OWN, replay.STAND_IN))
    drawn = replay.draw(cp.INSTRUCTIONS, "train", spec, base_words, row_interval_s=args.row_interval_s, **kw)
    drawn_coarse = (replay.draw(cp.INSTRUCTIONS, "train", spec, base_words, row_interval_s=args.merge_to_s, **kw)
                    if args.merge_to_s else None)
    print(f"{len(drawn.indices)} flights at {args.row_interval_s:g} s"
          + (f", {len(drawn_coarse.indices)} at {args.merge_to_s:g} s" if drawn_coarse else "")
          + f", {time.perf_counter() - started:.0f}s", flush=True)
    along = er.along_columns(base_words)
    current = {"words": base_words, "params": params}

    def fly_and_grade(label, batch_in, sentences, reproduces, interval):
        words = current["words"]
        batch, _ = closed_loop.replay_batch(batch_in, sentences, words)
        closed = er.closed_loop_columns(sentences, spec.step_s) if reproduces else None

        def per_flight(part, flown, verdicts):
            more = closed(part, flown, verdicts) if closed else [{} for _ in verdicts]
            return [{**a, **c} for a, c in zip(along(part, flown, verdicts), more)]
        by_airport = defaultdict(list)
        for j, f in enumerate(batch.signals):
            by_airport[f.airport].append(j)
        rows, reports = [], []
        for airport, members in sorted(by_airport.items()):
            r, graded = er.fly_airport(batch, members, current["params"], words, chunk=args.chunk, device=cpu,
                                       records=args.out / label / "records" / airport, split="train", checkpoint=label,
                                       extra_summary={"arm": label, "closed_loop": True, "row_interval_s": interval},
                                       per_flight=per_flight)
            rows += r
            reports += graded["trajectories"]
        return rows, reports

    def stats(rows, reports, grids, corrections=None, lateral_vs_observed=None):
        out = Counter(r["outcome"] for r in rows)
        verdict = Counter(t["verdict"] for t in reports)
        g = lambda f: sum(t[f] == "pass" for t in reports)
        xt = [abs(t["cross_track_m"]) for t in reports if t.get("cross_track_m") is not None]
        n = len(grids)
        res = {"flights": len(rows), "landed": out["landed"], "outcomes": dict(out), "eval_pass": verdict["pass"],
               "lateral_pass": g("lateral_result"), "vertical_pass": g("vertical_result"), "speed_pass": g("speed_result"),
               "speed_fail": sum(t["speed_result"] == "fail" for t in reports),
               "xtrack_p50": float(np.percentile(xt, 50)), "xtrack_p95": float(np.percentile(xt, 95)),
               "said_words_per_sentence": float(sum(int((x != UNCHANGED).sum()) for x in grids) / n),
               "said_words_by_column": {name: float(sum(int((x[:, c] != UNCHANGED).sum()) for x in grids) / n)
                                        for c, name in enumerate(COLUMNS)},
               "rows_per_sentence": float(sum(len(x) for x in grids) / n)}
        if corrections is not None:
            res["correction_words_per_sentence"] = {name: float(sum(int(c[:, k].sum()) for c in corrections) / n)
                                                    for k, name in enumerate(COLUMNS)}
        if lateral_vs_observed is not None:
            res["left_the_path_gt300m"] = int(sum(x > 300 for x in lateral_vs_observed))
        return res

    def merge(regenerated, run_words):
        """The 2 s closed-loop sentences' words on the coarse UTC grid (module docstring), from the first predicted step."""
        by_index = {index: pos for pos, index in enumerate(drawn_coarse.indices)}
        start_c = start_row(args.merge_to_s)
        pseudo, refused = {}, Counter()
        for j, index in enumerate(drawn.indices):
            sentence = regenerated.get(index)
            if sentence is None or index not in by_index:
                continue
            pos = by_index[index]
            reading, signals, geometry = drawn.readings[j], drawn.signals[j], drawn.geometries[j]
            first2 = drawn.sentences[j].first_row
            start2 = sentence.rows.start
            full = np.vstack([reading.words[first2: first2 + start2], sentence.rows.grid]).astype(np.int16)
            height = sentence.rows.states[: len(full), 2] - geometry.elevation_m
            courses = [c.course_deg for c in geometry.candidates]
            try:
                first, grid = on_utc_grid(full, signals.entry_time_utc, args.merge_to_s, spec.step_s, height,
                                          run_words, courses)
            except Refused as why:
                refused[why.reason] += 1
                continue
            if first != drawn_coarse.sentences[pos].first_row:
                refused["first row differs from the coarse draw"] += 1
                continue
            if len(grid) - start_c < 2:
                refused["too short for the closed loop"] += 1
                continue
            predicted = grid[start_c:].copy()
            predicted[0] = in_force(grid)[start_c]            # the first predicted step says every column
            if predicted[0, RUNWAY] == RUNWAY_GO_AROUND:
                refused["go-around at the first predicted step"] += 1
                continue
            pseudo[index] = SimpleNamespace(rows=SimpleNamespace(first_row=first, grid=predicted))
        return pseudo, refused

    results = []
    for index, arm in enumerate(arms()):
        if arm["pm"]:
            le.uninstall(ORIGINAL_EXECUTOR, ORIGINAL_PLANT)
        else:
            le.install(arm["assumed"], arm["true"])
        ProgressCorrector.ENABLED = arm["split"]
        closed_loop.Corrector = ProgressCorrector
        closed_loop.vertical_tolerance_m = _phase_tolerance
        current["words"] = Words(spec)
        current["params"] = (replace(params, bank_rate_deg_s=args.p_new) if (args.p_new and not arm["pm"]) else params)
        regenerated, refused = {}, Counter()
        for i, r in zip(drawn.indices, closed_loop.read_chunked(drawn, current["params"], current["words"], chunk=args.chunk, device=cpu)):
            if isinstance(r, Refused):
                refused[r.reason] += 1
            else:
                regenerated[i] = r
        rows_r, rep_r = fly_and_grade(f"arm{index}_closed_loop", drawn, regenerated, True, args.row_interval_s)
        sents = list(regenerated.values())
        entry = {"arm": arm["name"], "split_thresholds": arm["split"],
                 "closed_loop": stats(rows_r, rep_r, [s.rows.grid for s in sents], [s.rows.correction for s in sents],
                                      [float(np.max(np.abs(s.withheld.lateral_m))) for s in sents])
                 | {"refused": dict(refused), "row_interval_s": args.row_interval_s}}
        line = (f"  [{index}] {arm['name']:84s} closed loop {args.row_interval_s:g} s: landed {entry['closed_loop']['landed']:3d}"
                f"/{entry['closed_loop']['flights']}, eval {entry['closed_loop']['eval_pass']:3d}, words "
                f"{entry['closed_loop']['said_words_per_sentence']:.1f}")
        if drawn_coarse is not None:
            pseudo, refused_m = merge(regenerated, current["words"])
            rows_m, rep_m = fly_and_grade(f"arm{index}_merged", drawn_coarse, pseudo, False, args.merge_to_s)
            entry["merged"] = stats(rows_m, rep_m, [p.rows.grid for p in pseudo.values()]) | {
                "refused": dict(refused_m), "row_interval_s": args.merge_to_s}
            line += (f" | merged to {args.merge_to_s:g} s: landed {entry['merged']['landed']:3d}/{entry['merged']['flights']}, "
                     f"eval {entry['merged']['eval_pass']:3d}, words {entry['merged']['said_words_per_sentence']:.1f}")
        print(line + f", {time.perf_counter() - started:.0f}s", flush=True)
        results.append(entry)
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
    le.uninstall(ORIGINAL_EXECUTOR, ORIGINAL_PLANT)
    closed_loop.Corrector = ORIGINAL_CORRECTOR
    closed_loop.vertical_tolerance_m = _ORIGINAL_TOLERANCE
    print(f"→ {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
