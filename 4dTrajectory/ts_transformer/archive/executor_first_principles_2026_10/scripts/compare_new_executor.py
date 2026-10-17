#!/usr/bin/env python
"""The new lag-aware executor (lag_executor.py) on the same 500 flights, stored and regenerated words.

The lag plant is flown WITHOUT the compensation layer; the executor keeps its own laws except `lateral.word_rate`.

Original (autopilot/lateral.py):
    rate = min( |e| / max(to_go, 2 Δt),  sqrt(k |e|) ),  clamped to r_max,           k = 2 g p / V
    sqrt(k|e|) is the fastest rate whose bank can still be rolled out at p before the error e is gone:
    the turn that remains after the roll-out starts is r²/k.

Lag-aware (this file): the plant needs about τ more seconds to bring the bank back, and turns r·τ more in that time:
    r²/k + τ r = |e|   →   r_stop = k ( −τ + sqrt(τ² + 4|e|/k) ) / 2          (τ = 0 gives the original)
    "stop"  arm: only r_stop changes.
    "shift" arm: also the arrival rate aims τ earlier, |e| / max(to_go − τ, 2 Δt): a lag moves every turn τ later.
τ is the law's ASSUMED bank time constant; the plant's own τ is a separate number (mismatch arms).
The words, the bank cap and the roll rate p are the executor's; no word is looked at beyond the one in force.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_compensation as cc  # noqa: E402  (DEFAULT_TAUS)
import lag_executor as le          # noqa: E402
import compare_plants as cp        # noqa: E402
from ts_transformer.autopilot import closed_loop, replay                                  # noqa: E402
from ts_transformer.autopilot import executor as executor_module                          # noqa: E402
from ts_transformer.autopilot import flights as flights_module                            # noqa: E402
from ts_transformer.autopilot.plant import Plant                                          # noqa: E402
ORIGINAL_EXECUTOR, ORIGINAL_PLANT = executor_module.Executor, Plant
from ts_transformer.autopilot.spec import params_sha256                                   # noqa: E402
from ts_transformer.experiments import executor_replay as er                              # noqa: E402
from ts_transformer.instructions.artefact import closed_loop_path, closed_loop_sentences  # noqa: E402
from ts_transformer.instructions.labeller.records import Refused                          # noqa: E402
from ts_transformer.instructions.words import COLUMNS                                     # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                              # noqa: E402




def arms() -> list[dict]:
    t, b, l = cc.DEFAULT_TAUS
    return [
        {"name": "new executor, plant τ_bank 2.0, assumes 2.0", "true": (t, b, l), "assumed": (t, b, l), "p": None},
        {"name": "new executor, plant τ_bank 3.0, assumes 2.0", "true": (t, 3.0, l), "assumed": (t, b, l), "p": None},
        {"name": "new executor, plant τ_bank 1.0, assumes 2.0", "true": (t, 1.0, l), "assumed": (t, b, l), "p": None},
        {"name": "new executor, plant τ_bank 2.0, assumes 1.0", "true": (t, b, l), "assumed": (t, 1.0, l), "p": None},
        {"name": "new executor, plant τ_bank 2.0, assumes 2.0, roll rate p = 8°/s", "true": (t, b, l), "assumed": (t, b, l), "p": 8.0},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-airport", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    flights_module.arrival_manifest_path = lambda airport: arrival_manifest_path(airport, cp.HARVEST)
    cpu = torch.device("cpu")
    started = time.perf_counter()
    params, record, words = replay.open_spec(cp.EXECUTOR, cp.INSTRUCTIONS)
    spec = words.spec
    with np.load(closed_loop_path(cp.INSTRUCTIONS, "train", 2.0)) as data:
        assert str(data["executor_params_sha256"]) == params_sha256(params)
    drawn = replay.draw(cp.INSTRUCTIONS, "train", spec, words, per_airport=args.per_airport, seed=args.seed,
                        groups=(replay.OWN, replay.STAND_IN), row_interval_s=2.0)
    stored = {i: s for i, s in closed_loop_sentences(cp.INSTRUCTIONS, "train", 2.0, spec).items() if i in set(drawn.indices)}
    along = er.along_columns(words)
    print(f"{len(drawn.indices)} flights, {time.perf_counter() - started:.0f}s", flush=True)

    def fly_and_grade(label, sentences, reproduces, run_params=None):
        run_params = run_params or params
        batch, _ = closed_loop.replay_batch(drawn, sentences, words)
        closed = er.closed_loop_columns(sentences, spec.step_s) if reproduces else None

        def per_flight(part, flown, verdicts):
            more = closed(part, flown, verdicts) if closed else [{} for _ in verdicts]
            return [{**a, **c} for a, c in zip(along(part, flown, verdicts), more)]
        by_airport = defaultdict(list)
        for j, f in enumerate(batch.signals):
            by_airport[f.airport].append(j)
        rows, reports = [], []
        for airport, members in sorted(by_airport.items()):
            r, graded = er.fly_airport(batch, members, run_params, words, chunk=args.chunk, device=cpu,
                                       records=args.out / label / "records" / airport, split="train", checkpoint=label,
                                       extra_summary={"arm": label, "closed_loop": True, "row_interval_s": 2.0},
                                       per_flight=per_flight)
            rows += r
            reports += graded["trajectories"]
        return rows, reports

    def stats(rows, reports, sentences=None):
        out = Counter(r["outcome"] for r in rows)
        verdict = Counter(t["verdict"] for t in reports)
        g = lambda f: sum(t[f] == "pass" for t in reports)
        xt = [abs(t["cross_track_m"]) for t in reports if t.get("cross_track_m") is not None]
        res = {"flights": len(rows), "landed": out["landed"], "outcomes": dict(out), "eval_pass": verdict["pass"],
               "lateral_pass": g("lateral_result"), "vertical_pass": g("vertical_result"),
               "xtrack_p50": float(np.percentile(xt, 50)), "xtrack_p95": float(np.percentile(xt, 95))}
        if sentences is not None:
            said = Counter()
            for s in sentences.values():
                for name, n in zip(COLUMNS, s.rows.correction.sum(axis=0).tolist()):
                    said[name] += n
            res["correction_words_per_sentence"] = {k: v / len(sentences) for k, v in said.items()}
            lat = [float(np.max(np.abs(s.withheld.lateral_m))) for s in sentences.values()]
            res["largest_lateral_vs_observed_p50"] = float(np.percentile(lat, 50))
            res["left_the_path_gt300m"] = int(sum(x > 300 for x in lat))
        return res

    results = []
    for index, arm in enumerate(arms()):
        if args.only is not None and index not in args.only:
            continue
        le.install(arm["assumed"], arm["true"])
        run_params = params if arm["p"] is None else replace(params, bank_rate_deg_s=arm["p"])
        rows_s, rep_s = fly_and_grade(f"arm{index}_S", stored, False, run_params)
        regenerated, refused = {}, Counter()
        for i, r in zip(drawn.indices, closed_loop.read_chunked(drawn, run_params, words, chunk=args.chunk, device=cpu)):
            if isinstance(r, Refused):
                refused[r.reason] += 1
            else:
                regenerated[i] = r
        rows_r, rep_r = fly_and_grade(f"arm{index}_R", regenerated, True, run_params)
        results.append({"arm": arm["name"], "plant_taus": arm["true"], "assumed_taus": arm["assumed"], "p_deg_s": arm["p"],
                        "stored_words": stats(rows_s, rep_s),
                        "regenerated_words": stats(rows_r, rep_r, regenerated) | {"refused": dict(refused)}})
        s, r = results[-1]["stored_words"], results[-1]["regenerated_words"]
        print(f"  [{index}] {arm['name']:66s} stored: {s['landed']:3d} landed, eval {s['eval_pass']:3d} | "
              f"regenerated: {r['landed']:3d} landed, eval {r['eval_pass']:3d}, heading words "
              f"{r['correction_words_per_sentence']['heading']:.1f}/sentence, {time.perf_counter() - started:.0f}s",
              flush=True)
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
    le.uninstall(ORIGINAL_EXECUTOR, ORIGINAL_PLANT)
    print(f"→ {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
