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
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words                                     # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                              # noqa: E402




import math as _math
from aerodynamic_model.torch_dynamics import ISA_RHO0_KG_M3, isa_density          # noqa: E402
from ts_transformer.autopilot import speed as speed_module                          # noqa: E402
from evaluation.speed_gate import SPEED_GATE_UPPER_ADDITIVE_MS                     # noqa: E402

ORIGINAL_RATE = speed_module.Speed.rate


def window_middle_rate(self, state, speed_mps, unspecified, go_around, load_factor, aero_params, straight_m):
    """`Speed.rate` with "unspecified" at the middle of the gate's window [V_REF sqrt(n), V_REF + 20 kt]."""
    density = isa_density(state.height_m)
    ratio = torch.sqrt(ISA_RHO0_KG_M3 / density)
    ias = self.approach_ias_mps
    own = 0.5 * (ias * torch.sqrt(load_factor.clamp(min=1.0)) + ias + SPEED_GATE_UPPER_ADDITIVE_MS) * ratio
    wanted = torch.where(unspecified, torch.where(go_around, self.held_mps, own), speed_mps / torch.cos(state.gamma_rad))
    floor = self.margin * speed_module.stall_speed_mps(load_factor, state.mass_kg, density, aero_params[:, 0], aero_params[:, 1])
    landing = ((state.speed_mps.square() - own.square()) / (2.0 * straight_m.clamp(min=1.0))).clamp(
        self.pace_mps2, self.accel_max_mps2)
    rising = torch.where(unspecified, torch.full_like(landing, self.pace_mps2), torch.full_like(landing, self.accel_max_mps2))
    slowing = torch.where(unspecified & ~go_around, landing, rising)

    def toward(reference):
        faster = reference > state.speed_mps
        tau = self.band_mps / torch.where(faster, rising, slowing)
        return torch.minimum(torch.maximum((reference - state.speed_mps) / tau, -slowing), rising)
    return toward(torch.maximum(wanted, floor)), toward(wanted), {"stall_floor": floor > wanted}


def arms() -> list[dict]:
    t, b, l = cc.DEFAULT_TAUS
    base = {"true": (t, b, l), "assumed": (t, b, l), "p": None}
    return [
        {**base, "name": "CURRENT (official): original executor, point-mass plant, NO lag, deadband 30 m", "y": 30.0, "pm": True, "mid": False},
        {**base, "name": "NEW executor, lag plant, corrector deadband 20 m  (asked)", "y": 20.0, "pm": False, "mid": False},
        {**base, "name": "  original executor, point-mass plant, deadband 20 m  (deadband alone)", "y": 20.0, "pm": True, "mid": False},
        {**base, "name": "  NEW executor, lag plant, deadband 30 m  (executor alone)", "y": 30.0, "pm": False, "mid": False},
        {**base, "name": "  NEW executor + deadband 20 m + window-middle speed  (extra, not asked)", "y": 20.0, "pm": False, "mid": True},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-airport", type=int, default=100)
    ap.add_argument("--row-interval-s", type=float, default=2.0)
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
    base_words = words
    spec = words.spec
    with np.load(closed_loop_path(cp.INSTRUCTIONS, "train", 2.0)) as data:
        assert str(data["executor_params_sha256"]) == params_sha256(params)
    drawn = replay.draw(cp.INSTRUCTIONS, "train", spec, words, per_airport=args.per_airport, seed=args.seed,
                        groups=(replay.OWN, replay.STAND_IN), row_interval_s=args.row_interval_s)
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
                                       extra_summary={"arm": label, "closed_loop": True, "row_interval_s": args.row_interval_s},
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
            n = len(sentences)
            res["said_words_per_sentence"] = float(sum(int((s.rows.grid != UNCHANGED).sum()) for s in sentences.values()) / n)
            res["said_words_by_column"] = {name: float(sum(int((s.rows.grid[:, c] != UNCHANGED).sum())
                                                          for s in sentences.values()) / n)
                                           for c, name in enumerate(COLUMNS)}
            res["rows_per_sentence"] = float(sum(len(s.rows.grid) for s in sentences.values()) / n)
            lat = [float(np.max(np.abs(s.withheld.lateral_m))) for s in sentences.values()]
            res["largest_lateral_vs_observed_p50"] = float(np.percentile(lat, 50))
            res["left_the_path_gt300m"] = int(sum(x > 300 for x in lat))
        return res

    results = []
    for index, arm in enumerate(arms()):
        if args.only is not None and index not in args.only:
            continue
        if arm["pm"]:
            le.uninstall(ORIGINAL_EXECUTOR, ORIGINAL_PLANT)
        else:
            le.install(arm["assumed"], arm["true"])
        run_params = params
        words = Words(replace(base_words.spec, closed_loop_lateral_m=arm["y"]))
        speed_module.Speed.rate = window_middle_rate if arm["mid"] else ORIGINAL_RATE
        rows_s, rep_s = None, None   # the stored words are not flown: only the regenerated closed loop is compared
        regenerated, refused = {}, Counter()
        for i, r in zip(drawn.indices, closed_loop.read_chunked(drawn, run_params, words, chunk=args.chunk, device=cpu)):
            if isinstance(r, Refused):
                refused[r.reason] += 1
            else:
                regenerated[i] = r
        rows_r, rep_r = fly_and_grade(f"arm{index}_R", regenerated, True, run_params)
        results.append({"arm": arm["name"], "plant_taus": arm["true"], "assumed_taus": arm["assumed"], "deadband_m": arm["y"], "window_middle_speed": arm["mid"],
                        "stored_words": None,
                        "regenerated_words": stats(rows_r, rep_r, regenerated) | {"refused": dict(refused)}})
        r = results[-1]["regenerated_words"]
        print(f"  [{index}] {arm['name']:72s} landed {r['landed']:3d}/{r['flights']}, eval {r['eval_pass']:3d}, said words "
              f"{r['said_words_per_sentence']:.1f}/sentence, {time.perf_counter() - started:.0f}s", flush=True)
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
    le.uninstall(ORIGINAL_EXECUTOR, ORIGINAL_PLANT)
    speed_module.Speed.rate = ORIGINAL_RATE
    print(f"→ {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
