#!/usr/bin/env python
"""Can an executor that knows the actuator lag recover the landing ability? (diagnostic; nothing in the repo is edited)

The compensation sits between the executor's wanted controls c* (what its exact inversion asks for, assuming they act
at once) and the actuators. It holds an OBSERVER of the actuator state built from the commands it sent and the lag
constants it ASSUMES (never the plant's true state), and sends the one-cycle dead-beat command

    u = (c* − â e^{−Δt/τ̂}) / (1 − e^{−Δt/τ̂}),        â ← u + (â − u) e^{−Δt/τ̂}

so the actuator reaches c* at the end of the cycle when τ̂ = τ. u is clipped to the box the controls live in (bank to
the spec's cap unless --bank-clip-deg says otherwise, load factor and thrust fraction to the inversion's boxes), so a
saturated channel is only partly compensated. This is the same thing as the executor owning an actuator model; it lives
here as a plant wrapper so that no executor law changes.

Arms: the plant's TRUE constants and the ones the compensator ASSUMES are independent (mismatch tests).
Each arm flies (S) the artefact's stored words and (R) words REGENERATED with that plant in the loop, both graded by
`evaluation`; the words said per sentence is the cost of what the plant could not do on its own.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_plants as cp  # noqa: E402
from ts_transformer.autopilot import closed_loop, replay                                  # noqa: E402
from ts_transformer.autopilot import executor as executor_module                          # noqa: E402
from ts_transformer.autopilot import flights as flights_module                            # noqa: E402
from ts_transformer.autopilot import inverse                                              # noqa: E402
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS, Plant                       # noqa: E402
from ts_transformer.autopilot.spec import params_sha256                                   # noqa: E402
from ts_transformer.config import CONTROL_DYNAMICS_FIRST_ORDER_LAG                        # noqa: E402
from ts_transformer.experiments import executor_replay as er                              # noqa: E402
from ts_transformer.instructions.artefact import closed_loop_path, closed_loop_sentences  # noqa: E402
from ts_transformer.instructions.labeller.records import Refused                          # noqa: E402
from ts_transformer.instructions.words import COLUMNS                                     # noqa: E402
from ts_transformer.outputs.dynamics.rollout import rollout_control_endpoints             # noqa: E402
from ts_transformer.outputs.envelope import MAX_THRUST_FRACTION, MIN_THRUST_FRACTION      # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                              # noqa: E402

#: (thrust fraction, bank rad, load factor) — the executor's command order
DEFAULT_TAUS = (1.5, 2.0, 0.8)


def make_plant(true_taus, assumed_taus, compensate, bank_clip_deg):
    """A `Plant` class: the lag plant with ``true_taus``, a compensator over the channels flagged in ``compensate``
    (``(thrust, bank, load)`` bools) that assumes ``assumed_taus``."""
    dynamics = replace(EXECUTOR_DYNAMICS, control_dynamics_model=CONTROL_DYNAMICS_FIRST_ORDER_LAG,
                       control_thrust_time_constant_s=true_taus[0], control_bank_time_constant_s=true_taus[1],
                       control_load_time_constant_s=true_taus[2])
    flag = torch.tensor(compensate, dtype=torch.bool)
    low = torch.tensor([MIN_THRUST_FRACTION, -math.radians(bank_clip_deg), inverse.LOAD_FACTOR_MIN], dtype=torch.float64)
    high = torch.tensor([MAX_THRUST_FRACTION, math.radians(bank_clip_deg), inverse.LOAD_FACTOR_MAX], dtype=torch.float64)

    class CompensatedPlant(Plant):
        def __init__(self, inputs) -> None:
            super().__init__(inputs)
            self.actual: torch.Tensor | None = None      # the plant's own actuator state (the truth)
            self.observed: torch.Tensor | None = None    # the executor's observer of it

        def step(self, state, command, duration_s):
            if self.actual is None:
                self.actual, self.observed = command, command.clone()
            decay = torch.exp(-duration_s / torch.tensor(assumed_taus, dtype=state.dtype))
            dead_beat = (command - self.observed * decay) / (1.0 - decay)
            sent = torch.where(flag.to(state.device), dead_beat, command)
            sent = torch.maximum(torch.minimum(sent, high.to(state.device)), low.to(state.device))
            rollout = rollout_control_endpoints(
                sent[:, None, :], torch.full((len(state), 1), duration_s, dtype=state.dtype, device=state.device),
                {"initial_state": state, "initial_controls": self.actual, "aero_params": self.inputs.aero_params,
                 "frame_params": self.inputs.frame_params, "max_thrust_n": self.inputs.max_thrust_n}, dynamics)
            self.actual = rollout.actual_controls[:, -1]
            self.observed = sent + (self.observed - sent) * decay
            return rollout.geodetic_states[:, -1]

    return CompensatedPlant


def arms(bank_clip: float) -> list[dict]:
    t, b, l = DEFAULT_TAUS
    return [
        {"name": "lag, no compensation", "true": (t, b, l), "assumed": (t, b, l), "comp": (False, False, False), "clip": bank_clip},
        {"name": "lag + bank compensation", "true": (t, b, l), "assumed": (t, b, l), "comp": (False, True, False), "clip": bank_clip},
        {"name": "lag + all-channel compensation", "true": (t, b, l), "assumed": (t, b, l), "comp": (True, True, True), "clip": bank_clip},
        {"name": "all comp, bank clip 45° (grader's box)", "true": (t, b, l), "assumed": (t, b, l), "comp": (True, True, True), "clip": 45.0},
        # mismatch: the compensator assumes tau_bank = 2.0 s, the plant's is slower / faster
        {"name": "plant bank 3.0 s, no compensation", "true": (t, 3.0, l), "assumed": (t, 3.0, l), "comp": (False, False, False), "clip": bank_clip},
        {"name": "plant bank 3.0 s, comp assumes 2.0", "true": (t, 3.0, l), "assumed": (t, b, l), "comp": (True, True, True), "clip": bank_clip},
        {"name": "plant bank 1.0 s, no compensation", "true": (t, 1.0, l), "assumed": (t, 1.0, l), "comp": (False, False, False), "clip": bank_clip},
        {"name": "plant bank 1.0 s, comp assumes 2.0", "true": (t, 1.0, l), "assumed": (t, b, l), "comp": (True, True, True), "clip": bank_clip},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-airport", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--bank-clip-deg", type=float, default=32.0)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="arm indices to run")
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

    def fly_and_grade(label: str, sentences: dict, reproduces: bool):
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
            r, graded = er.fly_airport(batch, members, params, words, chunk=args.chunk, device=cpu,
                                       records=args.out / label / "records" / airport, split="train",
                                       checkpoint=label, extra_summary={"arm": label, "closed_loop": True,
                                                                        "row_interval_s": 2.0}, per_flight=per_flight)
            rows += r
            reports += graded["trajectories"]
        return rows, reports

    def stats(rows, reports, sentences=None):
        out = Counter(r["outcome"] for r in rows)
        verdict = Counter(t["verdict"] for t in reports)
        g = lambda f: sum(t[f] == "pass" for t in reports)
        xt = [abs(t["cross_track_m"]) for t in reports if t.get("cross_track_m") is not None]
        res = {"flights": len(rows), "landed": out["landed"], "outcomes": dict(out), "eval_pass": verdict["pass"],
               "lateral_pass": g("lateral_result"), "vertical_pass": g("vertical_result"), "speed_pass": g("speed_result"),
               "xtrack_p50": float(np.percentile(xt, 50)), "xtrack_p95": float(np.percentile(xt, 95))}
        if sentences is not None:
            words_said = Counter()
            for s in sentences.values():
                for name, n in zip(COLUMNS, s.rows.correction.sum(axis=0).tolist()):
                    words_said[name] += n
            res["correction_words_per_sentence"] = {k: v / len(sentences) for k, v in words_said.items()}
            lat = [float(np.max(np.abs(s.withheld.lateral_m))) for s in sentences.values()]
            res["largest_lateral_vs_observed_p50"] = float(np.percentile(lat, 50))
            res["left_the_path_gt300m"] = int(sum(x > 300 for x in lat))
        return res

    results = []
    # the point-mass baseline, on its own words
    executor_module.Plant = Plant
    rows, reports = fly_and_grade("point-mass", stored, True)
    results.append({"arm": "point-mass plant (baseline)", "stored_words": stats(rows, reports),
                    "regenerated_words": None})
    print(f"  baseline: {results[-1]['stored_words']['landed']} landed, {time.perf_counter() - started:.0f}s", flush=True)

    for index, arm in enumerate(arms(args.bank_clip_deg)):
        if args.only is not None and index not in args.only:
            continue
        executor_module.Plant = make_plant(arm["true"], arm["assumed"], arm["comp"], arm["clip"])
        # S: the stored words
        rows_s, rep_s = fly_and_grade(f"arm{index}_S", stored, False)
        # R: words regenerated with this plant in the loop, then flown again from them
        regenerated, refused = {}, Counter()
        results_r = closed_loop.read_chunked(drawn, params, words, chunk=args.chunk, device=cpu)
        for i, r in zip(drawn.indices, results_r):
            if isinstance(r, Refused):
                refused[r.reason] += 1
            else:
                regenerated[i] = r
        rows_r, rep_r = fly_and_grade(f"arm{index}_R", regenerated, True)
        results.append({"arm": arm["name"], "true_taus": arm["true"], "assumed_taus": arm["assumed"],
                        "compensated": arm["comp"], "bank_clip_deg": arm["clip"],
                        "stored_words": stats(rows_s, rep_s),
                        "regenerated_words": stats(rows_r, rep_r, regenerated) | {"refused": dict(refused)}})
        s, r = results[-1]["stored_words"], results[-1]["regenerated_words"]
        print(f"  [{index}] {arm['name']:42s} stored words: {s['landed']:3d} landed, eval {s['eval_pass']:3d} | "
              f"regenerated: {r['landed']:3d} landed, eval {r['eval_pass']:3d}, heading words "
              f"{r['correction_words_per_sentence']['heading']:.1f}/sentence, {time.perf_counter() - started:.0f}s", flush=True)
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"→ {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
