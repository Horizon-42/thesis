#!/usr/bin/env python
"""Three flights of the same 500 train flights, to separate "the plant" from "the correction words tuned to it".

  A  point-mass plant, the artefact's stored closed-loop sentences            (the baseline; reproduces the stored states)
  B  lag plant,        the SAME stored sentences                              (words tuned to the point-mass flight)
  C  lag plant,        sentences REGENERATED with the lag plant in the loop   (`closed_loop.read_chunked`: the corrector
                       sees the lag flight's lateral/vertical errors and says its heading/angle corrections for them)
  D  point-mass plant, sentences regenerated here — must equal the stored ones (checks this code and machine generate
                       the artefact's sentences before C is believed)

Each is flown from its first predicted step by the same executor and graded by `python -m evaluation`.
Diagnostic only; patches two module attributes of this process (see compare_plants.py), edits nothing.
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
import compare_plants as cp  # noqa: E402  (module-level names only: paths, LagPlant, LAG_DYNAMICS)
from ts_transformer.autopilot import closed_loop, replay                                  # noqa: E402
from ts_transformer.autopilot import executor as executor_module                          # noqa: E402
from ts_transformer.autopilot import flights as flights_module                            # noqa: E402
from ts_transformer.autopilot.frame import AirportCharts, read_state                      # noqa: E402
from ts_transformer.autopilot.plant import Plant                                          # noqa: E402
from ts_transformer.autopilot.spec import params_sha256                                   # noqa: E402
from ts_transformer.experiments import executor_replay as er                              # noqa: E402
from ts_transformer.instructions.artefact import closed_loop_path, closed_loop_sentences  # noqa: E402
from ts_transformer.instructions.labeller.records import Refused                          # noqa: E402
from ts_transformer.instructions.words import COLUMNS                                     # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-airport", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--taus", type=float, nargs=3, default=None, metavar=("THRUST", "BANK", "LOAD"))
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    if args.taus:
        cp.LAG_DYNAMICS = replace(cp.LAG_DYNAMICS, control_thrust_time_constant_s=args.taus[0],
                                  control_bank_time_constant_s=args.taus[1], control_load_time_constant_s=args.taus[2])
    flights_module.arrival_manifest_path = lambda airport: arrival_manifest_path(airport, cp.HARVEST)
    cpu = torch.device("cpu")
    started = time.perf_counter()

    params, record, words = replay.open_spec(cp.EXECUTOR, cp.INSTRUCTIONS)
    spec = words.spec
    with np.load(closed_loop_path(cp.INSTRUCTIONS, "train", 2.0)) as data:
        assert str(data["executor_params_sha256"]) == params_sha256(params)
    drawn = replay.draw(cp.INSTRUCTIONS, "train", spec, words, per_airport=args.per_airport, seed=args.seed,
                        groups=(replay.OWN, replay.STAND_IN), row_interval_s=2.0)
    stored = closed_loop_sentences(cp.INSTRUCTIONS, "train", 2.0, spec)
    print(f"{len(drawn.indices)} drawn flights, {time.perf_counter() - started:.0f}s", flush=True)

    def regenerate(plant_cls) -> dict[int, object]:
        executor_module.Plant = plant_cls
        results = closed_loop.read_chunked(drawn, params, words, chunk=args.chunk, device=cpu)
        return {index: r for index, r in zip(drawn.indices, results) if not isinstance(r, Refused)}, \
               Counter(r.reason for r in results if isinstance(r, Refused))

    # ---- D: regenerate under point-mass; it must be the stored artefact
    pm_new, pm_refused = regenerate(Plant)
    in_both = [i for i in drawn.indices if i in stored and i in pm_new]
    bad = []
    for i in in_both:
        a, b = stored[i], pm_new[i]
        same = (np.array_equal(a.rows.grid, b.rows.grid) and np.array_equal(a.rows.correction, b.rows.correction)
                and a.rows.states.shape == b.rows.states.shape
                and float(np.abs(a.rows.states - b.rows.states).max()) < 1e-6 and a.withheld.outcome == b.withheld.outcome)
        if not same:
            bad.append(i)
    print(f"D: point-mass regenerated {len(pm_new)} sentences ({dict(pm_refused)} refused); stored for {len(in_both)}; "
          f"differ from the stored artefact: {len(bad)}", flush=True)
    assert not bad, "this code does not regenerate the artefact's sentences under the point-mass plant"
    assert set(pm_new) == set(stored) & set(drawn.indices)

    # ---- C: regenerate under the lag plant
    lag_new, lag_refused = regenerate(cp.LagPlant)
    print(f"C: lag regenerated {len(lag_new)} sentences ({dict(lag_refused)} refused), "
          f"{time.perf_counter() - started:.0f}s", flush=True)

    # ---- fly + grade A, B, C
    tracks: dict[str, dict] = defaultdict(dict)
    rows: dict[str, list] = {}

    def recorder(label: str, check):
        def per_flight(part, flown, verdicts):
            charts = AirportCharts.of(part.geometries, dtype=flown.states.dtype, device=flown.states.device)
            for j, verdict in enumerate(verdicts):
                key = part.series[j].scenario.source["flight_key"]
                end = verdict.end_row
                one = AirportCharts(lat0_deg=charts.lat0_deg[j], lon0_deg=charts.lon0_deg[j],
                                    m_per_deg_lon=charts.m_per_deg_lon[j], elevation_m=charts.elevation_m[j])
                kin = read_state(flown.states[j, : end + 1], one)
                tracks[label][key] = {"e": kin.e_m.numpy(), "n": kin.n_m.numpy(), "h": kin.height_m.numpy(),
                                      "v": kin.speed_mps.numpy(), "outcome": verdict.outcome, "end_row": int(end)}
            return check(part, flown, verdicts)
        return per_flight

    along = er.along_columns(words)

    def fly(label: str, plant_cls, sentences, *, reproduces: bool) -> None:
        executor_module.Plant = plant_cls
        batch, missing = closed_loop.replay_batch(drawn, sentences, words)
        # `closed_loop_columns` raises unless a flight is flown again to its stored states and outcome: asked of the
        # arms whose sentences were made by the plant they fly on (A, C); B flies words made for another plant
        closed = er.closed_loop_columns(sentences, spec.step_s) if reproduces else None

        def check(part, flown, verdicts):
            more = closed(part, flown, verdicts) if closed else [{} for _ in verdicts]
            return [{**a, **c} for a, c in zip(along(part, flown, verdicts), more)]
        by_airport = defaultdict(list)
        for j, flight in enumerate(batch.signals):
            by_airport[flight.airport].append(j)
        rows[label] = []
        for airport, members in sorted(by_airport.items()):
            flown_rows, _ = er.fly_airport(
                batch, members, params, words, chunk=args.chunk, device=cpu,
                records=args.out / label / "records" / airport, split="train", checkpoint=f"{cp.EXECUTOR.name}+{label}",
                extra_summary={"executor_spec_sha256": record["sha256"], "row_interval_s": 2.0, "closed_loop": True,
                               "arm": label}, per_flight=recorder(label, check))
            rows[label] += flown_rows
        landed = sum(r["outcome"] == "landed" for r in rows[label])
        print(f"  {label}: {len(rows[label])} flown, {landed} landed, {time.perf_counter() - started:.0f}s", flush=True)

    fly("A_pm_stored", Plant, {i: stored[i] for i in pm_new}, reproduces=True)
    fly("B_lag_stored", cp.LagPlant, {i: stored[i] for i in pm_new}, reproduces=False)
    fly("C_lag_regenerated", cp.LagPlant, lag_new, reproduces=True)

    # ---- the words each arm said
    def words_said(sentences):
        out = Counter()
        for s in sentences.values():
            for name, n in zip(COLUMNS, s.rows.correction.sum(axis=0).tolist()):
                out[name] += n
        return {k: v / max(len(sentences), 1) for k, v in out.items()}
    wsum = {"A_pm_stored": words_said({i: stored[i] for i in pm_new}), "C_lag_regenerated": words_said(lag_new)}

    args.out.mkdir(exist_ok=True)
    (args.out / "rows.json").write_text(json.dumps(rows, default=str), encoding="utf-8")
    (args.out / "meta.json").write_text(json.dumps({
        "lag_time_constants_s": list(cp.LAG_DYNAMICS.control_time_constants_s), "per_airport": args.per_airport,
        "seed": args.seed, "flights_drawn": len(drawn.indices), "regenerated_point_mass_equals_stored": True,
        "lag_refused": dict(lag_refused), "point_mass_refused": dict(pm_refused),
        "correction_words_per_sentence": wsum, "elapsed_s": time.perf_counter() - started}, indent=1), encoding="utf-8")
    np.savez_compressed(args.out / "tracks.npz", **{f"{lab}|{k}|{n}": v for lab, per in tracks.items()
                                                    for k, rec in per.items() for n, v in rec.items()
                                                    if isinstance(v, np.ndarray)})
    json.dump({lab: {k: {"outcome": r["outcome"], "end_row": r["end_row"]} for k, r in per.items()}
               for lab, per in tracks.items()}, open(args.out / "track_index.json", "w"))
    print(f"→ {args.out}  ({time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
