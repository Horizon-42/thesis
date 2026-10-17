#!/usr/bin/env python
"""Fly the SAME stored closed-loop sentences on two plants and grade both with `evaluation`.

  point-mass       autopilot's own plant (autopilot/plant.py: EXECUTOR_DYNAMICS, instantaneous controls)
  first-order-lag  the same RHS with the three controls as actuator states (aerodynamic_model/torch_lag_dynamics.py)

Everything else is identical: the executor laws and limits, the words (the closed-loop sentences as stored, i.e. their
correction words were decided against the point-mass flight), the start state, the 1 s cycle, the integrator.

Diagnostic only: nothing under the repository is written; the harvest root is redirected to the copy
pulled from SP-AI (no code of the repository is edited, two module attributes are patched in this process).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import archive_paths   # noqa: E402  (README §2)
REPO = archive_paths.REPO
sys.path[:0] = [str(REPO / "4dTrajectory"), str(REPO)]

from ts_transformer.autopilot import closed_loop, replay                        # noqa: E402
from ts_transformer.autopilot import executor as executor_module                # noqa: E402
from ts_transformer.autopilot import flights as flights_module                  # noqa: E402
from ts_transformer.autopilot.frame import AirportCharts, read_state            # noqa: E402
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS, Plant             # noqa: E402
from ts_transformer.autopilot.spec import params_sha256                         # noqa: E402
from ts_transformer.config import CONTROL_DYNAMICS_FIRST_ORDER_LAG               # noqa: E402
from ts_transformer.experiments import executor_replay as er                    # noqa: E402
from ts_transformer.instructions.artefact import closed_loop_path, closed_loop_sentences  # noqa: E402
from ts_transformer.outputs.dynamics.rollout import rollout_control_endpoints   # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                    # noqa: E402

HARVEST = archive_paths.HARVEST
INSTRUCTIONS = REPO / "4dTrajectory/outputs/POOLED/instruction_language/v12_20261005"
EXECUTOR = REPO / "4dTrajectory/outputs/POOLED/executor/v17_20261005"

LAG_DYNAMICS = replace(EXECUTOR_DYNAMICS, control_dynamics_model=CONTROL_DYNAMICS_FIRST_ORDER_LAG)


class LagPlant(Plant):
    """`Plant` with the actuator states carried between cycles. At the first cycle the actuators sit AT the command
    (the point-mass plant's own `initial_controls`), so the lag acts on every change after that."""

    def __init__(self, inputs) -> None:
        super().__init__(inputs)
        self.actuators: torch.Tensor | None = None

    def step(self, state, command, duration_s):
        inputs = self.inputs
        if self.actuators is None:
            self.actuators = command
        rollout = rollout_control_endpoints(
            command[:, None, :],
            torch.full((len(state), 1), duration_s, dtype=state.dtype, device=state.device),
            {"initial_state": state, "initial_controls": self.actuators, "aero_params": inputs.aero_params,
             "frame_params": inputs.frame_params, "max_thrust_n": inputs.max_thrust_n},
            LAG_DYNAMICS)
        self.actuators = rollout.actual_controls[:, -1]
        return rollout.geodetic_states[:, -1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-airport", type=int, default=100)
    ap.add_argument("--split", default="train", choices=["train"])   # select/val are not read by this diagnostic
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--taus", type=float, nargs=3, metavar=("THRUST", "BANK", "LOAD"), default=None,
                    help="lag time constants s (default: TSConfig's 1.5 2.0 0.8)")
    args = ap.parse_args()
    global LAG_DYNAMICS
    if args.taus:
        LAG_DYNAMICS = replace(LAG_DYNAMICS, control_thrust_time_constant_s=args.taus[0],
                               control_bank_time_constant_s=args.taus[1], control_load_time_constant_s=args.taus[2])
    assert not args.out.exists(), f"{args.out} exists"
    # the pulled copy of the harvest (the repository's own harvest holds another generation under the same names)
    flights_module.arrival_manifest_path = lambda airport: arrival_manifest_path(airport, HARVEST)

    started = time.perf_counter()
    # The repository's gate (`require_conforming_*`) compares bit-exact sha256 digests of each reference flight's float64
    # inputs, written on the Linux box; on this Mac 50 of 336 digests differ, which no change of ours can fix. This
    # diagnostic skips that gate and instead holds the point-mass flight to its BEHAVIOUR on the stored inputs:
    # `closed_loop_columns` raises when a flight is not flown again to its stored closed-loop states (STATE_BOUND_M)
    # and stored outcome.
    params, record, words = replay.open_spec(EXECUTOR, INSTRUCTIONS)
    spec = words.spec
    with np.load(closed_loop_path(INSTRUCTIONS, args.split, 2.0)) as data:
        assert str(data["executor_params_sha256"]) == params_sha256(params), "closed loop flown by other params"
    batch = replay.draw(INSTRUCTIONS, args.split, spec, words, per_airport=args.per_airport, seed=args.seed,
                        groups=(replay.OWN, replay.STAND_IN), row_interval_s=2.0)
    stored = closed_loop_sentences(INSTRUCTIONS, args.split, 2.0, spec)
    batch, missing = closed_loop.replay_batch(batch, stored, words)
    print(f"{len(batch.sentences)} closed-loop sentences ({missing} flights had none); "
          f"{time.perf_counter() - started:.0f}s", flush=True)

    by_airport: dict[str, list[int]] = defaultdict(list)
    for j, flight in enumerate(batch.signals):
        by_airport[flight.airport].append(j)

    along = er.along_columns(words)
    closed = er.closed_loop_columns(stored, spec.step_s)
    tracks: dict[str, dict[str, dict]] = {"point-mass": {}, "first-order-lag": {}}
    rows: dict[str, list[dict]] = {"point-mass": [], "first-order-lag": []}
    reports: dict[str, dict] = {"point-mass": {}, "first-order-lag": {}}

    def recorder(plant_name: str):
        def per_flight(part, flown, verdicts):
            charts = AirportCharts.of(part.geometries, dtype=flown.states.dtype, device=flown.states.device)
            out = []
            for j, verdict in enumerate(verdicts):
                key = part.series[j].scenario.source["flight_key"]
                end = verdict.end_row
                kin = read_state(flown.states[j, : end + 1], replace_charts(charts, j))
                tracks[plant_name][key] = {
                    "e": kin.e_m.cpu().numpy(), "n": kin.n_m.cpu().numpy(), "h": kin.height_m.cpu().numpy(),
                    "v": kin.speed_mps.cpu().numpy(), "trk": kin.track_deg.cpu().numpy(),
                    "cmd": flown.commands[j, :end].cpu().numpy(), "outcome": verdict.outcome, "end_row": int(end)}
                out.append({})
            if plant_name == "point-mass":       # the stored reference is the point-mass flight: it must reproduce it
                merged = closed(part, flown, verdicts)
                return [{**a, **c} for a, c in zip(along(part, flown, verdicts), merged)]
            return [{**a, **o} for a, o in zip(along(part, flown, verdicts), out)]
        return per_flight

    def replace_charts(charts: AirportCharts, j: int) -> AirportCharts:
        # one flight's chart broadcast to the rows of its flown states
        return AirportCharts(lat0_deg=charts.lat0_deg[j], lon0_deg=charts.lon0_deg[j],
                             m_per_deg_lon=charts.m_per_deg_lon[j], elevation_m=charts.elevation_m[j])

    for plant_name, plant in (("point-mass", Plant), ("first-order-lag", LagPlant)):
        executor_module.Plant = plant
        for airport, members in sorted(by_airport.items()):
            flown_rows, graded = er.fly_airport(
                batch, members, params, words, chunk=args.chunk, device=torch.device("cpu"),
                records=args.out / plant_name / "records" / airport, split=args.split,
                checkpoint=f"{EXECUTOR.name}+{plant_name}",
                extra_summary={"executor_spec_sha256": record["sha256"], "row_interval_s": 2.0, "closed_loop": True,
                               "plant": plant_name},
                per_flight=recorder(plant_name))
            rows[plant_name] += flown_rows
            reports[plant_name][airport] = graded
            landed = sum(r["outcome"] == "landed" for r in flown_rows)
            print(f"  {plant_name:16s} {airport}: {len(flown_rows)} flown, {landed} landed, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)

    # ---- the pairwise track differences ----
    # Two readings: SAME ELAPSED SECONDS (it carries the along-track timing difference: a flight 8 s late is ~500 m
    # apart at 62 m/s) and TIME-FREE geometry (each lag point against the point-mass polyline: its distance to it, and
    # its height minus the point-mass height at the projection).
    def path_difference(a, b):
        """Per point of ``b`` against the polyline of ``a``: horizontal distance and height difference at the projection."""
        p = np.column_stack((b["e"], b["n"]))
        q = np.column_stack((a["e"], a["n"]))
        seg = q[1:] - q[:-1]
        length2 = np.maximum((seg ** 2).sum(1), 1e-9)
        rel = p[:, None, :] - q[None, :-1, :]
        t = np.clip((rel * seg[None]).sum(2) / length2[None], 0.0, 1.0)
        foot = q[None, :-1, :] + t[..., None] * seg[None]
        dist = np.hypot(*(p[:, None, :] - foot).transpose(2, 0, 1))
        j = dist.argmin(1)
        r = np.arange(len(p))
        h_on_a = a["h"][j] + t[r, j] * (a["h"][j + 1] - a["h"][j])
        return dist[r, j], b["h"] - h_on_a

    pairs = []
    for key, a in tracks["point-mass"].items():
        b = tracks["first-order-lag"][key]
        n = min(len(a["e"]), len(b["e"]))
        horizontal = np.hypot(a["e"][:n] - b["e"][:n], a["n"][:n] - b["n"][:n])
        vertical = b["h"][:n] - a["h"][:n]
        speed = b["v"][:n] - a["v"][:n]
        trk = (b["trk"][:n] - a["trk"][:n] + 180.0) % 360.0 - 180.0
        off, dh = path_difference(a, b)
        pairs.append({"flight_key": key, "airport": None,
                      "n_common_s": int(n), "outcome_point_mass": a["outcome"], "outcome_lag": b["outcome"],
                      "end_s_point_mass": a["end_row"], "end_s_lag": b["end_row"],
                      "max_horizontal_m": float(horizontal.max()), "final_horizontal_m": float(horizontal[-1]),
                      "max_vertical_m": float(np.abs(vertical).max()), "final_vertical_m": float(vertical[-1]),
                      "max_speed_mps": float(np.abs(speed).max()), "final_speed_mps": float(speed[-1]),
                      "max_track_deg": float(np.abs(trk).max()),
                      "path_max_offset_m": float(off.max()), "path_mean_offset_m": float(off.mean()),
                      "path_max_height_m": float(np.abs(dh).max()), "path_mean_height_m": float(np.abs(dh).mean()),
                      "path_p95_offset_m": float(np.percentile(off, 95))})
    np.savez_compressed(args.out / "tracks.npz", **{
        f"{plant}|{key}|{name}": value for plant, per in tracks.items() for key, rec in per.items()
        for name, value in rec.items() if isinstance(value, np.ndarray)})
    by_key_airport = {r["flight_key"]: r["airport"] for r in rows["point-mass"]}
    by_key_stratum = {r["flight_key"]: r["stratum"] for r in rows["point-mass"]}
    for p in pairs:
        p["airport"], p["stratum"] = by_key_airport[p["flight_key"]], by_key_stratum[p["flight_key"]]

    def strip(rs):
        return [{k: v for k, v in r.items()} for r in rs]

    (args.out / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
    (args.out / "rows.json").write_text(json.dumps({k: strip(v) for k, v in rows.items()}, default=str), encoding="utf-8")
    (args.out / "meta.json").write_text(json.dumps({
        "plants": {"point-mass": "autopilot/plant.py EXECUTOR_DYNAMICS (point-mass)",
                   "first-order-lag": {"model": "torch_lag_dynamics via control_dynamics_model=first-order-lag",
                                       "time_constants_s_thrust_bank_load": list(LAG_DYNAMICS.control_time_constants_s),
                                       "initial_actuators": "equal to the first command"}},
        "per_airport": args.per_airport, "seed": args.seed, "split": args.split, "flights": len(batch.sentences),
        "drawn": batch.drawn, "executor": str(EXECUTOR), "instructions": str(INSTRUCTIONS),
        "executor_params": asdict(params), "elapsed_s": time.perf_counter() - started}, default=str, indent=1),
        encoding="utf-8")
    print(f"→ {args.out}  ({time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
