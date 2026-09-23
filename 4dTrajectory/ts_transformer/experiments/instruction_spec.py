"""Instruction labeller, step 2: measure the vocabulary's measured values on the TRAIN signals
and write the spec (vocabulary design §8).

Every flight first goes through the labeller's own gate (`labeller.read.admit`): the measured
population is the labelled one, cut before the landing, with the ground-speed refusals applied.
Pass A reads, with a provisional spec (the choices of `measure.SUGGESTED`, stand-ins for the
measured values), the bank in turns, the speed's acceleration between holds, the course error
on the last stretch of the final, and the path angle of every move piece; it also lists the
track's and the altitude's wander for the bands the two tolerances are chosen from. Those set
the bank range, the acceleration bound, the course tolerance and the descent classes. Pass B,
with the course tolerance, fits the capture corridor on the aligned final and compares the
heading grids on the rows before it. Writes ``spec.json`` (with the labeller's source hash and
git state) and ``measurements.json`` into the signals directory (never over an existing file).

    python run_ts.py instruction_spec --dir 4dTrajectory/outputs/POOLED/instruction_language/<name>
"""

from __future__ import annotations

import argparse
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.instructions import measure
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import labeller_source_sha256, load_candidates, load_signals, write_spec
from ts_transformer.instructions.labeller.read import admit
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.repo_layout import REPO_ROOT

CHUNK = 1000


def _admitted(flights: list[Any], spec: VocabularySpec, geometry_data: dict[str, Any]) -> tuple[list[Any], Counter]:
    geometries = {code: AirportGeometry.from_dict(data) for code, data in geometry_data.items()}
    admitted, refused = [], Counter()
    for flight in flights:
        try:
            admitted.append(admit(flight, geometries[flight.airport], spec))
        except Refused as refusal:
            refused[refusal.reason] += 1
    return admitted, refused


def _pass_a(flights: list[Any], spec_data: dict[str, Any], geometry_data: dict[str, Any]) -> tuple[dict[str, np.ndarray], Counter]:
    spec = VocabularySpec.from_dict(spec_data)
    admitted, refused = _admitted(flights, spec, geometry_data)
    pooled: dict[str, list[np.ndarray]] = {}
    for flight in admitted:
        for name, values in measure.measure_flight(flight, spec).items():
            pooled.setdefault(name, []).append(values)
    return {name: np.concatenate(values) for name, values in pooled.items()}, refused


def _pass_b(flights: list[Any], spec_data: dict[str, Any], geometry_data: dict[str, Any], course_tolerance_deg: float,
            grids: dict[float, float]) -> tuple[dict[str, np.ndarray], dict[str, list[list[float]]]]:
    spec = VocabularySpec.from_dict(spec_data)
    admitted, _ = _admitted(flights, spec, geometry_data)
    pooled: dict[str, list[np.ndarray]] = {}
    grid_rows: dict[str, list[list[float]]] = {}
    for flight in admitted:
        arrays, rows = measure.measure_final(flight, spec, course_tolerance_deg, grids)
        for name, values in arrays.items():
            pooled.setdefault(name, []).append(values)
        for step, row in rows.items():
            grid_rows.setdefault(step, []).append(row)
    return {name: np.concatenate(values) for name, values in pooled.items()}, grid_rows


def _pool(results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([r[name] for r in results]) for name in results[0]}


def _grid_table(grid_rows: dict[str, list[list[float]]], grids: dict[float, float]) -> dict[str, Any]:
    table = {}
    for step, tolerance in grids.items():
        rows = grid_rows[f"{step:g}"]
        holds = np.array([r[0] for r in rows])
        changes = np.array([r[1] for r in rows])
        widths = np.array([w for r in rows for w in r[2:]])
        table[f"{step:g}"] = {"tolerance_deg": tolerance, "classes": int(round(360 / step)),
                              "holds_per_flight": measure.percentiles(holds),
                              "target_changes_per_flight": measure.percentiles(changes),
                              "hold_funnel_half_width_end_m": measure.percentiles(widths)}
    return table


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--dir", type=Path, required=True, help="the directory instruction_signals wrote")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    directory = args.dir if args.dir.is_absolute() else REPO_ROOT / args.dir
    for name in ("spec.json", "measurements.json"):
        if (directory / name).exists():
            parser.error(f"{directory / name} exists; an instruction artefact is never overwritten")
    started = time.perf_counter()
    labeller = labeller_source_sha256()          # the code the workers measure with
    flights = load_signals(directory, "train")
    print(f"{len(flights)} train flights", flush=True)
    provisional = measure.provisional_spec()
    geometry_data = {code: geometry.to_dict() for code, geometry in load_candidates(directory).items()}
    suggested = measure.SUGGESTED
    chunks = [flights[i: i + CHUNK] for i in range(0, len(flights), CHUNK)]

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        results = [f.result() for f in [pool.submit(_pass_a, chunk, provisional.to_dict(), geometry_data) for chunk in chunks]]
        first = _pool([arrays for arrays, _ in results])
        refused = sum((counts for _, counts in results), Counter())
        print(f"  pass A done ({sum(refused.values())} flights not admitted: {dict(refused)}), "
              f"{time.perf_counter() - started:.0f}s", flush=True)
        course_tolerance = measure.round_up(float(np.percentile(first["final_course_error_deg"], 99)), 1.0)
        grids = {g: g / 2 + measure.HEADING_WANDER_ALLOWANCE_DEG for g in measure.HEADING_GRIDS_DEG}
        results_b = [f.result() for f in [pool.submit(_pass_b, chunk, provisional.to_dict(), geometry_data,
                                                      course_tolerance, grids) for chunk in chunks]]
        second = _pool([arrays for arrays, _ in results_b])
        grid_rows: dict[str, list[list[float]]] = {}
        for _, rows in results_b:
            for step, items in rows.items():
                grid_rows.setdefault(step, []).extend(items)
        print(f"  pass B done, {time.perf_counter() - started:.0f}s", flush=True)

    angle, length = first["move_angle_deg"], first["move_length_m"]
    corridor = measure.fit_corridor(second["aligned_offset_m"], second["aligned_distance_m"])
    fits = {k: measure.fit_descent_classes(angle, length, k) for k in measure.DESCENT_CLASS_COUNTS}
    chosen = fits[measure.DESCENT_CLASSES]
    climbs = angle < measure.DESCENT_FLOOR_DEG
    order = np.argsort(-angle[climbs])
    cumulative = np.cumsum(length[climbs][order]) / length[climbs].sum()
    climb_centre = float(-angle[climbs][order][np.searchsorted(cumulative, 0.5)])
    measured = measure.MeasuredValues(
        turn_bank_min_deg=max(1.0, measure.round_down(float(np.percentile(first["turn_mean_bank_deg"], 5)), 1.0)),
        turn_bank_max_deg=min(89.0, measure.round_up(float(np.percentile(first["turn_row_bank_deg"], 99.9)), 1.0)),
        corridor_half_width_m=measure.round_up(corridor["half_width_m"], 5.0),
        corridor_widening_deg=measure.round_up(float(np.degrees(np.arctan(corridor["slope"]))), 0.05),
        corridor_course_tolerance_deg=course_tolerance,
        descent_angle_edges_deg=tuple(round(e, 2) for e in chosen["edges_deg"]),
        descent_angle_centres_deg=tuple(round(c, 2) for c in chosen["centres_deg"]),
        climb_angle_centre_deg=round(climb_centre, 2),
        speed_accel_max_mps2=measure.round_up(float(np.percentile(first["transition_accel_mps2"], 99.9)), 0.1),
    )
    spec = measure.build_spec(measured)
    measurements = {
        "train_flights": len(flights), "not_admitted": dict(refused.most_common()),
        "suggested": suggested, "measured": measured.to_dict(),
        "rules": {
            "heading_tolerance_deg": f"chosen: heading_step/2 + {measure.HEADING_WANDER_ALLOWANCE_DEG:g}° of wander "
                                     "(see sensitivity.heading_wander_p95_by_band)",
            "altitude_tolerance_m": "chosen: altitude_step/2 + altitude_fit_tolerance "
                                    "(see sensitivity.level_wander_p95_by_fit_tolerance)",
            "turn_bank_min_deg": "p5 of a turn's mean bank, down to 1°",
            "turn_bank_max_deg": "p99.9 of the bank on turning rows, up to 1°",
            "corridor_half_width_m": "p99 offset of the aligned final's nearest distance bin (0–3 km), up to 5 m",
            "corridor_widening_deg": "the smallest widening that keeps every bin's p99 inside at the bin's middle, up to 0.05°",
            "corridor_course_tolerance_deg": f"p99 of |track − course| on the last {measure.FINAL_MEASURE_M:.0f} m flown, up to 1°",
            "descent_angle_classes": f"{measure.DESCENT_CLASSES} classes, weighted k-means on tan(angle), weight = length²",
            "climb_angle_centre_deg": "length-weighted median of the climb pieces",
            "speed_accel_max_mps2": "p99.9 of |acceleration| on transition rows, up to 0.1",
        },
        "sensitivity": {
            "heading_wander_p95_by_band": {f"{h:g}": float(np.percentile(first[f"heading_wander_deg_band{h:g}"], 95))
                                           for h in measure.FREE_HOLD_HALF_RANGES_DEG},
            "level_wander_p95_by_fit_tolerance": {f"{t:g}": float(np.percentile(first[f"level_wander_m_fit{t:g}"], 95))
                                                  for t in measure.LEVEL_FIT_TOLERANCES_M},
        },
        "pass_a": {name: measure.percentiles(values) for name, values in first.items()
                   if name not in ("move_angle_deg", "move_length_m")},
        "move_pieces": {"angle_deg": measure.percentiles(angle), "length_m": measure.percentiles(length),
                        "climb_pieces": int(climbs.sum())},
        "descent_class_fits": {str(k): v for k, v in fits.items()},
        "corridor_fit": corridor,
        "heading_grids": _grid_table(grid_rows, grids),
        "elapsed_s": time.perf_counter() - started,
    }
    if labeller_source_sha256() != labeller:
        raise SystemExit("the labeller's code changed while the spec was being measured; measure again")
    source = {"labeller_source_sha256": labeller, "git": _git_state()}
    write_spec(directory, spec, measurements, source)
    print(f"spec {spec.sha256[:12]}:")
    for name, value in measured.to_dict().items():
        print(f"  {name:32s} {value}")
    for k, fit in fits.items():
        e = fit["end_height_error_m"]
        print(f"  descent K={k}: centres {[round(c, 2) for c in fit['centres_deg']]}  end-height error p50 "
              f"{e['p50']:.1f} m, p95 {e['p95']:.1f} m")
    for grid, row in measurements["heading_grids"].items():
        print(f"  heading grid {grid}°: tolerance {row['tolerance_deg']}°, target changes/flight "
              f"p50 {row['target_changes_per_flight']['p50']:.0f}, funnel end p95 "
              f"{row['hold_funnel_half_width_end_m']['p95']:.0f} m")
    print(f"wrote spec.json and measurements.json in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
