"""Where does a sentence's LAST event sit, and how should D61 judge it vertically? (plan §6.2 D61)

    python run_ts.py final_vertical_check --vocabulary <artefact>/instruction_vocabulary.json \\
        --executor <ckpt> --cohort <development_cohort.json> --cifp-file <FAACIFP18> [--limit N]

D61 asks whether the last instruction leaves the aircraft complying with LPV. The vertical half of
that was standing in as "altitude word = bin 0", i.e. cleared to within ±500 ft of the threshold —
a weak proxy, and one bound to a word that is being redesigned. This runner measures the three
candidate quantities at the LAST EVENT of every truth sentence, so the replacement is chosen on
numbers rather than on taste:

    d_to_go     along-track distance still to fly (positive before the threshold)
    h           height above the landing threshold
    angle_thr   atan(h / d)                                  — the angle to the THRESHOLD
    angle_aim   atan((h - TCH) / d)                          — the angle to the published aiming point
    dev_m       h - (TCH + d * tan(published glidepath))      — metres above/below the published path

Why the distinction is not pedantic: the published path crosses the threshold at TCH, so the angle
to the threshold carries a TCH bias of atan(TCH / d) — about 0.11° at 5 NM but 0.53° at 1 NM, which
is larger than the ±0.5° tolerance the angle would be judged with. If the last event sits close in,
the angle-to-threshold formulation cannot carry the criterion and the metre deviation must.

``dev_m`` is also the quantity AIM 5-4-5 states LPV's vertical limit in (50 m where the decision
altitude is at or above 250 ft, 35 m below it), so a criterion written on it cites the regulation
instead of inventing a tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.config import TSConfig
from ts_transformer.data.development_cohorts import load_development_cohort
from ts_transformer.experiments.support import REPO_ROOT, cohort_splits, rebuild_cohort
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.manoeuvre.instructions import course_frame, flight_runway, load_vocabulary, read_instructions
from ts_transformer.training.train import load_checkpoint_payload


def percentiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {}
    return {name: float(np.percentile(values, q)) for name, q in
            (("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))} | {
        "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def measure(*, vocabulary_path: Path, executor: Path, cohort_path: Path, cifp_file: Path,
            config_file: Path, limit: int) -> dict[str, Any]:
    from trajectory_data_process.harvest.airports import load_airport

    vocabulary, runways, _payload = load_vocabulary(vocabulary_path)
    payload = load_checkpoint_payload(executor)
    config = TSConfig.from_dict(payload["config"])
    cohort = load_development_cohort(cohort_path)
    splits = cohort_splits(payload, cohort, limit)
    series = rebuild_cohort(payload, config, [*splits["train"], *splits["val"]])

    airports = {item.airport for item in series}
    if len(airports) != 1:
        raise ValueError(f"the cohort spans {sorted(airports)}; one airport per runway vocabulary")
    airport = load_airport(airports.pop(), config_file=config_file, cifp_file=cifp_file)
    published = {}
    for ident in runways.idents:
        runway = airport.runway(ident)
        if runway.published_glidepath_deg is None:
            raise ValueError(f"{ident}: no published glidepath — D61's vertical check has no reference")
        published[ident] = (float(runway.published_glidepath_deg), float(runway.threshold_crossing_height_m))

    rows: list[dict[str, Any]] = []
    for item in series:
        reading = read_instructions(item, vocabulary, runways)
        frame = course_frame(item)
        last = float(reading.event_times_s[-1])
        d = float(np.interp(last, frame["t"], frame["to_go_m"]))
        h = float(np.interp(last, frame["t"], frame["height_m"]))
        gamma, tch = published[flight_runway(item)]
        if d <= 0:
            rows.append({"flight_id": item.flight_id, "past_threshold": True, "d_to_go_m": d, "height_m": h})
            continue
        rows.append({
            "flight_id": item.flight_id, "past_threshold": False, "runway": flight_runway(item),
            "d_to_go_m": d, "height_m": h,
            "angle_to_threshold_deg": math.degrees(math.atan2(h, d)),
            "angle_to_aiming_point_deg": math.degrees(math.atan2(h - tch, d)),
            "deviation_from_path_m": h - (tch + d * math.tan(math.radians(gamma))),
            "tch_bias_deg": math.degrees(math.atan2(tch, d)),
            "published_glidepath_deg": gamma,
        })

    on = [row for row in rows if not row["past_threshold"]]
    return {
        "flights": len(rows),
        "past_threshold": len(rows) - len(on),
        "published": {ident: {"glidepath_deg": g, "tch_m": t} for ident, (g, t) in published.items()},
        "d_to_go_m": percentiles(np.array([row["d_to_go_m"] for row in on])),
        "height_m": percentiles(np.array([row["height_m"] for row in on])),
        "angle_to_threshold_minus_published_deg": percentiles(
            np.array([row["angle_to_threshold_deg"] - row["published_glidepath_deg"] for row in on])),
        "angle_to_aiming_point_minus_published_deg": percentiles(
            np.array([row["angle_to_aiming_point_deg"] - row["published_glidepath_deg"] for row in on])),
        "deviation_from_path_m": percentiles(np.array([row["deviation_from_path_m"] for row in on])),
        "tch_bias_deg": percentiles(np.array([row["tch_bias_deg"] for row in on])),
        "within": {
            "angle_to_threshold_0p5deg": sum(
                abs(row["angle_to_threshold_deg"] - row["published_glidepath_deg"]) <= 0.5 for row in on),
            "angle_to_aiming_point_0p5deg": sum(
                abs(row["angle_to_aiming_point_deg"] - row["published_glidepath_deg"]) <= 0.5 for row in on),
            "deviation_35m": sum(abs(row["deviation_from_path_m"]) <= 35.0 for row in on),
            "deviation_50m": sum(abs(row["deviation_from_path_m"]) <= 50.0 for row in on),
            "of": len(on),
        },
    }


def render(out: dict[str, Any]) -> str:
    lines = [f"D61's vertical check at the LAST event of each truth sentence — {out['flights']} flights "
             f"({out['past_threshold']} already past the threshold, excluded)", ""]
    for name in ("d_to_go_m", "height_m", "tch_bias_deg", "angle_to_threshold_minus_published_deg",
                 "angle_to_aiming_point_minus_published_deg", "deviation_from_path_m"):
        row = out[name]
        lines.append(f"  {name:>44}: p5 {row['p5']:+9.2f}  p50 {row['p50']:+9.2f}  p95 {row['p95']:+9.2f}  "
                     f"(mean {row['mean']:+9.2f})")
    within = out["within"]
    lines += ["", f"  within ±0.5° of published, angle to THRESHOLD    : {within['angle_to_threshold_0p5deg']}/{within['of']}"
                 f" = {100 * within['angle_to_threshold_0p5deg'] / within['of']:.1f} %",
              f"  within ±0.5° of published, angle to AIMING POINT : {within['angle_to_aiming_point_0p5deg']}/{within['of']}"
              f" = {100 * within['angle_to_aiming_point_0p5deg'] / within['of']:.1f} %",
              f"  within ±35 m of the published path (AIM, DA < 250 ft) : {within['deviation_35m']}/{within['of']}"
              f" = {100 * within['deviation_35m'] / within['of']:.1f} %",
              f"  within ±50 m of the published path (AIM, DA ≥ 250 ft) : {within['deviation_50m']}/{within['of']}"
              f" = {100 * within['deviation_50m'] / within['of']:.1f} %"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_ts.py final_vertical_check", allow_abbrev=False, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cifp-file", type=Path, required=True)
    parser.add_argument("--config-file", type=Path,
                        default=REPO_ROOT / "trajectory_data_process" / "config" / "runway_thresholds.json")
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of each split (stated in the output)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    out = measure(vocabulary_path=absolute(args.vocabulary), executor=absolute(args.executor),
                  cohort_path=absolute(args.cohort), cifp_file=absolute(args.cifp_file),
                  config_file=absolute(args.config_file), limit=args.limit)
    out["limit"] = args.limit or None
    table = render(out)
    print(table)
    if args.out is not None:
        write_json_atomic(absolute(args.out), {"schema": "ts-final-vertical-check-v1", "written_utc": utc_now(), **out})
        print(f"  written to {absolute(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
