"""Measure absolute ground-track angles in airport-centred tangent ENU on the existing development roster."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from ts_transformer.experiments.measure_instruction_distribution import REPO, sha, write_json


def enu_basis(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Rows are local east, north and up unit vectors expressed in ECEF."""
    phi, lam = np.radians(lat), np.radians(lon)
    sp, cp, sl, cl = np.sin(phi), np.cos(phi), np.sin(lam), np.cos(lam)
    return np.stack((np.stack((-sl, cl, np.zeros_like(sl)), axis=-1),
                     np.stack((-sp * cl, -sp * sl, cp), axis=-1),
                     np.stack((cp * cl, cp * sl, sp), axis=-1)), axis=-2)


def absolute_angles(series, reference: dict) -> tuple[np.ndarray, np.ndarray]:
    """Undo channel transport factors, then rotate physical velocity into fixed airport ENU."""
    from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
    from ts_transformer.data.coordinate_frames import AirportENUFrame, AirportReference

    position = series.values[:, POSITION_IDX]
    velocity = series.values[:, VELOCITY_IDX]
    lat, lon = np.vectorize(series.frame.latlon_from_horizontal)(position[:, 0], position[:, 1])
    east_dot, north_dot = np.vectorize(series.frame.to_world_horizontal)(velocity[:, 0], velocity[:, 1])
    factors = np.array([series.frame.chart_velocity_factors(p, h)
                        for p, h in zip(lat, position[:, 2] + series.frame.alt0)])
    local = np.column_stack((east_dot / factors[:, 0], north_dot / factors[:, 1], velocity[:, 2]))
    earth_velocity = np.einsum("ni,nij->nj", local, enu_basis(lat, lon))
    airport_basis = enu_basis(np.asarray(reference["lat"]), np.asarray(reference["lon"]))
    airport_velocity = earth_velocity @ airport_basis.T
    angle = np.degrees(np.arctan2(airport_velocity[:, 1], airport_velocity[:, 0])) % 360
    # Audit against the repository's affine airport-enu chart, not the main measurement.
    airport_frame = AirportENUFrame.for_airport(AirportReference(
        code=series.airport, lat=reference["lat"], lon=reference["lon"], elevation_msl_m=reference["elevation_m"]))
    chart_angle = np.degrees(np.arctan2(north_dot, east_dot * airport_frame.m_per_deg_lon / series.frame.m_per_deg_lon)) % 360
    return angle, chart_angle


def describe(x: np.ndarray, weight: np.ndarray, edges: np.ndarray) -> dict:
    mass = np.histogram(x, edges, weights=weight)[0]
    share = mass / weight.sum()
    peaks = []
    for index in np.argsort(share)[::-1]:
        centre = float((edges[index] + edges[index + 1]) / 2)
        if all(abs((centre - p["centre_deg"] + 180) % 360 - 180) >= 20 for p in peaks):
            peaks.append({"left_deg": float(edges[index]), "right_deg": float(edges[index + 1]),
                          "centre_deg": centre, "share": float(share[index])})
        if len(peaks) == 3:
            break
    return {"rows": len(x), "histogram_counts": np.histogram(x, edges)[0].tolist(),
            "histogram_flight_weight": mass.tolist(), "separated_peak_bins": peaks,
            "top_ten_bins_share": float(np.sort(share)[-10:].sum()),
            "within_one_degree_of_east": float(np.average(np.abs((x + 180) % 360 - 180) <= 1, weights=weight))}


def run(analysis: Path, out: Path) -> None:
    from flight_scenarios.runway_target import airport_reference_point
    from ts_transformer.config import TSConfig
    from ts_transformer.data.dataset import build_series, load_flight_dicts
    from ts_transformer.data.development_cohorts import load_development_cohort
    from ts_transformer.manoeuvre.instructions import course_frame, min_rows, smooth

    provenance = json.loads((analysis / "extraction.json").read_text())
    old_summary = json.loads((analysis / "distribution_summary.json").read_text())
    for name, expected in provenance["source_hashes"].items():
        assert sha(REPO / name) == expected, f"source changed: {name}"
    cohort_path = REPO / provenance["cohort_path"]
    assert sha(cohort_path) == provenance["cohort_sha256"]
    cohort = load_development_cohort(cohort_path)
    allowed = {"train": set(cohort.train_flight_ids), "val": set(cohort.val_flight_ids)}
    assert not allowed["train"] & allowed["val"]
    with (analysis / "flight_audit.csv").open() as file:
        roster = {r["flight_id"]: r for r in csv.DictReader(file) if r["retained_in_distribution"] == "True"}
    assert all(key in allowed[r["split"]] for key, r in roster.items())
    config = TSConfig.from_dict(provenance["config"])
    assert json.loads(json.dumps(asdict(config))) == provenance["config"]
    out.mkdir(parents=True, exist_ok=False)
    edges = np.arange(0, 362, 2)
    result = {"definition": "airport-reference-point fixed tangent ENU; east 0, north 90, west 180, south 270; ground track",
              "source_extraction_sha256": sha(analysis / "extraction.json"),
              "flight_audit_sha256": sha(analysis / "flight_audit.csv"),
              "script_sha256": sha(Path(__file__)), "cohort_sha256": provenance["cohort_sha256"],
              "edges_deg": edges.tolist(), "airports": {}, "groups": {},
              "source_hashes": {str(p.relative_to(REPO)): sha(p) for p in [
                  REPO / "4dTrajectory/ts_transformer/data/coordinate_frames.py",
                  REPO / "4dTrajectory/ts_transformer/data/channels.py",
                  REPO / "flight_scenarios/runway_target.py",
                  REPO / "trajectory_data_process/config/runway_thresholds.json"]}}
    pooled = {k: [] for k in ("raw", "smoothed", "relative", "weight", "split")}
    checks = []
    started = time.monotonic()
    for airport, source in provenance["airports"].items():
        manifest = REPO / source["manifest"]
        assert sha(manifest) == source["manifest_sha256"]
        selected = sorted(key for key in roster if key.startswith(airport + ":"))
        reference = airport_reference_point(airport)
        arrays = {k: [] for k in pooled}
        max_difference = 0.
        for first in range(0, len(selected), 400):
            batch = selected[first:first + 400]
            flights = load_flight_dicts(manifest, include_flight_keys=set(batch), verbose=False)
            built, _ = build_series(flights, config, aircraft_type=config.aircraft_type)
            by_id = {item.dataset_id: item for item in built}
            assert set(by_id) == set(batch)
            for key in batch:
                item = by_id[key]
                count = len(item.times)
                assert count == int(roster[key]["rows"])
                assert np.allclose(np.diff(item.times), config.dt_s)
                raw, chart = absolute_angles(item, reference)
                smoothed = smooth(np.degrees(np.unwrap(np.radians(raw))), min_rows(6., config.dt_s)) % 360
                frame = course_frame(item)
                relative = smooth(frame["course_unwrapped_deg"], min_rows(6., config.dt_s)) % 360
                max_difference = max(max_difference, float(np.max(np.abs((raw - chart + 180) % 360 - 180))))
                assert np.isfinite(raw).all() and np.isfinite(smoothed).all()
                arrays["raw"].append(raw); arrays["smoothed"].append(smoothed)
                arrays["relative"].append(relative)
                arrays["weight"].append(np.full(count, 1. / count))
                arrays["split"].append(np.full(count, 0 if roster[key]["split"] == "train" else 1, dtype=np.int8))
                checks.append([key, roster[key]["split"], count])
            print(f"{airport}: {min(first+400, len(selected))}/{len(selected)} flights; {time.monotonic()-started:.0f}s", flush=True)
        arrays = {k: np.concatenate(parts) for k, parts in arrays.items()}
        result["airports"][airport] = {"reference": reference, "flights": len(selected),
                                       "max_tangent_vs_chart_angle_difference_deg": max_difference}
        for split_index, split in enumerate(("train", "val")):
            mask = arrays["split"] == split_index
            group = {"flights": int(round(arrays["weight"][mask].sum())), "rows": int(mask.sum()),
                     "signals": {kind: describe(arrays[kind][mask], arrays["weight"][mask], edges)
                                 for kind in ("raw", "smoothed", "relative")}}
            expected = old_summary["groups"][f"{airport}/{split}"]
            assert group["flights"] == expected["flights"] and group["rows"] == expected["rows"]
            result["groups"][f"{airport}/{split}"] = group
        for kind in pooled:
            pooled[kind].append(arrays[kind])
    pooled = {k: np.concatenate(parts) for k, parts in pooled.items()}
    for split_index, split in enumerate(("train", "val")):
        mask = pooled["split"] == split_index
        result["groups"][f"POOLED/{split}"] = {
            "flights": int(round(pooled["weight"][mask].sum())), "rows": int(mask.sum()),
            "signals": {kind: describe(pooled[kind][mask], pooled["weight"][mask], edges)
                        for kind in ("raw", "smoothed", "relative")}}
    write_json(out / "summary.json", result)
    with (out / "flight_audit.csv").open("w") as file:
        writer = csv.writer(file); writer.writerow(["flight_id", "split", "rows"]); writer.writerows(checks)
    with (out / "histograms.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "signal", "left_deg", "right_deg", "rows", "flight_weight"])
        for key, group in result["groups"].items():
            for kind, signal in group["signals"].items():
                writer.writerows(zip([key] * 180, [kind] * 180, edges[:-1], edges[1:],
                                     signal["histogram_counts"], signal["histogram_flight_weight"]))
    plot(result, out)


def plot(result: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.array(result["edges_deg"])
    names = ["POOLED", *result["airports"]]
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        for split, color in (("train", "#2266aa"), ("val", "#df7e22")):
            group = result["groups"][f"{name}/{split}"]
            mass = np.array(group["signals"]["smoothed"]["histogram_flight_weight"]) / group["flights"] * 100
            ax.stairs(mass, edges, color=color, label=f"{split}: {group['flights']:,} flights")
        ax.set(title=name, xlim=(0, 360), xlabel="Absolute ground-track angle (deg, airport ENU)",
               ylabel="Flight-balanced mass per 2 deg bin (%)")
        ax.set_xticks([0, 90, 180, 270, 360], ["0 E", "90 N", "180 W", "270 S", "360 E"])
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Absolute direction distributions | airport-centred ENU | 6 s smoothing | same development flights", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"absolute_heading.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), subplot_kw={"projection": "polar"}, constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        group = result["groups"][f"{name}/train"]
        for kind, color, label in (("smoothed", "#2266aa", "Absolute (airport ENU)"),
                                   ("relative", "#ad523a", "Relative (runway = 0)")):
            mass = np.array(group["signals"][kind]["histogram_flight_weight"]) / group["flights"] * 100
            centres = np.radians((edges[:-1] + edges[1:]) / 2)
            ax.plot(np.r_[centres, centres[0]+2*np.pi], np.r_[mass, mass[0]], color=color, label=label, linewidth=1.2)
        ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
        ax.set_xticks(np.radians([0, 90, 180, 270]), ["0", "90", "180", "270"])
        ax.set_title(name); ax.set_rlabel_position(45)
    axes.flat[0].legend(loc="upper left", bbox_to_anchor=(-.35, 1.25), fontsize=8)
    fig.suptitle("Effect of direction reference | train | each radius = mass per 2 deg bin (%)", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"absolute_vs_relative.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.analysis.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
