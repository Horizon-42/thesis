"""Summarise development signal caches and draw standalone distribution figures.

This is a descriptive analysis, not a model evaluation or a test release.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ts_transformer.experiments.measure_instruction_distribution import sha, write_json

KINDS = ("heading_deg", "speed_mps", "height_m", "cross_m")
QUANTILES = (0, .1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100)
EDGES = (np.arange(-180, 182, 2), np.arange(30, 252, 2), np.arange(-150, 6025, 25), np.arange(-30000, 30100, 100))
LABELS = ("Relative ground-track angle (deg)", "Ground speed (m/s)", "Height above runway target (m)", "Signed cross-track distance (m; right +)")


def distribution(x: np.ndarray, weight: np.ndarray, edges: np.ndarray) -> dict:
    order = np.argsort(x)
    cumulative = np.cumsum(weight[order])
    weighted = np.interp(np.asarray(QUANTILES) / 100 * cumulative[-1], cumulative, x[order])
    return {
        "rows": len(x), "quantiles": dict(zip(map(str, QUANTILES), np.percentile(x, QUANTILES).tolist())),
        "flight_weighted_quantiles": dict(zip(map(str, QUANTILES), weighted.tolist())),
        "histogram_counts": np.histogram(x, edges)[0].tolist(),
        "histogram_flight_weight": np.histogram(x, edges, weights=weight)[0].tolist(),
        "below_histogram": int((x < edges[0]).sum()), "above_histogram": int((x > edges[-1]).sum()),
    }


def spans(mask: np.ndarray) -> list[tuple[int, int]]:
    change = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def load_clean(cache: Path, airport: str) -> dict:
    with np.load(cache / f"{airport}.npz", allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    result["row_split"] = np.repeat(result["split"], np.diff(result["offsets"]))
    result["weight"] = np.repeat(1. / np.diff(result["offsets"]), np.diff(result["offsets"]))
    keep = np.ones(len(result["split"]), dtype=bool)
    refused = []
    plateaus = []
    for i, (a, b) in enumerate(zip(result["offsets"][:-1], result["offsets"][1:])):
        speed = result["read"][a:b, 1]
        assert np.isfinite(result["read"][a:b]).all()
        if speed.min() < 30 or speed.max() > 250:
            keep[i] = False
            refused.append({"id": str(result["flight_ids"][i]), "split": ("train", "val")[result["split"][i]],
                            "speed_min": float(speed.min()), "speed_max": float(speed.max())})
            continue
        h = result["read"][a:b, 2]
        t = result["time"][a:b]
        vertical_rate = np.gradient(h, t)
        for threshold in (.25, .5, 1.):
            for lo, hi in spans(np.abs(vertical_rate) <= threshold):
                if t[hi - 1] - t[lo] >= 10.:
                    plateaus.append({"airport": airport, "split": ("train", "val")[result["split"][i]],
                                     "flight_id": str(result["flight_ids"][i]), "rate_threshold_mps": threshold,
                                     "start_s": float(t[lo]), "end_s": float(t[hi - 1]),
                                     "median_height_m": float(np.median(h[lo:hi]))})
    result["keep_flight"] = keep
    result["keep"] = np.repeat(keep, np.diff(result["offsets"]))
    result["refused"] = refused
    result["plateaus"] = plateaus
    return result


def summarise(cache: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    extraction = json.loads((cache / "extraction.json").read_text())
    airports = list(extraction["airports"])
    assert len(airports) == 5, "wait for the complete extraction"
    summary = {"extraction_sha256": sha(cache / "extraction.json"),
               "cohort_sha256": extraction["cohort_sha256"], "script_sha256": sha(Path(__file__)),
               "quantile_percentages": QUANTILES, "histogram_edges": dict(zip(KINDS, [e.tolist() for e in EDGES])),
               "groups": {}, "before_speed_filter": {}, "refused": [], "plateaus": {}}
    pooled = {name: [] for name in ("raw", "read", "weight", "split", "airport")}
    all_plateaus = []
    audit_rows = []
    for airport_index, airport in enumerate(airports):
        assert sha(cache / f"{airport}.npz") == extraction["airports"][airport]["cache_sha256"]
        data = load_clean(cache, airport)
        summary["before_speed_filter"][airport] = {
            split: {"flights": int((data["split"] == split_index).sum()),
                    "signals": {kind: distribution(data["read"][data["row_split"] == split_index, j],
                                                   data["weight"][data["row_split"] == split_index], EDGES[j])
                                for j, kind in enumerate(KINDS)}}
            for split_index, split in enumerate(("train", "val"))}
        summary["refused"].extend(data["refused"])
        all_plateaus.extend(data["plateaus"])
        keep = data["keep"]
        for split_index, split in enumerate(("train", "val")):
            selected = keep & (data["row_split"] == split_index)
            n_flights = int((data["keep_flight"] & (data["split"] == split_index)).sum())
            group = {"flights": n_flights, "rows": int(selected.sum()), "signals": {}}
            for source in ("raw", "read"):
                group["signals"][source] = {kind: distribution(data[source][selected, j], data["weight"][selected], EDGES[j])
                                             for j, kind in enumerate(KINDS)}
            summary["groups"][f"{airport}/{split}"] = group
        for name in ("raw", "read", "weight"):
            pooled[name].append(data[name][keep])
        pooled["split"].append(data["row_split"][keep])
        pooled["airport"].append(np.full(int(keep.sum()), airport_index, dtype=np.int8))
        for i, key in enumerate(data["flight_ids"]):
            a, b = data["offsets"][i:i + 2]
            audit_rows.append([key, ("train", "val")[data["split"][i]], bool(data["keep_flight"][i]), int(b-a)])
        print(f"{airport}: {int(data['keep_flight'].sum())} retained, {len(data['refused'])} out-of-range flights", flush=True)
    pooled = {name: np.concatenate(parts) for name, parts in pooled.items()}
    for split_index, split in enumerate(("train", "val")):
        selected = pooled["split"] == split_index
        weight = pooled["weight"][selected]
        group = {"flights": int(round(weight.sum())), "rows": int(selected.sum()), "signals": {}}
        for source in ("raw", "read"):
            group["signals"][source] = {kind: distribution(pooled[source][selected, j], weight, EDGES[j])
                                         for j, kind in enumerate(KINDS)}
        x = pooled["read"][selected]
        group["shares_flight_weighted"] = {
            "heading_abs_le_1": float(np.average(np.abs(x[:, 0]) <= 1, weights=weight)),
            "heading_abs_le_10": float(np.average(np.abs(x[:, 0]) <= 10, weights=weight)),
            "heading_abs_gt_45": float(np.average(np.abs(x[:, 0]) > 45, weights=weight)),
            "heading_abs_ge_150": float(np.average(np.abs(x[:, 0]) >= 150, weights=weight)),
            "speed_60_to_140": float(np.average((x[:, 1] >= 60) & (x[:, 1] <= 140), weights=weight)),
            "height_below_zero": float(np.average(x[:, 2] < 0, weights=weight)),
            "height_le_300": float(np.average(x[:, 2] <= 300, weights=weight)),
            "height_gt_2000": float(np.average(x[:, 2] > 2000, weights=weight)),
            "cross_left": float(np.average(x[:, 3] < 0, weights=weight)),
            **{f"cross_abs_le_{limit}": float(np.average(np.abs(x[:, 3]) <= limit, weights=weight))
               for limit in (25, 50, 100, 250, 500, 1000, 5000, 10000, 20000)},
            "heading_abs_le_1_but_cross_abs_gt_250": float(np.average((np.abs(x[:, 0]) <= 1) & (np.abs(x[:, 3]) > 250), weights=weight)),
        }
        fine_edges = np.arange(-1000, 1010, 10)
        group["cross_fine"] = {"edges": fine_edges.tolist(), "flight_weight": np.histogram(x[:, 3], fine_edges, weights=weight)[0].tolist()}
        joint, cross_edges, heading_edges = np.histogram2d(x[:, 3], x[:, 0], bins=(np.arange(-30000, 30500, 500), np.arange(-180, 185, 5)), weights=weight)
        group["cross_heading"] = {"cross_edges": cross_edges.tolist(), "heading_edges": heading_edges.tolist(), "flight_weight": joint.tolist()}
        summary["groups"][f"POOLED/{split}"] = group
    for split in ("train", "val"):
        summary["plateaus"][split] = {}
        for threshold in (.25, .5, 1.):
            rows = [r for r in all_plateaus if r["split"] == split and r["rate_threshold_mps"] == threshold]
            x = np.array([r["median_height_m"] for r in rows])
            summary["plateaus"][split][str(threshold)] = {
                "runs": len(rows), "flights": len({r["flight_id"] for r in rows}),
                "quantiles": dict(zip(map(str, QUANTILES), np.percentile(x, QUANTILES).tolist())),
                "histogram_counts": np.histogram(x, EDGES[2])[0].tolist(),
            }
    write_json(out / "distribution_summary.json", summary)
    for name, rows, header in [
        ("flight_audit.csv", audit_rows, ["flight_id", "split", "retained_in_distribution", "rows"]),
        ("level_segments.csv", [list(r.values()) for r in all_plateaus], list(all_plateaus[0])),
    ]:
        with (out / name).open("w") as file:
            writer = csv.writer(file); writer.writerow(header); writer.writerows(rows)
    with (out / "histograms.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "signal", "kind", "left", "right", "rows", "flight_weight"])
        for group_name, group in summary["groups"].items():
            for source, signals in group["signals"].items():
                for j, kind in enumerate(KINDS):
                    block = signals[kind]
                    writer.writerows([group_name, source, kind, float(lo), float(hi), count, weight]
                                     for lo, hi, count, weight in zip(EDGES[j][:-1], EDGES[j][1:],
                                                                     block["histogram_counts"], block["histogram_flight_weight"]))
    plot(summary, out)
    assess_candidates(pooled, airports, out)
    print(json.dumps({k: {"flights": v["flights"], "rows": v["rows"],
                          "quantiles": {kind: v["signals"]["read"][kind]["flight_weighted_quantiles"] for kind in KINDS},
                          "shares": v["shares_flight_weighted"]}
                      for k, v in summary["groups"].items() if k.startswith("POOLED")}, indent=2), flush=True)


def assess_candidates(pooled: dict, airports: list[str], out: Path) -> None:
    """Descriptive quantisation audit of explicit proposals, not a tokenizer or rollout."""
    positive_heading = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                        33, 36, 39, 43, 45, 47, 51, 56, 61, 67, 73, 80, 88, 90,
                        96, 105, 115, 126, 135, 138, 151, 165]
    positive_cross = [25, 50, 75, 100, 150, 250, 500, 750, 1000, 1500, 2000, 3000,
                      4000, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 30000]
    candidates = {
        "heading_deg": sorted([-180, 0] + positive_heading + [-v for v in positive_heading]),
        "speed_mps": [30, 33, 36, 39, 43, 47, 51, 55, 60, 65, 70, 75, 80, 85, 90,
                      95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 150, 160, 175,
                      190, 205, 225, 245, 250],
        "height_m": [-150, -100, -75, -50, -25, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80,
                     90, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 475, 500, 550,
                     575, 600, 625, 650, 700, 750, 775, 800, 825, 850, 900, 925, 950,
                     1000, 1050, 1100, 1125, 1150, 1200, 1300, 1400, 1500, 1600, 1800,
                     2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
        "cross_m": sorted([0] + positive_cross + [-v for v in positive_cross]),
    }
    # Analytical interval-union checks, including the heading seam; not a sampled grid.
    angle = np.array(candidates["heading_deg"])
    delta = np.maximum(1, .05 * np.abs(angle))
    assert np.all(np.diff(np.r_[angle, angle[0] + 360]) <= delta + np.roll(delta, -1))
    speed = np.array(candidates["speed_mps"])
    assert speed[0] * .95 <= 30 and speed[-1] * 1.05 >= 250
    assert np.all(speed[:-1] * 1.05 >= speed[1:] * .95)
    result = {"status": "proposal_for_user_review_not_adopted", "targets": candidates,
              "counts": {k: len(v) for k, v in candidates.items()},
              "full_domain_interval_coverage": {"heading_deg": True, "speed_mps": True},
              "groups": {}}
    for kind_index, (kind, targets) in enumerate(candidates.items()):
        t = np.array(targets)
        x = pooled["read"][:, kind_index]
        index = np.searchsorted(t, x)
        left = np.clip(index - 1, 0, len(t) - 1)
        right = np.clip(index, 0, len(t) - 1)
        distance = lambda value: np.abs((x - value + 180) % 360 - 180) if kind == "heading_deg" else np.abs(x - value)
        if kind == "heading_deg":
            left = (index - 1) % len(t); right = index % len(t)
        dl, dr = distance(t[left]), distance(t[right])
        chosen = np.where(dl <= dr, left, right)
        error = np.minimum(dl, dr)
        if kind == "height_m":
            chosen = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)
            error = x - t[chosen]  # Signed gap above nearest lower target, NOT nearest rounding.
            assert np.all(error >= 0)
        if kind == "heading_deg":
            inside = (dl <= np.maximum(1, .05 * np.abs(t[left]))) | (dr <= np.maximum(1, .05 * np.abs(t[right])))
        elif kind == "speed_mps":
            inside = (dl <= .05 * t[left]) | (dr <= .05 * t[right])
        else:
            inside = (x >= t[0]) & (x <= t[-1])  # Range only; no envelope assumed.
        for split_index, split in enumerate(("train", "val")):
            for airport_index in range(-1, len(airports)):
                mask = pooled["split"] == split_index
                name = "POOLED" if airport_index == -1 else airports[airport_index]
                if airport_index >= 0:
                    mask &= pooled["airport"] == airport_index
                values, weight = error[mask], pooled["weight"][mask]
                order = np.argsort(values)
                quantiles = np.interp(np.array([50, 95, 99]) / 100 * weight.sum(), np.cumsum(weight[order]), values[order])
                result["groups"].setdefault(f"{name}/{split}", {})[kind] = {
                    "error_p50_p95_p99": quantiles.tolist(), "mean_error": float(np.average(values, weights=weight)),
                    "maximum_error": float(values.max()), "covered_share": float(np.average(inside[mask], weights=weight)),
                    "assigned_mass": (np.bincount(chosen[mask], weights=weight, minlength=len(t)) / weight.sum()).tolist(),
                }
    result["height_level_runs"] = {}
    for split in ("train", "val"):
        with (out / "level_segments.csv").open() as file:
            rows = [r for r in csv.DictReader(file) if r["split"] == split and float(r["rate_threshold_mps"]) == .5]
        x = np.array([float(r["median_height_m"]) for r in rows])
        t = np.array(candidates["height_m"])
        gap = x - t[np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)]
        result["height_level_runs"][split] = {"runs": len(x), "lower_target_gap_p50_p95_p99": np.percentile(gap, [50, 95, 99]).tolist()}
    write_json(out / "candidate_targets.json", result)


def plot(summary: dict, out: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    for j, kind in enumerate(KINDS):
        for split, color in (("train", "#2266aa"), ("val", "#df7e22")):
            group = summary["groups"][f"POOLED/{split}"]
            values = np.asarray(group["signals"]["read"][kind]["histogram_flight_weight"]) / group["flights"] * 100
            for row in (0, 1):
                axes[row, j].stairs(values, EDGES[j], label=split, color=color, linewidth=1.5)
        for row in (0, 1):
            axes[row, j].set(xlabel=LABELS[j], ylabel="Flight-balanced mass per bin (%)")
            axes[row, j].grid(alpha=.2)
        axes[0, j].legend()
    axes[0, 0].set_xlim(-180, 180); axes[1, 0].set_xlim(-20, 20)
    axes[0, 1].set_xlim(30, 250); axes[1, 1].set_xlim(50, 160)
    axes[0, 2].set_xlim(-150, 6000); axes[1, 2].set_xlim(-100, 500)
    axes[0, 3].set_xlim(-30000, 30000); axes[1, 3].set_xlim(-1000, 1000)
    fig.suptitle("Development distributions | angle/speed/height smoothed; cross-track geometric | equal weight per flight")
    fig.savefig(out / "distributions.png", dpi=180)
    fig.savefig(out / "distributions.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    for key, group in summary["groups"].items():
        if key.startswith("POOLED") or not key.endswith("/train"):
            continue
        for j, kind in enumerate(KINDS):
            mass = np.array(group["signals"]["read"][kind]["histogram_flight_weight"]) / group["flights"]
            axes[j].plot(EDGES[j][1:], np.cumsum(mass), label=key.split("/")[0])
    for j, ax in enumerate(axes):
        ax.set(xlabel=LABELS[j], ylabel="Flight-balanced cumulative share")
        ax.grid(alpha=.2); ax.legend()
    axes[2].set_xlim(-150, 3500)
    fig.suptitle("Airport differences | train only | equal weight per flight within each airport")
    fig.savefig(out / "airport_cdfs.png", dpi=180)
    fig.savefig(out / "airport_cdfs.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for split in ("train", "val"):
        group = summary["groups"][f"POOLED/{split}"]
        fine = group["cross_fine"]
        axes[0].stairs(np.array(fine["flight_weight"]) / group["flights"] * 100, fine["edges"], label=split)
    axes[0].set(xlabel=LABELS[3], ylabel="Flight-balanced mass per 10 m bin (%)", xlim=(-500, 500))
    axes[0].legend(); axes[0].grid(alpha=.2)
    group = summary["groups"]["POOLED/train"]
    joint = group["cross_heading"]
    mass = np.array(joint["flight_weight"]).T / group["flights"] * 100
    from matplotlib.colors import LogNorm
    mesh = axes[1].pcolormesh(joint["cross_edges"], joint["heading_edges"], np.ma.masked_equal(mass, 0), norm=LogNorm(), cmap="viridis")
    axes[1].set(xlabel=LABELS[3], ylabel=LABELS[0])
    fig.colorbar(mesh, ax=axes[1], label="Flight-balanced mass per cell (%)")
    fig.suptitle("Signed runway-centreline position | near-centre detail and joint distribution (train)")
    fig.savefig(out / "signed_cross_track.png", dpi=180)
    fig.savefig(out / "signed_cross_track.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for threshold in (.25, .5, 1.):
        block = summary["plateaus"]["train"][str(threshold)]
        mass = np.array(block["histogram_counts"]) / block["runs"] * 100
        for ax in axes:
            ax.stairs(mass, EDGES[2], label=f"|vertical rate| <= {threshold:g} m/s; n={block['runs']:,}")
            ax.set(xlabel=LABELS[2], ylabel="Share of approximately level runs (%)")
            ax.grid(alpha=.2)
    axes[0].set_xlim(-150, 3500); axes[1].set_xlim(-100, 500)
    axes[0].legend(fontsize=8)
    fig.suptitle("Approximately level segments | duration >= 10 s | train only | one median per run")
    fig.savefig(out / "level_heights.png", dpi=180)
    fig.savefig(out / "level_heights.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarise(args.cache, args.out)


if __name__ == "__main__":
    main()
