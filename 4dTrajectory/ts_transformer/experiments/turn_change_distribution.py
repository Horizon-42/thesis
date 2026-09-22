"""Measure signed turns over multiple horizons on the existing development flights."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from ts_transformer.experiments.absolute_heading_distribution import absolute_angles
from ts_transformer.experiments.measure_instruction_distribution import REPO, sha, write_json

HORIZONS = (2, 10, 30, 60, 120)
SIGNED_EDGES = np.arange(-360., 361., 1.)
ZOOM_EDGES = np.linspace(-10., 10., 201)
ABS_EDGES = np.arange(0., 361., 1.)
PERCENTILES = (50, 75, 90, 95, 99, 99.9)


def window_turns(unwrapped: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Net signed turn and total angular travel, excluding incomplete windows."""
    cumulative = np.r_[0., np.cumsum(np.abs(np.diff(unwrapped)))]
    return unwrapped[steps:] - unwrapped[:-steps], cumulative[steps:] - cumulative[:-steps]


def extract(analysis: Path, cache: Path) -> None:
    from flight_scenarios.runway_target import airport_reference_point
    from ts_transformer.config import TSConfig
    from ts_transformer.data.dataset import build_series, load_flight_dicts
    from ts_transformer.data.development_cohorts import load_development_cohort
    from ts_transformer.manoeuvre.instructions import min_rows, smooth

    prior = json.loads((analysis / "extraction.json").read_text())
    absolute = json.loads((analysis / "absolute_heading/summary.json").read_text())
    for name, expected in {**prior["source_hashes"], **absolute["source_hashes"]}.items():
        assert sha(REPO / name) == expected, name
    cohort_path = REPO / prior["cohort_path"]
    assert sha(cohort_path) == prior["cohort_sha256"]
    cohort = load_development_cohort(cohort_path)
    allowed = {"train": set(cohort.train_flight_ids), "val": set(cohort.val_flight_ids)}
    assert not allowed["train"] & allowed["val"]
    with (analysis / "flight_audit.csv").open() as f:
        roster = {r["flight_id"]: r for r in csv.DictReader(f) if r["retained_in_distribution"] == "True"}
    assert all(key in allowed[r["split"]] for key, r in roster.items())
    config = TSConfig.from_dict(prior["config"])
    assert config.dt_s == 2.
    cache.mkdir(parents=True, exist_ok=False)
    meta = {"source_extraction_sha256": sha(analysis / "extraction.json"),
            "source_flight_audit_sha256": sha(analysis / "flight_audit.csv"),
            "cohort_sha256": prior["cohort_sha256"], "dt_s": config.dt_s,
            "angle_definition": absolute["definition"], "airports": {},
            "absolute_angle_helper_sha256": sha(REPO / "4dTrajectory/ts_transformer/experiments/absolute_heading_distribution.py"),
            "script_sha256_at_extraction": sha(Path(__file__))}
    started = time.monotonic()
    for airport, source in prior["airports"].items():
        manifest = REPO / source["manifest"]
        assert sha(manifest) == source["manifest_sha256"]
        keys = sorted(k for k in roster if k.startswith(airport + ":"))
        raw_rows, smooth_rows, offsets, splits = [], [], [0], []
        reference = airport_reference_point(airport)
        for first in range(0, len(keys), 400):
            batch = keys[first:first+400]
            flights = load_flight_dicts(manifest, include_flight_keys=set(batch), verbose=False)
            built, _ = build_series(flights, config, aircraft_type=config.aircraft_type)
            by_id = {s.dataset_id: s for s in built}
            assert set(by_id) == set(batch)
            for key in batch:
                item = by_id[key]
                assert len(item.times) == int(roster[key]["rows"])
                assert np.allclose(np.diff(item.times), config.dt_s)
                angle, _ = absolute_angles(item, reference)
                unwrapped = np.degrees(np.unwrap(np.radians(angle)))
                raw_rows.append(unwrapped)
                smooth_rows.append(smooth(unwrapped, min_rows(6., config.dt_s)))
                offsets.append(offsets[-1] + len(angle))
                splits.append(0 if roster[key]["split"] == "train" else 1)
            print(f"{airport}: {min(first+400,len(keys))}/{len(keys)} flights; {time.monotonic()-started:.0f}s", flush=True)
        np.savez_compressed(cache / f"{airport}.npz", raw=np.concatenate(raw_rows),
                            smoothed=np.concatenate(smooth_rows), offsets=np.array(offsets),
                            flight_ids=np.array(keys), split=np.array(splits, dtype=np.int8))
        meta["airports"][airport] = {"reference": reference, "flights": len(keys), "rows": offsets[-1],
                                    "cache_sha256": sha(cache / f"{airport}.npz")}
        write_json(cache / "extraction.json", meta)


def describe(net: np.ndarray, travel: np.ndarray, weight: np.ndarray) -> dict:
    absolute = np.abs(net)
    order = np.argsort(absolute)
    percentiles = np.interp(np.array(PERCENTILES) / 100 * weight.sum(), np.cumsum(weight[order]), absolute[order])
    average = lambda condition: float(np.average(condition, weights=weight))
    return {
        "pairs": len(net), "flights": int(round(weight.sum())),
        "abs_quantiles_flight_weighted": dict(zip(map(str, PERCENTILES), percentiles.tolist())),
        "abs_quantiles_point_weighted": dict(zip(map(str, PERCENTILES), np.percentile(absolute, PERCENTILES).tolist())),
        "net_min_deg": float(net.min()), "net_max_deg": float(net.max()),
        "shares": {"within_1_deg": average(absolute <= 1), "within_5_deg": average(absolute <= 5),
                   "left_over_5_deg": average(net > 5), "right_over_5_deg": average(net < -5),
                   "over_30_deg": average(absolute > 30), "over_90_deg": average(absolute > 90),
                   "over_180_deg": average(absolute > 180), "over_360_deg": average(absolute > 360),
                   "net_within_5_but_travel_ge_30": average((absolute <= 5) & (travel >= 30))},
        "mean_abs_net_deg": average(absolute), "mean_total_travel_deg": average(travel),
        "signed_histogram_counts": np.histogram(net, SIGNED_EDGES)[0].tolist(),
        "signed_histogram_flight_weight": np.histogram(net, SIGNED_EDGES, weights=weight)[0].tolist(),
        "zoom_histogram_flight_weight": np.histogram(net, ZOOM_EDGES, weights=weight)[0].tolist(),
        "abs_histogram_flight_weight": np.histogram(absolute, ABS_EDGES, weights=weight)[0].tolist(),
        "wrapped_histogram_flight_weight": np.histogram((net+180)%360-180, np.arange(-180,181), weights=weight)[0].tolist(),
    }


def summarise(cache: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((cache / "extraction.json").read_text())
    result = {"extraction": meta, "script_sha256": sha(Path(__file__)), "horizons_s": HORIZONS,
              "main_signal": "6 s centred smoothing of unwrapped airport ENU ground-track angle",
              "signed_edges_deg": SIGNED_EDGES.tolist(), "zoom_edges_deg": ZOOM_EDGES.tolist(),
              "abs_edges_deg": ABS_EDGES.tolist(), "groups": {}}
    airports = list(meta["airports"])
    data = {}
    audit = []
    for airport in airports:
        assert sha(cache / f"{airport}.npz") == meta["airports"][airport]["cache_sha256"]
        with np.load(cache / f"{airport}.npz", allow_pickle=False) as f:
            data[airport] = {k: f[k] for k in f.files}
        d = data[airport]
        for i, key in enumerate(d["flight_ids"]):
            n = int(d["offsets"][i+1]-d["offsets"][i])
            audit.append([key, ("train", "val")[d["split"][i]], n,
                          *[max(0, n-int(h/meta["dt_s"])) for h in HORIZONS]])
    with (out / "flight_audit.csv").open("w") as f:
        writer = csv.writer(f); writer.writerow(["flight_id", "split", "rows", *[f"pairs_{h}s" for h in HORIZONS]])
        writer.writerows(audit)
    for horizon in HORIZONS:
        collected = {k: [] for k in ("raw_net", "raw_travel", "smoothed_net", "smoothed_travel", "weight",
                                    "common_weight", "split", "airport")}
        steps, common_steps = int(horizon/meta["dt_s"]), int(max(HORIZONS)/meta["dt_s"])
        for airport_index, airport in enumerate(airports):
            d = data[airport]
            for i, (a,b) in enumerate(zip(d["offsets"][:-1], d["offsets"][1:])):
                count = b-a-steps
                if count <= 0:
                    continue
                for source in ("raw", "smoothed"):
                    net, travel = window_turns(d[source][a:b], steps)
                    collected[source+"_net"].append(net)
                    collected[source+"_travel"].append(travel)
                collected["weight"].append(np.full(count, 1./count))
                common_count = max(0, b-a-common_steps)
                common_weight = np.zeros(count)
                if common_count:
                    common_weight[:common_count] = 1./common_count
                collected["common_weight"].append(common_weight)
                collected["split"].append(np.full(count, d["split"][i], dtype=np.int8))
                collected["airport"].append(np.full(count, airport_index, dtype=np.int8))
        arrays = {k: np.concatenate(v) for k,v in collected.items()}
        for split_index, split in enumerate(("train", "val")):
            for airport_index in range(-1, len(airports)):
                mask = arrays["split"] == split_index
                airport = "POOLED" if airport_index == -1 else airports[airport_index]
                if airport_index >= 0:
                    mask &= arrays["airport"] == airport_index
                group = {}
                for source in ("raw", "smoothed"):
                    group[source] = describe(arrays[source+"_net"][mask], arrays[source+"_travel"][mask], arrays["weight"][mask])
                common = mask & (arrays["common_weight"] > 0)
                group["common_starts_smoothed"] = describe(arrays["smoothed_net"][common], arrays["smoothed_travel"][common], arrays["common_weight"][common])
                result["groups"].setdefault(f"{airport}/{split}", {})[str(horizon)] = group
        print(f"Summarised {horizon}s windows", flush=True)
    write_json(out / "summary.json", result)
    with (out / "histograms.csv").open("w") as f:
        writer = csv.writer(f); writer.writerow(["group", "horizon_s", "source", "left_deg", "right_deg", "pairs", "flight_weight"])
        for key, horizons in result["groups"].items():
            for horizon, sources in horizons.items():
                for source, group in sources.items():
                    writer.writerows(zip([key]*720, [horizon]*720, [source]*720, SIGNED_EDGES[:-1], SIGNED_EDGES[1:],
                                         group["signed_histogram_counts"], group["signed_histogram_flight_weight"]))
    plot(result, out)


def plot(result: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        for extension in ("png", "svg"):
            fig.savefig(out / f"{name}.{extension}", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, len(HORIZONS), figsize=(20,7), constrained_layout=True)
    for j,h in enumerate(HORIZONS):
        for split,color in (("train", "#2266aa"),("val", "#df7e22")):
            g = result["groups"][f"POOLED/{split}"][str(h)]["smoothed"]
            for row, edges, field in ((0,SIGNED_EDGES,"signed"),(1,ZOOM_EDGES,"zoom")):
                mass = np.array(g[field+"_histogram_flight_weight"])/g["flights"]*100
                axes[row,j].stairs(mass, edges, label=split, color=color)
        axes[0,j].set(title=f"{h} s", xlim=(-360,360), yscale="log", ylim=(.0001,100), ylabel="Mass per 1 deg bin (%)")
        axes[1,j].set(xlim=(-10,10), ylabel="Mass per 0.1 deg bin (%)")
        for ax in axes[:,j]:
            ax.grid(alpha=.2); ax.set_xlabel("Net turn (deg; left +, right -)")
        axes[0,j].legend(fontsize=8)
    fig.suptitle("Turn relative to each window's initial direction | complete overlapping windows | equal weight per flight")
    save(fig,"turn_distributions")
    fig, axes = plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for q,color in (("50","#339966"),("95","#2266aa"),("99","#b54455")):
        for split,style in (("train","-"),("val","--")):
            y = [result["groups"][f"POOLED/{split}"][str(h)]["smoothed"]["abs_quantiles_flight_weighted"][q] for h in HORIZONS]
            axes[0,0].plot(HORIZONS,y,style,marker="o",color=color,label=f"P{q} {split}")
    for source, label in (("raw","Before extra smoothing"),("smoothed","6 s smoothing"),("common_starts_smoothed","6 s; same starts at every horizon")):
        y=[result["groups"]["POOLED/train"][str(h)][source]["abs_quantiles_flight_weighted"]["95"] for h in HORIZONS]
        axes[0,1].plot(HORIZONS,y,marker="o",label=label)
        axes[1,0].plot(HORIZONS,np.array(y)/HORIZONS,marker="o",label=label)
    for field,label in (("within_5_deg","Net change <= 5 deg"),("left_over_5_deg","Left > 5 deg"),("right_over_5_deg","Right > 5 deg")):
        y=[result["groups"]["POOLED/train"][str(h)]["smoothed"]["shares"][field]*100 for h in HORIZONS]
        axes[1,1].plot(HORIZONS,y,marker="o",label=label)
    for ax in axes.flat:
        ax.set_xlabel("Horizon (s)"); ax.set_xticks(HORIZONS); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0,0].set_ylabel("Absolute net turn (deg)")
    axes[0,1].set_ylabel("P95 absolute net turn (deg)")
    axes[1,0].set_ylabel("P95 absolute net turn / horizon (deg/s)")
    axes[1,1].set_ylabel("Flight-balanced share (%)")
    fig.suptitle("Time scale, smoothing and start-position effects | pooled development flights")
    save(fig,"turn_horizon_comparison")
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,airport in zip(axes.flat,["POOLED",*result["extraction"]["airports"]]):
        for h in HORIZONS:
            g=result["groups"][airport+"/train"][str(h)]["smoothed"]
            ax.plot(ABS_EDGES[1:],np.cumsum(g["abs_histogram_flight_weight"])/g["flights"]*100,label=f"{h} s")
        ax.set(title=airport,xlim=(0,180),ylim=(0,100),xlabel="Absolute net turn (deg)",ylabel="Flight-balanced cumulative share (%)")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Turn magnitude by airport | train | 1 deg CDF bins; values > 180 deg remain in denominator")
    save(fig,"turn_airport_cdfs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summarise-only", action="store_true")
    args=parser.parse_args()
    if not args.summarise_only:
        extract(args.analysis.resolve(),args.cache.resolve())
    summarise(args.cache.resolve(),args.out.resolve())


if __name__ == "__main__":
    main()
