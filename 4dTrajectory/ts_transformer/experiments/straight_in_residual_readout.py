"""Where does the straight-in residual live: along the track, across it, or in the vertical?

The control models fly the straight-in stratum laterally right (chamfer ~100 m, bank RMS
0.1-0.3 deg) yet keep an ADE of ~400-450 m; giving the model the TRUE arrival time (L3) moves
that only 402 -> 392 m, and the tower wind explains a tenth of the speed residual. So the
residual is neither lateral nor the total duration. This readout decomposes it, per flight,
on the observed sample clock over the common window t in [0, min(T_pred, T_true)]:

  along_m     (pred - obs) projected on the observed heading psi (+ = the model is AHEAD)
  cross_m     (pred - obs) projected on the observed right-hand normal (+ = right of track)
  vertical_m  alt_pred - alt_obs

and reads the speed schedule against remaining distance to the threshold:

  speed_by_distance   mean (V_pred - V_obs), horizontal, in remaining-distance bands
  decel_point_km      remaining distance at which the speed first drops below a threshold,
                      predicted minus observed (+ = the model decelerates FARTHER OUT)

Per stratum (approach_difficulty.strata_masks): RMS of each component, the median share of
along^2 in the horizontal error, the speed-residual and altitude-residual bands, and the
decel-point offsets at three thresholds. Reading rule, fixed first: if along_m dominates the
horizontal RMS and the decel offsets are systematically signed, the residual is the
deceleration SCHEDULE (where and how fast), a supervision problem, not an intent problem;
if vertical_m dominates, it is the height profile; if neither, it is scatter.

    python run_ts.py straight_in_residual_readout \
        native32=4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/L1_native32_pred_val \
        L3_cta=4dTrajectory/outputs/KRDU/experiments/l3_cta_20260907/L3_cta_pred_val --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ts_transformer.docs.compare_frame_arms as cfa  # noqa: E402
import ts_transformer.geometric_metrics as gm  # noqa: E402
from ts_transformer.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, strata_masks  # noqa: E402

STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED)
# Remaining horizontal distance to the threshold, km, outer edge first.
DISTANCE_BANDS_KM = ((25.0, 20.0), (20.0, 15.0), (15.0, 10.0), (10.0, 5.0), (5.0, 0.0))
# Speed thresholds for the deceleration point: two fixed (m/s) and one relative to the
# record's target speed (the threshold-crossing speed the scenario carries).
DECEL_THRESHOLDS = (("below_100mps", 100.0, None), ("below_85mps", 85.0, None), ("target_plus_10", None, 10.0))


def _arrays(states: list[dict]) -> dict[str, np.ndarray]:
    keys = ("t", "lat", "lon", "alt", "V", "psi", "gamma")
    return {k: np.array([s[k] for s in states], dtype=float) for k in keys}


def _interp(t: np.ndarray, src_t: np.ndarray, src: np.ndarray) -> np.ndarray:
    return np.interp(t, src_t, src)


def decel_distance_km(distance_km: np.ndarray, speed: np.ndarray, threshold: float) -> float | None:
    """Remaining distance at the first sample whose speed is below ``threshold``."""
    below = np.flatnonzero(speed < threshold)
    return float(distance_km[below[0]]) if below.size else None


def flight_decomposition(eval_record: dict, states: dict, true_final_time_s: float) -> dict:
    target = eval_record["target_state"]
    lat0, lon0 = float(target["lat"]), float(target["lon"])
    p, o = _arrays(states["predicted_states"]), _arrays(states["observed_states"])
    t_end = min(float(p["t"][-1]), true_final_time_s)
    m = (o["t"] >= 0.0) & (o["t"] <= t_end)
    if m.sum() < 3:
        return {}
    t = o["t"][m]
    oe, on = gm.chart_en(o["lat"][m], o["lon"][m], lat0, lon0)
    pe, pn = gm.chart_en(_interp(t, p["t"], p["lat"]), _interp(t, p["t"], p["lon"]), lat0, lon0)
    de, dn = np.asarray(pe) - np.asarray(oe), np.asarray(pn) - np.asarray(on)
    psi = o["psi"][m]
    along = de * np.cos(psi) + dn * np.sin(psi)
    cross = de * np.sin(psi) - dn * np.cos(psi)
    vertical = _interp(t, p["t"], p["alt"]) - o["alt"][m]
    horizontal2 = along ** 2 + cross ** 2
    # Speed schedule against remaining distance (observed distance for both, so the bands
    # compare the two speeds at the same place along the observed approach).
    dist_km = np.hypot(np.asarray(oe), np.asarray(on)) / 1000.0
    v_obs = o["V"][m] * np.cos(o["gamma"][m])
    v_pred = _interp(t, p["t"], p["V"] * np.cos(p["gamma"]))
    bands = {}
    for hi, lo in DISTANCE_BANDS_KM:
        sel = (dist_km < hi) & (dist_km >= lo)
        bands[f"{hi:g}-{lo:g}km"] = {
            "n": int(sel.sum()),
            "speed_residual_ms": float(np.mean(v_pred[sel] - v_obs[sel])) if sel.any() else None,
            "vertical_residual_m": float(np.mean(vertical[sel])) if sel.any() else None,
            "along_m": float(np.mean(along[sel])) if sel.any() else None,
        }
    # Deceleration points on each trajectory's OWN distance-to-threshold.
    pdist_km = np.hypot(*[np.asarray(x) for x in gm.chart_en(p["lat"], p["lon"], lat0, lon0)]) / 1000.0
    pv_all = p["V"] * np.cos(p["gamma"])
    odist_km = np.hypot(*[np.asarray(x) for x in gm.chart_en(o["lat"], o["lon"], lat0, lon0)]) / 1000.0
    ov_all = o["V"] * np.cos(o["gamma"])
    future = o["t"] >= 0.0
    decel = {}
    for name, absolute, relative in DECEL_THRESHOLDS:
        threshold = absolute if absolute is not None else float(target["V"]) + relative
        dp = decel_distance_km(pdist_km, pv_all, threshold)
        do = decel_distance_km(odist_km[future], ov_all[future], threshold)
        decel[name] = None if dp is None or do is None else dp - do
    return {
        "n_samples": int(m.sum()),
        "along_rms_m": float(np.sqrt(np.mean(along ** 2))),
        "along_mean_m": float(np.mean(along)),
        "cross_rms_m": float(np.sqrt(np.mean(cross ** 2))),
        "vertical_rms_m": float(np.sqrt(np.mean(vertical ** 2))),
        "vertical_mean_m": float(np.mean(vertical)),
        "horizontal_mean_m": float(np.mean(np.sqrt(horizontal2))),
        "along_share": float(np.sum(along ** 2) / np.sum(horizontal2)) if horizontal2.sum() > 0 else None,
        "bands": bands,
        "decel_offset_km": decel,
    }


def load_rows(pred_dir: Path) -> dict[str, dict]:
    rows = cfa.load_arm(pred_dir)
    for row in rows.values():
        eval_record = json.loads((pred_dir / row["eval_file"]).read_text())
        states = json.loads((pred_dir / row["states_file"]).read_text())
        row["decomp"] = flight_decomposition(eval_record, states, float(row["true_final_time_s"]))
    print(f"{pred_dir}: {len(rows)} scored flights, {sum(1 for r in rows.values() if r['decomp'])} decomposed")
    return rows


def _q(values: list[float]) -> dict[str, float]:
    a = np.array(values, dtype=float)
    return {"n": int(a.size), "p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "mean": float(a.mean())}


def stratum_readout(rows: dict[str, dict], keys: list[str]) -> dict[str, dict]:
    masks = strata_masks(rows, keys)
    out = {}
    for stratum in STRATA:
        sel = [k for k, on in zip(keys, masks[stratum]) if on and rows[k]["decomp"]]
        d = [rows[k]["decomp"] for k in sel]
        if len(d) < 3:
            out[stratum] = {"n": len(d)}
            continue
        block = {"n": len(d)}
        for name in ("along_rms_m", "along_mean_m", "cross_rms_m", "vertical_rms_m", "vertical_mean_m", "horizontal_mean_m", "along_share"):
            block[name] = _q([x[name] for x in d if x[name] is not None])
        block["bands"] = {}
        for band in d[0]["bands"]:
            block["bands"][band] = {
                q: _q([x["bands"][band][q] for x in d if x["bands"][band][q] is not None])
                for q in ("speed_residual_ms", "vertical_residual_m", "along_m")
            }
        block["decel_offset_km"] = {
            name: _q([x["decel_offset_km"][name] for x in d if x["decel_offset_km"][name] is not None])
            for name, _a, _r in DECEL_THRESHOLDS
        }
        # Does the deceleration point explain the along-track error? Signed offset against the
        # signed along error in the last band (a later deceleration = the model is ahead there),
        # and |offset| against the flight's along RMS.
        last = f"{DISTANCE_BANDS_KM[-1][0]:g}-{DISTANCE_BANDS_KM[-1][1]:g}km"
        pairs = [(x["decel_offset_km"]["target_plus_10"], x["bands"][last]["along_m"], x["along_rms_m"])
                 for x in d if x["decel_offset_km"]["target_plus_10"] is not None and x["bands"][last]["along_m"] is not None]
        off = np.array([q[0] for q in pairs]); along_last = np.array([q[1] for q in pairs]); rms = np.array([q[2] for q in pairs])
        r_signed = float(np.corrcoef(off, along_last)[0, 1]); r_abs = float(np.corrcoef(np.abs(off), rms)[0, 1])
        slope = float(np.polyfit(off, along_last, 1)[0])
        block["decel_explains"] = {"n": len(pairs), "r_signed_offset_vs_along_last_band": r_signed,
                                   "r2_signed": r_signed ** 2, "slope_m_per_km": slope,
                                   "r_abs_offset_vs_along_rms": r_abs, "r2_abs": r_abs ** 2}
        out[stratum] = block
    return out


def print_readout(label: str, readout: dict[str, dict]) -> None:
    for stratum, b in readout.items():
        print(f"\n## {label} — {stratum} (n={b['n']})")
        if b["n"] < 3:
            continue
        rows = [[name, f"{b[name]['p50']:.0f}", f"{b[name]['mean']:.0f}", f"{b[name]['p90']:.0f}"]
                for name in ("along_rms_m", "cross_rms_m", "vertical_rms_m", "horizontal_mean_m", "along_mean_m", "vertical_mean_m")]
        rows.append(["along_share (of horizontal²)", f"{b['along_share']['p50']:.2f}", f"{b['along_share']['mean']:.2f}", f"{b['along_share']['p90']:.2f}"])
        cfa.print_table(f"{stratum}: error components", ["component", "p50", "mean", "p90"], rows)
        rows = [[band, f"{v['speed_residual_ms']['p50']:+.1f} / {v['speed_residual_ms']['mean']:+.1f}",
                 f"{v['vertical_residual_m']['p50']:+.0f} / {v['vertical_residual_m']['mean']:+.0f}",
                 f"{v['along_m']['p50']:+.0f} / {v['along_m']['mean']:+.0f}", str(v['speed_residual_ms']['n'])]
                for band, v in b["bands"].items()]
        cfa.print_table(f"{stratum}: by remaining distance (p50 / mean)", ["band", "V_pred−V_obs m/s", "alt_pred−alt_obs m", "along m", "n"], rows)
        rows = [[name, f"{v['p10']:+.1f}", f"{v['p50']:+.1f}", f"{v['p90']:+.1f}", str(v['n'])] for name, v in b["decel_offset_km"].items()]
        cfa.print_table(f"{stratum}: deceleration point, pred − obs remaining km (+ = model slows farther out)", ["threshold", "p10", "p50", "p90", "n"], rows)
        e = b["decel_explains"]
        print(f"decel point vs along error (target+10 threshold, n={e['n']}): signed r = {e['r_signed_offset_vs_along_last_band']:+.2f} "
              f"(R² {e['r2_signed']:.2f}, slope {e['slope_m_per_km']:+.0f} m per km); |offset| vs along RMS r = {e['r_abs_offset_vs_along_rms']:+.2f} (R² {e['r2_abs']:.2f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", help="label=<prediction dir>")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    output = {"distance_bands_km": DISTANCE_BANDS_KM, "decel_thresholds": [t[0] for t in DECEL_THRESHOLDS], "arms": {}}
    for arm in args.arms:
        label, _, path = arm.partition("=")
        rows = load_rows(Path(path))
        readout = stratum_readout(rows, sorted(rows))
        print_readout(label, readout)
        output["arms"][label] = {"pred_dir": path, "strata": readout}
    if args.json is not None:
        args.json.write_text(json.dumps(output, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
