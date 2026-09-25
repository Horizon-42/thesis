"""Wind readout: does the METAR headwind at landing explain the along-track residual?

The straight-in residual of the control-output models is not lateral (chamfer ~100 m,
bank RMS 0.1-0.3 deg) but along-track: a speed/timing difference of wind scale (~3 m/s).
The point-mass rollout has no wind, so its V is an airspeed that the metrics compare
with an ADS-B GROUND speed. This readout joins each scored flight to the field's
ASOS/METAR report at its landing time (``evaluation.wind``, the same table the observed
speed gate uses) and asks how much of the residual the headwind explains.

Per flight of one prediction directory (``summary.json``-rostered):

  headwind_ms            wind component along the runway course at landing, + = headwind
                         (nearest report within 30 min; None when absent or variable)
  speed_residual_ms      mean over the common window t in [0, min(T_pred, T_true)] of
                         V_pred cos(gamma_pred) - V_obs cos(gamma_obs): predicted minus
                         observed horizontal ground speed on the observed sample times
  final_speed_residual_ms  the same over the last FINAL_WINDOW_S before the true crossing
  final_time_error_s     predicted - true duration (the summary row's number)
  endpoint_along_track_m predicted endpoint along the runway course, + past the threshold
                         (``compare_frame_arms.endpoint_geometry``)

Per stratum (``approach_difficulty.strata_masks``): n, headwind quantiles, and for each
residual the OLS slope on the headwind with its 95 % interval, Pearson r and R^2, plus
a binned table (tailwind / calm / headwind / strong headwind) of the residual means.

Reading rule, fixed before the numbers: if the wind IS the residual, the speed slope is
about +1 m/s per m/s of headwind (the model flies airspeed, the track shows ground
speed), the duration slope is negative (a headwind lengthens the true approach), and
R^2 says what share of the residual's variance the tower wind accounts for. A slope
near 0 means the along-track residual is something else. The limits are the table's:
a 10 m tower wind at the field, not the wind aloft over the last 25 km.

    python run_ts.py wind_residual_readout \
        L1_native32=4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/L1_native32_pred_val \
        --json 4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/wind_readout.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluation.cli import DEFAULT_METAR_ROOT  # noqa: E402
from evaluation.wind import load_wind_tables, wind_at_landing  # noqa: E402
import ts_transformer.inference.arm_readout as cfa  # noqa: E402
from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, strata_masks  # noqa: E402

FINAL_WINDOW_S = 120.0
# Headwind bins (m/s): a tailwind, calm, an ordinary headwind, a strong one.
WIND_BIN_EDGES_MS = (-2.0, 2.0, 6.0)
WIND_BIN_LABELS = ("tailwind < -2", "calm [-2, 2)", "headwind [2, 6)", "strong headwind >= 6")
RESIDUALS = ("speed_residual_ms", "final_speed_residual_ms", "final_time_error_s", "endpoint_along_track_m")
STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED)
# The record's target psi is the runway course in the chart frame (radians, east = 0,
# counter-clockwise); the wind table wants a compass bearing. Checked against the
# runway ident (two-digit tens of degrees) so a convention slip fails loudly.
RUNWAY_IDENT_TOLERANCE_DEG = 15.0


def runway_course_compass_deg(target_state: dict, runway_ident: str) -> float:
    compass = (90.0 - math.degrees(float(target_state["psi"]))) % 360.0
    ident_deg = 10.0 * int(runway_ident[:2])
    delta = abs((compass - ident_deg + 180.0) % 360.0 - 180.0)
    if delta > RUNWAY_IDENT_TOLERANCE_DEG:
        raise ValueError(f"runway {runway_ident}: course {compass:.1f} deg is {delta:.1f} deg "
                         f"off the ident; psi convention?")
    return compass


def _horizontal_speed(states: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([s["t"] for s in states], dtype=float)
    v = np.array([s["V"] * math.cos(s["gamma"]) for s in states], dtype=float)
    return t, v


def speed_residuals(states: dict, true_final_time_s: float) -> dict[str, float | None]:
    t_pred, v_pred = _horizontal_speed(states["predicted_states"])
    t_obs, v_obs = _horizontal_speed(states["observed_states"])
    t_end = min(float(t_pred[-1]), true_final_time_s)
    common = (t_obs >= 0.0) & (t_obs <= t_end)
    final = common & (t_obs >= true_final_time_s - FINAL_WINDOW_S)
    out: dict[str, float | None] = {}
    for name, mask in (("speed_residual_ms", common), ("final_speed_residual_ms", final)):
        if mask.sum() < 2:
            out[name] = None
            continue
        interp = np.interp(t_obs[mask], t_pred, v_pred)
        out[name] = float(np.mean(interp - v_obs[mask]))
    return out


def load_rows(pred_dir: Path, tables) -> dict[str, dict]:
    rows = cfa.load_arm(pred_dir)
    counts = {"estimated": 0, "unavailable": 0, "no_table": 0}
    for row in rows.values():
        eval_record = json.loads((pred_dir / row["eval_file"]).read_text())
        states = json.loads((pred_dir / row["states_file"]).read_text())
        row.update(speed_residuals(states, float(row["true_final_time_s"])))
        table = tables.get(row["arr_airport"])
        if table is None:
            row["headwind_ms"], row["wind"] = None, {"status": "no_table"}
            counts["no_table"] += 1
            continue
        course = runway_course_compass_deg(eval_record["target_state"], row["runway"])
        row["headwind_ms"], row["wind"] = wind_at_landing(table, row["landing_time_utc"], course)
        counts[row["wind"]["status"]] += 1
    print(f"{pred_dir}: {len(rows)} scored flights; wind {counts}")
    return rows


def ols(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    resid = y - fitted
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = math.sqrt(float(np.sum(resid ** 2)) / (n - 2) / sxx) if n > 2 and sxx > 0 else float("nan")
    r = float(np.corrcoef(x, y)[0, 1]) if n > 2 else float("nan")
    return {
        "n": n, "slope": float(slope), "slope_ci95": 1.96 * se, "intercept": float(intercept),
        "pearson_r": r, "r2": r * r,
        "residual_sd_before": float(np.std(y)), "residual_sd_after": float(np.std(resid)),
    }


def binned(x: np.ndarray, y: np.ndarray) -> list[dict]:
    bins = np.digitize(x, WIND_BIN_EDGES_MS)
    return [
        {"bin": label, "n": int((bins == i).sum()),
         "mean": float(y[bins == i].mean()) if (bins == i).any() else None}
        for i, label in enumerate(WIND_BIN_LABELS)
    ]


def stratum_readout(rows: dict[str, dict], keys: list[str]) -> dict[str, dict]:
    masks = strata_masks(rows, keys)
    headwind = np.array([np.nan if rows[k]["headwind_ms"] is None else rows[k]["headwind_ms"] for k in keys])
    out: dict[str, dict] = {}
    for stratum in STRATA:
        mask = masks[stratum] & ~np.isnan(headwind)
        block: dict = {"n": int(mask.sum()), "n_stratum": int(masks[stratum].sum())}
        if mask.sum() < 3:
            out[stratum] = block
            continue
        x = headwind[mask]
        block["headwind_ms_p10_p50_p90"] = [float(v) for v in np.percentile(x, (10, 50, 90))]
        for name in RESIDUALS:
            y = np.array([np.nan if rows[k][name] is None else rows[k][name] for k in keys])[mask]
            keep = ~np.isnan(y)
            block[name] = {**ols(x[keep], y[keep]), "bins": binned(x[keep], y[keep]),
                           "mean": float(y[keep].mean())}
        out[stratum] = block
    return out


def print_readout(label: str, readout: dict[str, dict]) -> None:
    for stratum, block in readout.items():
        print(f"\n## {label} — {stratum} (n={block['n']} with wind of {block['n_stratum']})")
        if "headwind_ms_p10_p50_p90" not in block:
            continue
        p10, p50, p90 = block["headwind_ms_p10_p50_p90"]
        print(f"headwind m/s p10/p50/p90: {p10:.1f} / {p50:.1f} / {p90:.1f}")
        header = ["residual", "mean", "slope per m/s headwind", "r", "R²", "sd before → after"] + list(WIND_BIN_LABELS)
        table = []
        for name in RESIDUALS:
            b = block[name]
            cells = [name, f"{b['mean']:.2f}", f"{b['slope']:+.3f} ± {b['slope_ci95']:.3f}",
                     f"{b['pearson_r']:+.2f}", f"{b['r2']:.3f}",
                     f"{b['residual_sd_before']:.2f} → {b['residual_sd_after']:.2f}"]
            cells += ["—" if e["mean"] is None else f"{e['mean']:.2f} (n={e['n']})" for e in b["bins"]]
            table.append(cells)
        cfa.print_table(f"{stratum}: residual ~ headwind", header, table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", help="label=<prediction dir> (summary.json inside)")
    parser.add_argument("--metar-root", type=Path, default=DEFAULT_METAR_ROOT)
    parser.add_argument("--json", type=Path, default=None, help="write every number here")
    args = parser.parse_args(argv)

    tables = load_wind_tables(args.metar_root)
    if not tables:
        raise SystemExit(f"{args.metar_root}: no METAR tables (fetch_iem_asos first)")
    output: dict = {"metar_root": str(args.metar_root), "final_window_s": FINAL_WINDOW_S,
                    "wind_bin_edges_ms": WIND_BIN_EDGES_MS, "arms": {}}
    for arm in args.arms:
        label, _, path = arm.partition("=")
        rows = load_rows(Path(path), tables)
        keys = sorted(rows)
        readout = stratum_readout(rows, keys)
        print_readout(label, readout)
        output["arms"][label] = {"pred_dir": path, "strata": readout}
    if args.json is not None:
        args.json.write_text(json.dumps(output, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
