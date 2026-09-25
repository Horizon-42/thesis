"""Does the current control head adapt its thrust to the airframe? Read-only (README §3.3).

Per flight of an existing control prediction directory: the dynamics type, ADE, the signed
duration error, the speed bias (predicted - observed V on the common absolute-time span from
the anchor), the head's time-mean thrust fraction, the thrust fraction the TRUTH needs under
the same flight model (the package's own actual-control inversion of the observed track),
and the time-mean specific force n_x = V'/g + sin(gamma) of both. Aggregated by airframe
class and by established-at-anchor, one block per prediction directory.

    conda activate aeroviz
    python docs/literature/control_normalization/measurements/per_class_readout.py \
        4dTrajectory/outputs/KRDU/experiments/b1_quantile_20260907/B1_point_matched_pred_val \
        4dTrajectory/outputs/KRDU/experiments/b1_quantile_20260907/B1_point_matched_s2024_pred_val
"""
import collections
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/4dTrajectory")
from aircraft.aero_params import aero_params_for_aircraft  # noqa: E402
from flight_scenarios.scenario import aircraft_for_code, aircraft_provider_of  # noqa: E402
from ts_transformer.config import CONTROL_THRUST_FRACTION  # noqa: E402
from ts_transformer.outputs.dynamics.inverse import actual_controls  # noqa: E402

CLASSES = {
    "A320fam": {"A319", "A320", "A321", "A20N", "A21N"},
    "B737fam": {"B737", "B738", "B739", "B38M", "B39M", "B734"},
    "regional": {"E75L", "CRJ9", "E170", "E190", "E145"},
    "bizjet": {"GLF6", "C550"},
}


def airframe_class(code):
    for name, members in CLASSES.items():
        if code in members:
            return name
    return "heavy"


def flight_row(pred_dir: Path, row):
    ev = json.loads((pred_dir / row["eval_file"]).read_text())
    st = json.loads((pred_dir / row["states_file"]).read_text())
    code = ev["source"]["dynamics_typecode"]
    ac = aircraft_for_code(code, provider=aircraft_provider_of(ev["source"]["dynamics_source"]))
    tmax = ac.engine.max_thrust_total_n
    pred = st["predicted_states"]
    obs = [s for s in st["observed_states"] if s["t"] >= 0.0]
    tp = np.array([s["t"] for s in pred]); vp = np.array([s["V"] for s in pred])
    to = np.array([s["t"] for s in obs]); vo = np.array([s["V"] for s in obs])
    end = min(tp[-1], to[-1])
    grid = np.linspace(0.0, end, 64)
    dv = np.interp(grid, tp, vp) - np.interp(grid, to, vo)
    segs = st["control_segments"]
    dur = np.array([s["duration_s"] for s in segs])
    frac = np.array([s["thrust"] for s in segs]) / tmax
    # What the TRUTH needs, under the same flight model: the actual-control inversion of the
    # observed track over the common span, time-averaged (the mean is robust to ADS-B noise
    # in the differentiated speed). n_x = (T - D)/W is the specific force along the path.
    aero = aero_params_for_aircraft(ac)
    aero_row = np.array([aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall])
    span_obs = [s for s in obs if s["t"] <= end]

    def as_array(samples):
        return np.array([[s["lat"], s["lon"], s["alt"], s["V"], s["psi"], s["gamma"], s["m"]] for s in samples])

    def n_x_mean(samples):
        t = np.array([s["t"] for s in samples]); v = np.array([s["V"] for s in samples])
        g = np.array([s["gamma"] for s in samples])
        return float(np.trapezoid(np.gradient(v, t) / 9.81 + np.sin(g), t) / (t[-1] - t[0]))

    t_obs = np.array([s["t"] for s in span_obs])
    req = actual_controls(as_array(span_obs), t_obs, aero_params=aero_row, max_thrust_n=tmax,
                          parameterization=CONTROL_THRUST_FRACTION)
    frac_req = float(np.trapezoid(req[:, 0], t_obs) / (t_obs[-1] - t_obs[0]))
    span_pred = [s for s in pred if s["t"] <= end]
    return dict(
        frac_req=frac_req, nx_pred=n_x_mean(span_pred), nx_obs=n_x_mean(span_obs),
        code=code, cls=airframe_class(code), est=bool(row["established_at_anchor"]),
        ade=row["ade_m"], fde=row["fde_m"], dt=row["final_time_error_s"],
        dv_mean=float(dv.mean()), dv_abs=float(np.abs(dv).mean()), dv_end=float(dv[-1]),
        frac_mean=float((frac * dur).sum() / dur.sum()),
        frac_floor_share=float(dur[frac <= -0.199].sum() / dur.sum()),
    )


def summarize(rows, label):
    print(f"\n== {label}: {len(rows)} flights")
    hdr = ("group", "n", "ADE_mean", "ADE_med", "dT_mean", "dV_mean", "dV_se", "dV_end",
           "frac", "frac_req", "dfrac", "nx_pred", "nx_obs", "dnx")
    print("  ".join(f"{h:>8}" for h in hdr))

    def line(name, rs):
        if not rs:
            return
        f = lambda k: [r[k] for r in rs]
        m = statistics.mean
        se = statistics.stdev(f("dv_mean")) / math.sqrt(len(rs)) if len(rs) > 1 else float("nan")
        print("  ".join([
            f"{name:>8}", f"{len(rs):>8d}", f"{m(f('ade')):>8.0f}",
            f"{statistics.median(f('ade')):>8.0f}",
            f"{m(f('dt')):>8.1f}", f"{m(f('dv_mean')):>8.2f}", f"{se:>8.2f}",
            f"{m(f('dv_end')):>8.2f}",
            f"{m(f('frac_mean')):>8.3f}", f"{m(f('frac_req')):>8.3f}",
            f"{m(f('frac_mean')) - m(f('frac_req')):>8.3f}",
            f"{m(f('nx_pred')):>8.4f}", f"{m(f('nx_obs')):>8.4f}",
            f"{m(f('nx_pred')) - m(f('nx_obs')):>8.4f}",
        ]))

    line("ALL", rows)
    for est in (True, False):
        sub = [r for r in rows if r["est"] == est]
        print(f"-- established_at_anchor={est}")
        for cls in ("B737fam", "A320fam", "regional", "heavy", "bizjet"):
            line(cls, [r for r in sub if r["cls"] == cls])


for pred_dir in sys.argv[1:]:
    pred_dir = Path(pred_dir)
    summary = json.loads((pred_dir / "summary.json").read_text())
    rows = [flight_row(pred_dir, r) for r in summary["results"] if r["status"] == "solved"]
    summarize(rows, pred_dir.name)
    counts = collections.Counter(r["code"] for r in rows if r["cls"] in ("heavy", "bizjet"))
    print("heavy/bizjet types:", dict(counts))
