#!/usr/bin/env python
"""What roll-in / roll-out response do the OBSERVED ADS-B tracks show, and what would each plant's flights show through
the SAME estimator?

ADS-B here is 1 Hz lon/lat/alt (no track angle, no roll), so bank is read from the position record:
    ψ̇ from the local second-order fit of the velocity, φ = atan(V ψ̇ / g)  (coordinated, no wind).
The same estimator runs on (a) the observed raw samples of the 500 flights and (b) the executor flights of each plant arm
(1 s cycle states, noise-free unless --noise-m). If the estimator separates the arms, the observed distribution can be
placed among them; if it does not, tau_bank is not identifiable from this record and the output says so.

Events: a roll-in is a run of |φ| rising from < LOW to a peak >= MIN_PEAK, rise time = 10 %→90 % of the peak; a roll-out
is the matching fall from the peak to < LOW.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

import archive_paths   # noqa: E402  (README §2)
REPO = archive_paths.REPO
sys.path[:0] = [str(REPO / "4dTrajectory"), str(REPO)]
from geokit import METRES_PER_DEG_LAT                                       # noqa: E402
from ts_transformer.instructions.artefact import load_candidates            # noqa: E402

G = 9.80665
IL = REPO / "4dTrajectory/outputs/POOLED/instruction_language/v12_20261005"
HARVEST = archive_paths.HARVEST
LOW_DEG, MIN_PEAK_DEG = 3.0, 12.0


def bank_series(e, n, window: int) -> np.ndarray:
    """Signed bank (deg) at each 1 s sample from positions, via a local quadratic fit over ``window`` samples."""
    ve = savgol_filter(e, window, 2, deriv=1)
    vn = savgol_filter(n, window, 2, deriv=1)
    ae = savgol_filter(e, window, 2, deriv=2)
    an = savgol_filter(n, window, 2, deriv=2)
    v2 = np.maximum(ve ** 2 + vn ** 2, 1.0)
    psi_dot = (vn * ae - ve * an) / v2                  # compass-clockwise positive (rad/s)
    return np.degrees(np.arctan(np.sqrt(v2) * psi_dot / G))


def events(phi: np.ndarray) -> list[dict]:
    """Roll-in / roll-out events of one flight's bank series (1 s samples)."""
    a = np.abs(phi)
    out = []
    i, n = 0, len(a)
    while i < n:
        if a[i] >= LOW_DEG:
            j = i
            while j < n and a[j] >= LOW_DEG and np.sign(phi[j]) == np.sign(phi[i]):
                j += 1
            seg = a[i:j]
            peak = seg.max()
            if peak >= MIN_PEAK_DEG and i > 0 and j < n:
                k = i + int(seg.argmax())
                up, down = a[i - 1:k + 1], a[k:j + 1]
                def cross(x, level, rising):
                    idx = np.flatnonzero(x >= level) if rising else np.flatnonzero(x <= level)
                    if not len(idx):
                        return None
                    t = idx[0] if rising else idx[0]
                    if t == 0:
                        return float(t)
                    y0, y1 = x[t - 1], x[t]
                    return float(t - 1 + (level - y0) / (y1 - y0)) if y1 != y0 else float(t)
                r10, r90 = cross(up, 0.1 * peak, True), cross(up, 0.9 * peak, True)
                f90, f10 = cross(down, 0.9 * peak, False), cross(down, 0.1 * peak, False)
                ev = {"peak_deg": float(peak)}
                if r10 is not None and r90 is not None and r90 > r10:
                    ev["rise_s"] = r90 - r10
                    ev["roll_in_rate"] = float(np.max(np.diff(a[i - 1:k + 1]))) if k > i else None
                if f90 is not None and f10 is not None and f10 > f90:
                    ev["fall_s"] = f10 - f90
                out.append(ev)
            i = j
        else:
            i += 1
    return out


def observed_enu(key: str, airport: str, frame) -> tuple[np.ndarray, np.ndarray]:
    """The arrival slice of an observed flight's raw samples, resampled on a 1 s grid, in the airport frame."""
    manifest = json.loads((HARVEST / airport / "arrivals/manifest.json").read_text())
    rec = next(r for r in manifest["records"] if r["flight_key"] == key)
    d = json.loads((HARVEST / airport / "tracks" / rec["source_file"]).read_text())
    s = np.asarray(d["samples"], dtype=float)[rec["first_sample_index"]:rec["last_sample_index"] + 1]
    t = s[:, 0]
    e = (s[:, 1] - frame.lon0) * frame.m_per_deg_lon
    n = (s[:, 2] - frame.lat0) * METRES_PER_DEG_LAT
    grid = np.arange(t[0], t[-1], 1.0)
    return np.interp(grid, t, e), np.interp(grid, t, n)


def summarize(evs: list[dict], name: str) -> dict:
    def q(key):
        v = np.array([e[key] for e in evs if e.get(key) is not None])
        return {"n": int(len(v)), "p25": float(np.percentile(v, 25)), "p50": float(np.percentile(v, 50)),
                "p75": float(np.percentile(v, 75))} if len(v) else {"n": 0}
    return {"arm": name, "events": len(evs), "peak_deg": q("peak_deg"), "rise_s": q("rise_s"), "fall_s": q("fall_s"),
            "roll_in_deg_per_s": q("roll_in_rate")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="compare_plants output with tracks.npz and rows.json")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="other compare_plants outputs (lag arms)")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--noise-along-m", type=float, default=0.0, help="white position noise along the track added to simulated flights")
    ap.add_argument("--noise-cross-m", type=float, default=0.0, help="white position noise across the track added to simulated flights")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = json.loads((args.run / "rows.json").read_text())["point-mass"]
    geoms = load_candidates(IL)
    keys = [(r["flight_key"], r["airport"]) for r in rows]

    arms: dict[str, list[dict]] = {"observed": []}
    for key, airport in keys:
        e, n = observed_enu(key, airport, geoms[airport].frame)
        if len(e) > args.window + 2:
            arms["observed"] += events(bank_series(e, n, args.window))

    def sim_arm(run: Path, plant: str, label: str):
        tr = np.load(run / "tracks.npz")
        evs = []
        for key, _ in keys:
            e, n = tr[f"{plant}|{key}|e"], tr[f"{plant}|{key}|n"]
            if args.noise_along_m or args.noise_cross_m:
                ve, vn = np.gradient(e), np.gradient(n)
                sp = np.maximum(np.hypot(ve, vn), 1e-6)
                ue, un = ve / sp, vn / sp
                al, cr = rng.normal(0, args.noise_along_m, len(e)), rng.normal(0, args.noise_cross_m, len(e))
                e, n = e + al * ue - cr * un, n + al * un + cr * ue
            if len(e) > args.window + 2:
                evs += events(bank_series(e, n, args.window))
        arms[label] = evs

    sim_arm(args.run, "point-mass", "A point-mass (p = executor's 5 deg/s)")
    sim_arm(args.run, "first-order-lag", "B lag tau (1.5, 2.0, 0.8)")
    for run in args.extra:
        taus = json.loads((run / "meta.json").read_text())["plants"]["first-order-lag"]["time_constants_s_thrust_bank_load"]
        sim_arm(run, "first-order-lag", f"lag tau {tuple(taus)}")

    result = [summarize(v, k) for k, v in arms.items()]
    args.out.write_text(json.dumps({"window": args.window, "noise_along_m": args.noise_along_m,
                                    "noise_cross_m": args.noise_cross_m, "arms": result}, indent=1))
    print(f"window {args.window} samples, simulated noise along {args.noise_along_m} m / cross {args.noise_cross_m} m")
    print(f"{'arm':40s} {'events':>6s} {'peak bank p50':>13s} {'rise 10→90% s p25/p50/p75':>27s} {'fall s p50':>10s} {'max d|φ|/dt p50':>16s}")
    for r in result:
        rise, fall, rate = r["rise_s"], r["fall_s"], r["roll_in_deg_per_s"]
        print(f"{r['arm']:40s} {r['events']:6d} {r['peak_deg'].get('p50', float('nan')):13.1f} "
              f"{rise.get('p25', float('nan')):8.1f}/{rise.get('p50', float('nan')):5.1f}/{rise.get('p75', float('nan')):5.1f}"
              f" (n={rise['n']:4d}) {fall.get('p50', float('nan')):10.1f} {rate.get('p50', float('nan')):16.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
