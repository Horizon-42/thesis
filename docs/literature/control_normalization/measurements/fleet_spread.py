"""Fleet spread of what one thrust fraction means, over the ts control cohort's types (README §3.2).

Read-only: arrival manifests -> icao24 -> ICAO type (the dataset's resolver) -> OpenAP-direct
filter (the dataset's rule) -> the dynamics the rollout flies (landing mass, installed
thrust, S, CLmax, CD0, k). The operating point is the model's own: 1.3 x its 1-g stall speed
at ~2000 ft ISA, on a 3 deg glide. Writes nothing.

    conda activate aeroviz
    python docs/literature/control_normalization/measurements/fleet_spread.py
"""
import collections
import json
import math
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, ROOT)

from aircraft.aero_params import aero_params_for_aircraft
from aircraft.identity import get_default_identity_resolver
from aircraft.query_aircraft_parameters import openap_support_kind
from flight_scenarios.scenario import aircraft_for_code

G = 9.81
RHO_600M = 1.225 * ((288.15 - 0.0065 * 600.0) / 288.15) ** 4.25588  # ~2000 ft, the RHS's ISA
GAMMA = math.radians(-3.0)
KT = 0.514444

resolver = get_default_identity_resolver()
counts = collections.Counter()
for icao in ("KRDU", "KSJC", "KSTL", "KSMF", "KMSY"):
    m = json.load(open(f"{ROOT}/trajectory_data_process/outputs/harvest/{icao}/arrivals/manifest.json"))
    for rec in m["records"]:
        ident = resolver.resolve(declared_type=None, icao24=rec.get("icao24"))
        if openap_support_kind(ident.typecode) == "direct":
            counts[ident.typecode] += 1

ref = json.load(open(f"{ROOT}/aircraft/reference_speeds.json"))["types"]

rows = []
for code, n in counts.most_common():
    ac = aircraft_for_code(code, provider="openap")
    aero = aero_params_for_aircraft(ac)
    m = ac.landing_mass
    W = m * G
    tmax = ac.engine.max_thrust_total_n
    ws = W / aero.S
    # Approach speed: 1.3 x the model's own 1-g stall speed at this density (the RHS's
    # CLmax), so the number is internal to the flight model, not a published V_ref.
    v_stall = math.sqrt(2 * W / (RHO_600M * aero.S * aero.Cl_max))
    v = 1.3 * v_stall
    q = 0.5 * RHO_600M * v * v
    cl = W / (q * aero.S)
    cd = aero.Cd0 + aero.k * cl * cl
    d = q * aero.S * cd
    t_eq = d + W * math.sin(GAMMA)          # thrust that holds V on a 3 deg glide
    rows.append(dict(
        code=code, n=n, m_t=m / 1000, tw=tmax / W, ws=ws, v_kt=v / KT,
        dw=d / W, delta_eq=t_eq / tmax,
        gain=tmax / m,                        # dVdot / d(delta), m/s^2 per unit fraction
        a_floor=(-0.2 * tmax - d) / m - G * math.sin(GAMMA),  # decel at delta = -0.2
        a_neutral=(0.2 * tmax - d) / m - G * math.sin(GAMMA),  # accel at the init delta = 0.2
        d_for_05kts=0.5 * KT * m / tmax,      # delta needed for 0.5 kt/s of speed change
    ))

total = sum(r["n"] for r in rows)
print(f"OpenAP-direct arrivals over 5 airports: {total}, types: {len(rows)}")
hdr = ("type", "n", "share", "m_t", "T/W", "W/S", "V_kt", "D/W", "d_eq", "gain", "a@-0.2", "a@0.2", "dd.5kt/s")
print("  ".join(f"{h:>7}" for h in hdr))
for r in rows:
    print("  ".join([
        f"{r['code']:>7}", f"{r['n']:>7d}", f"{100*r['n']/total:>6.1f}%", f"{r['m_t']:>7.1f}",
        f"{r['tw']:>7.3f}", f"{r['ws']:>7.0f}", f"{r['v_kt']:>7.0f}", f"{r['dw']:>7.3f}",
        f"{r['delta_eq']:>7.3f}", f"{r['gain']:>7.2f}", f"{r['a_floor']:>7.2f}",
        f"{r['a_neutral']:>7.2f}", f"{r['d_for_05kts']:>7.3f}",
    ]))


def wq(key, qs=(0.05, 0.5, 0.95)):
    s = sorted(rows, key=lambda r: r[key])
    out, acc = [], 0
    targets = list(qs)
    for r in s:
        acc += r["n"]
        while targets and acc >= targets[0] * total:
            out.append(r[key]); targets.pop(0)
    return out

for key in ("tw", "ws", "delta_eq", "gain", "a_floor", "a_neutral", "d_for_05kts"):
    lo, med, hi = wq(key)
    print(f"flight-weighted p5/p50/p95 {key:>12}: {lo:8.3f} {med:8.3f} {hi:8.3f}   ratio p95/p5 {hi/lo if lo else float('nan'):6.2f}")
print(f"min/max over types gain {min(r['gain'] for r in rows):.2f} / {max(r['gain'] for r in rows):.2f};"
      f" delta_eq {min(r['delta_eq'] for r in rows):.3f} / {max(r['delta_eq'] for r in rows):.3f}")
