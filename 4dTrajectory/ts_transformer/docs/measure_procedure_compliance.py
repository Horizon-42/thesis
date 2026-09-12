"""Which observed arrivals fly the coded RNAV(GPS) approach in full?  Read-only census.

Three NESTED tiers, mirroring how the optimizer reads the procedure
(``approach_constraints`` / ``aeroviz_backend.procedure_segments``):

  A  final segment: established in the k=0.5 LPV cone from UPSTREAM of the FAF down to
     300 m before the threshold (d_join >= d_FAF), and every inside-FAF sample within the
     glidepath window [-60, +120] m about TDZE + TCH + d*tan(GPA).
  B  A + the IF: closest approach to the IF <= 926 m (k*RNP = 0.5 x 1 NM, the optimizer's
     pre-FAF fix disc), altitude at the IF passage inside its coded constraint (30 m
     tolerance), and every sample between the IF and FAF passages at-or-above the FAF's
     coded floor (the optimizer's leg floor).
  C  B + a coded transition: for at least one transition, every transition fix INSIDE the
     flight's stored range (fix radial distance <= the first stored sample's) is passed in
     order within 926 m with its coded altitude met, the leg floor held up to each passage,
     and the IF's floor held from the last transition fix to the IF.  A flight with no
     transition fix in range is "C undecidable" (its transition lies outside the 30 km store).

Reported beside the tiers, not inside them: the track angle at the join against the course
(<= 30 deg) and the optimizer's join lower bound d_join >= 1.2 d_FAF.

Geometry mirrors docs/measure_procedure_adherence.py (FAS cone with the 9023 ft FPAP floor).
Samples: the FULL stored track (30 km crop) up to the roster's last_sample_index, so fixes
between 25 and 30 km are testable.  Altitudes: stored HAE -> MSL with the runway target's
hae_minus_msl_m, outliers repaired on read (harvest.altitude_filter.filtered_track).

    conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/measure_procedure_compliance.py KRDU --stride 1 --out /tmp/census_KRDU.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from geokit import FT_M, METRES_PER_DEG_LAT, NM_M

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from flight_scenarios.procedure_final import procedure_skeleton  # noqa: E402
from trajectory_data_process.harvest.altitude_filter import filtered_track  # noqa: E402

K_MARGIN = 0.5                                   # approach_constraints.DEFAULT_K_MARGIN
RNP_M = 1.0 * NM_M                               # procedure_segments._DEFAULT_RNP_NM
FIX_DISC_M = K_MARGIN * RNP_M                    # 926 m: the optimizer's pre-FAF fix disc
ALT_TOL_M = 30.0                                 # one 100 ft reporting quantum
D_GARP_LTP_M = (9023.0 + 1000.0) * FT_M
COURSE_WIDTH_M = max(350.0 * FT_M, math.tan(math.radians(1.5)) * D_GARP_LTP_M)
TAIL_MIN_D_M = 300.0
GP_BELOW_M, GP_ABOVE_M = 60.0, 120.0             # approach_constraints defaults
JOIN_MIN_FRACTION = 1.2                          # 1 + _JOIN_MIN_UPSTREAM_FRACTION
MAX_INTERCEPT_DEG = 30.0                         # STANDARD_INTERCEPT_MAX_DEG


def halfwidth_m(d):
    return COURSE_WIDTH_M * (d + D_GARP_LTP_M) / D_GARP_LTP_M


class RunwayProcedure:
    def __init__(self, airport: str, runway: str, target: dict):
        sk = procedure_skeleton(airport, runway)
        self.lat0, self.lon0 = target["lat"], target["lon"]
        self.m_lon = METRES_PER_DEG_LAT * math.cos(math.radians(self.lat0))
        course = math.radians(target["course_deg"])
        self.course_deg = float(target["course_deg"])
        self.ue, self.un = math.sin(course), math.cos(course)
        self.hae_minus_msl = target["hae_minus_msl_m"]
        self.elev = target["elevation_msl_m"]
        self.gpa = sk.glidepath_angle_deg or target["published_glidepath_deg"]
        self.tch = (sk.threshold_crossing_height_m if sk.threshold_crossing_height_m is not None
                    else target["threshold_crossing_height_m"])
        final = list(sk.final.fixes)
        faf_i = next(i for i, f in enumerate(final) if f.role == "FAF")
        self.faf = final[faf_i]
        self.d_faf = self.along(self.faf)
        self.if_fix = final[faf_i - 1] if faf_i >= 1 else None
        final_idents = {f.ident for f in final}
        self.transitions = [
            (t.ident, [f for f in t.fixes if f.ident not in final_idents]) for t in sk.transitions
        ]
        self.transitions = [(ident, fixes) for ident, fixes in self.transitions if fixes]

    def ne(self, fix):
        return ((fix.lon_deg - self.lon0) * self.m_lon,
                (fix.lat_deg - self.lat0) * METRES_PER_DEG_LAT)

    def along(self, fix):
        e, n = self.ne(fix)
        return -(e * self.ue + n * self.un)

    def radial(self, fix):
        return math.hypot(*self.ne(fix))

    def describe(self) -> dict:
        def fx(f):
            return {"ident": f.ident, "role": f.role, "d_along_km": round(self.along(f) / 1e3, 2),
                    "r_km": round(self.radial(f) / 1e3, 2),
                    "alt_min_m": f.altitude_min_m, "alt_max_m": f.altitude_max_m}
        return {"d_faf_km": round(self.d_faf / 1e3, 2), "gpa_deg": self.gpa, "tch_m": round(self.tch, 1),
                "if": fx(self.if_fix) if self.if_fix else None, "faf": fx(self.faf),
                "transitions": {ident: [fx(f) for f in fixes] for ident, fixes in self.transitions}}


def alt_ok(fix, h_at: float) -> bool:
    lo, hi = fix.altitude_min_m, fix.altitude_max_m
    return (lo is None or h_at >= lo - ALT_TOL_M) and (hi is None or h_at <= hi + ALT_TOL_M)


def judge(rp: RunwayProcedure, samples: list, last_idx: int) -> dict:
    s = np.asarray(samples[: last_idx + 1], dtype=float)
    e = (s[:, 1] - rp.lon0) * rp.m_lon
    n = (s[:, 2] - rp.lat0) * METRES_PER_DEG_LAT
    h = s[:, 3] - rp.hae_minus_msl
    d = -(e * rp.ue + n * rp.un)
    xt = e * rp.un - n * rp.ue
    r_first = float(math.hypot(e[0], n[0]))

    # ── A: the final segment ──────────────────────────────────────────────────────
    inside = (np.abs(xt) <= K_MARGIN * halfwidth_m(np.maximum(d, 0.0))) & (d > 0)
    stays = np.ones(len(s), bool)
    acc = True
    for i in range(len(s) - 1, -1, -1):
        if d[i] <= TAIL_MIN_D_M:
            continue
        acc = acc and bool(inside[i])
        stays[i] = acc
    joined = np.where(stays & (d > TAIL_MIN_D_M))[0]
    join_idx = int(joined[0]) if len(joined) else None
    d_join = float(d[join_idx]) if join_idx is not None else math.nan
    # the final segment proper: from the join onward (a downwind passes abeam with d <= d_FAF)
    on_final = np.arange(len(s)) >= (join_idx if join_idx is not None else len(s))
    tail = on_final & (d <= rp.d_faf) & (d > TAIL_MIN_D_M)
    dv = h - (rp.elev + rp.tch + d * math.tan(math.radians(rp.gpa)))
    gp_ok = bool(tail.any()) and bool(np.all((dv[tail] >= -GP_BELOW_M) & (dv[tail] <= GP_ABOVE_M)))
    established_before_faf = join_idx is not None and d_join >= rp.d_faf
    tier_a = established_before_faf and gp_ok

    # ── the join extras (reported, not in a tier) ──────────────────────────────────
    intercept_deg = math.nan
    if join_idx is not None:
        i0, i1 = max(join_idx - 1, 0), min(join_idx + 1, len(s) - 1)
        if i1 > i0:
            trk = math.degrees(math.atan2(e[i1] - e[i0], n[i1] - n[i0]))
            intercept_deg = abs((trk - rp.course_deg + 180.0) % 360.0 - 180.0)
    join_min_ok = join_idx is not None and d_join >= JOIN_MIN_FRACTION * rp.d_faf

    def passage(fix):
        e_f, n_f = rp.ne(fix)
        dist = np.hypot(e - e_f, n - n_f)
        i = int(np.argmin(dist))
        return i, float(dist[i])

    # ── B: A + the IF disc, its altitude, the IF→FAF leg floor ────────────────────
    i_faf, _ = passage(rp.faf)
    tier_b = False
    if_dist = math.nan
    i_if = None
    if rp.if_fix is not None:
        i_if, if_dist = passage(rp.if_fix)
        floor = rp.faf.altitude_min_m
        leg_ok = floor is None or bool(np.all(h[i_if: i_faf + 1] >= floor - ALT_TOL_M))
        b_checks = {"if_disc": if_dist <= FIX_DISC_M, "if_order": i_if < i_faf,
                    "if_alt": alt_ok(rp.if_fix, float(h[i_if])), "if_faf_floor": leg_ok}
        tier_b = tier_a and all(b_checks.values())
    else:
        b_checks = {}

    # ── C: B + one coded transition, on the fixes inside the stored range ─────────
    tier_c, c_path, c_decidable = False, None, False
    c_tested: dict[str, list[str]] = {}
    c_reasons: dict[str, str] = {}
    for ident, fixes in rp.transitions:
        in_range = [f for f in fixes if rp.radial(f) <= r_first]
        if not in_range:
            continue
        c_decidable = True
        c_tested[ident] = [f.ident for f in in_range]
        if not tier_b:
            continue
        prev_i, why = -1, None
        for f in in_range:
            i_f, dist_f = passage(f)
            start = max(prev_i, 0)
            floor_ok = f.altitude_min_m is None or bool(np.all(h[start: i_f + 1] >= f.altitude_min_m - ALT_TOL_M))
            if dist_f > FIX_DISC_M:
                why = f"{f.ident}:disc"
            elif i_f <= prev_i:
                why = f"{f.ident}:order"
            elif not alt_ok(f, float(h[i_f])):
                why = f"{f.ident}:alt"
            elif not floor_ok:
                why = f"{f.ident}:leg_floor"
            if why:
                break
            prev_i = i_f
        if why is None:
            if_floor = rp.if_fix.altitude_min_m
            if not (prev_i < i_if):
                why = "if:order"
            elif not (if_floor is None or bool(np.all(h[prev_i: i_if + 1] >= if_floor - ALT_TOL_M))):
                why = "if:leg_floor"
        c_reasons[ident] = why or "ok"
        if why is None:
            tier_c, c_path = True, ident
            break

    return {
        "r_first_km": round(r_first / 1e3, 2),
        "d_join_km": None if math.isnan(d_join) else round(d_join / 1e3, 2),
        "established_before_faf": bool(established_before_faf),
        "glidepath_window": bool(gp_ok),
        "A": bool(tier_a), "B": bool(tier_b), "C": bool(tier_c),
        "c_decidable": bool(c_decidable), "c_path": c_path, "c_fixes_in_range": c_tested,
        "c_reasons": c_reasons, "b_checks": {k: bool(v) for k, v in b_checks.items()},
        "if_dist_m": None if math.isnan(if_dist) else round(if_dist),
        "if_alt_m": None if i_if is None else round(float(h[i_if])),
        "if_floor_m": None if rp.if_fix is None else rp.if_fix.altitude_min_m,
        "intercept_deg": None if math.isnan(intercept_deg) else round(intercept_deg, 1),
        "join_ge_1p2_dfaf": bool(join_min_ok),
    }


def _count_reasons(rows: list[dict]) -> dict[str, int]:
    """Among tier-B flights that are C-decidable but not C: the failure kind of their BEST
    transition (the one that got furthest, by kind order disc < order < alt < floor)."""
    rank = {"disc": 0, "order": 1, "alt": 2, "leg_floor": 3, "ok": 9}
    out: dict[str, int] = {}
    for f in rows:
        if not (f["B"] and f["c_decidable"] and not f["C"]):
            continue
        best = max(f["c_reasons"].values(), key=lambda w: rank[w.split(":")[-1]])
        kind = best.split(":")[-1]
        out[kind] = out.get(kind, 0) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("airport")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    airport = args.airport.upper()
    harvest = ROOT / "trajectory_data_process/outputs/harvest" / airport
    manifest = json.loads((harvest / "arrivals/manifest.json").read_text())
    procs: dict[str, RunwayProcedure] = {}
    skipped: dict[str, str] = {}
    for runway, target in manifest["runway_targets"].items():
        try:
            procs[runway] = RunwayProcedure(airport, runway, target)
        except Exception as exc:  # no RNAV(GPS) document / no FAF: reported, not guessed
            skipped[runway] = f"{type(exc).__name__}: {exc}"
    flights = []
    t0 = time.time()
    records = manifest["records"][:: args.stride]
    for k, record in enumerate(records):
        rp = procs.get(record["runway"])
        if rp is None:
            continue
        track = filtered_track(json.loads((harvest / "tracks" / record["source_file"]).read_text()))
        verdict = judge(rp, track["samples"], int(record["last_sample_index"]))
        flights.append({"flight_key": record["flight_key"], "runway": record["runway"], **verdict})
        if (k + 1) % 1000 == 0:
            print(f"{airport}: {k + 1}/{len(records)} in {time.time() - t0:.0f} s", flush=True)
    summary = {}
    for runway in procs:
        rows = [f for f in flights if f["runway"] == runway]
        n = len(rows)
        cnt = lambda key: sum(1 for f in rows if f[key])  # noqa: E731
        c_dec = cnt("c_decidable")
        summary[runway] = {
            "n": n, "A": cnt("A"), "B": cnt("B"), "C": cnt("C"), "C_decidable": c_dec,
            "established_before_faf": cnt("established_before_faf"),
            "glidepath_window": cnt("glidepath_window"),
            "join_le_30deg": sum(1 for f in rows if f["intercept_deg"] is not None and f["intercept_deg"] <= MAX_INTERCEPT_DEG),
            "join_ge_1p2_dfaf": cnt("join_ge_1p2_dfaf"),
            "b_fail_among_A": {k: sum(1 for f in rows if f["A"] and f["b_checks"] and not f["b_checks"][k])
                               for k in ("if_disc", "if_order", "if_alt", "if_faf_floor")},
            "c_fail_among_B": _count_reasons(rows),
            "procedure": procs[runway].describe(),
        }
    out = {
        "airport": airport, "stride": args.stride, "n_flights": len(flights),
        "definitions": __doc__, "skipped_runways": skipped, "summary": summary, "flights": flights,
        "elapsed_s": round(time.time() - t0),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{airport}: {len(flights)} flights in {time.time() - t0:.0f} s -> {args.out}")
    for runway, sm in sorted(summary.items()):
        n = max(sm["n"], 1)
        print(f"  {runway:4s} n={sm['n']:6d}  A={sm['A']/n:6.1%}  B={sm['B']/n:6.1%}  C={sm['C']}/{sm['C_decidable']} decidable"
              f"  join<=30°={sm['join_le_30deg']/n:6.1%}  join>=1.2dFAF={sm['join_ge_1p2_dfaf']/n:6.1%}")
    for runway, why in skipped.items():
        print(f"  {runway:4s} skipped: {why}")


if __name__ == "__main__":
    main()
