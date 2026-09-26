"""How well do OBSERVED arrivals satisfy the optimizer's procedure-constraint rows?

Read-only over the harvest (``tracks/`` + ``arrivals/manifest.json``) and the CIFP
procedure documents the optimizer consumes. Answers, per runway:

* did the flight pass an OFF-AXIS entry fix of its runway's RNAV(GPS) procedure (an IAF /
  transition start, > 2 km off the extended centreline) — i.e. did it fly the coded
  transition at all;
* where along the final approach course it became ESTABLISHED (the earliest along-track
  distance from which it stays inside the k·halfwidth LPV cone to the threshold), against
  the FAF distance the optimizer's flexible join window keys on;
* on the inside-FAF tail, the share of samples (and of whole flights) inside the LPV cone
  (k = 0.5 and 1.0) and inside the glidepath window (the optimizer's −60/+120 m and the
  evaluation gate's ±22 m).

Geometry mirrors ``aeroviz_backend/procedure_segments._lpv_spec`` (runway length unknown ⇒
the 9023 ft FPAP floor, course width = max(350 ft, tan 1.5°·d_GARP)) and
``approach_constraints.lateral.lpv_course_halfwidth`` / ``vertical.glidepath_altitude``.
Altitudes: stored tracks are HAE, converted once with the manifest's ``hae_minus_msl_m``.

    conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/measure_procedure_adherence.py KRDU KSJC --stride 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from geokit import FT_M, METRES_PER_DEG_LAT, NM_M

ROOT = Path(__file__).resolve().parents[3]
K_MARGIN = 0.5                                   # approach_constraints.DEFAULT_K_MARGIN
D_GARP_LTP_M = (9023.0 + 1000.0) * FT_M          # FPAP floor + GARP offset
COURSE_WIDTH_M = max(350.0 * FT_M, math.tan(math.radians(1.5)) * D_GARP_LTP_M)
OFF_AXIS_M = 2000.0
TAIL_MIN_D_M = 300.0


def halfwidth_m(d: np.ndarray) -> np.ndarray:
    return COURSE_WIDTH_M * (d + D_GARP_LTP_M) / D_GARP_LTP_M


def rnav_gps_document(airport: str, runway: str) -> dict | None:
    root = ROOT / "aeroviz-4d/public/data/airports" / airport / "procedure-details"
    for entry in json.load(open(root / "index.json"))["runways"]:
        if entry["runwayIdent"] != f"RW{runway}":
            continue
        for proc in entry["procedures"]:
            if proc["procedureFamily"] == "RNAV_GPS":
                return json.load(open(root / f"{proc['procedureUid']}.json"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("airports", nargs="+")
    parser.add_argument("--stride", type=int, default=3, help="every Nth rostered arrival (landing-time order)")
    args = parser.parse_args()
    for airport in args.airports:
        measure(airport.upper(), args.stride)


def measure(airport: str, stride: int) -> None:
    harvest = ROOT / "trajectory_data_process/outputs/harvest" / airport
    manifest = json.load(open(harvest / "arrivals/manifest.json"))
    targets = manifest["runway_targets"]
    procs: dict[str, dict] = {}
    for runway, target in targets.items():
        doc = rnav_gps_document(airport, runway)
        if doc is None:
            print(f"{airport} {runway}: no RNAV(GPS) procedure document")
            continue
        profile = doc["verticalProfiles"][0]
        samples = {c["role"]: c for c in profile["constraintSamples"]}
        d_faf = samples["MAPt"]["distanceFromStartM"] - samples["FAF"]["distanceFromStartM"]
        lat0, lon0 = target["lat"], target["lon"]
        m_lon = METRES_PER_DEG_LAT * math.cos(math.radians(lat0))
        course = math.radians(target["course_deg"])
        u_e, u_n = math.sin(course), math.cos(course)
        off_axis = []
        for fix in doc["fixes"]:
            if {"FAF", "MAPt", "MAHF"} & set(fix["roleHints"]):
                continue
            e = (fix["position"]["lon"] - lon0) * m_lon
            n = (fix["position"]["lat"] - lat0) * METRES_PER_DEG_LAT
            if abs(e * u_n - n * u_e) > OFF_AXIS_M:
                off_axis.append((fix["ident"], fix["position"]["lat"], fix["position"]["lon"]))
        procs[runway] = dict(
            d_faf=d_faf, gpa=profile["glidepathAngleDeg"],
            tch=profile["thresholdCrossingHeightFt"] * FT_M, off_axis=off_axis,
        )
        print(f"{airport} {runway}: d_FAF {d_faf:.0f} m, GPA {profile['glidepathAngleDeg']}, "
              f"off-axis entry fixes {[f[0] for f in off_axis]}")

    stats: dict[str, dict] = {}
    for record in manifest["records"][::stride]:
        runway = record["runway"]
        if runway not in procs:
            continue
        target, proc = targets[runway], procs[runway]
        track = json.load(open(harvest / "tracks" / record["source_file"]))
        s = np.asarray(track["samples"], dtype=float)[record["first_sample_index"]: record["last_sample_index"] + 1]
        lat0, lon0 = target["lat"], target["lon"]
        m_lon = METRES_PER_DEG_LAT * math.cos(math.radians(lat0))
        e = (s[:, 1] - lon0) * m_lon
        n = (s[:, 2] - lat0) * METRES_PER_DEG_LAT
        h_msl = s[:, 3] - target["hae_minus_msl_m"]
        course = math.radians(target["course_deg"])
        u_e, u_n = math.sin(course), math.cos(course)
        d = -(e * u_e + n * u_n)                    # along-track distance back from the threshold
        xt = e * u_n - n * u_e                       # signed cross-track

        def closest_m(lat: float, lon: float) -> float:
            return float(np.min(np.hypot((s[:, 1] - lon) * m_lon, (s[:, 2] - lat) * METRES_PER_DEG_LAT)))

        d_iaf = min((closest_m(la, lo) for _, la, lo in proc["off_axis"]), default=math.nan)

        inside = (np.abs(xt) <= K_MARGIN * halfwidth_m(np.maximum(d, 0.0))) & (d > 0)
        stays = np.ones(len(s), bool)
        acc = True
        for i in range(len(s) - 1, -1, -1):
            if d[i] <= TAIL_MIN_D_M:
                continue
            acc = acc and bool(inside[i])
            stays[i] = acc
        joined = np.where(stays & (d > TAIL_MIN_D_M))[0]
        d_join = float(d[joined[0]]) if len(joined) else math.nan

        upstream = np.where(d >= proc["d_faf"])[0]
        tail = slice(int(upstream[-1]) + 1 if len(upstream) else 0, len(s))
        keep = d[tail] > TAIL_MIN_D_M
        d_t, xt_t, h_t = d[tail][keep], xt[tail][keep], h_msl[tail][keep]
        if len(d_t) < 3:
            continue
        hw = halfwidth_m(d_t)
        dv = h_t - (target["elevation_msl_m"] + proc["tch"] + d_t * math.tan(math.radians(proc["gpa"])))
        cone_k, cone_1 = np.abs(xt_t) <= K_MARGIN * hw, np.abs(xt_t) <= hw
        gp_opt, gp_gate = (dv >= -60.0) & (dv <= 120.0), np.abs(dv) <= 22.0

        st = stats.setdefault(runway, dict(n=0, iaf=0, est_faf=0, samples=0, cone_k=0, cone_1=0,
                                           cone_k_all=0, gp_opt=0, gp_opt_all=0, gp_gate=0, gp_gate_all=0,
                                           d_join=[], dv=[]))
        st["n"] += 1
        st["iaf"] += d_iaf <= NM_M
        st["est_faf"] += abs(xt_t[0]) <= K_MARGIN * hw[0]
        st["samples"] += len(d_t)
        st["cone_k"] += int(cone_k.sum()); st["cone_1"] += int(cone_1.sum()); st["cone_k_all"] += bool(cone_k.all())
        st["gp_opt"] += int(gp_opt.sum()); st["gp_opt_all"] += bool(gp_opt.all())
        st["gp_gate"] += int(gp_gate.sum()); st["gp_gate_all"] += bool(gp_gate.all())
        if not math.isnan(d_join):
            st["d_join"].append(d_join)
        st["dv"].extend(dv.tolist())

    print(f"\n{airport}: every {stride}th rostered arrival; tail = samples inside the FAF along-track, d > {TAIL_MIN_D_M:.0f} m")
    print("rwy     n | IAF<=1NM | est@FAF | cone k=.5 samp / all-flight | cone k=1 samp | GP[-60,+120] samp / all | |GP|<=22 samp / all | d_join p10/p50/p90 km | joined>=d_FAF | GP dev p05/p50/p95 m")
    for runway, st in sorted(stats.items()):
        n, m = st["n"], st["samples"]
        dj = np.array(st["d_join"]); dv = np.array(st["dv"])
        print(f"{runway:4s} {n:5d} | {st['iaf']/n:7.1%} | {st['est_faf']/n:6.1%} | {st['cone_k']/m:6.1%} / {st['cone_k_all']/n:6.1%}"
              f"          | {st['cone_1']/m:6.1%}       | {st['gp_opt']/m:6.1%} / {st['gp_opt_all']/n:6.1%}      | {st['gp_gate']/m:6.1%} / {st['gp_gate_all']/n:6.1%}"
              f"    | {np.percentile(dj,10)/1e3:5.1f}/{np.percentile(dj,50)/1e3:5.1f}/{np.percentile(dj,90)/1e3:5.1f}       | {np.mean(dj >= procs[runway]['d_faf']):6.1%}"
              f"       | {np.percentile(dv,5):+5.0f}/{np.percentile(dv,50):+5.0f}/{np.percentile(dv,95):+5.0f}")


if __name__ == "__main__":
    main()
