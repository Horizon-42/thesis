import json, sys, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
import archive_paths   # noqa: E402  (README §2)
RESULTS = archive_paths.RESULTS
sys.path.insert(0, str(HERE))
import compare_plants as cp
from ts_transformer.autopilot import replay, flights as flights_module
from ts_transformer.repo_layout import arrival_manifest_path
flights_module.arrival_manifest_path = lambda a: arrival_manifest_path(a, cp.HARVEST)
RUN = RESULTS / "run_train100"   # was the session's scratch area
params, record, words = replay.open_spec(cp.EXECUTOR, cp.INSTRUCTIONS)
drawn = replay.draw(cp.INSTRUCTIONS, "train", words.spec, words, per_airport=100, seed=1337, groups=(replay.OWN, replay.STAND_IN), row_interval_s=2.0)
tr = np.load(RUN / "tracks.npz")
ev = {}
for a in ["KMSY","KRDU","KSJC","KSMF","KSTL"]:
    ev.update({t["flight_key"]: t for t in json.load(open(RUN/"point-mass/records"/a/"evaluation_report.json"))["trajectories"]})
DIST = [8000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 600, 300, 100, 0]
prof = {d: [] for d in DIST}; by_airport = {}; rows = []
for j, i in enumerate(drawn.indices):
    key = drawn.series[j].scenario.source["flight_key"]
    geom = drawn.geometries[j]
    runway = drawn.signals[j].runway
    cand = next(c for c in geom.candidates if c.ident == runway)
    e, n, h = tr[f"point-mass|{key}|e"], tr[f"point-mass|{key}|n"], tr[f"point-mass|{key}|h"]
    course = math.radians(cand.course_deg)
    along = -((e - cand.threshold_e_m) * math.sin(course) + (n - cand.threshold_n_m) * math.cos(course))   # >0 before the threshold
    vp = cand.vertical_path
    path = cand.elevation_m + vp.crossing_height_m + along * math.tan(math.radians(vp.glidepath_deg))
    err = h - path
    ok = along > -50
    if not ok.any(): continue
    order = np.argsort(along)   # along ascending; interpolate err at the wanted distances
    a_s, e_s = along[order], err[order]
    for d in DIST:
        if a_s[0] <= d <= a_s[-1]:
            prof[d].append(float(np.interp(d, a_s, e_s)))
    rows.append((signals_airport := drawn.signals[j].airport, runway, ev[key].get("vertical_m"), vp.glidepath_deg, vp.crossing_height_m))
print("signed height above the FAS path by distance before the threshold (all flights that reach that distance)")
print(f'{"dist m":>7s} {"n":>4s} {"p10":>7s} {"p50":>7s} {"p90":>7s} {"mean":>7s}')
for d in DIST:
    v = np.array(prof[d])
    print(f'{d:7d} {len(v):4d} {np.percentile(v,10):7.1f} {np.percentile(v,50):7.1f} {np.percentile(v,90):7.1f} {v.mean():7.1f}')
import collections
g = collections.defaultdict(list)
for ap, rw, v, ang, tch in rows:
    if v is not None: g[(ap, rw)].append(v)
print("\nthe evaluation's vertical_m at the threshold, by runway (n>=8): mean / median / glidepath / TCH")
for (ap, rw), v in sorted(g.items()):
    if len(v) >= 8:
        r0 = next(r for r in rows if r[0] == ap and r[1] == rw)
        print(f'{ap} {rw:4s} n={len(v):3d}  mean {np.mean(v):6.1f}  median {np.median(v):6.1f}   {r0[3]:.2f} deg  TCH {r0[4]:.1f} m')

print("\nairport elevation E (the word datum) vs the runway's threshold elevation, and the mean error")
seen = {}
for j, i in enumerate(drawn.indices):
    geom = drawn.geometries[j]; runway = drawn.signals[j].runway
    cand = next(c for c in geom.candidates if c.ident == runway)
    seen[(drawn.signals[j].airport, runway)] = (geom.elevation_m, cand.elevation_m)
tab = []
for (ap, rw), v in sorted(g.items()):
    if len(v) >= 8:
        E, thr = seen[(ap, rw)]
        tab.append((E - thr, np.mean(v)))
        print(f'{ap} {rw:4s} E {E:6.1f} m   threshold {thr:6.1f} m   E - threshold {E - thr:5.1f} m   mean vertical error {np.mean(v):6.1f} m')
x = np.array(tab)
print("correlation (E - threshold) vs mean error:", np.corrcoef(x[:,0], x[:,1])[0,1].round(2))
