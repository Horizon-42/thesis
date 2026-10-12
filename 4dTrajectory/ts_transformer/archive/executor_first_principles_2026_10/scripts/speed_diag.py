import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
import archive_paths   # noqa: E402  (README §2)
RESULTS = archive_paths.RESULTS
sys.path.insert(0, str(HERE))
import compare_plants as cp
from ts_transformer.autopilot import replay, flights as flights_module
from ts_transformer.instructions.artefact import closed_loop_sentences
from ts_transformer.instructions.words import SPEED, UNCHANGED
from ts_transformer.repo_layout import arrival_manifest_path
flights_module.arrival_manifest_path = lambda a: arrival_manifest_path(a, cp.HARVEST)
RUN = RESULTS / "run_train100"   # was the session's scratch area
rows = {r["flight_key"]: r for r in json.loads((RUN/"rows.json").read_text())["point-mass"]}
obs = json.loads((RESULTS / "observed_verdicts_500.json").read_text())
ev = {}
for a in ["KMSY","KRDU","KSJC","KSMF","KSTL"]:
    ev.update({t["flight_key"]: t for t in json.loads((RUN/"point-mass/records"/a/"evaluation_report.json").read_text())["trajectories"]})
params, record, words = replay.open_spec(cp.EXECUTOR, cp.INSTRUCTIONS)
drawn = replay.draw(cp.INSTRUCTIONS, "train", words.spec, words, per_airport=100, seed=1337, groups=(replay.OWN, replay.STAND_IN), row_interval_s=2.0)
stored = closed_loop_sentences(cp.INSTRUCTIONS, "train", 2.0, words.spec)
info = {}
for j, i in enumerate(drawn.indices):
    key = drawn.series[j].scenario.source["flight_key"]
    col = stored[i].rows.grid[:, SPEED]
    said = col[col != UNCHANGED]
    last = int(said[-1]) if len(said) else None
    sp = words.speed_mps(last) if last is not None else None
    info[key] = {"last_speed_word_mps": sp, "specified": sp is not None, "group": drawn.groups[j],
                 "type": drawn.series[j].scenario.source.get("resolved_typecode"),
                 "dyn_type": drawn.series[j].scenario.source.get("dynamics_typecode")}
fail = [k for k, t in ev.items() if "speed" in t["violations"]]
ok = [k for k, t in ev.items() if t["speed_result"] == "pass"]
def share(ks, f): return round(sum(f(k) for k in ks) / len(ks), 3)
print("n speed-fail", len(fail), " n speed-pass", len(ok))
print("last speed word is a SPECIFIED ground-speed word: fail", share(fail, lambda k: info[k]["specified"]), " pass", share(ok, lambda k: info[k]["specified"]))
print("group (own / stand-in): fail", Counter(info[k]["group"] for k in fail), " pass share stand-in", share(ok, lambda k: info[k]["group"] != "own dynamics"))
print("types of the failing:", Counter(info[k]["type"] for k in fail).most_common(8))
print()

import math
from aerodynamic_model.torch_dynamics import ISA_RHO0_KG_M3
tab = {}
for j, i2 in enumerate(drawn.indices):
    key = drawn.series[j].scenario.source["flight_key"]
    tab[key] = (drawn.batch_ias if False else None)
batch, _ = __import__("ts_transformer.autopilot.closed_loop", fromlist=["x"]).replay_batch(drawn, {i2: stored[i2] for i2 in drawn.indices if i2 in stored}, words)
ias = {batch.series[j].scenario.source["flight_key"]: batch.approach_ias_mps[j] for j in range(len(batch.series))}
elev = {batch.series[j].scenario.source["flight_key"]: batch.geometries[j].elevation_m for j in range(len(batch.series))}
def rho(h):  # ISA density, same law the dynamics use (troposphere)
    T = 288.15 - 0.0065 * h
    return 1.225 * (T / 288.15) ** 4.25588
print(f'{"flight":38s} {"type":6s} {"grp":3s} {"own TAS":>8s} {"replay V":>9s} {"lower":>7s} {"upper":>7s} {"V-own":>6s} {"own-lower":>9s}')
res = []
for k in fail + ok[:0]:
    t = ev[k]
    own = ias[k] * math.sqrt(1.225 / rho(elev[k] + 15.0))
    res.append((t["crossing_speed_ms"] - own, own - t["bounds"]["speed_lower_ms"]))
    print(f'{k[:38]:38s} {str(info[k]["type"]):6s} {"own" if info[k]["group"].startswith("own") else "std":3s} {own:8.1f} {t["crossing_speed_ms"]:9.1f} '
          f'{t["bounds"]["speed_lower_ms"]:7.1f} {t["bounds"]["speed_upper_ms"]:7.1f} {t["crossing_speed_ms"]-own:6.1f} {own-t["bounds"]["speed_lower_ms"]:9.1f}')
a = np.array(res)
print("failing: replay V minus own target  p10/p50/p90", np.percentile(a[:,0],[10,50,90]).round(2), "  own target minus gate lower bound", np.percentile(a[:,1],[10,50,90]).round(2))
allr = []
for k in ev:
    if ev[k].get("crossing_speed_ms") is None: continue
    own = ias[k] * math.sqrt(1.225 / rho(elev[k] + 15.0))
    allr.append((ev[k]["crossing_speed_ms"] - own, own - ev[k]["bounds"]["speed_lower_ms"]))
b = np.array(allr)
print("all 500: replay V minus own target p10/p50/p90", np.percentile(b[:,0],[10,50,90]).round(2), "  own minus lower", np.percentile(b[:,1],[10,50,90]).round(2))
print("share with own target below the gate's lower bound: failing", (a[:,1] < 0).mean().round(2), " all", (b[:,1] < 0).mean().round(3))

print()
from aircraft.reference_speeds import reference_speed
print("per failing flight: type, approach_kt / min_kt / max_kt, MALW, initial mass, crossing mass, mass ratio, own IAS, gate lower (IAS)")
seen = 0
for j in range(len(batch.series)):
    k = batch.series[j].scenario.source["flight_key"]
    if k not in fail or seen >= 12: continue
    seen += 1
    ty = batch.series[j].scenario.source["resolved_typecode"]
    r = reference_speed(ty)
    m0 = float(batch.series[j].scenario.initial.m)
    mc = ev[k]["deviation"]["crossing_mass_kg"]
    print(f'{ty:5s} {r.approach_speed_kt:5.0f}/{r.approach_speed_min_kt:5.0f}/{r.approach_speed_max_kt:5.0f}  MALW {r.malw_kg:7.0f}  m0 {m0:7.0f}  mc {mc:7.0f}  mc/m0 {mc/m0:6.3f}  '
          f'own {ias[k]:5.1f}  lower {ev[k]["bounds"]["speed_lower_ms"]:5.1f}  group {info[k]["group"][:3]}')
