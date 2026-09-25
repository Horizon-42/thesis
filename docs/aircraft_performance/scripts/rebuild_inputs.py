"""Rebuild the decision inputs from the stored table, for re-applying the rule (final_mapping.py) to changed masses.

census.py / synonym_current.py can no longer rebuild the analysis's inputs: on code with the performance index they
read the index's own decisions (§11). This script rebuilds every WORK input final_mapping.py reads from the stored
table 2026-09-23_substitution_table.csv (per-type train/val counts, status and OpenAP surrogate as the analysis saw
them) and the FAA table, and recomputes the two mass-dependent inputs with the current code:
  pool.json               every native airframe's model stall speed at its landing mass (the native range, r_sub)
  synonym_current.json    each OpenAP synonym flown with OpenAP's copy of its surrogate's parameters (r_now)
MODE:
  current     the code's landing masses (since 2026-09-26: presets and own types at the published MALW)
  2026-09-24  presets at 0.85 x MTOW, as the analysis ran; final_mapping.py then reproduces every verdict of the
              stored table (one display column differs, C56X's stall_margin_now: the table was re-rendered after
              the index existed, and its own-parameter verdict comes before that margin is read)
  malw        what-if: every native airframe and synonym surrogate at its published MALW
Run after read_acd.py: python rebuild_inputs.py MODE (writes into AERO_SUB_WORK).
"""
import csv
import json
import math
import sys

from _paths import CODE, OUT, WORK, use_repo_code

use_repo_code()
from aircraft.aero_params import aero_params_for_aircraft, stall_speed_ms  # noqa: E402
from aircraft.aircraft_sets import AIRCRAFT_PRESETS  # noqa: E402
from aircraft.query_aircraft_parameters import PARAMETERS_PATH, load_json, openap_direct_typecodes  # noqa: E402
from flight_scenarios.scenario import aircraft_for_code  # noqa: E402

MODE = sys.argv[1]
assert MODE in ("current", "2026-09-24", "malw"), MODE
MS_TO_KT = 1 / 0.514444
KT, G, RHO0 = 0.514444, 9.81, 1.225
table = list(csv.DictReader(open(OUT / "2026-09-23_substitution_table.csv")))
ref = json.loads((CODE / "aircraft/reference_speeds.json").read_text())["types"]
acd = {}
for r in json.load(open(WORK / "acd.json")):
    acd.setdefault(r["ICAO_Code"], r)


def landing_mass(code, aircraft):
    if MODE == "malw":
        return ref[code]["malw_kg"] if code in ref else aircraft.landing_mass
    if MODE == "2026-09-24" and code in AIRCRAFT_PRESETS:
        return 0.85 * aircraft.mass.max_takeoff_kg
    return aircraft.landing_mass


# census counts (only the per-type train / val counts are read)
rows = []
for r in table:
    rows += [{"typecode": r["typecode"], "split": "train"}] * int(r["train_flights"])
    rows += [{"typecode": r["typecode"], "split": "val"}] * int(r["val_flights"])
(WORK / "census_rows.json").write_text(json.dumps(rows))

# missing types: status and OpenAP surrogate as the analysis saw them
with open(WORK / "missing_types.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["typecode", "status", "openap24_surrogate"])
    for r in table:
        syn = r["status_now"].startswith("OpenAP")
        w.writerow([r["typecode"], "openap_2_4_synonym" if syn else "no_dynamics(A320 fallback)",
                    r["status_now"].split()[-1] if syn else ""])


def cl_max_rule(code, mtow_kg):  # final_mapping.py's mirror of aero_params
    if code in {"A318", "A319", "A320", "A321", "A19N", "A20N", "A21N"}:
        return 3.0
    return 2.4 if mtow_kg > 100_000.0 else 2.7 if mtow_kg >= 30_000.0 else 2.2


# synonyms as flown: OpenAP's copy of the surrogate's parameters under the synonym's code
params = load_json(PARAMETERS_PATH)["typecodes"]
syn_now = {}
for r in table:
    if not r["status_now"].startswith("OpenAP"):
        continue
    t, surrogate = r["typecode"], r["status_now"].split()[-1]
    p = params[t]["parameters"]
    m = ref[surrogate]["malw_kg"] if MODE == "malw" else p["mass"]["mlw_kg"]
    s, mtow = p["geometry"]["wing_area_m2"], p["mass"]["mtow_kg"]
    syn_now[t] = {"vs_kt": math.sqrt(2 * m * G / (RHO0 * s * cl_max_rule(t, mtow))) / KT}
(WORK / "synonym_current.json").write_text(json.dumps(syn_now))

# the native pool
pool = {}
for code in sorted(set(openap_direct_typecodes()) | set(AIRCRAFT_PRESETS)):
    a = aircraft_for_code(code)
    aero = aero_params_for_aircraft(a)
    m = landing_mass(code, a)
    f = acd.get(code, {})
    pool[code] = {"source": "preset" if code in AIRCRAFT_PRESETS else "openap_direct", "mtow_kg": a.mass.max_takeoff_kg,
                  "landing_mass_kg": m, "TW": a.engine.max_thrust_total_n / (a.mass.max_takeoff_kg * 9.81),
                  "vs_model_kt": stall_speed_ms(m, wing_area_m2=aero.S, cl_max=aero.Cl_max) * MS_TO_KT,
                  "faa_vapp_kt": float(f["Approach_Speed_knot"]) if f.get("Approach_Speed_knot") not in (None, "N/A") else None,
                  "faa_engine": f.get("Physical_Class_Engine")}
(WORK / "pool.json").write_text(json.dumps(pool, indent=1))
margins = sorted((p["faa_vapp_kt"] / p["vs_model_kt"], c) for c, p in pool.items() if p["faa_vapp_kt"])
print(MODE, "native range", [(c, round(r, 3)) for r, c in margins[:3]], "…", [(c, round(r, 3)) for r, c in margins[-1:]])
