"""Every type the model has no native dynamics for, with its FAA ACD facts -> WORK/missing_types.csv.

Rows: types whose train/val flights fly the A320 fallback (known type, no dynamics) or an
OpenAP 2.4 synonym. Written in train-flight order; the source pack's coverage_matrix.csv
was built from this file.
"""
import collections
import csv
import json

from _paths import CODE, WORK

rows = json.loads((WORK / "census_rows.json").read_text())
acd = {}
for r in json.loads((WORK / "acd.json").read_text()):
    acd.setdefault(r["ICAO_Code"], r)
cat = {}
for r in json.loads((CODE / "aircraft/icao_doc8643.json").read_text())["records"]:
    cat.setdefault(r["typecode"], []).append(r)
OUTCOMES = ("no_dynamics->A320", "openap_synonym")
tr = collections.Counter((r["typecode"], r["outcome"], r["perf"]) for r in rows
                         if r["split"] == "train" and r["outcome"] in OUTCOMES)
va = collections.Counter((r["typecode"], r["outcome"], r["perf"]) for r in rows
                         if r["split"] == "val" and r["outcome"] in OUTCOMES)
keys = sorted(set(tr) | set(va), key=lambda k: (-tr.get(k, 0), k[0]))
with open(WORK / "missing_types.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["typecode", "status", "openap24_surrogate", "train_flights", "val_flights", "icao_desc", "wtc",
                "faa_engine_class", "faa_num_engines", "faa_mtow_lb", "faa_malw_lb", "faa_approach_speed_kt",
                "faa_model_bada", "faa_model", "icao_models"])
    for k in keys:
        t, oc, perf = k
        a = acd.get(t, {})
        c = cat.get(t, [{}])
        w.writerow([t, "no_dynamics(A320 fallback)" if oc != "openap_synonym" else "openap_2_4_synonym",
                    perf or "", tr.get(k, 0), va.get(k, 0), c[0].get("description"), c[0].get("wtc"),
                    a.get("Physical_Class_Engine"), a.get("Num_Engines"), a.get("MTOW_lb"), a.get("MALW_lb"),
                    a.get("Approach_Speed_knot"), a.get("Model_BADA"), a.get("Model_FAA"),
                    " / ".join(sorted({x["manufacturer"] + " " + x["model"] for x in c if x}))[:300]])
print(len(keys), "types")
