"""Build aircraft/performance_index.json from the analysis (WORK/final_mapping.json) + source packs.

Run after final_mapping.py (see the report's §11). Decisions (user, 2026-09-23/24): own parameters where primary documents (or Poll-Schumann) give
mass, wing area and thrust; a substitute airframe where the analysis found one; exclude the rest
(propeller aircraft, rotorcraft/military, no FAA approach speed, no same-class airframe).
OpenAP synonym types the analysis kept become explicit substitutes by their OpenAP surrogate.
"""
import csv
import json
from collections import Counter

from _paths import CODE, LIT, WORK

OUT = CODE / "aircraft/performance_index.json"
LB = 0.45359237
PS_SOURCE = "pycontrails_ps_20250328"
REPORT = "docs/aircraft_performance/2026-09-23_missing_performance_substitution.zh.md"


def csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


mapping = {r["typecode"]: r for r in json.loads((WORK / "final_mapping.json").read_text())}
acd = {}
for r in json.loads((WORK / "acd.json").read_text()):
    acd.setdefault(r["ICAO_Code"], r)
facts = {r["typecode"]: r for r in csv_rows(LIT / "airframe_facts.csv")}
ps = {r["ICAO"]: r for r in csv_rows(LIT / "excerpts/ps_params_rows_for_our_types.csv")}
doc8643 = {}
for r in json.loads((CODE / "aircraft/icao_doc8643.json").read_text())["records"]:
    doc8643.setdefault(r["typecode"], r)
ref_sources = json.loads((CODE / "aircraft/reference_speeds.json").read_text())["sources"]

# Source ids of the airframe pack: the markdown table under "## Sources".
pack = {}
text = (LIT / "airframe_facts_sources.md").read_text()
block = text.split("## Sources", 1)[1].split("\n## ", 1)[0]
for line in block.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) < 10 or cells[0] in ("id", "---") or set(cells[0]) <= {"-"}:
        continue
    sid, tier, title, publisher, revision, url, local, _bytes, sha, _provided = cells[:10]
    pack[sid.strip("`")] = {"title": title, "publisher": publisher, "revision": revision,
                            "url": url.strip("<>"), "local_path": local.strip("`"),
                            "sha256": sha.strip("`")}

used = set()


def num(x):
    return float(str(x).replace(",", ""))


def own_from_facts(t):
    f, a = facts[t], acd[t]
    used.update({"faa_acd_2024_10", f["wing_area_source"], f["thrust_source"]})
    return {
        "decision": "own",
        "name": a["Model_FAA"],
        "mtow_kg": round(num(a["MTOW_lb"]) * LB),
        "mlw_kg": round(num(a["MALW_lb"]) * LB),
        "wing_area_m2": num(f["wing_area_m2"]),
        "engines": int(f["n_engines"]),
        "max_thrust_n_each": round(num(f["rated_takeoff_thrust_kN_each"]) * 1000.0, 1),
        "sources": {"mass": "faa_acd_2024_10", "wing_area": f["wing_area_source"],
                    "thrust": f["thrust_source"]},
    }


def own_from_ps(t):
    p = ps[t]
    engines = int(doc8643[t]["engine_count"])
    used.add(PS_SOURCE)
    return {
        "decision": "own",
        "name": f"{p['Manufacturer']} {p['Type']}",
        "mtow_kg": round(num(p["MTOM_kg"])),
        "mlw_kg": round(num(p["MLM_kg"])),
        "wing_area_m2": num(p["Sref_m2"]),
        "engines": engines,
        # nominal_F00_ISA_kn is summed over all engines (PSAircraftEngineParams docstring)
        "max_thrust_n_each": round(num(p["nominal_F00_ISA_kn"]) * 1000.0 / engines, 1),
        "sources": {"mass": PS_SOURCE, "wing_area": PS_SOURCE, "thrust": PS_SOURCE},
    }


types = {}
for t, r in sorted(mapping.items()):
    v = r["verdict"]
    if v.startswith("own"):
        types[t] = own_from_facts(t)
    elif v.startswith("PS"):
        types[t] = own_from_ps(t)
    elif v.startswith("keep OpenAP synonym"):
        types[t] = {"decision": "substitute", "substitute": r["openap24"],
                    "basis": (f"OpenAP 2.4's own synonym, kept by the analysis: stall margin "
                              f"{r['r_current']}, same FAA approach category")}
    elif v.startswith("surrogate"):
        basis = (f"nearest airframe of the same propulsion class and FAA approach category: "
                 f"similarity distance {r['d']} (A320 {r['d_a320']}), stall margin {r['r_sub']}")
        if not r.get("tw_checked"):
            basis += "; thrust-to-weight not checked (no primary thrust)"
        types[t] = {"decision": "substitute", "substitute": r["sub"], "basis": basis}
    else:
        reason = {
            "exclude: propeller": "propeller aircraft (user decision 2026-09-23)",
            "exclude: not a fixed-wing": "rotorcraft or military type, not a fixed-wing civil approach",
            "no FAA approach speed": "no published approach speed (not in the FAA Aircraft Characteristics Database)",
            "no airframe of the same": "no modelled airframe of the same propulsion class and FAA approach category",
            "no acceptable surrogate": "no substitute with a stall margin above the model's stall branch",
        }
        key = next(k for k in reason if v.startswith(k))
        types[t] = {"decision": "exclude", "reason": reason[key]}

# Every OpenAP synonym the analysis never saw (no train/val flight) gets a decision too, by the
# analysis's own rule for OpenAP synonyms (final_mapping.py): keep the surrogate when its stall
# margin lies in the native range and the FAA approach category matches; a propeller type is
# excluded (user decision 2026-09-23); otherwise excluded. The loader requires a row for every
# synonym, so no synonym is ever left to a silent default.
pool = json.loads((WORK / "pool.json").read_text())
cands = {c: p for c, p in pool.items() if p["faa_vapp_kt"]}
native_r = [p["faa_vapp_kt"] / p["vs_model_kt"] for p in cands.values()]
R_LO, R_HI = min(native_r), max(native_r)


def speed_band(v_kt):
    return "A" if v_kt < 91 else "B" if v_kt < 121 else "C" if v_kt < 141 else "D" if v_kt < 166 else "E"


def aac(code):
    f = acd.get(code) or {}
    try:
        v = num(f["Approach_Speed_knot"])
    except (KeyError, TypeError, ValueError):
        return f.get("AAC") or ""
    return speed_band(v)


openap = json.loads((CODE / "aircraft/openap_aircraft_parameters.json").read_text())["typecodes"]
for code, record in sorted(openap.items()):
    if not record.get("openap_supported") or code in types:
        continue
    surrogate = record["parameters"].get("openap_performance_typecode") or code
    if surrogate == code:
        continue
    f = acd.get(code) or {}
    engine = f.get("Physical_Class_Engine")
    if engine in ("Piston", "Turboprop"):
        types[code] = {"decision": "exclude", "reason": "propeller aircraft (user decision 2026-09-23)"}
        continue
    try:
        vapp = num(f["Approach_Speed_knot"])
    except (KeyError, TypeError, ValueError):
        types[code] = {"decision": "exclude", "reason": "no published approach speed (not in the FAA Aircraft Characteristics Database)"}
        continue
    r_now = round(vapp / pool[surrogate]["vs_model_kt"], 2)
    if R_LO <= r_now <= R_HI and aac(code) == aac(surrogate):
        types[code] = {"decision": "substitute", "substitute": surrogate,
                       "basis": (f"OpenAP 2.4's own synonym, no observed flights; kept by the analysis "
                                 f"rule: stall margin {r_now}, same FAA approach category")}
    else:
        types[code] = {"decision": "exclude",
                       "reason": (f"OpenAP 2.4 synonym of {surrogate} with no observed flights; the "
                                  f"surrogate fails the analysis rule (stall margin {r_now}, approach "
                                  f"category {aac(code)} vs {aac(surrogate)})")}
types = dict(sorted(types.items()))

sources = {}
for sid in sorted(used):
    if sid == PS_SOURCE:
        sources[sid] = {
            "title": "Poll-Schumann aircraft performance parameters (ps-aircraft-params-20250328.csv)",
            "publisher": "pycontrails 0.63.5 (Contrails.org / Breakthrough Energy), Apache-2.0",
            "revision": "2025-03-28",
            "url": "https://pypi.org/project/pycontrails/0.63.5/",
            "local_path": "data/aircraft_performance/pycontrails/",
            "sha256": "2d462e5a8cc4f1fbf8671b8f850232562d0acf86981b24633dc1f96d2978a0db",
        }
    elif sid in pack:
        sources[sid] = pack[sid]
    else:
        s = ref_sources[sid]
        sources[sid] = {"title": s["title"], "publisher": s["publisher"], "revision": s["document"],
                        "url": s["url"], "local_path": s["local_path"], "sha256": s["sha256"]}

payload = {
    "schema": "aircraft-performance-index-v1",
    "generated": "2026-09-24",
    "decisions": REPORT,
    "sources": sources,
    "types": types,
}
OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(Counter(r["decision"] for r in types.values()), len(sources), "sources")
