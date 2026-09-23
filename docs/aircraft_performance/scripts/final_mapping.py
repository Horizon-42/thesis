"""Final substitution table for every type the model has no native dynamics for.

Inputs (all already on disk):
  WORK/census_rows.json     train/val flight -> resolved type + outcome (manifest metadata only)
  acd.json                  FAA Aircraft Characteristics Database 2024-10 (Vapp, MTOW, MALW, engine class)
  pool.json                 every airframe the model can fly today (OpenAP 2.4 direct + presets), model Vs
  <lit>/excerpts/ps_*       Poll-Schumann parameter rows (pycontrails 0.63.5, 2025-03-28 file)
  <lit>/airframe_facts.csv  wing area + rated thrust/power from TCDS / manufacturer docs
  <lit>/coverage_matrix.csv source coverage per type

Similarity (our reading of the point-mass equations): once divided by weight, the dynamics depend
on the landing wing loading over Cl_max (fixed by the approach speed, V_app^2 ~ (W/S)/Cl_max) and on
T/W; the drag polar is one global constant in this model. So
    d = 2 |ln(Vapp_t / Vapp_c)| + |ln(TW_t / TW_c)|      (T/W term only when both are known jets)
with the approach speeds of BOTH sides taken from the same FAA table.
"""
import collections
import csv
import json
import math

from _paths import LIT, WORK

KT = 0.514444
LB = 0.45359237
G = 9.81
RHO0 = 1.225


def num(x):
    try:
        v = float(str(x).replace(",", ""))
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))




def cl_max_rule(code, mtow_kg):  # mirror of aircraft.aero_params.aero_params_for_aircraft
    if code in {"A318", "A319", "A320", "A321", "A19N", "A20N", "A21N"}:
        return 3.0
    if mtow_kg > 100_000.0:
        return 2.4
    if mtow_kg >= 30_000.0:
        return 2.7
    return 2.2


def vs_kt(m, s, cl):
    return math.sqrt(2 * m * G / (RHO0 * s * cl)) / KT


rows = json.loads((WORK / "census_rows.json").read_text())
acd = {}
for r in json.loads((WORK / "acd.json").read_text()):
    acd.setdefault(r["ICAO_Code"], r)
pool = json.loads((WORK / "pool.json").read_text())


def speed_band(v_kt):
    """AAC band of an approach speed; every FAA ACD row but two follows exactly these bounds."""
    return "A" if v_kt < 91 else "B" if v_kt < 121 else "C" if v_kt < 141 else "D" if v_kt < 166 else "E"


def aac(code):
    """The FAA table's Aircraft Approach Category column. Its data dictionary says the category
    is set by the highest reported approach speed; where the column contradicts the table's own
    approach speed (B190, GL7T in the 2024-10 table) the speed decides."""
    f = acd.get(code) or {}
    v = num(f.get("Approach_Speed_knot"))
    col = f.get("AAC") or ""
    return speed_band(v) if v and col and speed_band(v) != col else col


AAC_CONTRADICTIONS = sorted(c for c, f in acd.items()
                            if num(f.get("Approach_Speed_knot")) and f.get("AAC")
                            and speed_band(num(f["Approach_Speed_knot"])) != f["AAC"])

ps = {r["ICAO"]: r for r in csv_rows(LIT / "excerpts/ps_params_rows_for_our_types.csv")}
ps_syn = {r["ICAO Aircraft Code"]: r["PS ATYP"] for r in csv_rows(LIT / "excerpts/ps_synonym_rows_for_our_types.csv")}
cov = {r["typecode"]: r for r in csv_rows(LIT / "coverage_matrix.csv")}
facts = {r["typecode"]: r for r in csv_rows(LIT / "airframe_facts.csv")}
missing = csv_rows(WORK / "missing_types.csv")
syn_now = json.loads((WORK / "synonym_current.json").read_text())   # OpenAP 2.4 synonym as flown today

train = collections.Counter(r["typecode"] for r in rows if r["split"] == "train")
val = collections.Counter(r["typecode"] for r in rows if r["split"] == "val")

cands = {c: p for c, p in pool.items() if p["faa_vapp_kt"]}
# The model's own acceptance range: stall margin Vapp/Vs_model over every airframe it flies natively.
native_r = sorted(p["faa_vapp_kt"] / p["vs_model_kt"] for p in cands.values())
R_LO, R_HI = native_r[0], native_r[-1]
STALL_ONSET = 1 / math.sqrt(0.9)  # AeroParams.stall_threshold = 0.9 of Cl_max
a320 = pool["A320"]

out = []
for m in missing:
    t = m["typecode"]
    f = acd.get(t) or {}
    seg = (cov.get(t) or {}).get("segment", "")
    vapp = num(f.get("Approach_Speed_knot"))
    mtow = num(f.get("MTOW_lb")); mtow = mtow * LB if mtow else None
    malw = num(f.get("MALW_lb")); malw = malw * LB if malw else None
    eng = f.get("Physical_Class_Engine")
    fa = facts.get(t) or {}
    # target airframe facts: primary documents (TCDS / manufacturer, airframe_facts) first,
    # the Poll-Schumann row only where they are missing (it disagrees with the TCDS for E75S S)
    ps_code = t if t in ps else ps_syn.get(t)
    ps_native = t in ps
    S_prim = num(fa.get("wing_area_m2"))
    S_t = S_prim or (num(ps[t]["Sref_m2"]) if ps_native else None)
    S_src = (fa.get("wing_area_source") or "") if S_prim else ("PS" if ps_native else "")
    thrust_prim = (num(fa["rated_takeoff_thrust_kN_each"]) * 1e3 * num(fa["n_engines"])
                   if num(fa.get("rated_takeoff_thrust_kN_each")) and num(fa.get("n_engines")) else None)
    thrust_t = thrust_prim or (num(ps[t]["nominal_F00_ISA_kn"]) * 1e3 if ps_native else None)
    thrust_src = (fa.get("thrust_source") or "") if thrust_prim else ("PS F00" if ps_native else "")
    power_t = None
    if num(fa.get("rated_takeoff_power_kW_each")) and num(fa.get("n_engines")):
        power_t = num(fa["rated_takeoff_power_kW_each"]) * num(fa["n_engines"])
    tw_t = thrust_t / (mtow * G) if thrust_t and mtow else None
    # stall margin with the type's own mass + wing area under the model's generic Cl_max rule
    r_geom = round(vapp / vs_kt(malw, S_t, cl_max_rule(t, mtow)), 2) if vapp and malw and S_t and mtow else None
    r_now = round(vapp / syn_now[t]["vs_kt"], 2) if vapp and t in syn_now else None
    ws_t = malw / S_t if malw and S_t else None

    rec = dict(typecode=t, segment=seg, status=m["status"], openap24=m["openap24_surrogate"],
               train=train.get(t, 0), val=val.get(t, 0), faa_engine=eng or "", faa_vapp_kt=vapp,
               vapp_ms=round(vapp * KT, 1) if vapp else None, aac=aac(t),
               mtow_kg=round(mtow) if mtow else None, malw_kg=round(malw) if malw else None,
               S_m2=S_t, S_src=S_src, thrust_total_kN=round(thrust_t / 1e3, 1) if thrust_t else None,
               thrust_src=thrust_src, power_total_kW=round(power_t) if power_t else None,
               tw=round(tw_t, 3) if tw_t else None, ws_landing=round(ws_t) if ws_t else None,
               r_a320=round(vapp / a320["vs_model_kt"], 2) if vapp else None,
               r_geom=r_geom, r_current=r_now if t in syn_now else (round(vapp / a320["vs_model_kt"], 2) if vapp else None))

    # option 0: the type's own m, S, T from primary documents (jets only: the model's T is a force)
    own = bool(eng == "Jet" and malw and S_prim and thrust_prim)
    if own:
        vs_own = vs_kt(malw, S_prim, cl_max_rule(t, mtow))
        rec.update(own_params=True, own_vs_model_kt=round(vs_own, 1),
                   r_own=round(vapp / vs_own, 2) if vapp else None)
    else:
        rec.update(own_params=False)

    # option 1: Poll-Schumann parameters (native row, or its own synonym list)
    if ps_code:
        p = ps[ps_code]
        mtom, mlm, s = num(p["MTOM_kg"]), num(p["MLM_kg"]), num(p["Sref_m2"])
        vs = vs_kt(mlm, s, cl_max_rule(ps_code, mtom))
        rec.update(ps_code=ps_code, ps_native=ps_native, ps_vs_model_kt=round(vs, 1),
                   r_ps=round(vapp / vs, 2) if vapp else None,
                   ps_tw=round(num(p["nominal_F00_ISA_kn"]) * 1e3 / (mtom * G), 3))

    # option 2: nearest airframe the model already flies. Candidates must share the propulsion
    # class (FAA Physical_Class_Engine) AND the FAA Aircraft Approach Category; the nearest one
    # over the whole pool is still recorded, as a reference, when no candidate qualifies.
    verdict = None
    if seg in ("rotorcraft", "military"):
        verdict = "exclude: not a fixed-wing civil approach"
    if vapp:
        jet_tw = tw_t is not None and eng == "Jet"

        def dist(c):
            p = cands[c]
            d_v = 2 * abs(math.log(vapp / p["faa_vapp_kt"]))
            d_tw = abs(math.log(tw_t / p["TW"])) if jet_tw else 0.0
            return d_v + d_tw, d_v, d_tw

        def rank(pool_c):  # exact ties in d (FAA speeds are whole knots) -> the closer take-off mass
            return sorted(pool_c, key=lambda c: (round(dist(c)[0], 9),
                                                 abs(math.log(mtow / cands[c]["mtow_kg"])) if mtow else 0.0))

        qualified = rank([c for c, p in cands.items() if p["faa_engine"] == eng and aac(c) == aac(t)])
        ranked = qualified or rank(list(cands))
        best = ranked[0]
        d, d_v, d_tw = dist(best)
        rb = vapp / cands[best]["vs_model_kt"]
        a = cands["A320"]
        rec["d_a320"] = round(2 * abs(math.log(vapp / a["faa_vapp_kt"]))
                              + (abs(math.log(tw_t / a["TW"])) if jet_tw else 0.0), 3)
        rec.update(sub=best, sub_second=ranked[1] if len(ranked) > 1 else "", d=round(d, 3),
                   d_v=round(d_v, 3), d_tw=round(d_tw, 3) if jet_tw else None, tw_checked=jet_tw,
                   class_match=bool(qualified), sub_engine=cands[best]["faa_engine"],
                   sub_vapp_kt=cands[best]["faa_vapp_kt"], sub_aac=aac(best), r_sub=round(rb, 2),
                   sub_tw=round(cands[best]["TW"], 3))
        if verdict:
            pass
        elif (r_now is not None and R_LO <= r_now <= R_HI
              and aac(t) == aac(m["openap24_surrogate"])):
            verdict = f"keep OpenAP synonym ({m['openap24_surrogate']})"
        elif own and R_LO <= rec["r_own"] <= R_HI:
            verdict = "own parameters (primary documents)"
        elif ps_code and ps_native:
            verdict = f"PS parameters ({ps_code})"
        elif not qualified:
            verdict = "no airframe of the same propulsion class and approach category: exclude or new data"
        elif rb < STALL_ONSET:
            verdict = "no acceptable surrogate: exclude or new data"
        elif R_LO <= rb <= R_HI:
            verdict = f"surrogate {best}"
        else:
            verdict = f"surrogate {best} (stall margin outside native range)"
    elif not verdict:
        verdict = "no FAA approach speed: exclude or new data"
    rec["verdict"] = verdict
    notes = []
    if t in AAC_CONTRADICTIONS:
        notes.append(f"FAA AAC column {f.get('AAC')} contradicts its approach speed; category {aac(t)} from the speed")
    if rec.get("d") is not None and rec.get("class_match") and verdict.startswith("surrogate") and rec["d"] >= rec["d_a320"]:
        notes.append("surrogate is not closer than A320 in similarity distance")
    if vapp and eng == "Jet" and not rec.get("tw_checked"):
        notes.append("thrust-to-weight not checked (no thrust from a primary document)")
    if verdict.startswith("surrogate") and "outside" in verdict and rec.get("r_geom"):
        notes.append(f"stall margin with own mass and wing area would be {rec['r_geom']}")
    if t in ps and S_prim and abs(num(ps[t]["Sref_m2"]) / S_prim - 1) > 0.02:
        notes.append(f"PS wing area {ps[t]['Sref_m2']} m2 differs from the primary document {S_prim} m2")
    rec["notes"] = "; ".join(notes)
    out.append(rec)

(WORK / "final_mapping.json").write_text(json.dumps(out, indent=1, default=str))
print("native stall-margin range", round(R_LO, 2), round(R_HI, 2), "stall onset", round(STALL_ONSET, 3))
v = collections.Counter()
for r in out:
    v[r["verdict"].split(" (")[0] if r["verdict"].startswith("surrogate") else r["verdict"].split(":")[0]] += r["train"]
print(v.most_common())
