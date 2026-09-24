"""Candidate pool (every airframe the project can fly today) + the model stall speed of each.

Vs_model = sqrt(2 m g / (rho0 S Cl_max)) with m = landing mass, exactly the project's
stall model (aircraft.aero_params.stall_speed_ms at n = 1).
"""
import json

from _paths import WORK, use_repo_code

use_repo_code()
from aircraft.aero_params import aero_params_for_aircraft, stall_speed_ms  # noqa: E402
from aircraft.aircraft_sets import AIRCRAFT_PRESETS  # noqa: E402
from aircraft.query_aircraft_parameters import openap_direct_typecodes  # noqa: E402
from flight_scenarios.scenario import aircraft_for_code  # noqa: E402

MS_TO_KT = 1 / 0.514444
acd = {}
for r in json.loads((WORK / "acd.json").read_text()):
    acd.setdefault(r["ICAO_Code"], r)

# Every airframe the model can fly natively: OpenAP direct types + presets (B3XM, with no
# published approach speed, is not an OpenAP direct type since 2026-09-24).
pool = sorted(set(openap_direct_typecodes()) | set(AIRCRAFT_PRESETS))
out = {}
for code in pool:
    a = aircraft_for_code(code)            # provider "auto": presets win, as in the pipeline
    aero = aero_params_for_aircraft(a)
    m = a.landing_mass
    vs = stall_speed_ms(m, wing_area_m2=aero.S, cl_max=aero.Cl_max) * MS_TO_KT
    f = acd.get(code, {})
    out[code] = {
        "source": "preset" if code in AIRCRAFT_PRESETS else "openap_direct",
        "name": a.name, "mtow_kg": a.mass.max_takeoff_kg, "landing_mass_kg": m,
        "S_m2": aero.S, "cl_max": aero.Cl_max, "T_total_N": a.engine.max_thrust_total_n,
        "TW": a.engine.max_thrust_total_n / (a.mass.max_takeoff_kg * 9.81),
        "WS_landing": m / aero.S, "vs_model_kt": vs,
        "vref_published_kt": a.approach.speeds.approach_speed_kt,
        "faa_vapp_kt": float(f["Approach_Speed_knot"]) if f.get("Approach_Speed_knot") not in (None, "N/A") else None,
        "faa_engine": f.get("Physical_Class_Engine"), "faa_mtow_lb": f.get("MTOW_lb"), "faa_malw_lb": f.get("MALW_lb"),
    }
(WORK / "pool.json").write_text(json.dumps(out, indent=1))
print(f"{'code':5s} {'src':6s} {'MTOWt':>6s} {'mLt':>6s} {'S':>6s} {'Cl':>4s} {'T/W':>5s} {'W/S':>5s} {'Vs_mod':>6s} {'Vapp':>5s} {'Vapp/Vs':>7s} {'Vref':>5s} eng")
for c, v in out.items():
    ratio = v["faa_vapp_kt"] / v["vs_model_kt"] if v["faa_vapp_kt"] else float("nan")
    print(f"{c:5s} {v['source'][:6]:6s} {v['mtow_kg']/1e3:6.1f} {v['landing_mass_kg']/1e3:6.1f} {v['S_m2']:6.1f} {v['cl_max']:4.1f} {v['TW']:5.2f} {v['WS_landing']:5.0f} {v['vs_model_kt']:6.1f} {v['faa_vapp_kt'] or 0:5.0f} {ratio:7.2f} {v['vref_model_kt']:5.0f} {v['faa_engine']}")
