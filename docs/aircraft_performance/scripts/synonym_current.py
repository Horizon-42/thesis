"""What each OpenAP 2.4 synonym type actually flies today -> WORK/synonym_current.json."""
import csv
import json

from _paths import WORK, use_repo_code

use_repo_code()
from aircraft.aero_params import aero_params_for_aircraft, stall_speed_ms  # noqa: E402
from flight_scenarios.scenario import aircraft_for_code  # noqa: E402

MS_TO_KT = 1 / 0.514444
out = {}
for row in csv.DictReader(open(WORK / "missing_types.csv")):
    if row["status"] != "openap_2_4_synonym":
        continue
    t = row["typecode"]
    a = aircraft_for_code(t)
    ae = aero_params_for_aircraft(a)
    out[t] = dict(code=a.code, m=a.landing_mass, S=ae.S, cl=ae.Cl_max, T=a.engine.max_thrust_total_n,
                  mtow=a.mass.max_takeoff_kg,
                  vs_kt=stall_speed_ms(a.landing_mass, wing_area_m2=ae.S, cl_max=ae.Cl_max) * MS_TO_KT)
(WORK / "synonym_current.json").write_text(json.dumps(out))
print(sorted(out))
