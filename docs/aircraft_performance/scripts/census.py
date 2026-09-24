"""Aircraft-dynamics census of the ts training data (train + val splits, test sealed).

Reads manifest metadata only (icao24); no trajectory file is opened.
Mirrors flight_scenarios.build._resolve_aircraft under aircraft_provider="auto".
"""
import collections
import json

from _paths import HARVEST, WORK, use_repo_code

use_repo_code()

from ts_transformer.data.splits import _split_fraction  # noqa: E402
from aircraft.identity import get_default_identity_resolver  # noqa: E402
from aircraft.aircraft_sets import AIRCRAFT_PRESETS  # noqa: E402
from aircraft.query_aircraft_parameters import (  # noqa: E402
    openap_support_kind, openap_performance_metadata,
)

# The ts split: seed 1337, val 0.15, test 0.15 (TSConfig defaults; split_seed documented in open-items)
SEED, TEST, VAL = 1337, 0.15, 0.15
AIRPORTS = ["KMSY", "KRDU", "KSJC", "KSMF", "KSTL"]


def split_of(dataset_id):
    f = _split_fraction(dataset_id, SEED)
    return "test" if f < TEST else "val" if f < TEST + VAL else "train"


resolver = get_default_identity_resolver()

rows = []
for ap in AIRPORTS:
    man = json.loads((HARVEST / ap / "arrivals/manifest.json").read_text())
    elig = set(json.loads((HARVEST / ap / "arrivals/lateral_pass_eligibility.json").read_text())["eligible_flight_keys"])
    for rec in man["records"]:
        if rec["flight_key"] not in elig:
            continue
        split = split_of(f"{ap}:{rec['flight_key']}")
        if split == "test":
            rows.append({"airport": ap, "split": "test"})
            continue
        ident = resolver.resolve(declared_type="UNK", icao24=rec["icao24"])
        tc = ident.typecode
        if tc is None:
            outcome, perf = "unresolved->A320", None
        elif tc in AIRCRAFT_PRESETS:
            outcome, perf = "preset", tc
        else:
            kind = openap_support_kind(tc)
            if kind == "direct":
                outcome, perf = "openap_direct", tc
            elif kind == "synonym":
                outcome, perf = "openap_synonym", openap_performance_metadata(tc)["performance_typecode"]
            else:
                outcome, perf = "no_dynamics->A320", None
        rows.append({
            "airport": ap, "split": split, "icao24": rec["icao24"], "typecode": tc,
            "outcome": outcome, "perf": perf, "failure": ident.failure_reason,
            "method": ident.typecode_method, "source": ident.typecode_source,
        })

out = WORK / "census_rows.json"
out.write_text(json.dumps(rows))
c = collections.Counter((r["split"], r.get("outcome")) for r in rows)
for k in sorted(c, key=str):
    print(k, c[k])
