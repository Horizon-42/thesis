#!/usr/bin/env python
"""Where the last angle word of each flight is READ (its piece's start, angle lead 0): its class and its distance to the
threshold, with the observed track's end distance; crossed with the vertical verdicts of `lead_eval_1000_combo`
(leads 4/4/4). Read only; writes --out."""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
import archive_paths   # noqa: E402  (README §2)
sys.path.insert(0, str(HERE))
import new_design_eval as nde   # noqa: E402

import numpy as np   # noqa: E402

from ts_transformer.autopilot import replay                                  # noqa: E402
from ts_transformer.autopilot import flights as flights_module               # noqa: E402
from ts_transformer.instructions.labeller.read import read_flight            # noqa: E402
from ts_transformer.instructions.labeller.records import Refused             # noqa: E402
from ts_transformer.instructions.words import ALTITUDE, ANGLE, Words         # noqa: E402
from ts_transformer.repo_layout import arrival_manifest_path                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
flights_module.arrival_manifest_path = lambda airport: arrival_manifest_path(airport, nde.HARVEST)
with np.load(nde.INSTRUCTIONS / "sentences_train.npz") as data:
    labelled = [int(i) for i in data["signal_index"]]
drawn = replay.draw_flights(nde.INSTRUCTIONS, "train", labelled, per_airport=200, seed=1337,
                            groups=(replay.OWN, replay.STAND_IN))
spec = nde.spec_of(4.0, 0.0, 0.0)
words = Words(spec)
combo = {x["row_interval_s"]: x["verdicts"] for x in json.loads((archive_paths.RESULTS / "lead_eval_1000_combo/results.json").read_text())
         if x["arm"] == "combo_a4_s4"}
rows = []
for flight in drawn.signals:
    geometry = drawn.geometries[flight.airport]
    c = geometry.candidates[geometry.candidate_index(flight.runway)]
    dist = np.hypot(flight.e_m - c.threshold_e_m, flight.n_m - c.threshold_n_m)
    try:
        reading = read_flight(flight, geometry, spec, words)
    except Refused:
        continue
    angles = sorted((i for i in reading.instructions if i.column == ANGLE), key=lambda i: i.row)
    last = angles[-1]
    key = flight.dataset_id.split(":", 1)[1]
    rows.append({"flight_key": key, "last_class": int(last.value), "last_deg": words.angle_deg(int(last.value)),
                 "last_read_m": float(dist[last.row]), "end_m": float(dist[-1]),
                 "seconds_before_end": float(flight.time_s[-1] - flight.time_s[last.row]),
                 "vertical_fail": {f"{k:g}": "vertical" in v[key]["violations"] for k, v in combo.items() if key in v}})
args.out.mkdir(parents=True, exist_ok=False)
(args.out / "rows.json").write_text(json.dumps(rows))
shallow = lambda r: r["last_deg"] < 2.75
print(f"flights {len(rows)}; last angle word class {Counter(r['last_class'] for r in rows)}")
print(f"last word shallower than 2.75 deg: {sum(map(shallow, rows))}")
for X in (300, 500, 800, 1000, 1500):
    within = [r for r in rows if r["last_read_m"] < X]
    print(f"  last word read within {X} m of the threshold: {len(within)} (shallow {sum(map(shallow, within))})")
for d in ("2", "4"):
    fails = [r for r in rows if r["vertical_fail"].get(d)]
    print(f"Δ {d}: vertical fails {len(fails)}; their last word shallow {sum(map(shallow, fails))}; read distance p10/50/90 "
          f"{np.percentile([r['last_read_m'] for r in fails], [10, 50, 90]).round(0)}; "
          f"shallow & within 500/1000 m: {sum(shallow(r) and r['last_read_m'] < 500 for r in fails)}/"
          f"{sum(shallow(r) and r['last_read_m'] < 1000 for r in fails)}")
    ok = [r for r in rows if r["vertical_fail"].get(d) is False]
    print(f"   passes: shallow last word {sum(map(shallow, ok))}/{len(ok)}")
