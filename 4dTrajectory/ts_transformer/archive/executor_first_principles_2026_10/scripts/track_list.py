"""Which track files `replay.draw_flights` will rebuild for the first N chunks of the train permutation."""
import json, sys
from pathlib import Path
import numpy as np
import archive_paths   # noqa: E402  (README §2)
REPO = archive_paths.REPO
sys.path[:0] = [str(REPO / "4dTrajectory"), str(REPO)]
from ts_transformer.instructions.artefact import load_signals
IL = REPO / "4dTrajectory/outputs/POOLED/instruction_language/v12_20261005"
CHUNKS, SEED, CHUNK = int(sys.argv[1]), 1337, 200
signals = load_signals(IL, "train")
with np.load(IL / "sentences_train.npz") as s:
    stored = sorted(int(i) for i in s["signal_index"])
order = [int(i) for i in np.random.default_rng(SEED).permutation(stored)]
picked = order[: CHUNKS * CHUNK]
by_airport = {}
for i in picked:
    by_airport.setdefault(signals[i].airport, []).append(signals[i].dataset_id.split(":", 1)[1])
print({a: len(v) for a, v in by_airport.items()}, file=sys.stderr)
print(signals[picked[0]].dataset_id, file=sys.stderr)
for a, ids in by_airport.items():
    m = json.load(open(REPO / f"trajectory_data_process/outputs/harvest/{a}/arrivals/manifest.json"))
    key = next(k for k, v in m.items() if isinstance(v, list) and v and isinstance(v[0], dict) and "source_file" in v[0])
    files = {r["flight_key"]: r["source_file"] for r in m[key]}
    # dataset_id of a harvest flight is its flight_key (flights.py: `keys = {item.dataset_id ...}` -> include_flight_keys)
    missing = [d for d in ids if d not in files]
    assert not missing, (a, missing[:3])
    for d in ids:
        print(f"{a}/tracks/{files[d]}")
