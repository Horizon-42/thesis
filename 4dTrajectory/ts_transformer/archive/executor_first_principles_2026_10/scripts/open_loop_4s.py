"""Fly the 4 s OPEN-LOOP observed words (no corrections) from the first predicted step, official executor, point-mass plant."""
import sys, json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent
import archive_paths   # noqa: E402  (README §2)
RESULTS = archive_paths.RESULTS
sys.path.insert(0, str(HERE))
import compare_plants as cp
from ts_transformer.autopilot import closed_loop, replay, flights as flights_module
from ts_transformer.autopilot.start import start_row
from ts_transformer.experiments import executor_replay as er
from ts_transformer.instructions.labeller.interval import in_force
from ts_transformer.instructions.words import RUNWAY, RUNWAY_GO_AROUND
from ts_transformer.repo_layout import arrival_manifest_path
flights_module.arrival_manifest_path = lambda a: arrival_manifest_path(a, cp.HARVEST)
params, record, words = replay.open_spec(cp.EXECUTOR, cp.INSTRUCTIONS)
out = Path(sys.argv[1]); interval = float(sys.argv[2])
batch = replay.draw(cp.INSTRUCTIONS, "train", words.spec, words, per_airport=150, seed=1337,
                    groups=(replay.OWN, replay.STAND_IN), row_interval_s=interval)
start = start_row(interval)
pseudo = {}
for j, index in enumerate(batch.indices):
    grid = batch.sentences[j].grid
    if len(grid) - start < 2: continue
    pred = grid[start:].copy(); pred[0] = in_force(grid)[start]
    if pred[0, RUNWAY] == RUNWAY_GO_AROUND: continue
    pseudo[index] = SimpleNamespace(rows=SimpleNamespace(first_row=batch.sentences[j].first_row, grid=pred))
b, _ = closed_loop.replay_batch(batch, pseudo, words)
along = er.along_columns(words)
by_airport = defaultdict(list)
for j, f in enumerate(b.signals): by_airport[f.airport].append(j)
rows, reports = [], []
for airport, members in sorted(by_airport.items()):
    r, g = er.fly_airport(b, members, params, words, chunk=250, device=torch.device("cpu"), records=out / "records" / airport,
                          split="train", checkpoint="open-loop", extra_summary={"arm": "open-loop"},
                          per_flight=lambda part, flown, verdicts: [dict(x) for x in along(part, flown, verdicts)])
    rows += r; reports += g["trajectories"]
o = Counter(r["outcome"] for r in rows)
print(f"open-loop observed words at {interval:g} s, official executor: flights {len(rows)}, landed {o['landed']}, "
      f"eval pass {sum(t['verdict']=='pass' for t in reports)}, outcomes {dict(o)}")
