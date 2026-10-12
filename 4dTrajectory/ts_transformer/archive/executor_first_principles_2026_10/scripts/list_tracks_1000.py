#!/usr/bin/env python
"""The track files a draw of --per-airport train flights needs (the seeded order of `replay.draw_flights` over the stored
sentences' flights, with --margin extra per airport for the flights the group test excludes), less the ones already
here: one path per line, relative to the remote harvest root, for `rsync --files-from`."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import new_design_eval as nde   # noqa: E402  (the worktree's packages)

import numpy as np   # noqa: E402

from ts_transformer.autopilot import replay   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-airport", type=int, required=True)
ap.add_argument("--margin", type=int, required=True)
args = ap.parse_args()
with np.load(nde.INSTRUCTIONS / "sentences_train.npz") as data:
    labelled = [int(i) for i in data["signal_index"]]
signals = replay.load_signals(nde.INSTRUCTIONS, "train")
order = [int(i) for i in np.random.default_rng(1337).permutation(sorted(labelled))]
wanted, out = {}, []
for i in order:
    s = signals[i]
    if wanted.get(s.airport, 0) >= args.per_airport + args.margin:
        continue
    wanted[s.airport] = wanted.get(s.airport, 0) + 1
    key = s.dataset_id.split(":", 1)[1]
    runway = key.split("_")[1]
    path = f"{s.airport}/tracks/assigned/{runway}/{key}.json"
    if not (nde.HARVEST / path).exists():
        out.append(path)
print("\n".join(out))
print(f"{len(out)} missing of {sum(wanted.values())} ({wanted}); example id {signals[order[0]].dataset_id}", file=sys.stderr)
