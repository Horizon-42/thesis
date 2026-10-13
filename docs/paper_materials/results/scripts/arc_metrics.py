"""Paper tables (2026-10-12): landing rate, arc ADE and arc FDE of single-aircraft free generation on the val days.

Each free-generation readout (`prior_free_generation`'s `sentences.npz` + `sentences.jsonl`) is compared, sentence by
sentence, with the OBSERVED track of its flight (the instruction artefact's val signals, the airport frame, MSL height).
Both paths start at the flight's first predicted step (the closed-loop sentence's ``first_row`` + its observed prefix,
the shared anchor) and run to their own last 2 s row: the generated path to the row where the judge ended the flight,
the observed path to the end of the arrival slice.

* arc ADE: `geometry.geometric_metrics.arc_aligned_ade_m` — the mean 3D distance at the same fractions 1/64 … 1 of each
  path's own horizontal arc length (the anchor not scored).
* arc FDE: the 3D distance at arc fraction 1, i.e. between the two paths' last points.

Every block is given for all sentences and for the sentences "inside" the prior's selection (`prior.selection.side`:
the flight's labelled outcome landed and its observed track without faults) — the readout's own `inside` part, the
set of stage B's reported val landing rate (94.6 %); the airport and stratum blocks are inside only.

A reference row flies the labelled closed-loop sentence itself (the artefact's flown states of the val sentences, the
executor on the labelled words) against the same observed track: the error of the labelling + executor alone.

usage (from a checkout's root, in the aeroviz env):
  python arc_metrics.py <instructions dir> <out.json> <name>=<free generation dir> ...
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "4dTrajectory"))

from ts_transformer.experiments.prior_free_generation import read_sentences  # noqa: E402
from ts_transformer.geometry.geometric_metrics import arc_aligned_ade_m  # noqa: E402
from ts_transformer.instructions.artefact import closed_loop_sentences, load_signals, load_spec  # noqa: E402
from ts_transformer.instructions.faults import faulty_flights  # noqa: E402
from ts_transformer.prior.selection import side  # noqa: E402

instructions, out_path = Path(sys.argv[1]), Path(sys.argv[2])
readouts = dict(item.split("=", 1) for item in sys.argv[3:])

SPLIT, INTERVAL_S = "val", 4.0
signals = load_signals(instructions, SPLIT)
sentences = closed_loop_sentences(instructions, SPLIT, INTERVAL_S, load_spec(instructions))
faulty = faulty_flights(instructions, SPLIT)


def side_of(index: int, selection: str) -> str:
    """The flight's side of the prior's selection, as the free-generation readout splits its sentences."""
    return side(selection, sentences[index].withheld.outcome, index in faulty)


def observed_from(index: int, first_flown: int) -> np.ndarray:
    """The observed track ``[N, 3]`` (e, n, height) from the 2 s row ``first_flown`` of the signals to their end."""
    s = signals[index]
    return np.column_stack([s.e_m, s.n_m, s.altitude_m])[first_flown:]


def pair(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    return arc_aligned_ade_m(pred, truth), float(np.linalg.norm(pred[-1] - truth[-1]))


def aggregate(rows: list[dict]) -> dict:
    def block(items):
        ade = np.array([r["arc_ade_m"] for r in items])
        fde = np.array([r["arc_fde_m"] for r in items])
        landed = sum(r["outcome"] == "landed" for r in items)
        return {"sentences": len(items), "landed": landed, "landing_rate": landed / len(items),
                "arc_ade_mean_m": float(ade.mean()), "arc_ade_median_m": float(np.median(ade)),
                "arc_fde_mean_m": float(fde.mean()), "arc_fde_median_m": float(np.median(fde))}
    inside = [r for r in rows if r["side"] == "inside"]
    out = {"all": block(rows), "inside": block(inside), "airports": {}, "strata": {}}
    for key, field in (("airports", "airport"), ("strata", "stratum")):
        for value in sorted({r[field] for r in inside}):
            out[key][value] = block([r for r in inside if r[field] == value])
    out["inside_landed_only"] = block([r for r in inside if r["outcome"] == "landed"])
    return out


result = {"split": SPLIT, "instructions": str(instructions), "truth": "observed track (signals), from the first "
          "predicted step to the end of the arrival slice", "arc_points": 64, "models": {}}
flights_read: set[int] = set()
for name, directory in readouts.items():
    config, stored = read_sentences(Path(directory))
    rows = []
    for item in stored:
        sentence = sentences[item.index].rows
        prefix = int(np.flatnonzero(sentence.on_interval)[sentence.start])  # 2 s rows before the first predicted step
        if not np.allclose(item.states[:prefix + 1, :3], sentence.states[:prefix + 1, :3]):
            raise ValueError(f"{name}: sentence {item.index}/{item.sample} does not start on its flight's observed rows")
        truth = observed_from(item.index, sentence.first_row + prefix)
        ade, fde = pair(item.states[prefix:, :3], truth)
        rows.append({"index": item.index, "sample": item.sample, "side": side_of(item.index, config["selection"]),
                     "airport": item.row["airport"],
                     "stratum": item.row["stratum"], "outcome": item.row["outcome"], "arc_ade_m": ade,
                     "arc_fde_m": fde})
        flights_read.add(item.index)
    result["models"][name] = {"directory": directory, "config_git": config["git"], **aggregate(rows)}

reference = []
for index in sorted(flights_read):
    sentence, withheld = sentences[index].rows, sentences[index].withheld
    prefix = int(np.flatnonzero(sentence.on_interval)[sentence.start])
    ade, fde = pair(sentence.flown_states[:, :3], observed_from(index, sentence.first_row + prefix))
    reference.append({"index": index, "sample": 0, "side": side_of(index, "landed"),
                      "airport": signals[index].airport, "stratum": withheld.stratum,
                      "outcome": withheld.outcome, "arc_ade_m": ade, "arc_fde_m": fde})
result["labelled_reference"] = aggregate(reference)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(result, indent=2))
print(json.dumps({name: {k: m[k] for k in ("all", "inside")} for name, m in result["models"].items()}
                 | {"reference": result["labelled_reference"]["inside"]}, indent=1))
