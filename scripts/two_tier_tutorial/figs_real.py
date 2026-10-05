"""Real labeller output for the figures: the open-loop heading words of one train flight, on the 2 s rows."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "4dTrajectory"), str(ROOT / "geokit" / "src")]

from ts_transformer.instructions.artefact import closed_loop_sentences, load_sentences, load_spec  # noqa: E402
from ts_transformer.instructions.words import HEADING, UNCHANGED, Words  # noqa: E402

ARTEFACT = Path("/home/supercomputing/studys/thesis/4dTrajectory/outputs/POOLED/instruction_language/v12_20261005")
FLIGHT = 10567          # the vectored KRDU 23R flight of the other figures (train split)


def heading_words_2s():
    """``(time since the first row of the closed-loop sentence in s, relative heading in deg)`` of each heading word of
    the labeller's open-loop reading at 2 s, from the capture of the sentence's first row to its last row."""
    spec = load_spec(ARTEFACT)
    words = Words(spec)
    data = load_sentences(ARTEFACT, "train", spec, ("signal_index", "instruction_flight", "instruction_column",
                                                    "instruction_value", "instruction_row"))
    n = int(np.flatnonzero(data["signal_index"] == FLIGHT)[0])
    pick = (data["instruction_flight"] == n) & (data["instruction_column"] == HEADING)
    rows, values = data["instruction_row"][pick], data["instruction_value"][pick]
    first = closed_loop_sentences(ARTEFACT, "train", 4.0, spec)[FLIGHT].rows.first_row
    keep = rows >= first
    t = (rows[keep] - first) * spec.step_s
    rel = np.array([words.heading_relative_deg(int(v)) for v in values[keep]])
    # the word in force at the first row is the last word said before it
    before = np.flatnonzero(~keep)
    if len(before) and (not keep.any() or rows[keep][0] > first):
        t = np.insert(t, 0, 0.0)
        rel = np.insert(rel, 0, words.heading_relative_deg(int(values[before[-1]])))
    return t, rel


if __name__ == "__main__":
    t, rel = heading_words_2s()
    print(len(t), t[:6], rel[:6])
