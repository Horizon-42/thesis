"""Extract the real data that the tutorial shows, from the formal artefact (TRAIN split only; the val days are read
one time for each stage, outline D85, and the test days are sealed).

    conda run -n aeroviz python scripts/two_tier_tutorial/extract_real_data.py

Writes ``data/tutorial_data.json`` beside this file: two real flights (a vectored approach and a go-around) with their
signals, closed-loop sentence, flown states and errors, the geometry of their airport, and a few train-split counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "4dTrajectory"), str(ROOT / "geokit" / "src")]

from ts_transformer.instructions.artefact import (  # noqa: E402
    STATE_COLUMNS, closed_loop_sentences, load_candidates, load_signals, load_spec,
)
from ts_transformer.instructions.words import COLUMNS, RUNWAY_GO_AROUND, UNCHANGED, Words  # noqa: E402

ARTEFACT = Path("/home/supercomputing/studys/thesis/4dTrajectory/outputs/POOLED/instruction_language/v12_20261005")
SPLIT = "train"            # D85: never the val days here
DELTA_S = 4.0              # the chosen row interval
FLIGHTS = {"vectored": 10567, "go_around": 9814}      # one-off pick on the train split: KRDU, landed, vectored with many corrections / with a go-around


def r(x, n=2):
    return [round(float(v), n) for v in x]


def decode(words: Words, grid: np.ndarray, idents: list[str]) -> list[list[str]]:
    """Each row's five words as text ('' = unchanged); the heading is relative to the course of the first runway."""
    out = []
    for row in grid:
        text = []
        for column, value in enumerate(row):
            value = int(value)
            if value == UNCHANGED:
                text.append("")
            elif column == 0:
                text.append("go-around" if value == RUNWAY_GO_AROUND else idents[value])
            elif column == 1:
                text.append(f"{words.heading_relative_deg(value):+.0f}")
            elif column == 2:
                level = words.altitude_level_m(value)
                text.append("no level-off" if level is None else f"{level:.0f}")
            elif column == 3:
                text.append(["level", "descent 1", "descent 2", "descent 3", "descent 4", "climb"][value])
            else:
                speed = words.speed_mps(value)
                text.append("unspecified" if speed is None else f"{speed:.0f}")
        out.append(text)
    return out


def main() -> None:
    spec = load_spec(ARTEFACT)
    words = Words(spec)
    geometries = load_candidates(ARTEFACT)
    signals = load_signals(ARTEFACT, SPLIT)
    sentences = closed_loop_sentences(ARTEFACT, SPLIT, DELTA_S, spec)
    every = int(round(DELTA_S / spec.step_s))
    data = {"spec": {"heading_step_deg": spec.heading_step_deg, "heading_lead_s": spec.heading_lead_s,
                     "turn_rate_max_deg_s": spec.turn_rate_max_deg_s, "bank_max_deg": spec.turn_bank_max_deg,
                     "altitude_segment_steps_m": list(spec.altitude_segment_steps_m),
                     "altitude_segment_tops_m": list(spec.altitude_segment_tops_m),
                     "fit_tolerance_m": spec.altitude_fit_tolerance_m,
                     "descent_edges_deg": list(spec.descent_angle_edges_deg),
                     "descent_centres_deg": list(spec.descent_angle_centres_deg),
                     "climb_centre_deg": spec.climb_angle_centre_deg,
                     "speed_step_mps": spec.speed_step_mps, "speed_min_mps": spec.speed_min_mps,
                     "speed_max_mps": spec.speed_max_mps, "speed_accel_max_mps2": spec.speed_accel_max_mps2,
                     "closed_loop_lateral_m": spec.closed_loop_lateral_m,
                     "closed_loop_vertical_m": spec.closed_loop_vertical_m,
                     "closed_loop_final_vertical_m": spec.closed_loop_final_vertical_m,
                     "levels_m": r(words.altitude_levels, 1), "level_bands_m": r(words.altitude_tolerances, 1)},
            "flights": {}}
    for name, index in FLIGHTS.items():
        flight, sentence = signals[index], sentences[index]
        geometry = geometries[flight.airport]
        idents = [c.ident for c in geometry.candidates]
        rows, withheld = sentence.rows, sentence.withheld
        m = len(rows.grid)
        first = rows.first_row
        # the sentence's states are on the 2 s rows from its first row; the Δ rows are marked
        states = rows.states
        observed = slice(first, first + len(states))
        data["flights"][name] = {
            "airport": flight.airport, "runway": flight.runway, "typecode": flight.typecode, "index": index,
            "stratum": withheld.stratum, "outcome": withheld.outcome, "delta_s": DELTA_S, "every": every,
            "start_row": rows.start, "elevation_m": geometry.elevation_m,
            "candidates": [{"ident": c.ident, "e": round(c.threshold_e_m, 1), "n": round(c.threshold_n_m, 1),
                            "course": round(c.course_deg, 2), "elev": round(c.elevation_m, 1),
                            "length": round(c.length_m, 0), "tch": c.vertical_path.crossing_height_m,
                            "gp": c.vertical_path.glidepath_deg, "da": c.vertical_path.decision_height_m}
                           for c in geometry.candidates],
            "runway_index": withheld.runway_index, "capture_row": int(withheld.capture_row),
            "go_around_rows": [int(v) for v in withheld.go_around_rows],
            "observed": {"t": r(flight.time_s[observed] - flight.time_s[first], 1), "e": r(flight.e_m[observed], 1),
                         "n": r(flight.n_m[observed], 1), "alt": r(flight.altitude_m[observed], 1),
                         "track": r(np.mod(flight.track_deg[observed], 360), 1),
                         "gs": r(flight.ground_speed_mps[observed], 1)},
            "states": {"columns": list(STATE_COLUMNS), "rows": [r(s, 2) for s in states],
                       "on_interval": [bool(v) for v in rows.on_interval]},
            "words": {"columns": list(COLUMNS), "grid": rows.grid.astype(int).tolist(),
                      "text": decode(words, rows.grid, idents),
                      "correction": rows.correction.astype(int).tolist()},
            "lateral_m": [None if not np.isfinite(v) else round(float(v), 1) for v in withheld.lateral_m],
            "vertical_m": [None if not np.isfinite(v) else round(float(v), 1) for v in withheld.vertical_m],
            "uncorrectable": withheld.uncorrectable.astype(int).tolist(),
            "matched_row": r(withheld.matched_row, 2),
        }
        print(name, flight.airport, flight.runway, m, "rows", int(rows.correction.sum()), "correction words")
    # train-split counts at each Δ (what the artefact holds)
    counts = {}
    for delta in (2.0, 4.0, 8.0):
        sent = closed_loop_sentences(ARTEFACT, SPLIT, delta, spec)
        n = len(sent)
        total_rows = sum(len(s.rows.grid) for s in sent.values())
        said = np.zeros(5)
        corrections = np.zeros(5)
        for s in sent.values():
            said += (s.rows.grid != UNCHANGED).sum(axis=0)
            corrections += s.rows.correction.sum(axis=0)
        counts[str(int(delta))] = {"sentences": n, "rows": int(total_rows),
                                   "rows_per_sentence": round(total_rows / n, 1),
                                   "words": said.astype(int).tolist(), "corrections": corrections.astype(int).tolist()}
        print(delta, counts[str(int(delta))])
    data["train_counts"] = counts
    # every candidate of the five airports, with the FAF's distance before its threshold (the CIFP document)
    from flight_scenarios.procedure_final import final_approach_fix  # noqa: E402
    procedures = Path("/home/supercomputing/studys/thesis/aeroviz-4d/public/data/airports")
    airports = {}
    for code, geometry in sorted(geometries.items()):
        candidates = []
        for c in geometry.candidates:
            faf = final_approach_fix(code, c.ident, root=procedures).distance_to_threshold_m
            candidates.append({"ident": c.ident, "e": round(c.threshold_e_m, 1), "n": round(c.threshold_n_m, 1),
                               "course": round(c.course_deg, 2), "elev": round(c.elevation_m, 1),
                               "length": round(c.length_m, 0), "tch": c.vertical_path.crossing_height_m,
                               "gp": c.vertical_path.glidepath_deg, "da": round(c.vertical_path.decision_height_m, 1),
                               "faf_m": round(faf, 0)})
        airports[code] = {"elevation_m": round(geometry.elevation_m, 1), "candidates": candidates}
    data["airports"] = airports
    out = HERE / "data" / "tutorial_data.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print("wrote", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
