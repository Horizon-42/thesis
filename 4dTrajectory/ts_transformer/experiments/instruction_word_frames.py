"""How airport-specific the labelled heading and level words are, in the absolute frame and in the labelled runway's
(two-tier design `docs/two_tier/two_tier_design.md` §11.2–§11.3; R48).

One split of an instruction artefact (never the test days: the artefact holds none). Every sentence's words are counted
per airport in two frames:

- **heading words** — every heading word said (row 0's included): its absolute class (5° compass, the artefact's
  reading) and its class relative to the labelled runway's course, ``(class − round(course / step)) mod classes`` — a
  re-indexing by the course rounded to the grid, the frame of design §3.3;
- **the intercept heading** — the heading word in force at the sentence's clearance row (``join_row``, the labeller's
  turn onset onto the final; the last heading word said at or before it), in both frames; a sentence cleared at row 0 has
  none;
- **level words** — every altitude word but "descend to land": its MSL level class, and the class less the labelled
  runway's threshold elevation rounded to the level step (height above the threshold).

Each measure's per-airport distributions are compared pairwise by the Jensen–Shannon divergence in bits (0: the same
distribution, 1: no class in common); the mean and the largest over the airport pairs are reported, with every pair. The
**rare share** of an airport is the share of its words whose class the other airports, pooled, use for less than
`RARE_SHARE` of theirs.

**Grid offsets**: every candidate runway's course against the heading grid — the nearest grid value and how far the
course is from it, beside the capture corridor's course tolerance (spec ``corridor_course_tolerance_deg``): a course
farther from the grid than the tolerance cannot be held inside the corridor by an absolute heading word.

A read, not a gate: no value here passes or fails anything (design D7). Writes ``word_frames.json`` into a NEW directory.

    python run_ts.py instruction_word_frames \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v6_20261002 --split train \\
        --out 4dTrajectory/outputs/POOLED/analyses/word_frames_<date>
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry, RunwayCandidate
from ts_transformer.instructions.artefact import (
    SPLITS, load_candidates, load_sentences, load_spec, signals_flights, spec_labeller_source,
)
from ts_transformer.instructions.words import ALTITUDE, HEADING, RUNWAY, UNCHANGED, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-instruction-word-frames-v1"
#: A class the other airports "almost never use": under this share of their pooled words.
RARE_SHARE = 0.002
MEASURES = ("heading", "intercept_heading", "level")
FRAMES = ("absolute", "runway")


def js_divergence_bits(first: Mapping[int, float], second: Mapping[int, float]) -> float:
    """The Jensen–Shannon divergence, in bits, of two distributions over integer classes (counts or shares; each is
    normalised)."""
    classes = sorted(set(first) | set(second))
    p = np.array([first.get(c, 0.0) for c in classes], dtype=float)
    q = np.array([second.get(c, 0.0) for c in classes], dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        keep = a > 0
        return float((a[keep] * np.log2(a[keep] / b[keep])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def course_shift(course_deg: float, step_deg: float, classes: int) -> int:
    """The heading classes a runway frame shifts by: the course rounded to the grid."""
    return int(round(course_deg / step_deg)) % classes


def intercept_heading(heading_column: np.ndarray, join_row: int) -> int | None:
    """The heading word in force at the clearance row: the last one said at or before it (the labeller says heading
    words only before the clearance row, and row 0 says one); None for a sentence cleared at row 0."""
    if join_row == 0:
        return None
    said = heading_column[: join_row + 1]
    said = said[said != UNCHANGED]
    return int(said[-1])


def sentence_candidate(geometry: AirportGeometry, rows: np.ndarray, runway_index: int, flight: Mapping[str, Any]
                       ) -> RunwayCandidate:
    """The sentence's runway: its recorded ``runway_index``, which its row-0 runway word must say and its flight must
    have landed on."""
    if int(rows[0, RUNWAY]) != runway_index:
        raise ValueError(f"{flight['dataset_id']}: row 0 says runway slot {int(rows[0, RUNWAY])}, the sentence records "
                         f"slot {runway_index}")
    candidate = geometry.candidates[runway_index]
    if candidate.ident != flight["runway"]:
        raise ValueError(f"{flight['dataset_id']}: sentence runway {candidate.ident} is not the landed "
                         f"{flight['runway']}")
    return candidate


def count_words(sentences: Mapping[str, np.ndarray], flights: Sequence[Mapping[str, Any]],
                geometries: Mapping[str, AirportGeometry], words: Words) -> dict[str, dict[str, dict[str, Counter]]]:
    """``{measure: {frame: {airport: Counter(class → words)}}}`` over every sentence (the grids: ``words.spec``)."""
    counts: dict[str, dict[str, dict[str, Counter]]] = {m: {f: {} for f in FRAMES} for m in MEASURES}
    grid, offsets = sentences["words"], sentences["offsets"]
    for k, signal in enumerate(sentences["signal_index"]):
        flight = flights[int(signal)]
        airport = flight["airport"]
        rows = grid[offsets[k]: offsets[k + 1]]
        candidate = sentence_candidate(geometries[airport], rows, int(sentences["runway_index"][k]), flight)
        shift = course_shift(candidate.course_deg, words.spec.heading_step_deg, words.n_heading)
        level_shift = int(round(candidate.elevation_m / words.spec.altitude_step_m))
        heading = rows[:, HEADING][rows[:, HEADING] != UNCHANGED].astype(int)
        levels = rows[:, ALTITUDE][(rows[:, ALTITUDE] != UNCHANGED) & (rows[:, ALTITUDE] != words.altitude_land)]
        levels = levels.astype(int)
        intercept = intercept_heading(rows[:, HEADING], int(sentences["join_row"][k]))
        found = {
            "heading": {"absolute": heading, "runway": (heading - shift) % words.n_heading},
            "intercept_heading": ({"absolute": np.array([intercept]),
                                   "runway": np.array([(intercept - shift) % words.n_heading])}
                                  if intercept is not None else {"absolute": np.array([], dtype=int),
                                                                 "runway": np.array([], dtype=int)}),
            "level": {"absolute": levels, "runway": levels - level_shift},
        }
        for measure, frames in found.items():
            for frame, classes in frames.items():
                counts[measure][frame].setdefault(airport, Counter()).update(int(c) for c in classes)
    return counts


def compare(by_airport: Mapping[str, Counter]) -> dict[str, Any]:
    """The pairwise Jensen–Shannon divergences (mean, largest, every pair) and each airport's rare share."""
    airports = sorted(code for code, counter in by_airport.items() if sum(counter.values()))
    pairs = {f"{a}-{b}": js_divergence_bits(by_airport[a], by_airport[b])
             for a, b in itertools.combinations(airports, 2)}
    rare = {}
    for code in airports:
        others = Counter()
        for other in airports:
            if other != code:
                others.update(by_airport[other])
        total, own = sum(others.values()), sum(by_airport[code].values())
        rare[code] = (sum(n for c, n in by_airport[code].items() if others.get(c, 0) < RARE_SHARE * total) / own
                      if total else None)
    values = list(pairs.values())
    return {"words": {code: sum(by_airport[code].values()) for code in airports},
            "mean_pairwise_js_bits": float(np.mean(values)) if values else None,
            "max_pairwise_js_bits": max(values) if values else None,
            "pairwise_js_bits": pairs, "rare_share": rare}


def grid_offsets(geometries: Mapping[str, AirportGeometry], step_deg: float, tolerance_deg: float
                 ) -> dict[str, list[dict[str, Any]]]:
    """Every candidate course's distance from the nearest heading grid value, beside the corridor's course tolerance."""
    out = {}
    for code, geometry in sorted(geometries.items()):
        rows = []
        for candidate in geometry.candidates:
            nearest = step_deg * round(candidate.course_deg / step_deg)
            offset = abs(candidate.course_deg - nearest)
            rows.append({"ident": candidate.ident, "course_deg": candidate.course_deg,
                         "nearest_grid_deg": nearest % 360.0, "offset_deg": offset,
                         "within_course_tolerance": offset <= tolerance_deg})
        out[code] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact")
    parser.add_argument("--split", choices=SPLITS, default="train", help="the split read (default train)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    directory = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a read is never overwritten")
    started = time.perf_counter()
    spec = load_spec(directory)
    words = Words(spec)
    geometries = load_candidates(directory)
    sentences = load_sentences(directory, args.split, spec)
    counts = count_words(sentences, signals_flights(directory, args.split), geometries, words)
    measures = {measure: {frame: compare(counts[measure][frame]) for frame in FRAMES} for measure in MEASURES}
    for measure in MEASURES:
        absolute, runway = (measures[measure][f]["mean_pairwise_js_bits"] for f in FRAMES)
        print(f"{measure}: mean pairwise JS {absolute:.3f} bits absolute, {runway:.3f} in the runway's frame"
              if absolute is not None else f"{measure}: one airport, no pair", flush=True)
    payload = {
        "schema": SCHEMA, "written_utc": utc_now(), "instructions": repo_relative(directory), "split": args.split,
        "spec_sha256": spec.sha256, "labeller_source_sha256": spec_labeller_source(directory),
        "sentences": int(len(sentences["signal_index"])),
        "constants": {"rare_share": RARE_SHARE, "heading_step_deg": spec.heading_step_deg,
                      "altitude_step_m": spec.altitude_step_m,
                      "corridor_course_tolerance_deg": spec.corridor_course_tolerance_deg},
        "measures": measures,
        "grid_offsets": grid_offsets(geometries, spec.heading_step_deg, spec.corridor_course_tolerance_deg),
        "git": git_state(), "seconds": time.perf_counter() - started,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "word_frames.json", payload)
    print(f"wrote {out / 'word_frames.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
