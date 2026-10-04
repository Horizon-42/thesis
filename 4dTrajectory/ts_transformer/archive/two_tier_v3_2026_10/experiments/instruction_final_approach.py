"""Where the labelled sentences put the clearance and "descend to land", and how close the observed flights fly to the
published glidepath after the capture row (two-tier design `docs/two_tier/two_tier_design.md` §11.4; R49).

One split of an instruction artefact (never the test days: the artefact holds none); every labelled sentence, read with
its signals' first ``len(words)`` rows and its landed runway (the sentence's row-0 runway word):

- **the clearance** — the sentence's ``join_row`` (the labeller's clearance: the turn onset onto the final): the share of
  all sentence rows before it, and the share of sentences cleared at row 0;
- **"descend to land"** — the first row its altitude column says it: the share of sentences that say it before the
  clearance row (and how long before, in seconds) and before the capture row (``capture_row``, the first row of the
  final run in the capture corridor), and its distance before the threshold along the course;
- **the published glidepath** — on the rows from the capture row to the sentence's end that are more than
  `FLARE_EXCLUDED_M` before the threshold, the observed height above the threshold less the glidepath's, against two
  references of the runway's published vertical path (`autopilot.runway_data`: TCH and glidepath angle):

  - ``flat``: ``TCH + d · tan(angle)`` — the glidepath the executor and the procedure masks use
    (`autopilot/vertical.py`, `prior/procedure.py`), which leaves the earth's curvature out;
  - ``straight_line``: the published path as a straight line in space from the TCH point — over the curved earth its
    height above the surface grows by ``d² / (2 R)`` more (the leading term), R the WGS84 radius of curvature along the
    runway's course at the airport's latitude (`curvature_radius_m`): about 15 m at 14 km, 31 m at 20 km.

  A sentence is read only with at least `MIN_ROWS` such rows. Reported for each reference: the rows' quantiles and their
  shares within each of `BANDS_M`; the flights' median deviation and the share of flights with at least `FLIGHT_SHARE`
  of their rows within the widest band; at the capture row, the deviation and the shares above / below the widest band;
  and the capture row's distance to the threshold.

Positions relative to the runway are the labeller's own (`instructions.airport.relative_to_runway`). The bands are
readout bands, not criteria: no value here passes or fails anything (design D7). Pooled and per airport. Writes
``final_approach.json`` into a NEW directory.

    python run_ts.py instruction_final_approach \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v6_20261002 --split train \\
        --out 4dTrajectory/outputs/POOLED/analyses/final_approach_<date>
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from geokit import wgs84_curvature_radii

from ts_transformer.autopilot.runway_data import VerticalPath, published_vertical_paths
from ts_transformer.instructions.airport import AirportGeometry, RunwayCandidate, relative_to_runway
from ts_transformer.instructions.artefact import (
    SPLITS, load_candidates, load_sentences, load_signals, load_spec, spec_labeller_source,
)
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import ALTITUDE, RUNWAY, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-instruction-final-approach-v1"
#: Rows closer to the threshold than this are left out of the glidepath reading (the flare).
FLARE_EXCLUDED_M = 300.0
#: A sentence is read on the glidepath only with at least this many rows left.
MIN_ROWS = 5
#: Readout bands of |height − glidepath| (descriptive, design D7).
BANDS_M = (30.0, 60.0)
#: A flight within the widest band on at least this share of its rows is counted as one that stays on the glidepath.
FLIGHT_SHARE = 0.9
QUANTILES = (10, 50, 90)
REFERENCES = ("flat", "straight_line")


@dataclass(frozen=True)
class SentenceReading:
    airport: str
    rows: int
    join_row: int
    land_row: int | None              # the first "descend to land" row, None when the sentence never says it
    land_before_threshold_m: float | None
    capture_row: int
    #: Height above each glidepath reference (`REFERENCES`) on the rows read (after the capture row, outside the flare);
    #: empty arrays when fewer than `MIN_ROWS`.
    deviation_m: Mapping[str, np.ndarray]
    capture_deviation_m: Mapping[str, float] | None     # at the capture row, for a sentence read on the glidepath
    capture_before_threshold_m: float | None


def curvature_radius_m(lat_deg: float, course_deg: float) -> float:
    """The earth's radius of curvature along a compass course at a latitude: Euler's formula on the WGS84 meridional
    and prime-vertical radii."""
    meridional, prime_vertical = wgs84_curvature_radii(lat_deg)
    course = math.radians(course_deg)
    return 1.0 / (math.cos(course) ** 2 / meridional + math.sin(course) ** 2 / prime_vertical)


def glidepath_heights_m(before_threshold_m: np.ndarray, path: VerticalPath, radius_m: float) -> dict[str, np.ndarray]:
    """Each reference's glidepath height above the threshold at each distance before it (module docstring)."""
    flat = path.crossing_height_m + before_threshold_m * math.tan(math.radians(path.glidepath_deg))
    return {"flat": flat, "straight_line": flat + before_threshold_m ** 2 / (2.0 * radius_m)}


def read_sentence(signals: FlightSignals, rows: np.ndarray, candidate: RunwayCandidate, path: VerticalPath,
                  radius_m: float, join_row: int, capture_row: int, land_class: int) -> SentenceReading:
    """One sentence (``rows``: its words) against its signals' first ``len(rows)`` rows and its landed runway."""
    n = len(rows)
    place = relative_to_runway(signals.e_m[:n], signals.n_m[:n], signals.track_deg[:n], signals.altitude_m[:n],
                               candidate)
    said = np.nonzero(rows[:, ALTITUDE] == land_class)[0]
    land_row = int(said[0]) if len(said) else None
    deviation = {reference: place.height_above_threshold_m - height
                 for reference, height in glidepath_heights_m(place.before_threshold_m, path, radius_m).items()}
    kept = (np.arange(n) >= capture_row) & (place.before_threshold_m > FLARE_EXCLUDED_M)
    read = int(kept.sum()) >= MIN_ROWS
    return SentenceReading(
        airport=signals.airport, rows=n, join_row=join_row, land_row=land_row,
        land_before_threshold_m=float(place.before_threshold_m[land_row]) if land_row is not None else None,
        capture_row=capture_row,
        deviation_m={reference: values[kept] if read else np.empty(0) for reference, values in deviation.items()},
        capture_deviation_m={reference: float(values[capture_row]) for reference, values in deviation.items()}
        if read else None,
        capture_before_threshold_m=float(place.before_threshold_m[capture_row]) if read else None)


def quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=float)
    return {f"p{q}": float(np.percentile(values, q)) for q in QUANTILES} if len(values) else None


def _glidepath(flights: Sequence[SentenceReading], reference: str) -> dict[str, Any]:
    """One reference's glidepath reading over the flights read on it."""
    rows = np.concatenate([r.deviation_m[reference] for r in flights]) if flights else np.empty(0)
    capture = np.array([r.capture_deviation_m[reference] for r in flights])
    widest = max(BANDS_M)
    return {
        "deviation_m": quantiles(rows),
        "rows_within_share": {f"{band:g}": float(np.mean(np.abs(rows) <= band)) if len(rows) else None
                              for band in BANDS_M},
        "flight_median_deviation_m": quantiles([float(np.median(r.deviation_m[reference])) for r in flights]),
        "flights_on_glidepath_share": (float(np.mean([np.mean(np.abs(r.deviation_m[reference]) <= widest) >= FLIGHT_SHARE
                                                      for r in flights])) if flights else None),
        "at_capture": {
            "deviation_m": quantiles(capture),
            "above_widest_band_share": float(np.mean(capture > widest)) if len(capture) else None,
            "below_widest_band_share": float(np.mean(capture < -widest)) if len(capture) else None,
        },
    }


def summarise(readings: Sequence[SentenceReading], step_s: float) -> dict[str, Any]:
    """The clearance, "descend to land" and glidepath readings of ``readings`` (module docstring)."""
    land = [r for r in readings if r.land_row is not None]
    early = [r for r in land if r.land_row < r.join_row]
    on_path = [r for r in readings if r.capture_deviation_m is not None]
    return {
        "sentences": len(readings),
        "clearance": {
            "rows_before_share": sum(r.join_row for r in readings) / sum(r.rows for r in readings),
            "cleared_at_row_0_share": float(np.mean([r.join_row == 0 for r in readings])),
        },
        "descend_to_land": {
            "sentences": len(land),
            "before_clearance_share": len(early) / len(land) if land else None,
            "lead_on_clearance_s": quantiles([(r.join_row - r.land_row) * step_s for r in early]),
            "before_capture_share": float(np.mean([r.land_row < r.capture_row for r in land])) if land else None,
            "before_threshold_m": quantiles([r.land_before_threshold_m for r in land]),
        },
        "glidepath": {
            "flights": len(on_path), "rows": int(sum(len(r.deviation_m["flat"]) for r in on_path)),
            "capture_before_threshold_m": quantiles([r.capture_before_threshold_m for r in on_path]),
            **{reference: _glidepath(on_path, reference) for reference in REFERENCES},
        },
    }


def sentence_candidate(geometry: AirportGeometry, rows: np.ndarray, runway_index: int, flight: FlightSignals
                       ) -> RunwayCandidate:
    """The sentence's runway: its recorded ``runway_index``, which its row-0 runway word must say and its flight must
    have landed on."""
    if int(rows[0, RUNWAY]) != runway_index:
        raise ValueError(f"{flight.dataset_id}: row 0 says runway slot {int(rows[0, RUNWAY])}, the sentence records "
                         f"slot {runway_index}")
    candidate = geometry.candidates[runway_index]
    if candidate.ident != flight.runway:
        raise ValueError(f"{flight.dataset_id}: sentence runway {candidate.ident} is not the landed {flight.runway}")
    return candidate


def read_split(sentences: Mapping[str, np.ndarray], signals: Sequence[FlightSignals],
               geometries: Mapping[str, AirportGeometry], paths: Mapping[str, Sequence[VerticalPath]],
               words: Words) -> list[SentenceReading]:
    """Every sentence of a split."""
    readings = []
    grid, offsets = sentences["words"], sentences["offsets"]
    for k, signal in enumerate(sentences["signal_index"]):
        flight = signals[int(signal)]
        geometry, index = geometries[flight.airport], int(sentences["runway_index"][k])
        rows = grid[offsets[k]: offsets[k + 1]]
        candidate = sentence_candidate(geometry, rows, index, flight)
        readings.append(read_sentence(flight, rows, candidate, paths[flight.airport][index],
                                      curvature_radius_m(geometry.frame.lat0, candidate.course_deg),
                                      int(sentences["join_row"][k]), int(sentences["capture_row"][k]),
                                      words.altitude_land))
    return readings


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
    geometries = load_candidates(directory)
    paths = {code: published_vertical_paths(geometry) for code, geometry in geometries.items()}
    readings = read_split(load_sentences(directory, args.split, spec), load_signals(directory, args.split), geometries,
                          paths, Words(spec))
    by_airport = defaultdict(list)
    for reading in readings:
        by_airport[reading.airport].append(reading)
    pooled = summarise(readings, spec.step_s)
    early = pooled["descend_to_land"]["before_clearance_share"]

    def median(reference: str) -> str:
        deviation = pooled["glidepath"][reference]["deviation_m"]
        return "n/a" if deviation is None else f"{deviation['p50']:+.0f} m"

    print(f"{len(readings)} sentences: rows before the clearance {pooled['clearance']['rows_before_share']:.1%}; "
          f"\"descend to land\" before it {'n/a' if early is None else f'{early:.1%}'}; after the capture row the median "
          f"height over the glidepath {median('flat')} (flat), {median('straight_line')} (straight line)", flush=True)
    payload = {
        "schema": SCHEMA, "written_utc": utc_now(), "instructions": repo_relative(directory), "split": args.split,
        "spec_sha256": spec.sha256, "labeller_source_sha256": spec_labeller_source(directory),
        "constants": {"flare_excluded_m": FLARE_EXCLUDED_M, "min_rows": MIN_ROWS, "bands_m": list(BANDS_M),
                      "flight_share": FLIGHT_SHARE, "step_s": spec.step_s,
                      "references": {"flat": "TCH + d·tan(angle): the executor's and the procedure masks' glidepath, "
                                             "no earth curvature",
                                     "straight_line": "flat + d²/(2R): the published straight path over the curved "
                                                      "earth, R along the course (WGS84)"}},
        "published_vertical_paths": {code: {c.ident: {"crossing_height_m": p.crossing_height_m,
                                                      "glidepath_deg": p.glidepath_deg,
                                                      "curvature_radius_m": curvature_radius_m(geometries[code].frame.lat0,
                                                                                               c.course_deg)}
                                            for c, p in zip(geometries[code].candidates, paths[code])}
                                     for code in sorted(geometries)},
        "pooled": pooled, "airports": {code: summarise(by_airport[code], spec.step_s) for code in sorted(by_airport)},
        "git": git_state(), "seconds": time.perf_counter() - started,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "final_approach.json", payload)
    print(f"wrote {out / 'final_approach.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
