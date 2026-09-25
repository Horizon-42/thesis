"""Instruction sentences for the frontend's Training module: a seeded sample of VAL flights per
airport, each with its words and every word's envelope drawn as geometry, from a frozen instruction
artefact (design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`; the words:
`docs/2026-09-23_instruction_vocabulary_design.zh.md`).

    python run_ts.py instruction_training_export \\
        --dir 4dTrajectory/outputs/POOLED/instruction_language/<an instruction-v3 artefact> \\
        --airports-root aeroviz-4d/public/data/airports \\
        --airport KMSY --airport KRDU --airport KSJC --airport KSMF --airport KSTL

Per airport, writes ``<airports-root>/<ICAO>/training/<set-id>/sample.json`` (refused if the
directory exists) and adds the set to ``<airports-root>/<ICAO>/training/index.json`` (refused if
the index already lists the id; every other set in it is kept as it is).

**The sample.** From the airport's labelled VAL flights (the test split is never opened), in a
permutation seeded by ``--seed``, each flight is read until ``--per-stratum`` straight-in and as
many vectored flights are found; the stratum is `readout.flight_record`'s. The pool, how many were
read and the rule are written into the file and the index, never left implicit.

**Every read flight is RE-READ with `read_flight` and must give the stored sentence** — its words
grid, runway, capture, clearance and "unspecified" rows — or the export stops, naming the flight.
The artefact's labeller hash is NOT required to be this code's (`require_current_labeller` is not
called): the frozen artefact stays exportable after unrelated code changes, and the re-read is what
says the sentence shown is the sentence stored.

**All envelope geometry comes from `instructions.display`**, which builds it from `envelope.py` and
`labeller/*`; the frontend draws numbers and computes none. This module only adds the geodesy: the
airport frame's metres → latitude / longitude (`AirportENUFrame.latlon_from_horizontal`), and MSL →
the ellipsoid height Cesium draws in (h = H + N, `flight_scenarios.datum.geoid_undulation_m`) for
the track and the altitude tubes that ride it. Units are SI throughout.

**Overlays.** What another model makes of a set's own flights — the executor's replay
(`executor_training_export`), the prior's predictions (`prior_training_export`) — is written beside
the set, never into it. The files, their schemas and the checks every writer and reader shares live
in `instructions.training_files` (not a runner: the backend imports it too).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from flight_scenarios.datum import geoid_undulation_m
from ts_transformer.instructions import display
from ts_transformer.instructions.airport import AirportGeometry, landing_cross_limit_m
from ts_transformer.instructions.artefact import (
    load_candidates, load_sentences, load_signals, load_spec, spec_labeller_source,
)
from ts_transformer.instructions.labeller.read import Admitted, Reading, admit, read_flight
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.readout import STRATA, VECTORED_TURN_DEG, flight_record
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import READING_RULE, VocabularySpec
from ts_transformer.instructions.training_files import (
    KIND_READBACK, SAMPLE_FILE, SAMPLE_SCHEMA, SPLIT, WORD_KINDS, band_payload, read_index, require_index_unchanged,
    require_stored_sentence, rounded, serialise, stored_sentence, words_in_force, write_set,
)
from ts_transformer.instructions.words import (
    ANGLE_LEVEL, APPROACH_CLEARED, APPROACH_GO_AROUND, APPROACH_NOT_CLEARED, COLUMNS, UNCHANGED, Words,
)
from ts_transformer.io_utils import utc_now
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

RUNNER = "ts_transformer.experiments.instruction_training_export"

#: The approach column's classes by name, in their class order.
APPROACH_NAMES = {APPROACH_NOT_CLEARED: "not cleared", APPROACH_CLEARED: "cleared", APPROACH_GO_AROUND: "go-around"}

#: The extended centrelines are drawn to the farthest any exported flight was from its own
#: threshold along its course, rounded up to this. A drawing extent, not an envelope value.
CENTRELINE_ROUND_M = 1000.0


# ---- numbers at display precision
def _pair(values: tuple[float, float] | None, digits: int) -> list[float] | None:
    return None if values is None else rounded(values, digits)


def _flags(values: np.ndarray) -> list[int]:
    return [int(bool(value)) for value in values]


def candidates_sha256(geometry: AirportGeometry) -> str:
    """The runway pointer's classes ARE the airport's candidates (vocabulary design §4.1), and the
    landing rule reads every runway end of the airport (§2.2), so the identity of both is the sha of
    the airport geometry — candidates and runway ends — the index's ``runwaySha256``."""
    canonical = json.dumps(geometry.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Globe:
    """The airport frame on the globe: metres → latitude / longitude, and MSL → the ellipsoid."""

    def __init__(self, geometry: AirportGeometry) -> None:
        self.frame = geometry.frame

    def latlon(self, e_m: np.ndarray, n_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lat, lon = self.frame.latlon_from_horizontal(np.asarray(e_m, dtype=np.float64), np.asarray(n_m, dtype=np.float64))
        return np.asarray(lat), np.asarray(lon)

    def line(self, line: display.Line) -> dict[str, list[float]]:
        lat, lon = self.latlon(line.e_m, line.n_m)
        return {"eM": rounded(line.e_m, 1), "nM": rounded(line.n_m, 1), "lon": rounded(lon, 7), "lat": rounded(lat, 7)}

    @staticmethod
    def undulation_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """N = h_HAE − H_MSL per point (EGM96)."""
        return geoid_undulation_m(lat, lon)


# ---- the draw
def reread(flight: FlightSignals, stored: dict[str, Any], geometry: AirportGeometry, spec: VocabularySpec,
           words: Words) -> Reading:
    """The flight read again, which must give the stored sentence exactly (`require_stored_sentence`)."""
    try:
        reading = read_flight(flight, geometry, spec, words)
    except Refused as refusal:
        raise ValueError(f"{flight.dataset_id}: the artefact holds its sentence, but this code refuses it "
                         f"({refusal}) — this labeller no longer reads like the artefact's") from None
    require_stored_sentence(flight.dataset_id, reading, stored)
    return reading


def draw(airport: str, flights: list[FlightSignals], sentences: dict[str, np.ndarray], geometry: AirportGeometry,
         spec: VocabularySpec, words: Words, per_stratum: int, seed: int
         ) -> tuple[list[tuple[str, FlightSignals, Reading]], dict[str, int]]:
    """The seeded permutation of the airport's labelled flights, read in order until both strata
    hold ``per_stratum``; every flight read is checked against its stored sentence."""
    pool = [k for k, index in enumerate(sentences["signal_index"]) if flights[int(index)].airport == airport]
    if not pool:
        raise ValueError(f"no labelled {SPLIT} flight lands at {airport}")
    wanted = {stratum: per_stratum for stratum in STRATA}
    chosen: list[tuple[str, FlightSignals, Reading]] = []
    read = 0
    for position in np.random.default_rng(seed).permutation(len(pool)):
        k = pool[int(position)]
        flight = flights[int(sentences["signal_index"][k])]
        reading = reread(flight, stored_sentence(sentences, k), geometry, spec, words)
        read += 1
        stratum = flight_record(reading)["stratum"]
        if wanted[stratum]:
            wanted[stratum] -= 1
            chosen.append((stratum, flight, reading))
        if not any(wanted.values()):
            break
    if any(wanted.values()):
        raise ValueError(f"{airport}: the {len(pool)} labelled {SPLIT} flights hold too few of a stratum; "
                         f"still wanted {wanted}")
    return chosen, {"pool": len(pool), "read": read}


# ---- one flight
def _turn_check(check: dict[str, Any]) -> dict[str, Any]:
    return {"progressOk": bool(check["progress_ok"]), "rateOk": bool(check["rate_ok"]),
            "meanRateDegS": round(float(check["mean_rate_deg_s"]), 4),
            "maxRateDegS": round(float(check["max_rate_deg_s"]), 4),
            "maxBankDeg": round(float(check["max_bank_deg"]), 3),
            "rateMinApplies": bool(check["rate_min_applies"])}


def heading_payload(item: display.HeadingEnvelope) -> dict[str, Any]:
    word = item.word
    return {"row": word.row, "value": word.value, "kind": word.kind, "targetDeg": item.target_deg,
            **band_payload(item.band), "check": {"rows": int(item.check["rows"]), "inside": int(item.check["inside"])}}


def flight_payload(original: FlightSignals, flight: Admitted, reading: Reading, stratum: str,
                   envelopes: display.FlightEnvelopes, globe: Globe, words: Words) -> dict[str, Any]:
    signals, smoothed, relative = flight.signals, flight.smoothed, flight.relative
    rows = signals.n_rows
    lat, lon = globe.latlon(signals.e_m, signals.n_m)
    undulation = globe.undulation_m(lat, lon)
    key = signals.dataset_id.split(":", 1)[1]
    callsign, runway, _icao24, _landing = key.rsplit("_", 3)
    if runway != signals.runway:
        raise ValueError(f"{signals.dataset_id}: the flight key names runway {runway}, the signals {signals.runway}")
    in_force = words_in_force(reading.words)
    unknown = sorted({item.kind for item in reading.instructions} - set(WORD_KINDS))
    if unknown:
        raise ValueError(f"{signals.dataset_id}: the labeller issued word kinds {unknown}, which the export "
                         f"contract (WORD_KINDS) does not name")
    # the capture's distance before the threshold: the labeller's check and the display's corridor read the same row of
    # the same admitted flight, so the file's two fields (the flight's, the corridor's) are written from one value
    capture_before_m = round(float(reading.checks["capture_before_threshold_m"]), 1)
    events = [{"row": item.row, "column": item.column, "value": item.value, "kind": item.kind}
              for item in sorted(reading.instructions, key=lambda item: (item.row, item.column))]
    capture = envelopes.capture_turn
    corridor = envelopes.corridor
    crossing = None
    if flight.cut_at_crossing:
        e, n = original.e_m[rows: rows + 1], original.n_m[rows: rows + 1]
        crossing_lat, crossing_lon = globe.latlon(e, n)
        crossing = {"row": rows, "eM": round(float(e[0]), 1), "nM": round(float(n[0]), 1),
                    "lon": round(float(crossing_lon[0]), 7), "lat": round(float(crossing_lat[0]), 7)}
    return {
        "datasetId": signals.dataset_id, "flightKey": key, "callsign": callsign, "typecode": signals.typecode,
        "runway": flight.candidate.ident, "runwayIndex": reading.runway_index, "stratum": stratum, "rows": rows,
        "captureRow": reading.capture_row, "joinRow": reading.join_row, "unspecifiedRow": reading.unspecified_row,
        "captureBeforeThresholdM": capture_before_m,
        "signals": {
            "tS": rounded(signals.time_s, 3), "eM": rounded(signals.e_m, 1), "nM": rounded(signals.n_m, 1),
            "lon": rounded(lon, 7), "lat": rounded(lat, 7), "altitudeHaeM": rounded(signals.altitude_m + undulation, 2),
            "raw": {"trackDeg": rounded(signals.track_deg, 3), "altitudeM": rounded(signals.altitude_m, 2),
                    "groundSpeedMps": rounded(signals.ground_speed_mps, 3), "verticalRateMps": rounded(signals.vertical_rate_mps, 3)},
            "smoothed": {"trackDeg": rounded(smoothed.track_deg, 3), "altitudeM": rounded(smoothed.altitude_m, 2),
                         "groundSpeedMps": rounded(smoothed.ground_speed_mps, 3), "distanceM": rounded(smoothed.distance_m, 1)},
            "beforeThresholdM": rounded(relative.before_threshold_m, 1), "rightOfCourseM": rounded(relative.right_of_course_m, 1),
        },
        "words": {"events": events, "inForce": [[int(v) for v in in_force[:, c]] for c in range(len(COLUMNS))]},
        "envelopes": {
            "heading": [heading_payload(item) for item in envelopes.heading],
            "approach": {
                "clearanceRow": reading.join_row, "captureRow": reading.capture_row,
                "captureBeforeThresholdM": capture_before_m,
                "captureTurn": None if capture is None else {
                    "startRow": capture.start_row, "endRow": capture.end_row,
                    "courseOnTrackDeg": round(capture.course_on_track_deg, 3), "check": _turn_check(capture.check)},
                "courseBandDeg": _pair(envelopes.course_band_deg, 3),
                "corridor": {"beforeThresholdM": round(corridor.before_threshold_m, 1),
                             "halfWidthAtCaptureM": round(corridor.half_width_at_capture_m, 2),
                             "halfWidthAtThresholdM": round(corridor.half_width_at_threshold_m, 2), "rows": corridor.rows,
                             "axis": globe.line(corridor.axis), "outline": globe.line(corridor.outline)},
                "landing": {"cutAtCrossing": flight.cut_at_crossing,
                            "lastRowBeforeThresholdM": round(float(relative.before_threshold_m[-1]), 1),
                            "crossing": crossing},
            },
            "altitude": [{
                "row": tube.word.row, "endRow": tube.end_row, "value": tube.word.value, "kind": tube.word.kind,
                "targetM": words.altitude_m(tube.word.value),
                "lowerM": rounded(tube.lower_m, 2), "upperM": rounded(tube.upper_m, 2),
                "lowerHaeM": rounded(tube.lower_m + undulation[tube.word.row: tube.end_row], 2),
                "upperHaeM": rounded(tube.upper_m + undulation[tube.word.row: tube.end_row], 2),
                "inside": _flags(tube.inside),
                "check": {"rows": int(tube.check["rows"]), "inside": int(tube.check["inside"]),
                          "contained": bool(tube.check["contained"]),
                          "tubeWidthEndM": round(float(tube.check["tube_width_end_m"]), 2)},
            } for tube in envelopes.altitude],
            "angle": [{"row": item.word.row, "value": item.word.value, "kind": item.word.kind,
                       "measuredDeg": None if item.measured_deg is None else round(item.measured_deg, 3)}
                      for item in envelopes.angle],
            "speed": [{
                "row": span.word.row, "endRow": span.end_row, "value": span.word.value, "kind": span.word.kind,
                "targetMps": span.target_mps, "arrivalRow": span.arrival_row,
                "transitionLowerMps": None if span.transition_lower_mps is None else rounded(span.transition_lower_mps, 3),
                "transitionUpperMps": None if span.transition_upper_mps is None else rounded(span.transition_upper_mps, 3),
                "bandMps": _pair(span.band_mps, 3),
                "bandInside": None if span.band_inside is None else _flags(span.band_inside),
                "check": None if span.check is None else {
                    "arrivalRows": int(span.check["arrival_rows"]), "cutBeforeArrival": bool(span.check["cut_before_arrival"]),
                    "transitionOk": bool(span.check["transition_ok"]), "accelOk": bool(span.check["accel_ok"]),
                    "bandRows": int(span.check["band_rows"]), "bandInside": int(span.check["band_inside"]),
                    "contained": bool(span.check["contained"])},
                "rangeMps": _pair(span.range_mps, 3),
            } for span in envelopes.speed],
        },
    }


# ---- the file
def vocabulary_block(spec: VocabularySpec, words: Words, labeller_sha256: str) -> dict[str, Any]:
    """What the frontend needs to name a word and print its tolerances — every class as a table,
    so the reader looks values up and decodes nothing."""
    def angle_name(index: int) -> str:
        if index == ANGLE_LEVEL:
            return "level"
        return f"descent {index}" if words.is_descent(index) else "climb"

    return {
        "readingRule": spec.reading_rule, "specSha256": spec.sha256, "labellerSourceSha256": labeller_sha256,
        "columns": list(COLUMNS), "unchanged": UNCHANGED, "stepS": spec.step_s,
        "smoothingS": {"track": spec.track_smoothing_s, "altitude": spec.altitude_smoothing_s,
                       "speed": spec.speed_smoothing_s},
        "classCounts": words.class_counts(),
        "approachClasses": [APPROACH_NAMES[index] for index in range(len(APPROACH_NAMES))],
        "headingTargetsDeg": [words.heading_deg(index) for index in range(words.n_heading)],
        "headingLeadS": spec.heading_lead_s, "headingLeadRows": spec.rows_exact(spec.heading_lead_s),
        "headingToleranceDeg": spec.heading_tolerance_deg,
        "turnOnsetRateDegS": spec.turn_onset_rate_deg_s,
        "turnRateMinDegS": spec.turn_rate_min_deg_s, "turnRateMaxDegS": spec.turn_rate_max_deg_s,
        "turnRateMinFromDeg": spec.turn_rate_min_from_deg, "turnBankMaxDeg": spec.turn_bank_max_deg,
        "interceptAngleDeg": spec.intercept_angle_deg,
        "corridorHalfWidthM": spec.corridor_half_width_m, "corridorWideningDeg": spec.corridor_widening_deg,
        "corridorCourseToleranceDeg": spec.corridor_course_tolerance_deg,
        "landingCrossLimitM": spec.landing_cross_limit_m, "landingMaxHeightM": spec.landing_max_height_m,
        "parallelCourseDeltaDeg": spec.parallel_course_delta_deg,
        "altitudeTargetsM": [words.altitude_m(index) for index in range(words.n_altitude_levels)],
        "altitudeLandValue": words.altitude_land, "altitudeToleranceM": spec.altitude_tolerance_m,
        "angleClasses": [{"value": index, "name": angle_name(index), "nominalDeg": words.angle_deg(index),
                          "lowDeg": words.angle_bounds(index)[0], "steepDeg": words.angle_bounds(index)[1]}
                         for index in range(words.n_descent + 2)],
        "angleLevelValue": ANGLE_LEVEL,
        "speedTargetsMps": [words.speed_mps(index) for index in range(words.n_speed_levels)],
        "speedUnspecifiedValue": words.speed_unspecified, "speedToleranceMps": spec.speed_tolerance_mps,
        "speedAccelMaxMps2": spec.speed_accel_max_mps2,
        "speedRangeMps": [spec.speed_min_mps - spec.speed_tolerance_mps, spec.speed_max_mps + spec.speed_tolerance_mps],
    }


def candidates_block(geometry: AirportGeometry, spec: VocabularySpec, globe: Globe,
                     centreline_m: float) -> list[dict[str, Any]]:
    return [{"index": index, "ident": candidate.ident, "thresholdEM": round(candidate.threshold_e_m, 1),
             "thresholdNM": round(candidate.threshold_n_m, 1), "courseDeg": candidate.course_deg,
             "elevationM": candidate.elevation_m, "lengthM": round(candidate.length_m, 1),
             # the landing rule's limit off this runway's centreline (§2.2: capped by a parallel runway)
             "landingCrossLimitM": round(landing_cross_limit_m(geometry, index, spec.landing_cross_limit_m,
                                                               spec.parallel_course_delta_deg), 1),
             "centreline": globe.line(display.centreline(candidate, centreline_m)),
             "runway": globe.line(display.runway(candidate))}
            for index, candidate in enumerate(geometry.candidates)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--dir", type=Path, required=True, help="the instruction artefact directory")
    parser.add_argument("--airports-root", type=Path, required=True,
                        help="the frontend's airports directory (…/public/data/airports)")
    parser.add_argument("--airport", action="append", required=True, help="an ICAO code; repeat for several")
    parser.add_argument("--per-stratum", type=int, default=20, help="flights per stratum (straight-in, vectored)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--set-id", default=READING_RULE.replace("-", "_"))
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    directory = args.dir if args.dir.is_absolute() else REPO_ROOT / args.dir
    root = args.airports_root if args.airports_root.is_absolute() else REPO_ROOT / args.airports_root
    airports = [code.upper() for code in args.airport]
    if args.per_stratum < 1:
        parser.error("--per-stratum must be at least 1")
    if len(set(airports)) != len(airports):
        parser.error(f"an airport is named twice in {airports}")
    started = time.perf_counter()
    export(directory, root, airports, args.per_stratum, args.seed, args.set_id, args.title)
    print(f"  done in {time.perf_counter() - started:.0f} s", flush=True)
    return 0


def export(directory: Path, root: Path, airports: list[str], per_stratum: int, seed: int, set_id: str,
           title: str | None) -> None:
    """The set ``set_id`` at every airport of ``airports``: every airport built and every refusal made before
    anything is written; then each airport's sample and index."""
    spec = load_spec(directory)
    words = Words(spec)
    geometries = load_candidates(directory)
    sentences = load_sentences(directory, SPLIT, spec)
    flights = load_signals(directory, SPLIT)
    labeller = spec_labeller_source(directory)
    missing = [code for code in airports if code not in geometries]
    if missing:
        raise ValueError(f"{missing} are not in {directory / 'candidates.json'} ({sorted(geometries)})")
    # every refusal before anything is written
    existing: dict[str, list[dict[str, Any]]] = {}
    for code in airports:
        out = root / code / "training" / set_id
        if out.exists():
            raise ValueError(f"{out} exists; an export is never overwritten")
        existing[code] = read_index(out.parent, code, set_id)
    artefact_name = repo_relative(directory)
    git = git_state()
    title = title or f"Instruction vocabulary · {READING_RULE} · spec {spec.sha256[:12]} · {SPLIT}"

    # Every airport is BUILT before any is written: a re-read that differs, or an envelope the
    # display refuses, stops the export with nothing on disk.
    built: dict[str, tuple[str, dict[str, Any], dict[str, int]]] = {}
    for code in airports:
        geometry = geometries[code]
        globe = Globe(geometry)
        chosen, counts = draw(code, flights, sentences, geometry, spec, words, per_stratum, seed)
        payloads, reach = [], 0.0
        for stratum, flight, reading in chosen:
            admitted = admit(flight, geometry, spec)
            envelopes = display.flight_envelopes(admitted, reading, spec, words)
            payloads.append(flight_payload(flight, admitted, reading, stratum, envelopes, globe, words))
            reach = max(reach, float(admitted.relative.before_threshold_m.max()))
        centreline_m = math.ceil(reach / CENTRELINE_ROUND_M) * CENTRELINE_ROUND_M
        cohort = {"split": SPLIT, "perStratum": per_stratum, "seed": seed,
                  "drawnFrom": (f"a permutation seeded {seed} of the {counts['pool']:,} labelled {SPLIT} flights "
                                f"at {code} in {artefact_name}, read in that order until {per_stratum} "
                                f"straight-in and {per_stratum} vectored were found ({counts['read']} read); "
                                f"vectored = the heading turned before the capture — word to word, and on from the "
                                f"last word to the course — adding up to at least {VECTORED_TURN_DEG:g}° "
                                f"(readout.flight_record)")}
        sha = candidates_sha256(geometry)
        sample = {
            "schema": SAMPLE_SCHEMA, "setId": set_id, "airport": code, "writtenUtc": utc_now(),
            "producedBy": {"runner": RUNNER, "artefact": artefact_name, "git": git},
            "cohort": {**cohort, "pool": counts["pool"], "read": counts["read"]},
            "vocabulary": vocabulary_block(spec, words, labeller),
            "airportFrame": {"code": code, "lat": geometry.frame.lat0, "lon": geometry.frame.lon0,
                             "elevationM": geometry.frame.alt0},
            "candidatesSha256": sha, "centrelineLengthM": centreline_m,
            "candidates": candidates_block(geometry, spec, globe, centreline_m),
            "flights": payloads,
        }
        entry = {"id": set_id, "kind": KIND_READBACK, "title": title, "file": f"{set_id}/{SAMPLE_FILE}",
                 "vocabularySha256": spec.sha256, "runwaySha256": sha, "readingRule": READING_RULE,
                 "flights": len(payloads), "cohort": cohort,
                 "source": {"artefact": artefact_name, "specSha256": spec.sha256, "labellerSourceSha256": labeller,
                            "exporter": RUNNER, "git": git}}
        built[code] = (serialise(sample), entry, counts)
    # no airport is written while another's index changed since the start (each write checks its own again)
    for code in airports:
        require_index_unchanged(root / code / "training", code, set_id, existing[code])
    for code, (text, entry, counts) in built.items():
        out = write_set(root / code / "training", code, entry, text, existing[code])
        print(f"  {code}: {entry['flights']} flights ({per_stratum} per stratum; {counts['read']} of "
              f"{counts['pool']:,} read), {out.stat().st_size / 1e6:.1f} MB → {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
