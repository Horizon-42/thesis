"""Instruction sentences for the frontend's Training module: a seeded sample of VAL flights per
airport, each with its words and every word's envelope drawn as geometry, from a frozen instruction
artefact (design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`; the words:
`docs/2026-09-23_instruction_vocabulary_design.zh.md`).

    python run_ts.py instruction_training_export \\
        --dir 4dTrajectory/outputs/POOLED/instruction_language/v2_20260924 \\
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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
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
from ts_transformer.instructions.words import (
    ANGLE_LEVEL, APPROACH_CLEARED, APPROACH_GO_AROUND, APPROACH_NOT_CLEARED, COLUMNS, UNCHANGED, Words,
)
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT

#: MIRROR of `aeroviz-4d/src/data/trainingSample.ts` (`TRAINING_INDEX_SCHEMA`,
#: `TRAINING_SAMPLE_SCHEMA`, `TRAINING_READABLE_SET_KIND`); the reader refuses anything else by
#: name, so these move together. A name changes with its file's shape, on both sides, in the same
#: change. Sample v5 is `instruction-v2`'s shape: a turn bounded by rate (the fastest turn on time
#: and the slowest begun late, in plan AND as the heading against time — no turn band), the judged
#: hold (`holdCheck`), the landing limits in the vocabulary and per candidate. The index keeps its
#: v1 shape: sets of every vocabulary sit in it.
INDEX_SCHEMA = "aeroviz-training-index-v1"
SAMPLE_SCHEMA = "aeroviz-training-sample-v5"
KIND_READBACK = "vocabulary-readback"
INDEX_FILE = "index.json"
SAMPLE_FILE = "sample.json"
RUNNER = "ts_transformer.experiments.instruction_training_export"

#: Only validation flights are drawn: train is what a prior will be fitted on, and test stays shut.
SPLIT = "val"

#: Why a word was issued — every `Instruction.kind` the labeller (`instructions/labeller/*`) writes
#: into a kept sentence. MIRROR of `TRAINING_WORD_KINDS` in the frontend reader, which names each
#: one; a kind outside this list stops the export by name rather than reaching a view unnamed.
WORD_KINDS = ("initial", "turn", "turn-split", "intercept", "intercept-split", "clear", "target", "step", "angle",
              "unspecified")

#: The approach column's classes by name, in their class order.
APPROACH_NAMES = {APPROACH_NOT_CLEARED: "not cleared", APPROACH_CLEARED: "cleared", APPROACH_GO_AROUND: "go-around"}

#: The extended centrelines are drawn to the farthest any exported flight was from its own
#: threshold along its course, rounded up to this. A drawing extent, not an envelope value.
CENTRELINE_ROUND_M = 1000.0


# ---- numbers at display precision
def _r(values: Any, digits: int) -> list[float]:
    return [round(float(value), digits) for value in np.asarray(values, dtype=np.float64).ravel()]


def _pair(values: tuple[float, float] | None, digits: int) -> list[float] | None:
    return None if values is None else _r(values, digits)


def _flags(values: np.ndarray) -> list[int]:
    return [int(bool(value)) for value in values]


def candidates_sha256(geometry: AirportGeometry) -> str:
    """The runway pointer's classes ARE the airport's candidates (vocabulary design §4.1), and the
    landing rule reads every runway end of the airport (§2.2), so the identity of both is the sha of
    the airport geometry — candidates and runway ends — the index's ``runwaySha256``."""
    canonical = json.dumps(geometry.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


class Globe:
    """The airport frame on the globe: metres → latitude / longitude, and MSL → the ellipsoid."""

    def __init__(self, geometry: AirportGeometry) -> None:
        self.frame = geometry.frame

    def latlon(self, e_m: np.ndarray, n_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lat, lon = self.frame.latlon_from_horizontal(np.asarray(e_m, dtype=np.float64), np.asarray(n_m, dtype=np.float64))
        return np.asarray(lat), np.asarray(lon)

    def line(self, line: display.Line) -> dict[str, list[float]]:
        lat, lon = self.latlon(line.e_m, line.n_m)
        return {"eM": _r(line.e_m, 1), "nM": _r(line.n_m, 1), "lon": _r(lon, 7), "lat": _r(lat, 7)}

    @staticmethod
    def undulation_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """N = h_HAE − H_MSL per point (EGM96)."""
        return np.asarray(geoid_undulation_m(list(lat), list(lon)), dtype=np.float64)


# ---- the draw
def stored_sentence(sentences: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    offsets = sentences["offsets"]
    return {"words": sentences["words"][offsets[index]: offsets[index + 1]],
            **{name: int(sentences[name][index]) for name in ("runway_index", "capture_row", "join_row", "unspecified_row")}}


def reread(flight: FlightSignals, stored: dict[str, Any], geometry: AirportGeometry, spec: VocabularySpec,
           words: Words) -> Reading:
    """The flight read again, which must give the stored sentence exactly."""
    try:
        reading = read_flight(flight, geometry, spec, words)
    except Refused as refusal:
        raise SystemExit(f"{flight.dataset_id}: the artefact holds its sentence, but this code refuses it "
                         f"({refusal}) — this labeller no longer reads like the artefact's") from None
    if reading.words.shape != stored["words"].shape:
        raise SystemExit(f"{flight.dataset_id}: re-read {reading.words.shape[0]} steps, the artefact holds "
                         f"{stored['words'].shape[0]}")
    differ = np.argwhere(reading.words != stored["words"])
    if len(differ):
        row, column = (int(v) for v in differ[0])
        raise SystemExit(f"{flight.dataset_id}: re-read {COLUMNS[column]} word {int(reading.words[row, column])} at "
                         f"step {row}, the artefact holds {int(stored['words'][row, column])} "
                         f"({len(differ)} cells differ)")
    for name in ("runway_index", "capture_row", "join_row", "unspecified_row"):
        if getattr(reading, name) != stored[name]:
            raise SystemExit(f"{flight.dataset_id}: re-read {name} {getattr(reading, name)}, the artefact holds {stored[name]}")
    return reading


def draw(airport: str, flights: list[FlightSignals], sentences: dict[str, np.ndarray], geometry: AirportGeometry,
         spec: VocabularySpec, words: Words, per_stratum: int, seed: int
         ) -> tuple[list[tuple[str, FlightSignals, Reading]], dict[str, int]]:
    """The seeded permutation of the airport's labelled flights, read in order until both strata
    hold ``per_stratum``; every flight read is checked against its stored sentence."""
    pool = [k for k, index in enumerate(sentences["signal_index"]) if flights[int(index)].airport == airport]
    if not pool:
        raise SystemExit(f"no labelled {SPLIT} flight lands at {airport}")
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
        raise SystemExit(f"{airport}: the {len(pool)} labelled {SPLIT} flights hold too few of a stratum; "
                         f"still wanted {wanted}")
    return chosen, {"pool": len(pool), "read": read}


# ---- one flight
def _profile(profile: display.HeadingProfile) -> dict[str, list[float]]:
    return {"tS": _r(profile.t_s, 3), "deg": _r(profile.deg, 3)}


def _turn(region: display.TurnRegion, globe: Globe) -> dict[str, Any]:
    return {"fromTrackDeg": round(region.from_track_deg, 3), "turnDeg": round(region.turn_deg, 3),
            "rateMinDegS": region.rate_min_deg_s, "rateMaxDegS": region.rate_max_deg_s,
            "bankMaxDeg": region.bank_max_deg, "startDelayMaxS": region.start_delay_max_s,
            "slowFinished": region.slow_finished,
            "region": globe.line(region.outline), "fastPath": globe.line(region.fast),
            "slowPath": globe.line(region.slow), "end": globe.line(region.end),
            "headingFast": _profile(region.heading_fast), "headingSlow": _profile(region.heading_slow),
            "headingRegion": _profile(region.heading_outline)}


def _turn_check(check: dict[str, Any]) -> dict[str, Any]:
    return {"progressOk": bool(check["progress_ok"]), "rateOk": bool(check["rate_ok"]),
            "meanRateDegS": round(float(check["mean_rate_deg_s"]), 4),
            "maxRateDegS": round(float(check["max_rate_deg_s"]), 4),
            "maxBankDeg": round(float(check["max_bank_deg"]), 3),
            "rateMinApplies": bool(check["rate_min_applies"])}


def heading_payload(item: display.HeadingEnvelope, globe: Globe) -> dict[str, Any]:
    word, check = item.word, item.turn_check
    split = word.kind.endswith("-split")
    return {
        "row": word.row, "value": word.value, "kind": word.kind, "targetDeg": item.target_deg,
        "split": {"part": int(word.info["part"]), "parts": int(word.info["parts"])} if split else None,
        "turnEndRow": item.turn_end_row, "holdStartRow": item.hold_start_row, "holdEndRow": item.hold_end_row,
        "fromTrackDeg": round(item.from_track_deg, 3), "targetOnTrackDeg": round(item.target_on_track_deg, 3),
        "holdBandDeg": _pair(item.hold_band_deg, 3),
        "turn": None if item.turn is None else _turn(item.turn, globe),
        "funnel": None if item.funnel is None else {
            "lengthM": round(item.funnel.length_m, 1), "startHalfWidthM": round(item.funnel.start_half_width_m, 1),
            "endHalfWidthM": round(item.funnel.end_half_width_m, 1), "axis": globe.line(item.funnel.axis),
            "outline": globe.line(item.funnel.outline)},
        "check": None if check is None else {
            "kind": check["kind"], "departureRow": int(check["departure_row"]), "arrivalRow": int(check["arrival_row"]),
            "turnDeg": round(float(check["turn_deg"]), 3), "parts": int(check["parts"]), **_turn_check(check)},
        "holdCheck": None if item.hold_check is None else {
            "holdStartRow": int(item.hold_check["hold_start"]), "holdEndRow": int(item.hold_check["hold_end"]),
            "rows": int(item.hold_check["rows"]), "inside": int(item.hold_check["inside"]),
            "halfWidthEndM": round(float(item.hold_check["half_width_end_m"]), 1)},
    }


def flight_payload(original: FlightSignals, flight: Admitted, reading: Reading, stratum: str,
                   envelopes: display.FlightEnvelopes, globe: Globe, words: Words) -> dict[str, Any]:
    signals, smoothed, relative = flight.signals, flight.smoothed, flight.relative
    rows = signals.n_rows
    lat, lon = globe.latlon(signals.e_m, signals.n_m)
    undulation = globe.undulation_m(lat, lon)
    key = signals.dataset_id.split(":", 1)[1]
    callsign, runway, _icao24, _landing = key.rsplit("_", 3)
    if runway != signals.runway:
        raise SystemExit(f"{signals.dataset_id}: the flight key names runway {runway}, the signals {signals.runway}")
    grid = reading.words
    # the word in force at each step: the value at the last row its column was issued at or before
    last_issue = np.maximum.accumulate(np.where(grid != UNCHANGED, np.arange(rows)[:, None], 0), axis=0)
    in_force = grid[last_issue, np.arange(len(COLUMNS))[None, :]]
    unknown = sorted({item.kind for item in reading.instructions} - set(WORD_KINDS))
    if unknown:
        raise SystemExit(f"{signals.dataset_id}: the labeller issued word kinds {unknown}, which the export "
                         f"contract (WORD_KINDS) does not name")
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
        "captureBeforeThresholdM": round(float(reading.checks["capture_before_threshold_m"]), 1),
        "signals": {
            "tS": _r(signals.time_s, 3), "eM": _r(signals.e_m, 1), "nM": _r(signals.n_m, 1),
            "lon": _r(lon, 7), "lat": _r(lat, 7), "altitudeHaeM": _r(signals.altitude_m + undulation, 2),
            "raw": {"trackDeg": _r(signals.track_deg, 3), "altitudeM": _r(signals.altitude_m, 2),
                    "groundSpeedMps": _r(signals.ground_speed_mps, 3), "verticalRateMps": _r(signals.vertical_rate_mps, 3)},
            "smoothed": {"trackDeg": _r(smoothed.track_deg, 3), "altitudeM": _r(smoothed.altitude_m, 2),
                         "groundSpeedMps": _r(smoothed.ground_speed_mps, 3), "distanceM": _r(smoothed.distance_m, 1)},
            "beforeThresholdM": _r(relative.before_threshold_m, 1), "rightOfCourseM": _r(relative.right_of_course_m, 1),
        },
        "words": {"events": events, "inForce": [[int(v) for v in in_force[:, c]] for c in range(len(COLUMNS))]},
        "envelopes": {
            "heading": [heading_payload(item, globe) for item in envelopes.heading],
            "approach": {
                "clearanceRow": reading.join_row, "captureRow": reading.capture_row,
                "captureBeforeThresholdM": round(corridor.before_threshold_m, 1),
                "interceptInserted": bool(reading.checks["intercept_inserted"]),
                "captureTurn": None if capture is None else {
                    "startRow": capture.start_row, "courseOnTrackDeg": round(capture.course_on_track_deg, 3),
                    "check": _turn_check(capture.check), "turn": _turn(capture.region, globe)},
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
                "lowerM": _r(tube.lower_m, 2), "upperM": _r(tube.upper_m, 2),
                "lowerHaeM": _r(tube.lower_m + undulation[tube.word.row: tube.end_row], 2),
                "upperHaeM": _r(tube.upper_m + undulation[tube.word.row: tube.end_row], 2),
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
                "transitionLowerMps": None if span.transition_lower_mps is None else _r(span.transition_lower_mps, 3),
                "transitionUpperMps": None if span.transition_upper_mps is None else _r(span.transition_upper_mps, 3),
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
        "headingToleranceDeg": spec.heading_tolerance_deg, "headingMaxTurnDeg": spec.heading_max_turn_deg,
        "turnRateMinDegS": spec.turn_rate_min_deg_s, "turnRateMaxDegS": spec.turn_rate_max_deg_s,
        "turnRateMinFromDeg": spec.turn_rate_min_from_deg, "turnBankMaxDeg": spec.turn_bank_max_deg,
        "turnStartDelayMaxS": spec.turn_start_delay_max_s,
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


def read_index(training: Path, airport: str, set_id: str) -> list[dict[str, Any]]:
    """The airport's index as it stands; refused when it is another schema's, another airport's,
    or already lists this set (an export is never overwritten)."""
    path = training / INDEX_FILE
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema"] != INDEX_SCHEMA:
        raise SystemExit(f"{path} is a {payload['schema']} file, not {INDEX_SCHEMA}")
    if payload["airport"] != airport:
        raise SystemExit(f"{path} is {payload['airport']}'s index, not {airport}'s")
    if any(item["id"] == set_id for item in payload["sets"]):
        raise SystemExit(f"{path} already lists set {set_id}; an export is never overwritten")
    return payload["sets"]


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

    spec = load_spec(directory)
    words = Words(spec)
    geometries = load_candidates(directory)
    sentences = load_sentences(directory, SPLIT, spec)
    flights = load_signals(directory, SPLIT)
    labeller = spec_labeller_source(directory)
    missing = [code for code in airports if code not in geometries]
    if missing:
        parser.error(f"{missing} are not in {directory / 'candidates.json'} ({sorted(geometries)})")
    # every refusal before anything is written
    existing: dict[str, list[dict[str, Any]]] = {}
    for code in airports:
        out = root / code / "training" / args.set_id
        if out.exists():
            parser.error(f"{out} exists; an export is never overwritten")
        existing[code] = read_index(out.parent, code, args.set_id)
    try:
        artefact_name = directory.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        artefact_name = directory.as_posix()
    git = _git_state()
    title = args.title or f"Instruction vocabulary · {READING_RULE} · spec {spec.sha256[:12]} · {SPLIT}"

    # Every airport is BUILT before any is written: a re-read that differs, or an envelope the
    # display refuses, stops the export with nothing on disk.
    built: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, int]]] = {}
    for code in airports:
        geometry = geometries[code]
        globe = Globe(geometry)
        chosen, counts = draw(code, flights, sentences, geometry, spec, words, args.per_stratum, args.seed)
        payloads, reach = [], 0.0
        for stratum, flight, reading in chosen:
            admitted = admit(flight, geometry, spec)
            envelopes = display.flight_envelopes(admitted, reading, spec, words)
            payloads.append(flight_payload(flight, admitted, reading, stratum, envelopes, globe, words))
            reach = max(reach, float(admitted.relative.before_threshold_m.max()))
        centreline_m = math.ceil(reach / CENTRELINE_ROUND_M) * CENTRELINE_ROUND_M
        cohort = {"split": SPLIT, "perStratum": args.per_stratum, "seed": args.seed,
                  "drawnFrom": (f"a permutation seeded {args.seed} of the {counts['pool']:,} labelled {SPLIT} flights "
                                f"at {code} in {artefact_name}, read in that order until {args.per_stratum} "
                                f"straight-in and {args.per_stratum} vectored were found ({counts['read']} read); "
                                f"vectored = heading turns before the capture adding up to at least "
                                f"{VECTORED_TURN_DEG:g}° (readout.flight_record)")}
        sha = candidates_sha256(geometry)
        sample = {
            "schema": SAMPLE_SCHEMA, "setId": args.set_id, "airport": code, "writtenUtc": utc_now(),
            "producedBy": {"runner": RUNNER, "artefact": artefact_name, "git": git},
            "cohort": {**cohort, "pool": counts["pool"], "read": counts["read"]},
            "vocabulary": vocabulary_block(spec, words, labeller),
            "airportFrame": {"code": code, "lat": geometry.frame.lat0, "lon": geometry.frame.lon0,
                             "elevationM": geometry.frame.alt0},
            "candidatesSha256": sha, "centrelineLengthM": centreline_m,
            "candidates": candidates_block(geometry, spec, globe, centreline_m),
            "flights": payloads,
        }
        entry = {"id": args.set_id, "kind": KIND_READBACK, "title": title, "file": f"{args.set_id}/{SAMPLE_FILE}",
                 "vocabularySha256": spec.sha256, "runwaySha256": sha, "readingRule": READING_RULE,
                 "flights": len(payloads), "cohort": cohort,
                 "source": {"artefact": artefact_name, "specSha256": spec.sha256, "labellerSourceSha256": labeller,
                            "exporter": RUNNER, "git": git}}
        built[code] = (sample, entry, counts)
    for code, (sample, entry, counts) in built.items():
        out = root / code / "training" / args.set_id
        out.mkdir(parents=True)
        write_json_atomic(out / SAMPLE_FILE, sample, allow_nan=False)
        write_json_atomic(out.parent / INDEX_FILE, {"schema": INDEX_SCHEMA, "writtenUtc": utc_now(), "airport": code,
                                                    "sets": [*existing[code], entry]})
        size = (out / SAMPLE_FILE).stat().st_size / 1e6
        print(f"  {code}: {entry['flights']} flights ({args.per_stratum} per stratum; {counts['read']} of "
              f"{counts['pool']:,} read), {size:.1f} MB → {out / SAMPLE_FILE}", flush=True)
    print(f"  done in {time.perf_counter() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
