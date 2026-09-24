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
the set, never into it: a file of its own schema under ``training/<overlay-id>/``, listed in
``training/overlays.json`` (`OVERLAYS_SCHEMA`) with the set it is drawn over and that set's sample
sha256. The helpers those runners share live here, with the files they bind to: `open_base_set`,
`base_flights`, `read_overlays`, `overlay_entry`, `serialise_overlay`, `write_overlay`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
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
#: change. Sample v7 (2026-09-24) is `instruction-v3`'s shape: a heading word is the band θ ± the
#: heading tolerance over the rows it is judged on (from its row plus the lead to the next heading
#: word's, `display.HeadingBand`) with each row's verdict — no turn region, turn end, hold funnel,
#: split part or inserted intercept any more; the capture turn is its rows and the labeller's check;
#: the vocabulary carries the lead. (v6 was `instruction-v2`'s: turns bounded by rate, judged holds;
#: a flight's ``typecode`` its own ICAO type or null, as in v7.) The index keeps its v1 shape: sets
#: of every vocabulary sit in it.
INDEX_SCHEMA = "aeroviz-training-index-v1"
SAMPLE_SCHEMA = "aeroviz-training-sample-v7"
KIND_READBACK = "vocabulary-readback"
INDEX_FILE = "index.json"
SAMPLE_FILE = "sample.json"
RUNNER = "ts_transformer.experiments.instruction_training_export"

#: MIRROR of `aeroviz-4d/src/data/trainingOverlays.ts` (`TRAINING_OVERLAYS_SCHEMA`, `TRAINING_OVERLAY_KINDS`): the
#: manifest of what is drawn OVER this airport's sets — another model's output on a set's own flights, each in a
#: file of its own schema (`executor_training_export`: the executor's replay; `prior_training_export`: the prior's
#: predictions). The index lists sets and keeps its shape; this file lists overlays, each naming the set it is
#: drawn over and that set's sample by its sha256, so a set re-exported under the same id is not mistaken for it.
OVERLAYS_SCHEMA = "aeroviz-training-overlays-v1"
OVERLAYS_FILE = "overlays.json"
KIND_EXECUTOR = "executor-replay"
KIND_PRIOR = "prior-prediction"
OVERLAY_KINDS = (KIND_EXECUTOR, KIND_PRIOR)

#: Only validation flights are drawn: train is what a prior will be fitted on, and test stays shut.
SPLIT = "val"

#: Why a word was issued — every `Instruction.kind` the labeller (`instructions/labeller/*`) writes
#: into a kept sentence. MIRROR of `TRAINING_WORD_KINDS` in the frontend reader, which names each
#: one; a kind outside this list stops the export by name rather than reaching a view unnamed.
WORD_KINDS = ("initial", "per-step", "clear", "target", "step", "angle", "unspecified")

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
def _turn_check(check: dict[str, Any]) -> dict[str, Any]:
    return {"progressOk": bool(check["progress_ok"]), "rateOk": bool(check["rate_ok"]),
            "meanRateDegS": round(float(check["mean_rate_deg_s"]), 4),
            "maxRateDegS": round(float(check["max_rate_deg_s"]), 4),
            "maxBankDeg": round(float(check["max_bank_deg"]), 3),
            "rateMinApplies": bool(check["rate_min_applies"])}


def band_payload(band: display.HeadingBand) -> dict[str, Any]:
    """A heading word's band as the reader takes it: its judged rows, θ ± the tolerance on the chart's branch, and each
    row's verdict (the executor's overlay writes its words' bands the same way)."""
    return {"firstRow": band.first_row, "stopRow": band.stop_row, "targetOnTrackDeg": round(band.target_on_track_deg, 3),
            "bandDeg": _pair(band.band_deg, 3), "inside": _flags(band.inside)}


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
            "heading": [heading_payload(item) for item in envelopes.heading],
            "approach": {
                "clearanceRow": reading.join_row, "captureRow": reading.capture_row,
                "captureBeforeThresholdM": round(corridor.before_threshold_m, 1),
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


# ---- overlays: another model's output, drawn over an exported set's own flights
@dataclass(frozen=True)
class BaseSet:
    """An exported set an overlay is drawn over: its index entry, its sample, and the sample file's sha256."""

    airport: str
    entry: dict[str, Any]
    sample: dict[str, Any]
    sha256: str

    @property
    def block(self) -> dict[str, Any]:
        """What an overlay records of its set: the frontend matches the set id, the sample's time of writing and
        the spec against the sample it loaded; `check-publication` matches the sha256 of the file on disk."""
        return {"setId": self.entry["id"], "sampleWrittenUtc": self.sample["writtenUtc"], "sampleSha256": self.sha256,
                "specSha256": self.sample["vocabulary"]["specSha256"]}


def open_base_set(training: Path, airport: str, set_id: str, spec: VocabularySpec) -> BaseSet:
    """A set this exporter writes and the frontend reads: listed in the airport's index as a read-back of this
    reading rule and ``spec``, its sample under `SAMPLE_SCHEMA`, drawn from `SPLIT`. Anything else is refused by
    name — an overlay drawn over a set the frontend refuses would never be seen."""
    path = training / INDEX_FILE
    index = json.loads(path.read_text(encoding="utf-8"))
    if index["schema"] != INDEX_SCHEMA or index["airport"] != airport:
        raise SystemExit(f"{path} is not {airport}'s {INDEX_SCHEMA} index")
    listed = [item for item in index["sets"] if item["id"] == set_id]
    if not listed:
        raise SystemExit(f"{path} lists no set {set_id}")
    (entry,) = listed
    if (entry["kind"], entry["readingRule"], entry["vocabularySha256"]) != (KIND_READBACK, READING_RULE, spec.sha256):
        raise SystemExit(f"set {set_id} at {airport} is a {entry['kind']} set of {entry['readingRule']} / spec "
                         f"{entry['vocabularySha256'][:12]}, not a {KIND_READBACK} set of {READING_RULE} / spec "
                         f"{spec.sha256[:12]}")
    file = training / entry["file"]
    raw = file.read_bytes()
    sample = json.loads(raw)
    if sample["schema"] != SAMPLE_SCHEMA:
        raise SystemExit(f"{file} is a {sample['schema']} file, not {SAMPLE_SCHEMA}: re-export the set first")
    found = (sample["setId"], sample["airport"], sample["vocabulary"]["specSha256"], sample["cohort"]["split"])
    if found != (set_id, airport, spec.sha256, SPLIT):
        raise SystemExit(f"{file} holds set {found[0]} at {found[1]}, spec {found[2][:12]}, split {found[3]}; expected "
                         f"{set_id} at {airport}, spec {spec.sha256[:12]}, split {SPLIT}")
    return BaseSet(airport, entry, sample, hashlib.sha256(raw).hexdigest())


def base_flights(base: BaseSet, flights: list[FlightSignals], sentences: dict[str, np.ndarray]) -> list[tuple[FlightSignals, int]]:
    """Each of the set's flights in the artefact — its signals and its sentence's position in ``sentences`` — in the
    set's order; refused unless the stored sentence is the set's own: every word at its step and column, and the
    same number of steps."""
    by_id = {flight.dataset_id: i for i, flight in enumerate(flights)}
    stored_at = {int(index): k for k, index in enumerate(sentences["signal_index"])}
    out = []
    for item in base.sample["flights"]:
        i = by_id.get(item["datasetId"])
        if i is None or i not in stored_at:
            raise SystemExit(f"{item['datasetId']} of set {base.entry['id']} has no labelled sentence in the artefact")
        k = stored_at[i]
        grid = stored_sentence(sentences, k)["words"]
        cells = [(int(r), int(c), int(grid[r, c])) for r, c in zip(*np.nonzero(grid != UNCHANGED))]
        events = [(event["row"], event["column"], event["value"]) for event in item["words"]["events"]]
        if len(grid) != item["rows"] or cells != events:
            raise SystemExit(f"{item['datasetId']}: the artefact's sentence ({len(grid)} steps) is not the one set "
                             f"{base.entry['id']} shows ({item['rows']} steps)")
        out.append((flights[i], k))
    return out


def read_overlays(training: Path, airport: str, overlay_id: str) -> list[dict[str, Any]]:
    """The airport's overlays as they stand (none yet: an empty list); refused when the manifest is another
    schema's or another airport's, or already lists this overlay — an overlay is never overwritten."""
    path = training / OVERLAYS_FILE
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema"] != OVERLAYS_SCHEMA:
        raise SystemExit(f"{path} is a {payload['schema']} file, not {OVERLAYS_SCHEMA}")
    if payload["airport"] != airport:
        raise SystemExit(f"{path} is {payload['airport']}'s overlays, not {airport}'s")
    if any(item["id"] == overlay_id for item in payload["overlays"]):
        raise SystemExit(f"{path} already lists overlay {overlay_id}; an overlay is never overwritten")
    return payload["overlays"]


def overlay_entry(overlay_id: str, kind: str, base: BaseSet, title: str, file_name: str, flights: int,
                  source: dict[str, Any]) -> dict[str, Any]:
    """One overlay as the manifest lists it: what it is, the set it is drawn over, and where its file is."""
    if kind not in OVERLAY_KINDS:
        raise ValueError(f"unknown overlay kind {kind!r}")
    return {"id": overlay_id, "kind": kind, "base": base.entry["id"], "baseSampleSha256": base.sha256, "title": title,
            "file": f"{overlay_id}/{file_name}", "flights": flights, "source": source}


def serialise_overlay(payload: dict[str, Any]) -> str:
    """An overlay's file as it is written: without indentation (its per-step arrays are long, and one number per line
    triples its size) and refusing NaN. Called while every airport is built, so a payload that cannot be written
    stops the run before the first airport is."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def write_overlay(training: Path, airport: str, overlay_id: str, entry: dict[str, Any], text: str,
                  existing: list[dict[str, Any]]) -> Path:
    """The overlay's file (``text``, `serialise_overlay`'s) in a directory of its own (refused if it exists), then the
    manifest with it added — refused when the manifest is no longer what ``existing`` read at the start of the run
    (another export wrote it meanwhile: adding to the old list would drop that one's entry)."""
    if read_overlays(training, airport, overlay_id) != existing:
        raise SystemExit(f"{training / OVERLAYS_FILE} changed since this run read it; run the export again")
    out = training / entry["file"]
    out.parent.mkdir()
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(out)
    write_json_atomic(training / OVERLAYS_FILE, {"schema": OVERLAYS_SCHEMA, "writtenUtc": utc_now(), "airport": airport,
                                                 "overlays": [*existing, entry]})
    return out


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
                                f"vectored = the heading turned before the capture — word to word, and on from the "
                                f"last word to the course — adding up to at least {VECTORED_TURN_DEG:g}° "
                                f"(readout.flight_record)")}
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
