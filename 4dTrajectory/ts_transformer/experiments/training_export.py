"""A23: the Training export of stage A (vocabulary §12.1 A23, outline §6) — for each airport, a set of train and select
flights the frontend's Training view reads: the observed track, the open-loop sentence on the 2 s rows, and at
Δ = 2, 4, 8 s the closed-loop sentence (each correction word marked) with its flown states, the judge's outcome and the
decision-altitude check, each flown again here; the attitudes; the envelopes of every word (the views compute none).

WHO (D86: no formal replay row is read). Of each split (`training_files.SPLITS`: train and select; `split_flights` takes
any split, val included, for a readout that a plan makes, outline §6 item 4), per airport, ``--per-stratum``
straight-in and as many vectored flights, in a permutation seeded by ``--seed`` of the flights with a closed-loop sentence
at every Δ (`choose`), each one's stratum the sentence file's (D70) and its kind whether its sentence has a go-around.

CHECKED, NOT TRUSTED. Every flight is read again by the labeller and must give its stored sentence (words and runway);
every closed-loop sentence is flown again here from its first predicted step (`training_flights`, the setup the live
executor shares, after the start's refusals; `replay.fly_batch`) and must give its stored states on every 2 s row
(within the executor conformance's bound) and its stored outcome (D74). The executor spec opens, and the closed-loop sentences are read, only for the code in the
process that passes the labeller's, the executor's and the closed loop's checks, run here first
(`closed_loop.require_conforming_closed_loop`, D73). One split's chosen flights, from their head to each Δ's closed-loop
sentence flown again, are `split_flights` (A36): the Training export of the prior calls it with its own Δ.

WRITES a set ``<root>/<airport>/training/<set-id>/sample.json`` and its entry in ``<root>/<airport>/training/index_v5.json``
(`training_files`), never the instruction-v3 view's ``training/index.json``; refused when the set exists. Every airport
is built before any is written. From a clean tree (the set records the commit).

    python run_ts.py training_export \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v9_20261004 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v14_20261004 --set-id closed_loop_v9_20261004
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from trajectory_data_process.harvest.airports import load_airport
from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.frame import ALT, LAT, LON
from ts_transformer.autopilot.judge import Verdict, flown_track, read_flown, words_said
from ts_transformer.autopilot.spec import EXECUTOR_SPEC_SCHEMA
from ts_transformer.experiments import training_flights
from ts_transformer.experiments.training_attitude import attitude_payload, executor_attitude, observed_attitude
from ts_transformer.instructions import training_files as files
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import (
    CANDIDATES_SCHEMA, CLOSED_LOOP_SCHEMA, SENTENCES_SCHEMA, ClosedLoopSentence, closed_loop_indices, load_sentences,
    signals_flights,
)
from ts_transformer.instructions.labeller.read import Reading, admit
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.labeller.speed import span_checks
from ts_transformer.instructions.labeller.vertical import tube_bounds, tube_checks
from ts_transformer.instructions.readout import KINDS, STRATA as READOUT_STRATA
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import READING_RULE, SPEC_SCHEMA, VocabularySpec
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, ANGLE_LEVEL, COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, SPEED, UNCHANGED, Words,
)
from ts_transformer.io_utils import file_sha256
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

#: The row intervals of the ablation (D25), each a closed-loop sentence of every exported flight.
ROW_INTERVALS_S = (2.0, 4.0, 8.0)
#: The strata a split's flights are drawn by (the sentence file's, D70), each ``--per-stratum`` flights.
STRATA = READOUT_STRATA


def choose(indices: list[set[int]], strata: dict[int, str], records: list[dict[str, Any]], airport: str,
           per_stratum: int, seed: int) -> list[str]:
    """The airport's flights of the split (module docstring): ``per_stratum`` of each stratum (``strata``: the sentence
    file's by signal index, D70) among those with a closed-loop sentence at every Δ (``indices``: each Δ's flights,
    `artefact.closed_loop_indices`; ``records``: the split's flight records, `artefact.signals_flights`), in a seeded
    permutation of their sorted ids; refused when a stratum has too few."""
    common = set.intersection(*indices)
    chosen = []
    for stratum in STRATA:
        pool = sorted(records[i]["dataset_id"] for i in common
                      if records[i]["airport"] == airport and strata[i] == stratum)
        if len(pool) < per_stratum:
            raise ValueError(f"{airport}: {len(pool)} {stratum} flights with a closed-loop sentence at every row interval, "
                             f"{per_stratum} asked")
        order = np.random.default_rng(seed).permutation(len(pool))
        chosen += [pool[i] for i in order[:per_stratum]]
    return chosen


# ---- the envelopes of a sentence's words on a track (the views draw them)
def envelopes(instructions: list[Instruction], track_deg: np.ndarray, distance_m: np.ndarray, altitude_msl_m: np.ndarray,
              ground_speed_mps: np.ndarray, geometry: AirportGeometry, spec: VocabularySpec, words: Words
              ) -> dict[str, list[dict[str, Any]]]:
    """The words' envelopes on a smoothed track (its 2 s rows): the heading bands (`envelope.heading_words_inside`'s
    rows and verdicts), the altitude tubes (`labeller.vertical`, on the height above E, D58) and the speed spans — the
    labeller's own checks, on the observed track, and the judge's, on a flown one."""
    rows = len(track_deg)
    heading = [(word.row, float(word.info["target_deg"])) for word in instructions if word.column == HEADING]
    height = np.asarray(altitude_msl_m) - geometry.elevation_m
    tubes = [files.tube_payload(word.row, end, words.altitude_level_m(word.value), low, high, geometry.elevation_m, check)
             for (word, end, low, high), check in zip(tube_bounds(instructions, distance_m, height, spec, words),
                                                      tube_checks(instructions, distance_m, height, spec, words))]
    speeds = [item for item in instructions if item.column == SPEED and words.speed_mps(item.value) is not None]
    spans = [files.speed_payload(check, words.speed_mps(word.value), spec.speed_tolerance_mps)
             for word, check in zip(sorted(speeds, key=lambda item: item.row),
                                    span_checks(instructions, ground_speed_mps, spec, words))]
    return {"heading": files.heading_bands(track_deg, heading, spec.rows_exact(spec.heading_lead_s), rows,
                                           spec.heading_tolerance_deg),
            "altitude": tubes, "speed": spans}


def said(column: int, value: int, runway_index: int, geometry: AirportGeometry, words: Words) -> dict[str, Any]:
    """What a word says, decoded with the vocabulary (`Words`) so the views decode nothing: the runway word its candidate
    or go-around; a heading word its class relative to the course of the runway in force (``runway_index``) and the
    compass track that makes (D8); an altitude word its level above E and in MSL, or "no level-off" (D58); an angle word
    its nominal angle (descending positive); a speed word its ground speed, or "unspecified"."""
    if column == RUNWAY:
        return {"goAround": True} if value == RUNWAY_GO_AROUND else {"runway": geometry.candidates[value].ident,
                                                                      "runwayIndex": value}
    if column == HEADING:
        course = geometry.candidates[runway_index].course_deg
        return {"relativeDeg": words.heading_relative_deg(value), "trackDeg": round(words.heading_track_deg(value, course), 3)}
    if column == ALTITUDE:
        level = words.altitude_level_m(value)
        return {"noLevelOff": level is None, "levelM": level,
                "mslM": None if level is None else round(words.altitude_msl_m(value, geometry.elevation_m), 1)}
    if column == ANGLE:
        return {"angleDeg": words.angle_deg(value), "climb": value == words.angle_climb, "level": value == ANGLE_LEVEL}
    return {"speedMps": words.speed_mps(value)}


def events(grid: np.ndarray, correction: np.ndarray | None, geometry: AirportGeometry, words: Words
           ) -> list[dict[str, Any]]:
    """Every word said in a sentence's grid (``[rows, 5]``, row 0 says every column), each with its row, column and
    value, whether the closed-loop reading added it (``correction``; an open-loop sentence has none) and what it says
    (`said`, a heading word with the runway in force at its row: go-around changes no runway, D27)."""
    out, runway = [], -1
    for row in range(len(grid)):
        if grid[row, RUNWAY] >= 0:
            runway = int(grid[row, RUNWAY])
        for column in np.flatnonzero(grid[row] != UNCHANGED):
            value = int(grid[row, column])
            out.append({"row": row, "column": int(column), "value": value,
                        "correction": bool(correction is not None and correction[row, column]),
                        "says": said(int(column), value, runway, geometry, words)})
    return out


# ---- one flight
def observed_payload(flight: FlightSignals, geometry: AirportGeometry, attitude: dict[str, np.ndarray | None]
                     ) -> dict[str, Any]:
    """The observed track on its 2 s rows, with the attitude of each row (`observed_attitude`, of the same rows)."""
    if len(attitude["headingDeg"]) != flight.n_rows:
        raise ValueError(f"{flight.dataset_id}: {len(attitude['headingDeg'])} attitude rows for {flight.n_rows} observed rows")
    lat, lon = geometry.frame.latlon_from_horizontal(flight.e_m, flight.n_m)
    return {"rows": flight.n_rows, "timeS": files.rounded(flight.time_s, 1), "eM": files.rounded(flight.e_m, 1),
            "nM": files.rounded(flight.n_m, 1), "latDeg": files.rounded(lat, 7), "lonDeg": files.rounded(lon, 7),
            "altitudeMslM": files.rounded(flight.altitude_m, 1), "trackDeg": files.rounded(flight.track_deg, 2),
            "groundSpeedMps": files.rounded(flight.ground_speed_mps, 2),
            "verticalRateMps": files.rounded(flight.vertical_rate_mps, 2), "attitude": attitude_payload(attitude)}


def open_loop_payload(flight: FlightSignals, reading: Reading, geometry: AirportGeometry, spec: VocabularySpec,
                      words: Words) -> dict[str, Any]:
    """The labelled (open-loop) sentence on the 2 s rows, with the labeller's envelopes on the observed track."""
    admitted = admit(flight, geometry, spec)
    smoothed = admitted.smoothed
    return {"rows": len(reading.words), "words": reading.words.astype(int).tolist(), "events": events(reading.words, None, geometry, words),
            "kinds": [{"row": item.row, "column": item.column, "kind": item.kind} for item in reading.instructions],
            "captureRow": reading.capture_row, "unspecifiedRow": reading.unspecified_row,
            "goAroundRows": list(reading.go_around_rows), "runwayAgainRows": list(reading.runway_again_rows),
            "envelopes": envelopes(reading.instructions, smoothed.track_deg, smoothed.distance_m, smoothed.altitude_m,
                                   smoothed.ground_speed_mps, geometry, spec, words)}


def flight_head(flight: FlightSignals, flight_key: str, split: str, stratum: str, kind: str, group: str, reading: Reading,
                geometry: AirportGeometry, attitude: dict[str, np.ndarray | None], words: Words) -> dict[str, Any]:
    """A set's flight before its closed-loop sentences: who it is, its observed track and its open-loop sentence."""
    return {"datasetId": flight.dataset_id, "flightKey": flight_key, "split": split, "stratum": stratum, "kind": kind,
            "typecode": flight.typecode, "group": group, "runway": flight.runway, "runwayIndex": reading.runway_index,
            "entryTimeUtc": flight.entry_time_utc, "observed": observed_payload(flight, geometry, attitude),
            "openLoop": open_loop_payload(flight, reading, geometry, words.spec, words), "closedLoop": {}}


def flown_states_row_cycles(sentence: ClosedLoopSentence, cycle_s: float, step_s: float) -> np.ndarray:
    """The cycle of each flown 2 s row of a stored sentence, from its first predicted step (the replay's cycle 0)."""
    return np.arange(len(sentence.rows.flown_states)) * int(round(step_s / cycle_s))


def flown_sentence(flown: Flown, j: int, geometry: AirportGeometry, reference: FlightSignals, said: np.ndarray,
                   sentence_step_s: float, aero: np.ndarray, words: Words, *, outcome: str, end_cycle: int,
                   last_cycle: int, crossing: dict[str, Any] | None) -> dict[str, Any]:
    """The block of a sentence the executor flew, in every stage (vocabulary §6 item 8; outline §6.2 item 7, D135):
    flight ``j`` of ``flown``, told the words ``said`` (``[rows, 5]``, rows ``sentence_step_s`` apart, the executor's own
    sentence rows); ended in ``outcome`` at the state row ``end_cycle``, drawn to the state ``last_cycle``
    (`training_flights.last_state_cycle`; a window ended at a loss of separation: the end of its row), with its
    ``crossing`` (`training_flights.crossing_payload`; None when it crossed nothing or no judge ended it). Its outcome and
    end cycle, the crossing with the DA check, the flight to its outcome on the 2 s rows from the first predicted step,
    unrounded (``track``: prior D127, for every stage; with its ``attitude``), and the judge's envelopes of the words on it
    (`envelopes`: each word from the flown row where the executor heard it, `judge.words_said`; None when fewer than two
    rows were flown), refused unless they end within the track. ``reference``: the observed flight, of which only the
    identity is read (`judge.flown_signals`). A stage adds its own fields beside it."""
    spec = words.spec
    cycles = np.arange(0, last_cycle + 1, int(round(spec.step_s / flown.cycle_s)))
    whole = flown_track(flown.states[j, : last_cycle + 1].cpu().numpy(), geometry)
    states = flown.states[j, cycles].cpu().numpy()
    judged = None
    smoothed = read_flown(flown, j, outcome, end_cycle, geometry, reference, spec)
    if smoothed is not None:
        heard = words_said(flown, j, replay.instructions_of(said, geometry, words), sentence_step_s, spec, end_cycle)
        reached = [word for word in heard.moved if word.row < len(smoothed.track_deg)]
        judged = envelopes(reached, smoothed.track_deg, smoothed.distance_m, smoothed.altitude_m,
                           smoothed.ground_speed_mps, geometry, spec, words)
        # a heading word whose lead runs past the flight's end has an empty band (``firstRow == stopRow``, maybe past
        # the track): not judged, and nothing to draw (`envelope.heading_word_rows`)
        ends = ([band["stopRow"] for band in judged["heading"] if band["stopRow"] > band["firstRow"]]
                + [tube["endRow"] for tube in judged["altitude"]] + [span["endRow"] for span in judged["speed"]])
        if max(ends, default=0) > len(cycles):
            raise ValueError(f"{reference.dataset_id}: an envelope ends at row {max(ends)}, past the flown track's "
                             f"{len(cycles)} rows")
    return {"outcome": outcome, "endCycle": int(end_cycle), "crossing": crossing,
            "track": {"rows": len(cycles), "lastCycle": int(last_cycle), "eM": files.unrounded(whole["e"][cycles]),
                      "nM": files.unrounded(whole["n"][cycles]), "latDeg": files.unrounded(states[:, LAT]),
                      "lonDeg": files.unrounded(states[:, LON]), "heightMslM": files.unrounded(states[:, ALT]),
                      "trackDeg": files.unrounded(np.mod(whole["track"][cycles], 360.0)),
                      "groundSpeedMps": files.unrounded(whole["ground_speed"][cycles]),
                      "verticalRateMps": files.unrounded(whole["vertical_rate"][cycles])},
            "attitude": attitude_payload(executor_attitude(flown, j, cycles, aero)), "envelopes": judged}


def replay_payload(flown: Flown, j: int, verdict: Verdict, part: replay.Batch, sentence: ClosedLoopSentence,
                   aero: np.ndarray, spec: VocabularySpec, words: Words) -> dict[str, Any]:
    """The closed-loop sentence flown again (module docstring): refused unless it gives its stored states on every 2 s
    row and its stored outcome (D74, D86); its block (`flown_sentence`: the flight to its outcome, a dynamics failure's
    failed state left out, as the live executor draws it) and stage A's own fields: whether it flew the sentence and the
    words not reached."""
    geometry = part.geometries[j]
    rows = flown_states_row_cycles(sentence, flown.cycle_s, spec.step_s)
    track = flown_track(flown.states[j, : rows[-1] + 1].cpu().numpy(), geometry)
    again = np.column_stack([track["e"][rows], track["n"][rows], track["height"][rows]])
    apart = float(np.abs(again - sentence.rows.flown_states[:, :3]).max())
    if not apart <= STATE_BOUND_M:                     # a NaN state is refused too
        raise ValueError(f"{part.signals[j].dataset_id}: flown again {apart:.3g} m from its closed-loop states")
    if verdict.outcome != sentence.withheld.outcome:
        raise ValueError(f"{part.signals[j].dataset_id}: flown again to {verdict.outcome}, stored "
                         f"{sentence.withheld.outcome}")
    block = flown_sentence(flown, j, geometry, part.signals[j], part.sentences[j].grid, part.row_interval_s, aero, words,
                           outcome=verdict.outcome, end_cycle=verdict.end_row,
                           last_cycle=training_flights.last_state_cycle(verdict.outcome, verdict.end_row),
                           crossing=training_flights.crossing_payload(verdict, flown, j, geometry))
    return {**block, "flewTheSentence": bool(verdict.flew_the_sentence),
            "notReached": 0 if verdict.words is None else int(verdict.words["not_reached"])}


def closed_loop_payload(stored: ClosedLoopSentence, replayed: dict[str, Any], interval_s: float,
                        geometry: AirportGeometry, words: Words) -> dict[str, Any]:
    """A closed-loop sentence as the Training view draws it: its rows, and — for the view's readouts, never a model's
    input — what is withheld (D82)."""
    sentence, withheld = stored.rows, stored.withheld
    states = sentence.states
    lat, lon = geometry.frame.latlon_from_horizontal(states[:, 0], states[:, 1])
    return {"rowIntervalS": interval_s, "firstRow": sentence.first_row, "startRow": sentence.start,
            # the row of ``states`` the executor flew from (the first predicted step; the replay's cycle 0)
            "flownFromRow": int(np.flatnonzero(sentence.on_interval)[sentence.start]),
            "words": sentence.grid.astype(int).tolist(), "events": events(sentence.grid, sentence.correction, geometry, words),
            "states": {"rows": len(states), "eM": files.rounded(states[:, 0], 1), "nM": files.rounded(states[:, 1], 1),
                       "latDeg": files.rounded(lat, 7), "lonDeg": files.rounded(lon, 7),
                       "heightMslM": files.rounded(states[:, 2], 1), "trackDeg": files.rounded(states[:, 3], 2),
                       "groundSpeedMps": files.rounded(states[:, 4], 2), "verticalRateMps": files.rounded(states[:, 5], 2),
                       "onInterval": [int(v) for v in sentence.on_interval]},
            "lateralM": files.nullable(withheld.lateral_m, 1), "verticalM": files.nullable(withheld.vertical_m, 1),
            "uncorrectable": withheld.uncorrectable.astype(int).tolist(),
            "observedRow": [int(v) for v in withheld.observed_row], "matchedRow": files.rounded(withheld.matched_row, 2),
            "timedOut": bool(withheld.timed_out), "replay": replayed}


# ---- the set's head
def vocabulary_block(spec: VocabularySpec, words: Words) -> dict[str, Any]:
    def angle_name(index: int) -> str:
        if index == 0:
            return "level"
        return "climb" if index == words.angle_climb else f"descent {index}"

    return {"readingRule": READING_RULE, "columns": list(COLUMNS), "stepS": spec.step_s,
            "headingStepDeg": spec.heading_step_deg, "headingLeadS": spec.heading_lead_s,
            "headingToleranceDeg": spec.heading_tolerance_deg,
            "altitudeLevelsM": files.rounded(words.altitude_levels, 1),
            "altitudeTolerancesM": files.rounded(words.altitude_tolerances, 1), "noLevelOff": words.altitude_no_level_off,
            "angleClasses": [{"index": k, "name": angle_name(k), "nominalDeg": words.angle_deg(k),
                              "lowDeg": words.angle_bounds(k)[0], "highDeg": words.angle_bounds(k)[1]}
                             for k in range(words.n_descent + 2)],
            "speed": {"minMps": spec.speed_min_mps, "stepMps": spec.speed_step_mps, "levels": words.n_speed_levels,
                      "unspecified": words.speed_unspecified, "toleranceMps": spec.speed_tolerance_mps},
            "closedLoopLateralM": spec.closed_loop_lateral_m, "closedLoopVerticalM": spec.closed_loop_vertical_m,
            "rowIntervalsS": list(ROW_INTERVALS_S)}


def candidate_hae_minus_msl_m(runway_ends_from: dict[str, str], geometry: AirportGeometry) -> dict[str, float]:
    """Each candidate's HAE − MSL offset, metres by ident, from the published runway data its candidates were read from
    (the artefact's ``signals.json`` ``runway_ends_from``: the runway configuration and the CIFP, refused unless the
    files are those, by sha256): what the data plane subtracts from the reported heights of a flight landing there. A
    flight's MSL height plus its runway's offset is the height its aircraft reported — the ellipsoid height Cesium draws
    in. Refused by name for a candidate the published data gives no offset (D78: a candidate needs no arrival, so the
    arrival manifest's runway targets cannot give it)."""
    for name, path in (("config", DEFAULT_CONFIG), ("cifp", DEFAULT_CIFP)):
        if file_sha256(path) != runway_ends_from[f"{name}_sha256"]:
            raise ValueError(f"{path} is not the {name} the artefact's candidates were read from "
                             f"(sha256 {runway_ends_from[f'{name}_sha256'][:12]} recorded)")
    published = {str(runway.ident).upper(): float(runway.hae_minus_msl_m)
                 for runway in load_airport(geometry.code, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways}
    missing = [c.ident for c in geometry.candidates if c.ident not in published]
    if missing:
        raise ValueError(f"{geometry.code}: the published runway data gives no height offset for candidate(s) {missing}")
    return {c.ident: published[c.ident] for c in geometry.candidates}


def candidates_block(geometry: AirportGeometry, hae_minus_msl_m: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for index, candidate in enumerate(geometry.candidates):
        lat, lon = geometry.frame.latlon_from_horizontal(candidate.threshold_e_m, candidate.threshold_n_m)
        path = candidate.vertical_path
        out.append({"index": index, "ident": candidate.ident, "thresholdEM": round(candidate.threshold_e_m, 1),
                    "thresholdNM": round(candidate.threshold_n_m, 1), "latDeg": round(float(lat), 7),
                    "lonDeg": round(float(lon), 7), "courseDeg": round(candidate.course_deg, 3),
                    "elevationM": round(candidate.elevation_m, 2), "lengthM": round(candidate.length_m, 1),
                    "haeMinusMslM": hae_minus_msl_m[candidate.ident],
                    "verticalPath": {"crossingHeightM": path.crossing_height_m, "glidepathDeg": path.glidepath_deg,
                                     "decisionHeightM": path.decision_height_m}})
    return out


#: The data trees every checkout links to the live data (outline §5 rule 1), by their place in a checkout.
LINKED_TREES = ("4dTrajectory/outputs", "aeroviz-4d/public/data/airports")


def this_checkout(recorded: str | Path, root: Path = REPO_ROOT) -> Path:
    """A path a runner recorded — perhaps in another checkout — as this checkout reads it. The linked data trees
    (`LINKED_TREES`) of the main checkout and of every worktree (``<main>/.claude/worktrees/<name>``) are links to the
    live data (outline §5 rule 1), so a path under one of them, in whichever checkout recorded it, is the same path under
    this checkout's tree (the name its checks and the sets it writes give it) — mapped by its place, never through the
    recording checkout's links, which go when that worktree is deleted. A relative path is the repository's; any other
    path as recorded. The main checkout is the parent of the live outputs tree's parent."""
    path = Path(os.path.normpath(recorded))
    if not path.is_absolute():
        return root / path
    main = (root / LINKED_TREES[0]).resolve().parent.parent
    if not path.is_relative_to(main):
        return path
    parts = path.relative_to(main).parts
    if parts[:2] == (".claude", "worktrees") and len(parts) > 3:
        parts = parts[3:]                                     # the same place in the worktree that recorded it
    inside = Path(*parts) if parts else Path()
    for tree in LINKED_TREES:
        if inside.is_relative_to(tree):
            return root / tree / inside.relative_to(tree)
    return path


FORMATS = {"spec": SPEC_SCHEMA, "sentences": SENTENCES_SCHEMA, "closedLoop": CLOSED_LOOP_SCHEMA,
           "candidates": CANDIDATES_SCHEMA, "executorSpec": EXECUTOR_SPEC_SCHEMA}


def sample_of(set_id: str, geometry: AirportGeometry, hae_minus_msl_m: dict[str, float], source: dict[str, Any],
              cohort: dict[str, Any], words: Words, cycle_s: float, flights: list[dict[str, Any]]) -> dict[str, Any]:
    """A set's sample: its head (formats, source, cohort, vocabulary, the airport frame, the candidates with their
    HAE − MSL) and its flights, each given its runway's HAE − MSL (the height its observed track is drawn at, and every
    flown track of it)."""
    for flight in flights:
        flight["haeMinusMslM"] = hae_minus_msl_m[flight["runway"]]
    return {"schema": files.SAMPLE_SCHEMA, "setId": set_id, "airport": geometry.code, "readingRule": READING_RULE,
            "formats": FORMATS, "source": source, "cohort": cohort, "vocabulary": vocabulary_block(words.spec, words),
            # the executor's control cycle: a replay's ``endCycle`` and a DA point's ``cycle`` count these
            "executor": {"cycleS": cycle_s},
            "airportFrame": {"code": geometry.code, "lat": geometry.frame.lat0, "lon": geometry.frame.lon0,
                             "elevationM": geometry.elevation_m},
            "candidatesSha256": files.candidates_sha256(geometry), "candidates": candidates_block(geometry, hae_minus_msl_m),
            "flights": flights}


def index_entry(set_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    """A set as its airport's index lists it."""
    return {"id": set_id, "kind": files.SET_KIND, "readingRule": READING_RULE,
            "title": f"Stage A · {READING_RULE} · closed loop at Δ 2/4/8 s · train + select",
            "file": f"{set_id}/{files.SAMPLE_FILE}", "flights": len(sample["flights"]), "formats": sample["formats"],
            "cohort": sample["cohort"], "source": sample["source"]}


def heads(flights: training_flights.SetFlights, stored: dict[int, ClosedLoopSentence], split: str,
          geometry: AirportGeometry, words: Words) -> dict[str, dict[str, Any]]:
    """Each flight's head (`flight_head`) by dataset id: its stratum its stored closed-loop sentence's (the sentence
    file's, D70) and its kind whether that sentence says a go-around."""
    drawn, out = flights.drawn, {}
    for j, (index, flight, series, reading) in enumerate(zip(drawn.indices, drawn.signals, drawn.series,
                                                             flights.readings)):
        sentence = stored[index]
        said_go_around = bool((sentence.rows.grid[:, RUNWAY] == RUNWAY_GO_AROUND).any())
        out[flight.dataset_id] = flight_head(flight, series.scenario.source["flight_key"], split,
                                             sentence.withheld.stratum, KINDS[said_go_around], drawn.groups[j], reading,
                                             geometry, observed_attitude(series), words)
    return out


def split_flights(instructions: Path, split: str, chosen: list[str], intervals: tuple[float, ...], params: Any,
                  words: Words, *, device: torch.device) -> tuple[list[dict[str, Any]], AirportGeometry]:
    """The payloads of the flights ``chosen`` of ``split`` (one airport's; any split), in that order: each one's head
    (`flight_head`, its stratum the stored sentence's at the first Δ given, D70, and its kind whether that closed-loop
    sentence says a go-around) and its closed-loop sentence at
    each Δ of ``intervals``, flown again against its stored states and outcome (`replay_payload`, D86). Shared with the
    Training export of the prior (`experiments/prior_training_export.py`)."""
    spec = words.spec
    if not intervals or not chosen:
        raise ValueError(f"{split}: split_flights needs at least one row interval and one flight")
    flights = training_flights.open_flights(instructions, split, chosen, words)
    drawn = flights.drawn
    airports = sorted({flight.airport for flight in drawn.signals})
    if len(airports) != 1:
        raise ValueError(f"{split}: the chosen flights land at {airports}, not at one airport")
    geometry = drawn.geometries[airports[0]]
    having = [closed_loop_indices(instructions, split, interval, spec) for interval in intervals]
    missing = sorted(flight.dataset_id for index, flight in zip(drawn.indices, drawn.signals)
                     if any(index not in one for one in having))
    if missing:
        raise ValueError(f"{split}: {missing[:3]} have no closed-loop sentence at every row interval of {list(intervals)}")
    per_flight: dict[str, dict[str, Any]] = {}
    for interval in intervals:          # one Δ's sentences loaded at a time (a sentence read is a view into its file)
        stored = training_flights.stored_closed_loop(instructions, split, interval, words)
        if not per_flight:
            per_flight = heads(flights, stored, split, geometry, words)
        batch, sentences = training_flights.closed_loop_batch(instructions, split, flights, stored, interval, params,
                                                              words)
        del stored
        flown, verdicts = replay.fly_batch(batch, params, words, device=device)
        aero = batch.inputs(params.start_rule, device).aero_params.cpu().numpy()
        for j, (flight, verdict) in enumerate(zip(batch.signals, verdicts)):
            replayed = replay_payload(flown, j, verdict, batch, sentences[j], aero[j], spec, words)
            per_flight[flight.dataset_id]["closedLoop"][f"{interval:g}"] = closed_loop_payload(
                sentences[j], replayed, interval, geometry, words)
    return [per_flight[d] for d in chosen], geometry


def build_airport(airport: str, instructions: Path, params: Any, words: Words, *, per_stratum: int,
                  seed: int, device: torch.device) -> tuple[dict[str, Any], int]:
    """The airport's set payload (without its head) and how many flights it holds."""
    flights_out: list[dict[str, Any]] = []
    geometry: AirportGeometry | None = None
    for split in files.SPLITS:
        labelled = load_sentences(instructions, split, words.spec, ("signal_index",))
        strata = dict(zip(labelled["signal_index"].tolist(), labelled["stratum"].tolist()))
        chosen = choose([closed_loop_indices(instructions, split, interval, words.spec) for interval in ROW_INTERVALS_S],
                        strata, signals_flights(instructions, split), airport, per_stratum, seed)
        payloads, geometry = split_flights(instructions, split, chosen, ROW_INTERVALS_S, params, words, device=device)
        flights_out += payloads
    return {"flights": flights_out, "geometry": geometry}, len(flights_out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports")
    parser.add_argument("--airports", nargs="+", default=None, help="default: every airport of the artefact")
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    git = git_state()
    if git["dirty"]:
        parser.error("the tree has uncommitted changes; a Training set is exported from a commit")
    params, record, words = closed_loop.require_conforming_closed_loop(instructions, executor)
    spec = words.spec
    signals_record = json.loads((instructions / "signals.json").read_text(encoding="utf-8"))
    airports = args.airports or sorted({source["airport"] for source in signals_record["sources"]})
    started = time.perf_counter()
    existing = {airport: files.FILES.read_index(args.root / airport / "training", airport, args.set_id)
                for airport in airports}
    built = {}
    for airport in airports:
        payload, count = build_airport(airport, instructions, params, words, per_stratum=args.per_stratum,
                                       seed=args.seed, device=torch.device(args.device))
        geometry = payload.pop("geometry")
        hae = candidate_hae_minus_msl_m(signals_record["runway_ends_from"], geometry)
        cohort = {"splits": {split: 2 * args.per_stratum for split in files.SPLITS}, "perStratum": args.per_stratum,
                  "strata": list(STRATA), "seed": args.seed,
                  "drawnFrom": "a seeded permutation, per split and stratum, of the flights with a closed-loop sentence "
                               "at every row interval"}
        source = {"instructions": repo_relative(instructions), "executor": repo_relative(executor),
                  "specSha256": spec.sha256, "executorSpecSha256": record["sha256"], "git": git}
        sample = sample_of(args.set_id, geometry, hae, source, cohort, words, params.cycle_s, payload["flights"])
        entry = index_entry(args.set_id, sample)
        built[airport] = (entry, files.serialise(sample))
        print(f"{airport}: {count} flights, {len(built[airport][1]) / 1e6:.1f} MB, {time.perf_counter() - started:.0f}s",
              flush=True)
    for airport, (entry, _) in built.items():          # every airport writable before any is written
        files.FILES.require_writable(args.root / airport / "training", airport, entry, existing[airport])
    for airport, (entry, text) in built.items():
        out = files.FILES.write_set(args.root / airport / "training", airport, entry, text, existing[airport])
        print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
