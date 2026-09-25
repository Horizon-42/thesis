"""The executor flies ONE SEGMENT of a Training flight's sentence, live, for the frontend's Training view
(aeroviz-4d `docs/36-2026-09-20-training-module.zh.md` §4.7; the executor: `4dTrajectory/ts_transformer/autopilot/`).

A SEGMENT is one word of one column in force: from the step it is said (``row``) to the step the next word of its
column is said (``endRow``), or to the sentence's end — the band the sentence bar draws. The executor

- starts from the observed aircraft's state at ``row`` (the data plane's flight first seen there,
  `dataset.series_from_row`, read by `outputs.dynamics.context.rollout_context` as every rollout reads a flight);
- is told the six words in force at ``row`` as its step 0 (a sentence's step 0 is what the aircraft is already doing,
  executor design §2.4), then the sentence's words of the steps after it, each where the observed aircraft heard it
  (the spec's word clock, design §11);
- flies to where the selected word's own envelope ends (``stopRow``): the point where the next word of its column was
  said (``endRow``) — for a HEADING word a lead later, since a heading word is judged from a lead after it is said to a
  lead after the next heading word is (vocabulary design §10.1), so the next heading word is told at ``endRow`` as in the
  sentence and its first lead is flown. The flight stops before the first cycle that starts a step its word clock puts
  at or past ``stopRow`` — where a word said there would be heard, so none is. When ``stopRow`` is the sentence's end, it is
  flown on to its outcome as the replay flies a whole sentence (a landing, a failure, or the time limit: the segment's
  steps × the spec's timeout factor).

Only the selected word is judged, by the executor's own judge on what was flown (`autopilot.judge.judge`): the
segment's sentence is the one the executor was told, renumbered from the segment's first step, so each word's
envelope is drawn from where the executor was told it — the selected word's from the observed state it was said at.

NOTHING HERE IS PRECOMPUTED, AND THE EXECUTOR IS USED AS IT IS. Every request rebuilds the flight from the arrival
manifest its artefact recorded (`flights.rebuild_series` refuses a moved manifest or a rebuilt flight that differs
from the stored signals; the last few rebuilt flights are kept), re-reads it with the labeller (it must give the stored
sentence), and flies it with the executor's own stepper (`executor.Executor`, one control cycle at a time) under the one
executor spec written by this code for this artefact's vocabulary (`replay.open_executor`), driven exactly as
`executor.fly` drives it — the spec's word clock read before each cycle, a step's words heard on the cycle that starts
it (`sentence.row_at`, as `judge.words_said` reads the clock) — and stopped at the segment's stop: nothing past it is
flown. No replay record and no Training overlay is read. The answer says how long each part took and how many cycles
were flown.

Which flight: the request names a Training set (``airport``, ``setId``) and a flight of it (``flightKey``); the set's
sample (under the frontend's airports root) names the artefact it was exported from and the split it was drawn from.

Units are SI; the flown track is returned every control cycle, in the airport frame and on the globe (geometric MSL
height, and the ellipsoid height Cesium draws in: h = H + N).
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aeroviz_backend.paths import REPO_ROOT
from flight_scenarios.datum import geoid_undulation_m
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Executor, Flown
from ts_transformer.autopilot.flights import rebuild_series
from ts_transformer.autopilot.frame import ALT, LAT, LON, AirportCharts
from ts_transformer.autopilot.judge import Verdict, flown_track, judge, read_flown
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.runway_data import published_crossing_heights
from ts_transformer.autopilot.sentence import DistanceClock, Sentences, TimeClock, TrackClock, row_at
from ts_transformer.data.dataset import FlightSeries, series_from_row
from ts_transformer.experiments.instruction_training_export import KIND_READBACK
from ts_transformer.instructions import display
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.labeller.read import Reading, admit, read_flight
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.signals import ROW_FIELDS, FlightSignals
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, APPROACH, COLUMNS, HEADING, SPEED, UNCHANGED, Words,
)

#: MIRROR of `aeroviz-4d/src/data/trainingAutopilot.ts` (`TRAINING_AUTOPILOT_SCHEMA`); the reader refuses anything
#: else by name. A name changes with the payload's shape, on both sides, in one change.
SCHEMA = "aeroviz-autopilot-segment-v2"
#: MIRROR of `trainingAutopilot.ts` (`TRAINING_AUTOPILOT_STATUSES`): the selected word's verdict.
STATUSES = ("inside", "outside", "not judged", "no check")
INSIDE, OUTSIDE, NOT_JUDGED, NO_CHECK = STATUSES
#: The end of a segment flown to its ``stopRow``: the executor reached it. Otherwise the end is the judge's outcome
#: (`judge.OUTCOMES`). MIRROR of `trainingAutopilot.ts` (`TRAINING_AUTOPILOT_SEGMENT_END`).
SEGMENT_END = "segment_end"
DEFAULT_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
#: Where executor specs are written (`run_ts.py executor_spec --dir`): each is a directory holding ``spec.json``.
DEFAULT_EXECUTOR_ROOT = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "executor"
#: Flights kept rebuilt between requests (a rebuild opens the arrival manifest and the flight's track, ~1 s): the
#: segments of one flight are usually flown one after another.
FLIGHT_CACHE_SIZE = 8
DEVICE = torch.device("cpu")


# ── the pure parts: which segment, what it is told, where it ends, the selected word's verdict ────────────────────

@dataclass(frozen=True)
class Segment:
    column: int                     # index into `COLUMNS`
    row: int                        # the step its word is said at
    end_row: int                    # the step the next word of its column is said at, or the sentence's length
    stop_row: int                   # where its own envelope ends: ``end_row``, a heading word's a lead later (capped)
    to_landing: bool                # ``stop_row`` is the sentence's end: flown on to the outcome
    grid: np.ndarray                # [n, 6] the segment's sentence, renumbered from ``row``
    instructions: list[Instruction]  # its words, renumbered the same way


def segment_of(reading: Reading, column: int, row: int, lead_rows: int) -> Segment:
    """The segment of ``column``'s word said at ``row``: step 0 holds the six words in force at ``row`` (each the
    reading's own word, moved to row 0); then the reading's words of the steps before ``stop_row`` (a heading word's:
    ``end_row`` plus ``lead_rows``, the heading lead in steps); and, unless that is the sentence's end, one silent step
    at ``stop_row`` itself — the point the word clock must reach to end it. A word said at the sentence's last step has
    no step to fly and is refused."""
    grid = np.asarray(reading.words)
    rows = len(grid)
    if not 0 <= column < len(COLUMNS):
        raise ValueError(f"column {column} is not one of the six {COLUMNS}")
    if not 0 <= row < rows:
        raise ValueError(f"step {row} is not a step of this {rows}-step sentence")
    if grid[row, column] == UNCHANGED:
        raise ValueError(f"no {COLUMNS[column]} word is said at step {row}: a segment starts where its word is said")
    later = np.nonzero(grid[row + 1:, column] != UNCHANGED)[0]
    end_row = row + 1 + int(later[0]) if len(later) else rows
    stop_row = min(end_row + (lead_rows if column == HEADING else 0), rows)
    to_landing = stop_row == rows
    if to_landing and rows - row < 2:
        raise ValueError(f"the {COLUMNS[column]} word said at step {row}, the sentence's last, has no step after it to fly")

    opening = []
    for index in range(len(COLUMNS)):
        said = [word for word in reading.instructions if word.column == index and word.row <= row]
        last = max(word.row for word in said)
        (word,) = [item for item in said if item.row == last]
        if word.value != grid[last, index]:
            raise ValueError(f"{reading.dataset_id}: the {COLUMNS[index]} word at step {last} is {word.value} in the "
                             f"reading's words but {grid[last, index]} in its grid")
        opening.append(replace(word, row=0))
    after = [replace(word, row=word.row - row) for word in reading.instructions if row < word.row < stop_row]

    length = stop_row - row + (0 if to_landing else 1)
    segment = np.full((length, len(COLUMNS)), UNCHANGED, dtype=grid.dtype)
    segment[0] = [word.value for word in opening]
    segment[1: stop_row - row] = grid[row + 1: stop_row]
    return Segment(column=column, row=row, end_row=end_row, stop_row=stop_row, to_landing=to_landing, grid=segment,
                   instructions=opening + after)


def told_words(segment: Segment) -> list[dict[str, int]]:
    """The words the executor was told, at the flight's steps, by step then column: the six in force at the segment's
    first step, then the sentence's words to its stop — as the sentence bar shows them (`trainingAutopilot.segmentWords`)."""
    return [{"row": segment.row + word.row, "column": word.column, "value": int(word.value)}
            for word in sorted(segment.instructions, key=lambda word: (word.row, word.column))]


def segment_reading(reading: Reading, segment: Segment) -> Reading:
    """The reading the executor flies and its judge reads: the segment's words; the flight's clearance, capture and
    "unspecified" rows renumbered the same way (negative: before the segment); the labeller's checks are the observed
    flight's, not the segment's, and are left out."""
    return Reading(dataset_id=reading.dataset_id, airport=reading.airport, runway_index=reading.runway_index,
                   words=segment.grid, instructions=segment.instructions,
                   capture_row=reading.capture_row - segment.row, join_row=reading.join_row - segment.row,
                   unspecified_row=reading.unspecified_row - segment.row,
                   cut_at_crossing=reading.cut_at_crossing and segment.to_landing, checks={})


def segment_signals(signals: FlightSignals, segment: Segment) -> FlightSignals:
    """The observed rows the segment's word clock reads: from its first step, one per step of its sentence."""
    stop = segment.row + len(segment.grid)
    return replace(signals, **{name: getattr(signals, name)[segment.row: stop] for name in ROW_FIELDS})


def fly_until(executor: Executor, sentences: Sentences, clock: TimeClock | DistanceClock | TrackClock, step_s: float,
              stop_steps: int | None) -> bool | None:
    """Drive ``executor`` one cycle at a time exactly as `executor.fly` does — the word clock read before each cycle,
    a step's words heard on the cycle that starts it — and stop before the first cycle that starts a step the clock puts
    at or past ``stop_steps`` (where a word said there would be heard). True when it got there; False when the flight was
    done or out of cycles first; None without a stop (flown to its outcome)."""
    for cycle in range(executor.cycles):
        sentence_s = clock.now(cycle, executor.now())
        if cycle % executor.step_rows == 0:
            if stop_steps is not None and int(row_at(sentence_s, step_s)[0]) >= stop_steps:
                return True
            step_start_s = sentence_s
        executor.cycle(sentences.at(step_start_s), sentence_s)
        if bool(executor.done.all()):
            break
    return None if stop_steps is None else False


def _check(name: str, ok: bool, inside: int | None = None, rows: int | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "inside": inside, "rows": rows}


def _settled(checks: list[dict[str, Any]], reason: str | None = None) -> dict[str, Any]:
    """A word judged by its checks: inside exactly when every one passed."""
    return {"status": INSIDE if all(check["ok"] for check in checks) else OUTSIDE, "checks": checks,
            "reason": reason, "heading": None}


def _not_judged(reason: str) -> dict[str, Any]:
    return {"status": NOT_JUDGED, "checks": [], "reason": reason, "heading": None}


def _no_check(reason: str) -> dict[str, Any]:
    return {"status": NO_CHECK, "checks": [], "reason": reason, "heading": None}


def selected_heading(judged: dict[str, Any]) -> dict[str, int]:
    """The judge's result for the selected heading word: the one told at the segment's step 0 (the next heading word,
    told at its ``end_row``, is judged on the rows the segment flies past it — none)."""
    (result,) = [item for item in judged["heading"] if item["row"] == 0]
    return result


def word_verdict(verdict: Verdict, segment: Segment, spec: Any, words: Words) -> dict[str, Any]:
    """The selected word's verdict — the word at the segment's step 0 in its column — read off the judge's verdict of
    the segment (`judge.judge`). The heading band's rows are the segment's (the caller draws them)."""
    column = segment.column
    word = segment.instructions[column]
    if verdict.words is None:
        return _not_judged(f"the flown segment did not pass the labeller's gate ({verdict.refused})")
    judged = verdict.words
    if column == HEADING:
        result = selected_heading(judged)
        if result["rows"] == 0:
            return _not_judged(f"no row to judge: its rows begin {spec.heading_lead_s:g} s after it was told, at or past "
                               "the clearance the executor was told or its capture")
        checks = [_check(f"track within ±{spec.heading_tolerance_deg:g}° of the word, {spec.heading_lead_s:g} s after it "
                         "was told, to the next heading word's", result["inside"] == result["rows"], result["inside"],
                         result["rows"])]
        if judged["intercepting_off_word_cycles"]:
            checks.append(_check(f"held until the capture — left for {judged['intercepting_off_word_cycles']} cycles to "
                                 "intercept the final on its own", False))
        return _settled(checks)
    if column == APPROACH:
        if word.kind != "clear":
            return _no_check("not cleared, or a go-around: no envelope of its own (a clearance is judged by its capture "
                             "turn and corridor)")
        corridor, capture = judged["corridor"], judged["capture_turn"]
        checks = ([_check("captured the final", False)] if capture is None else
                  [_check("capture turn monotone", capture["progress_ok"]),
                   _check("capture turn rate and bank", capture["rate_ok"])])
        checks += [_check("corridor entered", corridor["entered"]),
                   _check("corridor held", corridor["inside"] == corridor["rows"], corridor["inside"], corridor["rows"])]
        return _settled(checks)
    if column == ALTITUDE:
        (tube,) = [item for item in judged["vertical"] if item["row"] == 0]
        return _settled([_check("in its tube", tube["contained"], tube["inside"], tube["rows"])])
    if column == ANGLE:
        # an angle word re-anchors the tube of every altitude word it is in force over (the one in force at its step,
        # told with it at the segment's step 0, and any said before the segment's end): judged in each
        def named(target_m: float | None) -> str:
            return "descend to land" if target_m is None else f"{target_m:.0f} m"

        checks = [_check(f"in the tube of the altitude word {named(tube['target_m'])}", tube["contained"], tube["inside"],
                         tube["rows"]) for tube in judged["vertical"]]
        return _settled(checks, "judged in the altitude tubes it anchors")
    if column == SPEED:
        if words.speed_mps(word.value) is None:
            return _no_check("the pilot's own speed: no band to hold")
        (span,) = [item for item in judged["speed"] if item["row"] == 0]
        checks = [_check("transition monotone toward the target", span["transition_ok"]),
                  _check("band held" if not span["cut_before_arrival"] else "band not reached before the segment's end",
                         span["band_inside"] == span["band_rows"], span["band_inside"], span["band_rows"])]
        # the judge's verdict is the transition and the band (`contained`); its acceleration check is not part of it
        result = _settled(checks)
        if (result["status"] == INSIDE) != bool(span["contained"]):
            raise ValueError(f"the speed word's checks say {result['status']}, the judge says contained={span['contained']}")
        return result
    return _no_check("the runway pointer: judged by the landing, the flight's outcome")


# ── the flight: its inputs, rebuilt; the segment flown and cut ────────────────────────────────────────────────────

@dataclass(frozen=True)
class FlightContext:
    signals: FlightSignals
    series: FlightSeries
    reading: Reading
    geometry: AirportGeometry
    crossing_heights: tuple[float, ...]
    group: str
    approach_ias_mps: float
    observed_track_deg: np.ndarray     # the labeller's smoothed track of the observed flight (the heading chart's)
    observed_distance_m: np.ndarray    # and its smoothed distance flown (the altitude chart's axis)


def open_flight(artefact: Path, split: str, dataset_id: str, words: Words) -> FlightContext:
    """One labelled flight of ``artefact``: its stored signals, the flight rebuilt from the data plane and checked
    against them (`rebuild_series`), and the labeller's reading, which must give the stored sentence."""
    spec = words.spec
    signals = load_signals(artefact, split)
    found = [index for index, item in enumerate(signals) if item.dataset_id == dataset_id]
    if len(found) != 1:
        raise ValueError(f"{dataset_id} is not a labelled {split} flight of {artefact.name}")
    (index,) = found
    flight = signals[index]
    sentences = load_sentences(artefact, split, spec)
    stored = {int(value): k for k, value in enumerate(sentences["signal_index"])}
    if index not in stored:
        raise ValueError(f"{dataset_id} has no stored sentence in {artefact.name}")
    k = stored[index]
    geometry = load_candidates(artefact)[flight.airport]
    reading = read_flight(flight, geometry, spec, words)
    grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]
    if not np.array_equal(reading.words, grid) or reading.runway_index != int(sentences["runway_index"][k]):
        raise ValueError(f"{dataset_id}: the re-read sentence differs from the stored one")
    (series,) = rebuild_series(artefact, [flight])
    group = replay.group_of(series)
    if group not in (replay.OWN, replay.STAND_IN):
        raise ValueError(f"{dataset_id} cannot be flown: {group}")
    observed = admit(flight, geometry, spec)
    return FlightContext(signals=flight, series=series, reading=reading, geometry=geometry,
                         crossing_heights=published_crossing_heights(geometry), group=group,
                         approach_ias_mps=replay.flight_approach_ias_mps(series, group),
                         observed_track_deg=observed.smoothed.track_deg,
                         observed_distance_m=observed.smoothed.distance_m)


@dataclass(frozen=True)
class FlownSegment:
    segment: Segment
    reading: Reading                 # the segment's
    signals: FlightSignals           # the observed rows its clock read
    flown: Flown                     # to its stop, or its outcome
    verdict: Verdict
    reached_end: bool | None         # None when flown to its outcome (``stop_row`` is the sentence's end)
    fly_s: float                     # wall time the executor took, and the judge
    judge_s: float


def fly_segment(context: FlightContext, params: ExecutorParams, words: Words, column: int, row: int) -> FlownSegment:
    """``column``'s word said at ``row`` flown from the observed state there (module docstring), and judged."""
    spec, f64 = words.spec, torch.float64
    segment = segment_of(context.reading, column, row, spec.rows_exact(spec.heading_lead_s))
    reading = segment_reading(context.reading, segment)
    signals = segment_signals(context.signals, segment)
    batch = replay.Batch(signals=[signals], series=[series_from_row(context.series, row)], readings=[reading],
                         geometries=[context.geometry], crossing_heights=[context.crossing_heights],
                         approach_ias_mps=[context.approach_ias_mps], groups=[context.group], drawn={})
    started = time.perf_counter()
    # MIRROR of `replay.fly_sentences`' setup (its time limit, runways, charts and approach speeds), with the executor
    # stepped here (`fly_until`) in place of `executor.fly`: pinned against it by `test_autopilot_segment.StepperTest`
    executor = Executor(batch.inputs(DEVICE),
                        Runways.of(batch.geometries, batch.crossing_heights, dtype=f64, device=DEVICE),
                        AirportCharts.of(batch.geometries, dtype=f64, device=DEVICE),
                        torch.tensor(batch.approach_ias_mps, dtype=f64, device=DEVICE), params, words,
                        time_limit_s=torch.tensor([len(reading.words) * spec.step_s * params.timeout_factor], dtype=f64,
                                                  device=DEVICE))
    reached = fly_until(executor, Sentences([reading.words], words, device=DEVICE),
                        replay.word_clock(batch, params, spec.step_s, DEVICE), spec.step_s,
                        None if segment.to_landing else segment.stop_row - segment.row)
    flown = executor.flown()
    if reached:
        # stopped by the segment, not done by the executor: it ends with the last cycle flown
        flown = replace(flown, done_cycle=torch.full_like(flown.done_cycle, executor.count - 1))
    flown_at = time.perf_counter()
    verdict = judge(flown, 0, context.geometry, reading.runway_index, reading, signals, spec, words)
    return FlownSegment(segment=segment, reading=reading, signals=signals, flown=flown, verdict=verdict,
                        reached_end=reached, fly_s=flown_at - started, judge_s=time.perf_counter() - flown_at)


# ── the payload ────────────────────────────────────────────────────────────────────────────────────────────────────

def _r(values: Any, digits: int) -> list[float]:
    return [round(float(value), digits) for value in np.asarray(values, dtype=np.float64).ravel()]


def end_state_row(verdict: Verdict) -> int:
    """The last state row drawn: the outcome's (a dynamics failure's failed state is left out, as the export does)."""
    return verdict.end_row - 1 if verdict.outcome == "dynamics_failure" else verdict.end_row


def track_payload(result: FlownSegment, context: FlightContext, step_s: float) -> tuple[dict[str, Any], float]:
    """The flown segment every control cycle, on the flight's own clock and axes: its time from the flight's first row,
    its distance flown from the observed flight's at the segment's first step, and its track moved by whole turns onto
    the observed smoothed track's branch there, so each reads on the chart beside the observed one. Returned with that
    shift, in degrees."""
    flown, row = result.flown, result.segment.row
    last = end_state_row(result.verdict)
    states = flown.states[0, : last + 1].cpu().numpy()
    track = flown_track(states, context.geometry)
    lat, lon, height = states[:, LAT], states[:, LON], states[:, ALT]
    undulation = np.asarray(geoid_undulation_m(list(lat), list(lon)), dtype=np.float64)
    distance = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(track["e"]), np.diff(track["n"])))))
    shift = 360.0 * round((float(context.observed_track_deg[row]) - float(track["track"][0])) / 360.0)
    commands = flown.commands[0, :last].cpu().numpy()
    return {
        "tS": _r(row * step_s + np.arange(len(states)) * flown.cycle_s, 3),
        "eM": _r(track["e"], 1), "nM": _r(track["n"], 1), "lon": _r(lon, 7), "lat": _r(lat, 7),
        "altitudeM": _r(height, 2), "altitudeHaeM": _r(height + undulation, 2),
        "groundSpeedMps": _r(track["ground_speed"], 3), "verticalRateMps": _r(track["vertical_rate"], 3),
        "trackDeg": _r(track["track"] + shift, 3),
        "distanceM": _r(float(context.observed_distance_m[row]) + distance, 1),
        # the commands each cycle flew (one fewer than the states): the dynamics' bank turns left when positive
        "thrustFraction": _r(commands[:, 0], 4), "bankRightDeg": _r(-np.degrees(commands[:, 1]), 3),
        "loadFactor": _r(commands[:, 2], 4),
    }, shift


def heading_payload(result: FlownSegment, context: FlightContext, spec: Any,
                    shift: float) -> tuple[dict[str, Any] | None, list[float] | None]:
    """For a heading word the judge judged: its band over the rows it judged on the flown track, each row's verdict
    (`display.heading_band`, refused unless it gives back the judge's count), and that track as the judge read it —
    both in the flight's rows and on the chart's branch. None otherwise (and for a dynamics failure, whose judge read
    the failed state)."""
    verdict = result.verdict
    if result.segment.column != HEADING or verdict.words is None or verdict.outcome == "dynamics_failure":
        return None, None
    check = selected_heading(verdict.words)
    flight = read_flown(result.flown, 0, verdict.outcome, verdict.end_row, context.geometry, result.signals, spec)
    track = flight.smoothed.track_deg + shift
    word = result.segment.instructions[HEADING]
    first = spec.rows_exact(spec.heading_lead_s)
    band = display.heading_band(track, 0, float(word.info["target_deg"]), first, first + check["rows"],
                                spec.heading_tolerance_deg, check)
    row = result.segment.row
    return ({"firstRow": row + band.first_row, "stopRow": row + band.stop_row,
             "targetOnTrackDeg": round(band.target_on_track_deg, 3), "bandDeg": _r(band.band_deg, 3),
             "inside": [int(value) for value in band.inside]},
            _r(track, 3))


def segment_payload(result: FlownSegment, context: FlightContext, params: ExecutorParams, words: Words) -> dict[str, Any]:
    spec = words.spec
    segment, verdict, flown = result.segment, result.verdict, result.flown
    track, shift = track_payload(result, context, spec.step_s)
    band, judged_track = heading_payload(result, context, spec, shift)
    word = {**word_verdict(verdict, segment, spec, words), "heading": band}
    if segment.to_landing:
        observed_s = replay.observed_landing_s(context.signals, context.reading, context.geometry) - segment.row * spec.step_s
    else:
        observed_s = (segment.stop_row - segment.row) * spec.step_s
    crossing = verdict.crossing
    # the judge's outcome, unless the flight simply reached the segment's end (no event before it)
    reason = SEGMENT_END if result.reached_end and verdict.outcome == "timeout" else verdict.outcome
    offset = None
    if reason == SEGMENT_END:
        # where the executor was when its clock put it at the observed aircraft's point of the segment's stop
        observed, last = context.signals, segment.stop_row
        flown_end = flown_track(flown.states[0, -1:].cpu().numpy(), context.geometry)
        offset = {"horizontalM": round(float(math.hypot(flown_end["e"][0] - observed.e_m[last],
                                                          flown_end["n"][0] - observed.n_m[last])), 1),
                  "aboveM": round(float(flown_end["height"][0] - observed.altitude_m[last]), 1),
                  "groundSpeedMps": round(float(flown_end["ground_speed"][0] - observed.ground_speed_mps[last]), 2)}
    return {
        "segment": {
            "column": COLUMNS[segment.column], "row": segment.row, "endRow": segment.end_row,
            "stopRow": segment.stop_row, "toLanding": segment.to_landing, "observedS": round(observed_s, 3),
            "told": told_words(segment),
        },
        "end": {
            "reason": reason,
            "reachedSegmentEnd": result.reached_end,
            # the flown state minus the observed one at the step the segment ends at: only when it ended there
            "offsetFromObserved": offset,
            "flownS": round(verdict.end_row * flown.cycle_s, 3),
            "crossing": None if crossing is None else {"crossM": round(crossing["cross_m"], 2),
                                                       "heightM": round(crossing["height_m"], 2),
                                                       "atS": round(segment.row * spec.step_s
                                                                    + crossing["at_row"] * flown.cycle_s, 3)},
            "refused": verdict.refused,
        },
        "word": word,
        "limits": {"cycles": verdict.limits["cycles"]["cycles"],
                   "bound": {name: value["cycles"] for name, value in verdict.limits.items() if name != "cycles"}},
        "track": track,
        "judgedTrackDeg": judged_track,
    }


# ── the backend: which set, which spec, the cache, one flight at a time ────────────────────────────────────────────

def _field(record: dict[str, Any], name: str, where: str) -> Any:
    if name not in record:
        raise ValueError(f"{where} has no {name!r}")
    return record[name]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class AutopilotSegmentBackend:
    """``fly(payload)`` for ``POST /autopilot/segment``: ``{airport, setId, flightKey, column, row}``."""

    def __init__(self, *, airports_root: Path = DEFAULT_AIRPORTS_ROOT,
                 executor_root: Path = DEFAULT_EXECUTOR_ROOT) -> None:
        self.airports_root = Path(airports_root)
        self.executor_root = Path(executor_root)
        self._lock = threading.Lock()
        self._executors: dict[Path, tuple[Path, ExecutorParams, dict[str, Any], Words]] = {}
        self._flights: OrderedDict[tuple[Path, str], FlightContext] = OrderedDict()
        self._samples: dict[Path, tuple[int, dict[str, Any]]] = {}

    def _json(self, path: Path) -> dict[str, Any]:
        """A Training file, parsed once per version on disk (a sample is several MB)."""
        version = path.stat().st_mtime_ns
        if path not in self._samples or self._samples[path][0] != version:
            self._samples[path] = (version, json.loads(path.read_text(encoding="utf-8")))
        return self._samples[path][1]

    def training_set(self, airport: str, set_id: str) -> tuple[Path, str, dict[str, Any]]:
        """The set's artefact (absolute), the split its flights were drawn from, and its sample."""
        training = self.airports_root / airport / "training"
        index = self._json(training / "index.json")
        entries = [entry for entry in _field(index, "sets", "the Training index") if entry["id"] == set_id]
        if len(entries) != 1:
            raise FileNotFoundError(f"{airport} lists no Training set {set_id!r}")
        (entry,) = entries
        if entry["kind"] != KIND_READBACK:
            raise ValueError(f"Training set {set_id} is a {entry['kind']} set: only a {KIND_READBACK} set's flights are "
                             "labelled flights of an instruction artefact")
        sample = self._json(training / entry["file"])
        where = f"{set_id}'s sample"
        artefact = REPO_ROOT / _field(_field(sample, "producedBy", where), "artefact", where)
        split = _field(_field(sample, "cohort", where), "split", where)
        return artefact, split, sample

    def executor_for(self, artefact: Path) -> tuple[Path, ExecutorParams, dict[str, Any], Words]:
        """The one executor spec written by this code for ``artefact``'s vocabulary (`replay.open_executor`); refused,
        naming every spec and why, when there is none or more than one."""
        if artefact not in self._executors:
            usable, refused = [], []
            for directory in sorted(path.parent for path in self.executor_root.glob("*/spec.json")):
                try:
                    usable.append((directory, *replay.open_executor(directory, artefact)))
                except ValueError as error:
                    refused.append(f"{directory.name}: {error}")
            if len(usable) != 1:
                named = [directory.name for directory, *_ in usable]
                raise ValueError(f"{len(usable)} executor specs under {self.executor_root} fly {artefact.name} with this "
                                 f"code ({named}); one is needed. Refused: {refused}")
            self._executors[artefact] = usable[0]
        return self._executors[artefact]

    def flight(self, artefact: Path, split: str, dataset_id: str, words: Words) -> tuple[FlightContext, bool]:
        """The flight rebuilt, and whether it was kept from an earlier request."""
        key = (artefact, dataset_id)
        if key in self._flights:
            self._flights.move_to_end(key)
            return self._flights[key], True
        context = open_flight(artefact, split, dataset_id, words)
        self._flights[key] = context
        while len(self._flights) > FLIGHT_CACHE_SIZE:
            self._flights.popitem(last=False)
        return context, False

    def fly(self, payload: dict[str, Any]) -> dict[str, Any]:
        airport = str(_field(payload, "airport", "the request"))
        set_id = str(_field(payload, "setId", "the request"))
        flight_key = str(_field(payload, "flightKey", "the request"))
        column_name = _field(payload, "column", "the request")
        row = _field(payload, "row", "the request")
        if column_name not in COLUMNS:
            raise ValueError(f"column {column_name!r} is none of {COLUMNS}")
        if not isinstance(row, int) or isinstance(row, bool):
            raise ValueError(f"row must be an integer step, got {row!r}")
        asked = time.perf_counter()
        with self._lock:
            started = time.perf_counter()
            artefact, split, sample = self.training_set(airport, set_id)
            flights = [item for item in sample["flights"] if item["flightKey"] == flight_key]
            if len(flights) != 1:
                raise FileNotFoundError(f"Training set {set_id} at {airport} has no flight {flight_key}")
            dataset_id = flights[0]["datasetId"]
            directory, params, record, words = self.executor_for(artefact)
            opening = time.perf_counter()
            context, kept = self.flight(artefact, split, dataset_id, words)
            opened = time.perf_counter()
            result = fly_segment(context, params, words, COLUMNS.index(column_name), row)
            answering = time.perf_counter()
            body = segment_payload(result, context, params, words)
            finished = time.perf_counter()
        return {
            "ok": True, "schema": SCHEMA, "airport": airport, "setId": set_id, "flightKey": flight_key,
            "datasetId": dataset_id, "computedUtc": _utc_now(),
            # wall-clock seconds on the backend: waiting for the flight before it (one at a time); then, adding up to
            # ``computeS``: the set and the executor spec found, the flight rebuilt (or kept), the segment set up, the
            # executor's cycles (``cycles`` of them: what was computed, which may run past the judged outcome), the judge,
            # and the answer written
            "timing": {"waitS": round(started - asked, 3), "setupS": round(opening - started, 3),
                       "openS": round(opened - opening, 3), "flightKept": kept,
                       "prepareS": round(answering - opened - result.fly_s - result.judge_s, 3),
                       "flyS": round(result.fly_s, 3), "cycles": int(result.flown.commands.shape[1]),
                       "judgeS": round(result.judge_s, 3), "answerS": round(finished - answering, 3),
                       "computeS": round(finished - started, 3)},
            "executor": {"spec": directory.relative_to(REPO_ROOT).as_posix() if directory.is_relative_to(REPO_ROOT)
                         else directory.as_posix(),
                         "specSha256": record["sha256"], "sourceSha256": record["source"]["executor_source_sha256"],
                         "wordClock": params.word_clock, "cycleS": params.cycle_s,
                         "timeoutFactor": params.timeout_factor},
            "artefact": artefact.relative_to(REPO_ROOT).as_posix() if artefact.is_relative_to(REPO_ROOT)
            else artefact.as_posix(),
            "vocabularySpecSha256": words.spec.sha256,
            "group": context.group,
            **body,
        }
