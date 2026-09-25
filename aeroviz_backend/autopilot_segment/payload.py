"""The answer to ``POST /autopilot/segment``: the flown segment every control cycle, on the flight's own clock and axes,
how it ended, and the selected word's verdict with — for a heading word — its band as its judge read it.

Units are SI; the flown track is returned every control cycle, in the airport frame and on the globe (geometric MSL
height, and the ellipsoid height Cesium draws in: h = H + N).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from flight_scenarios.datum import geoid_undulation_m
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.frame import ALT, LAT, LON
from ts_transformer.autopilot.judge import Verdict, flown_track, read_flown, words_said
from ts_transformer.instructions import display
from ts_transformer.instructions.labeller.read import Admitted
from ts_transformer.instructions.words import COLUMNS, HEADING, Words

from aeroviz_backend.autopilot_segment.fly import FlightContext, FlownSegment
from aeroviz_backend.autopilot_segment.segment import told_words
from aeroviz_backend.autopilot_segment.verdict import HeadingFacts, selected_heading, word_verdict

#: MIRROR of `aeroviz-4d/src/data/trainingAutopilot.ts` (`TRAINING_AUTOPILOT_SCHEMA`); the reader refuses anything
#: else by name. A name changes with the payload's shape, on both sides, in one change.
SCHEMA = "aeroviz-autopilot-segment-v2"
#: The end of a segment flown to its ``stopRow``: the executor reached it. Otherwise the end is the judge's outcome
#: (`judge.OUTCOMES`, mirrored by `trainingOverlays.ts`'s `TRAINING_EXECUTOR_OUTCOMES`). MIRROR of `trainingAutopilot.ts`
#: (`TRAINING_AUTOPILOT_SEGMENT_END`).
SEGMENT_END = "segment_end"


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


def heading_facts(result: FlownSegment, judged: Admitted, spec: Any) -> HeadingFacts:
    """A selected heading word's `HeadingFacts`: the cycles the executor left it to intercept the final on its own while
    it was in force — before the cycle the next heading word was heard (the judge's own bookkeeping, `words_said`), and
    before the outcome — and the rows of the flown track its judge read (``judged``, `read_flown`)."""
    said = words_said(result.flown, 0, result.reading, spec)
    later = [cycle for word, cycle in zip(result.reading.instructions, said.cycles) if word.column == HEADING and word.row > 0]
    in_force = min(min(later, default=said.n_cycles), result.verdict.end_row)
    off = int(result.flown.modes["intercepting_off_word"][0, :in_force].sum())
    return HeadingFacts(off_word_cycles=off, judged_rows=len(judged.smoothed.track_deg))


def heading_payload(result: FlownSegment, judged: Admitted, spec: Any, shift: float) -> tuple[dict[str, Any], list[float]]:
    """A selected heading word's band over the rows its judge judged on the flown track, each row's verdict
    (`display.heading_band`, refused unless it gives back the judge's count), and that track as the judge read it — both
    in the flight's rows and on the chart's branch. A dynamics failure's too: its judge read the rows before the failed
    state (`read_flown`), which the track keeps."""
    check = selected_heading(result.verdict.words)
    track = judged.smoothed.track_deg + shift
    word = result.segment.instructions[HEADING]
    first = spec.rows_exact(spec.heading_lead_s)
    band = display.heading_band(track, 0, float(word.info["target_deg"]), first, first + check["rows"],
                                spec.heading_tolerance_deg, check)
    row = result.segment.row
    return ({"firstRow": row + band.first_row, "stopRow": row + band.stop_row,
             "targetOnTrackDeg": round(band.target_on_track_deg, 3), "bandDeg": _r(band.band_deg, 3),
             "inside": [int(value) for value in band.inside]},
            _r(track, 3))


def segment_payload(result: FlownSegment, context: FlightContext, words: Words) -> dict[str, Any]:
    spec = words.spec
    segment, verdict, flown = result.segment, result.verdict, result.flown
    track, shift = track_payload(result, context, spec.step_s)
    band = judged_track = facts = None
    if segment.column == HEADING and verdict.words is not None:
        judged = read_flown(flown, 0, verdict.outcome, verdict.end_row, context.geometry, result.signals, spec)
        facts = heading_facts(result, judged, spec)
        band, judged_track = heading_payload(result, judged, spec, shift)
    word = {**word_verdict(verdict, segment, spec, words, facts), "heading": band}
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
