"""The answer to ``POST /autopilot/segment``: the flown segment every control cycle, on the flight's own clock and axes,
how it ended, and the selected word's verdict with — for a heading word — its band as its judge read it.

The segment is the flight FROM THE SELECTED WORD ON: for the truth's word that is all that was flown; for a model's,
whose sentence was flown again from its first step (`segment.model_segment`), the part before the word — how its
aircraft got there — is flown and judged but not returned (``source`` says which sentence it was).

Units are SI; the flown track is returned every control cycle, in the airport frame and on the globe (geometric MSL
height, and the ellipsoid height Cesium draws in: plus the flight's runway's HAE − MSL offset, the one its observed
track was drawn with — `training_files.runway_hae_minus_msl_m`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.frame import ALT, LAT, LON
from ts_transformer.autopilot.judge import Verdict, flown_track, read_flown, words_said
from ts_transformer.instructions import display
from ts_transformer.instructions.labeller.read import Admitted
from ts_transformer.instructions.training_files import band_payload, rounded
from ts_transformer.instructions.words import COLUMNS, HEADING, Words

from aeroviz_backend.autopilot_segment.fly import FlightContext, FlownSegment
from aeroviz_backend.autopilot_segment.segment import told_words
from aeroviz_backend.autopilot_segment.verdict import HeadingFacts, selected_heading, word_verdict

#: MIRROR of `aeroviz-4d/src/data/trainingAutopilot.ts` (`TRAINING_AUTOPILOT_SCHEMA`); the reader refuses anything
#: else by name. A name changes with the payload's shape, on both sides, in one change.
SCHEMA = "aeroviz-autopilot-segment-v3"
#: The end of a segment flown to its ``stopRow``: the executor reached it. Otherwise the end is the judge's outcome
#: (`judge.OUTCOMES`, mirrored by `trainingOverlays.ts`'s `TRAINING_EXECUTOR_OUTCOMES`). MIRROR of `trainingAutopilot.ts`
#: (`TRAINING_AUTOPILOT_SEGMENT_END`).
SEGMENT_END = "segment_end"


def end_state_row(verdict: Verdict) -> int:
    """The last state row drawn: the outcome's (a dynamics failure's failed state is left out, as the export does)."""
    return verdict.end_row - 1 if verdict.outcome == "dynamics_failure" else verdict.end_row


def word_cycle(result: FlownSegment, step_s: float) -> int:
    """The flown state the selected word's segment starts at: its step in the flown sentence, in cycles (0 for the
    truth's)."""
    return result.segment.word_step * int(round(step_s / result.flown.cycle_s))


def track_payload(result: FlownSegment, context: FlightContext, step_s: float) -> tuple[dict[str, Any], float]:
    """The flown segment every control cycle, from the selected word on, on the flight's own clock and axes: its time
    from the flight's first row, its distance flown from the observed flight's at the step the executor started at, and
    its track moved by whole turns onto the observed smoothed track's branch there, so each reads on the chart beside the
    observed one. Returned with that shift, in degrees."""
    flown, start = result.flown, result.segment.start_row
    first, last = word_cycle(result, step_s), end_state_row(result.verdict)
    states = flown.states[0, : last + 1].cpu().numpy()
    track = flown_track(states, context.geometry)
    lat, lon, height = states[:, LAT], states[:, LON], states[:, ALT]
    distance = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(track["e"]), np.diff(track["n"])))))
    shift = 360.0 * round((float(context.observed_track_deg[start]) - float(track["track"][0])) / 360.0)
    commands = flown.commands[0, :last].cpu().numpy()
    shown = slice(first, None)
    return {
        "tS": rounded(result.segment.row * step_s + np.arange(len(states) - first) * flown.cycle_s, 3),
        "eM": rounded(track["e"][shown], 1), "nM": rounded(track["n"][shown], 1), "lon": rounded(lon[shown], 7),
        "lat": rounded(lat[shown], 7), "altitudeM": rounded(height[shown], 2),
        "altitudeHaeM": rounded(height[shown] + context.hae_minus_msl_m, 2),
        "groundSpeedMps": rounded(track["ground_speed"][shown], 3), "verticalRateMps": rounded(track["vertical_rate"][shown], 3),
        "trackDeg": rounded((track["track"] + shift)[shown], 3),
        "distanceM": rounded((float(context.observed_distance_m[start]) + distance)[shown], 1),
        # the commands each cycle flew (one fewer than the states): the dynamics' bank turns left when positive
        "thrustFraction": rounded(commands[shown, 0], 4), "bankRightDeg": rounded(-np.degrees(commands[shown, 1]), 3),
        "loadFactor": rounded(commands[shown, 2], 4),
    }, shift


def heading_facts(result: FlownSegment, judged: Admitted, spec: Any) -> HeadingFacts:
    """A selected heading word's `HeadingFacts`: the cycles the executor left it to intercept the final on its own while
    it was in force — from the cycle it was heard to the cycle the next heading word was (the judge's own bookkeeping,
    `words_said`), and before the outcome — and the rows of the flown track its judge read (``judged``, `read_flown`)."""
    said = words_said(result.flown, 0, result.reading, spec)
    step = result.segment.word_step
    heard = {(word.column, word.row): cycle for word, cycle in zip(result.reading.instructions, said.cycles)}
    later = [cycle for (column, row), cycle in heard.items() if column == HEADING and row > step]
    in_force = min(min(later, default=said.n_cycles), result.verdict.end_row)
    off = int(result.flown.modes["intercepting_off_word"][0, heard[(HEADING, step)]:in_force].sum())
    return HeadingFacts(off_word_cycles=off, judged_rows=len(judged.smoothed.track_deg))


def ended_at_stop(result: FlownSegment) -> bool:
    """The segment's stop ended the flight: it got there, and no event (a crossing, a dynamics failure, ground contact)
    ended it first — the executor flies on past an uncaptured crossing, so getting there is not enough."""
    return result.reached_end is True and result.verdict.outcome == "timeout"


def band_cut_by_stop(result: FlownSegment, judged_rows: int, spec: Any) -> bool:
    """The selected heading word's band was cut by the segment's stop. The stop is counted on the SENTENCE's steps
    (`fly_until`: the word clock at the next heading word's step plus the lead), the band on FLOWN steps (the judge: to a
    lead after the executor heard the next heading word). With the time and track clocks they agree — those move at most
    one sentence step a cycle, a step is two cycles and the lead two steps — but the distance clock has no such cap: it
    can jump past the next heading word to the stop in one cycle (the word never heard), or hear it on a step's first
    cycle and reach the stop on the next. Then the band runs to the track's end (``judged_rows``) short of where it
    should end. Only a flight the stop ended can be cut: one an event ended is judged to the event."""
    if not ended_at_stop(result):
        return False
    lead, step = spec.rows_exact(spec.heading_lead_s), result.segment.word_step
    band_stop = step + lead + selected_heading(result.verdict.words, step)["rows"]
    said = words_said(result.flown, 0, result.reading, spec)
    step_rows = int(round(spec.step_s / result.flown.cycle_s))
    heard = [cycle for word, cycle in zip(result.reading.instructions, said.cycles)
             if word.column == HEADING and word.row > step and cycle < said.n_cycles]
    needs = min(heard) // step_rows + lead if heard else None
    return band_stop == judged_rows and (needs is None or needs > judged_rows)


def heading_payload(result: FlownSegment, judged: Admitted, spec: Any, shift: float) -> tuple[dict[str, Any], list[float]]:
    """A selected heading word's band over the rows its judge judged on the flown track, each row's verdict
    (`display.heading_band`, refused unless it gives back the judge's count), and that track as the judge read it — both
    in the flight's rows and on the chart's branch — the track from the selected word's step on, as the segment is. A
    dynamics failure's too: its judge read the rows before the failed state (`read_flown`), which the track keeps."""
    step = result.segment.word_step
    check = selected_heading(result.verdict.words, step)
    track = judged.smoothed.track_deg + shift
    first = step + spec.rows_exact(spec.heading_lead_s)
    band = display.heading_band(track, step, float(result.segment.selected().info["target_deg"]), first,
                                first + check["rows"], spec.heading_tolerance_deg, check)
    start = result.segment.start_row
    # the exporters' band, its rows moved from the flown sentence's to the flight's
    return ({**band_payload(band), "firstRow": start + band.first_row, "stopRow": start + band.stop_row},
            rounded(track[step:], 3))


def segment_payload(result: FlownSegment, context: FlightContext, words: Words) -> dict[str, Any]:
    spec = words.spec
    segment, verdict, flown = result.segment, result.verdict, result.flown
    track, shift = track_payload(result, context, spec.step_s)
    band = judged_track = facts = None
    if segment.column == HEADING and verdict.words is not None:
        judged = read_flown(flown, 0, verdict.outcome, verdict.end_row, context.geometry, result.signals, spec)
        if band_cut_by_stop(result, len(judged.smoothed.track_deg), spec):
            raise ValueError(f"the heading word said at step {segment.row} was stopped at step {segment.stop_row} before its "
                             "band ended: the segment's stop is counted on the sentence's steps and the band on the flown "
                             "ones, and this executor spec lets them part — the stop needs to move to the flown axis")
        facts = heading_facts(result, judged, spec)
        band, judged_track = heading_payload(result, judged, spec, shift)
    word = {**word_verdict(verdict, segment, spec, words, facts), "heading": band}
    model = result.model
    # the observed aircraft's time over the same steps: a model's segment has no observed counterpart
    if model is not None:
        observed_s = None
    elif segment.to_landing:
        observed_s = replay.observed_landing_s(context.signals, context.reading, context.geometry) - segment.row * spec.step_s
    else:
        observed_s = (segment.stop_row - segment.row) * spec.step_s
    crossing = verdict.crossing
    # the judge's outcome, unless the flight simply reached the segment's end (no event before it)
    reason = SEGMENT_END if ended_at_stop(result) else verdict.outcome
    offset = None
    if reason == SEGMENT_END and model is None:
        # where the executor was when its clock put it at the observed aircraft's point of the segment's stop
        observed, last = context.signals, segment.stop_row
        flown_end = flown_track(flown.states[0, -1:].cpu().numpy(), context.geometry)
        offset = {"horizontalM": round(float(math.hypot(flown_end["e"][0] - observed.e_m[last],
                                                          flown_end["n"][0] - observed.n_m[last])), 1),
                  "aboveM": round(float(flown_end["height"][0] - observed.altitude_m[last]), 1),
                  "groundSpeedMps": round(float(flown_end["ground_speed"][0] - observed.ground_speed_mps[last]), 2)}
    started_s = segment.start_row * spec.step_s
    return {
        # which sentence was flown: the truth's, or one sample of a model's own (flown again from its first step)
        "source": ({"kind": "truth"} if model is None else
                   {"kind": "model", "overlayId": model.overlay_id, "sample": model.sample, "firstRow": model.first_row}),
        "segment": {
            "column": COLUMNS[segment.column], "row": segment.row, "endRow": segment.end_row,
            "stopRow": segment.stop_row, "toLanding": segment.to_landing, "observedS": None if observed_s is None else round(observed_s, 3),
            "told": told_words(segment),
        },
        "end": {
            "reason": reason,
            "reachedSegmentEnd": result.reached_end,
            # the flown state minus the observed one at the step the segment ends at: only the truth's, when it ended there
            "offsetFromObserved": offset,
            # from the selected word on
            "flownS": round((verdict.end_row - word_cycle(result, spec.step_s)) * flown.cycle_s, 3),
            "crossing": None if crossing is None else {"crossM": round(crossing["cross_m"], 2),
                                                       "heightM": round(crossing["height_m"], 2),
                                                       "atS": round(started_s + crossing["at_row"] * flown.cycle_s, 3)},
            "refused": verdict.refused,
        },
        "word": word,
        "limits": {"cycles": verdict.limits["cycles"]["cycles"],
                   "bound": {name: value["cycles"] for name, value in verdict.limits.items() if name != "cycles"}},
        "track": track,
        "judgedTrackDeg": judged_track,
    }
