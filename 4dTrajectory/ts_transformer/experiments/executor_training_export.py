"""The executor's replay of a Training set's flights, for the frontend: each flight's truth sentence flown from row 0
by a formal executor spec, its flown track, its outcome and every word's verdict (executor design §8, §11; the Training
module: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`).

    python run_ts.py executor_training_export \\
        --executor 4dTrajectory/outputs/POOLED/executor/<an executor spec of instruction-v3> \\
        --replay 4dTrajectory/outputs/POOLED/executor/<that spec>/replay-val \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<its instruction-v3 artefact> \\
        --airports-root aeroviz-4d/public/data/airports --set instruction_v3 \\
        --airport KMSY --airport KRDU --airport KSJC --airport KSMF --airport KSTL

Per airport, writes ``<airports-root>/<ICAO>/training/<overlay-id>/executor.json`` (schema `SCHEMA`; refused if the
directory exists) and adds the overlay to ``training/overlays.json`` (`instruction_training_export.OVERLAYS_SCHEMA`;
refused if it lists the id already). The set (``--set``) must be a read-back set of the executor's vocabulary drawn
from val, and the replay (``--replay``) the formal val replay of this spec (`executor_replay`).

**Every flight is RE-FLOWN, and must reproduce its row of the formal replay.** The replay's ``replay.json`` keeps each
flight's words as a list of verdicts without the words they belong to, so the set's flights are flown again with the
spec (`replay.open_executor`: refused unless this is the code that measured it), judged (`judge`), and each verdict is
matched to its word. The outcome, the flown-as-said flag, the verdict list — rebuilt from the words in
`replay.word_results`'s order — and the counts of words not judged, not reached and superseded must equal the formal
row, and the crossing must match it within `CROSSING_TOLERANCE`, or the export stops, naming the flight. The evaluation
verdicts (the replay's and the observed track's) are the formal row's: evaluation is not run again.

**Where it runs.** `replay.open_executor` refuses a spec measured by other executor code, by a hash over the executor's
files and the repository modules they import (`autopilot.spec.executor_source_files`), each labelled by its module
name, so the hash is the same from any checkout.

**A verdict per word.** Each word of the sentence gets one status (`STATUSES`): ``inside`` / ``outside`` its envelope
as the judge read it on the FLOWN track (a word's envelope re-drawn from where the executor was when it was said, not
the observed flight's); ``not judged`` (a heading word with no row to judge — its rows, from the step it was told plus
the lead to the next heading word's, begin at or past the clearance the executor was told or its capture, or the next
heading word was told on the same step — or a flown track the labeller's gate refuses); ``not reached`` (the flight
ended before the executor was told it, or after the flown track's last row); ``superseded`` (another word of its
column was said on the same flown step, and only the later one flew); ``no check`` (the runway pointer — judged by the
landing — an approach word other than the clearance, an angle word — it re-anchors its altitude tube — and
"unspecified" speed). The heading word the executor left to intercept the final on its own is failed with that check
added.

**What a heading word's verdict draws** (vocabulary design §10.1): the band θ ± the heading tolerance over the flown rows
the judge judged it on, and each row's verdict (`instructions.display.heading_band`: the judge's own check asked row by
row, refused unless it gives back the judge's count), with the flown track as the judge's gate read it (smoothed, to the
landing cut) — both on the chart branch the flown track is exported on, the judged track's step k the exported track's
point k. Not for a dynamics failure: its judge read the failed state (non-finite, or a stall), which the exported track
leaves out, so its words keep the judge's statuses and checks but draw no band and no judged track.

A flight the replay does not fly (no identified type, or a type publishing no approach speed) is listed with the reason
and no track. Units are SI; the flown track is exported every sentence step (2 s) to its outcome's row, with its
geometric MSL height and the ellipsoid height Cesium draws in (h = H + N).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flight_scenarios.datum import geoid_undulation_m
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.flights import rebuild_series
from ts_transformer.autopilot.frame import ALT, LAT, LON
from ts_transformer.autopilot.judge import Verdict, flown_track, read_flown, words_said
from ts_transformer.experiments.executor_replay import REPLAY_SCHEMA
from ts_transformer.experiments.instruction_training_export import (
    KIND_EXECUTOR, SPLIT, BaseSet, band_payload, base_flights, open_base_set, overlay_entry, read_overlays,
    serialise_overlay, write_overlay,
)
from ts_transformer.instructions import display
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.labeller.read import Admitted, Reading, read_flight
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, APPROACH, APPROACH_CLEARED, COLUMNS, HEADING, RUNWAY, SPEED, Words,
)
from ts_transformer.io_utils import utc_now
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: MIRROR of `aeroviz-4d/src/data/trainingOverlays.ts` (`TRAINING_EXECUTOR_SCHEMA`, `TRAINING_EXECUTOR_STATUSES`);
#: the reader refuses anything else by name. A name changes with its file's shape, on both sides, in one change: v2
#: (2026-09-24, instruction-v3) gives every heading word judged its band and per-row verdicts (``heading``) and every
#: flight judged its flown track as the judge read it (``judgedTrackDeg``); v1's heading verdicts were turns and holds.
SCHEMA = "aeroviz-training-executor-v2"
PAYLOAD_FILE = "executor.json"
RUNNER = "ts_transformer.experiments.executor_training_export"
STATUSES = ("inside", "outside", "not judged", "not reached", "superseded", "no check")
#: How closely a re-flown crossing must reproduce the formal replay's: the executor flies a batch at once, and a batch
#: of other flights than the replay's chunk reassociates its floating-point sums (seen: one unit in the last place of
#: the crossing height, a 3-flight batch against the replay's 500). Outcomes and word verdicts are compared exactly.
CROSSING_TOLERANCE = 1e-9


def no_check_reason(word: Instruction) -> str:
    """Why the judge checks a word against no envelope of its own (`checkable` is false)."""
    if word.column == RUNWAY:
        return "the runway pointer: judged by the landing, the flight's outcome"
    if word.column == APPROACH:
        return ("cleared at entry: the judge checks the capture and the corridor for a clearance said after step 0"
                if word.value == APPROACH_CLEARED else "not cleared: no envelope of its own")
    if word.column == ANGLE:
        return "re-anchors its altitude word's tube: judged there"
    return "the pilot's own speed: no band to hold"


def _r(values: Any, digits: int) -> list[float]:
    return [round(float(value), digits) for value in np.asarray(values, dtype=np.float64).ravel()]


def _cell(word: Instruction) -> tuple[int, int]:
    """A word's cell in the sentence — its step and column; one word per cell."""
    return int(word.row), int(word.column)


def _said_cell(word: Instruction) -> tuple[int, int]:
    """The sentence cell of a word moved to the flown row it was said at (`judge.said_at` keeps its own row)."""
    return int(word.info["sentence_row"]), int(word.column)


def checkable(word: Instruction, words: Words) -> bool:
    """Does the judge check this word against an envelope of its own?"""
    if word.column == APPROACH:
        return word.kind == "clear"
    if word.column == SPEED:
        return words.speed_mps(word.value) is not None
    return word.column in (HEADING, ALTITUDE)


def said_words(flown: Flown, index: int, verdict: Verdict, reading: Reading, observed: FlightSignals,
               geometry: AirportGeometry, spec: VocabularySpec
               ) -> tuple[dict[tuple[int, int], int], list[Instruction], Admitted]:
    """The judge's own word bookkeeping (`judge.words_said`, `judge.read_flown`): the cycle each word the executor's
    clock reached was said at (by its sentence cell), those words at the flown rows they were said at, the superseded
    dropped, and the flown track as the labeller's gate read it (its rows are the rows judged). Only for a flight whose
    flown track passed the gate (``verdict.words`` set)."""
    said = words_said(flown, index, reading, spec)
    flight = read_flown(flown, index, verdict.outcome, verdict.end_row, geometry, observed, spec)
    cycles = {_cell(word): cycle for word, cycle in zip(reading.instructions, said.cycles) if cycle < said.n_cycles}
    return cycles, said.moved, flight


def chart_shift_deg(flown: Flown, index: int, geometry: AirportGeometry, observed_track_deg: float) -> float:
    """The whole turns that put the flown track on the chart's branch: the observed smoothed track's at row 0
    (``observed_track_deg``), so the two read on one heading axis."""
    states = flown.states[index, :1].cpu().numpy()
    return 360.0 * round((observed_track_deg - float(flown_track(states, geometry)["track"][0])) / 360.0)


def _check(name: str, ok: bool, inside: int | None = None, rows: int | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "inside": inside, "rows": rows}


def word_verdicts(flown: Flown, index: int, verdict: Verdict, reading: Reading, observed: FlightSignals,
                  geometry: AirportGeometry, spec: VocabularySpec, words: Words, shift_deg: float
                  ) -> tuple[list[dict[str, Any]], list[tuple[str, bool]] | None, int, np.ndarray | None]:
    """One verdict per word of ``reading``'s sentence, in its order; the words judged as `replay.word_results`
    lists them (None when the flown track did not pass the gate) and the heading words not judged — for the caller
    to check against the judge's own count — and the flown track as the judge read it (smoothed, to the landing cut;
    None with the gate refusing it, or for a dynamics failure: see the module's note). ``shift_deg``
    (`chart_shift_deg`) moves that track and each heading word's band onto the chart's branch."""
    out = {_cell(word): {"row": int(word.row), "column": int(word.column), "value": int(word.value), "status": None,
                         "flownRow": None, "heading": None, "checks": [], "reason": None}
           for word in reading.instructions}
    by_cell = {_cell(word): word for word in reading.instructions}

    def settle(cell: tuple[int, int], status: str, reason: str | None = None) -> None:
        out[cell]["status"], out[cell]["reason"] = status, reason

    def remaining() -> list[tuple[int, int]]:
        return [cell for cell, item in out.items() if item["status"] is None]

    if verdict.words is None:
        for cell in remaining():
            if checkable(by_cell[cell], words):
                settle(cell, "not judged", f"the flown track did not pass the labeller's gate ({verdict.refused})")
            else:
                settle(cell, "no check", no_check_reason(by_cell[cell]))
        return list(out.values()), None, 0, None

    said, moved, flight = said_words(flown, index, verdict, reading, observed, geometry, spec)
    rows = flight.signals.n_rows
    flown_row = {_said_cell(word): int(word.row) for word in moved}
    for cell in remaining():
        if cell not in said:
            settle(cell, "not reached", "the flight ended before the executor's word clock reached this step")
        elif cell not in flown_row:
            settle(cell, "superseded", f"said on the flown step of the next {COLUMNS[cell[1]]} word, which flew instead")
        else:
            out[cell]["flownRow"] = flown_row[cell]
            if flown_row[cell] >= rows:
                settle(cell, "not reached", "said after the flown track's last step")
            elif not checkable(by_cell[cell], words):
                settle(cell, "no check", no_check_reason(by_cell[cell]))
    reached = [word for word in moved if word.row < rows]

    judged: list[tuple[str, bool]] = []
    not_judged = 0
    # the heading word in force when the executor left it to intercept the final on its own (the judge fails it)
    off_cycles = verdict.words["intercepting_off_word_cycles"]
    off_cell = None
    if off_cycles:
        # the heading word in force at the first cycle it happened: the last one said at or before that cycle
        first = int(np.nonzero(flown.modes["intercepting_off_word"][index, : verdict.end_row].cpu().numpy())[0][0])
        told = [_said_cell(word) for word in reached if word.column == HEADING and said[_said_cell(word)] <= first]
        off_cell = max(told, key=lambda cell: (said[cell], cell[0]))
        judged.append(("heading", False))

    # the heading words the judge judged, in its order (`judge.judge`: the words reached, as said), each on the rows it
    # judged: from its flown row plus the lead, as many as its result counts
    heading = [word for word in reached if word.column == HEADING]
    results = verdict.words["heading"]
    if [word.row for word in heading] != [result["row"] for result in results]:
        raise ValueError(f"{reading.dataset_id}: the judge's heading results are not the heading words said")
    lead = spec.rows_exact(spec.heading_lead_s)
    # the judge read the failed state of a dynamics failure, which the exported track leaves out: nothing to draw on
    drawn = verdict.outcome != "dynamics_failure"
    track = flight.smoothed.track_deg + shift_deg
    skipped = replay.skipped_by_clock(results)
    for number, (word, result) in enumerate(zip(heading, results)):
        cell = _said_cell(word)
        first_row = word.row + lead
        if drawn:
            band = display.heading_band(track, word.row, float(word.info["target_deg"]), first_row,
                                        first_row + result["rows"], spec.heading_tolerance_deg, result)
            out[cell]["heading"] = band_payload(band)
        if result["rows"] == 0:
            not_judged += 1
            if skipped[number]:
                reason = "no row to judge: the next heading word was told on the same flown step"
            elif first_row >= rows:
                reason = (f"no row to judge: its rows, {spec.heading_lead_s:g} s after it was told, begin past the end "
                          "of the flown track its judge read")
            else:
                reason = (f"no row to judge: its rows, {spec.heading_lead_s:g} s after it was told, begin at or past "
                          "the clearance the executor was told or its capture (the capture turn is judged in its place)")
            settle(cell, "not judged", reason)
            continue
        ok = result["inside"] == result["rows"]
        out[cell]["checks"] = [_check(f"track within ±{spec.heading_tolerance_deg:g}° of the word, "
                                      f"{spec.heading_lead_s:g} s after it was told, to the next word's",
                                      ok, result["inside"], result["rows"])]
        settle(cell, "inside" if ok else "outside")
        judged.append(("heading", ok))
    if off_cell is not None:
        out[off_cell]["checks"] = [*out[off_cell]["checks"],
                                   _check(f"held until the capture — left for {off_cycles} cycles to intercept the final "
                                          "on its own", False)]
        settle(off_cell, "outside")

    corridor, capture = verdict.words["corridor"], verdict.words["capture_turn"]
    if corridor["cleared"]:
        (clear,) = [word for word in reached if word.column == APPROACH and word.kind == "clear"]
        checks = ([_check("captured the final", False)] if capture is None else
                  [_check("capture turn monotone", capture["progress_ok"]),
                   _check("capture turn rate and bank", capture["rate_ok"])])
        checks += [_check("corridor entered", corridor["entered"]),
                   _check("corridor held to the landing", corridor["inside"] == corridor["rows"], corridor["inside"],
                          corridor["rows"])]
        ok = all(check["ok"] for check in checks)
        cell = _said_cell(clear)
        out[cell]["checks"] = checks
        settle(cell, "inside" if ok else "outside")
        judged.append(("approach", ok))

    tubes = verdict.words["vertical"]
    altitude = sorted((word for word in reached if word.column == ALTITUDE), key=lambda word: word.row)
    if [word.row for word in altitude] != [tube["row"] for tube in tubes]:
        raise ValueError(f"{reading.dataset_id}: the judge's altitude tubes are not the altitude words said")
    for word, tube in zip(altitude, tubes):
        cell = _said_cell(word)
        out[cell]["checks"] = [_check("in its tube", tube["contained"], tube["inside"], tube["rows"])]
        settle(cell, "inside" if tube["contained"] else "outside")
        judged.append(("altitude", bool(tube["contained"])))

    spans = verdict.words["speed"]
    speed = sorted((word for word in reached if word.column == SPEED and words.speed_mps(word.value) is not None),
                   key=lambda word: word.row)
    if [word.row for word in speed] != [span["row"] for span in spans]:
        raise ValueError(f"{reading.dataset_id}: the judge's speed spans are not the speed words said")
    for word, span in zip(speed, spans):
        cell = _said_cell(word)
        # the judge's verdict (`contained`) is the transition and the band; its acceleration check is not part of it
        out[cell]["checks"] = [_check("transition monotone toward the target", span["transition_ok"]),
                               _check("band held" if not span["cut_before_arrival"] else
                                      "band not reached before the next word", span["band_inside"] == span["band_rows"],
                                      span["band_inside"], span["band_rows"])]
        settle(cell, "inside" if span["contained"] else "outside")
        judged.append(("speed", bool(span["contained"])))

    unsettled = remaining()
    if unsettled:
        raise ValueError(f"{reading.dataset_id}: words at {unsettled} were said and reached but the judge judged none")
    # the reader's rule: a word is inside exactly when every check it lists passed
    for item in out.values():
        if item["status"] in ("inside", "outside") and (item["status"] == "inside") != all(c["ok"] for c in item["checks"]):
            raise ValueError(f"{reading.dataset_id}: the word at step {item['row']}, column {item['column']} is "
                             f"{item['status']} but its checks say {item['checks']}")
    return list(out.values()), judged, not_judged, track if drawn else None


def track_payload(flown: Flown, index: int, verdict: Verdict, geometry: AirportGeometry, spec: VocabularySpec,
                  shift_deg: float) -> dict[str, Any]:
    """The flown track every sentence step from row 0 to its outcome's row (a dynamics failure: to the row before
    it, as the records keep it), in the airport frame and on the globe; its track unwrapped and moved by
    ``shift_deg`` (`chart_shift_deg`) onto the observed smoothed track's branch, so the two read on one heading axis."""
    end = verdict.end_row - 1 if verdict.outcome == "dynamics_failure" else verdict.end_row
    states = flown.states[index, : end + 1].cpu().numpy()
    track = flown_track(states, geometry)
    step_rows = int(round(spec.step_s / flown.cycle_s))
    rows = list(range(0, end + 1, step_rows))
    if rows[-1] != end:
        rows.append(end)
    lat, lon, height = states[rows, LAT], states[rows, LON], states[rows, ALT]
    undulation = np.asarray(geoid_undulation_m(list(lat), list(lon)), dtype=np.float64)
    distance = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(track["e"]), np.diff(track["n"])))))
    heading = track["track"][rows] + shift_deg
    return {"tS": _r(np.asarray(rows) * flown.cycle_s, 3), "eM": _r(track["e"][rows], 1), "nM": _r(track["n"][rows], 1),
            "lon": _r(lon, 7), "lat": _r(lat, 7), "altitudeM": _r(height, 2), "altitudeHaeM": _r(height + undulation, 2),
            "groundSpeedMps": _r(track["ground_speed"][rows], 3), "trackDeg": _r(heading, 3),
            "distanceM": _r(distance[rows], 1)}


def gate_block(gates: dict[str, Any], airport: str) -> dict[str, Any]:
    """The formal replay's gate table for this airport and all airports, per group and stratum."""
    def cell(values: dict[str, Any]) -> dict[str, Any]:
        clears = values["clears"]
        return {"flights": values["flights"], "landed": values["landed"], "wordsJudged": values["words_judged"],
                "wordsInside": values["words_inside"], "observedPasses": values["observed_passes"],
                "evaluationPaired": values["replay_passes_where_observed_passes"],
                "clears": clears if isinstance(clears, dict) else None,
                "notGated": None if isinstance(clears, dict) else clears,
                "outcomes": values["outcomes"]}
    return {group: {place: {stratum: cell(values) for stratum, values in by_airport[place].items()}
                    for place in (airport, "all")}
            for group, by_airport in gates.items()}


def flight_payload(item: dict[str, Any], group: str, flown: Flown | None, index: int, verdict: Verdict | None,
                   reading: Reading | None, observed: FlightSignals, geometry: AirportGeometry, spec: VocabularySpec,
                   words: Words, formal: dict[str, Any] | None) -> dict[str, Any]:
    """One flight of the set: not flown (``group`` says why), or flown — re-flown here and checked against its
    ``formal`` replay row."""
    base = {"flightKey": item["flightKey"], "datasetId": item["datasetId"], "group": group}
    if flown is None:
        return {**base, "flown": False, "outcome": None, "flewTheSentence": None, "endS": None, "crossing": None,
                "refused": None, "evaluation": None, "alignment": None, "limits": None, "counts": None, "track": None,
                "judgedTrackDeg": None, "words": []}
    dataset_id = item["datasetId"]
    shift = chart_shift_deg(flown, index, geometry, item["signals"]["smoothed"]["trackDeg"][0])
    verdicts, judged, not_judged, judged_track = word_verdicts(flown, index, verdict, reading, observed, geometry, spec,
                                                               words, shift)
    counted = replay.word_results(verdict)
    theirs = None if counted is None else [list(pair) for pair in counted[0]]
    ours = None if judged is None else [list(pair) for pair in judged]
    formal_crossing = formal["crossing"]
    same_crossing = (verdict.crossing is None) == (formal_crossing is None) and (
        verdict.crossing is None or (verdict.crossing.keys() == formal_crossing.keys() and all(
            math.isclose(verdict.crossing[key], formal_crossing[key], rel_tol=CROSSING_TOLERANCE, abs_tol=CROSSING_TOLERANCE)
            for key in formal_crossing)))
    if (verdict.outcome, verdict.flew_the_sentence) != (formal["outcome"], formal["flew_the_sentence"]) or not same_crossing:
        raise SystemExit(f"{dataset_id}: re-flown {verdict.outcome} {verdict.crossing}, the formal replay holds "
                         f"{formal['outcome']} {formal_crossing} — this is not the replay's flight")
    if not (ours == theirs == formal["words"]):
        raise SystemExit(f"{dataset_id}: the words judged differ — rebuilt {ours}, judge {theirs}, formal {formal['words']}")
    if counted is not None and not (not_judged == counted[1] == formal["heading_words_not_judged"]):
        raise SystemExit(f"{dataset_id}: heading words not judged {not_judged}, judge {counted[1]}, formal "
                         f"{formal['heading_words_not_judged']}")
    statuses = [entry["status"] for entry in verdicts]
    if verdict.words is not None and statuses.count("not reached") != formal["words_not_reached"]:
        raise SystemExit(f"{dataset_id}: {statuses.count('not reached')} words not reached, the formal replay counts "
                         f"{formal['words_not_reached']}")
    if statuses.count("superseded") != formal["words_superseded_before_flown"]:
        raise SystemExit(f"{dataset_id}: {statuses.count('superseded')} words superseded, the formal replay counts "
                         f"{formal['words_superseded_before_flown']}")
    crossing = verdict.crossing
    return {
        **base, "flown": True, "outcome": verdict.outcome, "flewTheSentence": bool(verdict.flew_the_sentence),
        "endS": round(verdict.end_row * flown.cycle_s, 3),
        "crossing": None if crossing is None else {"crossM": round(crossing["cross_m"], 2),
                                                   "heightM": round(crossing["height_m"], 2),
                                                   "atS": round(crossing["at_row"] * flown.cycle_s, 3)},
        "refused": verdict.refused,
        "evaluation": {"replay": formal["replay_verdict"], "observed": formal["observed_verdict"]},
        "alignment": {"meanHorizontalDistanceM": round(formal["mean_horizontal_distance_m"], 1),
                      "meanVerticalDistanceM": round(formal["mean_vertical_distance_m"], 1),
                      "landingTimeMinusObservedS": (None if formal["landing_time_minus_observed_s"] is None
                                                    else round(formal["landing_time_minus_observed_s"], 1))},
        "limits": {"cycles": verdict.limits["cycles"]["cycles"],
                   "bound": {name: value["cycles"] for name, value in verdict.limits.items() if name != "cycles"}},
        "counts": {"wordsJudged": 0 if judged is None else len(judged),
                   "wordsInside": 0 if judged is None else sum(ok for _, ok in judged),
                   "headingWordsNotJudged": not_judged},
        "track": track_payload(flown, index, verdict, geometry, spec, shift),
        "judgedTrackDeg": None if judged_track is None else _r(judged_track, 3),
        "words": verdicts,
    }


def build_airport(base: BaseSet, flights: list[FlightSignals], sentences: dict[str, np.ndarray], instructions: Path,
                  geometry: AirportGeometry, params: Any, words: Words, formal: dict[str, dict[str, Any]],
                  device: torch.device) -> list[dict[str, Any]]:
    """Every flight of the set: those the replay flies re-flown in one batch and judged; the rest listed."""
    spec = words.spec
    located = base_flights(base, flights, sentences)
    signals = [flight for flight, _ in located]
    series = rebuild_series(instructions, signals)
    groups = [replay.group_of(item) for item in series]
    flyable = [j for j, group in enumerate(groups) if group in (replay.OWN, replay.STAND_IN)]
    readings = []
    for j in flyable:
        reading = read_flight(signals[j], geometry, spec, words)
        grid = sentences["words"][sentences["offsets"][located[j][1]]: sentences["offsets"][located[j][1] + 1]]
        if not np.array_equal(reading.words, grid):
            raise SystemExit(f"{signals[j].dataset_id}: the re-read sentence differs from the stored one")
        if signals[j].dataset_id not in formal:
            raise SystemExit(f"{signals[j].dataset_id} is flown by the replay's rule but has no row in the formal replay")
        if formal[signals[j].dataset_id]["group"] != groups[j]:
            raise SystemExit(f"{signals[j].dataset_id} flies on {groups[j]} here, on {formal[signals[j].dataset_id]['group']} "
                             "in the formal replay")
        readings.append(reading)
    for j, group in enumerate(groups):
        if j not in flyable and signals[j].dataset_id in formal:
            raise SystemExit(f"{signals[j].dataset_id} is not flown here ({group}) but has a row in the formal replay")
    batch = replay.Batch(signals=[signals[j] for j in flyable], series=[series[j] for j in flyable], readings=readings,
                         geometries=[geometry] * len(flyable),
                         approach_ias_mps=[replay.flight_approach_ias_mps(series[j], groups[j]) for j in flyable],
                         groups=[groups[j] for j in flyable], drawn={})
    flown, verdicts = replay.fly_batch(batch, params, words, device=device) if flyable else (None, [])
    position = {j: n for n, j in enumerate(flyable)}
    out = []
    for j, item in enumerate(base.sample["flights"]):
        if j in position:
            n = position[j]
            out.append(flight_payload(item, groups[j], flown, n, verdicts[n], readings[n], signals[j], geometry, spec,
                                      words, formal[signals[j].dataset_id]))
        else:
            out.append(flight_payload(item, groups[j], None, 0, None, None, signals[j], geometry, spec, words, None))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--executor", type=Path, required=True, help="the formal executor spec directory")
    parser.add_argument("--replay", type=Path, required=True, help="that spec's formal val replay (executor_replay --out)")
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact it flies")
    parser.add_argument("--airports-root", type=Path, required=True,
                        help="the frontend's airports directory (…/public/data/airports)")
    parser.add_argument("--set", required=True, help="the Training set whose flights are replayed, e.g. instruction_v3")
    parser.add_argument("--airport", action="append", required=True, help="an ICAO code; repeat for several")
    parser.add_argument("--overlay-id", default=None, help="default: executor_<the spec directory's name>")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    executor, replay_dir, instructions, root = (resolved(p) for p in (args.executor, args.replay, args.instructions,
                                                                       args.airports_root))
    airports = [code.upper() for code in args.airport]
    if len(set(airports)) != len(airports):
        parser.error(f"an airport is named twice in {airports}")
    overlay_id = args.overlay_id or f"executor_{executor.name}"
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor, instructions)
    spec = words.spec
    formal = json.loads((replay_dir / "replay.json").read_text(encoding="utf-8"))
    if formal["schema"] != REPLAY_SCHEMA:
        parser.error(f"{replay_dir / 'replay.json'} is a {formal['schema']} file, not {REPLAY_SCHEMA}")
    if (formal["executor_spec_sha256"], formal["split"]) != (record["sha256"], SPLIT):
        parser.error(f"{replay_dir} is the {formal['split']} replay of spec {formal['executor_spec_sha256'][:12]}, not "
                     f"the {SPLIT} replay of {record['sha256'][:12]}")
    rows = {row["dataset_id"]: row for row in formal["flights"]}
    geometries = load_candidates(instructions)
    flights = load_signals(instructions, SPLIT)
    sentences = load_sentences(instructions, SPLIT, spec)
    # every refusal before anything is flown or written
    bases, existing = {}, {}
    for code in airports:
        training = root / code / "training"
        if (training / overlay_id).exists():
            parser.error(f"{training / overlay_id} exists; an overlay is never overwritten")
        existing[code] = read_overlays(training, code, overlay_id)
        bases[code] = open_base_set(training, code, args.set, spec)

    def relative(path: Path) -> str:
        return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.as_posix()

    git = git_state()
    source = {"runner": RUNNER, "executor": relative(executor), "replay": relative(replay_dir),
              "instructions": relative(instructions), "git": git}
    title = (f"Executor · spec {record['sha256'][:12]} · the truth sentence flown from row 0, each word said where the "
             f"observed aircraft heard it (word clock {params.word_clock})")
    built = {}
    for code in airports:
        payloads = build_airport(bases[code], flights, sentences, instructions, geometries[code], params, words, rows,
                                 torch.device(args.device))
        flown = [item for item in payloads if item["flown"]]
        payload = {
            "schema": SCHEMA, "overlayId": overlay_id, "airport": code, "writtenUtc": utc_now(), "producedBy": source,
            "base": bases[code].block,
            "executor": {"specSha256": record["sha256"], "wordClock": params.word_clock, "cycleS": params.cycle_s,
                         "params": [{"name": name, "value": value} for name, value in _flat(asdict(params))]},
            "replay": {"split": formal["split"], "writtenUtc": formal["written_utc"], "gateShare": formal["gate_share"],
                       "drawn": formal["drawn"], "git": formal["git"]},
            "gate": gate_block(formal["gates"], code),
            "flights": payloads,
        }
        entry = overlay_entry(overlay_id, KIND_EXECUTOR, bases[code], title, PAYLOAD_FILE, len(payloads), source)
        built[code] = (serialise_overlay(payload), entry)
        landed = sum(item["outcome"] == "landed" for item in flown)
        print(f"  {code}: {len(flown)} of {len(payloads)} flights flown ({landed} landed), each its formal replay row; "
              f"{time.perf_counter() - started:.0f} s", flush=True)
    for code, (text, entry) in built.items():
        out = write_overlay(root / code / "training", code, overlay_id, entry, text, existing[code])
        print(f"  {code}: {out.stat().st_size / 1e6:.1f} MB → {out}", flush=True)
    return 0


def _flat(values: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """A nested parameter dict as ``(dotted name, value)`` rows, in its order."""
    rows: list[tuple[str, Any]] = []
    for name, value in values.items():
        if isinstance(value, dict):
            rows += _flat(value, f"{prefix}{name}.")
        else:
            rows.append((f"{prefix}{name}", value))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
