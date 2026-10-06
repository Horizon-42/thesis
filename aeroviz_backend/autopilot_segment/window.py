"""``POST /autopilot/window-segment``: a word's segment of a commanded aircraft's sentence in a window of a Training set
of stage C (post-training §8 C11), flown live by the same single-flight executor as stage A's
(`backend.AutopilotSegmentBackend`, whose request lock, page numbering, executor spec and flown-set caches this shares).

WHICH SENTENCE. The request names a window set (``airport``, ``setId``: `post.training_files.FILES.listed_set`, the airport's
``training/index_post_v3.json``), a window of it (``window``: its place in the set's ``windows``), one of the window's
commanded aircraft (``aircraft``: its ``datasetId``; the window format of stages C and D holds a list of them, frontend
D156 — stage C's one) and ``round``: a round of the set's ``model.rounds`` (``"start"`` or a round's number). The
commanded flight is the set's flight of that ``datasetId``; the sentence's Δ is the set's ``model.rowIntervalS``. The
other aircraft, commanded or recorded, fly their states of the round and do not react (frontend §5.5).

THE SENTENCE IS FLOWN AS THE EXPORT FLEW IT. The export flew the window through the start of a closed loop
(`autopilot.start.start_moved`) from the commanded flight's first predicted step, its start moved for window B; here the
same flight is set up on its stored closed-loop sentence (`training_flights`, A23's setup), window B's observed rows to
the first predicted step moved as the start moves them (`autopilot.start.moved_signals`, from which the start rule reads
the start state), and the sentence's words grid replaced by the round's words (`prior.on_words`). The other aircraft are
not flown: the executor reads only the words (vocabulary §6 item 5). The answer is checked against the export's flown
track, written unrounded (`prior.apart_from_exported`, prior D127 followed for windows: refused by name past the
executor's bound, or at another outcome or end cycle). A word after the aircraft's end (its
outcome, or the end of the row of its loss of separation) has no segment. An aircraft whose flight ended at a loss of
separation ends there live too (`fly_window_segment`): no segment runs past its last state, and none is judged — the
judge has no outcome for a flight its caller ended (post-training D93).

WARMED UP AT START (`warm_up`, run beside stage A's and stage B's): every listed window set opened at its Δ, each under
its own lock (stage A's `set_flown`), never the request lock.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ts_transformer.autopilot import replay
from ts_transformer.experiments import training_flights
from ts_transformer.autopilot.start import Move, moved_signals
from ts_transformer.experiments.post_window_loop import LOST_SEPARATION
from ts_transformer.instructions import training_files
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.io_utils import utc_now
from ts_transformer.post import training_files as post_files

from aeroviz_backend.autopilot_segment.errors import NotListed, RequestRefused, Superseded
from aeroviz_backend.autopilot_segment.fly import FlownSegment, fly_segment, segment_of
from aeroviz_backend.autopilot_segment.payload import segment_payload
from aeroviz_backend.autopilot_segment.prior import apart_from_exported, on_words

#: MIRROR of `aeroviz-4d/src/data/trainingWindowAutopilot.ts` (`TRAINING_WINDOW_AUTOPILOT_SCHEMA`); the reader refuses
#: anything else by name. A name changes with the payload's shape, on both sides, in one change. v1 (C11, 2026-10-05):
#: stage A's segment answer with the window and the round whose sentence it flew. v2 (frontend D156, 2026-10-07): the
#: request names the commanded aircraft (``aircraft``), the answer gives it back with the window's end in that round.
SCHEMA = "aeroviz-autopilot-window-segment-v2"
#: The round of a set's ``model.rounds`` that names the model at the start of the campaign (`post_training_export.START`).
START = "start"


def move_of(move: dict[str, float]) -> Move:
    """A window's ``startMove`` (``turnDeg``, ``heightM``, ``speedScale``) as `autopilot.start.Move`."""
    return Move(turn_deg=move["turnDeg"], height_m=move["heightM"], speed_scale=move["speedScale"])


def moved_start(batch: replay.Batch, j: int, move: dict[str, float], rule: str) -> tuple[replay.Batch, Any]:
    """Flight ``j`` of ``batch`` with its observed rows to its first predicted step moved by ``move`` (a window's
    ``startMove``, `move_of`), and the batch's inputs from them (the start rule ``rule``)."""
    observed = list(batch.observed)
    observed[j] = moved_signals(observed[j], batch.sentences[j].first_row, move_of(move))
    moved = dataclasses.replace(batch, observed=observed)
    return moved, moved.inputs(rule, replay_device())


def replay_device() -> Any:
    """The device the live executor's inputs are made on (stage A's setup's: `training_flights.CPU`)."""
    from ts_transformer.experiments import training_flights

    return training_flights.CPU


def fly_window_segment(batch: replay.Batch, inputs: Any, j: int, sentence: Any, column: int, row: int, params: Any,
                       words: Any, superseded: Callable[[], bool], end_cycle: int | None) -> FlownSegment:
    """`fly.fly_segment` for a window: ``end_cycle`` None, the same (a window the executor ended); else the window
    ended at a loss of separation at its state ``end_cycle`` (the export's ``track.lastCycle``), and the segment stops
    there at the latest, unjudged."""
    if end_cycle is None:
        return fly_segment(batch, inputs, j, sentence, column, row, params, words, superseded)
    segment = segment_of(sentence, column, row, batch.row_interval_s, params, words)
    stop = end_cycle if segment.stop_cycle is None else min(segment.stop_cycle, end_cycle)
    segment = dataclasses.replace(segment, stop_cycle=stop)
    started = time.perf_counter()
    try:
        flown, stopped = training_flights.fly_single(batch, inputs, j, params, words, stop_cycle=stop,
                                                     superseded=superseded)
    except InterruptedError as error:
        raise Superseded(f"a newer request came in: {error}") from None
    if not stopped:
        raise ValueError(f"the window's flight ended before its loss at cycle {end_cycle}: not the export's flight")
    return FlownSegment(segment, flown, True, None, time.perf_counter() - started, 0.0)


def is_moved(move: dict[str, float]) -> bool:
    """Whether a window's ``startMove`` moves anything (`Move`'s zero: no turn, no height, speed scale 1)."""
    return dataclasses.astuple(move_of(move)) != (0.0, 0.0, 1.0)


class WindowSegments:
    """``fly(payload)`` for ``POST /autopilot/window-segment``: ``{clientId, seq, airport, setId, window, aircraft,
    round, column, row}``, on ``backend``'s caches, lock and page numbering."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def listed(self, airport: str, set_id: str) -> dict[str, Any]:
        """The window set's sample (`post.training_files.FILES.listed_set`)."""
        from aeroviz_backend.autopilot_segment.backend import AIRPORT_CODE

        if not AIRPORT_CODE.fullmatch(airport):
            raise RequestRefused(f"airport {airport!r} is not an airport code")
        try:
            return post_files.FILES.listed_set(self.backend.airports_root / airport / "training", airport, set_id)[1]
        except training_files.NotListed as error:
            raise NotListed(str(error)) from None

    def warm_up(self, log: Callable[[str], None] = print) -> None:
        """Every window set opened at its Δ ahead of its first request (as stage A's warm-up does its sets): each set under
        its own lock (stage A's `set_flown`), never under the request lock; a set it cannot open is skipped with its
        reason."""
        started, opened_sets = time.perf_counter(), 0
        for index in sorted(self.backend.airports_root.glob(f"*/training/{post_files.INDEX_FILE}")):
            airport = index.parent.parent.name
            try:
                sets = post_files.FILES.index_sets(json.loads(index.read_text(encoding="utf-8")), index, airport)
            except Exception as error:       # noqa: BLE001 — a prefetch: logged; a request gets it whole
                log(f"window warm-up: {airport} skipped — {index}: {type(error).__name__}: {error}")
                continue
            for entry in sets:
                began = time.perf_counter()
                try:
                    sample = self.listed(airport, entry["id"])
                    instructions, _, params, _, words = self.backend.executor_for(sample)
                    self.backend.set_flown(sample, sample["cohort"]["split"], float(sample["model"]["rowIntervalS"]),
                                           instructions, params, words)
                except Exception as error:   # noqa: BLE001 — a prefetch: logged by type; a request gets it whole
                    log(f"window warm-up: {airport} {entry['id']} skipped — {type(error).__name__}: "
                        f"{str(error).split('; ')[0]}")
                    continue
                opened_sets += 1
                log(f"window warm-up: {airport} {entry['id']}: {len(sample['windows'])} windows opened in "
                    f"{time.perf_counter() - began:.1f} s")
        log(f"window warm-up: {opened_sets} sets ready in {time.perf_counter() - started:.1f} s")

    def fly(self, payload: dict[str, Any]) -> dict[str, Any]:
        from aeroviz_backend.autopilot_segment.backend import _field, _require_split, _whole

        backend = self.backend
        client = str(_field(payload, "clientId"))
        seq = _whole(_field(payload, "seq"), "seq")
        airport = str(_field(payload, "airport"))
        set_id = str(_field(payload, "setId"))
        place = _whole(_field(payload, "window"), "window")
        named = str(_field(payload, "aircraft"))
        which = _field(payload, "round")
        column_name = _field(payload, "column")
        row = _whole(_field(payload, "row"), "row")
        if column_name not in COLUMNS:
            raise RequestRefused(f"column {column_name!r} is none of {COLUMNS}")
        if which != START:
            _whole(which, "round")
        backend._claim(client, seq)
        superseded = lambda: backend._latest[client] != seq  # noqa: E731 — read each cycle
        asked = time.perf_counter()
        with backend._lock:
            if superseded():
                raise Superseded("a newer request from this page came in while this one waited")
            started = time.perf_counter()
            sample = self.listed(airport, set_id)
            interval = float(sample["model"]["rowIntervalS"])
            if place >= len(sample["windows"]):
                raise NotListed(f"Training set {set_id} at {airport} has {len(sample['windows'])} windows, no window "
                                f"{place}")
            window = sample["windows"][place]
            commanded = [a for a in window["commanded"] if a["datasetId"] == named]
            if len(commanded) != 1:
                raise NotListed(f"window {place} of set {set_id} commands no aircraft {named!r} (it commands "
                                f"{', '.join(a['datasetId'] for a in window['commanded'])})")
            aircraft = commanded[0]
            said = [r for r in aircraft["rounds"] if r["round"] == which]
            ended = [r for r in window["rounds"] if r["round"] == which]
            if len(said) != 1 or len(ended) != 1:
                raise NotListed(f"window {place} of set {set_id} has no round {which!r}")
            said, ended = said[0], ended[0]
            items = [item for item in sample["flights"] if item["datasetId"] == named]
            if len(items) != 1:
                raise ValueError(f"set {set_id} holds {len(items)} flights of window {place}'s {named}")
            item = items[0]
            _require_split(item["split"], backend.splits)        # before any check runs (A37)
            instructions, executor, params, record, words = backend.executor_for(sample)
            heard = row * int(round(interval / params.cycle_s))
            if heard > said["track"]["lastCycle"]:
                raise RequestRefused(f"the word at Δ row {row} is said at cycle {heard}, after the window's end at cycle "
                                     f"{said['track']['lastCycle']}")
            flown_set = backend.set_flown(sample, item["split"], interval, instructions, params, words)
            opened = time.perf_counter()
            j = flown_set.position[item["datasetId"]]
            batch, inputs, sentence = flown_set.batch, flown_set.inputs, flown_set.sentences[j]
            if is_moved(aircraft["startMove"]):
                batch, inputs = moved_start(batch, j, aircraft["startMove"], params.start_rule)
            batch, sentence = on_words(batch, j, sentence, np.array(said["words"], dtype=np.int16), words)
            lost = said["outcome"] == LOST_SEPARATION
            result = fly_window_segment(batch, inputs, j, sentence, COLUMNS.index(column_name), row, params, words,
                                        superseded, said["track"]["lastCycle"] if lost else None)
            answering = time.perf_counter()
            apart = apart_from_exported(result, said, batch, j, words.spec.step_s)
            body = segment_payload(result, batch.geometries[j], float(item["haeMinusMslM"]),
                                   np.asarray(inputs.aero_params[j].cpu().numpy()), apart)
            finished = time.perf_counter()
        return {
            "ok": True, "schema": SCHEMA, "airport": airport, "setId": set_id, "window": place, "aircraft": named,
            "round": which, "datasetId": item["datasetId"], "rowIntervalS": interval, "computedUtc": utc_now(),
            "reward": said["reward"], "windowEnd": {"losses": ended["losses"], "faultySteps": ended["faultySteps"]},
            "timing": {"waitS": round(started - asked, 3), "openS": round(opened - started, 3),
                       "flyS": round(result.fly_s, 3), "cycles": int(result.flown.commands.shape[1]),
                       "judgeS": round(result.judge_s, 3), "answerS": round(finished - answering, 3),
                       "computeS": round(finished - started, 3)},
            "executor": {"spec": executor.name, "specSha256": record["sha256"], "cycleS": params.cycle_s},
            "artefact": instructions.name,
            **body,
        }
