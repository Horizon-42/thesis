"""The flight: rebuilt from the data plane, the segment flown with the executor's own stepper, stopped at its stop.

NOTHING IS PRECOMPUTED, AND THE EXECUTOR IS USED AS IT IS. A flight is rebuilt from the arrival manifest its artefact
recorded (`flights.rebuild_series` refuses a moved manifest or a rebuilt flight that differs from the stored signals),
re-read with the labeller (it must give the stored sentence), and flown with `executor.Executor`, one control cycle at a
time, driven exactly as `executor.fly` drives it — the spec's word clock read before each cycle, a step's words heard
on the cycle that starts it (`sentence.row_at`, as `judge.words_said` reads the clock) — and stopped at the segment's
stop: nothing past it is flown. The executor starts from the observed aircraft's state at the segment's first step (the
data plane's flight first seen there, `dataset.series_from_row`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Executor, Flown
from ts_transformer.autopilot.flights import rebuild_series
from ts_transformer.autopilot.frame import AirportCharts
from ts_transformer.autopilot.judge import Verdict, judge
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.runway_data import published_crossing_heights
from ts_transformer.autopilot.sentence import DistanceClock, Sentences, TimeClock, TrackClock, row_at
from ts_transformer.data.dataset import FlightSeries, series_from_row
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.labeller.read import Reading, admit, read_flight
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.training_files import require_stored_sentence, stored_sentence
from ts_transformer.instructions.words import Words

from aeroviz_backend.autopilot_segment.errors import NotFlyable, Superseded
from aeroviz_backend.autopilot_segment.segment import Segment, segment_of, segment_reading, segment_signals

DEVICE = torch.device("cpu")


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
    geometry = load_candidates(artefact)[flight.airport]
    reading = read_flight(flight, geometry, spec, words)
    require_stored_sentence(dataset_id, reading, stored_sentence(sentences, stored[index]))
    (series,) = rebuild_series(artefact, [flight])
    group = replay.group_of(series)
    if group not in (replay.OWN, replay.STAND_IN):
        raise NotFlyable(f"{dataset_id} cannot be flown: {group}")
    observed = admit(flight, geometry, spec)
    return FlightContext(signals=flight, series=series, reading=reading, geometry=geometry,
                         crossing_heights=published_crossing_heights(geometry), group=group,
                         approach_ias_mps=replay.flight_approach_ias_mps(series, group),
                         observed_track_deg=observed.smoothed.track_deg,
                         observed_distance_m=observed.smoothed.distance_m)


def fly_until(executor: Executor, sentences: Sentences, clock: TimeClock | DistanceClock | TrackClock, step_s: float,
              stop_steps: int | None, superseded: Callable[[], bool]) -> bool:
    """Drive ``executor`` one cycle at a time exactly as `executor.fly` does — the word clock read before each cycle,
    a step's words heard on the cycle that starts it — and stop before the first cycle that starts a step the clock puts
    at or past ``stop_steps`` (where a word said there would be heard); without a stop, fly to the outcome. Whether it
    stopped there (False: the flight was done or out of cycles first, or there was no stop). Raises `Superseded`, before
    any cycle, as soon as ``superseded()`` says a newer request came in."""
    for cycle in range(executor.cycles):
        if superseded():
            raise Superseded(f"a newer request came in: stopped after {cycle} cycles")
        sentence_s = clock.now(cycle, executor.now())
        if cycle % executor.step_rows == 0:
            if stop_steps is not None and int(row_at(sentence_s, step_s)[0]) >= stop_steps:
                return True
            step_start_s = sentence_s
        executor.cycle(sentences.at(step_start_s), sentence_s)
        if bool(executor.done.all()):
            break
    return False


def segment_batch(context: FlightContext, segment: Segment) -> replay.Batch:
    """The segment as the replay's batch of one: its reading and observed rows, the flight from its first step."""
    return replay.Batch(signals=[segment_signals(context.signals, segment)],
                        series=[series_from_row(context.series, segment.row)],
                        readings=[segment_reading(context.reading, segment)], geometries=[context.geometry],
                        crossing_heights=[context.crossing_heights], approach_ias_mps=[context.approach_ias_mps],
                        groups=[context.group], drawn={})


def fly_batch_until(batch: replay.Batch, params: ExecutorParams, words: Words, stop_steps: int | None,
                    superseded: Callable[[], bool]) -> tuple[Flown, bool, float]:
    """``batch``'s one sentence flown to ``stop_steps`` (or its outcome): the flown record, whether it stopped there,
    and the executor's own wall time. The setup is `replay.fly_sentences`' — its time limit, runways, charts, approach
    speeds and word clock — with the executor stepped here (`fly_until`) in place of `executor.fly`: the setup is pinned
    against it by `test_autopilot_segment.SetupTest`, the stepping by `StepperTest`."""
    spec, f64 = words.spec, torch.float64
    (reading,) = batch.readings
    executor = Executor(batch.inputs(DEVICE),
                        Runways.of(batch.geometries, batch.crossing_heights, dtype=f64, device=DEVICE),
                        AirportCharts.of(batch.geometries, dtype=f64, device=DEVICE),
                        torch.tensor(batch.approach_ias_mps, dtype=f64, device=DEVICE), params, words,
                        time_limit_s=torch.tensor([len(reading.words) * spec.step_s * params.timeout_factor], dtype=f64,
                                                  device=DEVICE))
    sentences = Sentences([reading.words], words, device=DEVICE)
    clock = replay.word_clock(batch, params, spec.step_s, DEVICE)
    started = time.perf_counter()
    reached = fly_until(executor, sentences, clock, spec.step_s, stop_steps, superseded)
    fly_s = time.perf_counter() - started
    flown = executor.flown()
    if reached:
        # stopped by the segment, not done by the executor: it ends with the last cycle flown
        flown = replace(flown, done_cycle=torch.full_like(flown.done_cycle, executor.count - 1))
    return flown, reached, fly_s


@dataclass(frozen=True)
class FlownSegment:
    segment: Segment
    reading: Reading                 # the segment's: the sentence the executor was told
    signals: FlightSignals           # the observed rows its clock read
    flown: Flown                     # to its stop, or its outcome
    verdict: Verdict
    reached_end: bool | None         # None when flown to its outcome (``stop_row`` is the sentence's end)
    fly_s: float                     # wall time the executor's cycles took, and the judge
    judge_s: float


def fly_segment(context: FlightContext, params: ExecutorParams, words: Words, column: int, row: int,
                superseded: Callable[[], bool]) -> FlownSegment:
    """``column``'s word said at ``row`` flown from the observed state there, and judged."""
    spec = words.spec
    segment = segment_of(context.reading, column, row, spec.rows_exact(spec.heading_lead_s))
    batch = segment_batch(context, segment)
    flown, reached, fly_s = fly_batch_until(batch, params, words, None if segment.to_landing else segment.stop_row - segment.row,
                                            superseded)
    (reading,), (signals,) = batch.readings, batch.signals
    started = time.perf_counter()
    verdict = judge(flown, 0, context.geometry, reading.runway_index, reading, signals, spec, words)
    return FlownSegment(segment=segment, reading=reading, signals=signals, flown=flown, verdict=verdict,
                        reached_end=None if segment.to_landing else reached, fly_s=fly_s,
                        judge_s=time.perf_counter() - started)
