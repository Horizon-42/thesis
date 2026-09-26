"""The flight: rebuilt from the data plane, the segment flown with the executor's own stepper, stopped at its stop.

NOTHING IS PRECOMPUTED, AND THE EXECUTOR IS USED AS IT IS. A flight is rebuilt from the arrival manifest its artefact
recorded (`flights.rebuild_series` refuses a moved manifest or a rebuilt flight that differs from the stored signals),
re-read with the labeller (it must give the stored sentence), and flown with `executor.Executor`, one control cycle at a
time, driven exactly as `executor.fly` drives it — the spec's word clock read before each cycle, a step's words heard
on the cycle that starts it (`sentence.row_at`, as `judge.words_said` reads the clock) — and stopped at the segment's
stop: nothing past it is flown. The executor starts from the observed aircraft's state at the segment's first step (the
data plane's flight first seen there, `dataset.series_from_row`).

A MODEL's word (`segment.model_segment`) is flown as the prior's free generation flew it
(`experiments.prior_free_generation.speak_and_fly`): from the observed state at the sentence's first step, each word
heard at the step it was said (the time clock: `ClosedLoop.step` tells step k's words on its first cycle), under the
same time limit — the observed flight's remaining time from that step × the spec's timeout factor (`limits_s`) — and
judged against the runway the model points at, not the observed one. The executor is deterministic, so the flight is
the exported sample's own, which the frontend checks point for point.
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

from aeroviz_backend.autopilot_segment.errors import NotFlyable, RequestRefused, Superseded
from aeroviz_backend.autopilot_segment.segment import (
    ModelSentence, Segment, judged_reading, model_reading, model_segment, segment_of, segment_reading, segment_signals,
)

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


def segment_batch(context: FlightContext, segment: Segment, reading: Reading, signals: FlightSignals) -> replay.Batch:
    """The segment as the replay's batch of one: its reading and observed rows, the flight from the step the executor
    starts at."""
    return replay.Batch(signals=[signals], series=[series_from_row(context.series, segment.start_row)],
                        readings=[reading], geometries=[context.geometry],
                        crossing_heights=[context.crossing_heights], approach_ias_mps=[context.approach_ias_mps],
                        groups=[context.group], drawn={})


def model_time_limit_s(context: FlightContext, first_row: int, params: ExecutorParams, step_s: float) -> float:
    """A model's flight's time limit: the observed flight's remaining time from the sentence's first step × the spec's
    timeout factor. MIRROR of `experiments.prior_free_generation.limits_s` (a runner the backend does not import; pinned
    by `test_autopilot_segment.ModelSegmentTest`)."""
    return (len(context.reading.words) - first_row) * step_s * params.timeout_factor


def fly_batch_until(batch: replay.Batch, params: ExecutorParams, words: Words, stop_steps: int | None,
                    superseded: Callable[[], bool], *, model_limit_s: float | None = None) -> tuple[Flown, bool, float]:
    """``batch``'s one sentence flown to ``stop_steps`` (or its outcome): the flown record, whether it stopped there,
    and the executor's own wall time. The setup is `replay.fly_sentences`' — its time limit, runways, charts, approach
    speeds and word clock — with the executor stepped here (`fly_until`) in place of `executor.fly`: the setup is pinned
    against it by `test_autopilot_segment.SetupTest`, the stepping by `StepperTest`. ``model_limit_s``: a model's
    sentence, flown as its free generation flew it — its time limit, and every word heard at its own step (the time
    clock) — None for the truth."""
    spec, f64 = words.spec, torch.float64
    (reading,) = batch.readings
    limit_s = len(reading.words) * spec.step_s * params.timeout_factor if model_limit_s is None else model_limit_s
    executor = Executor(batch.inputs(DEVICE),
                        Runways.of(batch.geometries, batch.crossing_heights, dtype=f64, device=DEVICE),
                        AirportCharts.of(batch.geometries, dtype=f64, device=DEVICE),
                        torch.tensor(batch.approach_ias_mps, dtype=f64, device=DEVICE), params, words,
                        time_limit_s=torch.tensor([limit_s], dtype=f64, device=DEVICE))
    sentences = Sentences([reading.words], words, device=DEVICE)
    clock = (replay.word_clock(batch, params, spec.step_s, DEVICE) if model_limit_s is None
             else TimeClock(params.cycle_s))
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
    signals: FlightSignals           # the observed rows from where it started (its clock read the truth's)
    model: ModelSentence | None      # the model's sentence flown, or None for the truth's
    flown: Flown                     # to its stop, or its outcome
    verdict: Verdict
    reached_end: bool | None         # None when flown to its outcome (``stop_row`` is the sentence's end)
    fly_s: float                     # wall time the executor's cycles took, and the judge
    judge_s: float


def fly_segment(context: FlightContext, params: ExecutorParams, words: Words, column: int, row: int,
                superseded: Callable[[], bool], model: ModelSentence | None = None) -> FlownSegment:
    """``column``'s word said at ``row`` flown and judged: the truth's from the observed state there; a model's
    (``model``) as its sentence was flown, from its first step (`segment.model_segment`)."""
    spec = words.spec
    lead = spec.rows_exact(spec.heading_lead_s)
    if model is None:
        segment = segment_of(context.reading, column, row, lead)
        reading, signals = segment_reading(context.reading, segment), segment_signals(context.signals, segment)
        limit_s = None
    else:
        if context.group != replay.OWN:
            raise RequestRefused(f"a model's sentences are flown on the flight's own dynamics only, as its free generation "
                                 f"flies them: this flight flies on {context.group}")
        segment = model_segment(model, column, row, lead, words)
        reading = model_reading(context.signals, segment, words)
        # the judge reads the flown track against the runway the MODEL points at (the observed rows name the observed one)
        signals = replace(segment_signals(context.signals, segment),
                          runway=context.geometry.candidates[reading.runway_index].ident)
        limit_s = model_time_limit_s(context, model.first_row, params, spec.step_s)
    batch = segment_batch(context, segment, reading, signals)
    flown, reached, fly_s = fly_batch_until(batch, params, words,
                                            None if segment.to_landing else segment.stop_row - segment.start_row,
                                            superseded, model_limit_s=limit_s)
    started = time.perf_counter()
    verdict = judge(flown, 0, context.geometry, reading.runway_index, judged_reading(reading, segment), signals, spec, words)
    # a word said as or after the flight ended has nothing flown after it — the sentence bar offers it no Fly (its step's
    # time is not before the sample's end); at a crossing or failure the judge does not even read its step
    if model is not None and verdict.end_row <= segment.word_step * int(round(spec.step_s / flown.cycle_s)):
        raise RequestRefused(f"the model's flight had ended ({verdict.outcome}, {verdict.end_row * flown.cycle_s:g} s after "
                             f"step {segment.start_row}) by the time it said this word at step {row}")
    return FlownSegment(segment=segment, reading=reading, signals=signals, model=model, flown=flown, verdict=verdict,
                        reached_end=None if segment.to_landing else reached, fly_s=fly_s,
                        judge_s=time.perf_counter() - started)
