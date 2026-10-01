"""The flight: rebuilt from the data plane, the segment flown by the single-flight executor, stopped at its stop.

NOTHING IS PRECOMPUTED. A Training set's flights are rebuilt together from the arrival manifests their artefact recorded
(`flights.rebuild_series` refuses a moved manifest or a rebuilt flight that differs from the stored signals) the first
time the set is asked for — or when the backend warms up (`backend.AutopilotSegmentBackend.warm_up`) — each re-read with
the labeller (it must give the stored sentence), from the artefact split's stored files (`read_split`). A segment is flown by
`single.SingleExecutor` — the executor's single-flight way to fly, its cycle for one flight in plain floats (the spec
opens only for executor code that flies the spec's reference tracks within the bounds in every way, this one included:
`spec.require_conforming_executor`) —
one control cycle at a time, driven exactly as `executor.fly` drives it: the spec's word clock read before each cycle, a
step's words heard on the cycle that starts it (`sentence.row_at`, as `judge.words_said` reads the clock), and stopped at
the segment's stop: nothing past it is flown. The executor starts from the observed aircraft's state at the segment's
first step (the data plane's flight first seen there, `dataset.series_from_row`), set up as `replay.fly_sentences` sets
up the batch's (`flights.flight_inputs`, the runways and the chart from the artefact's candidates, the published
vertical paths, the approach speed, the time limit, the word clock).

A MODEL's word (`segment.model_segment`) is flown as the prior's free generation flew it
(`experiments.prior_free_generation.speak_and_fly`): from the observed state at the sentence's first step, each word
heard at the step it was said (the time clock: `ClosedLoop.step` tells step k's words on its first cycle), under the
same time limit — the observed flight's remaining time from that step × the spec's timeout factor (`limits_s`) — and
judged against the runway the model points at, not the observed one. The executor is deterministic and the single-flight
executor flies the torch executor's flight to round-off, so the flight is the exported sample's own, which the frontend
checks point for point. A sentence spoken under the procedure's altitudes
ends where its free generation ended it: at the first flown step whose end state sank more than the track tolerance below
the glidepath lower edge of the runway in force (`glidepath_stop`, the readout's rule read on the flown record, as the
readout reads it — after the flight); the flight is cut there and its answer's outcome is `BELOW_GLIDEPATH`.

A model's sentence spoken from an AUGMENTED start (`ModelSentence.augmentation`, its overlay's
`prior-generation-augmented` flight) is flown from that start, as its free generation flew it
(`prior_free_generation.prior_rows` with augmentations): the executor's state at the first step moved by
`prior.augment.augment_state` (`moved_inputs`), the observed rows the sentence is read against moved by
`augment_signals`, and the time limit the augmented one (× `augment.TIMEOUT_FACTOR`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs, rebuild_series
from ts_transformer.autopilot.frame import ALT, LAT, LON, PSI, SPEED
from ts_transformer.autopilot.judge import Verdict, flown_track, judge, read_flown
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.runway_data import VerticalPath, published_vertical_paths
from ts_transformer.autopilot.sentence import row_at
from ts_transformer.data.dataset import FlightSeries, series_from_row
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.labeller.read import Reading, admit, read_flight
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.training_files import require_stored_sentence, runway_hae_minus_msl_m, stored_sentence
from ts_transformer.instructions.words import RUNWAY, UNCHANGED, Words
from ts_transformer.outputs.dynamics.context import rollout_context
from ts_transformer.prior.augment import TIMEOUT_FACTOR, Augmentation, augment_signals, augment_state
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.procedure import RunwayProcedure, below_floor
from ts_transformer.repo_layout import arrival_manifest_path

from ts_transformer.autopilot import single
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
    vertical_paths: tuple[VerticalPath, ...]
    group: str
    approach_ias_mps: float
    observed_track_deg: np.ndarray     # the labeller's smoothed track of the observed flight (the heading chart's)
    observed_distance_m: np.ndarray    # and its smoothed distance flown (the altitude chart's axis)
    hae_minus_msl_m: float             # its runway's HAE − MSL offset (`training_files.runway_hae_minus_msl_m`)
    aero_params: np.ndarray            # its airframe's aero row as the executor flies it (`rollout_context`, as
                                       # `flight_inputs` reads it): what the attitude's attack reading reads


@dataclass(frozen=True)
class SplitFiles:
    """An artefact split's stored files as every set drawn from it reads them (`read_split`): its signals and where each
    dataset id sits among them, its stored sentences and the signal each is of, and the candidates."""

    split: str
    signals: list[FlightSignals]
    positions: dict[str, list[int]]
    sentences: dict[str, np.ndarray]
    stored: dict[int, int]            # signal index → stored sentence index
    geometries: dict[str, AirportGeometry]


def read_split(artefact: Path, split: str, words: Words) -> SplitFiles:
    """``artefact``'s ``split``: the whole split's signals (~0.8 s for val — thousands of flights, of which a set takes
    forty), its sentences read with ``words``' spec, and the candidates."""
    signals = load_signals(artefact, split)
    positions: dict[str, list[int]] = {}
    for index, item in enumerate(signals):
        positions.setdefault(item.dataset_id, []).append(index)
    sentences = load_sentences(artefact, split, words.spec)
    return SplitFiles(split=split, signals=signals, positions=positions, sentences=sentences,
                      stored={int(value): k for k, value in enumerate(sentences["signal_index"])},
                      geometries=load_candidates(artefact))


def open_flights(artefact: Path, files: SplitFiles, dataset_ids: Sequence[str], words: Words,
                 vertical_paths: Callable[[AirportGeometry], tuple[VerticalPath, ...]]
                 ) -> dict[str, FlightContext | NotFlyable | ValueError]:
    """Labelled flights of ``artefact`` (a Training set's; ``files``: its split's, `read_split`), each with its stored
    signals, the flight rebuilt from the data plane and checked against them (`rebuild_series`), the labeller's
    reading, which must give the stored sentence, and the offset that gives its heights back as the aircraft reported
    them — what they share read once: each airport's published vertical paths (``vertical_paths``, the backend's kept
    ones) and height offsets, and one `rebuild_series` over them all (one read of each arrival manifest). Each flight
    answers for itself, as one opened alone did: a flight the data cannot fly is its `NotFlyable`, one a check refuses
    its `ValueError` — the others open (`rebuilt_each`)."""
    spec = words.spec
    signals, positions, sentences, stored, geometries = (files.signals, files.positions, files.sentences, files.stored,
                                                          files.geometries)
    contexts: dict[str, FlightContext | NotFlyable | ValueError] = {}
    read: list[tuple[FlightSignals, Reading]] = []
    for dataset_id in dataset_ids:
        try:
            if dataset_id not in positions or len(positions[dataset_id]) != 1:
                raise ValueError(f"{dataset_id} is not a labelled {files.split} flight of {artefact.name}")
            (index,) = positions[dataset_id]
            if index not in stored:
                raise ValueError(f"{dataset_id} has no stored sentence in {artefact.name}")
            flight = signals[index]
            reading = read_flight(flight, geometries[flight.airport], spec, words)
            require_stored_sentence(dataset_id, reading, stored_sentence(sentences, stored[index]))
        except ValueError as error:
            contexts[dataset_id] = error
            continue
        read.append((flight, reading))
    airports = sorted({flight.airport for flight, _ in read})
    paths = {airport: vertical_paths(geometries[airport]) for airport in airports}
    offsets = {airport: runway_hae_minus_msl_m(artefact, airport, arrival_manifest_path(airport)) for airport in airports}
    for (flight, reading), rebuilt in zip(read, rebuilt_each(artefact, [flight for flight, _ in read])):
        if isinstance(rebuilt, ValueError):
            contexts[flight.dataset_id] = rebuilt
            continue
        group = replay.group_of(rebuilt)
        if group not in (replay.OWN, replay.STAND_IN):
            contexts[flight.dataset_id] = NotFlyable(f"{flight.dataset_id} cannot be flown: {group}")
            continue
        geometry = geometries[flight.airport]
        try:
            observed = admit(flight, geometry, spec)
        except ValueError as error:
            contexts[flight.dataset_id] = error
            continue
        contexts[flight.dataset_id] = FlightContext(
            signals=flight, series=rebuilt, reading=reading, geometry=geometry, vertical_paths=paths[flight.airport],
            group=group, approach_ias_mps=replay.flight_approach_ias_mps(rebuilt, group),
            observed_track_deg=observed.smoothed.track_deg, observed_distance_m=observed.smoothed.distance_m,
            hae_minus_msl_m=offsets[flight.airport][flight.runway], aero_params=rollout_context(rebuilt, 0)["aero_params"])
    return contexts


def rebuilt_each(artefact: Path, flights: list[FlightSignals]) -> list[FlightSeries | ValueError]:
    """Each flight rebuilt (`rebuild_series`): all in one call — one read of each arrival manifest — unless that call
    refuses one of them; then each alone, so the flight that fails answers for its refusal and the rest are flown."""
    try:
        return list(rebuild_series(artefact, flights))
    except ValueError:
        pass
    each: list[FlightSeries | ValueError] = []
    for flight in flights:
        try:
            each.extend(rebuild_series(artefact, [flight]))
        except ValueError as error:
            each.append(error)
    return each


def fly_until(executor: single.SingleExecutor, sentence: single.Sentence,
              clock: single.TimeClock | single.DistanceClock | single.TrackClock, step_s: float,
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
            if stop_steps is not None and int(row_at(sentence_s, step_s)) >= stop_steps:
                return True
            step_start_s = sentence_s
        executor.cycle(sentence.at(step_start_s), sentence_s)
        if executor.done:
            break
    return False


#: The word clock a model's sentence is flown on, whatever the spec's (`ExecutorParams.word_clock`): its free generation's,
#: every word heard at its own step (`single.TimeClock`, `fly_one_until`).
MODEL_WORD_CLOCK = "time"


def model_time_limit_s(context: FlightContext, first_row: int, params: ExecutorParams, step_s: float,
                       augmented: bool = False) -> float:
    """A model's flight's time limit: the observed flight's remaining time from the sentence's first step × the spec's
    timeout factor — or, from an ``augmented`` start, × `augment.TIMEOUT_FACTOR`. MIRROR of
    `experiments.prior_free_generation.limits_s` (a runner the backend does not import; pinned both ways by
    `test_autopilot_segment.ModelSegmentTest`)."""
    return (len(context.reading.words) - first_row) * step_s * (TIMEOUT_FACTOR if augmented else params.timeout_factor)


def moved_inputs(inputs: FlightInputs, geometry: AirportGeometry, augmentation: Augmentation) -> FlightInputs:
    """One flight's executor state at its first step moved like its augmented start (`augment_state`). MIRROR of
    `experiments.prior_free_generation.augmented_inputs` for one flight (pinned by `test_autopilot_segment.
    AugmentedStartTest`)."""
    state = inputs.initial_state.clone()
    row = state[0]
    moved = augment_state(float(row[LAT]), float(row[LON]), float(row[ALT]), float(row[SPEED]), float(row[PSI]),
                          geometry, augmentation)
    state[0, [LAT, LON, ALT, SPEED, PSI]] = torch.tensor(moved, dtype=state.dtype)
    return replace(inputs, initial_state=state)


def fly_one_until(context: FlightContext, segment: Segment, reading: Reading, signals: FlightSignals,
                  params: ExecutorParams, words: Words, stop_steps: int | None, superseded: Callable[[], bool], *,
                  model_limit_s: float | None = None, augmentation: Augmentation | None = None) -> tuple[Flown, bool, float]:
    """The segment's sentence (``reading``, the observed rows ``signals`` from its first step) flown from the observed
    state at its first step to ``stop_steps`` (or its outcome): the flown record, whether it stopped there, and the
    executor's own wall time. Set up as `replay.fly_sentences` sets up `executor.fly` — its time limit, runways, chart,
    approach speed and word clock — and stepped here (`fly_until`). ``model_limit_s``: a model's sentence, flown as its
    free generation flew it — its time limit, and every word heard at its own step (the time clock) — None for the
    truth; ``augmentation``: a model's sentence spoken from an augmented start, flown from the state moved like it."""
    spec = words.spec
    limit_s = len(reading.words) * spec.step_s * params.timeout_factor if model_limit_s is None else model_limit_s
    inputs = flight_inputs([series_from_row(context.series, segment.start_row)], device=DEVICE)
    if augmentation is not None:
        inputs = moved_inputs(inputs, context.geometry, augmentation)
    executor = single.SingleExecutor(inputs, context.geometry, context.vertical_paths, context.approach_ias_mps, params,
                                     words, time_limit_s=limit_s)
    rows = len(reading.words)
    clock = (single.word_clock(params, signals.e_m[:rows], signals.n_m[:rows], spec.step_s) if model_limit_s is None
             else single.TimeClock(params.cycle_s))
    started = time.perf_counter()
    reached = fly_until(executor, single.Sentence(reading.words, words), clock, spec.step_s, stop_steps, superseded)
    fly_s = time.perf_counter() - started
    flown = executor.flown()
    if reached:
        # stopped by the segment, not done by the executor: it ends with the last cycle flown
        flown = replace(flown, done_cycle=torch.full_like(flown.done_cycle, executor.count - 1))
    return flown, reached, fly_s


#: MIRROR of `experiments.prior_free_generation.BELOW_GLIDEPATH` (a runner the backend does not import; pinned by
#: `test_autopilot_segment.GlidepathStopTest`): a model's flight cut at the glidepath lower edge.
BELOW_GLIDEPATH = "below_glidepath"


def in_force(grid: np.ndarray) -> np.ndarray:
    """``[N, 6]``: each column's word in force at each row of a said grid, whose first row says every column (refused
    otherwise). MIRROR of `experiments.prior_free_generation.in_force` (pinned by `GlidepathStopTest`)."""
    grid = np.asarray(grid, dtype=np.int64)
    if (grid[0] == UNCHANGED).any():
        raise ValueError("a sentence's first row says every column")
    last = np.maximum.accumulate(np.where(grid != UNCHANGED, np.arange(len(grid))[:, None], 0), axis=0)
    return np.take_along_axis(grid, last, axis=0)


def glidepath_stop(flown: Flown, grid: np.ndarray, geometry: AirportGeometry, finals: tuple[RunwayProcedure, ...],
                   words: Words) -> int:
    """The first flown step whose end state is more than the track tolerance below the glidepath lower edge of the
    runway in force during it (-1: none) — one flight's `experiments.prior_free_generation.glidepath_stops`, a MIRROR
    (pinned by `GlidepathStopTest` flight for flight): the state after step k is the cycle boundary (k + 1) × the step's
    cycles, up to the one the flight ended at; the words step k flies are the row heard at its first cycle."""
    step_rows = round(words.spec.step_s / flown.cycle_s)
    boundaries = np.arange(step_rows, int(flown.done_cycle[0]) + 2, step_rows)
    track = flown_track(flown.states[0, boundaries].cpu().numpy(), geometry)
    force = in_force(grid)
    heard = np.minimum(row_at(flown.sentence_s[0, boundaries - step_rows].cpu().numpy(), words.spec.step_s), len(force) - 1)
    runway = force[heard, RUNWAY]
    below = np.zeros(len(boundaries), dtype=bool)
    for pointer in np.unique(runway):
        steps = runway == pointer
        below[steps], _ = below_floor(finals[pointer], track["e"][steps], track["n"][steps], track["height"][steps],
                                      words.spec)
    return int(np.flatnonzero(below)[0]) if below.any() else -1


def cut_at_step(flown: Flown, step: int, words: Words) -> Flown:
    """``flown`` over when its step ``step`` ended: the state after it (``states[(step + 1) × the step's cycles]``) is its
    last — ``done_cycle`` is the cycle at whose end the flight was done, whose state follows it (`executor.Flown`)."""
    step_rows = int(round(words.spec.step_s / flown.cycle_s))
    return replace(flown, done_cycle=torch.full_like(flown.done_cycle, (step + 1) * step_rows - 1))


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
    # a model's sentence spoken under the procedure's altitudes: the step the glidepath lower edge stopped it at, from
    # its first step (the flight is cut there); None: not stopped
    glidepath_step: int | None = None


def fly_segment(context: FlightContext, params: ExecutorParams, words: Words, column: int, row: int,
                superseded: Callable[[], bool], model: ModelSentence | None = None,
                procedure_masks: ProcedureMasks | None = None) -> FlownSegment:
    """``column``'s word said at ``row`` flown and judged: the truth's from the observed state there; a model's
    (``model``) as its sentence was flown, from its first step (`segment.model_segment`), under the procedure's masks it
    was spoken under (``procedure_masks``, built for the request: under the procedure's altitudes the flight is cut at
    the glidepath lower edge's stop)."""
    if (model is None) != (procedure_masks is None):
        raise ValueError("a model's sentence is flown with the procedure's masks it was spoken under, the truth's without")
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
        # from an augmented start, the observed rows as the model read them: moved like its start
        observed = context.signals if model.augmentation is None else augment_signals(context.signals, model.augmentation)
        reading = model_reading(observed, segment, model, words)
        # the judge reads the flown track against the runway the MODEL points at (the observed rows name the observed one)
        signals = replace(segment_signals(observed, segment),
                          runway=context.geometry.candidates[reading.runway_index].ident)
        limit_s = model_time_limit_s(context, model.first_row, params, spec.step_s, augmented=model.augmentation is not None)
    flown, reached, fly_s = fly_one_until(context, segment, reading, signals, params, words,
                                          None if segment.to_landing else segment.stop_row - segment.start_row,
                                          superseded, model_limit_s=limit_s,
                                          augmentation=None if model is None else model.augmentation)
    glidepath_step = None
    if procedure_masks is not None and procedure_masks.altitudes:
        stop = glidepath_stop(flown, segment.grid, context.geometry, procedure_masks.finals[context.geometry.code], words)
        if stop >= 0:
            # the free generation's end: the flight is over at that step's end state, whatever the executor flew after
            # (a stopped sample's sentence ends at its stop step, so only a segment flown to its outcome gets here)
            glidepath_step, flown = stop, cut_at_step(flown, stop, words)
    started = time.perf_counter()
    verdict = judge(flown, 0, context.geometry, reading.runway_index, judged_reading(reading, segment), signals, spec, words)
    stop_row = None if glidepath_step is None else int(flown.done_cycle[0]) + 1
    if glidepath_step is not None and (verdict.outcome, verdict.end_row) != ("timeout", stop_row):
        # the free generation's sample ends at the stop whatever came before it; the judge read an event first (an
        # uncaptured crossing or a stall the executor flew on past): the word cannot be judged as the sample ended
        raise RequestRefused(f"the judge ended the model's flight at {verdict.outcome} ({verdict.end_row * flown.cycle_s:g} s "
                             f"after step {segment.start_row}) before the glidepath lower edge stopped its sample at step "
                             f"{segment.start_row + glidepath_step}")
    # a word said as or after the flight ended has nothing flown after it — the sentence bar offers it no Fly (its step's
    # time is not before the sample's end); at a crossing or failure the judge does not even read its step
    if model is not None and verdict.end_row <= segment.word_step * int(round(spec.step_s / flown.cycle_s)):
        raise RequestRefused(f"the model's flight had ended ({verdict.outcome}, {verdict.end_row * flown.cycle_s:g} s after "
                             f"step {segment.start_row}) by the time it said this word at step {row}")
    if model is not None and verdict.words is not None:
        # the judge reads the flown track through the labeller's gate, which cuts it at a landing passage it finds on the
        # sentence's 2 s rows — which can come before the outcome read every cycle: a word past the cut was never judged
        read = read_flown(flown, 0, verdict.outcome, verdict.end_row, context.geometry, signals, spec).signals.n_rows
        if segment.word_step >= read:
            raise RequestRefused(f"the labeller's gate cuts the model's flown track at a landing passage after step "
                                 f"{segment.start_row + read - 1}, before it said this word at step {row}")
    return FlownSegment(segment=segment, reading=reading, signals=signals, model=model, flown=flown, verdict=verdict,
                        reached_end=None if segment.to_landing else reached, fly_s=fly_s,
                        judge_s=time.perf_counter() - started, glidepath_step=glidepath_step)
