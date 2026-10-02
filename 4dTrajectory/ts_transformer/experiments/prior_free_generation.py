"""Single-aircraft free generation (prior design §9.1): the prior speaks, the executor flies — step by step — and beside
it the executor flies the labelled words from the same row (the reference).

Each drawn flight (its own dynamics, the replay gate's group) is flown from the observed state at the prior's first
predicted step (row `prior.scene.N_LOOK`):

- ``--samples`` times with the prior speaking (`prior.generate.Speaker`): at every 2 s step it samples the step's six
  words from the positions so far — observed up to the first predicted step, then where the executor flew — the
  airport's landings before the step and the words it has said; the executor hears them at once and flies the step
  (`autopilot.executor.Executor`, two 1 s cycles; `autopilot.sentence.Spoken`);
- once with the labelled words: the words in force at the first predicted step, then the sentence as written, said on
  the executor spec's clock over the observed rows from that row (as the replay gate says them).

Every flight ends as the executor's judge reads it (`autopilot.judge.outcome_of`) — landed on the runway pointed at the
end, crossed without capture, crossed off the runway, ground contact, dynamics failure, or timeout (the observed
remaining time × the spec's timeout factor) — and the readout counts them in all, per airport and per airport ×
approach kind, with the runways pointed (first and last), the time to the end against the observed remaining time,
the words said per column, runway changes and go-arounds. Writes ``generation.json`` and ``sentences.npz`` (the words
the prior said) into a NEW directory; the val split only from a clean tree.

The prior speaks under the vocabulary's rules and the procedure's masks it was trained under (`prior.masks`, prior
design §5.1; its directory records them, `prior_train.load_prior` reads them) — ``--procedure-masks own``, the default;
``none`` or a comma-separated list of sets (`masks.SETS`) speaks under others instead, and the log says so. Under the
procedure's altitudes (post-training design §3) — the glidepath lower edge inside the FAF, the decision altitude and no
climbing back before the join — the prior's altitude and descent-angle columns are masked (`prior.generate.Speaker`),
and every sentence — the prior's and the labelled one — ends at the first flown step more than the track tolerance below
the glidepath lower edge (`glidepath_stops`), its outcome `BELOW_GLIDEPATH` whatever the executor made of the rest; the
readout records ``procedure_masks`` true. The check reads the flown states at the step boundaries after the flight, which is where an
in-loop check would have stopped it: what a stopped flight flies afterwards is discarded. Each of the prior's sentences
and each flight's observed track are read before the join (`procedure.pre_join_readout`): under the DA, climbing back
after the dip, under the MVA (`prior.mva`, a readout only).

``--augment-seed`` (post-training design §4): every drawn flight is flown from an augmented start instead of its own
(`prior.augment`: rotated about the airport, raised, sped up — one augmentation a flight, drawn with that seed until it is
plausible, `augmented_starts`); a flight with no plausible draw in `AUGMENT_TRIES` is left out and counted. The labelled
words are not flown then: they belong to the source flight's own start. An augmented start's time limit is its source's
observed remaining time × `augment.TIMEOUT_FACTOR` (design §4.5).

    python run_ts.py prior_free_generation --prior 4dTrajectory/outputs/POOLED/prior/<campaign>/<chosen run> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v4_20260924 \\
        --executor 4dTrajectory/outputs/POOLED/executor/<spec dir> --split select --per-airport 100 \\
        --out 4dTrajectory/outputs/POOLED/prior/<campaign>/<name>
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch

from aerodynamic_model.torch_dynamics import isa_density
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Executor, Flown, fly
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs
from ts_transformer.autopilot.frame import ALT, LAT, LON, MASS, PSI, SPEED, AirportCharts
from ts_transformer.autopilot.judge import OUTCOMES, flown_track, outcome_of
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
from ts_transformer.autopilot.sentence import DistanceClock, Sentences, Spoken, TimeClock, TrackClock, row_at
from ts_transformer.experiments.prior_train import load_prior, rosters
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import SPLITS, load_candidates, load_signals
from ts_transformer.instructions.labeller.read import Reading
from ts_transformer.instructions.readout import STRATA, flight_record
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import (
    APPROACH, APPROACH_CLEARED, APPROACH_GO_AROUND, COLUMNS, RUNWAY, UNCHANGED, Words,
)
from ts_transformer.prior import augment, mva
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.outputs.constraints.speed_floor import stall_speed_mps
from ts_transformer.prior.augment import Augmentation, augment_signals, augment_state, draw
from ts_transformer.prior.data import VARIANTS, airport_landings
from ts_transformer.prior.generate import Speaker, rows_for
from ts_transformer.prior.model import Prior
from ts_transformer.prior.masks import SETS, ProcedureMasks
from ts_transformer.prior.mva import MvaChart, airport_charts
from ts_transformer.prior.procedure import RunwayProcedure, below_floor, pre_join_readout
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import REPO_ROOT, git_state

GENERATION_SCHEMA = "ts-prior-free-generation-v5"
#: The outcome of a sentence that took its aircraft below the glidepath lower edge (post-training design §3.5): not the
#: executor's judge's — the runner's, over the judge's outcomes.
BELOW_GLIDEPATH = "below_glidepath"
FREE_OUTCOMES = (*OUTCOMES, BELOW_GLIDEPATH)


def limits_s(batch: replay.Batch, params: ExecutorParams, step_s: float, *, augmented: bool) -> list[float]:
    """Each flight's time limit from the first predicted step: its sentence's remaining rows × the step × the timeout
    factor — the executor spec's (the replay gate's convention, `replay.fly_sentences`), or an ``augmented`` start's
    (`augment.TIMEOUT_FACTOR`, design §4.5)."""
    factor = augment.TIMEOUT_FACTOR if augmented else params.timeout_factor
    return [(len(r.words) - N_LOOK) * step_s * factor for r in batch.readings]


def observed_remaining_s(reading: Reading, step_s: float) -> float:
    """From the first predicted step to the observed landing: the sentence's last row is at the landing."""
    return (len(reading.words) - 1 - N_LOOK) * step_s


def reference_grid(grid: np.ndarray) -> np.ndarray:
    """The labelled sentence from the first predicted step: its first row every column's word in force there, then
    the sentence as written."""
    grid = np.asarray(grid, dtype=np.int64)
    out = grid[N_LOOK:].copy()
    for column in range(6):
        written = np.flatnonzero(grid[: N_LOOK + 1, column] != UNCHANGED)
        out[0, column] = grid[written[-1], column]
    return out


#: How many augmentations a start may draw before it is given up (post-training design §4.3).
AUGMENT_TRIES = 10


def start_altitude_windows(instructions: Path) -> dict[str, tuple[float, float]]:
    """Each airport's train-day flights' altitude at the first predicted step (row `N_LOOK`, where the executor starts):
    its 1st and 99th percentiles — what an augmented start's altitude there must lie between to be like the data's
    (§4.3; a check, never a model input)."""
    heights: dict[str, list[float]] = defaultdict(list)
    for signals in load_signals(instructions, "train"):
        heights[signals.airport].append(float(signals.altitude_m[N_LOOK]))
    return {code: (float(np.percentile(h, 1)), float(np.percentile(h, 99))) for code, h in sorted(heights.items())}


class AugmentedStarts(NamedTuple):
    """Each flight's augmentation (None: none of `AUGMENT_TRIES` draws was plausible) and how many draws it took."""

    moves: list[Augmentation | None]
    draws: list[int]


def augmented_starts(signals: Sequence[FlightSignals], inputs: FlightInputs, rng: np.random.Generator,
                     windows: dict[str, tuple[float, float]]) -> AugmentedStarts:
    """One augmentation a flight (§4.2; ``inputs``: the executor's states at the first predicted step, `flight_inputs`),
    drawn until the moved start is plausible (§4.3) — its altitude at the first predicted step inside its airport's
    ``windows``, its airspeed above the executor's stall floor at its mass and the new altitude — at most
    `AUGMENT_TRIES` draws."""
    out = AugmentedStarts([], [])
    for j, flight in enumerate(signals):
        low, high = windows[flight.airport]
        state, aero = inputs.initial_state[j], inputs.aero_params[j]
        chosen, tries = None, 0
        while chosen is None and tries < AUGMENT_TRIES:
            augmentation = draw(rng)
            tries += 1
            altitude = float(state[ALT]) + augmentation.altitude_m
            floor = EXECUTOR_DYNAMICS.control_speed_floor_margin * float(stall_speed_mps(
                1.0, state[MASS], isa_density(torch.tensor(altitude, dtype=torch.float64)), aero[0], aero[1]))
            if low <= float(flight.altitude_m[N_LOOK]) + augmentation.altitude_m <= high \
                    and float(state[SPEED]) * augmentation.speed_scale >= floor:
                chosen = augmentation
        out.moves.append(chosen)
        out.draws.append(tries)
    return out


def augmented_inputs(inputs: FlightInputs, geometries: Sequence[AirportGeometry],
                     augmentations: Sequence[Augmentation | None]) -> FlightInputs:
    """The executor's states at the first predicted step moved by each flight's augmentation (`augment_state`); a flight
    whose augmentation is None keeps its own."""
    state = inputs.initial_state.clone()
    for j, (geometry, augmentation) in enumerate(zip(geometries, augmentations)):
        if augmentation is None:
            continue
        row = state[j]
        moved = augment_state(float(row[LAT]), float(row[LON]), float(row[ALT]), float(row[SPEED]), float(row[PSI]),
                              geometry, augmentation)
        state[j, [LAT, LON, ALT, SPEED, PSI]] = torch.tensor(moved, dtype=state.dtype)
    return replace(inputs, initial_state=state)


def _physics(batch: replay.Batch, device: torch.device) -> tuple[Runways, AirportCharts, torch.Tensor]:
    f64 = torch.float64
    return (Runways.of(batch.geometries, batch.vertical_paths, dtype=f64, device=device),
            AirportCharts.of(batch.geometries, dtype=f64, device=device),
            torch.tensor(batch.approach_ias_mps, dtype=f64, device=device))


class ClosedLoop:
    """A batch of flights flown from their first predicted step (``inputs``: the executor's state there) with the prior
    speaking, a step at a time (`step`): the speaker reads where the executor is, says the step's words, and the
    executor flies the step (its cycles). A flight the executor is done with hears nothing more and its row is frozen;
    one it has cleared or captured keeps its runway. Each flight flies until the executor is done with it or its time
    limit (``limits``, seconds). ``procedure_masks``: the procedure's masks the speaker speaks under (`prior.masks`)."""

    def __init__(self, model: Prior, flights: Sequence[FlightSignals], geometries: Sequence[AirportGeometry],
                 inputs: FlightInputs, runways: Runways, charts: AirportCharts, approach_ias_mps: torch.Tensor,
                 limits: Sequence[float], words: Words, params: ExecutorParams, landings: Any, *,
                 generator: torch.Generator, temperature: float, procedure_masks: ProcedureMasks) -> None:
        step_s, device = words.spec.step_s, inputs.initial_state.device
        self.step_s, self.params, self.device = step_s, params, device
        self.executor = Executor(inputs, runways, charts, approach_ias_mps, params, words,
                                 time_limit_s=torch.tensor(limits, dtype=torch.float64, device=device))
        self.speaker = Speaker(model, flights, geometries, landings, words,
                               max_rows=rows_for(max(limits) + step_s, step_s), generator=generator,
                               procedure_masks=procedure_masks, temperature=temperature)
        self.spoken = Spoken(len(limits), words, device=device)
        self.max_steps = rows_for(max(limits), step_s) - N_LOOK

    @property
    def steps(self) -> int:
        return self.spoken.steps

    @property
    def running(self) -> bool:
        executor = self.executor
        return self.steps < self.max_steps and not (bool(executor.done.all()) or executor.count == executor.cycles)

    def step(self) -> None:
        executor, speaker, count = self.executor, self.speaker, len(self.executor.done)
        done = executor.done.cpu().numpy()
        if self.steps:
            now = executor.now()
            speaker.append(now.e_m.cpu().numpy(), now.n_m.cpu().numpy(), now.height_m.cpu().numpy(), frozen=done)
        # a flight that is done hears nothing more; one the executor has cleared or captured keeps its runway
        said = speaker.speak(active=~done, runway_locked=executor.runway_locked.cpu().numpy())
        heard = torch.full((count,), self.steps * self.step_s, dtype=torch.float64, device=self.device)
        self.spoken.say(np.where(said > 0, said - 1, UNCHANGED))
        for _ in range(executor.step_rows):
            if executor.count == executor.cycles:
                break
            executor.cycle(self.spoken.at(heard), torch.full((count,), executor.count * self.params.cycle_s,
                                                             dtype=torch.float64, device=self.device))


def speak_and_fly(model: Prior, flights: Sequence[FlightSignals], geometries: Sequence[AirportGeometry],
                  inputs: FlightInputs, runways: Runways, charts: AirportCharts, approach_ias_mps: torch.Tensor,
                  limits: Sequence[float], words: Words, params: ExecutorParams, landings: Any, *,
                  generator: torch.Generator, temperature: float, procedure_masks: ProcedureMasks
                  ) -> tuple[Flown, np.ndarray, dict[int, np.ndarray], Speaker]:
    """`ClosedLoop` run to its end: ``(what was flown, the words said [B, steps, 6] with UNCHANGED where a column says
    nothing, the probability the prior put on what the masks removed [B, steps] per masked column, the speaker — the
    rows it read)``."""
    loop = ClosedLoop(model, flights, geometries, inputs, runways, charts, approach_ias_mps, limits, words, params,
                      landings, generator=generator, temperature=temperature, procedure_masks=procedure_masks)
    while loop.running:
        loop.step()
    return (loop.executor.flown(), loop.spoken.sentences(),
            {column: np.stack(masses, axis=1) for column, masses in loop.speaker.forbidden.items()}, loop.speaker)


def fly_reference(batch: replay.Batch, words: Words, params: ExecutorParams, *, device: torch.device) -> Flown:
    """Fly every flight's labelled words from its first predicted step, on the spec's clock over the observed rows from
    there."""
    step_s = words.spec.step_s
    grids = [reference_grid(r.words) for r in batch.readings]
    if params.word_clock == "time":
        clock: TimeClock | DistanceClock | TrackClock = TimeClock(params.cycle_s)
    else:
        rows = [len(r.words) for r in batch.readings]
        e_m = [f.e_m[N_LOOK:n] for f, n in zip(batch.signals, rows)]
        n_m = [f.n_m[N_LOOK:n] for f, n in zip(batch.signals, rows)]
        clock = (DistanceClock if params.word_clock == "distance" else TrackClock).of(e_m, n_m, step_s, params.cycle_s,
                                                                                    device=device)
    runways, charts, approach = _physics(batch, device)
    return fly(flight_inputs(batch.series, device=device, anchor=N_LOOK), Sentences(grids, words, device=device), clock,
               runways, charts, approach, params, words,
               time_limit_s=torch.tensor(limits_s(batch, params, step_s, augmented=False), dtype=torch.float64,
                                         device=device))


def steps_said(flown: Flown, index: int, steps: int, step_rows: int) -> int:
    """How many of the ``steps`` a flight's sentence said count: up to the step whose cycles ended the flight (a
    finished flight says nothing more)."""
    return min(steps, int(flown.done_cycle[index]) // step_rows + 1)


def in_force(grid: np.ndarray) -> np.ndarray:
    """``[N, 6]``: each column's word in force at each row of a said grid, whose first row says every column (refused
    otherwise). MIRROR of the value `autopilot.sentence._filled` gives, the executor's reading of a sentence — making
    that one public would move the executor's source hash and refuse every executor spec; `tests/test_prior_procedure.py`
    holds the two equal."""
    grid = np.asarray(grid, dtype=np.int64)
    if (grid[0] == UNCHANGED).any():
        raise ValueError("a sentence's first row says every column")
    last = np.maximum.accumulate(np.where(grid != UNCHANGED, np.arange(len(grid))[:, None], 0), axis=0)
    return np.take_along_axis(grid, last, axis=0)


class GlidepathStops(NamedTuple):
    """Where each sentence stopped (-1: it did not): the flown step whose end state sank below the edge, and the
    sentence row whose words that step flew. The two are one for the prior's sentences (a step says its row); a labelled
    sentence is heard on the spec's clock, where the row a step flies is where the observed aircraft heard it."""

    step: np.ndarray
    row: np.ndarray


def glidepath_stops(flown: Flown, grids: Sequence[np.ndarray], geometries: Sequence[AirportGeometry],
                    finals: Sequence[Sequence[RunwayProcedure]], words: Words) -> GlidepathStops:
    """Each sentence's first step whose flown end state is more than the track tolerance below the glidepath lower edge
    of the runway in force during it (post-training design §3.5). The state after step k is the cycle boundary (k + 1) ×
    the step's cycles, up to the one at which the executor was done with the flight; the words step k flies are the row
    heard at its first cycle (`fly`, `autopilot.sentence.row_at`), the last one past the sentence's end."""
    step_rows = round(words.spec.step_s / flown.cycle_s)
    stops = GlidepathStops(np.full(len(grids), -1, dtype=np.int64), np.full(len(grids), -1, dtype=np.int64))
    for j, grid in enumerate(grids):
        boundaries = np.arange(step_rows, int(flown.done_cycle[j]) + 2, step_rows)
        track = flown_track(flown.states[j, boundaries].cpu().numpy(), geometries[j])
        force = in_force(grid)
        heard = np.minimum(row_at(flown.sentence_s[j, boundaries - step_rows].cpu().numpy(), words.spec.step_s),
                           len(force) - 1)
        runway = force[heard, RUNWAY]
        below = np.zeros(len(boundaries), dtype=bool)
        for pointer in np.unique(runway):
            steps = runway == pointer
            below[steps], _ = below_floor(finals[j][pointer], track["e"][steps], track["n"][steps],
                                          track["height"][steps], words.spec)
        if below.any():
            k = int(np.flatnonzero(below)[0])
            stops.step[j], stops.row[j] = k, int(heard[k])
    return stops


def sentence_counts(grid: np.ndarray) -> dict[str, Any]:
    """What a said sentence (``grid``: [steps, 6], `UNCHANGED` where a column says nothing, the first step saying
    every column) did with its runway and its approach: the runway pointed first and last, how often it changed, the
    go-arounds said and whether it was cleared at its last step."""
    runway = grid[:, RUNWAY][grid[:, RUNWAY] != UNCHANGED]
    approach = grid[:, APPROACH][grid[:, APPROACH] != UNCHANGED]
    return {"first_runway": int(runway[0]), "last_runway": int(runway[-1]),
            "runway_changes": int((np.diff(runway) != 0).sum()),
            "go_arounds": int((grid[:, APPROACH] == APPROACH_GO_AROUND).sum()),
            "cleared_at_end": bool(approach[-1] == APPROACH_CLEARED)}


def flight_rows(batch: replay.Batch, flown: Flown, grids: Sequence[np.ndarray], words: Words, source: str,
                samples: Sequence[int | None], forbidden: dict[int, np.ndarray] | None,
                stops: GlidepathStops | None = None) -> list[dict[str, Any]]:
    """One row per flight: its outcome and what was said (``samples``: each flight's sample number, None for the
    labelled words; ``forbidden``: the prior's probability on what the masks forbid, per step, over the steps
    before the flight's end; ``stops``: `glidepath_stops`, None without the glidepath lower edge — a stopped sentence
    ends at the row its stopping step flew, its outcome `BELOW_GLIDEPATH`). The readouts before the join are the prior's
    sentences' alone, added by `prior_rows` under the procedure's masks."""
    rows = []
    step_rows = round(words.spec.step_s / flown.cycle_s)
    for j, (reading, grid, sample) in enumerate(zip(batch.readings, grids, samples)):
        stop = -1 if stops is None else int(stops.step[j])
        # the steps said up to the flight's end: the step whose cycles ended it (later ones said nothing), or the row
        # its stopping step flew
        steps = steps_said(flown, j, len(grid), step_rows) if stop < 0 else min(len(grid), int(stops.row[j]) + 1)
        grid = np.asarray(grid)[:steps]
        counts = sentence_counts(grid)
        outcome = outcome_of(flown, j, batch.geometries[j], counts["last_runway"], words.spec)
        said_after_first = (grid[1:] != UNCHANGED).sum(axis=0)
        rows.append({
            "dataset_id": batch.signals[j].dataset_id, "airport": batch.signals[j].airport,
            "stratum": flight_record(reading)["stratum"], "source": source, "sample": sample,
            "outcome": outcome.outcome if stop < 0 else BELOW_GLIDEPATH,
            "end_s": outcome.end_row * flown.cycle_s if stop < 0 else (stop + 1) * words.spec.step_s,
            "observed_remaining_s": observed_remaining_s(reading, words.spec.step_s),
            "observed_runway": reading.runway_index, "first_runway": counts["first_runway"],
            "last_runway": counts["last_runway"], "runway_changes": counts["runway_changes"], "steps_said": len(grid),
            "go_arounds": counts["go_arounds"], "cleared_at_end": counts["cleared_at_end"],
            "forbidden_mass": ({COLUMNS[c]: float(mass[j, :steps].mean()) for c, mass in forbidden.items()}
                               if forbidden is not None else None),
            "words_after_first": {name: int(said_after_first[c]) for c, name in enumerate(COLUMNS)},
        })
    return rows


def _mva(chart: MvaChart, geometry: AirportGeometry, e: np.ndarray, n: np.ndarray) -> np.ndarray:
    lat, lon = geometry.frame.latlon_from_horizontal(np.asarray(e), np.asarray(n))
    return chart.at(np.asarray(lon), np.asarray(lat))


def sentence_pre_join(grid: np.ndarray, steps: int, e: np.ndarray, n: np.ndarray, h: np.ndarray,
                      finals: Sequence[RunwayProcedure], chart: MvaChart, geometry: AirportGeometry,
                      words: Words) -> dict[str, Any]:
    """A sentence's readouts before the join (`procedure.pre_join_readout`) over the rows the speaker read: the observed
    ones up to the first predicted step, then one a step it said (``steps``); a row's words are the ones in force as
    its step was said (the first step's before it)."""
    rows = N_LOOK + steps
    force = in_force(np.asarray(grid)[:steps])[np.clip(np.arange(rows) - N_LOOK, 0, steps - 1)]
    return pre_join_readout(finals, force[:, RUNWAY], force[:, APPROACH], e[:rows], n[:rows], h[:rows],
                            _mva(chart, geometry, e[:rows], n[:rows]), N_LOOK, words.spec)


def observed_pre_join(signals: FlightSignals, reading: Reading, finals: Sequence[RunwayProcedure], chart: MvaChart,
                      geometry: AirportGeometry, words: Words) -> dict[str, Any]:
    """The observed track's readouts before the join (`procedure.pre_join_readout`): its labelled sentence's rows
    (contract C30: the signals' first rows), the labelled words in force at each."""
    rows = len(reading.words)
    force = in_force(reading.words)
    e, n = signals.e_m[:rows], signals.n_m[:rows]
    return pre_join_readout(finals, force[:, RUNWAY], force[:, APPROACH], e, n, signals.altitude_m[:rows],
                            _mva(chart, geometry, e, n), N_LOOK, words.spec)


def procedure_masks_named(choice: str, own: ProcedureMasks, geometries: dict[str, AirportGeometry]) -> ProcedureMasks:
    """The procedure's masks a readout speaks under: ``own`` (the model's), ``none``, or the comma-separated sets named."""
    if choice == "own":
        return own
    return ProcedureMasks.build(() if choice == "none" else choice.split(","), geometries)


def mva_charts(procedure_masks: ProcedureMasks, codes: Sequence[str]) -> dict[str, MvaChart]:
    """The airports' MVA charts (`mva.airport_charts`) the readouts before the join read — under the procedure's
    altitudes; none without them."""
    return airport_charts(sorted(codes)) if procedure_masks.altitudes else {}


def said_rows(batch: replay.Batch, flown: Flown, said: np.ndarray, forbidden: dict[int, np.ndarray], words: Words,
              samples: Sequence[int], procedure_masks: ProcedureMasks
              ) -> tuple[list[dict[str, Any]], list[np.ndarray], GlidepathStops | None]:
    """What `speak_and_fly` said under ``procedure_masks``, read: each sentence's `flight_rows` row, its words, and —
    under the procedure's altitudes — where the glidepath lower edge stopped it (`glidepath_stops`; nothing it said after
    the stop counts: the speaker went on). The one reading of a free sentence: the formal readout (`prior_rows`) and the
    Training export (`prior_generation_training_export`) both read theirs here."""
    grids = [said[j] for j in range(len(said))]
    stops = None
    if procedure_masks.altitudes:
        stops = glidepath_stops(flown, grids, batch.geometries, [procedure_masks.finals[g.code] for g in batch.geometries],
                                words)
        for j in np.flatnonzero(stops.step >= 0):
            grids[j][stops.row[j] + 1:] = UNCHANGED
    return flight_rows(batch, flown, grids, words, "prior", samples, forbidden, stops), grids, stops


def prior_rows(model: Prior, batch: replay.Batch, words: Words, params: ExecutorParams, landings: Any,
               samples: int, *, generator: torch.Generator, temperature: float, procedure_masks: ProcedureMasks,
               charts: dict[str, MvaChart], augmentations: Sequence[Augmentation] | None = None
               ) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Every flight of ``batch`` flown ``samples`` times with the prior speaking under ``procedure_masks`` (the executor
    on CPU): its `flight_rows` and the sentences said. Under the procedure's altitudes, the glidepath lower edge's stop and
    the readouts before the join (``charts``: `mva_charts`); ``augmentations``: each flight's, flown from its augmented
    start (None: its own)."""
    cpu = torch.device("cpu")
    index = [j for j in range(len(batch.readings)) for _ in range(samples)]
    repeated = replay.subset(batch, index)
    observed = repeated.signals                          # the source flights' own tracks, before any augmentation
    inputs = flight_inputs(repeated.series, device=cpu, anchor=N_LOOK)
    if augmentations is not None:
        moves = [augmentations[j] for j in index]
        repeated = replace(repeated, signals=[augment_signals(s, a) for s, a in zip(repeated.signals, moves)])
        inputs = augmented_inputs(inputs, repeated.geometries, moves)
    runways, executor_charts, approach = _physics(repeated, cpu)
    limits = limits_s(repeated, params, words.spec.step_s, augmented=augmentations is not None)
    flown, said, forbidden, speaker = speak_and_fly(model, repeated.signals, repeated.geometries, inputs, runways,
                                                    executor_charts, approach, limits, words, params, landings,
                                                    generator=generator, temperature=temperature,
                                                    procedure_masks=procedure_masks)
    rows, grids, _ = said_rows(repeated, flown, said, forbidden, words, [j % samples for j in range(len(said))],
                               procedure_masks)
    if procedure_masks.altitudes:
        finals = [procedure_masks.finals[g.code] for g in repeated.geometries]
        for j, row in enumerate(rows):
            geometry, chart = repeated.geometries[j], charts[repeated.geometries[j].code]
            row["pre_join"] = sentence_pre_join(grids[j], row["steps_said"], speaker.e[j], speaker.n[j], speaker.h[j],
                                                finals[j], chart, geometry, words)
            row["pre_join_observed"] = observed_pre_join(observed[j], repeated.readings[j], finals[j], chart,
                                                         geometry, words)
    return rows, grids


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Shares of each outcome and the rest, over ``rows``."""
    count = len(rows)
    landed = [r for r in rows if r["outcome"] == "landed"]
    timeouts = [r for r in rows if r["outcome"] == "timeout"]
    delta = [r["end_s"] - r["observed_remaining_s"] for r in landed]
    return {
        "flights": count,
        "outcomes": {name: sum(r["outcome"] == name for r in rows) / count for name in FREE_OUTCOMES},
        "first_runway_observed": sum(r["first_runway"] == r["observed_runway"] for r in rows) / count,
        "landed_on_observed_runway": (sum(r["last_runway"] == r["observed_runway"] for r in landed) / len(landed)
                                      if landed else None),
        "landing_minus_observed_s": ({"p10": float(np.percentile(delta, 10)), "p50": float(np.percentile(delta, 50)),
                                      "p90": float(np.percentile(delta, 90))} if delta else None),
        "words_after_first_per_flight": {name: float(np.mean([r["words_after_first"][name] for r in rows]))
                                         for name in COLUMNS},
        "runway_changes_per_flight": float(np.mean([r["runway_changes"] for r in rows])),
        "go_around_flights": sum(r["go_arounds"] > 0 for r in rows) / count,
        "cleared_at_end": sum(r["cleared_at_end"] for r in rows) / count,
        "timeouts_cleared": (sum(r["cleared_at_end"] for r in timeouts) / len(timeouts) if timeouts else None),
        "forbidden_mass_per_step": ({column: float(np.mean([r["forbidden_mass"][column] for r in rows]))
                                     for column in rows[0]["forbidden_mass"]} if rows[0]["forbidden_mass"] else None),
        "pre_join": ({f"{side}_{key}": sum(r[field][key] for r in rows) / count
                      for side, field in (("said", "pre_join"), ("observed", "pre_join_observed"))
                      for key in PRE_JOIN_LINES} if "pre_join" in rows[0] else None),
    }


#: The readouts before the join counted in a summary (`procedure.pre_join_readout`): the shares past each line.
PRE_JOIN_LINES = ("under_decision", "climbed_after_dip", "under_mva")


def grouped(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """`summarise` in all, per airport, per airport × approach kind."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for key in ("all", r["airport"], f"{r['airport']} {r['stratum']}", r["stratum"]):
            groups[key].append(r)
    return {key: summarise(value) for key, value in sorted(groups.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the chosen prior run (prior_select)")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--per-airport", type=int, default=0, help="0: every flight of the split flown on its own dynamics")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=64, help="flights a batch (× samples closed loops)")
    parser.add_argument("--procedure-masks", default="own",
                        help=f"the procedure's masks the prior speaks under: own (the ones it was trained under), none, "
                             f"or a comma-separated list of {list(SETS)}; under the procedure's altitudes the glidepath "
                             f"lower edge also stops a sentence below it and the sentences are read before the join")
    parser.add_argument("--augment-seed", type=int, default=None,
                        help="fly every flight from an augmented start drawn with this seed (post-training design §4)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executor flies on CPU")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, executor_dir, out = map(resolved, (args.prior, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    if args.split == "val" and git["dirty"]:
        parser.error("the val readout runs from a clean tree")
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor_dir, instructions)
    model, _, prior_config, own_masks = load_prior(prior_dir, instructions)
    if prior_config["smoke"]:
        parser.error(f"{prior_dir} is a smoke run")
    geometries = load_candidates(instructions)
    procedure_masks = procedure_masks_named(args.procedure_masks, own_masks, geometries)
    print(f"the procedure's masks: {list(procedure_masks.names) or 'none'}"
          + (" (the model's own)" if procedure_masks.names == own_masks.names else
             f" — NOT the model's own ({list(own_masks.names) or 'none'})"), flush=True)
    model.to(torch.device(args.device))
    landings = (airport_landings(instructions, rosters(instructions))
                if VARIANTS[model.config.variant].landing_context else None)
    batch = replay.draw(instructions, args.split, words.spec, words, per_airport=args.per_airport, seed=args.seed)
    print(f"{len(batch.readings)} {args.split} flights ({batch.drawn['excluded']} not flown), "
          f"{time.perf_counter() - started:.0f}s", flush=True)
    augmentations: list[Augmentation] | None = None
    left_out = 0
    if args.augment_seed is not None:
        # a readout's flights are fixed: a flight with no plausible draw is left out and counted, not replaced
        moves = augmented_starts(batch.signals, flight_inputs(batch.series, device=torch.device("cpu"), anchor=N_LOOK),
                                 np.random.default_rng(args.augment_seed), start_altitude_windows(instructions)).moves
        kept = [j for j, move in enumerate(moves) if move is not None]
        left_out = len(moves) - len(kept)
        batch, augmentations = replay.subset(batch, kept), [moves[j] for j in kept]
        print(f"augmented starts: {len(kept)} ({left_out} with no plausible draw left out)", flush=True)
    sources = ("prior",) if augmentations is not None else ("labelled", "prior")

    charts = mva_charts(procedure_masks, list(geometries))
    cpu = torch.device("cpu")
    generator = torch.Generator(device=torch.device(args.device)).manual_seed(args.seed)
    order = sorted(range(len(batch.readings)), key=lambda j: len(batch.readings[j].words))
    rows: list[dict[str, Any]] = []
    sentences: list[np.ndarray] = []
    for start in range(0, len(order), args.chunk):
        chunk = order[start: start + args.chunk]
        part = replay.subset(batch, chunk)
        if augmentations is None:
            reference = fly_reference(part, words, params, device=cpu)
            labelled = [reference_grid(r.words) for r in part.readings]
            stops = (glidepath_stops(reference, labelled, part.geometries,
                                     [procedure_masks.finals[g.code] for g in part.geometries], words)
                     if procedure_masks.altitudes else None)
            rows += flight_rows(part, reference, labelled, words, "labelled", [None] * len(part.readings), None, stops)
        said, grids = prior_rows(model, part, words, params, landings, args.samples, generator=generator,
                                 temperature=args.temperature, procedure_masks=procedure_masks, charts=charts,
                                 augmentations=None if augmentations is None else [augmentations[j] for j in chunk])
        rows += said
        sentences += grids
        done = start + len(part.readings)
        print(f"  {done}/{len(order)} flights, {time.perf_counter() - started:.0f}s", flush=True)

    readout = {source: grouped([r for r in rows if r["source"] == source]) for source in sources}
    landed_samples = Counter(r["dataset_id"] for r in rows if r["source"] == "prior" and r["outcome"] == "landed")
    spread = Counter(landed_samples[s.dataset_id] for s in batch.signals)
    out.mkdir(parents=True)
    lengths = np.array([len(g) for g in sentences], dtype=np.int64)
    np.savez_compressed(out / "sentences.npz", offsets=np.concatenate(([0], np.cumsum(lengths))),
                        words=np.concatenate(sentences).astype(np.int16),
                        dataset_id=np.array([r["dataset_id"] for r in rows if r["source"] == "prior"]))
    write_json_atomic(out / "generation.json", {
        "schema": GENERATION_SCHEMA, "written_utc": utc_now(), "git": git,
        "prior": {"directory": str(prior_dir), "variant": model.config.variant},
        "executor": {"directory": str(executor_dir), "sha256": record["sha256"]},
        "instructions": str(instructions), "split": args.split, "drawn": batch.drawn, "n_look": N_LOOK,
        "samples": args.samples, "temperature": args.temperature, "seed": args.seed,
        "procedure_masks": procedure_masks.altitudes,
        "timeout_factor": {"real": params.timeout_factor, "augmented": augment.TIMEOUT_FACTOR},
        "mva": {"charts_date": mva.CHARTS_DATE, "chart": mva.CHART, "facility": mva.FACILITY},
        "augment_seed": args.augment_seed, "augmented_left_out": left_out,
        "augmentations": None if augmentations is None else [
            {"dataset_id": s.dataset_id, **asdict(a)} for s, a in zip(batch.signals, augmentations)],
        "readout": readout, "landed_samples_per_flight": {str(k): v for k, v in sorted(spread.items())},
        "flights": rows, "elapsed_s": time.perf_counter() - started})
    for source in sources:
        print(f"{source}:")
        for key in ("all", *STRATA, *sorted({f"{r['airport']} {r['stratum']}" for r in rows})):
            if key in readout[source]:
                part = readout[source][key]
                shares = "  ".join(f"{name} {share:.3f}" for name, share in part["outcomes"].items() if share)
                print(f"  {key:22s} n={part['flights']:5d}  {shares}  first runway observed "
                      f"{part['first_runway_observed']:.3f}  cleared at the end {part['cleared_at_end']:.3f}")
        if readout[source]["all"]["forbidden_mass_per_step"]:
            print(f"  forbidden by the masks, mean probability a step: {readout[source]['all']['forbidden_mass_per_step']}")
        if readout[source]["all"]["pre_join"]:
            print(f"  before the join (said / observed): {readout[source]['all']['pre_join']}")
    print(f"landed samples per flight (of {args.samples}): {dict(sorted(spread.items()))}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
