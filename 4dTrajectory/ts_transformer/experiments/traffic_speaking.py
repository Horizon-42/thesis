"""The prior commanding one aircraft of a scene in the closed loop (multi-aircraft design §2.2 "一架由模型指挥", §2.6, §3.4,
§6.6 step 3) — shared by M3's readout and M4's post-training, not a runner.

A scene: one flight with a sentence the prior speaks to — its executor flies each step, as in single-aircraft free
generation (`prior_free_generation.ClosedLoop`) — and every other flight of its airport and split in the air in its time,
replayed along its record. The others' rows and the words said to them are data (`traffic_scene_data.FlightRows`; a
background flight's words "none"); the speaking aircraft's are the loop's. A traffic prior (`prior.model.with_traffic`)
reads the others through its traffic attention (`prior.scene_speaker.SceneSpeaker`); an other in the air more than
`HISTORY_S` before the speaking aircraft's first row is read from there (counted by the caller).

What the prior package does not reach is computed here, step by step, and handed to the speaker (design §9 item 18):

- **edge features** (`inference.scene_edges`) of the steps being encoded: the speaking aircraft on the loop's steps (its
  row r at the step its first row hangs on plus r steps: `prior.scene.hang`), at the loop's positions from its first
  predicted step, with the runway in force before the step (its own words); the others at their recorded row times, with
  the runway their labelled words have in force;
- **separation masks** (`inference.separation_masks`, design §3.4 layer 2; `speaking_masks`): the speed words and the
  approach clearance, against the others replayed at the step's instant (`traffic_census.traffic_at`) and the speaking
  aircraft's executor state — its speed along its course, its capture (established), the runway said at the step if it
  says one (else the one in force), its speed target in force (its speed word's; "unspecified" its own approach speed,
  the one its executor flies; before any speed word its present speed). A replayed flight with a sentence reads its
  labelled words as M0 step 6 did (`traffic_masks`): cleared under its approach word, its target its speed word's
  ("unspecified" its type's published approach speed, else its present speed); a background one is cleared once
  established, its target its present speed. The clearance mask applies to an aircraft not yet cleared. **Stated
  approximations**: the masks' own (`separation_masks`), and an "unspecified" speed is an indicated airspeed taken as the
  speed along the course while the executor flies it as true airspeed — about 5 % faster at 1 km, so the speed mask is
  optimistic for a follower flying "unspecified" (as in M0 step 6). The speaking aircraft's edge features read its rows at
  the loop's step times (its flown rows have no other), the others' at their recorded times — up to a second apart, as the
  scene samples read every flight's at its recorded times.

**Judged afterwards** (`judged`, `traffic_loop.Loop`): the speaking aircraft from its first predicted step to its own end
(`speaking_aircraft`), the others replayed. Ending it at the first loss it answers for afterwards is ending it then: the
others do not react, and what it says at a step never depends on a later one. Its steps before the first predicted one are
the record's, not the prior's, and are not judged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from geokit import NM_M

from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.autopilot.speed import approach_speed_ias_mps, speed_change_mps2
from ts_transformer.experiments.prior_free_generation import ClosedLoop, in_force
from ts_transformer.experiments.traffic_census import Track, track, traffic_at
from ts_transformer.experiments.traffic_labelled import own_end
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION, Controlled, Loop, Run, join
from ts_transformer.experiments.traffic_scene_data import AirportFlights, FlightRows, airport_flights
from ts_transformer.inference.scene_edges import SceneRows, scene_edges
from ts_transformer.inference.separation import VISUAL, Traffic
from ts_transformer.inference.separation_masks import clearance_check, speed_check
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import load_sentences, load_signals
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M, VocabularySpec
from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, RUNWAY, SPEED, Words
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import N_LOOK, SAMPLE_MAX_S, Landings, hang
from ts_transformer.prior.scene_data import Node
from ts_transformer.prior.scene_speaker import SceneSpeaker

#: The most of an other aircraft's history read before the speaking aircraft's first row: a training sample's longest span
#: (`prior.scene.SAMPLE_MAX_S`).
HISTORY_S = SAMPLE_MAX_S
#: The columns the separation masks take words from (design §3.4 layer 2).
MASK_COLUMNS = (APPROACH, SPEED)


@dataclass(frozen=True)
class SceneAirport:
    """One airport's flights of a split for the scene loop: as the samples place them (`FlightRows`: the model's inputs
    and edge features) and as the judge replays them (`traffic_census.Track`), by dataset id."""

    flights: AirportFlights
    tracks: dict[str, Track]


def scene_airports(directory: Path, split: str, spec: VocabularySpec, airports: Sequence[str],
                   landings: Mapping[str, Landings] | None, max_rows: int) -> tuple[dict[str, SceneAirport], dict[str, int]]:
    """Every airport's flights of ``split`` (`traffic_scene_data.airport_flights`: the counts are its) with their
    replayed tracks, established from the artefact's capture row (a background flight: the labeller's own rule,
    `traffic_census.track`)."""
    per_airport, counts = airport_flights(directory, split, spec, airports, landings, max_rows)
    signals = {s.dataset_id: (i, s) for i, s in enumerate(load_signals(directory, split))}
    sentences = load_sentences(directory, split, spec)
    offsets = sentences["offsets"]
    spoken = {int(i): k for k, i in enumerate(sentences["signal_index"])}
    out = {}
    for airport in per_airport:
        tracks = {}
        for key in airport.flights:
            i, flight = signals[key]
            k = spoken.get(i)
            tracks[key] = track(flight, None if k is None else int(offsets[k + 1] - offsets[k]),
                                None if k is None else int(sentences["capture_row"][k]), airport.geometry,
                                airport.separation.along_nm[flight.runway] * NM_M, spec, spec.step_s)
        out[airport.code] = SceneAirport(airport, tracks)
    return out, dict(counts)


@dataclass(frozen=True)
class Scene:
    """One speaking flight's scene: its airport, its key, the step its first row hangs on (epoch seconds), the others in
    the air at a step from there to the end of its time limit, and — an augmented scene's (`traffic_augment`) — flights
    moved in time or inserted (their rows and tracks, by their keys in ``others``)."""

    airport: SceneAirport
    key: str
    first_step_s: float
    others: tuple[str, ...]
    moved: tuple[tuple[FlightRows, Track], ...] = ()

    @property
    def speaking(self) -> FlightRows:
        return self.airport.flights.flights[self.key]

    def rows(self, key: str) -> FlightRows:
        """A flight of the scene as the samples place it (a moved one's own)."""
        return next((rows for rows, _ in self.moved if rows.presence.dataset_id == key), None) \
            or self.airport.flights.flights[key]

    def track(self, key: str) -> Track:
        """A flight of the scene as the judge replays it (a moved one's own)."""
        return next((track for _, track in self.moved if track.key == key), None) or self.airport.tracks[key]


def scene_of(airport: SceneAirport, key: str, limit_s: float, step_s: float) -> Scene:
    """``key``'s scene when it flies ``limit_s`` from its first predicted step (`Scene`)."""
    first = float(hang(airport.flights.flights[key].presence.start_s, step_s))
    end = first + N_LOOK * step_s + limit_s
    others = tuple(k for k, t in airport.tracks.items() if k != key and t.first_step_s <= end and t.last_step_s >= first)
    return Scene(airport, key, first, others)


def other_node(scene: Scene, key: str, step_s: float) -> Node:
    """An other aircraft as the speaker places it: its rows from the step its first row hangs on, counted from the
    speaking aircraft's row 0."""
    other = scene.rows(key)
    first = int(round((float(hang(other.presence.start_s, step_s)) - scene.first_step_s) / step_s))
    return Node(key, other.presence.speaking, first, **other.node_rows)


@dataclass(frozen=True)
class Aircraft:
    """The speaking aircraft's executor state at a step: position (airport frame), compass track, ground speed, capture."""

    e_m: float
    n_m: float
    height_m: float
    track_deg: float
    ground_speed_mps: float
    captured: bool


def speaking_masks(scene: Scene, column: int, classes: int, t_s: float, aircraft: Aircraft, runway: int,
                   speed_word: int | None, cleared: bool, approach_mps: float, words: Words, opening: bool) -> np.ndarray:
    """``[classes]``: the classes of ``column`` (one of `MASK_COLUMNS`) the separation masks leave the speaking
    aircraft at ``t_s`` (module docstring): on candidate ``runway``, its speed word in force (None: none yet) and whether
    it is cleared; ``approach_mps`` the speed its "unspecified" word flies; at the ``opening`` step "unchanged" is the
    model's to mask."""
    out = np.ones(classes, dtype=bool)
    airport = scene.airport
    geometry, separation = airport.flights.geometry, airport.flights.separation
    candidate = geometry.candidates[runway]
    relative = relative_to_runway(np.array([aircraft.e_m]), np.array([aircraft.n_m]), np.array([aircraft.track_deg]),
                                  np.array([aircraft.height_m]), candidate)
    angle = float(relative.track_minus_course_deg[0])
    keys = [k for k in scene.others if scene.track(k).on_step(t_s)]
    here = [scene.track(k) for k in keys]
    traffic = join(Traffic(np.array([aircraft.e_m]), np.array([aircraft.n_m]), np.array([aircraft.height_m]),
                           (candidate.ident,),
                           np.array([separation.along_nm[candidate.ident] * NM_M - float(relative.before_threshold_m[0])]),
                           np.array([angle]), np.array([float(relative.right_of_course_m[0])]),
                           np.array([aircraft.captured]), (scene.speaking.category,)),
                   traffic_at(here, t_s))
    levels = [words.speed_mps(i) for i in range(words.speed_unspecified)]
    said = [_labelled(scene.rows(k), t_s) for k in keys]
    if column == SPEED:
        speeds = np.array([aircraft.ground_speed_mps * math.cos(math.radians(angle)),
                           *(float(np.interp(t_s, t.presence.times_s, t.along_rate_mps)) for t in here)])
        candidates = np.append(levels, approach_mps)
        if len(candidates) != classes - 1:
            raise ValueError(f"{len(candidates)} speed words, the column has {classes} classes")
        targets = np.array([candidates[speed_word] if speed_word is not None else speeds[0],
                            *(_target(scene.rows(k), force, levels, words, float(speeds[m]))
                              for m, (k, force) in enumerate(zip(keys, said), start=1))])
        check = speed_check(traffic, 0, separation, speeds, targets, candidates, 0 if speed_word is None else speed_word,
                            speed_change_mps2(words.spec), ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M)
        if check is not None:
            out[1:] = check.allowed
            out[0] = opening or speed_word is None or check.unchanged_allowed
    elif column == APPROACH:
        if not cleared:
            others_cleared = [int(force[APPROACH]) - 1 == APPROACH_CLEARED if force is not None else t_s >= t.captured_s
                              for t, force in zip(here, said)]
            gate = clearance_check(traffic, 0, separation, np.array([False, *others_cleared], dtype=bool))
            if gate is not None and not gate.allowed:
                out[APPROACH_CLEARED + 1] = False
    else:
        raise ValueError(f"column {column} is not masked by separation")
    return out


def _labelled(other: FlightRows, t_s: float) -> np.ndarray | None:
    """A replayed flight's classes in force at ``t_s`` — its labelled words said at its rows up to then — or None for a
    background flight (no words)."""
    if not other.presence.speaking:
        return None
    force = other.node_rows["in_force"]
    row = int(np.searchsorted(other.presence.times_s, t_s, side="right")) - 1
    return force[min(max(row, 0) + 1, len(force) - 1)]


def _target(other: FlightRows, force: np.ndarray | None, levels: Sequence[float], words: Words, speed_mps: float) -> float:
    """A replayed flight's speed target: its speed word's ("unspecified" its type's published approach speed), else its
    present speed along its course."""
    if force is None or int(force[SPEED]) == 0:
        return speed_mps
    word = int(force[SPEED]) - 1
    if word < words.speed_unspecified:
        return float(levels[word])
    own = approach_speed_ias_mps(other.typecode, None) if other.typecode is not None else math.nan
    return speed_mps if math.isnan(own) else own


class SceneLoop(ClosedLoop):
    """`ClosedLoop` of flights, each speaking in its scene (module docstring): ``scenes`` one per flight, in its order;
    ``approach_mps`` the speed each flight's "unspecified" word flies (its executor's)."""

    def __init__(self, model: Prior, flights: Sequence[FlightSignals], geometries: Sequence[AirportGeometry],
                 inputs: Any, runways: Any, charts: Any, approach_ias_mps: Any, limits: Sequence[float], words: Words,
                 params: Any, landings: Any, *, scenes: Sequence[Scene], generator: Any, temperature: float,
                 procedure_masks: Any) -> None:
        if len(scenes) != len(flights) or any(s.key != f.dataset_id for s, f in zip(scenes, flights)):
            raise ValueError("a scene per flight, in the flights' order")
        self.scenes, self.words = list(scenes), words
        self.approach_mps = approach_ias_mps.cpu().numpy()
        #: per masked column, per step: which flights the separation masks took a word from ([B] bool)
        self.separation_masked: dict[int, list[np.ndarray]] = {column: [] for column in MASK_COLUMNS}
        self._state: tuple[int, list[Aircraft] | None] = (-1, None)
        super().__init__(model, flights, geometries, inputs, runways, charts, approach_ias_mps, limits, words, params,
                         landings, generator=generator, temperature=temperature, procedure_masks=procedure_masks)

    def _make_speaker(self, model: Prior, flights: Sequence[FlightSignals], geometries: Sequence[AirportGeometry],
                      landings: Any, words: Words, **options: Any) -> SceneSpeaker:
        others = [[other_node(scene, key, self.step_s) for key in scene.others] for scene in self.scenes]
        return SceneSpeaker(model, flights, geometries, landings, words, others=others, edges=self._edges,
                            masks=self._masks, mask_columns=MASK_COLUMNS, history=int(HISTORY_S // self.step_s),
                            **options)

    def _edges(self, first: int, last: int) -> np.ndarray:
        """Every scene's edge features on batch steps ``first … last − 1`` (module docstring), from the step before
        (an aircraft's motion is its displacement from its row before)."""
        speaker, step = self.speaker, self.step_s
        low = max(first - 1, 0)
        span = last - low
        width = len(speaker.model.traffic_features)
        out = np.zeros((len(self.scenes), last - first, speaker.aircraft, speaker.aircraft, width), dtype=np.float32)
        own_first, own_last = max(low - speaker.pre, 0), min(last - speaker.pre, speaker.rows)
        pointers = speaker.in_force[:, 0, own_first: max(own_last, own_first), RUNWAY].cpu().numpy()
        for b, scene in enumerate(self.scenes):
            airport = scene.airport
            geometry, separation = airport.flights.geometry, airport.flights.separation
            members = [scene.speaking, *(scene.rows(k) for k in scene.others)]
            shape = (len(members), span)
            time_s, e, n, h, along = (np.full(shape, np.nan) for _ in range(5))
            runway: list[list[str | None]] = [[None] * span for _ in members]
            if own_first < own_last:
                cols = slice(own_first + speaker.pre - low, own_last + speaker.pre - low)
                rows = np.arange(own_first, own_last)
                time_s[0, cols] = scene.first_step_s + rows * step
                e[0, cols], n[0, cols], h[0, cols] = speaker.e[b, rows], speaker.n[b, rows], speaker.h[b, rows]
                for j, pointer in zip(range(cols.start, cols.stop), pointers[b]):
                    if pointer:
                        candidate = geometry.candidates[int(pointer) - 1]
                        runway[0][j] = candidate.ident
                        before = relative_to_runway(e[0, j: j + 1], n[0, j: j + 1], np.zeros(1), h[0, j: j + 1],
                                                    candidate).before_threshold_m
                        along[0, j] = separation.along_nm[candidate.ident] * NM_M - float(before[0])
            for m, other in enumerate(members[1:], start=1):
                start = speaker.pre + int(round((float(hang(other.presence.start_s, step)) - scene.first_step_s) / step))
                lo, hi = max(low, start), min(last, start + len(other.runway))
                if lo >= hi:
                    continue
                cols, own = slice(lo - low, hi - low), slice(lo - start, hi - start)
                time_s[m, cols] = other.presence.times_s[own]
                e[m, cols], n[m, cols], h[m, cols] = other.e_m[own], other.n_m[own], other.height_m[own]
                along[m, cols] = other.along_m[own]
                runway[m][cols] = other.runway[own]
            rows_here = SceneRows(time_s, e, n, h, along, runway, [f.category for f in members])
            out[b, :, : len(members), : len(members)] = scene_edges(rows_here, separation)[first - low:]
        return out

    def close(self) -> None:
        """Let the speaker go: it holds the loop's callbacks and the loop holds it — a cycle only the cyclic collector
        frees, which does not watch the GPU, so the model's past of every batch piled up until it filled the GPU (the
        first formal run). Read what is wanted first."""
        self.speaker.edges_of = self.speaker.masks_of = None
        self.speaker = None

    def _now(self, row: int) -> list[Aircraft]:
        """Every flight's executor state at the start of the step at ``row`` (read once a step)."""
        if self._state[0] != row:
            now = self.executor.now()
            e, n, h = now.e_m.cpu().numpy(), now.n_m.cpu().numpy(), now.height_m.cpu().numpy()
            track_deg, ground = now.track_deg.cpu().numpy(), now.ground_speed_mps.cpu().numpy()
            captured = self.executor.lateral.captured.cpu().numpy()
            self._state = (row, [Aircraft(float(e[b]), float(n[b]), float(h[b]), float(track_deg[b]), float(ground[b]),
                                          bool(captured[b])) for b in range(len(e))])
        return self._state[1]

    def _masks(self, column: int, chosen: np.ndarray) -> np.ndarray:
        """``[B, classes]``: `speaking_masks` of every scene at the newest row; a flight the executor is done with, or
        with no runway in force or said, keeps every word."""
        speaker = self.speaker
        row = speaker.rows - 1
        classes = speaker.model.config.classes[column]
        out = np.ones((len(self.scenes), classes), dtype=bool)
        state, done = self._now(row), self.executor.done.cpu().numpy()
        for b, scene in enumerate(self.scenes):
            pointer = int(chosen[b, RUNWAY]) or int(speaker.value[b, RUNWAY])
            if done[b] or not pointer:
                continue
            speed = int(speaker.value[b, SPEED]) - 1
            out[b] = speaking_masks(scene, column, classes, scene.first_step_s + row * self.step_s, state[b],
                                    pointer - 1, speed if speed >= 0 else None,
                                    int(speaker.value[b, APPROACH]) - 1 == APPROACH_CLEARED, float(self.approach_mps[b]),
                                    self.words, row == N_LOOK)
        self.separation_masked[column].append(~out.all(axis=1))
        return out


def speaking_aircraft(scene: Scene, flown: Flown, j: int, grid: np.ndarray, outcome: str, end_row: int,
                      crossing: dict[str, float] | None, stop_step: int, step_s: float) -> Controlled:
    """Flight ``j`` of ``flown`` on the loop's steps from its first predicted step to its own end
    (`traffic_labelled.own_end`, ``stop_step`` the glidepath lower edge's, −1: none): its state every step, the runway its
    words have in force there (``grid``: the sentence said, its first row every column), the executor's capture after
    the cycle before each state (none at the first), its crossing when it landed."""
    step_rows = int(round(step_s / flown.cycle_s))
    ended, last_row = own_end(outcome, end_row, stop_step, step_rows)
    at_rows = np.arange(0, last_row + 1, step_rows)
    geometry, separation = scene.airport.flights.geometry, scene.airport.flights.separation
    flown_at = flown_track(flown.states[j, at_rows].cpu().numpy(), geometry)
    captured = np.concatenate(([False], flown.modes["captured"][j, at_rows[1:] - 1].cpu().numpy()))
    force = in_force(grid)
    idents = [geometry.candidates[int(force[min(k, len(force) - 1), RUNWAY])].ident for k in range(len(at_rows))]
    along, angle, right = (np.zeros(len(at_rows)) for _ in range(3))
    for ident in set(idents):
        on = np.array([r == ident for r in idents])
        candidate = geometry.candidates[geometry.candidate_index(ident)]
        relative = relative_to_runway(flown_at["e"][on], flown_at["n"][on], flown_at["track"][on], flown_at["height"][on],
                                      candidate)
        along[on] = separation.along_nm[ident] * NM_M - relative.before_threshold_m
        angle[on], right[on] = relative.track_minus_course_deg, relative.right_of_course_m
    times = scene.first_step_s + (N_LOOK + np.arange(len(at_rows))) * step_s
    landing = float(times[0] + crossing["at_row"] * flown.cycle_s) if ended == "landed" else None
    return Controlled(scene.speaking.presence, times, flown_at["e"], flown_at["n"], flown_at["height"], tuple(idents),
                      along, angle, right, flown_at["ground_speed"] * np.cos(np.radians(angle)), captured,
                      scene.speaking.category, ended, landing)


def judged(scene: Scene, aircraft: Controlled, step_s: float, reading: str = VISUAL
           ) -> tuple[str, dict[str, Any] | None, Run]:
    """The speaking aircraft judged with the others replayed (`traffic_loop.Loop` under ``reading``): its outcome
    (`traffic_loop.LOST_SEPARATION` when a loss it answers for ended it), that end, and the run."""
    run = Loop(scene.airport.flights.separation, reading, step_s).run(
        [aircraft], [scene.track(k) for k in scene.others])
    end = run.ended.get(aircraft.key)
    return (LOST_SEPARATION if end is not None else aircraft.outcome), end, run
