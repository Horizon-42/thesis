"""One commanded aircraft's view of its scene (multi-aircraft design §2.2 "一架由模型指挥", §2.6, §3.4) — what the window
loop reads and judges of each commanded aircraft (`traffic_window.Window.scene`); shared, not a runner. Written for the
one-aircraft scene loop of M3 and M4's first pass, archived 2026-10-02 when the window loop took that setting over as a
window with one commanded aircraft (`archive/one_commanded_scene_2026_10/`, design §6.6 step 9).

A scene: one flight with a sentence the prior speaks to and every other flight of its airport and split in the air in its
time (`Scene`), replayed along its record. The others' rows and the words said to them are data
(`traffic_scene_data.FlightRows`; a background flight's words "none"); the speaking aircraft's are the loop's. An other in
the air more than `HISTORY_S` before the speaking aircraft's first row is read from there.

What the prior package does not reach is computed here, step by step, and handed to the speaker (design §9 item 18):

- **edge features** (`inference.scene_edges`) of the steps being encoded: the speaking aircraft on the loop's steps (its
  row r at its first row's step plus r steps — every row is on a step, `prior.scene`), at the loop's positions from its
  first predicted step, with the runway in force before the step (its own words); the others at their recorded rows,
  with the runway their labelled words have in force;
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
  optimistic for a follower flying "unspecified" (as in M0 step 6). Every row, flown or recorded, is on a step
  (`prior.scene`).

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
from ts_transformer.experiments.prior_free_generation import in_force
from ts_transformer.experiments.traffic_census import Track, track, traffic_at
from ts_transformer.experiments.traffic_labelled import own_end
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION, Controlled, Loop, Run, join
from ts_transformer.experiments.traffic_scene_data import AirportFlights, FlightRows, airport_flights
from ts_transformer.inference.scene_edges import SceneRows
from ts_transformer.inference.separation import VISUAL, Traffic
from ts_transformer.inference.separation_masks import clearance_check, speed_check
from ts_transformer.instructions.airport import relative_to_runway
from ts_transformer.instructions.artefact import load_sentences, load_signals
from ts_transformer.instructions.spec import ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M, VocabularySpec
from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, RUNWAY, SPEED, Words
from ts_transformer.prior.scene import N_LOOK, SAMPLE_MAX_S, Landings

#: The most of an other aircraft's history read before the speaking aircraft's first row: a training sample's longest span
#: (`prior.scene.SAMPLE_MAX_S`).
HISTORY_S = SAMPLE_MAX_S
#: The columns the separation masks take words from (design §3.4 layer 2).
MASK_COLUMNS = (APPROACH, SPEED)
#: An inserted flight's key: its own, then this (`traffic_window_augment`).
INSERTED = "+inserted"


@dataclass(frozen=True)
class SceneAirport:
    """One airport's flights of a split for the window loop: as the samples place them (`FlightRows`: the model's inputs
    and edge features) and as the judge replays them (`traffic_census.Track`), by dataset id."""

    flights: AirportFlights
    tracks: dict[str, Track]


def scene_airports(directory: Path, split: str, spec: VocabularySpec, airports: Sequence[str],
                   landings: Mapping[str, Landings] | None, max_rows: int) -> tuple[dict[str, SceneAirport], dict[str, int]]:
    """Every airport's flights of ``split`` (`traffic_scene_data.airport_flights`: the counts are its) with their
    replayed tracks, established from the artefact's capture row (a background flight: the labeller's own rule,
    `traffic_census.track`)."""
    per_airport, counts = airport_flights(directory, split, spec, airports, landings, max_rows)
    return with_tracks(directory, split, spec, per_airport), dict(counts)


def with_tracks(directory: Path, split: str, spec: VocabularySpec, per_airport: Sequence[AirportFlights]
                ) -> dict[str, SceneAirport]:
    """`scene_airports` of flights already built (`traffic_scene_data.airport_flights` of ``split``)."""
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
    return out


@dataclass(frozen=True)
class Scene:
    """One speaking flight's scene: its airport, its key, its first row's step (epoch seconds), the others in
    the air at a step from there to the end of its time limit, and — an augmented scene's (`traffic_window_augment`) —
    flights moved in time or inserted (their rows and tracks, by their keys in ``others``)."""

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


@dataclass(frozen=True)
class Aircraft:
    """The speaking aircraft's executor state at a step: position (airport frame), compass track, ground speed, capture."""

    e_m: float
    n_m: float
    height_m: float
    track_deg: float
    ground_speed_mps: float
    captured: bool


@dataclass(frozen=True)
class OthersAt:
    """A scene's others on a step at an instant as the separation masks read them (module docstring), in the scene's
    order: their traffic (`traffic_census.traffic_at`), their speed along their course, their speed targets and whether
    they are cleared — the same for every sentence said in the scene and both masked columns (`others_at`)."""

    traffic: Traffic
    speeds: np.ndarray
    targets: np.ndarray
    cleared: np.ndarray


def others_at(scene: Scene, t_s: float, words: Words) -> OthersAt:
    """``scene``'s others on a step at ``t_s`` (`OthersAt`)."""
    keys = [k for k in scene.others if scene.track(k).on_step(t_s)]
    here = [scene.track(k) for k in keys]
    levels = [words.speed_mps(i) for i in range(words.speed_unspecified)]
    said = [_labelled(scene.rows(k), t_s) for k in keys]
    speeds = np.array([float(np.interp(t_s, t.presence.times_s, t.along_rate_mps)) for t in here])
    targets = np.array([_target(scene.rows(k), force, levels, words, float(speeds[m]))
                        for m, (k, force) in enumerate(zip(keys, said))])
    cleared = np.array([int(force[APPROACH]) - 1 == APPROACH_CLEARED if force is not None else t_s >= t.captured_s
                        for t, force in zip(here, said)], dtype=bool)
    return OthersAt(traffic_at(here, t_s), speeds, targets, cleared)


def speaking_masks(scene: Scene, column: int, classes: int, t_s: float, aircraft: Aircraft, runway: int,
                   speed_word: int | None, cleared: bool, approach_mps: float, words: Words, opening: bool,
                   others: OthersAt) -> np.ndarray:
    """``[classes]``: the classes of ``column`` (one of `MASK_COLUMNS`) the separation masks leave the speaking
    aircraft at ``t_s`` (module docstring): on candidate ``runway``, its speed word in force (None: none yet) and whether
    it is cleared; ``approach_mps`` the speed its "unspecified" word flies; at the ``opening`` step "unchanged" is the
    model's to mask; ``others``: the scene's others then (`others_at`)."""
    out = np.ones(classes, dtype=bool)
    airport = scene.airport
    geometry, separation = airport.flights.geometry, airport.flights.separation
    candidate = geometry.candidates[runway]
    relative = relative_to_runway(np.array([aircraft.e_m]), np.array([aircraft.n_m]), np.array([aircraft.track_deg]),
                                  np.array([aircraft.height_m]), candidate)
    angle = float(relative.track_minus_course_deg[0])
    traffic = join(Traffic(np.array([aircraft.e_m]), np.array([aircraft.n_m]), np.array([aircraft.height_m]),
                           (candidate.ident,),
                           np.array([separation.along_nm[candidate.ident] * NM_M - float(relative.before_threshold_m[0])]),
                           np.array([angle]), np.array([float(relative.right_of_course_m[0])]),
                           np.array([aircraft.captured]), (scene.speaking.category,)),
                   others.traffic)
    levels = [words.speed_mps(i) for i in range(words.speed_unspecified)]
    if column == SPEED:
        speeds = np.concatenate(([aircraft.ground_speed_mps * math.cos(math.radians(angle))], others.speeds))
        candidates = np.append(levels, approach_mps)
        if len(candidates) != classes - 1:
            raise ValueError(f"{len(candidates)} speed words, the column has {classes} classes")
        targets = np.concatenate(([candidates[speed_word] if speed_word is not None else speeds[0]], others.targets))
        check = speed_check(traffic, 0, separation, speeds, targets, candidates, 0 if speed_word is None else speed_word,
                            speed_change_mps2(words.spec), ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M)
        if check is not None:
            out[1:] = check.allowed
            out[0] = opening or speed_word is None or check.unchanged_allowed
    elif column == APPROACH:
        if not cleared:
            gate = clearance_check(traffic, 0, separation, np.concatenate(([False], others.cleared)))
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


def scene_landings(landings: Landings, scene: Scene) -> Landings:
    """An airport's landings as ``scene`` has them: each moved flight's landing at its moved time, an inserted one's
    added (module docstring)."""
    times, by_runway = list(landings.times_s), {runway: list(t) for runway, t in landings.by_runway.items()}
    for rows, _ in scene.moved:
        key, runway = rows.presence.dataset_id, rows.presence.runway
        if not key.endswith(INSERTED):
            original = scene.airport.flights.flights[key].presence.landing_s
            times.remove(original)
            by_runway[runway].remove(original)
        times.append(rows.presence.landing_s)
        by_runway[runway].append(rows.presence.landing_s)
    return Landings(np.sort(np.array(times, dtype=np.float64)),
                    {runway: np.sort(np.array(t, dtype=np.float64)) for runway, t in by_runway.items()})


def edge_rows(scene: Scene, e: np.ndarray, n: np.ndarray, h: np.ndarray, pointers: np.ndarray, rows: int, pre: int,
              low: int, last: int, step_s: float) -> SceneRows:
    """What ``scene``'s edge features on batch steps ``low … last − 1`` read (module docstring) — the one reading, for the
    loop's speaker and for a trainer scoring a sentence in its scene: the speaking aircraft from batch step ``pre`` over its
    first ``rows`` rows (its positions ``e``, ``n``, ``h`` and the runway class in force before each row, ``pointers``,
    0: none), on the loop's steps; the others at their recorded rows."""
    return SceneRows(*_edge_arrays(scene, e, n, h, pointers, rows, pre, low, last, step_s))


def _edge_arrays(scene: Scene, e: np.ndarray, n: np.ndarray, h: np.ndarray, pointers: np.ndarray, rows: int, pre: int,
                 low: int, last: int, step_s: float) -> tuple[Any, ...]:
    """`edge_rows`' fields, unchecked (the loop stacks a batch's scenes and checks them once)."""
    airport = scene.airport
    geometry, separation = airport.flights.geometry, airport.flights.separation
    members = [scene.speaking, *(scene.rows(k) for k in scene.others)]
    span = last - low
    shape = (len(members), span)
    time_s, east, north, height, along = (np.full(shape, np.nan) for _ in range(5))
    runway: list[list[str | None]] = [[None] * span for _ in members]
    own_first, own_last = max(low - pre, 0), min(last - pre, rows)
    if own_first < own_last:
        cols = slice(own_first + pre - low, own_last + pre - low)
        numbers = np.arange(own_first, own_last)
        time_s[0, cols] = scene.first_step_s + numbers * step_s
        east[0, cols], north[0, cols], height[0, cols] = e[numbers], n[numbers], h[numbers]
        for j, pointer in zip(range(cols.start, cols.stop), pointers[own_first: own_last]):
            if pointer:
                candidate = geometry.candidates[int(pointer) - 1]
                runway[0][j] = candidate.ident
                before = relative_to_runway(east[0, j: j + 1], north[0, j: j + 1], np.zeros(1), height[0, j: j + 1],
                                            candidate).before_threshold_m
                along[0, j] = separation.along_nm[candidate.ident] * NM_M - float(before[0])
    for m, other in enumerate(members[1:], start=1):
        start = pre + int(round((other.presence.start_s - scene.first_step_s) / step_s))
        lo, hi = max(low, start), min(last, start + len(other.runway))
        if lo >= hi:
            continue
        cols, own = slice(lo - low, hi - low), slice(lo - start, hi - start)
        time_s[m, cols] = other.presence.times_s[own]
        east[m, cols], north[m, cols], height[m, cols] = other.e_m[own], other.n_m[own], other.height_m[own]
        along[m, cols] = other.along_m[own]
        runway[m][cols] = other.runway[own]
    return time_s, east, north, height, along, runway, [f.category for f in members]


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
