"""Augmented windows for the loop where the prior commands every aircraft of a window (multi-aircraft design §5, §6.6
step 7 item 6), not a runner: a window changed so that it holds traffic the data rarely does — only in the closed loop
(reward and readouts), never teacher forcing (§5.1).

The three ways of "every aircraft of a window commanded" (§5.2, §9 item 7), a third each:

- **C, the flow compressed** — every commanded aircraft moved whole toward the window's opening: its first row's time
  after the opening × c, c ~ U[`COMPRESSION`] (the arrivals up to 1 / 0.6 ≈ 1.7 times as dense); the replayed ones as
  they were — the same arrivals from the same directions, closer together;
- **B, a start moved** — one commanded aircraft (drawn) moved as post-training stage 2 moves a start (`prior.augment`,
  as `traffic_augment` B: drawn until plausible), its time limit stage 2's; what the others read of it before it flies
  is its moved rows (`traffic_census.track` of them), never established there (the executor starts it uncaptured, as
  the qualification reads its own rows);
- **A, a flight inserted** — a flight of the draw at the same airport (one the window does not hold: it flies on its
  own dynamics, `traffic_window.draw_windows`), landing on a runway that must be spaced from a commanded aircraft's
  (drawn: its new follower; the same runway, a pair separated as one, a dependent parallel), moved whole to cross its
  threshold g × the required gap before the follower's recorded crossing on the approach clock (`traffic_augment` A's
  timing), g ~ U[`traffic_augment.GAP_RANGE`] — and commanded too, keyed `traffic_speaking.INSERTED` after its own key,
  over its own time limit; the window no longer replays the flight it came from.

**Every shift is whole seconds**: the rosters' landing times are whole seconds, and a flight's own landing leaves its
landing context by its exact time (`data.own_context`). A moved or inserted flight moves whole — its rows and record
(`traffic_augment.moved`) and its signals (`shifted`: its entry and landing times, so its rows' times, its landing in the
others' context and the landing direction at its first predicted step are the moved ones'). A moved flight keeps its
rows' inputs and words, as in `traffic_augment`.

**Qualification** (§5.3): every commanded aircraft, through its observed rows (to its first predicted step) judged not
established — as the loop judges its first step — against every other aircraft of the window along its record (a moved
one's moved), answers for no loss under the loop's reading: a window already lost when the prior starts to speak is not
the prior's. And the window never holds more aircraft on one step, along the records, than its airport's busiest step on
the training days (`busiest`: §5.2, a density the data has had). A kind that cannot apply (no flight to insert, no
plausible start) or a draw that fails is drawn again, at most `traffic_augment.TRIES` draws, else the window is left out;
the draws refused are counted by why.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from geokit import NM_M

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.flights import flight_inputs
from ts_transformer.data.day_split import parse_utc
from ts_transformer.experiments.prior_free_generation import augmented_starts
from ts_transformer.experiments.traffic_augment import GAP_RANGE, TRIES, moved
from ts_transformer.experiments.traffic_census import track
from ts_transformer.experiments.traffic_loop import recorded
from ts_transformer.experiments.traffic_scene_data import FlightRows
from ts_transformer.experiments.traffic_speaking import INSERTED, judged
from ts_transformer.experiments.traffic_window import Window, window_of
from ts_transformer.inference.runway_schedule import DEPENDENT, SAME, SINGLE
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.prior.augment import Augmentation, augment_signals
from ts_transformer.prior.scene import N_LOOK, SceneIndex, hang, in_scene, presence, scene_steps

KINDS = ("C", "B", "A")
COMPRESSION = (0.6, 1.0)
#: A commanded aircraft's part in its window's augmentation: moved in time (C), its start moved (B), inserted (A).
ROLES = ("shifted", "moved", "inserted")
#: Why a draw was refused (module docstring).
REFUSALS = ("no_plausible_start", "no_flight_to_insert", "busier_than_the_data", "starts_in_a_loss")


@dataclass(frozen=True)
class AugmentedWindow:
    """A window augmented (module docstring): its kind, the window, and per commanded aircraft (in the window's order)
    its flight in the draw's batch (an inserted one's: the flight it came from), its signals (moved or shifted where
    it moved), its moved start (B's; else None), its time limit and its part (`ROLES`; None: as drawn); what was drawn
    and how many draws it took."""

    kind: str
    window: Window
    places: tuple[int, ...]
    signals: tuple[FlightSignals, ...]
    moves: tuple[Augmentation | None, ...]
    limits: tuple[float, ...]
    roles: tuple[str | None, ...]
    drawn: dict[str, Any]
    draws: int


def shifted(signals: FlightSignals, dt_s: int, key: str) -> FlightSignals:
    """``signals`` moved ``dt_s`` whole seconds and keyed ``key``: its entry time (to the microsecond) and landing time
    (whole seconds, the roster's), its rows as they were."""
    def later(value: str, timespec: str) -> str:
        return (parse_utc(value) + timedelta(seconds=dt_s)).isoformat(timespec=timespec).replace("+00:00", "Z")

    return dataclasses.replace(signals, dataset_id=key, entry_time_utc=later(signals.entry_time_utc, "microseconds"),
                               landing_time_utc=later(signals.landing_time_utc, "seconds"))


def busiest(directory: Path, spec: VocabularySpec, airports: Sequence[str], step_s: float) -> dict[str, int]:
    """Each airport's most aircraft on one step of the training days — every arrival, a background one too, from the
    step its first row hangs on to its last's (`prior.scene.in_scene`; a flight with a sentence over its sentence's
    rows): the cap on an augmented window (§5.2)."""
    geometries = load_candidates(directory)
    sentences = load_sentences(directory, "train", spec)
    rows = {int(i): int(sentences["offsets"][k + 1] - sentences["offsets"][k])
            for k, i in enumerate(sentences["signal_index"])}
    flights: dict[str, list[Any]] = {code: [] for code in airports}
    for i, flight in enumerate(load_signals(directory, "train")):
        if flight.airport in flights:
            flights[flight.airport].append(presence(flight, rows.get(i), geometries[flight.airport]))
    return {code: max(at_once(segment, step_s) for segment in SceneIndex(part).segments(step_s))
            for code, part in flights.items()}


def at_once(flights: Sequence[Any], step_s: float) -> int:
    """The most of ``flights`` (their presences) in the scene on one step."""
    first = min(float(hang(p.start_s, step_s)) for p in flights)
    last = max(float(hang(p.end_s, step_s)) for p in flights)
    return int(in_scene(flights, scene_steps(first, last, step_s), step_s).max())


def qualifies(window: Window, signals: Mapping[str, FlightSignals], step_s: float) -> bool:
    """No commanded aircraft of ``window`` answers for a loss through its observed rows (``signals``: each one's, a
    moved one's moved), judged not established, against every other aircraft along its record (module docstring)."""
    geometry, separation = window.airport.flights.geometry, window.airport.flights.separation
    cut = slice(0, N_LOOK + 1)
    for key in window.commanded:
        rows = window.rows(key)
        seen = presence(signals[key], len(rows.presence.times_s), geometry)
        whole = recorded(seen, signals[key], len(seen.times_s), 0.0, geometry, separation, rows.category, step_s)
        observed = dataclasses.replace(
            whole, times_s=whole.times_s[cut], e_m=whole.e_m[cut], n_m=whole.n_m[cut], height_m=whole.height_m[cut],
            runway=whole.runway[cut], along_m=whole.along_m[cut],
            track_minus_course_deg=whole.track_minus_course_deg[cut], right_of_course_m=whole.right_of_course_m[cut],
            along_speed_mps=whole.along_speed_mps[cut], established=whole.established[cut], outcome="timeout",
            landing_s=None)
        if judged(window.scene(key, step_s), observed, step_s)[1] is not None:
            return False
    return True


def augment_window(window: Window, batch: replay.Batch, places: Sequence[int], pool: Mapping[str, int],
                   limits: Sequence[float], moved_limits: Sequence[float], most: int, rng: np.random.Generator,
                   windows: dict[str, tuple[float, float]], spec: VocabularySpec, kinds: Sequence[str] = KINDS
                   ) -> tuple[AugmentedWindow | None, Counter]:
    """``window`` augmented (module docstring), and the draws refused by why (`REFUSALS`). ``batch``: the draw's flights,
    the window's commanded ones at ``places`` (in its order); ``pool``: the flights an insertion is drawn from, by key,
    their places in ``batch`` (the draw's flights at this airport); ``limits`` / ``moved_limits``: every flight's time
    limit in ``batch``, the executor spec's and stage 2's; ``most``: the airport's busiest step (`busiest`); ``windows``:
    stage 2's altitude windows; ``kinds``: those drawn from, each alike. None after `TRIES` draws refused."""
    step_s = spec.step_s
    airport = window.airport
    geometry, separation = airport.flights.geometry, airport.flights.separation
    own = dict(zip(window.commanded, places))
    refused: Counter = Counter()
    for draw in range(1, TRIES + 1):
        kind = kinds[int(rng.integers(len(kinds)))]
        place, signals = dict(own), {k: batch.signals[j] for k, j in own.items()}
        limit = {k: limits[j] for k, j in own.items()}
        move: dict[str, Augmentation] = {}
        role: dict[str, str] = {}
        parts: list[tuple[FlightRows, Any]] = []
        left_out: list[str] = []
        if kind == "C":
            c = float(rng.uniform(*COMPRESSION))
            shift = {k: int(round((c - 1.0) * (window.rows(k).presence.start_s - window.opens_s)))
                     for k in window.commanded}
            for k in window.commanded:
                parts.append(moved(window.rows(k), window.track(k), float(shift[k]), k, step_s))
                signals[k] = shifted(signals[k], shift[k], k)
                role[k] = "shifted"
            drawn: dict[str, Any] = {"compression": c, "shift_s": shift}
        elif kind == "B":
            key = window.commanded[int(rng.integers(len(window.commanded)))]
            j = own[key]
            start = augmented_starts([signals[key]], flight_inputs(batch.series[j: j + 1], device=torch.device("cpu"),
                                                                   anchor=N_LOOK), rng, windows).moves[0]
            if start is None:
                refused["no_plausible_start"] += 1
                continue
            signals[key] = augment_signals(signals[key], start)
            rows = window.rows(key)
            record = dataclasses.replace(
                track(signals[key], len(rows.presence.times_s), None, geometry,
                      separation.along_nm[rows.presence.runway] * NM_M, spec, step_s),
                captured_s=math.inf)
            parts.append((dataclasses.replace(rows, presence=record.presence), record))
            move[key], role[key], limit[key] = start, "moved", moved_limits[j]
            drawn = {"moved": key, **dataclasses.asdict(start)}
        else:
            follower = window.commanded[int(rng.integers(len(window.commanded)))]
            behind = window.track(follower)
            present = set(window.commanded) | set(window.others)
            options = sorted(k for k in pool if k not in present
                             and separation.relation(airport.tracks[k].presence.runway, behind.presence.runway)
                             in (SAME, SINGLE, DEPENDENT))
            if not options:
                refused["no_flight_to_insert"] += 1
                continue
            source_key = options[int(rng.integers(len(options)))]
            source = airport.tracks[source_key]
            gap = float(rng.uniform(*GAP_RANGE))
            clock = (separation.approach_time_s(behind.presence.runway, behind.presence.landing_s)
                     - gap * separation.gap_s(source.presence.runway, source.category, behind.presence.runway,
                                              behind.category))
            dt = int(round(clock + separation.along_nm[source.presence.runway] * NM_M / separation.speed_mps
                           - source.presence.landing_s))
            key = source_key + INSERTED
            j = pool[source_key]
            parts.append(moved(airport.flights.flights[source_key], source, float(dt), key, step_s))
            place[key], signals[key], limit[key], role[key] = j, shifted(batch.signals[j], dt, key), limits[j], "inserted"
            left_out.append(source_key)
            drawn = {"inserted": source_key, "follower": follower, "gap": gap, "shift_s": dt}
        placed = Window(airport, window.opens_s, tuple(place), (), tuple(parts))
        commanded = tuple(sorted(place, key=lambda k: (placed.first_step_s(k, step_s), k)))
        candidate = window_of(airport, window.opens_s, commanded, [limit[k] for k in commanded], step_s, parts,
                              left_out)
        if at_once([candidate.track(k).presence for k in (*candidate.commanded, *candidate.others)], step_s) > most:
            refused["busier_than_the_data"] += 1
            continue
        if not qualifies(candidate, signals, step_s):
            refused["starts_in_a_loss"] += 1
            continue
        return (AugmentedWindow(kind, candidate, tuple(place[k] for k in commanded),
                                tuple(signals[k] for k in commanded), tuple(move.get(k) for k in commanded),
                                tuple(limit[k] for k in commanded), tuple(role.get(k) for k in commanded), drawn, draw),
                refused)
    return None, refused
