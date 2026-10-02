"""Augmented windows for the window loop (multi-aircraft design §5, §6.6 step 7 item 6 and step 9's "9.4 的代码"), not a
runner: a window changed so that it holds traffic the data rarely does — only in the closed loop (reward and readouts),
never teacher forcing (§5.1).

**A window's kind is drawn once**, and only its parameters are drawn again when a draw is refused (a kind that took over
another's refused windows would read a different population); the kinds are a third each (§5.4 item 2, §9 item 7), by
how the draw commands (`KINDS_OF`, `traffic_window.COMMANDED`): every aircraft of a window commanded — C, B, A, A's
inserted flight commanded too; one aircraft a window commanded — D, B, A, A's inserted flight replayed along its record
(the one-aircraft setting's, as M4's first runner had them).

The ways of "every aircraft of a window commanded":

- **C, the flow compressed** — every commanded aircraft moved whole toward the window's opening: its first row's time
  after the opening × c, c ~ U[`COMPRESSION`] (the arrivals up to 1 / 0.6 ≈ 1.7 times as dense). The replayed ones stay
  as they were — among them the flights entering in the window that the executor cannot fly — so the commanded arrivals
  keep their order and directions among themselves and may pass a replayed one;
- **B, a start moved** — one commanded aircraft (drawn) moved as post-training stage 2 moves a start (`prior.augment`,
  as `traffic_augment` B: drawn until plausible), its time limit stage 2's; what the others read of it before it flies
  is its moved rows (`traffic_census.track` of them), never established there (the executor starts it uncaptured, as
  the qualification reads its own rows);
- **A, a flight inserted** — a flight of the draw at the same airport and split (one the window does not hold: it flies
  on its own dynamics, `traffic_window.draw_windows`; as `traffic_augment` A draws from the scene's split), landing on a runway that must be spaced from a commanded aircraft's
  (drawn: its new follower; the same runway, a pair separated as one, a dependent parallel), moved whole to cross its
  threshold g × the required gap before the follower's recorded crossing on the approach clock (`traffic_augment` A's
  timing), g ~ U[`traffic_augment.GAP_RANGE`] — and commanded too, keyed `traffic_speaking.INSERTED` after its own key,
  over its own time limit; the window no longer replays the flight it came from (its recorded landing stays in the
  landing contexts, as in `traffic_augment`).

The ways of "one aircraft a window commanded" (design §5.2):

- **D, the leader moved** — the replayed flight landing just before the commanded one on its runway or one separated as
  one with it, in the air at its first predicted step (`leader`: by landing order, not by where the approach clock has
  them before they turn final — design §2.5's reading), moved whole by δ drawn evenly from the whole steps in
  [−`traffic_augment.SHIFT_S`, `traffic_augment.SHIFT_S`] but 0 (a leader not moved is not an augmentation); its moved
  landing is the one in the commanded aircraft's landing context (`traffic_speaking.scene_landings`);
- **B, the start moved** — as above;
- **A, a flight inserted** — as above, drawn and timed alike, but REPLAYED along its record: the commanded aircraft's new
  leader (one aircraft is commanded).

**Every shift is whole steps** (even seconds, so whole seconds too): a moved flight's rows stay on the scene's steps
(`prior.scene`, `traffic_augment.moved`), the rosters' landing times are whole seconds, and a flight's own landing leaves
its landing context by its exact time (`data.own_context`). A moved or inserted flight moves whole — its rows and record
(`traffic_augment.moved`) and its signals (`shifted`: its entry and landing times, so its rows' times, its landing in the
others' context and the landing direction at its first predicted step are the moved ones'). A moved flight keeps its
rows' inputs and words, as in `traffic_augment`.

**Qualification** (§5.3): every commanded aircraft, through its observed rows (to its first predicted step) judged not
established — as the loop judges its first step — against every other aircraft of the window along its record (a moved
one's moved), answers for no loss under the loop's reading: a window already lost when the prior starts to speak is not
the prior's. **Stated reading**: this differs from the loop — the other commanded aircraft are read along their records,
before their first predicted steps too, where the loop's judge does not see them yet (stricter there), and after them,
where the loop reads the paths the model flies them on — and a draw refused drops the whole window (a window as drawn
only leaves out the aircraft starting in a loss). The window never holds more aircraft on one step, along the records,
than the data has had (§5.2): its airport's busiest step on the training days (`busiest`), or — where that is busier —
the same span as recorded (the window's commanded aircraft unmoved, and every flight the augmented window replays: a
longer time limit can bring in recorded traffic, which is not added): only what the augmentation adds is capped. Every
commanded aircraft's time limit must fit the model's positions (a moved start's stage-2 limit can pass them). A draw
that cannot apply (no plausible start, no flight to insert) or fails is drawn again, at most `traffic_augment.TRIES`
draws, else the window is left out; the draws refused are counted by why.
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
from ts_transformer.experiments.traffic_augment import GAP_RANGE, SHIFT_S, TRIES, moved
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
from ts_transformer.prior.generate import rows_for
from ts_transformer.prior.scene import N_LOOK, SceneIndex, in_scene, presence, scene_steps

#: The ways of augmenting (design §5.2), and each draw's (module docstring), in the order a kind is drawn from.
KINDS = ("A", "B", "C", "D")
KINDS_OF = {"every": ("C", "B", "A"), "one": ("D", "B", "A")}
COMPRESSION = (0.6, 1.0)
#: A commanded aircraft's part in its window's augmentation: moved in time (C), its start moved (B), inserted (A).
ROLES = ("shifted", "moved", "inserted")
#: Why a draw was refused (module docstring).
REFUSALS = ("no_plausible_start", "no_flight_to_insert", "no_leader_to_move", "longer_than_the_model",
            "busier_than_the_data", "lost_before_the_model_speaks")


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


def whole_steps(dt_s: float, step_s: float) -> int:
    """``dt_s`` rounded to the nearest whole step, in (whole) seconds."""
    return int(round(dt_s / step_s) * step_s)


def busiest(directory: Path, spec: VocabularySpec, airports: Sequence[str], step_s: float) -> dict[str, int]:
    """Each airport's most aircraft on one step of the training days — every arrival, a background one too, from its
    first row's step to its last's (`prior.scene.in_scene`; a flight with a sentence over its sentence's
    rows): the cap on an augmented window (§5.2)."""
    geometries = load_candidates(directory)
    sentences = load_sentences(directory, "train", spec)
    rows = {int(i): int(sentences["offsets"][k + 1] - sentences["offsets"][k])
            for k, i in enumerate(sentences["signal_index"])}
    flights: dict[str, list[Any]] = {code: [] for code in airports}
    for i, flight in enumerate(load_signals(directory, "train")):
        if flight.airport in flights:
            flights[flight.airport].append(presence(flight, rows.get(i), geometries[flight.airport], step_s))
    return {code: max(at_once(segment, step_s) for segment in SceneIndex(part).segments())
            for code, part in flights.items()}


def at_once(flights: Sequence[Any], step_s: float) -> int:
    """The most of ``flights`` (their presences) in the scene on one step."""
    first = min(p.start_s for p in flights)
    last = max(p.end_s for p in flights)
    return int(in_scene(flights, scene_steps(first, last, step_s)).max())


def leader(window: Window, key: str, step_s: float) -> str | None:
    """The replayed flight landing just before commanded aircraft ``key`` on its landed runway or one separated as one
    with it, among those in the air at its first predicted step; None without one."""
    t0 = window.first_step_s(key) + N_LOOK * step_s
    own = window.track(key).presence
    separation = window.airport.flights.separation
    before = [(window.track(k).presence.landing_s, k) for k in window.others
              if window.track(k).on_step(t0) and separation.one_runway(window.track(k).presence.runway, own.runway)
              and window.track(k).presence.landing_s < own.landing_s]
    return max(before)[1] if before else None


def qualifies(window: Window, signals: Mapping[str, FlightSignals], step_s: float) -> bool:
    """No commanded aircraft of ``window`` answers for a loss through its observed rows (``signals``: each one's, a
    moved one's moved), judged not established, against every other aircraft along its record (module docstring)."""
    geometry, separation = window.airport.flights.geometry, window.airport.flights.separation
    cut = slice(0, N_LOOK + 1)
    for key in window.commanded:
        rows = window.rows(key)
        seen = presence(signals[key], len(rows.presence.times_s), geometry, step_s)
        whole = recorded(seen, signals[key], len(seen.times_s), 0.0, geometry, separation, rows.category, step_s)
        observed = dataclasses.replace(
            whole, times_s=whole.times_s[cut], e_m=whole.e_m[cut], n_m=whole.n_m[cut], height_m=whole.height_m[cut],
            runway=whole.runway[cut], along_m=whole.along_m[cut],
            track_minus_course_deg=whole.track_minus_course_deg[cut], right_of_course_m=whole.right_of_course_m[cut],
            along_speed_mps=whole.along_speed_mps[cut], established=whole.established[cut], outcome="timeout",
            landing_s=None)
        if judged(window.scene(key), observed, step_s)[1] is not None:
            return False
    return True


def augment_window(window: Window, batch: replay.Batch, places: Sequence[int], pool: Mapping[str, int],
                   limits: Sequence[float], moved_limits: Sequence[float], most: int, max_rows: int,
                   rng: np.random.Generator, windows: dict[str, tuple[float, float]], spec: VocabularySpec,
                   kinds: Sequence[str] = KINDS_OF["every"], commanded: str = "every"
                   ) -> tuple[AugmentedWindow | None, str, Counter]:
    """``window`` augmented (module docstring), its kind and the draws refused by why (`REFUSALS`): ``kinds`` those the
    kind is drawn from, each alike — of the kinds of a draw that commanded ``commanded`` (`KINDS_OF`). ``batch``: the draw's
    flights, the window's commanded ones at ``places`` (in its order); ``pool``: the flights an insertion is drawn from,
    by key, their places in ``batch`` (the draw's flights at this airport); ``limits`` / ``moved_limits``: every flight's
    time limit in ``batch``, the executor spec's and stage 2's; ``most``: the airport's busiest training step
    (`busiest`); ``max_rows``: the model's positions; ``windows``: stage 2's altitude windows. None after `TRIES` draws
    refused."""
    step_s = spec.step_s
    airport = window.airport
    geometry, separation = airport.flights.geometry, airport.flights.separation
    if not set(kinds) <= set(KINDS_OF[commanded]):
        raise ValueError(f"kinds {tuple(kinds)}: a draw that commanded {commanded!r} is augmented by {KINDS_OF[commanded]}")
    own = dict(zip(window.commanded, places))
    kind = kinds[int(rng.integers(len(kinds)))]
    refused: Counter = Counter()
    for draw in range(1, TRIES + 1):
        place, signals = dict(own), {k: batch.signals[j] for k, j in own.items()}
        limit = {k: limits[j] for k, j in own.items()}
        move: dict[str, Augmentation] = {}
        role: dict[str, str] = {}
        parts: list[tuple[FlightRows, Any]] = []
        left_out: list[str] = []
        if kind == "C":
            c = float(rng.uniform(*COMPRESSION))
            shift = {k: whole_steps((c - 1.0) * (window.rows(k).presence.start_s - window.opens_s), step_s)
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
        elif kind == "D":
            follower = window.commanded[int(rng.integers(len(window.commanded)))]
            lead = leader(window, follower, step_s)
            if lead is None:
                refused["no_leader_to_move"] += 1
                continue
            reach = int(SHIFT_S // step_s)
            steps = int(rng.integers(-reach, reach))
            dt = step_s * float(steps + 1 if steps >= 0 else steps)
            parts.append(moved(window.rows(lead), window.track(lead), dt, lead, step_s))
            drawn = {"leader": lead, "follower": follower, "shift_s": dt}
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
            dt = whole_steps(clock + separation.along_nm[source.presence.runway] * NM_M / separation.speed_mps
                             - source.presence.landing_s, step_s)
            key = source_key + INSERTED
            j = pool[source_key]
            parts.append(moved(airport.flights.flights[source_key], source, float(dt), key, step_s))
            if commanded == "every":                    # commanded too; one aircraft a window: replayed
                place[key], signals[key], limit[key], role[key] = (j, shifted(batch.signals[j], dt, key), limits[j],
                                                                   "inserted")
            left_out.append(source_key)
            drawn = {"inserted": source_key, "follower": follower, "gap": gap, "shift_s": dt}
        if rows_for(max(limit.values()) + step_s, step_s) > max_rows:
            refused["longer_than_the_model"] += 1
            continue
        placed = Window(airport, window.opens_s, tuple(place), (), tuple(parts))
        order = tuple(sorted(place, key=lambda k: (placed.first_step_s(k), k)))
        candidate = window_of(airport, window.opens_s, order, [limit[k] for k in order], step_s, parts,
                              left_out)
        # the same span as recorded: an inserted flight replayed (one aircraft a window) is what the augmentation adds
        cap = max(most, at_once([airport.tracks[k].presence for k in (*window.commanded, *candidate.others)
                                 if k in airport.tracks], step_s))
        count = at_once([candidate.track(k).presence for k in (*candidate.commanded, *candidate.others)], step_s)
        if count > cap:
            refused["busier_than_the_data"] += 1
            continue
        drawn["most_at_once"] = {"cap": cap, "window": count}
        if not qualifies(candidate, signals, step_s):
            refused["lost_before_the_model_speaks"] += 1
            continue
        return (AugmentedWindow(kind, candidate, tuple(place[k] for k in order),
                                tuple(signals[k] for k in order), tuple(move.get(k) for k in order),
                                tuple(limit[k] for k in order), tuple(role.get(k) for k in order), drawn, draw),
                kind, refused)
    return None, kind, refused
