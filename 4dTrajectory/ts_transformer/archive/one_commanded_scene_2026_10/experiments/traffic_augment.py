"""Augmented scenes for the loop where the prior commands one aircraft (multi-aircraft design §5, §6.6 step 5), not a runner:
a speaking flight's scene changed so that it has traffic to deal with that the data rarely holds. Only in the closed loop
— reward and readouts — never teacher forcing (§5.1: an inserted flight's words were not said to the traffic it now has).

The three ways of "one aircraft commanded" (§5.2, §9 item 7), a third each:

- **D, the leader moved** — the replayed flight landing just before the speaking one on its runway or one separated as
  one with it (`Separation.one_runway`), in the air at its first predicted step, moved whole by δ drawn evenly from the
  whole steps in [−`SHIFT_S`, `SHIFT_S`] but 0 (a leader not moved is not an augmentation);
- **B, the start moved** — the speaking flight's own start moved as post-training stage 2 moves it (`prior.augment`:
  rotated about the airport, raised, sped up; drawn until plausible, `prior_free_generation.augmented_starts`), the
  others as they were;
- **A, a flight inserted** — a flight of the same airport and split with a sentence, not in the scene, landing on a
  runway that must be spaced from the speaking flight's (the same, a pair separated as one, a dependent parallel: a
  required distance above 0), moved whole to cross its threshold ``g`` × the required gap before the speaking flight's
  recorded crossing ON THE APPROACH CLOCK (`Separation.approach_time_s`, `gap_s`: staggered thresholds counted, trap T3),
  g ~ U[`GAP_RANGE`] — the move rounded to the nearest whole step: its new leader.

**Every move is whole steps** (`moved`): a moved flight's rows stay on the scene's steps (`prior.scene`).

Leaders are by landing order (on the approach clock for A), not by where the approach clock has them before they turn
final: on a downwind the clock does not say who lands first (design §2.5) — a reading of §5.2's "the leader", stated.

**What moves with the scene**: the speaking flight's landing context (the landings before each of its steps, for a variant
that reads it) is the scene's — the moved leader's landing at its moved time, the inserted flight's landing added
(`traffic_speaking.scene_landings`). A moved or inserted flight's own inputs keep the landing context of its own time (design §5.2 asks the
scene's; rebuilding a replayed flight's rows is left out — its inputs reach the speaking one only through the traffic
attention), stated. §5.3's checks that do not bear on one commanded aircraft are not made: losses between replayed flights
end no one, and a replayed flight needs no dynamics.

A moved or inserted flight keeps its rows' inputs (its landing context its own time's, as stage 2's moved start keeps its
own) and its labelled words; an inserted one is keyed `INSERTED` after its own key.

**Qualification** (§5.3): through the speaking flight's observed rows (to its first predicted step), judged as the loop
judges its first step — not established (its executor starts uncaptured) — no loss it answers for, under the loop's
reading — a scene already lost when the prior starts to speak is not the prior's; a kind that cannot
apply (no leader to move, no flight to insert) or a draw that fails is drawn again, at most `TRIES` draws, else the flight
is left out (the caller counts it).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from geokit import NM_M

from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.experiments.prior_free_generation import augmented_starts
from ts_transformer.experiments.traffic_census import Track
from ts_transformer.experiments.traffic_loop import Controlled, recorded
from ts_transformer.experiments.traffic_scene_data import FlightRows
from ts_transformer.experiments.traffic_speaking import INSERTED, Scene, judged
from ts_transformer.inference.runway_schedule import DEPENDENT, SAME, SINGLE
from ts_transformer.inference.separation import VISUAL
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.prior.augment import Augmentation, augment_signals
from ts_transformer.prior.scene import N_LOOK, presence

KINDS = ("D", "B", "A")
SHIFT_S = 60.0
GAP_RANGE = (0.5, 2.0)
TRIES = 10


@dataclass(frozen=True)
class Augmented:
    """A speaking flight's augmented scene: the kind, the scene, the speaking flight's signals and start (B's moved
    ones; else its own, ``augmentation`` None), what was drawn, and how many draws it took."""

    kind: str
    scene: Scene
    signals: FlightSignals
    augmentation: Augmentation | None
    drawn: dict[str, Any]
    draws: int


def moved(rows: FlightRows, track: Track, dt_s: float, key: str, step_s: float) -> tuple[FlightRows, Track]:
    """A flight moved ``dt_s`` in time — whole steps, or refused: its rows stay on the scene's steps — and keyed ``key``:
    its rows, inputs and words as they were."""
    if dt_s % step_s:
        raise ValueError(f"{key}: a move of {dt_s} s is not whole {step_s:g} s steps")
    shifted = dataclasses.replace(rows.presence, dataset_id=key, times_s=rows.presence.times_s + dt_s,
                                  landing_s=rows.presence.landing_s + dt_s)
    return (dataclasses.replace(rows, presence=shifted),
            dataclasses.replace(track, presence=shifted, captured_s=track.captured_s + dt_s))


def leader(scene: Scene, step_s: float) -> str | None:
    """The replayed flight landing just before the speaking one on its landed runway, among those in the air at its first
    predicted step; None without one."""
    t0 = scene.first_step_s + N_LOOK * step_s
    own = scene.track(scene.key).presence
    separation = scene.airport.flights.separation
    before = [(scene.track(k).presence.landing_s, k) for k in scene.others
              if scene.track(k).on_step(t0) and separation.one_runway(scene.track(k).presence.runway, own.runway)
              and scene.track(k).presence.landing_s < own.landing_s]
    return max(before)[1] if before else None


def qualifies(scene: Scene, signals: FlightSignals, step_s: float) -> bool:
    """No loss the speaking flight answers for through its observed rows (to its first predicted step), under the
    loop's reading: ``signals`` its rows (its own, or its moved start's)."""
    own = scene.track(scene.key)
    rows = N_LOOK + 1
    seen = presence(signals, len(own.presence.times_s), scene.airport.flights.geometry, step_s)
    never = len(seen.times_s)                     # not established: as the loop's first step
    whole: Controlled = recorded(seen, signals, never, 0.0, scene.airport.flights.geometry,
                                 scene.airport.flights.separation, scene.speaking.category, step_s)
    cut = slice(0, rows)
    observed = dataclasses.replace(
        whole, times_s=whole.times_s[cut], e_m=whole.e_m[cut], n_m=whole.n_m[cut], height_m=whole.height_m[cut],
        runway=whole.runway[cut], along_m=whole.along_m[cut], track_minus_course_deg=whole.track_minus_course_deg[cut],
        right_of_course_m=whole.right_of_course_m[cut], along_speed_mps=whole.along_speed_mps[cut],
        established=whole.established[cut], outcome="timeout", landing_s=None)
    _, end, _ = judged(scene, observed, step_s, VISUAL)
    return end is None


def augment(scene: Scene, signals: FlightSignals, inputs: FlightInputs, pool: Sequence[str],
            rng: np.random.Generator, windows: dict[str, tuple[float, float]], step_s: float,
            kinds: Sequence[str] = KINDS) -> Augmented | None:
    """``scene`` augmented (module docstring): ``signals`` the speaking flight's, ``inputs`` its executor's state at the
    first predicted step (one flight), ``pool`` the keys an insertion is drawn from, ``windows`` stage 2's altitude
    windows, ``kinds`` those drawn from (each alike). None after `TRIES` draws that do not qualify."""
    separation = scene.airport.flights.separation
    own = scene.track(scene.key)
    for draw in range(1, TRIES + 1):
        kind = kinds[int(rng.integers(len(kinds)))]
        if kind == "D":
            key = leader(scene, step_s)
            if key is None:
                continue
            reach = int(SHIFT_S // step_s)
            steps = int(rng.integers(-reach, reach))
            dt = step_s * float(steps + 1 if steps >= 0 else steps)
            candidate = dataclasses.replace(scene, moved=(moved(scene.rows(key), scene.track(key), dt, key, step_s),))
            if qualifies(candidate, signals, step_s):
                return Augmented(kind, candidate, signals, None, {"leader": key, "shift_s": dt}, draw)
        elif kind == "B":
            move = augmented_starts([signals], inputs, rng, windows).moves[0]
            if move is None:
                continue
            start = augment_signals(signals, move)
            if qualifies(scene, start, step_s):
                return Augmented(kind, scene, start, move, dataclasses.asdict(move), draw)
        else:
            present = set(scene.others) | {scene.key}
            options = [k for k in pool if k not in present and scene.airport.flights.flights[k].presence.speaking
                       and separation.relation(scene.airport.tracks[k].presence.runway, own.presence.runway)
                       in (SAME, SINGLE, DEPENDENT)]
            if not options:
                continue
            key = options[int(rng.integers(len(options)))]
            source = scene.airport.tracks[key]
            gap = float(rng.uniform(*GAP_RANGE))
            required = separation.distance_nm(source.presence.runway, source.category, own.presence.runway,
                                              own.category) * NM_M
            clock = (separation.approach_time_s(own.presence.runway, own.presence.landing_s)
                     - gap * separation.gap_s(source.presence.runway, source.category, own.presence.runway, own.category))
            dt = step_s * round((clock + separation.along_nm[source.presence.runway] * NM_M / separation.speed_mps
                                 - source.presence.landing_s) / step_s)
            inserted = moved(scene.airport.flights.flights[key], source, dt, key + INSERTED, step_s)
            candidate = dataclasses.replace(scene, others=(*scene.others, key + INSERTED), moved=(inserted,))
            if qualifies(candidate, signals, step_s):
                return Augmented(kind, candidate, signals, None, {"inserted": key, "gap": gap, "required_m": required,
                                                                   "shift_s": dt}, draw)
    return None
