"""Augmented scenes for the loop where the prior commands one aircraft (multi-aircraft design §5, §6.6 step 5), not a runner:
a speaking flight's scene changed so that it has traffic to deal with that the data rarely holds. Only in the closed loop
— reward and readouts — never teacher forcing (§5.1: an inserted flight's words were not said to the traffic it now has).

The three ways of "one aircraft commanded" (§5.2, §9 item 7), a third each:

- **D, the leader moved** — the replayed flight landing just before the speaking one on its runway (in the air at its
  first predicted step) moved whole by δ ~ U[−`SHIFT_S`, `SHIFT_S`];
- **B, the start moved** — the speaking flight's own start moved as post-training stage 2 moves it (`prior.augment`:
  rotated about the airport, raised, sped up; drawn until plausible, `prior_free_generation.augmented_starts`), the
  others as they were;
- **A, a flight inserted** — a flight of the same airport and split with a sentence, not in the scene, landing in the
  speaking flight's landing direction, moved whole to land ``g`` × the distance the rules require (at the approach speed,
  `traffic_census.APPROACH_SPEED_MPS`) before the speaking flight's recorded landing, g ~ U[`GAP_RANGE`]: its new leader.

Leaders are by landing order, not the approach clock: before they turn final (on a downwind) the clock does not say who
lands first (design §2.5) — a reading of §5.2's "the leader" / "the gap to the leader", stated.

A moved or inserted flight keeps its rows' inputs (its landing context its own time's, as stage 2's moved start keeps its
own) and its labelled words; an inserted one is keyed `INSERTED` after its own key.

**Qualification** (§5.3): through the speaking flight's observed rows (to its first predicted step) no loss it answers
for, under the loop's reading — a scene already lost when the prior starts to speak is not the prior's; a kind that cannot
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
from ts_transformer.experiments.traffic_census import APPROACH_SPEED_MPS, Track
from ts_transformer.experiments.traffic_loop import Controlled, recorded
from ts_transformer.experiments.traffic_scene_data import FlightRows
from ts_transformer.experiments.traffic_speaking import Scene, judged
from ts_transformer.inference.runway_schedule import UNRELATED
from ts_transformer.inference.separation import VISUAL
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.prior.augment import Augmentation, augment_signals
from ts_transformer.prior.scene import N_LOOK, hung_span, presence

KINDS = ("D", "B", "A")
SHIFT_S = 60.0
GAP_RANGE = (0.5, 2.0)
TRIES = 10
INSERTED = "+inserted"


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
    """A flight moved ``dt_s`` in time and keyed ``key``: its rows, inputs and words as they were."""
    shifted = dataclasses.replace(rows.presence, dataset_id=key, times_s=rows.presence.times_s + dt_s,
                                  landing_s=rows.presence.landing_s + dt_s)
    first, last = hung_span(shifted, step_s)
    return (dataclasses.replace(rows, presence=shifted),
            dataclasses.replace(track, presence=shifted, first_step_s=first, last_step_s=last,
                                captured_s=track.captured_s + dt_s))


def leader(scene: Scene, step_s: float) -> str | None:
    """The replayed flight landing just before the speaking one on its landed runway, among those in the air at its first
    predicted step; None without one."""
    t0 = scene.first_step_s + N_LOOK * step_s
    own = scene.track(scene.key).presence
    before = [(scene.track(k).presence.landing_s, k) for k in scene.others
              if scene.track(k).on_step(t0) and scene.track(k).presence.runway == own.runway
              and scene.track(k).presence.landing_s < own.landing_s]
    return max(before)[1] if before else None


def qualifies(scene: Scene, signals: FlightSignals, step_s: float) -> bool:
    """No loss the speaking flight answers for through its observed rows (to its first predicted step), under the
    loop's reading: ``signals`` its rows (its own, or its moved start's)."""
    own = scene.track(scene.key)
    rows = N_LOOK + 1
    seen = presence(signals, len(own.presence.times_s), scene.airport.flights.geometry)
    capture = int(np.searchsorted(seen.times_s, own.captured_s))
    whole: Controlled = recorded(seen, signals, capture, 0.0, scene.airport.flights.geometry,
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
            dt = float(rng.uniform(-SHIFT_S, SHIFT_S))
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
                       and separation.relation(scene.airport.tracks[k].presence.runway, own.presence.runway) != UNRELATED]
            if not options:
                continue
            key = options[int(rng.integers(len(options)))]
            source = scene.airport.tracks[key]
            gap = float(rng.uniform(*GAP_RANGE))
            required = separation.distance_nm(source.presence.runway, source.category, own.presence.runway,
                                              own.category) * NM_M
            dt = own.presence.landing_s - gap * required / APPROACH_SPEED_MPS - source.presence.landing_s
            inserted = moved(scene.airport.flights.flights[key], source, dt, key + INSERTED, step_s)
            candidate = dataclasses.replace(scene, others=(*scene.others, key + INSERTED), moved=(inserted,))
            if qualifies(candidate, signals, step_s):
                return Augmented(kind, candidate, signals, None, {"inserted": key, "gap": gap, "required_m": required,
                                                                   "shift_s": dt}, draw)
    return None
