"""The windows of stage D (multi-aircraft control D146, §3.1; MC1).

**A window** is an anchor — a flight with a closed-loop sentence at the chosen Δ, stage C's real window of it
(`post.scene.real_windows`) — and every other flight of the same airport and split with such a sentence whose row 0 is
in [the anchor's row 0, the anchor's row 0 + L): its commanded aircraft (`post.scene.Window.joined`, each joining at its
own row 0 on the window's steps). Every other flight of the airport and split in the air at a step is recorded and flies
its record. With L = 0 a window is stage C's real window. A window is drawn from each flight as anchor (`window_of`).

**A compressed window** (`compressed`): each commanded aircraft other than the anchor moved earlier, its offset from
the anchor's row 0 multiplied by c (c drawn uniformly in [c_min, 1]) and rounded to a whole Δ; its record moves with it
(its rows, its row 0, its first predicted step), its states and words do not change; the recorded aircraft do not move.

**Left out** (`left_out`, post-training D113 for each commanded aircraft): a window in which a commanded aircraft, on its
record at its first predicted step with no runway in force, loses separation that it answers for (the rules) against the
records of every other aircraft of the window — its other commanded aircraft on their records (moved, in a compressed
window) and its recorded aircraft. A stated limit (§3.1): in the loop the earlier aircraft fly the model's states, so an
aircraft can still open inside a loss that they made.

The kind of a window (`Drawn.kind`, `REAL` or `COMPRESSED`, with its c) is stage D's own: the window it holds is a real
window of stage C's code (`post.scene.REAL`) with its commanded aircraft joined.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ts_transformer.inference.runway_schedule import Separation
from ts_transformer.inference.separation import Loss
from ts_transformer.instructions.labeller.interval import OBSERVATION_S
from ts_transformer.post.scene import REAL, AircraftAt, Joined, Window
from ts_transformer.post.traffic import answered_loss, scene_aircraft, step_losses, traffic
from ts_transformer.prior.procedure import Final

#: The kinds of a window of stage D (module docstring; their counts are the user's, O16).
REAL_KIND, COMPRESSED = "real", "compressed"
KINDS = (REAL_KIND, COMPRESSED)


@dataclass(frozen=True)
class Drawn:
    """A window of stage D: the window (its commanded aircraft joined), its kind and its c (1 for a real window)."""

    window: Window
    kind: str
    c: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS or not 0.0 < self.c <= 1.0 or (self.kind == REAL_KIND) != (self.c == 1.0):
            raise ValueError(f"a window of stage D is {KINDS[0]} (c 1) or {KINDS[1]} (c in (0, 1)), not "
                             f"{self.kind} c {self.c}")


class Anchors:
    """The real windows of one split (`post.scene.real_windows`), by airport in the order of their row 0 (then their
    place in the signals): what `window_of` draws the commanded aircraft of a window from."""

    def __init__(self, windows: Sequence[Window]) -> None:
        if any(w.kind != REAL or w.joined for w in windows):
            raise ValueError("the anchors of stage D's windows are stage C's real windows")
        self.by_airport: dict[str, list[Window]] = {}
        for window in windows:
            self.by_airport.setdefault(window.scene.geometry.code, []).append(window)
        for items in self.by_airport.values():
            items.sort(key=lambda w: (w.row0_s, w.signal_index))
        self.row0_s = {code: np.array([w.row0_s for w in items]) for code, items in self.by_airport.items()}

    def window_of(self, anchor: Window, span_s: float) -> Window:
        """``anchor``'s window of span ``span_s`` (L, s; module docstring): every other flight of its airport and split
        with a sentence whose row 0 is in [the anchor's row 0, + L), in the order of their row 0."""
        if span_s < 0.0:
            raise ValueError(f"a window's span is not negative, got {span_s:g} s")
        code = anchor.scene.geometry.code
        items, row0 = self.by_airport[code], self.row0_s[code]
        low, high = np.searchsorted(row0, anchor.row0_s, side="left"), np.searchsorted(row0, anchor.row0_s + span_s,
                                                                                         side="left")
        joined = tuple(Joined(w.commanded, w.signal_index) for w in items[low:high]
                       if w.signal_index != anchor.signal_index)
        return replace(anchor, joined=joined, span_s=float(span_s))


def compressed(window: Window, rng: np.random.Generator, c_min: float) -> Drawn:
    """``window`` compressed (module docstring): c drawn uniformly in [``c_min``, 1), each commanded aircraft after the
    anchor moved to its offset × c, rounded to a whole Δ."""
    if not 0.0 < c_min < 1.0:
        raise ValueError(f"c_min is in (0, 1), got {c_min:g}")
    c = float(rng.uniform(c_min, 1.0))
    interval = window.scene.interval_s
    moved = []
    for item in window.joined:
        offset = int(round((item.record.first_step_s - window.first_step_s) / interval))
        shift = (int(round(offset * c)) - offset) * interval
        moved.append(Joined(item.record.shifted(shift, interval), item.signal_index, shift_s=shift))
    return Drawn(replace(window, joined=tuple(moved)), COMPRESSED, c)


def commanded_at(window: Window, time_s: float, *, without: int | None = None) -> AircraftAt:
    """The window's commanded aircraft in the air at ``time_s`` on their records (moved, in a compressed window), in
    their order, but the ``without``-th."""
    interval = window.scene.interval_s
    return AircraftAt.of([record.at_step(time_s, interval) for k, record in enumerate(window.commanded_all)
                          if k != without and record.start_s <= time_s <= record.end_s])


def opening_loss(window: Window, member: int, separation: Separation, finals: Sequence[Final], step_s: float
                 ) -> Loss | None:
    """The loss of separation that the window's commanded aircraft ``member`` answers for at its first predicted step,
    on its record with no runway in force, against the records of every other aircraft of the window (module
    docstring); None without one."""
    record = window.commanded_all[member]
    time_s = record.first_step_s
    step = int(round((time_s - window.row0_s) / window.scene.interval_s))
    own = AircraftAt.of([record.at_step(time_s, window.scene.interval_s)])      # R not in force at its first step
    others = scene_aircraft(window.others_at(step), commanded_at(window, time_s, without=member))
    aircraft = scene_aircraft(own, others)
    scene = traffic(aircraft, window.scene.geometry, separation, finals, step_s)
    return answered_loss(step_losses(scene, aircraft.last_step, separation), 0)


def left_out(window: Window, separation: Separation, finals: Sequence[Final], step_s: float) -> list[int]:
    """The commanded aircraft of ``window`` that open inside a loss they answer for (module docstring): the window is
    left out of the draw when any does."""
    return [k for k in range(len(window.commanded_all))
            if opening_loss(window, k, separation, finals, step_s) is not None]


def first_step_rows(window: Window) -> np.ndarray:
    """``[commanded]`` each commanded aircraft's first predicted step on the window's steps."""
    return window.join_steps() + int(round(OBSERVATION_S / window.scene.interval_s))
