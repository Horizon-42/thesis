"""The losses of separation of a window of stage D and the aircraft that answer for them (multi-aircraft control §2.4;
D144, D145).

The judge is stage C's (`post.traffic.step_losses`, VISUAL): at a step, every pair of the judged set, the window's
commanded aircraft first. A loss that holds a commanded aircraft is of one pair (`classify`, `PAIRS`):

- ``commanded_commanded``: two commanded aircraft, each answering when the rules make it responsible;
- ``commanded_answers``: a commanded and a recorded aircraft, the commanded one responsible;
- ``records_kept``: a commanded and a recorded aircraft, only the recorded one responsible, and the two records kept
  their separation at that step (the same pair judged on the records, `losses_on_records`): the commanded one answers
  (D145: the model made the loss, and the commanded aircraft is the only one that can be moved);
- ``recorded_only``: as the last, the records also losing it at that step: it earns nothing and costs nothing (D145).

**On the records** (`on_records`): the commanded aircraft on their records (moved, in a compressed window) and the
recorded ones on theirs. A commanded aircraft is never "over its threshold" (`AircraftAt.last_step` false), as in the
loop (post-training §9 item 8), so the records and the loop are judged by the same rules.

**Who answers in the loop** (`Answering`, the window loop's rule of who answers, post-training §9 item 5): of each
loss, the commanded aircraft that the rules make responsible, and the commanded one of a loss of ``records_kept``. The
loop asks it only for the aircraft that are said in that row and not silent: an aircraft in its observed rows answers
for nothing, a silent one never again (D144, D145).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ts_transformer.inference.runway_schedule import Separation
from ts_transformer.inference.separation import Loss
from ts_transformer.post.scene import AircraftAt, Window
from ts_transformer.post.traffic import scene_aircraft, step_losses, traffic
from ts_transformer.prior.procedure import Final

#: The pairs of a loss that holds a commanded aircraft (module docstring), in the order of the census.
PAIRS = ("commanded_commanded", "commanded_answers", "records_kept", "recorded_only")


@dataclass(frozen=True)
class Positions:
    """A caller's positions of a window's commanded aircraft: ``at(member, time_s)`` → `AircraftAt` fields of that
    aircraft at that step (`post.scene.Recorded.at_step`'s), or None when it is not in the air then; ``end_s`` the last
    time any of them is; ``records``: whether they are the records themselves."""

    at: Callable[[int, float], "tuple | None"]
    end_s: float
    records: bool


def on_records(window: Window) -> Positions:
    """The commanded aircraft on their records (moved, in a compressed window), never over their thresholds (module
    docstring)."""
    records = window.commanded_all
    interval = window.scene.interval_s

    def at(member: int, time_s: float):
        record = records[member]
        if not record.start_s <= time_s <= record.end_s:
            return None
        key, here, before, known, runway, category, _, go_around = record.at_step(time_s, interval)
        return key, here, before, known, runway, category, False, go_around

    return Positions(at, max(r.end_s for r in records), True)


def judged_step(window: Window, positions: Positions, step: int, separation: Separation, finals: Sequence[Final],
                step_s: float) -> tuple[AircraftAt, int, list[Loss]] | None:
    """The judged set of ``window``'s step ``step`` (its commanded aircraft at ``positions`` first, its recorded ones),
    the count of its commanded ones, and its losses; None when no commanded aircraft is in the air then."""
    time_s = window.step_s(step)
    present = [item for item in (positions.at(k, time_s) for k in range(len(window.commanded_all)))
               if item is not None]
    if not present:
        return None
    aircraft = scene_aircraft(AircraftAt.of(present), window.others_at(step))
    if len(aircraft) < 2:
        return aircraft, len(present), []
    scene = traffic(aircraft, window.scene.geometry, separation, finals, step_s)
    return aircraft, len(present), step_losses(scene, aircraft.last_step, separation)


def classify(i: int, j: int, responsible: Sequence[int], commanded: int, over: frozenset[int] = frozenset()
             ) -> str | None:
    """The pair of a loss between aircraft ``i`` and ``j`` of a step (the first ``commanded`` commanded, module
    docstring), before the records are read: ``recorded_only`` stands for both pairs of a loss that only the recorded
    aircraft is responsible for; None when the loss holds no commanded aircraft. A commanded aircraft in ``over`` — one
    that landed, judged once over its threshold (`post_window_loop.WindowLoop._judged`) — counts as a recorded one: it
    answers for nothing (as an aircraft in its observed rows, D145)."""
    def held(k: int) -> bool:
        return k < commanded and k not in over

    if held(i) and held(j):
        return "commanded_commanded"
    if not (held(i) or held(j)):
        return None
    return "commanded_answers" if any(held(k) for k in responsible) else "recorded_only"


def losses_on_records(window: Window, step: int, separation: Separation, finals: Sequence[Final], step_s: float
                      ) -> frozenset[frozenset[str]]:
    """The pairs (by their keys) that lose separation at ``window``'s step ``step`` on the records (module
    docstring)."""
    judged = judged_step(window, on_records(window), step, separation, finals, step_s)
    if judged is None:
        return frozenset()
    aircraft, _, losses = judged
    return frozenset(frozenset((aircraft.keys[x.i], aircraft.keys[x.j])) for x in losses)


class Answering:
    """Stage D's rule of who answers for a loss (module docstring; D145), for the window loop: ``(window, step, loss,
    aircraft, commanded)`` → the places in ``aircraft`` (the judged set, its ``commanded`` commanded aircraft first) of
    the commanded aircraft that answer. ``separations`` and ``finals`` by airport; the losses on the records of a step
    are judged once (by the window and the step)."""

    def __init__(self, separations: Mapping[str, Separation], finals: Mapping[str, Sequence[Final]], step_s: float
                 ) -> None:
        self.separations, self.finals, self.step_s = separations, finals, step_s
        #: by (the window's id, the step): the window (so that its id is not taken again) and its losses on the records
        self._on_records: dict[tuple[int, int], tuple[Window, frozenset[frozenset[str]]]] = {}

    def __call__(self, window: Window, step: int, loss: Loss, aircraft: AircraftAt, commanded: int
                 ) -> tuple[int, ...]:
        pair = classify(loss.i, loss.j, loss.responsible, commanded)
        if pair != "recorded_only":
            return tuple(k for k in loss.responsible if k < commanded)
        if frozenset((aircraft.keys[loss.i], aircraft.keys[loss.j])) in self.records(window, step):
            return ()
        return (min(loss.i, loss.j),)                       # the commanded one: the records kept their separation

    def records(self, window: Window, step: int) -> frozenset[frozenset[str]]:
        """`losses_on_records` of ``window``'s step ``step``, judged once."""
        key = (id(window), step)
        if key not in self._on_records:
            code = window.scene.geometry.code
            self._on_records[key] = (window, losses_on_records(window, step, self.separations[code], self.finals[code],
                                                                self.step_s))
        return self._on_records[key][1]
