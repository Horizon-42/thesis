"""Branch points found backward from each event (multi-aircraft control D182, MC12): stage D's search of where a varied
aircraft's words can still change its event. The loop's work (the second pass's copies, the continuations flown) is the
shared branch round's (`experiments.post_branches.backward_round`); this module decides.

- **The setting** (`MultiSettings.branching`): `GRID` (D143's points, the behaviour before D182) or `BACKWARD`.
- **The event** of a varied aircraft v (`event`): D143's — the tick of v's own end or of the earliest loss that it is
  in — and for an aircraft that landed after a go-around, its first go-around if earlier. Its kind: `LOSS` (with the
  other aircraft of that loss), `GO_AROUND`, or `END` (v's outcome).
- **The candidate points** (`points`): back from the event every `COARSE_S`, the latest first, at most `POINTS`, never
  before v's first predicted step; ticks of the window's grid.
- **A continuation avoids the event** (`avoids`, D182 (3)), judged at `AFTER_S` after it (`judged_until`): v answers for
  no loss, for every kind of event (a worse end, a loss of separation, does not take the event's place), and for a
  loss the pair keeps its separation, for a go-around v says none, for another end v's same outcome does not happen.
- **The search** (`search`): the first point going back where a continuation avoids the event is the decision point,
  a group; then the point `FINE_S` after it, a group when a continuation of it avoids the event; then point B, the first
  point (from the decision point back, within the same points) where a continuation avoids the event with no go-around
  said by v from the point to the judged tick, a group — B is the decision point when one of its avoiding
  continuations says none, and then one group. Every other point tried gives none; an event that no point avoids gives
  none. The caller flies a point's continuations to the judged tick (``trial``), and a point chosen to its window's end
  right after it is tried (``chosen``), so that one point's continuations are flown at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ts_transformer.multi.credit import Aircraft
from ts_transformer.multi.credit import event as d143_event

#: `MultiSettings.branching`: D143's points (the default, the behaviour before D182) or the backward search.
GRID, BACKWARD = "grid", "backward"
BRANCHINGS = (GRID, BACKWARD)
#: The search's steps (D182): back from the event every 32 s, at most 8 points; one point 16 s after the decision point;
#: each continuation tried judged at 60 s after the event.
COARSE_S, FINE_S, POINTS, AFTER_S = 32.0, 16.0, 8, 60.0
#: The kinds of an event (module docstring).
LOSS, GO_AROUND, END = "loss", "go_around", "end"


@dataclass(frozen=True)
class Event:
    """A varied aircraft's event (module docstring): its tick and kind; for a loss the other aircraft's key, for an end
    v's outcome."""

    tick: int
    kind: str
    partner: str | None = None
    outcome: str | None = None


@dataclass(frozen=True)
class Trial:
    """One continuation of a point, judged at the event + `AFTER_S`: whether it avoids the event, and whether v said a
    go-around from the point to then."""

    avoids: bool
    go_around: bool


@dataclass(frozen=True)
class Found:
    """A search's points (ticks): the decision point, the fine point and point B that give groups (None: no group; B is
    None when it is the decision point), and every point tried, in order."""

    decision: int | None
    fine: int | None
    b: int | None
    tried: tuple[int, ...]

    @property
    def groups(self) -> tuple[int, ...]:
        """The points that give groups, in the order they are chosen."""
        return tuple(p for p in (self.decision, self.fine, self.b) if p is not None)


def ticks_of(seconds: float, interval_s: float) -> int:
    """``seconds`` in ticks of the window's grid (Δ = ``interval_s``), refused unless whole."""
    ticks = seconds / interval_s
    if ticks != int(ticks) or ticks < 1:
        raise ValueError(f"{seconds:g} s is no whole number of {interval_s:g} s ticks")
    return int(ticks)


def event(aircraft: Sequence[Aircraft], v: int, outcome: str, first_go_around: int | None) -> Event:
    """Aircraft ``v``'s event (module docstring) from the window's commanded aircraft of the first sentence: ``outcome``
    v's outcome, ``first_go_around`` the tick of its first go-around word when it landed after one (else None)."""
    tick = d143_event(aircraft, v)
    if first_go_around is not None and first_go_around < tick:
        return Event(first_go_around, GO_AROUND)
    key = aircraft[v].key
    if aircraft[v].loss_step == tick:
        return Event(tick, LOSS, partner=aircraft[v].other)
    partner = next((a.key for a in aircraft if a.loss_step == tick and a.other == key), None)
    if partner is not None:
        return Event(tick, LOSS, partner=partner)
    return Event(tick, END, outcome=outcome)


def points(happened: Event, first_step: int, interval_s: float) -> list[int]:
    """The candidate points of an event (module docstring), the latest first."""
    step = ticks_of(COARSE_S, interval_s)
    return [t for t in (happened.tick - step * j for j in range(1, POINTS + 1)) if t >= first_step]


def judged_until(happened: Event, interval_s: float) -> int:
    """The tick at which a continuation is judged: the event + `AFTER_S`."""
    return happened.tick + ticks_of(AFTER_S, interval_s)


def avoids(happened: Event, *, answered: str | None, partner_answered_v: bool, ended: str | None,
           go_around: bool) -> bool:
    """Whether a continuation avoids ``happened`` (module docstring, D182 (3)), from what it did by the judged tick: the
    other aircraft of the loss v answered for (``answered``, None: none), whether the event's partner answered for a loss
    with v (``partner_answered_v``), v's outcome if it ended (``ended``) and whether it said a go-around
    (``go_around``). Every kind of event requires that v answered for no loss."""
    if answered is not None:
        return False
    if happened.kind == LOSS:
        return not partner_answered_v
    if happened.kind == GO_AROUND:
        return not go_around
    return ended != happened.outcome


def search(candidates: Sequence[int], fine_ticks: int, trial: Callable[[int], Sequence[Trial]],
           chosen: Callable[[int], None]) -> Found:
    """The search of one varied aircraft (module docstring) over its ``candidates`` (the latest first): ``trial(p)``
    flies point p's continuations to the judged tick and judges them; ``chosen(p)`` flies the point last tried to its
    window's end and makes its group."""
    tried: list[int] = []
    decision = fine = b = None
    for k, p in enumerate(candidates):
        trials = trial(p)
        tried.append(p)
        if any(t.avoids for t in trials):
            decision = p
            chosen(p)
            break
    if decision is None:
        return Found(None, None, None, tuple(tried))
    at_decision = any(t.avoids and not t.go_around for t in trials)
    point = decision + fine_ticks
    tried.append(point)
    if any(t.avoids for t in trial(point)):
        fine = point
        chosen(point)
    if not at_decision:
        for p in candidates[k + 1:]:
            tried.append(p)
            if any(t.avoids and not t.go_around for t in trial(p)):
                b = p
                chosen(p)
                break
    return Found(decision, fine, b, tuple(tried))
