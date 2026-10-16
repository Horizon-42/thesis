"""Branch points found backward from each event (multi-aircraft control D182, MC12): stage D's search of where a varied
aircraft's words can still change its event. The loop's work (the second pass's copies, the continuations flown) is the
shared branch round's (`experiments.post_branches.backward_round`); this module decides.

- **The setting** (`MultiSettings.branching`): `GRID` (D143's points, the behaviour before D182), `BACKWARD` or
  `MAJORITY` (D192: the search with the majority point in place of point B).
- **The event** of a varied aircraft v (`event`): D143's — the tick of v's own end or of the earliest loss that it is
  in — and for an aircraft that landed after a go-around, its first go-around if earlier. Its kind: `LOSS` (with the
  other aircraft of that loss), `GO_AROUND`, or `END` (v's outcome).
- **The candidate points** (`points`): back from the event every `COARSE_S`, the latest first, at most the search's
  count (`POINTS`, D182's; D217: `MultiSettings.search_points`), never before v's first predicted step; ticks of the
  window's grid.
- **A continuation avoids the event** (`avoids`, D182 (3)), judged at the judged span after it (`judged_until`;
  `AFTER_S`, D182's; D217: `MultiSettings.judged_after_s`): v answers for
  no loss, for every kind of event (a worse end, a loss of separation, does not take the event's place), and for a
  loss the pair keeps its separation, for a go-around v says none, for another end v's same outcome does not happen.
- **The search** (`search`): the first point going back where a continuation avoids the event is the decision point,
  a group; then the point `FINE_S` after it, a group when a continuation of it avoids the event (none with D217's
  `MultiSettings.fine_point` off); then point B, the first
  point (from the decision point back, within the same points) where a continuation avoids the event with no go-around
  said by v from the point to the judged tick, a group — B is the decision point when one of its avoiding
  continuations says none, and then one group. Every other point tried gives none; an event that no point avoids gives
  none. With `MAJORITY` (D192) the majority point takes B's place: the first point (from the decision point back,
  within the same points) where at least ``majority`` continuations avoid the event, go-arounds counted as avoiding, a
  group; the decision point when it has that many, and then one group; none when no point has. The search is a
  generator of its steps, so that a caller steps the searches of a batch together (D188): it asks
  for a point's continuations flown to the judged tick and judged (`TRY`), and says which point tried is chosen, to be
  flown on to its window's end (`CHOOSE`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Sequence

from ts_transformer.multi.credit import Aircraft
from ts_transformer.multi.credit import event as d143_event

#: `MultiSettings.branching`: D143's points (the default, the behaviour before D182), the backward search with point B
#: (D182) or with the majority point (D192).
GRID, BACKWARD, MAJORITY = "grid", "backward", "majority"
BRANCHINGS = (GRID, BACKWARD, MAJORITY)
#: The search's steps (D182): back from the event every 32 s, at most 8 points; one point 16 s after the decision point;
#: each continuation tried judged at 60 s after the event. D217: the count, the fine point and the judged span are
#: `MultiSettings`' settings, these their defaults (the behaviour before D217).
COARSE_S, FINE_S, POINTS, AFTER_S = 32.0, 16.0, 8, 60.0
#: The kinds of an event (module docstring).
LOSS, GO_AROUND, END = "loss", "go_around", "end"
#: The kinds of a search's groups (`Found.kinds`): the decision point, the fine point, point B, the majority point.
DECISION, FINE, B, MAJORITY_POINT = "decision", "fine", "b", "majority"
#: What a search asks of its caller (`search`): a point's continuations tried, or the point last tried chosen.
TRY, CHOOSE = "try", "choose"


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
    None when it is the decision point), every point tried, in order, and the majority point (`MAJORITY` only; None:
    none; the decision point when it is the majority point, and then no group of its own)."""

    decision: int | None
    fine: int | None
    b: int | None
    tried: tuple[int, ...]
    majority: int | None = None

    @property
    def groups(self) -> tuple[int, ...]:
        """The points that give groups, in the order they are chosen."""
        return tuple(p for _, p in self._chosen())

    @property
    def kinds(self) -> tuple[str, ...]:
        """The kind of each point of `groups`, in the same order."""
        return tuple(kind for kind, _ in self._chosen())

    def _chosen(self) -> list[tuple[str, int]]:
        behind = self.majority if self.majority != self.decision else None
        return [(kind, p) for kind, p in ((DECISION, self.decision), (FINE, self.fine), (B, self.b),
                                          (MAJORITY_POINT, behind)) if p is not None]


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


def points(happened: Event, first_step: int, interval_s: float, count: int = POINTS) -> list[int]:
    """The candidate points of an event (module docstring), the latest first, at most ``count``."""
    step = ticks_of(COARSE_S, interval_s)
    return [t for t in (happened.tick - step * j for j in range(1, count + 1)) if t >= first_step]


def judged_until(happened: Event, interval_s: float, after_s: float = AFTER_S) -> int:
    """The tick at which a continuation is judged: the event + ``after_s``."""
    return happened.tick + ticks_of(after_s, interval_s)


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


def out_of_reach(failed: int, continuations: int, need: int) -> bool:
    """Whether a point with ``failed`` of its ``continuations`` already judged not to avoid the event can no longer be
    chosen, ``need`` of them having to avoid it (D194 (3)): 1 for the decision point, the fine point and point B, the
    majority for the majority point (D192). A continuation that does not avoid it by some tick does not avoid it at the
    judged tick: a loss answered for, an end and a go-around said stay (`avoids`). The point's continuations can then
    end at that tick: the search's answer is the same."""
    return continuations - failed < need


def search(candidates: Sequence[int], fine_ticks: int | None, majority: int | None = None
           ) -> Generator[tuple[str, int], Sequence[Trial] | None, Found]:
    """The search of one varied aircraft (module docstring) over its ``candidates`` (the latest first), step by step:
    it yields ``(TRY, p)``, and the caller flies point p's continuations to the judged tick and sends their trials back;
    it yields ``(CHOOSE, p)`` for the point it tried last, and the caller flies that point's continuations on to their
    window's end, makes its group and sends nothing; it returns what it found. ``majority``: the avoiding continuations
    that make the majority point (D192), which then takes point B's place; None: point B (D182). ``fine_ticks``: the
    fine point's ticks after the decision point; None: no fine point (D217)."""

    def behind(trials: Sequence[Trial]) -> bool:
        """Whether a point's ``trials`` make it point B, or the majority point."""
        if majority is None:
            return any(t.avoids and not t.go_around for t in trials)
        return sum(t.avoids for t in trials) >= majority

    tried: list[int] = []
    decision = fine = b = None
    for k, p in enumerate(candidates):
        trials = yield TRY, p
        tried.append(p)
        if any(t.avoids for t in trials):
            decision = p
            yield CHOOSE, p
            break
    if decision is None:
        return Found(None, None, None, tuple(tried))
    at_decision = behind(trials)
    if fine_ticks is not None:
        point = decision + fine_ticks
        tried.append(point)
        if any(t.avoids for t in (yield TRY, point)):
            fine = point
            yield CHOOSE, point
    if not at_decision:
        for p in candidates[k + 1:]:
            tried.append(p)
            if behind((yield TRY, p)):
                b = p
                yield CHOOSE, p
                break
    if majority is None:
        return Found(decision, fine, b, tuple(tried))
    return Found(decision, fine, None, tuple(tried), majority=decision if at_decision else b)
