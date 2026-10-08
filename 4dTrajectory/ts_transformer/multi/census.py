"""The census of stage D's windows (multi-aircraft control MC1, §11; O16): what a window of span L holds, for the user to
choose L, the kinds of window and c_min.

For each window (`windows.Drawn`): its commanded aircraft; the recorded aircraft in the air at each commanded aircraft's
first predicted step; whether it is left out (`windows.left_out`, D146) and by which commanded aircraft (the anchor or
one after it); and the losses of separation on its steps (`window_losses`) — at each Δ step from the anchor's first
predicted step to the last row of its last commanded aircraft, every loss of the judge (`post.traffic.step_losses`,
VISUAL) that holds a commanded aircraft, by pair:

- ``commanded_commanded``: two commanded aircraft, each answering when the rules make it responsible;
- ``commanded_answers``: a commanded and a recorded aircraft, the commanded one responsible;
- ``records_kept``: a commanded and a recorded aircraft, only the recorded one responsible, and the two records kept
  their separation at that step (the same pair judged on the records): the loop charges the commanded aircraft (D145);
- ``recorded_only``: as the last, the records also losing it at that step: it earns nothing and costs nothing (D145).

On the records themselves (`on_records`) every loss is the records', so ``records_kept`` is never counted there.

The positions of the commanded aircraft are a caller's (`Positions`): their records (moved, in a compressed window) for
the census on the records; their stored closed-loop states for the baseline of §5 item 4 (`closed_loop_positions`: the
executor's own losses of separation), judged to the end of those states. A commanded aircraft is never "over its
threshold" on either side (`AircraftAt.last_step` false), as in the loop (post-training §9 item 8), so both judge the
same rules. A window's counts are whether it has a loss of each pair and the steps with one; `summary` gives the counts
of a set of windows by airport: no criterion (D7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping, Sequence

import numpy as np

from ts_transformer.inference.runway_schedule import Separation
from ts_transformer.inference.separation import Loss
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.grammar import apply
from ts_transformer.instructions.labeller.interval import OBSERVATION_S
from ts_transformer.instructions.words import Words
from ts_transformer.multi.separation import PAIRS, Positions, classify, judged_step, losses_on_records, on_records
from ts_transformer.multi.windows import Drawn, first_step_rows, left_out
from ts_transformer.post.scene import AircraftAt, Window
from ts_transformer.prior.procedure import Final

@dataclass
class WindowCount:
    """One window's census (module docstring)."""

    airport: str
    kind: str
    commanded: int
    recorded_at_first_steps: list[int]
    left_out_by: list[int]
    #: by pair: the steps judged with a loss of that pair
    loss_steps: dict[str, int] = field(default_factory=lambda: {pair: 0 for pair in PAIRS})
    steps: int = 0


def closed_loop_positions(window: Window, sentences: Mapping[int, ClosedLoopSentence], words: Words) -> Positions:
    """The commanded aircraft at their closed-loop sentences' states (vocabulary §6 item 3: observed before the first
    predicted step, flown from it, to the last row said) — the baseline of §5 item 4 (`flown_positions`; a stored
    sentence gives no landed runway here, so no commanded aircraft of the baseline is judged over its threshold: a
    stated limit)."""
    return flown_positions(window, [(sentences[i].rows.states, sentences[i].rows.grid, sentences[i].rows.start, None)
                                    for i in window.signal_indices], words)


def flown_positions(window: Window, flown: Sequence[tuple[np.ndarray, np.ndarray, int, int | None]], words: Words
                    ) -> Positions:
    """The commanded aircraft at the states they flew — for each, in the window's order, its states on the 2 s rows
    from its row 0 (`instructions.artefact.STATE_COLUMNS`), the words said to it from its first predicted step, the Δ
    row of that step and its landed runway (None unless the executor ended it `landed`): a closed-loop sentence's
    (`closed_loop_positions`) or the window loop's (stage D's readouts, §5 item 4) — each at its own row 0 in the window
    (moved, in a compressed window), its R and G those of its words in force before the row
    (`instructions.grammar.apply` along its words: R only from the row after its first predicted step, D23). An aircraft
    that landed is judged once more as the window loop judges it (`post_window_loop.WindowLoop._judged`): at the first
    step at or after its last row, over its threshold at its last state on its landed runway."""
    geometry, interval, step_s = window.scene.geometry, window.scene.interval_s, words.spec.step_s
    every = int(round(interval / step_s))
    tables = []
    for record, (states, grid, start, landed) in zip(window.commanded_all, flown, strict=True):
        state, in_force = None, []
        for q in range(start + len(grid) + 1):                    # before the words of each Δ row, and after the last
            in_force.append((state.runway, state.go_around) if state is not None else (-1, False))
            if start <= q < start + len(grid):
                height = float(states[q * every, 2]) - geometry.elevation_m
                state = apply(state, grid[q - start], height, words, len(geometry.candidates))
        # the step at which a landed aircraft is judged over its threshold: the first at or after its last row
        over = None if landed is None else -(-(len(states) - 1) // every) * every
        tables.append((record, states, in_force, landed, over))
    end_s = max(record.first_step_s - OBSERVATION_S + (len(states) - 1 if over is None else over) * step_s
                for record, states, _, _, over in tables)

    def at(member: int, time_s: float):
        record, states, in_force, landed, over = tables[member]
        r = int(round((time_s - (record.first_step_s - OBSERVATION_S)) / step_s))
        if r == over:
            return (record.key, tuple(states[-1, :3]), tuple(states[-2, :3]), True, landed, record.category, True,
                    False)
        if r < 0 or r >= len(states) or (over is not None and r > over):
            return None
        runway, go_around = in_force[min(r // every, len(in_force) - 1)]
        before = tuple(states[r - 1, :3]) if r else (0.0, 0.0, 0.0)
        return (record.key, tuple(states[r, :3]), before, r > 0, runway, record.category, False, go_around)

    return Positions(at, end_s, False)


def loop_positions(window: Window, ends: Sequence, start: int, words: Words) -> Positions:
    """The commanded aircraft of ``window`` at the states the window loop flew them (``ends``: their `WindowResult`s,
    in the window's order; ``start`` the Δ rows to a first predicted step), as the loop judges them
    (`post_window_loop.WindowLoop._landed`, `_judged`): `flown_positions` of each one's states and words, and its landed
    runway only in a window of several commanded aircraft (one that landed is judged once more over its threshold, as a
    recorded one). The positions stage D's readout and its Training export hand the census (`judged_steps`; frontend
    D178 (6))."""
    several = len(ends) > 1
    return flown_positions(window, [(e.states, e.words, start,
                                     int(e.crossing["runway_index"]) if several and e.crossing is not None else None)
                                    for e in ends], words)


@dataclass(frozen=True)
class StepLoss:
    """A loss of a judged step that holds a commanded aircraft: its pair as `classify` reads it (``recorded_only``
    before the records are read: `window_losses` splits off ``records_kept``), its two aircraft's keys and the judge's
    loss."""

    pair: str
    keys: tuple[str, str]
    loss: Loss


@dataclass(frozen=True)
class JudgedStep:
    """A step of a window at which a commanded aircraft is in the air: the judged set (its commanded aircraft first),
    the count of its commanded ones and its losses that hold a commanded aircraft, in the judge's order."""

    step: int
    aircraft: AircraftAt
    commanded: int
    losses: tuple[StepLoss, ...]


def judged_steps(window: Window, positions: Positions, separation: Separation, finals: Sequence[Final], step_s: float
                 ) -> Iterator[JudgedStep]:
    """The census's step loop (module docstring): each Δ step of ``window`` from the anchor's first predicted step to
    the end of ``positions`` at which a commanded aircraft is in the air, with its losses that hold one — a commanded
    aircraft that landed (`flown_positions`) judged over its threshold counts as recorded. The readout's census
    (`window_losses`) and the Training export's losses (frontend D177 (12)) both read it."""
    interval = window.scene.interval_s
    first = int(first_step_rows(window)[0])
    last = int(round((positions.end_s - window.row0_s) / interval))
    for step in range(first, last + 1):
        judged = judged_step(window, positions, step, separation, finals, step_s)
        if judged is None:
            continue
        aircraft, commanded, losses = judged
        over = frozenset(int(k) for k in np.flatnonzero(aircraft.last_step[:commanded]))   # a landed one (flown_positions)
        found = []
        for loss in losses:
            pair = classify(loss.i, loss.j, loss.responsible, commanded, over)
            if pair is not None:
                found.append(StepLoss(pair, (aircraft.keys[loss.i], aircraft.keys[loss.j]), loss))
        yield JudgedStep(step, aircraft, commanded, tuple(found))


def window_losses(window: Window, positions: Positions, separation: Separation, finals: Sequence[Final], step_s: float,
                  count: WindowCount) -> None:
    """The losses of ``window``'s steps into ``count`` (module docstring; `judged_steps`), its commanded aircraft at
    ``positions``, to the end of those positions."""
    for judged in judged_steps(window, positions, separation, finals, step_s):
        count.steps += 1
        found = set()
        kept_on_records = None
        for item in judged.losses:
            pair = item.pair
            if pair == "recorded_only" and not positions.records:     # the same pair on the records, at that step
                if kept_on_records is None:
                    kept_on_records = losses_on_records(window, judged.step, separation, finals, step_s)
                if frozenset(item.keys) not in kept_on_records:
                    pair = "records_kept"
            found.add(pair)
        for pair in found:
            count.loss_steps[pair] += 1


def window_count(drawn: Drawn, separation: Separation, finals: Sequence[Final], step_s: float,
                 positions: Positions | None = None) -> WindowCount:
    """One window's census (module docstring), its commanded aircraft on their records unless ``positions`` are
    given."""
    window = drawn.window
    firsts = first_step_rows(window)
    count = WindowCount(airport=window.scene.geometry.code, kind=drawn.kind, commanded=len(window.commanded_all),
                        recorded_at_first_steps=[len(window.others_at(int(s))) for s in firsts],
                        left_out_by=left_out(window, separation, finals, step_s))
    window_losses(window, on_records(window) if positions is None else positions, separation, finals, step_s, count)
    return count


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)), "max": float(values.max())}


def summary(counts: Sequence[WindowCount]) -> dict[str, dict]:
    """The census of ``counts`` by airport (and all of them): the windows, the commanded aircraft of a window, the
    recorded aircraft at a first predicted step, the windows left out (by the anchor, by a later aircraft), and by
    pair the windows with a loss and the share of the steps judged with one (the windows left out not counted)."""
    out: dict[str, dict] = {}
    groups: dict[str, list[WindowCount]] = {"all": list(counts)}
    for c in counts:
        groups.setdefault(c.airport, []).append(c)
    for code, items in sorted(groups.items()):
        kept = [c for c in items if not c.left_out_by]
        steps = sum(c.steps for c in kept)
        out[code] = {
            "windows": len(items),
            "commanded": _quantiles([c.commanded for c in items]),
            "recorded_at_first_steps": _quantiles([n for c in items for n in c.recorded_at_first_steps]),
            "left_out": {"windows": len(items) - len(kept),
                         "by_anchor": sum(0 in c.left_out_by for c in items),
                         "by_a_later_aircraft": sum(any(k > 0 for k in c.left_out_by) for c in items)},
            "losses": {pair: {"windows": sum(c.loss_steps[pair] > 0 for c in kept),
                              "share_of_windows": (sum(c.loss_steps[pair] > 0 for c in kept) / len(kept)
                                                   if kept else None),
                              "share_of_steps": sum(c.loss_steps[pair] for c in kept) / steps if steps else None}
                       for pair in PAIRS},
            "steps_judged": steps,
        }
    return out
