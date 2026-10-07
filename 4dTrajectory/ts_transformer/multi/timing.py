"""The readouts of the time the aircraft take (multi-aircraft control §2.5, §5 item 4; D151, O18): a readout, never a term
of the reward, for the user to see whether the model spreads the aircraft out.

For a window's commanded aircraft (`Landed`, `landed`): the time each one whose outcome is `landed` crossed its
threshold in the loop (its crossing's cycle row from its first predicted step, as the window loop counts a loop landing,
D147; a silent aircraft that crosses is not a landing, D147 item 3) and on its record (its recorded landing, moved in a
compressed window), and its runways in each.

- **The delay** of a landing (`delays`): its time in the loop less its time on the record, s.
- **The spacing at the threshold** (`spacings`): the time between successive landings on one runway that hold a
  commanded aircraft, s, every landing counted, the recorded aircraft's on their records (they land between the
  commanded ones) — in the loop (the landed commanded aircraft on their landed runways) and on the records of the same
  window (every commanded aircraft on its recorded runway), side by side.
- **The landing order** (`order`): of the pairs of aircraft landed in the loop, how many land in the order of their
  records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ts_transformer.post.scene import Recorded


@dataclass(frozen=True)
class Landed:
    """One commanded aircraft that landed in the loop: its key, its runway and time in the loop, and on its record."""

    key: str
    runway: int
    loop_s: float
    record_runway: int
    record_s: float


def landed(records: Sequence[Recorded], crossings: Sequence[Mapping[str, Any] | None], cycle_s: float) -> list[Landed]:
    """The window's commanded aircraft that landed (``crossings``: each one's crossing, None where it did not land), in
    the window's order (module docstring)."""
    return [Landed(record.key, int(crossing["runway_index"]), record.first_step_s + crossing["at_row"] * cycle_s,
                   record.runway_index, record.landing_s)
            for record, crossing in zip(records, crossings, strict=True) if crossing is not None]


def delays(items: Sequence[Landed]) -> list[float]:
    """Each landing's time in the loop less its time on its record, s."""
    return [item.loop_s - item.record_s for item in items]


def spacings(times: Sequence[tuple[int, float, bool]]) -> list[float]:
    """The time between successive landings on one runway that hold a commanded one (``times``: each landing's runway,
    time and whether it is a commanded aircraft's), s."""
    out: list[float] = []
    for runway in sorted({r for r, _, _ in times}):
        on = sorted((t, commanded) for r, t, commanded in times if r == runway)
        out += [b - a for (a, x), (b, y) in zip(on, on[1:]) if x or y]
    return out


def recorded_landings(window: Any, last_s: float) -> list[tuple[int, float, bool]]:
    """The landings of a window's recorded aircraft on their records — the flights of its scene but its commanded ones
    in the air at a step of the window (§3.1), from its row 0 to ``last_s`` (its last landing, in the loop or on a
    record): each one's runway and time, not commanded. (The scene holds every day of the split: a flight of another
    day is never one.)"""
    commanded = {record.key for record in window.commanded_all}
    return [(flight.runway_index, flight.landing_s, False) for flight in window.scene.flights
            if flight.key not in commanded and flight.end_s >= window.row0_s and flight.start_s <= last_s]


def loop_spacings(items: Sequence[Landed], recorded: Sequence[tuple[int, float, bool]]) -> list[float]:
    """`spacings` of the landings in the loop: the commanded aircraft landed there, among the recorded ones."""
    return spacings([(item.runway, item.loop_s, True) for item in items] + list(recorded))


def record_spacings(records: Sequence[Recorded], recorded: Sequence[tuple[int, float, bool]]) -> list[float]:
    """`spacings` on the records: the same window's commanded aircraft on their records, among the recorded ones."""
    return spacings([(record.runway_index, record.landing_s, True) for record in records] + list(recorded))


def order(items: Sequence[Landed]) -> tuple[int, int]:
    """Of the pairs of aircraft landed in the loop, those that land in the order of their records, and the pairs."""
    same = pairs = 0
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            pairs += 1
            same += (items[a].loop_s - items[b].loop_s) * (items[a].record_s - items[b].record_s) > 0
    return same, pairs


def quantiles(values: Sequence[float], qs: Sequence[float]) -> dict[str, float | int]:
    """The count, the mean and the percentiles ``qs`` (in %) of ``values``; the count alone when there is none."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "mean": float(values.mean()),
            **{f"p{q:g}": float(np.percentile(values, q)) for q in qs}}
