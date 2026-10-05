"""The faults of an observed track (vocabulary D111, §6 item 3): the points of a flight's stored signals that no aircraft
flies — a jump, a position that stops while the aircraft flies on, a reversal — read from the stored rows alone, by one
definition, when an artefact is read; the artefact is not changed. Stage B's selection leaves a marked flight out (prior
D111).

A 2 s step (the distance between two consecutive rows) is set against the median of the steps around it: up to
`AROUND_STEPS` on each side, fewer at the ends of the track, never itself.

- a **jump**: the step is more than `JUMP_RATIO` times that median (a step among held positions, whose median is 0, too);
- a **held position**: the step is less than that median / `JUMP_RATIO`;
- a **reversal**: the track turns more than `REVERSAL_DEG` between two consecutive moves (a repeated position is no move).

A fault's row is the row its step starts from — a held stretch is reported from its last good row — and a reversal's
the row where the track turns. The rule reads the horizontal positions only. Torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ts_transformer.instructions.artefact import load_signals
from ts_transformer.instructions.signals import FlightSignals

#: A step more than this many times the median of the steps around it is a jump, less than its inverse a held position.
JUMP_RATIO = 3.0
#: The steps on each side of a step that its median reads.
AROUND_STEPS = 5
#: A turn of the track between two consecutive moves above this is a reversal.
REVERSAL_DEG = 120.0
JUMP, HELD, REVERSAL = "jump", "held position", "reversal"


@dataclass(frozen=True)
class Fault:
    """One faulty point of a track: what (`JUMP`, `HELD`, `REVERSAL`) and its 2 s row."""

    kind: str
    row: int


def track_faults(signals: FlightSignals) -> tuple[Fault, ...]:
    """The faults of a flight's stored track (module docstring), in row order (a jump before a reversal at one row).
    Refused for a track of fewer than three rows (no step has another to be set against) or a position that is not
    finite (no fault could be read there)."""
    e, n = np.asarray(signals.e_m, dtype=np.float64), np.asarray(signals.n_m, dtype=np.float64)
    if len(e) < 3 or not (np.isfinite(e).all() and np.isfinite(n).all()):
        raise ValueError(f"{signals.dataset_id}: a track of {len(e)} rows, all positions finite, is needed to read "
                         f"its faults")
    step = np.hypot(np.diff(e), np.diff(n))
    around = np.delete(sliding_window_view(np.pad(step, AROUND_STEPS, constant_values=np.nan), 2 * AROUND_STEPS + 1),
                       AROUND_STEPS, axis=1)                            # never the step itself; fewer at the ends
    median = np.nanmedian(around, axis=1)
    faults = [Fault(JUMP, int(k)) for k in np.flatnonzero(step > JUMP_RATIO * median)]
    faults += [Fault(HELD, int(k)) for k in np.flatnonzero(step * JUMP_RATIO < median)]
    moves = np.flatnonzero(step > 0.0)
    track = np.degrees(np.arctan2(np.diff(e)[moves], np.diff(n)[moves]))
    turn = np.abs(np.remainder(np.diff(track) + 180.0, 360.0) - 180.0)
    faults += [Fault(REVERSAL, int(moves[k + 1])) for k in np.flatnonzero(turn > REVERSAL_DEG)]
    order = {JUMP: 0, HELD: 1, REVERSAL: 2}
    return tuple(sorted(faults, key=lambda fault: (fault.row, order[fault.kind])))


def faulty_flights(directory: Path, split: str) -> dict[int, tuple[Fault, ...]]:
    """The flights of an artefact's ``split`` with a faulty track, by their place in the split's signals (the key a
    closed-loop sentence has, vocabulary §6 item 3), each with its faults."""
    out = {}
    for index, signals in enumerate(load_signals(directory, split)):
        faults = track_faults(signals)
        if faults:
            out[index] = faults
    return out
