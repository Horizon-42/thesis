"""The inputs of a row in a closed loop (prior design §2, §7 item 2; D96 item 4): the one function a loop of the speaker
reads its next Δ row with — free generation, and the post-training's loops of several aircraft.

From each aircraft's states on the data's 2 s rows (`instructions.artefact.STATE_COLUMNS`: the state at the row and the
2 s row before it) and the speaker's words in force (`inputs.Heard`), `LoopRows` gives the inputs of the row
(`batch.RowTensors`, through the prior's one input function, `inputs.state_inputs`; no landings, D206) and the position
the procedure masks read (`speaker.Position`, the height above E). A sentence of the artefact gives the same rows
through `inputs.sentence_rows` (the tests compare them bit for bit).

**Join ticks** (multi-aircraft control D150, §6.3 items 2 and 5; prior §7 item 2). The aircraft of a loop may join it at
their own ticks (a tick a Δ row of the loop's clock): at tick t, aircraft b's inputs are those of its own row t − j_b, at
its own seconds; an aircraft that has not joined (t < j_b) has no row present. With every join tick 0, the rows are the
rows without join ticks, bit for bit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import STATE_COLUMNS
from ts_transformer.prior.batch import RowTensors, row_tensors
from ts_transformer.prior.inputs import Heard, state_inputs
from ts_transformer.prior.speaker import Position

#: The columns of a state row that the inputs read: the position and the MSL height.
POSITION = [STATE_COLUMNS.index(name) for name in ("e_m", "n_m", "height_m")]


class LoopRows:
    """The inputs of each Δ row of a batch of aircraft in a loop (module docstring). For each aircraft: its airport's
    geometry; ``start`` the first predicted step's Δ row of each aircraft's own rows, one for the batch; ``join_ticks``
    each aircraft's join tick (module docstring), every aircraft's 0 when not given. A row's state inputs are computed
    for the aircraft of each airport at once (`state_inputs` on several rows, one each: multi-aircraft control D186
    (3)); each row's numbers are those of the row alone (elementwise numpy, the same bits for one row as for many: the
    tests compare them)."""

    def __init__(self, geometries: Sequence[AirportGeometry], start: int, *, variant: str, interval_s: float,
                 step_s: float, device: torch.device, join_ticks: np.ndarray | None = None) -> None:
        self.geometries = list(geometries)
        count = len(self.geometries)
        if not isinstance(start, (int, np.integer)) or isinstance(start, bool):
            raise ValueError(f"the first predicted step is one Δ row for the batch, got {start!r}")
        self.start = int(start)
        ticks = np.zeros(count, dtype=np.int64) if join_ticks is None else np.asarray(join_ticks)
        if ticks.shape != (count,) or ticks.dtype.kind not in "iu" or (ticks < 0).any():
            raise ValueError(f"a join tick is a whole number >= 0 for each of the {count} aircraft, got {join_ticks!r}")
        self.join_ticks = ticks.astype(np.int64)
        self.variant, self.interval_s, self.step_s, self.device = variant, interval_s, step_s, device
        self.elevations = np.array([g.elevation_m for g in self.geometries])
        # the aircraft of each airport, in their order: their state inputs computed at once
        airports: dict[int, list[int]] = {}
        for b, geometry in enumerate(self.geometries):
            airports.setdefault(id(geometry), []).append(b)
        self._airports = [np.array(group) for group in airports.values()]

    def __call__(self, t: int, at: np.ndarray, before: np.ndarray, known: np.ndarray | bool,
                 heard: Sequence[Heard]) -> tuple[RowTensors, Position]:
        """The inputs of tick ``t`` of every aircraft (its own Δ row t − j, module docstring) at its states ``at`` [B, 6]
        (`STATE_COLUMNS`), ``before`` [B, 6] its 2 s row before, ``known`` whether that row is one of its own (row 0 has
        no motion, D60), ``heard`` its words in force (the speaker's); and the positions the procedure masks read. An
        aircraft that has not joined gets a row not present, from the finite states the caller gives it."""
        count = len(self.geometries)
        known = np.broadcast_to(np.asarray(known, dtype=bool), (count,))
        own_row = t - self.join_ticks                       # each aircraft's own Δ row
        own, candidates = [None] * count, [None] * count
        for group in self._airports:
            o, c = state_inputs(at[group][:, POSITION], before[group][:, POSITION], known[group],
                                self.geometries[group[0]], self.variant, self.step_s)
            for k, b in enumerate(group):
                own[b], candidates[b] = o[k], c[k]
        words = [h.inputs(float(r * self.interval_s)) for h, r in zip(heard, own_row)]
        tensors = row_tensors(own_row * self.interval_s, np.stack(own), candidates,
                              np.array([w[0] for w in words]), np.array([w[1] for w in words]),
                              np.stack([w[2] for w in words]), np.stack([w[3] for w in words]),
                              np.stack([w[4] for w in words]), own_row == self.start, own_row >= self.start,
                              self.device, present=own_row >= 0)
        return tensors, Position(at[:, POSITION[0]], at[:, POSITION[1]], at[:, POSITION[2]] - self.elevations)

    def select(self, indices: Sequence[int], device: torch.device | None = None) -> LoopRows:
        """The rows of the aircraft ``indices`` (repeats permitted), in that order, with their join ticks: a copy's
        (`Speaker.copy`); its rows' tensors on ``device`` (this one's when not given)."""
        indices = list(indices)
        return LoopRows([self.geometries[i] for i in indices], self.start, variant=self.variant,
                        interval_s=self.interval_s, step_s=self.step_s, device=self.device if device is None else device,
                        join_ticks=self.join_ticks[indices])

    @staticmethod
    def together(parts: Sequence[tuple[LoopRows, Sequence[int], int]], device: torch.device) -> LoopRows:
        """The rows of aircraft of several loops as one loop's (a speaking loop's flights together, multi-aircraft
        control D188): ``parts`` each rows, its aircraft (repeats permitted) and the ticks its clock moves later by, its
        join ticks with it; each aircraft's own rows the same rows. On ``device``."""
        index = [list(aircraft) for _, aircraft, _ in parts]
        first = parts[0][0]
        ticks = np.concatenate([rows.join_ticks[i] + shift for (rows, _, shift), i in zip(parts, index)])
        return LoopRows([rows.geometries[b] for (rows, _, _), i in zip(parts, index) for b in i], first.start,
                        variant=first.variant, interval_s=first.interval_s, step_s=first.step_s, device=device,
                        join_ticks=ticks)
