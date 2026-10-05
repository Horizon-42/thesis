"""The inputs of a row in a closed loop (prior design §2, §7 item 2; D96 item 4): the one function a loop of the speaker
reads its next Δ row with — free generation, and the post-training's loops of several aircraft.

From each aircraft's states on the data's 2 s rows (`instructions.artefact.STATE_COLUMNS`: the state at the row and the
2 s row before it), its airport's landings before the row's time (without its own, D63) and the speaker's words in force
(`inputs.Heard`), `LoopRows` gives the inputs of the row (`batch.RowTensors`, through the prior's one input function,
`inputs.state_inputs`) and the position the procedure masks read (`speaker.Position`, the height above E). A sentence of
the artefact gives the same rows through `inputs.sentence_rows` (the tests compare them bit for bit).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import STATE_COLUMNS
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.prior.batch import RowTensors, row_tensors
from ts_transformer.prior.inputs import Heard, state_inputs
from ts_transformer.prior.landings import LandingIndex
from ts_transformer.prior.speaker import Position

#: The columns of a state row that the inputs read: the position and the MSL height.
POSITION = [STATE_COLUMNS.index(name) for name in ("e_m", "n_m", "height_m")]


class LoopRows:
    """The inputs of each Δ row of a batch of aircraft in a loop (module docstring). For each aircraft: its airport's
    geometry, its key in the tracks rosters (`inputs.own_flight_key`: its own landing left out), the UTC time of its 2 s
    row 0 (``entry_utc_s``), the 2 s row of the signals its Δ row 0 is (``first_rows``) and its first predicted step's Δ
    row (``start``)."""

    def __init__(self, geometries: Sequence[AirportGeometry], landings: Mapping[str, LandingIndex], keys: Sequence[str],
                 entry_utc_s: np.ndarray, first_rows: np.ndarray, start: np.ndarray | int, *, variant: str,
                 interval_s: float, step_s: float, device: torch.device) -> None:
        self.geometries, self.keys = list(geometries), list(keys)
        self.landings = {g.code: landings[g.code] for g in self.geometries}
        count = len(self.geometries)
        self.entry_utc_s = np.asarray(entry_utc_s, dtype=np.float64)
        self.first_rows = np.asarray(first_rows, dtype=np.int64)
        self.start = np.broadcast_to(np.asarray(start, dtype=np.int64), (count,)).copy()
        if not (len(self.keys) == len(self.entry_utc_s) == len(self.first_rows) == count):
            raise ValueError("one key, entry time and first row for each aircraft")
        self.variant, self.interval_s, self.step_s, self.device = variant, interval_s, step_s, device
        self.every = interval_rows(interval_s, step_s)
        self.elevations = np.array([g.elevation_m for g in self.geometries])
        self._runways = [[self.landings[g.code].runways.index(c.ident) for c in g.candidates] for g in self.geometries]

    def __call__(self, t: int, at: np.ndarray, before: np.ndarray, known: np.ndarray | bool,
                 heard: Sequence[Heard]) -> tuple[RowTensors, Position]:
        """The inputs of Δ row ``t`` of every aircraft at its states ``at`` [B, 6] (`STATE_COLUMNS`), ``before`` [B, 6]
        its 2 s row before, ``known`` whether that row is one of its own (row 0 has no motion, D60), ``heard`` its words in
        force (the speaker's); and the positions the procedure masks read."""
        count = len(self.geometries)
        known = np.broadcast_to(np.asarray(known, dtype=bool), (count,))
        utc = self.entry_utc_s + (self.first_rows + t * self.every) * self.step_s
        own, candidates = [], []
        for b, geometry in enumerate(self.geometries):
            counts = self.landings[geometry.code].counts_before(utc[b: b + 1], without=self.keys[b])
            o, c = state_inputs(at[b: b + 1, POSITION], before[b: b + 1, POSITION], known[b: b + 1],
                                counts[:, self._runways[b]], geometry, self.variant, self.step_s)
            own.append(o[0])
            candidates.append(c[0])
        words = [h.inputs(t * self.interval_s) for h in heard]
        tensors = row_tensors(np.full(count, t * self.interval_s), np.stack(own), candidates,
                              np.array([w[0] for w in words]), np.array([w[1] for w in words]),
                              np.stack([w[2] for w in words]), np.stack([w[3] for w in words]),
                              np.stack([w[4] for w in words]), t == self.start, t >= self.start, self.device)
        return tensors, Position(at[:, POSITION[0]], at[:, POSITION[1]], at[:, POSITION[2]] - self.elevations)

    def select(self, indices: Sequence[int]) -> LoopRows:
        """The rows of the aircraft ``indices`` (repeats permitted), in that order: a copy's (`Speaker.copy`)."""
        indices = list(indices)
        return LoopRows([self.geometries[i] for i in indices], self.landings, [self.keys[i] for i in indices],
                        self.entry_utc_s[indices], self.first_rows[indices], self.start[indices], variant=self.variant,
                        interval_s=self.interval_s, step_s=self.step_s, device=self.device)
