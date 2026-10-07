"""The inputs of a row in a closed loop (prior design §2, §7 item 2; D96 item 4): the one function a loop of the speaker
reads its next Δ row with — free generation, and the post-training's loops of several aircraft.

From each aircraft's states on the data's 2 s rows (`instructions.artefact.STATE_COLUMNS`: the state at the row and the
2 s row before it), its own landings before the row's time (D105: the caller's, one index for each aircraft — free
generation gives each flight its airport's roster, a window its scene's; without its own landing, D63) and the speaker's words in force
(`inputs.Heard`), `LoopRows` gives the inputs of the row (`batch.RowTensors`, through the prior's one input function,
`inputs.state_inputs`) and the position the procedure masks read (`speaker.Position`, the height above E). A sentence of
the artefact gives the same rows through `inputs.sentence_rows` (the tests compare them bit for bit).

**Join ticks** (multi-aircraft control D150, §6.3 items 2 and 5; prior §7 item 2). The aircraft of a loop may join it at
their own ticks (a tick a Δ row of the loop's clock): at tick t, aircraft b's inputs are those of its own row t − j_b, at
its own UTC time (its entry time and its first row; in a compressed window its moved entry time) and its own seconds; an
aircraft that has not joined (t < j_b) has no row present. A landing added to chosen aircraft's landings while the loop
runs (`add_landing`) replaces each one's index by one that holds it (`LandingIndex.with_landing`, with the index's
checks); the counts of a row read the landings before its time, so it changes only later rows. With every join tick 0 and
no landing added, the rows are the rows without join ticks, bit for bit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import STATE_COLUMNS
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.prior.batch import RowTensors, row_tensors
from ts_transformer.prior.inputs import Heard, state_inputs
from ts_transformer.prior.landings import Landing, LandingIndex
from ts_transformer.prior.speaker import Position

#: The columns of a state row that the inputs read: the position and the MSL height.
POSITION = [STATE_COLUMNS.index(name) for name in ("e_m", "n_m", "height_m")]


class LoopRows:
    """The inputs of each Δ row of a batch of aircraft in a loop (module docstring). For each aircraft: its airport's
    geometry, its landings (D105: an index of landings on its airport's candidates), its key in the tracks rosters
    (`inputs.own_flight_key`: its own landing, which its landings hold, left out), the UTC time of its 2 s row 0
    (``entry_utc_s``) and the 2 s row of the signals its Δ row 0 is (``first_rows``); ``start`` the first predicted
    step's Δ row of each aircraft's own rows, one for the batch; ``join_ticks`` each aircraft's join tick (module
    docstring), every aircraft's 0 when not given."""

    def __init__(self, geometries: Sequence[AirportGeometry], landings: Sequence[LandingIndex], keys: Sequence[str],
                 entry_utc_s: np.ndarray, first_rows: np.ndarray, start: int, *, variant: str,
                 interval_s: float, step_s: float, device: torch.device,
                 join_ticks: np.ndarray | None = None) -> None:
        self.geometries, self.landings, self.keys = list(geometries), list(landings), list(keys)
        count = len(self.geometries)
        self.entry_utc_s = np.asarray(entry_utc_s, dtype=np.float64)
        self.first_rows = np.asarray(first_rows, dtype=np.int64)
        if not isinstance(start, (int, np.integer)) or isinstance(start, bool):
            raise ValueError(f"the first predicted step is one Δ row for the batch, got {start!r}")
        self.start = int(start)
        if not (len(self.landings) == len(self.keys) == len(self.entry_utc_s) == len(self.first_rows) == count):
            raise ValueError("one index of landings, key, entry time and first row for each aircraft")
        ticks = np.zeros(count, dtype=np.int64) if join_ticks is None else np.asarray(join_ticks)
        if ticks.shape != (count,) or ticks.dtype.kind not in "iu" or (ticks < 0).any():
            raise ValueError(f"a join tick is a whole number >= 0 for each of the {count} aircraft, got {join_ticks!r}")
        self.join_ticks = ticks.astype(np.int64)
        self.variant, self.interval_s, self.step_s, self.device = variant, interval_s, step_s, device
        self.every = interval_rows(interval_s, step_s)
        self.elevations = np.array([g.elevation_m for g in self.geometries])
        # each aircraft's candidates as columns of its landings (refused when its landings are not on its candidates)
        self._runways = [[index.runways.index(c.ident) for c in g.candidates]
                         for g, index in zip(self.geometries, self.landings)]

    def __call__(self, t: int, at: np.ndarray, before: np.ndarray, known: np.ndarray | bool,
                 heard: Sequence[Heard]) -> tuple[RowTensors, Position]:
        """The inputs of tick ``t`` of every aircraft (its own Δ row t − j, module docstring) at its states ``at`` [B, 6]
        (`STATE_COLUMNS`), ``before`` [B, 6] its 2 s row before, ``known`` whether that row is one of its own (row 0 has
        no motion, D60), ``heard`` its words in force (the speaker's); and the positions the procedure masks read. An
        aircraft that has not joined gets a row not present, from the finite states the caller gives it."""
        count = len(self.geometries)
        known = np.broadcast_to(np.asarray(known, dtype=bool), (count,))
        own_row = t - self.join_ticks                       # each aircraft's own Δ row
        utc = self.entry_utc_s + (self.first_rows + own_row * self.every) * self.step_s
        own, candidates = [], []
        for b, geometry in enumerate(self.geometries):
            counts = self.landings[b].counts_before(utc[b: b + 1], without=self.keys[b])
            o, c = state_inputs(at[b: b + 1, POSITION], before[b: b + 1, POSITION], known[b: b + 1],
                                counts[:, self._runways[b]], geometry, self.variant, self.step_s)
            own.append(o[0])
            candidates.append(c[0])
        words = [h.inputs(float(r * self.interval_s)) for h, r in zip(heard, own_row)]
        tensors = row_tensors(own_row * self.interval_s, np.stack(own), candidates,
                              np.array([w[0] for w in words]), np.array([w[1] for w in words]),
                              np.stack([w[2] for w in words]), np.stack([w[3] for w in words]),
                              np.stack([w[4] for w in words]), own_row == self.start, own_row >= self.start,
                              self.device, present=own_row >= 0)
        return tensors, Position(at[:, POSITION[0]], at[:, POSITION[1]], at[:, POSITION[2]] - self.elevations)

    def add_landing(self, aircraft: Sequence[int], landing: Landing) -> None:
        """``landing`` added to the landings of the aircraft ``aircraft`` (module docstring): each one's index replaced
        by one that holds it."""
        for b in aircraft:
            self.landings[b] = self.landings[b].with_landing(landing)

    def select(self, indices: Sequence[int]) -> LoopRows:
        """The rows of the aircraft ``indices`` (repeats permitted), in that order, with their landings as they are now
        and their join ticks: a copy's (`Speaker.copy`)."""
        indices = list(indices)
        return LoopRows([self.geometries[i] for i in indices], [self.landings[i] for i in indices],
                        [self.keys[i] for i in indices],
                        self.entry_utc_s[indices], self.first_rows[indices], self.start, variant=self.variant,
                        interval_s=self.interval_s, step_s=self.step_s, device=self.device,
                        join_ticks=self.join_ticks[indices])
