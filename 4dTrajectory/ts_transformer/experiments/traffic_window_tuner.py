"""The multi-aircraft post-training in windows (multi-aircraft design §6.6 step 7 item 7, the 7.6 plan): a window's words
scored as the window speaker read them — shared by the window M4 runner, not a runner.

**What each aircraft read** (`window_flights`): a commanded aircraft's rows rebuilt as `data.chain_record` rebuilds a
one-aircraft sentence — its positions and its words in force, every row it was in the scene (the rows past its judged
end too: the other aircraft read them) — with its landing context switched where the speaker switched it: a landing in
the loop re-reads the rows not encoded yet (`WindowSpeaker.set_context`), so each row reads the context it was encoded
with, a row a landing came after it read before (`WindowRecord.context_changes`).

**A window scored whole** (`window_layout`, `window_logits`): the window samples laid out on one batch's steps as the
loop placed them (`traffic_window.window_places`), every commanded aircraft at its rows and every replayed one at its
own (`window_speaker.window_inputs` — the speaker's one layout), the edge features by the loop's code
(`traffic_window.window_edges`); one encoding, the heads read on each commanded aircraft's rows: ``[N, 1, rows,
classes]`` per column, as `train.batch_logits` gives them for aircraft alone — so the words' distributions are the
ones they were sampled from, to rounding (tests).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from ts_transformer.experiments.traffic_window import (
    Window, WindowRecord, _with_landing, window_edges, window_landings, window_places,
)
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import RUNWAY, UNCHANGED
from ts_transformer.prior.data import Flight, chain_record
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import N_LOOK, Landings
from ts_transformer.prior.window_speaker import window_inputs


def window_flight(record: WindowRecord, window: Window, signals: FlightSignals, geometry: AirportGeometry,
                  landings: Mapping[str, Landings] | None, airport: int, capture_row: int, counted: int, step_s: float
                  ) -> Flight:
    """One commanded aircraft's rows as the speaker read them (module docstring), its own words the targets of its first
    ``counted`` steps (what it is trained on); ``landings``: each airport's landing context, None for a variant without
    it."""
    said = record.grid
    classes = np.where(said != UNCHANGED, said + 1, 0)
    asked = np.zeros(said.shape, dtype=bool)
    asked[:counted] = True
    context = None if landings is None else window_landings(window, record.key, landings, step_s)

    def built(context: Landings | None) -> Flight:
        return chain_record(signals, record.e, record.n, record.h, said, classes, asked, geometry, context, airport,
                            capture_row, step_s)

    flight = built(context)
    if not record.context_changes:
        return flight
    features, relative = flight.features.copy(), flight.relative.copy()
    for first, landing_s, runway in record.context_changes:
        context = _with_landing(context, landing_s, runway)
        again = built(context)
        features[first:], relative[first:] = again.features[first:], again.relative[first:]
    return dataclasses.replace(flight, features=features, relative=relative)


@dataclass(frozen=True)
class WindowLayout:
    """Window samples laid out on one batch's steps (`window_layout`): the model's inputs with each commanded aircraft's
    targets, the edge features, each scene's airport, and where each commanded aircraft sits (its scene, its place among
    the scene's aircraft, the step of its row 0, its rows)."""

    inputs: dict[str, torch.Tensor]
    edges: torch.Tensor
    airport: torch.Tensor
    scene: np.ndarray
    slot: np.ndarray
    start: np.ndarray
    rows: np.ndarray


def window_layout(model: Prior, windows: Sequence[Window], flights: Sequence[Flight], records: Sequence[WindowRecord],
                  step_s: float, device: torch.device) -> WindowLayout:
    """``windows`` (window samples, each its commanded aircraft's ``flights`` and ``records`` in its order, window after
    window) laid out as the loop laid them out (module docstring)."""
    places = window_places(windows, step_s)
    scene = np.array([w for w, window in enumerate(windows) for _ in window.commanded], dtype=np.int64)
    keys = [k for window in windows for k in window.commanded]
    if [r.key for r in records] != keys or len(flights) != len(keys):
        raise ValueError("a flight and a record per commanded aircraft, window after window, each in its order")
    slot = np.concatenate([np.arange(len(window.commanded)) for window in windows])
    rows = np.array([f.rows for f in flights], dtype=np.int64)
    width = max(rows)
    aircraft = max(len(window.commanded) + len(window.others) for window in windows)

    def padded(name: str) -> torch.Tensor:
        values = [getattr(f, name) for f in flights]
        out = np.zeros((len(values), width) + values[0].shape[1:], dtype=values[0].dtype)
        for i, v in enumerate(values):
            out[i, : len(v)] = v
        return torch.as_tensor(out, device=device)

    own = {name: padded(name) for name in ("features", "relative", "in_force", "since", "targets")}
    own["in_force"], own["targets"] = own["in_force"].long(), own["targets"].long()
    last = int((places.start + rows).max())
    inputs = window_inputs(own, rows, places.start, scene, slot, places.nodes,
                           [len(window.commanded) for window in windows], aircraft, 0, last)
    positions = np.zeros((3, len(flights), width))
    pointers = np.zeros((len(flights), width), dtype=np.int64)
    for i, (record, flight) in enumerate(zip(records, flights)):
        positions[:, i, : len(record.e)] = record.e, record.n, record.h
        pointers[i, : flight.rows] = flight.in_force[:, RUNWAY]
    edges = window_edges(windows, keys, scene, False, positions[0], positions[1], positions[2], pointers, rows,
                         places.start, 0, last, step_s, aircraft, len(model.traffic_features))
    airport = torch.tensor([model.config.airports.index(window.airport.flights.code) for window in windows],
                           device=device)
    return WindowLayout(inputs, torch.as_tensor(edges, device=device), airport, scene, slot, places.start, rows)


def window_logits(model: Prior, layout: WindowLayout, *, checkpoint: bool = False) -> list[torch.Tensor]:
    """Each commanded aircraft's logits of its own words in its window (teacher forcing), ``[N, 1, rows, classes]`` per
    column as `train.batch_logits` gives them for aircraft alone: one encoding of the windows (`Prior.encode`;
    ``checkpoint``: the layers recomputed in the backward), the heads read on each commanded aircraft's rows (padded past
    its last with its last step's place, never read)."""
    x = layout.inputs
    h, tokens, valid = model.encode(x["features"], x["relative"], x["static"], x["in_force"], x["since"],
                                    layout.airport, x["present"], x["rows"], layout.edges, checkpoint)
    device = h.device
    width = int(layout.rows.max())
    steps = torch.as_tensor(np.minimum(layout.start[:, None] + np.arange(width)[None, :],
                                       (layout.start + layout.rows - 1)[:, None]), device=device)
    scene = torch.as_tensor(layout.scene, device=device)[:, None]
    slot = torch.as_tensor(layout.slot, device=device)[:, None]
    own_h = h[scene, slot, steps][:, None]
    own_tokens = tokens[scene, slot, steps][:, None]
    targets = x["targets"][scene, slot, steps][:, None]
    first = (torch.arange(width, device=device) == N_LOOK)[None, None, :]
    return model.logits(own_h, own_tokens, valid[scene[:, 0]], targets, first)
