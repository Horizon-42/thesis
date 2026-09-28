"""The prior speaking in a scene (multi-aircraft design §2.2 "一架由模型指挥", §2.6, §6.6 step 2): each scene of the batch
has one aircraft the prior speaks for — the `generate.Speaker`'s flight, the scene's aircraft 0 — and the others of the
scene replayed along their records: their rows and the words said to them as data (`scene_data.Node`: a flight with a
sentence with its labelled words in force, a background one with "none"), each at its own rows, on the scene's steps as
the scene samples place them (design §2.3). A traffic prior (`model.with_traffic`) reads the others through its traffic
attention; everything the speaking aircraft's rows, words and masks are is the `Speaker`'s.

**Steps.** The speaking aircraft of every scene are at the same row, as a `Speaker`'s flights are. The batch's steps start
``pre`` steps before their row 0 — the most any replayed aircraft of any scene is in the air before it (`Node.first_step`
counts from the speaking aircraft's row 0, negative before it), at most ``history`` — so a replayed aircraft arrives with
its history, as a flight carried into a sample does: batch step ``s`` is the speaking aircraft's row ``s − pre``. An
aircraft in the air longer before it is read from ``history`` steps before, at its own rows there (the caller counts
them). A scene holds nothing before its own aircraft enter.

**What the prior package does not reach is asked of the caller** (the separation rules are `inference`'s, design §9
item 18):

- ``edges(first, last)`` — every scene's edge features on batch steps ``first … last − 1`` (``[B, steps, A, A, E]``, in
  the order of the scene's aircraft, the speaking one first; `inference.scene_edges`). It is asked as the steps are
  encoded, so never of a step not reached yet: the speaking aircraft's row ``r`` (at step ``pre + r``) is the speaker's
  ``e``, ``n``, ``h`` and ``in_force[:, 0, r]``;
- ``masks(column, chosen)`` — for each of ``mask_columns``, the classes each scene's speaking aircraft may say given this
  step's classes so far (``chosen`` [B, 6]; the separation masks, design §3.4 layer 2): what the vocabulary's rules, the
  procedure's masks and these allow together is said, and the probability on what they remove recorded as the others'.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import Words
from ts_transformer.prior.data import STATIC_FEATURES, own_context
from ts_transformer.prior.generate import Speaker
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import Landings
from ts_transformer.prior.scene_data import Node


#: The most steps encoded in one `Prior.extend`: the pre-roll is encoded in blocks — the edge networks' tensors grow with
#: steps × aircraft², and a scene of 9 aircraft with a 600-step pre-roll filled the GPU in one (the step 3–4 review).
BLOCK_STEPS = 64


class SceneSpeaker(Speaker):
    """A `Speaker` of one aircraft per scene, with the scene's other aircraft replayed (module docstring)."""

    def __init__(self, model: Prior, flights: Sequence[FlightSignals], geometries: Sequence[AirportGeometry],
                 landings: Mapping[str, Landings] | None, words: Words, *, others: Sequence[Sequence[Node]],
                 edges: Callable[[int, int], np.ndarray], masks: Callable[[int, np.ndarray], np.ndarray],
                 mask_columns: Sequence[int], history: int, max_rows: int, generator: torch.Generator,
                 procedure_masks: ProcedureMasks, temperature: float = 1.0,
                 scene_landings: Sequence[Landings] | None = None) -> None:
        """``others``: per scene, its replayed aircraft (`Node.first_step` from the speaking aircraft's row 0);
        ``history``: the most steps read before the speaking aircraft's row 0; ``scene_landings``: each scene's landings
        (an augmented scene moves or adds some; None: the airport's, ``landings``) for a variant with a landing context;
        the rest is `Speaker`'s (module docstring for ``edges``, ``masks``, ``mask_columns``)."""
        if len(others) != len(flights):
            raise ValueError(f"{len(others)} scenes' other aircraft for {len(flights)} speaking aircraft")
        self.mask_columns = tuple(mask_columns)
        self.scene_landings = scene_landings
        super().__init__(model, flights, geometries, landings, words, max_rows=max_rows, generator=generator,
                         procedure_masks=procedure_masks, temperature=temperature)
        self.others = [tuple(scene) for scene in others]
        self.edges_of, self.masks_of = edges, masks
        #: the steps before the speaking aircraft's row 0 (module docstring)
        self.pre = min(history, max([0] + [-node.first_step for scene in self.others for node in scene]))
        self.aircraft = 1 + max(len(scene) for scene in self.others)
        self.steps_encoded = 0
        self.past = model.no_past(len(flights) * self.aircraft, self.pre + max_rows)

    @staticmethod
    def _check(model: Prior) -> None:
        if not model.traffic_features:
            raise ValueError("a scene is spoken to through a traffic prior's traffic attention (`model.with_traffic`)")

    def _more_masked_columns(self) -> tuple[int, ...]:
        return self.mask_columns

    def _contexts(self, flights: Sequence[FlightSignals], landings: Mapping[str, Landings] | None
                  ) -> list[Landings | None]:
        if landings is None or self.scene_landings is None:
            return super()._contexts(flights, landings)
        return [own_context(f, scene) for f, scene in zip(flights, self.scene_landings)]

    def _newest(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every scene's steps not encoded yet, encoded `BLOCK_STEPS` at a time: the speaking aircraft's newest row's
        ``(h, tokens, valid)``."""
        last = self.pre + self.rows
        while self.steps_encoded < last:
            h, tokens, valid = self._encode(self.steps_encoded, min(last, self.steps_encoded + BLOCK_STEPS))
        self.encoded = self.rows
        return h[:, :1, -1:], tokens[:, :1, -1:], valid

    def _encode(self, first: int, last: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch steps ``first … last − 1`` encoded (the speaking aircraft's rows, the others' at theirs, the caller's
        edge features of those steps): ``(h, tokens, valid)`` of every aircraft on them."""
        own = {"features": self.features[:, 0], "relative": self.relative[:, 0], "in_force": self.in_force[:, 0],
               "since": self.since[:, 0]}
        inputs = scene_inputs(own, [self.rows] * len(self.geometries), self.others, self.pre, self.aircraft, first, last)
        edges = torch.as_tensor(self.edges_of(first, last), dtype=inputs["features"].dtype, device=self.features.device)
        expected = (len(self.geometries), last - first, self.aircraft, self.aircraft, len(self.model.traffic_features))
        if tuple(edges.shape) != expected:
            raise ValueError(f"edge features {tuple(edges.shape)}, the steps encoded need {expected}")
        h, tokens, valid, self.past = self.model.extend(inputs["features"], inputs["relative"], inputs["static"],
                                                        inputs["in_force"], inputs["since"], self.airport,
                                                        inputs["present"], inputs["rows"], edges, self.past)
        self.steps_encoded = last
        return h, tokens, valid

    def _allowed(self, column: int, chosen: np.ndarray, opening: bool, classes: int,
                 runway_locked: np.ndarray) -> np.ndarray:
        out = super()._allowed(column, chosen, opening, classes, runway_locked)
        if column in self.mask_columns:
            out &= self.masks_of(column, chosen)
        return out


def scene_inputs(own: Mapping[str, torch.Tensor], own_rows: Sequence[int], others: Sequence[Sequence[Node]], pre: int,
                 aircraft: int, first: int, last: int) -> dict[str, torch.Tensor]:
    """Batch steps ``first … last − 1`` of scenes laid out as the speaker places them (module docstring) — the one layout,
    for the speaker and for a trainer scoring its sentences in their scenes: each scene's speaking aircraft as aircraft 0
    from step ``pre`` over its first ``own_rows[b]`` rows (``own``: its rows' ``features`` ``[B, R, F]``, ``relative``
    ``[B, R, K, W]``, ``in_force`` and ``since`` ``[B, R, 6]`` and, to score them, ``targets``), each other aircraft
    (`Node`) from step ``pre`` + its first step, at its own rows (one in the air before step 0 enters there, its rows
    counted on). Returns ``features``, ``relative``, ``static``, ``in_force``, ``since``, ``present``, ``rows`` (and
    ``targets`` when ``own`` has them; the others' are 0) ``[B, A, steps, …]`` on ``own``'s device."""
    base = own["features"]
    count, steps, device = len(others), last - first, base.device
    out = {"features": torch.zeros((count, aircraft, steps, base.shape[-1]), device=device),
           "relative": torch.zeros((count, aircraft, steps) + tuple(own["relative"].shape[2:]), device=device),
           "in_force": torch.zeros((count, aircraft, steps, 6), dtype=torch.long, device=device),
           "since": torch.zeros((count, aircraft, steps, 6), device=device),
           "static": torch.zeros((count, aircraft, len(STATIC_FEATURES)), device=device),
           "present": torch.zeros((count, aircraft, steps), dtype=torch.bool, device=device),
           "rows": torch.zeros((count, aircraft, steps), dtype=torch.long, device=device)}
    names = ["features", "relative", "in_force", "since"]
    if "targets" in own:
        out["targets"] = torch.zeros((count, aircraft, steps, 6), dtype=torch.long, device=device)
        names.append("targets")
    # the speaking aircraft: row s − pre at step s
    low, high = max(first, pre), min(last, pre + base.shape[1])
    if low < high:
        span, rows = slice(low - first, high - first), slice(low - pre, high - pre)
        for name in names:
            out[name][:, 0, span] = own[name][:, rows]
        numbers = torch.arange(low - pre, high - pre, device=device)
        out["rows"][:, 0, span] = numbers
        out["present"][:, 0, span] = numbers[None, :] < torch.as_tensor(list(own_rows), device=device)[:, None]
    # the others, at their own rows
    for b, scene in enumerate(others):
        for k, node in enumerate(scene, start=1):
            start = pre + node.first_step
            low, high = max(first, start), min(last, start + node.rows)
            if low >= high:
                continue
            span, rows = slice(low - first, high - first), slice(low - start, high - start)
            out["features"][b, k, span] = torch.as_tensor(node.features[rows], device=device)
            out["relative"][b, k, span, : node.relative.shape[1]] = torch.as_tensor(node.relative[rows], device=device)
            out["in_force"][b, k, span] = torch.as_tensor(node.in_force[rows], device=device)
            out["since"][b, k, span] = torch.as_tensor(node.since[rows], device=device)
            out["static"][b, k] = torch.as_tensor(node.static, device=device)
            out["present"][b, k, span] = True
            out["rows"][b, k, span] = torch.arange(low - start, high - start, device=device)
    return out
