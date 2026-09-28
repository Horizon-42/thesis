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
                 procedure_masks: ProcedureMasks, temperature: float = 1.0) -> None:
        """``others``: per scene, its replayed aircraft (`Node.first_step` from the speaking aircraft's row 0);
        ``history``: the most steps read before the speaking aircraft's row 0; the rest is `Speaker`'s (module docstring
        for ``edges``, ``masks``, ``mask_columns``)."""
        if len(others) != len(flights):
            raise ValueError(f"{len(others)} scenes' other aircraft for {len(flights)} speaking aircraft")
        self.mask_columns = tuple(mask_columns)
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
        count, aircraft, steps, device = len(self.geometries), self.aircraft, last - first, self.features.device
        features = torch.zeros((count, aircraft, steps, self.features.shape[-1]), device=device)
        relative = torch.zeros((count, aircraft, steps) + tuple(self.relative.shape[3:]), device=device)
        in_force = torch.zeros((count, aircraft, steps, 6), dtype=torch.long, device=device)
        since = torch.zeros((count, aircraft, steps, 6), device=device)
        static = torch.zeros((count, aircraft, self.static.shape[-1]), device=device)
        present = torch.zeros((count, aircraft, steps), dtype=torch.bool, device=device)
        rows = torch.zeros((count, aircraft, steps), dtype=torch.long, device=device)
        # the speaking aircraft: row s − pre at step s
        low, high = max(first, self.pre), last
        if low < high:
            span, own = slice(low - first, high - first), slice(low - self.pre, high - self.pre)
            features[:, 0, span], relative[:, 0, span] = self.features[:, 0, own], self.relative[:, 0, own]
            in_force[:, 0, span], since[:, 0, span] = self.in_force[:, 0, own], self.since[:, 0, own]
            present[:, 0, span] = True
            rows[:, 0, span] = torch.arange(low - self.pre, high - self.pre, device=device)
        # the others, at their own rows
        for b, scene in enumerate(self.others):
            for k, node in enumerate(scene, start=1):
                start = self.pre + node.first_step
                low, high = max(first, start), min(last, start + node.rows)
                if low >= high:
                    continue
                span, own = slice(low - first, high - first), slice(low - start, high - start)
                features[b, k, span] = torch.as_tensor(node.features[own], device=device)
                relative[b, k, span, : node.relative.shape[1]] = torch.as_tensor(node.relative[own], device=device)
                in_force[b, k, span] = torch.as_tensor(node.in_force[own], device=device)
                since[b, k, span] = torch.as_tensor(node.since[own], device=device)
                static[b, k] = torch.as_tensor(node.static, device=device)
                present[b, k, span] = True
                rows[b, k, span] = torch.arange(low - start, high - start, device=device)
        edges = torch.as_tensor(self.edges_of(first, last), dtype=features.dtype, device=device)
        expected = (count, steps, aircraft, aircraft, len(self.model.traffic_features))
        if tuple(edges.shape) != expected:
            raise ValueError(f"edge features {tuple(edges.shape)}, the steps encoded need {expected}")
        h, tokens, valid, self.past = self.model.extend(features, relative, static, in_force, since, self.airport,
                                                        present, rows, edges, self.past)
        self.steps_encoded = last
        return h, tokens, valid

    def _allowed(self, column: int, chosen: np.ndarray, opening: bool, classes: int,
                 runway_locked: np.ndarray) -> np.ndarray:
        out = super()._allowed(column, chosen, opening, classes, runway_locked)
        if column in self.mask_columns:
            out &= self.masks_of(column, chosen)
        return out
