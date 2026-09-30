"""The prior speaking to every commanded aircraft of a window (multi-aircraft design §2.2 "窗口内由模型指挥", §2.6,
§6.6 step 7 item 2): each scene of the batch has one or more aircraft the prior speaks for, each from its own row 0 —
its first `N_LOOK` rows observed, then the caller's — and the scene's other aircraft replayed along their records, as in
`scene_speaker`. With one commanded aircraft a scene it says what `scene_speaker.SceneSpeaker` says, word for word, up to
each aircraft's end (tests).

**Steps and places.** A scene's first commanded aircraft is its reference: batch step ``pre`` is the references' row 0
(``pre`` the most any replayed aircraft of any scene is in the air before its reference, at most ``history``, as in
`scene_speaker`), and each other commanded aircraft enters its ``offset`` in steps after its reference. On the model's
aircraft axis a scene holds its reference first, its replayed aircraft next — laid out by `scene_speaker.scene_inputs`,
the one layout — then its other commanded aircraft. A commanded aircraft is there from its row 0 over the rows it has:
the caller appends one a step while it flies (`advance`); one no longer appended — its flight over — has left.

**One encoding a step, the words picked in rounds** (design §2.6, §9 item 24): the model encodes every aircraft of every
scene once a step; the caller ranks the aircraft that speak at the step (``rank``: its approach clock's order, front
first; −1 silent) and the words are picked round by round — a round's aircraft, one a scene, their six columns in order
as `generate.Speaker` picks them — so the caller's masks in a later round read the words an earlier one just said
(`value`: the classes in force, this step's so far). Every round samples EVERY commanded aircraft of the batch, the
ones not in it from a distribution that picks "unchanged": a round's draw has one shape, and each aircraft's draw reads
only its own row, whatever else the batch holds or has left.

**Per aircraft**: its rows, its words in force, the vocabulary's rules and the procedure's masks at its own row, and what
the masks removed at each of its steps (``forbidden``, ``allowed``: ``[N, steps]``, indexed by its step after its first
predicted one).

What the prior package does not reach is asked of the caller, as in `scene_speaker` (design §9 item 18):
``edges(first, last)`` — every scene's edge features on batch steps ``first … last − 1``, ``[S, steps, A, A, E]`` in the
places above; ``masks(column, chosen, speaking)`` — for each of ``mask_columns``, the classes each aircraft of the round
(``speaking`` [N] bool) may say given this step's classes so far (``chosen`` [N, 6]; the separation masks), every class
for the others.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import Words
from ts_transformer.prior.data import SINCE_SCALE, STEP_FEATURES, VARIANTS, own_context, rows_inputs
from ts_transformer.prior.generate import VOCABULARY_COLUMNS, vocabulary_allowed
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import N_LOOK, Landings, utc_s
from ts_transformer.prior.scene_data import Node
from ts_transformer.prior.scene_speaker import BLOCK_STEPS, scene_inputs


class WindowSpeaker:
    """The prior speaking to several aircraft a scene (module docstring): `speak` at the step, `advance` to the next."""

    def __init__(self, model: Prior, flights: Sequence[FlightSignals], geometries: Sequence[AirportGeometry],
                 landings: Mapping[str, Landings] | None, words: Words, *, scenes: Sequence[int],
                 offsets: Sequence[int], others: Sequence[Sequence[Node]], edges: Callable[[int, int], np.ndarray],
                 masks: Callable[[int, np.ndarray, np.ndarray], np.ndarray], mask_columns: Sequence[int], history: int,
                 max_rows: int, generator: torch.Generator, procedure_masks: ProcedureMasks, temperature: float = 1.0,
                 scene_landings: Sequence[Landings] | None = None) -> None:
        """``flights`` (with ``geometries``): the commanded aircraft; ``scenes``: each one's scene (0 … S − 1, a scene's
        first one its reference); ``offsets``: each one's row 0 in steps after its reference's (the reference's 0);
        ``others``: per scene, its replayed aircraft (`Node.first_step` from its reference's row 0); ``history``: the most
        steps read before the references' row 0; ``max_rows``: the most rows any aircraft will have; ``landings``: each
        airport's landing context, None for a variant without it; ``scene_landings``: each AIRCRAFT's scene's landings
        (an augmented scene moves or adds some; None: the airport's)."""
        if not model.traffic_features:
            raise ValueError("a scene is spoken to through a traffic prior's traffic attention (`model.with_traffic`)")
        if VARIANTS[model.config.variant].landing_context != (landings is not None):
            raise ValueError(f"variant {model.config.variant} and the landing context given disagree")
        if max_rows > model.config.max_rows:
            raise ValueError(f"{max_rows} rows, the model's positions end at {model.config.max_rows}")
        count = len(flights)
        scene, offset = np.asarray(scenes, dtype=np.int64), np.asarray(offsets, dtype=np.int64)
        if len(scene) != count or len(offset) != count or len(geometries) != count:
            raise ValueError("a scene, an offset and a geometry per commanded aircraft")
        members = [np.flatnonzero(scene == w) for w in range(len(others))]
        if len(set(scene.tolist())) != len(others) or any(offset[m[0]] != 0 for m in members) or (offset < 0).any():
            raise ValueError("every scene commands an aircraft, its first at offset 0, the rest at or after it")
        self.model, self.geometries, self.generator, self.temperature = model, list(geometries), generator, temperature
        self.words, self.edges_of, self.masks_of = words, edges, masks
        self.mask_columns = tuple(mask_columns)
        #: each set of the procedure's masks on these aircraft (`ProcedureMasks.speaking`)
        self.procedure = procedure_masks.speaking(self.geometries, words)
        masked = list(VOCABULARY_COLUMNS)
        masked += [c for rules in self.procedure for c in rules.columns if c not in masked]
        masked += [c for c in self.mask_columns if c not in masked]
        self.contexts = [None if landings is None else
                         own_context(f, landings[f.airport] if scene_landings is None else scene_landings[i])
                         for i, f in enumerate(flights)]
        self.entry_s = np.array([utc_s(f.entry_time_utc) for f in flights])
        self.step_s = words.spec.step_s
        self.time_s = np.arange(max_rows) * self.step_s
        for f in flights:
            if not np.allclose(f.time_s[: N_LOOK + 1], self.time_s[: N_LOOK + 1]):
                raise ValueError(f"{f.dataset_id}: its rows are not {self.step_s} s apart from 0")
        device = model.candidates.device
        slots, width = model.config.candidate_slots, len(VARIANTS[model.config.variant].relative_features)
        # each aircraft's rows: the observed ones, then the ones appended
        self.e, self.n, self.h = (np.zeros((count, max_rows)) for _ in range(3))
        for i, f in enumerate(flights):
            self.e[i, : N_LOOK + 1], self.n[i, : N_LOOK + 1] = f.e_m[: N_LOOK + 1], f.n_m[: N_LOOK + 1]
            self.h[i, : N_LOOK + 1] = f.altitude_m[: N_LOOK + 1]
        self.features = torch.zeros((count, max_rows, len(STEP_FEATURES)), device=device)
        self.relative = torch.zeros((count, max_rows, slots, width), device=device)
        self.in_force = torch.zeros((count, max_rows, 6), dtype=torch.long, device=device)
        self.since = torch.zeros((count, max_rows, 6), device=device)
        #: rows each aircraft has: an aircraft is there at its rows below it
        self.rows = np.full(count, N_LOOK + 1)
        self._inputs(np.arange(count), 0)
        for row in range(N_LOOK + 1):
            for rules in self.procedure:
                rules.track(self.e[:, row], self.n[:, row], self.h[:, row])
        # the words said: the class in force per column (0: none yet) and the row it was said at
        self.value = np.zeros((count, 6), dtype=np.int64)
        self.said_row = np.zeros((count, 6), dtype=np.int64)
        # places (module docstring)
        self.scene, self.offset = scene, offset
        self.others = [tuple(nodes) for nodes in others]
        self.references = np.array([m[0] for m in members])
        self.pre = min(history, max([0] + [-node.first_step for nodes in self.others for node in nodes]))
        self.start = self.pre + offset
        self.slot = np.zeros(count, dtype=np.int64)
        for w, m in enumerate(members):
            self.slot[m[1:]] = 1 + len(self.others[w]) + np.arange(len(m) - 1)
        self.aircraft = max(len(m) + len(nodes) for m, nodes in zip(members, self.others))
        self.airport = torch.tensor([model.config.airports.index(flights[i].airport) for i in self.references],
                                    device=device)
        self.past = model.no_past(len(others) * self.aircraft, self.pre + int(offset.max()) + max_rows)
        self.steps_encoded = 0
        #: the batch step spoken at: the references' first predicted step first
        self.step = self.pre + N_LOOK
        steps = max_rows - N_LOOK
        #: per masked column, each aircraft's step: the probability the model put on what the masks removed, and the
        #: classes they allowed, bit-packed (`generate.allowed_classes` unpacks them)
        self.forbidden = {c: np.zeros((count, steps)) for c in masked}
        self.allowed = {c: np.zeros((count, steps, (model.config.classes[c] + 7) // 8), dtype=np.uint8)
                        for c in masked}

    @property
    def row(self) -> np.ndarray:
        """``[N]``: each aircraft's row at the step (negative before its row 0)."""
        return self.step - self.start

    @property
    def there(self) -> np.ndarray:
        """``[N]`` bool: which aircraft are in the scene at the step."""
        row = self.row
        return (row >= 0) & (row < self.rows)

    def _inputs(self, index: np.ndarray, first: int) -> None:
        """The rows of ``index`` from ``first`` to each one's last (`data.rows_inputs`), the aircraft of one airport
        geometry with the same rows together."""
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in index:
            groups[(id(self.geometries[i]), int(self.rows[i]))].append(int(i))
        for (_, rows), members in groups.items():
            at = np.array(members)
            contexts = self.contexts[members[0]]
            f, r = rows_inputs(self.e[at, :rows], self.n[at, :rows], self.h[at, :rows], self.time_s[:rows],
                               self.entry_s[at], first, self.geometries[members[0]],
                               None if contexts is None else [self.contexts[i] for i in members])
            where = torch.as_tensor(at, device=self.features.device)
            self.features[where, first:rows] = torch.as_tensor(f, device=self.features.device)
            self.relative[where, first:rows, : r.shape[2]] = torch.as_tensor(r, device=self.relative.device)

    def advance(self, index: np.ndarray, e: np.ndarray, n: np.ndarray, height: np.ndarray) -> None:
        """On to the next step, with the row there of the aircraft at ``index`` (each past its first predicted step: its
        next row, where it is — ``[len(index)]`` each); one past its first predicted step and not given has left."""
        index = np.asarray(index, dtype=np.int64)
        step = self.step + 1
        row = step - self.start[index]
        if (row <= N_LOOK).any() or (row != self.rows[index]).any():
            raise ValueError("a row is appended to an aircraft past its first predicted step, its next row")
        self.e[index, row], self.n[index, row], self.h[index, row] = e, n, height
        self.rows[index] += 1
        self._appended(index, row)
        newest = np.clip(self.rows - 1, 0, None)
        ar = np.arange(len(self.rows))
        for rules in self.procedure:
            rules.track(self.e[ar, newest], self.n[ar, newest], self.h[ar, newest])
        self.step = step

    def _appended(self, index: np.ndarray, row: np.ndarray) -> None:
        """The appended rows' inputs (`_inputs` from each one's new row) and the words in force at the row before, as the
        prior said them (`data.sentence_steps`)."""
        by_row: dict[int, list[int]] = defaultdict(list)
        for i, r in zip(index, row):
            by_row[int(r)].append(int(i))
        for r, members in by_row.items():
            self._inputs(np.array(members), r)
        where, rows = torch.as_tensor(index, device=self.features.device), torch.as_tensor(row, device=self.features.device)
        self.in_force[where, rows] = torch.as_tensor(self.value[index], device=self.features.device)
        self.since[where, rows] = torch.as_tensor(np.log1p(row[:, None] - self.said_row[index]) / SINCE_SCALE,
                                                  dtype=torch.float32, device=self.features.device)

    @torch.no_grad()
    def speak(self, rank: np.ndarray, runway_locked: np.ndarray) -> np.ndarray:
        """``[N, 6]``: the classes said at the step (0: unchanged, else the word + 1) — by the aircraft ranked (``rank``
        ≥ 0: there, at or past its first predicted step; one a scene a rank), round by round in their ranks, each column
        given the ones before it; the others say nothing. One whose runway is locked says no other runway."""
        rank = np.asarray(rank, dtype=np.int64)
        speaking = rank >= 0
        row = self.row
        if (speaking & ~(self.there & (row >= N_LOOK))).any():
            raise ValueError("an aircraft speaks while it is there, from its first predicted step")
        for p in range(int(rank.max()) + 1 if speaking.any() else 0):
            if len(set(self.scene[rank == p].tolist())) != int((rank == p).sum()):
                raise ValueError(f"two aircraft of one scene in round {p}")
        said = np.zeros((len(rank), 6), dtype=np.int64)
        if not speaking.any():
            return said
        h, tokens, valid = self._newest()
        opening = row == N_LOOK
        for p in range(int(rank.max()) + 1):
            now = rank == p
            chosen = self._sample(h, tokens, valid, now, opening, row, runway_locked)
            said[now] = chosen[now]
            written = now[:, None] & (chosen > 0)
            self.value = np.where(written, chosen, self.value)
            self.said_row = np.where(written, row[:, None], self.said_row)
        return said

    def _newest(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every scene's steps not encoded yet, encoded `BLOCK_STEPS` at a time: each commanded aircraft's
        ``(h [N, 1, 1, d], tokens [N, 1, 1, slots, d], valid [N, slots])`` at the step."""
        last = self.step + 1
        while self.steps_encoded < last:
            h, tokens, valid = self._encode(self.steps_encoded, min(last, self.steps_encoded + BLOCK_STEPS))
        scene = torch.as_tensor(self.scene, device=h.device)
        slot = torch.as_tensor(self.slot, device=h.device)
        return h[scene, slot, -1][:, None, None], tokens[scene, slot, -1][:, None, None], valid[scene]

    def _encode(self, first: int, last: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch steps ``first … last − 1`` encoded (module docstring: the references and the replayed as
        `scene_inputs` lays them out, the other commanded aircraft after them): ``(h, tokens, valid)`` of every aircraft
        on them."""
        refs = self.references
        own = {"features": self.features[refs], "relative": self.relative[refs], "in_force": self.in_force[refs],
               "since": self.since[refs]}
        inputs = scene_inputs(own, self.rows[refs].tolist(), self.others, self.pre, self.aircraft, first, last)
        extra = np.flatnonzero(self.slot > 0)
        rows = np.arange(first, last)[None, :] - self.start[extra, None]
        k, t = np.nonzero((rows >= 0) & (rows < self.rows[extra, None]))
        if len(k):
            device = inputs["features"].device
            a = torch.as_tensor(extra[k], device=device)
            r = torch.as_tensor(rows[k, t], device=device)
            w, s, t = (torch.as_tensor(v, device=device) for v in (self.scene[extra[k]], self.slot[extra[k]], t))
            for name in ("features", "relative", "in_force", "since"):
                inputs[name][w, s, t] = getattr(self, name)[a, r]
            inputs["present"][w, s, t] = True
            inputs["rows"][w, s, t] = r
        edges = torch.as_tensor(self.edges_of(first, last), dtype=inputs["features"].dtype,
                                device=inputs["features"].device)
        expected = (len(self.others), last - first, self.aircraft, self.aircraft, len(self.model.traffic_features))
        if tuple(edges.shape) != expected:
            raise ValueError(f"edge features {tuple(edges.shape)}, the steps encoded need {expected}")
        h, tokens, valid, self.past = self.model.extend(inputs["features"], inputs["relative"], inputs["static"],
                                                        inputs["in_force"], inputs["since"], self.airport,
                                                        inputs["present"], inputs["rows"], edges, self.past)
        self.steps_encoded = last
        return h, tokens, valid

    def _sample(self, h: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor, now: np.ndarray, opening: np.ndarray,
                row: np.ndarray, runway_locked: np.ndarray) -> np.ndarray:
        """``[N, 6]``: one round's classes (module docstring) — the aircraft ``now`` sampled as `generate.Speaker`
        samples, every other one "unchanged"."""
        model, device = self.model, self.features.device
        count = len(now)
        first = torch.as_tensor(opening, device=device)[:, None, None]
        silent = torch.as_tensor(~now, device=device)[:, None]
        chosen = torch.zeros((count, 1, 1, 6), dtype=torch.long, device=device)
        at = np.flatnonzero(now)
        step = row[at] - N_LOOK
        for c in range(6):
            logit = model.logits(h, tokens, valid, chosen, first)[c][:, 0, 0].double()
            probability = torch.softmax(logit / self.temperature, dim=-1)
            if c in self.forbidden:
                allowed_np = self._allowed(c, chosen[:, 0, 0].cpu().numpy(), opening, logit.shape[1], runway_locked,
                                           now, row)
                self.allowed[c][at, step] = np.packbits(allowed_np[at], axis=1, bitorder="little")
                allowed = torch.as_tensor(allowed_np, device=device)
                self.forbidden[c][at, step] = (probability * ~allowed).sum(dim=1).cpu().numpy()[at]
                probability = torch.softmax((logit / self.temperature).masked_fill(~allowed, float("-inf")), dim=-1)
            unchanged = torch.zeros_like(probability)
            unchanged[:, 0] = 1.0
            probability = torch.where(silent, unchanged, probability)
            chosen[:, 0, 0, c] = torch.multinomial(probability, 1, generator=self.generator)[:, 0]
        return chosen[:, 0, 0].cpu().numpy()

    def _allowed(self, column: int, chosen: np.ndarray, opening: np.ndarray, classes: int, runway_locked: np.ndarray,
                 now: np.ndarray, row: np.ndarray) -> np.ndarray:
        """``[N, classes]``: the classes of ``column`` each aircraft of the round (``now``) may say after the columns
        before it — the vocabulary's rules, every set of the procedure's masks masking it and the caller's masks together
        (`generate.Speaker._allowed`, each at its own row); every class for the others."""
        at = np.flatnonzero(now)
        newest = np.clip(row, 0, self.e.shape[1] - 1)
        ar = np.arange(len(now))
        position = (self.e[ar, newest], self.n[ar, newest], self.h[ar, newest])
        out = (vocabulary_allowed(column, chosen, opening, classes, runway_locked, self.value, position[2], self.words,
                                  rows=at)
               if column in VOCABULARY_COLUMNS else np.ones((len(now), classes), dtype=bool))
        for rules in self.procedure:
            if column in rules.columns:
                for flag in (True, False):
                    part = now & (opening == flag)
                    if part.any():
                        out[part] &= rules.allowed(column, chosen, self.value, position, flag, classes)[part]
        if column in self.mask_columns:
            out &= self.masks_of(column, chosen, now)
        out[~now] = True
        return out
