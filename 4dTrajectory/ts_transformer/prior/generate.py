"""The prior speaking (prior design §5, §9.1): at each step it samples the step's six words in column order — the
runway, then the approach given it, and so on (the ordered heads read what the earlier columns just chose) — from what
it can know before the step: the positions so far, the airport's landings before the step, the words it has said.

The positions are the flight's observed rows up to the first predicted step (`scene.N_LOOK`); after it, the caller
appends where the aircraft is at the start of each step (`Speaker.append`) — in a closed loop, where the executor flew
the words said. No executor here: the prior never reads it (framework document §2); the runner joins the two.

The inputs are built row by row with the training data's own functions (`data.row_inputs`, the words said as
`data.sentence_steps` shifts them), so a speaker's rows are the rows `data.flight_steps` builds from the same positions
and words — the tests hold it to that. The model encodes them row by row too (`Prior.extend`: each layer's keys and
values of the rows already read are kept), the arithmetic of encoding them all again.

**The vocabulary's compatibility rules are a mask** (vocabulary design §2.1, §2.5: "checked in labelling, a mask in
decoding"): a runway changed under a clearance takes the approach with it; the altitude and descent-angle classes in
force must agree at the aircraft's altitude ("descend to land" with a descent class). The approach column is masked
given the runway just sampled, the angle column given the altitude just sampled — the ordered heads come in that order
— by the labeller's own check (`instructions.grammar.step_allowed`). The probability the model put on what the mask
removed is recorded (`forbidden`): how much of the grammar it has learned.

**The runway column is masked by the listener's rule** where the caller says the runway is locked: an executor refuses
a runway change once it has been cleared (since the last go-around) or has captured the line (executor design §4.6) —
stricter than the vocabulary's rule, which allows it with the approach changed in the same step; the stricter rule wins,
so the listener never hears what it refuses. A flight the caller marks inactive (its flight is over) says nothing.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.grammar import step_allowed
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, RUNWAY, UNCHANGED, Words
from ts_transformer.prior.data import SINCE_SCALE, STEP_FEATURES, VARIANTS, own_context, row_inputs
from ts_transformer.prior.model import Prior, self_edges
from ts_transformer.prior.scene import N_LOOK, Landings, utc_s


class Speaker:
    """A batch of flights the prior speaks to, all at the same row: rows 0 … `N_LOOK` observed, one more row per
    `append`. `speak` samples the six words of the newest row."""

    #: What `take` keeps per flight (arrays, tensors and lists, first axis the flight); the masses the masks removed
    #: (`forbidden`, one entry per step) are taken with them.
    PER_FLIGHT = ("geometries", "contexts", "entry_s", "e", "n", "h", "features", "relative", "in_force", "since",
                  "airport", "static", "value", "said_row")

    def __init__(self, model: Prior, flights: Sequence[FlightSignals], geometries: Sequence[AirportGeometry],
                 landings: Mapping[str, Landings] | None, words: Words, *, max_rows: int, generator: torch.Generator,
                 temperature: float = 1.0) -> None:
        """``landings``: each airport's landing context (`data.airport_landings`), None for a variant without it;
        ``max_rows``: the most rows any flight will have (the model's position table must hold them)."""
        if VARIANTS[model.config.variant].landing_context != (landings is not None):
            raise ValueError(f"variant {model.config.variant} and the landing context given disagree")
        if max_rows > model.config.max_rows:
            raise ValueError(f"{max_rows} rows, the model's positions end at {model.config.max_rows}")
        self.model, self.geometries, self.generator, self.temperature = model, list(geometries), generator, temperature
        self.words = words
        #: per step: the probability the model put, before the mask, on what the mask removed — [B] per column
        self.forbidden: dict[int, list[np.ndarray]] = {RUNWAY: [], APPROACH: [], ANGLE: []}
        device = model.candidates.device
        count, slots = len(flights), model.config.candidate_slots
        width = len(VARIANTS[model.config.variant].relative_features)
        self.contexts = [own_context(f, landings[f.airport]) if landings is not None else None for f in flights]
        self.entry_s = [utc_s(f.entry_time_utc) for f in flights]
        self.step_s = words.spec.step_s
        self.time_s = np.arange(max_rows) * self.step_s          # every flight's rows, on the vocabulary's step
        for f in flights:
            if not np.allclose(f.time_s[: N_LOOK + 1], self.time_s[: N_LOOK + 1]):
                raise ValueError(f"{f.dataset_id}: its rows are not {self.step_s} s apart from 0")
        # positions, row by row (the observed ones, then the ones appended)
        self.e, self.n, self.h = (np.zeros((count, max_rows)) for _ in range(3))
        for b, f in enumerate(flights):
            self.e[b, : N_LOOK + 1], self.n[b, : N_LOOK + 1] = f.e_m[: N_LOOK + 1], f.n_m[: N_LOOK + 1]
            self.h[b, : N_LOOK + 1] = f.altitude_m[: N_LOOK + 1]
        self.features = torch.zeros((count, 1, max_rows, len(STEP_FEATURES)), device=device)
        self.relative = torch.zeros((count, 1, max_rows, slots, width), device=device)
        self.in_force = torch.zeros((count, 1, max_rows, 6), dtype=torch.long, device=device)
        self.since = torch.zeros((count, 1, max_rows, 6), device=device)
        self.airport = torch.tensor([model.config.airports.index(f.airport) for f in flights], device=device)
        self.static = torch.zeros((count, 1, 0), device=device)
        self.rows = N_LOOK + 1
        self._inputs(0)
        # the rows the model has encoded, and each layer's keys and values of them (`Prior.extend`)
        self.encoded, self.past = 0, model.no_past(count, max_rows)
        # the words said: the class in force per column (0: none yet) and the row it was said at
        self.value = np.zeros((count, 6), dtype=np.int64)
        self.said_row = np.zeros((count, 6), dtype=np.int64)

    def _inputs(self, first: int) -> None:
        """Rows ``first`` … ``rows − 1``'s features and candidate relations (`data.row_inputs`)."""
        for b, geometry in enumerate(self.geometries):
            features, relative = row_inputs(self.e[b, : self.rows], self.n[b, : self.rows], self.h[b, : self.rows],
                                            self.time_s[: self.rows], self.entry_s[b], first, geometry,
                                            self.contexts[b])
            self.features[b, 0, first: self.rows] = torch.as_tensor(features)
            self.relative[b, 0, first: self.rows, : relative.shape[1]] = torch.as_tensor(relative)

    def take(self, index: np.ndarray) -> None:
        """Keep the flights at ``index`` — a closed loop's branches: a flight taken twice is spoken to on as two copies
        of itself, each with the rows it read and the words it said."""
        rows = torch.as_tensor(index, device=self.features.device)
        for layer, past in enumerate(self.past):                # one layer at a time: one copy alive at once
            self.past[layer] = past.take(rows)
        for name in self.PER_FLIGHT:
            value = getattr(self, name)
            if isinstance(value, list):
                setattr(self, name, [value[i] for i in index])
            else:
                setattr(self, name, value[rows] if isinstance(value, torch.Tensor) else value[index])
        self.forbidden = {column: [mass[index] for mass in masses] for column, masses in self.forbidden.items()}

    def append(self, e: np.ndarray, n: np.ndarray, height: np.ndarray, frozen: np.ndarray) -> None:
        """The next row's position of every flight, ``[B]`` each (after at least one `speak`); a ``frozen`` flight (its
        flight is over — its state may no longer be finite) repeats its last row."""
        if self.rows <= N_LOOK or not self.value.all():
            raise ValueError("a row is appended after the first predicted step was said")
        row = self.rows
        self.e[:, row] = np.where(frozen, self.e[:, row - 1], e)
        self.n[:, row] = np.where(frozen, self.n[:, row - 1], n)
        self.h[:, row] = np.where(frozen, self.h[:, row - 1], height)
        self.rows += 1
        self._inputs(row)
        # the words in force at the row before, as the prior said them (`data.sentence_steps`)
        self.in_force[:, 0, row] = torch.as_tensor(self.value)
        self.since[:, 0, row] = torch.as_tensor(np.log1p(row - self.said_row) / SINCE_SCALE, dtype=torch.float32)

    @torch.no_grad()
    def speak(self, active: np.ndarray, runway_locked: np.ndarray) -> np.ndarray:
        """``[B, 6]``: the classes sampled at the newest row (0: unchanged, else the word + 1), each column given the
        ones before it; an inactive flight says nothing (all 0); a flight whose runway is locked says no other runway."""
        model, rows, device = self.model, self.rows, self.features.device
        count, new = len(self.geometries), slice(self.encoded, self.rows)
        present = torch.ones((count, 1, rows - self.encoded), dtype=torch.bool, device=device)
        h, tokens, valid, self.past = model.extend(self.features[:, :, new], self.relative[:, :, new], self.static,
                                                   self.in_force[:, :, new], self.since[:, :, new], self.airport,
                                                   present, self_edges(count, 1, rows - self.encoded, device),
                                                   self.past)
        self.encoded = rows
        h, tokens = h[:, :, -1:], tokens[:, :, -1:]
        opening = rows - 1 == N_LOOK
        first = torch.tensor([opening], device=device)
        chosen = torch.zeros((count, 1, 1, 6), dtype=torch.long, device=device)
        for c in range(6):
            logit = model.logits(h, tokens, valid, chosen, first)[c][:, 0, 0].double()
            probability = torch.softmax(logit / self.temperature, dim=-1)
            if c in self.forbidden:
                allowed = torch.as_tensor(self._allowed(c, chosen[:, 0, 0].cpu().numpy(), opening, logit.shape[1],
                                                        runway_locked), device=device)
                self.forbidden[c].append((probability * ~allowed).sum(dim=1).cpu().numpy())
                probability = torch.softmax((logit / self.temperature).masked_fill(~allowed, float("-inf")), dim=-1)
            chosen[:, 0, 0, c] = torch.multinomial(probability, 1, generator=self.generator)[:, 0]
        said = np.where(active[:, None], chosen[:, 0, 0].cpu().numpy(), 0)
        written = said > 0
        self.value = np.where(written, said, self.value)
        self.said_row = np.where(written, rows - 1, self.said_row)
        return said


    def _allowed(self, column: int, chosen: np.ndarray, opening: bool, classes: int,
                 runway_locked: np.ndarray) -> np.ndarray:
        """``[B, classes]``: the classes of ``column`` each flight may say after the columns before it (``chosen``:
        this step's classes so far, [B, 6]) — the runway: another runway, or none where locked; the approach and the
        angle: what the grammar allows. At the first predicted step every column is said, so "unchanged" (class 0) is
        not asked about (the model already masks it)."""
        spec = self.words.spec
        out = np.ones((len(chosen), classes), dtype=bool)
        if column == RUNWAY:
            if not opening:
                # the runway in force is not said again (the labeller drops a word equal to the one in force, and a
                # runway said again would ask the approach to change under a clearance); a locked one stays
                out[np.arange(len(chosen)), self.value[:, RUNWAY]] = False
                out[runway_locked, 1:] = False
            return out
        for b in range(len(chosen)):
            if column == APPROACH and (opening or chosen[b, RUNWAY] == 0):
                continue                                     # the runway rule asks only of a later step that changes it
            step = np.where(chosen[b] > 0, chosen[b] - 1, UNCHANGED)
            # the columns after this one are not sampled yet; at the first step (every column said) the speed, the
            # only one after the angle, stands in with any word: it enters no rule
            step[column + 1:] = 0 if opening else UNCHANGED
            in_force = None if opening else self.value[b] - 1
            height = float(self.h[b, self.rows - 1])
            for k in range(1 if opening else 0, classes):
                step[column] = k - 1 if k else UNCHANGED
                if column == ANGLE and not opening and step[ALTITUDE] == UNCHANGED and k == 0:
                    continue                                 # nothing said in either column: nothing to check
                out[b, k] = step_allowed(in_force, step, height, spec, self.words)
        return out


def rows_for(remaining_s: float, step_s: float) -> int:
    """The rows a flight has when it is spoken to for ``remaining_s`` after the first predicted step."""
    return N_LOOK + 1 + math.ceil(remaining_s / step_s)
