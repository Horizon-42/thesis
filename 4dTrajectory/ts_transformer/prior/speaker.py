"""The speaker (prior design §4, §7 item 3; milestone B4): it says one row for each aircraft of a batch, column by
column in the order of the columns, from the inputs of the row, with the model's cache from row to row
(`model.Prior.extend`), under the masks of §4, with a random generator that the caller gives.

**Masks.** A word is said only where every mask permits it:

- the model's own: "unchanged" in each column and "go-around" at the first predicted step, a padded candidate;
- the procedure masks (`procedure.ProcedureMasks`, the altitude and angle columns), under the runway and G in force
  after the row's runway word;
- the masks of a caller: for each column a ``[B, words]`` bool of the words it permits (`instructions.grammar.
  column_words` order, the heads' class order), the same at every column of the row; the prior does not know what they
  mean;
- the grammar (D62, `instructions.grammar.column_mask`): a word only where some words of the later columns, each among
  the words the other masks permit there, make the row pass the rules. In the runway column the later columns' masks
  depend on the runway word asked (the procedure of R, and G), so the grammar is asked once for each runway word, with
  the later columns' masks under it (at the heading column after that word: some permitted heading word must be able to
  follow). A row thus never reaches a column with no permitted word; one that does is a
  defect, and raises.

`forbidden` records, for each column, the probability that the model put on the words the masks removed, at each row
said (`speak`): how much of the masks the model has not learned.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence

import numpy as np
import torch

from ts_transformer.instructions.grammar import InForce, column_mask, column_words
from ts_transformer.instructions.words import COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words
from ts_transformer.prior.batch import RUNWAY_FIXED_CLASSES, RUNWAY_GO_AROUND_CLASS, RowTensors
from ts_transformer.prior.inputs import Heard
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import Final, ProcedureMasks


class Position(NamedTuple):
    """Where each aircraft of the batch is at its newest row: airport frame (m) and the height above E (m), ``[B]``."""

    e_m: np.ndarray
    n_m: np.ndarray
    height_m: np.ndarray


def class_words(column: int, classes: np.ndarray) -> np.ndarray:
    """The heads' classes of ``column`` as the vocabulary's words (the inverse of `batch.target_classes`)."""
    classes = np.asarray(classes, dtype=np.int64)
    if column == RUNWAY:
        return np.where(classes == 0, UNCHANGED,
                        np.where(classes == RUNWAY_GO_AROUND_CLASS, RUNWAY_GO_AROUND, classes - RUNWAY_FIXED_CLASSES))
    return classes - 1


#: The most go-arounds a flight may say in free generation (D68); a caller forbids "go-around" after them
#: (`go_around_bound`).
MOST_GO_AROUNDS = 2


def go_around_bound(go_arounds: np.ndarray, words: Words, most_candidates: int,
                    most: int = MOST_GO_AROUNDS) -> np.ndarray:
    """A caller's mask of the runway column (`column_words` order) for aircraft that have said ``go_arounds`` go-arounds:
    "go-around" forbidden to those that have said ``most`` (D68), every other word permitted."""
    words_of = column_words(RUNWAY, words, most_candidates)
    out = np.ones((len(go_arounds), len(words_of)), dtype=bool)
    out[:, list(words_of).index(RUNWAY_GO_AROUND)] = np.asarray(go_arounds) < most
    return out


class Speaker:
    """A batch of aircraft the prior speaks to, all at the same row. `observe` encodes the rows before the first
    predicted step; `speak` says the next row (the first predicted step first)."""

    def __init__(self, model: Prior, words: Words, finals: Sequence[Sequence[Final]], *, capacity: int,
                 generator: torch.Generator, temperature: float = 1.0) -> None:
        """``finals``: each aircraft's airport's finals, one for each candidate in the pointer's order (the procedure
        masks it speaks under, §4, kept fresh for this batch: their state is this batch's); ``capacity``: the most rows
        any aircraft will have."""
        if not temperature > 0.0:
            raise ValueError(f"temperature {temperature}: it must be positive")
        for finals_b in finals:
            geometry = finals_b[0].geometry
            if [f.index for f in finals_b] != list(range(len(geometry.candidates))):
                raise ValueError(f"{geometry.code}: the finals are not one for each candidate, in the pointer's order")
        self.model, self.words = model.eval(), words
        self.procedure = ProcedureMasks(finals, words)
        self.n_candidates = np.array([len(f) for f in finals], dtype=np.int64)
        self.generator, self.temperature = generator, temperature
        self.past = model.no_past(len(finals), capacity)
        #: each aircraft's words in force (`inputs.Heard`, the training sentences' own walk): a loop reads the inputs of
        #: the next row from it, so the speaker and the inputs never keep two copies
        self.heard = [Heard(finals_b[0].geometry, words) for finals_b in finals]
        self.forbidden: dict[int, list[np.ndarray]] = {c: [] for c in range(len(COLUMNS))}
        #: per row said: each aircraft's probability of "go-around" in the distribution its runway word was drawn from
        #: (the masks applied; the readout's "probability of go-around", §12 B4)
        self.go_around_probability: list[np.ndarray] = []
        #: per row said: whether the masks permitted each aircraft "go-around" (no G in force, the bound of D68 not met)
        self.go_around_permitted: list[np.ndarray] = []
        #: the go-arounds each aircraft has said (D68: a caller bounds them with `go_around_bound`)
        self.go_arounds = np.zeros(len(finals), dtype=np.int64)

    @torch.no_grad()
    def observe(self, rows: RowTensors, positions: Sequence[Position]) -> None:
        """Encode observed rows (each aircraft's rows before its first predicted step), ``positions`` one a row; the
        procedure masks take them on."""
        if rows.first.any():
            raise ValueError("observed rows hold no first predicted step: the speaker says it")
        _, _, self.past = self.model.extend(rows, self.past)
        for at in positions:
            self.procedure.track(at.e_m, at.n_m, at.height_m, self._go_around())

    @torch.no_grad()
    def speak(self, row: RowTensors, at: Position, caller: Mapping[int, np.ndarray] | None = None) -> np.ndarray:
        """Say one row (``row``: its inputs, ``[B, 1]``; ``at``: where each aircraft is): ``[B, 5]`` words, in the
        vocabulary's values. ``caller``: the masks of a caller, by column."""
        if row.present.shape[1] != 1:
            raise ValueError(f"a speaker says one row at a time, not {row.present.shape[1]}")
        caller = dict(caller or {})
        self.procedure.track(at.e_m, at.n_m, at.height_m, self._go_around())
        h, tokens, self.past = self.model.extend(row, self.past)
        g = h
        count = len(self.in_force)
        said = np.zeros((count, 0), dtype=np.int64)
        for column in range(len(COLUMNS)):
            logits = self.model.column_logits(column, g, tokens, row.valid, row.first)[:, 0]
            allowed = self._allowed(column, said, at, caller)
            permitted = torch.as_tensor(allowed[:, : logits.shape[-1]], device=logits.device)
            if allowed[:, logits.shape[-1]:].any():
                raise ValueError(f"column {COLUMNS[column]}: a mask permits a word the model has no class for")
            masked = logits.masked_fill(~permitted, float("-inf"))
            if torch.isneginf(masked).all(dim=-1).any():
                raise ValueError(f"column {COLUMNS[column]}: an aircraft has no permitted word (D62)")
            probabilities = torch.softmax(logits, dim=-1)
            self.forbidden[column].append(probabilities.masked_fill(permitted, 0.0).sum(dim=-1).cpu().numpy())
            drawn = torch.softmax(masked / self.temperature, dim=-1)
            if column == RUNWAY:
                self.go_around_probability.append(drawn[:, RUNWAY_GO_AROUND_CLASS].cpu().numpy())
                self.go_around_permitted.append(permitted[:, RUNWAY_GO_AROUND_CLASS].cpu().numpy())
            chosen = torch.multinomial(drawn, 1, generator=self.generator)
            g = self.model.after_choice(column, g, chosen, tokens)
            said = np.concatenate((said, class_words(column, chosen.cpu().numpy())), axis=1)
        for heard, step, height, time_s in zip(self.heard, said, at.height_m, row.time_s[:, 0].tolist()):
            heard.hear(step, float(height), time_s)
        self.go_arounds += said[:, RUNWAY] == RUNWAY_GO_AROUND
        return said

    @property
    def in_force(self) -> list[InForce | None]:
        """Each aircraft's words in force (the grammar's), None before its first predicted step."""
        return [heard.state for heard in self.heard]

    def _go_around(self) -> np.ndarray:
        return np.array([f is not None and f.go_around for f in self.in_force])

    def _allowed(self, column: int, said: np.ndarray, at: Position, caller: Mapping[int, np.ndarray]) -> np.ndarray:
        """``[B, words]``: the words of ``column`` that every mask permits, after the row's earlier words ``said``
        (module docstring)."""
        if column != RUNWAY:
            runway, go_around = self._runway_after(said[:, RUNWAY])
            later = {c: self._others(c, runway, go_around, at, caller) for c in range(column + 1, len(COLUMNS))}
            return (column_mask(self.in_force, said, column, at.height_m, self.n_candidates, self.words, later)
                    & self._others(column, runway, go_around, at, caller))
        # a runway word is permitted where some permitted heading word of the row can follow it: the grammar asked at
        # the heading column after it, with the later columns' masks under the runway and G it leaves (exact, and one
        # heading column's work for each runway word, not the whole row's)
        count = len(self.in_force)
        runway_words = column_words(RUNWAY, self.words, int(self.n_candidates.max()))
        out = np.zeros((count, len(runway_words)), dtype=bool)
        for w, word in enumerate(runway_words):
            runway, go_around = self._runway_after(np.full(count, word))
            later = {c: self._others(c, runway, go_around, at, caller) for c in range(HEADING + 1, len(COLUMNS))}
            heading = column_mask(self.in_force, np.full((count, 1), word), HEADING, at.height_m, self.n_candidates,
                                  self.words, later)
            out[:, w] = (heading & self._others(HEADING, runway, go_around, at, caller)).any(axis=1)
        if RUNWAY in caller:
            out &= caller[RUNWAY]
        return out

    def _runway_after(self, word: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The runway and G in force after each aircraft's runway word ``word`` (the runway in force for "unchanged"
        and "go-around"; at the first predicted step, where those are never said, candidate 0 stands for the
        procedure masks of a row that says no candidate — `column_mask` refuses such a row). A candidate beyond an
        aircraft's own airport (a batch of airports of different sizes) is never permitted to it (`column_mask`); its
        procedure masks are read on the airport's last candidate, a stand-in that decides nothing."""
        runway = np.array([w if w >= 0 else (f.runway if f is not None else 0) for w, f in zip(word, self.in_force)])
        go_around = np.array([w == RUNWAY_GO_AROUND or (w == UNCHANGED and f is not None and f.go_around)
                              for w, f in zip(word, self.in_force)])
        return np.minimum(runway, self.n_candidates - 1), go_around

    def _others(self, column: int, runway: np.ndarray, go_around: np.ndarray, at: Position,
                caller: Mapping[int, np.ndarray]) -> np.ndarray:
        """The words of ``column`` the procedure masks and the caller's permit."""
        width = len(column_words(column, self.words, int(self.n_candidates.max())))
        out = np.ones((len(self.in_force), width), dtype=bool)
        if column in self.procedure.columns:
            out &= self.procedure.permitted(column, runway, go_around, at.e_m, at.n_m, at.height_m)
        if column in caller:
            out &= caller[column]
        return out

