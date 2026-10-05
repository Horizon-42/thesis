"""The speaker (prior design §4, §7 item 3; milestone B4): it says one row for each aircraft of a batch, column by
column in the order of the columns, from the inputs of the row, with the model's cache from row to row
(`model.Prior.extend`), under the masks of §4, with the random numbers that the caller gives (§4 "Drawing", D96).

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

**Drawing (D96).** The caller gives one uniform number in [0, 1) for each aircraft and column of the row; the speaker
says the word whose cumulative probability, in the class order of the heads, under the distribution with every mask
applied (and the temperature), first passes the number (`draw`). An aircraft's words thus depend only on its own numbers
and inputs (the probabilities up to their last bits: a batch of other aircraft can move a number that lies within the
float tolerance of a boundary, post-training §6.4).

**What the speaker keeps of each row said.** `permitted`: for each column the words every mask permitted (an object the
caller keeps and gives back, `Permitted`; `train.masked_log_probability` gives the log-probability of words under it);
`forbidden`: the probability the model put on the words the masks removed (how much of the masks it has not learned);
the probability and the permission of "go-around" (D72); the words the procedure masks blocked (B6). The caller's input
of the added modules (§7 item 5, ``extra``) is passed to the model at every row it encodes or says. `copy` gives chosen
aircraft of a speaker, repeats permitted: their cache, procedure masks, words in force, go-arounds and records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import torch

from ts_transformer.instructions.grammar import InForce, column_mask, column_words
from ts_transformer.instructions.words import COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words
from ts_transformer.prior.batch import RUNWAY_FIXED_CLASSES, RUNWAY_GO_AROUND_CLASS, RowTensors
from ts_transformer.prior.inputs import Heard
from ts_transformer.prior.model import Past, Prior
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


def draw(probabilities: torch.Tensor, numbers: np.ndarray) -> torch.Tensor:
    """``[B, 1]`` the class whose cumulative probability (in the class order) first passes each aircraft's number
    (``numbers`` [B], uniform in [0, 1)) under ``probabilities`` [B, classes] (§4 "Drawing", D96): a number past the
    float sum of the probabilities says the last class of positive probability."""
    cumulative = probabilities.double().cumsum(dim=-1)
    u = torch.as_tensor(np.asarray(numbers, dtype=np.float64), device=probabilities.device)[:, None]
    passed = (cumulative <= u).sum(dim=-1)
    positive = probabilities > 0
    last = positive.shape[-1] - 1 - positive.flip(-1).to(torch.int64).argmax(dim=-1)
    return torch.minimum(passed, last)[:, None]


@dataclass(frozen=True)
class Permitted:
    """The words every mask permitted at each row a speaker said (D96 item 3): for each column ``[B, rows, classes]``
    bool (the heads' class order; a class past an aircraft's own — a runway class of a candidate its airport does not
    have, or one past the width of the batch it was said in — is False), each row's time (``time_s`` [B, rows], the
    aircraft's own seconds, as its inputs give it), each row's own-state inputs (``own`` [B, rows, features]: which
    aircraft and row a record is, checked where it is read) and the temperature it drew with. The caller keeps it and
    gives it back (`train.masked_log_probability`); it does not read it."""

    masks: tuple[np.ndarray, ...]
    time_s: np.ndarray
    own: np.ndarray
    temperature: float

    def select(self, indices: Sequence[int]) -> Permitted:
        """The rows of the aircraft ``indices`` (repeats permitted), in that order."""
        indices = list(indices)
        return Permitted(tuple(mask[indices] for mask in self.masks), self.time_s[indices], self.own[indices],
                         self.temperature)


def runway_after(word: np.ndarray, in_force: Sequence[InForce | None], n_candidates: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The runway and G in force after each aircraft's runway word ``word``, its words in force ``in_force`` before the
    row (the runway in force for "unchanged" and "go-around"; at the first predicted step, where those are never said,
    candidate 0 stands for the procedure masks of a row that says no candidate — `column_mask` refuses such a row). A
    candidate beyond an aircraft's own airport (a batch of airports of different sizes) is never permitted to it
    (`column_mask`); its procedure masks are read on the airport's last candidate, a stand-in that decides nothing."""
    runway = np.array([w if w >= 0 else (f.runway if f is not None else 0) for w, f in zip(word, in_force)])
    go_around = np.array([w == RUNWAY_GO_AROUND or (w == UNCHANGED and f is not None and f.go_around)
                          for w, f in zip(word, in_force)])
    return np.minimum(runway, np.asarray(n_candidates) - 1), go_around


class Speaker:
    """A batch of aircraft the prior speaks to, all at the same row. `observe` encodes the rows before the first
    predicted step; `speak` says the next row (the first predicted step first)."""

    def __init__(self, model: Prior, words: Words, finals: Sequence[Sequence[Final]], *, capacity: int,
                 temperature: float = 1.0) -> None:
        """``finals``: each aircraft's airport's finals, one for each candidate in the pointer's order (the procedure
        masks it speaks under, §4, kept fresh for this batch: their state is this batch's); ``capacity``: the rows the
        cache has room for at first (it grows as a sentence needs, `model.Past.grown`)."""
        if not temperature > 0.0:
            raise ValueError(f"temperature {temperature}: it must be positive")
        if model.training:
            raise ValueError("the speaker's model is training (dropout on): a speaker speaks in eval mode")
        for finals_b in finals:
            geometry = finals_b[0].geometry
            if [f.index for f in finals_b] != list(range(len(geometry.candidates))):
                raise ValueError(f"{geometry.code}: the finals are not one for each candidate, in the pointer's order")
        self.model, self.words = model, words
        self.procedure = ProcedureMasks(finals, words)
        self.n_candidates = np.array([len(f) for f in finals], dtype=np.int64)
        self.temperature = temperature
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
        #: per row said: for each column the procedure masks rule (`ProcedureMasks.columns`), ``[B, words]`` the words
        #: they blocked, under the runway and G after the row's runway word (the Training view shows them, §12 B6)
        self.procedure_blocked: list[dict[int, np.ndarray]] = []
        #: per row said: for each column ``[B, classes]`` the words every mask permitted, and ``[B]`` the row's time
        #: (`permitted` stacks them; the runway column's width is the batch's, padded with False when stacked)
        self.permitted_rows: list[tuple[np.ndarray, ...]] = []
        self.permitted_times: list[np.ndarray] = []
        self.permitted_own: list[np.ndarray] = []
        #: per row said: ``[B, 5]`` the probability of each word said in the distribution it was drawn from
        self.drawn_probability: list[np.ndarray] = []
        #: the go-arounds each aircraft has said (D68: a caller bounds them with `go_around_bound`)
        self.go_arounds = np.zeros(len(finals), dtype=np.int64)

    @torch.no_grad()
    def observe(self, rows: RowTensors, positions: Sequence[Position], extra: Any = None) -> None:
        """Encode observed rows (each aircraft's rows before its first predicted step), ``positions`` one a row; the
        procedure masks take them on. ``extra``: the caller's input of the added modules (§7 item 5)."""
        if rows.first.any():
            raise ValueError("observed rows hold no first predicted step: the speaker says it")
        if self.model.training:
            raise ValueError("the speaker's model is training (dropout on): it speaks in eval mode")
        _, _, self.past = self.model.extend(rows, self.past, extra)
        for at in positions:
            self.procedure.track(at.e_m, at.n_m, at.height_m, self._go_around())

    @torch.no_grad()
    def speak(self, row: RowTensors, at: Position, numbers: np.ndarray, caller: Mapping[int, np.ndarray] | None = None,
              extra: Any = None) -> np.ndarray:
        """Say one row (``row``: its inputs, ``[B, 1]``; ``at``: where each aircraft is): ``[B, 5]`` words, in the
        vocabulary's values. ``numbers``: ``[B, 5]`` the caller's uniform numbers in [0, 1) (`draw`); ``caller``: the
        masks of a caller, by column; ``extra``: the caller's input of the added modules (§7 item 5)."""
        if row.present.shape[1] != 1:
            raise ValueError(f"a speaker says one row at a time, not {row.present.shape[1]}")
        if self.model.training:
            raise ValueError("the speaker's model is training (dropout on): it speaks in eval mode")
        numbers = np.asarray(numbers, dtype=np.float64)
        if numbers.shape != (len(self.heard), len(COLUMNS)) or not ((numbers >= 0.0) & (numbers < 1.0)).all():
            raise ValueError(f"the numbers of a row: [{len(self.heard)}, {len(COLUMNS)}] in [0, 1), not "
                             f"{numbers.shape}")
        if self.permitted_times and not (row.time_s[:, 0].cpu().numpy() > self.permitted_times[-1]).all():
            raise ValueError("a row said at or before the time of the row said last: rows are said in time order")
        caller = dict(caller or {})
        self.procedure.track(at.e_m, at.n_m, at.height_m, self._go_around())
        h, tokens, self.past = self.model.extend(row, self.past, extra)
        g = h
        count = len(self.in_force)
        said = np.zeros((count, 0), dtype=np.int64)
        permitted_row, drawn_row = [], np.zeros((count, len(COLUMNS)))
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
            chosen = draw(drawn, numbers[:, column])
            permitted_row.append(permitted.cpu().numpy())
            drawn_row[:, column] = drawn.gather(-1, chosen)[:, 0].double().cpu().numpy()
            g = self.model.after_choice(column, g, chosen, tokens)
            said = np.concatenate((said, class_words(column, chosen.cpu().numpy())), axis=1)
        runway, go_around = self._runway_after(said[:, RUNWAY])
        self.procedure_blocked.append({c: ~self.procedure.permitted(c, runway, go_around, at.e_m, at.n_m, at.height_m)
                                       for c in self.procedure.columns})
        for heard, step, height, time_s in zip(self.heard, said, at.height_m, row.time_s[:, 0].tolist()):
            heard.hear(step, float(height), time_s)
        self.go_arounds += said[:, RUNWAY] == RUNWAY_GO_AROUND
        self.permitted_rows.append(tuple(permitted_row))
        self.permitted_times.append(row.time_s[:, 0].cpu().numpy().copy())
        self.permitted_own.append(row.own[:, 0].cpu().numpy().copy())
        self.drawn_probability.append(drawn_row)
        return said

    def permitted(self) -> Permitted:
        """The words every mask permitted at each row said so far (D96 item 3), for the caller to keep."""
        if not self.permitted_rows:
            raise ValueError("the speaker has said no row yet")
        masks = []
        for c in range(len(COLUMNS)):
            width = max(row[c].shape[1] for row in self.permitted_rows)
            masks.append(np.stack([np.pad(row[c], ((0, 0), (0, width - row[c].shape[1]))) for row in self.permitted_rows],
                                  axis=1))
        return Permitted(tuple(masks), np.stack(self.permitted_times, axis=1), np.stack(self.permitted_own, axis=1),
                         self.temperature)

    def copy(self, indices: Sequence[int]) -> Speaker:
        """A speaker of the aircraft ``indices`` of this one (repeats permitted), in that order (D96 item 5): their cache,
        the state of their procedure masks, their words heard, their go-arounds and their records. Continued with the
        same inputs and numbers, a copy of every aircraft says what this speaker says, bit for bit."""
        indices = list(indices)
        out = object.__new__(Speaker)
        out.model, out.words, out.temperature = self.model, self.words, self.temperature
        out.procedure = self.procedure.select(indices)
        out.n_candidates = self.n_candidates[indices].copy()
        index = torch.as_tensor(indices, device=self.past[0].keys.device)
        out.past = [Past(p.keys[index].clone(), p.values[index].clone(), p.present[index].clone(), p.rows)
                    for p in self.past]
        out.heard = [self.heard[i].copy() for i in indices]
        out.forbidden = {c: [row[indices] for row in rows] for c, rows in self.forbidden.items()}
        out.go_around_probability = [row[indices] for row in self.go_around_probability]
        out.go_around_permitted = [row[indices] for row in self.go_around_permitted]
        out.procedure_blocked = [{c: mask[indices] for c, mask in row.items()} for row in self.procedure_blocked]
        out.permitted_rows = [tuple(mask[indices] for mask in row) for row in self.permitted_rows]
        out.permitted_times = [row[indices] for row in self.permitted_times]
        out.permitted_own = [row[indices] for row in self.permitted_own]
        out.drawn_probability = [row[indices] for row in self.drawn_probability]
        out.go_arounds = self.go_arounds[indices].copy()
        return out

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
        return runway_after(word, self.in_force, self.n_candidates)

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

