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
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np
import torch

from ts_transformer.instructions.grammar import InForce, column_mask, column_words
from ts_transformer.instructions.words import COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words
from ts_transformer.prior.batch import RUNWAY_FIXED_CLASSES, RUNWAY_GO_AROUND_CLASS, RowTensors
from ts_transformer.prior.inputs import Heard
from ts_transformer.prior.model import Past, Prior, require_eval
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

    @staticmethod
    def join(records: Sequence[Permitted]) -> Permitted:
        """The records of several speakers (or copies) as one, their aircraft in the order given (D106 item 4): each
        column as wide as the widest record's, the rows as many as the longest's; a padded class is not permitted, a
        padded row has no time (NaN: no row asked matches it) and no inputs. One temperature for all."""
        if not records:
            raise ValueError("no records to join")
        if any(r.temperature != records[0].temperature for r in records):
            raise ValueError(f"records drawn at the temperatures {sorted({r.temperature for r in records})}: one batch "
                             f"has one")
        rows = max(r.time_s.shape[1] for r in records)

        def padded(value: np.ndarray, width: int, fill: Any) -> np.ndarray:
            pad = [(0, 0), (0, rows - value.shape[1])] + ([(0, width - value.shape[2])] if value.ndim == 3 else [])
            return np.pad(value, pad, constant_values=fill)

        masks = []
        for c in range(len(COLUMNS)):
            width = max(r.masks[c].shape[2] for r in records)
            masks.append(np.concatenate([padded(r.masks[c], width, False) for r in records]))
        own = np.concatenate([padded(r.own, r.own.shape[2], 0.0) for r in records])
        return Permitted(tuple(masks), np.concatenate([padded(r.time_s, 0, np.nan) for r in records]), own,
                         records[0].temperature)


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
    predicted step; `speak` says the next row (the first predicted step first). Rows come in time order, each later
    than the last encoded or said; no row is observed after the first is said. A refused row changes nothing: the
    cache, the procedure masks, the words heard and the records change only once every column of it has a word (as
    vocabulary D80 for the start)."""

    def __init__(self, model: Prior, words: Words, finals: Sequence[Sequence[Final]], *, capacity: int,
                 temperature: float = 1.0) -> None:
        """``finals``: each aircraft's airport's finals, one for each candidate in the pointer's order (the procedure
        masks it speaks under, §4, kept fresh for this batch: their state is this batch's); ``capacity``: the rows the
        cache has room for at first (it grows as a sentence needs, `model.Past.grown`)."""
        if not temperature > 0.0:
            raise ValueError(f"temperature {temperature}: it must be positive")
        require_eval(model, "the speaker")
        for finals_b in finals:
            geometry = finals_b[0].geometry
            if [f.index for f in finals_b] != list(range(len(geometry.candidates))):
                raise ValueError(f"{geometry.code}: the finals are not one for each candidate, in the pointer's order")
        self.model, self.words = model, words
        self.procedure = ProcedureMasks(finals, words)
        self._n_candidates = np.array([len(f) for f in finals], dtype=np.int64)
        self.temperature = temperature
        self.past = model.no_past(len(finals), capacity)
        # each aircraft's words in force (`inputs.Heard`, the training sentences' own walk): a loop reads the inputs of
        # the next row from it (`heard`, a copy: the speaker's own changes only as it says a row)
        self._heard = [Heard(finals_b[0].geometry, words) for finals_b in finals]
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
        # the go-arounds each aircraft has said (D68: a caller bounds them with `go_around_bound`; `go_arounds`)
        self._go_arounds = np.zeros(len(finals), dtype=np.int64)
        # each aircraft's time of the last row encoded or said: a row comes later
        self._last_s = np.full(len(finals), -np.inf)

    def _in_time_order(self, time_s: torch.Tensor) -> np.ndarray:
        """``[B, R]`` the rows' times, refused unless each aircraft's come after its last row and after each other."""
        times = time_s.cpu().numpy().astype(np.float64)
        if not ((times[:, 0] > self._last_s).all() and (np.diff(times, axis=1) > 0).all()):
            raise ValueError("a row at or before the time of the last row encoded or said: rows come in time order")
        return times

    @torch.no_grad()
    def observe(self, rows: RowTensors, positions: Sequence[Position], extra: Any = None) -> None:
        """Encode observed rows (each aircraft's rows before its first predicted step), ``positions`` one a row; the
        procedure masks take them on. ``extra``: the caller's input of the added modules (§7 item 5). Refused, before any
        change, unless ``positions`` are one for each row and each aircraft."""
        count, length = rows.present.shape[:2]
        if len(positions) != length or any(len(np.atleast_1d(v)) != count
                                           for at in positions for v in (at.e_m, at.n_m, at.height_m)):
            raise ValueError(f"the positions of observed rows: one for each of the {length} rows and {count} aircraft, "
                             f"got {len(positions)} rows of {sorted({len(np.atleast_1d(at.e_m)) for at in positions})}")
        if rows.first.any():
            raise ValueError("observed rows hold no first predicted step: the speaker says it")
        if self.permitted_rows:
            raise ValueError("a row was said: the rows before the first predicted step are observed before it")
        require_eval(self.model, "the speaker")
        times = self._in_time_order(rows.time_s)
        procedure = self.procedure.select(range(len(self._heard)))
        for at in positions:
            procedure.track(at.e_m, at.n_m, at.height_m, self._go_around())
        _, _, past = self.model.extend(rows, self.past, extra)
        self.past, self.procedure, self._last_s = past, procedure, times[:, -1]

    @torch.no_grad()
    def speak(self, row: RowTensors, at: Position, numbers: np.ndarray, caller: Mapping[int, np.ndarray] | None = None,
              extra: Any = None, accept: Callable[[np.ndarray], None] | None = None) -> np.ndarray:
        """Say one row (``row``: its inputs, ``[B, 1]``; ``at``: where each aircraft is): ``[B, 5]`` words, in the
        vocabulary's values. ``numbers``: ``[B, 5]`` the caller's uniform numbers in [0, 1) (`draw`); ``caller``: the
        masks of a caller, by column; ``extra``: the caller's input of the added modules (§7 item 5); ``accept``: the
        caller's last step with the row's words before the speaker keeps it (a closed loop's executor, vocabulary D80):
        when it refuses, the speaker is as it was (its cache's rows past the kept ones are written over). A row whose mark of
        the first predicted step is not, for each aircraft, whether nothing is in force yet is refused before any change
        (the masks of the first step read the mark: a later row marked first would draw without "unchanged")."""
        if row.present.shape[1] != 1:
            raise ValueError(f"a speaker says one row at a time, not {row.present.shape[1]}")
        nothing = np.array([heard.state is None for heard in self._heard])
        marked = row.first[:, 0].cpu().numpy()
        if (marked != nothing).any():
            wrong = np.flatnonzero(marked != nothing).tolist()
            raise ValueError(f"aircraft {wrong[:5]}: the row's mark of the first predicted step is not whether nothing is "
                             f"in force yet (the first step is the row said first)")
        require_eval(self.model, "the speaker")
        numbers = np.asarray(numbers, dtype=np.float64)
        if numbers.shape != (len(self._heard), len(COLUMNS)) or not ((numbers >= 0.0) & (numbers < 1.0)).all():
            raise ValueError(f"the numbers of a row: [{len(self._heard)}, {len(COLUMNS)}] in [0, 1), not "
                             f"{numbers.shape}")
        times = self._in_time_order(row.time_s)
        caller = dict(caller or {})
        # every change is made on copies, and kept only once every column of the row has a word
        procedure = self.procedure.select(range(len(self._heard)))
        procedure.track(at.e_m, at.n_m, at.height_m, self._go_around())
        h, tokens, past = self.model.extend(row, self.past, extra)
        g = h
        count = len(self.in_force)
        said = np.zeros((count, 0), dtype=np.int64)
        permitted_row, forbidden_row, drawn_row = [], [], np.zeros((count, len(COLUMNS)))
        for column in range(len(COLUMNS)):
            logits = self.model.column_logits(column, g, tokens, row.valid, row.first)[:, 0]
            allowed = self._allowed(column, said, at, caller, procedure)
            permitted = torch.as_tensor(allowed[:, : logits.shape[-1]], device=logits.device)
            if allowed[:, logits.shape[-1]:].any():
                raise ValueError(f"column {COLUMNS[column]}: a mask permits a word the model has no class for")
            masked = logits.masked_fill(~permitted, float("-inf"))
            if torch.isneginf(masked).all(dim=-1).any():
                raise ValueError(f"column {COLUMNS[column]}: an aircraft has no permitted word (D62)")
            probabilities = torch.softmax(logits, dim=-1)
            forbidden_row.append(probabilities.masked_fill(permitted, 0.0).sum(dim=-1).cpu().numpy())
            drawn = torch.softmax(masked / self.temperature, dim=-1)
            if column == RUNWAY:
                go_around_drawn = drawn[:, RUNWAY_GO_AROUND_CLASS].cpu().numpy()
                go_around_permitted = permitted[:, RUNWAY_GO_AROUND_CLASS].cpu().numpy()
            chosen = draw(drawn, numbers[:, column])
            permitted_row.append(permitted.cpu().numpy())
            drawn_row[:, column] = drawn.gather(-1, chosen)[:, 0].double().cpu().numpy()
            g = self.model.after_choice(column, g, chosen, tokens)
            said = np.concatenate((said, class_words(column, chosen.cpu().numpy())), axis=1)
        runway, go_around = self._runway_after(said[:, RUNWAY])
        blocked = {c: ~procedure.permitted(c, runway, go_around, at.e_m, at.n_m, at.height_m) for c in procedure.columns}
        heard = [aircraft.copy() for aircraft in self._heard]
        for heard_b, step, height, time_s in zip(heard, said, at.height_m, row.time_s[:, 0].tolist()):
            heard_b.hear(step, float(height), time_s)
        procedure.after_row(at.e_m, at.n_m, at.height_m, go_around)      # a row that ends G starts the stretch (D64)
        if accept is not None:
            accept(said)
        # the row is said: kept
        self.past, self.procedure, self._heard, self._last_s = past, procedure, heard, times[:, 0]
        self._go_arounds = self._go_arounds + (said[:, RUNWAY] == RUNWAY_GO_AROUND)
        for column, value in enumerate(forbidden_row):
            self.forbidden[column].append(value)
        self.go_around_probability.append(go_around_drawn)
        self.go_around_permitted.append(go_around_permitted)
        self.procedure_blocked.append(blocked)
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
        out._n_candidates = self._n_candidates[indices].copy()
        index = torch.as_tensor(indices, device=self.past[0].keys.device)
        out.past = [Past(p.keys[index].clone(), p.values[index].clone(), p.present[index].clone(), p.rows)
                    for p in self.past]
        out._heard = [self._heard[i].copy() for i in indices]
        out.forbidden = {c: [row[indices] for row in rows] for c, rows in self.forbidden.items()}
        out.go_around_probability = [row[indices] for row in self.go_around_probability]
        out.go_around_permitted = [row[indices] for row in self.go_around_permitted]
        out.procedure_blocked = [{c: mask[indices] for c, mask in row.items()} for row in self.procedure_blocked]
        out.permitted_rows = [tuple(mask[indices] for mask in row) for row in self.permitted_rows]
        out.permitted_times = [row[indices] for row in self.permitted_times]
        out.permitted_own = [row[indices] for row in self.permitted_own]
        out.drawn_probability = [row[indices] for row in self.drawn_probability]
        out._go_arounds = self._go_arounds[indices].copy()
        out._last_s = self._last_s[indices].copy()
        return out

    # ---- what a loop reads of the speaker, and does not change (§7 item 3, D106 item 6)
    @property
    def heard(self) -> tuple[Heard, ...]:
        """Each aircraft's words in force as the inputs read them (`inputs.Heard`, `prior.loop.LoopRows`): copies."""
        return tuple(heard.copy() for heard in self._heard)

    @property
    def in_force(self) -> list[InForce | None]:
        """Each aircraft's words in force (the grammar's), None before its first predicted step."""
        return [heard.state for heard in self._heard]

    @property
    def go_arounds(self) -> np.ndarray:
        """The go-arounds each aircraft has said (a copy)."""
        return self._go_arounds.copy()

    @property
    def n_candidates(self) -> np.ndarray:
        """Each aircraft's number of candidates (a copy)."""
        return self._n_candidates.copy()

    def _go_around(self) -> np.ndarray:
        return np.array([f is not None and f.go_around for f in self.in_force])

    def _allowed(self, column: int, said: np.ndarray, at: Position, caller: Mapping[int, np.ndarray],
                 procedure: ProcedureMasks) -> np.ndarray:
        """``[B, words]``: the words of ``column`` that every mask permits, after the row's earlier words ``said``
        (module docstring)."""
        if column != RUNWAY:
            runway, go_around = self._runway_after(said[:, RUNWAY])
            later = {c: self._others(c, runway, go_around, at, caller, procedure)
                     for c in range(column + 1, len(COLUMNS))}
            return (column_mask(self.in_force, said, column, at.height_m, self._n_candidates, self.words, later)
                    & self._others(column, runway, go_around, at, caller, procedure))
        # a runway word is permitted where some permitted heading word of the row can follow it: the grammar asked at
        # the heading column after it, with the later columns' masks under the runway and G it leaves (exact, and one
        # heading column's work for each runway word, not the whole row's)
        count = len(self.in_force)
        runway_words = column_words(RUNWAY, self.words, int(self._n_candidates.max()))
        out = np.zeros((count, len(runway_words)), dtype=bool)
        for w, word in enumerate(runway_words):
            runway, go_around = self._runway_after(np.full(count, word))
            later = {c: self._others(c, runway, go_around, at, caller, procedure)
                     for c in range(HEADING + 1, len(COLUMNS))}
            heading = column_mask(self.in_force, np.full((count, 1), word), HEADING, at.height_m, self._n_candidates,
                                  self.words, later)
            out[:, w] = (heading & self._others(HEADING, runway, go_around, at, caller, procedure)).any(axis=1)
        if RUNWAY in caller:
            out &= caller[RUNWAY]
        return out

    def _runway_after(self, word: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return runway_after(word, self.in_force, self._n_candidates)

    def _others(self, column: int, runway: np.ndarray, go_around: np.ndarray, at: Position,
                caller: Mapping[int, np.ndarray], procedure: ProcedureMasks) -> np.ndarray:
        """The words of ``column`` the procedure masks (``procedure``, the row's) and the caller's permit."""
        width = len(column_words(column, self.words, int(self._n_candidates.max())))
        out = np.ones((len(self.in_force), width), dtype=bool)
        if column in procedure.columns:
            out &= procedure.permitted(column, runway, go_around, at.e_m, at.n_m, at.height_m)
        if column in caller:
            out &= caller[column]
        return out

