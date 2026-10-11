"""The prior's network (prior design §2, §3, §7 items 3–5): one aircraft's rows, a causal time attention with RoPE on
seconds, candidate tokens, and five heads said in the order of the columns.

**A row's input** (`batch.RowTensors`) is the sum of: the projected own state, the time since each column's word, G
and the heading in force (sine, cosine); one embedding for each other column's word in force (index 0 "none yet");
the runway in force as its candidate's token of the row, projected ("none yet": a learned vector); and what the row
reads of the airport's candidates (an attention over their tokens, `CandidatePool`). There is no airport embedding,
no row-position embedding and no time-from-row-0 input (D5, D16).

**Candidate tokens** (D23, D24, D41). At every row each candidate becomes a vector through one network shared by every
candidate. No candidate carries its slot, and the pool is an attention whose weights sum to one over the real
candidates: the order of the candidates changes only the order of the runway scores, and their number is the
airport's own — not a sum that grows with it.

**Layers.** Pre-normed and residual: the time attention → a module added by a caller (`Layer.added`, §7 item 5; none
in the prior itself) → the feed-forward. Along time a row reads the present rows up to itself, and always itself, so
no row has every key masked. The rotation of a row's query and key uses its time in seconds (`rope_angles`, D16): the
attention reads only time differences, so a shift of every time changes no output. The angles are computed in float64
and the times are the aircraft's own (from its row 0), so the float32 inputs lose no row.

**RoPE base** (set at implementation, prior §2): `ROPE_BASE` 10,000. With a head of 32 the frequencies run from
1 rad/s (a period of 6.3 s: a 2 s row is a third of it) to 10,000^(−15/16) rad/s (a period of approximately 35,000 s,
ten times the longest sentence with a go-around).

**Heads** (§3). Five columns in `COLUMNS` order. Column k's head reads the hidden state plus the embeddings of the
classes the earlier columns chose at this row (the truth in teacher forcing, the sampled ones when the prior speaks).
The runway head scores each candidate, ``q(g) · token_j / √d``, after "unchanged" and "go-around"
(`batch.RUNWAY_FIXED_CLASSES`); a padded slot is −inf. At the first predicted step "unchanged" is −inf in every
column and "go-around" in the runway column (§4: the grammar's first step; the speaker's masks come on top).

**Row by row** (`Prior.extend`): every layer reads a row and the rows before it only, so a speaker that adds one row a
step keeps each layer's rotated keys and values (`Past`, allocated once for every row it will hold, written in place)
and encodes only the new rows — the same arithmetic as encoding every row again (`encode` is `extend` from nothing),
to rounding of the attention's summation order.

**Copies share their front.** A speaker can take several copies of one line (the continuations of a branch point).
The copies are the same up to the row where they part, so their cache rows are the same too. The cache holds those
rows once, as a front (`Front`). Each line holds only its own rows after the front. `Past.together` chooses the length
of the front so that the cache holds the fewest rows. A line reads its front in one of two ways (`Past.attend`):
`COPIED` copies the front back to the line, one layer at a time; the tensors are the same as for a line that holds all
its rows, bit for bit. `MERGED` reads the front once for all its lines and merges it with each line's own rows; there
is no copy, but the last bits of the attention sums can differ.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from ts_transformer.instructions.words import COLUMNS, RUNWAY, Words
from ts_transformer.prior.batch import (
    IN_FORCE_WORDS, OWN_FEATURES, RUNWAY_FIXED_CLASSES, RUNWAY_GO_AROUND_CLASS, RUNWAY_UNCHANGED_CLASS, RowTensors,
    variant_features,
)

#: The columns after the runway, whose words are classes of the vocabulary spec (`Words.class_counts`).
WORD_COLUMNS = COLUMNS[1:]
ROPE_BASE = 10_000.0
#: The number of rows by which a speaker's cache grows when it is full (64 rows = 256 s at 4 s a row). The attention
#: reads only the rows written, so the spare room changes no result.
CACHE_STEP_ROWS = 64
#: How a line reads a front that it shares with other lines (`Past.attend`). `COPIED`: the front is copied to each
#: line, one layer at a time; the result is the same, bit for bit, as for a line that holds all its rows. `MERGED`:
#: the front is read once for all its lines, and the scores of the front and of the line's own rows are normalised
#: together; there is no copy, but the last bits of a sum can differ.
COPIED, MERGED = "copied", "merged"
FRONT_ATTENTIONS = (COPIED, MERGED)


@dataclass(frozen=True)
class PriorConfig:
    """The shape of the network.

    ``word_values``: the number of classes in each of `WORD_COLUMNS`, "unchanged" not counted. It comes from the
    vocabulary spec (`from_words`). It is never a constant of the prior.

    ``variant``: the name of the candidate feature set (D39). `full` has the candidate features only.
    `constants` adds the runway constant features.
    """

    word_values: tuple[int, ...]
    variant: str
    d_model: int = 192
    layers: int = 4
    heads: int = 6
    feedforward: int = 768
    dropout: float = 0.1
    rope_base: float = ROPE_BASE

    def __post_init__(self) -> None:
        variant_features(self.variant)
        if len(self.word_values) != len(WORD_COLUMNS) or min(self.word_values) < 1:
            raise ValueError(f"word_values {self.word_values}: one positive count for each of {WORD_COLUMNS}")
        if self.d_model % self.heads or (self.d_model // self.heads) % 2:
            raise ValueError(f"d_model {self.d_model} is not {self.heads} heads of an even width (RoPE rotates pairs)")

    @classmethod
    def from_words(cls, words: Words, variant: str, **shape: Any) -> PriorConfig:
        counts = words.class_counts()
        return cls(tuple(counts[name] for name in WORD_COLUMNS), variant, **shape)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "word_values": list(self.word_values)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriorConfig:
        expected = {item.name for item in fields(cls)}
        if set(data) != expected:
            raise ValueError(f"not a prior config: missing {sorted(expected - set(data))}, "
                             f"unexpected {sorted(set(data) - expected)}")
        return cls(**{**data, "word_values": tuple(data["word_values"])})


def rope_angles(time_s: torch.Tensor, head: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The RoPE rotation angles for the times ``time_s`` ``[..., T]`` (seconds): ``(cos, sin)``, each
    ``[..., T, head / 2]``.

    Pair i of a head turns at the frequency ``base ** (-i / (head / 2))`` rad/s. The angles are computed in float64
    and returned in the dtype of ``time_s``."""
    half = head // 2
    frequency = base ** (-torch.arange(half, dtype=torch.float64, device=time_s.device) / half)
    angle = time_s.double()[..., None] * frequency
    return angle.cos().to(time_s.dtype), angle.sin().to(time_s.dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[..., T, head]`` by the angles ``cos`` and ``sin`` of `rope_angles`.

    Element i of the first half of the head and element i of the second half make one pair. Each pair turns by its
    own angle."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class Front(NamedTuple):
    """The first `start` rows that the continuations of a branch point share, stored once.

    At a branch point the speaker copies a window K times (K = 8) and speaks each copy on with other random draws.
    Before the branch point the K copies are the same, so their cached keys and values of those rows are the same.
    Without a front each copy holds them: K times the memory. With a front the cache holds them once, and each copy
    holds only the rows it says after the branch point.

    A line (one aircraft of one copy) that has no copy gets a front of its own. So every line uses one front: B lines
    use F fronts (F <= B). Example: a window of 3 aircraft, copied 8 times, has B = 24 lines. The 8 copies of one
    aircraft use one front, so F = 3. Training and plain speaking have no copies and no front (`Past.front` is None).

    ``runs`` is for the `MERGED` attention. It reads the lines of a front in one batched product, which needs the same
    number of lines n for every front in the product. ``runs`` has one group for each n: (the first front of the group,
    n, the lines of the group's fronts, front after front). The groups follow each other in front order, the largest n
    first. In the example there is one group, with n = 8.
    """

    keys: torch.Tensor      # [F, heads, start, head]: the rotated keys of the front's rows
    values: torch.Tensor    # [F, heads, start, head]: the values of the front's rows
    present: torch.Tensor   # [F, start] bool: which rows of the front exist
    of: torch.Tensor        # [B] int: the front that each line uses
    runs: tuple[tuple[int, int, torch.Tensor], ...]


class Past(NamedTuple):
    """The cache of one layer for B lines: the rotated keys and values of the time attention.

    A line is one aircraft's sequence of rows (one item of the batch). Its time attention reads only its own rows.
    The rows of a line are in two places:

    - The first `start` rows are in ``front``, shared with the copies of the line. None: no copies, `start` is 0.
    - The other rows are in ``keys`` and ``values`` ``[B, heads, room, head]`` and ``present`` ``[B, room]``. Row
      ``start + j`` of the line is at index ``j``. The indexes up to ``rows - start`` are written; the rest of
      ``room`` is spare.

    Example: a line has 250 rows and its branch point is at row 200. Then `start` is 200 and ``rows`` is 250. The 50
    rows after the branch point are at indexes 0 to 49.

    ``kin`` and ``kin_rows`` ``[B]`` say which lines are copies of one another. Lines with the same ``kin`` are
    copies. ``kin_rows[i]`` is the number of leading rows that line i has in common with its copies (the rows up to
    the branch point). `together` uses them to choose `start`. ``attention`` (`COPIED` or `MERGED`) is how `attend`
    reads the front.

    The tensors are written in place. To branch, make new storage with `together`: two extends from the same `Past`
    overwrite each other's rows. A gradient cannot flow back through an extend that a later one overwrote. The speaker
    extends without gradients, and training encodes a whole sentence at once."""

    keys: torch.Tensor
    values: torch.Tensor
    present: torch.Tensor
    rows: int
    front: Front | None
    kin: np.ndarray
    kin_rows: np.ndarray
    attention: str

    @property
    def start(self) -> int:
        """The rows of each line's front."""
        return 0 if self.front is None else self.front.keys.shape[2]

    def grown(self, capacity: int) -> Past:
        """The same rows with room for ``capacity`` own rows in each line (new storage). A line that outgrows its room
        is moved, never cut. The front does not change."""
        extra = capacity - self.keys.shape[2]
        pad = self.keys.new_zeros((*self.keys.shape[:2], extra, self.keys.shape[3]))
        return self._replace(keys=torch.cat((self.keys, pad), dim=2), values=torch.cat((self.values, pad), dim=2),
                             present=torch.cat((self.present, self.present.new_zeros((self.present.shape[0], extra))),
                                               dim=1))

    def attend(self, q: torch.Tensor, dropout_p: float) -> torch.Tensor:
        """The time attention ``[B, heads, R, head]`` for the queries ``q`` of the last R rows written. A row reads the
        rows of its line that are present and not after it, and itself.

        Without a front, one attention reads the line's own rows. With a front: `COPIED` copies the front to each line
        for this layer and reads one attention for each line, as for a line that holds all its rows (the same tensors,
        bit for bit). `MERGED` reads each front once for all its lines (`Front.runs`) and reads the own rows of each
        line apart; the scores of both are normalised together and the values are summed. There is no copy, but the
        order of the sums differs, so the last bits can differ."""
        count = q.shape[2]
        start, written = self.start, self.rows - self.start
        steps = torch.arange(start, self.rows, device=q.device)
        queries = steps[-count:, None]
        own = (steps[None, :] <= queries) & (self.present[:, None, :written] | (steps[None, :] == queries))
        keys, values = self.keys[:, :, :written], self.values[:, :, :written]
        if self.front is None:
            return functional.scaled_dot_product_attention(q, keys, values, attn_mask=own[:, None],
                                                           dropout_p=dropout_p)
        if dropout_p:
            raise ValueError("a front held once is read without dropout (the speaker reads in eval mode)")
        front = self.front
        # every row of a front is before the rows written: a query reads the front rows that are present
        allowed = torch.cat((front.present.index_select(0, front.of)[:, None, :].expand(-1, count, -1), own), dim=2)
        if self.attention == COPIED:
            whole = []
            for held, mine in ((front.keys, keys), (front.values, values)):
                line = held.new_empty((len(front.of), held.shape[1], self.rows, held.shape[3]))
                torch.index_select(held, 0, front.of, out=line[:, :, :start])
                line[:, :, start:] = mine
                whole.append(line)
            return functional.scaled_dot_product_attention(q, *whole, attn_mask=allowed[:, None])
        batch, heads, _, head = q.shape

        def by_front(x: torch.Tensor, n: int, lines: torch.Tensor) -> torch.Tensor:
            """``x`` [B, heads, R, w] of a run's lines as [its fronts, heads, n R, w]: a front's lines one after another."""
            fronts = len(lines) // n
            return x.index_select(0, lines).view(fronts, n, heads, count, -1).transpose(1, 2).reshape(
                fronts, heads, n * count, -1)

        def by_line(x: torch.Tensor, n: int) -> torch.Tensor:
            """`by_front` undone: [fronts, heads, n R, w] → [the run's lines, heads, R, w]."""
            fronts = x.shape[0]
            return x.view(fronts, heads, n, count, -1).transpose(1, 2).reshape(fronts * n, heads, count, -1)

        scores = q.new_empty((batch, heads, count, self.rows))
        for first, n, lines in front.runs:
            held = front.keys[first:first + len(lines) // n]
            scores[:, :, :, :start].index_copy_(0, lines, by_line(by_front(q, n, lines) @ held.transpose(-1, -2),
                                                                  n))
        scores[:, :, :, start:] = q @ keys.transpose(-1, -2)
        weights = torch.softmax((scores * head ** -0.5).masked_fill(~allowed[:, None], float("-inf")), dim=-1)
        out = weights[:, :, :, start:] @ values
        for first, n, lines in front.runs:
            held = front.values[first:first + len(lines) // n]
            out.index_add_(0, lines, by_line(by_front(weights[:, :, :, :start], n, lines) @ held, n))
        return out

    def _read(self, lines: torch.Tensor, first: int, last: int, keys: torch.Tensor, values: torch.Tensor,
              present: torch.Tensor) -> None:
        """Write the rows [first, last) of the lines ``lines`` into ``keys``, ``values`` and ``present``.

        The three are views of the rows of these lines, in order. A row inside the front is read from the front, any
        other row from the line's own rows. ``lines`` is on this past's device. When the output is on the same device,
        `torch.index_select` writes straight into it, with no copy between."""
        start = self.start
        pieces = []
        if first >= last:
            return
        if first < start:
            pieces.append((self.front.keys, self.front.values, self.front.present, self.front.of.index_select(0, lines),
                           first, min(last, start), 0))
        if last > start:
            pieces.append((self.keys, self.values, self.present, lines, max(first, start) - start, last - start,
                           max(start - first, 0)))
        for held_keys, held_values, held_present, index, a, b, at in pieces:
            into = slice(at, at + b - a)
            for held, out in ((held_keys, keys[:, :, into]), (held_values, values[:, :, into])):
                if out.device == held.device:
                    torch.index_select(held[:, :, a:b], 0, index, out=out)
                else:
                    out.copy_(torch.index_select(held[:, :, a:b], 0, index))
            if present.device == held_present.device:
                torch.index_select(held_present[:, a:b], 0, index, out=present[:, into])
            else:
                present[:, into] = torch.index_select(held_present[:, a:b], 0, index)

    @staticmethod
    def together(parts: Sequence[tuple[Past, torch.Tensor, int]], room: int, device: torch.device,
                 attention: str) -> Past:
        """Join the lines of several pasts into one past, in new storage on ``device``.

        ``parts`` is a list of (past, lines, shift). ``lines`` ``[K]`` are the lines taken from the past (repeats are
        allowed; on the past's device). ``shift`` is the number of rows by which the part moves later; below 0 the
        first rows are left out, and none of the part's lines has them present. After the shift, the rows written of
        every part end at one row. A line reads the same as before: its present rows are the same rows at the same
        times (the rotation uses a row's time, not its place), and the rows moved in before them are not present.
        ``room`` is the number of spare rows that each line keeps.

        **The front.** A line taken more than once and its copies are a kin: they are the same up to every row
        written. Lines of one kin in a part, each taken once, stay a kin up to the rows moved. The front has the
        number of rows S that makes the past smallest. A front of S rows is held once for each kin of two lines or more
        that are the same up to S rows or more, and once for each other line. Each line holds its own rows from S on.
        ``attention`` is how the lines read their front (`FRONT_ATTENTIONS`)."""
        if attention not in FRONT_ATTENTIONS:
            raise ValueError(f"attention {attention!r} is none of {FRONT_ATTENTIONS}")
        ends = {past.rows + shift for past, _, shift in parts}
        if len(ends) != 1:
            raise ValueError(f"the parts' rows, moved, end at the rows {sorted(ends)}, not one")
        (rows,) = ends
        # each line's kin (a label unique over the parts) and the rows it is the same as its kin to
        label, same, offset = [], [], 0
        for past, index, shift in parts:
            lines = index.cpu().numpy()
            copies = np.bincount(lines, minlength=len(past.kin))[lines] > 1
            label.append(np.where(copies, offset + lines, offset + len(past.kin) + past.kin[lines]))
            same.append(np.where(copies, rows, np.clip(past.kin_rows[lines] + shift, 0, rows)))
            offset += 2 * len(past.kin)
        kins, kin, members = np.unique(np.concatenate(label), return_inverse=True, return_counts=True)
        kin_rows = np.concatenate(same).astype(np.int64)
        kin_same = np.zeros(len(kins), dtype=np.int64)
        kin_same[kin] = kin_rows
        batch = len(kin)

        def held(s: int) -> int:                         # the rows held with a front of s rows, room aside
            fronts = int(np.where((members > 1) & (kin_same >= s), 1, members).sum())
            return fronts * s + batch * (rows - s)

        start = min(sorted({0, *kin_same[members > 1].tolist()}), key=held)
        like = parts[0][0].keys
        heads, head = like.shape[1], like.shape[3]
        keys = like.new_zeros((batch, heads, rows - start + room, head), device=device)
        values = torch.zeros_like(keys)
        present = torch.zeros((batch, rows - start + room), dtype=torch.bool, device=device)
        part_of = np.concatenate([np.full(len(index), p) for p, (_, index, _) in enumerate(parts)])
        line_of = np.concatenate([index.cpu().numpy() for _, index, _ in parts])
        at = 0
        for past, index, shift in parts:                 # each line's own rows
            lines = slice(at, at + len(index))
            first = max(start, shift)
            past._read(index, first - shift, rows - shift, keys[lines, :, first - start:rows - start],
                       values[lines, :, first - start:rows - start], present[lines, first - start:rows - start])
            at += len(index)
        front = None
        if start:
            # a front for each kin that shares one, and for each other line; the fronts of as many lines in a run
            unit = np.where(((members > 1) & (kin_same >= start))[kin], kin, len(kins) + np.arange(batch))
            _, of, lines_of = np.unique(unit, return_inverse=True, return_counts=True)
            order = np.argsort(-lines_of, kind="stable")              # the fronts by their lines, most first
            place = np.empty_like(order)
            place[order] = np.arange(len(order))
            of = place[of]
            counts = lines_of[order]
            by_front = np.argsort(of, kind="stable")                 # the lines, front after front
            first_line = by_front[np.concatenate(([0], np.cumsum(counts)[:-1]))]
            front_keys = like.new_zeros((len(order), heads, start, head), device=device)
            front_values = torch.zeros_like(front_keys)
            front_present = torch.zeros((len(order), start), dtype=torch.bool, device=device)
            for p, (past, _, shift) in enumerate(parts):
                mine = np.flatnonzero(part_of[first_line] == p)
                if not len(mine):
                    continue
                index = torch.as_tensor(line_of[first_line[mine]], device=past.keys.device)
                into = torch.as_tensor(mine, device=device)
                first = max(0, shift)
                k = like.new_zeros((len(mine), heads, start, head), device=device)
                v, pr = torch.zeros_like(k), torch.zeros((len(mine), start), dtype=torch.bool, device=device)
                past._read(index, first - shift, start - shift, k[:, :, first:], v[:, :, first:], pr[:, first:])
                front_keys.index_copy_(0, into, k)
                front_values.index_copy_(0, into, v)
                front_present.index_copy_(0, into, pr)
            runs, f, at = [], 0, 0
            for n in sorted(set(counts.tolist()), reverse=True):
                fronts = int((counts == n).sum())
                runs.append((f, n, torch.as_tensor(by_front[at:at + fronts * n], device=device)))
                f, at = f + fronts, at + fronts * n
            front = Front(front_keys, front_values, front_present, torch.as_tensor(of, device=device), tuple(runs))
        return Past(keys, values, present, rows, front, kin.astype(np.int64), kin_rows, attention)

    @classmethod
    def nothing(cls, batch: int, heads: int, head: int, capacity: int, like: torch.Tensor) -> Past:
        """A past with no rows and room for ``capacity`` rows. ``like`` gives the device and dtype. Every line is its
        own kin, and there is no front."""
        return cls(like.new_zeros((batch, heads, capacity, head)), like.new_zeros((batch, heads, capacity, head)),
                   torch.zeros((batch, capacity), dtype=torch.bool, device=like.device), 0, None,
                   np.arange(batch, dtype=np.int64), np.zeros(batch, dtype=np.int64), COPIED)


class Layer(nn.Module):
    """Time attention → the added module, if a caller set one (§7 item 5) → feed-forward."""

    def __init__(self, d: int, heads: int, feedforward: int, dropout: float, rope_base: float) -> None:
        super().__init__()
        self.heads, self.rope_base, self.attention_dropout = heads, rope_base, dropout
        self.time_norm = nn.LayerNorm(d)
        self.time_qkv = nn.Linear(d, 3 * d)
        self.time_out = nn.Linear(d, d)
        self.feedforward_norm = nn.LayerNorm(d)
        self.feedforward = nn.Sequential(nn.Linear(d, feedforward), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(feedforward, d))
        self.dropout = nn.Dropout(dropout)
        #: A module a caller adds (§7 item 5): ``added(x, extra)`` → what it adds to the residual stream after the time
        #: attention, ``[B, R, d]``; ``extra`` is what the caller passed to `Prior.encode` / `Prior.extend`.
        self.added: nn.Module | None = None

    def extend(self, x: torch.Tensor, time_s: torch.Tensor, present: torch.Tensor, past: Past, extra: Any
               ) -> tuple[torch.Tensor, Past]:
        """The rows ``x`` [B, R, d] at ``time_s`` [B, R] that follow the ``past`` ones, written into the past in
        place: ``(the rows out, the past with them)``."""
        batch, rows, d = x.shape
        head = d // self.heads
        start, end = past.rows - past.start, past.rows - past.start + rows       # in each line's own rows
        room = past.keys.shape[2]
        if end > room:                               # a line longer than its room: grown by steps
            past = past.grown(room + CACHE_STEP_ROWS * math.ceil((end - room) / CACHE_STEP_ROWS))
        y = self.time_norm(x)
        q, k, v = self.time_qkv(y).reshape(batch, rows, 3, self.heads, head).permute(2, 0, 3, 1, 4)
        cos, sin = rope_angles(time_s, head, self.rope_base)
        cos, sin = cos[:, None], sin[:, None]
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        past.keys[:, :, start:end], past.values[:, :, start:end] = k, v
        past.present[:, start:end] = present
        past = past._replace(rows=past.rows + rows)
        y = past.attend(q, self.attention_dropout if self.training else 0.0)
        x = x + self.dropout(self.time_out(y.transpose(1, 2).reshape(batch, rows, d)))
        if self.added is not None:
            x = x + self.added(x, extra)
        return x + self.dropout(self.feedforward(self.feedforward_norm(x))), past


class CandidatePool(nn.Module):
    """What a row reads of the airport's candidate tokens: one attention, its query the row, over the real candidates."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.query, self.key, self.value, self.out = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``x`` [B, R, d], ``tokens`` [B, R, K, d], ``valid`` [B, K] → [B, R, d]."""
        scores = torch.einsum("brd,brkd->brk", self.query(x), self.key(tokens)) / math.sqrt(x.shape[-1])
        weights = torch.softmax(scores.masked_fill(~valid[:, None, :], float("-inf")), dim=-1)
        return self.out(torch.einsum("brk,brkd->brd", weights, self.value(tokens)))


class Prior(nn.Module):
    def __init__(self, config: PriorConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.state = nn.Linear(len(OWN_FEATURES) + len(COLUMNS) + 1 + 2, d)
        self.in_force = nn.ModuleDict({name: nn.Embedding(1 + config.word_values[WORD_COLUMNS.index(name)], d)
                                       for name in IN_FORCE_WORDS})
        self.candidate = nn.Sequential(nn.Linear(len(variant_features(config.variant)), d), nn.GELU(),
                                       nn.Linear(d, d))
        self.pool = CandidatePool(d)
        self.runway_in_force = nn.Linear(d, d)
        self.no_runway = nn.Parameter(torch.zeros(d))
        self.layers = nn.ModuleList(Layer(d, config.heads, config.feedforward, config.dropout, config.rope_base)
                                    for _ in range(config.layers))
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({name: nn.Linear(d, 1 + count)
                                    for name, count in zip(WORD_COLUMNS, config.word_values)})
        self.runway_fixed = nn.Linear(d, RUNWAY_FIXED_CLASSES)
        self.runway_query = nn.Linear(d, d, bias=False)
        # what a column chose at this row, for the heads after it
        self.chosen = nn.ModuleDict({name: nn.Embedding(1 + count, d)
                                     for name, count in zip(WORD_COLUMNS, config.word_values)})
        self.runway_chosen = nn.Linear(d, d)
        self.runway_fixed_chosen = nn.Embedding(RUNWAY_FIXED_CLASSES, d)

    # ---- the rows
    def encode(self, rows: RowTensors, extra: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(h [B, R, d], tokens [B, R, K, d])`` of every row: `extend` from nothing."""
        h, tokens, _ = self.extend(rows, self.no_past(rows.present.shape[0], rows.present.shape[1]), extra)
        return h, tokens

    def no_past(self, batch: int, capacity: int) -> list[Past]:
        """Every layer's `Past` before the first row, room for ``capacity`` rows."""
        head = self.config.d_model // self.config.heads
        return [Past.nothing(batch, self.config.heads, head, capacity, self.no_runway) for _ in self.layers]

    def extend(self, rows: RowTensors, past: list[Past], extra: Any = None
               ) -> tuple[torch.Tensor, torch.Tensor, list[Past]]:
        """The rows of ``rows`` that follow the ``past`` ones (`no_past` before the first): ``(h, tokens, the past
        with these rows)``."""
        x, tokens = self._inputs(rows)
        after = []
        for layer, before in zip(self.layers, past):
            x, kept = layer.extend(x, rows.time_s, rows.present, before, extra)
            after.append(kept)
        return self.norm(x), tokens, after

    def _inputs(self, rows: RowTensors) -> tuple[torch.Tensor, torch.Tensor]:
        """The first layer's input of the rows and their candidate tokens (module docstring)."""
        tokens = self.candidate(rows.candidates) * rows.valid[:, None, :, None]
        x = self.state(torch.cat((rows.own, rows.since, rows.go_around[..., None].to(rows.own.dtype),
                                  rows.heading_in_force), dim=-1))
        for k, name in enumerate(IN_FORCE_WORDS):
            x = x + self.in_force[name](rows.words_in_force[..., k] + 1)
        said = rows.runway_in_force >= 0
        x = x + torch.where(said[..., None], self.runway_in_force(self._token(tokens, rows.runway_in_force)),
                            self.no_runway)
        return x + self.pool(x, tokens, rows.valid), tokens

    @staticmethod
    def _token(tokens: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """``tokens`` [B, R, K, d] of the candidate ``candidate`` [B, R] at each row (a negative index: slot 0, for
        the caller to replace)."""
        index = candidate.clamp(min=0)[..., None, None].expand(*candidate.shape, 1, tokens.shape[-1])
        return tokens.gather(2, index)[..., 0, :]

    # ---- the heads
    def column_logits(self, column: int, g: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor,
                      first: torch.Tensor) -> torch.Tensor:
        """Column ``column``'s logits ``[B, R, classes]`` from ``g`` (the hidden state plus the earlier columns'
        choices, `after_choice`); ``first`` [B, R]: the first predicted step."""
        if column == RUNWAY:
            scores = torch.einsum("brd,brkd->brk", self.runway_query(g), tokens) / math.sqrt(g.shape[-1])
            scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
            logit = torch.cat((self.runway_fixed(g), scores), dim=-1)
            blocked = (RUNWAY_UNCHANGED_CLASS, RUNWAY_GO_AROUND_CLASS)
        else:
            logit = self.heads[COLUMNS[column]](g)
            blocked = (0,)
        at_first = torch.zeros(logit.shape[-1], dtype=torch.bool, device=g.device)
        at_first[list(blocked)] = True
        return logit.masked_fill(first[..., None] & at_first, float("-inf"))

    def after_choice(self, column: int, g: torch.Tensor, chosen: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """``g`` with the class ``chosen`` [B, R] of column ``column`` added, for the columns after it."""
        if column == RUNWAY:
            candidate = chosen - RUNWAY_FIXED_CLASSES
            fixed = self.runway_fixed_chosen(chosen.clamp(max=RUNWAY_FIXED_CLASSES - 1))
            said = self.runway_chosen(self._token(tokens, candidate))
            return g + torch.where((candidate >= 0)[..., None], said, fixed)
        return g + self.chosen[COLUMNS[column]](chosen)

    def logits(self, h: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor, chosen: torch.Tensor,
               first: torch.Tensor) -> list[torch.Tensor]:
        """The five columns' logits ``[B, R, classes]``, each head reading the classes ``chosen`` [B, R, 5] of the
        columns before it."""
        out, g = [], h
        for column in range(len(COLUMNS)):
            out.append(self.column_logits(column, g, tokens, valid, first))
            g = self.after_choice(column, g, chosen[..., column], tokens)
        return out

    def forward(self, rows: RowTensors, extra: Any = None) -> list[torch.Tensor]:
        """Teacher forcing: the five columns' logits, each head reading the targets of the columns before it."""
        h, tokens = self.encode(rows, extra)
        return self.logits(h, tokens, rows.valid, rows.targets, rows.first)

    # ---- §7 item 5
    def add_at_each_layer(self, make: Callable[[int], nn.Module]) -> None:
        """Set ``make(i)`` as layer i's added module (`Layer.added`); refused where one is set already."""
        for i, layer in enumerate(self.layers):
            if layer.added is not None:
                raise ValueError(f"layer {i} has an added module already")
            layer.added = make(i)


def require_eval(model: nn.Module, who: str) -> None:
    """Refuse ``model`` when any of its modules is in training mode (D107): the speaker speaks with dropout off, and the
    log-probability under its records is that of the policy that spoke only with dropout off (an added module in
    training mode inside an eval model is refused too). ``who``: the caller, named in the refusal."""
    training = [name or "the model" for name, module in model.named_modules() if module.training]
    if training:
        raise ValueError(f"{who}: the prior's modules {training[:3]}{' …' if len(training) > 3 else ''} are in training "
                         f"mode (dropout on); {who} reads the model in eval mode (D107)")
