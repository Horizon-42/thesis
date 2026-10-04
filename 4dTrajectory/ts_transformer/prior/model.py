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
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, NamedTuple

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


@dataclass(frozen=True)
class PriorConfig:
    """The network's shape. ``word_values``: each of `WORD_COLUMNS`' values, "unchanged" not counted, from the
    vocabulary spec (`from_words`), never a constant of the prior. Configuration A of D40 is the default."""

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
    """``(cos, sin)`` ``[..., T, head / 2]`` of the rotation at the times ``time_s`` [..., T] (seconds), in the
    dtype of ``time_s``; computed in float64."""
    half = head // 2
    frequency = base ** (-torch.arange(half, dtype=torch.float64, device=time_s.device) / half)
    angle = time_s.double()[..., None] * frequency
    return angle.cos().to(time_s.dtype), angle.sin().to(time_s.dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` [..., T, head] rotated pair by pair (the first half of the head against the second)."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class Past(NamedTuple):
    """A layer's rotated time-attention keys and values (``[B, heads, capacity, head]``) and which rows are present
    (``[B, capacity]``), of which the first ``rows`` are encoded.

    Written in place: one `Past` is one line of rows. Two extends from the same `Past` share its storage (a branch
    overwrites its sibling's rows: copy the tensors to branch), and a gradient cannot flow back through an extend
    that a later one wrote over (the speaker extends without gradients; training encodes a sentence whole)."""

    keys: torch.Tensor
    values: torch.Tensor
    present: torch.Tensor
    rows: int

    @classmethod
    def nothing(cls, batch: int, heads: int, head: int, capacity: int, like: torch.Tensor) -> Past:
        """No rows yet, room for ``capacity`` (``like``: a tensor of the device and dtype)."""
        return cls(like.new_zeros((batch, heads, capacity, head)), like.new_zeros((batch, heads, capacity, head)),
                   torch.zeros((batch, capacity), dtype=torch.bool, device=like.device), 0)


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
        start, end = past.rows, past.rows + rows
        if end > past.keys.shape[2]:
            raise ValueError(f"{end} rows, the past holds {past.keys.shape[2]}")
        y = self.time_norm(x)
        q, k, v = self.time_qkv(y).reshape(batch, rows, 3, self.heads, head).permute(2, 0, 3, 1, 4)
        cos, sin = rope_angles(time_s, head, self.rope_base)
        cos, sin = cos[:, None], sin[:, None]
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        past.keys[:, :, start:end], past.values[:, :, start:end] = k, v
        past.present[:, start:end] = present
        past = past._replace(rows=end)
        steps = torch.arange(end, device=x.device)
        queries = steps[start:, None]
        allowed = (steps[None, :] <= queries) & (past.present[:, None, :end] | (steps[None, :] == queries))
        y = functional.scaled_dot_product_attention(q, past.keys[:, :, :end], past.values[:, :, :end],
                                                    attn_mask=allowed[:, None],
                                                    dropout_p=self.attention_dropout if self.training else 0.0)
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
