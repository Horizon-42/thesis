"""The prior's network (prior design §5–§6): a scene of aircraft over steps, candidate-runway tokens, and six word
heads said in order.

**Shapes.** A batch holds scenes: ``[B, A, T]`` — scene, aircraft, step. Design §9 step 1 trains single-aircraft
scenes (``A = 1``); the layers are the scene model's all the same.

**Candidate tokens.** At every step each candidate runway of the airport becomes a vector: one small network, shared
by every candidate, reads its geometry (`data.CANDIDATE_FEATURES`) and the aircraft's relation to it at the step (the
variant's `data.Variant.relative_features`). No candidate carries its slot, so the order of the candidates changes
nothing but the order of the runway scores. An aircraft-step's input is the sum of: the projected state and the steps
since each column's word; one embedding per column of the word in force (the runway in force is its candidate's vector
of this step, "none yet" a learned vector); the SUM of the airport's candidate vectors (padded slots count nothing);
the projected static attributes (none in this version, `data.STATIC_FEATURES`); the airport's embedding; the step's
position.

**Layers** (§6): causal attention along each aircraft's steps → attention among the aircraft present at the same step
→ feed-forward, each pre-normed and residual. Along time a step reads its aircraft's present steps up to itself, and
always itself, so no row has every key masked (a fully masked row is NaN in some attention kernels, and a NaN value
times a zero weight is still NaN); an absent step's time-attention output is zero. Among aircraft the edge features
(``edges [B, T, A, A, E]``) enter twice: a small network turns them into one bias per head (whom to listen to), another adds them to the value read
(where that aircraft is) — attention on a fully connected graph. Every aircraft always attends to itself, so a step
with one aircraft present reads only its own value. Edge feature 0 is "this is the aircraft itself".

**Heads** (§5). Six columns in `COLUMNS` order. With ordered heads, column k's head reads the hidden state plus the
embeddings of the classes the earlier columns chose at this step (the truth in teacher forcing), so the words said
together fit each other; unordered, every head reads the hidden state alone. The runway head scores each candidate,
``g · W · c_j`` (its input, the candidate's vector of the step), beside an "unchanged" score, softmaxed over the
airport's candidates (padded slots −inf). **The first predicted step** (row `scene.N_LOOK`) says every column: its
"unchanged" is masked. Rows before it are never asked.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional

from ts_transformer.instructions.words import COLUMNS, RUNWAY
from ts_transformer.prior.data import CANDIDATE_FEATURES, STATIC_FEATURES, STEP_FEATURES, VARIANTS
from ts_transformer.prior.scene import N_LOOK

#: Edge features among aircraft: [is the aircraft itself]. Design §4.4's relative quantities join from step 3 on.
EDGE_FEATURES = ("self",)


@dataclass(frozen=True)
class PriorConfig:
    classes: tuple[int, ...]       # per column, "unchanged" + its values
    airports: tuple[str, ...]
    candidate_slots: int
    variant: str                   # a `data.VARIANTS` name
    d_model: int = 192
    layers: int = 4
    heads: int = 6
    feedforward: int = 768
    dropout: float = 0.1
    max_rows: int = 2048

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PriorConfig:
        return cls(**{**data, "classes": tuple(data["classes"]), "airports": tuple(data["airports"])})


def asked_entries(present: torch.Tensor) -> torch.Tensor:
    """``[B, A, T]``: the aircraft-steps the prior speaks at — present, from the first predicted step on."""
    rows = torch.arange(present.shape[-1], device=present.device)
    return present & (rows >= N_LOOK)


def first_step_rows(rows: int, device: torch.device) -> torch.Tensor:
    """``[rows]`` bool: the row that is the first predicted step (`N_LOOK`), of rows 0 … rows − 1."""
    return torch.arange(rows, device=device) == N_LOOK


def self_edges(batch: int, aircraft: int, rows: int, device: torch.device) -> torch.Tensor:
    """``[B, T, A, A, len(EDGE_FEATURES)]`` of scenes with no relation but each aircraft's to itself."""
    eye = torch.eye(aircraft, device=device)[None, None, :, :, None]
    return eye.expand(batch, rows, aircraft, aircraft, 1).contiguous()


class SceneLayer(nn.Module):
    """Time attention → aircraft attention (edge bias + edge value) → feed-forward."""

    def __init__(self, d: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.attention_dropout = dropout
        self.time_norm = nn.LayerNorm(d)
        self.time_qkv = nn.Linear(d, 3 * d)
        self.time_out = nn.Linear(d, d)
        self.aircraft_norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.edge_bias = nn.Sequential(nn.Linear(len(EDGE_FEATURES), d), nn.GELU(), nn.Linear(d, heads))
        self.edge_value = nn.Sequential(nn.Linear(len(EDGE_FEATURES), d), nn.GELU(), nn.Linear(d, d))
        self.feedforward_norm = nn.LayerNorm(d)
        self.feedforward = nn.Sequential(nn.Linear(d, feedforward), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(feedforward, d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, present: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """``x`` [B, A, T, d], ``present`` [B, A, T] bool, ``edges`` [B, T, A, A, E]."""
        batch, aircraft, rows, d = x.shape
        head = d // self.heads
        # along time: each aircraft's present steps up to this one, and always this one
        y = self.time_norm(x).reshape(batch * aircraft, rows, d)
        q, k, v = self.time_qkv(y).reshape(batch * aircraft, rows, 3, self.heads, head).permute(2, 0, 3, 1, 4)
        steps = torch.arange(rows, device=x.device)
        keys = present.reshape(batch * aircraft, 1, rows) | (steps[:, None] == steps[None, :])
        allowed = (steps[None, :] <= steps[:, None]) & keys                     # [B·A, T, T]: query, key
        y = functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None],
                                                    dropout_p=self.attention_dropout if self.training else 0.0)
        y = self.time_out(y.transpose(1, 2).reshape(batch, aircraft, rows, d))
        x = x + self.dropout(y.masked_fill(~present[..., None], 0.0))
        # among aircraft at one step: [B, T, A, ...]
        y = self.aircraft_norm(x).transpose(1, 2)
        q, k, v = self.qkv(y).reshape(batch, rows, aircraft, 3, self.heads, head).unbind(dim=3)
        scores = torch.einsum("btihc,btjhc->bthij", q, k) / math.sqrt(head)
        scores = scores + self.edge_bias(edges).permute(0, 1, 4, 2, 3)
        others = present.transpose(1, 2)[:, :, None, None, :]                       # [B, T, 1, 1, A]: key present
        itself = torch.eye(aircraft, dtype=torch.bool, device=x.device)[None, None, None]
        weights = torch.softmax(scores.masked_fill(~(others | itself), float("-inf")), dim=-1)
        weights = self.dropout(weights)
        read = torch.einsum("bthij,btjhc->btihc", weights, v)
        read = read + torch.einsum("bthij,btijhc->btihc", weights,
                                   self.edge_value(edges).reshape(batch, rows, aircraft, aircraft, self.heads, head))
        x = x + self.dropout(self.out(read.reshape(batch, rows, aircraft, d)).transpose(1, 2))
        return x + self.dropout(self.feedforward(self.feedforward_norm(x)))


class Prior(nn.Module):
    def __init__(self, config: PriorConfig, candidates: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        variant = VARIANTS[config.variant]
        self.ordered = variant.ordered_heads
        d = config.d_model
        self.state = nn.Linear(len(STEP_FEATURES) + len(COLUMNS), d)
        self.static = nn.Linear(len(STATIC_FEATURES), d, bias=False) if STATIC_FEATURES else None
        self.words = nn.ModuleDict({name: nn.Embedding(config.classes[c], d)
                                    for c, name in enumerate(COLUMNS) if c != RUNWAY})
        self.candidate = nn.Sequential(nn.Linear(len(CANDIDATE_FEATURES) - 1 + len(variant.relative_features), d),
                                       nn.GELU(), nn.Linear(d, d))
        self.pool = nn.Linear(d, d)
        self.runway_in_force = nn.Linear(d, d)
        self.no_runway = nn.Parameter(torch.zeros(d))
        self.airport = nn.Embedding(len(config.airports), d)
        self.position = nn.Embedding(config.max_rows, d)
        self.layers = nn.ModuleList(SceneLayer(d, config.heads, config.feedforward, config.dropout)
                                    for _ in range(config.layers))
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({name: nn.Linear(d, config.classes[c])
                                    for c, name in enumerate(COLUMNS) if c != RUNWAY})
        self.runway_query = nn.Linear(d, d, bias=False)
        self.runway_unchanged = nn.Linear(d, 1)
        # what a column chose at this step, for the heads after it (ordered heads)
        self.chosen = nn.ModuleDict({name: nn.Embedding(config.classes[c], d)
                                     for c, name in enumerate(COLUMNS) if c != RUNWAY})
        self.runway_chosen = nn.Linear(d, d)
        self.runway_kept = nn.Parameter(torch.zeros(d))
        # the airports' candidate runways, fixed context: [A, slots, CANDIDATE_FEATURES] (the last one: valid)
        self.register_buffer("candidates", candidates, persistent=True)

    def encode(self, features: torch.Tensor, relative: torch.Tensor, static: torch.Tensor, in_force: torch.Tensor,
               since: torch.Tensor, airport: torch.Tensor, present: torch.Tensor, edges: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(h [B, A, T, d], tokens [B, A, T, slots, d], valid [B, slots])``."""
        batch, aircraft, rows, slots = relative.shape[:4]
        table = self.candidates[airport]                                          # [B, slots, features]
        valid = table[..., -1] > 0.5                                              # [B, slots]
        geometry = table[:, None, None, :, :-1].expand(batch, aircraft, rows, slots, -1)
        tokens = self.candidate(torch.cat((geometry, relative), dim=-1)) * valid[:, None, None, :, None]

        x = self.state(torch.cat((features, since), dim=-1))
        for c, name in enumerate(COLUMNS):
            if c != RUNWAY:
                x = x + self.words[name](in_force[..., c])
        x = x + self._runway(in_force[..., RUNWAY], tokens, self.runway_in_force, self.no_runway)
        x = x + self.pool(tokens.sum(dim=3))
        if self.static is not None:
            x = x + self.static(static)[:, :, None, :]
        x = x + self.airport(airport)[:, None, None, :] + self.position(torch.arange(rows, device=x.device))
        for layer in self.layers:
            x = layer(x, present, edges)
        return self.norm(x), tokens, valid

    @staticmethod
    def _runway(pointer: torch.Tensor, tokens: torch.Tensor, project: nn.Module, none: torch.Tensor) -> torch.Tensor:
        """The runway class ``pointer`` (0 = none / unchanged, k = slot k − 1) as a vector: its candidate's vector of
        the step, projected, or ``none``."""
        index = (pointer - 1).clamp(min=0)[..., None, None].expand(*pointer.shape, 1, tokens.shape[-1])
        said = tokens.gather(3, index)[..., 0, :]
        return torch.where(pointer[..., None] > 0, project(said), none)

    def logits(self, h: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor, chosen: torch.Tensor,
               first: torch.Tensor) -> list[torch.Tensor]:
        """Six logits tensors ``[B, A, T, classes]`` of the rows of ``h`` (``[B, A, T, d]``); ``chosen`` [B, A, T, 6]
        are the classes the columns choose at each step (the truth in teacher forcing, the ones already sampled when
        the prior speaks), read by the later columns' heads when the heads are ordered; ``first`` [T] bool: which rows
        are the first predicted step (`first_step_rows`), where "unchanged" is masked."""
        out, g = [], h
        for c, name in enumerate(COLUMNS):
            if c == RUNWAY:
                scores = torch.einsum("batd,batsd->bats", self.runway_query(g), tokens)
                logit = torch.cat((self.runway_unchanged(g), scores), dim=-1)
                allowed = torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1)[:, None, None, :]
                logit = logit.masked_fill(~allowed, float("-inf"))
            else:
                logit = self.heads[name](g)
            logit = logit.masked_fill(first[:, None] & (torch.arange(logit.shape[-1], device=h.device) == 0),
                                      float("-inf"))
            out.append(logit)
            if self.ordered:
                if c == RUNWAY:
                    g = g + self._runway(chosen[..., RUNWAY], tokens, self.runway_chosen, self.runway_kept)
                else:
                    g = g + self.chosen[name](chosen[..., c])
        return out

    def forward(self, features: torch.Tensor, relative: torch.Tensor, static: torch.Tensor, in_force: torch.Tensor,
                since: torch.Tensor, airport: torch.Tensor, present: torch.Tensor, edges: torch.Tensor,
                chosen: torch.Tensor) -> list[torch.Tensor]:
        """``features`` [B, A, T, step], ``relative`` [B, A, T, slots, relative], ``static`` [B, A, static],
        ``in_force`` / ``chosen`` [B, A, T, 6] long, ``since`` [B, A, T, 6], ``airport`` [B] long, ``present``
        [B, A, T] bool, ``edges`` [B, T, A, A, E] → six logits tensors [B, A, T, classes]."""
        h, tokens, valid = self.encode(features, relative, static, in_force, since, airport, present, edges)
        return self.logits(h, tokens, valid, chosen, first_step_rows(h.shape[2], h.device))
