"""The traffic attention (post-training §3, §8 C5; D29, D98): a module at each layer of the prior (prior §7 item 5,
`Prior.add_at_each_layer`) through which the commanded aircraft's row reads the other aircraft of its step.

**Its input** (`Traffic`, the ``extra`` that the caller gives the model and the speaker, prior §7 items 3–5): for each
aircraft and row, one token for each other aircraft of the step — `post.edges.tokens`: the edge features to the
commanded aircraft and the other aircraft's own motion, from its recorded state only (D98) — padded to the most other
aircraft of the batch, with a mask of the real ones.

A sample without a scene (the single-aircraft samples of the data term, D36) gives the model no traffic (``extra``
None): every row then has no other aircraft.

**The module** (`TrafficAttention`): a token network shared by the tokens, then an attention from the row (its query,
after a layer norm) to the row's own tokens, with ``heads`` heads. Its output layer starts at zero, so the prior with the
module added gives every output of the base, bit for bit (the start of the post-training, D29); and where a row has no
other aircraft the output is zero at any weights, so a window without traffic is free generation (§2 item 1). The
attention is a weighted sum over the tokens, so the order of the other aircraft changes nothing but the summation order.

**Its own learning rate** (§2 item 5): `parameter_groups` puts the modules' parameters in a group of their own.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.post.edges import EDGES_SCHEMA, TOKEN_FEATURES
from ts_transformer.prior.model import Prior

#: The name of this module's shape and input (a post-trained checkpoint's identity holds it, §4 item 3).
TRAFFIC_ATTENTION_SCHEMA = "post-traffic-attention-v1"


@dataclass(frozen=True)
class TrafficConfig:
    """The shape of the module at each layer: the token network's hidden width and the attention's heads (the width of
    the attention is the prior's ``d_model``)."""

    hidden: int
    heads: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema": TRAFFIC_ATTENTION_SCHEMA, "edges_schema": EDGES_SCHEMA, **asdict(self)}


@dataclass(frozen=True)
class Traffic:
    """The input of the modules for a batch of rows: ``tokens`` [B, R, N, len(TOKEN_FEATURES)] float32 and ``present``
    [B, R, N] bool (a real other aircraft; padding is False)."""

    tokens: torch.Tensor
    present: torch.Tensor

    def __post_init__(self) -> None:
        if self.tokens.dim() != 4 or self.tokens.shape[-1] != len(TOKEN_FEATURES):
            raise ValueError(f"traffic tokens are [B, R, N, {len(TOKEN_FEATURES)}], not {tuple(self.tokens.shape)}")
        if self.present.shape != self.tokens.shape[:3] or self.present.dtype != torch.bool:
            raise ValueError("present is a bool [B, R, N] beside the tokens")


def traffic_of(rows: Sequence[Sequence[np.ndarray]], device: torch.device) -> Traffic:
    """`Traffic` from each aircraft's rows of tokens (``rows[b][r]``: ``[N_br, len(TOKEN_FEATURES)]``, `edges.tokens`),
    every aircraft with the same number of rows; padded to the largest N (at least 1, so that a batch without traffic
    has a shape)."""
    counts = {len(r) for r in rows}
    if len(counts) != 1:
        raise ValueError("every aircraft of a batch has the same number of rows")
    width = max([1] + [len(t) for r in rows for t in r])
    tokens = np.zeros((len(rows), counts.pop(), width, len(TOKEN_FEATURES)), dtype=np.float32)
    present = np.zeros(tokens.shape[:3], dtype=bool)
    for b, aircraft in enumerate(rows):
        for r, row in enumerate(aircraft):
            tokens[b, r, :len(row)] = row
            present[b, r, :len(row)] = True
    return Traffic(torch.as_tensor(tokens, device=device), torch.as_tensor(present, device=device))


class TrafficAttention(nn.Module):
    """One layer's module (module docstring): ``forward(x, extra)`` → ``[B, R, d]``."""

    def __init__(self, d: int, config: TrafficConfig) -> None:
        super().__init__()
        if d % config.heads:
            raise ValueError(f"{config.heads} heads do not divide the prior's width {d}")
        self.heads = config.heads
        self.token = nn.Sequential(nn.Linear(len(TOKEN_FEATURES), config.hidden), nn.GELU(),
                                   nn.Linear(config.hidden, d))
        self.norm = nn.LayerNorm(d)
        self.query, self.key, self.value = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, extra: Traffic | None) -> torch.Tensor:
        if extra is None:                   # a sample without a scene (module docstring)
            return torch.zeros_like(x)
        batch, rows, d = x.shape
        if extra.tokens.shape[:2] != (batch, rows):
            raise ValueError(f"traffic of {tuple(extra.tokens.shape[:2])} rows for {(batch, rows)} rows")
        head = d // self.heads
        others = extra.tokens.shape[2]
        embedded = self.token(extra.tokens.to(x.dtype))                                    # [B, R, N, d]
        q = self.query(self.norm(x)).reshape(batch, rows, self.heads, head)
        k = self.key(embedded).reshape(batch, rows, others, self.heads, head)
        v = self.value(embedded).reshape(batch, rows, others, self.heads, head)
        scores = torch.einsum("brhc,brnhc->brhn", q, k) / math.sqrt(head)
        any_other = extra.present.any(dim=-1)                                            # [B, R]
        # a row without other aircraft attends to a padded slot (finite weights), and its output is set to zero below
        mask = extra.present | ~any_other[..., None]
        weights = torch.softmax(scores.masked_fill(~mask[:, :, None, :], float("-inf")), dim=-1)
        read = torch.einsum("brhn,brnhc->brhc", weights, v).reshape(batch, rows, d)
        return torch.where(any_other[..., None], self.out(read), torch.zeros_like(x))


def add_traffic_attention(model: Prior, config: TrafficConfig) -> None:
    """Add a `TrafficAttention` at each layer of ``model`` (prior §7 item 5)."""
    model.add_at_each_layer(lambda _: TrafficAttention(model.config.d_model, config))


def traffic_modules(model: Prior) -> list[TrafficAttention]:
    """The traffic modules of ``model``, one for each layer; refused unless every layer has one."""
    found = [m for m in model.modules() if isinstance(m, TrafficAttention)]
    if len(found) != model.config.layers:
        raise ValueError(f"{len(found)} traffic modules in a prior of {model.config.layers} layers")
    return found


def parameter_groups(model: Prior, prior_lr: float, traffic_lr: float) -> list[dict[str, Any]]:
    """The optimizer's parameter groups (§2 item 5): the prior's own parameters at ``prior_lr``, the traffic modules' at
    ``traffic_lr``; every parameter in exactly one group."""
    traffic: list[nn.Parameter] = [p for m in traffic_modules(model) for p in m.parameters()]
    ids = {id(p) for p in traffic}
    own: Iterable[nn.Parameter] = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": list(own), "lr": prior_lr, "name": "prior"},
            {"params": traffic, "lr": traffic_lr, "name": "traffic"}]
