"""Version 2's module (multi-aircraft control D153, D212; §6.4): a commanded aircraft's row reads the rows of the other
commanded aircraft of its window at the same tick — what the model is about to tell them.

**Where it is.** Beside the traffic attention of each layer (post-training §9 item 7, `traffic_attention.add_beside`):
it gets the row as the traffic module gets it (after the layer's time attention) and the same input, and its output is
added to the traffic module's. Its parameters are in the traffic group of `traffic_attention.parameter_groups` (D212
(4)).

**What a row reads.** Its input is the window input of the traffic module's input (``Traffic.windows``: the rows of
each window; ``Traffic.key``: the rows that the other rows of their window read). At each tick a row's query (after a
layer norm) reads, with ``heads`` heads, the keys and values of the other rows of its window that are keys at that tick:
never itself, never a row of another window. A row that is in no window, or whose window has no other key at the tick,
adds zero, so a window of one commanded aircraft is version 1's. The keys are the rows of commanded aircraft that have
joined and have not ended (D212 (1)): the window loop gives them at each tick (`WindowLoop(…, joint=True)`), a window
sample gives the rows of its sentences (`post.branches.samples`). A row at tick t is computed from the inputs at the
start of row t and holds no word of row t, so no row reads a word said in its own row (D148).

**At the start** its output layer is zero, so the model says what it says without the module, bit for bit. The order
of a window's rows changes only the order of a sum.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from ts_transformer.post.traffic_attention import Traffic, add_beside
from ts_transformer.prior.model import Prior

#: The name of this module's shape and input (a checkpoint of version 2 holds it, §7 row 3).
JOINT_SCHEMA = "multi-joint-attention-v1"


@dataclass(frozen=True)
class JointConfig:
    """The shape of the module at each layer: its heads (its width is the prior's ``d_model``)."""

    heads: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema": JOINT_SCHEMA, **asdict(self)}


class JointAttention(nn.Module):
    """One layer's module (module docstring): ``forward(x, extra)`` → ``[B, R, d]``."""

    def __init__(self, d: int, config: JointConfig) -> None:
        super().__init__()
        if config.heads <= 0 or d % config.heads:
            raise ValueError(f"{config.heads} heads do not divide the prior's width {d}")
        self.heads = config.heads
        self.norm = nn.LayerNorm(d)
        self.query, self.key, self.value = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, extra: Traffic | None) -> torch.Tensor:
        if extra is None or extra.windows is None:          # no window: no other commanded aircraft
            return torch.zeros_like(x)
        _, rows, d = x.shape
        windows = extra.windows                                               # [G, M]
        count, most = windows.shape
        head = d // self.heads
        real = windows >= 0
        index = windows.clamp(min=0)
        y = self.norm(x)

        def grouped(z: torch.Tensor) -> torch.Tensor:        # [B, R, d] → [G, R, heads, M, head]
            return z[index].reshape(count, most, rows, self.heads, head).permute(0, 2, 3, 1, 4)

        q, k, v = grouped(self.query(y)), grouped(self.key(y)), grouped(self.value(y))
        keys = (extra.key[index] & real[..., None]).transpose(1, 2)                                  # [G, R, M]
        others = ~torch.eye(most, dtype=torch.bool, device=x.device)
        allowed = keys[:, :, None, :] & others                                                       # [G, R, M, M]
        reads = allowed.any(dim=-1)                                                                  # [G, R, M]
        # a row without a key attends to every slot (finite weights); its output is set to zero below
        mask = (allowed | ~reads[..., None])[:, :, None]
        scores = q @ k.transpose(-1, -2) / math.sqrt(head)                                           # [G, R, heads, M, M]
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        read = (weights @ v).permute(0, 3, 1, 2, 4).reshape(count, most, rows, d)                     # [G, M, R, d]
        out = torch.where(reads.transpose(1, 2)[..., None], self.out(read), torch.zeros_like(read))
        return torch.zeros_like(x).index_copy(0, windows[real], out[real])


def add_joint_attention(model: Prior, config: JointConfig) -> None:
    """Put a `JointAttention` beside the traffic attention of each layer of ``model`` (module docstring)."""
    add_beside(model, lambda i: JointAttention(model.config.d_model, config))


def joint_modules(model: Prior) -> list[JointAttention]:
    """The modules of ``model``, one for each layer; refused unless every layer has one."""
    found = [m for m in model.modules() if isinstance(m, JointAttention)]
    if len(found) != model.config.layers:
        raise ValueError(f"{len(found)} joint modules in a prior of {model.config.layers} layers")
    return found
