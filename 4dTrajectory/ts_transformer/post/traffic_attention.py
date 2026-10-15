"""The traffic attention (post-training §3, §8 C5; D29, D98): a module at each layer of the prior (prior §7 item 5,
`Prior.add_at_each_layer`) through which the commanded aircraft's row reads the other aircraft of its step.

**Its input** (`Traffic`, the ``extra`` that the caller gives the model and the speaker, prior §7 items 3–5): for each
aircraft and row, one token for each other aircraft of the step — `post.edges.tokens`: the edge features to the
commanded aircraft and the other aircraft's own motion, from its recorded state only (D98) — padded to the most other
aircraft of the batch, with a mask of the real ones.

A sample without a scene (the single-aircraft samples of the data term, D36) gives the model no traffic (``extra``
None): every row then has no other aircraft.

**The modules** (`TrafficAttention`, one at each layer): one token network shared by the layers (`TrafficTokens`,
D116) — the tokens of a step are embedded once in a forward pass, by the first layer's module, which keeps them on the
input for the layers after it — then, in each layer, its own attention from the row (its query, after a layer norm) to
the row's own tokens, with ``heads`` heads. Its output layer starts at zero, so the prior with the
module added gives every output of the base, bit for bit (the start of the post-training, D29); and where a row has no
other aircraft the output is zero at any weights, so a window without traffic is free generation (§2 item 1). The
attention is a weighted sum over the tokens, so the order of the other aircraft changes nothing but the summation order.

**Its own learning rate** (§2 item 5): `parameter_groups` puts the modules' parameters in a group of their own.

**A token part that a caller adds** (multi-aircraft control D149, D152; post-training §9 item 7): its own inputs of each
token beside the edge features (``Traffic.part``, ``[B, R, N, width]``; `traffic_of`'s rows then hold them after the
edge features), through its own projection (`TokenPart`, added to the shared token network by `add_token_part`, under
its own format name) whose weights start at zero, so the start says what the module says without it. Without a part,
the module, its tokens and its shape are as before, bit for bit: a checkpoint of stage C loads as it did.

**A module beside it, and its window input** (multi-aircraft control D153, D212; post-training §9 item 7). A caller may
put a module of its own beside each layer's module (`add_beside`, ``TrafficAttention.beside``): it gets the same row
and input, and its output is added to the module's; its parameters are in the traffic group of `parameter_groups`. Its
input travels with the module's: ``Traffic.windows``, the rows of the batch of each window of commanded aircraft (rows
of one window are on the same ticks), and ``Traffic.key``, the rows that the other rows of their window read at the same
tick. The traffic module reads neither. Without them and without a module beside, the module is as before, bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.post.edges import EDGES_SCHEMA, TOKEN_FEATURES
from ts_transformer.prior.model import Prior

#: The name of this module's shape and input (a post-trained checkpoint's identity holds it, §4 item 3). v2 (D116): one
#: token network shared by the layers.
TRAFFIC_ATTENTION_SCHEMA = "post-traffic-attention-v2"


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
    #: a caller's token part (module docstring): ``[B, R, N, width]`` float32 beside the tokens, or None
    part: torch.Tensor | None = None
    #: a window input (module docstring): ``[G, M]`` int64, the rows of each window, −1 after its last; no row in two
    #: windows. None: no window
    windows: torch.Tensor | None = None
    #: ``[B, R]`` bool beside ``windows``: the rows that the other rows of their window read at the same tick
    key: torch.Tensor | None = None
    #: the tokens embedded by the shared token network in the present forward pass (the first layer's module sets it)
    embedded: torch.Tensor | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.tokens.dim() != 4 or self.tokens.shape[-1] != len(TOKEN_FEATURES):
            raise ValueError(f"traffic tokens are [B, R, N, {len(TOKEN_FEATURES)}], not {tuple(self.tokens.shape)}")
        if self.present.shape != self.tokens.shape[:3] or self.present.dtype != torch.bool:
            raise ValueError("present is a bool [B, R, N] beside the tokens")
        if self.part is not None and (self.part.dim() != 4 or self.part.shape[:3] != self.tokens.shape[:3]):
            raise ValueError(f"a token part is [B, R, N, width] beside the tokens, not {tuple(self.part.shape)}")
        if (self.windows is None) != (self.key is None):
            raise ValueError("a window input is the windows and the key rows together")
        if self.windows is not None:
            rows = self.windows[self.windows >= 0]
            if (self.windows.dim() != 2 or self.windows.dtype != torch.int64 or self.key.dtype != torch.bool
                    or self.key.shape != self.tokens.shape[:2] or len(rows.unique()) != len(rows)
                    or bool((rows >= self.tokens.shape[0]).any())):
                raise ValueError("windows are [G, M] int64 rows of the batch (−1 after a window's last), each row in "
                                 "one window at most; the key rows are [B, R] bool beside the tokens")

    def select(self, indices: torch.Tensor) -> Traffic:
        """The rows of the samples ``indices`` (the first axis), the part with them; with a window input, whole windows
        only (a selection that splits a window is refused), each window's rows in their new places."""
        if self.windows is None:
            return Traffic(self.tokens[indices], self.present[indices],
                           None if self.part is None else self.part[indices])
        chosen = [int(i) for i in indices]
        place = {row: k for k, row in enumerate(chosen)}
        kept = []
        for window in self.windows.tolist():
            rows = [row for row in window if row >= 0]
            inside = [row in place for row in rows]
            if any(inside) and not all(inside):
                raise ValueError("a selection of samples with a window input keeps whole windows")
            if all(inside) and rows:
                kept.append([place[row] for row in rows])
        width = max([1] + [len(rows) for rows in kept])
        windows = torch.full((len(kept), width), -1, dtype=torch.int64, device=self.windows.device)
        for g, rows in enumerate(kept):
            windows[g, :len(rows)] = torch.tensor(rows, dtype=torch.int64)
        return Traffic(self.tokens[indices], self.present[indices], None if self.part is None else self.part[indices],
                       windows, self.key[indices])


def traffic_of(rows: Sequence[Sequence[np.ndarray]], device: torch.device, part_width: int = 0,
               windows: Sequence[Sequence[int]] | None = None, key: np.ndarray | torch.Tensor | None = None) -> Traffic:
    """`Traffic` from each aircraft's rows of tokens (``rows[b][r]``: ``[N_br, len(TOKEN_FEATURES) + part_width]``, the
    edge features (`edges.tokens`) and then a caller's token part, module docstring), every aircraft with the same
    number of rows; padded to the largest N (at least 1, so that a batch without traffic has a shape). Without a part
    (``part_width`` 0), ``Traffic.part`` is None. ``windows`` and ``key``: a window input (module docstring), each
    window's rows and the ``[B, R]`` key rows; None: none."""
    counts = {len(r) for r in rows}
    if len(counts) != 1:
        raise ValueError("every aircraft of a batch has the same number of rows")
    features = len(TOKEN_FEATURES)
    if any(t.shape[1] != features + part_width for r in rows for t in r if len(t)):
        raise ValueError(f"a token has {features} edge features and a part of {part_width}")
    width = max([1] + [len(t) for r in rows for t in r])
    tokens = np.zeros((len(rows), counts.pop(), width, features + part_width), dtype=np.float32)
    present = np.zeros(tokens.shape[:3], dtype=bool)
    for b, aircraft in enumerate(rows):
        for r, row in enumerate(aircraft):
            tokens[b, r, :len(row)] = row
            present[b, r, :len(row)] = True
    if (windows is None) != (key is None):
        raise ValueError("a window input is the windows and the key rows together")
    grouped = keys = None
    if windows is not None:
        grouped = torch.full((len(windows), max([1] + [len(w) for w in windows])), -1, dtype=torch.int64)
        for g, window in enumerate(windows):
            grouped[g, :len(window)] = torch.as_tensor(list(window), dtype=torch.int64)
        grouped, keys = grouped.to(device), torch.as_tensor(key, device=device)
    if not part_width:
        return Traffic(torch.as_tensor(tokens, device=device), torch.as_tensor(present, device=device), None, grouped,
                       keys)
    return Traffic(torch.as_tensor(tokens[..., :features], device=device), torch.as_tensor(present, device=device),
                   torch.as_tensor(np.ascontiguousarray(tokens[..., features:]), device=device), grouped, keys)


class TokenPart(nn.Module):
    """A caller's token part (module docstring): its inputs → ``[d]``, a projection whose weights start at zero, under
    the caller's format name ``schema``."""

    def __init__(self, d: int, width: int, schema: str) -> None:
        super().__init__()
        self.schema, self.width = schema, width
        self.project = nn.Linear(width, d)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, part: torch.Tensor) -> torch.Tensor:
        return self.project(part)


class TrafficTokens(nn.Module):
    """The token network that the layers share (D116): a token's features → ``[d]``; with a caller's token part
    (`add_token_part`), its projection added."""

    def __init__(self, d: int, config: TrafficConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(TOKEN_FEATURES), config.hidden), nn.GELU(), nn.Linear(config.hidden, d))
        self.part: TokenPart | None = None

    def forward(self, tokens: torch.Tensor, part: torch.Tensor | None = None) -> torch.Tensor:
        if (part is None) != (self.part is None):
            raise ValueError("a token part is given exactly when the module has one (`add_token_part`)")
        if self.part is None:
            return self.net(tokens)
        return self.net(tokens) + self.part(part.to(tokens.dtype))


class TrafficAttention(nn.Module):
    """One layer's module (module docstring): ``forward(x, extra)`` → ``[B, R, d]``. ``first``: the first layer's,
    which embeds the step's tokens with the shared ``tokens`` network for every layer of the forward pass."""

    def __init__(self, d: int, config: TrafficConfig, tokens: TrafficTokens, *, first: bool) -> None:
        super().__init__()
        if d % config.heads:
            raise ValueError(f"{config.heads} heads do not divide the prior's width {d}")
        self.heads, self.first = config.heads, first
        self.tokens = tokens
        self.norm = nn.LayerNorm(d)
        self.query, self.key, self.value = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        #: a caller's module beside this one (module docstring, `add_beside`): ``beside(x, extra)`` → ``[B, R, d]``,
        #: added to this module's output; None: none
        self.beside: nn.Module | None = None

    def forward(self, x: torch.Tensor, extra: Traffic | None) -> torch.Tensor:
        own = self._own(x, extra)
        return own if self.beside is None else own + self.beside(x, extra)

    def _own(self, x: torch.Tensor, extra: Traffic | None) -> torch.Tensor:
        """This module's own output (module docstring)."""
        if extra is None:                   # a sample without a scene (module docstring)
            return torch.zeros_like(x)
        batch, rows, d = x.shape
        if extra.tokens.shape[:2] != (batch, rows):
            raise ValueError(f"traffic of {tuple(extra.tokens.shape[:2])} rows for {(batch, rows)} rows")
        head = d // self.heads
        others = extra.tokens.shape[2]
        if self.first:                      # once a forward pass: the layers after it read what it embedded
            object.__setattr__(extra, "embedded", self.tokens(extra.tokens.to(x.dtype), extra.part))   # [B, R, N, d]
        embedded = extra.embedded
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
    """Add a `TrafficAttention` at each layer of ``model`` (prior §7 item 5), all of them on one `TrafficTokens`."""
    tokens = TrafficTokens(model.config.d_model, config)
    model.add_at_each_layer(lambda i: TrafficAttention(model.config.d_model, config, tokens, first=i == 0))


def add_token_part(model: Prior, width: int, schema: str) -> TokenPart:
    """Add a caller's token part (module docstring) of ``width`` inputs under the format name ``schema`` to ``model``'s
    traffic attention (one, shared by the layers); refused when it has one."""
    tokens = traffic_modules(model)[0].tokens
    if tokens.part is not None:
        raise ValueError(f"the traffic attention has a token part already ({tokens.part.schema})")
    tokens.part = TokenPart(model.config.d_model, width, schema).to(next(tokens.parameters()).device)
    return tokens.part


def add_beside(model: Prior, make: Callable[[int], nn.Module]) -> None:
    """Put a caller's module ``make(i)`` beside the traffic attention of each layer i of ``model`` (module docstring),
    on that module's device; refused when a layer has one."""
    for i, module in enumerate(traffic_modules(model)):
        if module.beside is not None:
            raise ValueError(f"layer {i}'s traffic attention has a module beside it already")
        module.beside = make(i).to(module.out.weight.device)


def traffic_modules(model: Prior) -> list[TrafficAttention]:
    """The traffic modules of ``model``, one for each layer; refused unless every layer has one."""
    found = [m for m in model.modules() if isinstance(m, TrafficAttention)]
    if len(found) != model.config.layers:
        raise ValueError(f"{len(found)} traffic modules in a prior of {model.config.layers} layers")
    return found


def parameter_groups(model: Prior, prior_lr: float, traffic_lr: float) -> list[dict[str, Any]]:
    """The optimizer's parameter groups (§2 item 5): the prior's own parameters at ``prior_lr``, the traffic modules' at
    ``traffic_lr``; every parameter in exactly one group."""
    traffic: list[nn.Parameter] = list({id(p): p for m in traffic_modules(model) for p in m.parameters()}.values())
    ids = {id(p) for p in traffic}
    own: Iterable[nn.Parameter] = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": list(own), "lr": prior_lr, "name": "prior"},
            {"params": traffic, "lr": traffic_lr, "name": "traffic"}]
