"""The prior's network (prior design §3): a causal transformer over a flight's steps and six word heads.

Each step's input is the sum of: the projected state and runway features and the steps since each column's word,
one embedding per column of the word in force, the airport's embedding, the projected candidate runways, and the
step's position. A step attends to itself and the steps before it only. Each head gives the logits of its column
(class 0 = unchanged); the runway head's candidate slots an airport does not have are masked out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from ts_transformer.prior.data import CANDIDATE_FEATURES, N_FEATURES
from ts_transformer.instructions.words import RUNWAY


@dataclass(frozen=True)
class PriorConfig:
    classes: tuple[int, ...]       # per column, "unchanged" + its values
    airports: tuple[str, ...]
    candidate_slots: int
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


class Prior(nn.Module):
    def __init__(self, config: PriorConfig, candidates: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.state = nn.Linear(N_FEATURES + len(config.classes), d)
        self.words = nn.ModuleList(nn.Embedding(classes, d) for classes in config.classes)
        self.airport = nn.Embedding(len(config.airports), d)
        self.candidate = nn.Linear(config.candidate_slots * len(CANDIDATE_FEATURES), d)
        self.position = nn.Embedding(config.max_rows, d)
        layer = nn.TransformerEncoderLayer(d, config.heads, config.feedforward, config.dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList(nn.Linear(d, classes) for classes in config.classes)
        # the airports' candidate runways, fixed context: [A, slots, features]
        self.register_buffer("candidates", candidates, persistent=True)

    def forward(self, features: torch.Tensor, in_force: torch.Tensor, since: torch.Tensor, airport: torch.Tensor,
                padding: torch.Tensor) -> list[torch.Tensor]:
        """``features`` [B, T, N_FEATURES], ``in_force`` [B, T, 6] long, ``since`` [B, T, 6], ``airport`` [B] long,
        ``padding`` [B, T] bool (True past a flight's end) → six logits tensors [B, T, classes]."""
        batch, rows = in_force.shape[:2]
        x = self.state(torch.cat((features, since), dim=-1))
        for column, embedding in enumerate(self.words):
            x = x + embedding(in_force[..., column])
        context = self.airport(airport) + self.candidate(self.candidates[airport].flatten(1))
        x = x + context[:, None, :] + self.position(torch.arange(rows, device=x.device))[None]
        causal = torch.triu(torch.ones(rows, rows, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.norm(self.encoder(x, mask=causal, src_key_padding_mask=padding, is_causal=True))
        logits = [head(h) for head in self.heads]
        valid = self.candidates[airport][..., -1] > 0.5                       # [B, slots]
        runway_mask = torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1)  # class 0 (unchanged) always valid
        logits[RUNWAY] = logits[RUNWAY].masked_fill(~runway_mask[:, None, :], float("-inf"))
        return logits
