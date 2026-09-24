"""The prior's network (prior design §3; the second version §8.2 items 3–4): a causal transformer over a flight's
steps, candidate-runway tokens, and six word heads.

**Candidate tokens.** At every step each candidate runway of the airport becomes a vector: one small network, shared
by every candidate, reads its geometry (`data.CANDIDATE_FEATURES`) and where the aircraft is relative to it at the
step (`data.RELATIVE_FEATURES` and the input set's direction block). No candidate carries its slot, so the order of
the candidates changes nothing but the order of the runway scores. The step's input is the sum of: the projected state
and the steps since each column's word; one embedding per column of the word in force (the runway in force is its
candidate's vector of this step, "none yet" a learned vector); the SUM of the airport's candidate vectors (the padded
slots count nothing); the airport's embedding; the step's position. A step attends to itself and the steps before it.

**Heads.** Five linear heads (class 0 = unchanged). The runway head scores each candidate, ``h · W · c_j`` (the
step's hidden state, the candidate's vector of the step), beside an "unchanged" score ``w · h``, softmaxed over the
airport's candidates (padded slots −inf).

**Step 0** (`predicted_entries`): the runway is the one word asked — "unchanged" is not a class there, the sentence
must say one; every column of `data.GIVEN_AT_HANDOVER` is the hand-over's word, given: its distribution there is the
point mass on that word (its input), so its likelihood is 1 and it trains nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from ts_transformer.instructions.words import COLUMNS, RUNWAY
from ts_transformer.prior.data import CANDIDATE_FEATURES, GIVEN_AT_HANDOVER, INPUT_SETS


@dataclass(frozen=True)
class PriorConfig:
    classes: tuple[int, ...]       # per column, "unchanged" + its values
    airports: tuple[str, ...]
    candidate_slots: int
    inputs: str                    # a `data.INPUT_SETS` name
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


def predicted_entries(padding: torch.Tensor) -> torch.Tensor:
    """``[B, T, 6]``: the entries the prior is asked for — every real step's six columns, except at step 0, where
    the columns of `GIVEN_AT_HANDOVER` are given."""
    asked = (~padding)[..., None].repeat(1, 1, len(COLUMNS))
    asked[:, 0, list(GIVEN_AT_HANDOVER)] = False
    return asked


class Prior(nn.Module):
    def __init__(self, config: PriorConfig, candidates: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        inputs = INPUT_SETS[config.inputs]
        d = config.d_model
        self.state = nn.Linear(len(inputs.step_features) + len(COLUMNS), d)
        self.words = nn.ModuleDict({name: nn.Embedding(config.classes[c], d)
                                    for c, name in enumerate(COLUMNS) if c != RUNWAY})
        self.candidate = nn.Sequential(nn.Linear(len(CANDIDATE_FEATURES) - 1 + len(inputs.relative_features), d),
                                       nn.GELU(), nn.Linear(d, d))
        self.pool = nn.Linear(d, d)
        self.runway_in_force = nn.Linear(d, d)
        self.no_runway = nn.Parameter(torch.zeros(d))
        self.airport = nn.Embedding(len(config.airports), d)
        self.position = nn.Embedding(config.max_rows, d)
        layer = nn.TransformerEncoderLayer(d, config.heads, config.feedforward, config.dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({name: nn.Linear(d, config.classes[c])
                                    for c, name in enumerate(COLUMNS) if c != RUNWAY})
        self.runway_query = nn.Linear(d, d, bias=False)
        self.runway_unchanged = nn.Linear(d, 1)
        # the airports' candidate runways, fixed context: [A, slots, CANDIDATE_FEATURES] (the last one: valid)
        self.register_buffer("candidates", candidates, persistent=True)

    def forward(self, features: torch.Tensor, relative: torch.Tensor, in_force: torch.Tensor, since: torch.Tensor,
                airport: torch.Tensor, padding: torch.Tensor) -> list[torch.Tensor]:
        """``features`` [B, T, step features], ``relative`` [B, T, slots, relative features], ``in_force`` [B, T, 6]
        long, ``since`` [B, T, 6], ``airport`` [B] long, ``padding`` [B, T] bool (True past a flight's end) → six
        logits tensors [B, T, classes]."""
        batch, rows, slots = relative.shape[:3]
        table = self.candidates[airport]                                        # [B, slots, features]
        valid = table[..., -1] > 0.5                                            # [B, slots]
        geometry = table[:, None, :, :-1].expand(batch, rows, slots, -1)
        tokens = self.candidate(torch.cat((geometry, relative), dim=-1)) * valid[:, None, :, None]   # [B, T, slots, d]

        x = self.state(torch.cat((features, since), dim=-1))
        for c, name in enumerate(COLUMNS):
            if c != RUNWAY:
                x = x + self.words[name](in_force[..., c])
        pointer = in_force[..., RUNWAY]                                         # 0 = none yet, k = slot k − 1
        index = (pointer - 1).clamp(min=0)[..., None, None].expand(batch, rows, 1, tokens.shape[-1])
        said = tokens.gather(2, index)[:, :, 0]
        x = x + torch.where(pointer[..., None] > 0, self.runway_in_force(said), self.no_runway)
        x = x + self.pool(tokens.sum(dim=2))
        x = x + self.airport(airport)[:, None, :] + self.position(torch.arange(rows, device=x.device))[None]
        causal = torch.triu(torch.ones(rows, rows, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.norm(self.encoder(x, mask=causal, src_key_padding_mask=padding, is_causal=True))

        logits = []
        for c, name in enumerate(COLUMNS):
            if c == RUNWAY:
                scores = torch.einsum("btd,btsd->bts", self.runway_query(h), tokens)
                logit = torch.cat((self.runway_unchanged(h), scores), dim=-1)
                allowed = torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1)[:, None, :].repeat(1, rows, 1)
                allowed[:, 0, 0] = False                                        # step 0 says a runway
                logit = logit.masked_fill(~allowed, float("-inf"))
            else:
                logit = self.heads[name](h)
                if c in GIVEN_AT_HANDOVER:                                      # step 0: the hand-over's word
                    given = torch.full_like(logit[:, :1], float("-inf")).scatter(-1, in_force[:, :1, c, None], 0.0)
                    logit = torch.cat((given, logit[:, 1:]), dim=1)
            logits.append(logit)
        return logits
