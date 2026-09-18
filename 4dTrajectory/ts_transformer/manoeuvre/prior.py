"""The prior (plan §2.5): a causal transformer over one flight's code sequence that says which
code comes next, whether the flight has landed instead, and — when it has — when inside the
last segment.

    [TYPE] [RWY]  [BOS, x_0] [c_1, x_1] … [c_T, x_T]   →   at position t: p(c_{t+1}), p(landed after t), fraction

Tokens: the code embedding (K learned rows + BOS) + the state token (`sequences.state_token`,
six features through a linear layer) + the position (the segment index). The two context
tokens sit in front; the causal mask lets every sequence position read them and its past.
Not the iTransformer / PatchTST backbone: those are channel-series models, this is a token
sequence model (plan §2.5).

**One loss skeleton, two variants** (`PriorConfig.continuous`):

* discrete (the design): ``next_code`` cross-entropy over the K codes at every position that
  has a next segment; ``landed`` BCE at every position (1 where no full segment follows);
  ``landed_fraction`` L1 where landed. Decoding: top-1 code → its z through the codebook.
* continuous (gate P's control, §3.3): the SAME backbone regresses the next segment's UNROUNDED
  encoder vector (`Codebook.encode_continuous`, the bounded pre-round coordinates in the same
  space as z) with an MSE in place of the cross-entropy; the executor is handed that vector as
  ``manoeuvre_z``. The landed heads are the same. This is PatchTST-with-feedback in the plan's
  words: if it is not beaten in the closed loop, the discrete bottleneck bought nothing.

`bigram_nll` is gate T(ii)'s baseline: a Laplace-smoothed bigram over the code sequence
(BOS → c_1 … c_T → LANDED) fitted on train and scored on val, in nats per token, the number
the prior's val NLL must beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ts_transformer.manoeuvre.context import RUNWAY_TOKEN_WIDTH, TypeVocabulary, runway_token
from ts_transformer.manoeuvre.sequences import STATE_TOKEN_FEATURES, CodeSequence

PRIOR_SCHEMA = "ts-manoeuvre-prior-v1"
#: A padded target position (torch's ignore index).
IGNORE = -100


@dataclass(frozen=True)
class PriorConfig:
    """The prior's shape and objective. ``code_count`` is the codebook's K; ``z_dim`` the
    codebook's z width (read only under ``continuous``)."""

    code_count: int
    z_dim: int
    type_count: int
    continuous: bool = False
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_positions: int = 64
    landed_loss_weight: float = 1.0
    landed_fraction_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.code_count < 2 or self.type_count < 1 or self.z_dim < 1:
            raise ValueError("code_count ≥ 2, type_count ≥ 1 and z_dim ≥ 1")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be a multiple of n_heads")

    @property
    def bos(self) -> int:
        """The BOS row of the code embedding table (after the K codes)."""
        return self.code_count

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriorConfig:
        return cls(**data)


@dataclass(frozen=True)
class PriorBatch:
    """Padded to the longest flight: ``codes_in`` ``[B, L]`` (BOS then c_1…c_T), ``states``
    ``[B, L, 6]``, ``valid`` ``[B, L]``, ``next_code`` ``[B, L]`` (c_{t+1} or IGNORE), ``landed``
    ``[B, L]`` (1.0 where no full segment follows, IGNORE-masked by ``valid``), ``landed_fraction``
    ``[B]``, ``next_z`` ``[B, L, Z]`` (the continuous target, zeros where none), ``type_index``
    ``[B]``, ``runway`` ``[B, 2]``."""

    codes_in: torch.Tensor
    states: torch.Tensor
    valid: torch.Tensor
    next_code: torch.Tensor
    landed: torch.Tensor
    landed_fraction: torch.Tensor
    next_z: torch.Tensor
    type_index: torch.Tensor
    runway: torch.Tensor

    def to(self, device: torch.device) -> PriorBatch:
        return PriorBatch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})


def collate(sequences: Sequence[CodeSequence], config: PriorConfig, vocabulary: TypeVocabulary,
            *, continuous_targets: Sequence[np.ndarray] | None = None) -> PriorBatch:
    """The batch of ``sequences``. ``continuous_targets`` (one ``[T, Z]`` array per sequence, the
    unrounded encoder vectors) are required under ``config.continuous`` and refused otherwise."""
    if config.continuous != (continuous_targets is not None):
        raise ValueError("continuous targets are given exactly when the prior is continuous")
    batch = len(sequences)
    length = max(item.length for item in sequences) + 1
    if length > config.max_positions:
        raise ValueError(f"a flight has {length - 1} segments; the prior holds {config.max_positions - 1}")
    codes_in = torch.full((batch, length), config.bos, dtype=torch.long)
    states = torch.zeros(batch, length, len(STATE_TOKEN_FEATURES))
    valid = torch.zeros(batch, length, dtype=torch.bool)
    next_code = torch.full((batch, length), IGNORE, dtype=torch.long)
    landed = torch.zeros(batch, length)
    landed_fraction = torch.zeros(batch)
    next_z = torch.zeros(batch, length, config.z_dim)
    type_index = torch.zeros(batch, dtype=torch.long)
    runway = torch.zeros(batch, RUNWAY_TOKEN_WIDTH)
    for row, item in enumerate(sequences):
        count = item.length
        codes_in[row, 1 : count + 1] = torch.from_numpy(item.codes)
        states[row, : count + 1] = torch.from_numpy(item.states)
        valid[row, : count + 1] = True
        next_code[row, :count] = torch.from_numpy(item.codes)          # position t predicts c_{t+1}
        landed[row, count] = 1.0                                         # after the last segment
        landed_fraction[row] = item.landed_fraction
        if continuous_targets is not None:
            target = np.asarray(continuous_targets[row], dtype=np.float32)
            if target.shape != (count, config.z_dim):
                raise ValueError(f"{item.dataset_id}: continuous targets are [{count}, {config.z_dim}], got {target.shape}")
            next_z[row, :count] = torch.from_numpy(target)
        type_index[row] = vocabulary.index(item.typecode)
        runway[row] = torch.from_numpy(runway_token(item.runway_course_rad))
    return PriorBatch(codes_in, states, valid, next_code, landed, landed_fraction, next_z, type_index, runway)


@dataclass(frozen=True)
class PriorOutput:
    """``next_logits`` ``[B, L, K]`` (discrete) or ``next_z`` ``[B, L, Z]`` (continuous),
    ``landed_logit`` ``[B, L]``, ``landed_fraction`` ``[B, L]`` in (0, 1)."""

    landed_logit: torch.Tensor
    landed_fraction: torch.Tensor
    next_logits: torch.Tensor | None = None
    next_z: torch.Tensor | None = None


class ManoeuvrePrior(nn.Module):
    CONTEXT_TOKENS = 2   # type, runway

    def __init__(self, config: PriorConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.code_embedding = nn.Embedding(config.code_count + 1, d)          # K codes + BOS
        self.state_embedding = nn.Linear(len(STATE_TOKEN_FEATURES), d)
        self.position_embedding = nn.Embedding(config.max_positions, d)
        self.type_embedding = nn.Embedding(config.type_count, d)
        self.runway_embedding = nn.Linear(RUNWAY_TOKEN_WIDTH, d)
        self.context_position = nn.Parameter(torch.zeros(self.CONTEXT_TOKENS, d))
        layer = nn.TransformerEncoderLayer(d, config.n_heads, dim_feedforward=config.d_ff, dropout=config.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.next_head = nn.Linear(d, config.z_dim if config.continuous else config.code_count)
        self.landed_head = nn.Linear(d, 2)                                    # logit, fraction pre-activation
        nn.init.normal_(self.context_position, std=0.02)

    def forward(self, batch: PriorBatch) -> PriorOutput:
        b, length = batch.codes_in.shape
        positions = torch.arange(length, device=batch.codes_in.device)
        tokens = self.code_embedding(batch.codes_in) + self.state_embedding(batch.states) + self.position_embedding(positions)
        context = torch.stack((self.type_embedding(batch.type_index), self.runway_embedding(batch.runway)), dim=1) + self.context_position
        sequence = torch.cat((context, tokens), dim=1)
        total = self.CONTEXT_TOKENS + length
        # causal over the sequence positions; the context tokens are visible to every position
        # and see only each other
        mask = torch.triu(torch.ones(total, total, dtype=torch.bool, device=sequence.device), diagonal=1)
        mask[: self.CONTEXT_TOKENS, : self.CONTEXT_TOKENS] = False
        padding = torch.cat((torch.zeros(b, self.CONTEXT_TOKENS, dtype=torch.bool, device=sequence.device), ~batch.valid), dim=1)
        encoded = self.norm(self.transformer(sequence, mask=mask, src_key_padding_mask=padding))[:, self.CONTEXT_TOKENS :]
        landed = self.landed_head(encoded)
        head = self.next_head(encoded)
        return PriorOutput(
            landed_logit=landed[..., 0], landed_fraction=torch.sigmoid(landed[..., 1]),
            next_logits=None if self.config.continuous else head, next_z=head if self.config.continuous else None,
        )

    def loss(self, output: PriorOutput, batch: PriorBatch) -> dict[str, torch.Tensor]:
        """``total`` and its parts: ``next`` (CE in nats per position with a next segment, or
        the MSE of the continuous vector), ``landed`` (BCE per valid position),
        ``landed_fraction`` (L1 where landed)."""
        has_next = batch.next_code != IGNORE
        if self.config.continuous:
            error = (output.next_z - batch.next_z).square().sum(dim=-1)
            next_term = (error * has_next).sum() / has_next.sum().clamp(min=1)
        else:
            next_term = F.cross_entropy(output.next_logits.flatten(0, 1), batch.next_code.flatten(), ignore_index=IGNORE)
        landed_term = F.binary_cross_entropy_with_logits(output.landed_logit, batch.landed, reduction="none")
        landed_term = (landed_term * batch.valid).sum() / batch.valid.sum()
        at_landing = batch.landed > 0.5
        fraction_term = ((output.landed_fraction - batch.landed_fraction.unsqueeze(1)).abs() * at_landing).sum() / at_landing.sum().clamp(min=1)
        total = next_term + self.config.landed_loss_weight * landed_term + self.config.landed_fraction_loss_weight * fraction_term
        return {"total": total, "next": next_term, "landed": landed_term, "landed_fraction": fraction_term}

    @torch.no_grad()
    def next_code_accuracy(self, output: PriorOutput, batch: PriorBatch) -> float:
        if self.config.continuous:
            raise ValueError("a continuous prior predicts a vector, not a code")
        has_next = batch.next_code != IGNORE
        hits = (output.next_logits.argmax(dim=-1) == batch.next_code) & has_next
        return float(hits.sum()) / max(int(has_next.sum()), 1)


# ── decoding one step ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Step:
    """The prior's answer at the last valid position of each flight: ``landed_probability``
    ``[B]``, and either ``code`` ``[B]`` (+ ``probabilities`` ``[B, K]``) or ``z`` ``[B, Z]``."""

    landed_probability: torch.Tensor
    landed_fraction: torch.Tensor
    code: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None
    z: torch.Tensor | None = None


@torch.no_grad()
def last_position_step(model: ManoeuvrePrior, batch: PriorBatch) -> Step:
    """Decode the next segment for every flight from its LAST valid position (top-1, plan §2.5:
    sampling is a fan reading, not a prediction)."""
    output = model(batch)
    last = batch.valid.sum(dim=1) - 1
    rows = torch.arange(len(last), device=last.device)
    landed = torch.sigmoid(output.landed_logit[rows, last])
    fraction = output.landed_fraction[rows, last]
    if model.config.continuous:
        return Step(landed_probability=landed, landed_fraction=fraction, z=output.next_z[rows, last])
    probabilities = torch.softmax(output.next_logits[rows, last], dim=-1)
    return Step(landed_probability=landed, landed_fraction=fraction, code=probabilities.argmax(dim=-1), probabilities=probabilities)


# ── the fit ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FitResult:
    """``history``: one row per epoch (train / val loss parts, val accuracy, the learning rate);
    ``best_epoch`` / ``best_val_next``: the kept epoch, selected on the val ``next`` term (the
    NLL, or the continuous MSE); ``state_dict``: that epoch's weights (CPU)."""

    history: list[dict[str, Any]]
    best_epoch: int
    best_val_next: float
    state_dict: dict[str, torch.Tensor]
    stopped_early: bool


def _batches(sequences: Sequence[CodeSequence], targets: Sequence[np.ndarray] | None, config: PriorConfig,
             vocabulary: TypeVocabulary, *, batch_size: int, order: np.ndarray, device: torch.device):
    for start in range(0, len(order), batch_size):
        chosen = order[start : start + batch_size]
        yield collate(
            [sequences[i] for i in chosen], config, vocabulary,
            continuous_targets=None if targets is None else [targets[i] for i in chosen],
        ).to(device)


@torch.no_grad()
def evaluate(model: ManoeuvrePrior, sequences: Sequence[CodeSequence], targets: Sequence[np.ndarray] | None,
             vocabulary: TypeVocabulary, *, batch_size: int, device: torch.device) -> dict[str, float]:
    """The loss parts (token-weighted over the set), the next-code accuracy (discrete) and the
    landed-decision accuracy."""
    model.eval()
    sums: dict[str, float] = {}
    weights: dict[str, float] = {}
    hits = has_next = landed_hits = valid = 0
    order = np.arange(len(sequences))
    for batch in _batches(sequences, targets, model.config, vocabulary, batch_size=batch_size, order=order, device=device):
        output = model(batch)
        terms = model.loss(output, batch)
        next_count = float((batch.next_code != IGNORE).sum())
        valid_count = float(batch.valid.sum())
        for name, weight in (("next", next_count), ("landed", valid_count), ("landed_fraction", float((batch.landed > 0.5).sum()))):
            sums[name] = sums.get(name, 0.0) + float(terms[name]) * weight
            weights[name] = weights.get(name, 0.0) + weight
        if not model.config.continuous:
            mask = batch.next_code != IGNORE
            hits += int(((output.next_logits.argmax(dim=-1) == batch.next_code) & mask).sum())
            has_next += int(mask.sum())
        landed_hits += int((((output.landed_logit > 0.0) == (batch.landed > 0.5)) & batch.valid).sum())
        valid += int(batch.valid.sum())
    out = {name: sums[name] / max(weights[name], 1.0) for name in sums}
    out["total"] = out["next"] + model.config.landed_loss_weight * out["landed"] + model.config.landed_fraction_loss_weight * out["landed_fraction"]
    out["landed_accuracy"] = landed_hits / max(valid, 1)
    if not model.config.continuous:
        out["next_code_accuracy"] = hits / max(has_next, 1)
    return out


def fit(
    model: ManoeuvrePrior, train: Sequence[CodeSequence], val: Sequence[CodeSequence], vocabulary: TypeVocabulary,
    *, train_targets: Sequence[np.ndarray] | None = None, val_targets: Sequence[np.ndarray] | None = None,
    epochs: int, patience: int, batch_size: int, learning_rate: float, seed: int, device: torch.device,
    log=None,
) -> FitResult:
    """Adam, shuffled batches per epoch (seeded), the epoch kept on the val ``next`` term,
    early stop after ``patience`` epochs without improvement. ``log`` is called with each
    epoch's row when given."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_val, best_epoch, best_state, since_best = math.inf, 0, None, 0
    stopped_early = False
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        sums: dict[str, float] = {}
        count = 0
        for batch in _batches(train, train_targets, model.config, vocabulary, batch_size=batch_size, order=order, device=device):
            optimizer.zero_grad()
            terms = model.loss(model(batch), batch)
            terms["total"].backward()
            optimizer.step()
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * len(batch.codes_in)
            count += len(batch.codes_in)
        validation = evaluate(model, val, val_targets, vocabulary, batch_size=batch_size, device=device)
        row = {"epoch": epoch, "train": {name: value / count for name, value in sums.items()}, "val": validation,
               "learning_rate": learning_rate}
        history.append(row)
        if log is not None:
            log(row)
        if validation["next"] < best_val:
            best_val, best_epoch, since_best = validation["next"], epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                stopped_early = True
                break
    model.load_state_dict(best_state)
    return FitResult(history=history, best_epoch=best_epoch, best_val_next=best_val, state_dict=best_state,
                     stopped_early=stopped_early)


# ── the bigram baseline (gate T(ii)) ─────────────────────────────────────────

def bigram_nll(train: Sequence[CodeSequence], val: Sequence[CodeSequence], code_count: int, *, alpha: float = 1.0) -> dict[str, float]:
    """A Laplace-smoothed bigram over ``BOS → c_1 → … → c_T → LANDED`` fitted on ``train``, scored
    on ``val``: ``nll_per_token`` in nats over every transition (the LANDED one included) and
    ``nll_per_code`` over the code transitions alone — the two numbers a prior's ``next`` and
    ``landed`` terms are read against."""
    bos, landed = code_count, code_count + 1
    counts = np.full((code_count + 1, code_count + 1), alpha, dtype=np.float64)   # from {codes, BOS} to {codes, LANDED}
    for item in train:
        chain = [bos, *item.codes.tolist(), landed]
        for a, b in zip(chain[:-1], chain[1:]):
            counts[a, b if b != landed else code_count] += 1.0
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    total = codes_only = 0.0
    tokens = code_tokens = 0
    for item in val:
        chain = [bos, *item.codes.tolist(), landed]
        for a, b in zip(chain[:-1], chain[1:]):
            nll = -math.log(probabilities[a, b if b != landed else code_count])
            total += nll
            tokens += 1
            if b != landed:
                codes_only += nll
                code_tokens += 1
    return {"nll_per_token": total / max(tokens, 1), "nll_per_code": codes_only / max(code_tokens, 1),
            "tokens": tokens, "code_tokens": code_tokens, "alpha": alpha}


__all__ = [
    "IGNORE", "PRIOR_SCHEMA", "FitResult", "ManoeuvrePrior", "PriorBatch", "PriorConfig", "PriorOutput", "Step",
    "bigram_nll", "collate", "evaluate", "fit", "last_position_step",
]
