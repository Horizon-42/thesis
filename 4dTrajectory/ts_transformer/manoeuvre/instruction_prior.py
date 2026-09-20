"""The instruction prior (two-tier v3 stage B, plan §5.2.2 "预训练"; B′-dev3): a causal
Transformer over one flight's sentence that says, at every position, the five words in force at
the NEXT event — one softmax per kind (factorised heads), the terminal kind among them
instead.

    [TYPE] [RWY]  [w_0, x_0] [w_1, x_1] … [w_T, x_T]
        →  at t: p(heading_{t+1}), p(altitude_{t+1}), p(speed_{t+1}), p(intercept_{t+1}),
                 p(runway_{t+1}), p(duration_{t+1}), p(terminal_{t+1})

Tokens: the sum of the five word embeddings (the intercept's table has no extra row for "no
capture yet"; the runway's classes are the cohort's thresholds, `instructions.RunwayVocabulary`,
which is why `PriorConfig.words` carries their count rather than reading it off the spec), the
state token (`instruction_sequences.state_token`, six features through a
linear layer) and the position; two context tokens (aircraft type, runway) sit in front and
every position reads them and its past. Teacher forcing on the truth's sentence; the closed-loop
fine-tuning (D55) feeds `InstructionSequence`s whose states were flown and whose targets are the
truth's words at the same absolute time — the same batch, the same loss.

Readings (`evaluate`): per kind the next-word NLL and top-1; the FLIP rate (the prior changes a
word the truth holds — an unforced instruction), the MISS rate (the truth changes and the prior
holds) and the change recall; top-k coverage per kind and JOINT top-K coverage over the five
kinds (the truth's 6-tuple among the K best product-of-marginals candidates, searched inside
each kind's top-8 — what a K-candidate decoder can reach, D55 / D56); the landing decision.
`hold_baseline` is the reading a prior must beat: "the words stay", and a per-kind Laplace
bigram P(next | current).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ts_transformer.manoeuvre.context import TypeVocabulary
from ts_transformer.manoeuvre.instruction_sequences import STATE_TOKEN_FEATURES, InstructionSequence
from ts_transformer.manoeuvre.instructions import INSTRUCTION_KINDS, NO_INTERCEPT

PRIOR_SCHEMA = "ts-instruction-prior-v1"
#: A padded target position (torch's ignore index).
IGNORE = -100
#: The per-kind top-k coverages read (a k at or past a kind's class count is 1 by construction
#: and is reported as None), and the joint top-K (searched inside each kind's top-`JOINT_SEARCH`,
#: which bounds the K a marginal rank can be trusted to: the check at import).
TOP_K = (1, 2, 4, 8)
JOINT_TOP_K = (2, 4, 8)
JOINT_SEARCH = 8
#: Positions `_joint_ranks` scores at once (its candidate tensor is [rows, Π_kind min(JOINT_SEARCH,
#: classes)] — at most JOINT_SEARCH ** len(INSTRUCTION_KINDS), in practice far less: the intercept
#: has 4 classes and the runway one per threshold).
JOINT_CHUNK = 2048
if max(JOINT_TOP_K) > JOINT_SEARCH:
    raise RuntimeError("the joint top-K is searched inside each kind's top-JOINT_SEARCH: K cannot exceed it")
if NO_INTERCEPT != -1:
    raise RuntimeError("the intercept shift puts NO_INTERCEPT on row 0: it must be -1")


@dataclass(frozen=True)
class PriorConfig:
    """The prior's shape. ``words`` are the word counts per kind (the intercept head gets one
    more class, "none") — `Vocabulary.words` for the four spec'd kinds plus the RUNWAY's class
    count, which is the cohort's (`instructions.RunwayVocabulary`), not the spec's;
    ``vocabulary_sha256`` binds the checkpoint to the artefact."""

    words: dict[str, int]
    type_count: int
    vocabulary_sha256: str
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_positions: int = 128

    def __post_init__(self) -> None:
        if tuple(self.words) != INSTRUCTION_KINDS or any(count < 2 for count in self.words.values()):
            raise ValueError(f"words are the counts of {INSTRUCTION_KINDS}, each ≥ 2, got {self.words}")
        if self.type_count < 1 or self.max_positions < 2:
            raise ValueError("type_count ≥ 1 and max_positions ≥ 2")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be a multiple of n_heads")

    def classes(self, kind: str) -> int:
        """The head's classes: the kind's own words. Nothing is shifted — since D73 took the
        intercept out, every kind is in force from the first event and no column holds a
        "nothing said" value."""
        return self.words[kind]

    def to_dict(self) -> dict[str, Any]:
        return {name: (dict(getattr(self, name)) if name == "words" else getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriorConfig:
        return cls(**data)


def shift_words(words: np.ndarray) -> np.ndarray:
    """Word indices as the embedding tables index them.

    A no-op since D73: every kind is in force from the first event, so no column carries a
    "nothing said" value and none needs shifting. Kept as the ONE place that would change if a
    kind ever became optional again — the callers stay honest about where that would go."""
    return np.asarray(words, dtype=np.int64)


def unshift_words(indices: np.ndarray) -> np.ndarray:
    """The inverse of `shift_words` — a no-op, for the same reason."""
    return np.asarray(indices, dtype=np.int64)



@dataclass(frozen=True)
class PriorBatch:
    """Padded to the longest flight: ``words`` ``[B, L, 5]`` (the inputs, shifted), ``states``
    ``[B, L, 6]``, ``valid`` ``[B, L]``, ``targets`` ``[B, L, 5]`` (the next position’s words,
    shifted; IGNORE where none — the last position of a sequence that ends at the landing),
    ``type_index`` ``[B]``."""

    words: torch.Tensor
    states: torch.Tensor
    valid: torch.Tensor
    targets: torch.Tensor
    type_index: torch.Tensor

    def to(self, device: torch.device) -> PriorBatch:
        return PriorBatch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})

    @property
    def has_next(self) -> torch.Tensor:
        """``[B, L]``: the positions that predict a next word (every valid one but the last)."""
        return self.targets[..., 0] != IGNORE


def collate(sequences: Sequence[InstructionSequence], config: PriorConfig, types: TypeVocabulary) -> PriorBatch:
    batch = len(sequences)
    length = max(item.length for item in sequences)
    if length > config.max_positions:
        raise ValueError(f"a flight has {length} positions; the prior holds {config.max_positions}")
    kinds = len(INSTRUCTION_KINDS)
    words = torch.zeros(batch, length, kinds, dtype=torch.long)
    states = torch.zeros(batch, length, len(STATE_TOKEN_FEATURES))
    valid = torch.zeros(batch, length, dtype=torch.bool)
    targets = torch.full((batch, length, kinds), IGNORE, dtype=torch.long)
    type_index = torch.zeros(batch, dtype=torch.long)
    for row, item in enumerate(sequences):
        count = item.length
        words[row, :count] = torch.from_numpy(shift_words(item.words))
        states[row, :count] = torch.from_numpy(np.asarray(item.states, dtype=np.float32))
        valid[row, :count] = True
        # event k predicts the words at event k + 1. A sequence that ends at the landing has no
        # k + 1 for its last event — the LANDING is said by that event's own terminal word (D72),
        # not by a separate head — so its last event carries no target. A rolled prefix's last
        # event does have a next: the truth continues past where the loop stopped.
        with_next = count - 1 if item.ends_at_landing else count
        targets[row, :with_next] = torch.from_numpy(shift_words(item.targets[:with_next]))
        type_index[row] = types.index(item.typecode)
    return PriorBatch(words, states, valid, targets, type_index)


@dataclass(frozen=True)
class PriorOutput:
    """``logits[kind]`` ``[B, L, classes]`` for each kind, the terminal kind among them."""

    logits: dict[str, torch.Tensor]


class InstructionPrior(nn.Module):
    #: ONE context token, the aircraft type. The runway's COURSE used to sit here and was
    #: removed with D66: the state's course is absolute while the heading word is relative, so a
    #: course in the context is the runway's identity handed to a model that must SAY the runway.
    #: The frame now comes from the runway WORD, which is where the sentence puts it.
    CONTEXT_TOKENS = 1   # type

    def __init__(self, config: PriorConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.word_embeddings = nn.ModuleDict({kind: nn.Embedding(config.classes(kind), d) for kind in INSTRUCTION_KINDS})
        self.state_embedding = nn.Linear(len(STATE_TOKEN_FEATURES), d)
        self.position_embedding = nn.Embedding(config.max_positions, d)
        self.type_embedding = nn.Embedding(config.type_count, d)
        self.context_position = nn.Parameter(torch.zeros(self.CONTEXT_TOKENS, d))
        layer = nn.TransformerEncoderLayer(d, config.n_heads, dim_feedforward=config.d_ff, dropout=config.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({kind: nn.Linear(d, config.classes(kind)) for kind in INSTRUCTION_KINDS})
        nn.init.normal_(self.context_position, std=0.02)

    def forward(self, batch: PriorBatch) -> PriorOutput:
        b, length, _ = batch.words.shape
        positions = torch.arange(length, device=batch.words.device)
        tokens = sum(self.word_embeddings[kind](batch.words[..., column]) for column, kind in enumerate(INSTRUCTION_KINDS))
        tokens = tokens + self.state_embedding(batch.states) + self.position_embedding(positions)
        context = self.type_embedding(batch.type_index).unsqueeze(1) + self.context_position
        sequence = torch.cat((context, tokens), dim=1)
        total = self.CONTEXT_TOKENS + length
        # causal over the sequence positions; the context tokens are visible to every position
        # and see only each other
        mask = torch.triu(torch.ones(total, total, dtype=torch.bool, device=sequence.device), diagonal=1)
        mask[: self.CONTEXT_TOKENS, : self.CONTEXT_TOKENS] = False
        padding = torch.cat((torch.zeros(b, self.CONTEXT_TOKENS, dtype=torch.bool, device=sequence.device), ~batch.valid), dim=1)
        encoded = self.norm(self.transformer(sequence, mask=mask, src_key_padding_mask=padding))[:, self.CONTEXT_TOKENS :]
        return PriorOutput(
            logits={kind: head(encoded) for kind, head in self.heads.items()},
        )

    def loss(self, output: PriorOutput, batch: PriorBatch) -> dict[str, torch.Tensor]:
        """``total`` and its parts: one cross-entropy per kind (nats per position with a next
        word). The terminal kind is one of them, so there is no separate landing term any more
        (D72): "does the sentence stop here" is one question with three answers."""
        has_next = batch.has_next
        count = has_next.sum().clamp(min=1)
        terms: dict[str, torch.Tensor] = {}
        for column, kind in enumerate(INSTRUCTION_KINDS):
            terms[kind] = F.cross_entropy(
                output.logits[kind].flatten(0, 1), batch.targets[..., column].flatten(), ignore_index=IGNORE, reduction="sum"
            ) / count
        terms["next"] = sum(terms[kind] for kind in INSTRUCTION_KINDS)
        terms["total"] = terms["next"]
        return terms


# ── readings ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FitResult:
    history: list[dict[str, Any]]
    best_epoch: int
    best_val_next: float
    state_dict: dict[str, torch.Tensor]
    stopped_early: bool


def _batches(sequences: Sequence[InstructionSequence], config: PriorConfig, types: TypeVocabulary, *,
             batch_size: int, order: np.ndarray, device: torch.device):
    for start in range(0, len(order), batch_size):
        chunk = [sequences[int(index)] for index in order[start : start + batch_size]]
        yield collate(chunk, config, types).to(device)


def _joint_ranks(logits: dict[str, torch.Tensor], targets: torch.Tensor, has_next: torch.Tensor) -> torch.Tensor:
    """The truth 6-tuple’s rank (0 = best; a tie counts as beaten) among the product-of-marginals
    candidates built from each kind's top-`JOINT_SEARCH` words, at every position with a next;
    a truth word outside a kind's top-`JOINT_SEARCH` ranks past every candidate
    (`JOINT_SEARCH ** len(INSTRUCTION_KINDS)`). Scored `JOINT_CHUNK` positions at a time."""
    rows = has_next.flatten()
    logp = {kind: F.log_softmax(logits[kind].flatten(0, 1)[rows].float(), dim=-1) for kind in INSTRUCTION_KINDS}
    target = {column: targets[..., column].flatten()[rows] for column in range(len(INSTRUCTION_KINDS))}
    total = int(rows.sum())
    out = torch.empty(total, dtype=torch.long, device=targets.device)
    for start in range(0, total, JOINT_CHUNK):
        stop = min(start + JOINT_CHUNK, total)
        truth_score = torch.zeros(stop - start, device=targets.device)
        covered = torch.ones(stop - start, dtype=torch.bool, device=targets.device)
        candidate = None
        for column, kind in enumerate(INSTRUCTION_KINDS):
            chunk = logp[kind][start:stop]
            top_values, top_indices = chunk.topk(min(JOINT_SEARCH, chunk.shape[-1]), dim=-1)        # [n, S]
            hit = target[column][start:stop]
            truth_score = truth_score + chunk.gather(1, hit.unsqueeze(1)).squeeze(1)
            covered &= (top_indices == hit.unsqueeze(1)).any(dim=1)
            shape = [stop - start] + [1] * column + [-1] + [1] * (len(INSTRUCTION_KINDS) - column - 1)
            term = top_values.view(shape)
            candidate = term if candidate is None else candidate + term
        better = (candidate.flatten(1) >= truth_score.unsqueeze(1)).sum(dim=1) - 1       # the truth itself is a candidate
        out[start:stop] = torch.where(covered, better, torch.full_like(better, JOINT_SEARCH ** len(INSTRUCTION_KINDS)))
    return out


def _rate(numerator: int, denominator: int) -> float | None:
    """A share, or None when nothing was there to count (never a 0.0 that reads as measured)."""
    return numerator / denominator if denominator else None


@torch.no_grad()
def evaluate(model: InstructionPrior, sequences: Sequence[InstructionSequence], types: TypeVocabulary, *,
             batch_size: int, device: torch.device, joint: bool = True) -> dict[str, Any]:
    """
    CAVEAT while D66 is unimplemented: the labeller writes the runway word CONSTANT for a flight,
    and the same word is an input column, so ``top1["runway"]`` is ~1.0, ``change_share`` is 0 and
    the joint top-K scores a tuple whose runway element is free. ``next`` is now the sum of FIVE
    cross-entropies and is not comparable with any number from before D62. Any readout that quotes
    these must say so, or the runway column reads as skill.
The loss parts (token-weighted over the set) and the readings the module docstring names.
    Denominators: the per-kind NLL, top-1 / top-k and `change_share` are over the positions with a
    next word; `flip_rate` over the positions where the truth HOLDS the kind's word; `miss_rate`
    and `change_recall` over the positions where it CHANGES; the landing figures over every
    valid position. ``joint`` (the joint top-K coverage, the costly reading) can be left out of
    an epoch's validation pass."""
    if not sequences:
        raise ValueError("evaluate needs at least one sequence")
    model.eval()
    sums: dict[str, float] = {}
    count_next = count_valid = 0
    hits = {kind: 0 for kind in INSTRUCTION_KINDS}
    top = {kind: {k: 0 for k in TOP_K} for kind in INSTRUCTION_KINDS}
    flips = {kind: 0 for kind in INSTRUCTION_KINDS}
    misses = {kind: 0 for kind in INSTRUCTION_KINDS}
    recalled = {kind: 0 for kind in INSTRUCTION_KINDS}
    held = {kind: 0 for kind in INSTRUCTION_KINDS}
    changed = {kind: 0 for kind in INSTRUCTION_KINDS}
    joint_hits = {k: 0 for k in JOINT_TOP_K}

    for batch in _batches(sequences, model.config, types, batch_size=batch_size, order=np.arange(len(sequences)), device=device):
        output = model(batch)
        terms = model.loss(output, batch)
        has_next = batch.has_next
        n_next, n_valid = int(has_next.sum()), int(batch.valid.sum())
        for kind in (*INSTRUCTION_KINDS, "next"):
            sums[kind] = sums.get(kind, 0.0) + float(terms[kind]) * n_next
        count_next += n_next
        count_valid += n_valid
        for column, kind in enumerate(INSTRUCTION_KINDS):
            logits = output.logits[kind]
            target = batch.targets[..., column]
            current = batch.words[..., column]
            predicted = logits.argmax(dim=-1)
            hits[kind] += int(((predicted == target) & has_next).sum())
            ranked = logits.topk(min(max(TOP_K), logits.shape[-1]), dim=-1).indices
            for k in TOP_K:
                top[kind][k] += int(((ranked[..., :k] == target.unsqueeze(-1)).any(dim=-1) & has_next).sum())
            holds = (target == current) & has_next
            moves = (target != current) & has_next
            held[kind] += int(holds.sum())
            changed[kind] += int(moves.sum())
            flips[kind] += int(((predicted != current) & holds).sum())
            misses[kind] += int(((predicted == current) & moves).sum())
            recalled[kind] += int(((predicted == target) & moves).sum())
        if joint and n_next:
            ranks = _joint_ranks(output.logits, batch.targets, has_next)
            for k in JOINT_TOP_K:
                joint_hits[k] += int((ranks < k).sum())

    out: dict[str, Any] = {name: _rate(sums[name], count_next) for name in (*INSTRUCTION_KINDS, "next")}
    out["total"] = out["next"]                      # D72: the terminal kind is one of the five
    out["positions_with_next"] = count_next
    out["valid_positions"] = count_valid
    out["top1"] = {kind: _rate(hits[kind], count_next) for kind in INSTRUCTION_KINDS}
    out["top_k"] = {kind: {str(k): (_rate(top[kind][k], count_next) if k < model.config.classes(kind) else None) for k in TOP_K}
                    for kind in INSTRUCTION_KINDS}
    out["flip_rate"] = {kind: _rate(flips[kind], held[kind]) for kind in INSTRUCTION_KINDS}
    out["miss_rate"] = {kind: _rate(misses[kind], changed[kind]) for kind in INSTRUCTION_KINDS}
    out["change_recall"] = {kind: _rate(recalled[kind], changed[kind]) for kind in INSTRUCTION_KINDS}
    out["change_share"] = {kind: _rate(changed[kind], count_next) for kind in INSTRUCTION_KINDS}
    out["joint_top_k"] = {str(k): _rate(joint_hits[k], count_next) for k in JOINT_TOP_K} if joint else None
    # the landing: its base rate (one positive per flight) beside the accuracy, so the accuracy is not read as skill
    return out


def fit(
    model: InstructionPrior, train: Sequence[InstructionSequence], val: Sequence[InstructionSequence], types: TypeVocabulary,
    *, epochs: int, patience: int, batch_size: int, learning_rate: float, seed: int, device: torch.device, log=None,
) -> FitResult:
    """Adam, shuffled batches per epoch (seeded), the epoch kept on the val ``next`` term (the
    four cross-entropies), early stop after ``patience`` epochs without improvement. The
    ``train`` terms of a row are flight-weighted means of the batch losses (a monitor); the
    ``val`` terms are `evaluate`'s token-weighted ones."""
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
        for batch in _batches(train, model.config, types, batch_size=batch_size, order=order, device=device):
            optimizer.zero_grad()
            terms = model.loss(model(batch), batch)
            terms["total"].backward()
            optimizer.step()
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * len(batch.words)
            count += len(batch.words)
        validation = evaluate(model, val, types, batch_size=batch_size, device=device, joint=False)
        row = {"epoch": epoch, "train": {name: value / count for name, value in sums.items()}, "val": validation,
               "learning_rate": learning_rate}
        history.append(row)
        if log is not None:
            log(row)
        if validation["next"] is not None and validation["next"] < best_val:
            best_val, best_epoch, since_best = validation["next"], epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                stopped_early = True
                break
    if best_state is None:
        raise ValueError(f"no epoch improved the val next-word term ({len(history)} run): it was never finite, or epochs = 0")
    model.load_state_dict(best_state)
    return FitResult(history=history, best_epoch=best_epoch, best_val_next=best_val, state_dict=best_state, stopped_early=stopped_early)


# ── the baselines a prior must beat ──────────────────────────────────────────

def hold_baseline(train: Sequence[InstructionSequence], val: Sequence[InstructionSequence], config: PriorConfig,
                  *, alpha: float = 1.0) -> dict[str, Any]:
    """Per kind on ``val``: the accuracy of "the word stays" (top-1 of a prior that never
    speaks) and the NLL of a Laplace-smoothed bigram P(next | current) fitted on ``train`` —
    the same conditional quantity as the prior's per-kind term."""
    out: dict[str, Any] = {"hold_accuracy": {}, "bigram_nll": {}, "alpha": alpha,
                           "positions_with_next": sum(item.length - 1 for item in val)}
    for column, kind in enumerate(INSTRUCTION_KINDS):
        classes = config.classes(kind)
        counts = np.full((classes, classes), alpha, dtype=np.float64)
        for item in train:
            words = shift_words(item.words)[:, column]
            targets = shift_words(item.targets)[:, column]
            for a, b in zip(words[:-1], targets[:-1]):
                counts[a, b] += 1.0
        conditional = counts / counts.sum(axis=1, keepdims=True)
        holds = total = 0
        nll = 0.0
        for item in val:
            words = shift_words(item.words)[:, column]
            targets = shift_words(item.targets)[:, column]
            for a, b in zip(words[:-1], targets[:-1]):
                holds += int(a == b)
                nll -= math.log(conditional[a, b])
                total += 1
        out["hold_accuracy"][kind] = _rate(holds, total)
        out["bigram_nll"][kind] = _rate(nll, total)
    out["bigram_nll"]["next"] = None if not out["positions_with_next"] else sum(out["bigram_nll"][kind] for kind in INSTRUCTION_KINDS)
    return out


__all__ = [
    "IGNORE", "JOINT_CHUNK", "JOINT_SEARCH", "JOINT_TOP_K", "PRIOR_SCHEMA", "TOP_K", "FitResult", "InstructionPrior", "PriorBatch",
    "PriorConfig", "PriorOutput", "collate", "evaluate", "fit", "hold_baseline", "shift_words", "unshift_words",
]
