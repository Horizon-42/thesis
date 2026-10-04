"""Training the prior (prior design §5; D31, D40): teacher forcing, the loss of each step and column, and the stop on
the select days.

**The loss** (§7 item 4) is the five columns' cross entropy at every asked row (`batch.RowTensors.asked`: present, at
or after the first predicted step), each head reading the truth of the columns before it. The per-step loss of a set
of sentences is the sum over its asked rows and the five columns, divided by its asked rows.

**The loop** trains on a run's train sentences (`runs.RunData`) with AdamW, a linear warm-up and gradient clipping,
in batches of sentences of similar length of at most ``tokens_per_batch`` padded rows, shuffled each epoch. After each
epoch it reads the per-step loss on the run's select sentences, keeps the state of the epoch with the smallest one
(D31) and stops after ``patience`` epochs without a smaller one, or at ``max_epochs``. It reads nothing else.
`TrainConfig`'s defaults are configuration A (D40).
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.instructions.words import COLUMNS
from ts_transformer.prior.batch import RowTensors, SentenceRows, collate, require_words
from ts_transformer.prior.model import Prior
from ts_transformer.prior.runs import RunData


@dataclass(frozen=True)
class TrainConfig:
    tokens_per_batch: int = 16_384
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    clip_norm: float = 1.0
    max_epochs: int = 30
    patience: int = 3
    seed: int = 1337

    def __post_init__(self) -> None:
        for name in ("tokens_per_batch", "warmup_steps", "max_epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def length_groups(lengths: Sequence[int], tokens: int, rng: np.random.Generator | None) -> list[list[int]]:
    """Indices in groups of similar length, each at most ``tokens`` padded rows (a longer one is a group of its own).
    With ``rng`` the sentences of one length are drawn in a new order and the groups shuffled, so each call makes new
    groups; without it, in length order."""
    groups: list[list[int]] = []
    current: list[int] = []
    ties = rng.permutation(len(lengths)) if rng is not None else np.arange(len(lengths))
    for i in sorted(range(len(lengths)), key=lambda i: (lengths[i], ties[i])):
        if current and lengths[i] * (len(current) + 1) > tokens:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    if rng is not None:
        rng.shuffle(groups)
    return groups


def step_nll(logits: Sequence[torch.Tensor], rows: RowTensors) -> torch.Tensor:
    """``[B, R, 5]``: the teacher-forced loss of each row and column (§7 item 4), 0 where the row is not asked."""
    nll = logits[0].new_zeros((*rows.asked.shape, len(logits)))
    for c, logit in enumerate(logits):
        nll[..., c][rows.asked] = nn.functional.cross_entropy(logit[rows.asked], rows.targets[..., c][rows.asked],
                                                              reduction="none")
    return nll


def column_nll(logits: Sequence[torch.Tensor], rows: RowTensors) -> torch.Tensor:
    """``[5]``: each column's cross entropy summed over the asked rows."""
    return step_nll(logits, rows).sum(dim=(0, 1))


def batch_nll(model: Prior, rows: RowTensors) -> tuple[torch.Tensor, int]:
    """``(each column's summed loss [5], the asked rows)`` of one batch, teacher-forced."""
    return column_nll(model(rows), rows), int(rows.asked.sum())


@torch.no_grad()
def evaluate(model: Prior, sentences: Sequence[SentenceRows], tokens_per_batch: int, device: torch.device
             ) -> dict[str, Any]:
    """The per-step loss of ``sentences``, in all and for each column, with dropout off."""
    model.eval()
    total, steps = torch.zeros(len(COLUMNS), dtype=torch.float64, device=device), 0
    for group in length_groups([s.rows for s in sentences], tokens_per_batch, None):
        nll, asked = batch_nll(model, collate([sentences[i] for i in group], device))
        total += nll.double()
        steps += asked
    per_column = (total / steps).cpu().numpy()
    return {"loss_per_step": float(per_column.sum()), "per_column": dict(zip(COLUMNS, per_column.tolist())),
            "steps": steps}


class TrainResult(NamedTuple):
    state: dict[str, torch.Tensor]       # the weights of the best epoch
    best_epoch: int
    history: list[dict[str, Any]]        # one row an epoch


def train(model: Prior, data: RunData, config: TrainConfig, device: torch.device, log: Callable[[str], None]
          ) -> TrainResult:
    """Train ``model`` (on ``device``; the caller seeds its initial weights) on ``data`` (module docstring). The seed
    sets the batch order and dropout."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(1.0, (step + 1) / config.warmup_steps))
    for sentences in (data.train, data.select):
        require_words(sentences, model.config.word_values)
    lengths = [s.rows for s in data.train]
    best, best_state, best_epoch, stale, history = float("inf"), {}, 0, 0, []
    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        total, steps = 0.0, 0
        for group in length_groups(lengths, config.tokens_per_batch, rng):
            nll, asked = batch_nll(model, collate([data.train[i] for i in group], device))
            loss = nll.sum() / asked
            if not torch.isfinite(loss):
                raise FloatingPointError(f"epoch {epoch}: the loss is {float(loss.detach())}")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimiser.step()
            schedule.step()
            total += float(loss.detach()) * asked
            steps += asked
        select = evaluate(model, data.select, config.tokens_per_batch, device)
        if not np.isfinite(select["loss_per_step"]):
            raise FloatingPointError(f"epoch {epoch}: the select loss is {select['loss_per_step']}")
        row = {"epoch": epoch, "train_loss_per_step": total / steps, "select_loss_per_step": select["loss_per_step"],
               "select_per_column": select["per_column"], "seconds": time.perf_counter() - started}
        history.append(row)
        improved = select["loss_per_step"] < best
        if improved:
            best, best_epoch, stale = select["loss_per_step"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        log(f"epoch {epoch:2d}  train {row['train_loss_per_step']:.4f}  select {row['select_loss_per_step']:.4f}"
            f"{'  (best)' if improved else ''}  {row['seconds']:.0f}s")
        if stale >= config.patience:
            break
    return TrainResult(best_state, best_epoch, history)
