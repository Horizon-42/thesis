"""Training the prior (prior design §7): teacher forcing on the training flights, early stopping on the val split's
negative log-likelihood per predicted step, the best state kept. The loss is the six columns' cross entropy summed
over the aircraft-steps the prior speaks at (`model.asked_entries`: from the first predicted step on) and the columns
asked there (`Flight.asked`), per such step — a column left out drops from the sum, not from the count of steps, so a
closed-loop batch weighs a step the same whichever of its columns are asked. Design §9 step 1: every scene is one
flight.

Closed-loop fine-tuning (design §9.2) takes the same step on chains (`FineTuner`): one pass at a time, from the weights
it is given, at its own learning rate."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.prior.data import Split, batches
from ts_transformer.prior.model import Prior, asked_entries, self_edges


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


def to_batch(split: Split, indices: Sequence[int], device: torch.device) -> dict[str, torch.Tensor]:
    """The flights at ``indices`` as single-aircraft scenes (``A = 1``), padded to the longest and to the split's
    candidate slots: features, relative, static, in_force, since, targets, asked, airport, present, edges."""
    flights = [split.flights[i] for i in indices]
    rows = max(f.rows for f in flights)
    slots = split.candidates.shape[1]
    count = len(flights)
    features = np.zeros((count, 1, rows, flights[0].features.shape[1]), dtype=np.float32)
    relative = np.zeros((count, 1, rows, slots, flights[0].relative.shape[2]), dtype=np.float32)
    static = np.stack([f.static for f in flights])[:, None, :].astype(np.float32)
    in_force = np.zeros((count, 1, rows, 6), dtype=np.int64)
    since = np.zeros((count, 1, rows, 6), dtype=np.float32)
    targets = np.zeros((count, 1, rows, 6), dtype=np.int64)
    asked = np.zeros((count, 1, rows, 6), dtype=bool)
    present = np.zeros((count, 1, rows), dtype=bool)
    for b, f in enumerate(flights):
        features[b, 0, : f.rows], in_force[b, 0, : f.rows], since[b, 0, : f.rows] = f.features, f.in_force, f.since
        relative[b, 0, : f.rows, : f.relative.shape[1]] = f.relative
        targets[b, 0, : f.rows], asked[b, 0, : f.rows], present[b, 0, : f.rows] = f.targets, f.asked, True
    tensors = {"features": features, "relative": relative, "static": static, "in_force": in_force, "since": since,
               "targets": targets, "asked": asked, "present": present,
               "airport": np.array([f.airport for f in flights], dtype=np.int64)}
    batch = {name: torch.as_tensor(value, device=device) for name, value in tensors.items()}
    batch["edges"] = self_edges(count, 1, rows, device)
    return batch


def batch_logits(model: Prior, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Teacher forcing: the heads see the truth's classes of the earlier columns."""
    return model(batch["features"], batch["relative"], batch["static"], batch["in_force"], batch["since"],
                 batch["airport"], batch["present"], batch["edges"], batch["targets"])


def column_nll(logits: list[torch.Tensor], targets: torch.Tensor, present: torch.Tensor,
               asked: torch.Tensor) -> torch.Tensor:
    """Summed negative log-likelihood per column over the aircraft-steps the prior speaks at and the columns asked
    there (``asked``, ``[B, A, T, 6]``): [6]."""
    speaks = asked_entries(present)
    columns = []
    for c, logit in enumerate(logits):
        entries = speaks & asked[..., c]
        columns.append(nn.functional.cross_entropy(logit[entries], targets[..., c][entries], reduction="sum"))
    return torch.stack(columns)


def batch_nll(model: Prior, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """``(summed NLL per column [6], the aircraft-steps the prior speaks at)`` of one batch."""
    nll = column_nll(batch_logits(model, batch), batch["targets"], batch["present"], batch["asked"])
    return nll, int(asked_entries(batch["present"]).sum())


@torch.no_grad()
def evaluate(model: Prior, split: Split, config: TrainConfig, device: torch.device) -> dict[str, Any]:
    """The split's negative log-likelihood per predicted step, in all and per column."""
    model.eval()
    total, steps = torch.zeros(6, dtype=torch.float64, device=device), 0
    for indices in batches(split.flights, config.tokens_per_batch, None):
        nll, speaks = batch_nll(model, to_batch(split, indices, device))
        total += nll.double()
        steps += speaks
    per_column = (total / steps).cpu().numpy()
    return {"nll_per_step": float(per_column.sum()), "per_column": per_column.tolist(), "steps": steps}


def _step(model: Prior, batch: dict[str, torch.Tensor], optimiser: torch.optim.Optimizer,
          schedule: torch.optim.lr_scheduler.LRScheduler, clip_norm: float, where: str) -> tuple[float, int]:
    """One optimiser step on ``batch``: ``(its loss per aircraft-step, its aircraft-steps)``."""
    nll, speaks = batch_nll(model, batch)
    loss = nll.sum() / speaks
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{where}: the loss is {float(loss)} (a label under a -inf mask?)")
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimiser.step()
    schedule.step()
    return float(loss.detach()), speaks


def train(model: Prior, train_split: Split, val_split: Split, config: TrainConfig, device: torch.device,
          log: Callable[[str], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train until the val likelihood has not improved for ``patience`` epochs; returns the best state and the
    per-epoch history. The seed sets the batch order and dropout (the caller seeds the initial weights with it)."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(1.0, (step + 1) / config.warmup_steps))
    best, best_state, stale, history = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        total, steps = 0.0, 0
        for indices in batches(train_split.flights, config.tokens_per_batch, rng):
            loss, speaks = _step(model, to_batch(train_split, indices, device), optimiser, schedule, config.clip_norm,
                                 f"epoch {epoch}")
            total += loss * speaks
            steps += speaks
        val = evaluate(model, val_split, config, device)
        if not np.isfinite(val["nll_per_step"]):
            raise FloatingPointError(f"epoch {epoch}: the val NLL is {val['nll_per_step']}")
        row = {"epoch": epoch, "train_nll_per_step": total / steps, "val_nll_per_step": val["nll_per_step"],
               "val_per_column": val["per_column"], "seconds": time.perf_counter() - started}
        history.append(row)
        improved = val["nll_per_step"] < best
        if improved:
            best, best_state, stale = val["nll_per_step"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        log(f"epoch {epoch:2d}  train {row['train_nll_per_step']:.4f}  val {row['val_nll_per_step']:.4f}"
            f"{'  (best)' if improved else ''}  {row['seconds']:.0f}s")
        if stale >= config.patience:
            break
    return best_state, history


@dataclass(frozen=True)
class FineTuneConfig:
    """Closed-loop fine-tuning's optimiser (design §9.2, §10): AdamW, a third of pretraining's learning rate."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    clip_norm: float = 1.0
    tokens_per_batch: int = 16_384


class FineTuner:
    """The model's optimiser across closed-loop rounds: `one_pass` trains one pass over the flights it is given (in
    length buckets, shuffled), the optimiser's state and the warm-up carried from pass to pass. The seed sets the batch
    order and dropout."""

    def __init__(self, model: Prior, config: FineTuneConfig, device: torch.device, *, seed: int) -> None:
        torch.manual_seed(seed)
        self.model, self.config, self.device = model, config, device
        self.rng = np.random.default_rng(seed)
        self.optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
        self.schedule = torch.optim.lr_scheduler.LambdaLR(
            self.optimiser, lambda step: min(1.0, (step + 1) / config.warmup_steps))
        self.passes = 0

    def one_pass(self, split: Split) -> dict[str, Any]:
        """One pass over ``split``: its negative log-likelihood per aircraft-step as trained, and what it took."""
        self.passes += 1
        self.model.train()
        started = time.perf_counter()
        total, steps, count = 0.0, 0, 0
        for indices in batches(split.flights, self.config.tokens_per_batch, self.rng):
            loss, speaks = _step(self.model, to_batch(split, indices, self.device), self.optimiser, self.schedule,
                                 self.config.clip_norm, f"pass {self.passes}")
            total += loss * speaks
            steps += speaks
            count += 1
        self.model.eval()
        return {"nll_per_step": total / steps, "steps": steps, "batches": count,
                "seconds": time.perf_counter() - started}
