"""Training the prior (prior design §4): teacher forcing on the train split, early stopping on the val split's
negative log-likelihood per step, the best state kept."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.prior.data import Split, batches
from ts_transformer.prior.model import Prior


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
    """The flights at ``indices``, padded to the longest: features, in_force, since, targets, airport, padding."""
    flights = [split.flights[i] for i in indices]
    rows = max(f.rows for f in flights)
    features = np.zeros((len(flights), rows, flights[0].features.shape[1]), dtype=np.float32)
    in_force = np.zeros((len(flights), rows, 6), dtype=np.int64)
    since = np.zeros((len(flights), rows, 6), dtype=np.float32)
    targets = np.zeros((len(flights), rows, 6), dtype=np.int64)
    padding = np.ones((len(flights), rows), dtype=bool)
    for b, f in enumerate(flights):
        features[b, : f.rows], in_force[b, : f.rows], since[b, : f.rows] = f.features, f.in_force, f.since
        targets[b, : f.rows], padding[b, : f.rows] = f.targets, False
    tensors = {"features": features, "in_force": in_force, "since": since, "targets": targets, "padding": padding,
               "airport": np.array([f.airport for f in flights], dtype=np.int64)}
    return {name: torch.as_tensor(value, device=device) for name, value in tensors.items()}


def column_nll(logits: list[torch.Tensor], targets: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    """Summed negative log-likelihood per column over the real steps: [6]."""
    real = ~padding
    return torch.stack([nn.functional.cross_entropy(logit[real], targets[..., c][real], reduction="sum")
                        for c, logit in enumerate(logits)])


@torch.no_grad()
def evaluate(model: Prior, split: Split, config: TrainConfig, device: torch.device) -> dict[str, Any]:
    """The split's negative log-likelihood per step, in all and per column."""
    model.eval()
    total, steps = torch.zeros(6, dtype=torch.float64, device=device), 0
    for indices in batches(split.flights, config.tokens_per_batch, None):
        batch = to_batch(split, indices, device)
        logits = model(batch["features"], batch["in_force"], batch["since"], batch["airport"], batch["padding"])
        total += column_nll(logits, batch["targets"], batch["padding"]).double()
        steps += int((~batch["padding"]).sum())
    per_column = (total / steps).cpu().numpy()
    return {"nll_per_step": float(per_column.sum()), "per_column": per_column.tolist(), "steps": steps}


def train(model: Prior, train_split: Split, val_split: Split, config: TrainConfig, device: torch.device,
          log: Callable[[str], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train until the val likelihood has not improved for ``patience`` epochs; returns the best state and the
    per-epoch history."""
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
            batch = to_batch(train_split, indices, device)
            logits = model(batch["features"], batch["in_force"], batch["since"], batch["airport"], batch["padding"])
            real = int((~batch["padding"]).sum())
            loss = column_nll(logits, batch["targets"], batch["padding"]).sum() / real
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimiser.step()
            schedule.step()
            total += float(loss.detach()) * real
            steps += real
        val = evaluate(model, val_split, config, device)
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
