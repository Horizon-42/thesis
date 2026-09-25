"""Training the prior (prior design §7): teacher forcing on the training flights, early stopping on the val split's
negative log-likelihood per predicted step, the best state kept. The loss is the six columns' cross entropy summed
over the aircraft-steps the prior speaks at (`model.asked_entries`: from the first predicted step on) and the columns
asked there (`Flight.asked`), per such step — a column left out drops from the sum, not from the count of steps, so a
closed-loop batch weighs a step the same whichever of its columns are asked. Design §9 step 1: every scene is one
flight.

Closed-loop fine-tuning (design §9.2) takes the same step on chains (`FineTuner`): one pass at a time, from the weights
it is given, at its own learning rate. The landing reward (design §9.3, `RewardTuner`) weighs each sentence the prior
said by its advantage, keeps the model near a frozen reference and trains on the data beside it."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

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


def flight_nll(logits: list[torch.Tensor], targets: torch.Tensor, present: torch.Tensor,
               asked: torch.Tensor) -> torch.Tensor:
    """`column_nll` per scene rather than per column: ``[B]``, summed over the scene's asked cells."""
    speaks = asked_entries(present)
    total = logits[0].new_zeros(len(targets))
    for c, logit in enumerate(logits):
        entries = speaks & asked[..., c]
        nll = nn.functional.cross_entropy(logit[entries], targets[..., c][entries], reduction="none")
        total = total.index_add(0, entries.nonzero()[:, 0], nll)
    return total


def flight_kl(logits: list[torch.Tensor], reference: list[torch.Tensor], targets: torch.Tensor, present: torch.Tensor,
              asked: torch.Tensor) -> torch.Tensor:
    """``[B]``: how far the model is from the ``reference`` on each scene's own words, summed over its asked cells —
    per cell ``exp(r − p) − (r − p) − 1`` of the two log-probabilities of the word (an estimate of the KL divergence
    from the samples, ≥ 0, GRPO's)."""
    speaks = asked_entries(present)
    total = logits[0].new_zeros(len(targets))
    for c, (logit, fixed) in enumerate(zip(logits, reference)):
        entries = speaks & asked[..., c]
        word = targets[..., c][entries][:, None]
        gap = (torch.log_softmax(fixed[entries], dim=-1).gather(1, word)
               - torch.log_softmax(logit[entries], dim=-1).gather(1, word))[:, 0]
        total = total.index_add(0, entries.nonzero()[:, 0], torch.exp(gap) - gap - 1.0)
    return total


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


@dataclass(frozen=True)
class RewardConfig:
    """The landing reward's optimiser and weights (design §9.3, §10): a tenth of §9.2's learning rate (the reward term's
    gradient is noisy), a pull to the reference of 0.04, the data term at 1."""

    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 20
    clip_norm: float = 1.0
    tokens_per_batch: int = 16_384          # half the sentences said, half the data
    kl_weight: float = 0.04
    data_weight: float = 1.0


class RewardTuner:
    """The model's optimiser across landing-reward rounds (design §9.3). `one_pass` goes once over the sentences of a
    round — each a flight the prior spoke to, its words as targets, with its advantage — in length buckets, shuffled;
    every update pairs a batch of them with a batch of teacher-forced data flights:

        loss = mean over sentences of (advantage × the NLL of its own words, per step)
             + kl_weight × mean over sentences of (`flight_kl` to the frozen reference, per step)
             + data_weight × the data batch's NLL per step (pretraining's loss)

    — minimising the first term raises the words of a sentence that did better than its flight's others and lowers the
    words of one that did worse.

    The sentences' words are scored under the model's unmasked distribution (the masks removed ~0.06 % a step, readouts
    §5) with dropout off — the distribution they were sampled from, so the pull to the reference measures how far the
    weights moved and not dropout's noise (at the first update it is 0); one pass, no importance ratio (the sentences
    come from the weights at the start of the pass). The seed sets the batch order."""

    def __init__(self, model: Prior, reference: Prior, config: RewardConfig, device: torch.device, *, seed: int) -> None:
        torch.manual_seed(seed)
        self.model, self.reference, self.config, self.device = model, reference.eval(), config, device
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self.rng = np.random.default_rng(seed)
        self.optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
        self.schedule = torch.optim.lr_scheduler.LambdaLR(
            self.optimiser, lambda step: min(1.0, (step + 1) / config.warmup_steps))
        self.passes = 0
        self._data: Iterator[list[int]] = iter(())

    def _data_batch(self, data: Split) -> list[int]:
        """The next batch of data flights, the data reshuffled whenever it runs out."""
        indices = next(self._data, None)
        if indices is None:
            self._data = batches(data.flights, self.config.tokens_per_batch // 2, self.rng)
            indices = next(self._data)
        return indices

    def one_pass(self, sentences: Split, advantages: np.ndarray, data: Split) -> dict[str, Any]:
        """One pass over ``sentences`` (``advantages``: one per sentence) beside ``data``: the mean of each term as
        trained, and what it took."""
        if len(advantages) != len(sentences.flights):
            raise ValueError(f"{len(advantages)} advantages for {len(sentences.flights)} sentences")
        if not sentences.flights:
            raise ValueError("no sentence to train on: no flight's sentences differ in reward")
        self.passes += 1
        self.model.eval()
        started = time.perf_counter()
        sums = {"reward": 0.0, "kl": 0.0, "data": 0.0}
        count = 0
        for indices in batches(sentences.flights, self.config.tokens_per_batch // 2, self.rng):
            batch = to_batch(sentences, indices, self.device)
            logits = batch_logits(self.model, batch)
            with torch.no_grad():
                reference = batch_logits(self.reference, batch)
            steps = asked_entries(batch["present"]).sum(dim=(1, 2)).to(logits[0].dtype)
            advantage = torch.as_tensor(advantages[indices], dtype=logits[0].dtype, device=self.device)
            reward = (advantage * flight_nll(logits, batch["targets"], batch["present"], batch["asked"]) / steps).mean()
            kl = (flight_kl(logits, reference, batch["targets"], batch["present"], batch["asked"]) / steps).mean()
            nll, speaks = batch_nll(self.model, to_batch(data, self._data_batch(data), self.device))
            data_loss = nll.sum() / speaks
            loss = reward + self.config.kl_weight * kl + self.config.data_weight * data_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"pass {self.passes}: the loss is {float(loss)}")
            self.optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
            self.optimiser.step()
            self.schedule.step()
            for name, value in (("reward", reward), ("kl", kl), ("data", data_loss)):
                sums[name] += float(value.detach())
            count += 1
        return {**{f"{name}_mean": value / count for name, value in sums.items()}, "batches": count,
                "sentences": len(sentences.flights), "seconds": time.perf_counter() - started}
