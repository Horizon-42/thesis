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

from ts_transformer.instructions.words import COLUMNS, RUNWAY
from ts_transformer.prior.batch import RowTensors, SentenceRows, collate, require_words
from ts_transformer.prior.model import Prior
from ts_transformer.prior.runs import RunData
from ts_transformer.prior.speaker import Permitted


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


def masked_log_probability(model: Prior, rows: RowTensors, permitted: Permitted, extra: Any = None) -> torch.Tensor:
    """``[B, R, 5]``: the log-probability of the words ``rows.targets`` at each asked row, under the speaker's records
    ``permitted`` (`speaker.Permitted`: the words every mask permitted at each row it said, the rows' times, the
    temperature), teacher-forced with the caller's input of the added modules ``extra``, with gradients (D96 item 3); 0
    at a row that is not asked. Each asked row reads the record of the row said at its time, refused when there is
    none or when its own-state inputs are not the record's (records of another aircraft or another row); the caller
    keeps records and rows of the same aircraft in the same order. A record narrower than the batch's classes permits none of
    the extra classes, one wider is refused unless its extra classes are all blocked. At the parameters that spoke, the
    probability is the one the speaker drew the word from (within the float tolerance); a word the record blocks has
    probability 0."""
    count, length = rows.asked.shape
    if permitted.time_s.shape[0] != count:
        raise ValueError(f"records of {permitted.time_s.shape[0]} aircraft for a batch of {count}")
    asked, times, own = rows.asked.cpu().numpy(), rows.time_s.cpu().numpy(), rows.own.cpu().numpy()
    said = []                                   # for each aircraft, (its asked rows, the record row of each)
    for b in range(count):
        where = np.flatnonzero(asked[b])
        at = {float(t): k for k, t in enumerate(permitted.time_s[b])}
        missing = [float(times[b, r]) for r in where if float(times[b, r]) not in at]
        if missing:
            raise ValueError(f"aircraft {b}: no record of the rows at {missing[:3]} s")
        rows_said = np.array([at[float(times[b, r])] for r in where], dtype=np.int64)
        if not np.array_equal(own[b, where], permitted.own[b, rows_said]):
            raise ValueError(f"aircraft {b}: its rows' inputs are not those its records were said with (another "
                             f"aircraft's records, or another sentence)")
        said.append((where, rows_said))
    logits = model(rows, extra)
    out = logits[0].new_zeros((count, length, len(logits)))
    for column, logit in enumerate(logits):
        width, stored = logit.shape[-1], permitted.masks[column]
        if stored.shape[-1] > width and stored[..., width:].any():
            raise ValueError(f"column {COLUMNS[column]}: the records permit a class past the batch's {width}")
        stored = np.pad(stored[..., :width], ((0, 0), (0, 0), (0, max(0, width - stored.shape[-1]))))
        record = np.ones((count, length, width), dtype=bool)
        for b, (where, rows_said) in enumerate(said):
            record[b, where] = stored[b, rows_said]
        masked = logit.masked_fill(~torch.as_tensor(record, device=logit.device), float("-inf"))
        log_p = torch.log_softmax(masked / permitted.temperature, dim=-1)
        chosen = log_p.gather(-1, rows.targets[..., column: column + 1])[..., 0]
        out[..., column] = torch.where(rows.asked, chosen, torch.zeros_like(chosen))
    return out


@torch.no_grad()
def first_step_runway(model: Prior, sentences: Sequence[SentenceRows], tokens_per_batch: int, device: torch.device
                      ) -> dict[str, Any]:
    """The runway word of the first predicted step (§5: a readout of a fold, not used for the choice): of
    ``sentences``, how many the model's most probable runway class there (the head's own, teacher-forced up to that
    row; the runway head reads no column before it) is the labelled one."""
    model.eval()
    right = 0
    for group in length_groups([s.rows for s in sentences], tokens_per_batch, None):
        rows = collate([sentences[i] for i in group], device)
        runway = model(rows)[RUNWAY]
        right += int((runway[rows.first].argmax(dim=-1) == rows.targets[..., RUNWAY][rows.first]).sum())
    return {"sentences": len(sentences), "top1": right, "share": right / len(sentences)}


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


def largest_batches(sentences: Sequence[SentenceRows], tokens_per_batch: int) -> list[list[int]]:
    """The batches of `length_groups` that may need the most memory: the one with the most padded row-candidates
    (the candidate tokens) and the one of the longest sentences (the time attention, whose cost can grow with the
    square of the rows) — one batch when they are the same."""
    groups = length_groups([s.rows for s in sentences], tokens_per_batch, None)
    most = max(groups, key=lambda group: max(sentences[i].rows for i in group) * len(group)
               * max(sentences[i].candidates.shape[1] for i in group))
    longest = max(groups, key=lambda group: max(sentences[i].rows for i in group))
    return [most] if longest == most else [most, longest]


def largest_batch_memory(model: Prior, sentences: Sequence[SentenceRows], config: TrainConfig, device: torch.device
                         ) -> dict[str, Any]:
    """One teacher-forced forward and backward pass of each of `largest_batches`, the gradients cleared: the peak GPU
    memory the allocator reserved on a CUDA ``device`` (None on the CPU), the memory the training adds on top (AdamW's
    two moments, the gradients and the copy of the best state: four times the parameters) and each batch's shape — the
    check at the formal size before a run (§12 B3)."""
    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    shapes = []
    for group in largest_batches(sentences, config.tokens_per_batch):
        rows = collate([sentences[i] for i in group], device)
        nll, asked = batch_nll(model, rows)
        (nll.sum() / asked).backward()
        model.zero_grad(set_to_none=True)
        shapes.append({"sentences": len(group), "rows": int(rows.present.shape[1]),
                       "candidates": int(rows.valid.shape[1])})
    peak = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_reserved(device))
    state = 4 * sum(p.numel() * p.element_size() for p in model.parameters())
    return {"batches": shapes, "gpu_peak_reserved_bytes": peak, "training_state_bytes": state}
