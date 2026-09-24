"""The prior's readout (prior design §5): per-step negative log-likelihood per column, the steps where a word changes
(the probability of a change, the value's top-1 / top-5), the steps where none does (a change wrongly the most
likely class), and the same likelihood under two baselines counted from train:

- repeat: a column changes with its train frequency (from step 1 on; add-one smoothed), to a value with its train
  frequency among changes;
- previous word: as repeat, but the value by its train frequency after the column's previous word.

Step 0 writes every column: both baselines take its values' train frequency at step 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ts_transformer.instructions.words import COLUMNS
from ts_transformer.prior.data import Split, batches
from ts_transformer.prior.model import Prior
from ts_transformer.prior.train import TrainConfig, to_batch

TOP_K = 5


def _stacked(split: Split) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(targets, in_force, first)``: every step of the split, and whether it is a flight's step 0."""
    targets = np.concatenate([f.targets for f in split.flights]).astype(np.int64)
    in_force = np.concatenate([f.in_force for f in split.flights]).astype(np.int64)
    first = np.concatenate([np.arange(f.rows) == 0 for f in split.flights])
    return targets, in_force, first


@dataclass(frozen=True)
class Baselines:
    first: list[np.ndarray]        # per column: probability of each class at step 0
    change: np.ndarray             # [6]: probability of a change from step 1 on
    unigram: list[np.ndarray]      # per column: probability of each class given a change
    bigram: list[np.ndarray]       # per column: [previous class, class] given a change

    @classmethod
    def count(cls, split: Split) -> Baselines:
        targets, in_force, first = _stacked(split)
        later = ~first
        firsts, changes, unigrams, bigrams = [], [], [], []
        for c, classes in enumerate(split.classes):
            t = targets[:, c]
            firsts.append(_smoothed(np.bincount(t[first], minlength=classes)[1:]))
            changed = later & (t > 0)
            changes.append((changed.sum() + 1) / (later.sum() + 2))
            unigrams.append(_smoothed(np.bincount(t[changed], minlength=classes)[1:]))
            pairs = np.zeros((classes, classes - 1))
            np.add.at(pairs, (in_force[changed, c], t[changed] - 1), 1.0)
            bigrams.append((pairs + 1.0) / (pairs + 1.0).sum(axis=1, keepdims=True))
        return cls(firsts, np.array(changes), unigrams, bigrams)

    def nll_per_step(self, split: Split) -> dict[str, dict[str, float]]:
        targets, in_force, first = _stacked(split)
        out = {"repeat": {}, "previous word": {}}
        for c, name in enumerate(COLUMNS):
            t = targets[:, c]
            later = ~first
            kept = later & (t == 0)
            changed = later & (t > 0)
            common = (-np.log(self.first[c][t[first] - 1]).sum() - np.log(1.0 - self.change[c]) * kept.sum()
                      - np.log(self.change[c]) * changed.sum())
            out["repeat"][name] = float((common - np.log(self.unigram[c][t[changed] - 1]).sum()) / len(t))
            out["previous word"][name] = float(
                (common - np.log(self.bigram[c][in_force[changed, c], t[changed] - 1]).sum()) / len(t))
        for scores in out.values():
            scores["all"] = sum(scores[name] for name in COLUMNS)
        return out


def _smoothed(counts: np.ndarray) -> np.ndarray:
    return (counts + 1.0) / (counts.sum() + len(counts))


@torch.no_grad()
def model_readout(model: Prior, split: Split, config: TrainConfig, device: torch.device) -> dict[str, Any]:
    model.eval()
    classes = split.classes
    nll = np.zeros(6)
    steps = 0
    change_steps, change_prob, top1, top_k = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    kept_steps, false_change = np.zeros(6), np.zeros(6)
    for indices in batches(split.flights, config.tokens_per_batch, None):
        batch = to_batch(split, indices, device)
        logits = model(batch["features"], batch["in_force"], batch["since"], batch["airport"], batch["padding"])
        real = ~batch["padding"]
        steps += int(real.sum())
        for c, logit in enumerate(logits):
            log_p = torch.log_softmax(logit[real], dim=-1)
            target = batch["targets"][..., c][real]
            nll[c] += float(-log_p.gather(1, target[:, None]).sum())
            changed = target > 0
            change_steps[c] += int(changed.sum())
            change_prob[c] += float((1.0 - log_p[changed, 0].exp()).sum())
            ranked = log_p[changed, 1:].topk(min(TOP_K, classes[c] - 1), dim=-1).indices + 1
            top1[c] += int((ranked[:, 0] == target[changed]).sum())
            top_k[c] += int((ranked == target[changed][:, None]).any(dim=1).sum())
            kept_steps[c] += int((~changed).sum())
            false_change[c] += int((log_p[~changed].argmax(dim=-1) != 0).sum())
    per_column = {}
    for c, name in enumerate(COLUMNS):
        per_column[name] = {"nll_per_step": nll[c] / steps, "change_steps": int(change_steps[c]),
                            "mean_change_probability_where_changed": change_prob[c] / change_steps[c],
                            "top1_given_change": top1[c] / change_steps[c],
                            f"top{TOP_K}_given_change": top_k[c] / change_steps[c],
                            "false_change_share_where_kept": false_change[c] / kept_steps[c]}
    return {"steps": steps, "nll_per_step": float(nll.sum() / steps), "perplexity_per_step": float(np.exp(nll.sum() / steps)),
            "per_column": per_column}
