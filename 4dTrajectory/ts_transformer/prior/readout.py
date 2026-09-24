"""The prior's readout (prior design §5; the second version's basis §8.3): at step 0 only the runway counts — the
other columns are the hand-over's, given (`model.predicted_entries`) — and from step 1 every column.

Per column, over the entries the prior is asked for: the negative log-likelihood per step (and its part from step 1
on); the steps where a word is said (the probability of a change, the value's top-1 / top-5); the steps where none is
(a change wrongly the most likely class). The runway's choice at step 0 on its own: its likelihood per flight, top-1
and top-2, per airport. The same likelihood under two baselines counted from the training flights:

- repeat: a column changes with its training frequency (from step 1 on; add-one smoothed), to a value with its
  training frequency among changes;
- previous word: as repeat, but the value by its training frequency after the column's previous word.

Both give the runway at step 0 its training frequency at step 0, pooled over the airports' slots (the first
version's rule); the runway's step-0 block adds a per-airport reference — each airport's own runway frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ts_transformer.instructions.words import COLUMNS, RUNWAY
from ts_transformer.prior.data import Split, batches
from ts_transformer.prior.model import Prior, predicted_entries
from ts_transformer.prior.train import TrainConfig, batch_logits, to_batch

TOP_K = 5


def _stacked(split: Split) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(targets, in_force, first)``: every step of the split, and whether it is a flight's step 0."""
    targets = np.concatenate([f.targets for f in split.flights]).astype(np.int64)
    in_force = np.concatenate([f.in_force for f in split.flights]).astype(np.int64)
    first = np.concatenate([np.arange(f.rows) == 0 for f in split.flights])
    return targets, in_force, first


def _smoothed(counts: np.ndarray) -> np.ndarray:
    return (counts + 1.0) / (counts.sum() + len(counts))


@dataclass(frozen=True)
class Baselines:
    first_runway: np.ndarray       # probability of each runway slot at step 0, pooled over the airports
    change: np.ndarray             # [6]: probability of a change from step 1 on
    unigram: list[np.ndarray]      # per column: probability of each value given a change
    bigram: list[np.ndarray]       # per column: [previous class, value] given a change
    airport_runway: np.ndarray     # [A, slots]: each airport's runway frequency at step 0 over its own candidates

    @classmethod
    def count(cls, split: Split) -> Baselines:
        targets, in_force, first = _stacked(split)
        later = ~first
        changes, unigrams, bigrams = [], [], []
        for c, classes in enumerate(split.classes):
            t = targets[:, c]
            changed = later & (t > 0)
            changes.append((changed.sum() + 1) / (later.sum() + 2))
            unigrams.append(_smoothed(np.bincount(t[changed], minlength=classes)[1:]))
            pairs = np.zeros((classes, classes - 1))
            np.add.at(pairs, (in_force[changed, c], t[changed] - 1), 1.0)
            bigrams.append((pairs + 1.0) / (pairs + 1.0).sum(axis=1, keepdims=True))
        slots = split.candidates.shape[1]
        chosen = np.array([f.targets[0, RUNWAY] - 1 for f in split.flights])
        airport = np.array([f.airport for f in split.flights])
        per_airport = np.zeros((len(split.airports), slots))
        for a in range(len(split.airports)):
            valid = split.candidates[a, :, -1] > 0.5
            per_airport[a, valid] = _smoothed(np.bincount(chosen[airport == a], minlength=slots)[valid])
        return cls(_smoothed(np.bincount(chosen, minlength=slots)), np.array(changes), unigrams, bigrams, per_airport)

    def nll_per_step(self, split: Split) -> dict[str, dict[str, float]]:
        targets, in_force, first = _stacked(split)
        later = ~first
        out: dict[str, dict[str, float]] = {"repeat": {}, "previous word": {}}
        for c, name in enumerate(COLUMNS):
            t = targets[:, c]
            kept = later & (t == 0)
            changed = later & (t > 0)
            common = -np.log(1.0 - self.change[c]) * kept.sum() - np.log(self.change[c]) * changed.sum()
            if c == RUNWAY:
                common -= np.log(self.first_runway[t[first] - 1]).sum()
            out["repeat"][name] = float((common - np.log(self.unigram[c][t[changed] - 1]).sum()) / len(t))
            out["previous word"][name] = float(
                (common - np.log(self.bigram[c][in_force[changed, c], t[changed] - 1]).sum()) / len(t))
        for scores in out.values():
            scores["all"] = sum(scores[name] for name in COLUMNS)
        return out

    def airport_runway_reference(self, split: Split) -> dict[str, Any]:
        """Each airport's own step-0 runway frequency on ``split``: its likelihood per flight and top-1."""
        chosen = np.array([f.targets[0, RUNWAY] - 1 for f in split.flights])
        airport = np.array([f.airport for f in split.flights])
        probability = self.airport_runway[airport, chosen]
        hit = self.airport_runway[airport].argmax(axis=1) == chosen
        return {"nll_per_flight": float(-np.log(probability).mean()), "top1": float(hit.mean()),
                "per_airport": {code: {"flights": int((airport == a).sum()), "top1": float(hit[airport == a].mean())}
                                for a, code in enumerate(split.airports) if (airport == a).any()}}


@torch.no_grad()
def model_readout(model: Prior, split: Split, config: TrainConfig, device: torch.device) -> dict[str, Any]:
    model.eval()
    classes = split.classes
    nll, nll_later = np.zeros(6), np.zeros(6)
    steps = 0
    change_steps, change_prob, top1, top_k = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    kept_steps, false_change = np.zeros(6), np.zeros(6)
    runway_nll, runway_hits, runway_top2, runway_airport = [], [], [], []
    for indices in batches(split.flights, config.tokens_per_batch, None):
        batch = to_batch(split, indices, device)
        logits = batch_logits(model, batch)
        asked = predicted_entries(batch["padding"])
        rows = asked.shape[1]
        later_rows = (torch.arange(rows, device=device) >= 1)[None].expand(asked.shape[:2])
        steps += int((~batch["padding"]).sum())
        for c, logit in enumerate(logits):
            entries = asked[..., c]
            log_p = torch.log_softmax(logit[entries], dim=-1)
            target = batch["targets"][..., c][entries]
            truth = -log_p.gather(1, target[:, None])[:, 0]
            nll[c] += float(truth.sum())
            nll_later[c] += float(truth[later_rows[entries]].sum())
            changed = target > 0
            change_steps[c] += int(changed.sum())
            change_prob[c] += float((1.0 - log_p[changed, 0].exp()).sum())
            ranked = log_p[changed, 1:].topk(min(TOP_K, classes[c] - 1), dim=-1).indices + 1
            top1[c] += int((ranked[:, 0] == target[changed]).sum())
            top_k[c] += int((ranked == target[changed][:, None]).any(dim=1).sum())
            kept_steps[c] += int((~changed).sum())
            false_change[c] += int((log_p[~changed].argmax(dim=-1) != 0).sum())
        first = torch.log_softmax(logits[RUNWAY][:, 0], dim=-1)                  # [B, classes]: step 0's runway
        target = batch["targets"][:, 0, RUNWAY]
        runway_nll += (-first.gather(1, target[:, None])[:, 0]).tolist()
        best = first.topk(2, dim=-1).indices
        runway_hits += (best[:, 0] == target).tolist()
        runway_top2 += (best == target[:, None]).any(dim=1).tolist()
        runway_airport += batch["airport"].tolist()
    per_column = {}
    for c, name in enumerate(COLUMNS):
        per_column[name] = {"nll_per_step": nll[c] / steps, "nll_from_step1_per_step": nll_later[c] / steps,
                            "change_steps": int(change_steps[c]),
                            "mean_change_probability_where_changed": change_prob[c] / change_steps[c],
                            "top1_given_change": top1[c] / change_steps[c],
                            f"top{TOP_K}_given_change": top_k[c] / change_steps[c],
                            "false_change_share_where_kept": false_change[c] / kept_steps[c]}
    hits, airport = np.array(runway_hits), np.array(runway_airport)
    step0 = {"flights": len(hits), "nll_per_flight": float(np.mean(runway_nll)), "top1": float(hits.mean()),
             "top2": float(np.mean(runway_top2)),
             "per_airport": {code: {"flights": int((airport == a).sum()), "top1": float(hits[airport == a].mean())}
                             for a, code in enumerate(split.airports) if (airport == a).any()}}
    return {"steps": steps, "flights": len(split.flights), "nll_per_step": float(nll.sum() / steps),
            "perplexity_per_step": float(np.exp(nll.sum() / steps)),
            "nll_from_step1_per_step": float(nll_later.sum() / steps), "per_column": per_column,
            "step0_runway": step0}
