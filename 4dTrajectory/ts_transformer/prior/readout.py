"""The prior's readout (prior design §8), over the steps it speaks at: the first predicted step (row `scene.N_LOOK`,
every column said) and the steps after it.

Per column: the negative log-likelihood per predicted step (over all of them; and over the steps after the first);
on the steps after the first where a word is said, the probability of a change and the value's top-1 / top-5; where
none is, a change wrongly the most likely class; at the first step, the word's top-1. The same likelihood under two
baselines counted from the training flights:

- repeat: at the first step each column's value with its training frequency there (the runway: its airport's own
  frequency over its candidates); after it a column changes with its training frequency (add-one smoothed), to a value
  with its training frequency among changes;
- previous word: as repeat, but after the first step the value by its training frequency after the column's word in
  force.

**The runway at the first predicted step** on its own: likelihood per flight, top-1 and top-2, and read in two layers
(`runway_breakdown`): the DIRECTION (the airport's landing direction, `data.runway_context.direction_groups`) and, where
the direction is right, the SIDE (which runway of the group); apart for the flights already established on the final
at that step and those not yet; per airport. The same breakdown for three causal rules (`data.runway_context`, design
§8): B0 the airport's most used runway (train-day landings), B1 the runway with the most landings in the previous 30
minutes, B3 the runway of the last landing from the same entry sector — from the tracks roster's landings less the
sealed test days (`scene.context_landings`), the sectors of the artefact's development flights (`runway_rules`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ts_transformer.data.runway_context import ContextLanding, RunwayContext, direction_groups
from ts_transformer.instructions.artefact import load_candidates, load_day_split
from ts_transformer.instructions.words import COLUMNS, RUNWAY
from ts_transformer.prior.data import Split, batches
from ts_transformer.prior.model import Prior, asked_entries
from ts_transformer.prior.scene import CONTEXT_WINDOW_S, N_LOOK, context_landings
from ts_transformer.prior.train import TrainConfig, batch_logits, to_batch

TOP_K = 5
#: The runway rules compared with the prior (design §8), and their windows: B1's is `scene.CONTEXT_WINDOW_S`, B3 looks
#: back an hour (runway intent R0's defaults).
RULES = ("B0_majority", "B1_active_config", "B3_same_sector_last")
SECTOR_WINDOW = timedelta(minutes=60)


def _stacked(split: Split) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(targets, in_force, first, later)`` over every flight's predicted steps: whether each is the first."""
    targets = np.concatenate([f.targets[N_LOOK:] for f in split.flights]).astype(np.int64)
    in_force = np.concatenate([f.in_force[N_LOOK:] for f in split.flights]).astype(np.int64)
    first = np.concatenate([np.arange(f.rows - N_LOOK) == 0 for f in split.flights])
    return targets, in_force, first, ~first


def _smoothed(counts: np.ndarray) -> np.ndarray:
    return (counts + 1.0) / (counts.sum() + len(counts))


def _share_by(hits: np.ndarray, keys: np.ndarray, names: Mapping[Any, str]) -> dict[str, dict[str, Any]]:
    return {name: {"flights": int((keys == key).sum()), "top1": float(hits[keys == key].mean())}
            for key, name in names.items() if (keys == key).any()}


def runway_breakdown(predicted: np.ndarray, truth: np.ndarray, split: Split) -> dict[str, Any]:
    """Top-1 of the first step's runway (slots), in all, by direction and side (design §8), established or not, per
    airport."""
    airport = np.array([f.airport for f in split.flights])
    established = np.array([f.established for f in split.flights])
    groups = [direction_groups(dict(zip(split.runways[a], split.courses[a]))) for a in range(len(split.airports))]
    group = lambda a, slot: groups[a][split.runways[a][slot]]                       # noqa: E731
    direction = np.array([group(a, p) == group(a, t) for a, p, t in zip(airport, predicted, truth)])
    hit = predicted == truth
    return {"flights": len(hit), "top1": float(hit.mean()), "direction": float(direction.mean()),
            "side_given_direction": float(hit[direction].mean()) if direction.any() else None,
            "by_establishment": _share_by(hit, established, {True: "established", False: "not established"}),
            "direction_by_establishment": _share_by(direction, established,
                                                    {True: "established", False: "not established"}),
            "per_airport": _share_by(hit, airport, dict(enumerate(split.airports)))}


@dataclass(frozen=True)
class Baselines:
    first: list[np.ndarray]        # per column: each value's probability at the first step (the runway's: see next)
    first_runway: np.ndarray       # [A, slots]: each airport's runway frequency at the first step over its candidates
    change: np.ndarray             # [6]: probability of a change after the first step
    unigram: list[np.ndarray]      # per column: probability of each value given a change
    bigram: list[np.ndarray]       # per column: [class in force, value] given a change

    @classmethod
    def count(cls, split: Split) -> Baselines:
        targets, in_force, first, later = _stacked(split)
        changes, unigrams, bigrams, firsts = [], [], [], []
        for c, classes in enumerate(split.classes):
            t = targets[:, c]
            changed = later & (t > 0)
            changes.append((changed.sum() + 1) / (later.sum() + 2))
            unigrams.append(_smoothed(np.bincount(t[changed], minlength=classes)[1:]))
            pairs = np.zeros((classes, classes - 1))
            np.add.at(pairs, (in_force[changed, c], t[changed] - 1), 1.0)
            bigrams.append((pairs + 1.0) / (pairs + 1.0).sum(axis=1, keepdims=True))
            firsts.append(_smoothed(np.bincount(t[first], minlength=classes)[1:]))
        slots = split.candidates.shape[1]
        chosen = targets[first, RUNWAY] - 1
        airport = np.array([f.airport for f in split.flights])
        per_airport = np.zeros((len(split.airports), slots))
        for a in range(len(split.airports)):
            valid = split.candidates[a, :, -1] > 0.5
            per_airport[a, valid] = _smoothed(np.bincount(chosen[airport == a], minlength=slots)[valid])
        return cls(firsts, per_airport, np.array(changes), unigrams, bigrams)

    def nll_per_step(self, split: Split) -> dict[str, dict[str, float]]:
        targets, in_force, first, later = _stacked(split)
        airport = np.array([f.airport for f in split.flights])
        out: dict[str, dict[str, float]] = {"repeat": {}, "previous word": {}}
        for c, name in enumerate(COLUMNS):
            t = targets[:, c]
            kept = later & (t == 0)
            changed = later & (t > 0)
            common = -np.log(1.0 - self.change[c]) * kept.sum() - np.log(self.change[c]) * changed.sum()
            if c == RUNWAY:
                common -= np.log(self.first_runway[airport, t[first] - 1]).sum()
            else:
                common -= np.log(self.first[c][t[first] - 1]).sum()
            out["repeat"][name] = float((common - np.log(self.unigram[c][t[changed] - 1]).sum()) / len(t))
            out["previous word"][name] = float(
                (common - np.log(self.bigram[c][in_force[changed, c], t[changed] - 1]).sum()) / len(t))
        for scores in out.values():
            scores["all"] = sum(scores[name] for name in COLUMNS)
        return out

    def airport_runway(self, split: Split) -> np.ndarray:
        """Each flight's first-step runway by its airport's training frequency (slots)."""
        return np.array([int(self.first_runway[f.airport].argmax()) for f in split.flights])


@dataclass(frozen=True)
class RunwayRules:
    """Each airport's causal rules, and the share of its context landings whose entry sector is known (B3's
    evidence: a landing of a flight not among the labelled development flights has none)."""

    contexts: Mapping[str, RunwayContext]
    sector_known: Mapping[str, float]


def runway_rules(directory: Path, tracks: Mapping[str, Path], splits: Mapping[str, Split]) -> RunwayRules:
    """Each airport's causal rules over its context landings (``tracks``: airport → tracks roster): the tracks
    roster's landings on its candidates less the sealed test days, each with the entry sector of its flight where the
    flight is among ``splits``' flights (the labelled development flights), else none; B0's majority from the
    train-day landings."""
    days = load_day_split(directory)
    sectors = {f.dataset_id: f.sector for split in splits.values() for f in split.flights}
    contexts, known = {}, {}
    for code, geometry in load_candidates(directory).items():
        pool = context_landings(tracks[code], [c.ident for c in geometry.candidates], days)
        keys = [f"{code}:{r.flight_key}" for r in pool.records]
        landings = [ContextLanding(datetime.fromtimestamp(r.time_s, tz=timezone.utc), r.runway,
                                   sectors[key] if key in sectors else None) for r, key in zip(pool.records, keys)]
        majority = Counter(r.runway for r in pool.records if r.split == "train")
        contexts[code] = RunwayContext(landings, {c.ident: c.course_deg for c in geometry.candidates}, majority, [],
                                       window=timedelta(seconds=CONTEXT_WINDOW_S), sector_window=SECTOR_WINDOW,
                                       metar_delay=timedelta(0), calm_kt=0.0)
        known[code] = sum(key in sectors for key in keys) / len(keys)
    return RunwayRules(contexts, known)


def rules_readout(split: Split, rules: RunwayRules) -> dict[str, Any]:
    """`RULES` at every flight's first predicted step, as `runway_breakdown`s; a rule's stated fallback is counted."""
    truth = np.array([f.targets[N_LOOK, RUNWAY] - 1 for f in split.flights])
    picks: dict[str, list[int]] = {rule: [] for rule in RULES}
    fallback: Counter = Counter()
    for flight in split.flights:
        code = split.airports[flight.airport]
        chosen = rules.contexts[code].picks(datetime.fromtimestamp(flight.first_step_s, tz=timezone.utc),
                                            sector=flight.sector, track_course_deg=flight.course_deg)
        for rule in RULES:
            picks[rule].append(split.runways[flight.airport].index(chosen[rule].runway))
            fallback[rule] += chosen[rule].fallback
    return {rule: {**runway_breakdown(np.array(picks[rule]), truth, split),
                   "fallback_share": fallback[rule] / len(split.flights)} for rule in RULES}


@torch.no_grad()
def model_readout(model: Prior, split: Split, config: TrainConfig, device: torch.device) -> dict[str, Any]:
    model.eval()
    classes = split.classes
    nll, nll_later = np.zeros(6), np.zeros(6)
    steps = 0
    change_steps, change_prob, top1, top_k = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    kept_steps, false_change, first_hits = np.zeros(6), np.zeros(6), np.zeros(6)
    runway_nll, runway_best, runway_top2 = [], [], []
    order = []
    for indices in batches(split.flights, config.tokens_per_batch, None):
        batch = to_batch(split, indices, device)
        logits = batch_logits(model, batch)
        asked = asked_entries(batch["present"])
        rows = asked.shape[-1]
        later = asked & (torch.arange(rows, device=device) > N_LOOK)
        steps += int(asked.sum())
        for c, logit in enumerate(logits):
            log_p = torch.log_softmax(logit, dim=-1)
            target = batch["targets"][..., c]
            truth = -log_p.gather(-1, target[..., None])[..., 0]
            nll[c] += float(truth[asked].sum())
            nll_later[c] += float(truth[later].sum())
            first_hits[c] += int((log_p[:, :, N_LOOK].argmax(dim=-1) == target[:, :, N_LOOK]).sum())
            changed = later & (target > 0)
            kept = later & (target == 0)
            change_steps[c] += int(changed.sum())
            change_prob[c] += float((1.0 - log_p[..., 0].exp())[changed].sum())
            ranked = log_p[changed][:, 1:].topk(min(TOP_K, classes[c] - 1), dim=-1).indices + 1
            top1[c] += int((ranked[:, 0] == target[changed]).sum())
            top_k[c] += int((ranked == target[changed][:, None]).any(dim=1).sum())
            kept_steps[c] += int(kept.sum())
            false_change[c] += int((log_p[kept].argmax(dim=-1) != 0).sum())
        first = torch.log_softmax(logits[RUNWAY][:, 0, N_LOOK], dim=-1)             # [B, classes]
        target = batch["targets"][:, 0, N_LOOK, RUNWAY]
        runway_nll += (-first.gather(1, target[:, None])[:, 0]).tolist()
        best = first.topk(2, dim=-1).indices
        runway_best += (best[:, 0] - 1).tolist()
        runway_top2 += (best == target[:, None]).any(dim=1).tolist()
        order += list(indices)
    flights = len(split.flights)
    later_steps = steps - flights
    per_column = {}
    for c, name in enumerate(COLUMNS):
        # a column that never changes after the first step (the runway, in the labelled sentences) has no change
        # metrics: None, not a division by zero
        changes = change_steps[c] or None
        per_column[name] = {"nll_per_step": nll[c] / steps, "nll_after_first_per_step": nll_later[c] / later_steps,
                            "first_step_top1": first_hits[c] / flights,
                            "change_steps": int(change_steps[c]),
                            "mean_change_probability_where_changed": changes and change_prob[c] / changes,
                            "top1_given_change": changes and top1[c] / changes,
                            f"top{TOP_K}_given_change": changes and top_k[c] / changes,
                            "false_change_share_where_kept": false_change[c] / kept_steps[c]}
    back = np.argsort(order)                                                     # batch order → the split's order
    predicted = np.array(runway_best)[back]
    truth = np.array([f.targets[N_LOOK, RUNWAY] - 1 for f in split.flights])
    first_runway = {**runway_breakdown(predicted, truth, split), "nll_per_flight": float(np.mean(runway_nll)),
                    "top2": float(np.mean(runway_top2))}
    return {"steps": steps, "flights": flights, "nll_per_step": float(nll.sum() / steps),
            "perplexity_per_step": float(np.exp(nll.sum() / steps)),
            "nll_after_first_per_step": float(nll_later.sum() / later_steps), "per_column": per_column,
            "first_step_runway": first_runway}


def full_readout(model: Prior, split: Split, baselines: Baselines, rules: RunwayRules,
                 config: TrainConfig, device: torch.device) -> dict[str, Any]:
    """The model, the two baselines, the airport's own runway frequency and the runway rules on ``split`` (with each
    airport's share of context landings whose entry sector is known, B3's evidence)."""
    truth = np.array([f.targets[N_LOOK, RUNWAY] - 1 for f in split.flights])
    return {"model": model_readout(model, split, config, device), "baselines": baselines.nll_per_step(split),
            "airport_runway_frequency": runway_breakdown(baselines.airport_runway(split), truth, split),
            "rules": rules_readout(split, rules), "rules_sector_known_share": dict(rules.sector_known)}
