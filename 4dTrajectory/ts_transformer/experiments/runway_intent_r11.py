"""Runway-intent R1.1: a candidate-symmetric runway head against R1's per-runway head, on the same samples.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §14 (the design and the gates, written before this
ran), after §13: R1's head keeps one column per runway, so "the recent landings went to X" is learned
for each X from the training days alone, and it read the raw wind and the time of day as a
fingerprint of the usual configuration — on KSJC's two 30L-closure days it read 55-57 % against B1's
97-98 %. R1.1 re-reads every R1 sample (`runway_intent_r1.build_samples`) as one row per candidate
runway in that runway's own terms (`data.runway_features.candidate_rows`), and a gradient-boosted
CONDITIONAL LOGIT (`ListwiseBooster`) scores the rows with ONE set of trees and takes a softmax
across the sample's candidates: a rule learned on one runway applies to all of them. R1's head is
retrained here on the same samples with its own settings — its numbers are checked against R1's
artifact — so every comparison is paired.

Three trainings per head, as in R1: ``day_a`` / ``day_b`` (the operating-day partitions) and
``flight`` (the per-flight split, the leakage control, read paired with ``day_a``).

R1.1b (plan §15.3) adds two heads on the same table: R1.1's rows still re-identify a runway through
two per-runway constants — its training share (``prior_share``) and the operator's share of its
landings — and on the closure days the trees trusted them over the recent landings. ``r11_noid``
drops both; ``r11_lift`` drops the prior and moves the operator's share to its deviation from it.

    python run_ts.py runway_intent_r11 --airport KSJC \\
        --output-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r11_20260913/KSJC
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ts_transformer.data.runway_features import (
    CANDIDATE_ROW_GROUPS,
    CANDIDATE_ROW_NAMES,
    ENTRY,
    candidate_rows,
)
from ts_transformer.experiments.runway_intent_r1 import (
    HGB_SETTINGS,
    PERMUTATION_REPEATS,
    _fraction,
    add_sample_arguments,
    build_samples,
    by_day,
    level_accuracy,
    score,
)
from ts_transformer.experiments.support import REPO_ROOT

# v2: R1.1b's identity-free heads beside R1.1's (plan §15.3)
SCHEMA = "ts-runway-intent-r11-v2"
PARTITIONS = ("day_a", "day_b", "flight")
#: The candidate-symmetric heads — R1.1 as pre-registered (§14) and R1.1b's two variants (§15.3),
#: each a selection / transform of the one candidate table (`head_table`) — and R1's head.
SYMMETRIC_HEADS = ("r11", "r11_noid", "r11_lift")
HEADS = SYMMETRIC_HEADS + ("r1",)
#: The two columns that carry a runway's base rate, i.e. which runway a row is (§15.2).
IDENTITY_COLUMNS = ("prior_share", "airline_share")
HEAD_GROUPS = {**CANDIDATE_ROW_GROUPS, "airline_lift": "airline"}
#: R1's budget and tree shape (`HGB_SETTINGS`), so the two heads differ in the representation only.
HEAD_SETTINGS = dict(n_iter=HGB_SETTINGS["max_iter"], learning_rate=HGB_SETTINGS["learning_rate"],
                     max_leaf_nodes=HGB_SETTINGS["max_leaf_nodes"],
                     l2_regularization=HGB_SETTINGS["l2_regularization"], min_samples_leaf=20)
#: p(1 − p) of a candidate the head is sure of underflows the Newton weight; floored here.
HESSIAN_FLOOR = 1e-6
#: A child needs at least this hessian mass (sklearn's `min_hessian_to_split`).
MIN_CHILD_HESSIAN = 1e-3
#: Feature values are binned once, on the training rows, into at most this many bins (sklearn's).
MAX_BINS = 255
#: The airline share: training days hashed into this many blocks (out of fold), and this many
#: pseudo-landings placed at the partition's prior.
AIRLINE_BLOCKS = 5
AIRLINE_PRIOR_WEIGHT = 10.0
DEFAULT_R1_CAMPAIGN = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments" / "runway_intent_r1_20260913"


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = np.exp(scores - scores.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class _Tree:
    """One histogram tree, as arrays over its nodes (node 0 the root): an internal node sends a row
    left when its bin of ``feature`` is at most ``bin``; a leaf carries ``value``."""

    feature: np.ndarray
    bin: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    is_leaf: np.ndarray

    def predict(self, binned: np.ndarray) -> np.ndarray:
        node = np.zeros(len(binned), dtype=np.int64)
        active = np.arange(len(binned))
        while active.size:
            here = node[active]
            inner = ~self.is_leaf[here]
            active, here = active[inner], here[inner]
            left = binned[active, self.feature[here]] <= self.bin[here]
            node[active] = np.where(left, self.left[here], self.right[here])
        return self.value[node]


class ListwiseBooster:
    """A gradient-boosted conditional logit. Every (sample, candidate) row gets a score from ONE
    set of trees, a sample's probabilities are the softmax of its candidates' scores, and each step
    is a Newton step on that listwise log-loss (``g = p - y``, ``h = p(1 - p)``): a histogram tree
    grown leaf-wise to ``max_leaf_nodes`` on the second-order split gain ``G²/(H + λ)``, each leaf
    worth ``-G/(H + λ)`` — the tree sklearn's HistGradientBoosting grows, with the gradient of this
    loss, which sklearn cannot take (a one-iteration sklearn regressor per step reproduced it but
    spent its time in per-call overhead: KSMF, the smallest airport, took over 10 minutes)."""

    def __init__(self, *, n_iter: int, learning_rate: float, max_leaf_nodes: int,
                 l2_regularization: float, min_samples_leaf: int) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.min_samples_leaf = min_samples_leaf
        self.edges: list[np.ndarray] = []
        self.trees: list[_Tree] = []
        self.train_scores_: np.ndarray | None = None

    def _fit_bins(self, rows: np.ndarray) -> None:
        self.edges = []
        for column in rows.T:
            values = np.unique(column)
            if len(values) <= MAX_BINS:
                cuts = (values[:-1] + values[1:]) / 2.0
            else:
                cuts = np.unique(np.quantile(column, np.linspace(0.0, 1.0, MAX_BINS + 1)[1:-1]))
            self.edges.append(cuts)

    def _bin(self, rows: np.ndarray) -> np.ndarray:
        """Bin ``k`` holds ``cuts[k-1] < x <= cuts[k]``: a split at bin ``b`` is ``x <= cuts[b]``."""
        binned = np.empty(rows.shape, dtype=np.uint8)
        for f, cuts in enumerate(self.edges):
            binned[:, f] = np.searchsorted(cuts, rows[:, f], side="left")
        return binned

    def _grow(self, binned: np.ndarray, cells: np.ndarray, gradient: np.ndarray,
              hessian: np.ndarray) -> tuple[_Tree, np.ndarray]:
        """One tree on the rows' gradients; returns it and each training row's leaf value."""
        width = binned.shape[1]
        size = width * 256
        lam, floor = self.l2_regularization, self.min_samples_leaf

        def histograms(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            flat = cells[rows].ravel()
            shape = (width, 256)
            return (np.bincount(flat, np.repeat(gradient[rows], width), size).reshape(shape),
                    np.bincount(flat, np.repeat(hessian[rows], width), size).reshape(shape),
                    np.bincount(flat, minlength=size).reshape(shape).astype(np.float64))

        def best(hist: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, int, int]:
            G, H, N = hist
            g_all, h_all, n_all = G[0].sum(), H[0].sum(), N[0].sum()
            gl, hl, nl = (np.cumsum(a, axis=1)[:, :-1] for a in (G, H, N))
            gr, hr, nr = g_all - gl, h_all - hl, n_all - nl
            gain = gl ** 2 / (hl + lam) + gr ** 2 / (hr + lam) - g_all ** 2 / (h_all + lam)
            ok = (nl >= floor) & (nr >= floor) & (hl >= MIN_CHILD_HESSIAN) & (hr >= MIN_CHILD_HESSIAN)
            gain = np.where(ok, gain, -np.inf)
            k = int(np.argmax(gain))
            f, b = divmod(k, gain.shape[1])
            return float(gain[f, b]), f, b

        feature, split_bin, left, right = [-1], [0], [-1], [-1]
        members = {0: np.arange(len(binned))}
        hists = {0: histograms(members[0])}
        heap: list[tuple[float, int, int, int]] = []
        gain, f, b = best(hists[0])
        if gain > 0.0:
            heap.append((-gain, 0, f, b))
        leaves = 1
        while heap and leaves < self.max_leaf_nodes:
            _, node, f, b = heapq.heappop(heap)
            rows = members.pop(node)
            parent = hists.pop(node)
            goes_left = binned[rows, f] <= b
            children = (rows[goes_left], rows[~goes_left])
            small = 0 if len(children[0]) <= len(children[1]) else 1
            small_hist = histograms(children[small])
            big_hist = tuple(p - s for p, s in zip(parent, small_hist))
            ids = (len(feature), len(feature) + 1)
            feature[node], split_bin[node], left[node], right[node] = f, b, ids[0], ids[1]
            for side, child in enumerate(ids):
                feature.append(-1)
                split_bin.append(0)
                left.append(-1)
                right.append(-1)
                members[child] = children[side]
                hists[child] = small_hist if side == small else big_hist
                gain, cf, cb = best(hists[child])
                if gain > 0.0:
                    heapq.heappush(heap, (-gain, child, cf, cb))
            leaves += 1
        value = np.zeros(len(feature))
        row_value = np.empty(len(binned))
        for node, rows in members.items():
            G, H, _ = hists[node]
            value[node] = -G[0].sum() / (H[0].sum() + lam)
            row_value[rows] = value[node]
        tree = _Tree(
            feature=np.asarray(feature, dtype=np.int64), bin=np.asarray(split_bin, dtype=np.int64),
            left=np.asarray(left, dtype=np.int64), right=np.asarray(right, dtype=np.int64),
            value=value, is_leaf=np.asarray(feature) < 0,
        )
        return tree, row_value

    def fit(self, table: np.ndarray, truth: np.ndarray) -> "ListwiseBooster":
        """``table`` ``[n, C, F]`` (`candidate_rows`), ``truth`` ``[n]`` candidate indices."""
        n, count, width = table.shape
        rows = table.reshape(n * count, width)
        assert np.isfinite(rows).all(), "a candidate row holds a non-finite value"
        self._fit_bins(rows)
        binned = self._bin(rows)
        cells = binned.astype(np.int64) + (np.arange(width, dtype=np.int64) * 256)[None, :]
        onehot = np.zeros((n, count))
        onehot[np.arange(n), truth] = 1.0
        scores = np.zeros((n, count))
        self.trees = []
        for _ in range(self.n_iter):
            prob = _softmax(scores)
            gradient = (prob - onehot).ravel()
            hessian = np.maximum(prob * (1.0 - prob), HESSIAN_FLOOR).ravel()
            tree, row_value = self._grow(binned, cells, gradient, hessian)
            scores += self.learning_rate * row_value.reshape(n, count)
            self.trees.append(tree)
        self.train_scores_ = scores
        return self

    def decision_function(self, table: np.ndarray) -> np.ndarray:
        n, count, width = table.shape
        binned = self._bin(table.reshape(n * count, width))
        total = np.zeros(n * count)
        for tree in self.trees:
            total += tree.predict(binned)
        return self.learning_rate * total.reshape(n, count)

    def predict_proba(self, table: np.ndarray) -> np.ndarray:
        return _softmax(self.decision_function(table))


def prior_share(majority: Counter, candidates: Sequence[str]) -> np.ndarray:
    """``[C]``: each candidate's share of the partition's training landings (R1's static majority
    counts, `RunwayContext.majority_counts`)."""
    counts = np.array([majority.get(r, 0) for r in candidates], dtype=np.float64)
    return counts / counts.sum()


def airline_block(day: str) -> int:
    return int(_fraction(f"r11:airline:{day}") * AIRLINE_BLOCKS)


def airline_shares(
    operators: np.ndarray, labels: np.ndarray, days: np.ndarray, keys: np.ndarray,
    train: np.ndarray, prior: np.ndarray,
) -> np.ndarray:
    """``[n, C]``: each sample's operator's share of the partition's TRAINING landings on each
    candidate, counted once per flight and smoothed toward ``prior`` with `AIRLINE_PRIOR_WEIGHT`
    pseudo-landings. OUT OF FOLD for a training sample: the training days are hashed into
    `AIRLINE_BLOCKS` blocks and its share counts the other blocks' flights only, so its own label —
    and its day's — never enters its own feature (a leave-one-out share would move with the label,
    and a tree reads that). A validation sample counts every training day. A flight with no
    callsign has no operator and reads the prior."""
    count = len(prior)
    blocks = np.array([airline_block(day) for day in days])
    per_block: dict[tuple[int, str], np.ndarray] = {}
    counted: set[str] = set()
    for i in np.flatnonzero(train):
        if keys[i] in counted:
            continue
        counted.add(keys[i])
        per_block.setdefault((int(blocks[i]), str(operators[i])), np.zeros(count))[labels[i]] += 1.0
    total: dict[str, np.ndarray] = {}
    for (_, operator), cell in per_block.items():
        total[operator] = total.get(operator, np.zeros(count)) + cell
    out = np.empty((len(labels), count))
    for i, operator in enumerate(operators):
        if not operator:
            out[i] = prior
            continue
        counts = total.get(str(operator), np.zeros(count))
        if train[i]:
            counts = counts - per_block.get((int(blocks[i]), str(operator)), np.zeros(count))
        out[i] = (counts + AIRLINE_PRIOR_WEIGHT * prior) / (counts.sum() + AIRLINE_PRIOR_WEIGHT)
    return out


def head_columns(head: str) -> tuple[str, ...]:
    """The column names of ``head``'s rows (`head_table`)."""
    if head == "r11":
        return CANDIDATE_ROW_NAMES
    kept = tuple(name for name in CANDIDATE_ROW_NAMES if name not in IDENTITY_COLUMNS)
    return kept if head == "r11_noid" else kept + ("airline_lift",)


def head_table(table: np.ndarray, head: str) -> np.ndarray:
    """``head``'s rows, from the full candidate table: R1.1's as they are; ``r11_noid`` without the
    `IDENTITY_COLUMNS`; ``r11_lift`` without the prior and with the operator's share replaced by its
    DEVIATION from the prior — about 0 on every runway for an operator that lands like everyone else,
    so it no longer carries the runway's base rate, while a real preference (a terminal side) stays."""
    if head == "r11":
        return table
    index = {name: i for i, name in enumerate(CANDIDATE_ROW_NAMES)}
    kept = table[:, :, [index[name] for name in head_columns("r11_noid")]]
    if head == "r11_noid":
        return kept
    lift = table[:, :, index["airline_share"]] - table[:, :, index["prior_share"]]
    return np.concatenate([kept, lift[:, :, None]], axis=2)


def grouped_permutation_rows(
    head: ListwiseBooster, table: np.ndarray, truth: np.ndarray, names: Sequence[str], *,
    group_of: np.ndarray, multi: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """R1's grouped permutation importance on a candidate table whose columns are ``names``: one
    group's columns are permuted together across SAMPLES (every candidate row of a sample moves with
    it). The static prior is constant across samples within a partition, so no permutation can move
    it; it is left out rather than reported as zero."""
    base = level_accuracy(head.predict_proba(table).argmax(axis=1), truth, group_of, multi)
    rng = np.random.default_rng(0)
    out: dict[str, dict[str, float | None]] = {}
    for group in sorted({HEAD_GROUPS[name] for name in names} - {"static prior"}):
        columns = [i for i, name in enumerate(names) if HEAD_GROUPS[name] == group]
        drops_exact, drops_side = [], []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = table.copy()
            shuffled[:, :, columns] = table[rng.permutation(len(table))][:, :, columns]
            cell = level_accuracy(head.predict_proba(shuffled).argmax(axis=1), truth, group_of, multi)
            drops_exact.append(base["exact"] - cell["exact"])
            if base["side_given_direction"] is not None and cell["side_given_direction"] is not None:
                drops_side.append(base["side_given_direction"] - cell["side_given_direction"])
        out[group] = {"exact_drop": float(np.mean(drops_exact)),
                      "side_drop": float(np.mean(drops_side)) if drops_side else None}
    return out


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_sample_arguments(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--r1-campaign-dir", type=Path, default=DEFAULT_R1_CAMPAIGN,
                        help="R1's campaign folder: the retrained R1 head is checked against its artifact")
    args = parser.parse_args(argv)

    s = build_samples(args)
    candidates, count = s.candidates, len(s.candidates)
    kw = dict(group_of=s.group_of, multi=s.multi)
    priors = {part: prior_share(s.contexts[part].majority_counts, candidates) for part in PARTITIONS}
    tables = {
        part: candidate_rows(
            s.space, s.X, group_of=s.group_of, prior_share=priors[part],
            b1_pick=s.picks[part]["B1_active_config"], minutes_since=s.minutes_since,
            airline_share=airline_shares(s.operators, s.y, s.days, s.keys, s.folds[part] == "train", priors[part]),
        )
        for part in PARTITIONS
    }

    probs: dict[str, dict[str, np.ndarray]] = {head: {} for head in HEADS}
    boosters: dict[str, ListwiseBooster] = {}      # the day_a heads, for their importance
    results: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for part in PARTITIONS:
        train, val = s.folds[part] == "train", s.folds[part] == "val"
        for head in SYMMETRIC_HEADS:
            rows = head_table(tables[part], head)
            started = time.perf_counter()
            booster = ListwiseBooster(**HEAD_SETTINGS).fit(rows[train], s.y[train])
            timings[f"{head}/{part}"] = time.perf_counter() - started
            probs[head][part] = booster.predict_proba(rows)
            if part == "day_a":
                boosters[head] = booster
        reference = HistGradientBoostingClassifier(**HGB_SETTINGS).fit(s.X[train], s.y[train])
        prob = np.zeros((len(s.y), count))
        prob[:, reference.classes_] = reference.predict_proba(s.X)
        probs["r1"][part] = prob
        rules = {rule: picks[val] for rule, picks in s.picks[part].items()}
        results[part] = {
            "train_samples": int(train.sum()), "val_samples": int(val.sum()),
            "train_flights": int(len(set(s.keys[train]))), "val_flights": int(len(set(s.keys[val]))),
            "prior_share": dict(zip(candidates, priors[part].tolist())),
            "minority_runways": s.minority_names[part],
            "validation": {
                head: score(probs[head][part][val], s.y[val], rules, s.b1_prob[val], s.anchors[val],
                            minority=s.minority[part], **kw)
                for head in HEADS
            },
        }

    # R1's head, retrained here, must be R1's head: its numbers against R1's own artifact.
    r1_path = args.r1_campaign_dir / s.airport / "runway_intent_r1.json"
    r1_check: dict[str, Any] = {"artifact": str(r1_path)}
    if r1_path.is_file():
        stored = json.loads(r1_path.read_text(encoding="utf-8"))["models"]
        r1_check["identical"] = {
            part: _jsonable(results[part]["validation"]["r1"]) == stored[part]["validation"] for part in PARTITIONS
        }

    both = (s.folds["day_a"] == "val") & (s.folds["flight"] == "val")
    rules_a = {rule: picks[both] for rule, picks in s.picks["day_a"].items()}
    leakage = {
        head: {part: score(probs[head][part][both], s.y[both], rules_a, s.b1_prob[both], s.anchors[both],
                           minority=s.minority["day_a"], **kw)
               for part in ("day_a", "flight")}
        for head in HEADS
    }
    validation_by_day = {
        part: by_day(s.folds[part] == "val", s.days, s.y, s.anchors, {
            **{head: probs[head][part].argmax(axis=1) for head in HEADS},
            "B1_active_config": s.picks[part]["B1_active_config"],
        }, candidates)
        for part in ("day_a", "day_b")
    }
    leakage_by_day = {
        head: by_day(both, s.days, s.y, s.anchors, {
            "day_a": probs[head]["day_a"].argmax(axis=1), "flight": probs[head]["flight"].argmax(axis=1),
            "B1_active_config": s.picks["day_a"]["B1_active_config"],
        }, candidates)
        for head in HEADS
    }
    val_a = s.folds["day_a"] == "val"
    importance = {
        head: grouped_permutation_rows(boosters[head], head_table(tables["day_a"], head)[val_a], s.y[val_a],
                                       head_columns(head), **kw)
        for head in SYMMETRIC_HEADS
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": SCHEMA,
        "airport": s.airport,
        "candidates": candidates,
        "direction_groups": s.pool.rules.groups,
        "minority_runways": s.minority_names,
        "anchors": {"entry": "the arrival-slice entry (25 km ring)", "rings_km": s.radii_km,
                    "rule": "first sample within R km of the airport reference"},
        "candidate_row": {"columns": list(CANDIDATE_ROW_NAMES), "groups": HEAD_GROUPS,
                          "heads": {head: list(head_columns(head)) for head in SYMMETRIC_HEADS}},
        "head": {**HEAD_SETTINGS, "hessian_floor": HESSIAN_FLOOR, "min_child_hessian": MIN_CHILD_HESSIAN,
                 "max_bins": MAX_BINS, "tree": "histogram, leaf-wise, Newton (ListwiseBooster)"},
        "r1_head": HGB_SETTINGS,
        "airline_share": {"blocks": AIRLINE_BLOCKS, "prior_weight": AIRLINE_PRIOR_WEIGHT,
                          "rule": "out of fold by training-day block; counted per flight"},
        "split": {"paired_leakage_samples": int(both.sum()), "paired_leakage_flights": int(len(set(s.keys[both])))},
        "r1_reference_check": r1_check,
        "models": results,
        "leakage": leakage,
        "validation_by_day": validation_by_day,
        "leakage_by_day": leakage_by_day,
        "importance_day_a": importance,
    }
    (out / "runway_intent_r11.json").write_text(json.dumps(document, indent=1), encoding="utf-8")

    lines = [f"{s.airport}: {len(s.usable)} usable flights, {len(s.y)} samples, {count} candidates; "
             f"R1 head matches R1's artifact: {r1_check.get('identical', 'no artifact')}; "
             f"fit seconds {', '.join(f'{k} {t:.0f}' for k, t in timings.items())}"]
    for part in PARTITIONS:
        for anchor in ("all", ENTRY):
            cells = []
            for head in HEADS:
                v = results[part]["validation"][head][anchor]
                m = v["model"]
                side = m["side_given_direction"]
                cells.append(f"{head} exact {m['exact']:.1%} side {'n/a' if side is None else f'{side:.1%}'} "
                             f"minority {'n/a' if m['minority_accuracy'] is None else format(m['minority_accuracy'], '.1%')} "
                             f"NLL {m['nll']:.3f} ECE {m['ece']:.3f}")
            best = max(c["exact"] for c in results[part]["validation"]["r11"][anchor]["rules"].values())
            lines.append(f"  {part:<6} {anchor:<5} " + " | ".join(cells) + f" | best rule exact {best:.1%}")
    for head in HEADS:
        a, f = leakage[head]["day_a"]["all"]["model"], leakage[head]["flight"]["all"]["model"]
        lines.append(f"  paired ({document['split']['paired_leakage_flights']} flights) {head}: "
                     f"day_a {a['exact']:.1%} / flight {f['exact']:.1%}")
    for part, cells in validation_by_day.items():
        trailing = [f"{day} ({c['flights']} flights, {c['top_runway']} {c['top_share']:.0%}): "
                    + " ".join(f"{head} {c['exact'][head]:.1%}" for head in HEADS)
                    + f" B1 {c['exact']['B1_active_config']:.1%}"
                    for day, c in cells.items()
                    if c["exact"]["B1_active_config"] - min(c["exact"][head] for head in HEADS) >= 0.05]
        lines.append(f"  {part} days a head trails B1 by >= 5 points: " + ("; ".join(trailing) or "none"))
    for head, groups in importance.items():
        lines.append(f"  importance ({head} day_a, exact / side drop): " + ", ".join(
            f"{g} {c['exact_drop']:+.3f}/{'n/a' if c['side_drop'] is None else format(c['side_drop'], '+.3f')}"
            for g, c in groups.items()))
    text = "\n".join(lines)
    (out / "runway_intent_r11.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
