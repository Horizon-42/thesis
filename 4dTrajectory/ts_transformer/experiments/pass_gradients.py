"""Which part of the loss sets a round's pass (the user, 2026-10-10: is the reward's signal too small for the step, or
the reward itself?) — the gradient of each part at the start of a round's pass, stages C and D.

THE READ: a round whose groups are on disk (a round left half done: a round's groups are deleted after its pass), the
model that spoke it (the round before's checkpoint, or the start for round 0), its pass's updates in the pass's own
order (`post_train.update_pairs` with the round's numbers; the data term's dropout seeded from the round's seed, its
masks the pass's in distribution only: another device's generator, or with ``--every`` above 1 updates left out; of a
campaign of several passes a round, D169, the first pass's order), every ``--every``-th of them, at that model
(no step taken: every ratio is 1, so no word is clipped). Of each update, by the optimizer's parameter groups
(`parameter_groups`: the prior's own, the traffic modules') and over all: the gradient of the surrogate (the reward's
advantages), of the pull to the base (times `post.loss.KL_WEIGHT`) and of the data term (times the campaign's weight),
`post.loss.update_gradients`: each part's norm and its share of the update's direction, ⟨g_part, g⟩ / |g|² with g the
sum of the parts (the shares add up to 1), and the cosine of each pair. Under AdamW a parameter's step is about its
learning rate whatever its gradient's size: the shares say which part points the update, the learning rate how far it
goes. The means over the updates read, by group. And whether the updates read point one way (the user, 2026-10-10:
after a 3× learning rate did worse, is the reward's direction noise?), for the surrogate, the data term (a reference:
one objective on every batch) and their sum with the pull, by group and over all (`Consistency`): the mean pairwise
cosine of the updates' gradients, the cosine of the sums of the updates of even and odd place, and the coherence
|Σ g_i| / Σ |g_i| (1 when every update points one way, about 1/√n when they are independent).

Writes one JSON (``ts-pass-gradients-v2``) and prints the means. From a clean checkout unless ``--smoke``; the
closed loop's (D69) and the edge features' (D104) checks first.

    python run_ts.py pass_gradients --campaign 4dTrajectory/outputs/POOLED/multi/<campaign> --round <r> \\
        --out <a new file>.json [--every K] [--device cpu] [--threads T]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments import post_train, window_behaviour
from ts_transformer.experiments.post_window_loop import checked_edges
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.post.branches import MINORITY_SIDE
from ts_transformer.post.loss import DATA_WEIGHT, PassStart, update_gradients
from ts_transformer.post.traffic_attention import parameter_groups
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of the read.
PASS_GRADIENTS_SCHEMA = "ts-pass-gradients-v2"
#: The parts of the loss, in `post.loss.update_gradients`'s order.
PARTS = ("surrogate", "pull", "data")
#: Over every parameter group.
ALL = "all"
#: The gradients whose consistency across the updates is read: the surrogate's, the data term's and the update's whole
#: (the pull, a hundredth of the direction, left out apart).
CONSISTENT = ("surrogate", "data", "whole")


def _flat(grads: Sequence[torch.Tensor | None], parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """One part's gradient of ``parameters`` as one vector (zeros where it does not reach a parameter)."""
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1).double()
                      for g, p in zip(grads, parameters)])


def measured(parts: Mapping[str, Sequence[torch.Tensor | None]], groups: Mapping[str, Sequence[int]],
             parameters: Sequence[torch.nn.Parameter]) -> dict[str, dict[str, Any]]:
    """One update's parts (`update_gradients`) by group (``groups``: each group's indices in ``parameters``) and over
    all: each part's norm and share of the direction, the cosine of each pair (module docstring); a share or a cosine
    is None where its denominator is 0."""
    out = {}
    for name, indices in {**groups, ALL: range(len(parameters))}.items():
        chosen = [parameters[i] for i in indices]
        vectors = {part: _flat([parts[part][i] for i in indices], chosen) for part in PARTS}
        whole = sum(vectors.values())
        squared = float(whole @ whole)
        norms = {part: float(v.norm()) for part, v in vectors.items()}
        out[name] = {
            "norm": norms, "whole_norm": squared ** 0.5,
            "share": {part: (float(v @ whole) / squared if squared else None) for part, v in vectors.items()},
            "cosine": {f"{a}|{b}": (float(vectors[a] @ vectors[b]) / (norms[a] * norms[b]) if norms[a] * norms[b] else
                                    None) for i, a in enumerate(PARTS) for b in PARTS[i + 1:]}}
    return out


class Consistency:
    """Whether the updates read point one way (module docstring), streamed by part (`CONSISTENT`) and by span of the
    flat gradient (``spans``: each group's and `ALL`'s): each update's gradient g_i goes to the sum of the updates of
    its place's parity (H0, H1) and, as a unit vector, is dotted with the sum of the units before it (U), so that the
    mean pairwise cosine is 2 Σ_i ĝ_i·U_{i−1} / (m(m−1)) over the m updates whose gradient there is not 0; the halves'
    cosine is cos(H0, H1); the coherence |H0 + H1| / Σ_i |g_i|. None where a part reaches no parameter of a span (or
    fewer than two updates do)."""

    def __init__(self, spans: Mapping[str, slice]) -> None:
        self.spans = dict(spans)
        size = max(span.stop for span in self.spans.values())
        self.updates = 0
        self.halves = {part: [torch.zeros(size, dtype=torch.float64) for _ in range(2)] for part in CONSISTENT}
        self.units = {part: {name: torch.zeros(span.stop - span.start, dtype=torch.float64)
                             for name, span in self.spans.items()} for part in CONSISTENT}
        self.pairs = {part: dict.fromkeys(self.spans, 0.0) for part in CONSISTENT}
        self.norms = {part: dict.fromkeys(self.spans, 0.0) for part in CONSISTENT}
        self.counts = {part: dict.fromkeys(self.spans, 0) for part in CONSISTENT}

    def add(self, vectors: Mapping[str, torch.Tensor]) -> None:
        """One update's flat gradients (``vectors``, each part of `CONSISTENT`)."""
        for part in CONSISTENT:
            g = vectors[part]
            self.halves[part][self.updates % 2] += g
            for name, span in self.spans.items():
                piece = g[span]
                norm = float(piece.norm())
                if not norm:
                    continue
                unit = piece / norm
                self.pairs[part][name] += float(unit @ self.units[part][name])
                self.units[part][name] += unit
                self.norms[part][name] += norm
                self.counts[part][name] += 1
        self.updates += 1

    def result(self) -> dict[str, dict[str, dict[str, float | None]]]:
        """By span and part: the mean pairwise cosine, the halves' cosine and the coherence (class docstring)."""
        out: dict[str, dict[str, dict[str, float | None]]] = {}
        for name, span in self.spans.items():
            out[name] = {}
            for part in CONSISTENT:
                m = self.counts[part][name]
                h0, h1 = (h[span] for h in self.halves[part])
                n0, n1 = float(h0.norm()), float(h1.norm())
                out[name][part] = {
                    "updates": m,
                    "pairwise_cosine": 2 * self.pairs[part][name] / (m * (m - 1)) if m >= 2 else None,
                    "halves_cosine": float(h0 @ h1) / (n0 * n1) if n0 * n1 else None,
                    "coherence": float((h0 + h1).norm()) / self.norms[part][name] if m else None}
        return out


def spans_of(groups: Mapping[str, Sequence[int]], parameters: Sequence[torch.nn.Parameter]) -> dict[str, slice]:
    """Each group's span of the flat gradient (`_flat`'s order: ``parameters``) and `ALL`'s."""
    starts = [0]
    for p in parameters:
        starts.append(starts[-1] + p.numel())
    return {**{name: slice(starts[indices[0]], starts[indices[-1] + 1]) for name, indices in groups.items()},
            ALL: slice(0, starts[-1])}


def means(updates: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The means over ``updates`` of each group's whole norm, norms, shares and cosines (a None left out of its mean;
    None where every one is None)."""
    def mean(values: list[float | None]) -> float | None:
        held = [v for v in values if v is not None]
        return sum(held) / len(held) if held else None

    groups = updates[0]["parts"]
    return {name: {"whole_norm": mean([u["parts"][name]["whole_norm"] for u in updates]),
                   **{key: {k: mean([u["parts"][name][key][k] for u in updates]) for k in groups[name][key]}
                      for key in ("norm", "share", "cosine")}} for name in groups}


def _shown(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a post_train or multi_train campaign")
    parser.add_argument("--round", type=int, required=True, help="a round whose groups are on disk (left half done)")
    parser.add_argument("--out", type=Path, required=True, help="a new JSON file")
    parser.add_argument("--every", type=int, default=1, help="every K-th update of the pass, from its first")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="the CPU's threads (torch.set_num_threads)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a tree with changes too")
    args = parser.parse_args(argv)
    campaign, out = (p if p.is_absolute() else REPO_ROOT / p for p in (args.campaign, args.out))
    if args.every < 1 or args.threads < 1 or args.round < 0:
        parser.error("--every and --threads are at least 1, --round at least 0")
    if out.exists():
        parser.error(f"{out} exists: the read writes a new file")
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    try:
        settings, stage = window_behaviour.opened_campaign(record)
    except ValueError as refused:
        parser.error(f"{campaign}: {refused}")
    directory = campaign / f"round_{args.round}"
    if not sorted(directory.glob("groups_*.pt")):
        parser.error(f"{directory} holds no groups: a round's groups are deleted after its pass; read a round left "
                     f"half done")
    if args.round > post_train.done_rounds(campaign):
        parser.error(f"{campaign} has done {post_train.done_rounds(campaign)} rounds: round {args.round} has no model "
                     f"that spoke it")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a read that is not a smoke needs a clean tree")
    torch.set_num_threads(args.threads)
    inputs = post_train.inputs_here(record["inputs"])
    instructions, executor = Path(inputs["instructions"]), Path(inputs["executor"])
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    device = torch.device(args.device)
    context = post_train.open_context(Path(inputs["prior"]), instructions, executor, edges_reference, device,
                                      Path(inputs["procedure_root"]), formal=not record["inputs"]["smoke"], splits=())
    model_round = None if args.round == 0 else args.round - 1
    model = window_behaviour.model_of(context, settings, stage, campaign, model_round)
    named = parameter_groups(model, settings.prior_lr, settings.traffic_lr)
    parameters = [p for group in named for p in group["params"]]
    groups, at = {}, 0
    for group in named:
        groups[group["name"]] = range(at, at + len(group["params"]))
        at += len(group["params"])
    start = PassStart(model)
    # MIRROR of `post_train.run_campaign` and the stages' `train` (`STAGE_C.train`; stage D's `train_pass` with its data and
    # group weights): the pass's numbers, its dropout's seed, the rows a group counts and each group's weight
    segment_rows = (post_train.segment_rows_of(settings, context.interval_s) if stage is post_train.STAGE_C else None)
    data_weight = DATA_WEIGHT if stage is post_train.STAGE_C else settings.data_weight          # the stage's pass
    weighted = stage is not post_train.STAGE_C and settings.group_weight == MINORITY_SIDE
    torch.manual_seed(post_train.pass_seed(settings.seed, args.round))             # the data term's dropout
    pairs = post_train.update_pairs(directory, context.data, settings, np.random.default_rng([settings.seed,
                                                                                               args.round, 1]),
                                    device, part_width=stage.part_width, segment_rows=segment_rows,
                                    minority_mean=post_train.minority_mean(directory) if weighted else None)
    updates = []
    consistency = Consistency(spans_of(groups, parameters))
    for u, (pieces, data) in enumerate(pairs):
        if u % args.every:
            continue
        parts = update_gradients(model, start, context.base, pieces, data, parameters, data_weight)
        updates.append({"update": u, "groups": len(pieces), "parts": measured(parts, groups, parameters)})
        flat = {part: _flat(parts[part], parameters) for part in PARTS}
        consistency.add({"surrogate": flat["surrogate"], "data": flat["data"], "whole": sum(flat.values())})
        print(json.dumps({"update": u, "share": updates[-1]["parts"][ALL]["share"]}), flush=True)
    if not updates:
        raise SystemExit(f"{directory}: its groups give no update (no informative group)")
    summary, across = means(updates), consistency.result()
    write_json_atomic(out, {
        "schema": PASS_GRADIENTS_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign), "round": args.round,
        "model_round": model_round, "every": args.every, "updates_read": len(updates), "updates_seen": u + 1,
        "group_files": len(sorted(directory.glob("groups_*.pt"))),
        "learning_rates": {g["name"]: g["lr"] for g in named},
        "git": git, "checks": opened["checks"], "smoke": args.smoke, "device": str(device), "threads": args.threads,
        "means": summary, "consistency": across, "updates": updates})
    for name, held in summary.items():
        print(f"{name}: share " + ", ".join(f"{part} {_shown(held['share'][part])}" for part in PARTS)
              + " | norm " + ", ".join(f"{part} {held['norm'][part]:.4g}" for part in PARTS)
              + " | cosine " + ", ".join(f"{pair} {_shown(value)}" for pair, value in held["cosine"].items()),
              flush=True)
    for name, held in across.items():
        print(f"{name}: across updates " + "; ".join(
            f"{part} pairwise {_shown(c['pairwise_cosine'])} halves {_shown(c['halves_cosine'])} coherence "
            f"{_shown(c['coherence'])}" for part, c in held.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
