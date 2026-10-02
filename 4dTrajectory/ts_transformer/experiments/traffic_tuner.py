"""The multi-aircraft post-training's optimiser (multi-aircraft design §6.2, §6.6 step 6): the second stage's recipe
(`prior.train.RewardTuner`) with each sentence scored in its scene — shared by M4's runner, not a runner.

**Scoring a sentence in its scene** (`scene_logits`): the traffic model and the model frozen at the pass's start read the
speaking aircraft's rows (`data.chain_record`: what it read as it spoke, its own words the targets) with its scene's
others replayed, laid out and given edge features by the same code the loop's speaker used
(`prior.scene_speaker.scene_inputs`, `traffic_speaking.speaking_edges`) — so the start's distribution is the one each
word was sampled from, to rounding. The pull's reference (base) has no traffic attention: it reads the aircraft alone
(`train.to_batch`), its own distribution per aircraft and step (design §6.2). All three under the masks each sentence was
said under (the separation masks among them).

**One pass** (`SceneRewardTuner.one_pass`) is `RewardTuner.one_pass`'s, term for term — the clipped-ratio surrogate
against the pass's start, ``kl_weight`` × the pull to the reference, ``data_weight`` × the data term, one update per batch
of sentences grouped as the second stage groups them (their own rows, ``tokens_per_batch`` / 2: the same sentences an
update) — with two differences:

- a batch laid out in its scenes can be far larger than the aircraft alone, so it is scored in parts of at most
  `SCORE_BUDGET` (the GPU memory measured, `PAIR_COST`), the model's layers recomputed in the backward (`Prior.encode`)
  and its heads read only on the speaking aircraft's rows, each part's share of the batch's mean backpropagated as it
  goes: the gradients add up to the batch's;
- the data term is the scene samples' (M2's, `traffic_scene_data.build_split` of the training days: the traffic
  attention reads the others there too), `DATA_BATCHES` of M2's batches an update (together about as many asked steps
  as the second stage's data batch), their NLL per asked step.

**Several passes** (design §6.6 step 6 item 14): ``passes`` sweeps over a round's sentences, each in a new order of
batches, every one against the model frozen once at the start — the one the sentences were sampled from, so the clip
bounds how far the sweeps go together (PPO's epochs); one pass is the one pass it always was.

The traffic attention has its own learning rate (design §8: from zero, at the others' 1e-5 it would take ten thousand
updates to move); the warm-up and the clipping are the second stage's. `RewardTuner.one_pass` itself is untouched: the
single-aircraft runners stay bit for bit (design §1.2), so the loop below mirrors it rather than sharing it.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.experiments.traffic_prior_train import asked_steps, scene_batch, scene_batches
from ts_transformer.experiments.traffic_scene_data import Built
from ts_transformer.experiments.traffic_speaking import HISTORY_S, Scene, other_node, speaking_edges
from ts_transformer.instructions.words import RUNWAY
from ts_transformer.prior.data import Split, batches
from ts_transformer.prior.model import Prior, asked_entries
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.prior.scene_speaker import scene_inputs
from ts_transformer.prior.train import (
    RewardConfig, RewardTuner, TrainConfig, allowed_tensors, batch_logits, column_nll, flight_kl, flight_surrogate,
    masked, to_batch,
)

#: M2's training batch, padded aircraft-steps (`traffic_prior_train`: pretraining's token budget).
M2_BATCH = TrainConfig().tokens_per_batch
#: What a part scored with gradients holds on the GPU, measured (RTX 4060, 8 GB; the traffic model on augmented: d 192,
#: 4 layers, each recomputed in the backward — `Prior.encode`; the heads on the speaking aircraft's rows only): about
#: 64 KB an aircraft-step and 8 KB an ordered pair of aircraft a step (the edge layers' [B, T, A, A, d]; without the
#: recomputation 145 KB and 31 KB, and one 16-aircraft scene of 900 steps did not fit). A part costs its padded
#: aircraft-steps plus its pairs × `PAIR_COST`.
PAIR_COST = 1 / 8
#: The most a part costs, about 3 GB (a sentence costing more is a part of its own: the training days' largest scene,
#: 16 aircraft × 1,400 steps, measured 4.5 GB alone).
SCORE_BUDGET = 48 * 1024
#: M2's scene batches an update's data term reads: two hold about 8,800 asked steps (M2's ratio, 3.7 padded to one
#: asked), about the second stage's data batch (8,192 rows of single flights).
DATA_BATCHES = 2


@dataclass(frozen=True)
class SceneSplit:
    """A round's sentences as the tuner reads them: ``split`` (`Split`: each sentence's `data.chain_record` and the
    candidate table) and, per sentence, its scene and the positions it read (``[rows, 3]``: e, n, height)."""

    split: Split
    scenes: list[Scene]
    positions: list[np.ndarray]

    def __post_init__(self) -> None:
        if not len(self.split.flights) == len(self.scenes) == len(self.positions):
            raise ValueError(f"{len(self.split.flights)} sentences, {len(self.scenes)} scenes, "
                             f"{len(self.positions)} positions")
        for flight, positions in zip(self.split.flights, self.positions):
            if len(positions) != flight.rows:
                raise ValueError(f"{flight.dataset_id}: {flight.rows} rows, {len(positions)} positions")

    @property
    def flights(self) -> list[Any]:
        return self.split.flights

    def subset(self, indices: Sequence[int]) -> SceneSplit:
        split = self.split
        return SceneSplit(Split([split.flights[i] for i in indices], split.airports, split.candidates, split.runways,
                                split.courses, split.classes, split.variant),
                          [self.scenes[i] for i in indices], [self.positions[i] for i in indices])


@dataclass(frozen=True)
class SceneBatch:
    """Sentences laid out in their scenes (`scene_layout`): the model's inputs and edge features, and where the speaking
    aircraft's rows sit (from step ``pre``, ``rows`` padded as `to_batch` pads them)."""

    inputs: dict[str, torch.Tensor]
    edges: torch.Tensor
    airport: torch.Tensor
    pre: int
    rows: int


def _pre(scene: Scene, step_s: float) -> tuple[int, list[Any]]:
    """A scene's others as the speaker places them and its pre-roll (the speaker's rule: the most steps an other is in
    the air before the speaking aircraft's row 0, at most `HISTORY_S`)."""
    others = [other_node(scene, key, step_s) for key in scene.others]
    return min(int(HISTORY_S // step_s), max([0] + [-node.first_step for node in others])), others


def scene_layout(sentences: SceneSplit, indices: Sequence[int], single: Mapping[str, torch.Tensor], step_s: float
                 ) -> SceneBatch:
    """The sentences at ``indices`` in their scenes (module docstring), ``single`` their `to_batch` (the speaking
    aircraft's rows and targets, padded)."""
    placed = [_pre(sentences.scenes[i], step_s) for i in indices]
    pre = max(p for p, _ in placed)
    rows = single["features"].shape[2]
    aircraft = 1 + max(len(others) for _, others in placed)
    own = {name: single[name][:, 0] for name in ("features", "relative", "in_force", "since", "targets")}
    inputs = scene_inputs(own, [sentences.flights[i].rows for i in indices], [others for _, others in placed], pre,
                          aircraft, 0, pre + rows)
    edges = []
    for (own_pre, _), i in zip(placed, indices):
        position, flight = sentences.positions[i], sentences.flights[i]
        edges.append((pre - own_pre, speaking_edges(sentences.scenes[i], position[:, 0], position[:, 1],
                                                    position[:, 2], flight.in_force[:, RUNWAY], own_pre, step_s)))
    features = edges[0][1].shape[-1]
    out = np.zeros((len(indices), pre + rows, aircraft, aircraft, features), dtype=np.float32)
    for b, (offset, edge) in enumerate(edges):
        steps, members = edge.shape[:2]
        out[b, offset: offset + steps, :members, :members] = edge
    device = single["features"].device
    return SceneBatch(inputs, torch.as_tensor(out, device=device), single["airport"], pre, rows)


def scene_logits(model: Prior, layout: SceneBatch, *, checkpoint: bool = False) -> list[torch.Tensor]:
    """The speaking aircraft's logits of its own words in its scene (teacher forcing), ``[B, 1, rows, classes]`` as
    `train.batch_logits` gives them for the aircraft alone: `Prior.forward`'s, its heads read only on the speaking
    aircraft's rows (a head reads its own aircraft-step alone); ``checkpoint``: the layers recomputed in the backward
    (`Prior.encode`)."""
    x = layout.inputs
    h, tokens, valid = model.encode(x["features"], x["relative"], x["static"], x["in_force"], x["since"],
                                    layout.airport, x["present"], x["rows"], layout.edges, checkpoint)
    own = (slice(None), slice(0, 1), slice(layout.pre, layout.pre + layout.rows))
    return model.logits(h[own], tokens[own], valid, x["targets"][own], x["rows"][own] == N_LOOK)


def scene_size(sentences: SceneSplit, i: int, step_s: float) -> tuple[int, int, int]:
    """A sentence laid out alone: its aircraft, its pre-roll and its rows."""
    pre, others = _pre(sentences.scenes[i], step_s)
    return 1 + len(others), pre, sentences.flights[i].rows


def part_cost(count: int, aircraft: int, steps: int) -> float:
    """What ``count`` sentences laid out together on ``aircraft`` × ``steps`` cost (`PAIR_COST`)."""
    return count * aircraft * steps * (1 + aircraft * PAIR_COST)


def parts(sentences: SceneSplit, indices: Sequence[int], step_s: float, budget: float) -> list[list[int]]:
    """``indices`` in parts costing at most ``budget`` (`part_cost`) as `scene_layout` lays them out together — the most
    aircraft, the longest pre-roll plus the most rows — a costlier sentence a part of its own; by size."""
    sizes = {i: scene_size(sentences, i, step_s) for i in indices}
    out: list[list[int]] = [[]]
    aircraft = pre = rows = 0
    for i in sorted(indices, key=lambda i: (part_cost(1, sizes[i][0], sizes[i][1] + sizes[i][2]), i)):
        a, p, r = sizes[i]
        if out[-1] and part_cost(len(out[-1]) + 1, max(aircraft, a), max(pre, p) + max(rows, r)) > budget:
            out.append([])
            aircraft = pre = rows = 0
        out[-1].append(i)
        aircraft, pre, rows = max(aircraft, a), max(pre, p), max(rows, r)
    return out


def traffic_parameters(model: Prior) -> list[nn.Parameter]:
    """The traffic attention's parameters, every layer's."""
    if not model.traffic_features:
        raise ValueError("the model has no traffic attention")
    return [p for layer in model.layers for p in layer.traffic.parameters()]


class SceneRewardTuner(RewardTuner):
    """`RewardTuner` with each sentence scored in its scene and scene samples as the data (module docstring);
    ``traffic_learning_rate`` the traffic attention's, ``step_s`` the vocabulary's step."""

    def __init__(self, model: Prior, reference: Prior, config: RewardConfig, device: torch.device, *, seed: int,
                 traffic_learning_rate: float, step_s: float) -> None:
        if reference.traffic_features:
            raise ValueError("the pull's reference reads the aircraft alone: a single-aircraft prior")
        self.traffic_learning_rate, self.step_s = traffic_learning_rate, step_s
        self._scene_data: Iterator[list[int]] = iter(())
        super().__init__(model, reference, config, device, seed=seed)

    def restart(self, rng: np.random.Generator) -> None:
        """The next pass draws from ``rng`` — its batches' order and its data term's from the start (a round's own
        stream: a round run on its own draws what it would within one run)."""
        self.rng = rng
        self._scene_data = iter(())

    def state(self) -> dict[str, Any]:
        """What the next round's pass continues from beside the weights: the optimiser's state (AdamW's moments), the
        warm-up's step and the passes made."""
        return {"optimiser": self.optimiser.state_dict(), "schedule": self.schedule.state_dict(), "passes": self.passes}

    def load_state(self, state: Mapping[str, Any]) -> None:
        """`state` put back (the model already holding the weights it was saved with)."""
        self.optimiser.load_state_dict(state["optimiser"])
        self.schedule.load_state_dict(state["schedule"])
        self.passes = state["passes"]

    def _parameter_groups(self) -> Any:
        traffic = traffic_parameters(self.model)
        own = {id(p) for p in traffic}
        return [{"params": traffic, "lr": self.traffic_learning_rate},
                {"params": [p for p in self.model.parameters() if id(p) not in own]}]

    def _scene_scored(self, sentences: SceneSplit, part: Sequence[int],
                      allowed: Sequence[Mapping[int, np.ndarray]] | None, start: Prior | None
                      ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None,
                                 list[torch.Tensor], torch.Tensor]:
        """A part's sentences: their `to_batch`, the model's logits in their scenes, the start's (None: not asked), the
        reference's alone — all under the masks they were said under when ``allowed`` is given — and each one's asked
        steps."""
        batch = to_batch(sentences.split, part, self.device)
        layout = scene_layout(sentences, part, batch, self.step_s)
        with torch.no_grad():                   # first: their memory is freed before the model's is held
            started = None if start is None else scene_logits(start, layout)
            reference = batch_logits(self.reference, batch)
        logits = scene_logits(self.model, layout, checkpoint=torch.is_grad_enabled())
        if allowed is not None:
            masks = allowed_tensors([allowed[i] for i in part], batch["targets"].shape[2],
                                    [logit.shape[-1] for logit in logits], self.device)
            logits, reference = masked(logits, masks), masked(reference, masks)
            started = None if started is None else masked(started, masks)
        speaks = asked_entries(batch["present"], batch["rows"]).expand_as(batch["present"])
        return batch, logits, started, reference, speaks.sum(dim=(1, 2)).to(logits[0].dtype)

    def _parts(self, sentences: SceneSplit, indices: Sequence[int]) -> list[list[int]]:
        return parts(sentences, indices, self.step_s, SCORE_BUDGET)

    def distance(self, sentences: SceneSplit, allowed: Sequence[Mapping[int, np.ndarray]] | None = None) -> float:
        """`RewardTuner.distance` with the model in the sentences' scenes."""
        if not sentences.flights:
            raise ValueError("no sentence to measure the distance on")
        self.model.eval()
        means = []
        with torch.no_grad():
            for indices in batches(sentences.flights, self.config.tokens_per_batch // 2, None):
                total = 0.0
                for part in self._parts(sentences, indices):
                    batch, logits, _, reference, steps = self._scene_scored(sentences, part, allowed, None)
                    total += float((flight_kl(logits, reference, batch["targets"], batch["present"], batch["asked"])
                                    / steps).sum())
                means.append(total / len(indices))
        return float(np.mean(means))

    def _data_backward(self, data: Sequence[Built], slots: int) -> float:
        """The data term of one update, backpropagated: `DATA_BATCHES` scene batches' NLL per asked step, × the data
        weight (the data reshuffled whenever it runs out)."""
        groups = []
        for _ in range(DATA_BATCHES):
            indices = next(self._scene_data, None)
            if indices is None:
                self._scene_data = scene_batches(data, M2_BATCH, self.rng)
                indices = next(self._scene_data)
            groups.append(indices)
        asked = sum(asked_steps(data[i]) for indices in groups for i in indices)
        total = 0.0
        for indices in groups:
            batch = scene_batch(data, indices, slots, self.device)
            logits = self.model(batch["features"], batch["relative"], batch["static"], batch["in_force"],
                                batch["since"], batch["airport"], batch["present"], batch["rows"], batch["edges"],
                                batch["targets"], checkpoint=True)
            nll = column_nll(logits, batch["targets"], batch["present"], batch["asked"])
            part = self.config.data_weight * nll.sum() / asked
            if not torch.isfinite(part):
                raise FloatingPointError(f"pass {self.passes}: the data term is {float(part)}")
            part.backward()
            total += float(part.detach())
        return total / self.config.data_weight

    def one_pass(self, sentences: SceneSplit, advantages: np.ndarray, data: Sequence[Built],
                 allowed: Sequence[Mapping[int, np.ndarray]] | None = None, *, slots: int,
                 passes: int = 1) -> dict[str, Any]:
        """`RewardTuner.one_pass` over ``sentences`` in their scenes beside the scene samples ``data`` (module
        docstring; ``slots``: the model's candidate slots), ``passes`` times — each sweep in a new order of batches, every
        one against the model the sentences were sampled from, frozen once (the clipped ratio's denominator: PPO's
        several epochs, the clip bounding how far they go together). The same record over all updates, and each sweep's
        own under ``sweeps``; one pass is the one pass it always was."""
        if len(advantages) != len(sentences.flights):
            raise ValueError(f"{len(advantages)} advantages for {len(sentences.flights)} sentences")
        if allowed is not None and len(allowed) != len(sentences.flights):
            raise ValueError(f"{len(allowed)} sentences' masks for {len(sentences.flights)} sentences")
        if not sentences.flights:
            raise ValueError("no sentence to train on: no scene's sentences differ in reward")
        return self.sweeps(lambda: batches(sentences.flights, self.config.tokens_per_batch // 2, self.rng),
                           lambda indices: self._parts(sentences, indices),
                           lambda part, start: self._scene_scored(sentences, part, allowed, start)
                           + (advantages[part], None),
                           len, data, slots=slots, passes=passes, sentences=len(sentences.flights))

    def sweeps(self, groups: Callable[[], Iterator[list[int]]], parts: Callable[[list[int]], list[list[int]]],
               scored: Callable[[list[int], Prior], tuple[Any, ...]], counted: Callable[[list[int]], int],
               data: Sequence[Built], *, slots: int, passes: int, sentences: int) -> dict[str, Any]:
        """`one_pass`' sweeps over units a caller lays out (the scene's sentences, or a window's samples: `traffic_window_
        tuner`): each sweep's update batches (``groups``, drawn when it starts), each batch's parts (``parts``), a part
        scored (``scored``: its `to_batch`, the model's logits, the start's, the reference's, each sentence's asked steps,
        its advantages, as `_scene_scored` gives them, and each sentence's own extra loss — a probe's word learned, `traffic_
        window_tuner`; None: none), and the sentences a batch counts (``counted``: each one's mean over the batch)."""
        if passes < 1:
            raise ValueError(f"{passes} passes")
        self.model.eval()
        started = time.perf_counter()
        # the model the pass's sentences were sampled from, frozen: the clipped ratio's denominator
        start = copy.deepcopy(self.model).eval()
        for parameter in start.parameters():
            parameter.requires_grad_(False)
        sums = {"reward": 0.0, "kl": 0.0, "data": 0.0, "imitation": 0.0}
        count, trace, clipped_trace, clipped, words, sweeps = 0, [], [], 0, 0, []
        for _ in range(passes):
            self.passes += 1
            sweep_started, first, sweep_clipped, sweep_words = time.perf_counter(), count, 0, 0
            for indices in groups():
                self.optimiser.zero_grad(set_to_none=True)
                reward, kl, imitated, batch_clipped, batch_words = 0.0, 0.0, 0.0, 0, 0
                size = counted(indices)
                for part in parts(indices):
                    batch, logits, sampled_from, reference, steps, advantages, imitation = scored(part, start)
                    advantage = torch.as_tensor(advantages, dtype=logits[0].dtype, device=self.device)
                    surrogate, part_clipped, part_words = flight_surrogate(logits, sampled_from, batch["targets"],
                                                                           batch["present"], batch["asked"], advantage,
                                                                           self.config.clip_ratio)
                    distance = flight_kl(logits, reference, batch["targets"], batch["present"], batch["asked"])
                    part_reward = (surrogate / steps).sum() / size
                    part_kl = (distance / steps).sum() / size
                    loss = part_reward + self.config.kl_weight * part_kl
                    if imitation is not None:
                        part_imitation = (imitation / steps).sum() / size
                        loss = loss + part_imitation
                        imitated += float(part_imitation.detach())
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"pass {self.passes}: the loss is {float(loss)}")
                    loss.backward()
                    reward, kl = reward + float(part_reward.detach()), kl + float(part_kl.detach())
                    batch_clipped, batch_words = batch_clipped + part_clipped, batch_words + part_words
                data_loss = self._data_backward(data, slots)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
                self.optimiser.step()
                self.schedule.step()
                clipped, words = clipped + batch_clipped, words + batch_words
                sweep_clipped, sweep_words = sweep_clipped + batch_clipped, sweep_words + batch_words
                clipped_trace.append(batch_clipped / batch_words)
                for name, value in (("reward", reward), ("kl", kl), ("data", data_loss), ("imitation", imitated)):
                    sums[name] += value
                count += 1
                trace.append(kl)
            sweeps.append({"batches": count - first, "kl_mean": float(np.mean(trace[first:])),
                           "kl_max": max(trace[first:]), "clipped_share": sweep_clipped / sweep_words,
                           "seconds": time.perf_counter() - sweep_started})
        return {**{f"{name}_mean": value / count for name, value in sums.items()}, "kl_max": max(trace),
                "batches": count, "sentences": sentences, "seconds": time.perf_counter() - started,
                "kl_trace": trace, "clipped_share": clipped / words, "clipped_trace": clipped_trace, "sweeps": sweeps}
