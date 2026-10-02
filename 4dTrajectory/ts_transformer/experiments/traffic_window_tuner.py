"""The multi-aircraft post-training in windows (multi-aircraft design §6.6 step 7 item 7, the 7.6 plan): a window's words
scored as the window speaker read them — shared by the window M4 runner, not a runner.

**What each aircraft read** (`window_flights`): a commanded aircraft's rows rebuilt as `data.chain_record` rebuilds a
one-aircraft sentence — its positions and its words in force, every row it was in the scene (the rows past its judged
end too: the other aircraft read them) — with its landing context switched where the speaker switched it: a landing in
the loop re-reads the rows not encoded yet (`WindowSpeaker.set_context`), so each row reads the context it was encoded
with, a row a landing came after it read before (`WindowRecord.context_changes`).

**A window scored whole** (`window_layout`, `window_logits`): the window samples laid out on one batch's steps as the
loop placed them (`traffic_window.window_places`), every commanded aircraft at its rows and every replayed one at its
own (`window_speaker.window_inputs` — the speaker's one layout), the edge features by the loop's code
(`traffic_window.window_edges`); one encoding, the heads read on each commanded aircraft's rows: ``[N, 1, rows,
classes]`` per column, as `train.batch_logits` gives them for aircraft alone — so the words' distributions are the
ones they were sampled from, to rounding (tests). Encoded a block of steps at a time (`PAIRS_PER_BLOCK`): a busy
window's pair tensors do not fit the GPU whole.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.experiments.traffic_go_around import IMITATION_WEIGHT
from ts_transformer.experiments.traffic_scene_data import Built
from ts_transformer.experiments.traffic_tuner import SCORE_BUDGET, SceneRewardTuner, part_cost
from ts_transformer.experiments.traffic_window import (
    Window, WindowRecord, _with_landing, window_edges, window_landings, window_places,
)
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import APPROACH, APPROACH_GO_AROUND, RUNWAY, UNCHANGED
from ts_transformer.prior.data import Flight, Split, chain_record
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import N_LOOK, Landings
from ts_transformer.prior.train import RewardConfig, allowed_tensors, batch_logits, flight_kl, masked, to_batch
from ts_transformer.prior.window_speaker import window_inputs

#: The aircraft pairs a block of steps holds in the encoding (`Prior.encode`'s ``pairs_per_block``). Measured on the RTX
#: 4060 (2026-09-30, round 1's windows of the formal run, every commanded aircraft trained, the layers recomputed): the
#: largest window sample, 23 aircraft (cost 112,654), ran out of the 8 GB whole and peaks at 1.51 GB above the models in
#: blocks of 32,768 pairs (131,072: 2.27 GB, 16,384: 1.51 GB); 19 aircraft whole 4.61 GB, in blocks 1.06 GB; no slower;
#: the gradients equal the whole encoding's to 1e-6 of the largest.
PAIRS_PER_BLOCK = 32_768


def window_flight(record: WindowRecord, window: Window, signals: FlightSignals, geometry: AirportGeometry,
                  landings: Mapping[str, Landings] | None, airport: int, capture_row: int, counted: int, step_s: float,
                  forced: int | None = None) -> Flight:
    """One commanded aircraft's rows as the speaker read them (module docstring), its own words the targets of its first
    ``counted`` steps (what it is trained on) — but the approach word a probe said for it at own step ``forced``, which
    it did not sample (`WindowLoop`'s probes; learned apart, `WindowRewardTuner`); ``signals``: the loop's own
    (`WindowLoop.flights`: an augmented window's moved or shifted — its entry time, its own landing and its first step are
    read off them); ``landings``: each airport's landing context, None for a variant without it."""
    said = record.grid
    classes = np.where(said != UNCHANGED, said + 1, 0)
    asked = np.zeros(said.shape, dtype=bool)
    asked[:counted] = True
    if forced is not None and forced < counted:
        asked[forced, APPROACH] = False
    context = None if landings is None else window_landings(window, record.key, landings, step_s)

    def built(context: Landings | None) -> Flight:
        return chain_record(signals, record.e, record.n, record.h, said, classes, asked, geometry, context, airport,
                            capture_row, step_s)

    flight = built(context)
    if not record.context_changes:
        return flight
    features, relative = flight.features.copy(), flight.relative.copy()
    for first, landing_s, runway in record.context_changes:
        context = _with_landing(context, landing_s, runway)
        again = built(context)
        features[first:], relative[first:] = again.features[first:], again.relative[first:]
    return dataclasses.replace(flight, features=features, relative=relative)


@dataclass(frozen=True)
class WindowLayout:
    """Window samples laid out on one batch's steps (`window_layout`): the model's inputs with each commanded aircraft's
    targets, the edge features, each scene's airport, and where each commanded aircraft sits (its scene, its place among
    the scene's aircraft, the step of its row 0, its rows)."""

    inputs: dict[str, torch.Tensor]
    edges: torch.Tensor
    airport: torch.Tensor
    scene: np.ndarray
    slot: np.ndarray
    start: np.ndarray
    rows: np.ndarray


def window_layout(model: Prior, windows: Sequence[Window], flights: Sequence[Flight], records: Sequence[WindowRecord],
                  step_s: float, device: torch.device) -> WindowLayout:
    """``windows`` (window samples, each its commanded aircraft's ``flights`` and ``records`` in its order, window after
    window) laid out as the loop laid them out (module docstring)."""
    places = window_places(windows, step_s)
    scene = np.array([w for w, window in enumerate(windows) for _ in window.commanded], dtype=np.int64)
    keys = [k for window in windows for k in window.commanded]
    if [r.key for r in records] != keys or len(flights) != len(keys):
        raise ValueError("a flight and a record per commanded aircraft, window after window, each in its order")
    slot = np.concatenate([np.arange(len(window.commanded)) for window in windows])
    rows = np.array([f.rows for f in flights], dtype=np.int64)
    width = max(rows)
    aircraft = max(len(window.commanded) + len(window.others) for window in windows)

    def padded(name: str) -> torch.Tensor:
        values = [getattr(f, name) for f in flights]
        tail = values[0].shape[1:]
        if name == "relative":                             # an airport's candidates in the model's slots (`to_batch`)
            tail = (model.config.candidate_slots, max(v.shape[2] for v in values))
        out = np.zeros((len(values), width) + tail, dtype=values[0].dtype)
        for i, v in enumerate(values):
            out[(i, slice(0, len(v))) + tuple(slice(0, n) for n in v.shape[1:])] = v
        return torch.as_tensor(out, device=device)

    own = {name: padded(name) for name in ("features", "relative", "in_force", "since", "targets")}
    own["in_force"], own["targets"] = own["in_force"].long(), own["targets"].long()
    last = int((places.start + rows).max())
    inputs = window_inputs(own, rows, places.start, scene, slot, places.nodes,
                           [len(window.commanded) for window in windows], aircraft, 0, last)
    positions = np.zeros((3, len(flights), width))
    pointers = np.zeros((len(flights), width), dtype=np.int64)
    for i, (record, flight) in enumerate(zip(records, flights)):
        positions[:, i, : len(record.e)] = record.e, record.n, record.h
        pointers[i, : flight.rows] = flight.in_force[:, RUNWAY]
    edges = window_edges(windows, keys, scene, False, positions[0], positions[1], positions[2], pointers, rows,
                         places.start, 0, last, step_s, aircraft, len(model.traffic_features))
    airport = torch.tensor([model.config.airports.index(window.airport.flights.code) for window in windows],
                           device=device)
    return WindowLayout(inputs, torch.as_tensor(edges, device=device), airport, scene, slot, places.start, rows)


def window_logits(model: Prior, layout: WindowLayout, *, checkpoint: bool = False,
                  aircraft: Sequence[int] | None = None, rows: int | None = None) -> list[torch.Tensor]:
    """Commanded aircraft's logits of their own words in their windows (teacher forcing), ``[N, 1, rows, classes]`` per
    column as `train.batch_logits` gives them for aircraft alone: one encoding of the windows (`Prior.encode`;
    ``checkpoint``: the layers recomputed in the backward), the heads read on each aircraft's rows — ``aircraft`` (the
    commanded ones' places; every one by default) over their first ``rows`` (their longest by default; padded past an
    aircraft's last with its last step's place, never read)."""
    x = layout.inputs
    h, tokens, valid = model.encode(x["features"], x["relative"], x["static"], x["in_force"], x["since"],
                                    layout.airport, x["present"], x["rows"], layout.edges, checkpoint,
                                    pairs_per_block=PAIRS_PER_BLOCK)
    device = h.device
    read = np.arange(len(layout.rows)) if aircraft is None else np.asarray(aircraft, dtype=np.int64)
    width = int(layout.rows[read].max()) if rows is None else rows
    start, own_rows = layout.start[read], layout.rows[read]
    steps = torch.as_tensor(np.minimum(start[:, None] + np.arange(width)[None, :], (start + own_rows - 1)[:, None]),
                            device=device)
    scene = torch.as_tensor(layout.scene[read], device=device)[:, None]
    slot = torch.as_tensor(layout.slot[read], device=device)[:, None]
    own_h = h[scene, slot, steps][:, None]
    own_tokens = tokens[scene, slot, steps][:, None]
    targets = x["targets"][scene, slot, steps][:, None]
    first = (torch.arange(width, device=device) == N_LOOK)[None, None, :]
    return model.logits(own_h, own_tokens, valid[scene[:, 0]], targets, first)


def window_advantages(rows: Sequence[Mapping[str, object]], samples: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(each row's advantage, each row's probe gain, the rows trained on)`` of a round's window sentences
    (`traffic_window_generation.WindowSentences.rows`, one draw's: an aircraft of a window over its ``samples`` samples
    — refused otherwise).

    - The advantage: its reward less the mean of the same aircraft of the same window over the samples spoken the same
      way — three groups: its unprobed samples, its probes that said no go-around (the model's words throughout, but
      kept because its margin never fell) and its probes that did (`landing_reward.group_advantages`' rule: not divided
      by the spread). A sentence with a probe's go-around is the model's words under a go-around it did not say; weighed
      against the others, every word after the go-around was pushed down with it, the better ones too (step 8 small
      tests, readout 2026-10-02 §3) — among themselves, the words after one go-around are weighed against those after
      another.
    - The probe gain (0 but for a sentence a probe said a go-around for): its reward less the mean of the aircraft's
      unprobed samples — how much better the go-around did than the model's own words; the go-around word is learned
      where it is above 0 (`WindowRewardTuner._window_scored`). An aircraft with a probe's go-around and no unprobed
      sample is refused.
    - Trained: of an aircraft none of whose samples starts in a loss it answers for (no word of it made that one), each
      such group's rows whose rewards differ, and a probe's sentence whose gain is above 0."""
    groups: dict[tuple[object, object], list[int]] = {}
    for k, row in enumerate(rows):
        groups.setdefault((row["window"], row["dataset_id"]), []).append(k)
    advantages, gains = np.zeros(len(rows)), np.zeros(len(rows))
    trained: set[int] = set()
    for (window, key), members in groups.items():
        if sorted(rows[k]["sample"] for k in members) != list(range(samples)):
            raise ValueError(f"window {window}, {key}: samples {[rows[k]['sample'] for k in members]}, not 0 … "
                             f"{samples - 1} once each")
        rewards = {k: float(rows[k]["reward"]) for k in members}
        unprobed = [k for k in members if not rows[k]["probed"]]
        unfired = [k for k in members if rows[k]["probed"] and rows[k]["forced"] is None]
        forced = [k for k in members if rows[k]["forced"] is not None]
        if forced and not unprobed:
            raise ValueError(f"window {window}, {key}: a probe's go-around and no unprobed sample to weigh it against")
        answerable = not any(rows[k]["starts_in_a_loss"] for k in members)
        for group in (unprobed, unfired, forced):
            if group:
                own = np.array([rewards[k] for k in group])
                advantages[group] = own - own.mean()
                if own.min() != own.max() and answerable:
                    trained.update(group)
        if forced:
            gains[forced] = np.array([rewards[k] for k in forced]) - np.mean([rewards[k] for k in unprobed])
            if answerable:
                trained.update(k for k in forced if gains[k] > 0.0)
    return advantages, gains, np.array(sorted(trained), dtype=np.int64)


@dataclass(frozen=True)
class WindowSplit:
    """A round's window samples trained on, as the tuner reads them: the candidate table (``table``, a `Split` whose own
    flights are not read), and per window sample its window, its commanded aircraft's rows as the speaker read them
    (`window_flight`: the trained ones' own words asked over their counted steps, the others' over none) and records,
    which of them are trained (their places in the sample) with their advantages, the masks they spoke under, the own
    step a probe said a go-around for each at (−1: none; inside its counted steps, else −1) and their probe gains
    (`window_advantages`)."""

    table: Split
    windows: list[Window]
    flights: list[list[Flight]]
    records: list[list[WindowRecord]]
    trained: list[list[int]]
    advantages: list[np.ndarray]
    allowed: list[list[Mapping[int, np.ndarray]]]
    forced: list[list[int]]
    gains: list[np.ndarray]

    def __post_init__(self) -> None:
        for window, flights, records, trained, advantages, allowed, forced, gains in zip(
                self.windows, self.flights, self.records, self.trained, self.advantages, self.allowed, self.forced,
                self.gains, strict=True):
            if not (len(window.commanded) == len(flights) == len(records)) or not trained \
                    or not (len(trained) == len(advantages) == len(allowed) == len(forced) == len(gains)) \
                    or sorted(set(trained)) != list(trained) or not 0 <= trained[0] <= trained[-1] < len(flights):
                raise ValueError("a window sample: a flight and a record per commanded aircraft, one or more trained "
                                 "(each once, in order), an advantage, masks, a probe's step and a probe gain each")
            for k, masks, step in zip(trained, allowed, forced):
                steps = counted_rows(flights[k]) - N_LOOK
                if any(len(bits) != steps for bits in masks.values()):
                    raise ValueError(f"{records[k].key}: masks over {[len(b) for b in masks.values()]} steps, "
                                     f"{steps} counted")
                if not -1 <= step < steps or (step >= 0 and flights[k].asked[N_LOOK + step, APPROACH]):
                    raise ValueError(f"{records[k].key}: a probe's go-around at step {step} of {steps} counted, asked")

    @property
    def sentences(self) -> int:
        return sum(len(t) for t in self.trained)


def scoring_cost(window: Window, flights: Sequence[Flight], step_s: float) -> float:
    """A window sample's scoring cost alone (`part_cost`: its aircraft, its pre-roll and span)."""
    aircraft, pre, span = window_size(window, flights, step_s)
    return part_cost(1, aircraft, pre + span)


def counted_rows(flight: Flight) -> int:
    """A trained aircraft's rows to its last counted step (`window_flight` asks every column of those)."""
    return N_LOOK + int(flight.asked.any(axis=-1).sum())


def counted_flight(flight: Flight) -> Flight:
    """A trained aircraft's rows cut at its last counted step — the sentence as R32 trains it (the model is causal: its
    later rows, read by the others, change none of these)."""
    rows = counted_rows(flight)
    return dataclasses.replace(flight, **{name: getattr(flight, name)[:rows] for name in
                                          ("features", "relative", "in_force", "since", "targets", "asked")})


def window_size(window: Window, flights: Sequence[Flight], step_s: float) -> tuple[int, int, int]:
    """A window sample laid out alone: its aircraft, its pre-roll and its steps after it."""
    places = window_places([window], step_s)
    rows = np.array([f.rows for f in flights])
    return (len(window.commanded) + len(window.others), places.pre,
            int((places.start + rows).max()) - places.pre)


def window_parts(split: WindowSplit, indices: Sequence[int], step_s: float, budget: float) -> list[list[int]]:
    """``indices`` (window samples) in parts costing at most ``budget`` scored together (`traffic_tuner.part_cost`: the
    most aircraft, the longest pre-roll plus the longest span), a costlier one a part of its own; by size."""
    sizes = {s: window_size(split.windows[s], split.flights[s], step_s) for s in indices}
    out: list[list[int]] = [[]]
    aircraft = pre = span = 0
    for s in sorted(indices, key=lambda s: (part_cost(1, sizes[s][0], sizes[s][1] + sizes[s][2]), s)):
        a, p, r = sizes[s]
        if out[-1] and part_cost(len(out[-1]) + 1, max(aircraft, a), max(pre, p) + max(span, r)) > budget:
            out.append([])
            aircraft = pre = span = 0
        out[-1].append(s)
        aircraft, pre, span = max(aircraft, a), max(pre, p), max(span, r)
    return out


def update_batches(split: WindowSplit, tokens: int, rng: np.random.Generator | None) -> list[list[int]]:
    """Window samples in update batches as `data.batches` groups sentences: by their trained aircraft's longest rows to
    their counted steps (`counted_rows`: a sentence as R32 cuts it), each batch at most ``tokens`` padded trained steps (a
    sample holding more is a batch of its own); shuffled when ``rng`` is given."""
    longest = {s: max(counted_rows(split.flights[s][k]) for k in trained) for s, trained in enumerate(split.trained)}
    groups: list[list[int]] = []
    current: list[int] = []
    rows = count = 0
    for s in sorted(longest, key=lambda s: (longest[s], s)):
        grown_rows, grown = max(rows, longest[s]), count + len(split.trained[s])
        if current and grown_rows * grown > tokens:
            groups.append(current)
            current, grown_rows, grown = [], longest[s], len(split.trained[s])
        current.append(s)
        rows, count = grown_rows, grown
    if current:
        groups.append(current)
    if rng is not None:
        rng.shuffle(groups)
    return groups


class WindowRewardTuner(SceneRewardTuner):
    """`traffic_tuner.SceneRewardTuner` over window samples (module docstring, the 7.6 plan): each part's samples scored
    whole (`window_layout`, `window_logits`), the loss on its trained aircraft's own words — the clipped ratio against
    the model frozen at the pass's start, the pull to the reference reading each aircraft alone, each over the masks it
    spoke under — each sentence's mean over its counted steps, averaged over the update batch's trained sentences; the
    data term, the optimiser and the passes the scene tuner's; a probe's go-around word learned apart
    (``imitation_weight``, `_window_scored`)."""

    def __init__(self, model: Prior, reference: Prior, config: RewardConfig, device: torch.device, *, seed: int,
                 traffic_learning_rate: float, step_s: float, imitation_weight: float = IMITATION_WEIGHT) -> None:
        self.imitation_weight = imitation_weight
        super().__init__(model, reference, config, device, seed=seed, traffic_learning_rate=traffic_learning_rate,
                         step_s=step_s)

    def _window_scored(self, split: WindowSplit, part: Sequence[int], start: Prior | None
                       ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None,
                                  list[torch.Tensor], torch.Tensor, np.ndarray, torch.Tensor | None]:
        """A part's window samples: the trained aircraft's `to_batch`, their logits in their windows (the model's, the
        start's — None: not asked — and the reference's alone) under their masks, their asked steps, advantages, and
        each sentence's probe term (None when no sentence of the part was probed): ``imitation_weight`` × its probe gain
        × the negative log-probability of the go-around a probe said for it, where its gain is above 0 — the word it did
        not sample, out of the clipped ratio (`window_flight`), learned as a cross-entropy weighted by how much better
        the go-around did than the aircraft's unprobed samples (`window_advantages`; multi-aircraft design §6.6 step 8
        item 10)."""
        flights = [f for s in part for f in split.flights[s]]
        records = [r for s in part for r in split.records[s]]
        layout = window_layout(self.model, [split.windows[s] for s in part], flights, records, self.step_s,
                               self.device)
        offsets = np.cumsum([0] + [len(split.flights[s]) for s in part])
        chosen = [int(offsets[n] + k) for n, s in enumerate(part) for k in split.trained[s]]
        # the trained sentences as R32 cuts them; the layout keeps every row, the others read them
        trained = dataclasses.replace(split.table, flights=[counted_flight(flights[i]) for i in chosen])
        batch = to_batch(trained, range(len(chosen)), self.device)
        rows = batch["targets"].shape[2]
        with torch.no_grad():                   # first: their memory is freed before the model's is held
            started = None if start is None else window_logits(start, layout, aircraft=chosen, rows=rows)
            reference = batch_logits(self.reference, batch)
        logits = window_logits(self.model, layout, checkpoint=torch.is_grad_enabled(), aircraft=chosen, rows=rows)
        masks = allowed_tensors([a for s in part for a in split.allowed[s]], rows, [x.shape[-1] for x in logits],
                                self.device)
        logits, reference = masked(logits, masks), masked(reference, masks)
        started = None if started is None else masked(started, masks)
        steps = batch["asked"].any(dim=-1).sum(dim=(1, 2)).to(logits[0].dtype)
        advantages = np.concatenate([split.advantages[s] for s in part])
        forced = np.concatenate([np.asarray(split.forced[s], dtype=np.int64) for s in part])
        gains = np.concatenate([split.gains[s] for s in part])
        learned = np.flatnonzero((forced >= 0) & (gains > 0.0))
        imitation = None
        if len(learned):
            log_p = torch.log_softmax(logits[APPROACH][learned, 0, N_LOOK + forced[learned]], dim=-1)
            weight = torch.as_tensor(self.imitation_weight * gains[learned], dtype=log_p.dtype, device=self.device)
            imitation = torch.zeros(len(advantages), dtype=log_p.dtype, device=self.device)
            imitation = imitation.index_put((torch.as_tensor(learned, device=self.device),),
                                            -weight * log_p[:, APPROACH_GO_AROUND + 1])
        return batch, logits, started, reference, steps, advantages, imitation

    def _window_parts(self, split: WindowSplit, indices: Sequence[int]) -> list[list[int]]:
        """Parts by the scene tuner's budget (`SCORE_BUDGET`, `PAIR_COST`: the WHOLE encoding's cost, measured on
        scenes) — in blocks of steps a part holds about a fifth of that (the largest window sample: 13 KB a cost unit
        against 67 KB whole), so the parts are smaller than the GPU allows: slower, never larger."""
        return window_parts(split, indices, self.step_s, SCORE_BUDGET)

    def window_distance(self, split: WindowSplit) -> float:
        """`RewardTuner.distance` with each trained aircraft in its window: the mean over update batches of the mean
        over their trained sentences of the KL to the reference per counted step."""
        if not split.windows:
            raise ValueError("no window sample to measure the distance on")
        self.model.eval()
        means = []
        with torch.no_grad():
            for indices in update_batches(split, self.config.tokens_per_batch // 2, None):
                total = 0.0
                for part in self._window_parts(split, indices):
                    batch, logits, _, reference, steps, _, _ = self._window_scored(split, part, None)
                    total += float((flight_kl(logits, reference, batch["targets"], batch["present"], batch["asked"])
                                    / steps).sum())
                means.append(total / sum(len(split.trained[s]) for s in indices))
        return float(np.mean(means))

    def window_pass(self, split: WindowSplit, data: Sequence[Built], *, slots: int, passes: int = 1) -> dict[str, Any]:
        """`SceneRewardTuner.one_pass` over window samples (`sweeps`): the update batches `update_batches`', each
        sentence its trained aircraft's."""
        if not split.windows:
            raise ValueError("no window sample to train on: no aircraft's group of sentences differs in reward and no "
                             "probe's go-around did better than its unprobed samples")
        return self.sweeps(lambda: iter(update_batches(split, self.config.tokens_per_batch // 2, self.rng)),
                           lambda indices: self._window_parts(split, indices),
                           lambda part, start: self._window_scored(split, part, start),
                           lambda indices: sum(len(split.trained[s]) for s in indices), data, slots=slots,
                           passes=passes, sentences=split.sentences)
