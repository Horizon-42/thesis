"""Training with a value function (post-training §2 item 10, §8 C22; D171): the value network V, what it reads beyond
the model, the samples it reads and the advantages it gives.

**V** is a copy of a campaign's start model (the prior with its traffic attention, `Value.of`) with a token part of its
own (`VALUE_FEATURES`, `traffic_attention.add_token_part`: its projection starts at zero) and a head: from the last
layer's output at a row (`Prior.encode`) and the time left at that row to one number, the reward expected at the
sentence's end. After the copy V shares no weight with the model. It reads its inputs with dropout off (eval mode, as
the surrogate, D107), its gradients on.

**What V reads beyond the model** (`value_part`, `time_left`; the user, 2026-10-07): (a) for each recorded aircraft of
a row's tokens, in the order of the row's tokens, its edge features (`edges.EDGE_FEATURES`) at the row + 30, 60 and
120 s, each against the commanded aircraft's state at the row, each with a flag that it is in the air then (0 and the
flag 0 where it is not); and the time to its recorded landing over `LANDING_SCALE_S` (0 once it has landed); (b) the
time left at the row before the judge's time limit (the limit in force, its go-arounds' extensions included, less the
time flown) over `TIME_LEFT_SCALE_S`, to the head. Nothing else: not the commanded aircraft's own record. This is the
one exception to outline principle 7 and vocabulary D90: V never speaks, no readout reads it, and the model's inputs
never hold these features (they are kept apart from the tokens the model reads, `ValueSample`).

**The advantages** (`advantages`): V at the round's start reads every sample once: v_t at each counted row t (the
rows said, from the first predicted step up to the event), 0 after the event. δ_t = v_{t+1} − v_t, plus the reward
(D30) at the last counted row; A_t = Σ_l λ^l δ_{t+l} (λ = `GAE_LAMBDA`, no discount: γ = 1); the target R_t = A_t + v_t.
Not normalised.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.inference.runway_schedule import Separation
from ts_transformer.post.branches import Sentence
from ts_transformer.post.edges import EDGE_FEATURES, tokens
from ts_transformer.post.loss import Samples
from ts_transformer.post.scene import AircraftAt, Window
from ts_transformer.post.traffic_attention import Traffic, add_token_part, traffic_of
from ts_transformer.prior.batch import RowTensors, collate
from ts_transformer.prior.model import Prior
from ts_transformer.prior.speaker import Permitted

#: The format of a round's V file (`value.pt`: V's weights, its optimizer, its shape, the round's identity).
VALUE_SCHEMA = "ts-post-value-v1"
#: The format name of V's token part (`traffic_attention.TokenPart`).
VALUE_PART_SCHEMA = "post-value-part-v1"
#: The times after a row at which V reads each recorded aircraft, s.
HORIZONS_S = (30.0, 60.0, 120.0)
#: The scale of the time to a recorded aircraft's landing, s.
LANDING_SCALE_S = 120.0
#: The scale of the time left before the judge's time limit, s (a go-around's extension, `executor.GO_AROUND_EXTRA_S`).
TIME_LEFT_SCALE_S = 900.0
#: GAE's λ (§2 item 10 point 4).
GAE_LAMBDA = 0.95
#: The width of V's head.
HEAD_HIDDEN = 64
#: V's token part, in order: for each horizon, the edge features and the flag that the aircraft is in the air then; the
#: time to its recorded landing.
VALUE_FEATURES = (*(f"{name}_{int(h)}s" for h in HORIZONS_S for name in (*EDGE_FEATURES, "in_air")), "to_landing")


def value_part(window: Window, step: int, own: AircraftAt, others: AircraftAt, separation: Separation,
               step_s: float) -> np.ndarray:
    """``[len(others), len(VALUE_FEATURES)]`` float32: V's token part of each recorded aircraft ``others`` at the
    window's step ``step`` (module docstring), ``own`` the commanded aircraft there; ``step_s`` the motion's row."""
    scene, time_s = window.scene, window.step_s(step)
    records = [scene.flight(key) for key in others.keys]
    width = len(EDGE_FEATURES) + 1
    out = np.zeros((len(others), len(VALUE_FEATURES)), dtype=np.float32)
    for k, horizon in enumerate(HORIZONS_S):
        later = time_s + horizon
        alive = [i for i, r in enumerate(records) if r.start_s <= later <= r.end_s]
        if alive:
            then = AircraftAt.of([records[i].at_step(later, scene.interval_s) for i in alive])
            out[alive, k * width:k * width + len(EDGE_FEATURES)] = tokens(
                own, then, scene.geometry, separation, step_s)[:, :len(EDGE_FEATURES)]
            out[alive, k * width + len(EDGE_FEATURES)] = 1.0
    out[:, -1] = [max(r.landing_s - time_s, 0.0) / LANDING_SCALE_S for r in records]
    return out


def time_left(limit_s: float, flown_s: float) -> float:
    """The time left before the judge's time limit ``limit_s`` (in force, its go-arounds' included) after ``flown_s``
    flown, over `TIME_LEFT_SCALE_S` (module docstring)."""
    return (limit_s - flown_s) / TIME_LEFT_SCALE_S


@dataclass(frozen=True)
class ValueSample:
    """A window's sentence in a campaign with a value function (D171): the window (its place in the round), its
    sentence (its rows with the words said as targets, the speaker's records, the model's tokens at each row, its
    reward), and what V reads beyond it at each row: V's token part of each recorded aircraft (`value_part`, in the
    order of the row's tokens) and the time left (`time_left`)."""

    window: int
    sentence: Sentence
    part: list[np.ndarray]
    time_left: np.ndarray

    def __post_init__(self) -> None:
        if len(self.part) != len(self.sentence.tokens) or len(self.time_left) != len(self.sentence.tokens):
            raise ValueError("V reads one token part and one time left at each row of the sentence's tokens")
        if any(len(p) != len(t) for p, t in zip(self.part, self.sentence.tokens)):
            raise ValueError("V's token part holds one row for each recorded aircraft of the row's tokens")


@dataclass(frozen=True)
class ValueBatch:
    """Samples as the model's loss and V read them: the model's `Samples` (``advantage``: the rows' A_t), V's traffic
    input (the tokens with V's part), the time left at each row, and each row's target R_t."""

    samples: Samples
    traffic: Traffic
    time_left: torch.Tensor
    target: torch.Tensor


def value_batch(kept: Sequence[ValueSample], device: torch.device) -> tuple[Samples, Traffic, torch.Tensor, torch.Tensor]:
    """The samples of ``kept`` before their advantages: the model's `Samples` (every row said counts, the advantage 0
    for the caller to fill), V's traffic input, the time left ``[B, R]`` and the rewards ``[B]``."""
    if not kept:
        raise ValueError("no sample")
    rows = collate([k.sentence.rows for k in kept], device)
    length = rows.asked.shape[1]
    width = kept[0].sentence.tokens[0].shape[1]
    empty, empty_part = (np.zeros((0, width), dtype=np.float32),
                         np.zeros((0, width + len(VALUE_FEATURES)), dtype=np.float32))
    model_tokens = [k.sentence.tokens + [empty] * (length - len(k.sentence.tokens)) for k in kept]
    value_tokens = [[np.concatenate((t, p), axis=1) for t, p in zip(k.sentence.tokens, k.part)]
                    + [empty_part] * (length - len(k.part)) for k in kept]
    left = np.zeros((len(kept), length), dtype=np.float32)
    for b, k in enumerate(kept):
        left[b, :len(k.time_left)] = k.time_left
    samples = Samples(rows, Permitted.join([k.sentence.permitted for k in kept]), traffic_of(model_tokens, device),
                      torch.zeros(rows.asked.shape, dtype=torch.float32, device=device), rows.asked)
    return (samples, traffic_of(value_tokens, device, len(VALUE_FEATURES)), torch.as_tensor(left, device=device),
            torch.tensor([k.sentence.reward for k in kept], dtype=torch.float32, device=device))


class Value(nn.Module):
    """V (module docstring): a copy of a start model with V's token part, and its head."""

    def __init__(self, prior: Prior) -> None:
        super().__init__()
        self.prior = prior
        add_token_part(self.prior, len(VALUE_FEATURES), VALUE_PART_SCHEMA)
        d = prior.config.d_model
        self.head = nn.Sequential(nn.Linear(d + 1, HEAD_HIDDEN), nn.GELU(), nn.Linear(HEAD_HIDDEN, 1))

    @classmethod
    def of(cls, model: Prior) -> Value:
        """V from the model ``model`` (a copy: no weight shared), in eval mode; its head drawn from torch's present
        numbers (the caller seeds them)."""
        out = cls(copy.deepcopy(model)).to(next(model.parameters()).device)
        for parameter in out.parameters():
            parameter.requires_grad_(True)
        return out.eval()

    def shape(self) -> dict[str, Any]:
        """Its shape (`value.pt`): the copy's configuration, the token part's width and format, the head's width."""
        return {"prior": self.prior.config.to_dict(), "part": {"schema": VALUE_PART_SCHEMA, "width": len(VALUE_FEATURES)},
                "head_hidden": HEAD_HIDDEN}

    def forward(self, rows: RowTensors, traffic: Traffic, left: torch.Tensor) -> torch.Tensor:
        """``[B, R]``: the value at every row (dropout off: refused in training mode)."""
        if self.training:
            raise ValueError("V reads its inputs with dropout off (eval mode, D171)")
        h, _ = self.prior.encode(rows, traffic)
        return self.head(torch.cat((h, left[..., None].to(h.dtype)), dim=-1))[..., 0]


def advantages(values: torch.Tensor, counted: torch.Tensor, reward: torch.Tensor, lam: float = GAE_LAMBDA
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(A [B, R], R_t [B, R])`` of samples whose V values at each row are ``values`` [B, R], counted at ``counted``
    [B, R] (one run of rows each, from the first predicted step to the event), rewarded ``reward`` [B] (module
    docstring); 0 at a row not counted."""
    values = values.detach().to(torch.float64)
    counted_np = counted.cpu().numpy()
    advantage = torch.zeros_like(values)
    for b in range(values.shape[0]):
        rows = np.flatnonzero(counted_np[b])
        if not len(rows):
            raise ValueError(f"sample {b} counts no row")
        if rows[-1] - rows[0] + 1 != len(rows):
            raise ValueError(f"sample {b}: its counted rows are not one run")
        v = values[b, rows[0]:rows[-1] + 1]
        delta = torch.cat((v[1:], v.new_zeros(1))) - v
        delta[-1] += float(reward[b])
        running = v.new_zeros(())
        out = torch.empty_like(v)
        for i in range(len(v) - 1, -1, -1):
            running = delta[i] + lam * running
            out[i] = running
        advantage[b, rows[0]:rows[-1] + 1] = out
    target = (advantage + values) * counted
    return advantage.to(torch.float32), target.to(torch.float32)


def value_loss(value: Value, batch: ValueBatch, rows: torch.Tensor) -> torch.Tensor:
    """V's loss of one piece of an update: the sum over its counted rows of (V − R_t)², over ``rows`` (the counted
    rows of the whole update: an update's loss is its mean, in pieces as `post.loss.update_step`)."""
    counted = batch.samples.counted
    predicted = value(batch.samples.rows, batch.traffic, batch.time_left)
    return (((predicted - batch.target) ** 2) * counted).sum() / rows
