"""The closure prediction output (scene design §五 P1.c): the network regresses a few
DECISION quantities and the trajectory is drawn in closed form.

A closure model reads the same history as the other outputs and returns a
``ClosurePrediction`` — one decision vector per flight in physical units — instead of
states or controls:

    d_join (m)            the runway distance at which the path settles on the localizer
    via_d, via_xt (m)     the via pose in runway axes (a base-turn point, an intercept …)
    via_cos, via_sin      the via heading RELATIVE to the inbound course, as a (near-)unit
                          vector — exact away from the origin, softened at it
    slowness[K_s + 1]     ground slowness (s/m) at K_s + 1 knots over the path's progress —
                          the duration is their integral over the path, no separate head
    height[K_h]           height (m) at the first K_h of K_h + 1 knots; the last is the
                          threshold, pinned to 0

``reconstruct`` turns a decision vector and the anchor state into the dense trajectory
(``closure_geometry.via_dubins`` for the path, ``closure_profile`` for the clock and the
heights, velocities from the tangent and the ground speed), and the export writes it
through the record contract's reference-shaped branch (no controls). A via inside one
turn radius of the anchor is not a decision — the label puts a straight-in flight's via
AT the anchor, and a predicted via a few tens of metres off it with a degree of heading
error makes ``via_dubins`` fly a full circle (measured on the first C_pred arm: 827 of
the 837 straight-in paths longer than 1.3× the truth) — so ``reconstruct`` draws the
plain CSC from the anchor to the join instead and records ``csc-via-at-anchor``. Training is a
per-flight regression on the labels ``fit_labels`` produces from the truth
(``docs/p1_closure_oracle.py labels`` writes them for a whole cohort); nothing
differentiable goes through the geometry, and the model never sees a label as an input.
Flights whose label is not identifiable (``canonical`` false) or whose family residual
is above the cap carry ``closure_valid = 0`` and do not enter the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

import closure_geometry as cg
import closure_profile as cp
from approach_difficulty import approach_difficulty
from batch_contract import LossComponents
from config import CLOSURE_LABEL_KNOTS, TSConfig
from geometric_metrics import cumulative_arc_m
from intent_conditioning import truth_join_point

if TYPE_CHECKING:   # dataset imports this module lazily; a module-level import would cycle
    from dataset import FlightSeries, Normalizer

LABEL_SCHEMA = "closure-labels-v2"  # v2: per-profile position / time errors, geometry kind + length ratio
LABEL_RESIDUAL_MAX_M = 1_000.0      # above this F3 residual the flight is a fallback, not a label
LABEL_KNOTS = CLOSURE_LABEL_KNOTS   # the profile knot counts a labels file carries

# The decision vector's scales: the loss reads every group at these, and the head's
# outputs are bounded to them (a decision the family cannot draw is not a prediction).
D_JOIN_SCALE_M = 10_000.0
D_JOIN_MAX_M = 40_000.0             # d_join ∈ [D_JOIN_MIN_M, D_JOIN_MIN_M + D_JOIN_MAX_M]
VIA_SCALE_M = 10_000.0
VIA_MAX_M = 40_000.0                # via_d, via_xt ∈ ±VIA_MAX_M
HEIGHT_SCALE_M = 1_000.0
SLOWNESS_MIN = 1.0 / cp.SPEED_MAX_MPS
SLOWNESS_MAX = 1.0 / cp.SPEED_MIN_MPS

KIND_VIA_AT_ANCHOR = "csc-via-at-anchor"     # the via was inside one turn radius: dropped

CONTEXT_DECISION = "closure_decision"
CONTEXT_VALID = "closure_valid"
CONTEXT_PATH_LENGTH = "closure_path_length_m"
CONTEXT_COURSE = "runway_heading_rad"
CLOSURE_CONTEXT_KEYS = (CONTEXT_DECISION, CONTEXT_VALID, CONTEXT_PATH_LENGTH, CONTEXT_COURSE)


# ── the decision vector ──────────────────────────────────────────────────────

def decision_names(config: TSConfig) -> tuple[str, ...]:
    return ("d_join", "via_d", "via_xt", "via_cos", "via_sin",
            *(f"slowness_{k}" for k in range(config.closure_slowness_knots + 1)),
            *(f"height_{k}" for k in range(config.closure_height_knots)))


def decision_width(config: TSConfig) -> int:
    return 5 + (config.closure_slowness_knots + 1) + config.closure_height_knots


@dataclass(frozen=True)
class Decision:
    """One decision vector split into its named parts (numpy, physical units)."""
    d_join_m: float
    via_d_m: float
    via_xt_m: float
    via_heading_rel_rad: float
    slowness_knots: np.ndarray       # [K_s + 1] s/m
    height_knots: np.ndarray         # [K_h + 1] m, the last 0


def split_decision(vector: np.ndarray, config: TSConfig) -> Decision:
    v = np.asarray(vector, dtype=np.float64)
    ks, kh = config.closure_slowness_knots, config.closure_height_knots
    return Decision(
        d_join_m=float(v[0]), via_d_m=float(v[1]), via_xt_m=float(v[2]),
        via_heading_rel_rad=math.atan2(float(v[4]), float(v[3])),
        slowness_knots=v[5:5 + ks + 1].copy(),
        height_knots=np.append(v[5 + ks + 1:5 + ks + 1 + kh], 0.0),
    )


def decision_from_label(label: dict[str, Any], config: TSConfig) -> np.ndarray:
    """The decision vector of one labels-file entry (``fit_labels`` output)."""
    geometry = label["geometry"]["params"]
    profile_s = label["profile"][str(config.closure_slowness_knots)]
    profile_h = label["profile"][str(config.closure_height_knots)]
    heading = float(geometry["via_heading_rel"])
    return np.array([
        geometry["d_join"], geometry["via_d"], geometry["via_xt"], math.cos(heading), math.sin(heading),
        *profile_s["slowness_knots"], *profile_h["height_knots"][:-1],
    ], dtype=np.float64)


# ── labels ───────────────────────────────────────────────────────────────────

def fit_labels(series: "FlightSeries", anchor: int) -> dict[str, Any]:
    """The closure decoder's label for one flight from its truth: the canonical F3
    geometry (join at the localizer entry, via in the chart and in runway axes), the
    profiles at every ``LABEL_KNOTS`` on the truth path with their residuals, ``valid``
    (canonical and within ``LABEL_RESIDUAL_MAX_M``), and the difficulty covariates."""
    psi = float(series.scenario.target.psi)
    a = series.values[anchor]
    truth_xy = np.concatenate([a[None, :2], series.supervision_values[anchor + 1:, :2]], 0)
    truth_t = np.concatenate([[0.0], series.supervision_times[anchor + 1:] - series.times[anchor]])
    truth_u = np.concatenate([[a[2]], series.supervision_values[anchor + 1:, 2]])
    anchor_pose = cg.AnchorPose.from_state(a, psi)
    join = truth_join_point(series)
    d_join0 = float(cg.runway_axes_np(join[0], join[1], psi)[0])
    f0 = cg.rule_template(anchor_pose, psi, d_join0)
    f1 = cg.fit_rule_template(anchor_pose, psi, truth_xy, d_join0)
    f2 = cg.fit_dubins_join(anchor_pose, psi, truth_xy, d_join0, seed=f1)
    f3, spread = cg.fit_via_dubins(anchor_pose, psi, truth_xy, d_join0, seeds=(f1, f2))
    f, length = cp.progress(truth_xy)
    profile: dict[str, dict[str, Any]] = {}
    for k in LABEL_KNOTS:
        slowness = cp.fit_slowness_knots(f, length, truth_t, k)
        heights = cp.fit_height_knots(f, truth_u, k)
        times = cg.strictly_increasing(cp.times_from_slowness(f, length, slowness))
        fitted_u = cp.height_from_knots(f, heights)
        profile[str(k)] = {
            "slowness_knots": slowness.tolist(), "height_knots": heights.tolist(),
            "position_error_m": float(np.mean(np.abs(fitted_u - truth_u))),
            "duration_error_s": float(times[-1] - truth_t[-1]),
            "time_error_s": float(np.mean(np.abs(times - truth_t))),
        }
    error = cg.path_error_m(f3.horizontal, truth_xy)
    return {
        "flight_id": series.flight_id, "runway": series.scenario.source.get("runway"),
        "anchor": {"d_m": anchor_pose.d, "xt_m": anchor_pose.xt, "heading_rad": anchor_pose.heading,
                   "speed_mps": anchor_pose.speed_mps, "u_m": float(a[2])},
        "d_join_truth_m": d_join0,
        "geometry": {"kind": f3.kind, "params": f3.params, "error_m": error,
                     "via_label_spread_m": spread, "rule_kind": f0.kind,
                     "length_ratio": f3.length / float(cumulative_arc_m(truth_xy)[-1])},
        "profile": profile, "duration_s": float(truth_t[-1]), "path_length_m": length,
        "valid": bool(f3.params["canonical"] and error <= LABEL_RESIDUAL_MAX_M),
        "difficulty": approach_difficulty(series, anchor).to_dict(),
    }


@dataclass(frozen=True)
class ClosureLabels:
    """A labels file: the airport it was fitted for and its flights keyed by
    ``FlightSeries.flight_id`` (a flight key is unique within an airport only)."""
    airport: str
    flights: dict[str, dict[str, Any]]


def load_labels(path: str | Path) -> ClosureLabels:
    """Refuses another schema (the payload changed with the schema name); the airport is
    normalised the way ``FlightSeries.airport`` is."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != LABEL_SCHEMA:
        raise ValueError(f"{path}: expected closure labels schema {LABEL_SCHEMA!r}, got {payload.get('schema')!r}")
    return ClosureLabels(str(payload["airport"] or "").strip().upper(), payload["flights"])


def check_airport(series: "FlightSeries", labels: ClosureLabels) -> None:
    """A labels file of another airport is refused: flight keys are unique within an
    airport only, so a match across airports would be chance, never identity."""
    if labels.airport != series.airport:
        raise ValueError(f"closure labels were fitted for {labels.airport!r}, this flight is at {series.airport!r}")


def label_context(series: "FlightSeries", labels: ClosureLabels, config: TSConfig) -> dict[str, np.ndarray]:
    """The per-flight context rows: the decision vector and its validity (0 for a flight
    the labels file does not carry or marks invalid — such a flight is in the batch but
    out of the loss), the label's path length (turns slowness error into seconds), and
    the runway course the replay reconstructs with."""
    check_airport(series, labels)
    label = labels.flights.get(series.flight_id)
    width = decision_width(config)
    if label is None or not label["valid"]:
        decision, valid, length = np.zeros(width), 0.0, 0.0
    else:
        decision, valid, length = decision_from_label(label, config), 1.0, float(label["path_length_m"])
    return {
        CONTEXT_DECISION: decision.astype(np.float32),
        CONTEXT_VALID: np.array(valid, dtype=np.float32),
        CONTEXT_PATH_LENGTH: np.array(length, dtype=np.float32),
        CONTEXT_COURSE: np.array(float(series.scenario.target.psi), dtype=np.float64),
    }


def probe_closure_context(batch_size: int, device: torch.device, config: TSConfig) -> dict[str, torch.Tensor]:
    """One representative closure context for shape/throughput probes."""
    decision = torch.zeros((batch_size, decision_width(config)), dtype=torch.float32, device=device)
    decision[:, 0] = 8_000.0
    decision[:, 3] = 1.0
    decision[:, 5:5 + config.closure_slowness_knots + 1] = 1.0 / 80.0
    return {
        CONTEXT_DECISION: decision,
        CONTEXT_VALID: torch.ones(batch_size, dtype=torch.float32, device=device),
        CONTEXT_PATH_LENGTH: torch.full((batch_size,), 20_000.0, dtype=torch.float32, device=device),
        CONTEXT_COURSE: torch.zeros(batch_size, dtype=torch.float64, device=device),
    }


# ── the model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClosurePrediction:
    """One decision vector per flight, ``[B, decision_width]``, in physical units."""
    decision: torch.Tensor


def decode_raw(raw: torch.Tensor, config: TSConfig) -> torch.Tensor:
    """Free network outputs → the physical decision vector, every group bounded to what
    the family can draw: d_join in ``[D_JOIN_MIN_M, D_JOIN_MIN_M + D_JOIN_MAX_M]``
    (sigmoid), via distances in ``±VIA_MAX_M`` (tanh), the heading as a near-unit vector
    (``raw / sqrt(|raw|² + 1e-6)``: exact away from the origin, its gradient bounded at
    it — ``split_decision`` reads it through ``atan2``, which needs no unit norm), slowness inside
    the profile's speed bounds (sigmoid), heights at the km scale (free)."""
    ks = config.closure_slowness_knots
    d_join = cg.D_JOIN_MIN_M + torch.sigmoid(raw[:, 0:1]) * D_JOIN_MAX_M
    via = torch.tanh(raw[:, 1:3]) * VIA_MAX_M
    heading = raw[:, 3:5] / torch.sqrt(raw[:, 3:5].pow(2).sum(dim=1, keepdim=True) + 1e-6)
    slowness = SLOWNESS_MIN + torch.sigmoid(raw[:, 5:5 + ks + 1]) * (SLOWNESS_MAX - SLOWNESS_MIN)
    heights = raw[:, 5 + ks + 1:] * HEIGHT_SCALE_M
    return torch.cat([d_join, via, heading, slowness, heights], dim=1)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class ClosureOutputModel(nn.Module):
    """The backbone's flattened tokens → an MLP → the decision vector. The backbone's
    state head is discarded; the context (labels) is accepted and ignored — a label is
    never an input."""

    def __init__(self, config: TSConfig, feature_encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.feature_encoder = feature_encoder
        self.feature_encoder.discard_state_head()
        width = config.enc_in * config.d_model
        self.head = nn.Sequential(
            nn.Linear(width, config.d_model), nn.GELU(), nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, decision_width(config)),
        )
        # Start every flight at one plausible decision — the join 8 km out, the via 8 km
        # out on the centreline heading inbound, 80 m/s throughout, heights at 0 — so
        # the first validation replays draw approaches, not loops.
        with torch.no_grad():
            last = self.head[-1]
            last.weight.mul_(0.1)
            last.bias.zero_()
            last.bias[0] = _logit((8_000.0 - cg.D_JOIN_MIN_M) / D_JOIN_MAX_M)
            last.bias[1] = math.atanh(8_000.0 / VIA_MAX_M)
            last.bias[3] = 1.0
            last.bias[5:5 + config.closure_slowness_knots + 1] = _logit(
                (1.0 / 80.0 - SLOWNESS_MIN) / (SLOWNESS_MAX - SLOWNESS_MIN))

    def forward(self, history: torch.Tensor, context: dict[str, torch.Tensor] | None = None) -> ClosurePrediction:
        del context
        return ClosurePrediction(decode_raw(self.head(self.feature_encoder.encode_features(history)), self.config))


# ── the loss ─────────────────────────────────────────────────────────────────

def closure_loss_components(prediction: ClosurePrediction, normalized_anchor_state, target_states, state_weights,
                            target_final_time_s, flight_weights: torch.Tensor, config: TSConfig,
                            normalizer: "Normalizer", dynamics: dict[str, torch.Tensor] | None = None,
                            dense_supervision=None, training_stage=None, *, multipliers=None):
    """L1 regression of the decision vector against the flight's label, over the valid
    flights: ``state`` = the geometry (the join distance and the via's mean distance
    error at the 10 km scale, the heading as its unit vector), ``final_time`` = the
    slowness knots turned into seconds with the label's path length, at
    ``closure_timing_scale_s``, ``kinematic`` = the height knots at the km scale,
    ``terminal`` = 0 (nothing to add: the path ends at the threshold by construction).
    Weights are the flight weights times ``closure_valid``. A batch with no valid flight
    contributes zero (the dataset refuses a labels file with no valid flight at all, and
    reports the covered share, so a low share is visible, never silent)."""
    del normalized_anchor_state, target_states, state_weights, target_final_time_s, normalizer
    del dense_supervision, training_stage, multipliers
    if dynamics is None or CONTEXT_DECISION not in dynamics:
        raise ValueError("the closure loss needs the per-flight label context")
    ks = config.closure_slowness_knots
    pred, target = prediction.decision, dynamics[CONTEXT_DECISION].to(prediction.decision.dtype)
    weight = flight_weights * dynamics[CONTEXT_VALID].to(flight_weights.dtype)
    total = weight.sum().clamp(min=1e-6)
    delta = (pred - target).abs()
    via_mean = 0.5 * (delta[:, 1] + delta[:, 2])
    geometry = delta[:, 0] / D_JOIN_SCALE_M + via_mean / VIA_SCALE_M + delta[:, 3] + delta[:, 4]
    seconds = delta[:, 5:5 + ks + 1].mean(dim=1) * dynamics[CONTEXT_PATH_LENGTH].to(pred.dtype)
    timing = seconds / config.closure_timing_scale_s
    height = delta[:, 5 + ks + 1:].mean(dim=1) / HEIGHT_SCALE_M

    def weighted(term: torch.Tensor) -> torch.Tensor:
        return (term * weight).sum() / total

    heading_error = torch.atan2(pred[:, 4], pred[:, 3]) - torch.atan2(target[:, 4], target[:, 3])
    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
    return LossComponents(
        state=weighted(geometry) * config.closure_geometry_loss_weight,
        final_time=weighted(timing) * config.closure_timing_loss_weight,
        kinematic=weighted(height) * config.closure_height_loss_weight,
        terminal=pred.new_zeros(()),
        diagnostics={
            "closure_d_join_mae_m": weighted(delta[:, 0]).detach(),
            "closure_via_mae_m": weighted(via_mean).detach(),
            "closure_heading_mae_deg": weighted(heading_error.abs()).detach() * (180.0 / math.pi),
            "closure_duration_mae_s": weighted(seconds).detach(),
            "closure_height_mae_m": weighted(delta[:, 5 + ks + 1:].mean(dim=1)).detach(),
            "closure_valid_share": (weight > 0).float().mean().detach(),
        },
    )


# ── reconstruction ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Reconstruction:
    """The dense trajectory after the anchor: offsets from the anchor (s, strictly
    increasing), physical channel values ``[N, 6]`` (positions, chart velocities), the
    path, and which construction drew it."""
    offsets_s: np.ndarray
    values: np.ndarray
    path: cg.ClosurePath
    construction: str

    @property
    def final_time_s(self) -> float:
        return float(self.offsets_s[-1])


def reconstruct(vector: np.ndarray, anchor_channels: np.ndarray, psi: float, config: TSConfig) -> Reconstruction:
    """Draw the trajectory a decision vector describes from the anchor state (physical
    channel space). The via-Dubins path when the via lies at least one turn radius from
    the anchor and the path exists; the plain CSC to the join when the via is inside
    that radius (``KIND_VIA_AT_ANCHOR``: a straight-in flight's label puts it AT the
    anchor, and a near miss there is a full circle) or the via path does not exist; else
    the straight line to the threshold — the construction used is reported."""
    decision = split_decision(vector, config)
    anchor = cg.AnchorPose.from_state(anchor_channels, psi)
    via_e, via_n = cg.chart_from_axes_np(decision.via_d_m, decision.via_xt_m, psi)
    via_heading = cg.wrap_angle(psi + decision.via_heading_rel_rad)
    path, construction = None, cg.KIND_VIA_DUBINS
    if np.hypot(via_e - anchor.position[0], via_n - anchor.position[1]) >= anchor.radius:
        path = cg.via_dubins(anchor, psi, decision.d_join_m, float(via_e), float(via_n), via_heading)
    else:
        construction = KIND_VIA_AT_ANCHOR
    if path is None:
        path = cg.dubins_join(anchor, psi, decision.d_join_m, 0.0)
        construction = construction if construction == KIND_VIA_AT_ANCHOR else cg.KIND_DOWNWIND_DUBINS
    if path is None:
        path, construction = cg.straight_path(anchor), cg.KIND_STRAIGHT
    if path.length <= 0.0:
        raise ValueError("the anchor sits on the threshold: no path to draw")
    f = path.arc / path.length
    times = cg.strictly_increasing(cp.times_from_slowness(f, path.length, decision.slowness_knots))
    heights = cp.height_from_knots(f, decision.height_knots)
    speed = cp.speed_from_slowness(f, decision.slowness_knots)
    step = np.gradient(path.horizontal, axis=0)
    tangent = step / np.maximum(np.hypot(*step.T), 1e-9)[:, None]
    values = np.zeros((len(times), 6))
    values[:, 0], values[:, 1], values[:, 2] = path.horizontal[:, 0], path.horizontal[:, 1], heights
    values[:, 3], values[:, 4] = tangent[:, 0] * speed, tangent[:, 1] * speed
    values[:, 5] = np.gradient(heights, times)
    return Reconstruction(times[1:], values[1:], path, construction)


def sample_at_progress(reconstruction: Reconstruction, points: int) -> tuple[np.ndarray, np.ndarray]:
    """The reconstruction on ``points`` equal fractions of its own duration:
    ``(values [points, 6], segment_durations_s [points])`` — the replay's grid."""
    duration = reconstruction.final_time_s
    query = np.arange(1, points + 1, dtype=np.float64) / points * duration
    values = np.column_stack([np.interp(query, reconstruction.offsets_s, reconstruction.values[:, c]) for c in range(6)])
    return values, np.full(points, duration / points)


def replay_batch(prediction: ClosurePrediction, anchors_physical: np.ndarray, context: dict[str, torch.Tensor],
                 config: TSConfig, points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The training-time deployable replay: every decision reconstructed and sampled on
    ``points`` fractions of its duration → ``(predicted [B, points, 6], segment
    durations [B, points], predicted final times [B])``."""
    decisions = prediction.decision.detach().cpu().numpy().astype(np.float64)
    courses = context[CONTEXT_COURSE].detach().cpu().numpy().astype(np.float64)
    predicted = np.zeros((len(decisions), points, 6), dtype=np.float32)
    durations = np.zeros((len(decisions), points), dtype=np.float64)
    final = np.zeros(len(decisions), dtype=np.float64)
    for row, (vector, anchor, psi) in enumerate(zip(decisions, anchors_physical, courses)):
        rec = reconstruct(vector, anchor, float(psi), config)
        predicted[row], durations[row] = sample_at_progress(rec, points)
        final[row] = rec.final_time_s
    return predicted, durations, final
