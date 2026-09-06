"""Scene context → arrays (scene design §五 P2.c).

``scene_arrays`` turns one ``SceneContext`` into what a model consumes, all in the ego's
threshold chart and all from the OBSERVED half of every neighbour — ``future_label`` is
never read here (``tests/test_scene_features.py`` pins that by scrambling it):

* ``neighbour_static``: ``[N_MAX, len(STATIC_NAMES)]`` — the entity-level quantities
  (distance to the ego, distance to the threshold, ETA, its lead over the ego's ETA,
  established, age of the last sample, runway axes at the last sample);
  ``neighbour_valid``: ``[N_MAX]``, True where a slot holds a neighbour.
* ``scalars``: ``[len(SCALAR_NAMES)]`` — the runway-use scalars, with missing values
  (no landing yet, no lead) encoded as the stated sentinels.

The SEQUENCE half — each neighbour's ``[N_MAX, L, 6]`` chart track resampled onto the
ego's lookback grid, plus its ``[N_MAX, L]`` validity mask — was deleted 2026-09-07
(package audit T2): the L4 explainability gate did not pass, and its only consumer reads
the entity and scalar halves. Adding it back means re-deriving it against a model that
actually consumes a sequence.

Everything is metres, seconds and metres per second; scaling is the model's business.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flight_scenarios.scene_context import N_MAX, Neighbour, SceneContext

STATIC_NAMES = ("distance_to_ego_m", "distance_to_threshold_m", "eta_s", "eta_lead_s", "established",
                "age_s", "d_m", "xt_m", "height_m", "ground_speed_mps", "cos_heading", "sin_heading")
SCALAR_NAMES = ("since_last_landing_same_runway_s", "landings_recent", "landings_recent_same_runway",
                "same_runway_share_recent", "airborne_in_radius", "established_on_ego_final", "ahead_by_eta",
                "lead_eta_s", "lead_gap_s", "ego_eta_s", "hour_sin", "hour_cos", "weekday")
NO_LANDING_SENTINEL_S = 3_600.0      # since_last_landing when the runway has no landing yet in the roster
NO_LEAD_SENTINEL_S = 1_800.0         # lead ETA / gap when nobody is ahead by ETA
SHARE_UNKNOWN = 0.5                  # same-runway share when nothing landed recently


@dataclass(frozen=True)
class SceneArrays:
    neighbour_valid: np.ndarray   # [N_MAX] bool
    neighbour_static: np.ndarray  # [N_MAX, S] float32
    scalars: np.ndarray           # [K] float32


def static_row(neighbour: Neighbour, ego_eta_s: float) -> np.ndarray:
    o = neighbour.observed
    return np.array([
        neighbour.distance_to_ego_m, o.distance_to_threshold_m, o.eta_s, ego_eta_s - o.eta_s, float(o.established),
        o.age_s, o.d_m, o.xt_m, float(o.height_m[-1]), o.ground_speed_mps, np.cos(o.heading_rad), np.sin(o.heading_rad),
    ], dtype=np.float64)


def scalar_row(scene: SceneContext) -> np.ndarray:
    s = scene.scalars
    angle = 2.0 * np.pi * s.hour_utc / 24.0
    return np.array([
        NO_LANDING_SENTINEL_S if s.since_last_landing_same_runway_s is None else min(s.since_last_landing_same_runway_s, NO_LANDING_SENTINEL_S),
        s.landings_recent, s.landings_recent_same_runway,
        SHARE_UNKNOWN if s.same_runway_share_recent is None else s.same_runway_share_recent,
        s.airborne_in_radius, s.established_on_ego_final, s.ahead_by_eta,
        NO_LEAD_SENTINEL_S if s.lead_eta_s is None else s.lead_eta_s,
        NO_LEAD_SENTINEL_S if s.lead_gap_s is None else s.lead_gap_s,
        scene.ego_eta_s, np.sin(angle), np.cos(angle), s.weekday,
    ], dtype=np.float64)


def scene_arrays(scene: SceneContext, *, n_max: int = N_MAX) -> SceneArrays:
    """The entity and scalar arrays for one scene, in the ego's own threshold chart."""
    valid = np.zeros(n_max, dtype=bool)
    static = np.zeros((n_max, len(STATIC_NAMES)), dtype=np.float32)
    for slot, neighbour in enumerate(scene.neighbours[:n_max]):
        valid[slot] = True
        static[slot] = static_row(neighbour, scene.ego_eta_s)
    return SceneArrays(valid, static, scalar_row(scene).astype(np.float32))
