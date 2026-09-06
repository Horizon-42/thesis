"""Scene context → arrays (scene design §五 P2.c).

``scene_arrays`` turns one ``SceneContext`` into what a model consumes, all in the ego's
threshold chart and all from the OBSERVED half of every neighbour — ``future_label`` is
never read here (``tests/test_scene_features.py`` pins that by scrambling it):

* ``neighbours``: ``[N_MAX, L, 6]`` — each neighbour's chart position (e, n, height) and
  velocity (finite differences of the position) resampled onto the ego's own lookback
  grid (``L`` steps of ``dt_s`` ending at t₀), zero where the neighbour has no sample;
  ``neighbour_mask``: ``[N_MAX, L]``, True where the grid time lies inside the
  neighbour's sampled span; ``neighbour_valid``: ``[N_MAX]``.
* ``neighbour_static``: ``[N_MAX, len(STATIC_NAMES)]`` — the entity-level quantities
  (distance to the ego, distance to the threshold, ETA, its lead over the ego's ETA,
  established, age of the last sample, runway axes at the last sample).
* ``scalars``: ``[len(SCALAR_NAMES)]`` — the runway-use scalars, with missing values
  (no landing yet, no lead) encoded as the stated sentinels.

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
    neighbours: np.ndarray        # [N_MAX, L, 6] float32
    neighbour_mask: np.ndarray    # [N_MAX, L] bool
    neighbour_valid: np.ndarray   # [N_MAX] bool
    neighbour_static: np.ndarray  # [N_MAX, S] float32
    scalars: np.ndarray           # [K] float32


def _series_on_grid(neighbour: Neighbour, grid_rel_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = neighbour.observed
    inside = (grid_rel_s >= o.t_rel_s[0]) & (grid_rel_s <= o.t_rel_s[-1])
    e = np.interp(grid_rel_s, o.t_rel_s, o.e_m)
    n = np.interp(grid_rel_s, o.t_rel_s, o.n_m)
    h = np.interp(grid_rel_s, o.t_rel_s, o.height_m)
    rows = np.zeros((len(grid_rel_s), 6), dtype=np.float64)
    rows[:, 0], rows[:, 1], rows[:, 2] = e, n, h
    if inside.sum() >= 2:
        idx = np.flatnonzero(inside)
        rows[idx, 3] = np.gradient(e[idx], grid_rel_s[idx])
        rows[idx, 4] = np.gradient(n[idx], grid_rel_s[idx])
        rows[idx, 5] = np.gradient(h[idx], grid_rel_s[idx])
    rows[~inside] = 0.0
    return rows, inside


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


def scene_arrays(scene: SceneContext, *, seq_len: int, dt_s: float, n_max: int = N_MAX) -> SceneArrays:
    """The arrays for one scene on the ego's lookback grid (``seq_len`` steps of ``dt_s``
    ending at t₀, the ts window's own clock)."""
    grid = -dt_s * np.arange(seq_len - 1, -1, -1, dtype=np.float64)
    neighbours = np.zeros((n_max, seq_len, 6), dtype=np.float32)
    mask = np.zeros((n_max, seq_len), dtype=bool)
    valid = np.zeros(n_max, dtype=bool)
    static = np.zeros((n_max, len(STATIC_NAMES)), dtype=np.float32)
    for slot, neighbour in enumerate(scene.neighbours[:n_max]):
        rows, inside = _series_on_grid(neighbour, grid)
        neighbours[slot], mask[slot], valid[slot] = rows, inside, True
        static[slot] = static_row(neighbour, scene.ego_eta_s)
    return SceneArrays(neighbours, mask, valid, static, scalar_row(scene).astype(np.float32))
