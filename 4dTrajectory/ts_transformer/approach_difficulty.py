"""How hard the flight in front of the anchor actually is, measured from the truth.

ADE/FDE alone do not compare across airports, because the airports do not pose the same
problem. Measured on the five-airport fleet (2026-08-20), the share of validation flights
that are already established straight-in at the evaluation anchor runs from 41 % (KSMF) to
**78 %** (KSJC), and within a matched stratum every airport scores about the same:
412-509 m median ADE on "straight, under 13 km left to fly". Reweighting each airport to the
pooled mix moves KSJC from 483 m — the best of the five, by 1.7x — to 1526 m, the worst.
The headline number was reporting the route mix, not the model.

So every prediction row carries the covariates that mix is made of, computed from the
OBSERVED track the error is scored against and never from the prediction:

``anchor_range_m``      straight-line horizontal distance from the anchor to the threshold
``remaining_path_m``    horizontal arc length the aircraft actually flew from there
``route_tortuosity``    the ratio — 1.0 is a straight-in, 2.4 is a full downwind and base
``anchor_cross_track_m`` signed offset from the extended centreline, positive to the RIGHT
                        of the inbound approach course
``established_at_anchor`` on the centreline, before the threshold, and tracking inbound

Horizontal only, deliberately: tortuosity is a lateral notion and folding the descent into it
would make a steep approach look vectored. The truth curve is the same one
``common_physical_time_flight_metrics`` scores against — the supervision samples after the
anchor — so a covariate and the error it explains describe one curve, not two.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from channels import POSITION_IDX, VELOCITY_IDX

if TYPE_CHECKING:  # avoid a dataset <-> difficulty import cycle at runtime
    from dataset import FlightSeries

DIFFICULTY_SCHEMA = "ts-approach-difficulty-v1"

# Established on the extended centreline. 500 m is the width the fleet's own distribution
# picks out: at 15 km to go the interquartile cross-track spread is 12-125 m at four of the
# five airports, while an aircraft still on a downwind sits 8-10 km off. Nothing in the data
# lands near 500 m, so the exact value does not decide any flight's label.
ESTABLISHED_CROSS_TRACK_M = 500.0
# ...and tracking inbound, not outbound on a teardrop over the same ground.
ESTABLISHED_TRACK_TOLERANCE_DEG = 30.0
# The tortuosity above which a route is no longer a straight-in. 1.05 is what the fleet's
# own bimodal distribution separates at; every readout that reports a stratum uses it.
STRAIGHT_TORTUOSITY = 1.05
# The stratum labels ARE the schema every readout keys on: a consumer that restates one
# and drifts gets a silently empty stratum, not an error.
STRATUM_ALL = "all"
STRATUM_STRAIGHT_IN = f"straight-in (tortuosity < {STRAIGHT_TORTUOSITY})"
STRATUM_VECTORED = f"vectored (tortuosity >= {STRAIGHT_TORTUOSITY}, not established)"
STRATUM_ESTABLISHED = "established at anchor"
STRATUM_NEAR = "remaining path < 13 km"
STRATUM_FAR = "remaining path >= 13 km"


def strata_masks(rows: dict[str, dict[str, Any]], keys: list[str]) -> dict[str, np.ndarray]:
    """The standard readout strata over rows carrying the difficulty covariates.

    One source for every readout and measurement that reports "straight-in" against
    "vectored": a stratum defined twice is a comparison between two different populations.
    ``rows`` maps a key to a scored row (a ``summary.json`` result, say); ``keys`` fixes
    the order the masks are aligned to.
    """
    tortuosity = np.array([rows[key]["route_tortuosity"] for key in keys])
    established = np.array([bool(rows[key]["established_at_anchor"]) for key in keys])
    remaining = np.array([rows[key]["remaining_path_m"] for key in keys])
    return {
        STRATUM_ALL: np.ones(len(keys), dtype=bool),
        STRATUM_STRAIGHT_IN: tortuosity < STRAIGHT_TORTUOSITY,
        STRATUM_VECTORED: (tortuosity >= STRAIGHT_TORTUOSITY) & ~established,
        STRATUM_ESTABLISHED: established,
        STRATUM_NEAR: remaining < 13_000.0,
        STRATUM_FAR: remaining >= 13_000.0,
    }


@dataclass(frozen=True)
class ApproachDifficulty:
    """The route the anchor still has to fly, as the observed track flew it."""

    anchor_range_m: float
    remaining_path_m: float
    route_tortuosity: float
    anchor_cross_track_m: float
    established_at_anchor: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def difficulty_policy() -> dict[str, Any]:
    """The thresholds ``established_at_anchor`` encodes, for the summary manifest.

    Published beside the rows because a boolean whose definition lives only in source is a
    boolean that gets quoted without it.
    """
    return {
        "schema_version": DIFFICULTY_SCHEMA,
        "established_cross_track_m": ESTABLISHED_CROSS_TRACK_M,
        "established_track_tolerance_deg": ESTABLISHED_TRACK_TOLERANCE_DEG,
    }


def approach_difficulty(series: "FlightSeries", anchor: int) -> ApproachDifficulty:
    """Difficulty covariates for one flight at one anchor."""
    target = series.scenario.target
    if target is None:
        raise ValueError(
            f"{series.flight_id}: the scenario carries no target, so there is no approach "
            "course to place the anchor against"
        )
    anchor_time = float(series.times[anchor])
    future = series.supervision_times > anchor_time
    truth = np.asarray(series.supervision_values, dtype=np.float64)[future][:, POSITION_IDX]
    anchor_state = np.asarray(series.values[anchor], dtype=np.float64)
    # Positions RELATIVE TO THE TARGET: range, cross-track and distance-to-go are all
    # measured from the threshold, which is the chart origin only under the
    # threshold-anchored frames.
    path = np.vstack([anchor_state[list(POSITION_IDX)], truth]) - series.target_chart

    anchor_range_m = float(np.hypot(*path[0, :2]))
    if anchor_range_m <= 0.0:
        raise ValueError(
            f"{series.flight_id}: anchor sits on the threshold, so there is no approach "
            "left to characterise"
        )
    remaining_path_m = float(
        np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()
    )

    # The chart's two horizontal axes are frame-dependent (east/north or along/cross
    # runway); world EN is the one both frames agree on, so the course rotation happens
    # there and this stays correct under either coordinate_frame setting.
    east, north = series.frame.to_world_horizontal(float(path[0, 0]), float(path[0, 1]))
    course = float(target.psi)
    cosine, sine = math.cos(course), math.sin(course)
    # Inbound direction is (cos, sin); its right-hand normal is (sin, -cos). The anchor
    # lies BEFORE the threshold when its projection on the inbound direction is negative.
    anchor_cross_track_m = east * sine - north * cosine
    distance_to_go_m = -(east * cosine + north * sine)

    velocity = anchor_state[list(VELOCITY_IDX)]
    track_east, track_north = series.frame.to_world_horizontal(
        float(velocity[0]), float(velocity[1])
    )
    track_error_deg = abs(
        math.degrees(
            _wrapped(math.atan2(track_north, track_east) - course)
        )
    )
    established = bool(
        abs(anchor_cross_track_m) < ESTABLISHED_CROSS_TRACK_M
        and distance_to_go_m > 0.0
        and track_error_deg <= ESTABLISHED_TRACK_TOLERANCE_DEG
    )

    return ApproachDifficulty(
        anchor_range_m=anchor_range_m,
        remaining_path_m=remaining_path_m,
        route_tortuosity=remaining_path_m / anchor_range_m,
        anchor_cross_track_m=float(anchor_cross_track_m),
        established_at_anchor=established,
    )


def _wrapped(radians: float) -> float:
    """Angle difference folded onto (-pi, pi] — a course test must not wrap at the cut."""
    return (radians + math.pi) % (2.0 * math.pi) - math.pi


def difficulty_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch-level route mix, so a per-airport ADE is never read without it."""
    tortuosity = np.array([row["route_tortuosity"] for row in rows], dtype=np.float64)
    remaining = np.array([row["remaining_path_m"] for row in rows], dtype=np.float64)
    established = np.array(
        [row["established_at_anchor"] for row in rows], dtype=bool
    )
    return {
        **difficulty_policy(),
        "flights": len(rows),
        "established_at_anchor_fraction": float(established.mean()),
        "route_tortuosity": _spread(tortuosity),
        "remaining_path_m": _spread(remaining),
    }


def _spread(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }
