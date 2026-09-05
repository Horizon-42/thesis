"""Train-anchor eligibility policies, separate from temporal window construction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable

import numpy as np

from channels import IDX
from config import (
    PREDICTION_CLOSURE,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
)

if TYPE_CHECKING:
    from dataset import FlightSeries

SEA_LEVEL_DENSITY_KG_M3 = 1.225
GRAVITY_MPS2 = 9.81
CONTROL_ANCHOR_STALL_MARGIN = 1.10


def _all_temporal_candidates(
    _series: FlightSeries, anchors: Sequence[int]
) -> list[int]:
    return list(anchors)


def _airborne_control_candidates(
    series: FlightSeries, anchors: Sequence[int]
) -> list[int]:
    """Retain anchors where the observed state is inside the airborne model domain.

    The control ODE divides by airspeed and is not defined for stopped ground reports.
    Its existing optimization counterpart uses the same 1.10 x sea-level stall-speed
    floor. Channel horizontal derivatives differ from physical ENU velocity only by the
    local chart transport factors (well below this 10% margin in terminal airspace).
    """
    candidate_indices = np.fromiter(anchors, dtype=np.int64)
    if not len(candidate_indices):
        return []
    velocity_indices = [IDX["edot"], IDX["ndot"], IDX["udot"]]
    observed_speed = np.linalg.norm(
        series.values[candidate_indices][:, velocity_indices], axis=1
    )
    scenario = series.scenario
    stall_speed = math.sqrt(
        2.0 * float(scenario.initial.m) * GRAVITY_MPS2
        / (
            SEA_LEVEL_DENSITY_KG_M3
            * float(scenario.aero.S)
            * float(scenario.aero.Cl_max)
        )
    )
    return candidate_indices[
        observed_speed >= CONTROL_ANCHOR_STALL_MARGIN * stall_speed
    ].tolist()


AnchorEligibility = Callable[["FlightSeries", Sequence[int]], list[int]]
_POLICIES: dict[str, tuple[str, AnchorEligibility]] = {
    PREDICTION_STATE: ("temporal-only-v1", _all_temporal_candidates),
    # Labels are fitted at the fixed anchor; TSConfig refuses random anchors for closure,
    # so this policy only ever names the fixed-anchor population.
    PREDICTION_CLOSURE: ("temporal-only-v1", _all_temporal_candidates),
    PREDICTION_CONTROL: (
        "airborne-1.10-stall-margin-v1",
        _airborne_control_candidates,
    ),
}


def random_train_anchor_eligibility_policy(config: TSConfig) -> str:
    return _POLICIES[config.prediction_output][0]


def eligible_random_train_anchors(
    series: FlightSeries,
    anchors: Sequence[int],
    config: TSConfig,
) -> list[int]:
    """Apply the output strategy's declared random-train-anchor domain."""
    return _POLICIES[config.prediction_output][1](series, anchors)
