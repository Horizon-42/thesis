"""The runway data the executor reads beyond the vocabulary and the candidates' geometry: each candidate's published
vertical path — its threshold crossing height (TCH), where "descend to land" crosses the threshold, and its glidepath
angle, whose lower edge "descend to land" does not descend under before the threshold (`vertical`).

Torch-free, so the labeller's first runner checks the runways it has already loaded (`experiments.instruction_signals`:
a candidate without a published vertical path is refused before anything is written). The runway data is the harvest's
(`trajectory_data_process.harvest.airports.load_airport`, the FAA CIFP's vertical path) at the configuration and CIFP
the harvest and the evaluator read by default — the evaluation's own reference plane, and the glidepath the second
stage's lower edge is drawn from (`prior.procedure`). The paths come from the evaluation's CLI, the one place they are
defined; that module decides nothing about a flight, so it is left out of the executor's source hash
(`spec.UNHASHED_IMPORTS`): a replay records the crossing heights it flew to, and the executor spec records every
candidate's vertical path as the spec was written (`experiments.executor_spec`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from trajectory_data_process.harvest.airports import Runway, load_airport
from ts_transformer.instructions.airport import AirportGeometry


@dataclass(frozen=True)
class VerticalPath:
    """One runway end's published vertical path."""

    crossing_height_m: float        # the TCH, m above the threshold
    glidepath_deg: float            # the glidepath angle, deg


def vertical_paths(geometry: AirportGeometry, runways: Sequence[Runway]) -> tuple[VerticalPath, ...]:
    """Each candidate's published vertical path, in the candidates' order, from ``runways`` (the harvest's runway ends
    for the airport). A candidate that publishes no TCH or no glidepath is refused: "descend to land" has no crossing
    point or no glidepath there."""
    by_ident = {runway.ident: runway for runway in runways}
    missing = [candidate.ident for candidate in geometry.candidates if candidate.ident not in by_ident]
    if missing:
        raise ValueError(f"{geometry.code} candidates {missing} are not among the harvest's runways")
    paths = []
    for candidate in geometry.candidates:
        runway = by_ident[candidate.ident]
        if runway.threshold_crossing_height_m is None or runway.published_glidepath_deg is None:
            raise ValueError(f"{geometry.code} {candidate.ident} publishes no threshold crossing height or glidepath")
        paths.append(VerticalPath(crossing_height_m=float(runway.threshold_crossing_height_m),
                                  glidepath_deg=float(runway.published_glidepath_deg)))
    return tuple(paths)


def published_vertical_paths(geometry: AirportGeometry) -> tuple[VerticalPath, ...]:
    """`vertical_paths` from the harvest's runway data for ``geometry``'s airport at the default configuration and CIFP."""
    return vertical_paths(geometry, load_airport(geometry.code, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways)
