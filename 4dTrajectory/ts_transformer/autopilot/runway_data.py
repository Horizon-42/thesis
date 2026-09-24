"""The one piece of runway data the executor reads beyond the vocabulary and the candidates' geometry: each
candidate's published threshold crossing height (TCH), where "descend to land" crosses the threshold (`vertical`).

Torch-free, so the labeller's first runner checks the runways it has already loaded (`experiments.instruction_signals`:
a candidate without a TCH is refused before anything is written). The runway data is the harvest's
(`trajectory_data_process.harvest.airports.load_airport`, the FAA CIFP's vertical path) at the configuration and CIFP
the harvest and the evaluator read by default — the evaluation's own reference plane. The paths come from the
evaluation's CLI, the one place they are defined; that module decides nothing about a flight, so it is left out of the
executor's source hash (`spec.UNHASHED_IMPORTS`), and a replay records the heights it flew to instead.
"""

from __future__ import annotations

from typing import Sequence

from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from trajectory_data_process.harvest.airports import Runway, load_airport
from ts_transformer.instructions.airport import AirportGeometry


def crossing_heights(geometry: AirportGeometry, runways: Sequence[Runway]) -> tuple[float, ...]:
    """Each candidate's published TCH (m above its threshold), in the candidates' order, from ``runways`` (the
    harvest's runway ends for the airport). A candidate that publishes none is refused: "descend to land" has no
    crossing point there."""
    by_ident = {runway.ident: runway for runway in runways}
    missing = [candidate.ident for candidate in geometry.candidates if candidate.ident not in by_ident]
    if missing:
        raise ValueError(f"{geometry.code} candidates {missing} are not among the harvest's runways")
    heights = []
    for candidate in geometry.candidates:
        height = by_ident[candidate.ident].threshold_crossing_height_m
        if height is None:
            raise ValueError(f"{geometry.code} {candidate.ident} publishes no threshold crossing height")
        heights.append(float(height))
    return tuple(heights)


def published_crossing_heights(geometry: AirportGeometry) -> tuple[float, ...]:
    """`crossing_heights` from the harvest's runway data for ``geometry``'s airport at the default configuration and CIFP."""
    return crossing_heights(geometry, load_airport(geometry.code, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways)
