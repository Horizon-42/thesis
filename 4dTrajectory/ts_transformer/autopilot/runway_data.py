"""The runway data the JUDGE reads beyond the vocabulary and the candidates' geometry (vocabulary §5.8): each candidate's
published vertical path — its threshold crossing height (TCH) and glidepath angle, the glidepath the decision-altitude
check measures the height against — and its decision altitude as a height above the threshold. The executor's laws read
none of it (§5.2, D9).

Torch-free, so the labeller's first runner checks the runways it has already loaded (`experiments.instruction_signals`:
a candidate without a TCH, a glidepath or a decision altitude is refused before anything is written, §4.2). The runway
data is the harvest's (`trajectory_data_process.harvest.airports.load_airport`, the FAA CIFP's vertical path and the
plate's minima) at the configuration and CIFP the harvest and the evaluator read by default. The decision altitude is
the harvest's `Runway.decision_height_above_threshold_m`: the LPV line's, or the LNAV/VNAV line's where the runway
publishes no LPV (KRDU 32, KSMF 35R — Claude's reading of "the LPV DA" of §2 for those two candidates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from trajectory_data_process.harvest.airports import Runway, load_airport
from ts_transformer.instructions.airport import AirportGeometry


@dataclass(frozen=True)
class VerticalPath:
    """One runway end's published vertical path and decision altitude."""

    crossing_height_m: float        # the TCH, m above the threshold
    glidepath_deg: float            # the glidepath angle, deg
    decision_height_m: float        # the DA, m above the threshold


def vertical_paths(geometry: AirportGeometry, runways: Sequence[Runway]) -> tuple[VerticalPath, ...]:
    """Each candidate's published vertical path, in the candidates' order, from ``runways`` (the harvest's runway ends
    for the airport). A candidate without a TCH, a glidepath or vertically guided minima is refused: the judge cannot
    check its decision altitude."""
    by_ident = {runway.ident: runway for runway in runways}
    missing = [candidate.ident for candidate in geometry.candidates if candidate.ident not in by_ident]
    if missing:
        raise ValueError(f"{geometry.code} candidates {missing} are not among the harvest's runways")
    paths = []
    for candidate in geometry.candidates:
        runway = by_ident[candidate.ident]
        if runway.threshold_crossing_height_m is None or runway.published_glidepath_deg is None:
            raise ValueError(f"{geometry.code} {candidate.ident} publishes no threshold crossing height or glidepath")
        if not runway.published_minima.vertically_guided:
            raise ValueError(f"{geometry.code} {candidate.ident} publishes no decision altitude "
                             f"({runway.published_minima.note})")
        paths.append(VerticalPath(crossing_height_m=float(runway.threshold_crossing_height_m),
                                  glidepath_deg=float(runway.published_glidepath_deg),
                                  decision_height_m=float(runway.decision_height_above_threshold_m)))
    return tuple(paths)


def published_vertical_paths(geometry: AirportGeometry) -> tuple[VerticalPath, ...]:
    """`vertical_paths` from the harvest's runway data for ``geometry``'s airport at the default configuration and CIFP."""
    return vertical_paths(geometry, load_airport(geometry.code, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP).runways)
