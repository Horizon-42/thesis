"""Shared CLI wiring for the two entry points (``__main__`` and ``visualize``).

Both read the same inputs and must judge them identically, so the gate and
established-criteria flags are defined once here — the HTML report recomputing
with different knobs than the JSON report would be a silent disagreement.
"""

from __future__ import annotations

import argparse

from evaluation.arrival import EstablishedCriteria
from evaluation.thresholds import DeviationThresholds


def add_judgement_args(parser: argparse.ArgumentParser) -> None:
    """The gate + established-criteria flags, with the package defaults."""
    gates = DeviationThresholds()
    parser.add_argument("--lateral-max-m", type=float, default=gates.lateral_max_m,
                        help="lateral gate (8260.58D Formula 3-1-1 course semiwidth floor)")
    parser.add_argument("--vertical-below-max-m", type=float,
                        default=gates.vertical_below_max_m,
                        help="vertical gate below target (8260.58D WCH window)")
    parser.add_argument("--vertical-above-max-m", type=float,
                        default=gates.vertical_above_max_m,
                        help="vertical gate above target (8260.58D WCH window)")

    criteria = EstablishedCriteria()
    parser.add_argument("--fit-window-m", type=float, nargs=2, default=list(criteria.window_m),
                        metavar=("OUTER", "INNER"),
                        help="observed fit window along-track (negative = before threshold); "
                             "a methodological choice — report the sensitivity, "
                             "never one number")
    parser.add_argument("--max-cross-track-m", type=float, default=criteria.max_cross_track_m,
                        help="established: max median |cross-track| of the fitted segment")
    parser.add_argument("--glidepath-range-deg", type=float, nargs=2,
                        default=list(criteria.glidepath_range_deg), metavar=("LOW", "HIGH"),
                        help="established: acceptable fitted glidepath range")
    parser.add_argument("--max-vertical-rms-m", type=float, default=criteria.max_vertical_rms_m,
                        help="established: max RMS residual of the vertical fit")


def thresholds_from_args(args: argparse.Namespace) -> DeviationThresholds:
    return DeviationThresholds(
        lateral_max_m=args.lateral_max_m,
        vertical_below_max_m=args.vertical_below_max_m,
        vertical_above_max_m=args.vertical_above_max_m,
    )


def criteria_from_args(args: argparse.Namespace) -> EstablishedCriteria:
    return EstablishedCriteria(
        window_m=tuple(args.fit_window_m),
        max_cross_track_m=args.max_cross_track_m,
        glidepath_range_deg=tuple(args.glidepath_range_deg),
        max_vertical_rms_m=args.max_vertical_rms_m,
    )
