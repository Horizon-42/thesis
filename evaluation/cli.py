"""Shared authoritative-context CLI wiring."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.context import ContextKey, contexts_for_airport
from evaluation.records import TrajectoryRecord
from evaluation.thresholds import AssessmentContext
from trajectory_data_process.harvest.airports import load_airport

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "trajectory_data_process/config/runway_thresholds.json"
DEFAULT_CIFP = REPO_ROOT / "data/CIFP/CIFP_260806/FAACIFP18"


def add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="FAA NASR-backed U.S. runway configuration")
    parser.add_argument("--cifp", type=Path, default=DEFAULT_CIFP,
                        help="current FAA CIFP file used for LPV FAS facts")


def contexts_from_args(
    records: list[TrajectoryRecord], args: argparse.Namespace
) -> dict[ContextKey, AssessmentContext]:
    airports: set[str] = set()
    for record in records:
        value = record.source.get("arr_airport") or record.source.get("airport")
        if not isinstance(value, str):
            raise ValueError("every record requires source.arr_airport")
        code = value.upper()
        if not code.startswith("K"):
            raise ValueError(
                f"{code}: this implementation is limited to U.S. FAA airport data"
            )
        airports.add(code)
    contexts: dict[ContextKey, AssessmentContext] = {}
    for code in sorted(airports):
        airport = load_airport(code, config_file=args.config, cifp_file=args.cifp)
        contexts.update(contexts_for_airport(airport))
    return contexts
