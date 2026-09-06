"""Shared authoritative-context CLI wiring."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from evaluation.context import ContextKey, contexts_for_airport
from evaluation.records import load_record, record_files, roster_context_keys
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


def contexts_for_codes(
    codes: Iterable[str], args: argparse.Namespace
) -> dict[ContextKey, AssessmentContext]:
    """Assessment contexts for the given airport codes."""
    contexts: dict[ContextKey, AssessmentContext] = {}
    for code in sorted(set(codes)):
        if not code.startswith("K"):
            raise ValueError(
                f"{code}: this implementation is limited to U.S. FAA airport data"
            )
        airport = load_airport(code, config_file=args.config, cifp_file=args.cifp)
        contexts.update(contexts_for_airport(airport))
    return contexts


def contexts_for_input(
    path: str | Path, args: argparse.Namespace
) -> dict[ContextKey, AssessmentContext]:
    """The assessment contexts an input needs, without holding the input.

    A batch's ``summary.json`` carries each row's ``arr_airport``, so the airports it
    spans are read off the roster and the records themselves are streamed. A loose
    record file is read once for its airport; a directory without a manifest is
    rejected by :func:`record_files` with its own message.
    """
    keys = roster_context_keys(path)
    if keys is not None:
        return contexts_for_codes((airport for airport, _runway in keys), args)
    [file] = record_files(path)
    return contexts_for_codes([load_record(file).airport], args)
