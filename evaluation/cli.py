"""Shared authoritative-context CLI wiring."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from evaluation.context import ContextKey, contexts_for_airport
from evaluation.records import load_record, record_files, roster_context_keys
from evaluation.thresholds import AssessmentContext
from evaluation.wind import WindTable, load_wind_tables
from trajectory_data_process.harvest.airports import load_airport

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "trajectory_data_process/config/runway_thresholds.json"
DEFAULT_CIFP = REPO_ROOT / "data/CIFP/CIFP_260806/FAACIFP18"
# ``trajectory_data_process/metar/fetch_iem_asos.py`` writes <ICAO>/*.csv here. An absent
# directory means no wind correction: observed rows are judged on the ground-speed
# proxy and the report counts them (wind_counts.unavailable).
DEFAULT_METAR_ROOT = REPO_ROOT / "data/metar"


def add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="FAA NASR-backed U.S. runway configuration")
    parser.add_argument("--cifp", type=Path, default=DEFAULT_CIFP,
                        help="current FAA CIFP file used for LPV FAS facts")
    parser.add_argument("--metar-root", type=Path, default=DEFAULT_METAR_ROOT,
                        help="<ICAO>/*.csv ASOS wind archives (fetch_iem_asos); observed "
                             "crossing speeds are corrected by the headwind when present")


def winds_for_input(
    contexts: dict[ContextKey, AssessmentContext], args: argparse.Namespace
) -> dict[str, WindTable]:
    """The METAR tables for the airports an input spans (those that have one)."""
    wanted = {airport for airport, _runway in contexts}
    return {
        airport: table
        for airport, table in load_wind_tables(args.metar_root).items()
        if airport in wanted
    }


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
