"""Shared paths for the substitution analysis (docs/aircraft_performance/scripts/).

REPO   the checkout whose code and git-ignored data are read (default: this checkout;
       set AERO_SUB_REPO to the main tree when running from a docs-only worktree)
WORK   intermediate JSON/CSV (default /tmp/aero_substitution; AERO_SUB_WORK overrides)
LIT    the source pack, docs/literature/aircraft_performance/ next to this folder
OUT    docs/aircraft_performance/, where the final table is written
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
LIT = OUT.parent / "literature" / "aircraft_performance"
REPO = Path(os.environ.get("AERO_SUB_REPO", HERE.parents[2]))
WORK = Path(os.environ.get("AERO_SUB_WORK", "/tmp/aero_substitution"))
WORK.mkdir(parents=True, exist_ok=True)
ACD_XLSX = REPO / "data/reference_speeds/faa/FAA-Aircraft-Characteristics-Database-2024-10.xlsx"
HARVEST = REPO / "trajectory_data_process/outputs/harvest"


def use_repo_code():
    """Import the project's own aircraft / flight_scenarios / ts_transformer packages from REPO."""
    sys.path[:0] = [str(REPO / "4dTrajectory"), str(REPO)]
