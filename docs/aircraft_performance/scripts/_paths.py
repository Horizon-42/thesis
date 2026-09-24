"""Shared paths for the substitution analysis (docs/aircraft_performance/scripts/).

CODE   this checkout: the aircraft / flight_scenarios / ts_transformer code that is imported
REPO   where the git-ignored data is read (harvest rosters, the FAA spreadsheet); default this
       checkout, set AERO_SUB_REPO to the main tree when running from a worktree without data
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
CODE = HERE.parents[2]
REPO = Path(os.environ.get("AERO_SUB_REPO", CODE))
WORK = Path(os.environ.get("AERO_SUB_WORK", "/tmp/aero_substitution"))
WORK.mkdir(parents=True, exist_ok=True)
ACD_XLSX = REPO / "data/reference_speeds/faa/FAA-Aircraft-Characteristics-Database-2024-10.xlsx"
HARVEST = REPO / "trajectory_data_process/outputs/harvest"


def use_repo_code():
    """Import the project's own aircraft / flight_scenarios / ts_transformer packages from CODE."""
    sys.path[:0] = [str(CODE / "4dTrajectory"), str(CODE)]
