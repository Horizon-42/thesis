"""Where this repository keeps things, resolved from the package's own location.

The ONE definition the CLI, the benchmark and the experiment runners read (2026-09-10,
review §4.5): until then `run_ts_pipeline`, `batch_benchmark`, `cli/common` and every
runner each computed the repository root from their own `__file__`.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
#: The package CLI, for the runners that train or predict in a subprocess.
TS_SCRIPT = TS_DIR / "__main__.py"
#: The runners' CLI (`python run_ts.py <runner>`), for a runner that spawns another runner — NOT
#: `TS_SCRIPT`, which knows only train / predict / … and refuses a runner's name.
RUN_TS = REPO_ROOT / "run_ts.py"
HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
OPT_OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"
COMPARISON_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
CZML_SCRIPT = REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"


def arrival_manifest_path(airport: str, harvest_root: Path = HARVEST_ROOT) -> Path:
    """The harvest's arrivals manifest for ``airport`` under ``harvest_root``."""
    return harvest_root / airport.upper() / "arrivals" / "manifest.json"


def discover_k_airports(harvest_root: Path = HARVEST_ROOT) -> list[str]:
    """The K-airports with a harvested arrivals manifest under ``harvest_root``, sorted."""
    if not harvest_root.exists():
        return []
    return sorted(
        child.name.upper()
        for child in harvest_root.iterdir()
        if child.is_dir()
        and child.name.upper().startswith("K")
        and arrival_manifest_path(child.name, harvest_root).exists()
    )
