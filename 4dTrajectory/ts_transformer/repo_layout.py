"""Where this repository keeps things, resolved from the package's own location.

The ONE definition the CLI, the benchmark and the experiment runners read (2026-09-10,
review §4.5): until then `run_ts_pipeline`, `batch_benchmark`, `cli/common` and every
runner each computed the repository root from their own `__file__`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from trajectory_data_process.harvest.generations import (
    FROZEN_GENERATIONS,
    frozen_generation_roots,
)

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


def checkpoint_arrival_manifests(
    payload: dict[str, Any], harvest_root: Path = HARVEST_ROOT
) -> list[Path]:
    """The arrival manifests a checkpoint trained on, found by the digest it recorded.

    A checkpoint names its data by content (``data_provenance.manifests[*]
    .arrival_manifest_sha256``), so the manifest is looked up the same way: the live
    harvest's, else a frozen generation's (``trajectory_data_process.harvest.generations``)
    whose bytes hash to that digest. Nothing matching raises by airport and digest --
    a replay against any other bytes would be refused by the provenance check anyway,
    with a less useful message.
    """
    return [
        checkpoint_arrival_manifest(payload, entry["airport"], harvest_root)
        for entry in payload["data_provenance"]["manifests"]
    ]


def checkpoint_arrival_manifest(
    payload: dict[str, Any], airport: str, harvest_root: Path = HARVEST_ROOT
) -> Path:
    """One airport's manifest of :func:`checkpoint_arrival_manifests`."""
    digests = [
        entry["arrival_manifest_sha256"]
        for entry in payload["data_provenance"]["manifests"]
        if entry["airport"] == airport.upper()
    ]
    if len(digests) != 1:
        raise ValueError(
            f"this checkpoint's data_provenance names {len(digests)} {airport.upper()} "
            "arrival manifests; it can only be replayed on an airport it trained on"
        )
    [digest] = digests
    match = arrival_manifest_with_digest(airport, digest, harvest_root)
    if match is None:
        raise ValueError(
            f"no harvest generation holds the {airport} arrival manifest this checkpoint "
            f"trained on (sha256 {digest}); searched {harvest_root} and the registered "
            f"frozen generations {list(FROZEN_GENERATIONS)}"
        )
    return match


def arrival_manifest_with_digest(
    airport: str, digest: str, harvest_root: Path = HARVEST_ROOT
) -> Path | None:
    """``airport``'s arrival manifest whose bytes hash to ``digest``: the live harvest's,
    else a frozen generation's; None when no generation holds those bytes."""
    for root in (harvest_root, *frozen_generation_roots()):
        path = arrival_manifest_path(airport, root)
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            return path
    return None


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
