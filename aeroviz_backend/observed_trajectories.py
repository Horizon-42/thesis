"""Select a bounded observed-flight roster and return playback-ready CZML.

The browser must never parse the airport-wide ``trajectories.czml`` before it
can apply its sample count: KRDU's canonical artifact is larger than V8's
single-string limit. This backend reads the small harvest manifest first,
samples manifest rows, and opens only the selected per-flight source records.
"""

from __future__ import annotations

import hashlib
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aeroviz_backend import paths
from generate_czml import build_czml
from trajectory_data_process.harvest.czml import observed_czml_flights
from trajectory_data_process.harvest.store import HarvestPaths, read_manifest


DEFAULT_HARVEST_ROOT = paths.REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
DEFAULT_LIMIT = 200
MAX_TRAJECTORIES_PER_RESPONSE = 1000
_START_DT = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)
_AIRPORT_RE = re.compile(r"^[A-Z0-9]{3,4}$")
_RUNWAY_RE = re.compile(r"^[0-9]{2}[LRC]?$")


class ObservedTrajectoryBackend:
    def __init__(
        self,
        *,
        harvest_root: Path = DEFAULT_HARVEST_ROOT,
        max_trajectories: int = MAX_TRAJECTORIES_PER_RESPONSE,
    ) -> None:
        self.harvest_root = Path(harvest_root)
        self.max_trajectories = int(max_trajectories)
        if self.max_trajectories < 1:
            raise ValueError("max_trajectories must be positive")

    def query(
        self,
        airport: str,
        *,
        runway: str | None = None,
        limit: int = DEFAULT_LIMIT,
        seed: int = 0,
    ) -> list[dict[str, Any]]:
        code = _normalize_airport(airport)
        runway_ident = _normalize_runway(runway)
        requested_limit = int(limit)
        if requested_limit < 0 or requested_limit > self.max_trajectories:
            raise ValueError(
                f"limit must be between 0 and {self.max_trajectories}"
            )

        harvest_paths = HarvestPaths(self.harvest_root, code)
        manifest = read_manifest(harvest_paths)
        eligible = [
            row
            for row in manifest["records"]
            if row["outcome"] == "assigned"
            and (runway_ident is None or str(row["runway"]).upper() == runway_ident)
        ]

        if requested_limit == 0:
            if len(eligible) > self.max_trajectories:
                raise ValueError(
                    f"{code} {runway_ident or 'all runways'} has {len(eligible)} "
                    "trajectories, which exceeds the safe response maximum of "
                    f"{self.max_trajectories}; request a positive limit"
                )
            selected = eligible
        elif requested_limit < len(eligible):
            selected = _stable_sample(
                eligible,
                requested_limit,
                airport=code,
                runway=runway_ident,
                seed=int(seed),
            )
        else:
            selected = eligible

        flights = list(observed_czml_flights(harvest_paths, selected))
        return build_czml(flights, _START_DT)


def _normalize_airport(value: str) -> str:
    code = str(value).strip().upper()
    if not _AIRPORT_RE.fullmatch(code):
        raise ValueError("airport must be a 3-4 character ICAO-style code")
    return code


def _normalize_runway(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    ident = str(value).strip().upper()
    if not _RUNWAY_RE.fullmatch(ident):
        raise ValueError("runway must look like 05, 05L, 23R, or 32")
    return ident


def _stable_sample(
    rows: list[dict[str, Any]],
    count: int,
    *,
    airport: str,
    runway: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    material = f"{airport}|{runway or '*'}|{seed}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(stable_seed).sample(rows, count)
