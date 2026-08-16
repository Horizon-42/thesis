"""Select a bounded observed-flight roster and return playback-ready metadata + CZML.

The browser must never parse the airport-wide ``trajectories.czml`` before it
can apply its sample count: KRDU's canonical artifact is larger than V8's
single-string limit. This backend reads the small harvest manifest first,
filters the complete eligible roster by runway and terminal verdict, samples
exactly once, and opens only the selected per-flight source records.

The observed evaluation report is large too. It is parsed and cached server-side;
the response carries only the selected flights' verdicts and the small aggregate
needed by the baseline UI. The browser downloads the full report only on an
explicit Details request.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from aeroviz_backend import paths
from generate_czml import build_czml
from trajectory_data_process.harvest.czml import observed_czml_flights
from trajectory_data_process.harvest.store import HarvestPaths, read_manifest


DEFAULT_HARVEST_ROOT = (
    paths.REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
)
DEFAULT_EVALUATION_ROOT = (
    paths.REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
)
DEFAULT_LIMIT = 200
MAX_TRAJECTORIES_PER_RESPONSE = 1000
RESPONSE_SCHEMA_VERSION = "observed-trajectories-v1"
_START_DT = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)
_AIRPORT_RE = re.compile(r"^[A-Z0-9]{3,4}$")
_RUNWAY_RE = re.compile(r"^[0-9]{2}[LRC]?$")
_VERDICTS = {"pass", "fail", "undecided"}


@dataclass(frozen=True)
class _ObservedEvaluation:
    by_flight_key: dict[str, str]
    summary: dict[str, Any] | None


class ObservedTrajectoryBackend:
    def __init__(
        self,
        *,
        harvest_root: Path = DEFAULT_HARVEST_ROOT,
        evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
        max_trajectories: int = MAX_TRAJECTORIES_PER_RESPONSE,
    ) -> None:
        self.harvest_root = Path(harvest_root)
        self.evaluation_root = Path(evaluation_root)
        self.max_trajectories = int(max_trajectories)
        self._evaluation_cache: dict[
            str, tuple[int, int, _ObservedEvaluation | None]
        ] = {}
        if self.max_trajectories < 1:
            raise ValueError("max_trajectories must be positive")

    def query(
        self,
        airport: str,
        *,
        runway: str | None = None,
        verdict: str | None = None,
        limit: int = DEFAULT_LIMIT,
        seed: int = 0,
        flight_keys: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        code = _normalize_airport(airport)
        runway_ident = _normalize_runway(runway)
        verdict_filter = _normalize_verdict(verdict)
        requested_limit = int(limit)
        if requested_limit < 0 or requested_limit > self.max_trajectories:
            raise ValueError(
                f"limit must be between 0 and {self.max_trajectories}"
            )
        requested_keys = _normalize_flight_keys(
            flight_keys,
            max_trajectories=self.max_trajectories,
        )

        harvest_paths = HarvestPaths(self.harvest_root, code)
        manifest = read_manifest(harvest_paths)
        eligible = [
            row
            for row in manifest["records"]
            if row["outcome"] == "assigned"
            and (runway_ident is None or str(row["runway"]).upper() == runway_ident)
        ]

        evaluation = self._evaluation(code)
        counts = {"pass": 0, "fail": 0, "undecided": 0}
        matched = 0
        if evaluation is not None:
            for row in eligible:
                key = str(row["flight_key"])
                published = evaluation.by_flight_key.get(key)
                if published is not None:
                    matched += 1
                counts[published or "undecided"] += 1
            if verdict_filter is not None:
                eligible = [
                    row
                    for row in eligible
                    if evaluation.by_flight_key.get(str(row["flight_key"]), "undecided")
                    == verdict_filter
                ]

        if requested_keys is not None:
            eligible_by_key = {
                str(row["flight_key"]): row
                for row in eligible
            }
            missing = [key for key in requested_keys if key not in eligible_by_key]
            if missing:
                preview = ", ".join(missing[:3])
                suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
                raise ValueError(
                    f"{len(missing)} requested flight_key value(s) were not found in "
                    f"{code} {runway_ident or 'eligible arrivals'}: {preview}{suffix}"
                )
            selected = [eligible_by_key[key] for key in requested_keys]
        elif requested_limit == 0:
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
        czml = build_czml(flights, _START_DT)
        verdicts = None
        if evaluation is not None:
            verdicts = {
                "counts": counts,
                "byFlightId": {
                    str(row["flight_key"]): evaluation.by_flight_key.get(
                        str(row["flight_key"]), "undecided"
                    )
                    for row in selected
                },
                "matched": matched,
                "total": sum(counts.values()),
            }
        return {
            "schemaVersion": RESPONSE_SCHEMA_VERSION,
            "czml": czml,
            "verdicts": verdicts,
            "evaluation": evaluation.summary if evaluation is not None else None,
        }

    def _evaluation(self, airport: str) -> _ObservedEvaluation | None:
        path = (
            self.evaluation_root
            / airport
            / "comparison"
            / "observed"
            / "evaluation_report.json"
        )
        try:
            stat = path.stat()
        except OSError:
            return None
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._evaluation_cache.get(airport)
        if cached is not None and cached[:2] == signature:
            return cached[2]

        evaluation: _ObservedEvaluation | None = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                rows = payload.get("trajectories")
            else:
                rows = None
            if (
                isinstance(payload, dict)
                and payload.get("subject") == "observed"
                and isinstance(rows, list)
            ):
                by_flight_key: dict[str, str] = {}
                for row in rows:
                    if not isinstance(row, dict) or not isinstance(
                        row.get("flight_key"), str
                    ):
                        continue
                    raw_verdict = row.get("verdict")
                    if raw_verdict == "indeterminate":
                        by_flight_key[row["flight_key"]] = "undecided"
                    elif raw_verdict in {"pass", "fail"}:
                        by_flight_key[row["flight_key"]] = raw_verdict
                evaluation = _ObservedEvaluation(
                    by_flight_key=by_flight_key,
                    summary=_evaluation_summary(payload),
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            evaluation = None

        self._evaluation_cache[airport] = (*signature, evaluation)
        return evaluation


def _normalize_airport(value: str) -> str:
    code = str(value).strip().upper()
    if not _AIRPORT_RE.fullmatch(code):
        raise ValueError("airport must be a 3-4 character ICAO-style code")
    return code


def _evaluation_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    total = payload.get("total")
    counts = payload.get("verdict_counts")
    observed = payload.get("observed")
    lateral = payload.get("lateral_m")
    vertical = payload.get("vertical_m")
    if not _non_negative_int(total) or not isinstance(counts, dict):
        return None
    if not all(
        _non_negative_int(counts.get(key))
        for key in ("pass", "fail", "indeterminate")
    ):
        return None
    if observed is not None and (
        not isinstance(observed, dict)
        or not _finite_number(observed.get("event_estimated_rate"))
    ):
        return None
    if lateral is not None and (
        not isinstance(lateral, dict) or not _finite_number(lateral.get("mean"))
    ):
        return None
    if vertical is not None and (
        not isinstance(vertical, dict) or not _finite_number(vertical.get("mean_abs"))
    ):
        return None
    return {
        "total": total,
        "verdict_counts": counts,
        "observed": observed,
        "lateral_m": lateral,
        "vertical_m": vertical,
    }


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _normalize_runway(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    ident = str(value).strip().upper()
    if not _RUNWAY_RE.fullmatch(ident):
        raise ValueError("runway must look like 05, 05L, 23R, or 32")
    return ident


def _normalize_verdict(value: str | None) -> str | None:
    if value is None or not str(value).strip() or str(value).strip().lower() == "all":
        return None
    verdict = str(value).strip().lower()
    if verdict not in _VERDICTS:
        raise ValueError("verdict must be all, pass, fail, or undecided")
    return verdict


def _normalize_flight_keys(
    values: Sequence[str] | None,
    *,
    max_trajectories: int,
) -> list[str] | None:
    if values is None:
        return None
    keys = [str(value).strip() for value in values]
    if not keys or any(not key for key in keys):
        raise ValueError("flight_key values must be non-empty")
    if len(keys) > max_trajectories:
        raise ValueError(
            f"at most {max_trajectories} flight_key values may be requested"
        )
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate flight_key values are not allowed")
    return keys


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
