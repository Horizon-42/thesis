"""Select a bounded observed-flight roster and return playback-ready metadata + CZML.

The browser must never parse the airport-wide ``trajectories.czml`` before it
can apply its sample count: KRDU's canonical artifact is larger than V8's
single-string limit. This backend reads the small harvest manifest first,
filters the complete eligible roster by runway and terminal verdict, samples
exactly once, and opens only the selected per-flight source records.

Two windows of the same measurement are served, chosen by ``window``:

``full``
    The complete reconstructed track, from the first received state vector.
    This is what the Observe/Baseline layer shows and what ``trajectories.czml``
    contains.
``arrival``
    The model arrival slice only — terminal-ring entry to the landing anchor —
    with time rebased so ``t = 0`` is the entry. This is the window every
    modeling artifact lives in, so it is the one the comparison overlay's
    reference must use: an optimizer/prediction group's ``t = 0`` is the entry,
    while a full track's ``t = 0`` is the first reception, a median 45 s and
    5 km earlier. Drawn against each other on one clock, the group renders that
    far ahead of the reference it is supposed to be compared with.

The slice is taken at READ time. ``tracks/`` is the canonical measurement store
and is never edited (the same rule the altitude-outlier repair follows), and no
derived artifact is written: the comparison reference is served live from here.

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
from flight_scenarios.identity import flight_key
from generate_czml import build_czml
from trajectory_data_process.harvest.arrivals import (
    arrival_manifest_path,
    load_arrival_flights,
)
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
# v2 carries ``trackWindow``. The bump is load-bearing rather than cosmetic: a v1
# backend ignores an unknown ``window`` argument and answers a comparison-reference
# request with full tracks, which is exactly the misalignment this contract exists
# to prevent — and it would look like a rendering quirk, not a version skew.
RESPONSE_SCHEMA_VERSION = "observed-trajectories-v2"
TRACK_WINDOW_FULL = "full"
TRACK_WINDOW_ARRIVAL = "arrival"
_TRACK_WINDOWS = (TRACK_WINDOW_FULL, TRACK_WINDOW_ARRIVAL)
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
        window: str | None = None,
    ) -> dict[str, Any]:
        code = _normalize_airport(airport)
        runway_ident = _normalize_runway(runway)
        verdict_filter = _normalize_verdict(verdict)
        track_window = _normalize_window(window)
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
        eligible = _roster(harvest_paths, track_window, runway_ident)

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

        flights = _window_flights(harvest_paths, track_window, selected)
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
            "trackWindow": track_window,
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


def _roster(
    paths: HarvestPaths,
    window: str,
    runway_ident: str | None,
) -> list[dict[str, Any]]:
    """The flights a window can serve, read from the manifest that DEFINES that window.

    ``full`` is rostered by ``tracks/manifest.json`` (every assigned landing) and
    ``arrival`` by ``arrivals/manifest.json`` (the model-ready subset: assigned, with a
    published TCH and glidepath, cropped to the final terminal entry). Taking the arrival
    window's roster from the tracks manifest would offer flights that have no arrival
    slice at all, and the loader would then raise on a request the caller had every
    reason to believe was valid.
    """
    if window == TRACK_WINDOW_ARRIVAL:
        manifest = json.loads(
            arrival_manifest_path(paths).read_text(encoding="utf-8")
        )
        rows = list(manifest["records"])
    else:
        rows = [
            row
            for row in read_manifest(paths)["records"]
            if row["outcome"] == "assigned"
        ]
    if runway_ident is None:
        return rows
    return [row for row in rows if str(row["runway"]).upper() == runway_ident]


def _window_flights(
    paths: HarvestPaths,
    window: str,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Renderer-ready flight dicts for ``selected``, in request order.

    The arrival window goes through ``load_arrival_flights`` — the SAME loader the
    scenario, optimizer and training paths use — rather than re-slicing here. That is the
    whole point: the reference cannot drift from the data the model was fed, because there
    is no second implementation of the slice to drift from. It also inherits the loader's
    source-hash check and identity round trip for free.

    No fitted threshold extension is attached in the arrival window: the reference is the
    measured slice, matching what the model consumed and what the evaluation graded.
    """
    if window != TRACK_WINDOW_ARRIVAL:
        return list(observed_czml_flights(paths, selected))
    keys = [str(row["flight_key"]) for row in selected]
    flights = load_arrival_flights(
        arrival_manifest_path(paths), include_flight_keys=set(keys)
    )
    by_key = {flight_key(flight, 0): flight for flight in flights}
    return [by_key[key] for key in keys]


def _normalize_window(value: str | None) -> str:
    if value is None or not str(value).strip():
        return TRACK_WINDOW_FULL
    window = str(value).strip().lower()
    if window not in _TRACK_WINDOWS:
        raise ValueError(f"window must be one of {', '.join(_TRACK_WINDOWS)}")
    return window


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
