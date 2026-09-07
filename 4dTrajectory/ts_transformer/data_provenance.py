"""The fingerprint of the arrival data a run was trained on.

A checkpoint that cannot say which rosters produced it cannot be trusted to predict on
anything; ``arrival_data_provenance`` is the digest that answers it, and
``require_matching_data_provenance`` is what refuses a stale one. Pure hashing over JSON —
no torch, no numpy, no trajectory values — so `evaluation_protocol` can compare two
fingerprints without importing the data plane.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from trajectory_data_process.harvest.arrivals import resolve_arrival_manifest


ARRIVAL_DATA_PROVENANCE_SCHEMA = "ts-arrival-data-v3-eligibility-bound"


def manifest_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve one or more manifests, rejecting duplicate airport inputs."""
    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not raw_paths:
        raise ValueError("at least one arrival manifest is required")

    resolved: list[tuple[str, Path]] = []
    seen_airports: set[str] = set()
    for raw_path in raw_paths:
        manifest_path = resolve_arrival_manifest(raw_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(
                f"{manifest_path} is not an arrival manifest object; legacy flight-array "
                "inputs are no longer supported"
            )
        airport = str(manifest.get("airport") or "").strip().upper()
        if not airport:
            raise ValueError(f"{manifest_path} does not declare an airport")
        if airport in seen_airports:
            raise ValueError(f"multiple arrival manifests supplied for airport {airport}")
        seen_airports.add(airport)
        resolved.append((airport, manifest_path))
    return [path for _airport, path in sorted(resolved)]


def arrival_data_provenance(
    paths: str | Path | Sequence[str | Path],
    *,
    eligibility_rosters: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Fingerprint the exact canonical arrival rosters used by a training run.

    The manifest digest catches any roster, slice, target, or metadata change.  Keeping
    each flight's canonical source digest as well makes the checkpoint independently
    auditable without reopening every source track.
    """
    roster_by_airport: dict[str, tuple[Path, bytes, dict[str, Any]]] = {}
    for raw_roster in eligibility_rosters or ():
        roster_path = Path(raw_roster).resolve()
        roster_bytes = roster_path.read_bytes()
        roster = json.loads(roster_bytes)
        airport = str(roster.get("airport") or "").strip().upper()
        if not airport:
            raise ValueError(f"{roster_path} does not declare an airport")
        if airport in roster_by_airport:
            raise ValueError(f"multiple eligibility rosters supplied for airport {airport}")
        roster_by_airport[airport] = (roster_path, roster_bytes, roster)

    manifest_entries: list[dict[str, Any]] = []
    manifest_airports: set[str] = set()
    for manifest_path in manifest_paths(paths):
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        records = manifest.get("records") if isinstance(manifest, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"{manifest_path} lacks an arrival records roster")
        airport = str(manifest.get("airport") or "").strip().upper()
        manifest_airports.add(airport)

        eligible_keys: set[str] | None = None
        eligibility: dict[str, Any] | None = None
        if eligibility_rosters is not None:
            if airport not in roster_by_airport:
                raise ValueError(f"no eligibility roster supplied for airport {airport}")
            roster_path, roster_bytes, roster = roster_by_airport[airport]
            from lateral_eligibility import (  # local import keeps the loader policy-agnostic
                LATERAL_PASS_POLICY,
                LATERAL_PASS_ROSTER_SCHEMA,
            )
            if roster.get("schema_version") != LATERAL_PASS_ROSTER_SCHEMA:
                raise ValueError(f"{roster_path} has the wrong eligibility schema")
            sources = roster.get("sources")
            manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
            if (
                not isinstance(sources, dict)
                or sources.get("arrival_manifest_sha256") != manifest_digest
            ):
                raise ValueError(f"{roster_path} was built for a different arrival manifest")
            if roster.get("policy") != LATERAL_PASS_POLICY:
                raise ValueError(f"{roster_path} has the wrong eligibility policy")
            keys = roster.get("eligible_flight_keys")
            if (
                not isinstance(keys, list)
                or any(not isinstance(key, str) or not key for key in keys)
                or len(set(keys)) != len(keys)
            ):
                raise ValueError(f"{roster_path} has invalid eligible flight identities")
            eligible_keys = set(keys)
            eligibility = {
                "schema_version": roster["schema_version"],
                "policy": roster["policy"],
                "roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
                "evaluation_report_sha256": sources.get("evaluation_report_sha256"),
                "counts": roster.get("counts"),
            }

        source_records: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, row in enumerate(records):
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}: arrival record {index} is not an object")
            key = row.get("flight_key")
            source_sha256 = row.get("source_sha256")
            if not isinstance(key, str) or not key:
                raise ValueError(f"{manifest_path}: arrival record {index} lacks flight_key")
            if key in seen:
                raise ValueError(f"{manifest_path} lists duplicate flight_key {key!r}")
            if (
                not isinstance(source_sha256, str)
                or len(source_sha256) != 64
                or any(char not in "0123456789abcdef" for char in source_sha256.lower())
            ):
                raise ValueError(
                    f"{manifest_path}: arrival record {index} has invalid source_sha256"
                )
            seen.add(key)
            if eligible_keys is not None and key not in eligible_keys:
                continue
            source_records.append(
                {"flight_key": key, "source_sha256": source_sha256.lower()}
            )

        source_records.sort(key=lambda item: item["flight_key"])
        if eligible_keys is not None:
            missing_eligible = eligible_keys - seen
            if missing_eligible:
                raise ValueError(
                    f"{roster_path} names flight absent from arrival manifest: "
                    f"{min(missing_eligible)!r}"
                )
        manifest_entries.append({
            "airport": airport,
            "arrival_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "arrival_candidate_count": len(records),
            "eligibility": eligibility,
            "source_records": source_records,
        })

    unused_rosters = roster_by_airport.keys() - manifest_airports
    if unused_rosters:
        raise ValueError(
            f"eligibility roster supplied without arrival manifest for {min(unused_rosters)}"
        )

    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": manifest_entries,
    }


def provenance_manifest_digests(provenance: dict[str, Any]) -> dict[str, str]:
    """Compact airport -> manifest digest view used by import-light runners."""
    if provenance.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA:
        raise ValueError("data_provenance is not a multi-airport TS fingerprint")
    manifests = provenance.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("data_provenance has no arrival manifests")
    result: dict[str, str] = {}
    for entry in manifests:
        if not isinstance(entry, dict):
            raise ValueError("data_provenance manifest entry is not an object")
        airport = entry.get("airport")
        digest = entry.get("arrival_manifest_sha256")
        if not isinstance(airport, str) or not isinstance(digest, str):
            raise ValueError("data_provenance manifest entry lacks airport or digest")
        if airport in result:
            raise ValueError(f"data_provenance repeats airport {airport}")
        result[airport] = digest
    return result


def provenance_eligibility_digests(
    provenance: dict[str, Any],
) -> dict[str, str]:
    """Compact airport -> pre-split eligibility artifact digest."""
    if provenance.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA:
        raise ValueError("data_provenance is not a multi-airport TS fingerprint")
    result: dict[str, str] = {}
    for entry in provenance.get("manifests", []):
        if not isinstance(entry, dict):
            raise ValueError("data_provenance manifest entry is not an object")
        airport = entry.get("airport")
        eligibility = entry.get("eligibility")
        digest = (
            eligibility.get("roster_sha256")
            if isinstance(eligibility, dict)
            else None
        )
        if not isinstance(airport, str) or (
            digest is not None and not isinstance(digest, str)
        ):
            raise ValueError("data_provenance has invalid eligibility identity")
        if digest is not None:
            result[airport] = digest
    return result


def checkpoint_data_provenance(
    payload: dict[str, Any], manifests: Sequence[str | Path]
) -> dict[str, Any]:
    """Today's fingerprint of a checkpoint's own data, built the way it was trained.

    The pre-split lateral-pass roster is PART of the data identity — the whole v5 cohort
    is eligibility-bound — and a fingerprint taken without it lists 14 435 KRDU arrivals
    where the checkpoint carries 14 378, which reads as "the manifest changed". Whether to
    read the roster is the checkpoint's own answer, not a caller's flag: a checkpoint
    trained before the sidecar existed must not be handed one either. Every runner that
    replays a checkpoint fingerprints through here; the L5.a fitter did not, and died at
    startup on every v5 checkpoint (2026-09-07).
    """
    from lateral_eligibility import default_lateral_pass_roster_path  # policy module; kept off this module's import graph

    rosters = (
        [default_lateral_pass_roster_path(path) for path in manifests]
        if provenance_eligibility_digests(payload["data_provenance"]) else None
    )
    return arrival_data_provenance(manifests, eligibility_rosters=rosters)


def require_matching_data_provenance(
    checkpoint_payload: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_subset: bool = False,
) -> None:
    """Reject stale data; prediction may verify an exact airport subset of training data."""
    stored = checkpoint_payload.get("data_provenance")
    if not isinstance(stored, dict):
        raise ValueError(
            "checkpoint has no arrival-data provenance; retrain it against the current "
            "arrivals/manifest.json"
        )
    if not allow_subset and stored != current:
        raise ValueError(
            "checkpoint training data does not match the current arrival manifests; "
            "retrain instead of reusing this checkpoint"
        )
    if allow_subset:
        stored_entries = {
            entry["airport"]: entry for entry in stored.get("manifests", [])
            if isinstance(entry, dict) and isinstance(entry.get("airport"), str)
        }
        current_entries = current.get("manifests")
        if (
            current.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA
            or not isinstance(current_entries, list)
            or not current_entries
            or any(
                not isinstance(entry, dict)
                or stored_entries.get(entry.get("airport")) != entry
                for entry in current_entries
            )
        ):
            raise ValueError(
                "prediction data is not an exact airport subset of the checkpoint training "
                "data; retrain or use the matching manifests"
            )
