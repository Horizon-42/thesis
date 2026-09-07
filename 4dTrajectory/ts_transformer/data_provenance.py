"""The fingerprint of the arrival data a run was trained on.

A checkpoint that cannot say which rosters produced it cannot be trusted to predict on
anything; ``arrival_data_provenance`` is the digest that answers it, and
``require_matching_data_provenance`` is what refuses a stale one. Pure hashing over JSON —
no torch, no numpy, no trajectory values — so `evaluation_protocol` can compare two
fingerprints without importing the data plane.

**The identity of an eligibility roster is its eligible SET, never the roster file's
bytes.** The roster embeds UPSTREAM provenance (which observed evaluation report it was
joined against), so regenerating that report moves the file's bytes while the eligible set
stays identical — and a byte-bound identity then refuses every checkpoint trained before
the regeneration on data that did not change. That is exactly what happened on 2026-09-07
(observed reports v6 -> v9, five airports, eligible sets byte-for-byte identical), which
blocked predict, evaluate-fit and every replay runner. Hence ``eligible_set_sha256``: the
compared identity is a digest of the eligible flight keys. The roster's byte facts stay
auditable through ``eligibility_sources`` — in ``data_selection.pre_split_eligibility`` and
nowhere that is compared for equality.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from trajectory_data_process.harvest.arrivals import resolve_arrival_manifest


ARRIVAL_DATA_PROVENANCE_SCHEMA = "ts-arrival-data-v4-eligible-set"
#: The retired byte-bound schema. Checkpoints carrying it stay usable exactly: their
#: eligible set is re-verified against today's rosters through the checkpoint's OWN split
#: identity digests (`_require_unchanged_eligible_sets`), never against stored roster bytes.
LEGACY_ELIGIBILITY_BOUND_SCHEMA = "ts-arrival-data-v3-eligibility-bound"
READABLE_ARRIVAL_DATA_PROVENANCE_SCHEMAS = (
    ARRIVAL_DATA_PROVENANCE_SCHEMA,
    LEGACY_ELIGIBILITY_BOUND_SCHEMA,
)


def eligible_set_digest(keys: Iterable[str]) -> str:
    """The content identity of a set of flight identities: sorted, newline-joined, sha256.

    THE one definition. `splits.data_selection_audit` hashes its split rosters with it, the
    provenance's ``eligible_set_sha256`` is it, and the legacy re-verification recomputes
    the checkpoint's stored split digests with it — a second implementation anywhere would
    make a v3 checkpoint unverifiable the day the two drifted.
    """
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


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


def _roster_document(raw_roster: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    """One eligibility roster's path, bytes and parsed document."""
    roster_path = Path(raw_roster).resolve()
    roster_bytes = roster_path.read_bytes()
    roster = json.loads(roster_bytes)
    if not isinstance(roster, dict):
        raise ValueError(f"{roster_path} is not an eligibility roster object")
    if not str(roster.get("airport") or "").strip():
        raise ValueError(f"{roster_path} does not declare an airport")
    return roster_path, roster_bytes, roster


def _roster_eligible_keys(roster_path: Path, roster: dict[str, Any]) -> list[str]:
    """Validate one lateral-pass roster's policy and identities; return its eligible keys."""
    from lateral_eligibility import (  # local import keeps the loader policy-agnostic
        LATERAL_PASS_POLICY,
        LATERAL_PASS_ROSTER_SCHEMA,
    )
    if roster.get("schema_version") != LATERAL_PASS_ROSTER_SCHEMA:
        raise ValueError(f"{roster_path} has the wrong eligibility schema")
    if roster.get("policy") != LATERAL_PASS_POLICY:
        raise ValueError(f"{roster_path} has the wrong eligibility policy")
    keys = roster.get("eligible_flight_keys")
    if (
        not isinstance(keys, list)
        or any(not isinstance(key, str) or not key for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError(f"{roster_path} has invalid eligible flight identities")
    return keys


def roster_eligible_set_digest(roster: str | Path) -> str:
    """The content identity of one eligibility roster FILE, read off its eligible set."""
    roster_path, _roster_bytes, document = _roster_document(roster)
    return eligible_set_digest(_roster_eligible_keys(roster_path, document))


def eligibility_sources(
    rosters: Sequence[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """The byte-level facts about each roster: WHERE it came from, never WHAT it selects.

    Auditable, and deliberately outside the compared identity: ``evaluation_report_sha256``
    moves whenever the observed evaluation is regenerated, with the eligible set unchanged.
    Recorded in ``data_selection.pre_split_eligibility`` so a run can still be traced back
    to the exact report it was joined against.
    """
    entries: list[dict[str, Any]] = []
    for raw_roster in rosters or ():
        roster_path, roster_bytes, roster = _roster_document(raw_roster)
        sources = roster.get("sources")
        if not isinstance(sources, dict):
            raise ValueError(f"{roster_path} has no sources block")
        entries.append({
            "airport": str(roster["airport"]).strip().upper(),
            "roster_path": str(roster_path),
            "roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
            "evaluation_report": sources.get("evaluation_report"),
            "evaluation_report_sha256": sources.get("evaluation_report_sha256"),
            "evaluation_report_schema": sources.get("evaluation_report_schema"),
            # The reject tallies live here, with the report they were read off — never in
            # the compared identity, which they do not describe.
            "counts": roster.get("counts"),
        })
    return entries


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
    roster_by_airport: dict[str, tuple[Path, dict[str, Any]]] = {}
    for raw_roster in eligibility_rosters or ():
        roster_path, _roster_bytes, roster = _roster_document(raw_roster)
        airport = str(roster["airport"]).strip().upper()
        if airport in roster_by_airport:
            raise ValueError(f"multiple eligibility rosters supplied for airport {airport}")
        roster_by_airport[airport] = (roster_path, roster)

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
            roster_path, roster = roster_by_airport[airport]
            sources = roster.get("sources")
            manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
            if (
                not isinstance(sources, dict)
                or sources.get("arrival_manifest_sha256") != manifest_digest
            ):
                raise ValueError(f"{roster_path} was built for a different arrival manifest")
            keys = _roster_eligible_keys(roster_path, roster)
            eligible_keys = set(keys)
            # No `counts`: three of the roster's five are REJECT tallies read off the
            # observed evaluation (`excluded_lateral_indeterminate`, `evaluation_only`),
            # so a re-graded flight that never was eligible would move them and refuse the
            # checkpoint — the same defect this schema exists to remove. The two that do
            # describe the eligible cohort are already compared as
            # `arrival_candidate_count` and `source_records`; all five stay auditable in
            # `eligibility_sources`.
            eligibility = {
                "schema_version": roster["schema_version"],
                "policy": roster["policy"],
                "eligible_set_sha256": eligible_set_digest(keys),
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


def provenance_eligible_set_digests(
    provenance: dict[str, Any],
) -> dict[str, str]:
    """Compact airport -> pre-split eligible-set digest, for a CURRENT fingerprint."""
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
        eligibility = entry.get("eligibility")
        digest = (
            eligibility.get("eligible_set_sha256")
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


def provenance_has_eligibility(provenance: dict[str, Any]) -> bool:
    """Did this run apply a pre-split eligibility roster? True for either stored schema."""
    if provenance.get("schema_version") not in READABLE_ARRIVAL_DATA_PROVENANCE_SCHEMAS:
        raise ValueError("data_provenance is not a multi-airport TS fingerprint")
    manifests = provenance.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("data_provenance has no arrival manifests")
    return any(
        isinstance(entry, dict) and isinstance(entry.get("eligibility"), dict)
        for entry in manifests
    )


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
        if provenance_has_eligibility(payload["data_provenance"]) else None
    )
    return arrival_data_provenance(manifests, eligibility_rosters=rosters)


def _eligible_keys(entry: dict[str, Any]) -> set[str]:
    """The eligible flight keys a manifest entry stands for.

    ``source_records`` IS the eligible set: the roster's keys are validated to exist in the
    manifest, so the records that survive the roster filter are exactly the eligible ones.
    """
    return {record["flight_key"] for record in entry.get("source_records", [])}


def _require_comparable_eligibility(
    stored: dict[str, Any], current: dict[str, Any]
) -> None:
    """Refuse a fingerprint taken WITHOUT the roster the checkpoint recorded.

    Otherwise the comparison fails for the right reason with the wrong message: the current
    entry then lists every arrival candidate (14 435 KRDU) against the checkpoint's eligible
    14 378 and it reads as "the manifest changed" (`code-health-followups.md` §19).
    """
    bound = {
        entry.get("airport")
        for entry in stored.get("manifests", [])
        if isinstance(entry, dict) and isinstance(entry.get("eligibility"), dict)
    }
    rosterless = sorted(
        entry["airport"]
        for entry in current.get("manifests", [])
        if isinstance(entry, dict)
        and entry.get("eligibility") is None
        and entry.get("airport") in bound
    )
    if rosterless:
        raise ValueError(
            f"the current arrival fingerprint for {', '.join(rosterless)} was taken WITHOUT "
            "the pre-split eligibility roster this checkpoint recorded; build it with "
            "data_provenance.checkpoint_data_provenance(payload, manifests)"
        )


def _require_unchanged_eligible_sets(
    stored: dict[str, Any], current: dict[str, Any]
) -> None:
    """Verify a v3 checkpoint's eligible SET against today's rosters, per airport.

    A v3 provenance bound the roster's BYTES, which move for reasons that are not the
    checkpoint's data (module docstring). What it ALSO carries is the eligible set itself:
    ``source_records`` IS that set. So the identity is re-derivable from the checkpoint
    alone, and this compares it. Airports the caller did not supply are not verified —
    `allow_subset` checks the supplied ones, nothing else.

    This runs BEFORE the full comparison, which would catch the same difference through
    `source_records` and report it as "the manifest changed". Its job is the accurate
    message: which airport, and the two set digests. It deliberately reads NOTHING else
    from the payload — an earlier version rebuilt the checkpoint's `TSConfig` to recompute
    the `data_selection` split digests, which re-imposed the model-recipe contract on a
    reader that needs none of it and raised `TypeError` on three real pooled checkpoints
    whose stored configs this build no longer accepts.
    """
    current_entries = {
        entry["airport"]: entry
        for entry in current.get("manifests", [])
        if isinstance(entry, dict) and isinstance(entry.get("airport"), str)
    }
    for entry in stored["manifests"]:
        airport = entry.get("airport")
        if airport not in current_entries:
            continue
        was, now = _eligible_keys(entry), _eligible_keys(current_entries[airport])
        if was != now:
            raise ValueError(
                f"the eligible flight set for {airport} changed since this checkpoint was "
                f"trained: {len(was)} flights ({eligible_set_digest(was)}) against today's "
                f"{len(now)} ({eligible_set_digest(now)}), "
                f"{len(was - now)} dropped and {len(now - was)} added — retrain instead of "
                "reusing this checkpoint"
            )


def _stored_in_current_form(
    checkpoint_payload: dict[str, Any],
    stored: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """The checkpoint's stored fingerprint, expressed the way today's is built.

    A v3 fingerprint differs from a v4 one only in HOW it names the eligibility roster: by
    the file's bytes rather than by the set the file selects. Once the set is verified, the
    v3 entry says the same thing as today's, so it is rewritten into the current form and
    the rest of the comparison (manifest digest, candidate count, source records, policy)
    runs unchanged. Nothing is written back to the checkpoint — the freeze-test ledger
    hashes the STORED object and must keep matching.
    """
    schema = stored.get("schema_version")
    if schema == ARRIVAL_DATA_PROVENANCE_SCHEMA:
        return stored
    if schema != LEGACY_ELIGIBILITY_BOUND_SCHEMA:
        raise ValueError(
            f"checkpoint arrival-data provenance schema {schema!r} is not readable by this "
            f"build (expected {ARRIVAL_DATA_PROVENANCE_SCHEMA!r} or the legacy "
            f"{LEGACY_ELIGIBILITY_BOUND_SCHEMA!r}); retrain it"
        )
    manifests = stored.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("checkpoint data_provenance has no arrival manifests")
    if provenance_has_eligibility(stored):
        _require_unchanged_eligible_sets(stored, current)
    upgraded: list[dict[str, Any]] = []
    for entry in manifests:
        eligibility = entry.get("eligibility") if isinstance(entry, dict) else None
        if not isinstance(eligibility, dict):
            upgraded.append(entry)
            continue
        upgraded.append({
            **entry,
            "eligibility": {
                "schema_version": eligibility.get("schema_version"),
                "policy": eligibility.get("policy"),
                "eligible_set_sha256": eligible_set_digest(_eligible_keys(entry)),
            },
        })
    return {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA, "manifests": upgraded}


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
    _require_comparable_eligibility(stored, current)
    stored = _stored_in_current_form(checkpoint_payload, stored, current)
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
