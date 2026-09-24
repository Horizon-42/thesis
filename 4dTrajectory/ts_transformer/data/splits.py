"""Which split a flight belongs to, and the audit that records the answer.

**The split is BY FLIGHT, never by window.** Consecutive windows of one approach overlap
by ``seq_len - 1`` samples, so splitting windows at random puts near-duplicates of a
validation window in the training set and the val loss becomes a memorisation score.

Each flight's split is a pure function of ``(config.resolved_split_seed,
airport:flight_id)``, hashed with ``hashlib`` rather than the salted builtin ``hash()``, so
a flight that is in the test set stays there across harvests and across processes.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Sequence

from aircraft.performance_index import performance_index_identity
from aircraft.query_aircraft_parameters import (
    openap_direct_typecodes,
    openap_source_label,
)
from ts_transformer.config import TSConfig
from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA, eligible_set_digest

if TYPE_CHECKING:   # annotations only: importing `dataset` here would drag torch in
    from ts_transformer.data.dataset import BuildReport, FlightSeries


DATA_SELECTION_SCHEMA = "ts-data-selection-v3-performance-index"
#: How a flight's split is decided, as recorded in every data-selection audit.
SPLIT_ASSIGNMENT_METHOD = "sha256(seed:airport-qualified-flight-id)"


def _split_fraction(flight_id: str, seed: int) -> float:
    """One flight's deterministic position in [0, 1), independent of every other flight.

    hashlib, not the builtin ``hash()`` — that one is salted per process (PYTHONHASHSEED),
    so it would deal every run a different split.
    """
    digest = hashlib.sha256(f"{seed}:{flight_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_name_for_dataset_id(dataset_id: str, config: TSConfig) -> str:
    """Return the locked outer split without reading any trajectory values."""
    fraction = _split_fraction(dataset_id, config.resolved_split_seed)
    if fraction < config.test_fraction:
        return "test"
    if fraction < config.test_fraction + config.val_fraction:
        return "val"
    return "train"


def flight_keys_by_split(
    data_provenance: dict[str, Any], config: TSConfig
) -> dict[str, list[str]]:
    """Resolve airport-qualified split identities from manifest metadata only.

    Arrival provenance already carries the authoritative roster and source digests. This
    helper deliberately does not open any source trajectory file; callers can pass the
    returned keys to :func:`load_flight_dicts` before loading model inputs.
    """
    if data_provenance.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA:
        raise ValueError("data_provenance is not a TS arrival-data fingerprint")
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for manifest in data_provenance.get("manifests", []):
        airport = str(manifest.get("airport") or "").strip().upper()
        for record in manifest.get("source_records", []):
            dataset_id = f"{airport}:{record['flight_key']}"
            result[split_name_for_dataset_id(dataset_id, config)].append(dataset_id)
    return result


def data_selection_audit(
    series: Sequence[FlightSeries],
    report: BuildReport,
    config: TSConfig,
    outer_split_keys: dict[str, list[str]],
) -> dict[str, Any]:
    """Describe the post-split fleet selection without opening outer-test tracks.

    ``outer_split_keys`` comes from manifest metadata only. ``series`` must contain only
    development rows loaded by the caller; the sealed test population is recorded by
    identity/hash and its aircraft eligibility is intentionally deferred until release.
    """
    selected: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for item in series:
        selected[split_name_for_dataset_id(item.dataset_id, config)].append(item.dataset_id)
    if selected["test"]:
        raise ValueError(
            "data-selection audit received outer-test trajectory series during development"
        )

    # THE set digest (`data_provenance.eligible_set_digest`), not a local copy of it: these
    # identities and the arrival provenance's ``eligible_set_sha256`` describe the same
    # cohort, and two implementations would be two answers to "is this the same data?".
    digest = eligible_set_digest
    direct_typecodes = openap_direct_typecodes()
    direct_digest = hashlib.sha256("\n".join(direct_typecodes).encode()).hexdigest()
    return {
        "schema_version": DATA_SELECTION_SCHEMA,
        "aircraft_filter": config.aircraft_filter,
        "identity_standard": "ICAO Doc 8643",
        "performance_provider": openap_source_label(),
        # Decides every type with no native model (`modelled` / `all-flights`); a changed index
        # changes which flights have dynamics.
        "performance_index": performance_index_identity(),
        "openap_direct_typecodes": {
            "count": len(direct_typecodes),
            "sha256": direct_digest,
            "values": list(direct_typecodes),
        },
        "split_policy": {
            "method": SPLIT_ASSIGNMENT_METHOD,
            "split_seed": config.resolved_split_seed,
            "eligibility_applied_before_split_assignment": True,
            "aircraft_filter_applied_after_split_assignment": True,
            "outer_test_tracks_loaded": False,
            "outer_test_aircraft_filter_status": "deferred_until_test_release",
        },
        "splits": {
            name: {
                "eligible_roster_flights": len(outer_split_keys[name]),
                "eligible_identity_sha256": digest(outer_split_keys[name]),
                "selected_flights": (
                    len(selected[name]) if name != "test" else None
                ),
                "selected_identity_sha256": (
                    digest(selected[name]) if name != "test" else None
                ),
            }
            for name in ("train", "val", "test")
        },
        "development_build": report.to_dict(),
    }


def split_by_flight(
    series: Sequence[FlightSeries], config: TSConfig
) -> tuple[list[FlightSeries], list[FlightSeries], list[FlightSeries]]:
    """Deterministic train / val / test split at FLIGHT granularity.

    Each flight's split is a pure function of ``(config.resolved_split_seed,
    airport:flight_id)`` — never of its POSITION in the list or of the model-training seed.
    A positional shuffle looks deterministic but reshuffles the whole assignment the moment
    one flight is added to or dropped from the harvest, silently promoting old test flights
    into training on the next retrain. The cost of per-flight hashing is that the realised
    fractions only approximate ``val_fraction`` / ``test_fraction`` (exact in expectation);
    the win is that a flight, once in the test set, stays there for every future harvest with
    the same split seed.
    """
    train, val, test = [], [], []
    for s in series:
        split = split_name_for_dataset_id(s.dataset_id, config)
        if split == "test":
            test.append(s)
        elif split == "val":
            val.append(s)
        else:
            train.append(s)
    if not train or not val:
        raise ValueError(
            f"split of {len(series)} flight(s) left train={len(train)}, val={len(val)}, "
            f"test={len(test)} at val={config.val_fraction}, test={config.test_fraction} — "
            f"too few flights for these fractions (training needs non-empty train AND val)"
        )
    return train, val, test


def cross_validation_folds(
    series: Sequence[FlightSeries], n_splits: int, *, seed: int
) -> list[list[FlightSeries]]:
    """Deterministic airport-stratified folds over an already locked outer-train set."""
    if n_splits < 2:
        raise ValueError(f"cross validation needs at least 2 folds, got {n_splits}")
    if len(series) < n_splits:
        raise ValueError(f"cannot split {len(series)} flight(s) into {n_splits} folds")

    by_airport: dict[str, list[FlightSeries]] = {}
    for item in series:
        by_airport.setdefault(item.airport or "<unknown>", []).append(item)

    folds: list[list[FlightSeries]] = [[] for _ in range(n_splits)]
    for airport, group in sorted(by_airport.items()):
        if len(group) < n_splits:
            raise ValueError(
                f"airport {airport} has only {len(group)} outer-train flight(s), fewer than "
                f"the requested {n_splits} folds"
            )
        ordered = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"cv:{seed}:{airport}:{item.dataset_id}".encode()
            ).digest(),
        )
        for index, item in enumerate(ordered):
            folds[index % n_splits].append(item)
    if any(not fold for fold in folds):
        raise ValueError("airport-stratified cross validation produced an empty fold")
    return folds
