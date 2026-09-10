"""Explicit train/validation flight rosters for controlled development experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEVELOPMENT_COHORT_SCHEMA = "ts-development-cohort-v1"


@dataclass(frozen=True)
class DevelopmentCohort:
    name: str
    train_flight_ids: tuple[str, ...]
    val_flight_ids: tuple[str, ...]
    selection: dict[str, Any]

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> DevelopmentCohort:
        if document.get("schema_version") != DEVELOPMENT_COHORT_SCHEMA:
            raise ValueError("development cohort has the wrong schema")
        splits = document["splits"]
        train_ids = tuple(splits["train"])
        val_ids = tuple(splits["val"])
        if set(train_ids) & set(val_ids):
            raise ValueError("development cohort repeats a flight across train and validation")
        return cls(
            name=str(document["name"]),
            train_flight_ids=train_ids,
            val_flight_ids=val_ids,
            selection=dict(document["selection"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DEVELOPMENT_COHORT_SCHEMA,
            "name": self.name,
            "selection": self.selection,
            "splits": {
                "train": list(self.train_flight_ids),
                "val": list(self.val_flight_ids),
            },
        }

    def development_flight_ids(
        self, outer_split_keys: dict[str, Sequence[str]]
    ) -> set[str]:
        train = set(outer_split_keys["train"])
        val = set(outer_split_keys["val"])
        if not set(self.train_flight_ids) <= train:
            raise ValueError("development cohort train roster does not match the locked split")
        if not set(self.val_flight_ids) <= val:
            raise ValueError("development cohort validation roster does not match the locked split")
        if not self.train_flight_ids or not self.val_flight_ids:
            raise ValueError("development cohort requires non-empty train and validation rosters")
        return set(self.train_flight_ids + self.val_flight_ids)


def load_development_cohort(path: str | Path) -> DevelopmentCohort:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return DevelopmentCohort.from_dict(document)


def write_development_cohort(
    path: str | Path,
    *,
    name: str,
    train_flight_ids: Sequence[str],
    val_flight_ids: Sequence[str],
    selection: dict[str, Any],
) -> DevelopmentCohort:
    cohort = DevelopmentCohort(
        name=name,
        train_flight_ids=tuple(sorted(train_flight_ids)),
        val_flight_ids=tuple(sorted(val_flight_ids)),
        selection=selection,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cohort.to_dict(), indent=2), encoding="utf-8")
    return cohort


def development_cohort_audit(
    path: str | Path, cohort: DevelopmentCohort
) -> dict[str, Any]:
    payload = Path(path).read_bytes()

    def digest(values: Sequence[str]) -> str:
        return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()

    return {
        "schema_version": DEVELOPMENT_COHORT_SCHEMA,
        "name": cohort.name,
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "selection": cohort.selection,
        "splits": {
            "train": {
                "flights": len(cohort.train_flight_ids),
                "identity_sha256": digest(cohort.train_flight_ids),
            },
            "val": {
                "flights": len(cohort.val_flight_ids),
                "identity_sha256": digest(cohort.val_flight_ids),
            },
        },
    }
