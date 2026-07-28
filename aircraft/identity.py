"""Modular aircraft identity resolution with explicit source provenance.

Resolution and performance are intentionally separate:

``declared type / FAA registry / OpenSky`` -> ICAO Doc 8643 typecode
``ICAO typecode`` -> OpenAP performance (owned by ``query_aircraft_parameters``)

An identity may therefore be known even when OpenAP has no corresponding dynamics.
That distinction is preserved for scenario audit instead of being erased by A320.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .icao_type_designators import ICAO_STANDARD, IcaoTypeDesignatorCatalog


PACKAGE_DIR = Path(__file__).resolve().parent
ICAO_CATALOG_PATH = PACKAGE_DIR / "icao_doc8643.json"
FAA_REGISTRY_PATH = PACKAGE_DIR / "faa_aircraft_identity.json"
OPENSKY_LOOKUP_PATH = PACKAGE_DIR / "aircraft_id_lookup.json"


@dataclass(frozen=True, slots=True)
class AircraftIdentity:
    typecode: str | None
    identity_source: str
    typecode_source: str | None = None
    typecode_standard: str = ICAO_STANDARD
    typecode_standard_date: str | None = None
    identity_source_date: str | None = None
    typecode_method: str | None = None
    confidence: str | None = None
    registration: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    faa_model_code: str | None = None
    failure_reason: str | None = None

    def audit_fields(self) -> dict[str, Any]:
        return asdict(self)


class AircraftIdentityResolver:
    def __init__(
        self,
        *,
        catalog: IcaoTypeDesignatorCatalog,
        faa_registry: Mapping[str, Any],
        opensky_lookup: Mapping[str, Any],
    ) -> None:
        self.catalog = catalog
        self.faa_registry = dict(faa_registry)
        self.opensky_lookup = dict(opensky_lookup)

    @classmethod
    def from_paths(
        cls,
        *,
        icao_path: str | Path = ICAO_CATALOG_PATH,
        faa_path: str | Path = FAA_REGISTRY_PATH,
        opensky_path: str | Path = OPENSKY_LOOKUP_PATH,
    ) -> "AircraftIdentityResolver":
        catalog = IcaoTypeDesignatorCatalog.from_json(icao_path)
        faa_registry = _load_versioned_json(faa_path, "FAA aircraft identity")
        opensky_lookup = _load_versioned_json(opensky_path, "OpenSky aircraft lookup")
        return cls(
            catalog=catalog,
            faa_registry=faa_registry,
            opensky_lookup=opensky_lookup,
        )

    def resolve(self, *, declared_type: str | None, icao24: str | None) -> AircraftIdentity:
        failures: list[str] = []
        declared = (declared_type or "").strip().upper()
        if declared and declared != "UNK":
            try:
                typecode = self.catalog.normalize_typecode(declared)
            except KeyError as exc:
                failures.append(str(exc))
            else:
                return self._identity(
                    typecode=typecode,
                    identity_source="declared_type",
                    typecode_source="declared_type",
                    typecode_method="direct_designator",
                    confidence="high",
                )

        normalized_icao24 = (icao24 or "").strip().upper()
        faa_record = self._faa_record(normalized_icao24)
        if faa_record is not None:
            typecode = faa_record.get("typecode")
            if typecode:
                try:
                    normalized_typecode = self.catalog.normalize_typecode(str(typecode))
                except KeyError as exc:
                    failures.append(f"FAA crosswalk: {exc}")
                else:
                    return self._faa_identity(
                        faa_record,
                        typecode=normalized_typecode,
                    )

        opensky_typecode = (
            self.opensky_lookup.get("icao24_to_typecode", {}).get(normalized_icao24)
            if normalized_icao24
            else None
        )
        if opensky_typecode:
            try:
                normalized_typecode = self.catalog.normalize_typecode(str(opensky_typecode))
            except KeyError as exc:
                failures.append(f"OpenSky lookup: {exc}")
            else:
                if faa_record is not None:
                    return self._faa_identity(
                        faa_record,
                        typecode=normalized_typecode,
                        method="opensky_icao24_validated",
                        confidence="medium",
                        typecode_source="opensky",
                    )
                return self._identity(
                    typecode=normalized_typecode,
                    identity_source="opensky",
                    typecode_source="opensky",
                    identity_source_date=_source_date(self.opensky_lookup.get("source", {})),
                    typecode_method="opensky_icao24_validated",
                    confidence="medium",
                )

        if faa_record is not None:
            return self._faa_identity(
                faa_record,
                typecode=None,
                method=None,
                confidence=None,
                failure_reason="; ".join(failures) or "FAA model has no unambiguous ICAO typecode",
            )
        return self._identity(
            typecode=None,
            identity_source="unresolved",
            failure_reason="; ".join(failures) or "no aircraft identity source matched",
        )

    def _faa_record(self, icao24: str) -> dict[str, Any] | None:
        if not icao24:
            return None
        model_code = self.faa_registry.get("icao24_to_model_code", {}).get(icao24)
        if not model_code:
            return None
        model = self.faa_registry.get("models", {}).get(model_code)
        if not model:
            return None
        registration = self.faa_registry.get("icao24_to_registration", {}).get(icao24)
        return {
            **model,
            "faa_model_code": model_code,
            "registration": registration,
        }

    def _faa_identity(
        self,
        record: Mapping[str, Any],
        *,
        typecode: str | None,
        method: str | None = None,
        confidence: str | None = None,
        typecode_source: str | None = None,
        failure_reason: str | None = None,
    ) -> AircraftIdentity:
        resolved_method = method or record.get("typecode_method")
        if typecode is not None and typecode_source is None:
            typecode_source = (
                "faa_registry+opensky_evidence"
                if resolved_method == "registration_crosswalk"
                else "faa_registry+icao_doc8643"
            )
        return self._identity(
            typecode=typecode,
            identity_source="faa_registry",
            typecode_source=typecode_source,
            identity_source_date=_source_date(self.faa_registry.get("source", {})),
            typecode_method=resolved_method,
            confidence=confidence or record.get("confidence"),
            registration=record.get("registration"),
            manufacturer=record.get("manufacturer"),
            model=record.get("model"),
            faa_model_code=record.get("faa_model_code"),
            failure_reason=failure_reason,
        )

    def _identity(self, **values: Any) -> AircraftIdentity:
        return AircraftIdentity(
            typecode_standard_date=self.catalog.last_updated,
            **values,
        )


def _load_versioned_json(path: str | Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {label} database {path}; regenerate authoritative aircraft identity data"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported {label} schema in {path}")
    return payload


def _source_date(source: Mapping[str, Any]) -> str | None:
    # Cache generation/retrieval time says when we copied bytes, not when the
    # authority's records became effective.  Only expose an actual source date.
    value = (
        source.get("effective_date")
        or source.get("snapshot_date")
        or source.get("last_updated")
    )
    return str(value) if value else None


@lru_cache(maxsize=1)
def get_default_identity_resolver() -> AircraftIdentityResolver:
    return AircraftIdentityResolver.from_paths()
