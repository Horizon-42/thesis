"""ICAO Doc 8643 type-designator validation and conservative model matching.

This module owns only the *standardisation* layer.  It does not know how aircraft
performance is modelled and it does not treat a registry's make/model string as an
OpenAP key.  Registry adapters may ask the catalog for a canonical ICAO typecode;
the performance provider consumes that typecode afterwards.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ICAO_STANDARD = "ICAO Doc 8643"


def _compact(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "").upper()
    return "".join(character for character in text if character.isalnum())


def _manufacturer_family(value: str | None) -> str:
    compact = _compact(value)
    aliases = (
        ("AIRBUSCANADA", "AIRBUS_CANADA"),
        ("AIRBUS", "AIRBUS"),
        ("BOEING", "BOEING"),
        ("BOMBARDIER", "BOMBARDIER"),
        ("CANADAIR", "BOMBARDIER"),
        ("EMBRAER", "EMBRAER"),
        ("CESSNA", "CESSNA"),
        ("TEXTRONAVIATION", "CESSNA"),
        ("PIPER", "PIPER"),
        ("PILATUS", "PILATUS"),
        ("GULFSTREAM", "GULFSTREAM"),
    )
    for prefix, family in aliases:
        if compact.startswith(prefix):
            return family
    return compact


@dataclass(frozen=True, slots=True)
class TypecodeMatch:
    typecode: str
    method: str
    confidence: str
    standard: str = ICAO_STANDARD


class IcaoTypeDesignatorCatalog:
    """An immutable view of one versioned ICAO Doc 8643 snapshot."""

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        source: Mapping[str, Any] | None = None,
    ) -> None:
        normalized: list[dict[str, str | None]] = []
        by_typecode: dict[str, list[dict[str, str | None]]] = {}
        by_family: dict[str, list[tuple[str, str]]] = {}
        for raw in records:
            record = {
                "manufacturer": str(
                    raw.get("manufacturer") or raw.get("ManufacturerCode") or ""
                ).strip(),
                "model": str(raw.get("model") or raw.get("ModelFullName") or "").strip(),
                "typecode": str(raw.get("typecode") or raw.get("Designator") or "")
                .strip()
                .upper(),
                "description": _optional_string(
                    raw.get("description") or raw.get("Description")
                ),
                "wtc": _optional_string(raw.get("wtc") or raw.get("WTC")),
            }
            if not record["typecode"]:
                continue
            normalized.append(record)
            by_typecode.setdefault(record["typecode"], []).append(record)
            by_family.setdefault(
                _manufacturer_family(record["manufacturer"]), []
            ).append((_compact(record["model"]), record["typecode"]))

        self._records = tuple(normalized)
        self._by_typecode = by_typecode
        self._by_family = {
            family: tuple(values) for family, values in by_family.items()
        }
        self.source = dict(source or {})

    @classmethod
    def from_json(cls, path: str | Path) -> "IcaoTypeDesignatorCatalog":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported ICAO catalog schema in {path}")
        return cls(payload.get("records", []), source=payload.get("source", {}))

    @property
    def records(self) -> tuple[dict[str, str | None], ...]:
        return self._records

    @property
    def last_updated(self) -> str | None:
        value = self.source.get("last_updated")
        return str(value) if value else None

    def contains(self, typecode: str | None) -> bool:
        return (typecode or "").strip().upper() in self._by_typecode

    def normalize_typecode(self, typecode: str) -> str:
        normalized = typecode.strip().upper()
        if normalized not in self._by_typecode:
            raise KeyError(f"typecode {normalized!r} is not present in {ICAO_STANDARD}")
        return normalized

    def match_faa_model(self, manufacturer: str, model: str) -> TypecodeMatch | None:
        """Map an FAA certificated model only when the ICAO result is unambiguous.

        Exact ICAO make/model matches win.  A small set of certificated-family rules
        covers Airbus/Bombardier identifiers whose certification names intentionally
        differ from the operational type name.  Finally, a unique model-prefix match
        handles values such as ``EMB-505`` versus ``EMB-505 Phenom 300``.  Ambiguous
        variants (for example one certification model spanning two operational types)
        deliberately return ``None`` and require an audited registry crosswalk.
        """
        family = _manufacturer_family(manufacturer)
        model_key = _compact(model)
        if not model_key:
            return None

        candidates = self._by_family.get(family, ())
        exact = self._unique_typecode(
            typecode for candidate_model, typecode in candidates if candidate_model == model_key
        )
        if exact is not None:
            return TypecodeMatch(exact, "icao_exact_model", "high")

        family_typecode = self._certificated_family_typecode(family, model_key)
        if family_typecode is not None and self.contains(family_typecode):
            return TypecodeMatch(
                family_typecode,
                "faa_certificated_model_rule",
                "high",
            )

        prefix = self._unique_typecode(
            typecode
            for candidate_model, typecode in candidates
            if min(len(model_key), len(candidate_model)) >= 5
            and (
                model_key.startswith(candidate_model)
                or candidate_model.startswith(model_key)
            )
        )
        if prefix is not None:
            return TypecodeMatch(prefix, "icao_unique_model_prefix", "medium")
        return None

    @staticmethod
    def _unique_typecode(typecodes: Iterable[str]) -> str | None:
        typecodes = {str(typecode).upper() for typecode in typecodes}
        return next(iter(typecodes)) if len(typecodes) == 1 else None

    @staticmethod
    def _certificated_family_typecode(family: str, model_key: str) -> str | None:
        if family == "AIRBUS":
            match = re.fullmatch(r"A(318|319|320|321)[A-Z0-9]*", model_key)
            if match:
                member = match.group(1)
                is_neo = model_key.endswith("N") or model_key.endswith("NX")
                if is_neo and member in {"319", "320", "321"}:
                    return {"319": "A19N", "320": "A20N", "321": "A21N"}[member]
                return f"A{member}"

        if family in {"AIRBUS_CANADA", "BOMBARDIER"}:
            if model_key.startswith("BD5001A10"):
                return "BCS1"
            if model_key.startswith("BD5001A11"):
                return "BCS3"
        return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
