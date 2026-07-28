#!/usr/bin/env python3
"""Build authoritative aircraft identity assets from versioned source snapshots.

The inputs remain separate because they answer different questions:

* FAA registry: which U.S. civil aircraft owns a Mode-S/ICAO24 address?
* ICAO Doc 8643: what is the canonical operational type designator?
* OpenSky snapshot: lower-authority registration history used only as crosswalk
  evidence when FAA certification names and ICAO operational names differ.

Every emitted typecode is validated against the supplied ICAO Doc 8643 snapshot.
Ambiguous FAA models remain unresolved instead of being guessed.

Run from the repository root with the project environment, for example::

    conda run -n aeroviz python -m aircraft.build_aircraft_identity_database \
      --faa-zip /path/to/ReleasableAircraft.zip \
      --icao-json /path/to/icao-aircraft-types.json \
      --icao-stats /path/to/icao-stats.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from aircraft.icao_type_designators import IcaoTypeDesignatorCatalog


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OPENSKY_CSV = REPO_ROOT / "data" / "AIRCRAFT" / "aircraftDatabase.csv"
DEFAULT_ICAO_OUTPUT = PACKAGE_DIR / "icao_doc8643.json"
DEFAULT_FAA_OUTPUT = PACKAGE_DIR / "faa_aircraft_identity.json"

FAA_SOURCE_URL = "https://registry.faa.gov/database/ReleasableAircraft.zip"
ICAO_TYPES_URL = "https://doc8643.icao.int/External/AircraftTypes"
ICAO_STATS_URL = "https://doc8643.icao.int/External/Stats"

CROSSWALK_MIN_SUPPORT = 2
CROSSWALK_MIN_SHARE = 0.95


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_icao_records(raw_records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in raw_records:
        typecode = str(raw.get("Designator") or raw.get("typecode") or "").strip().upper()
        if not typecode:
            continue
        records.append(
            {
                "manufacturer": str(
                    raw.get("ManufacturerCode") or raw.get("manufacturer") or ""
                ).strip(),
                "model": str(raw.get("ModelFullName") or raw.get("model") or "").strip(),
                "typecode": typecode,
                "description": _clean(raw.get("Description") or raw.get("description")),
                "aircraft_description": _clean(raw.get("AircraftDescription")),
                "engine_type": _clean(raw.get("EngineType")),
                "engine_count": _clean(raw.get("EngineCount")),
                "wtc": _clean(raw.get("WTC") or raw.get("wtc")),
                "wtg": _clean(raw.get("WTG")),
            }
        )
    return records


def build_icao_payload(
    raw_records: Iterable[Mapping[str, Any]],
    stats: Mapping[str, Any],
    *,
    retrieved_at_utc: str,
    raw_sha256: str,
) -> dict[str, Any]:
    records = normalize_icao_records(raw_records)
    return {
        "schema_version": 1,
        "source": {
            "standard": "ICAO Doc 8643",
            "aircraft_types_url": ICAO_TYPES_URL,
            "stats_url": ICAO_STATS_URL,
            "last_updated": stats.get("LastUpdated"),
            "next_update": stats.get("NextUpdate"),
            "retrieved_at_utc": retrieved_at_utc,
            "raw_sha256": raw_sha256,
        },
        "counts": {
            "records": len(records),
            "unique_typecodes": len({record["typecode"] for record in records}),
        },
        "records": records,
    }


def build_faa_payload(
    *,
    master_rows: Iterable[Mapping[str, str]],
    reference_rows: Iterable[Mapping[str, str]],
    opensky_rows: Iterable[Mapping[str, str]],
    catalog: IcaoTypeDesignatorCatalog,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    model_references = {
        row.get("CODE", "").strip(): {
            "manufacturer": row.get("MFR", "").strip(),
            "model": row.get("MODEL", "").strip(),
        }
        for row in reference_rows
        if row.get("CODE", "").strip()
    }

    registration_to_model_code: dict[str, str] = {}
    icao24_to_model_code: dict[str, str] = {}
    icao24_to_registration: dict[str, str] = {}
    active_model_codes: set[str] = set()
    for row in master_rows:
        model_code = row.get("MFR MDL CODE", "").strip()
        if not model_code:
            continue
        registration = "N" + row.get("N-NUMBER", "").strip().upper()
        icao24 = row.get("MODE S CODE HEX", "").strip().upper()
        if registration != "N":
            registration_to_model_code[registration] = model_code
        if icao24:
            icao24_to_model_code[icao24] = model_code
            if registration != "N":
                icao24_to_registration[icao24] = registration
        active_model_codes.add(model_code)

    evidence: dict[str, Counter[str]] = defaultdict(Counter)
    for row in opensky_rows:
        registration = row.get("registration", "").strip().upper()
        typecode = row.get("typecode", "").strip().upper()
        model_code = registration_to_model_code.get(registration)
        if model_code and catalog.contains(typecode):
            evidence[model_code][typecode] += 1

    resolved_by_method: Counter[str] = Counter()
    models: dict[str, dict[str, Any]] = {}
    for model_code in sorted(active_model_codes):
        reference = model_references.get(model_code, {"manufacturer": "", "model": ""})
        manufacturer = reference["manufacturer"]
        model = reference["model"]
        output: dict[str, Any] = {
            "manufacturer": manufacturer,
            "model": model,
        }

        match = catalog.match_faa_model(manufacturer, model)
        if match is not None:
            output.update(
                typecode=match.typecode,
                typecode_method=match.method,
                confidence=match.confidence,
            )
            resolved_by_method[match.method] += 1
        else:
            supported = evidence.get(model_code)
            if supported:
                typecode, count = supported.most_common(1)[0]
                total = sum(supported.values())
                share = count / total
                if count >= CROSSWALK_MIN_SUPPORT and share >= CROSSWALK_MIN_SHARE:
                    output.update(
                        typecode=catalog.normalize_typecode(typecode),
                        typecode_method="registration_crosswalk",
                        # OpenSky registration history is useful corroborating evidence,
                        # but unlike the FAA registry and ICAO catalog it is not an
                        # authority.  Do not overstate this inferred crosswalk.
                        confidence="medium",
                        crosswalk_support=count,
                        crosswalk_total=total,
                        crosswalk_share=round(share, 6),
                    )
                    resolved_by_method["registration_crosswalk"] += 1
        models[model_code] = output

    resolved_icao24 = sum(
        1
        for model_code in icao24_to_model_code.values()
        if models.get(model_code, {}).get("typecode")
    )
    return {
        "schema_version": 1,
        "source": dict(source),
        "crosswalk_policy": {
            "standard": "ICAO Doc 8643",
            "opensky_role": "registration history evidence only",
            "minimum_support": CROSSWALK_MIN_SUPPORT,
            "minimum_dominant_share": CROSSWALK_MIN_SHARE,
            "ambiguous_models": "left unresolved",
        },
        "counts": {
            "icao24_records": len(icao24_to_model_code),
            "active_model_codes": len(models),
            "resolved_model_codes": sum(bool(model.get("typecode")) for model in models.values()),
            "resolved_icao24_records": resolved_icao24,
            "resolved_by_method": dict(sorted(resolved_by_method.items())),
        },
        "icao24_to_model_code": dict(sorted(icao24_to_model_code.items())),
        "icao24_to_registration": dict(sorted(icao24_to_registration.items())),
        "models": models,
    }


def _clean(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _zip_csv_rows(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with archive.open(member) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            return list(csv.DictReader(text))


def _csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _snapshot_date(archive: zipfile.ZipFile, member: str) -> str:
    year, month, day, *_ = archive.getinfo(member).date_time
    return f"{year:04d}-{month:02d}-{day:02d}"


def write_json(payload: Mapping[str, Any], path: Path, *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o644)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":") if compact else None,
                indent=None if compact else 2,
            )
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faa-zip", type=Path, required=True)
    parser.add_argument("--icao-json", type=Path, required=True)
    parser.add_argument("--icao-stats", type=Path, required=True)
    parser.add_argument("--opensky-csv", type=Path, default=DEFAULT_OPENSKY_CSV)
    parser.add_argument("--icao-output", type=Path, default=DEFAULT_ICAO_OUTPUT)
    parser.add_argument("--faa-output", type=Path, default=DEFAULT_FAA_OUTPUT)
    parser.add_argument("--retrieved-at-utc", default=now_utc())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_icao = json.loads(args.icao_json.read_text(encoding="utf-8"))
    stats = json.loads(args.icao_stats.read_text(encoding="utf-8"))
    icao_payload = build_icao_payload(
        raw_icao,
        stats,
        retrieved_at_utc=args.retrieved_at_utc,
        raw_sha256=sha256_file(args.icao_json),
    )
    catalog = IcaoTypeDesignatorCatalog(
        icao_payload["records"],
        source=icao_payload["source"],
    )

    with zipfile.ZipFile(args.faa_zip) as archive:
        master_rows = _zip_csv_rows(archive, "MASTER.txt")
        reference_rows = _zip_csv_rows(archive, "ACFTREF.txt")
        snapshot_date = _snapshot_date(archive, "MASTER.txt")

    faa_payload = build_faa_payload(
        master_rows=master_rows,
        reference_rows=reference_rows,
        opensky_rows=_csv_rows(args.opensky_csv),
        catalog=catalog,
        source={
            "authority": "Federal Aviation Administration",
            "url": FAA_SOURCE_URL,
            # FAA does not expose a separate legal "effective" field for this bulk
            # export.  The MASTER member timestamp identifies the official snapshot.
            "snapshot_date": snapshot_date,
            "retrieved_at_utc": args.retrieved_at_utc,
            "faa_zip_sha256": sha256_file(args.faa_zip),
            "icao_last_updated": catalog.last_updated,
            "opensky_csv_sha256": sha256_file(args.opensky_csv),
        },
    )

    write_json(icao_payload, args.icao_output, compact=False)
    write_json(faa_payload, args.faa_output, compact=True)
    print(
        f"ICAO: {icao_payload['counts']['records']} records, "
        f"{icao_payload['counts']['unique_typecodes']} designators -> {args.icao_output}"
    )
    print(
        f"FAA: {faa_payload['counts']['icao24_records']} addresses, "
        f"{faa_payload['counts']['resolved_icao24_records']} standardized -> {args.faa_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
