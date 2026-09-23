#!/usr/bin/env python3
"""Build authoritative aircraft identity assets from versioned source snapshots.

The inputs remain separate because they answer different questions:

* FAA registry: which U.S. civil aircraft owns a Mode-S/ICAO24 address?
* ICAO Doc 8643: what is the canonical operational type designator?
* OpenSky snapshot: lower-authority registration history used only as crosswalk
  evidence when FAA certification names and ICAO operational names differ.
* The documented crosswalk (``faa_icao_crosswalk.json``): FAA models whose designator is
  established by the model's TCDS or FAA Order JO 7360.1 together with the Doc 8643
  record, including serial-number splits the TCDS prints and models an authority shows
  to be ambiguous. Its rows OVERRIDE the name matcher and the registration crosswalk.

Every emitted typecode is validated against the supplied ICAO Doc 8643 snapshot.
Ambiguous FAA models remain unresolved instead of being guessed.

Run from the repository root with the project environment, for example::

    conda run -n aeroviz python -m aircraft.build_aircraft_identity_database \
      --faa-zip /path/to/ReleasableAircraft.zip \
      --icao-json /path/to/icao-aircraft-types.json \
      --icao-stats /path/to/icao-stats.json

or, rebuilding only the FAA identity against the ICAO snapshot already in the package::

    conda run -n aeroviz python -m aircraft.build_aircraft_identity_database \
      --faa-zip /path/to/ReleasableAircraft.zip --icao-catalog aircraft/icao_doc8643.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from aircraft.icao_type_designators import (
    ICAO_CATALOG_SCHEMA,
    ICAO_STANDARD,
    IcaoTypeDesignatorCatalog,
    compact_name,
)
from aircraft.identity import (
    DOCUMENTED_METHOD,
    DOCUMENTED_UNRESOLVED_METHOD,
    FAA_IDENTITY_SCHEMA,
    REGISTRATION_CROSSWALK_METHOD,
    SERIAL_SPLIT_METHOD,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OPENSKY_CSV = REPO_ROOT / "data" / "AIRCRAFT" / "aircraftDatabase.csv"
DEFAULT_ICAO_OUTPUT = PACKAGE_DIR / "icao_doc8643.json"
DEFAULT_FAA_OUTPUT = PACKAGE_DIR / "faa_aircraft_identity.json"
DEFAULT_CROSSWALK = PACKAGE_DIR / "faa_icao_crosswalk.json"
CROSSWALK_SCHEMA = 1

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
        "schema_version": ICAO_CATALOG_SCHEMA,
        "source": {
            "standard": ICAO_STANDARD,
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


def load_documented_crosswalk(
    path: Path, catalog: IcaoTypeDesignatorCatalog
) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    """``{(FAA manufacturer, FAA model): row}`` for every registry spelling a row lists, and
    the file's sha256, validated.

    Every designator a row emits must be the one the snapshot gives the Doc 8643 record
    the row cites -- a row whose evidence no longer matches the snapshot raises. A serial
    split's pattern must capture exactly one group and its ranges must not overlap.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CROSSWALK_SCHEMA:
        raise ValueError(f"{path}: expected documented-crosswalk schema {CROSSWALK_SCHEMA}")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload["models"]:
        name = row["faa_records"][0]
        if row["decision"] == "typecode":
            cited = [(row["typecode"], record) for record in row["icao_records"]]
        elif row["decision"] == "serial_split":
            _check_serial_split(row, name)
            cited = [(split["typecode"], split["icao_record"]) for split in row["serial_split"]]
        elif row["decision"] == "unresolved":
            cited = []
        else:
            raise ValueError(f"{path}: {name} has unknown decision {row['decision']!r}")
        if row["decision"] != "unresolved" and not cited:
            raise ValueError(f"{path}: {name} names no Doc 8643 record")
        for typecode, (manufacturer, model) in cited:
            if typecode not in catalog.record_typecodes(manufacturer, model):
                raise ValueError(
                    f"{path}: {name} -> {typecode}, but Doc 8643 record {manufacturer!r} | "
                    f"{model!r} gives {sorted(catalog.record_typecodes(manufacturer, model))}"
                )
        for manufacturer, model in row["faa_records"]:
            if (manufacturer, model) in rows:
                raise ValueError(f"{path}: {(manufacturer, model)} is listed twice")
            rows[(manufacturer, model)] = row
    return rows, sha256_file(path)


def _check_serial_split(row: Mapping[str, Any], name: object) -> None:
    if re.compile(row["serial_pattern"]).groups != 1:
        raise ValueError(f"{name}: serial_pattern must capture exactly the serial number")
    splits = row["serial_split"]
    for split in splits:
        last = split["last"] if split["last"] is not None else float("inf")
        excepted = set(split["except"])
        if (
            not split["first"] <= last
            or not all(split["first"] <= number <= last for number in excepted)
            or last - split["first"] + 1 <= len(excepted)
        ):
            raise ValueError(f"{name}: serial range {split} is empty or excepts outside itself")
    for index, a in enumerate(splits):
        for b in splits[index + 1:]:
            low = max(a["first"], b["first"])
            high = min(
                a["last"] if a["last"] is not None else float("inf"),
                b["last"] if b["last"] is not None else float("inf"),
            )
            if low > high:
                continue
            if high == float("inf") or any(
                number not in a["except"] and number not in b["except"]
                for number in range(low, int(high) + 1)
            ):
                raise ValueError(f"{name}: serial ranges {a} and {b} overlap")


def serial_split_typecode(serial: str, pattern: str, rule: list[Mapping[str, Any]]) -> str | None:
    """The designator a documented serial split gives ``serial``; None when the serial does
    not have the documented form (nothing is guessed out of it) or lies outside every range."""
    match = re.fullmatch(pattern, serial, flags=re.ASCII)
    if match is None or not (match.group(1).isascii() and match.group(1).isdigit()):
        return None
    number = int(match.group(1))
    for split in rule:
        last = split["last"] if split["last"] is not None else number
        if split["first"] <= number <= last and number not in split["except"]:
            return split["typecode"]
    return None


def build_faa_payload(
    *,
    master_rows: Iterable[Mapping[str, str]],
    reference_rows: Iterable[Mapping[str, str]],
    opensky_rows: Iterable[Mapping[str, str]],
    catalog: IcaoTypeDesignatorCatalog,
    source: Mapping[str, Any],
    documented_crosswalk: Mapping[tuple[str, str], Mapping[str, Any]],
    documented_crosswalk_sha256: str | None,
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
    icao24_to_serial: dict[str, str] = {}
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
            icao24_to_serial[icao24] = row.get("SERIAL NUMBER", "").strip()
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

        documented = documented_crosswalk.get((manufacturer, model))
        match = None if documented is not None else catalog.match_faa_model(manufacturer, model)
        if documented is not None:
            if documented["decision"] == "typecode":
                output.update(
                    typecode=documented["typecode"],
                    typecode_method=DOCUMENTED_METHOD,
                    confidence="high",
                )
                resolved_by_method[DOCUMENTED_METHOD] += 1
            elif documented["decision"] == "serial_split":
                # Per aircraft, below: the model itself spans several designators.
                output.update(
                    typecode_method=SERIAL_SPLIT_METHOD,
                    serial_pattern=documented["serial_pattern"],
                    serial_split=[
                        {key: split[key] for key in ("first", "last", "except", "typecode")}
                        for split in documented["serial_split"]
                    ],
                )
            else:
                output.update(
                    typecode_method=DOCUMENTED_UNRESOLVED_METHOD,
                    unresolved_reason=documented["reason"],
                )
        elif match is not None:
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
                        typecode_method=REGISTRATION_CROSSWALK_METHOD,
                        # OpenSky registration history is useful corroborating evidence,
                        # but unlike the FAA registry and ICAO catalog it is not an
                        # authority.  Do not overstate this inferred crosswalk.
                        confidence="medium",
                        crosswalk_support=count,
                        crosswalk_total=total,
                        crosswalk_share=round(share, 6),
                    )
                    resolved_by_method[REGISTRATION_CROSSWALK_METHOD] += 1
        models[model_code] = output

    icao24_documented_typecode: dict[str, str] = {}
    for icao24, model_code in icao24_to_model_code.items():
        rule = models[model_code].get("serial_split")
        if rule is None:
            continue
        typecode = serial_split_typecode(
            icao24_to_serial[icao24], models[model_code]["serial_pattern"], rule
        )
        if typecode is not None:
            icao24_documented_typecode[icao24] = typecode

    resolved_icao24 = sum(
        1
        for icao24, model_code in icao24_to_model_code.items()
        if models.get(model_code, {}).get("typecode") or icao24 in icao24_documented_typecode
    )
    return {
        "schema_version": FAA_IDENTITY_SCHEMA,
        "source": dict(source),
        "crosswalk_policy": {
            "standard": ICAO_STANDARD,
            "opensky_role": "registration history evidence only",
            "minimum_support": CROSSWALK_MIN_SUPPORT,
            "minimum_dominant_share": CROSSWALK_MIN_SHARE,
            "ambiguous_models": "left unresolved",
            "documented_crosswalk": {
                "role": "overrides the name matcher and the registration crosswalk",
                "registry_spellings": len(documented_crosswalk),
                "sha256": documented_crosswalk_sha256,
            },
        },
        "counts": {
            "icao24_records": len(icao24_to_model_code),
            "active_model_codes": len(models),
            "resolved_model_codes": sum(bool(model.get("typecode")) for model in models.values()),
            "resolved_icao24_records": resolved_icao24,
            "resolved_by_method": dict(sorted(resolved_by_method.items())),
            "serial_split_resolved_icao24_records": len(icao24_documented_typecode),
        },
        "icao24_to_model_code": dict(sorted(icao24_to_model_code.items())),
        "icao24_to_registration": dict(sorted(icao24_to_registration.items())),
        "icao24_documented_typecode": dict(sorted(icao24_documented_typecode.items())),
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


def _print_unlisted_spellings(
    models: Mapping[str, Mapping[str, Any]], documented: Mapping[tuple[str, str], Any]
) -> None:
    """Name every registry spelling the crosswalk does not list whose model string matches a
    listed one: a new snapshot's spelling of a documented model would otherwise fall through
    to the heuristics unnoticed (it may also be a different aircraft, e.g. ROCKWELL 700)."""
    listed = {compact_name(model) for _, model in documented}
    unlisted = sorted(
        (model["manufacturer"], model["model"], model.get("typecode"))
        for model in models.values()
        if (model["manufacturer"], model["model"]) not in documented
        and compact_name(model["model"]) in listed
    )
    if unlisted:
        print(f"notice: {len(unlisted)} registry spelling(s) share a documented model string but "
              "are not in the crosswalk -- check each is a different aircraft:")
        for manufacturer, model, typecode in unlisted:
            print(f"  {manufacturer} | {model} -> {typecode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faa-zip", type=Path, required=True)
    parser.add_argument("--icao-json", type=Path, default=None,
                        help="raw ICAO Doc 8643 download (with --icao-stats): rebuild the catalog too")
    parser.add_argument("--icao-stats", type=Path, default=None)
    parser.add_argument("--icao-catalog", type=Path, default=None,
                        help="an existing icao_doc8643.json: rebuild only the FAA identity against it")
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--opensky-csv", type=Path, default=DEFAULT_OPENSKY_CSV)
    parser.add_argument("--icao-output", type=Path, default=DEFAULT_ICAO_OUTPUT)
    parser.add_argument("--faa-output", type=Path, default=DEFAULT_FAA_OUTPUT)
    parser.add_argument("--retrieved-at-utc", default=now_utc())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_icao_args = (args.icao_json, args.icao_stats)
    if (
        args.icao_catalog is not None and any(raw_icao_args)
        or args.icao_catalog is None and not all(raw_icao_args)
    ):
        raise SystemExit("give either --icao-json with --icao-stats, or --icao-catalog alone")
    if args.icao_catalog is not None:
        icao_payload = None
        catalog = IcaoTypeDesignatorCatalog.from_json(args.icao_catalog)
    else:
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
    documented, documented_sha256 = load_documented_crosswalk(args.crosswalk, catalog)

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
            "icao_raw_sha256": catalog.source["raw_sha256"],
            "opensky_csv_sha256": sha256_file(args.opensky_csv),
        },
        documented_crosswalk=documented,
        documented_crosswalk_sha256=documented_sha256,
    )

    if icao_payload is not None:
        write_json(icao_payload, args.icao_output, compact=False)
        print(
            f"ICAO: {icao_payload['counts']['records']} records, "
            f"{icao_payload['counts']['unique_typecodes']} designators -> {args.icao_output}"
        )
    write_json(faa_payload, args.faa_output, compact=True)
    _print_unlisted_spellings(faa_payload["models"], documented)
    print(
        f"FAA: {faa_payload['counts']['icao24_records']} addresses, "
        f"{faa_payload['counts']['resolved_icao24_records']} standardized -> {args.faa_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
