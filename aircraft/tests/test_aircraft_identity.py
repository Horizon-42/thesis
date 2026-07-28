import json

import pytest

from aircraft.build_aircraft_identity_database import build_faa_payload
from aircraft.icao_type_designators import IcaoTypeDesignatorCatalog
from aircraft.identity import AircraftIdentityResolver, get_default_identity_resolver
from aircraft.query_aircraft_parameters import (
    get_aircraft_parameters,
    openap_direct_typecodes,
    openap_performance_metadata,
    openap_support_kind,
)


ICAO_RECORDS = [
    {
        "manufacturer": "AIRBUS",
        "model": "A-321neo",
        "typecode": "A21N",
        "description": "L2J",
        "wtc": "M",
    },
    {
        "manufacturer": "BOEING",
        "model": "737-8",
        "typecode": "B38M",
        "description": "L2J",
        "wtc": "M",
    },
    {
        "manufacturer": "BOEING",
        "model": "737-900",
        "typecode": "B739",
        "description": "L2J",
        "wtc": "M",
    },
    {
        "manufacturer": "BOMBARDIER",
        "model": "BD-500 CSeries CS300",
        "typecode": "BCS3",
        "description": "L2J",
        "wtc": "M",
    },
]


def _catalog() -> IcaoTypeDesignatorCatalog:
    return IcaoTypeDesignatorCatalog(
        ICAO_RECORDS,
        source={"last_updated": "10 July 2026"},
    )


def test_icao_catalog_normalizes_only_official_designators():
    catalog = _catalog()

    assert catalog.normalize_typecode(" b38m ") == "B38M"
    with pytest.raises(KeyError, match="not present in ICAO Doc 8643"):
        catalog.normalize_typecode("NOTREAL")


@pytest.mark.parametrize(
    ("manufacturer", "model", "expected"),
    [
        ("BOEING", "737-8", "B38M"),
        ("AIRBUS S A S", "A321-271NX", "A21N"),
        ("AIRBUS CANADA LP", "BD-500-1A11", "BCS3"),
    ],
)
def test_icao_catalog_maps_faa_certificated_models_conservatively(
    manufacturer: str,
    model: str,
    expected: str,
):
    match = _catalog().match_faa_model(manufacturer, model)

    assert match is not None
    assert match.typecode == expected
    assert match.standard == "ICAO Doc 8643"


def test_identity_resolver_prefers_faa_and_audits_icao_normalization():
    resolver = AircraftIdentityResolver(
        catalog=_catalog(),
        faa_registry={
            "source": {"effective_date": "2026-07-27"},
            "icao24_to_model_code": {"AD63F7": "13844FN"},
            "icao24_to_registration": {"AD63F7": "N961AN"},
            "models": {
                "13844FN": {
                    "manufacturer": "BOEING",
                    "model": "737-9GPER",
                    "typecode": "B739",
                    "typecode_method": "registration_crosswalk",
                    "confidence": "medium",
                }
            },
        },
        opensky_lookup={"icao24_to_typecode": {"AD63F7": "A320"}},
    )

    identity = resolver.resolve(declared_type="UNK", icao24="ad63f7")

    assert identity.typecode == "B739"
    assert identity.identity_source == "faa_registry"
    assert identity.identity_source_date == "2026-07-27"
    assert identity.typecode_standard == "ICAO Doc 8643"
    assert identity.typecode_standard_date == "10 July 2026"
    assert identity.typecode_method == "registration_crosswalk"
    assert identity.typecode_source == "faa_registry+opensky_evidence"
    assert identity.faa_model_code == "13844FN"
    assert identity.registration == "N961AN"


def test_identity_resolver_rejects_non_icao_opensky_typecode():
    resolver = AircraftIdentityResolver(
        catalog=_catalog(),
        faa_registry={"source": {}, "icao24_to_model_code": {}, "models": {}},
        opensky_lookup={"icao24_to_typecode": {"ABC123": "NOTREAL"}},
    )

    identity = resolver.resolve(declared_type="UNK", icao24="abc123")

    assert identity.typecode is None
    assert identity.identity_source == "unresolved"
    assert "NOTREAL" in identity.failure_reason


def test_opensky_cache_build_time_is_not_reported_as_source_date():
    resolver = AircraftIdentityResolver(
        catalog=_catalog(),
        faa_registry={"source": {}, "icao24_to_model_code": {}, "models": {}},
        opensky_lookup={
            "source": {"generated_at_utc": "2026-07-28T21:32:40Z"},
            "icao24_to_typecode": {"ABC123": "B38M"},
        },
    )

    identity = resolver.resolve(declared_type="UNK", icao24="abc123")

    assert identity.identity_source == "opensky"
    assert identity.identity_source_date is None


def test_identity_resolver_distinguishes_faa_identity_from_opensky_typecode():
    resolver = AircraftIdentityResolver(
        catalog=_catalog(),
        faa_registry={
            "source": {"effective_date": "2026-07-27"},
            "icao24_to_model_code": {"ABC123": "UNKNOWN"},
            "icao24_to_registration": {"ABC123": "N123ZZ"},
            "models": {"UNKNOWN": {"manufacturer": "UNKNOWN", "model": "MODEL"}},
        },
        opensky_lookup={"icao24_to_typecode": {"ABC123": "B38M"}},
    )

    identity = resolver.resolve(declared_type="UNK", icao24="abc123")

    assert identity.typecode == "B38M"
    assert identity.identity_source == "faa_registry"
    assert identity.typecode_source == "opensky"
    assert identity.registration == "N123ZZ"


def test_identity_resolver_loads_versioned_json_sources(tmp_path):
    catalog_path = tmp_path / "icao.json"
    faa_path = tmp_path / "faa.json"
    opensky_path = tmp_path / "opensky.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {"last_updated": "10 July 2026"},
                "records": ICAO_RECORDS,
            }
        ),
        encoding="utf-8",
    )
    faa_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {"effective_date": "2026-07-27"},
                "icao24_to_model_code": {},
                "models": {},
            }
        ),
        encoding="utf-8",
    )
    opensky_path.write_text(
        json.dumps({"schema_version": 1, "icao24_to_typecode": {}}),
        encoding="utf-8",
    )

    resolver = AircraftIdentityResolver.from_paths(
        icao_path=catalog_path,
        faa_path=faa_path,
        opensky_path=opensky_path,
    )

    assert resolver.catalog.normalize_typecode("A21N") == "A21N"


def test_generated_faa_snapshot_emits_only_official_icao_designators():
    resolver = get_default_identity_resolver()
    emitted = {
        model["typecode"]
        for model in resolver.faa_registry["models"].values()
        if model.get("typecode")
    }

    assert emitted
    assert all(resolver.catalog.contains(typecode) for typecode in emitted)
    assert resolver.catalog.source["standard"] == "ICAO Doc 8643"
    assert resolver.faa_registry["source"]["authority"] == (
        "Federal Aviation Administration"
    )


def test_openap_synonym_typecode_retains_icao_identity():
    aircraft = get_aircraft_parameters("A306")

    assert aircraft.code == "A306"
    assert aircraft.geometry.wing_area_m2 > 0.0
    assert aircraft.engine.max_thrust_total_n > 0.0
    metadata = openap_performance_metadata("A306")
    assert metadata["source"].startswith("openap-")
    assert metadata["performance_typecode"] == "A332"
    assert metadata["uses_synonym"] is True


def test_openap_direct_model_wins_when_type_also_appears_in_synonym_table():
    # OpenAP 2.4 lists CRJ9 in both available_aircraft() and aircraft_synonym.
    # prop.aircraft() uses the direct model, so provenance must say the same.
    aircraft = get_aircraft_parameters("CRJ9")

    assert aircraft.code == "CRJ9"
    metadata = openap_performance_metadata("CRJ9")
    assert metadata["performance_typecode"] == "CRJ9"
    assert metadata["uses_synonym"] is False


def test_openap_support_kind_distinguishes_native_synonym_and_unsupported_models():
    assert openap_support_kind("A320") == "direct"
    assert openap_support_kind("A306") == "synonym"
    assert openap_support_kind("BCS3") is None
    assert "A320" in openap_direct_typecodes()
    assert "A306" not in openap_direct_typecodes()


def test_faa_builder_uses_direct_icao_match_and_rejects_ambiguous_crosswalk():
    catalog = _catalog()
    payload = build_faa_payload(
        master_rows=[
            {"N-NUMBER": "1AA", "MFR MDL CODE": "DIRECT", "MODE S CODE HEX": "A00001"},
            {"N-NUMBER": "2AA", "MFR MDL CODE": "CROSS", "MODE S CODE HEX": "A00002"},
            {"N-NUMBER": "3AA", "MFR MDL CODE": "CROSS", "MODE S CODE HEX": "A00003"},
            {"N-NUMBER": "4AA", "MFR MDL CODE": "AMBIG", "MODE S CODE HEX": "A00004"},
            {"N-NUMBER": "5AA", "MFR MDL CODE": "AMBIG", "MODE S CODE HEX": "A00005"},
        ],
        reference_rows=[
            {"CODE": "DIRECT", "MFR": "BOEING", "MODEL": "737-8"},
            {"CODE": "CROSS", "MFR": "BOEING", "MODEL": "737-9GPER"},
            {"CODE": "AMBIG", "MFR": "UNKNOWN", "MODEL": "MODEL"},
        ],
        opensky_rows=[
            {"registration": "N2AA", "typecode": "B739"},
            {"registration": "N3AA", "typecode": "B739"},
            {"registration": "N4AA", "typecode": "B739"},
            {"registration": "N5AA", "typecode": "A21N"},
        ],
        catalog=catalog,
        source={"effective_date": "2026-07-27"},
    )

    assert payload["models"]["DIRECT"]["typecode"] == "B38M"
    assert payload["models"]["DIRECT"]["typecode_method"] == "icao_exact_model"
    assert payload["models"]["CROSS"]["typecode"] == "B739"
    assert payload["models"]["CROSS"]["typecode_method"] == "registration_crosswalk"
    assert payload["models"]["CROSS"]["confidence"] == "medium"
    assert "typecode" not in payload["models"]["AMBIG"]
    assert payload["icao24_to_registration"]["A00001"] == "N1AA"
