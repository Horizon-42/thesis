import json

import pytest

from aircraft.build_aircraft_identity_database import (
    CROSSWALK_SCHEMA,
    DEFAULT_CROSSWALK,
    build_faa_payload,
    load_documented_crosswalk,
    sha256_file,
)
from aircraft.icao_type_designators import ICAO_CATALOG_SCHEMA, IcaoTypeDesignatorCatalog
from aircraft.identity import (
    DOCUMENTED_METHOD,
    DOCUMENTED_UNRESOLVED_METHOD,
    FAA_IDENTITY_SCHEMA,
    ICAO_CATALOG_PATH,
    OPENSKY_LOOKUP_SCHEMA,
    REGISTRATION_CROSSWALK_METHOD,
    AircraftIdentityResolver,
    get_default_identity_resolver,
)
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
            "icao24_documented_typecode": {},
            "models": {
                "13844FN": {
                    "manufacturer": "BOEING",
                    "model": "737-9GPER",
                    "typecode": "B739",
                    "typecode_method": REGISTRATION_CROSSWALK_METHOD,
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
    assert identity.typecode_method == REGISTRATION_CROSSWALK_METHOD
    assert identity.typecode_source == "faa_registry+opensky_evidence"
    assert identity.faa_model_code == "13844FN"
    assert identity.registration == "N961AN"


def test_identity_resolver_rejects_non_icao_opensky_typecode():
    resolver = AircraftIdentityResolver(
        catalog=_catalog(),
        faa_registry={"source": {}, "icao24_to_model_code": {}, "models": {},
                      "icao24_documented_typecode": {}},
        opensky_lookup={"icao24_to_typecode": {"ABC123": "NOTREAL"}},
    )

    identity = resolver.resolve(declared_type="UNK", icao24="abc123")

    assert identity.typecode is None
    assert identity.identity_source == "unresolved"
    assert "NOTREAL" in identity.failure_reason


def test_opensky_cache_build_time_is_not_reported_as_source_date():
    resolver = AircraftIdentityResolver(
        catalog=_catalog(),
        faa_registry={"source": {}, "icao24_to_model_code": {}, "models": {},
                      "icao24_documented_typecode": {}},
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
            "icao24_documented_typecode": {},
            "models": {"UNKNOWN": {"manufacturer": "UNKNOWN", "model": "MODEL"}},
        },
        opensky_lookup={"icao24_to_typecode": {"ABC123": "B38M"}},
    )

    identity = resolver.resolve(declared_type="UNK", icao24="abc123")

    assert identity.typecode == "B38M"
    assert identity.identity_source == "faa_registry"
    assert identity.typecode_source == "opensky"
    assert identity.registration == "N123ZZ"


def _write_identity_sources(tmp_path, *, faa_built_against="a" * 64):
    paths = {name: tmp_path / f"{name}.json" for name in ("icao", "faa", "opensky")}
    paths["icao"].write_text(
        json.dumps(
            {
                "schema_version": ICAO_CATALOG_SCHEMA,
                "source": {"last_updated": "10 July 2026", "raw_sha256": "a" * 64},
                "records": ICAO_RECORDS,
            }
        ),
        encoding="utf-8",
    )
    paths["faa"].write_text(
        json.dumps(
            {
                "schema_version": FAA_IDENTITY_SCHEMA,
                "source": {"effective_date": "2026-07-27", "icao_raw_sha256": faa_built_against},
                "icao24_to_model_code": {},
                "icao24_documented_typecode": {},
                "models": {},
            }
        ),
        encoding="utf-8",
    )
    paths["opensky"].write_text(
        json.dumps({"schema_version": OPENSKY_LOOKUP_SCHEMA, "icao24_to_typecode": {}}),
        encoding="utf-8",
    )
    return {"icao_path": paths["icao"], "faa_path": paths["faa"], "opensky_path": paths["opensky"]}


def test_identity_resolver_loads_versioned_json_sources(tmp_path):
    resolver = AircraftIdentityResolver.from_paths(**_write_identity_sources(tmp_path))

    assert resolver.catalog.normalize_typecode("A21N") == "A21N"


def test_identity_resolver_refuses_an_faa_identity_built_against_another_icao_snapshot(tmp_path):
    with pytest.raises(ValueError, match="built against the Doc 8643 snapshot b{64}"):
        AircraftIdentityResolver.from_paths(
            **_write_identity_sources(tmp_path, faa_built_against="b" * 64)
        )


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
        documented_crosswalk={},
        documented_crosswalk_sha256=None,
    )

    assert payload["models"]["DIRECT"]["typecode"] == "B38M"
    assert payload["models"]["DIRECT"]["typecode_method"] == "icao_exact_model"
    assert payload["models"]["CROSS"]["typecode"] == "B739"
    assert payload["models"]["CROSS"]["typecode_method"] == REGISTRATION_CROSSWALK_METHOD
    assert payload["models"]["CROSS"]["confidence"] == "medium"
    assert "typecode" not in payload["models"]["AMBIG"]
    assert payload["icao24_to_registration"]["A00001"] == "N1AA"


DOCUMENTED_ICAO_RECORDS = ICAO_RECORDS + [
    {"manufacturer": "BOMBARDIER", "model": "BD-100 Challenger 300", "typecode": "CL30"},
    {"manufacturer": "BOMBARDIER", "model": "BD-100 Challenger 350", "typecode": "CL35"},
    {"manufacturer": "CESSNA", "model": "700 Citation Longitude", "typecode": "C700"},
    {"manufacturer": "CESSNA", "model": "525 CitationJet", "typecode": "C525"},
    {"manufacturer": "CESSNA", "model": "525A Citation CJ2", "typecode": "C25A"},
    {"manufacturer": "BEECH", "model": "300 (B300) Super King Air 350", "typecode": "B350"},
    {"manufacturer": "BEECH", "model": "300 Super King Air", "typecode": "BE30"},
]

DOCUMENTED_ROWS = [
    {"faa_records": [["TEXTRON AVIATION INC", "700"]], "decision": "typecode",
     "typecode": "C700", "icao_records": [["CESSNA", "700 Citation Longitude"]], "evidence": []},
    {"faa_records": [["TEXTRON AVIATION INC", "B300"], ["RAYTHEON AIRCRAFT COMPANY", "B300"]],
     "decision": "typecode", "typecode": "B350",
     "icao_records": [["BEECH", "300 (B300) Super King Air 350"]], "evidence": []},
    {"faa_records": [["BOMBARDIER INC", "BD-100-1A10"]], "decision": "serial_split",
     "serial_pattern": "([0-9]{5})",
     "serial_split": [
         {"first": 20002, "last": 20500, "except": [], "typecode": "CL30",
          "icao_record": ["BOMBARDIER", "BD-100 Challenger 300"]},
         {"first": 20501, "last": None, "except": [20777], "typecode": "CL35",
          "icao_record": ["BOMBARDIER", "BD-100 Challenger 350"]}],
     "evidence": []},
    {"faa_records": [["CESSNA", "525"]], "decision": "unresolved",
     "reason": "Model 525 spans C525 and C25M", "evidence": []},
]


def _documented(tmp_path, rows=DOCUMENTED_ROWS):
    path = tmp_path / "crosswalk.json"
    path.write_text(json.dumps({"schema_version": CROSSWALK_SCHEMA, "models": rows}), encoding="utf-8")
    catalog = IcaoTypeDesignatorCatalog(DOCUMENTED_ICAO_RECORDS, source={"last_updated": "10 July 2026"})
    return catalog, *load_documented_crosswalk(path, catalog)


def test_documented_crosswalk_overrides_matcher_and_registration_evidence(tmp_path):
    """A TCDS/JO row wins on EVERY registry spelling it lists (RAYTHEON B300 had a BE30
    registration vote); a serial split types each aircraft by its serial; a row an authority
    marks ambiguous stays unresolved even where the OpenSky registration history votes (the
    2026-09-23 fix of CESSNA 525 -> C25A, the Model 525A)."""
    catalog, documented, digest = _documented(tmp_path)
    master = [
        {"N-NUMBER": "700A", "MFR MDL CODE": "T700", "MODE S CODE HEX": "A00001", "SERIAL NUMBER": "700-0001"},
        {"N-NUMBER": "300A", "MFR MDL CODE": "BD100", "MODE S CODE HEX": "A00002", "SERIAL NUMBER": "20400"},
        {"N-NUMBER": "350A", "MFR MDL CODE": "BD100", "MODE S CODE HEX": "A00003", "SERIAL NUMBER": "20600"},
        {"N-NUMBER": "350B", "MFR MDL CODE": "BD100", "MODE S CODE HEX": "A00004", "SERIAL NUMBER": "20777"},
        {"N-NUMBER": "350C", "MFR MDL CODE": "BD100", "MODE S CODE HEX": "A00005", "SERIAL NUMBER": "S-20800"},
        {"N-NUMBER": "350D", "MFR MDL CODE": "BD100", "MODE S CODE HEX": "A00008", "SERIAL NUMBER": "２０６００"},
        {"N-NUMBER": "525A", "MFR MDL CODE": "C525", "MODE S CODE HEX": "A00006", "SERIAL NUMBER": "525-0001"},
        {"N-NUMBER": "525B", "MFR MDL CODE": "C525", "MODE S CODE HEX": "A00007", "SERIAL NUMBER": "525-0002"},
        {"N-NUMBER": "350E", "MFR MDL CODE": "RB300", "MODE S CODE HEX": "A00009", "SERIAL NUMBER": "FL-1"},
        {"N-NUMBER": "350F", "MFR MDL CODE": "RB300", "MODE S CODE HEX": "A0000A", "SERIAL NUMBER": "FL-2"},
    ]
    payload = build_faa_payload(
        master_rows=master,
        reference_rows=[
            {"CODE": "T700", "MFR": "TEXTRON AVIATION INC", "MODEL": "700"},
            {"CODE": "BD100", "MFR": "BOMBARDIER INC", "MODEL": "BD-100-1A10"},
            {"CODE": "C525", "MFR": "CESSNA", "MODEL": "525"},
            {"CODE": "RB300", "MFR": "RAYTHEON AIRCRAFT COMPANY", "MODEL": "B300"},
        ],
        opensky_rows=[
            {"registration": "N525A", "typecode": "C25A"}, {"registration": "N525B", "typecode": "C25A"},
            {"registration": "N350E", "typecode": "BE30"}, {"registration": "N350F", "typecode": "BE30"},
        ],
        catalog=catalog,
        source={},
        documented_crosswalk=documented,
        documented_crosswalk_sha256=digest,
    )
    assert payload["schema_version"] == FAA_IDENTITY_SCHEMA
    assert payload["models"]["T700"]["typecode"] == "C700"
    assert payload["models"]["T700"]["typecode_method"] == DOCUMENTED_METHOD
    assert payload["models"]["RB300"]["typecode"] == "B350"
    assert "typecode" not in payload["models"]["BD100"]
    # 20777 is an exception; "S-20800" and the full-width "２０６００" do not have the
    # documented form: none is guessed.
    assert payload["icao24_documented_typecode"] == {"A00002": "CL30", "A00003": "CL35"}
    assert payload["counts"]["serial_split_resolved_icao24_records"] == 2
    assert "typecode" not in payload["models"]["C525"]
    assert payload["models"]["C525"]["typecode_method"] == DOCUMENTED_UNRESOLVED_METHOD

    resolver = AircraftIdentityResolver(
        catalog=catalog, faa_registry=payload,
        opensky_lookup={"icao24_to_typecode": {"A00004": "CL35", "A00006": "C25A"}},
    )
    assert resolver.resolve(declared_type=None, icao24="a00002").typecode == "CL30"
    challenger = resolver.resolve(declared_type=None, icao24="a00003")
    assert (challenger.typecode, challenger.typecode_source) == ("CL35", "faa_registry+faa_documents+icao_doc8643")
    for icao24 in ("a00004", "a00006"):      # OpenSky has a value for both; neither is used
        identity = resolver.resolve(declared_type=None, icao24=icao24)
        assert identity.typecode is None and identity.identity_source == "faa_registry"
    assert "serial number not covered" in resolver.resolve(declared_type=None, icao24="a00004").failure_reason
    assert "C525 and C25M" in resolver.resolve(declared_type=None, icao24="a00006").failure_reason


def test_documented_crosswalk_refuses_a_row_its_cited_icao_record_contradicts(tmp_path):
    rows = [{**DOCUMENTED_ROWS[0], "typecode": "C525"}]
    with pytest.raises(ValueError, match="Doc 8643 record"):
        _documented(tmp_path, rows)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"serial_pattern": "[0-9]{5}"}, "exactly the serial number"),
        ({"serial_split": [{**DOCUMENTED_ROWS[2]["serial_split"][0], "last": 20000}]}, "is empty"),
        ({"serial_split": [{**DOCUMENTED_ROWS[2]["serial_split"][0], "last": 20003,
                            "except": [20002, 20003]}]}, "is empty"),
        ({"serial_split": [DOCUMENTED_ROWS[2]["serial_split"][0],
                           {**DOCUMENTED_ROWS[2]["serial_split"][1], "first": 20500}]}, "overlap"),
    ],
)
def test_documented_crosswalk_refuses_a_malformed_serial_split(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        _documented(tmp_path, [{**DOCUMENTED_ROWS[2], **change}])


def test_documented_crosswalk_refuses_a_spelling_listed_twice(tmp_path):
    rows = [DOCUMENTED_ROWS[0], {**DOCUMENTED_ROWS[1], "faa_records": [["TEXTRON AVIATION INC", "700"]]}]
    with pytest.raises(ValueError, match="listed twice"):
        _documented(tmp_path, rows)


def test_shipped_identity_was_built_from_the_shipped_crosswalk_and_catalog():
    """Editing the crosswalk (or the catalog) without rebuilding the FAA identity would leave
    the resolver on stale rows; the shipped crosswalk must also still validate."""
    catalog = IcaoTypeDesignatorCatalog.from_json(ICAO_CATALOG_PATH)
    load_documented_crosswalk(DEFAULT_CROSSWALK, catalog)
    registry = get_default_identity_resolver().faa_registry
    assert registry["crosswalk_policy"]["documented_crosswalk"]["sha256"] == sha256_file(DEFAULT_CROSSWALK)
    assert registry["source"]["icao_raw_sha256"] == catalog.source["raw_sha256"]
