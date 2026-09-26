"""The performance index: one decision per type with no native model, validated at load."""

import json

import pytest

from aircraft.aircraft_sets import AIRCRAFT_PRESETS
from aircraft.performance_index import (
    PERFORMANCE_INDEX_PATH,
    PERFORMANCE_INDEX_SCHEMA,
    index_entry,
    load_performance_index,
    performance_index_identity,
)
from aircraft.query_aircraft_parameters import openap_support_kind
from aircraft.reference_speeds import reference_speed


def _shipped() -> dict:
    return json.loads(PERFORMANCE_INDEX_PATH.read_text(encoding="utf-8"))


def _write(tmp_path, types: dict, sources: dict | None = None):
    payload = {"schema": PERFORMANCE_INDEX_SCHEMA, "generated": "test", "decisions": "test",
               "sources": _shipped()["sources"] if sources is None else sources, "types": types}
    path = tmp_path / "performance_index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_shipped_index_loads_and_never_shadows_a_native_airframe():
    index = load_performance_index()
    for code, entry in index.items():
        assert code not in AIRCRAFT_PRESETS and openap_support_kind(code) != "direct", code
        if entry.decision == "own":
            assert entry.aircraft.code == code
            assert entry.aircraft.approach.speeds == reference_speed(code)
        if entry.decision == "substitute":
            assert entry.substitute in AIRCRAFT_PRESETS or openap_support_kind(entry.substitute) == "direct"


def test_own_parameters_come_from_the_cited_documents():
    # A220-300: FAA ACD MALW 133,600 lb; ACP wing area 112.3 m2; PW1524G-3 108.54 kN (engine TCDS).
    bcs3 = index_entry("BCS3")
    assert bcs3.decision == "own"
    assert bcs3.aircraft.landing_mass == 60600.0
    assert bcs3.aircraft.geometry.wing_area_m2 == 112.3
    assert bcs3.aircraft.engine.max_thrust_total_n == 2 * 108540.0
    assert "airbus_a220_acp_i013" in bcs3.reason


def test_a_substitute_names_the_airframe_it_is_flown_as_and_why():
    glf4 = index_entry("GLF4")
    assert glf4.decision == "substitute" and glf4.substitute == "CRJ9"
    assert "similarity distance" in glf4.reason
    # OpenAP's own synonyms become explicit substitutes (LJ45 is cached with GLF6's data).
    assert index_entry("LJ45").substitute == "GLF6"


def test_excluded_types_say_why():
    assert index_entry("PC12").decision == "exclude"
    assert "propeller" in index_entry("PC12").reason
    assert index_entry("ZZZZ") is None


@pytest.mark.parametrize("row, message", [
    ({"decision": "keep"}, "decision must be one of"),
    ({"decision": "substitute", "substitute": "BCS3", "basis": "x"}, "not a preset or OpenAP-direct"),
    ({"decision": "exclude"}, "an exclude row has keys"),
    ({"decision": "own", "name": "x", "mtow_kg": 1.0, "wing_area_m2": 1.0,          # E55P's published MALW is 7,568 kg
      "engines": 2, "max_thrust_n_each": 1.0,
      "sources": {"mtow": "faa_acd_2024_10", "wing_area": "faa_acd_2024_10", "thrust": "faa_acd_2024_10"}},
     "published MALW .* above mtow_kg"),
    ({"decision": "own", "name": "x", "mtow_kg": 10_000.0, "wing_area_m2": 1.0,
      "engines": 2, "max_thrust_n_each": 1.0,
      "sources": {"mtow": "nowhere", "wing_area": "faa_acd_2024_10", "thrust": "faa_acd_2024_10"}},
     "must name a listed source"),
    ({"decision": "own", "name": "x", "mtow_kg": 10_000.0, "mlw_kg": 7_000.0, "wing_area_m2": 1.0,
      "engines": 2, "max_thrust_n_each": 1.0,
      "sources": {"mtow": "faa_acd_2024_10", "wing_area": "faa_acd_2024_10", "thrust": "faa_acd_2024_10"}},
     "keys"),                                            # the landing mass is the published row's, never the index's
])
def test_a_malformed_row_is_refused(tmp_path, row, message):
    with pytest.raises(ValueError, match=message):
        load_performance_index(_write(tmp_path, {"E55P": row}))


def test_an_own_type_lands_at_its_published_malw_the_mass_its_approach_speed_is_scaled_from():
    from aircraft.reference_speeds import reference_speed

    for code in ("B722", "E135", "A30B", "BCS3"):
        entry = index_entry(code)
        assert entry.decision == "own"
        assert entry.aircraft.landing_mass == reference_speed(code).malw_kg
    assert index_entry("B722").aircraft.landing_mass == 68039.0     # FAA ACD MALW 150,000 lb (Poll-Schumann had 71.6 t)


def test_a_row_for_a_natively_flown_airframe_is_refused(tmp_path):
    with pytest.raises(ValueError, match="flown natively"):
        load_performance_index(_write(tmp_path, {"A320": {"decision": "exclude", "reason": "x"}}))


def test_every_openap_synonym_is_decided_by_the_index():
    # The model never flies a synonym's borrowed data under the synonym's code, so every
    # synonym needs a decision (the loader refuses a file that leaves one out).
    records = json.loads((PERFORMANCE_INDEX_PATH.parent / "openap_aircraft_parameters.json").read_text())
    synonyms = {code for code, record in records["typecodes"].items()
                if record.get("openap_supported")
                and (record["parameters"].get("openap_performance_typecode") or code) != code}
    assert len(synonyms) == 21
    assert all(index_entry(code) is not None for code in synonyms)
    assert index_entry("AT72").decision == "exclude"            # a propeller type
    assert index_entry("MD11").substitute == "B773"            # no flights; OpenAP's surrogate kept


def test_a_synonym_left_undecided_is_refused(tmp_path):
    types = {code: row for code, row in _shipped()["types"].items() if code != "MD11"}
    with pytest.raises(ValueError, match="no decision for the OpenAP synonyms"):
        load_performance_index(_write(tmp_path, types))


def test_the_identity_names_the_shipped_file():
    identity = performance_index_identity()
    assert identity["types"] == len(_shipped()["types"])
    assert len(identity["sha256"]) == 64

