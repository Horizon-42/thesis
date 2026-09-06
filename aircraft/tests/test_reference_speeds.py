"""The published reference-speed table: its contract, its math, and the packaged data."""

from __future__ import annotations

import json
import math

import pytest

from aircraft.reference_speeds import (
    REFERENCE_SPEEDS_SCHEMA,
    ReferenceSpeed,
    load_reference_speeds,
    reference_speed,
)

_SOURCES = {"faa": {"title": "t"}, "oem": {"title": "t"}}
_ROW = {
    "approach_speed_kt": 144, "approach_speed_min_kt": 140, "approach_speed_max_kt": 144,
    "approach_speed_source": "faa", "malw_kg": 66350, "malw_source": "faa",
    "min_mass_kg": 41412, "min_mass_kind": "OEW", "min_mass_source": "oem",
}


def _write(tmp_path, types, sources=_SOURCES, schema=REFERENCE_SPEEDS_SCHEMA):
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"schema": schema, "sources": sources, "types": types}))
    return path


def test_the_published_pair_scales_with_the_square_root_of_mass():
    entry = ReferenceSpeed("B738", 144.0, 140.0, 144.0, 66350.0, 41412.0, "OEW", "faa", "faa", "oem")
    assert entry.vref_kt(66350.0, edge="low") == pytest.approx(140.0)
    assert entry.vref_kt(66350.0, edge="high") == pytest.approx(144.0)
    # Empty weight: sqrt(41412/66350) = 0.790 -> 110.6 kt on the lower edge.
    assert entry.vref_kt(41412.0, edge="low") == pytest.approx(140.0 * math.sqrt(41412.0 / 66350.0))
    with pytest.raises(ValueError, match="mass"):
        entry.vref_kt(0.0, edge="low")


def test_a_table_row_is_validated_and_its_sources_must_be_listed(tmp_path):
    table = load_reference_speeds(_write(tmp_path, {"b738": _ROW}))
    assert table["B738"].min_mass_kind == "OEW" and table["B738"].approach_speed_min_kt == 140.0
    for broken, message in [
        ({**_ROW, "approach_speed_min_kt": 150}, "min <= main <= max"),
        ({**_ROW, "min_mass_kg": 70000}, "below malw_kg"),
        ({**_ROW, "malw_source": "nowhere"}, "not a listed source"),
        ({**_ROW, "min_mass_source": None}, "min_mass_kind and min_mass_source"),
        ({**_ROW, "approach_speed_kt": True}, "must be a number"),
        ({k: v for k, v in _ROW.items() if k != "min_mass_kg"}, "without min_mass_kg"),
        ({**_ROW, "min_mass_kgs": 41412}, "unknown keys"),
        ({**_ROW, "corroboration": [{"source": "nowhere", "vat_kt": 147}]}, "names no listed source"),
    ]:
        with pytest.raises(ValueError, match=message):
            load_reference_speeds(_write(tmp_path, {"B738": broken}))
    with pytest.raises(ValueError, match="expected schema"):
        load_reference_speeds(_write(tmp_path, {"B738": _ROW}, schema="other"))


def test_a_type_without_a_published_minimum_mass_is_allowed_and_says_so(tmp_path):
    row = {k: v for k, v in _ROW.items() if not k.startswith("min_mass")}
    row["corroboration"] = [{"source": "oem", "note": "listed source ids are accepted"}]
    entry = load_reference_speeds(_write(tmp_path, {"C56X": row}))["C56X"]
    assert entry.min_mass_kg is None and entry.min_mass_kind is None and entry.min_mass_source is None


def test_the_packaged_table_covers_the_fleet_and_pins_verified_values():
    """Spot checks read directly off the FAA Aircraft Characteristics Database
    (October 2024) rows and the manufacturer documents (docs/reference_speeds)."""
    b738 = reference_speed("b738")
    assert b738 is not None
    assert (b738.approach_speed_kt, b738.approach_speed_min_kt, b738.approach_speed_max_kt) == (144.0, 140.0, 144.0)
    assert b738.malw_kg == pytest.approx(146_275 * 0.45359237, abs=1.0)
    assert b738.min_mass_kg == pytest.approx(41_412.0) and b738.min_mass_kind == "OEW"
    a320 = reference_speed("A320")
    assert a320.approach_speed_kt == 136.0 and a320.malw_kg == pytest.approx(66_000.0, abs=1.0)
    e75l = reference_speed("E75L")
    assert e75l.approach_speed_kt == 126.0 and e75l.malw_kg == pytest.approx(34_000.0, abs=1.0)
    assert e75l.min_mass_kg == pytest.approx(21_500.0)
    crj9 = reference_speed("CRJ9")
    assert (crj9.approach_speed_min_kt, crj9.approach_speed_max_kt) == (132.0, 141.0)
    assert crj9.min_mass_kg == pytest.approx(20_412.0) and crj9.min_mass_kind == "MFW"
    for code in ("B38M", "B737", "B739", "A319", "A321", "A21N", "B39M", "A20N", "C56X", "B763", "B752"):
        assert reference_speed(code) is not None, code
    assert reference_speed("ZZZZ") is None
