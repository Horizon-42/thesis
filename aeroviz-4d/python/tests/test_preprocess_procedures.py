from __future__ import annotations

from pathlib import Path

import pytest

from preprocess_procedures import (
    FixRecord,
    ProcedureLeg,
    build_route_points,
    decode_cifp_coordinate,
    generate_procedure_geojson,
    procedure_exists,
)


CIFP_ROOT = Path(__file__).parents[3] / "data" / "CIFP" / "CIFP_260319"


def test_decode_cifp_coordinate() -> None:
    assert decode_cifp_coordinate("N35483156") == pytest.approx(35.8087667)
    assert decode_cifp_coordinate("W078525864") == pytest.approx(-78.8829556)
    assert decode_cifp_coordinate("N3552280160") == pytest.approx(35.8744489)
    assert decode_cifp_coordinate("W07848070690") == pytest.approx(-78.8019630)


def test_krdu_r05ly_exists_in_cifp_index() -> None:
    assert procedure_exists(CIFP_ROOT / "IN_CIFP.txt", "KRDU", "SIAP", "R05LY")


def test_generate_krdu_r05ly_geojson() -> None:
    collection = generate_procedure_geojson(
        cifp_root=CIFP_ROOT,
        airport="KRDU",
        procedure_type="SIAP",
        procedure="R05LY",
        branch="R",
        nominal_speed_kt=140.0,
        tunnel_half_width_nm=0.3,
        tunnel_half_height_ft=300.0,
        sample_spacing_m=250.0,
    )

    route_features = [
        feature
        for feature in collection["features"]
        if feature["properties"]["featureType"] == "procedure-route"
    ]
    fix_features = [
        feature
        for feature in collection["features"]
        if feature["properties"]["featureType"] == "procedure-fix"
    ]

    assert collection["metadata"]["sourceCycle"] == "2603"
    assert len(route_features) == 1
    assert len(fix_features) >= 3
    assert route_features[0]["geometry"]["coordinates"][0][0] == pytest.approx(-78.9264722)
    assert route_features[0]["properties"]["samples"][0]["fixIdent"] == "SCHOO"
    assert route_features[0]["properties"]["samples"][-1]["fixIdent"] == "RW05L"
    assert route_features[0]["properties"]["warnings"]


def test_unresolved_fix_is_warned_and_skipped() -> None:
    legs = [
        ProcedureLeg(
            sequence=10,
            branch="R",
            fix_ident="MISSING",
            leg_type="TF",
            role="IF",
            altitude_ft=3000,
            source_line=1,
        )
    ]
    fixes = {
        "KNOWN": FixRecord(
            ident="KNOWN",
            lon=-78.8,
            lat=35.8,
            altitude_ft=1000,
            source_line=2,
        )
    }

    route_points, warnings = build_route_points(legs, fixes, nominal_speed_kt=140.0)

    assert route_points == []
    assert any("unresolved fix MISSING" in warning for warning in warnings)
