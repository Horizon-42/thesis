from __future__ import annotations

from pathlib import Path

import pytest

from preprocess_procedures import (
    FixRecord,
    ProcedureLeg,
    build_route_points,
    decode_cifp_coordinate,
    generate_procedure_geojson,
    generate_procedures_geojson,
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
    assert route_features[0]["properties"]["runwayIdent"] == "RW05L"
    assert route_features[0]["properties"]["procedureFamily"] == "RNAV_GPS"
    assert route_features[0]["properties"]["branchType"] == "final"
    assert route_features[0]["properties"]["legCoverage"]["skippedLegTypes"] == ["CA", "DF", "HM"]
    assert route_features[0]["properties"]["warnings"]


def test_generate_multi_runway_rnav_geojson() -> None:
    collection = generate_procedures_geojson(
        cifp_root=CIFP_ROOT,
        airport="KRDU",
        procedure_type="SIAP",
        procedures=["R05LY", "R05RY", "R23LY", "R23RY", "R32"],
        include_transitions=False,
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
    route_ids = [feature["properties"]["routeId"] for feature in route_features]
    runways = {feature["properties"]["runwayIdent"] for feature in route_features}

    assert len(route_features) == 5
    assert route_ids == [
        "KRDU-R05LY-R",
        "KRDU-R05RY-R",
        "KRDU-R23LY-R",
        "KRDU-R23RY-R",
        "KRDU-R32-R",
    ]
    assert runways == {"RW05L", "RW05R", "RW23L", "RW23R", "RW32"}
    assert collection["metadata"]["procedureFamilies"] == ["RNAV_GPS"]


def test_generate_transitions_hidden_by_default() -> None:
    collection = generate_procedures_geojson(
        cifp_root=CIFP_ROOT,
        airport="KRDU",
        procedure_type="SIAP",
        procedures=["R05LY"],
        include_transitions=True,
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
    by_branch = {feature["properties"]["branchIdent"]: feature for feature in route_features}

    assert list(by_branch) == ["R", "ACHWDR", "AOTTOS"]
    assert by_branch["R"]["properties"]["defaultVisible"] is True
    assert by_branch["ACHWDR"]["properties"]["defaultVisible"] is False
    assert by_branch["AOTTOS"]["properties"]["branchType"] == "transition"


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
