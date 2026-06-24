from __future__ import annotations

import json
from pathlib import Path

import pytest

import cifp_parser
import preprocess_procedures as procedures_module
from preprocess_procedures import (
    FixRecord,
    ProcedureLeg,
    build_fix_index,
    build_route_points,
    build_branch_document,
    build_procedure_detail_document,
    decode_cifp_coordinate,
    discover_rnav_procedures,
    generate_procedure_geojson,
    generate_procedures_geojson,
    infer_chart_targets,
    parse_leg_altitude_ft,
    parse_leg_altitude_qualifier,
    parse_leg_course_deg,
    parse_path_point_vertical_metadata,
    parse_procedure_legs,
    procedure_detail_documents_to_geojson,
    publish_local_chart_manifest,
    publish_procedure_details_assets,
    procedure_exists,
    sanitize_public_chart_filename,
)


CIFP_ROOT = Path(__file__).parents[3] / "data" / "CIFP" / "CIFP_260319"


def test_decode_cifp_coordinate() -> None:
    assert decode_cifp_coordinate("N35483156") == pytest.approx(35.8087667)
    assert decode_cifp_coordinate("W078525864") == pytest.approx(-78.8829556)
    assert decode_cifp_coordinate("N3552280160") == pytest.approx(35.8744489)
    assert decode_cifp_coordinate("W07848070690") == pytest.approx(-78.8019630)


def test_parse_leg_altitude_ignores_transition_altitude_for_unconstrained_if() -> None:
    # KRDU R32 CONCA is an initial fix with NO altitude constraint; the
    # "18000" sits in the Transition Altitude field (cols 95-99), a
    # procedure-wide constant.  It must NOT be read as a crossing altitude.
    # Regression for docs/33-cifp-transition-altitude-misparse-postmortem.md.
    line = (
        "SUSAP KRDUK7FR32   ACONCA 010CONCAK7PC0E  A    IF"
        "                                             18000                 A JS   008442407"
    )

    assert parse_leg_altitude_ft(line) is None
    assert parse_leg_altitude_qualifier(line) is None


def test_parse_leg_altitude_ignores_transition_altitude_with_speed_field() -> None:
    # Same misparse, with a speed limit packed after the transition
    # altitude ("18000210" = transition altitude 18000 + speed 210).
    line = (
        "SUSAP KRDUK7FR23LY ADUWON 010DUWONK7PC0E  A    IF"
        "                                             18000210              A-JS   008112102"
    )

    assert parse_leg_altitude_ft(line) is None


def test_parse_leg_altitude_prefers_signed_constraint_over_hold_course() -> None:
    line = (
        "SUSAP KRDUK7FR32   R      070DUHAMK7EA0EE  R   HM"
        "                     17290040    + 02200                           A JS   008562407"
    )

    assert parse_leg_altitude_ft(line) == 2200
    assert parse_leg_altitude_qualifier(line) == "atOrAbove"


def test_parse_leg_altitude_reads_at_or_above_crossing_constraint() -> None:
    # KRDU R32 NOSIC has a real "at or above 3400" constraint (Alt1) plus
    # the 18000 transition altitude; the real constraint must win and the
    # qualifier must reflect the "+" descriptor.
    line = (
        "SUSAP KRDUK7FR32   ANOSIC 010NOSICK7PC0EE AR   HF"
        "                     32420040    + 03400     18000                 A JS   008462407"
    )

    assert parse_leg_altitude_ft(line) == 3400
    assert parse_leg_altitude_qualifier(line) == "atOrAbove"


def test_block_altitude_descriptor_keeps_both_window_bounds() -> None:
    """A block ("B") altitude descriptor is a WINDOW (at or below Altitude 1,
    at or above Altitude 2). KSTL H30RZ FDRKO codes "B 07000 06000"; the parser
    must keep BOTH bounds so the 6000-7000 ft block is not silently collapsed to
    a single 7000 ft value (an audit gap fixed alongside the front/back
    canonical altitude consolidation)."""
    document = build_procedure_detail_document(
        cifp_root=CIFP_ROOT,
        airport="KSTL",
        procedure_type="SIAP",
        procedure="H30RZ",
        nominal_speed_kt=140.0,
        tunnel_half_width_nm=0.3,
        tunnel_half_height_ft=300.0,
        sample_spacing_m=250.0,
    )

    block_altitudes = {
        leg["path"]["endFixRef"]: leg["constraints"]["altitude"]
        for branch in document["branches"]
        for leg in branch["legs"]
        if (leg["constraints"]["altitude"] or {}).get("qualifier") == "block"
    }
    # FDRKO: "B 07000 06000" -> at or below 7000, at or above 6000.
    assert block_altitudes["fix:FDRKO"] == {
        "qualifier": "block",
        "valueFt": 7000,
        "rawText": "6000-7000 ft",
    }
    # TEEBE: "B 09000 08000".
    assert block_altitudes["fix:TEEBE"] == {
        "qualifier": "block",
        "valueFt": 9000,
        "rawText": "8000-9000 ft",
    }


def test_parse_leg_course_reads_ca_course_field() -> None:
    line = (
        "SUSAP 00R K4FR30   R      040         0  M     CA"
        "                     3050        + 01000                           B PS   785032404"
    )

    assert parse_leg_course_deg(line) == pytest.approx(305.0)


def test_krdu_r05ly_exists_in_cifp_index() -> None:
    assert procedure_exists(CIFP_ROOT / "IN_CIFP.txt", "KRDU", "SIAP", "R05LY")


def test_discover_rnav_procedures_is_airport_specific() -> None:
    assert discover_rnav_procedures(CIFP_ROOT / "IN_CIFP.txt", "KRDU", "SIAP") == [
        "H05LZ",
        "R05LY",
        "H05RZ",
        "R05RY",
        "H23LZ",
        "R23LY",
        "H23RZ",
        "R23RY",
        "R32",
    ]
    assert discover_rnav_procedures(CIFP_ROOT / "IN_CIFP.txt", "KAAA", "SIAP") == [
        "R03",
        "R21",
    ]
    assert discover_rnav_procedures(CIFP_ROOT / "IN_CIFP.txt", "KAFN", "SIAP") == [
        "RNV-B",
        "RNV-C",
    ]


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


def test_geojson_projection_uses_procedure_detail_document_as_source() -> None:
    document = build_procedure_detail_document(
        cifp_root=CIFP_ROOT,
        airport="KRDU",
        procedure_type="SIAP",
        procedure="R32",
        nominal_speed_kt=140.0,
        tunnel_half_width_nm=0.3,
        tunnel_half_height_ft=300.0,
        sample_spacing_m=250.0,
    )

    collection = procedure_detail_documents_to_geojson(
        airport="KRDU",
        procedure_type="SIAP",
        source_cycle="2603",
        documents=[document],
        include_transitions=False,
        branch="R",
    )

    route_feature = next(
        feature
        for feature in collection["features"]
        if feature["properties"]["featureType"] == "procedure-route"
    )
    fix_features = [
        feature
        for feature in collection["features"]
        if feature["properties"]["featureType"] == "procedure-fix"
    ]
    fix_by_id = {fix["fixId"]: fix for fix in document["fixes"]}
    final_branch = next(branch for branch in document["branches"] if branch["branchIdent"] == "R")
    rendered_legs = [
        leg
        for leg in final_branch["legs"]
        if leg["quality"].get("renderedInPlanView") is True
    ]

    assert collection["metadata"]["canonicalDataLayer"] == "procedure-details"
    assert route_feature["properties"]["source"] == "procedure-details"
    assert route_feature["properties"]["samples"] == [
        {
            "sequence": leg["sequence"],
            "fixIdent": fix_by_id[leg["path"]["endFixRef"]]["ident"],
            "legType": leg["path"]["pathTerminator"],
            "role": leg["roleAtEnd"],
            "altitudeFt": (
                None
                if leg["constraints"]["altitude"] is None
                else leg["constraints"]["altitude"]["valueFt"]
            ),
            "geometryAltitudeFt": leg["constraints"]["geometryAltitudeFt"],
            "distanceFromStartM": sample["distanceFromStartM"],
            "timeSeconds": sample["timeSeconds"],
            "sourceLine": leg["quality"]["sourceLine"],
        }
        for leg, sample in zip(rendered_legs, route_feature["properties"]["samples"])
    ]
    assert [
        [fix_by_id[leg["path"]["endFixRef"]]["position"]["lon"], fix_by_id[leg["path"]["endFixRef"]]["position"]["lat"]]
        for leg in rendered_legs
    ] == [[feature["geometry"]["coordinates"][0], feature["geometry"]["coordinates"][1]] for feature in fix_features]


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
    by_branch = {feature["properties"]["branchKey"]: feature for feature in route_features}

    assert list(by_branch) == ["R", "ACHWDR", "AOTTOS"]
    assert by_branch["ACHWDR"]["properties"]["branchIdent"] == "CHWDR"
    assert by_branch["ACHWDR"]["properties"]["branchProcedureType"] == "A"
    assert by_branch["ACHWDR"]["properties"]["branchTransitionIdent"] == "CHWDR"
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


def test_branch_document_exports_rf_path_metadata() -> None:
    legs = [
        ProcedureLeg(
            sequence=10,
            branch="R",
            fix_ident="START",
            leg_type="IF",
            role="IF",
            altitude_ft=3000,
            source_line=1,
        ),
        ProcedureLeg(
            sequence=20,
            branch="R",
            fix_ident="END",
            leg_type="RF",
            role="FAF",
            altitude_ft=2200,
            source_line=2,
            turn_direction="LEFT",
            arc_radius_nm=2.0,
            center_lat_deg=35.82,
            center_lon_deg=-78.86,
        ),
    ]
    fixes = {
        "START": FixRecord("START", -78.9, 35.8, 3000, 1),
        "END": FixRecord("END", -78.84, 35.84, 2200, 2),
    }

    branch = build_branch_document(
        branch_ident="R",
        branch_order=1,
        final_branch_ident="R",
        branch_legs=legs,
        branch_route_points=[],
        fix_records=fixes,
        branch_warnings=[],
        final_fix_idents={"START", "END"},
    )

    rf_path = branch["legs"][1]["path"]
    assert rf_path["pathTerminator"] == "RF"
    assert rf_path["turnDirection"] == "LEFT"
    assert rf_path["arcRadiusNm"] == 2.0
    assert rf_path["centerLatDeg"] == 35.82
    assert rf_path["centerLonDeg"] == -78.86


def test_branch_document_exports_course_path_metadata() -> None:
    legs = [
        ProcedureLeg(
            sequence=10,
            branch="R",
            fix_ident="RW",
            leg_type="CA",
            role="Route",
            altitude_ft=1000,
            source_line=1,
            course_deg=305.0,
        ),
    ]
    fixes = {
        "RW": FixRecord("RW", -78.9, 35.8, 800, 1),
    }

    branch = build_branch_document(
        branch_ident="R",
        branch_order=1,
        final_branch_ident="R",
        branch_legs=legs,
        branch_route_points=[],
        fix_records=fixes,
        branch_warnings=[],
        final_fix_idents={"RW"},
    )

    assert branch["legs"][0]["path"]["pathTerminator"] == "CA"
    assert branch["legs"][0]["path"]["courseDeg"] == 305.0


def test_cifparse_rf_leg_metadata_resolves_center_fix() -> None:
    legs = parse_procedure_legs(
        CIFP_ROOT / "FAACIFP18",
        airport="KATL",
        procedure="ZELAN4",
        branch="4RW27R",
    )
    rf_leg = next(leg for leg in legs if leg.leg_type == "RF")

    assert rf_leg.fix_ident == "MPASS"
    assert rf_leg.turn_direction == "RIGHT"
    assert rf_leg.arc_radius_nm == pytest.approx(3.46)
    assert rf_leg.center_fix_ident == "CFZJF"
    assert rf_leg.center_fix_region_code == "K7"

    fix_records = build_fix_index(CIFP_ROOT / "FAACIFP18", "KATL", legs)
    assert "CFZJF" in fix_records

    branch = build_branch_document(
        branch_ident="4RW27R",
        branch_order=1,
        final_branch_ident="4RW27R",
        branch_legs=legs,
        branch_route_points=[],
        fix_records=fix_records,
        branch_warnings=[],
        final_fix_idents={leg.fix_ident for leg in legs},
    )
    rf_path = next(leg["path"] for leg in branch["legs"] if leg["path"]["pathTerminator"] == "RF")

    assert rf_path["turnDirection"] == "RIGHT"
    assert rf_path["arcRadiusNm"] == pytest.approx(3.46)
    assert rf_path["centerFixRef"] == "fix:CFZJF"
    assert rf_path["centerLatDeg"] == pytest.approx(fix_records["CFZJF"].lat)
    assert rf_path["centerLonDeg"] == pytest.approx(fix_records["CFZJF"].lon)


def test_cifparse_ca_leg_metadata_exports_course() -> None:
    legs = parse_procedure_legs(
        CIFP_ROOT / "FAACIFP18",
        airport="KRDU",
        procedure="R32",
        branch="R",
    )
    ca_leg = next(leg for leg in legs if leg.leg_type == "CA")

    assert ca_leg.course_deg == pytest.approx(324.2)


def test_cifparse_final_leg_metadata_exports_vertical_angle() -> None:
    legs = parse_procedure_legs(
        CIFP_ROOT / "FAACIFP18",
        airport="KRDU",
        procedure="R32",
        branch="R",
    )
    runway_leg = next(leg for leg in legs if leg.fix_ident == "RW32")

    assert runway_leg.vertical_angle_deg == pytest.approx(-3.5)


def test_path_point_metadata_exports_tch_and_glidepath_angle() -> None:
    metadata = parse_path_point_vertical_metadata(
        CIFP_ROOT / "FAACIFP18",
        airport="KRDU",
        procedure="R05LY",
        runway="RW05L",
    )

    assert metadata is not None
    assert metadata.glidepath_angle_deg == pytest.approx(3.0)
    assert metadata.threshold_crossing_height_ft == pytest.approx(57.4)
    assert metadata.source_line == 300869


def test_cifp_source_line_map_is_cached_per_process() -> None:
    cifp_parser.clear_cifp_parser_caches()
    faacifp_path = CIFP_ROOT / "FAACIFP18"

    first = cifp_parser.build_exact_source_line_map(faacifp_path)
    second = cifp_parser.build_exact_source_line_map(faacifp_path)

    assert first is second
    assert cifp_parser.build_exact_source_line_map.cache_info().hits == 1


def test_build_katl_zelan4_procedure_detail_document_exports_real_rf_metadata() -> None:
    document = build_procedure_detail_document(
        cifp_root=CIFP_ROOT,
        airport="KATL",
        procedure_type="SID",
        procedure="ZELAN4",
        nominal_speed_kt=180.0,
        tunnel_half_width_nm=0.3,
        tunnel_half_height_ft=300.0,
        sample_spacing_m=250.0,
    )

    rf_legs = [
        leg
        for branch in document["branches"]
        for leg in branch["legs"]
        if leg["path"]["pathTerminator"] == "RF"
    ]

    assert document["procedureUid"] == "KATL-ZELAN4-UNKNOWN"
    assert len(rf_legs) == 1

    rf_path = rf_legs[0]["path"]
    assert rf_path["startFixRef"] == "fix:CPARK"
    assert rf_path["endFixRef"] == "fix:MPASS"
    assert rf_path["turnDirection"] == "RIGHT"
    assert rf_path["arcRadiusNm"] == pytest.approx(3.46)
    assert rf_path["centerFixRef"] == "fix:CFZJF"
    assert rf_path["centerLatDeg"] == pytest.approx(33.692475)
    assert rf_path["centerLonDeg"] == pytest.approx(-84.5128917)

    fix_by_id = {fix["fixId"]: fix for fix in document["fixes"]}
    assert "fix:CFZJF" in fix_by_id
    assert fix_by_id["fix:CFZJF"]["position"] == {
        "lon": pytest.approx(-84.5128917),
        "lat": pytest.approx(33.692475),
    }


def test_build_krdu_r05ly_procedure_detail_document() -> None:
    document = build_procedure_detail_document(
        cifp_root=CIFP_ROOT,
        airport="KRDU",
        procedure_type="SIAP",
        procedure="R05LY",
        nominal_speed_kt=140.0,
        tunnel_half_width_nm=0.3,
        tunnel_half_height_ft=300.0,
        sample_spacing_m=250.0,
    )

    assert document["procedureUid"] == "KRDU-R05LY-RW05L"
    assert document["airport"]["faa"] == "RDU"
    assert document["procedure"]["chartName"] == "RNAV(GPS) Y RWY 05L"
    assert document["runway"]["ident"] == "RW05L"
    assert document["runway"]["threshold"]["elevationFt"] == 367
    assert len(document["branches"]) >= 3
    assert any(branch["branchRole"] == "final" for branch in document["branches"])
    assert document["verticalProfiles"][0]["glidepathAngleDeg"] == pytest.approx(3.0)
    assert document["verticalProfiles"][0]["thresholdCrossingHeightFt"] == pytest.approx(57.4)
    assert document["verticalProfiles"][0]["pathPointSourceLine"] == 300869
    assert document["validation"]["expectedFAF"] == "fix:WEPAS"
    assert document["displayHints"]["defaultVisibleBranchIds"] == ["branch:R"]


def test_krdu_r05ly_matches_published_rnav_gps_chart() -> None:
    """Golden cross-reference of the parser output for KRDU R05LY against the
    authoritative published FAA chart **RNAV (GPS) Y RWY 5L** (00516RY5L.pdf,
    Amdt 2, dated 25FEB21).

    Every asserted number is annotated with what the chart prints. Unlike the
    self-referential snapshot/structure tests, this guards the *parser*: a
    systematic misparse (wrong column offset, dropped altitude descriptor, a
    transition-altitude leak like docs/33-...) would change one of these values
    and fail here, even though the parser still "agrees with itself".
    """
    document = build_procedure_detail_document(
        cifp_root=CIFP_ROOT,
        airport="KRDU",
        procedure_type="SIAP",
        procedure="R05LY",
        nominal_speed_kt=140.0,
        tunnel_half_width_nm=0.3,
        tunnel_half_height_ft=300.0,
        sample_spacing_m=250.0,
    )

    branches = {branch["branchIdent"]: branch for branch in document["branches"]}

    def leg_to(branch_ident: str, fix_ident: str) -> dict:
        for leg in branches[branch_ident]["legs"]:
            if leg["path"]["endFixRef"] == f"fix:{fix_ident}":
                return leg
        raise AssertionError(f"{fix_ident} not found on branch {branch_ident}")

    def crossing_ft(branch_ident: str, fix_ident: str) -> int:
        return leg_to(branch_ident, fix_ident)["constraints"]["altitude"]["valueFt"]

    # Header block --------------------------------------------------------
    assert document["procedure"]["chartName"] == "RNAV(GPS) Y RWY 05L"
    assert document["procedure"]["procedureFamily"] == "RNAV_GPS"
    # Chart: TDZE 384, threshold (LDA side) charted elevation 367.
    assert document["runway"]["threshold"]["elevationFt"] == 367

    # Final approach (branch R): SCHOO (IF) -> WEPAS (FAF) -> RW05L (MAPt) ---
    final_roles = [leg["roleAtEnd"] for leg in branches["R"]["legs"][:3]]
    assert final_roles == ["IF", "FAF", "MAPt"]
    # Chart profile crossing altitudes:
    assert crossing_ft("R", "SCHOO") == 3000   # chart: SCHOO 3000 (IF)
    assert crossing_ft("R", "WEPAS") == 2200    # chart: WEPAS 2200 (FAF)
    # Chart glidepath crosses the threshold at TCH 57 ft above the 367 ft
    # threshold -> 424 ft MSL (367 + 57).
    assert crossing_ft("R", "RW05L") == 424
    assert leg_to("R", "WEPAS")["constraints"]["altitude"]["qualifier"] == "atOrAbove"

    # Coded final glidepath -- chart: GP 3.00 deg, TCH 57.
    vertical = document["verticalProfiles"][0]
    assert vertical["glidepathAngleDeg"] == pytest.approx(3.0)
    assert vertical["thresholdCrossingHeightFt"] == pytest.approx(57.4, abs=0.5)

    # CHWDR transition (chart: CHWDR 5000 -> BOULE 4000 -> SCHOO) -----------
    assert crossing_ft("CHWDR", "CHWDR") == 5000  # chart: CHWDR 5000
    assert crossing_ft("CHWDR", "BOULE") == 4000  # chart: BOULE 4000

    # OTTOS transition (chart: OTTOS IAF, 6000 holding depiction) -----------
    assert crossing_ft("OTTOS", "OTTOS") == 6000  # chart: OTTOS 6000 (2T0K hold)

    # Missed approach (chart: "Climb to 1000 then climbing left turn to 2200
    # direct DUHAM and hold") ----------------------------------------------
    missed = {leg["path"]["pathTerminator"]: leg for leg in branches["R"]["legs"][3:]}
    assert set(missed) == {"CA", "DF", "HM"}
    # CA climb course == the 054 deg final approach course on the chart.
    assert missed["CA"]["path"]["courseDeg"] == pytest.approx(54.0, abs=1.0)
    assert missed["DF"]["path"]["endFixRef"] == "fix:DUHAM"   # direct DUHAM
    assert missed["HM"]["roleAtEnd"] == "MAHF"                # and hold
    assert document["validation"]["expectedMissedHoldFix"] == "fix:DUHAM"


def test_chart_filename_inference_and_sanitizing() -> None:
    assert infer_chart_targets("00516RY5L.PDF#nameddest=(RDU).pdf") == ("R05LY", "RW05L")
    assert infer_chart_targets("00516RRZ23L.PDF#nameddest=(RDU).pdf") == ("H23LZ", "RW23L")
    assert infer_chart_targets("00516R32.PDF#nameddest=(RDU).pdf") == ("R32", "RW32")
    assert sanitize_public_chart_filename("00516RY5L.PDF#nameddest=(RDU).pdf") == (
        "00516RY5L.PDF-nameddest-RDU.pdf"
    )


def test_publish_procedure_details_assets_writes_manifest_and_chart_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail_root = tmp_path / "procedure-details"
    chart_root = tmp_path / "charts"
    source_chart_root = tmp_path / "chart-source" / "KRDU"
    source_chart_root.mkdir(parents=True)
    (source_chart_root / "00516RY5L.PDF#nameddest=(RDU).pdf").write_bytes(
        b"%PDF-1.4\n% fake chart for test\n"
    )

    monkeypatch.setattr(
        procedures_module,
        "airport_procedure_details_dir",
        lambda airport: detail_root,
    )
    monkeypatch.setattr(
        procedures_module,
        "airport_procedure_details_path",
        lambda airport, file_name: detail_root / file_name,
    )
    monkeypatch.setattr(
        procedures_module,
        "airport_charts_dir",
        lambda airport: chart_root,
    )
    monkeypatch.setattr(
        procedures_module,
        "airport_chart_path",
        lambda airport, file_name: chart_root / file_name,
    )

    result = publish_procedure_details_assets(
        cifp_root=CIFP_ROOT,
        airport="KRDU",
        procedure_type="SIAP",
        procedures=["R05LY"],
        nominal_speed_kt=140.0,
        tunnel_half_width_nm=0.3,
        tunnel_half_height_ft=300.0,
        sample_spacing_m=250.0,
        chart_root=tmp_path / "chart-source",
    )

    assert result["procedureCount"] == 1
    assert result["chartCount"] == 1
    assert result["procedureIndexPath"].exists()
    assert result["chartIndexPath"].exists()
    assert (detail_root / "KRDU-R05LY-RW05L.json").exists()

    index = json.loads(result["procedureIndexPath"].read_text(encoding="utf-8"))
    chart_index = json.loads(result["chartIndexPath"].read_text(encoding="utf-8"))

    assert index["airport"] == "KRDU"
    assert index["runways"][0]["runwayIdent"] == "RW05L"
    assert index["runways"][0]["procedures"][0]["procedureUid"] == "KRDU-R05LY-RW05L"
    assert chart_index["charts"][0]["procedureUid"] == "KRDU-R05LY-RW05L"
    assert chart_index["charts"][0]["url"].endswith("00516RY5L.PDF-nameddest-RDU.pdf")


def test_publish_chart_manifest_links_existing_public_rnp_chart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chart_dir = tmp_path / "public-charts"
    chart_dir.mkdir(parents=True)
    (chart_dir / "00516RRZ23L.PDF").write_bytes(b"%PDF-1.4\n% fake public RNP chart\n")

    monkeypatch.setattr(
        procedures_module,
        "airport_charts_dir",
        lambda airport: chart_dir,
    )
    monkeypatch.setattr(
        procedures_module,
        "airport_chart_path",
        lambda airport, file_name: chart_dir / file_name,
    )

    manifest = publish_local_chart_manifest(
        airport="KRDU",
        chart_root=tmp_path / "empty-source",
        documents=[
            {
                "procedureUid": "KRDU-H23LZ-RW23L",
                "procedure": {
                    "procedureIdent": "H23LZ",
                    "runwayIdent": "RW23L",
                    "chartName": "RNAV(RNP) Z RWY 23L",
                },
                "runway": {"ident": "RW23L"},
            }
        ],
    )

    assert len(manifest["charts"]) == 1
    chart = manifest["charts"][0]
    assert chart["procedureUid"] == "KRDU-H23LZ-RW23L"
    assert chart["procedureIdent"] == "H23LZ"
    assert chart["runwayIdent"] == "RW23L"
    assert chart["title"] == "RNAV(RNP) Z RWY 23L"
    assert chart["originalFileName"] == "00516RRZ23L.PDF"
    assert chart["url"] == "/data/airports/KRDU/charts/00516RRZ23L.pdf"
