"""HTML payload keeps stable identity and enforces its overlay cap."""

from __future__ import annotations

import pytest

from evaluation import AssessmentContext, record_from_dict
from evaluation.tests.test_evaluation import payload
from evaluation.visualize import _sample_evenly, build_payload, render_html


def contexts():
    context = AssessmentContext(
        benchmark="rnp_apch_lnav_vnav_baro", airport="KRDU", runway="05L",
        runway_course_deg=0.0, runway_width_m=45.72,
        runway_source="faa_nasr_apt_rwy", runway_source_cycle="2026-08-06",
        procedure_source="faa_terminal_procedure", procedure_source_cycle="2026-08-06",
        threshold_elevation_hae_m=130.0,
        threshold_elevation_msl_m=100.0,
        threshold_crossing_height_m=30.0,
        baro_vnav_approved=True,
    )
    return {("KRDU", "05L"): context}


def records_with_repeated_callsign():
    records = []
    for index in range(2):
        value = payload()
        value["source"]["flight_key"] = f"TEST1_05L_abc12{index}_20260812T000000Z"
        records.append(record_from_dict(value))
    return records


def test_overlay_labels_preserve_stable_flight_identity_for_repeated_callsigns():
    result = build_payload(records_with_repeated_callsign(), contexts=contexts())
    assert [track["id"] for track in result["tracks"]] == ["TEST1", "TEST1"]
    assert len({track["label"] for track in result["tracks"]}) == 2
    assert all(track["flight_key"] in track["label"] for track in result["tracks"])


def test_non_positive_overlay_caps_are_rejected():
    with pytest.raises(ValueError, match="greater than zero"):
        _sample_evenly([1, 2], 0)
    with pytest.raises(ValueError, match="greater than zero"):
        build_payload(records_with_repeated_callsign(), contexts=contexts(), max_tracks=-1)


def test_rendered_payload_is_strict_json_and_escapes_script_close():
    value = payload()
    value["source"]["id"] = "</script><img src=x>"
    result = build_payload([record_from_dict(value)], contexts=contexts())
    page = render_html(result, title="Test", source_label="batch")
    assert "const DATA=" in page
    assert "7.5 m half-FSD threshold bound" in page
    assert "remain indeterminate" not in page
    assert "missing LPV bound" not in page
    assert "</script><img" not in page
    assert "<\\/script><img" in page
