"""HTML payload: one streaming path over a rostered batch, stable identity, overlay cap."""

from __future__ import annotations

import pytest

from evaluation.tests.factories import assessment_context, trajectory_payload, write_batch
from evaluation.visualize import _sample_evenly, build_payload, render_html


def contexts():
    return {
        ("KRDU", "05L"): assessment_context(
            benchmark="rnp_apch_lnav_vnav_baro"
        )
    }


def payloads_with_repeated_callsign() -> list[dict]:
    payloads = []
    for index in range(2):
        value = trajectory_payload()
        value["source"]["flight_key"] = f"TEST1_05L_abc12{index}_20260812T000000Z"
        payloads.append(value)
    return payloads


def test_overlay_labels_preserve_stable_flight_identity_for_repeated_callsigns(tmp_path):
    batch = write_batch(tmp_path, payloads_with_repeated_callsign())
    result = build_payload(batch, contexts=contexts())
    assert result["report"]["total"] == 2
    assert [track["id"] for track in result["tracks"]] == ["TEST1", "TEST1"]
    assert len({track["label"] for track in result["tracks"]}) == 2
    assert all(track["flight_key"] in track["label"] for track in result["tracks"])
    assert (result["tracksShown"], result["tracksTotal"]) == (2, 2)


def test_the_overlay_cap_samples_the_drawable_records_and_the_report_keeps_all(tmp_path):
    batch = write_batch(tmp_path, [trajectory_payload() for _ in range(5)])
    result = build_payload(batch, contexts=contexts(), max_tracks=2)
    assert result["report"]["total"] == 5
    assert (result["tracksShown"], result["tracksTotal"]) == (2, 5)


def test_an_unsolved_record_is_reported_but_not_drawn(tmp_path):
    unsolved = trajectory_payload()
    unsolved.update(final_time_s=None, states=[], controls=[], reason="solver failed")
    batch = write_batch(tmp_path, [trajectory_payload(), unsolved])
    result = build_payload(batch, contexts=contexts())
    assert result["report"]["total"] == 2
    assert (result["tracksShown"], result["tracksTotal"]) == (1, 1)


def test_non_positive_overlay_caps_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="greater than zero"):
        _sample_evenly([1, 2], 0)
    batch = write_batch(tmp_path, payloads_with_repeated_callsign())
    with pytest.raises(ValueError, match="greater than zero"):
        build_payload(batch, contexts=contexts(), max_tracks=-1)


def test_rendered_payload_is_strict_json_and_escapes_script_close(tmp_path):
    value = trajectory_payload()
    value["source"]["id"] = "</script><img src=x>"
    result = build_payload(write_batch(tmp_path, [value]), contexts=contexts())
    page = render_html(result, title="Test", source_label="batch")
    assert "const DATA=" in page
    assert "22 m RNAV/RNP terminal vertical bound" in page
    # The note states the CURRENT speed policy -- every subject graded, the observed
    # baseline on its ground-speed proxy. Positive pins: the old negative pins on
    # retired wording could never fail again, and the note they guarded was the one
    # that had actually gone stale.
    assert "[1.23·Vs(n), 1.23·Vs1g + 20 kt]" in page
    assert "corrected by the field's METAR headwind into an airspeed estimate" in page
    assert "V judged (m/s)" in page
    assert "</script><img" not in page
    assert "<\\/script><img" in page
