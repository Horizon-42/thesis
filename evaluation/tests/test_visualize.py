"""HTML visualization: payload structure, track sampling, end-to-end file output."""

import json
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation import load_records  # noqa: E402
from evaluation.visualize import (  # noqa: E402
    _sample_evenly,
    build_payload,
    main as visualize_main,
)


def _path_states(*, lon_offset_deg=0.0, alt_top=1000.0, t_end=100.0, n=5):
    states = []
    for k in range(n):
        f = k / (n - 1)
        states.append({
            "t": t_end * f, "lat": 35.0 + 0.1 * f, "lon": -78.5 + lon_offset_deg,
            "alt": alt_top + (130.0 - alt_top) * f,
            "V": 100.0, "psi": math.pi / 2, "gamma": -0.05, "m": 60000.0,
        })
    return states


def _solved_payload(flight_id="AFR074", *, reference_file=None):
    states = _path_states(lon_offset_deg=0.001, t_end=80.0)
    control = {"thrust": 1e5, "bank_rad": 0.0, "load_factor": 1.0}
    payload = {
        "source": {"id": flight_id},
        "initial_state": {k: v for k, v in states[0].items() if k != "t"},
        "target_state": {k: v for k, v in states[-1].items() if k != "t"},
        "final_time_s": 80.0,
        "states": states,
        "controls": [control] * len(states),
    }
    if reference_file:
        payload["reference_file"] = reference_file
    return payload


def _reference_payload():
    states = _path_states()
    return {
        "source": {"id": "AFR074"},
        "initial_state": {k: v for k, v in states[0].items() if k != "t"},
        "target_state": {k: v for k, v in states[-1].items() if k != "t"},
        "final_time_s": 100.0,
        "states": states,
        "controls": [],
    }


def _failed_payload():
    return {
        "source": {"id": "BAD001"},
        "initial_state": _solved_payload()["initial_state"],
        "target_state": None,
        "final_time_s": None,
        "states": [],
        "controls": [],
        "reason": "ValueError: infeasible",
    }


def _write_batch(tmp_path):
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "AFR074_reference_eval.json").write_text(
        json.dumps(_reference_payload()), encoding="utf-8")
    (tmp_path / "a_eval.json").write_text(
        json.dumps(_solved_payload(reference_file="references/AFR074_reference_eval.json")),
        encoding="utf-8")
    (tmp_path / "b_eval.json").write_text(json.dumps(_failed_payload()), encoding="utf-8")


def test_build_payload_embeds_report_and_tracks(tmp_path):
    _write_batch(tmp_path)
    payload = build_payload(load_records(tmp_path))
    assert payload["report"]["total"] == 2 and payload["report"]["solved"] == 1
    assert payload["tracksShown"] == payload["tracksTotal"] == 1
    track = payload["tracks"][0]
    assert track["id"] == "AFR074"
    assert len(track["optimized"]["lat"]) == 101
    assert len(track["reference"]["lat"]) == 101       # the pointed-at observed path
    assert track["target"]["lat"] == pytest.approx(35.1)


def test_sample_evenly_keeps_ends_and_caps():
    items = list(range(10))
    assert _sample_evenly(items, 20) == items          # no cap needed
    sampled = _sample_evenly(items, 4)
    assert len(sampled) == 4 and sampled[0] == 0 and sampled[-1] == 9


def test_cli_writes_html_with_embedded_data(tmp_path, capsys):
    _write_batch(tmp_path)
    out = tmp_path / "report.html"
    visualize_main(["--input", str(tmp_path), "--output", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "const DATA" in text and "cdn.plot.ly" in text
    assert "AFR074" in text and "Trajectory Evaluation Report" in text
    # Aviation abbreviations are expanded in parentheses in the page body.
    assert "WCH (Wheel Crossing Height)" in text
    assert "ADS-B (Automatic Dependent Surveillance–Broadcast)" in text
    assert "1/1 track overlays" in capsys.readouterr().out


def test_degenerate_record_excluded_from_tracks_and_html_is_escaped(tmp_path):
    # A single-sample solved record must not reach the resampler; hostile strings in
    # source ids / reasons must not break out of the embedded <script> JSON.
    _write_batch(tmp_path)
    single = {"t": 0.0, "lat": 35.6, "lon": -78.5, "alt": 2000.0, "V": 130.0,
              "psi": 1.5, "gamma": -0.05, "m": 60000.0}
    degenerate = {
        "source": {"id": "</script><img src=x>"},
        "initial_state": {k: v for k, v in single.items() if k != "t"},
        "target_state": {k: v for k, v in single.items() if k != "t"},
        "final_time_s": 0.0,
        "states": [single],
        "controls": [{"thrust": 1e5, "bank_rad": 0.0, "load_factor": 1.0}],
    }
    (tmp_path / "c_eval.json").write_text(json.dumps(degenerate), encoding="utf-8")

    out = tmp_path / "report.html"
    visualize_main(["--input", str(tmp_path), "--output", str(out)])
    text = out.read_text(encoding="utf-8")
    # The degenerate record is in the report but NOT among the track overlays.
    payload = build_payload(load_records(tmp_path))
    assert payload["report"]["total"] == 3
    assert payload["tracksTotal"] == 1 and payload["tracks"][0]["id"] == "AFR074"
    # The raw close-tag sequence never appears inside the data blob (escaped as <\/).
    assert "</script><img" not in text
    assert "<\\/script><img" in text
