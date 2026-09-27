"""The observed-traffic census (`experiments/traffic_census`, multi-aircraft design §6.4 step 4): what it counts on
hand-built tracks, and its runner on a tmp artefact (never a live root)."""

import dataclasses
import json
import math

import numpy as np
import pytest

from ts_transformer.inference.runway_schedule import CWT_ON_APPROACH_NM, Separation

STEP_S = 2.0


def _track(key: str, times: np.ndarray, along: np.ndarray, rate: float, *, n: float = 0.0, landing_s: float,
           captured_s: float = 0.0):
    from ts_transformer.experiments.traffic_census import Track
    from ts_transformer.prior.scene import Presence, hung_span

    seen = Presence(key, "KXXX", "R", landing_s, True, times, np.abs(along))
    first, last = hung_span(seen, STEP_S)
    return Track(seen, first, last, along.copy(), np.full(len(times), n), np.full(len(times), 500.0), along,
                 np.full(len(times), rate), captured_s, "artefact", "F")


def test_the_census_counts_an_in_trail_loss_the_landing_intervals_the_closing_speed_and_the_swaps():
    from ts_transformer.experiments.traffic_census import census_airport

    # L at 70 m/s and F at 80 m/s on one final, 7 km apart at t = 0: under 3 NM from t = 146 s; G, never established and
    # 20 km off, enters later than both and lands first
    t = np.arange(0.0, 200.0 + 1e-9, STEP_S)
    leader = _track("L", t, -15_000.0 + 70.0 * t, 70.0, landing_s=15_000.0 / 70.0)
    t_f = np.arange(0.0, 260.0 + 1e-9, STEP_S)
    follower = _track("F", t_f, -22_000.0 + 80.0 * t_f, 80.0, landing_s=22_000.0 / 80.0)
    t_g = np.arange(50.0, 150.0 + 1e-9, STEP_S)
    late = _track("G", t_g, -9_000.0 + 70.0 * (t_g - 50.0), 70.0, n=20_000.0, landing_s=160.0, captured_s=math.inf)
    separation = Separation(same_nm=3.0, speed_mps=70.0, along_nm={"R": 0.0}, wake_nm=CWT_ON_APPROACH_NM)
    census = census_airport([leader, follower, late], separation, STEP_S)

    losses = census["losses"]
    assert (losses["episodes"], losses["pairs_with_a_loss"], losses["by_kind"]) == (1, 1, {"in_trail": 1})
    episode = census["episodes"][0]
    assert episode["steps"] == len(range(146, 201, 2)) and episode["responsible"] == [{"key": "F", "speaking": True}]
    assert census["closing_speed_on_a_final_mps"]["p50"] == pytest.approx(10.0)
    # at G's landing L is 3.8 km behind, at L's F is 4.86 km: both under 3 NM, neither a TBL 5-5-2 (F behind F) loss
    landing = census["at_landing"]
    assert (landing["judged"], landing["below_required"], landing["wake_losses"], landing["required_nm"]) == (
        2, 2, 0, {"3": 2})
    assert landing["interval_m"]["min"] == pytest.approx(3_800.0)
    assert census["order_swaps"]["same_runway"] == {"swapped": 2, "pairs": 3, "share": pytest.approx(2 / 3)}
    assert census["A_max"] == 3


def _artefact(tmp_path, typecode: str = "A320"):
    """The two fixture flights of the prior census test (one labelled, one background) in a tmp artefact, with a tmp
    arrivals manifest recorded as the one the signals were read from."""
    from ts_transformer.instructions.artefact import (
        labeller_source_sha256, write_candidates, write_sentences, write_signals, write_spec,
    )
    from ts_transformer.instructions.labeller.read import read_flight
    from ts_transformer.io_utils import file_sha256
    from ts_transformer.tests.support import (
        fixture_days, fly_legs, instruction_airport, instruction_flight, instruction_spec,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runway_targets": {"09": {"lat": 35.0, "lon": -78.0, "course_deg": 90.0}}}),
                        encoding="utf-8")
    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    flights = [dataclasses.replace(instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0),
                                                      dataset_id=f"KXXX:{name}"), typecode=typecode) for name in "ab"]
    directory = tmp_path / "artefact"
    directory.mkdir()
    write_signals(directory, {"train": flights},
                  {"counts": {"train": {"built_usable": 2}}, "test_days": {"flights_not_opened": 0},
                   "sources": [{"airport": "KXXX", "arrival_manifest_sha256": file_sha256(manifest)}]}, fixture_days())
    write_candidates(directory, {"KXXX": instruction_airport()})
    spec = instruction_spec()
    write_spec(directory, spec, {"n": 1}, {"labeller_source_sha256": labeller_source_sha256(),
                                           "git": {"head": "test", "dirty": False}})
    write_sentences(directory, "train", spec, [read_flight(flights[0], instruction_airport(), spec)], [0])
    return directory, manifest


def test_the_runner_judges_an_artefact_and_refuses_what_it_should(tmp_path, monkeypatch):
    from ts_transformer.experiments import traffic_census

    directory, manifest = _artefact(tmp_path)
    monkeypatch.setattr(traffic_census, "arrival_manifest_path", lambda code: manifest)
    out = tmp_path / "census"
    assert traffic_census.main(["--instructions", str(directory), "--out", str(out)]) == 0
    census = json.loads((out / "census.json").read_text(encoding="utf-8"))
    assert census["instructions"] == str(directory) and census["arrival_manifest_sha256"]["KXXX"]
    airport = census["airports"]["KXXX"]
    assert (airport["flights"]["speaking"], airport["flights"]["background"]) == (1, 1)
    with pytest.raises(SystemExit):                                                  # never overwritten
        traffic_census.main(["--instructions", str(directory), "--out", str(out)])
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="not the manifest"):                        # not what the signals were read from
        traffic_census.main(["--instructions", str(directory), "--out", str(tmp_path / "again")])


def test_the_runner_stops_on_a_type_neither_table_lists(tmp_path, monkeypatch):
    from ts_transformer.experiments import traffic_census

    directory, manifest = _artefact(tmp_path, typecode="ZZZZ")
    monkeypatch.setattr(traffic_census, "arrival_manifest_path", lambda code: manifest)
    with pytest.raises(SystemExit, match="ZZZZ"):
        traffic_census.main(["--instructions", str(directory), "--out", str(tmp_path / "census")])
    assert json.loads((tmp_path / "census" / "types_not_listed.json").read_text(encoding="utf-8"))["types"] == ["ZZZZ"]
