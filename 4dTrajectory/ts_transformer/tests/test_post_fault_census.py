"""Stage C, C1: the census of the recorded aircraft with a faulty observed track (post-training §8 C1; vocabulary
D111)."""

from __future__ import annotations

import numpy as np
import pytest

from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.post.fault_census import STEPS_BEFORE_EVENT, fault_census, reads_fault, window_faults
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import airport_scenes, real_windows
from ts_transformer.tests.post_support import categories, finals, scene_artefact, straight_in, train_noon
from ts_transformer.tests.support import INSTRUCTION_STEP_S

DELTA = 4.0


def _built(tmp_path, seconds):
    flights = {"train": [straight_in(f"KXXX:{k}", train_noon(s)) for k, s in enumerate(seconds)]}
    spec = scene_artefact(tmp_path / "art", flights, interval_s=DELTA)
    geometries = load_candidates(tmp_path / "art")
    scenes, signals = airport_scenes(tmp_path / "art", "train", spec, DELTA, geometries, categories)
    windows = real_windows(tmp_path / "art", "train", spec, DELTA, scenes, signals)
    return windows, scenes["KXXX"], airport_separation(geometries["KXXX"]), finals(geometries["KXXX"])


def test_a_recorded_aircraft_reads_a_faulty_point_at_its_row_and_through_its_motion():
    from ts_transformer.post.scene import Recorded

    flight = Recorded(key="K:x", airport="K", runway_index=0, category=None, start_s=100.0, step_s=2.0,
                      first_step_s=116.0, landing_s=200.0, e_m=np.zeros(50), n_m=np.zeros(50), height_m=np.zeros(50),
                      go_around=np.zeros(50, dtype=bool))
    times = 100.0 + 2.0 * np.arange(10)
    assert np.flatnonzero(reads_fault(flight, times, frozenset({3}))).tolist() == [3, 4]   # the row, the motion after it


def test_the_steps_and_tokens_that_read_a_faulty_point(tmp_path):
    windows, scene, separation, fin = _built(tmp_path, [0.0, 120.0])
    window = windows[1]                                                   # flight 1; flight 0 is in the air ahead
    leader = scene.flights[0]
    marked = {leader.key: frozenset({70, 71})}
    found = window_faults(window, marked, separation, fin, INSTRUCTION_STEP_S)
    times = window.row0_s + np.arange(int((window.commanded.end_s - window.row0_s) // DELTA) + 1) * DELTA
    present = (times >= leader.start_s) & (times <= leader.end_s)
    expected = reads_fault(leader, times[present], marked[leader.key])
    assert found.marked_in_air and found.faulty_steps == found.faulty_tokens == int(expected.sum()) > 0
    clean = window_faults(window, {}, separation, fin, INSTRUCTION_STEP_S)
    assert not clean.marked_in_air and clean.faulty_steps == clean.faulty_tokens == 0
    own_only = window_faults(window, {window.commanded.key: frozenset({70})}, separation, fin, INSTRUCTION_STEP_S)
    assert not own_only.marked_in_air and own_only.faulty_tokens == 0          # the commanded aircraft is no token


def test_a_loss_on_the_records_and_whether_it_reads_a_faulty_point(tmp_path):
    windows, scene, separation, fin = _built(tmp_path, [0.0, 8.0])            # 8 s apart: inside 3 NM
    window = windows[1]
    clean = window_faults(window, {}, separation, fin, INSTRUCTION_STEP_S)
    assert clean.loss and not clean.loss_reads_fault
    # the loss is at the first step after the first predicted step: the leader's rows there, and 2 Δ before
    event = window.first_step_s + DELTA
    leader = scene.flights[0]
    near = int(round((event - leader.start_s) / leader.step_s))
    reads = window_faults(window, {leader.key: frozenset({near})}, separation, fin, INSTRUCTION_STEP_S)
    assert reads.loss and reads.loss_reads_fault
    before = int(round((event - STEPS_BEFORE_EVENT * DELTA - leader.start_s) / leader.step_s))
    assert window_faults(window, {leader.key: frozenset({before})}, separation, fin, INSTRUCTION_STEP_S).loss_reads_fault
    far = window_faults(window, {leader.key: frozenset({before - 10})}, separation, fin, INSTRUCTION_STEP_S)
    assert far.loss and not far.loss_reads_fault
    own_row = int(round((event - window.commanded.start_s) / window.commanded.step_s))
    own = window_faults(window, {window.commanded.key: frozenset({own_row})}, separation, fin, INSTRUCTION_STEP_S)
    assert own.loss_reads_fault                                             # the commanded aircraft's own fault counts


def test_the_census_adds_the_windows_of_each_airport(tmp_path):
    windows, scene, separation, fin = _built(tmp_path, [0.0, 120.0, 7_200.0])
    leader = scene.flights[0]
    counted = fault_census(windows, {"KXXX": {leader.key: frozenset({70})}}, {"KXXX": separation}, {"KXXX": fin},
                           INSTRUCTION_STEP_S)["KXXX"]
    assert counted["windows"] == 3 and counted["windows_with_marked_in_air"] == 1
    assert counted["tokens_reading_fault"] > 0 and counted["losses_on_records"] == 0
    assert counted["steps"] == sum(int((w.commanded.end_s - w.row0_s) // DELTA) + 1 for w in windows)
    with pytest.raises(ValueError, match="real windows"):
        from dataclasses import replace

        fault_census([replace(windows[0], kind="A")], {"KXXX": {}}, {"KXXX": separation}, {"KXXX": fin},
                     INSTRUCTION_STEP_S)


def test_the_judge_is_skipped_only_where_no_minimum_can_bind():
    """A step is not judged where no other aircraft is within the largest minimum (8 NM), widened by the largest offset
    across the courses of a pair separated as one (2,500 ft): the threshold's wake gap is along the approach clock."""
    from geokit import FT_M, NM_M

    from ts_transformer.inference.runway_schedule import CWT_ON_APPROACH_NM, SINGLE, FAA_PARALLEL_REGIMES
    from ts_transformer.post.fault_census import FARTHEST_MINIMUM_M

    assert FAA_PARALLEL_REGIMES[0].relation == SINGLE
    assert FARTHEST_MINIMUM_M == pytest.approx(np.hypot(8.0 * NM_M, 2_500.0 * FT_M))
    assert FARTHEST_MINIMUM_M > max(CWT_ON_APPROACH_NM.values()) * NM_M


def test_a_window_with_nobody_near_has_no_loss(tmp_path):
    windows, scene, separation, fin = _built(tmp_path, [0.0, 7_200.0])
    assert not window_faults(windows[1], {}, separation, fin, INSTRUCTION_STEP_S).loss      # nobody near: no loss
