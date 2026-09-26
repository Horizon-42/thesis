"""The glidepath lower edge (post-training design §3, `prior.procedure`): where it binds, the word rules, the flown-track
check, the final read from the coded approach, the mirrored constants; its mask in the speaker (`prior.generate`), the
stop in the closed loop (`experiments.prior_free_generation.glidepath_stops`), and the stage-0 check's pieces
(`experiments.prior_procedure_check`)."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flight_scenarios.fas_geometry import FasCourseGeometry, course_halfwidth_m, fas_course_geometry
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.frame import ALT, GAMMA, LAT, LON, MASS, PSI, SPEED
from ts_transformer.experiments import prior_procedure_check
from ts_transformer.experiments.prior_free_generation import (
    BELOW_GLIDEPATH, flight_rows, glidepath_stops, in_force, speak_and_fly,
)
from ts_transformer.experiments.prior_procedure_check import (
    MAX_FORBIDDEN_WORDS, MAX_STOPPED_REPLAYS, check_labelled, labelled_rows, passes,
)
from ts_transformer.instructions.airport import RunwayCandidate
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import ALTITUDE, RUNWAY, UNCHANGED, Words
from ts_transformer.prior import procedure
from ts_transformer.prior.generate import Speaker
from ts_transformer.prior.procedure import (
    GLIDEPATH_BELOW_M, THRESHOLD_TOLERANCE_M, RunwayProcedure, airport_procedures, altitude_word_allowed, below_floor,
    runway_procedure, track_tolerance_m, word_tolerance_m,
)
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import REPO_ROOT
from ts_transformer.tests.support import instruction_airport, instruction_spec
from ts_transformer.tests.test_autopilot import _params
from ts_transformer.tests.test_prior import _model as _prior_model, _signals, _two_runways
from ts_transformer.tests.test_prior_speaker import _flight, _model as _speaker_model

FAF_D_M = 10_000.0
TAN = math.tan(math.radians(3.0))
RUNWAY_09 = instruction_airport().candidates[0]                   # threshold at the origin, flown eastbound


def _final(crossing_m: float = 115.0, candidate: RunwayCandidate = RUNWAY_09, faf_d_m: float = FAF_D_M,
           cone: FasCourseGeometry | None = None) -> RunwayProcedure:
    return RunwayProcedure(candidate=candidate, crossing_m=crossing_m, glidepath_tan=TAN, faf_d_m=faf_d_m,
                           cone=cone or fas_course_geometry(candidate.length_m))


def _at(d: float, xt: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The airport-frame position ``d`` before runway 09's threshold, ``xt`` north of its centreline."""
    return np.array([-d]), np.array([xt])


def test_the_mirrored_constants_equal_their_owners():
    optimization = REPO_ROOT / "4dTrajectory" / "optimization"
    if str(optimization) not in sys.path:
        sys.path.insert(0, str(optimization))
    import approach_constraints
    import scenario_optimization

    assert GLIDEPATH_BELOW_M == approach_constraints.DEFAULT_GLIDEPATH_BELOW_M
    assert THRESHOLD_TOLERANCE_M == scenario_optimization._FRAME_ANCHOR_TOLERANCE_M


def test_the_floor_is_the_glidepath_less_the_lower_edge_inside_the_faf_and_the_cone_only():
    final = _final()
    for d in (0.0, 3_000.0, FAF_D_M):
        assert final.floor_m(*_at(d))[0] == pytest.approx(115.0 + d * TAN - GLIDEPATH_BELOW_M)
    half = float(course_halfwidth_m(5_000.0, final.cone))
    assert not np.isnan(final.floor_m(*_at(5_000.0, 0.99 * half))[0])
    assert not np.isnan(final.floor_m(*_at(5_000.0, -0.99 * half))[0])        # either side of the centreline
    for d, xt in ((FAF_D_M + 1.0, 0.0), (-100.0, 0.0), (5_000.0, 1.01 * half)):
        assert np.isnan(final.floor_m(*_at(d, xt))[0])                       # outside the FAF, past the threshold, off


def test_the_word_rules_allow_the_step_holding_the_floor_and_land_only_from_above_it():
    spec = instruction_spec()
    words = Words(spec)
    step, tolerance = spec.altitude_step_m, word_tolerance_m(spec)
    assert tolerance == step / 2
    # a crossing that puts the floor 5 km out exactly on a step: the step below it is 1.5 tolerances short
    d = 5_000.0
    final = _final(crossing_m=20 * step - d * TAN + GLIDEPATH_BELOW_M)
    floor = float(final.floor_m(*_at(d))[0])
    assert floor == pytest.approx(20 * step)
    e, n = _at(d)
    assert altitude_word_allowed(final, np.array([20, 21, 19]), e, n, np.zeros(1), words).tolist() == [True, True, False]
    land = np.array([words.altitude_land])
    assert altitude_word_allowed(final, land, e, n, np.array([floor - tolerance + 0.1]), words)[0]
    assert not altitude_word_allowed(final, land, e, n, np.array([floor - tolerance - 0.1]), words)[0]
    # outside the FAF there is no floor: any word
    far = _at(FAF_D_M + 500.0)
    assert altitude_word_allowed(final, np.array([0, words.altitude_land]), *far, np.zeros(1), words).all()


def test_a_flown_row_is_below_only_beyond_the_track_tolerance():
    spec = instruction_spec()
    final, e, n = _final(), *_at(4_000.0)
    floor = float(final.floor_m(e, n)[0])
    tolerance = track_tolerance_m(spec)
    assert tolerance == word_tolerance_m(spec) + spec.altitude_tolerance_m
    below, got = below_floor(final, e, n, np.array([floor - tolerance + 0.1]), spec)
    assert not below[0] and got[0] == pytest.approx(floor)
    assert below_floor(final, e, n, np.array([floor - tolerance - 0.1]), spec)[0][0]
    assert not below_floor(final, *_at(FAF_D_M + 1.0), np.array([-1_000.0]), spec)[0][0]      # no floor there


def _skeleton(geometry, threshold_e_m: float, faf_d_m: float, course_deg: float = 90.0, threshold_n_m: float = 0.0):
    """A coded approach whose threshold is at ``(threshold_e_m, threshold_n_m)`` and whose FAF is ``faf_d_m`` before it
    on a final flown on ``course_deg``."""
    frame = geometry.frame
    travel = math.radians(course_deg)
    threshold = frame.latlon_from_horizontal(threshold_e_m, threshold_n_m)
    faf = frame.latlon_from_horizontal(threshold_e_m - faf_d_m * math.sin(travel),
                                       threshold_n_m - faf_d_m * math.cos(travel))
    return SimpleNamespace(procedure_uid="KXXX-R-RW", threshold_lat_deg=threshold[0], threshold_lon_deg=threshold[1],
                           faf=SimpleNamespace(lat_deg=faf[0], lon_deg=faf[1]))


def test_the_final_is_read_from_the_coded_approach_and_the_published_glidepath(monkeypatch):
    geometry = instruction_airport()
    candidate = geometry.candidates[0]
    runway = SimpleNamespace(ident="09", threshold_crossing_height_m=15.0, published_glidepath_deg=3.0)
    monkeypatch.setattr(procedure, "procedure_skeleton", lambda code, ident, root: _skeleton(geometry, 0.5, 9_800.0))
    final = runway_procedure(geometry, candidate, runway)
    assert final.faf_d_m == pytest.approx(9_799.5, abs=1e-6) and final.ident == "09"   # measured from the candidate
    assert final.crossing_m == candidate.elevation_m + 15.0
    assert final.glidepath_tan == pytest.approx(TAN)
    assert final.cone == fas_course_geometry(candidate.length_m)
    # a document whose threshold is another runway's is refused, and so is a runway without a published glidepath
    far = THRESHOLD_TOLERANCE_M + 1.0
    monkeypatch.setattr(procedure, "procedure_skeleton", lambda code, ident, root: _skeleton(geometry, far, 9_800.0))
    with pytest.raises(ValueError, match="from the candidate's"):
        runway_procedure(geometry, candidate, runway)
    monkeypatch.setattr(procedure, "procedure_skeleton", lambda code, ident, root: _skeleton(geometry, 0.0, 9_800.0))
    with pytest.raises(ValueError, match="no threshold crossing height or glidepath"):
        runway_procedure(geometry, candidate, SimpleNamespace(ident="09", threshold_crossing_height_m=15.0,
                                                              published_glidepath_deg=None))


def test_an_airport_s_finals_follow_the_runway_pointer_whatever_order_the_runways_come_in(monkeypatch):
    geometry = _two_runways()                                    # 09 at the origin, 27 3 km east flown westbound
    skeletons = {"09": _skeleton(geometry, 0.0, 9_000.0), "27": _skeleton(geometry, 3_000.0, 11_000.0, 270.0, 500.0)}
    monkeypatch.setattr(procedure, "procedure_skeleton", lambda code, ident, root: skeletons[ident])
    runways = [SimpleNamespace(ident=ident, threshold_crossing_height_m=tch, published_glidepath_deg=3.0)
               for ident, tch in (("27", 16.0), ("09", 15.0))]            # the harvest's order, not the pointer's
    finals = airport_procedures(geometry, runways)
    assert [f.ident for f in finals] == ["09", "27"]
    assert [round(f.faf_d_m) for f in finals] == [9_000, 11_000]
    assert [f.crossing_m for f in finals] == [geometry.candidates[0].elevation_m + 15.0,
                                              geometry.candidates[1].elevation_m + 16.0]


def test_in_force_carries_each_column_forward_and_needs_a_first_row_that_says_every_column():
    grid = np.full((5, 6), UNCHANGED)
    grid[0] = [0, 0, 10, 30, 2, 5]
    grid[2, ALTITUDE] = 20
    grid[4, RUNWAY] = 1
    force = in_force(grid)
    assert force[:, ALTITUDE].tolist() == [30, 30, 20, 20, 20]
    assert force[:, RUNWAY].tolist() == [0, 0, 0, 0, 1]
    grid[0, ALTITUDE] = UNCHANGED
    with pytest.raises(ValueError, match="first row says every column"):
        in_force(grid)


def test_the_check_counts_the_words_where_the_prior_would_say_them():
    spec = instruction_spec()
    words, final = Words(spec), _final()
    rows = N_LOOK + 4
    d = np.linspace(9_000.0, 6_000.0, rows)                  # inside the FAF all along
    h = final.crossing_m + d * TAN                           # on the glidepath
    grid = np.full((rows, 6), UNCHANGED)
    grid[0] = 5                                               # said before the first predicted step: not counted there…
    grid[N_LOOK + 2, ALTITUDE] = words.altitude_land          # …but it is in force at N_LOOK (forbidden: 150 m)
    force, said = in_force(grid)[:, ALTITUDE], grid[:, ALTITUDE]

    def check(heights):
        return check_labelled({"e": -d, "n": np.zeros(rows), "h": heights, "in_force": force, "said": said,
                               "row": np.arange(rows), "flight": np.zeros(rows, dtype=int)}, final, words)

    on = check(h)
    assert on["words"] == {"level": 1, "land": 1, "level_forbidden": 1, "land_forbidden": 0}
    # the level word in force is forbidden at the one silent step before the landing word replaces it
    assert on["forced_steps"] == 1 and on["steps"] == 3 and on["flights_forced"] == 1
    assert on["observed_below"] == 0
    low = check(h - GLIDEPATH_BELOW_M - track_tolerance_m(spec) - 1.0)
    assert low["observed_below"] == 1 and low["observed_depth_m"][0] == pytest.approx(track_tolerance_m(spec) + 1.0)
    assert low["words"]["land_forbidden"] == 1


def test_a_labelled_flight_s_rows_are_its_signals_first_rows(monkeypatch):
    """Contract C30: a sentence's words line up with the FIRST ``len(words)`` rows of its signals."""
    words = Words(instruction_spec())
    rows, extra = N_LOOK + 3, 5
    flights = [SimpleNamespace(airport="KXXX", e_m=np.arange(rows + extra) + 100.0 * k, n_m=np.zeros(rows + extra),
                               altitude_m=np.full(rows + extra, 500.0 + k)) for k in range(2)]
    grid = np.full((rows, 6), UNCHANGED, dtype=np.int16)
    grid[0] = 3
    sentences = {"signal_index": np.array([1, 0]), "offsets": np.array([0, rows, 2 * rows]),
                 "words": np.concatenate([grid, grid]), "runway_index": np.array([0, 0])}
    monkeypatch.setattr(prior_procedure_check, "load_signals", lambda directory, split: flights)
    monkeypatch.setattr(prior_procedure_check, "load_sentences", lambda directory, split, spec: sentences)
    pooled = labelled_rows("train", None, words)[("KXXX", 0)]
    assert pooled["e"].tolist() == [*(np.arange(rows) + 100.0), *np.arange(rows)]      # sentence 0 is signal 1
    assert pooled["h"].tolist() == [501.0] * rows + [500.0] * rows
    assert pooled["flight"].tolist() == [0] * rows + [1] * rows and pooled["row"].tolist() == [*range(rows)] * 2


def test_the_check_passes_only_within_both_lines():
    assert passes({"forbidden_share": MAX_FORBIDDEN_WORDS, "stopped_share": MAX_STOPPED_REPLAYS})
    assert not passes({"forbidden_share": MAX_FORBIDDEN_WORDS + 1e-9, "stopped_share": 0.0})
    assert not passes({"forbidden_share": 0.0, "stopped_share": MAX_STOPPED_REPLAYS + 1e-9})


def _flown(d: np.ndarray, h: np.ndarray, done_cycle: int, sentence_s: np.ndarray | None = None):
    """A flown record on runway 09's centreline: state ``c`` (a cycle boundary) at ``d[c]`` before the threshold, height
    ``h[c]``; the executor done at the end of cycle ``done_cycle``; cycle ``c``'s sentence time ``sentence_s[c]`` (default
    the time clock: ``c`` s)."""
    frame = instruction_airport().frame
    states = np.zeros((1, len(d), 7))
    for c, (distance, height) in enumerate(zip(d, h)):
        states[0, c, [LAT, LON]] = frame.latlon_from_horizontal(-distance, 0.0)
        states[0, c, [ALT, SPEED, PSI, GAMMA, MASS]] = height, 70.0, 0.0, 0.0, 60_000.0
    clock = np.arange(len(d) - 1, dtype=np.float64) if sentence_s is None else sentence_s
    return SimpleNamespace(states=torch.as_tensor(states), done_cycle=torch.tensor([done_cycle]), cycle_s=1.0,
                           sentence_s=torch.as_tensor(clock)[None])


def test_a_sentence_stops_at_the_first_step_whose_end_state_is_below_the_edge_of_the_runway_in_force():
    words = Words(instruction_spec())
    final, cycles = _final(), 20
    d = np.linspace(9_000.0, 7_000.0, cycles + 1)
    h = final.crossing_m + d * TAN
    sink = GLIDEPATH_BELOW_M + track_tolerance_m(words.spec) + 5.0
    grid = np.zeros((cycles // 2, 6), dtype=np.int64)            # runway pointer 0 (09) in force throughout
    geometry = [instruction_airport()]

    def stop(heights, done=cycles - 1, finals=((final,),), said=grid):
        stops = glidepath_stops(_flown(d, heights, done), [said], geometry, finals, words)
        assert stops.row[0] == stops.step[0]                       # the time clock: a step flies its own row
        return int(stops.step[0])

    assert stop(h) == -1
    dipped = h.copy()
    dipped[7] -= sink                                            # an odd cycle: inside a step, not at its end
    assert stop(dipped) == -1
    dipped[8] -= sink                                            # the end of step 3 (cycles 6 and 7)
    assert stop(dipped) == 3
    assert stop(dipped, done=5) == -1                            # the executor was done before that step ended
    # the runway in force during the step decides whose edge: under another runway's final, no stop
    away = replace(final, candidate=replace(RUNWAY_09, threshold_e_m=50_000.0))
    said = grid.copy()
    said[3:, RUNWAY] = 1
    assert stop(dipped, finals=((final, away),), said=said) == -1
    assert stop(dipped, finals=((final, away),), said=grid) == 3


def test_a_labelled_sentence_on_the_track_clock_stops_at_the_row_its_step_flew():
    """A labelled sentence is heard where the observed aircraft heard it: step k may fly another row than k — the stop
    reads that row's runway and names that row."""
    words = Words(instruction_spec())
    final, cycles = _final(), 20
    d = np.linspace(9_000.0, 7_000.0, cycles + 1)
    h = final.crossing_m + d * TAN
    h[8] -= GLIDEPATH_BELOW_M + track_tolerance_m(words.spec) + 5.0           # the end of step 3
    clock = np.arange(cycles, dtype=np.float64) + 4.0                         # two rows ahead: step 3 flies row 5
    away = replace(final, candidate=replace(RUNWAY_09, threshold_e_m=50_000.0))
    grid = np.zeros((cycles // 2, 6), dtype=np.int64)
    geometry = [instruction_airport()]
    stops = glidepath_stops(_flown(d, h, cycles - 1, clock), [grid], geometry, ((final, away),), words)
    assert (int(stops.step[0]), int(stops.row[0])) == (3, 5)
    said = grid.copy()
    said[5:, RUNWAY] = 1                                                      # row 5 points at the other runway
    stops = glidepath_stops(_flown(d, h, cycles - 1, clock), [said], geometry, ((final, away),), words)
    assert stops.step[0] == -1 and stops.row[0] == -1


def test_in_force_is_the_executor_s_reading_of_a_sentence():
    from ts_transformer.autopilot.sentence import _filled

    rng = np.random.default_rng(3)
    grid = np.where(rng.random((30, 6)) < 0.8, UNCHANGED, rng.integers(0, 9, (30, 6)))
    grid[0] = rng.integers(0, 9, 6)
    assert np.array_equal(in_force(grid), _filled(grid)[0])


def test_the_speaker_masks_the_altitude_column_by_the_runway_just_sampled_and_the_word_in_force():
    words, geometry = Words(instruction_spec()), _two_runways()
    signals = _signals(N_LOOK + 3)
    e, n = signals.e_m[N_LOOK], signals.n_m[N_LOOK]
    # runway 09's edge lies over the aircraft (a cone wide enough to hold it), runway 27's nowhere near it
    wide = _final(crossing_m=1_500.0, faf_d_m=20_000.0,
                  cone=FasCourseGeometry(d_fpap_m=3_000.0, d_garp_m=3_300.0, course_width_m=2_000.0))
    nowhere = _final(candidate=geometry.candidates[1], crossing_m=110.0, faf_d_m=0.0)
    floor = float(wide.floor_m(np.array([e]), np.array([n]))[0])
    assert floor > signals.altitude_m[N_LOOK] + 200.0
    speaker = Speaker(_prior_model(variant="no-context"), [signals] * 2, [geometry] * 2, None, words,
                      max_rows=N_LOOK + 3, generator=torch.Generator().manual_seed(0), finals=[(wide, nowhere)] * 2)
    classes = words.n_altitude_levels + 2
    chosen = np.zeros((2, 6), dtype=np.int64)
    chosen[:, RUNWAY] = [1, 2]                                    # this step's runway: 09, then 27
    out = speaker._above_the_glidepath(chosen, True, classes)
    levels = np.arange(words.n_altitude_levels) * words.spec.altitude_step_m
    assert np.array_equal(out[0, 1:-1], levels >= floor - word_tolerance_m(words.spec))
    assert not out[0, -1] and out[0, 0]                          # below the edge: no landing; "unchanged" not asked
    assert out[1].all()                                           # runway 27: no edge here
    # later steps: "unchanged" only where the word in force still holds, for the runway in force when none is said
    speaker.value[:, RUNWAY] = 1
    speaker.value[:, ALTITUDE] = [1 + words.altitude_index(600.0), 1 + words.altitude_index(3000.0)]
    later = speaker._above_the_glidepath(np.zeros((2, 6), dtype=np.int64), False, classes)
    assert later[:, 0].tolist() == [False, True]
    with pytest.raises(ValueError, match="flights' finals"):
        Speaker(_prior_model(variant="no-context"), [signals] * 2, [geometry] * 2, None, words, max_rows=N_LOOK + 3,
                generator=torch.Generator(), finals=[(wide, nowhere)])
    with pytest.raises(ValueError, match="in the pointer's order"):
        Speaker(_prior_model(variant="no-context"), [signals] * 2, [geometry] * 2, None, words, max_rows=N_LOOK + 3,
                generator=torch.Generator(), finals=[(nowhere, wide)] * 2)


def _closed_loop(finals):
    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words = Words(one)
    flown, said, forbidden, speaker = speak_and_fly(_speaker_model(words), [signals], [geometry], inputs, runways,
                                                    charts, approach, [200.0], words, _params(), None,
                                                    generator=torch.Generator().manual_seed(2), temperature=1.0,
                                                    finals=finals)
    return one, geometry, signals, words, flown, said[0], forbidden, speaker


def test_a_closed_loop_speaks_above_the_edge_and_stops_at_the_first_step_that_ends_below_it():
    # 2 km up: the aircraft is far below it once it turns onto the final (in the first 200 s)
    final = _final(crossing_m=2_000.0, faf_d_m=12_000.0)
    one, geometry, signals, words, flown, grid, forbidden, speaker = _closed_loop([(final,)])
    assert forbidden[ALTITUDE].shape == (1, len(grid)) and forbidden[ALTITUDE].max() > 0
    force = in_force(grid)
    for k in range(len(grid)):                                   # every word in force was allowed where it was said from
        row = N_LOOK + k
        assert altitude_word_allowed(final, force[k:k + 1, ALTITUDE], speaker.e[0, row:row + 1],
                                     speaker.n[0, row:row + 1], speaker.h[0, row:row + 1], words)[0]
    # the post-hoc scan stops where a check in the loop would have: the speaker's row after step k is the state the
    # scan reads for step k, and the stop is the first such row below the edge
    stops = glidepath_stops(flown, [grid], [geometry], [(final,)], words)
    step = int(stops.step[0])
    assert step >= 0 and stops.row[0] == step
    rows = N_LOOK + 1 + np.arange(step + 1)
    below, _ = below_floor(final, speaker.e[0, rows], speaker.n[0, rows], speaker.h[0, rows], one)
    assert np.flatnonzero(below).tolist()[:1] == [step]
    # the stopped sentence ends at its stop, whatever the executor made of the rest
    reading = read_flight(signals, geometry, one, words)
    batch = replay.Batch(signals=[signals], series=[], readings=[reading], geometries=[geometry], crossing_heights=[],
                         approach_ias_mps=[], groups=[], drawn={})
    row = flight_rows(batch, flown, [grid], words, "prior", [0], None, stops)[0]
    assert row["outcome"] == BELOW_GLIDEPATH and row["steps_said"] == step + 1
    assert row["end_s"] == (step + 1) * one.step_s
    assert flight_rows(batch, flown, [grid], words, "prior", [0], None)[0]["outcome"] != BELOW_GLIDEPATH


def test_an_edge_that_never_binds_leaves_the_closed_loop_draw_for_draw_as_without_one():
    low = _final(crossing_m=-5_000.0, faf_d_m=12_000.0)
    *_, flown, grid, forbidden, _ = _closed_loop([(low,)])
    *_, flown_bare, grid_bare, forbidden_bare, _ = _closed_loop(None)
    assert np.array_equal(grid, grid_bare) and torch.equal(flown.states, flown_bare.states)
    assert ALTITUDE not in forbidden_bare and not forbidden[ALTITUDE].any()
    stops = glidepath_stops(flown, [grid], [instruction_airport()], [(low,)], Words(instruction_spec()))
    assert stops.step.tolist() == [-1] and stops.row.tolist() == [-1]
