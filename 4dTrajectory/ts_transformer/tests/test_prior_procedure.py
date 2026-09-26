"""The procedure's altitudes (post-training design §3, `prior.procedure`): where the glidepath lower edge binds, the join
and the dip, the word rules 1–5, the flown-track check, the readouts before the join, the final read from the coded
approach, the mirrored constants; the masks in the speaker (`prior.generate`), the stop in the closed loop
(`experiments.prior_free_generation.glidepath_stops`), and the stage-0 check's pieces
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
    BELOW_GLIDEPATH, ProcedureMasks, flight_rows, glidepath_stops, in_force, sentence_pre_join, speak_and_fly,
)
from ts_transformer.experiments.prior_procedure_check import (
    MAX_FORBIDDEN_WORDS, MAX_STOPPED_REPLAYS, check_labelled, labelled_rows, passes,
)
from ts_transformer.instructions.airport import RunwayCandidate
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, APPROACH, APPROACH_CLEARED, APPROACH_GO_AROUND, APPROACH_NOT_CLEARED, HEADING, RUNWAY, UNCHANGED,
    Words,
)
from ts_transformer.prior import procedure
from ts_transformer.prior.generate import Speaker
from ts_transformer.prior.mva import MvaChart
from ts_transformer.prior.procedure import (
    GLIDEPATH_BELOW_M, THRESHOLD_TOLERANCE_M, RunwayProcedure, airport_procedures, altitude_word_allowed,
    angle_word_allowed, below_floor, climb_barred, pre_join, pre_join_readout, runway_procedure, track_tolerance_m,
    word_tolerance_m,
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


#: A chart with no sector: no point has an MVA.
NO_CHART = MvaChart("none", ())


def _final(crossing_m: float = 115.0, candidate: RunwayCandidate = RUNWAY_09, faf_d_m: float = FAF_D_M,
           cone: FasCourseGeometry | None = None, decision_m: float | None = None) -> RunwayProcedure:
    """A final on runway 09's course; its DA 46 m above the crossing point unless given (the TCH 15 m, a DA 61 m up)."""
    return RunwayProcedure(candidate=candidate, crossing_m=crossing_m, glidepath_tan=TAN, faf_d_m=faf_d_m,
                           cone=cone or fas_course_geometry(candidate.length_m),
                           decision_m=crossing_m + 46.0 if decision_m is None else decision_m)


#: Rules 3 and 4 off: joined, the climb not barred.
FINAL_ONLY = {"joined": np.array(True), "barred": np.array(False)}


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
    assert altitude_word_allowed(final, np.array([20, 21, 19]), e, n, np.zeros(1), words,
                                 **FINAL_ONLY).tolist() == [True, True, False]
    land = np.array([words.altitude_land])
    assert altitude_word_allowed(final, land, e, n, np.array([floor - tolerance + 0.1]), words, **FINAL_ONLY)[0]
    assert not altitude_word_allowed(final, land, e, n, np.array([floor - tolerance - 0.1]), words, **FINAL_ONLY)[0]
    # outside the FAF there is no floor: any word
    far = _at(FAF_D_M + 500.0)
    assert altitude_word_allowed(final, np.array([0, words.altitude_land]), *far, np.zeros(1), words,
                                 **FINAL_ONLY).all()


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
    runway = SimpleNamespace(ident="09", threshold_crossing_height_m=15.0, published_glidepath_deg=3.0,
                             decision_height_above_threshold_m=61.0)
    monkeypatch.setattr(procedure, "procedure_skeleton", lambda code, ident, root: _skeleton(geometry, 0.5, 9_800.0))
    final = runway_procedure(geometry, candidate, runway)
    assert final.faf_d_m == pytest.approx(9_799.5, abs=1e-6) and final.ident == "09"   # measured from the candidate
    assert final.crossing_m == candidate.elevation_m + 15.0
    assert final.glidepath_tan == pytest.approx(TAN)
    assert final.cone == fas_course_geometry(candidate.length_m)
    assert final.decision_m == candidate.elevation_m + 61.0
    assert final.entry_m == pytest.approx(final.crossing_m + final.faf_d_m * TAN)
    # a document whose threshold is another runway's is refused, and so is a runway without a published glidepath
    far = THRESHOLD_TOLERANCE_M + 1.0
    monkeypatch.setattr(procedure, "procedure_skeleton", lambda code, ident, root: _skeleton(geometry, far, 9_800.0))
    with pytest.raises(ValueError, match="from the candidate's"):
        runway_procedure(geometry, candidate, runway)
    monkeypatch.setattr(procedure, "procedure_skeleton", lambda code, ident, root: _skeleton(geometry, 0.0, 9_800.0))
    with pytest.raises(ValueError, match="no threshold crossing height or glidepath"):
        runway_procedure(geometry, candidate, SimpleNamespace(ident="09", threshold_crossing_height_m=15.0,
                                                              published_glidepath_deg=None,
                                                              decision_height_above_threshold_m=61.0))


def test_an_airport_s_finals_follow_the_runway_pointer_whatever_order_the_runways_come_in(monkeypatch):
    geometry = _two_runways()                                    # 09 at the origin, 27 3 km east flown westbound
    skeletons = {"09": _skeleton(geometry, 0.0, 9_000.0), "27": _skeleton(geometry, 3_000.0, 11_000.0, 270.0, 500.0)}
    monkeypatch.setattr(procedure, "procedure_skeleton", lambda code, ident, root: skeletons[ident])
    runways = [SimpleNamespace(ident=ident, threshold_crossing_height_m=tch, published_glidepath_deg=3.0,
                               decision_height_above_threshold_m=61.0)
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
        joined, dipped = pre_join(final, -d, np.zeros(rows), heights, spec)
        return check_labelled({"e": -d, "n": np.zeros(rows), "h": heights, "in_force": force, "said": said,
                               "angle_in_force": np.full(rows, 1), "angle_said": np.full(rows, UNCHANGED),
                               "approach": np.full(rows, APPROACH_CLEARED), "joined": joined, "dipped": dipped,
                               "row": np.arange(rows), "flight": np.zeros(rows, dtype=int),
                               "pre_join": np.array([{"under_decision": False, "climbed_after_dip": False,
                                                      "under_mva": True}], dtype=object)}, final, words)

    on = check(h)
    assert on["words"] == {"level": 1, "land": 1, "angle": 1, "level_forbidden_edge": 1, "land_forbidden_edge": 0,
                           "level_forbidden_decision": 0, "level_forbidden_climb": 0, "angle_forbidden_climb": 0,
                           "forbidden": 1}
    assert on["observed_pre_join"] == {"under_decision": 0, "climbed_after_dip": 0, "under_mva": 1}
    # the level word in force is forbidden at the one silent step before the landing word replaces it
    assert on["forced_steps"] == 1 and on["steps"] == 3 and on["flights_forced"] == 1
    assert on["observed_below"] == 0
    low = check(h - GLIDEPATH_BELOW_M - track_tolerance_m(spec) - 1.0)
    assert low["observed_below"] == 1 and low["observed_depth_m"][0] == pytest.approx(track_tolerance_m(spec) + 1.0)
    assert low["words"]["land_forbidden_edge"] == 1


def test_a_labelled_flight_s_rows_are_its_signals_first_rows(monkeypatch):
    """Contract C30: a sentence's words line up with the FIRST ``len(words)`` rows of its signals."""
    words = Words(instruction_spec())
    rows, extra = N_LOOK + 3, 5
    flights = [SimpleNamespace(airport="KXXX", e_m=np.arange(rows + extra) + 100.0 * k, n_m=np.zeros(rows + extra),
                               altitude_m=np.full(rows + extra, 500.0 + k)) for k in range(2)]
    grid = np.full((rows, 6), UNCHANGED, dtype=np.int16)
    grid[0] = 3
    grid[0, RUNWAY] = 0                                        # the one candidate
    sentences = {"signal_index": np.array([1, 0]), "offsets": np.array([0, rows, 2 * rows]),
                 "words": np.concatenate([grid, grid]), "runway_index": np.array([0, 0])}
    monkeypatch.setattr(prior_procedure_check, "load_signals", lambda directory, split: flights)
    monkeypatch.setattr(prior_procedure_check, "load_sentences", lambda directory, split, spec: sentences)
    monkeypatch.setattr(prior_procedure_check, "load_candidates", lambda directory: {"KXXX": instruction_airport()})
    masks = ProcedureMasks({"KXXX": (_final(),)}, {"KXXX": NO_CHART})
    pooled = labelled_rows("train", None, words, masks)[("KXXX", 0)]
    assert pooled["e"].tolist() == [*(np.arange(rows) + 100.0), *np.arange(rows)]      # sentence 0 is signal 1
    assert pooled["h"].tolist() == [501.0] * rows + [500.0] * rows
    assert pooled["flight"].tolist() == [0] * rows + [1] * rows and pooled["row"].tolist() == [*range(rows)] * 2
    assert len(pooled["pre_join"]) == 2 and pooled["pre_join"][0]["mva_under_m"] is None       # one per flight


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
    nowhere = _final(candidate=geometry.candidates[1], crossing_m=110.0, faf_d_m=0.0, decision_m=-1_000.0)
    floor = float(wide.floor_m(np.array([e]), np.array([n]))[0])
    assert floor > signals.altitude_m[N_LOOK] + 200.0
    speaker = Speaker(_prior_model(variant="no-context"), [signals] * 2, [geometry] * 2, None, words,
                      max_rows=N_LOOK + 3, generator=torch.Generator().manual_seed(0), finals=[(wide, nowhere)] * 2)
    classes = words.n_altitude_levels + 2
    chosen = np.zeros((2, 6), dtype=np.int64)
    chosen[:, RUNWAY] = [1, 2]                                    # this step's runway: 09, then 27
    chosen[:, APPROACH] = 1 + APPROACH_CLEARED
    out = speaker._procedure_altitude(chosen, True, classes)
    levels = np.arange(words.n_altitude_levels) * words.spec.altitude_step_m
    assert np.array_equal(out[0, 1:-1], levels >= floor - word_tolerance_m(words.spec))
    assert not out[0, -1] and out[0, 0]                          # below the edge: no landing; "unchanged" not asked
    assert out[1].all()                                           # runway 27: no edge here, a DA under the ground
    # later steps: "unchanged" only where the word in force still holds, for the runway in force when none is said
    speaker.value[:, RUNWAY] = 1
    speaker.value[:, ALTITUDE] = [1 + words.altitude_index(600.0), 1 + words.altitude_index(3000.0)]
    speaker.value[:, APPROACH] = 1 + APPROACH_CLEARED
    later = speaker._procedure_altitude(np.zeros((2, 6), dtype=np.int64), False, classes)
    assert later[:, 0].tolist() == [False, True]
    with pytest.raises(ValueError, match="flights' finals"):
        Speaker(_prior_model(variant="no-context"), [signals] * 2, [geometry] * 2, None, words, max_rows=N_LOOK + 3,
                generator=torch.Generator(), finals=[(wide, nowhere)])
    with pytest.raises(ValueError, match="in the pointer's order"):
        Speaker(_prior_model(variant="no-context"), [signals] * 2, [geometry] * 2, None, words, max_rows=N_LOOK + 3,
                generator=torch.Generator(), finals=[(nowhere, wide)] * 2)


def _closed_loop(finals, seed: int = 2):
    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words = Words(one)
    flown, said, forbidden, speaker = speak_and_fly(_speaker_model(words), [signals], [geometry], inputs, runways,
                                                    charts, approach, [200.0], words, _params(), None,
                                                    generator=torch.Generator().manual_seed(seed), temperature=1.0,
                                                    finals=finals)
    return one, geometry, signals, words, flown, said[0], forbidden, speaker


def test_a_closed_loop_speaks_above_the_edge_and_stops_at_the_first_step_that_ends_below_it():
    # 2 km up: the aircraft is far below it once it turns onto the final (in the first 200 s); a DA under the ground.
    # Under the entry height from the start, it may not climb before the join: the untrained model's draws with seed 8
    # take it into the cone (step 65), most seeds' do not
    final = _final(crossing_m=2_000.0, faf_d_m=12_000.0, decision_m=-1_000.0)
    one, geometry, signals, words, flown, grid, forbidden, speaker = _closed_loop([(final,)], seed=8)
    assert forbidden[ALTITUDE].shape == (1, len(grid)) and forbidden[ALTITUDE].max() > 0
    force = in_force(grid)
    joined, dipped = pre_join(final, speaker.e[0], speaker.n[0], speaker.h[0], one)
    # the speaker's state, kept a row at a time, is the function's over its rows
    assert (speaker.joined[0, 0], speaker.dipped[0, 0]) == (joined[speaker.rows - 1], dipped[speaker.rows - 1])
    for k in range(len(grid)):                                   # every word in force was allowed where it was said from
        row = N_LOOK + k
        barred = climb_barred(joined[row], dipped[row], force[k, APPROACH])
        assert altitude_word_allowed(final, force[k:k + 1, ALTITUDE], speaker.e[0, row:row + 1],
                                     speaker.n[0, row:row + 1], speaker.h[0, row:row + 1], words,
                                     joined=joined[row], barred=barred)[0]
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
    batch = replay.Batch(signals=[signals], series=[], readings=[reading], geometries=[geometry], vertical_paths=[],
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


def _approach(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions ``d`` before runway 09's threshold on its centreline."""
    return -np.asarray(d, dtype=np.float64), np.zeros(len(d))


def test_the_join_is_the_first_row_in_the_region_and_a_dip_counts_only_before_it():
    spec = instruction_spec()
    final = _final()
    entry = final.entry_m
    assert entry == pytest.approx(115.0 + FAF_D_M * TAN)
    d = np.array([20_000.0, 16_000.0, 12_000.0, 9_000.0, 6_000.0])       # joins at the fourth row
    e, n = _approach(d)
    above = np.array([entry + 100.0, entry + 50.0, entry, entry - 20.0, entry - 300.0])
    joined, dipped = pre_join(final, e, n, above, spec)
    assert joined.tolist() == [False, False, False, True, True]
    assert not dipped.any()                                 # low only from the join on: no dip
    low = above.copy()
    low[1] = entry - word_tolerance_m(spec) - 1.0           # under the entry height less the tolerance, before the join
    joined, dipped = pre_join(final, e, n, low, spec)
    assert dipped.tolist() == [False, True, True, True, True]
    edge = above.copy()
    edge[1] = entry - word_tolerance_m(spec) + 1.0          # within the tolerance: no dip
    assert not pre_join(final, e, n, edge, spec)[1].any()


def test_the_climb_is_barred_after_a_dip_before_the_join_unless_a_go_around_is_in_force():
    joined = np.array([False, False, True])
    dipped = np.array([False, True, True])
    approach = np.array([APPROACH_NOT_CLEARED, APPROACH_CLEARED, APPROACH_CLEARED])
    assert climb_barred(joined, dipped, approach).tolist() == [False, True, False]
    assert not climb_barred(np.array(False), np.array(True), np.array(APPROACH_GO_AROUND))


def test_before_the_join_a_level_under_the_decision_altitude_is_forbidden_and_landing_is_not():
    spec = instruction_spec()
    words = Words(spec)
    step, tolerance = spec.altitude_step_m, word_tolerance_m(spec)
    final = _final(decision_m=10 * step)                   # the DA on a step
    e, n = _at(FAF_D_M + 5_000.0)                            # outside the FAF: no glidepath edge here
    levels = np.array([10, 9, words.altitude_land])
    assert 9 * step < final.decision_m - tolerance
    free = {"barred": np.array(False)}
    assert altitude_word_allowed(final, levels, e, n, np.zeros(1), words, joined=np.array(False),
                                 **free).tolist() == [True, False, True]
    assert altitude_word_allowed(final, levels, e, n, np.zeros(1), words, joined=np.array(True), **free).all()


def test_after_a_dip_no_level_above_the_aircraft_and_no_climb_class_may_be_said():
    spec = instruction_spec()
    words = Words(spec)
    step, tolerance = spec.altitude_step_m, word_tolerance_m(spec)
    final = _final(decision_m=-1_000.0)
    e, n = _at(FAF_D_M + 5_000.0)
    h = np.array([20 * step])
    levels = np.array([20, 21, 19, words.altitude_land])
    barred = {"joined": np.array(False), "barred": np.array(True)}
    assert altitude_word_allowed(final, levels, e, n, h, words, **barred).tolist() == [True, False, True, True]
    assert altitude_word_allowed(final, levels, e, n, h - tolerance - 0.1, words, **barred).tolist() == [
        False, False, True, True]                             # 20 × step is now more than the tolerance above
    assert altitude_word_allowed(final, levels, e, n, h, words, joined=np.array(False), barred=np.array(False)).all()
    angles = np.arange(words.angle_climb + 1)
    assert angle_word_allowed(angles, np.array(True), words).tolist() == [True] * words.angle_climb + [False]
    assert angle_word_allowed(angles, np.array(False), words).all()


def test_the_speaker_masks_the_climb_once_the_observed_rows_dipped_under_the_entry_height():
    spec = instruction_spec()
    words, geometry = Words(spec), _two_runways()
    signals = _signals(N_LOOK + 3)
    h = float(signals.altitude_m[N_LOOK])
    # runway 09's FAF far out and its entry height well above the aircraft; the cone too narrow to hold it; DA below
    high = _final(crossing_m=h + 400.0 - 30_000.0 * TAN, faf_d_m=30_000.0, decision_m=h - 200.0)
    nowhere = _final(candidate=geometry.candidates[1], crossing_m=110.0, faf_d_m=0.0, decision_m=-1_000.0)
    assert high.entry_m > h + 100.0
    speaker = Speaker(_prior_model(variant="no-context"), [signals], [geometry], None, words, max_rows=N_LOOK + 3,
                      generator=torch.Generator().manual_seed(0), finals=[(high, nowhere)])
    rows = N_LOOK + 1
    joined, dipped = pre_join(high, speaker.e[0, :rows], speaker.n[0, :rows], speaker.h[0, :rows], spec)
    assert speaker.joined[0, 0] == joined[-1] and speaker.dipped[0, 0] == dipped[-1]
    assert not joined[-1] and dipped[-1]
    classes = words.n_altitude_levels + 2
    chosen = np.zeros((1, 6), dtype=np.int64)
    chosen[0, RUNWAY], chosen[0, APPROACH] = 1, 1 + APPROACH_CLEARED
    out = speaker._procedure_altitude(chosen, True, classes)
    levels = np.arange(words.n_altitude_levels) * spec.altitude_step_m
    tolerance = word_tolerance_m(spec)
    assert np.array_equal(out[0, 1:-1], (levels <= h + tolerance) & (levels >= high.decision_m - tolerance))
    assert out[0, -1]                                         # "descend to land" stays
    chosen[0, HEADING], chosen[0, ALTITUDE] = 1, 1 + words.altitude_index(h)     # the columns before the angle
    angle = speaker._allowed(ANGLE, chosen, True, words.angle_climb + 2, np.zeros(1, dtype=bool))
    assert not angle[0, 1 + words.angle_climb]
    chosen[0, APPROACH] = 1 + APPROACH_GO_AROUND              # a go-around may climb
    assert speaker._procedure_altitude(chosen, True, classes)[0, 1:-1][levels > h + 100.0].all()


def test_the_readout_before_the_join_reads_the_dip_the_decision_altitude_and_the_mva_where_not_cleared():
    spec = instruction_spec()
    final = _final(decision_m=300.0)
    entry = final.entry_m
    d = np.linspace(24_000.0, 8_000.0, N_LOOK + 9)            # joins at the last rows
    e, n = _approach(d)
    h = np.full(len(d), entry + 200.0)
    h[N_LOOK + 1] = entry - 50.0                              # the dip
    h[N_LOOK + 4] = 300.0 - 60.0                              # 60 m under the DA, the lowest since the dip
    runway = np.zeros(len(d), dtype=np.int64)
    approach = np.full(len(d), APPROACH_NOT_CLEARED)
    approach[N_LOOK + 5:] = APPROACH_CLEARED
    mva = np.full(len(d), np.nan)
    mva[N_LOOK + 2] = h[N_LOOK + 2] + 70.0                    # 70 m under the MVA, not cleared
    mva[N_LOOK + 6] = h[N_LOOK + 6] + 500.0                   # cleared: not read
    got = pre_join_readout((final,), runway, approach, e, n, h, mva, N_LOOK, spec)
    assert got["decision_under_m"] == pytest.approx(60.0) and got["under_decision"]
    assert got["climb_after_dip_m"] == pytest.approx(entry + 200.0 - (300.0 - 60.0))    # the rows after the lowest
    assert got["climbed_after_dip"]
    assert got["mva_under_m"] == pytest.approx(70.0) and got["under_mva"]
    # rows before the first predicted step only set the dip; none of them is read
    early = h.copy()
    early[2] = 300.0 - 500.0
    assert pre_join_readout((final,), runway, approach, e, n, early, mva, N_LOOK, spec)["decision_under_m"] \
        == pytest.approx(60.0)
    quiet = pre_join_readout((final,), runway, approach, e, n, np.full(len(d), entry + 200.0), np.full(len(d), np.nan),
                             N_LOOK, spec)
    assert quiet["climb_after_dip_m"] is None and quiet["mva_under_m"] is None and not quiet["under_decision"]


def test_a_sentence_is_read_before_the_join_on_the_rows_the_speaker_read_with_the_words_of_each_step():
    spec = instruction_spec()
    words = Words(spec)
    final = _final(decision_m=300.0)
    steps = 4
    d = np.linspace(24_000.0, 20_000.0, N_LOOK + steps + 3)     # rows past the sentence's steps are not read
    e, n = _approach(d)
    h = np.full(len(d), final.entry_m + 200.0)
    h[N_LOOK + steps:] = 0.0                                    # far under the DA, but after the last step said
    h[N_LOOK - 2] = 200.0                                       # 100 m under the DA, but before the first step
    grid = np.full((steps, 6), UNCHANGED)
    grid[0] = [0, APPROACH_NOT_CLEARED, 9, 20, 1, 3]
    grid[2, APPROACH] = APPROACH_CLEARED
    geometry = instruction_airport()
    got = sentence_pre_join(grid, steps, e, n, h, (final,), NO_CHART, geometry, words)
    assert got["decision_under_m"] == pytest.approx(300.0 - h[N_LOOK])
    # every row of the sentence under a 1 km MVA: only the rows before the clearance count, and they are all as deep
    chart = SimpleNamespace(at=lambda lon, lat: np.full(len(lon), 1_000.0))
    deep = sentence_pre_join(grid, steps, e, n, h, (final,), chart, geometry, words)
    assert deep["mva_under_m"] == pytest.approx(1_000.0 - h[N_LOOK]) and deep["under_mva"] == (
        1_000.0 - h[N_LOOK] > track_tolerance_m(spec))


def test_a_climb_is_read_only_inside_a_stretch_of_barred_rows_from_the_first_predicted_step():
    """The observed rows before the first predicted step set the dip but their climb is not the prior's; a go-around's
    climb ends the stretch, and the next one starts from where it resumes."""
    spec = instruction_spec()
    final = _final(decision_m=-1_000.0)
    entry = final.entry_m
    d = np.linspace(30_000.0, 20_000.0, N_LOOK + 6)            # never joins
    e, n = _approach(d)
    runway = np.zeros(len(d), dtype=np.int64)
    cleared = np.full(len(d), APPROACH_CLEARED)
    mva = np.full(len(d), np.nan)
    # the observed rows climb from well under the entry height; from the first predicted step the aircraft only descends
    h = np.concatenate([np.linspace(entry - 400.0, entry - 200.0, N_LOOK), entry - 190.0 - 10.0 * np.arange(6)])
    got = pre_join_readout((final,), runway, cleared, e, n, h, mva, N_LOOK, spec)
    assert got["climb_after_dip_m"] == pytest.approx(0.0) and not got["climbed_after_dip"]
    # a go-around between two barred stretches: its climb is not read, the next stretch starts where it resumes
    h = np.full(len(d), entry - 300.0)
    h[N_LOOK + 2: N_LOOK + 4] = entry - 100.0                   # climbing under the go-around
    h[N_LOOK + 4:] = [entry - 90.0, entry - 80.0]               # then 10 m more after it
    around = cleared.copy()
    around[N_LOOK + 2: N_LOOK + 4] = APPROACH_GO_AROUND
    got = pre_join_readout((final,), runway, around, e, n, h, mva, N_LOOK, spec)
    assert got["climb_after_dip_m"] == pytest.approx(10.0) and not got["climbed_after_dip"]
    # the same climb with no go-around is read
    got = pre_join_readout((final,), runway, cleared, e, n, h, mva, N_LOOK, spec)
    assert got["climb_after_dip_m"] == pytest.approx(220.0) and got["climbed_after_dip"]
