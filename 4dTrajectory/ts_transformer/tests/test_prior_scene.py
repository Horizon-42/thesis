"""Scenes, the landing context (`prior.scene`) and the step-0 census runner."""

import json

import numpy as np
import pytest

from ts_transformer.prior.scene import (
    CONTEXT_WINDOW_S, LEADER_RANGE_M, N_LOOK, Landings, Presence, SceneIndex, context_landings, presence, utc_s,
)
from ts_transformer.tests.support import fixture_days, fly_legs, instruction_airport, instruction_flight, landing_on


def _place(name: str, start: float, end: float, runway: str = "09", landing: float | None = None,
           speaking: bool = True, to_go: tuple[float, float] = (20_000.0, 0.0)) -> Presence:
    """A flight present over ``[start, end]`` (2 s rows), flying straight in from ``to_go[0]`` to ``to_go[1]`` metres."""
    times = np.arange(start, end + 1e-9, 2.0)
    return Presence(name, "KXXX", runway, end if landing is None else landing, speaking, times,
                    np.linspace(to_go[0], to_go[1], len(times)))


def test_the_scene_index_finds_overlapping_flights_and_chains_segments():
    flights = [_place("a", 0.0, 100.0), _place("b", 50.0, 400.0), _place("c", 390.0, 500.0), _place("d", 600.0, 700.0)]
    index = SceneIndex(reversed(flights))
    assert [p.dataset_id for p in index.overlapping(95.0, 120.0)] == ["a", "b"]
    assert [p.dataset_id for p in index.overlapping(450.0, 450.0)] == ["c"]
    # a long flight that started well before the window is still found
    assert [p.dataset_id for p in index.overlapping(300.0, 300.0)] == ["b"]
    assert [[p.dataset_id for p in s] for s in index.segments()] == [["a", "b", "c"], ["d"]]


def _roster(tmp_path, rows):
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")
    return path


def test_the_landing_context_leaves_out_test_days_other_runways_and_non_landings(tmp_path):
    days = fixture_days()
    rows = [{"outcome": "assigned", "runway": "09", "landing_time_utc": landing_on("train"), "flight_key": "a"},
            {"outcome": "assigned", "runway": "27", "landing_time_utc": landing_on("val"), "flight_key": "b"},
            {"outcome": "assigned", "runway": "09", "landing_time_utc": landing_on("test"), "flight_key": "c"},
            {"outcome": "assigned", "runway": "18", "landing_time_utc": landing_on("train"), "flight_key": "d"},
            {"outcome": "not_landing", "runway": None, "landing_time_utc": None, "flight_key": "e"}]
    pool = context_landings(_roster(tmp_path, rows), ["09", "27"], days)
    landings = pool.landings()
    assert pool.sealed == 1 and {r: len(t) for r, t in landings.by_runway.items()} == {"09": 1, "27": 1}
    assert context_landings(_roster(tmp_path, rows[:1]), ["09", "27"], days).landings().by_runway["27"].tolist() == []
    assert [r.split for r in pool.records] == sorted(["train", "val"], key=lambda s: landing_on(s))
    assert landings.times_s.tolist() == sorted(np.concatenate(list(landings.by_runway.values())).tolist())
    t = utc_s(landing_on("train"))
    # strictly before t: the landing at t itself is not yet seen; one second later it is
    assert landings.count_before(np.array([t, t + 1.0, t + CONTEXT_WINDOW_S + 1.0]), CONTEXT_WINDOW_S).tolist() == [0, 1, 0]
    assert landings.count_before(np.array([t + 1.0]), CONTEXT_WINDOW_S, "27").tolist() == [0]
    with pytest.raises(KeyError):                                     # not a candidate
        landings.count_before(np.array([t + 1.0]), CONTEXT_WINDOW_S, "18")
    since = landings.since_last(np.array([t, t + 5.0]), "09")
    assert np.isnan(since[0]) and since[1] == 5.0
    empty = context_landings(_roster(tmp_path, rows[:1]), ["09", "27"], days).landings()
    assert np.isnan(empty.since_last(np.array([t + 5.0]), "27")).all()


def test_a_flight_is_present_from_its_entry_to_its_last_sentence_row():
    flight = instruction_flight(*fly_legs([(30, 0.0, 70.0, 0.0)], 90.0, 500.0, -300.0, 0.0), dataset_id="KXXX:a")
    start = utc_s(flight.entry_time_utc)
    spoken = presence(flight, 10, instruction_airport(), 2.0)
    assert (spoken.start_s, spoken.end_s, spoken.speaking) == (start, start + flight.time_s[9], True)
    silent = presence(flight, None, instruction_airport(), 2.0)
    assert (silent.end_s, silent.speaking) == (start + flight.time_s[-1], False)
    assert silent.landing_s == utc_s(flight.landing_time_utc)
    # it ends 300 m before runway 09's threshold, flying east along the centreline (fly_legs' end point)
    threshold = instruction_airport().candidates[instruction_airport().candidate_index("09")]
    assert silent.to_threshold_m[-1] == pytest.approx(np.hypot(-300.0 - threshold.threshold_e_m, threshold.threshold_n_m))
    assert silent.to_threshold_at(np.array([start + 1.0]))[0] == pytest.approx(silent.to_threshold_m[:2].mean())


def test_the_census_counts_others_leaders_and_the_landing_window():
    from ts_transformer.experiments.prior_scene_census import census_airport

    ego = _place("ego", 1000.0, 1100.0, landing=1100.0, to_go=(20_000.0, 0.0))
    # the others are background aircraft, so the ego's steps are the only predicted steps
    ahead = _place("ahead", 900.0, 1050.0, landing=1050.0, to_go=(20_000.0, 0.0), speaking=False)  # lands first
    far = _place("far", 1000.0, 1060.0, landing=1060.0, to_go=(8_000.0, 0.0), speaking=False)     # > 10 km ahead early
    behind = _place("behind", 1040.0, 1300.0, landing=1300.0, speaking=False)                      # lands after the ego
    other_runway = _place("other", 1000.0, 1100.0, runway="27", landing=1080.0, speaking=False)
    rows = ego.times_s[N_LOOK:]
    ego_to_go = ego.to_threshold_m[N_LOOK:]
    landings = Landings(np.array([1016.0 - CONTEXT_WINDOW_S + 10.0]), {"09": np.array([1016.0 - CONTEXT_WINDOW_S + 10.0])})
    none = np.zeros((0, 2))
    out = census_airport([ego], [ahead, far, behind, other_runway], landings, none)
    assert out["predicted_steps"] == len(rows)
    gaps = []
    for p in (ahead, far):
        here = (rows >= p.start_s) & (rows <= p.end_s)
        gap = np.where(here, ego_to_go - p.to_threshold_at(rows), np.inf)
        gaps.append(np.where((gap > 0) & (gap <= LEADER_RANGE_M), gap, np.inf))
    nearest = np.minimum(*gaps)
    assert 0 < np.isfinite(nearest).sum() < len(rows)          # the rule both finds and rejects leaders here
    assert out["leader_share"] == pytest.approx(np.isfinite(nearest).mean())
    assert out["leader_gap_m"]["p50"] == pytest.approx(np.median(nearest[np.isfinite(nearest)]))
    # others: 'other' all along, 'ahead' until 1050, 'far' until 1060, 'behind' from 1040
    total = 1 + (rows <= 1050.0) + (rows <= 1060.0) + (rows >= 1040.0)
    assert out["others_total"]["mean"] == pytest.approx(total.mean())
    assert out["other_speaking"]["max"] == 0.0
    assert out["landing_in_window_share"] == pytest.approx((rows <= 1016.0 + 10.0).mean())
    assert out["window_reaches_a_test_day_share"] == 0.0
    assert out["segments"]["count"] == 1 and out["segments"]["flights"]["max"] == 5


def test_the_census_sees_a_window_reach_back_into_a_test_day():
    from ts_transformer.data.day_split import operating_day_span_s
    from ts_transformer.experiments.prior_scene_census import census_airport

    test_start, test_end = operating_day_span_s("2026-06-01")          # 2026-06-01T09Z … 06-02T09Z
    start = test_end + 600.0                                           # ten minutes into the next day
    ego = _place("ego", start, start + 2000.0)
    landings = Landings(np.zeros(0), {"09": np.zeros(0)})
    out = census_airport([ego], [], landings, np.array([[test_start, test_end]]))
    rows = ego.times_s[N_LOOK:]
    assert out["window_reaches_a_test_day_share"] == pytest.approx((rows - CONTEXT_WINDOW_S < test_end).mean())
    assert 0.0 < out["window_reaches_a_test_day_share"] < 1.0


def test_the_census_runner_reads_the_train_days_of_an_artefact(tmp_path, monkeypatch):
    from ts_transformer.experiments import prior_scene_census
    from ts_transformer.instructions.artefact import (
        labeller_source_sha256, write_candidates, write_sentences, write_signals, write_spec,
    )
    from ts_transformer.instructions.labeller.read import read_flight
    from ts_transformer.tests.support import instruction_spec

    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    flights = [instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0), dataset_id=f"KXXX:{name}")
               for name in "ab"]
    directory = tmp_path / "artefact"
    directory.mkdir()
    write_signals(directory, {"train": flights}, {"counts": {"train": {"built_usable": 2}},
                                                  "test_days": {"flights_not_opened": 0}}, fixture_days())
    write_candidates(directory, {"KXXX": instruction_airport()})
    spec = instruction_spec()
    write_spec(directory, spec, {"n": 1}, {"labeller_source_sha256": labeller_source_sha256(),
                                           "git": {"head": "test", "dirty": False}})
    write_sentences(directory, "train", spec, [read_flight(flights[0], instruction_airport(), spec)], [0])
    roster = _roster(tmp_path, [{"outcome": "assigned", "runway": "09", "landing_time_utc": landing_on("test"),
                                 "flight_key": "t"}])
    monkeypatch.setattr(prior_scene_census, "tracks_manifest_path", lambda code: roster)
    assert prior_scene_census.main(["--instructions", str(directory), "--out", str(tmp_path / "census")]) == 0
    census = json.loads((tmp_path / "census" / "census.json").read_text(encoding="utf-8"))
    airport = census["airports"]["KXXX"]
    assert airport["flights"] == {"speaking": 1, "background": 1}
    # the two fixture flights fly the same minutes: the refused one is background at every step
    assert airport["other_background"]["histogram"]["1"] == airport["predicted_steps"]
    assert census["landing_context"]["KXXX"]["sealed_test_day_landings_left_out"] == 1
    assert census["landing_context"]["KXXX"]["landings_kept"] == 0 and airport["landing_in_window_share"] == 0.0
    with pytest.raises(SystemExit):
        prior_scene_census.main(["--instructions", str(directory), "--out", str(tmp_path / "census")])



# ---- several aircraft on the scene's steps (multi-aircraft design §2.1, §2.3)
def _minutes(name: str, start: float, end: float, speaking: bool = True):
    from ts_transformer.prior.scene import Presence

    times = np.arange(start * 60.0, end * 60.0 + 1e-9, 2.0)
    return Presence(name, "KXXX", "05", end * 60.0 + 30.0, speaking, times, np.zeros(len(times)))


def test_the_steps_are_whole_multiples_and_a_flight_off_them_is_refused_by_name():
    import dataclasses

    from ts_transformer.prior.scene import scene_steps

    assert list(scene_steps(3.4, 10.1, 2.0)) == [4.0, 6.0, 8.0, 10.0]
    flight = instruction_flight(*fly_legs([(30, 0.0, 70.0, 0.0)], 90.0, 500.0, -300.0, 0.0), dataset_id="KXXX:a")
    assert utc_s(flight.entry_time_utc) % 2.0 == 0.0          # the fixture's rows are on the steps
    seen = presence(flight, None, instruction_airport(), 2.0)
    assert np.array_equal(seen.times_s, utc_s(flight.entry_time_utc) + 2.0 * np.arange(flight.n_rows))
    # rows cut at a flight's own first sample (an artefact before 2026-10-02), or not one step apart
    late = dataclasses.replace(flight, entry_time_utc=flight.entry_time_utc.replace(":00Z", ":01.250Z"))
    with pytest.raises(ValueError, match="not on the scene's 2 s steps"):
        presence(late, None, instruction_airport(), 2.0)
    with pytest.raises(ValueError, match="not on the scene's 2 s steps"):
        presence(dataclasses.replace(flight, time_s=flight.time_s * 1.5), None, instruction_airport(), 2.0)


def test_a_segment_is_cut_as_the_design_draws_it():
    """Figure 1's made-up segment (experiments/scene_sample_figure): cuts at 15:44 and 31:50, D and H carried."""
    from ts_transformer.prior.scene import samples

    segment = [_minutes("A", 0.0, 9.0), _minutes("B", 3.0, 12.5), _minutes("C", 7.0, 15.7), _minutes("D", 12.0, 21.0),
               _minutes("E", 18.0, 27.0), _minutes("F", 19.5, 28.5), _minutes("X", 21.5, 26.0, speaking=False),
               _minutes("G", 23.0, 31.8), _minutes("H", 28.0, 37.5), _minutes("I", 33.5, 42.0),
               _minutes("J", 37.0, 46.0), _minutes("K", 40.0, 49.0)]
    cut = samples(segment, 2.0)
    assert [s.loss_start_s for s in cut] == [0.0, 944.0, 1910.0] and cut[-1].loss_end_s == 49 * 60.0
    assert [[f.dataset_id for f in s.carried] for s in cut] == [[], ["D"], ["H"]]
    assert [len(s.flights) for s in cut] == [4, 6, 4]            # every flight with a step in the loss span


def test_a_short_segment_is_one_sample_and_the_cut_step_is_the_later_samples():
    from ts_transformer.prior.scene import samples

    assert len(samples([_minutes("A", 0.0, 20.0)], 2.0)) == 1
    # two aircraft through the whole window (one entering at minute 15): every step ties, the earliest is taken, and the
    # one whose first row is on the cut is only in the later sample, not carried
    cut = samples([_minutes("A", 0.0, 30.0), _minutes("E", 15.0, 30.0)], 2.0)
    assert [s.loss_start_s for s in cut] == [0.0, 900.0]
    assert [f.dataset_id for f in cut[0].flights] == ["A"]
    assert ([f.dataset_id for f in cut[1].flights], [f.dataset_id for f in cut[1].carried]) == (["A", "E"], ["A"])
