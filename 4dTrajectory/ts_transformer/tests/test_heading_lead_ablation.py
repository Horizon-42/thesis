"""The heading-lead ablation runner (`experiments/heading_lead_ablation.py`): what a cell changes, the reference and the
grid, relabelling, the two largest distances, the reproduction check and the summaries."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot.frame import ALT, GAMMA, LAT, LON, MASS, PSI, SPEED
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.experiments import heading_lead_ablation as ablation
from ts_transformer.instructions.words import HEADING, Words
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec


def _params(**changes) -> ExecutorParams:
    # method A's values from the test vocabulary (lead 4 s, bank limit 32°)
    return replace(ExecutorParams(cycle_s=1.0, heading_time_constant_s=4.0, bank_rate_deg_s=8.0, path_time_constant_s=2.0,
                                  path_rate_factor=2.0, timeout_factor=1.5, word_clock="track"), **changes)


def test_a_cell_moves_only_the_lead_and_the_bank_limit_and_derives_the_executors_turn_parameters():
    base, params = instruction_spec(), _params()
    cell = ablation.Cell(8.0, 25.0, None)
    vocabulary = cell.vocabulary(base)
    moved = {k for k, v in vocabulary.to_dict().items() if base.to_dict()[k] != v}
    assert moved == {"heading_lead_s", "turn_bank_max_deg"} and vocabulary.sha256 != base.sha256
    derived = cell.executor(params, vocabulary)
    assert derived == replace(params, heading_time_constant_s=8.0, bank_rate_deg_s=25.0 / 8.0)
    # p decoupled from the lead: τ_ψ still the lead
    decoupled = ablation.Cell(2.0, 32.0, 3.0)
    assert decoupled.executor(params, decoupled.vocabulary(base)) == replace(params, heading_time_constant_s=2.0,
                                                                             bank_rate_deg_s=3.0)
    assert (cell.name, decoupled.name) == ("L8_bank25_pderived", "L2_bank32_p3")
    # a lead under 2Δt leaves the executor's own turns faster than its loop: refused before anything is flown
    with pytest.raises(ValueError, match="lead is too short"):
        ablation.Cell(0.0, 32.0, None).executor(params, ablation.Cell(0.0, 32.0, None).vocabulary(base))


def test_the_reference_is_the_formal_values_first_and_the_grid_holds_every_cell_once():
    spec, params = instruction_spec(), _params()
    reference = ablation.reference_cell(spec, params)
    assert reference == ablation.Cell(4.0, 32.0, None)
    cells = ablation.cells_of(reference, [2.0, 4.0], [25.0, 32.0], [None, 3.0])
    assert cells[0] == reference and len(cells) == len(set(cells)) == 8
    assert ablation.Cell(4.0, 25.0, 3.0) in cells
    # a spec whose p is not method A's has no cell that reproduces it
    with pytest.raises(ValueError, match="not method A"):
        ablation.reference_cell(spec, _params(bank_rate_deg_s=6.0))
    assert [ablation.parse_rate(v) for v in (ablation.DERIVED, "3")] == [None, 3.0]


# downwind 270°, base, final onto runway 09 (the executor tests' flight): two 90° left turns at 2.25°/s
def _turn(degrees: float, speed: float) -> list[tuple[int, float, float, float]]:
    side, per_row = math.copysign(1.0, degrees), 4.5
    steady = int(round((abs(degrees) - 4.0 * per_row) / per_row))
    return [(2, side * per_row / 3.0, speed, 0.0), (2, side * per_row * 2.0 / 3.0, speed, 0.0),
            (steady, side * per_row, speed, 0.0), (2, side * per_row * 2.0 / 3.0, speed, 0.0),
            (2, side * per_row / 3.0, speed, 0.0)]


DOWNWIND_BASE_FINAL = [(60, 0.0, 100.0, 0.0), *_turn(-90.0, 100.0), (20, 0.0, 90.0, 0.0), *_turn(-90.0, 85.0),
                       (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]


def test_relabelling_reads_every_flight_under_the_cells_vocabulary_and_lists_the_refused():
    landing = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    # the same flight ending 3 km north of the centreline: never on the final
    astray = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 3000.0), dataset_id="KXXX:astray")
    batch = SimpleNamespace(signals=[landing, astray], geometries=[instruction_airport()] * 2)
    first_turn_word = {}
    for lead in (4.0, 8.0):
        vocabulary = ablation.Cell(lead, 32.0, None).vocabulary(instruction_spec())
        keep, readings, refused = ablation.relabel(batch, vocabulary, Words(vocabulary))
        assert keep == [0] and refused == [(1, "not on the final at the end")]
        first_turn_word[lead] = min(i.row for i in readings[0].instructions if i.column == HEADING and i.row > 0)
    # a word says where the track is a lead later: 4 s more lead, said two rows earlier
    assert first_turn_word[4.0] - first_turn_word[8.0] == 2


def test_a_polyline_is_read_at_points_no_farther_apart_than_the_spacing_ending_on_every_vertex():
    points = np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 3.0]])
    dense = ablation.densify(points, 5.0)
    assert dense == pytest.approx(np.array([[0, 0], [4, 0], [8, 0], [12, 0], [12, 3]]))
    assert ablation.densify(points[:1], 5.0) == pytest.approx(points[:1])


def _states(geometry, e: np.ndarray, n: np.ndarray) -> torch.Tensor:
    lat, lon = geometry.frame.latlon_from_horizontal(e, n)
    states = np.zeros((1, len(e), 7))
    states[0, :, LAT], states[0, :, LON], states[0, :, ALT] = lat, lon, 500.0
    states[0, :, SPEED], states[0, :, PSI], states[0, :, GAMMA], states[0, :, MASS] = 70.0, 0.0, 0.0, 60000.0
    return torch.tensor(states, dtype=torch.float64)


def test_the_largest_distances_read_the_rows_both_reach_in_time_and_both_tracks_as_lines_without_it():
    geometry = instruction_airport()          # runway 09's threshold at the frame's origin
    # observed: 11 rows 2 s apart on the centreline, 140 m a row, the last 600 m before the threshold (it landed next)
    rows = 11
    observed = SimpleNamespace(time_s=np.arange(rows) * 2.0, e_m=-2000.0 + np.arange(rows) * 140.0, n_m=np.zeros(rows))
    # flown: 1 s cycles, 70 m each, 30 m north, to 30 m past the threshold — but 400 m north at cycle 9 (an overshoot)
    e = -2000.0 + np.arange(30) * 70.0
    n = np.full(30, 30.0)
    n[9] = 400.0
    flown = SimpleNamespace(states=_states(geometry, e, n), cycle_s=1.0)
    batch = SimpleNamespace(signals=[observed], readings=[SimpleNamespace(words=np.zeros((rows, 6)), runway_index=0)],
                            geometries=[geometry])
    # the threshold (e = 0) is crossed between cycles 28 and 29
    crossed = SimpleNamespace(outcome="landed", end_row=29, crossing={"at_row": 2000.0 / 70.0})
    landed = ablation.track_gaps(batch, flown, [crossed])[0]
    # time-aligned: the rows sit on even cycles, so the overshoot falls between them; time-free: the overshoot is 400 m
    # from the observed line, and the observed track closed to the threshold ends where the flown one does
    assert landed["max_horizontal_distance_m"] == pytest.approx(30.0, abs=0.01)
    assert landed["hausdorff_horizontal_distance_m"] == pytest.approx(400.0, abs=0.01)
    # on the centreline all the way, the flown line ends at its crossing, as the observed one at the threshold: the two
    # lines are one, read within half the point spacing (the cycle past the plane, 30 m beyond it, is not read)
    on_line = SimpleNamespace(states=_states(geometry, e, np.zeros(30)), cycle_s=1.0)
    assert ablation.track_gaps(batch, on_line, [crossed])[0]["hausdorff_horizontal_distance_m"] <= ablation.DENSIFY_M / 2
    # a dynamics failure at cycle 9 is read to cycle 8 (e = −1440 m), as its record is: the overshoot is not read, and
    # the threshold the observed aircraft reached is that far from where the flown track stopped
    cut = ablation.track_gaps(batch, flown, [SimpleNamespace(outcome="dynamics_failure", end_row=9)])[0]
    assert cut["max_horizontal_distance_m"] == pytest.approx(30.0, abs=0.01)
    assert cut["hausdorff_horizontal_distance_m"] == pytest.approx(math.hypot(1440.0, 30.0), abs=0.01)


def _row(dataset_id: str, **changes):
    row = {"dataset_id": dataset_id, "flight_key": dataset_id, "airport": "KXXX", "group": "own dynamics",
           "stratum": "vectored", "outcome": "landed", "flew_the_sentence": True, "crossing": {"cross_m": 1.5},
           "words": [("heading", True), ("speed", False)], "heading_words_not_judged": 0,
           "heading_words_told_with_a_skipped_word": {"judged": 0, "inside": 0}, "words_not_reached": 0,
           "words_superseded_before_flown": 0, "intercepting_off_word_cycles": 0, "refused": None,
           "limits": {"bank_cap": {"cycles": 2, "wanted_minus_given": 0.0},
                      "bank_rate": {"cycles": 1, "wanted_minus_given": 0.0},
                      "cycles": {"cycles": 10, "wanted_minus_given": 0.0}},
           "recorded": True, "mean_horizontal_distance_m": float("nan"), "mean_vertical_distance_m": 5.0,
           "landing_time_minus_observed_s": 3.0, "replay_verdict": "pass", "observed_verdict": "pass",
           "heading_words": 7, "max_horizontal_distance_m": 100.0, "hausdorff_horizontal_distance_m": 200.0}
    return {**row, **changes}


def _stored(row):
    """A row as the formal replay stored it: through JSON, without the ablation's own fields."""
    import json

    return json.loads(json.dumps({k: v for k, v in row.items() if k not in ablation.ABLATION_FIELDS}))


def test_the_reference_must_reproduce_every_field_the_formal_replay_stored_but_its_verdicts():
    rows = [_row("a"), _row("b", replay_verdict="fail")]
    stored = [_stored(_row("a")), _stored(_row("b"))]
    result = ablation.reproduction(rows, stored)
    assert result["executor_fields_differ"] == [] and result["evaluation_verdicts_differ"] == ["b"]
    assert result["observed_verdicts_differ"] == []
    assert "crossing" in result["fields_compared"] and "replay_verdict" not in result["fields_compared"]
    assert ablation.reproduction([_row("a", crossing={"cross_m": 1.6}), _row("b")], stored)["executor_fields_differ"] == ["a"]
    with pytest.raises(ValueError, match="other flights"):
        ablation.reproduction([_row("a"), _row("c")], stored)
    # a field the replay stores that this runner's rows lack — or one they gained — is never skipped
    with pytest.raises(ValueError, match="same fields"):
        ablation.reproduction([_row("a", new_field=1), _row("b", new_field=1)], stored)
    with pytest.raises(ValueError, match="same fields"):
        ablation.reproduction(rows, [{**stored[0], "new_field": 1}, {**stored[1], "new_field": 1}])


def test_a_cells_summary_counts_per_group_and_stratum_and_pairs_against_the_reference():
    reference = [_row("a"), _row("b", stratum="straight-in"), _row("c", observed_verdict="fail"),
                 _row("s", group="stand-in dynamics")]
    cell = [_row("a", outcome="timeout", replay_verdict="fail", landing_time_minus_observed_s=None),
            _row("b", stratum="straight-in", mean_horizontal_distance_m=250.0), _row("c", observed_verdict="fail"),
            _row("s", group="stand-in dynamics", outcome="timeout")]
    refused = [{"dataset_id": "r", "reason": "too short", "group": "own dynamics", "stratum": "vectored"}]
    summary = ablation.by_group(cell, refused)
    own = summary["own dynamics"]["all"]
    assert own["flights"] == 3 and own["refused"] == 1 and own["landed"] == pytest.approx(2 / 3)
    assert own["words_judged"] == 6 and own["words_inside"] == pytest.approx(0.5)
    assert own["observed_passes"] == 2 and own["replay_passes_where_observed_passes"] == pytest.approx(0.5)
    # the distances and the landing time are the landed flights' (b, c); a NaN is counted apart, never averaged
    assert own["landing_time_minus_observed_s"]["n"] == 2
    assert own["mean_horizontal_distance_m"] == {"n": 1, "non_finite": 1, "mean": 250.0, "p50": 250.0, "p90": 250.0}
    assert own["limit_bound_cycle_share"] == {"bank_cap": pytest.approx(0.2), "bank_rate": pytest.approx(0.1)}
    assert summary["own dynamics"]["straight-in"]["flights"] == 1 and summary["own dynamics"]["straight-in"]["refused"] == 0
    assert summary["stand-in dynamics"]["all"]["landed"] == 0.0
    assert ablation.paired_changes(cell, reference, "all") == {
        "flights": 3, "landed_lost": 1, "landed_gained": 0, "evaluation_pairs": 2, "evaluation_lost": 1,
        "evaluation_gained": 0}
    assert ablation.paired_changes(cell, reference, "straight-in")["landed_lost"] == 0
    gates = {"own dynamics": {"all": {"all": {"clears": {"landed": True}}},
                              "KXXX": {"all": {"clears": {}}, "vectored": {"clears": {"landed": True, "words": False}},
                                       "straight-in": {"clears": {"landed": True, "words": True}}}}}
    assert ablation.gates_cleared(gates, "all") == {"cells": 2, "clear": 1}
    assert ablation.gates_cleared(gates, "vectored") == {"cells": 1, "clear": 0}
    # the table prints every stratum's own counts
    strata = ("all", "straight-in", "vectored")
    result = {"sample": {"flights": 5, "by_group": {}}, "reference": "L4_bank32_pderived",
              "reference_reproduction": {"flights": 4, "evaluation_verdicts_differ": [], "observed_verdicts_differ": []},
              "cells": {"L8_bank32_pderived": {"params": _params(heading_time_constant_s=8.0, bank_rate_deg_s=4.0).__dict__,
                                               "summary": summary,
                                               "gates_cleared": {s: ablation.gates_cleared(gates, s) for s in strata},
                                               "against_reference": {s: ablation.paired_changes(cell, reference, s)
                                                                     for s in strata}}}}
    table = ablation.render(result).splitlines()
    rows = [line for line in table if line.startswith("L8_bank32_pderived")]
    assert len(rows) == 3 and " 1/2 " in rows[0] and " 0/1 " in rows[2]
