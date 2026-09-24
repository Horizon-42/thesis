"""The prior's second version (`prior/`, prior design §8): the steps it is trained on — positions only, the
hand-over's words given at step 0, every candidate's relative place — its selection set and batches, the network's
causality, candidate symmetry and step-0 structure, the baselines' arithmetic, and the two runners end to end on a
synthetic artefact (every write into ``tmp_path``)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import (
    labeller_source_sha256, write_candidates, write_sentences, write_signals, write_spec,
)
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import (
    ALTITUDE, APPROACH, APPROACH_CLEARED, APPROACH_NOT_CLEARED, COLUMNS, HEADING, RUNWAY, SPEED, UNCHANGED, Words,
)
from ts_transformer.prior import data as prior_data
from ts_transformer.prior.data import (
    GIVEN_AT_HANDOVER, INPUT_SETS, Flight, Split, batches, column_classes, flight_steps, partition, selection_ids,
)
from ts_transformer.prior.model import Prior, PriorConfig, predicted_entries
from ts_transformer.prior.readout import Baselines
from ts_transformer.prior.train import column_nll
from ts_transformer.tests.support import (
    fixture_days, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec,
)


def _grid(words, rows=6):
    grid = np.full((rows, 6), UNCHANGED, dtype=np.int16)
    grid[0] = [1, APPROACH_NOT_CLEARED, words.heading_index(270.0), words.altitude_index(1200.0), 0,
               words.speed_index(100.0)]
    grid[3, HEADING] = words.heading_index(180.0)
    grid[3, APPROACH] = APPROACH_CLEARED
    grid[5, SPEED] = words.speed_unspecified
    return grid


def _two_runways() -> AirportGeometry:
    """`instruction_airport` with a second candidate: runway 27, the reciprocal, 3 km east and 500 m north."""
    data = instruction_airport().to_dict()
    other = {"ident": "27", "threshold_e_m": 3000.0, "threshold_n_m": 500.0, "course_deg": 270.0, "elevation_m": 110.0,
             "length_m": 3000.0}
    data["candidates"].append(other)
    data["runway_ends"].append({key: other[key] for key in ("ident", "threshold_e_m", "threshold_n_m", "course_deg")})
    return AirportGeometry.from_dict(data)


def _signals(rows=6):
    """Westbound at 90 m/s, 300 m north of runway 09's centreline, descending 2 m/s; turning 3°/row from row 3."""
    signals = instruction_flight(*fly_legs([(3, 0.0, 90.0, -2.0), (rows - 4, -3.0, 90.0, -2.0)], 270.0, 900.0, -5000.0,
                                           300.0))
    signals.track_deg[:] += 7.0                     # a "fitted" track that is not the displacement's direction
    return signals


def test_step_0_gives_the_hand_over_and_asks_only_the_runway():
    words = Words(spec())
    grid = _grid(words)
    _, _, in_force, since, targets = flight_steps(_signals(), grid, _two_runways(), INPUT_SETS["V1"])
    given = list(GIVEN_AT_HANDOVER)
    assert sorted(given + [RUNWAY]) == list(range(6))
    assert in_force[0, RUNWAY] == 0 and (in_force[0, given] == grid[0, given] + 1).all()   # the hand-over, given
    assert (targets[0] == grid[0] + 1).all() and (since[0] == 0).all()
    assert (in_force[1] == grid[0] + 1).all() and (in_force[4, HEADING] == grid[3, HEADING] + 1)
    assert in_force[3, HEADING] == grid[0, HEADING] + 1                  # step 3's own word is its target, not input
    assert targets[1].sum() == 0 and targets[3, HEADING] == grid[3, HEADING] + 1 and targets[3, ALTITUDE] == 0
    assert since[4, HEADING] == pytest.approx(math.log1p(1) / prior_data.SINCE_SCALE)
    assert since[3, HEADING] == pytest.approx(math.log1p(3) / prior_data.SINCE_SCALE)
    bad = grid.copy()
    bad[0, SPEED] = UNCHANGED
    with pytest.raises(ValueError, match="step 0"):
        flight_steps(_signals(), bad, _two_runways(), INPUT_SETS["V1"])


def test_every_candidate_is_placed_from_positions_and_the_direction_from_the_step_before():
    signals, geometry = _signals(), _two_runways()
    features, relative, *_ = flight_steps(signals, _grid(Words(spec())), geometry, INPUT_SETS["V2d"])
    names = INPUT_SETS["V2d"]
    assert features.shape == (6, len(names.step_features)) and relative.shape == (6, 2, len(names.relative_features))
    de, dn = np.diff(signals.e_m[:6]), np.diff(signals.n_m[:6])
    motion = np.degrees(np.arctan2(de, dn))
    for k, candidate in enumerate(geometry.candidates):
        place = relative_to_runway(signals.e_m[:6], signals.n_m[:6], np.zeros(6), signals.altitude_m[:6], candidate)
        assert relative[:, k, 0] == pytest.approx(np.arcsinh(place.before_threshold_m / 1000.0), abs=1e-6)
        assert relative[:, k, 1] == pytest.approx(np.arcsinh(place.right_of_course_m / 1000.0), abs=1e-6)
        assert relative[:, k, 2] == pytest.approx(place.height_above_threshold_m / 1000.0, abs=1e-6)
        off = np.radians(motion - candidate.course_deg)
        assert relative[1:, k, 3] == pytest.approx(np.sin(off), abs=1e-6)
        assert relative[1:, k, 4] == pytest.approx(np.cos(off), abs=1e-6)
        assert (relative[0, k, 3:5] == 0).all() and relative[0, k, 5] == 1 and (relative[1:, k, 5] == 0).all()
    # the state: positions, then the displacement over its 2 s — none at row 0, a flag there
    velocity = features[:, len(prior_data.STATE_FEATURES):]
    assert velocity[1:, 0] == pytest.approx(np.hypot(de, dn) / 2.0 / prior_data.SPEED_SCALE_MPS, abs=1e-6)
    assert velocity[1:, 1] == pytest.approx(np.sin(np.radians(motion)), abs=1e-6)
    assert velocity[1:, 3] == pytest.approx(np.diff(signals.altitude_m[:6]) / 2.0 / 10.0, abs=1e-6)
    assert (velocity[0, :4] == 0).all() and velocity[0, 4] == 1 and (velocity[1:, 4] == 0).all()


def test_only_variant_0_reads_the_fitted_columns_and_a_step_reads_no_later_row():
    signals, geometry, grid = _signals(), _two_runways(), _grid(Words(spec()))
    fitted, fitted_relative, *_ = flight_steps(signals, grid, geometry, INPUT_SETS["V0"])
    track = np.radians(signals.track_deg[:6])
    assert fitted[:, 4] == pytest.approx(np.sin(track), abs=1e-6)
    assert fitted[:, 6] == pytest.approx(signals.ground_speed_mps[:6] / 100.0, abs=1e-6)
    assert fitted_relative[:, 0, 3] == pytest.approx(np.sin(track - np.radians(90.0)), abs=1e-6)
    later = _signals()
    later.e_m[4:] += 500.0
    later.altitude_m[4:] += 100.0
    later.track_deg[:] += 30.0                           # the fitted columns move with the future; positions do not
    later.ground_speed_mps[:] += 10.0
    for name in ("V1", "V1d", "V2", "V2d"):
        before, before_relative, *_ = flight_steps(signals, grid, geometry, INPUT_SETS[name])
        after, after_relative, *_ = flight_steps(later, grid, geometry, INPUT_SETS[name])
        assert np.array_equal(before[:4], after[:4]) and np.array_equal(before_relative[:4], after_relative[:4])
        assert not np.array_equal(before[4:], after[4:])
    assert INPUT_SETS["V1"].step_features == prior_data.STATE_FEATURES
    assert INPUT_SETS["V1"].relative_features == prior_data.RELATIVE_FEATURES


def _flight(identifier, airport, rows=4, width=len(prior_data.STATE_FEATURES)):
    return Flight(identifier, airport, np.zeros((rows, width), np.float32),
                  np.zeros((rows, 1, len(prior_data.RELATIVE_FEATURES)), np.float32), np.zeros((rows, 6), np.int16),
                  np.zeros((rows, 6), np.float32), np.zeros((rows, 6), np.int16))


def test_the_selection_set_is_drawn_per_airport_seeded_and_apart_from_the_training_flights():
    flights = [_flight(f"A:{i:03d}", 0) for i in range(40)] + [_flight(f"B:{i:03d}", 1) for i in range(15)]
    split = Split(flights, ("A", "B"), np.zeros((2, 1, len(prior_data.CANDIDATE_FEATURES)), np.float32),
                  (2,) * 6, "V1")
    ids = selection_ids(split, 0.1, 1337)
    assert ids == selection_ids(split, 0.1, 1337) and ids != selection_ids(split, 0.1, 2024)
    assert sum(i.startswith("A:") for i in ids) == 4 and sum(i.startswith("B:") for i in ids) == 2   # round(1.5) = 2
    assert list(ids) == sorted(ids)
    rest, chosen = partition(split, ids)
    assert {f.dataset_id for f in chosen.flights} == set(ids) and len(rest.flights) == 55 - len(ids)
    assert not {f.dataset_id for f in rest.flights} & set(ids) and rest.inputs == "V1"
    with pytest.raises(ValueError, match="not in the split"):
        partition(split, ids + ("C:000",))


def test_batches_hold_every_flight_once_under_the_token_budget():
    flights = [_flight(f"F{i}", 0, n) for i, n in enumerate([5, 50, 7, 300, 40, 41])]
    groups = list(batches(flights, 100, np.random.default_rng(0)))
    assert sorted(i for g in groups for i in g) == list(range(6))
    for g in groups:
        assert len(g) == 1 or max(flights[i].rows for i in g) * len(g) <= 100


def _model(inputs="V2d", slots=3, valid=2):
    words = Words(spec())
    classes = column_classes(words, slots)
    candidates = torch.zeros(1, slots, len(prior_data.CANDIDATE_FEATURES))
    candidates[0, :valid] = torch.randn(valid, len(prior_data.CANDIDATE_FEATURES), generator=torch.Generator().manual_seed(1))
    candidates[0, :valid, -1] = 1.0                                      # the rest are empty slots
    torch.manual_seed(0)
    config = PriorConfig(classes=classes, airports=("KXXX",), candidate_slots=slots, inputs=inputs, d_model=32, layers=2,
                         heads=4, feedforward=64, dropout=0.0)
    return Prior(config, candidates).eval()


def _inputs(model, rows=7, pointer=1):
    inputs = INPUT_SETS[model.config.inputs]
    generator = torch.Generator().manual_seed(2)
    features = torch.randn(1, rows, len(inputs.step_features), generator=generator)
    relative = torch.randn(1, rows, model.config.candidate_slots, len(inputs.relative_features), generator=generator)
    in_force = torch.zeros(1, rows, 6, dtype=torch.long)
    in_force[0, :, [c for c in GIVEN_AT_HANDOVER]] = 1
    in_force[0, 1:, RUNWAY] = pointer
    return {"features": features, "relative": relative, "in_force": in_force, "since": torch.zeros(1, rows, 6),
            "airport": torch.zeros(1, dtype=torch.long), "padding": torch.zeros(1, rows, dtype=torch.bool)}


@torch.no_grad()
def _run(model, batch):
    return model(batch["features"], batch["relative"], batch["in_force"], batch["since"], batch["airport"],
                 batch["padding"])


def test_a_step_sees_only_itself_and_the_steps_before_it():
    model = _model()
    batch = _inputs(model)
    before = _run(model, batch)
    for name in ("features", "relative"):
        changed = dict(batch)
        changed[name] = batch[name].clone()
        changed[name][0, 5:] += 3.0
        after = _run(model, changed)
        for a, b in zip(before, after):
            assert torch.allclose(a[:, :5], b[:, :5], atol=1e-5) and not torch.allclose(a[:, 5:], b[:, 5:])


def test_step_0_says_a_runway_and_takes_the_hand_over_as_given():
    model = _model()
    batch = _inputs(model)
    batch["in_force"][0, 0, HEADING] = 7
    logits = _run(model, batch)
    runway = logits[RUNWAY][0]
    assert torch.isinf(runway[0, 0]) and torch.isfinite(runway[1:, 0]).all()   # "unchanged" only after step 0
    assert torch.isfinite(runway[:, 1:3]).all() and torch.isinf(runway[:, 3]).all()   # the empty slot never
    for c in GIVEN_AT_HANDOVER:
        p = torch.softmax(logits[c][0, 0], dim=-1)
        assert p[batch["in_force"][0, 0, c]] == 1.0 and p.sum() == 1.0      # the point mass on the given word
        assert torch.isfinite(logits[c][0, 1:]).all()
    targets = batch["in_force"].clone()
    targets[0, 0, RUNWAY] = 2
    asked = predicted_entries(batch["padding"])
    assert asked[0, 0].tolist() == [c == RUNWAY for c in range(6)] and asked[0, 1:].all()
    nll = column_nll(logits, targets, batch["padding"])
    step0 = -torch.log_softmax(logits[RUNWAY][0, 0], dim=-1)[2]
    later = sum(float(-torch.log_softmax(logits[RUNWAY][0, t], dim=-1)[targets[0, t, RUNWAY]]) for t in range(1, 7))
    assert float(nll[RUNWAY]) == pytest.approx(float(step0) + later, rel=1e-5)
    for c in GIVEN_AT_HANDOVER:                                          # step 0 adds nothing there
        expected = sum(float(-torch.log_softmax(logits[c][0, t], dim=-1)[targets[0, t, c]]) for t in range(1, 7))
        assert float(nll[c]) == pytest.approx(expected, rel=1e-5)


def test_the_candidates_order_changes_only_the_order_of_the_runway_scores():
    model = _model(slots=3, valid=3)
    batch = _inputs(model, pointer=2)
    logits = _run(model, batch)
    order = [2, 0, 1]                                                    # new slot j holds old candidate order[j]
    permuted = _model(slots=3, valid=3)
    permuted.load_state_dict({**model.state_dict(), "candidates": model.candidates[:, order]})
    moved = dict(batch)
    moved["relative"] = batch["relative"][:, :, order]
    moved["in_force"] = batch["in_force"].clone()
    moved["in_force"][0, 1:, RUNWAY] = order.index(1) + 1                 # the same runway (old slot 1) in force
    after = _run(permuted, moved)
    for c in range(6):
        if c == RUNWAY:
            assert torch.allclose(after[c][..., 0], logits[c][..., 0], atol=1e-5)
            assert torch.allclose(after[c][..., 1:], logits[c][..., 1:][..., order], atol=1e-5)
        else:
            assert torch.allclose(after[c], logits[c], atol=1e-5)


def test_a_padded_candidate_slot_adds_nothing():
    model = _model(slots=3, valid=2)
    batch = _inputs(model)
    logits = _run(model, batch)
    noisy = dict(batch)
    noisy["relative"] = batch["relative"].clone()
    noisy["relative"][:, :, 2] += 5.0 * torch.randn_like(noisy["relative"][:, :, 2])
    for before, after in zip(logits, _run(model, noisy)):
        assert torch.equal(torch.isinf(before), torch.isinf(after))
        assert torch.allclose(before[torch.isfinite(before)], after[torch.isfinite(after)], atol=1e-6)


def test_the_rule_takes_the_fewest_inputs_within_the_seed_line_of_the_leader():
    from ts_transformer.experiments.prior_select import choose

    def runs(nll, replicate):
        out = {(name, 1337): {"readout": {"model": {"nll_per_step": value}}} for name, value in nll.items()}
        leader = min(nll, key=nll.get)
        out[(leader, 2024)] = {"readout": {"model": {"nll_per_step": replicate}}}
        return out

    # V2d leads by 0.010; its seeds differ by 0.012: V1d (0.008 behind) is inside the line, V1 (0.013) is not
    rule = choose(runs({"V1": 0.513, "V1d": 0.508, "V2": 0.505, "V2d": 0.500}, 0.512), 1337, 2024)
    assert rule["leader"] == "V2d" and rule["seed_line"] == pytest.approx(0.012)
    assert rule["within_seed_line"] == ["V1d", "V2", "V2d"] and rule["chosen"] == "V1d"
    # a seed line of zero keeps the leader
    assert choose(runs({"V1": 0.513, "V1d": 0.508, "V2": 0.505, "V2d": 0.500}, 0.500), 1337, 2024)["chosen"] == "V2d"
    with pytest.raises(SystemExit, match="no replicate"):
        choose({key: value for key, value in runs({"V1": 1, "V1d": 1, "V2": 1, "V2d": 0.5}, 0.5).items()
                if key[1] == 1337}, 1337, 2024)


def test_the_baselines_count_changes_and_values_from_train_and_the_runway_alone_at_step_0():
    classes = (3, 3, 4, 3, 3, 3)
    targets = np.zeros((4, 6), dtype=np.int16)
    targets[0] = [1, 1, 1, 1, 1, 1]
    targets[2, HEADING] = 3                                              # heading changes once, to value 2
    in_force = np.vstack(([0, 1, 1, 1, 1, 1], [1] * 6, [1] * 6, [1, 1, 3, 1, 1, 1])).astype(np.int16)
    flight = Flight("F", 0, np.zeros((4, 4), np.float32), np.zeros((4, 2, 3), np.float32), in_force,
                    np.zeros((4, 6), np.float32), targets)
    table = np.zeros((1, 2, len(prior_data.CANDIDATE_FEATURES)), np.float32)
    table[0, :, -1] = 1.0
    split = Split([flight], ("KXXX",), table, classes, "V1")
    base = Baselines.count(split)
    assert base.change[HEADING] == pytest.approx((1 + 1) / (3 + 2))
    assert base.unigram[HEADING].tolist() == pytest.approx([1 / 4, 1 / 4, 2 / 4])
    assert base.bigram[HEADING][1].tolist() == pytest.approx([1 / 4, 1 / 4, 2 / 4])   # after value 0 (class 1)
    assert base.first_runway.tolist() == pytest.approx([2 / 3, 1 / 3])
    nll = base.nll_per_step(split)
    # heading: nothing at step 0 (given), two kept steps and one change
    expected = (-2 * math.log(1 - base.change[HEADING]) - math.log(base.change[HEADING] * 2 / 4)) / 4
    assert nll["repeat"]["heading"] == pytest.approx(expected)
    runway = (-math.log(2 / 3) - 3 * math.log(1 - base.change[RUNWAY])) / 4
    assert nll["repeat"]["runway"] == pytest.approx(runway)
    reference = base.airport_runway_reference(split)
    assert reference["top1"] == 1.0 and reference["nll_per_flight"] == pytest.approx(-math.log(2 / 3))


# ---- the runners end to end, on a synthetic artefact (every write in tmp_path)
VECTORED = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
STRAIGHT = [(40, 0.0, 90.0, 0.0), (30, 0.0, 80.0, 0.0), (110, 0.0, 72.0, -72.0 * np.tan(np.radians(3.0)))]


def _artefact(directory):
    one = spec()
    directory.mkdir(parents=True)

    def flights(prefix, count, split):
        return [instruction_flight(*(fly_legs(VECTORED, 270.0, 1110.0, -400.0, 0.0) if i % 2
                                     else fly_legs(STRAIGHT, 90.0, 1200.0, -300.0, 0.0)),
                                   dataset_id=f"KXXX:{prefix}{i}_09_abc{i:03d}_20260101T000000Z", split=split)
                for i in range(count)]

    items = {"train": flights("T", 12, "train"), "val": flights("V", 4, "val")}
    write_signals(directory, items, {"note": "test"}, fixture_days())
    write_candidates(directory, {"KXXX": _two_runways()})
    write_spec(directory, one, {"n": 1}, {"labeller_source_sha256": labeller_source_sha256(),
                                          "git": {"head": "test", "dirty": False}})
    for split, signals in items.items():
        readings = [read_flight(flight, _two_runways(), one) for flight in signals]
        assert all(reading.words is not None for reading in readings)
        write_sentences(directory, split, one, readings, list(range(len(signals))))
    return one


def test_the_runners_train_every_variant_choose_by_the_rule_and_read_val_once(tmp_path, capsys):
    from ts_transformer.experiments import prior_select, prior_train

    artefact, campaign = tmp_path / "artefact", tmp_path / "campaign"
    _artefact(artefact)
    common = ["--instructions", str(artefact), "--device", "cpu", "--limit", "20", "--max-epochs", "1",
              "--tokens-per-batch", "2048", "--selection-share", "0.25"]
    for name in ("V1", "V1d", "V2", "V2d", "V0"):
        assert prior_train.main([*common, "--inputs", name, "--out", str(campaign / f"{name}_s1337")]) == 0
    capsys.readouterr()
    assert prior_select.main(["--campaign", str(campaign), "--leader"]) == 0
    leader = capsys.readouterr().out.strip()
    assert leader in prior_data.CHOOSABLE
    with pytest.raises(SystemExit, match="no replicate"):
        prior_select.main(["--campaign", str(campaign), "--device", "cpu"])
    assert prior_train.main([*common, "--inputs", leader, "--seed", "2024", "--out", str(campaign / f"{leader}_s2024")]) == 0
    assert prior_select.main(["--campaign", str(campaign), "--device", "cpu"]) == 0

    config = json.loads((campaign / "V1_s1337" / "config.json").read_text())
    assert config["smoke"] and config["selection"]["flights"] == 3 and config["flights"]["train"] == 9
    choice = json.loads((campaign / "choice.json").read_text())
    nll = choice["nll_per_step"]
    assert choice["leader"] == leader == min(nll, key=nll.get)
    assert choice["chosen"] in choice["within_seed_line"]
    assert all(INPUT_SETS[choice["chosen"]].added <= INPUT_SETS[name].added for name in choice["within_seed_line"])
    assert len(choice["runs"]) == 6 and set(choice["future_information"]["minus"]) == set(prior_data.CHOOSABLE)
    readout = json.loads((campaign / choice["chosen_directory"] / "readout.json").read_text())
    assert readout["split"] == "val" and readout["model"]["flights"] == 4
    step0 = readout["model"]["step0_runway"]
    assert readout["model"]["per_column"]["runway"]["change_steps"] == 4 == step0["flights"]
    assert 0.0 <= step0["top1"] <= step0["top2"] == 1.0                  # two candidates: the second pick is certain
    handed = step0["by_handover_approach"]
    assert sum(part["flights"] for part in handed.values()) == 4 and set(handed) <= {"not cleared", "cleared"}
    assert sum(1 for path in campaign.glob("*/readout.json")) == 1        # val is read once, on the chosen run
    with pytest.raises(SystemExit):
        prior_select.main(["--campaign", str(campaign), "--device", "cpu"])   # never twice
