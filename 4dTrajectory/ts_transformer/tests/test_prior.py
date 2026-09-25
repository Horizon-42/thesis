"""The prior (`prior/`, prior design §3–§9, step 1): the steps it is trained on — only what is known before a step,
the first predicted step saying every column, the words said shifted by a step, every candidate's relative place and
the airport's landings before the step — its batches, the network's causality, candidate symmetry, ordered heads and
the aircraft attention, the baselines' and the runway breakdown's arithmetic, the runway rules, and the two runners end
to end on a synthetic artefact (every write into ``tmp_path``)."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from ts_transformer.data.day_split import SealedDay
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
    VARIANTS, Flight, Split, batches, column_classes, flight_steps, load_split, sentence_steps,
)
from ts_transformer.prior.model import EDGE_FEATURES, Prior, PriorConfig, asked_entries, self_edges
from ts_transformer.prior.readout import Baselines, runway_breakdown
from ts_transformer.prior.scene import N_LOOK, Landings, utc_s
from ts_transformer.prior.train import column_nll
from ts_transformer.tests.support import (
    fixture_days, fly_legs, instruction_airport, instruction_flight, instruction_spec as spec, landing_on,
)

ROWS = N_LOOK + 6


def _grid(words, rows=ROWS):
    """Row 0 says every column; row 3 (observed only) turns and clears; row N_LOOK + 2 changes the speed and row
    N_LOOK + 3 the heading."""
    grid = np.full((rows, 6), UNCHANGED, dtype=np.int16)
    grid[0] = [1, APPROACH_NOT_CLEARED, words.heading_index(270.0), words.altitude_index(1200.0), 0,
               words.speed_index(100.0)]
    grid[3, HEADING] = words.heading_index(180.0)
    grid[3, APPROACH] = APPROACH_CLEARED
    grid[N_LOOK + 2, SPEED] = words.speed_unspecified
    grid[N_LOOK + 3, HEADING] = words.heading_index(90.0)
    return grid


def _two_runways() -> AirportGeometry:
    """`instruction_airport` with a second candidate: runway 27, the reciprocal, 3 km east and 500 m north."""
    data = instruction_airport().to_dict()
    other = {"ident": "27", "threshold_e_m": 3000.0, "threshold_n_m": 500.0, "course_deg": 270.0, "elevation_m": 110.0,
             "length_m": 3000.0}
    data["candidates"].append(other)
    data["runway_ends"].append({key: other[key] for key in ("ident", "threshold_e_m", "threshold_n_m", "course_deg")})
    return AirportGeometry.from_dict(data)


def _signals(rows=ROWS):
    """Westbound at 90 m/s, 300 m north of runway 09's centreline, descending 2 m/s; turning 3°/row from row 3."""
    signals = instruction_flight(*fly_legs([(3, 0.0, 90.0, -2.0), (rows - 4, -3.0, 90.0, -2.0)], 270.0, 900.0, -5000.0,
                                           300.0))
    signals.track_deg[:] += 7.0                     # a fitted track that is not the displacement's direction
    return signals


def _landings(times_09=(), times_27=(), own=True) -> Landings:
    """Landings on 09 and 27 — with the fixture flight's own (09, `landing_on("train")`) unless ``own`` is False."""
    times_09 = [*times_09, *([utc_s(landing_on("train"))] if own else [])]
    runways = {"09": np.array(sorted(times_09), dtype=np.float64), "27": np.array(sorted(times_27), dtype=np.float64)}
    return Landings(np.sort(np.concatenate(list(runways.values()))), runways)


def test_the_first_predicted_step_says_every_column_and_the_words_said_shift_by_a_step():
    words = Words(spec())
    grid = _grid(words)
    in_force, since, targets = sentence_steps(grid)
    assert (targets[:N_LOOK] == 0).all() and (in_force[: N_LOOK + 1] == 0).all() and (since[: N_LOOK + 1] == 0).all()
    # the first predicted step: the word in force there in every column — row 3's turn and clearance included
    in_force_at_first = grid[0].astype(np.int64)
    in_force_at_first[[HEADING, APPROACH]] = grid[3, [HEADING, APPROACH]]
    assert (targets[N_LOOK] == in_force_at_first + 1).all()
    # after it: the words said so far, as the prior said them — at the first predicted step, not at row 3
    assert (in_force[N_LOOK + 1] == in_force_at_first + 1).all()
    assert since[N_LOOK + 1] == pytest.approx(np.full(6, math.log1p(1) / prior_data.SINCE_SCALE))
    assert targets[N_LOOK + 1].sum() == 0 and targets[N_LOOK + 2, SPEED] == words.speed_unspecified + 1
    assert targets[N_LOOK + 2, [c for c in range(6) if c != SPEED]].sum() == 0
    assert in_force[N_LOOK + 2, SPEED] == in_force_at_first[SPEED] + 1     # its own word is its target, not input
    assert in_force[N_LOOK + 3, SPEED] == words.speed_unspecified + 1
    assert since[N_LOOK + 3, SPEED] == pytest.approx(math.log1p(1) / prior_data.SINCE_SCALE)
    assert since[N_LOOK + 3, HEADING] == pytest.approx(math.log1p(3) / prior_data.SINCE_SCALE)
    bad = grid.copy()
    bad[0, SPEED] = UNCHANGED
    with pytest.raises(ValueError, match="step 0"):
        sentence_steps(bad)
    with pytest.raises(ValueError, match="no predicted step"):
        sentence_steps(grid[:N_LOOK])


def test_every_candidate_is_placed_from_positions_and_the_landings_before_the_step():
    signals, geometry = _signals(), _two_runways()
    clock = utc_s(signals.entry_time_utc) + signals.time_s[:ROWS]
    landings = _landings(times_09=[clock[0] - 4000.0, clock[0] - 60.0, clock[5]], times_27=[clock[0] - 100_000.0])
    features, relative, static, *_ = flight_steps(signals, _grid(Words(spec())), geometry, landings)
    width = len(VARIANTS["full"].relative_features)
    assert features.shape == (ROWS, len(prior_data.STEP_FEATURES)) and relative.shape == (ROWS, 2, width)
    assert static.shape == (len(prior_data.STATIC_FEATURES),) == (0,)
    de, dn = np.diff(signals.e_m[:ROWS]), np.diff(signals.n_m[:ROWS])
    motion = np.degrees(np.arctan2(de, dn))
    for k, candidate in enumerate(geometry.candidates):
        place = relative_to_runway(signals.e_m[:ROWS], signals.n_m[:ROWS], np.zeros(ROWS), signals.altitude_m[:ROWS],
                                   candidate)
        assert relative[:, k, 0] == pytest.approx(np.arcsinh(place.before_threshold_m / 1000.0), abs=1e-6)
        assert relative[:, k, 1] == pytest.approx(np.arcsinh(place.right_of_course_m / 1000.0), abs=1e-6)
        assert relative[:, k, 2] == pytest.approx(place.height_above_threshold_m / 1000.0, abs=1e-6)
        off = np.radians(motion - candidate.course_deg)
        assert relative[1:, k, 3] == pytest.approx(np.sin(off), abs=1e-6)
        assert relative[1:, k, 4] == pytest.approx(np.cos(off), abs=1e-6)
        assert (relative[0, k, 3:5] == 0).all() and relative[0, k, 5] == 1 and (relative[1:, k, 5] == 0).all()
    # the landing context: 09 had one landing in the 30 minutes before row 0 (and one before that); the one at row 5
    # counts only from row 6 on — strictly before the step
    count, since_last, none = relative[:, 0, 6], relative[:, 0, 7], relative[:, 0, 8]
    assert count[:6] == pytest.approx(np.full(6, math.log1p(1))) and count[6:] == pytest.approx(np.full(ROWS - 6, math.log1p(2)))
    assert since_last[0] == pytest.approx(math.log1p(1.0) / prior_data.LANDING_SINCE_SCALE)
    assert since_last[6] == pytest.approx(math.log1p(2.0 / 60.0) / prior_data.LANDING_SINCE_SCALE, abs=1e-6)
    assert (none == 0).all()
    # 27: its only landing long ago — none in the window, time since it known
    assert (relative[:, 1, 6] == 0).all() and (relative[:, 1, 8] == 0).all()
    assert relative[0, 1, 7] == pytest.approx(math.log1p(100_000.0 / 60.0) / prior_data.LANDING_SINCE_SCALE, abs=1e-6)
    # a runway that has had no landing: the flag, and zeros
    _, empty, *_ = flight_steps(signals, _grid(Words(spec())), geometry, _landings())
    assert (empty[:, :, 6:8] == 0).all() and (empty[:, :, 8] == 1).all()
    # the flight's own landing is never its context — here moved into its rows — and it must be in the pool
    early = _signals()
    early.landing_time_utc = datetime.fromtimestamp(clock[5], tz=timezone.utc).isoformat().replace("+00:00", "Z")
    shifted = Landings(np.sort(np.append(landings.times_s, clock[5] + 0.0)), {
        "09": np.sort(np.append(landings.by_runway["09"], clock[5])), "27": landings.by_runway["27"]})
    _, own, *_ = flight_steps(early, _grid(Words(spec())), geometry, shifted)
    assert np.array_equal(own[:, :, 6:], relative[:, :, 6:])
    with pytest.raises(ValueError, match="the flight's own must be there"):
        flight_steps(signals, _grid(Words(spec())), geometry, _landings(own=False))
    # the variant without the context has none of it
    _, bare, *_ = flight_steps(signals, _grid(Words(spec())), geometry, None)
    assert bare.shape[2] == len(VARIANTS["no-context"].relative_features) == 6
    assert np.array_equal(bare, relative[:, :, :6])
    # the state: positions, then the displacement over its 2 s — none at row 0, a flag there
    assert features[1:, 4] == pytest.approx(np.hypot(de, dn) / 2.0 / prior_data.SPEED_SCALE_MPS, abs=1e-6)
    assert features[1:, 5] == pytest.approx(np.sin(np.radians(motion)), abs=1e-6)
    assert features[1:, 7] == pytest.approx(np.diff(signals.altitude_m[:ROWS]) / 2.0 / 10.0, abs=1e-6)
    assert (features[0, 4:8] == 0).all() and features[0, 8] == 1 and (features[1:, 8] == 0).all()


def test_a_row_reads_no_later_position_and_no_later_landing():
    signals, geometry, grid = _signals(), _two_runways(), _grid(Words(spec()))
    clock = utc_s(signals.entry_time_utc) + signals.time_s[:ROWS]
    landings = _landings(times_09=[clock[0] - 60.0])
    before, before_relative, *_ = flight_steps(signals, grid, geometry, landings)
    later = _signals()
    later.e_m[10:] += 500.0
    later.altitude_m[10:] += 100.0
    later.track_deg[:] += 30.0                           # the fitted columns are never read
    later.ground_speed_mps[:] += 10.0
    more = _landings(times_09=[clock[0] - 60.0, clock[10], clock[12]], times_27=[clock[11]])
    after, after_relative, *_ = flight_steps(later, grid, geometry, more)
    # positions moved from row 10 on: rows before it read none of it
    assert np.array_equal(before[:10], after[:10]) and np.array_equal(before_relative[:10], after_relative[:10])
    assert not np.array_equal(before[10:], after[10:])
    # landings at rows 10, 11, 12: a row sees only those strictly before it
    context = slice(len(prior_data.RELATIVE_FEATURES), None)
    assert np.array_equal(before_relative[:11, :, context], after_relative[:11, :, context])
    assert not np.array_equal(before_relative[11:, :, context], after_relative[11:, :, context])


def _flight(identifier, airport, rows=ROWS, width=len(prior_data.STEP_FEATURES), established=False, target=1):
    targets = np.zeros((rows, 6), np.int16)
    targets[N_LOOK] = target
    return Flight(identifier, airport, np.zeros((rows, width), np.float32),
                  np.zeros((rows, 2, len(prior_data.RELATIVE_FEATURES)), np.float32), np.zeros(0, np.float32),
                  np.zeros((rows, 6), np.int16), np.zeros((rows, 6), np.float32), targets, _asked(rows), 0.0, 0, 0.0,
                  established)


def _asked(rows):
    return (np.arange(rows) >= N_LOOK)[:, None].repeat(6, axis=1)


def test_batches_hold_every_flight_once_under_the_token_budget():
    flights = [_flight(f"F{i}", 0, n) for i, n in enumerate([15, 50, 17, 300, 40, 41])]
    groups = list(batches(flights, 100, np.random.default_rng(0)))
    assert sorted(i for g in groups for i in g) == list(range(6))
    for g in groups:
        assert len(g) == 1 or max(flights[i].rows for i in g) * len(g) <= 100


def _model(variant="full", slots=3, valid=2):
    words = Words(spec())
    classes = column_classes(words, slots)
    candidates = torch.zeros(1, slots, len(prior_data.CANDIDATE_FEATURES))
    candidates[0, :valid] = torch.randn(valid, len(prior_data.CANDIDATE_FEATURES), generator=torch.Generator().manual_seed(1))
    candidates[0, :valid, -1] = 1.0                                      # the rest are empty slots
    torch.manual_seed(0)
    config = PriorConfig(classes=classes, airports=("KXXX",), candidate_slots=slots, variant=variant, d_model=32,
                         layers=2, heads=4, feedforward=64, dropout=0.0)
    return Prior(config, candidates).eval()


def _inputs(model, rows=ROWS, aircraft=1, pointer=1):
    """Random inputs for ``aircraft`` aircraft over ``rows`` steps; from N_LOOK + 1 the runway ``pointer`` in force;
    the targets: at the first predicted step a word in every column, after it some changes."""
    width = len(VARIANTS[model.config.variant].relative_features)
    generator = torch.Generator().manual_seed(2)
    features = torch.randn(1, aircraft, rows, len(prior_data.STEP_FEATURES), generator=generator)
    relative = torch.randn(1, aircraft, rows, model.config.candidate_slots, width, generator=generator)
    in_force = torch.zeros(1, aircraft, rows, 6, dtype=torch.long)
    in_force[:, :, N_LOOK + 1:] = 1
    in_force[:, :, N_LOOK + 1:, RUNWAY] = pointer
    targets = torch.zeros(1, aircraft, rows, 6, dtype=torch.long)
    targets[:, :, N_LOOK] = 1
    targets[:, :, N_LOOK, RUNWAY] = pointer
    targets[:, :, N_LOOK + 2, HEADING] = 5
    return {"features": features, "relative": relative, "static": torch.zeros(1, aircraft, 0), "in_force": in_force,
            "since": torch.zeros(1, aircraft, rows, 6), "airport": torch.zeros(1, dtype=torch.long),
            "present": torch.ones(1, aircraft, rows, dtype=torch.bool), "edges": self_edges(1, aircraft, rows, torch.device("cpu")),
            "targets": targets}


@torch.no_grad()
def _run(model, batch):
    return model(batch["features"], batch["relative"], batch["static"], batch["in_force"], batch["since"],
                 batch["airport"], batch["present"], batch["edges"], batch["targets"])


def test_a_step_sees_only_itself_and_the_steps_before_it():
    model = _model()
    batch = _inputs(model)
    before = _run(model, batch)
    t = N_LOOK + 2
    for name in ("features", "relative", "in_force", "targets"):
        changed = dict(batch)
        changed[name] = batch[name].clone()
        if name in ("in_force", "targets"):
            changed[name][:, :, t + 1:] = 2                              # another class in every column
        else:
            changed[name][:, :, t + 1:] += 3.0
        after = _run(model, changed)
        for a, b in zip(before, after):
            assert torch.equal(torch.isinf(a[..., : t + 1, :]), torch.isinf(b[..., : t + 1, :]))
            finite = torch.isfinite(a[..., : t + 1, :])
            assert torch.allclose(a[..., : t + 1, :][finite], b[..., : t + 1, :][finite], atol=1e-5)


def test_the_first_predicted_step_says_every_column_and_the_rows_before_it_are_not_asked():
    model = _model()
    batch = _inputs(model)
    logits = _run(model, batch)
    for c, logit in enumerate(logits):
        assert torch.isinf(logit[0, 0, N_LOOK, 0])                        # "unchanged" is not a class there
        assert torch.isfinite(logit[0, 0, N_LOOK + 1:, 0]).all()
    runway = logits[RUNWAY][0, 0]
    assert torch.isfinite(runway[N_LOOK:, 1:3]).all() and torch.isinf(runway[:, 3]).all()   # the empty slot never
    asked = asked_entries(batch["present"])
    assert not asked[0, 0, :N_LOOK].any() and asked[0, 0, N_LOOK:].all()
    nll = column_nll(logits, batch["targets"], batch["present"], torch.ones_like(batch["targets"], dtype=torch.bool))
    for c in range(6):
        expected = sum(float(-torch.log_softmax(logits[c][0, 0, t], dim=-1)[batch["targets"][0, 0, t, c]])
                       for t in range(N_LOOK, ROWS))
        assert float(nll[c]) == pytest.approx(expected, rel=1e-5)


def test_ordered_heads_read_the_earlier_columns_choices_and_unordered_heads_none():
    for variant, ordered in (("full", True), ("unordered", False)):
        model = _model(variant)
        batch = _inputs(model)
        before = _run(model, batch)
        t = N_LOOK + 1
        changed = dict(batch)
        changed["targets"] = batch["targets"].clone()
        changed["targets"][0, 0, t, ALTITUDE] = 7                         # what the altitude column chose at t
        after = _run(model, changed)
        for c in range(6):
            same = torch.allclose(before[c][0, 0, t], after[c][0, 0, t], atol=1e-6)
            assert same == (c <= ALTITUDE or not ordered), (variant, COLUMNS[c])


def test_the_candidates_order_changes_only_the_order_of_the_runway_scores():
    model = _model(slots=3, valid=3)
    batch = _inputs(model, pointer=2)
    logits = _run(model, batch)
    order = [2, 0, 1]                                                    # new slot j holds old candidate order[j]
    permuted = _model(slots=3, valid=3)
    permuted.load_state_dict({**model.state_dict(), "candidates": model.candidates[:, order]})
    moved = dict(batch)
    moved["relative"] = batch["relative"][:, :, :, order]
    moved["in_force"] = batch["in_force"].clone()
    moved["in_force"][..., N_LOOK + 1:, RUNWAY] = order.index(1) + 1     # the same runway (old slot 1) in force
    moved["targets"] = batch["targets"].clone()
    moved["targets"][..., N_LOOK, RUNWAY] = order.index(1) + 1
    after = _run(permuted, moved)
    for c in range(6):
        if c == RUNWAY:
            assert torch.allclose(after[c][..., 0], logits[c][..., 0], atol=1e-5, equal_nan=True)
            assert torch.allclose(after[c][..., 1:], logits[c][..., 1:][..., order], atol=1e-5)
        else:
            assert torch.allclose(after[c], logits[c], atol=1e-5)


def test_a_padded_candidate_slot_adds_nothing():
    model = _model(slots=3, valid=2)
    batch = _inputs(model)
    logits = _run(model, batch)
    noisy = dict(batch)
    noisy["relative"] = batch["relative"].clone()
    noisy["relative"][..., 2, :] += 5.0 * torch.randn_like(noisy["relative"][..., 2, :])
    for before, after in zip(logits, _run(model, noisy)):
        assert torch.equal(torch.isinf(before), torch.isinf(after))
        assert torch.allclose(before[torch.isfinite(before)], after[torch.isfinite(after)], atol=1e-6)


def test_the_aircraft_attention_is_symmetric_in_the_aircraft_and_reads_only_the_present_ones():
    model = _model()
    batch = _inputs(model, aircraft=3)
    generator = torch.Generator().manual_seed(3)
    batch["features"] = torch.randn(batch["features"].shape, generator=generator)
    batch["edges"] = torch.randn(1, ROWS, 3, 3, len(EDGE_FEATURES), generator=generator)
    logits = _run(model, batch)
    # every aircraft's output depends on the others (the scene is read)
    alone = dict(batch)
    alone["present"] = batch["present"].clone()
    alone["present"][0, 1:] = False
    solo = _run(model, alone)
    assert not torch.allclose(solo[HEADING][0, 0, N_LOOK:], logits[HEADING][0, 0, N_LOOK:], atol=1e-4)
    # absent aircraft — throughout, or before they enter the scene — never make a present one's output NaN, in eval or
    # in training (a fully masked attention row is NaN in some kernels, and NaN times a zero weight is still NaN)
    entering = dict(alone)
    entering["present"] = alone["present"].clone()
    entering["present"][0, 1, 5:] = True
    for training in (False, True):
        model.train(training)
        for scene in (alone, entering):
            out = model(*(scene[name] for name in ("features", "relative", "static", "in_force", "since", "airport",
                                                   "present", "edges", "targets")))
            for logit in out:
                assert not torch.isnan(logit[scene["present"]]).any()
    model.eval()
    # an absent aircraft's inputs change nothing for the others
    moved = dict(alone)
    moved["features"] = alone["features"].clone()
    moved["features"][0, 1:] += 10.0
    for a, b in zip(solo, _run(model, moved)):
        finite = torch.isfinite(a[0, 0])
        assert torch.allclose(a[0, 0][finite], b[0, 0][finite], atol=1e-5)
    # the aircraft's order changes only the order of the outputs
    order = [2, 0, 1]
    swapped = {name: value[:, order] for name, value in batch.items()
               if name in ("features", "relative", "static", "in_force", "since", "present", "targets")}
    swapped.update(airport=batch["airport"], edges=batch["edges"][:, :, order][:, :, :, order])
    for a, b in zip(logits, _run(model, swapped)):
        finite = torch.isfinite(a[:, order])
        assert torch.allclose(a[:, order][finite], b[finite], atol=1e-5)


def test_the_baselines_count_the_first_step_and_the_changes_after_it_from_train():
    classes = (3, 3, 4, 3, 3, 3)
    rows = N_LOOK + 4
    targets = np.zeros((rows, 6), dtype=np.int16)
    targets[N_LOOK] = [1, 1, 1, 1, 1, 1]
    targets[N_LOOK + 2, HEADING] = 3                                     # heading changes once, to value 2
    in_force = np.zeros((rows, 6), dtype=np.int16)
    in_force[N_LOOK + 1:] = 1
    in_force[N_LOOK + 3, HEADING] = 3
    flight = Flight("F", 0, np.zeros((rows, 9), np.float32), np.zeros((rows, 2, 6), np.float32), np.zeros(0, np.float32),
                    in_force, np.zeros((rows, 6), np.float32), targets, _asked(rows), 0.0, 0, 0.0, True)
    table = np.zeros((1, 2, len(prior_data.CANDIDATE_FEATURES)), np.float32)
    table[0, :, -1] = 1.0
    split = Split([flight], ("KXXX",), table, (("09", "27"),), ((90.0, 270.0),), classes, "full")
    base = Baselines.count(split)
    assert base.change[HEADING] == pytest.approx((1 + 1) / (3 + 2))
    assert base.unigram[HEADING].tolist() == pytest.approx([1 / 4, 1 / 4, 2 / 4])
    assert base.bigram[HEADING][1].tolist() == pytest.approx([1 / 4, 1 / 4, 2 / 4])   # after value 0 (class 1)
    assert base.first[HEADING].tolist() == pytest.approx([2 / 4, 1 / 4, 1 / 4])
    assert base.first_runway[0].tolist() == pytest.approx([2 / 3, 1 / 3])
    nll = base.nll_per_step(split)
    # heading: the first step's word, then two kept steps and one change
    expected = (-math.log(2 / 4) - 2 * math.log(1 - base.change[HEADING]) - math.log(base.change[HEADING] * 2 / 4)) / 4
    assert nll["repeat"]["heading"] == pytest.approx(expected)
    runway = (-math.log(2 / 3) - 3 * math.log(1 - base.change[RUNWAY])) / 4
    assert nll["repeat"]["runway"] == pytest.approx(runway)
    assert base.airport_runway(split).tolist() == [0]


def test_the_runway_breakdown_reads_direction_then_side():
    table = np.zeros((1, 3, len(prior_data.CANDIDATE_FEATURES)), np.float32)
    table[0, :, -1] = 1.0
    flights = [_flight(f"F{i}", 0, established=i < 2) for i in range(4)]
    # 09L, 09R, 27: two directions
    split = Split(flights, ("KXXX",), table, (("09L", "09R", "27"),), ((90.0, 90.0, 270.0),), (1,) * 6, "full")
    truth = np.array([0, 1, 2, 0])
    out = runway_breakdown(np.array([0, 0, 2, 2]), truth, split)
    assert out["top1"] == 0.5 and out["direction"] == 0.75 and out["side_given_direction"] == pytest.approx(2 / 3)
    assert out["by_establishment"] == {"established": {"flights": 2, "top1": 0.5},
                                       "not established": {"flights": 2, "top1": 0.5}}
    assert out["direction_by_establishment"]["not established"]["top1"] == 0.5


def test_a_speaker_builds_the_rows_training_builds_from_the_same_positions_and_words():
    """`prior.generate.Speaker`, given the observed positions row by row, feeds the model exactly the rows
    `data.flight_steps` builds from those positions and the words it said (its first step as row 0's words)."""
    from ts_transformer.prior.generate import Speaker

    rows = ROWS + 6
    signals, geometry = _signals(rows), _two_runways()
    clock = utc_s(signals.entry_time_utc) + signals.time_s[:rows]
    landings = _landings(times_09=[clock[0] - 60.0, clock[N_LOOK + 3]], times_27=[clock[2]])
    model = _model(slots=3, valid=2)
    speaker = Speaker(model, [signals], [geometry], {"KXXX": landings}, Words(spec()), max_rows=rows,
                      generator=torch.Generator().manual_seed(0))
    on, off = np.array([True]), np.array([False])
    with pytest.raises(ValueError, match="after the first predicted step was said"):
        speaker.append(signals.e_m[[N_LOOK + 1]], signals.n_m[[N_LOOK + 1]], signals.altitude_m[[N_LOOK + 1]], off)
    said = [speaker.speak(on, off)[0]]
    assert (said[0] > 0).all()                                           # the first predicted step says every column
    for t in range(N_LOOK + 1, rows):
        speaker.append(signals.e_m[[t]], signals.n_m[[t]], signals.altitude_m[[t]], off)
        said.append(speaker.speak(on, off)[0])
    grid = np.full((rows, 6), UNCHANGED, dtype=np.int64)
    grid[0] = said[0] - 1
    grid[N_LOOK + 1:] = np.where(np.array(said[1:]) > 0, np.array(said[1:]) - 1, UNCHANGED)
    features, relative, _, in_force, since, _ = flight_steps(signals, grid, geometry, landings)
    assert np.allclose(speaker.features[0, 0, :rows].numpy(), features, atol=1e-6)
    assert np.allclose(speaker.relative[0, 0, :rows, :2].numpy(), relative, atol=1e-6)
    assert (speaker.relative[0, 0, :rows, 2:] == 0).all()                # the model's empty slot
    assert np.array_equal(speaker.in_force[0, 0, :rows].numpy(), in_force)
    assert np.allclose(speaker.since[0, 0, :rows].numpy(), since, atol=1e-6)
    # the same seed says the same words
    again = Speaker(model, [signals], [geometry], {"KXXX": landings}, Words(spec()), max_rows=rows,
                    generator=torch.Generator().manual_seed(0))
    assert np.array_equal(again.speak(on, off)[0], said[0])
    with pytest.raises(ValueError, match="landing context given disagree"):
        Speaker(model, [signals], [geometry], None, Words(spec()), max_rows=rows, generator=torch.Generator())


def test_a_speaker_keeps_a_locked_runway_says_nothing_when_done_and_freezes_a_finished_flight():
    """The runway column: never the runway in force again, no other where the listener locks it; an inactive flight says
    nothing; a frozen flight's row repeats its last position (its state may be non-finite)."""
    from ts_transformer.prior.generate import Speaker

    rows = ROWS + 40
    signals, geometry = _signals(rows), _two_runways()
    model = _model(variant="no-context", slots=3, valid=2)
    speaker = Speaker(model, [signals] * 8, [geometry] * 8, None, Words(spec()), max_rows=rows + 1,
                      generator=torch.Generator().manual_seed(1))
    active, locked = np.ones(8, dtype=bool), np.arange(8) < 4
    speaker.speak(active, np.zeros(8, dtype=bool))
    changed = False
    for t in range(N_LOOK + 1, rows):
        before = speaker.value[:, RUNWAY].copy()
        speaker.append(signals.e_m[[t] * 8], signals.n_m[[t] * 8], signals.altitude_m[[t] * 8], np.zeros(8, dtype=bool))
        said = speaker.speak(active, locked)
        assert (said[locked, RUNWAY] == 0).all()                         # locked: no runway word at all
        assert not ((said[:, RUNWAY] > 0) & (said[:, RUNWAY] == before)).any()   # never the one in force again
        changed |= bool((said[~locked, RUNWAY] > 0).any())
    assert changed                                   # the untrained model does change runways where it may
    assert all(mass[locked].min() > 0 for mass in speaker.forbidden[RUNWAY][1:])   # what the lock removed, recorded
    # done: says nothing, its row frozen at its last position whatever the listener reports
    done = np.arange(8) == 0
    speaker.append(np.full(8, np.nan), np.full(8, np.nan), np.full(8, np.nan), np.ones(8, dtype=bool))
    said = speaker.speak(~done, np.zeros(8, dtype=bool))
    assert (said[0] == 0).all()
    assert np.isfinite(speaker.e[:, speaker.rows - 1]).all() and speaker.e[0, speaker.rows - 1] == speaker.e[0, speaker.rows - 2]


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

    items = {"train": flights("T", 12, "train"), "select": flights("S", 4, "select"), "val": flights("V", 4, "val")}
    write_signals(directory, items, {"note": "test"}, fixture_days())
    write_candidates(directory, {"KXXX": _two_runways()})
    write_spec(directory, one, {"n": 1}, {"labeller_source_sha256": labeller_source_sha256(),
                                          "git": {"head": "test", "dirty": False}})
    for split, signals in items.items():
        readings = [read_flight(flight, _two_runways(), one) for flight in signals]
        assert all(reading.words is not None for reading in readings)
        write_sentences(directory, split, one, readings, list(range(len(signals))))
    return one


def _tracks(path):
    """A tracks roster: the fixture flights' own landings (09, midday of their split's day), landings on 09 two hours
    before them, one on a test day (sealed)."""
    rows = [{"outcome": "assigned", "runway": "09", "flight_key": f"L{split}",
             "landing_time_utc": landing_on(split).replace("T12:", "T10:")} for split in ("train", "select", "val", "test")]
    rows += [{"outcome": "assigned", "runway": "09", "flight_key": f"own{split}", "landing_time_utc": landing_on(split)}
             for split in ("train", "select", "val")]
    rows.append({"outcome": "assigned", "runway": "27", "flight_key": "Ltrain27",
                 "landing_time_utc": f"{fixture_days().days['train'][0]}T10:30:00Z"})
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")
    return path


def test_the_split_refuses_a_test_day_flight(tmp_path):
    one = _artefact(tmp_path / "artefact")
    record = json.loads((tmp_path / "artefact" / "signals.json").read_text(encoding="utf-8"))
    record["splits"]["train"]["flights"][0]["landing_time_utc"] = landing_on("test")
    (tmp_path / "artefact" / "signals.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SealedDay):
        load_split(tmp_path / "artefact", "train", one, Words(one), "full", landings={"KXXX": _landings()})


def test_the_runners_train_every_variant_choose_by_the_rule_and_read_val_once(tmp_path, capsys, monkeypatch):
    from ts_transformer.experiments import prior_select, prior_train

    artefact, campaign = tmp_path / "artefact", tmp_path / "campaign"
    _artefact(artefact)
    roster = _tracks(tmp_path / "tracks.json")
    monkeypatch.setattr(prior_train, "tracks_manifest_path", lambda code: roster)
    common = ["--instructions", str(artefact), "--device", "cpu", "--limit", "20", "--max-epochs", "1",
              "--tokens-per-batch", "2048"]
    for name in prior_select.PREFERENCE:
        assert prior_train.main([*common, "--variant", name, "--out", str(campaign / f"{name}_s1337")]) == 0
    with pytest.raises(SystemExit, match="no run of"):
        prior_select.main(["--campaign", str(campaign), "--device", "cpu"])
    assert prior_train.main([*common, "--variant", "full", "--seed", "2024", "--out", str(campaign / "full_s2024")]) == 0
    capsys.readouterr()
    assert prior_select.main(["--campaign", str(campaign), "--device", "cpu"]) == 0

    config = json.loads((campaign / "full_s1337" / "config.json").read_text())
    assert config["smoke"] and config["flights"] == {"train": 12, "select": 4, "val": 4}
    assert config["instructions"]["day_split"] == fixture_days().to_dict() and set(config["tracks_rosters"]) == {"KXXX"}
    selection = json.loads((campaign / "full_s1337" / "selection_readout.json").read_text())
    assert selection["split"] == "select" and selection["model"]["flights"] == 4
    assert set(selection["rules"]) == {"B0_majority", "B1_active_config", "B3_same_sector_last"}
    # the fixture roster's landings are not the artefact's flights: none has a known entry sector
    assert selection["rules_sector_known_share"] == {"KXXX": 0.0}
    choice = json.loads((campaign / "choice.json").read_text())
    nll = choice["nll_per_step"]
    assert set(prior_select.PREFERENCE) == set(VARIANTS)
    assert choice["leader"] == min(nll, key=nll.get) and choice["chosen"] == choice["within_seed_line"][0]
    assert len(choice["runs"]) == 4
    readout = json.loads((campaign / choice["chosen_directory"] / "readout.json").read_text())
    assert readout["split"] == "val" and readout["model"]["flights"] == 4
    first = readout["model"]["first_step_runway"]
    assert first["flights"] == 4 and 0.0 <= first["top1"] <= first["top2"] == 1.0   # two candidates
    assert readout["model"]["per_column"]["runway"]["first_step_top1"] == first["top1"]
    assert sum(part["flights"] for part in first["by_establishment"].values()) == 4
    assert sum(1 for path in campaign.glob("*/readout.json")) == 1        # val is read once, on the chosen run
    with pytest.raises(SystemExit):
        prior_select.main(["--campaign", str(campaign), "--device", "cpu"])   # never twice


def test_the_rule_prefers_fewer_inputs_then_ordered_heads_within_the_seed_line():
    from ts_transformer.experiments.prior_select import choose

    def runs(nll, replicate):
        out = {(name, 1337): {"readout": {"model": {"nll_per_step": value}}} for name, value in nll.items()}
        out[("full", 2024)] = {"readout": {"model": {"nll_per_step": replicate}}}
        return out

    # unordered leads by 0.004 over full; the seeds differ by 0.005: full and unordered inside, no-context not
    rule = choose(runs({"full": 0.504, "no-context": 0.520, "unordered": 0.500}, 0.509), 1337, 2024)
    assert rule["leader"] == "unordered" and rule["seed_line"] == pytest.approx(0.005)
    assert rule["within_seed_line"] == ["full", "unordered"] and rule["chosen"] == "full"
    # no-context within the line: fewer inputs win
    assert choose(runs({"full": 0.500, "no-context": 0.503, "unordered": 0.510}, 0.505), 1337, 2024)["chosen"] == "no-context"
    # a seed line of zero keeps the leader
    assert choose(runs({"full": 0.504, "no-context": 0.520, "unordered": 0.500}, 0.504), 1337, 2024)["chosen"] == "unordered"
    with pytest.raises(SystemExit, match="no run of"):
        choose({key: value for key, value in runs({"full": 1, "no-context": 1, "unordered": 1}, 1).items()
                if key[1] == 1337}, 1337, 2024)
    with pytest.raises(SystemExit, match="no seed line"):
        choose(runs({"full": 1, "no-context": 1, "unordered": 1}, 1), 1337, 1337)
