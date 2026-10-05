"""Stage C, C2: the edge features and the tokens of the traffic attention (post-training §3, §4 item 1, §8 C2; D23,
D31, D98)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.post.conformance import (
    EDGES_REFERENCE_SCHEMA, reference_steps, require_conforming_edges, write_edge_reference,
)
from ts_transformer.post.edges import SCALE_M, TOKEN_FEATURES, tokens
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import AircraftAt, airport_scenes, real_windows
from ts_transformer.tests.post_support import airport, at_step, categories, scene_artefact, straight_in, train_noon
from ts_transformer.tests.support import INSTRUCTION_STEP_S

DELTA = 4.0
F = {name: k for k, name in enumerate(TOKEN_FEATURES)}


def _stream():
    """Five train flights onto 09 and 09L (891 m apart: dependent parallels), some in the air together."""
    return {"train": [straight_in("KXXX:a", train_noon(0.0)), straight_in("KXXX:b", train_noon(100.0)),
                      straight_in("KXXX:c", train_noon(150.0), runway="09L", offset_n=891.0),
                      straight_in("KXXX:d", train_noon(210.0), typecode=None),
                      straight_in("KXXX:e", train_noon(260.0), speed_mps=65.0)]}


@pytest.fixture
def built(tmp_path):
    spec = scene_artefact(tmp_path / "art", _stream(), interval_s=DELTA)
    geometries = load_candidates(tmp_path / "art")
    scenes, signals = airport_scenes(tmp_path / "art", "train", spec, DELTA, geometries, categories)
    windows = real_windows(tmp_path / "art", "train", spec, DELTA, scenes, signals)
    return geometries["KXXX"], airport_separation(geometries["KXXX"]), scenes["KXXX"], windows


def _own(window, time_s):
    return AircraftAt.of([window.commanded.at_step(time_s, DELTA)])


def _step_tokens(window, geometry, separation, time_s):
    return tokens(_own(window, time_s), window.scene.others_at(time_s, window.commanded.key), geometry, separation,
                  INSTRUCTION_STEP_S)


def test_no_feature_reads_a_later_row(built):
    geometry, separation, scene, windows = built
    window = windows[3]
    time_s = window.first_step_s + 40.0
    before = _step_tokens(window, geometry, separation, time_s)
    assert len(before) >= 2

    def later_changed(flight):
        row = flight.row_at(time_s) if flight.start_s <= time_s <= flight.end_s else -1
        keep = slice(0, row + 1) if row >= 0 else slice(0, 0)
        noise = np.arange(len(flight.e_m)) * 37.0
        noise[keep] = 0.0
        return replace(flight, e_m=flight.e_m + noise, n_m=flight.n_m - noise, height_m=flight.height_m + noise)

    changed = replace(window, commanded=later_changed(window.commanded),
                      scene=scene.with_flights([later_changed(f) for f in scene.flights]))
    assert np.array_equal(before, _step_tokens(changed, geometry, separation, time_s))


def test_a_runway_changes_nothing_up_to_its_first_predicted_step(built):
    """D23 in a scene: a change of one aircraft's runway (a recorded aircraft's record, the commanded aircraft's word)
    leaves every token at the steps up to its first predicted step the same, bit for bit."""
    geometry, separation, scene, windows = built
    window = windows[2]                                   # c, on 09L, in the air with a, b, d, e
    for target in ("KXXX:d", "KXXX:e"):                 # both enter after c's row 0
        flights = [replace(f, runway_index=1 - f.runway_index) if f.key == target else f for f in scene.flights]
        changed = replace(window, scene=scene.with_flights(flights))
        limit = scene.flights[scene.index[target]].first_step_s
        compared, differ = 0, 0
        for step in range(400):
            time_s = window.step_s(step)
            if time_s > window.commanded.end_s:
                break
            same = np.array_equal(_step_tokens(window, geometry, separation, time_s),
                                  _step_tokens(changed, geometry, separation, time_s))
            if time_s <= limit:
                assert same, (target, step)
                compared += 1
            else:
                differ += not same
        assert compared and differ           # and the runway is read once it is in force
    swapped = replace(window, commanded=replace(window.commanded, runway_index=0))
    for step in range(5):          # rows 0 … 16 s: up to the commanded aircraft's first predicted step
        time_s = window.step_s(step)
        assert np.array_equal(_step_tokens(window, geometry, separation, time_s),
                              _step_tokens(swapped, geometry, separation, time_s))


def test_a_permutation_of_the_other_aircraft_permutes_their_tokens():
    geometry = airport()
    separation = airport_separation(geometry)
    rng = np.random.default_rng(3)
    own = at_step(rng, 1)
    others = at_step(rng, 7, runway=1)
    order = rng.permutation(7)
    permuted = AircraftAt(keys=tuple(others.keys[k] for k in order), at=others.at[order], before=others.before[order],
                          known=others.known[order], runway_index=others.runway_index[order],
                          category=tuple(others.category[k] for k in order), last_step=others.last_step[order])
    assert np.array_equal(tokens(own, others, geometry, separation, 2.0)[order],
                          tokens(own, permuted, geometry, separation, 2.0))


def test_an_unknown_motion_gives_the_flag_and_zeros():
    geometry = airport()
    separation = airport_separation(geometry)
    rng = np.random.default_rng(4)
    own, others = at_step(rng, 1), at_step(rng, 3)
    unknown = replace(others, known=np.array([False, True, True]), before=others.before * np.array([[0.0], [1], [1]]))
    got = tokens(own, unknown, geometry, separation, 2.0)
    motion_features = ["closing", "cpa_time", "cpa_horizontal", "cpa_vertical", "ground_speed", "vertical_rate",
                       "direction_sin", "direction_cos"]
    assert got[0, F["motion_unknown"]] == 1.0 and (got[1:, F["motion_unknown"]] == 0.0).all()
    assert (got[0, [F[n] for n in motion_features]] == 0.0).all()
    assert (got[1:, F["ground_speed"]] > 0.0).all()
    blind = tokens(replace(own, known=np.array([False])), others, geometry, separation, 2.0)
    assert (blind[:, F["motion_unknown"]] == 1.0).all()
    assert (blind[:, [F[n] for n in ("front", "left", "closing", "direction_sin", "direction_cos")]] == 0.0).all()
    assert (blind[:, F["height"]] != 0.0).all()


def test_the_runway_features_read_the_rules():
    geometry = airport()
    separation = airport_separation(geometry)
    own = AircraftAt.of([("me", (-12_000.0, 0.0, 800.0), (-12_150.0, 0.0, 803.0), True, 0, "F", False)])
    others = AircraftAt.of([
        ("ahead", (-6_000.0, 0.0, 500.0), (-6_150.0, 0.0, 503.0), True, 0, "F", False),
        ("parallel", (-15_000.0, 891.0, 900.0), (-15_150.0, 891.0, 903.0), True, 1, "F", False),
        ("unsaid", (-9_000.0, 0.0, 700.0), (-9_150.0, 0.0, 703.0), True, -1, "F", False)])
    got = tokens(own, others, geometry, separation, 2.0)
    assert got[0, F["same"]] == 1.0 and got[1, F["dependent"]] == 1.0
    assert got[0, F["clock_ahead"]] == pytest.approx(np.arcsinh(6_000.0 / SCALE_M), abs=1e-6)
    assert got[0, F["required"]] == pytest.approx(np.arcsinh(1.0), abs=1e-6)       # the radar minimum, 3 NM
    assert got[0, F["front"]] == pytest.approx(np.arcsinh(6_000.0 / SCALE_M), abs=1e-6)
    assert got[1, F["left"]] == pytest.approx(np.arcsinh(891.0 / SCALE_M), abs=1e-6)
    assert got[1, F["clock_ahead"]] < 0.0 and got[1, F["required"]] > 0.0        # the diagonal stagger
    assert got[2, F["clock_incomparable"]] == 1.0
    assert (got[2, [F[n] for n in ("same", "single", "dependent", "independent", "unrelated", "required")]] == 0).all()
    assert got[0, F["closing"]] == 0.0 and got[0, F["cpa_time"]] == 1.0             # same speed: never closer


def test_the_edge_reference_is_read_again_and_a_difference_refused(built, tmp_path):
    geometry, _, _, windows = built
    steps = reference_steps(windows, np.random.default_rng(1337), per_airport=3, every_s=40.0)
    assert len(steps) > 5 and any(len(s.others) for s in steps)
    path = tmp_path / "ref" / "edges.npz"
    write_edge_reference(path, steps, {"KXXX": geometry}, INSTRUCTION_STEP_S)
    checked = require_conforming_edges(path)
    assert checked.steps == len(steps) and checked.tokens == sum(len(s.others) for s in steps)
    assert checked.max_difference == 0.0
    with pytest.raises(FileExistsError):
        write_edge_reference(path, steps, {"KXXX": geometry}, INSTRUCTION_STEP_S)
    with np.load(path) as arrays:
        data = {name: arrays[name] for name in arrays.files}
    # the check reads the reference's own geometry: another artefact's candidates change nothing
    assert require_conforming_edges(path).max_difference == 0.0
    for name, change in (("bad", lambda d: d["tokens"] + np.float32(1e-3)),
                         ("nan", lambda d: np.full_like(d["tokens"], np.nan))):
        np.savez(tmp_path / f"{name}.npz", **{**data, "tokens": change(data)})
        with pytest.raises(ValueError, match="does not conform"):
            require_conforming_edges(tmp_path / f"{name}.npz")
    moved = json.loads(str(data["geometries"]))
    moved["KXXX"]["candidates"][0]["course_deg"] += 20.0
    np.savez(tmp_path / "geometry.npz", **{**data, "geometries": np.array(json.dumps(moved))})
    with pytest.raises(ValueError, match="does not conform"):
        require_conforming_edges(tmp_path / "geometry.npz")
    np.savez(tmp_path / "old.npz", **{**data, "schema": np.array("post-edges-reference-v0")})
    with pytest.raises(ValueError, match=EDGES_REFERENCE_SCHEMA):
        require_conforming_edges(tmp_path / "old.npz")
