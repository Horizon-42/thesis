"""The scene samples the scene prior reads (`prior/scene_data`, `experiments/traffic_scene_data`, multi-aircraft design §2.3,
§2.5): nodes on a sample's steps, a batch of samples, and samples built from a tmp artefact (never a live root)."""

import dataclasses
import json
from datetime import timedelta

import numpy as np
import pytest
import torch

from ts_transformer.prior.scene import N_LOOK


def _node(key, first_step, rows, *, speaking=True, width=3, slots=1):
    from ts_transformer.prior.scene_data import Node

    asked = np.zeros((rows, 6), dtype=bool)
    asked[N_LOOK:] = speaking
    return Node(key, speaking, first_step, np.full((rows, width), float(first_step + 1), dtype=np.float32),
                np.zeros((rows, slots, 2), dtype=np.float32), np.zeros(0, dtype=np.float32),
                np.ones((rows, 6), dtype=np.int64), np.zeros((rows, 6), dtype=np.float32),
                np.ones((rows, 6), dtype=np.int64), asked)


def _placed(sample):
    from ts_transformer.prior.scene_data import placed

    return {name: value[0] for name, value in placed([sample], 1).items() if name != "airport"}


def test_nodes_sit_on_their_steps_with_their_own_rows_and_only_the_loss_steps_count():
    from ts_transformer.prior.scene_data import SceneSample

    carried = _node("A", 0, 30)                      # from before the cut: its first 12 steps are context
    late = _node("B", 20, 30)                        # past the sample's last step at 40: rows 20+ dropped
    background = _node("C", 5, 10, speaking=False)
    sample = SceneSample(0, (carried, late, background), 40, 12, 40)
    laid = _placed(sample)
    assert laid["present"].sum(axis=1).tolist() == [30, 20, 10]
    assert laid["rows"][1, 20:].tolist() == list(range(20)) and laid["rows"][1, :20].tolist() == [0] * 20
    asked = laid["asked"].any(axis=-1)
    assert asked[0].tolist() == [False] * 12 + [True] * 18 + [False] * 10             # from the loss start, to its end
    assert asked[1].tolist() == [False] * (20 + N_LOOK) + [True] * (20 - N_LOOK)      # from its own first predicted row
    assert not asked[2].any()                                                        # background: never
    assert sample.asks and not SceneSample(0, (background,), 40, 12, 40).asks
    with pytest.raises(ValueError, match="aircraft axis"):
        SceneSample(0, tuple(_node(str(k), 0, 10) for k in range(19)), 10, 0, 10)
    with pytest.raises(ValueError, match="outside the sample"):
        SceneSample(0, (_node("A", 40, 10),), 40, 0, 40)


def test_a_batch_pads_aircraft_steps_and_candidate_slots():
    from ts_transformer.prior.scene_data import SceneSample, batch_tensors

    one = SceneSample(0, (_node("A", 0, 12),), 12, 0, 12)
    two = SceneSample(1, (_node("B", 0, 20, slots=2), _node("C", 3, 10, slots=2)), 20, 0, 20)
    edges = [np.ones((12, 1, 1, 4), dtype=np.float32), np.full((20, 2, 2, 4), 2.0, dtype=np.float32)]
    batch = batch_tensors([one, two], edges, 3, torch.device("cpu"))
    assert tuple(batch["features"].shape) == (2, 2, 20, 3) and tuple(batch["relative"].shape) == (2, 2, 20, 3, 2)
    assert tuple(batch["edges"].shape) == (2, 20, 2, 2, 4) and batch["airport"].tolist() == [0, 1]
    assert not batch["present"][0, 1].any() and not batch["present"][0, 0, 12:].any()
    assert float(batch["edges"][0, 12:].abs().sum()) == 0.0 and float(batch["edges"][1].min()) == 2.0


def _artefact(tmp_path, entries_s, labelled, background_rows_before=0):
    """Flights of one vectored approach onto the fixture's runway 09 entering at ``entries_s`` (seconds after the first),
    those at ``labelled`` with a sentence, the rest background (flying ``background_rows_before`` more rows first); a
    tmp arrivals manifest recorded as the signals'."""
    from ts_transformer.data.day_split import parse_utc
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
    base = instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0))
    longer = instruction_flight(*fly_legs([(background_rows_before, 0.0, 100.0, 0.0), *legs] if background_rows_before
                                          else legs, 270.0, 1110.0, -400.0, 0.0))
    entry = parse_utc(base.landing_time_utc) - timedelta(seconds=float(base.time_s[-1]) + 2.0)

    def at(n, later_s):
        flight = base if n in labelled else longer
        enters = entry + timedelta(seconds=later_s)
        lands = enters + timedelta(seconds=float(flight.time_s[-1]) + 2.0)
        return dataclasses.replace(flight, dataset_id=f"KXXX:f{n}", entry_time_utc=enters.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                   landing_time_utc=lands.strftime("%Y-%m-%dT%H:%M:%SZ"))

    flights = [at(n, s) for n, s in enumerate(entries_s)]
    directory = tmp_path / "artefact"
    directory.mkdir()
    write_signals(directory, {"train": flights},
                  {"counts": {"train": {"built_usable": len(flights)}}, "test_days": {"flights_not_opened": 0},
                   "sources": [{"airport": "KXXX", "arrival_manifest_sha256": file_sha256(manifest)}]}, fixture_days())
    write_candidates(directory, {"KXXX": instruction_airport()})
    spec = instruction_spec()
    write_spec(directory, spec, {"n": 1}, {"labeller_source_sha256": labeller_source_sha256(),
                                           "git": {"head": "test", "dirty": False}})
    write_sentences(directory, "train", spec, [read_flight(flights[n], instruction_airport(), spec) for n in labelled],
                    list(labelled))
    return directory, manifest, spec, flights


def test_samples_built_from_an_artefact(tmp_path, monkeypatch):
    """Two labelled flights 30 s apart and a background one between them make one sample; a flight an hour later its
    own. In the edge inputs a labelled flight has no runway until its first predicted step's words are in force."""
    from ts_transformer.experiments import traffic_scene_data
    from ts_transformer.experiments.traffic_scene_data import build_split, edges
    from ts_transformer.inference.scene_edges import EDGE_FEATURES

    directory, manifest, spec, flights = _artefact(tmp_path, [0.0, 30.0, 10.0, 3_600.0], [0, 1, 3])
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    built, counts = build_split(directory, "train", spec, ("KXXX",), None, 2_048)
    assert counts == {"with_a_sentence": 3, "background": 1, "background_left_out_for_its_length": 0,
                      "without_a_type": 0, "samples": 2, "most_aircraft": 3}
    first, alone = built
    sample = first.sample
    assert sample.keys == ("KXXX:f0", "KXXX:f2", "KXXX:f1") and [n.speaking for n in sample.nodes] == [True, False, True]
    laid = _placed(sample)
    starts = [int(np.argmax(p)) for p in laid["present"]]
    assert starts == [0, 5, 15]                                        # 10 s and 30 s later: 5 and 15 steps
    asked = laid["asked"].any(axis=-1)
    assert int(np.argmax(asked[2])) == 15 + N_LOOK and not asked[1].any()
    scene = first.rows
    assert scene.runway[0][N_LOOK] is None and scene.runway[0][N_LOOK + 1] == "09" and all(r is None for r in scene.runway[1])
    assert np.isfinite(scene.along_m[0, N_LOOK + 1]) and np.isnan(scene.along_m[0, N_LOOK])
    assert np.array_equal(~np.isnan(scene.time_s), laid["present"])
    values = edges(first)
    assert values.shape == (sample.steps, 3, 3, len(EDGE_FEATURES)) and np.isfinite(values).all()
    # each flight's rows are held once: the sample's nodes share its flights' arrays
    assert all(np.shares_memory(node.features, f.node_rows["features"]) for node, f in zip(sample.nodes, first.flights))
    assert len(alone.sample.keys) == 1
    # the model's positions bound a sentence's length: refused, not trimmed
    with pytest.raises(ValueError, match="positions hold 200"):
        build_split(directory, "train", spec, ("KXXX",), None, 200)
    # two flights under one dataset id would leave one out of the scene: refused
    twins = [flights[0], dataclasses.replace(flights[1], dataset_id=flights[0].dataset_id), *flights[2:]]
    monkeypatch.setattr(traffic_scene_data, "load_signals", lambda d, split: twins)
    with pytest.raises(ValueError, match="share a dataset id"):
        build_split(directory, "train", spec, ("KXXX",), None, 2_048)


def test_a_background_track_longer_than_the_positions_is_left_out_and_counted(tmp_path, monkeypatch):
    from ts_transformer.experiments import traffic_scene_data
    from ts_transformer.experiments.traffic_scene_data import build_split

    directory, manifest, spec, flights = _artefact(tmp_path, [0.0, 10.0], [0], background_rows_before=40)
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    assert flights[1].n_rows > 240 >= flights[0].n_rows
    built, counts = build_split(directory, "train", spec, ("KXXX",), None, 240)
    assert counts["background_left_out_for_its_length"] == 1 and counts["background"] == 0
    assert [b.sample.keys for b in built] == [("KXXX:f0",)]


def test_a_long_segment_is_cut_and_a_flight_across_the_cut_is_context_before_it(tmp_path, monkeypatch):
    """Five labelled flights entering 6 minutes apart, each in the scene ~7.7 minutes: one 31-minute segment, cut once
    between its 15th and 20th minute; the flight across the cut starts the second sample's steps, its rows before the cut
    only context."""
    from ts_transformer.experiments import traffic_scene_data
    from ts_transformer.experiments.traffic_scene_data import build_split

    entries = [360.0 * n for n in range(5)]
    directory, manifest, spec, _ = _artefact(tmp_path, entries, list(range(5)))
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    built, counts = build_split(directory, "train", spec, ("KXXX",), None, 2_048)
    assert counts["samples"] == 2
    second = _placed(built[1].sample)
    carried = [a for a, p in enumerate(second["present"]) if p[0]]
    assert len(carried) >= 1
    a = carried[0]
    asked = second["asked"][a].any(axis=-1)
    loss_from = int(np.argmax(second["asked"].any(axis=-1).any(axis=0)))
    assert loss_from > 0 and not asked[:loss_from].any() and asked[loss_from:].any()
    assert second["rows"][a, 0] == 0                                     # it starts at its own row 0: context


def test_scene_batches_group_samples_by_padded_size():
    from types import SimpleNamespace

    from ts_transformer.experiments.traffic_prior_train import scene_batches

    sizes = [(1, 100), (2, 100), (3, 100), (1, 900), (4, 400)]
    built = [SimpleNamespace(sample=SimpleNamespace(keys=("x",) * a, steps=t)) for a, t in sizes]
    groups = list(scene_batches(built, 600, None))
    assert sorted(i for g in groups for i in g) == list(range(5))
    for group in groups:
        aircraft = max(sizes[i][0] for i in group)
        steps = max(sizes[i][1] for i in group)
        assert len(group) == 1 or aircraft * steps * len(group) <= 600


def test_the_scene_training_runner_writes_a_scene_prior_its_loader_reads(tmp_path, monkeypatch):
    """Two epochs on the fixture's samples (no landing context, CPU, SMOKE): a v4 checkpoint whose edge features are
    `inference.scene_edges`' and which `load_prior` reads back."""
    from ts_transformer.experiments import prior_train, traffic_prior_train, traffic_scene_data
    from ts_transformer.experiments.prior_train import SCENE_CHECKPOINT_SCHEMA, load_prior
    from ts_transformer.inference.scene_edges import EDGE_FEATURES

    directory, manifest, _, _ = _artefact(tmp_path, [0.0, 30.0, 10.0, 3_600.0], [0, 1, 3])
    roster = tmp_path / "tracks.json"
    roster.write_text(json.dumps({"records": []}), encoding="utf-8")
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    monkeypatch.setattr(prior_train, "tracks_manifest_path", lambda code: roster)
    # the fixture's days: every flight is on a train day; val holds none, so train is read for both
    real = traffic_scene_data.build_split
    monkeypatch.setattr(traffic_prior_train, "build_split",
                        lambda d, split, *rest: real(d, "train", *rest))
    out = tmp_path / "scene"
    args = ["--instructions", str(directory), "--out", str(out), "--variant", "no-context", "--device", "cpu",
            "--limit", "5", "--max-epochs", "2", "--warmup-steps", "1"]
    assert traffic_prior_train.main(args) == 0
    loaded = load_prior(out, directory)
    assert loaded.payload["schema"] == SCENE_CHECKPOINT_SCHEMA and loaded.model.edge_features == EDGE_FEATURES
    assert loaded.payload["edge_source_sha256"] == traffic_scene_data.edge_source_sha256()
    epochs = json.loads((out / "history.json").read_text(encoding="utf-8"))["epochs"]
    assert len(epochs) == 2 and all(row["batches"] >= 1 and row["asked_steps"] > 0 for row in epochs)
    with pytest.raises(SystemExit):                                                   # never overwritten
        traffic_prior_train.main(args)
