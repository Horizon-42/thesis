"""Multi-aircraft M2's readout (`experiments/traffic_scene_readout`): the three readings of the same cells, the paired
differences and the whole runner on a tmp artefact (never a live root)."""

import json

import numpy as np
import pytest
import torch

from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.test_traffic_scene_data import _artefact


def _priors(directory, spec, *edge_features):
    """An untrained small base prior and scene prior of the fixture artefact (one airport), and its split."""
    from ts_transformer.instructions.artefact import load_candidates
    from ts_transformer.instructions.words import Words
    from ts_transformer.prior.data import candidate_table, column_classes, load_split
    from ts_transformer.prior.model import Prior, PriorConfig

    geometries = load_candidates(directory)
    slots = max(len(g.candidates) for g in geometries.values())
    config = PriorConfig(classes=column_classes(Words(spec), slots), airports=("KXXX",), candidate_slots=slots,
                         variant="no-context", d_model=32, layers=2, heads=4, feedforward=64, dropout=0.0)
    table = torch.as_tensor(candidate_table(geometries, ("KXXX",), slots))
    torch.manual_seed(0)
    base = Prior(config, table).eval()
    torch.manual_seed(1)
    scene = Prior(config, table, edge_features).eval()
    split = load_split(directory, "train", spec, Words(spec), "no-context", landings={}, airports=("KXXX",))
    return base, scene, split, slots


def test_the_scene_and_alone_readings_ask_base_s_cells_and_a_lone_aircraft_sees_nothing_else(tmp_path, monkeypatch):
    """Two labelled flights and a background one in one sample, a labelled flight alone in another: the scene samples ask
    exactly base's cells; where an aircraft is alone its scene reading is its alone reading (KL 0); where it has
    company the untrained network's edges move its words (KL > 0)."""
    from ts_transformer.experiments import traffic_scene_data
    from ts_transformer.experiments.traffic_scene_data import build_split
    from ts_transformer.experiments.traffic_scene_readout import base_cells, scene_cells
    from ts_transformer.inference.scene_edges import EDGE_FEATURES

    directory, manifest, spec, _ = _artefact(tmp_path, [0.0, 30.0, 10.0, 3_600.0], [0, 1, 3])
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    base, scene, split, slots = _priors(directory, spec, *EDGE_FEATURES)
    built, _ = build_split(directory, "train", spec, ("KXXX",), None, 2_048)
    cells = scene_cells(scene, built, slots, 4_096, torch.device("cpu"))
    base_nll = base_cells(base, split, 4_096, torch.device("cpu"))
    assert sorted(cells) == sorted(f.dataset_id for f in split.flights)
    for k, flight in enumerate(split.flights):
        own = cells[flight.dataset_id]
        assert np.array_equal(np.isnan(own["scene"]), np.isnan(base_nll[k]))
        assert np.array_equal(np.isnan(own["scene"]), ~flight.asked)
        assert not np.isnan(own["scene"][N_LOOK:]).all() and np.isnan(own["scene"][:N_LOOK]).all()
    lonely = cells["KXXX:f3"]                                          # an hour later: its own sample
    asked = ~np.isnan(lonely["scene"])
    np.testing.assert_allclose(lonely["scene"][asked], lonely["alone"][asked], atol=1e-5)
    assert np.abs(lonely["kl"][asked]).max() < 1e-5
    company = cells["KXXX:f0"]
    assert np.nanmax(company["kl"]) > 1e-4 and np.nanmin(company["kl"]) > -1e-6


def test_a_cell_asked_by_two_samples_is_refused(tmp_path, monkeypatch):
    from ts_transformer.experiments import traffic_scene_data
    from ts_transformer.experiments.traffic_scene_data import build_split
    from ts_transformer.experiments.traffic_scene_readout import scene_cells
    from ts_transformer.inference.scene_edges import EDGE_FEATURES

    directory, manifest, spec, _ = _artefact(tmp_path, [0.0, 3_600.0], [0, 1])
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    _, scene, _, slots = _priors(directory, spec, *EDGE_FEATURES)
    built, _ = build_split(directory, "train", spec, ("KXXX",), None, 2_048)
    with pytest.raises(ValueError, match="asked by two samples"):
        scene_cells(scene, [built[0], built[0]], slots, 4_096, torch.device("cpu"))


def test_the_paired_means_split_by_flag_start_after_the_first_predicted_row_and_resample_clusters():
    from ts_transformer.experiments.traffic_interaction import READ, READ_FROM
    from ts_transformer.experiments.traffic_scene_readout import PAIRED, paired
    from ts_transformer.instructions.words import COLUMNS

    rows = READ_FROM + 4

    def flight(cluster, flagged_value, other_value):
        values = np.full((rows, 6), np.nan)
        values[:READ_FROM] = 100.0                                    # asked but before the first row read
        values[READ_FROM: READ_FROM + 2] = flagged_value
        values[READ_FROM + 2:] = other_value
        flag = np.arange(rows) < READ_FROM + 2
        return {**{name: values for name in PAIRED}, "leader": flag, "busy": ~flag, "stratum": np.zeros(rows, int),
                "cluster": cluster}

    out = paired([flight(("A", 0), -0.1, 0.02), flight(("A", 1), -0.3, 0.04)], np.random.default_rng(0))
    speed = out[COLUMNS[READ[0]]]
    assert speed["leader"]["flagged"]["steps"] == 4 and speed["leader"]["other"]["steps"] == 4
    assert speed["leader"]["flagged"]["scene_minus_base"]["mean"] == pytest.approx(-0.2)
    assert speed["leader"]["other"]["kl_scene_alone"]["mean"] == pytest.approx(0.03)
    low, high = speed["leader"]["flagged"]["scene_minus_base"]["95"]
    assert -0.3 - 1e-9 <= low <= -0.2 <= high <= -0.1 + 1e-9
    assert speed["busy"]["flagged"]["scene_minus_base"]["mean"] == pytest.approx(0.03)


def test_base_s_matched_gap_is_m1_s_to_the_bit_and_every_quantity_is_drawn_alike():
    """Two airports of random flights: the readout's matched gap of base equals M1's (`read_group` airport by airport,
    then the pool, one generator), intervals included; a reading shifted by a constant has the same gap and intervals."""
    from ts_transformer.experiments.traffic_interaction import (
        BOOTSTRAP_SEED, READ, READ_FROM, STRATA, pooled_flights, read_group,
    )
    from ts_transformer.experiments.traffic_scene_readout import report

    rng = np.random.default_rng(3)

    def flight(airport, hour):
        rows = READ_FROM + 12
        base = rng.gamma(1.0, 0.2, size=(rows, 6))
        base[:N_LOOK] = np.nan
        flag = rng.random(rows) < 0.4
        return {"base": base, "scene": base + 0.01, "alone": base + 0.02, "scene_minus_base": np.full_like(base, 0.01),
                "scene_minus_alone": np.full_like(base, -0.01), "kl_scene_alone": np.abs(base - 0.2),
                "leader": flag, "busy": ~flag, "stratum": rng.integers(0, STRATA, rows), "cluster": (airport, hour)}

    groups = {airport: [flight(airport, h % 4) for h in range(12)] for airport in ("KAAA", "KBBB")}
    airports, pooled = report(groups)

    def m1(flights):
        return [{"nll": np.where(np.arange(len(f["stratum"]))[:, None] >= READ_FROM, f["base"][:, list(READ)], np.nan),
                 "p_change": np.zeros((len(f["stratum"]), len(READ))), "changed": np.zeros((len(f["stratum"]), len(READ)),
                                                                                            dtype=bool),
                 **{k: f[k] for k in ("leader", "busy", "stratum", "cluster")}} for f in flights]

    draws = np.random.default_rng(BOOTSTRAP_SEED)
    reference = {airport: read_group(m1(flights), STRATA, draws) for airport, flights in groups.items()}
    reference_pool = read_group(m1(pooled_flights(groups)), 2 * STRATA, draws)
    for column in ("speed", "heading", "approach"):
        for flag in ("leader", "busy"):
            assert pooled["matched"]["base"][column][flag] == reference_pool[column][flag]["nll"]
            for airport in groups:
                assert airports[airport]["matched"]["base"][column][flag] == reference[airport][column][flag]["nll"]
            shifted, own = pooled["matched"]["scene"][column][flag], pooled["matched"]["base"][column][flag]
            assert shifted["difference"] == pytest.approx(own["difference"])
            assert shifted["difference_95"] == pytest.approx(own["difference_95"])


def test_the_nll_per_step_is_the_training_runners_measure():
    from ts_transformer.experiments.traffic_scene_readout import nll_per_step

    one = np.full((3, 6), np.nan)
    one[1, 0], one[1, 2], one[2, 0] = 1.0, 2.0, 3.0                   # two rows ask: (1 + 3) / 2 and 2 / 2
    out = nll_per_step([one])
    assert out["steps"] == 2 and out["nll_per_step"] == pytest.approx(3.0)


def test_the_readout_runner_reads_a_scene_prior_against_a_base_prior(tmp_path, monkeypatch):
    """The runner on the fixture's days (every flight on a train day, read as the split): a scene prior trained by its
    runner for one epoch and an untrained base of the same variant; the record holds every reading and quantity."""
    from ts_transformer.experiments import prior_train, traffic_prior_train, traffic_scene_data, traffic_scene_readout
    from ts_transformer.tests.test_training_overlays import _prior_dir

    directory, manifest, _, _ = _artefact(tmp_path, [0.0, 30.0, 10.0, 3_600.0], [0, 1, 3])
    roster = tmp_path / "tracks.json"
    roster.write_text(json.dumps({"records": []}), encoding="utf-8")
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    monkeypatch.setattr(prior_train, "tracks_manifest_path", lambda code: roster)
    real = traffic_scene_data.build_split
    monkeypatch.setattr(traffic_prior_train, "build_split", lambda d, split, *rest: real(d, "train", *rest))
    scene = tmp_path / "scene"
    assert traffic_prior_train.main(["--instructions", str(directory), "--out", str(scene), "--variant", "no-context",
                                     "--device", "cpu", "--limit", "5", "--max-epochs", "1", "--warmup-steps", "1"]) == 0
    _prior_dir(tmp_path / "base", directory, roster, variant="no-context")
    monkeypatch.setattr(traffic_scene_readout, "SPLIT", "train")
    out = tmp_path / "readout"
    args = ["--scene", str(scene), "--base", str(tmp_path / "base"), "--instructions", str(directory), "--out", str(out)]
    assert traffic_scene_readout.main(args) == 0
    record = json.loads((out / "scene_readout.json").read_text(encoding="utf-8"))
    pooled = record["pooled"]
    assert set(pooled["nll_per_step"]) == {"base", "scene", "alone"} and pooled["flights"] == 3
    assert set(pooled["paired"]["speed"]["leader"]["flagged"]) >= {"steps", "scene_minus_base", "kl_scene_alone"}
    assert set(pooled["matched"]) == {"base", "scene", "alone", "scene_minus_base", "scene_minus_alone", "kl_scene_alone"}
    assert record["smoke"] == {"scene": True, "base": False}
    with pytest.raises(SystemExit):                                                   # never overwritten
        traffic_scene_readout.main(args)
    with pytest.raises(SystemExit, match="--scene takes"):                           # the two roles are not swapped
        traffic_scene_readout.main(["--scene", str(tmp_path / "base"), "--base", str(scene), "--instructions",
                                    str(directory), "--out", str(tmp_path / "swapped")])
