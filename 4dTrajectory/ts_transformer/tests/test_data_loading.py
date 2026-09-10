"""Loading and splitting: the manifest roster, flight identity, the outer split and the CV folds.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import hashlib
import json
from pathlib import Path

import pytest

import ts_transformer.cli.common as cli_common
import ts_transformer.data.dataset as dataset_module
import ts_transformer.data.splits as splits
import ts_transformer.cli.benchmark_batch as batch_probe
from ts_transformer.config import TSConfig
from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series
from ts_transformer.data.splits import (
    cross_validation_folds,
    split_by_flight,
    split_name_for_dataset_id,
)
from ts_transformer.data.synthetic import synthetic_arrivals
# Imported, never restated: a schema version pinned by hand in a fixture is a version
# the fixture cannot check, and this one gates every loader that reads the roster.
from trajectory_data_process.harvest.arrivals import (
    SCHEMA_VERSION as ARRIVAL_MANIFEST_SCHEMA,
)

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def _write_arrival_manifest(root: Path, ids: list[str], *, airport: str = "KRDU") -> Path:
    from ts_transformer.data.dataset import flight_key

    arrivals = root / "arrivals"
    tracks = root / "tracks"
    records = tracks / "assigned" / "05L"
    records.mkdir(parents=True)
    roster = []
    source_roster = []
    runway_target = {
        "lat": 35.8, "lon": -78.8, "elevation_hae_m": 100.0,
        "elevation_msl_m": 133.5, "hae_minus_msl_m": -33.5,
        "course_deg": 50.0, "threshold_crossing_height_m": 15.0,
        "published_glidepath_deg": 3.0,
        "position_source": "faa_cifp_path_point",
        "vertical_source": "faa_cifp_path_point",
    }
    for index, ident in enumerate(ids):
        flight = {
            "flight_key": None,
            "callsign": ident,
            "icao24": f"abc{index:03d}",
            "runway": "05L",
            "landing_time_utc": f"2026-01-01T00:00:{index:02d}Z",
            "altitude_source": "opensky_history_geoaltitude_m",
            "samples": [[0.0, -78.8, 35.8, 500.0]],
        }
        key = flight_key(
            {
                "id": ident,
                "runway": "05L",
                "icao24": flight["icao24"],
                "landing_time_utc": flight["landing_time_utc"],
            },
            index,
        )
        flight["flight_key"] = key
        relative = f"assigned/05L/{key}.json"
        source_text = json.dumps(flight)
        (tracks / relative).write_text(source_text, encoding="utf-8")
        # The v2 tracks roster row (outcome, runway, landing time): what the lead lookup
        # (intent_conditioning.lead_landings) reads.
        source_roster.append({
            "flight_key": key, "file": relative, "outcome": "assigned", "runway": "05L",
            "landing_time_utc": flight["landing_time_utc"],
        })
        roster.append({
            "flight_key": key,
            "source_file": relative,
            "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "first_sample_index": 0,
            "last_sample_index": 0,
            "runway": "05L",
            "landing_time_utc": flight["landing_time_utc"],
            "arrival_truncated": False,
            "cut_samples": 0,
            "arrival_duration_s": 0.0,
            "entry_time_utc": flight["landing_time_utc"],
        })
    (tracks / "manifest.json").write_text(
        json.dumps({"records": source_roster}), encoding="utf-8"
    )
    manifest = arrivals / "manifest.json"
    arrivals.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": ARRIVAL_MANIFEST_SCHEMA,
                "airport": airport,
                "source_manifest": "../tracks/manifest.json",
                "altitude_source": "opensky_history_geoaltitude_m",
                "runway_targets": {"05L": runway_target},
                "records": roster,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_split_by_flight_is_disjoint_and_reproducible():
    series, config = _series(n_flights=10)
    train_a, val_a, test_a = split_by_flight(series, config)
    train_b, val_b, test_b = split_by_flight(series, config)

    ids = lambda group: [s.flight_id for s in group]  # noqa: E731
    assert ids(train_a) == ids(train_b) and ids(val_a) == ids(val_b)  # seeded -> stable

    everything = ids(train_a) + ids(val_a) + ids(test_a)
    assert len(everything) == len(set(everything)) == len(series)  # partition, no overlap
    for split_name, group in (("train", train_a), ("val", val_a), ("test", test_a)):
        assert all(split_name_for_dataset_id(item.dataset_id, config) == split_name
                   for item in group)


def test_split_seed_locks_outer_split_across_training_seeds():
    series, _config = _series(n_flights=100)
    first = TSConfig(seed=1337, split_seed=1337)
    repeated = TSConfig(seed=2027, split_seed=1337)

    def split_ids(config):
        return tuple(
            tuple(item.dataset_id for item in group)
            for group in split_by_flight(series, config)
        )

    assert split_ids(first) == split_ids(repeated)


def test_pooled_prediction_filters_checkpoint_split_to_current_airport_subset():
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{"airport": "KAAA"}],
    }
    assert cli_common.split_keys_for_current_data(
        ["KAAA:flight-a", "KBBB:flight-b", "KAAA:flight-c"], provenance
    ) == ["KAAA:flight-a", "KAAA:flight-c"]


def test_windows_never_straddle_two_flights():
    # The index is (series, anchor) pairs, so a window cannot span flights. Guard it
    # explicitly: concatenating series into one array would be an easy "optimisation" that
    # silently teaches the model to fly from one aircraft's track into another's.
    series, config = _series(n_flights=3)
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    for s_idx, anchor in dataset.index:
        assert anchor - config.seq_len + 1 >= 0
        assert anchor < series[s_idx].n_samples


def test_ts_load_uses_only_the_arrival_manifest_roster(tmp_path):
    # An orphan beside the roster is deliberately ignored: no glob can leak rejected or
    # stale flights into the train/validation/test split.
    from ts_transformer.data.dataset import load_flight_dicts

    _write_arrival_manifest(tmp_path, ["A", "B", "C"])
    (tmp_path / "tracks" / "assigned" / "05L" / "orphan.json").write_text(
        json.dumps({"id": "ORPHAN"}), encoding="utf-8"
    )

    flights = load_flight_dicts(tmp_path, verbose=False)
    assert [f["id"] for f in flights] == ["A", "B", "C"]
    # Scene context rides along from the same roster: each flight's previous same-runway
    # landing (the fixture lands one per second on 05L).
    from ts_transformer.data.intent_conditioning import LeadLanding

    assert [f["lead_landing"] for f in flights] == [
        LeadLanding(None), LeadLanding("2026-01-01T00:00:00Z"),
        LeadLanding("2026-01-01T00:00:01Z"),
    ]


def test_ts_load_aggregates_multiple_airport_manifests(tmp_path):
    from ts_transformer.data.dataset import load_flight_dicts

    first = _write_arrival_manifest(tmp_path / "first", ["A"], airport="KAAA")
    second = _write_arrival_manifest(tmp_path / "second", ["B"], airport="KBBB")
    flights = load_flight_dicts([second, first], verbose=False)
    assert [(flight["arr_airport"], flight["id"]) for flight in flights] == [
        ("KAAA", "A"), ("KBBB", "B")
    ]


def test_ts_load_filters_qualified_keys_before_opening_source_tracks(tmp_path):
    manifest_path = _write_arrival_manifest(tmp_path, ["A", "B"], airport="KRDU")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_key = manifest["records"][0]["flight_key"]
    excluded = manifest["records"][1]
    (tmp_path / "tracks" / excluded["source_file"]).write_text(
        "excluded source track must stay closed", encoding="utf-8"
    )

    flights = dataset_module.load_flight_dicts(
        manifest_path,
        include_flight_keys={f"KRDU:{selected_key}"},
        verbose=False,
    )

    assert [flight["id"] for flight in flights] == ["A"]


def test_manifest_split_keys_are_resolved_without_loading_trajectory_values():
    config = TSConfig(seed=1337)
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": "KRDU",
            "source_records": [
                {"flight_key": f"flight-{index}", "source_sha256": f"{index:064x}"}
                for index in range(20)
            ],
        }],
    }

    resolved = splits.flight_keys_by_split(provenance, config)

    assert set(resolved) == {"train", "val", "test"}
    assert sum(map(len, resolved.values())) == 20
    assert all(
        split_name_for_dataset_id(key, config) == split
        for split, keys in resolved.items()
        for key in keys
    )


def test_batch_probe_opens_outer_train_track_files_only(tmp_path):
    manifest_path = _write_arrival_manifest(
        tmp_path, [f"F{index:02d}" for index in range(40)]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = TSConfig(seed=1337)
    counts = {"train": 0, "val": 0, "test": 0}
    for row in manifest["records"]:
        split = split_name_for_dataset_id(f"KRDU:{row['flight_key']}", config)
        counts[split] += 1
        if split != "train":
            # A filtered loader would fail JSON/SHA validation if it opened this test/val file.
            source_path = tmp_path / "tracks" / row["source_file"]
            source_path.write_text("not opened by the batch probe", encoding="utf-8")

    flights, audit = batch_probe.load_outer_train_flights([manifest_path], config)
    assert len(flights) == counts["train"]
    assert audit["split_counts_from_manifest_rosters"] == counts
    assert audit["loaded_source_tracks"] == {
        "train": counts["train"], "validation": 0, "test": 0,
    }


def test_ts_load_rejects_duplicate_manifest_identity(tmp_path):
    from ts_transformer.data.dataset import load_flight_dicts

    manifest_path = _write_arrival_manifest(tmp_path, ["A"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"].append(dict(manifest["records"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate flight_key"):
        load_flight_dicts(manifest_path, verbose=False)


def test_ts_load_rejects_legacy_json_input(tmp_path):
    from ts_transformer.data.dataset import load_flight_dicts

    legacy = tmp_path / "KRDU_05L_landings.json"
    legacy.write_text(json.dumps([{"id": "OLD"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="arrival manifest"):
        load_flight_dicts(legacy, verbose=False)


def test_flight_identity_separates_the_same_callsign_on_different_runways():
    # Regression: FlightSeries.flight_id was flight["id"] (the callsign). Across a
    # multi-runway harvest that collides, and the collision is invisible until you notice
    # `predict --split test` returning 48 flights for an 18-flight split — because every
    # namesake on every runway matched. It also leaks the split: three copies of one id
    # land in train, val and test at once.
    from ts_transformer.data.dataset import flight_key

    series, _ = _series(n_flights=6)
    ids = [s.flight_id for s in series]
    assert len(ids) == len(set(ids))

    same_callsign_other_runway = flight_key(
        {"id": "AAL1", "runway": "23R", "icao24": "a1b2c3",
         "landing_time_utc": "2026-06-18T21:37:36Z"}, 0)
    assert same_callsign_other_runway != flight_key(
        {"id": "AAL1", "runway": "05L", "icao24": "a1b2c3",
         "landing_time_utc": "2026-06-18T21:37:36Z"}, 0)


def test_split_membership_selects_exactly_the_flights_it_names():
    # The consumer of the leak above: predict --split test filters by flight_id, so the
    # filter must return exactly as many series as the split names.
    series, config = _series(n_flights=12)
    train_group, val_group, test_group = split_by_flight(series, config)
    wanted = {s.dataset_id for s in test_group}
    assert len([s for s in series if s.dataset_id in wanted]) == len(test_group)


def test_dataset_identity_qualifies_flight_key_with_airport():
    series, _ = _series(n_flights=1)
    assert series[0].dataset_id == f"{AIRPORT}:{series[0].flight_id}"


def test_cross_validation_folds_are_disjoint_and_cover_outer_train():
    series, config = _series(n_flights=30)
    outer_train, _outer_val, _outer_test = split_by_flight(series, config)
    folds = cross_validation_folds(outer_train, 3, seed=config.seed)
    identities = [item.dataset_id for fold in folds for item in fold]
    assert len(identities) == len(set(identities)) == len(outer_train)
    assert set(identities) == {item.dataset_id for item in outer_train}
