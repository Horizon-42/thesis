"""The config contract's ranges and defaults, and the arrival-data provenance.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import hashlib
import json

import pytest

from ts_transformer.config import TSConfig
from ts_transformer.data.data_provenance import (
    ARRIVAL_DATA_PROVENANCE_SCHEMA,
    arrival_data_provenance,
    require_matching_data_provenance,
)
# Imported, never restated: a schema version pinned by hand in a fixture is a version
# the fixture cannot check, and this one gates every loader that reads the roster.


def test_config_rejects_a_head_count_that_does_not_divide_d_model():
    with pytest.raises(ValueError, match="n_heads"):
        TSConfig(d_model=100, n_heads=8)


def test_default_config_uses_selected_normalized_output_and_physics_losses():
    config = TSConfig()
    assert config.n_segments == 16
    assert TSConfig(model="patchtst").n_segments == 256
    assert config.epochs == 180
    assert config.patience == 20
    assert config.lr_plateau_patience == 3
    assert config.lr_plateau_factor == 0.5
    assert config.state_endpoint_loss_weight == 0.25
    assert config.kinematic_consistency_loss_weight == 3.0
    assert config.terminal_loss_weight == 0.02


@pytest.mark.parametrize(
    "field",
    [
        "state_endpoint_loss_weight",
        "kinematic_consistency_loss_weight",
        "terminal_loss_weight",
    ],
)
def test_negative_physics_loss_weights_are_rejected(field):
    with pytest.raises(ValueError, match=field):
        TSConfig(**{field: -0.1})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seq_len", 0),
        ("n_segments", 0),
        ("dt_s", 0.0),
        ("batch_size", 0),
        ("epochs", 0),
        ("learning_rate", 0.0),
        ("final_time_scale_s", 0.0),
        ("patience", 0),
        ("d_model", 0),
        ("n_heads", 0),
        ("e_layers", 0),
    ],
)
def test_non_positive_training_parameters_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        TSConfig(**{field: value})


def test_vendor_pred_len_contract_tracks_n_segments():
    assert TSConfig(n_segments=99).pred_len == 99


def test_arrival_data_provenance_binds_manifest_and_per_flight_sources(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "airport": "KRDU",
        "records": [
            {"flight_key": "B", "source_sha256": "b" * 64},
            {"flight_key": "A", "source_sha256": "a" * 64},
        ],
    }), encoding="utf-8")

    provenance = arrival_data_provenance(manifest)
    entry = provenance["manifests"][0]
    assert provenance["schema_version"] == ARRIVAL_DATA_PROVENANCE_SCHEMA
    assert entry["airport"] == "KRDU"
    assert entry["arrival_manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert entry["source_records"] == [
        {"flight_key": "A", "source_sha256": "a" * 64},
        {"flight_key": "B", "source_sha256": "b" * 64},
    ]
    require_matching_data_provenance({"data_provenance": provenance}, provenance)
    with pytest.raises(ValueError, match="does not match"):
        changed = json.loads(json.dumps(provenance))
        changed["manifests"][0]["arrival_manifest_sha256"] = "c" * 64
        require_matching_data_provenance(
            {"data_provenance": provenance},
            changed,
        )


def test_prediction_provenance_accepts_only_exact_training_airport_subset():
    stored = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [
            {"airport": "KAAA", "arrival_manifest_sha256": "a" * 64, "source_records": []},
            {"airport": "KBBB", "arrival_manifest_sha256": "b" * 64, "source_records": []},
        ],
    }
    subset = {**stored, "manifests": [stored["manifests"][1]]}
    require_matching_data_provenance(
        {"data_provenance": stored}, subset, allow_subset=True
    )
    changed = json.loads(json.dumps(subset))
    changed["manifests"][0]["arrival_manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="subset"):
        require_matching_data_provenance(
            {"data_provenance": stored}, changed, allow_subset=True
        )
