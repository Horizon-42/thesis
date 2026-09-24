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


def test_a_retired_output_field_is_dropped_at_its_old_default_and_named_anywhere_else():
    """The third retirement kind (2026-09-18): the closure, plan and segment-plan views left
    the contract with their outputs, so every state/control checkpoint written before then
    carries fields `TSConfig` no longer declares. Each is dropped at the default the
    ownership rule pinned it to — nothing read it there — and a moved value is NAMED, because
    it belongs to a run of the retired output itself (measured: no stored state or control
    artifact moves one).
    """
    from dataclasses import fields as dataclass_fields

    from ts_transformer.config import (
        PREDICTION_CONTROL,
        RETIRED_CONSTANT_FIELDS,
        RETIRED_FALLBACK_FIELD,
        RETIRED_FIELD_NAMES,
        RETIRED_OUTPUT_FIELDS,
        RETIRED_SERIALIZED_FIELDS,
    )

    live = {field.name for field in dataclass_fields(TSConfig)}
    retired = {name: default
               for names in RETIRED_OUTPUT_FIELDS.values() for name, default in names.items()}
    assert not (retired.keys() & live), "a retired output's field is still declared"
    assert not (retired.keys() & set(RETIRED_SERIALIZED_FIELDS))
    assert not (retired.keys() & set(RETIRED_CONSTANT_FIELDS))
    assert RETIRED_FIELD_NAMES == (
        set(RETIRED_SERIALIZED_FIELDS) | set(RETIRED_CONSTANT_FIELDS) | retired.keys()
        | {RETIRED_FALLBACK_FIELD}
    ), "the flat set the CLI reads must cover every kind (the A320 fallback is the fourth)"

    stored = TSConfig(prediction_output=PREDICTION_CONTROL).to_dict()
    at_defaults = {**stored, **retired}
    assert TSConfig.from_dict(at_defaults).prediction_output == PREDICTION_CONTROL

    moved = {"closure_labels_path": "/labels.json", "plan_rolled_share": 0.5,
             "plan_fan_components": 4, "segment_plan_attention": "segments"}
    assert moved.keys() <= retired.keys()
    for name, value in moved.items():
        with pytest.raises(ValueError, match=f"{name}=") as info:
            TSConfig.from_dict({**at_defaults, name: value})
        assert "retired" in str(info.value) and repr(value) in str(info.value)


def test_a_config_of_a_retired_output_is_refused_by_its_output_not_by_its_fields():
    """A closure/plan/segment-plan run DID move its own fields. Loading one must fail on the
    output — the reason — and not on whichever field `from_dict` happened to reach first."""
    from ts_transformer.config import PREDICTION_OUTPUTS_RETIRED, RETIRED_OUTPUT_FIELDS

    stored = TSConfig().to_dict()
    for output in PREDICTION_OUTPUTS_RETIRED:
        moved = {name: ("moved" if isinstance(default, str) else 7)
                 for name, default in RETIRED_OUTPUT_FIELDS[output].items()}
        with pytest.raises(ValueError, match="prediction_output"):
            TSConfig.from_dict({**stored, **moved, "prediction_output": output})


def test_a_stored_retired_plan_token_is_refused_by_name_with_its_own_archive():
    """21 stored L1 CONTROL checkpoints carry `plan_conditioning='waypoints'` / `'truth-next'`,
    and the 2026-09-19/20 stage B executors carry `'manoeuvre-code'`. Their token builders are
    archived, so they cannot load — but the refusal must say retired and point at THAT value's
    archive, or the reader reads a live vocabulary value as corruption."""
    from pathlib import Path

    from ts_transformer import config as config_module
    from ts_transformer.config import PLAN_CONDITIONINGS, PLAN_CONDITIONINGS_RETIRED

    ts_dir = Path(config_module.__file__).resolve().parent
    assert not set(PLAN_CONDITIONINGS_RETIRED) & set(PLAN_CONDITIONINGS)
    assert set(PLAN_CONDITIONINGS_RETIRED) == {"truth-next", "waypoints", "manoeuvre-code"}
    for value, archive in PLAN_CONDITIONINGS_RETIRED.items():
        assert (ts_dir / archive).is_dir(), f"{value} points at {archive}, which is not there"
        with pytest.raises(ValueError, match="retired") as info:
            TSConfig(prediction_output="control", plan_conditioning=value)
        assert archive in str(info.value)
        # ...and through `from_dict`, which is the door a stored checkpoint comes in by
        with pytest.raises(ValueError, match="retired") as info:
            TSConfig.from_dict({**TSConfig(prediction_output="control").to_dict(), "plan_conditioning": value})
        assert archive in str(info.value)


def test_the_config_grid_tolerance_mirrors_the_row_tolerance():
    """`config._GRID_TOLERANCE_S` is a mirror of `data/time_grids.ROW_TOLERANCE_S` (config is a leaf and
    cannot import it): the horizon, token span and step are judged against the same grid the rows use."""
    from ts_transformer import config
    from ts_transformer.data.time_grids import ROW_TOLERANCE_S
    assert config._GRID_TOLERANCE_S == ROW_TOLERANCE_S



# ── the aircraft filter: drop a flight only where dynamics are used (2026-09-24) ──

def test_the_default_aircraft_filter_is_decided_by_the_output_and_stored_as_a_value():
    from ts_transformer.config import (
        AIRCRAFT_FILTER_ALL_FLIGHTS, AIRCRAFT_FILTER_MODELLED, PREDICTION_CONTROL,
    )
    assert TSConfig().aircraft_filter == AIRCRAFT_FILTER_ALL_FLIGHTS
    assert TSConfig(prediction_output=PREDICTION_CONTROL).aircraft_filter == AIRCRAFT_FILTER_MODELLED
    assert TSConfig().to_dict()["aircraft_filter"] == AIRCRAFT_FILTER_ALL_FLIGHTS


def test_a_control_output_refuses_a_filter_that_keeps_flights_without_dynamics():
    from ts_transformer.config import AIRCRAFT_FILTER_ALL_FLIGHTS, PREDICTION_CONTROL
    with pytest.raises(ValueError, match="inverse-dynamics labels need every flight's aircraft dynamics"):
        TSConfig(prediction_output=PREDICTION_CONTROL, aircraft_filter=AIRCRAFT_FILTER_ALL_FLIGHTS)


def test_a_stored_config_of_the_retired_a320_fallback_is_refused_by_name():
    from ts_transformer.config import (
        AIRCRAFT_FILTER_MODELLED, AIRCRAFT_FILTER_OPENAP_DIRECT, RETIRED_AIRCRAFT_FILTER,
        RETIRED_FALLBACK_FIELD,
    )
    stored = TSConfig().to_dict()
    with pytest.raises(ValueError, match=f"aircraft_filter={RETIRED_AIRCRAFT_FILTER!r} is retired"):
        TSConfig.from_dict({**stored, "aircraft_filter": RETIRED_AIRCRAFT_FILTER,
                            RETIRED_FALLBACK_FIELD: "A320"})
    # under a filter that would have flown it, the fallback is refused...
    with pytest.raises(ValueError, match="the fallback is retired"):
        TSConfig.from_dict({**stored, "aircraft_filter": AIRCRAFT_FILTER_MODELLED,
                            RETIRED_FALLBACK_FIELD: "A320"})
    # ...under openap-direct it was never read, so it is dropped
    direct = {**stored, "aircraft_filter": AIRCRAFT_FILTER_OPENAP_DIRECT, RETIRED_FALLBACK_FIELD: "A320"}
    assert RETIRED_FALLBACK_FIELD not in TSConfig.from_dict(direct).to_dict()


def test_a_stored_config_must_name_its_aircraft_filter():
    # The default moved (2026-09-24); a config without the field would read today's.
    stored = TSConfig().to_dict()
    del stored["aircraft_filter"]
    with pytest.raises(ValueError, match="missing aircraft_filter"):
        TSConfig.from_dict(stored)
