"""An eligibility roster's identity is the SET it selects, never the file's bytes.

2026-09-07: the observed evaluation reports were regenerated (v6 -> v9) over eligible sets
that did not change by one flight. That moved every roster's `sources.evaluation_report_sha256`,
hence its bytes, hence the `roster_sha256` the byte-bound provenance compared — and every
checkpoint trained before the regeneration was refused by `predict`, `evaluate-fit` and every
replay runner, on data that had not changed. The tests here are what makes that impossible:
byte moves are invisible, one changed flight is not, and a checkpoint carrying the retired
byte-bound fingerprint is re-verified rather than refused.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cli.common as cli_common
import cli.train as cli_train
from config import TSConfig
from data_provenance import (
    ARRIVAL_DATA_PROVENANCE_SCHEMA,
    LEGACY_ELIGIBILITY_BOUND_SCHEMA,
    arrival_data_provenance,
    eligible_set_digest,
    require_matching_data_provenance,
)
from dataset import build_series
from flight_scenarios.identity import flight_key
from lateral_eligibility import (
    EVALUATION_REPORT_SCHEMA,
    build_lateral_pass_roster,
    default_evaluation_report_path,
    default_lateral_pass_roster_path,
)
from splits import data_selection_audit, flight_keys_by_split
from synthetic import synthetic_arrivals
from train import train

_CLI_SPEC = importlib.util.spec_from_file_location(
    "ts_transformer_cli_eligible_set_test",
    Path(__file__).resolve().parents[1] / "__main__.py",
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
ts_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(ts_cli)

AIRPORT, RUNWAY = "KRDU", "05L"


def _harvest(
    tmp_path: Path, flights: list[dict], *, ineligible: int = 2
) -> tuple[Path, Path]:
    """A real arrivals manifest plus the lateral-pass roster the evaluator derives from it.

    The last ``ineligible`` flights fail the lateral gate, so the roster genuinely filters
    and a swapped key can keep the counts unchanged.
    """
    arrivals = tmp_path / "arrivals"
    approach = tmp_path / "approach"
    arrivals.mkdir(parents=True, exist_ok=True)
    approach.mkdir(parents=True, exist_ok=True)
    keys = [flight_key(flight, index) for index, flight in enumerate(flights)]
    manifest = arrivals / "manifest.json"
    manifest.write_text(
        json.dumps({
            "airport": AIRPORT,
            "records": [
                {
                    "flight_key": key,
                    "source_sha256": hashlib.sha256(key.encode()).hexdigest(),
                }
                for key in keys
            ],
        }),
        encoding="utf-8",
    )
    report = default_evaluation_report_path(manifest)
    report.write_text(
        json.dumps({
            "schema_version": EVALUATION_REPORT_SCHEMA,
            "subject": "observed",
            "trajectories": [
                {
                    "airport": AIRPORT,
                    "flight_key": key,
                    "lateral_result": "pass" if index < len(keys) - ineligible else "fail",
                }
                for index, key in enumerate(keys)
            ],
        }),
        encoding="utf-8",
    )
    roster = default_lateral_pass_roster_path(manifest)
    build_lateral_pass_roster(manifest, report, roster)
    return manifest, roster


def _reserialise(roster: Path) -> None:
    """Rewrite the roster the way a regenerated observed evaluation does.

    Same eligible set; a new upstream report digest, the keys in another order, and a
    different indentation — i.e. every byte-level property the retired identity compared.
    """
    document = json.loads(roster.read_text(encoding="utf-8"))
    document["sources"]["evaluation_report_sha256"] = "f" * 64
    document["sources"]["evaluation_report_schema"] = "evaluation-report-v9-regenerated"
    document["eligible_flight_keys"] = list(reversed(document["eligible_flight_keys"]))
    roster.write_text(json.dumps(document, indent=4) + "\n\n", encoding="utf-8")


def _swap_one_eligible_key(roster: Path, manifest: Path) -> None:
    """Change WHICH flights are eligible without changing HOW MANY."""
    document = json.loads(roster.read_text(encoding="utf-8"))
    candidates = [
        row["flight_key"]
        for row in json.loads(manifest.read_text(encoding="utf-8"))["records"]
    ]
    replacement = next(
        key for key in candidates if key not in set(document["eligible_flight_keys"])
    )
    document["eligible_flight_keys"] = sorted(
        document["eligible_flight_keys"][1:] + [replacement]
    )
    roster.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _v3_payload(provenance: dict, config: TSConfig) -> dict:
    """A checkpoint payload carrying the RETIRED byte-bound fingerprint.

    The eligibility block below is a frozen literal of the v3 shape — a mirror of the
    schema this build no longer writes, which is the only way a test can still produce
    one. Everything else (the split identities the re-verification recomputes) comes from
    the real `splits` audit, so this fixture cannot drift from what training wrote.
    """
    manifests = []
    for entry in provenance["manifests"]:
        eligibility = entry["eligibility"]
        manifests.append({
            **entry,
            "eligibility": {
                "schema_version": eligibility["schema_version"],
                "policy": eligibility["policy"],
                "roster_sha256": "9" * 64,          # the roster file as it was that day
                "evaluation_report_sha256": "8" * 64,
                "counts": eligibility["counts"],
            },
        })
    outer = flight_keys_by_split(provenance, config)
    audit = data_selection_audit(
        [
            SimpleNamespace(dataset_id=key)
            for key in outer["train"] + outer["val"]
        ],
        SimpleNamespace(to_dict=lambda: {}),
        config,
        outer,
    )
    return {
        "config": config.to_dict(),
        "data_provenance": {
            "schema_version": LEGACY_ELIGIBILITY_BOUND_SCHEMA,
            "manifests": manifests,
        },
        "data_selection": audit,
    }


def test_a_reserialised_roster_gives_the_identical_provenance(tmp_path: Path) -> None:
    """The failure of 2026-09-07, in one assertion: new bytes, same eligible set."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=6, seed=3)
    manifest, roster = _harvest(tmp_path, flights)
    before_bytes = roster.read_bytes()

    before = arrival_data_provenance(manifest, eligibility_rosters=[roster])
    _reserialise(roster)
    after = arrival_data_provenance(manifest, eligibility_rosters=[roster])

    assert roster.read_bytes() != before_bytes
    assert after == before
    assert after["manifests"][0]["eligibility"]["eligible_set_sha256"] == (
        eligible_set_digest(
            row["flight_key"] for row in after["manifests"][0]["source_records"]
        )
    )


def test_one_swapped_eligible_key_is_refused_although_the_counts_are_unchanged(
    tmp_path: Path,
) -> None:
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=6, seed=3)
    manifest, roster = _harvest(tmp_path, flights)
    stored = arrival_data_provenance(manifest, eligibility_rosters=[roster])

    _swap_one_eligible_key(roster, manifest)
    current = arrival_data_provenance(manifest, eligibility_rosters=[roster])

    assert current["manifests"][0]["eligibility"]["counts"] == (
        stored["manifests"][0]["eligibility"]["counts"]
    )
    assert current["manifests"][0]["eligibility"]["eligible_set_sha256"] != (
        stored["manifests"][0]["eligibility"]["eligible_set_sha256"]
    )
    with pytest.raises(ValueError, match="does not match the current arrival manifests"):
        require_matching_data_provenance({"data_provenance": stored}, current)
    with pytest.raises(ValueError, match="not an exact airport subset"):
        require_matching_data_provenance(
            {"data_provenance": stored}, current, allow_subset=True
        )


@pytest.mark.parametrize("allow_subset", [False, True])
def test_a_legacy_checkpoint_is_accepted_against_a_reserialised_roster(
    tmp_path: Path, allow_subset: bool
) -> None:
    """A v3 fingerprint is re-verified by its own split identities, not by roster bytes."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=6, seed=3)
    manifest, roster = _harvest(tmp_path, flights)
    config = TSConfig()
    payload = _v3_payload(
        arrival_data_provenance(manifest, eligibility_rosters=[roster]), config
    )

    _reserialise(roster)
    current = arrival_data_provenance(manifest, eligibility_rosters=[roster])

    require_matching_data_provenance(payload, current, allow_subset=allow_subset)


def test_a_legacy_checkpoint_is_refused_when_the_eligible_set_moved(
    tmp_path: Path,
) -> None:
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=6, seed=3)
    manifest, roster = _harvest(tmp_path, flights)
    config = TSConfig()
    payload = _v3_payload(
        arrival_data_provenance(manifest, eligibility_rosters=[roster]), config
    )

    _swap_one_eligible_key(roster, manifest)
    current = arrival_data_provenance(manifest, eligibility_rosters=[roster])

    with pytest.raises(ValueError, match=f"eligible flight set changed.*{AIRPORT}"):
        require_matching_data_provenance(payload, current, allow_subset=True)


def test_the_predict_cli_runs_a_legacy_checkpoint_against_a_reserialised_roster(
    tmp_path: Path, monkeypatch
) -> None:
    """The blocked command itself: `predict` on a v3 checkpoint whose roster was rewritten.

    Trained here, then rewritten into the retired fingerprint — the shape every checkpoint
    on disk before 2026-09-08 carries — and predicted through the real CLI with the real
    manifest and roster on disk, so the provenance is built the way the command builds it.
    """
    import cli.predict as predict_module

    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    manifest, roster = _harvest(tmp_path, flights, ineligible=2)
    config = TSConfig(
        seq_len=20, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        epochs=1, patience=1, batch_size=16, device="cpu", horizon_mode="normalized",
    )
    provenance = arrival_data_provenance(manifest, eligibility_rosters=[roster])
    outer = flight_keys_by_split(provenance, config)
    eligible = set(outer["train"] + outer["val"] + outer["test"])
    assert len(eligible) == len(flights) - 2       # the roster really filtered
    development = set(outer["train"] + outer["val"])
    series, _report = build_series(
        [
            flight for index, flight in enumerate(flights)
            if f"{AIRPORT}:{flight_key(flight, index)}" in development
        ],
        config,
        airport=AIRPORT,
    )
    run = tmp_path / "run"
    train(
        series,
        config,
        output_dir=run,
        data_provenance=provenance,
        reserved_test_keys=outer["test"],
        data_selection=_v3_payload(provenance, config)["data_selection"],
        verbose=False,
    )
    checkpoint = run / "checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["data_provenance"] = _v3_payload(provenance, config)["data_provenance"]
    torch.save(payload, checkpoint)

    _reserialise(roster)
    monkeypatch.setattr(
        predict_module,
        "load_flight_dicts",
        lambda _data, include_flight_keys=None: flights,
    )

    out = tmp_path / "prediction"
    assert ts_cli.main([
        "predict", "--checkpoint", str(checkpoint),
        "--data", str(manifest), "--eligibility-roster", str(roster),
        "--airport", AIRPORT, "--output-dir", str(out),
        "--split", "val", "--device", "cpu",
    ]) == 0
    assert json.loads((out / "summary.json").read_text())["results"]


def _publication_plan(
    tmp_path: Path, *, metadata: dict, checkpoint_payload: dict | None = None
):
    """One publisher job against a real harvest, its checkpoint written by torch."""
    import publish_ts_experiment_trajectories as publisher

    run = tmp_path / "experiments" / "campaign" / "run"
    run.mkdir(parents=True)
    checkpoint = run / "checkpoint.pt"
    if checkpoint_payload is None:
        checkpoint.write_bytes(b"checkpoint")
    else:
        torch.save(checkpoint_payload, checkpoint)
    (run / "checkpoint_metadata.json").write_text(
        json.dumps({
            **metadata,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    index = tmp_path / "experiments" / "index.json"
    index.write_text(
        json.dumps({
            "root": str(tmp_path / "experiments"),
            "entries": [{
                "path": "campaign/run",
                "campaign_id": "campaign",
                "run_id": "run",
                "kind": "training",
                "status": "completed",
                "artifacts": ["checkpoint.pt"],
            }],
        }),
        encoding="utf-8",
    )
    return publisher.PublicationPlan(
        publisher.discover_checkpoints(index)[0],
        AIRPORT,
        "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
    )


def test_the_publisher_preflight_compares_the_eligible_set_not_the_roster_bytes(
    tmp_path: Path,
) -> None:
    """The publisher's own byte comparison blocked every publication on 2026-09-07."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=6, seed=3)
    manifest, roster = _harvest(tmp_path / "harvest" / AIRPORT, flights)
    provenance = arrival_data_provenance(manifest, eligibility_rosters=[roster])
    plan = _publication_plan(tmp_path, metadata={
        "arrival_manifests": {
            AIRPORT: hashlib.sha256(manifest.read_bytes()).hexdigest()
        },
        "eligible_sets": {
            AIRPORT: provenance["manifests"][0]["eligibility"]["eligible_set_sha256"]
        },
    })

    _reserialise(roster)
    assert plan.preflight_error() is None
    assert "--eligibility-roster" in dict(plan.commands())["predict"]

    _swap_one_eligible_key(roster, manifest)
    assert "eligible set changed for KRDU" in (plan.preflight_error() or "")


def test_the_publisher_preflight_reads_a_legacy_checkpoint_through_the_package(
    tmp_path: Path,
) -> None:
    """Metadata that predates `eligible_sets` is answered by the checkpoint, not by bytes."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=6, seed=3)
    manifest, roster = _harvest(tmp_path / "harvest" / AIRPORT, flights)
    provenance = arrival_data_provenance(manifest, eligibility_rosters=[roster])
    plan = _publication_plan(
        tmp_path,
        metadata={
            "arrival_manifests": {
                AIRPORT: hashlib.sha256(manifest.read_bytes()).hexdigest()
            },
            "eligibility_rosters": {AIRPORT: "9" * 64},
        },
        checkpoint_payload=_v3_payload(provenance, TSConfig()),
    )

    _reserialise(roster)
    assert plan.preflight_error() is None

    _swap_one_eligible_key(roster, manifest)
    assert "eligible flight set changed" in (plan.preflight_error() or "")


def test_the_training_audit_still_records_the_roster_byte_facts(
    tmp_path: Path, monkeypatch
) -> None:
    """The bytes stay AUDITABLE: `pre_split_eligibility` names the report they came from."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=6, seed=3)
    manifest, roster = _harvest(tmp_path, flights)
    captured: dict = {}

    monkeypatch.setattr(
        cli_common, "load_flight_dicts", lambda _data, include_flight_keys: [{}]
    )
    monkeypatch.setattr(
        cli_common,
        "build_series_or_exit",
        lambda args, config, parser, _flights: (
            [
                SimpleNamespace(dataset_id=key)
                for key in flight_keys_by_split(
                    arrival_data_provenance(manifest, eligibility_rosters=[roster]),
                    config,
                )["train"]
            ],
            SimpleNamespace(to_dict=lambda: {}),
        ),
    )
    monkeypatch.setattr(
        cli_train, "run_training", lambda _series, _config, **kwargs: captured.update(kwargs)
    )

    assert ts_cli.main([
        "train",
        "--data", str(manifest), "--eligibility-roster", str(roster),
        "--output-dir", str(tmp_path / "run"),
    ]) == 0

    entry = captured["data_selection"]["pre_split_eligibility"][0]
    sources = entry["roster_sources"]
    assert entry["eligible_set_sha256"] == eligible_set_digest(
        json.loads(roster.read_text(encoding="utf-8"))["eligible_flight_keys"]
    )
    assert sources["roster_sha256"] == hashlib.sha256(roster.read_bytes()).hexdigest()
    assert sources["evaluation_report_sha256"] == hashlib.sha256(
        default_evaluation_report_path(manifest).read_bytes()
    ).hexdigest()
    # The provenance the checkpoint is compared against carries neither.
    assert "roster_sha256" not in entry and "evaluation_report_sha256" not in entry
