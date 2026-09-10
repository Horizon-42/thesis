"""The experiment index.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import json

import ts_transformer.experiment_index as experiment_index


def test_experiment_index_keeps_legacy_incomplete_and_formal_runs_distinct(tmp_path):
    legacy_complete = tmp_path / "legacy" / "run_complete"
    legacy_complete.mkdir(parents=True)
    (legacy_complete / "checkpoint.pt").write_bytes(b"checkpoint")
    (legacy_complete / "history.json").write_text(
        json.dumps({"config": {"prediction_output": "state", "seed": 1337}}),
        encoding="utf-8",
    )
    legacy_incomplete = tmp_path / "legacy" / "run_aborted"
    legacy_incomplete.mkdir(parents=True)
    (legacy_incomplete / "experiment_notes.md").write_text("aborted", encoding="utf-8")
    formal = tmp_path / "openap_direct" / "control"
    formal.mkdir(parents=True)
    (formal / experiment_index.RUN_MANIFEST_NAME).write_text(
        json.dumps({
            "schema_version": experiment_index.RUN_MANIFEST_SCHEMA,
            "campaign_id": "openap-direct-20260729",
            "run_id": "control",
            "status": "completed",
            "config": {
                "prediction_output": "control",
                "aircraft_filter": "openap-direct",
                "seed": 1337,
            },
            "artifacts": {"checkpoint.pt": {}},
        }),
        encoding="utf-8",
    )

    document = experiment_index.rebuild_index(tmp_path)
    by_run = {entry["run_id"]: entry for entry in document["entries"]}

    assert by_run["run_complete"]["status"] == "completed"
    assert by_run["run_aborted"]["status"] == "incomplete"
    assert by_run["control"]["campaign_id"] == "openap-direct-20260729"
    assert by_run["control"]["aircraft_filter"] == "openap-direct"
    assert (tmp_path / "index.json").is_file()
    assert (tmp_path / "INDEX.md").is_file()
