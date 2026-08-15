"""Build the evaluation-owned lateral-pass roster consumed by TS data loading.

This module is the only seam that understands ``evaluation.lateral_result``.  It emits
stable flight identities and provenance; model, loss, and trajectory-loading code never
import evaluation policy.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


LATERAL_PASS_ROSTER_SCHEMA = "ts-lateral-pass-eligibility-v1"
EVALUATION_REPORT_SCHEMA = "terminal-approach-evaluation-v4"
LATERAL_PASS_POLICY = "evaluation.lateral_result == pass"
LATERAL_RESULTS = {"pass", "fail", "indeterminate"}
ROSTER_NAME = "lateral_pass_eligibility.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def default_lateral_pass_roster_path(arrival_manifest: str | Path) -> Path:
    return Path(arrival_manifest).resolve().parent / ROSTER_NAME


def default_evaluation_report_path(arrival_manifest: str | Path) -> Path:
    manifest = Path(arrival_manifest).resolve()
    return manifest.parent.parent / "approach" / "evaluation_report.json"


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload, value


def build_lateral_pass_roster(
    arrival_manifest: str | Path,
    evaluation_report: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Join one arrival roster to observed evaluation by exact ``flight_key``.

    Every model candidate must have exactly one evaluation row. Evaluation-only rows are
    harmless because some evaluated tracks are not model-ready arrivals, but they are
    counted explicitly. Vertical and overall verdict fields are intentionally ignored.
    """
    manifest_path = Path(arrival_manifest).resolve()
    report_path = Path(evaluation_report).resolve()
    manifest_bytes, manifest = _read_object(manifest_path)
    report_bytes, report = _read_object(report_path)

    airport = str(manifest.get("airport") or "").strip().upper()
    if not airport:
        raise ValueError(f"{manifest_path} does not declare an airport")
    if report.get("schema_version") != EVALUATION_REPORT_SCHEMA:
        raise ValueError(f"{report_path} has the wrong evaluation schema")
    if report.get("subject") != "observed":
        raise ValueError(f"{report_path} is not an observed evaluation report")

    manifest_rows = manifest.get("records")
    evaluation_rows = report.get("trajectories")
    if not isinstance(manifest_rows, list):
        raise ValueError(f"{manifest_path} lacks an arrival records roster")
    if not isinstance(evaluation_rows, list):
        raise ValueError(f"{report_path} lacks trajectory results")

    candidate_keys: list[str] = []
    candidate_seen: set[str] = set()
    for index, row in enumerate(manifest_rows):
        key = row.get("flight_key") if isinstance(row, dict) else None
        if not isinstance(key, str) or not key:
            raise ValueError(f"{manifest_path}: arrival record {index} lacks flight_key")
        if key in candidate_seen:
            raise ValueError(f"{manifest_path} repeats flight_key {key!r}")
        candidate_seen.add(key)
        candidate_keys.append(key)

    by_key: dict[str, str] = {}
    for index, row in enumerate(evaluation_rows):
        if not isinstance(row, dict):
            raise ValueError(f"{report_path}: trajectory {index} is not an object")
        key = row.get("flight_key")
        result = row.get("lateral_result")
        row_airport = str(row.get("airport") or "").strip().upper()
        if not isinstance(key, str) or not key:
            raise ValueError(f"{report_path}: trajectory {index} lacks flight_key")
        if key in by_key:
            raise ValueError(f"{report_path} repeats flight_key {key!r}")
        if row_airport != airport:
            raise ValueError(
                f"{report_path}: trajectory {key!r} belongs to {row_airport or '<missing>'}, "
                f"not {airport}"
            )
        if result not in LATERAL_RESULTS:
            raise ValueError(
                f"{report_path}: trajectory {key!r} has invalid lateral_result {result!r}"
            )
        by_key[key] = result

    missing = sorted(candidate_seen - by_key.keys())
    if missing:
        raise ValueError(
            f"{manifest_path} has {len(missing)} arrival(s) missing evaluation result; "
            f"first: {missing[0]!r}"
        )

    outcomes = Counter(by_key[key] for key in candidate_keys)
    eligible = sorted(key for key in candidate_keys if by_key[key] == "pass")
    document = {
        "schema_version": LATERAL_PASS_ROSTER_SCHEMA,
        "airport": airport,
        "policy": LATERAL_PASS_POLICY,
        "sources": {
            "arrival_manifest": str(manifest_path),
            "arrival_manifest_sha256": _sha256(manifest_bytes),
            "evaluation_report": str(report_path),
            "evaluation_report_sha256": _sha256(report_bytes),
            "evaluation_report_schema": report["schema_version"],
        },
        "counts": {
            "arrival_candidates": len(candidate_keys),
            "eligible_lateral_pass": outcomes["pass"],
            "excluded_lateral_fail": outcomes["fail"],
            "excluded_lateral_indeterminate": outcomes["indeterminate"],
            "evaluation_only": len(by_key.keys() - candidate_seen),
        },
        "eligible_flight_keys": eligible,
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return document


def ensure_lateral_pass_roster(arrival_manifest: str | Path) -> Path:
    """Refresh a derived roster only when either bound source changed."""
    manifest = Path(arrival_manifest).resolve()
    report = default_evaluation_report_path(manifest)
    output = default_lateral_pass_roster_path(manifest)
    if output.is_file():
        try:
            existing_bytes, existing = _read_object(output)
            del existing_bytes
            sources = existing.get("sources")
            if (
                existing.get("schema_version") == LATERAL_PASS_ROSTER_SCHEMA
                and isinstance(sources, dict)
                and sources.get("arrival_manifest_sha256") == _sha256(manifest.read_bytes())
                and sources.get("evaluation_report_sha256") == _sha256(report.read_bytes())
            ):
                return output
        except ValueError:
            pass
    build_lateral_pass_roster(manifest, report, output)
    return output

