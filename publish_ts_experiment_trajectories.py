#!/usr/bin/env python
"""Publish train/validation trajectories for indexed TS checkpoints.

The script is intentionally an orchestration layer.  It does not define another trajectory
or evaluation format; every job calls the existing predictor, the shared ``evaluation`` package,
and the existing comparison-CZML publisher in that order.  Publications are resumable and kept
under a checkpoint-specific raw-output directory. ``--result-source prediction`` publishes a
primary result under Prediction; the default ``experiment`` source includes checkpoint metadata
for the Experiments picker.

Outer-test is not a valid option here.  This command is for development train/validation
inspection only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
_TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(_TS_DIR) not in sys.path:
    sys.path.insert(0, str(_TS_DIR))

from run_naming import category_display_label, run_display_name  # noqa: E402

EXPERIMENT_ROOT = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments"
EXPERIMENT_INDEX = EXPERIMENT_ROOT / "index.json"
RAW_OUTPUT_ROOT = (
    REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiment_predictions"
)
HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
FRONTEND_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
TS_SCRIPT = REPO_ROOT / "4dTrajectory" / "ts_transformer" / "__main__.py"
CZML_SCRIPT = REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"

PUBLICATION_SCHEMA = "ts-experiment-publication-v1"
PUBLICATION_INDEX_SCHEMA = "ts-experiment-publication-index-v1"
PUBLICATION_MANIFEST = "publication.json"
DEVELOPMENT_SPLITS = ("train", "val")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@lru_cache(maxsize=None)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(value: str, *, limit: int = 56) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return (normalized or "experiment")[:limit]


def _path_for_manifest(path: Path) -> str:
    """Use a repository-relative path when possible, otherwise an absolute path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class ExperimentCheckpoint:
    experiment_id: str
    campaign: str
    run_id: str
    checkpoint: Path
    checkpoint_sha256: str
    arrival_manifests: dict[str, str]
    eligibility_rosters: dict[str, str]
    config: dict[str, Any]

    @property
    def token(self) -> str:
        return hashlib.sha256(self.experiment_id.encode("utf-8")).hexdigest()[:12]

    @property
    def directory_name(self) -> str:
        return f"{_safe_stem(self.run_id)}_{self.token}"

    @property
    def checkpoint_relative(self) -> str:
        return _path_for_manifest(self.checkpoint)


def _checkpoint_config(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = _load_object(manifest_path)
        config = manifest.get("config")
        if isinstance(config, dict):
            return config
    history_path = directory / "history.json"
    if history_path.is_file():
        history = _load_object(history_path)
        config = history.get("config")
        if isinstance(config, dict):
            return config
    return {}


def discover_checkpoints(
    index_path: Path = EXPERIMENT_INDEX,
    *,
    selected_ids: set[str] | None = None,
    campaigns: set[str] | None = None,
) -> list[ExperimentCheckpoint]:
    """Return completed, indexed training checkpoints in stable path order."""
    document = _load_object(index_path)
    root = Path(document.get("root") or index_path.parent).resolve()
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{index_path} has no entries list")

    checkpoints: list[ExperimentCheckpoint] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "training" or entry.get("status") != "completed":
            continue
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, list) or "checkpoint.pt" not in artifacts:
            continue
        experiment_id = str(entry.get("path") or "")
        if not experiment_id:
            continue
        if selected_ids is not None and experiment_id not in selected_ids:
            continue
        campaign = str(entry.get("campaign_id") or Path(experiment_id).parts[0])
        if campaigns is not None and campaign not in campaigns:
            continue

        directory = root / experiment_id
        checkpoint = directory / "checkpoint.pt"
        metadata_path = directory / "checkpoint_metadata.json"
        if not checkpoint.is_file() or not metadata_path.is_file():
            continue
        metadata = _load_object(metadata_path)
        manifests = metadata.get("arrival_manifests")
        if not isinstance(manifests, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in manifests.items()
        ):
            raise ValueError(f"{metadata_path} has no valid arrival_manifests map")
        rosters = metadata.get("eligibility_rosters")
        if rosters is None:
            rosters = {}
        if not isinstance(rosters, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in rosters.items()
        ):
            raise ValueError(f"{metadata_path} has no valid eligibility_rosters map")
        declared_sha = metadata.get("checkpoint_sha256")
        checkpoint_sha = declared_sha if isinstance(declared_sha, str) else _sha256(checkpoint)
        checkpoints.append(ExperimentCheckpoint(
            experiment_id=experiment_id,
            campaign=campaign,
            run_id=str(entry.get("run_id") or directory.name),
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            arrival_manifests=dict(sorted(manifests.items())),
            eligibility_rosters=dict(sorted(rosters.items())),
            config=_checkpoint_config(directory),
        ))
    return checkpoints


@dataclass(frozen=True)
class PublicationPlan:
    experiment: ExperimentCheckpoint
    airport: str
    split: str
    result_source: str = "experiment"
    raw_output_root: Path = RAW_OUTPUT_ROOT
    harvest_root: Path = HARVEST_ROOT
    frontend_airports_root: Path = FRONTEND_AIRPORTS_ROOT
    device: str = "auto"
    record_retention: str = "archive"

    def __post_init__(self) -> None:
        if self.split not in DEVELOPMENT_SPLITS:
            raise ValueError(
                f"experiment publication accepts development splits {DEVELOPMENT_SPLITS}, "
                f"got {self.split!r}"
            )
        if self.result_source not in {"prediction", "experiment"}:
            raise ValueError(f"unknown result source {self.result_source!r}")
        if self.record_retention not in {"loose", "archive"}:
            raise ValueError(f"unknown record retention {self.record_retention!r}")

    @property
    def data_manifest(self) -> Path:
        return self.harvest_root / self.airport / "arrivals" / "manifest.json"

    @property
    def eligibility_roster(self) -> Path:
        return self.data_manifest.parent / "lateral_pass_eligibility.json"

    @property
    def output_dir(self) -> Path:
        return (
            self.raw_output_root
            / self.experiment.directory_name
            / self.result_source
            / self.airport
            / self.split
        )

    @property
    def summary(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def evaluation_report(self) -> Path:
        return self.output_dir / "evaluation_report.json"

    @property
    def publication_manifest(self) -> Path:
        return self.output_dir / PUBLICATION_MANIFEST

    @property
    def records_archive(self) -> Path:
        return self.output_dir / "prediction_records.tar.gz"

    @property
    def category(self) -> str:
        run = _safe_stem(self.experiment.run_id, limit=42).lower()
        return f"{self.result_source}_{run}_{self.experiment.token}_{self.split}"

    @property
    def comparison_dir(self) -> Path:
        return self.frontend_airports_root / self.airport / "comparison" / self.category

    @property
    def comparison_index(self) -> Path:
        return self.comparison_dir / "comparison_index.json"

    @property
    def category_label(self) -> str:
        return _publication_label(
            self.split, self.result_source, self.experiment.config, self.experiment.run_id
        )

    @property
    def experiment_metadata(self) -> dict[str, Any]:
        return _publication_experiment_metadata(
            experiment_id=self.experiment.experiment_id,
            campaign=self.experiment.campaign,
            checkpoint=self.experiment.checkpoint_relative,
            config=self.experiment.config,
            run_id=self.experiment.run_id,
        )

    def commands(self) -> list[tuple[str, list[str]]]:
        py = sys.executable
        predict = [
            py, str(TS_SCRIPT), "predict",
            "--checkpoint", str(self.experiment.checkpoint),
            "--data", str(self.data_manifest),
        ]
        if self.airport in self.experiment.eligibility_rosters:
            predict += ["--eligibility-roster", str(self.eligibility_roster)]
        predict += [
            "--output-dir", str(self.output_dir),
            "--split", self.split,
            "--device", self.device,
        ]
        publish = [
            py, str(CZML_SCRIPT),
            "--summary", str(self.summary),
            "--output-dir", str(self.comparison_dir),
            "--airport", self.airport,
            "--category", self.category,
            "--category-label", self.category_label,
            "--dataset-split", self.split,
            "--evaluation-report", str(self.evaluation_report),
            "--result-source", self.result_source,
        ]
        if self.result_source == "experiment":
            publish += [
                "--experiment-id", self.experiment.experiment_id,
                "--experiment-group", self.experiment.campaign,
                "--experiment-checkpoint", self.experiment.checkpoint_relative,
            ]
        return [
            ("predict", predict),
            ("evaluate", [
                py, "-m", "evaluation",
                "--input", str(self.output_dir),
                "--output", str(self.evaluation_report),
            ]),
            ("publish-czml", publish),
        ]

    def preflight_error(self) -> str | None:
        if not self.experiment.checkpoint.is_file():
            return f"missing checkpoint {self.experiment.checkpoint}"
        if _sha256(self.experiment.checkpoint) != self.experiment.checkpoint_sha256:
            return "checkpoint SHA-256 differs from checkpoint_metadata.json"
        if not self.data_manifest.is_file():
            return f"missing arrival manifest {self.data_manifest}"
        expected = self.experiment.arrival_manifests.get(self.airport)
        if expected is None:
            return f"checkpoint provenance does not include airport {self.airport}"
        actual = _sha256(self.data_manifest)
        if actual != expected:
            return (
                f"arrival manifest SHA-256 mismatch for {self.airport}: "
                f"checkpoint={expected}, current={actual}"
            )
        expected_roster = self.experiment.eligibility_rosters.get(self.airport)
        if expected_roster is not None:
            if not self.eligibility_roster.is_file():
                return f"missing eligibility roster {self.eligibility_roster}"
            actual_roster = _sha256(self.eligibility_roster)
            if actual_roster != expected_roster:
                return (
                    f"eligibility roster SHA-256 mismatch for {self.airport}: "
                    f"checkpoint={expected_roster}, current={actual_roster}"
                )
        return None

    def is_complete(self) -> bool:
        if not self.publication_manifest.is_file():
            return False
        try:
            manifest = _load_object(self.publication_manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            manifest.get("status") == "completed" and
            manifest.get("checkpointSha256") == self.experiment.checkpoint_sha256 and
            manifest.get("resultSource") == self.result_source and
            self.summary.is_file() and
            self.evaluation_report.is_file() and
            self.comparison_index.is_file()
        )


def _publication_label(
    split: str, result_source: str, config: dict[str, Any], run_id: str
) -> str:
    kind = "Experiment" if result_source == "experiment" else "Predicted"
    return category_display_label(
        split, run_display_name(config, extra=(run_id,)), kind=kind
    )


def _publication_experiment_metadata(
    *,
    experiment_id: str,
    campaign: str | None,
    checkpoint: str,
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    return {
        "id": experiment_id,
        "group": campaign,
        "checkpoint": checkpoint,
        "label": run_display_name(config, extra=(run_id,)),
        "model": config.get("model"),
        "predictionOutput": config.get("prediction_output", "state"),
        "horizonMode": config.get("horizon_mode", "normalized"),
        "seed": config.get("seed"),
    }


def _apply_category_refresh(
    manifest_path: Path,
    category_key: str,
    label: str,
    result_source: str,
    experiment_metadata: dict[str, Any] | None,
) -> bool:
    """Patch one category's derived label/metadata in a categories.json; True if found."""
    if not manifest_path.is_file():
        return False
    document = _load_object(manifest_path)
    categories = document.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"{manifest_path} must contain a categories array")
    changed = False
    found = False
    updated_categories: list[Any] = []
    for value in categories:
        if not isinstance(value, dict) or value.get("key") != category_key:
            updated_categories.append(value)
            continue
        found = True
        updated = dict(value)
        if updated.get("label") != label:
            updated["label"] = label
            changed = True
        if updated.get("resultSource") != result_source:
            updated["resultSource"] = result_source
            changed = True
        if experiment_metadata is not None:
            if updated.get("experiment") != experiment_metadata:
                updated["experiment"] = experiment_metadata
                changed = True
        elif "experiment" in updated:
            del updated["experiment"]
            changed = True
        updated_categories.append(updated)
    if found and changed:
        document["categories"] = updated_categories
        _write_json_atomic(manifest_path, document)
    return found


def refresh_category_metadata(plan: PublicationPlan) -> bool:
    """Refresh derived labels/metadata without regenerating archived trajectories."""
    return _apply_category_refresh(
        plan.comparison_dir.parent / "categories.json",
        plan.category,
        plan.category_label,
        plan.result_source,
        plan.experiment_metadata if plan.result_source == "experiment" else None,
    )


def refresh_labels_from_manifests(
    output_root: Path, frontend_airports_root: Path
) -> tuple[int, int]:
    """Recompute labels/metadata for every completed publication under ``output_root``.

    Reads only the stored publication manifests (which carry the run's exact config), so
    it needs neither the experiment index nor the checkpoints and never regenerates
    trajectories — a pure metadata refresh for already-published categories.
    """
    seen = 0
    patched = 0
    for manifest_path in sorted(output_root.rglob(PUBLICATION_MANIFEST)):
        document = _load_object(manifest_path)
        if (
            document.get("schemaVersion") != PUBLICATION_SCHEMA
            or document.get("status") != "completed"
        ):
            continue
        seen += 1
        config = document.get("config") or {}
        run_id = document.get("runId") or ""
        split = document.get("split") or ""
        result_source = document.get("resultSource") or "experiment"
        metadata = None
        if result_source == "experiment":
            metadata = _publication_experiment_metadata(
                experiment_id=document.get("experimentId") or run_id,
                campaign=document.get("campaign"),
                checkpoint=document.get("checkpoint") or "",
                config=config,
                run_id=run_id,
            )
        found = _apply_category_refresh(
            frontend_airports_root / document["airport"] / "comparison" / "categories.json",
            document["category"],
            _publication_label(split, result_source, config, run_id),
            result_source,
            metadata,
        )
        if found:
            patched += 1
            print(f"  ✓ refreshed {document['airport']}/{document['category']}")
        else:
            print(
                f"  ⚠ no category {document['category']} at "
                f"{document['airport']} (publication {manifest_path})"
            )
    return seen, patched


def _publication_document(
    plan: PublicationPlan,
    *,
    status: str,
    failure: str | None = None,
    completed_steps: Iterable[str] = (),
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schemaVersion": PUBLICATION_SCHEMA,
        "updatedAtUtc": _utc_now(),
        "status": status,
        "failure": failure,
        "experimentId": plan.experiment.experiment_id,
        "campaign": plan.experiment.campaign,
        "runId": plan.experiment.run_id,
        "checkpoint": plan.experiment.checkpoint_relative,
        "checkpointSha256": plan.experiment.checkpoint_sha256,
        "airport": plan.airport,
        "split": plan.split,
        "resultSource": plan.result_source,
        "category": plan.category,
        "rawOutputDir": _path_for_manifest(plan.output_dir),
        "frontendDir": _path_for_manifest(plan.comparison_dir),
        "completedSteps": list(completed_steps),
        "config": plan.experiment.config,
        "recordRetention": plan.record_retention,
    }
    if plan.records_archive.is_file():
        document["recordsArchive"] = {
            "file": plan.records_archive.name,
            "bytes": plan.records_archive.stat().st_size,
            "sha256": _sha256(plan.records_archive),
        }
    if status == "completed":
        summary = _load_object(plan.summary)
        report = _load_object(plan.evaluation_report)
        document["accuracy"] = summary.get("accuracy")
        document["evaluation"] = {
            key: value
            for key, value in report.items()
            if key not in {"trajectories", "reference"}
        }
    return document


def _loose_prediction_records(output_dir: Path) -> list[Path]:
    records = sorted(output_dir.glob("*_states.json"))
    records.extend(sorted(output_dir.glob("*_eval.json")))
    references = output_dir / "references"
    if references.is_dir():
        records.extend(sorted(references.glob("*_reference_eval.json")))
    return records


def archive_prediction_records(plan: PublicationPlan, *, replace: bool = False) -> int:
    """Atomically archive regenerable per-flight records after CZML/evaluation publication.

    ``summary.json``, aggregate evaluation, flyability, publication metadata and all frontend
    assets remain directly readable.  The archive retains the exact per-flight JSON contract for
    later inspection while avoiding tens of thousands of loose files and substantially reducing
    disk use.  Source files are removed only after the completed archive has been reopened and its
    member roster exactly matches the intended set.
    """
    records = _loose_prediction_records(plan.output_dir)
    if not records:
        return 0
    relative_names = [str(path.relative_to(plan.output_dir)) for path in records]
    if plan.records_archive.is_file() and not replace:
        # Recovery path for an interrupted post-archive cleanup: never replace the already
        # complete archive with only the loose subset that happened not to be deleted yet.
        with tarfile.open(plan.records_archive, "r:gz") as archive:
            archived_names = {
                member.name for member in archive.getmembers() if member.isfile()
            }
        missing = set(relative_names) - archived_names
        if missing:
            raise RuntimeError(
                f"existing prediction archive is missing loose record {sorted(missing)[0]}"
            )
    else:
        temporary = plan.records_archive.with_suffix(plan.records_archive.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        try:
            with tarfile.open(temporary, "w:gz") as archive:
                for path, relative_name in zip(records, relative_names):
                    archive.add(path, arcname=relative_name, recursive=False)
            with tarfile.open(temporary, "r:gz") as archive:
                archived_names = [
                    member.name for member in archive.getmembers() if member.isfile()
                ]
            if archived_names != relative_names:
                raise RuntimeError("prediction record archive roster verification failed")
            temporary.replace(plan.records_archive)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    for path in records:
        path.unlink()
    references = plan.output_dir / "references"
    if references.is_dir() and not any(references.iterdir()):
        references.rmdir()
    _sha256.cache_clear()
    return len(records)


def rebuild_publication_index(output_root: Path = RAW_OUTPUT_ROOT) -> dict[str, Any]:
    publications: list[dict[str, Any]] = []
    if output_root.exists():
        for path in sorted(output_root.rglob(PUBLICATION_MANIFEST)):
            try:
                document = _load_object(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if document.get("schemaVersion") == PUBLICATION_SCHEMA:
                publications.append(document)
    result = {
        "schemaVersion": PUBLICATION_INDEX_SCHEMA,
        "generatedAtUtc": _utc_now(),
        "publications": publications,
        "counts": {
            status: sum(item.get("status") == status for item in publications)
            for status in ("completed", "failed", "blocked", "running")
        },
    }
    _write_json_atomic(output_root / "index.json", result)
    return result


def run_publication(
    plan: PublicationPlan,
    *,
    dry_run: bool,
    force: bool,
    fail_fast: bool,
) -> str:
    context = f"{plan.experiment.experiment_id} · {plan.airport} · {plan.split}"
    if not force and plan.is_complete():
        if not dry_run:
            refresh_category_metadata(plan)
            if plan.record_retention == "archive":
                archived = archive_prediction_records(plan)
            else:
                archived = 0
            if archived:
                _write_json_atomic(
                    plan.publication_manifest,
                    _publication_document(
                        plan,
                        status="completed",
                        completed_steps=("predict", "evaluate", "publish-czml", "archive-records"),
                    ),
                )
                rebuild_publication_index(plan.raw_output_root)
                print(f"  ✓ archived {archived} per-flight records for {context}")
        print(f"  ✓ reuse {context}")
        return "completed"

    error = plan.preflight_error()
    if error:
        print(f"  ⚠ blocked {context}: {error}")
        if not dry_run:
            _write_json_atomic(
                plan.publication_manifest,
                _publication_document(plan, status="blocked", failure=error),
            )
            rebuild_publication_index(plan.raw_output_root)
        return "blocked"

    commands = plan.commands()
    if dry_run:
        print(f"\n━━ {context}")
        for label, command in commands:
            print(f"  [{label}] {' '.join(command)}")
        return "planned"

    completed: list[str] = []
    _write_json_atomic(
        plan.publication_manifest,
        _publication_document(plan, status="running"),
    )
    rebuild_publication_index(plan.raw_output_root)
    try:
        for label, command in commands:
            print(f"\n=== [{context} · {label}] ===\n{' '.join(command)}", flush=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            completed.append(label)
        if plan.record_retention == "archive":
            archived = archive_prediction_records(plan, replace=True)
            completed.append("archive-records")
            print(f"  ✓ archived {archived} per-flight records -> {plan.records_archive}")
        refresh_category_metadata(plan)
        _write_json_atomic(
            plan.publication_manifest,
            _publication_document(plan, status="completed", completed_steps=completed),
        )
        rebuild_publication_index(plan.raw_output_root)
        return "completed"
    except KeyboardInterrupt as exc:
        failure = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(
            plan.publication_manifest,
            _publication_document(
                plan,
                status="failed",
                failure=failure,
                completed_steps=completed,
            ),
        )
        rebuild_publication_index(plan.raw_output_root)
        print(f"  ✗ interrupted {context}: {failure}", file=sys.stderr)
        raise
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(
            plan.publication_manifest,
            _publication_document(
                plan,
                status="failed",
                failure=failure,
                completed_steps=completed,
            ),
        )
        rebuild_publication_index(plan.raw_output_root)
        print(f"  ✗ failed {context}: {failure}", file=sys.stderr)
        if fail_fast:
            raise
        return "failed"


def _experiment_root(index_path: Path) -> Path:
    document = _load_object(index_path)
    return Path(document.get("root") or index_path.parent).resolve()


def _normalize_checkpoint_id(
    value: str,
    *,
    experiment_root: Path = EXPERIMENT_ROOT,
) -> str:
    normalized = value.strip().replace("\\", "/").strip("/")
    suffix = "/checkpoint.pt"
    if normalized.endswith(suffix):
        normalized = normalized[:-len(suffix)]
    candidate = Path(normalized)
    candidate_paths = (
        (candidate,) if candidate.is_absolute() else (REPO_ROOT / candidate,)
    )
    for candidate_path in candidate_paths:
        try:
            return candidate_path.resolve().relative_to(experiment_root.resolve()).as_posix()
        except ValueError:
            continue
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-index", type=Path, default=EXPERIMENT_INDEX)
    parser.add_argument("--output-root", type=Path, default=RAW_OUTPUT_ROOT)
    parser.add_argument("--harvest-root", type=Path, default=HARVEST_ROOT)
    parser.add_argument("--frontend-airports-root", type=Path, default=FRONTEND_AIRPORTS_ROOT)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help=(
            "indexed experiment run ID or repository-relative checkpoint.pt path; "
            "repeat to publish selected checkpoints"
        ),
    )
    parser.add_argument("--campaign", action="append", default=None)
    parser.add_argument("--airport", action="append", default=None)
    parser.add_argument(
        "--split",
        action="append",
        choices=DEVELOPMENT_SPLITS,
        default=None,
        help="development split; repeat as needed (default: train and val)",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--result-source",
        choices=("prediction", "experiment"),
        default="experiment",
        help="frontend result category (default: experiment)",
    )
    parser.add_argument(
        "--record-retention",
        choices=("archive", "loose"),
        default="archive",
        help="archive per-flight JSON after successful publication (default), or keep loose files",
    )
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--refresh-labels-only",
        action="store_true",
        help=(
            "recompute frontend labels/metadata for every completed publication under "
            "--output-root from its stored manifest; no prediction, no CZML, no archive"
        ),
    )
    args = parser.parse_args(argv)

    if args.refresh_labels_only:
        seen, patched = refresh_labels_from_manifests(
            args.output_root, args.frontend_airports_root
        )
        print(f"refreshed {patched} of {seen} completed publications under {args.output_root}")
        return 0

    if args.max_checkpoints is not None and args.max_checkpoints <= 0:
        parser.error("--max-checkpoints must be positive")
    experiment_root = _experiment_root(args.experiment_index)
    selected_ids = (
        {
            _normalize_checkpoint_id(value, experiment_root=experiment_root)
            for value in args.checkpoint
        }
        if args.checkpoint else None
    )
    checkpoints = discover_checkpoints(
        args.experiment_index,
        selected_ids=selected_ids,
        campaigns=set(args.campaign) if args.campaign else None,
    )
    if selected_ids is not None:
        missing = selected_ids - {checkpoint.experiment_id for checkpoint in checkpoints}
        if missing:
            parser.error(f"checkpoint is not a completed indexed run: {sorted(missing)[0]}")
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[:args.max_checkpoints]
    if not checkpoints:
        parser.error("no completed indexed checkpoints matched the selection")

    requested_airports = {value.strip().upper() for value in args.airport} if args.airport else None
    splits = tuple(args.split or DEVELOPMENT_SPLITS)
    plans: list[PublicationPlan] = []
    for checkpoint in checkpoints:
        airports = sorted(checkpoint.arrival_manifests)
        if requested_airports is not None:
            airports = [airport for airport in airports if airport in requested_airports]
        for airport in airports:
            for split in splits:
                plans.append(PublicationPlan(
                    checkpoint,
                    airport,
                    split,
                    result_source=args.result_source,
                    raw_output_root=args.output_root.resolve(),
                    harvest_root=args.harvest_root.resolve(),
                    frontend_airports_root=args.frontend_airports_root.resolve(),
                    device=args.device,
                    record_retention=args.record_retention,
                ))
    if not plans:
        parser.error("no checkpoint provenance contains the requested airport(s)")

    print(
        f"{len(checkpoints)} checkpoint(s), {len(plans)} airport/split publication job(s); "
        f"source={args.result_source}; outer-test is sealed"
    )
    counts: dict[str, int] = {}
    for plan in plans:
        status = run_publication(
            plan,
            dry_run=args.dry_run,
            force=args.force,
            fail_fast=args.fail_fast,
        )
        counts[status] = counts.get(status, 0) + 1
    print(f"\n✓ publication batch finished: {counts}")
    return 1 if counts.get("failed", 0) or counts.get("blocked", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
