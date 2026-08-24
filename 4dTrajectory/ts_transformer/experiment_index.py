"""Immutable run manifests and a repository-local experiment index.

The directory tree remains the source of truth.  New formal runs opt into an
``experiment_manifest.json`` whose status is updated atomically; ``rebuild_index`` also
discovers legacy run/comparison directories without rewriting them.  This gives old and
new experiments one searchable index while keeping every artifact in its original place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # run_naming -> config -> aerodynamic_model
    sys.path.insert(0, str(_REPO_ROOT))

from run_naming import run_display_name  # noqa: E402


RUN_MANIFEST_NAME = "experiment_manifest.json"
RUN_MANIFEST_SCHEMA = "ts-experiment-run-v1"
INDEX_SCHEMA = "ts-experiment-index-v1"
INDEX_JSON_NAME = "index.json"
INDEX_MARKDOWN_NAME = "INDEX.md"

_RUN_ARTIFACTS = (
    "checkpoint.pt",
    "checkpoint_metadata.json",
    "history.json",
    "fit_evaluation.json",
    "data_selection.json",
    "cv_results.json",
    "best_config.json",
)
_DISCOVERY_MARKERS = (
    RUN_MANIFEST_NAME,
    "checkpoint.pt",
    "history.json",
    "fit_evaluation.json",
    "data_selection.json",
    "report.json",
    "experiment_notes.md",
)
_OCCUPIED_RUN_MARKERS = tuple(
    name for name in _DISCOVERY_MARKERS if name != "experiment_notes.md"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_identity(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    commit = run("rev-parse", "HEAD").strip()
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": _sha256_bytes(status.encode()),
    }


def begin_run(
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    campaign_id: str,
    run_id: str,
    config: dict[str, Any],
    data_selection: dict[str, Any],
    command: Sequence[str],
) -> Path:
    """Create an immutable identity for a formal run and refuse directory reuse."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    occupied = [
        name
        for name in (*_OCCUPIED_RUN_MARKERS, "test_release.json")
        if (output / name).exists()
    ]
    if occupied:
        raise RuntimeError(
            f"formal experiment directory {output} is already occupied by {occupied[0]}; "
            "choose a new run id/directory instead of mixing or overwriting artifacts"
        )
    source = _git_identity(Path(repo_root).resolve())
    if source["dirty"]:
        raise RuntimeError(
            "formal experiments require a clean committed worktree; commit the exact code "
            "and data-contract changes before starting this run"
        )
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "status": "running",
        "created_at_utc": _utc_now(),
        "completed_at_utc": None,
        "output_dir": str(output),
        "source": source,
        "command": list(command),
        "config": config,
        "data_selection": data_selection,
        "artifacts": {},
        "failure": None,
    }
    path = output / RUN_MANIFEST_NAME
    _write_json_atomic(path, manifest)
    return path


def finish_run(manifest_path: str | Path, *, failure: str | None = None) -> dict[str, Any]:
    """Close a run manifest and fingerprint every standard artifact that exists."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    output = path.parent
    artifacts: dict[str, Any] = {}
    for name in _RUN_ARTIFACTS:
        artifact = output / name
        if artifact.is_file():
            artifacts[name] = {
                "bytes": artifact.stat().st_size,
                "sha256": _sha256_bytes(artifact.read_bytes()),
            }
    manifest.update({
        "status": "failed" if failure is not None else "completed",
        "completed_at_utc": _utc_now(),
        "artifacts": artifacts,
        "failure": failure,
    })
    _write_json_atomic(path, manifest)
    return manifest


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _legacy_entry(directory: Path, root: Path) -> dict[str, Any]:
    history = _load_json(directory / "history.json") or {}
    metadata = _load_json(directory / "checkpoint_metadata.json") or {}
    config = history.get("config") if isinstance(history.get("config"), dict) else {}
    has_checkpoint = (directory / "checkpoint.pt").is_file()
    has_history = (directory / "history.json").is_file()
    has_report = (directory / "report.json").is_file()
    if has_checkpoint and has_history:
        kind, status = "training", "completed"
    elif has_report:
        kind, status = "comparison", "completed"
    elif any(
        child.is_file()
        for artifact_name in ("checkpoint.pt", "history.json", "report.json")
        for child in directory.rglob(artifact_name)
        if child.parent != directory
    ):
        kind, status = "collection", "completed"
    else:
        kind, status = "training", "incomplete"
    return {
        "path": str(directory.relative_to(root)),
        "campaign_id": None,
        "run_id": directory.name,
        "kind": kind,
        "status": status,
        "legacy": True,
        "display_name": run_display_name(config) if config else None,
        "prediction_output": config.get("prediction_output") or metadata.get("prediction_output"),
        "aircraft_filter": config.get("aircraft_filter", "legacy-unspecified"),
        "seed": config.get("seed"),
        "split_sha256": metadata.get("split_sha256"),
        "artifacts": sorted(
            name for name in _DISCOVERY_MARKERS if (directory / name).is_file()
        ),
    }


def _manifest_entry(directory: Path, root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    return {
        "path": str(directory.relative_to(root)),
        "campaign_id": manifest.get("campaign_id"),
        "run_id": manifest.get("run_id") or directory.name,
        "kind": "training",
        "status": manifest.get("status", "unknown"),
        "legacy": False,
        "display_name": run_display_name(config) if config else None,
        "prediction_output": config.get("prediction_output"),
        "aircraft_filter": config.get("aircraft_filter"),
        "seed": config.get("seed"),
        "source": manifest.get("source"),
        "created_at_utc": manifest.get("created_at_utc"),
        "completed_at_utc": manifest.get("completed_at_utc"),
        "artifacts": sorted((manifest.get("artifacts") or {}).keys()),
    }


def rebuild_index(root: str | Path) -> dict[str, Any]:
    """Scan experiment directories and atomically write JSON + Markdown indexes."""
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    directories = {
        marker.parent
        for name in _DISCOVERY_MARKERS
        for marker in root_path.rglob(name)
        if marker.parent != root_path
    }
    # A run directory may contain a comparison/prediction subdirectory. Keep the deepest
    # real artifact directories as their own entries but never duplicate one directory.
    entries = []
    for directory in sorted(directories):
        manifest = _load_json(directory / RUN_MANIFEST_NAME)
        if manifest and manifest.get("schema_version") == RUN_MANIFEST_SCHEMA:
            entries.append(_manifest_entry(directory, root_path, manifest))
        else:
            entries.append(_legacy_entry(directory, root_path))
    campaign_by_top_directory = {
        Path(entry["path"]).parts[0]: entry["campaign_id"]
        for entry in entries
        if not entry["legacy"] and entry.get("campaign_id")
    }
    for entry in entries:
        if entry["campaign_id"] is None:
            entry["campaign_id"] = campaign_by_top_directory.get(
                Path(entry["path"]).parts[0]
            )
    document = {
        "schema_version": INDEX_SCHEMA,
        "generated_at_utc": _utc_now(),
        "root": str(root_path),
        "entries": entries,
        "counts": {
            status: sum(entry["status"] == status for entry in entries)
            for status in ("completed", "running", "failed", "incomplete", "unknown")
        },
    }
    _write_json_atomic(root_path / INDEX_JSON_NAME, document)

    lines = [
        "# TS Transformer experiment index",
        "",
        f"Generated: `{document['generated_at_utc']}`",
        "",
        "| Status | Kind | Campaign | Run | Name | Seed | Directory |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for entry in entries:
        lines.append(
            "| {status} | {kind} | {campaign} | {run} | {name} | {seed} | [{path}]({path}) |".format(
                status=entry["status"],
                kind=entry["kind"],
                campaign=entry.get("campaign_id") or "legacy",
                run=entry["run_id"],
                name=entry.get("display_name") or "—",
                seed=entry.get("seed") if entry.get("seed") is not None else "—",
                path=entry["path"],
            )
        )
    markdown = root_path / INDEX_MARKDOWN_NAME
    temporary = markdown.with_suffix(markdown.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(markdown)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    document = rebuild_index(args.root)
    print(
        f"indexed {len(document['entries'])} experiment directories under {args.root}; "
        f"counts={document['counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
