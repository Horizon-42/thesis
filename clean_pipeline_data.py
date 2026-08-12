#!/usr/bin/env python
"""Delete only allow-listed, regenerable pipeline publications and views.

This cleaner deliberately does *not* scan an entire output root and call everything in
it disposable. It selects only these producer-owned artifacts:

  1. ``flight_scenarios/outputs/<ICAO>_*_scenarios.json``;
  2. canonical optimizer categories ``fitted_adsb``, ``runway``, ``runway_cons`` and
     their ``shared_references``;
  3. standalone ``4dTrajectory/outputs/<ICAO>/ts_pred_*`` prediction publications;
  4. frontend comparison and observed CZML publications; and
  5. harvest ``arrivals/`` and ``approach/`` derived views.

Never selected: downloaded ``tracks/``, training checkpoints/history, formal experiment
directories/manifests, checkpoint-adjacent ``test_release.json``, parked/manual/unknown
model outputs, static airport data, git-tracked files, and ``data/archive``.

Airport scope is mandatory. Files are moved into a same-filesystem staging directory
first; a staging failure restores every moved file before the command exits. Nothing is
removed without an interactive confirmation or ``--yes``.

Usage:
    conda run -n aeroviz python clean_pipeline_data.py --airport KRDU --dry-run
    conda run -n aeroviz python clean_pipeline_data.py --airport KRDU
    conda run -n aeroviz python clean_pipeline_data.py --all-airports --yes
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parent

# Pipeline roots come from the producer scripts. The humanizer/pruner are import-light
# helpers only; this cleaner does not use the archiver's broader production scan.
from archive_pipeline_data import _human, _prune_empty_dirs  # noqa: E402
from prepare_scenario_inputs import HARVEST_TRACKS_ROOT, SCENARIOS_DIR  # noqa: E402
from run_scenario_optimization import (  # noqa: E402
    COMPARISON_AIRPORTS_ROOT,
    OPT_OUTPUTS_ROOT,
)

# Despite its historical name in the runner, this is the harvest root, not the
# per-airport ``tracks/`` directory.
HARVEST_ROOT = HARVEST_TRACKS_ROOT

OPTIMIZER_OUTPUT_DIRS = frozenset({
    "fitted_adsb",
    "runway",
    "runway_cons",
    "shared_references",
})
PROTECTED_EXPERIMENT_FILES = frozenset({
    "checkpoint.pt",
    "checkpoint_metadata.json",
    "history.json",
    "experiment_manifest.json",
    "test_release.json",
})


@functools.lru_cache(maxsize=1)
def _tracked_files() -> frozenset[Path]:
    """Absolute paths of every git-tracked file.

    The docstring promises this script never touches anything tracked by git. That
    guarantee is enforced HERE, by construction, rather than trusted to .gitignore: the
    output roots are all git-ignored today, but a single ``git add`` of a curated result
    under one of them would otherwise be deleted silently. Raises if git can't answer —
    the safety guarantee depends on it, so we refuse to delete rather than guess.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "cannot list git-tracked files, so the 'never delete tracked files' guarantee "
            "cannot be verified — refusing to delete. Run inside the git working tree."
        ) from exc
    return frozenset((REPO_ROOT / rel).resolve() for rel in out.split("\0") if rel)


def _tree_files(root: Path) -> list[Path]:
    """Every UNTRACKED file under ``root`` (recursive), or [] if it does not exist.

    Git-tracked files are excluded (see :func:`_tracked_files`); the count skipped is
    surfaced by :func:`deletion_groups`, never dropped silently.
    """
    if not root.exists():
        return []
    tracked = _tracked_files()
    return sorted(p for p in root.rglob("*") if p.is_file() and p.resolve() not in tracked)


def _is_airport_code(code: str) -> bool:
    return len(code) == 4 and code.isalnum()


def _selected(code: str, airports: set[str] | None) -> bool:
    """Select only ICAO-shaped airport namespaces, never roots such as POOLED."""
    return _is_airport_code(code) and (airports is None or code.upper() in airports)


def _harvest_category_files(
    category: str, airports: set[str] | None
) -> tuple[list[Path], list[Path]]:
    """Files and existing per-airport dirs for one canonical harvest category.

    Default cleanup needs only ``arrivals`` and ``approach``. Scanning those roots
    directly avoids walking every downloaded track just to preserve it.
    """
    files: list[Path] = []
    roots: list[Path] = []
    if not HARVEST_ROOT.exists():
        return files, roots
    for airport_dir in sorted(HARVEST_ROOT.iterdir()):
        category_dir = airport_dir / category
        if (
            not airport_dir.is_dir()
            or not _selected(airport_dir.name, airports)
            or not category_dir.is_dir()
        ):
            continue
        files.extend(_tree_files(category_dir))
        roots.append(category_dir)
    return files, roots


def _scenario_files(airports: set[str] | None) -> list[Path]:
    if not SCENARIOS_DIR.exists():
        return []
    tracked = _tracked_files()
    files: list[Path] = []
    for path in sorted(SCENARIOS_DIR.glob("*_scenarios.json")):
        airport = path.name.split("_arrivals", 1)[0].upper()
        if (
            path.is_file()
            and _selected(airport, airports)
            and path.resolve() not in tracked
        ):
            files.append(path)
    return files


def _model_output_files(
    airports: set[str] | None,
) -> tuple[list[Path], list[Path], list[str]]:
    """Allow-listed optimizer/prediction outputs; never training or experiments."""
    files: list[Path] = []
    containers: list[Path] = []
    kept: list[str] = []
    if not OPT_OUTPUTS_ROOT.exists():
        return files, containers, kept
    for airport_dir in sorted(OPT_OUTPUTS_ROOT.iterdir()):
        if not airport_dir.is_dir() or not _selected(airport_dir.name, airports):
            continue
        for candidate in sorted(airport_dir.iterdir()):
            if not candidate.is_dir() or not (
                candidate.name in OPTIMIZER_OUTPUT_DIRS
                or candidate.name.startswith("ts_pred_")
            ):
                continue
            if candidate.name.startswith("ts_pred_"):
                summary = candidate / "summary.json"
                if candidate.name.endswith("_test"):
                    kept.append(
                        f"final-test prediction {candidate.relative_to(REPO_ROOT)}/"
                    )
                    continue
                if not summary.is_file():
                    kept.append(
                        f"protected model output {candidate.relative_to(REPO_ROOT)}/ "
                        "(missing prediction summary)"
                    )
                    continue
                try:
                    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    kept.append(
                        f"protected model output {candidate.relative_to(REPO_ROOT)}/ "
                        "(unreadable prediction summary)"
                    )
                    continue
                split = (
                    summary_payload.get("split")
                    if isinstance(summary_payload, dict)
                    else None
                )
                if split == "test":
                    kept.append(
                        f"final-test prediction {candidate.relative_to(REPO_ROOT)}/"
                    )
                    continue
                if split != "val":
                    kept.append(
                        f"protected model output {candidate.relative_to(REPO_ROOT)}/ "
                        f"(prediction split is {split!r}, not 'val')"
                    )
                    continue
            protected = next(
                (
                    path for path in candidate.rglob("*")
                    if path.is_file() and path.name in PROTECTED_EXPERIMENT_FILES
                ),
                None,
            )
            if protected is not None:
                kept.append(
                    f"protected model output {candidate.relative_to(REPO_ROOT)}/ "
                    f"(contains {protected.name})"
                )
                continue
            files.extend(_tree_files(candidate))
            containers.append(candidate)
    return files, containers, kept


def _comparison_protection_reason(directory: Path) -> str | None:
    """Why a comparison tree must stay intact, or ``None`` when fully owned."""
    registry = directory / "categories.json"
    if not registry.is_file():
        return "missing categories registry"
    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable categories registry"
    categories = payload.get("categories") if isinstance(payload, dict) else None
    if not isinstance(categories, list):
        return "invalid categories registry"

    declared_dirs: set[str] = set()
    for category in categories:
        if not isinstance(category, dict):
            return "invalid categories registry"
        category_dir = category.get("dir")
        if (
            not isinstance(category_dir, str)
            or not category_dir
            or Path(category_dir).name != category_dir
            or category_dir in {".", ".."}
        ):
            return "invalid categories registry"
        declared_dirs.add(category_dir)
        if (
            category.get("resultSource") == "experiment"
            or category.get("datasetSplit") == "test"
        ):
            return "contains an experiment or final-test publication"

    if any(not (directory / name).is_dir() for name in declared_dirs):
        return "categories registry references a missing directory"
    for child in directory.iterdir():
        if child == registry:
            continue
        if child.is_symlink() or not child.is_dir() or child.name not in declared_dirs:
            return "contains content not owned by the categories registry"
        if child.name.startswith("experiment_") or child.name.endswith("_test"):
            return "contains an experiment or final-test publication"
    return None


def deletion_groups(*, airports: set[str] | None):
    """``(label, files)`` groups to delete, the container dirs to remove once
    emptied, and the notes about what is deliberately being kept."""
    groups: list[tuple[str, list[Path]]] = []
    containers: list[Path] = []
    kept: list[str] = [
        "experiment and training outputs under 4dTrajectory/outputs/ "
        "(checkpoints, histories, test ledgers, formal/manual/unknown runs)",
        f"harvest tracks {HARVEST_ROOT.relative_to(REPO_ROOT)}/*/tracks/ "
        "(downloaded source data; never selected by this cleaner)",
    ]

    groups.append(("scenarios    (allow-listed *_scenarios.json)", _scenario_files(airports)))
    model_files, model_dirs, model_kept = _model_output_files(airports)
    groups.append(("optimizer + standalone predictions (allow-listed)", model_files))
    containers.extend(model_dirs)
    kept.extend(model_kept)

    comparison_files: list[Path] = []
    observed_files: list[Path] = []
    if COMPARISON_AIRPORTS_ROOT.exists():
        for airport_dir in sorted(COMPARISON_AIRPORTS_ROOT.iterdir()):
            if not airport_dir.is_dir() or not _selected(airport_dir.name, airports):
                continue
            comparison_dir = airport_dir / "comparison"
            if comparison_dir.exists():
                protection = _comparison_protection_reason(comparison_dir)
                if protection is None:
                    comparison_files += _tree_files(comparison_dir)
                    containers.append(comparison_dir)
                else:
                    kept.append(
                        f"frontend comparison {comparison_dir.relative_to(REPO_ROOT)}/ "
                        f"({protection})"
                    )
            # The observed producer owns exactly this canonical filename. Prefix
            # lookalikes may be curated copies and are deliberately out of scope.
            trajectories = airport_dir / "trajectories.czml"
            if (
                trajectories.is_file()
                and trajectories.resolve() not in _tracked_files()
            ):
                observed_files.append(trajectories)
            landings_dir = airport_dir / "landings"
            if landings_dir.exists():
                observed_files += _tree_files(landings_dir)
                containers.append(landings_dir)
    groups.append(("frontend comparison (airports/*/comparison)", comparison_files))
    groups.append(("frontend observed layer (airports/*/{trajectories.czml, landings})",
                   observed_files))

    arrival_files, arrival_dirs = _harvest_category_files("arrivals", airports)
    approach_files, approach_dirs = _harvest_category_files("approach", airports)
    groups.extend(
        [
            ("harvest arrivals  (harvest/*/arrivals; derived)", arrival_files),
            ("harvest approach  (harvest/*/approach; derived)", approach_files),
        ]
    )
    containers.extend(arrival_dirs)
    containers.extend(approach_dirs)

    # Surface, never silently drop, any git-tracked file the guard excluded from the scan
    # (0 today — the roots are git-ignored — but a future ``git add`` under one must be
    # visible, not quietly skipped).
    scanned_roots = [SCENARIOS_DIR, *containers]
    tracked_in_scan = sum(
        1 for f in _tracked_files()
        if any(root == f or root in f.parents for root in scanned_roots)
    )
    if tracked_in_scan:
        kept.append(f"{tracked_in_scan} git-tracked file(s) under these roots "
                    f"(never deleted — commit them elsewhere if that is wrong)")

    return groups, containers, kept


def _validate_plan(files: list[Path]) -> list[Path]:
    """Reject duplicates, protected sentinels, and paths outside producer roots."""
    roots = tuple(
        root.resolve()
        for root in (
            SCENARIOS_DIR,
            OPT_OUTPUTS_ROOT,
            COMPARISON_AIRPORTS_ROOT,
            HARVEST_ROOT,
        )
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if path.name in PROTECTED_EXPERIMENT_FILES:
            raise RuntimeError(f"refusing to select protected experiment file {path}")
        if not any(resolved.is_relative_to(root) for root in roots):
            raise RuntimeError(f"cleanup target escapes producer roots: {path}")
        if resolved in _tracked_files():
            raise RuntimeError(f"cleanup target is git-tracked: {path}")
        seen.add(resolved)
        unique.append(path)
    return sorted(unique)


def _move_file(source: Path, destination: Path) -> Path:
    """Replace hook kept separate so rollback behavior can be failure-tested."""
    return source.replace(destination)


def delete_files_transactionally(files: list[Path], *, staging_root: Path) -> None:
    """Stage every target before commit; restore all staged files on move failure."""
    selected = _validate_plan(files)
    if staging_root.exists():
        raise FileExistsError(f"cleanup staging path already exists: {staging_root}")
    staging_root.mkdir(parents=True)
    moved: list[tuple[Path, Path]] = []
    try:
        for source in selected:
            relative = source.absolute().relative_to(REPO_ROOT.absolute())
            destination = staging_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _move_file(source, destination)
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            _move_file(destination, source)
        shutil.rmtree(staging_root)
        raise
    shutil.rmtree(staging_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--airport",
        action="append",
        metavar="ICAO",
        help="clean one airport; repeat for more than one",
    )
    scope.add_argument(
        "--all-airports",
        action="store_true",
        help="clean allow-listed derived data for every airport",
    )
    parser.add_argument("--yes", action="store_true",
                        help="delete without the interactive confirmation")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the deletion plan without deleting anything")
    args = parser.parse_args()

    airports = None if args.all_airports else {
        value.strip().upper() for value in args.airport
    }
    if airports is not None and any(not _is_airport_code(code) for code in airports):
        parser.error("--airport must be a four-character alphanumeric ICAO code")
    groups, containers, kept = deletion_groups(airports=airports)
    selected_files = _validate_plan([path for _, files in groups for path in files])

    total_files = len(selected_files)
    total_bytes = sum(path.stat().st_size for path in selected_files)

    scope_label = "all airports" if airports is None else ", ".join(sorted(airports))
    print(f"\n━━ clean regenerable pipeline data  ·  {scope_label}  ·  {REPO_ROOT}")
    for label, files in groups:
        size = sum(f.stat().st_size for f in files)
        print(f"   {label:<58} {len(files):>6} files  {_human(size):>10}")
    print(f"   {'TOTAL':<58} {total_files:>6} files  {_human(total_bytes):>10}")
    for note in kept:
        print(f"   · keeping {note}")
    print("   · static airport layers and data/archive snapshots are never touched")

    if total_files == 0:
        print("\n✓ nothing to delete — the working tree is already clean")
        return
    if args.dry_run:
        print("\n(dry-run — nothing deleted)")
        return

    if not args.yes:
        if not sys.stdin.isatty():
            print("✗ refusing to delete without confirmation in a non-interactive "
                  "shell — pass --yes", file=sys.stderr)
            sys.exit(1)
        answer = input(f"\nType 'delete' to remove {total_files} files "
                       f"({_human(total_bytes)}): ")
        if answer.strip().lower() != "delete":
            print("aborted — nothing deleted")
            return

    staging = REPO_ROOT / f".pipeline-clean-staging-{uuid4().hex}"
    delete_files_transactionally(selected_files, staging_root=staging)
    for label, files in groups:
        if files:
            print(f"   ✓ {label}: {len(files)} files deleted")

    # Tidy only explicitly selected producer containers. Never recurse from the broad
    # model or harvest roots, where protected experiment/source directories also live.
    for container in containers:
        _prune_empty_dirs(container)
        if container.exists() and not any(container.iterdir()):
            container.rmdir()

    print(f"\n✓ deleted {total_files} files ({_human(total_bytes)})")


if __name__ == "__main__":
    main()
