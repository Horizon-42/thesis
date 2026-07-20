#!/usr/bin/env python
"""Delete the generated data of the optimization + ts_transformer pipelines.

The destructive sibling of ``archive_pipeline_data.py`` (which MOVES the same
data into a named snapshot — prefer it when the run might still be worth
keeping). This script wipes the whole chain's history so the next batch starts
from a clean slate, with nothing stale left to mix in:

  1. flight_scenarios/outputs/                       scenarios (step 1)
  2. 4dTrajectory/outputs/<ICAO>/…                   optimizer categories
                                                     (asdb/runway/runway_cons)
                                                     AND ts training + prediction
                                                     dirs (ts_*/, ts_pred_*/)
  3. aeroviz-4d/public/data/airports/<ICAO>/comparison/     frontend comparison CZML
  4. aeroviz-4d/public/data/airports/<ICAO>/trajectories.czml*   frontend observed layer
     aeroviz-4d/public/data/airports/<ICAO>/landings/            (per-runway CZML)

NOT touched, ever: the static airport layers (airport.json, runway.geojson,
waypoints.geojson, procedures*, charts/, obstacles.geojson, local-terrain/),
``data/archive/`` snapshots, and anything tracked by git.

Kept by default, deletable by flag:

  * ``--include-downloads``  also wipes trajectory_data_process/outputs/ — the RAW
    OpenSky downloads (landings/, raw_tracks/, history_rows/, manifests/,
    source_responses/). OFF by default: re-creating them needs OpenSky history
    access and hours of downloading; everything else above is recomputable from
    them.
  * ``--include-parked``     also wipes the ``_``-prefixed dirs under
    4dTrajectory/outputs (parked research artifacts, e.g. _pre_b3_transport,
    _ablation_norm — the ablation numbers quoted in the ts README live there).

Nothing is deleted without either an interactive confirmation or ``--yes``.

Usage:
    python clean_pipeline_data.py --dry-run              # preview only
    python clean_pipeline_data.py                        # plan + confirm + delete
    python clean_pipeline_data.py --yes                  # no prompt (scripts)
    python clean_pipeline_data.py --include-downloads    # ALSO drop the raw downloads
"""

from __future__ import annotations

import argparse
import functools
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Single sources: the pipeline roots come from the runner script, the humanize +
# empty-dir pruning helpers from the archiver (repo-root scripts import flat).
from archive_pipeline_data import _human, _prune_empty_dirs  # noqa: E402
from run_scenario_pipeline import (  # noqa: E402
    COMPARISON_AIRPORTS_ROOT,
    LANDINGS_DIR,
    OPT_OUTPUTS_ROOT,
    SCENARIOS_DIR,
)

DOWNLOADS_ROOT = LANDINGS_DIR.parent  # trajectory_data_process/outputs


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


def deletion_groups(*, include_parked: bool, include_downloads: bool):
    """``(label, files)`` groups to delete, the container dirs to remove once
    emptied, and the notes about what is deliberately being kept."""
    groups: list[tuple[str, list[Path]]] = []
    containers: list[Path] = []
    kept: list[str] = []

    groups.append(("scenarios    (flight_scenarios/outputs)", _tree_files(SCENARIOS_DIR)))

    # Parked research artifacts are ``_``-prefixed dirs at ANY level under the
    # outputs root (e.g. outputs/KRDU/_pre_b3_transport, outputs/KRDU/_ablation_norm
    # — the previous ts generation and the ablation numbers the ts README quotes).
    def _under_parked(rel_dir_parts: tuple[str, ...]) -> bool:
        return any(part.startswith("_") for part in rel_dir_parts)

    opt_files = _tree_files(OPT_OUTPUTS_ROOT)
    if not include_parked:
        opt_files = [f for f in opt_files
                     if not _under_parked(f.relative_to(OPT_OUTPUTS_ROOT).parts[:-1])]
        # Note each TOP-MOST parked dir once (a parked dir nested inside another is
        # already covered by its ancestor's note).
        for d in sorted(OPT_OUTPUTS_ROOT.rglob("_*") if OPT_OUTPUTS_ROOT.exists() else []):
            if d.is_dir() and not _under_parked(d.relative_to(OPT_OUTPUTS_ROOT).parts[:-1]):
                kept.append(f"parked research artifacts {d.relative_to(REPO_ROOT)}/ "
                            f"(pass --include-parked to delete)")
    groups.append(("optimization + ts (4dTrajectory/outputs)", opt_files))

    comparison_files: list[Path] = []
    observed_files: list[Path] = []
    if COMPARISON_AIRPORTS_ROOT.exists():
        for airport_dir in sorted(COMPARISON_AIRPORTS_ROOT.iterdir()):
            if not airport_dir.is_dir():
                continue
            comparison_dir = airport_dir / "comparison"
            if comparison_dir.exists():
                comparison_files += _tree_files(comparison_dir)
                containers.append(comparison_dir)
            # glob bypasses _tree_files, so apply the same git-tracked guard here.
            observed_files += [f for f in sorted(airport_dir.glob("trajectories.czml*"))
                               if f.resolve() not in _tracked_files()]
            landings_dir = airport_dir / "landings"
            if landings_dir.exists():
                observed_files += _tree_files(landings_dir)
                containers.append(landings_dir)
    groups.append(("frontend comparison (airports/*/comparison)", comparison_files))
    groups.append(("frontend observed layer (airports/*/{trajectories.czml*, landings})",
                   observed_files))

    if include_downloads:
        groups.append(("RAW DOWNLOADS (trajectory_data_process/outputs)",
                       _tree_files(DOWNLOADS_ROOT)))
    else:
        kept.append(f"raw downloads {DOWNLOADS_ROOT.relative_to(REPO_ROOT)}/ "
                    f"(pass --include-downloads to delete — re-creating them needs "
                    f"OpenSky history access)")

    # Surface, never silently drop, any git-tracked file the guard excluded from the scan
    # (0 today — the roots are git-ignored — but a future ``git add`` under one must be
    # visible, not quietly skipped).
    scanned_roots = [SCENARIOS_DIR, OPT_OUTPUTS_ROOT, *containers]
    if include_downloads:
        scanned_roots.append(DOWNLOADS_ROOT)
    tracked_in_scan = sum(
        1 for f in _tracked_files()
        if any(root == f or root in f.parents for root in scanned_roots)
    )
    if tracked_in_scan:
        kept.append(f"{tracked_in_scan} git-tracked file(s) under these roots "
                    f"(never deleted — commit them elsewhere if that is wrong)")

    return groups, containers, kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--include-downloads", action="store_true",
        help="ALSO delete the raw OpenSky downloads (trajectory_data_process/outputs). "
             "OFF by default — everything else is recomputable from them; they are not",
    )
    parser.add_argument(
        "--include-parked", action="store_true",
        help="ALSO delete the _-prefixed parked dirs under 4dTrajectory/outputs "
             "(_pre_b3_transport, _ablation_norm, …)",
    )
    parser.add_argument("--yes", action="store_true",
                        help="delete without the interactive confirmation")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the deletion plan without deleting anything")
    args = parser.parse_args()

    groups, containers, kept = deletion_groups(
        include_parked=args.include_parked, include_downloads=args.include_downloads)

    total_files = sum(len(files) for _, files in groups)
    total_bytes = sum(f.stat().st_size for _, files in groups for f in files)

    print(f"\n━━ clean pipeline data  ·  {REPO_ROOT}")
    for label, files in groups:
        size = sum(f.stat().st_size for f in files)
        print(f"   {label:<58} {len(files):>6} files  {_human(size):>10}")
    print(f"   {'TOTAL':<58} {total_files:>6} files  {_human(total_bytes):>10}")
    for note in kept:
        print(f"   · keeping {note}")
    print("   · static airport layers and data/archive snapshots are never touched")
    print("   · reversible alternative: python archive_pipeline_data.py archive <name>")

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

    for label, files in groups:
        for f in files:
            f.unlink()
        if files:
            print(f"   ✓ {label}: {len(files)} files deleted")

    # Tidy the emptied trees: prune below the anchors we actually deleted from, and drop
    # the emptied per-airport container dirs (comparison/, landings/) themselves. The
    # downloads tree is pruned only when --include-downloads deleted from it — otherwise it
    # is "kept untouched" and must stay exactly as found, empty subdirs and all.
    prune_anchors = [SCENARIOS_DIR, OPT_OUTPUTS_ROOT]
    if args.include_downloads:
        prune_anchors.append(DOWNLOADS_ROOT)
    for anchor in prune_anchors:
        _prune_empty_dirs(anchor)
    for container in containers:
        _prune_empty_dirs(container)
        if container.exists() and not any(container.iterdir()):
            container.rmdir()

    print(f"\n✓ deleted {total_files} files ({_human(total_bytes)})")


if __name__ == "__main__":
    main()
