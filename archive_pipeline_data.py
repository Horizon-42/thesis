#!/usr/bin/env python
"""Archive / restore everything ``run_scenario_pipeline.py`` produces.

The pipeline writes its data production to three places (all git-ignored):

  1. flight_scenarios/outputs/*_scenarios.json                 (step 1)
  2. 4dTrajectory/outputs/<ICAO>/<category>/…                  (steps 2–4:
        states, eval, summary.json, evaluation_report.{json,html}, references/)
  3. aeroviz-4d/public/data/airports/<ICAO>/comparison/…       (step 5:
        comparison_*.czml, comparison_index.json, categories.json, report)

This script moves that whole set into a NAMED snapshot under ``data/archive/``,
so recovery is a straight replay back into the working tree — nothing is
recomputed. Only the pipeline's outputs are touched; the shared
``airports/<ICAO>/`` static data (runway.geojson, procedures, trajectories.czml,
…) is left in place because only the ``comparison/`` sub-folder of each airport
comes from this pipeline.

A snapshot lives at ``data/archive/<name>/`` and is one of two forms:

  * TREE (default) — the repo-relative tree mirrored verbatim. Move (not copy)
    is the default: the outputs run ~6 GB on the same filesystem, so a move is
    an instant relink and never doubles disk use.
  * COMPRESSED (--compress) — a single ``archive.tar.zst`` (tar piped through
    ``zstd -19 --long=27``; long-range matching dedups the pipeline's repeated
    reference tracks + the eval-vs-states rollout overlap, ~10× smaller).
    Falls back to a stdlib ``archive.tar.xz`` when ``zstd`` is not on PATH.

``restore`` auto-detects the form; both carry an ``_archive_manifest.json``.

Usage:
    # snapshot the current pipeline data as "hs-run-jul06" (working tree emptied):
    python archive_pipeline_data.py archive hs-run-jul06

    # same, but compressed to ~10× smaller (a single archive.tar.zst):
    python archive_pipeline_data.py archive hs-run-jul06 --compress

    # bring it back (snapshot consumed; --keep leaves a copy in the archive):
    python archive_pipeline_data.py restore hs-run-jul06

    # see what has been archived:
    python archive_pipeline_data.py list

    # preview either without moving anything:
    python archive_pipeline_data.py archive trapezoidal --compress --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT_DEFAULT = REPO_ROOT / "data" / "archive"

# The metadata file written at each snapshot root (skipped on a tree restore).
MANIFEST_NAME = "_archive_manifest.json"

# Compressed-snapshot filenames + the zstd knobs. --long=27 (128 MiB window) is
# deliberately the largest window that zstd decompresses WITHOUT an explicit
# --long flag, so restore needs no special handling.
TARBALL_ZSTD = "archive.tar.zst"
TARBALL_XZ = "archive.tar.xz"
ZSTD_ARGS = ["-19", "--long=27", "-T0"]

# The three source anchors (repo-relative). Empty directories BELOW these are
# pruned after a move; the anchors themselves are kept.
_SCENARIOS_DIR = REPO_ROOT / "flight_scenarios" / "outputs"
_OPT_OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"
_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"


# ── enumerating the pipeline's data production ────────────────────────────────

def _iter_tree(root: Path) -> list[Path]:
    """Every file under ``root`` (recursive), or [] if it does not exist."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def production_groups() -> list[tuple[str, list[Path]]]:
    """The pipeline outputs currently in the working tree, grouped by source.

    Returns ``(label, files)`` with files as absolute paths under REPO_ROOT;
    ``.DS_Store`` noise is dropped (it is not pipeline data)."""
    comparison_files: list[Path] = []
    for comparison_dir in sorted(_AIRPORTS_ROOT.glob("*/comparison")):
        comparison_files += _iter_tree(comparison_dir)

    groups = [
        ("scenarios   (flight_scenarios/outputs)",
         sorted(_SCENARIOS_DIR.glob("*_scenarios.json"))),
        ("optimization (4dTrajectory/outputs)",
         _iter_tree(_OPT_OUTPUTS_ROOT)),
        ("comparison   (airports/*/comparison)",
         comparison_files),
    ]
    return [(label, [f for f in files if f.name != ".DS_Store"])
            for label, files in groups]


# ── helpers ───────────────────────────────────────────────────────────────────

def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _move_file(src: Path, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not force:
            raise FileExistsError(dst)
        dst.unlink()
    shutil.move(str(src), str(dst))


def _copy_file(src: Path, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        raise FileExistsError(dst)
    shutil.copy2(src, dst)


def _prune_empty_dirs(anchor: Path) -> None:
    """Remove empty directories strictly BELOW ``anchor`` (the anchor stays)."""
    if not anchor.exists():
        return
    for path in sorted((p for p in anchor.rglob("*") if p.is_dir()),
                       key=lambda p: len(p.parts), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()


def _prune_source_dirs() -> None:
    """Tidy up the anchors after a move (empty category/comparison dirs)."""
    for anchor in (_SCENARIOS_DIR, _OPT_OUTPUTS_ROOT):
        _prune_empty_dirs(anchor)
    for comparison_dir in sorted(_AIRPORTS_ROOT.glob("*/comparison")):
        _prune_empty_dirs(comparison_dir)
        if comparison_dir.exists() and not any(comparison_dir.iterdir()):
            comparison_dir.rmdir()


def _fail(msg: str) -> None:
    print(f"✗ {msg}", file=sys.stderr)
    sys.exit(1)


# ── compression (tar + zstd, stdlib tar.xz fallback) ──────────────────────────

def _zstd_available() -> bool:
    return bool(shutil.which("zstd") and shutil.which("tar"))


def _tar_base() -> list[str]:
    """``tar`` plus ``--no-mac-metadata`` on bsdtar (macOS). Without it, bsdtar
    stores every file's extended attributes as a separate AppleDouble ``._name``
    member — OS cruft (provenance/quarantine), not pipeline data, which doubles
    the archive's member count and surfaces as stray ``._*`` files if the archive
    is ever extracted off macOS. GNU tar has no such flag and needs none."""
    cmd = ["tar"]
    try:
        version = subprocess.run(["tar", "--version"], capture_output=True,
                                 text=True).stdout
        if "bsdtar" in version:
            cmd.append("--no-mac-metadata")
    except OSError:
        pass
    return cmd


def _compress(relpaths: list[Path], snapshot: Path) -> tuple[str, int]:
    """Write the repo-relative ``relpaths`` into a single tarball in ``snapshot``.

    Returns ``(codec, compressed_bytes)``. Uses ``tar | zstd -19 --long=27`` when
    available (its long-range window dedups the repeated reference tracks), else
    a stdlib ``tar.xz``."""
    snapshot.mkdir(parents=True, exist_ok=True)
    if _zstd_available():
        tarball = snapshot / TARBALL_ZSTD
        listfile = snapshot / "_filelist.txt"
        listfile.write_text("\n".join(str(r) for r in relpaths) + "\n")
        try:
            with open(tarball, "wb") as out:
                tar = subprocess.Popen(
                    [*_tar_base(), "-cf", "-", "-C", str(REPO_ROOT), "-T", str(listfile)],
                    stdout=subprocess.PIPE)
                zstd = subprocess.Popen(
                    ["zstd", *ZSTD_ARGS, "-q", "-c"], stdin=tar.stdout, stdout=out)
                tar.stdout.close()  # let tar see SIGPIPE if zstd dies
                zstd_rc = zstd.wait()
                tar_rc = tar.wait()
        finally:
            listfile.unlink(missing_ok=True)
        if tar_rc or zstd_rc:
            tarball.unlink(missing_ok=True)
            _fail(f"compression failed (tar={tar_rc}, zstd={zstd_rc})")
        return "zstd", tarball.stat().st_size

    import tarfile
    print("   (zstd not found — falling back to stdlib tar.xz; slower)")
    tarball = snapshot / TARBALL_XZ
    with tarfile.open(tarball, "w:xz") as tf:
        for rel in relpaths:
            tf.add(REPO_ROOT / rel, arcname=str(rel))
    return "xz", tarball.stat().st_size


def _find_tarball(snapshot: Path) -> tuple[Path | None, str | None]:
    for codec, name in (("zstd", TARBALL_ZSTD), ("xz", TARBALL_XZ)):
        path = snapshot / name
        if path.exists():
            return path, codec
    return None, None


def _extract(tarball: Path, codec: str) -> None:
    """Unpack a snapshot tarball back into the working tree (REPO_ROOT)."""
    if codec == "zstd":
        zstd = subprocess.Popen(
            ["zstd", "-d", "--long=27", "-q", "-c", str(tarball)], stdout=subprocess.PIPE)
        tar = subprocess.Popen(
            [*_tar_base(), "-xf", "-", "-C", str(REPO_ROOT)], stdin=zstd.stdout)
        zstd.stdout.close()
        tar_rc = tar.wait()
        zstd_rc = zstd.wait()
        if tar_rc or zstd_rc:
            _fail(f"extraction failed (zstd={zstd_rc}, tar={tar_rc})")
        return

    import tarfile
    with tarfile.open(tarball, "r:xz") as tf:
        try:
            tf.extractall(REPO_ROOT, filter="data")  # py3.12+
        except TypeError:
            tf.extractall(REPO_ROOT)


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_archive(name: str, archive_root: Path, *,
                copy: bool, compress: bool, force: bool, dry_run: bool) -> None:
    snapshot = archive_root / name
    if snapshot.exists() and not dry_run:
        if not force:
            _fail(f"snapshot {name!r} already exists at {snapshot} "
                  f"(use --force to replace it)")
        shutil.rmtree(snapshot)

    groups = production_groups()
    total_files = sum(len(files) for _, files in groups)
    total_bytes = sum(f.stat().st_size for _, files in groups for f in files)

    verb = "copy" if copy else "move"
    fmt = "compressed tar.zst" if compress else "tree"
    print(f"\n━━ archive {name!r}  ·  {verb} · {fmt}  ·  {snapshot}")
    for label, files in groups:
        size = sum(f.stat().st_size for f in files)
        print(f"   {label:<40} {len(files):>6} files  {_human(size):>10}")
    print(f"   {'TOTAL':<40} {total_files:>6} files  {_human(total_bytes):>10}")

    if total_files == 0:
        _fail("nothing to archive — no pipeline outputs found in the working tree")

    if dry_run:
        dest = (snapshot / (TARBALL_ZSTD if _zstd_available() else TARBALL_XZ)
                if compress else snapshot)
        note = "  (est ~10× smaller)" if compress else ""
        print(f"   (dry-run — nothing moved)")
        print(f"   -> {dest}{note}")
        return

    relpaths = [f.relative_to(REPO_ROOT) for _, files in groups for f in files]

    codec: str | None = None
    compressed_bytes: int | None = None
    if compress:
        print(f"   compressing {total_files} files …", flush=True)
        codec, compressed_bytes = _compress(relpaths, snapshot)
        if not copy:
            for rel in relpaths:
                (REPO_ROOT / rel).unlink(missing_ok=True)
    else:
        place = _copy_file if copy else _move_file
        done = 0
        for rel in relpaths:
            place(REPO_ROOT / rel, snapshot / rel, force=True)
            done += 1
            if done % 2000 == 0:
                print(f"   … {done}/{total_files}", flush=True)

    if not copy:
        _prune_source_dirs()

    manifest = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": verb,
        "format": "compressed" if compress else "tree",
        "codec": codec,
        "repo_root": str(REPO_ROOT),
        "file_count": total_files,
        "total_bytes": total_bytes,
        "compressed_bytes": compressed_bytes,
        "sources": [label for label, _ in groups],
    }
    (snapshot / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    if compress:
        ratio = total_bytes / compressed_bytes if compressed_bytes else 0
        print(f"✓ archived {total_files} files: {_human(total_bytes)} -> "
              f"{_human(compressed_bytes)} ({ratio:.1f}× smaller) -> {snapshot}"
              + ("" if copy else "  (working tree emptied)"))
    else:
        print(f"✓ archived {total_files} files ({_human(total_bytes)}) -> {snapshot}"
              + ("" if copy else "  (working tree emptied)"))


def cmd_restore(name: str, archive_root: Path, *,
                keep: bool, force: bool, dry_run: bool) -> None:
    snapshot = archive_root / name
    if not snapshot.exists():
        _fail(f"no snapshot {name!r} under {archive_root} "
              f"(run `list` to see what is archived)")
    tarball, codec = _find_tarball(snapshot)

    # Refuse to restore over pipeline outputs already sitting in the working tree
    # (a fresh run) — that would silently mix two runs. --force overrides.
    existing = sum(len(files) for _, files in production_groups())
    if existing and not force:
        _fail(f"the working tree already has {existing} pipeline output file(s) — "
              f"restoring would overwrite/mix them; archive them under another name "
              f"first, or pass --force")

    if tarball is not None:
        _restore_compressed(name, snapshot, tarball, codec, keep=keep, dry_run=dry_run)
    else:
        _restore_tree(name, snapshot, keep=keep, dry_run=dry_run)


def _restore_compressed(name: str, snapshot: Path, tarball: Path, codec: str, *,
                        keep: bool, dry_run: bool) -> None:
    manifest_path = snapshot / MANIFEST_NAME
    count = json.loads(manifest_path.read_text()).get("file_count", "?") \
        if manifest_path.exists() else "?"
    print(f"\n━━ restore {name!r}  ·  unpack {codec} · {count} files  ·  {tarball}"
          + ("  (snapshot preserved)" if keep else "  (snapshot consumed)"))
    if dry_run:
        print(f"   (dry-run — would extract into {REPO_ROOT})")
        return
    _extract(tarball, codec)
    if not keep:
        shutil.rmtree(snapshot)
    print(f"✓ restored {count} files into the working tree"
          + ("" if keep else f"  ·  snapshot {name!r} removed"))


def _restore_tree(name: str, snapshot: Path, *, keep: bool, dry_run: bool) -> None:
    files = [p for p in sorted(snapshot.rglob("*"))
             if p.is_file() and p != snapshot / MANIFEST_NAME]
    if not files:
        _fail(f"snapshot {name!r} is empty")
    total_bytes = sum(f.stat().st_size for f in files)
    verb = "copy" if keep else "move"
    print(f"\n━━ restore {name!r}  ·  {verb} back · {len(files)} files  "
          f"{_human(total_bytes)}"
          + ("  (snapshot preserved)" if keep else "  (snapshot consumed)"))
    if dry_run:
        print(f"   (dry-run — would restore into {REPO_ROOT})")
        return
    place = _copy_file if keep else _move_file
    for i, f in enumerate(files, 1):
        place(f, REPO_ROOT / f.relative_to(snapshot), force=True)
        if i % 2000 == 0:
            print(f"   … {i}/{len(files)}", flush=True)
    if not keep:
        shutil.rmtree(snapshot)
    print(f"✓ restored {len(files)} files ({_human(total_bytes)}) into the working tree"
          + ("" if keep else f"  ·  snapshot {name!r} removed"))


def cmd_list(archive_root: Path) -> None:
    if not archive_root.exists() or not any(archive_root.iterdir()):
        print(f"(no snapshots under {archive_root})")
        return
    print(f"snapshots under {archive_root}:\n")
    for snapshot in sorted(p for p in archive_root.iterdir() if p.is_dir()):
        manifest_path = snapshot / MANIFEST_NAME
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            files = m.get("file_count", "?")
            created = m.get("created_utc", "?")
            if m.get("format") == "compressed":
                comp = m.get("compressed_bytes") or 0
                raw = m.get("total_bytes") or 0
                ratio = f"{raw / comp:.1f}×" if comp else "?"
                tag = f"{m.get('codec', '?')}, {ratio}"
                print(f"  {snapshot.name:<24} {files:>6} files  "
                      f"{_human(comp):>10}  [{tag}]  {created}")
            else:
                print(f"  {snapshot.name:<24} {files:>6} files  "
                      f"{_human(m.get('total_bytes', 0)):>10}  [tree]  {created}")
        else:
            fs = [p for p in snapshot.rglob("*") if p.is_file()]
            size = sum(f.stat().st_size for f in fs)
            print(f"  {snapshot.name:<24} {len(fs):>6} files  {_human(size):>10}  "
                  f"(no manifest)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--archive-root", type=Path, default=ARCHIVE_ROOT_DEFAULT,
        help=f"where snapshots live (default: {ARCHIVE_ROOT_DEFAULT})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_arc = sub.add_parser("archive", help="move pipeline outputs into a named snapshot")
    p_arc.add_argument("name", help="snapshot name (e.g. hs-run-jul06)")
    p_arc.add_argument("--compress", action="store_true",
                       help="store a single zstd tarball (~10× smaller) instead of a tree")
    p_arc.add_argument("--copy", action="store_true",
                       help="copy instead of move (leaves the working tree in place)")
    p_arc.add_argument("--force", action="store_true",
                       help="replace an existing snapshot of the same name")
    p_arc.add_argument("--dry-run", action="store_true",
                       help="print the plan without moving anything")

    p_res = sub.add_parser("restore", help="restore a named snapshot into the working tree")
    p_res.add_argument("name", help="snapshot name to restore")
    p_res.add_argument("--keep", action="store_true",
                       help="preserve the snapshot (default consumes it)")
    p_res.add_argument("--force", action="store_true",
                       help="restore even if the working tree already has outputs")
    p_res.add_argument("--dry-run", action="store_true",
                       help="print the plan without moving anything")

    sub.add_parser("list", help="list archived snapshots")

    args = parser.parse_args()
    if args.command == "archive":
        cmd_archive(args.name, args.archive_root, copy=args.copy,
                    compress=args.compress, force=args.force, dry_run=args.dry_run)
    elif args.command == "restore":
        cmd_restore(args.name, args.archive_root,
                    keep=args.keep, force=args.force, dry_run=args.dry_run)
    elif args.command == "list":
        cmd_list(args.archive_root)


if __name__ == "__main__":
    main()
