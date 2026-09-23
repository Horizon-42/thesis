"""Frozen harvest generations: the data a stored checkpoint was trained on, kept as written.

The live harvest (``outputs/harvest``) is rebuilt whenever the roster rules change or new
data is merged in, and every ts checkpoint fingerprints the exact bytes it trained on
(``ts_transformer.data.data_provenance``: each airport's arrival-manifest SHA-256 and each
source track's). A checkpoint can therefore only be replayed against the generation it
was trained on. Freezing keeps that generation: its root is MOVED aside (never edited or
copied) and ``FROZEN.json`` records the SHA-256 of every airport's arrival and tracks
manifest at the moment it was frozen.

Reading a frozen roster is not a compatibility path. The loader code is the same for
every roster; the schema string only says which rule WROTE a roster, and the version gate
exists so that NEW training never picks up a roster written under an older rule. A frozen
roster is identified by its content instead -- the same identity the checkpoint pins --
so ``arrivals.load_arrival_flights`` reads it as written and nothing converts it.

A generation is registered in ``FROZEN_GENERATIONS`` (a reviewed code change, never a
directory scan: the repo never discovers data by globbing).

    python -m trajectory_data_process.harvest.generations freeze \\
        trajectory_data_process/outputs/harvest-v5-20260823 --reason "..."
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs"
FROZEN_MARKER = "FROZEN.json"
FROZEN_SCHEMA = "harvest-frozen-generation-v1"
# An airport directory of a harvest root; anything else there (e.g. the hidden
# ``.KSJC-reclassify-*`` a killed reclassification leaves behind) is not part of the data.
_AIRPORT_DIR = re.compile(r"[A-Z]{4}")

#: Every frozen generation, by directory name under ``OUTPUTS_ROOT``.
#: ``harvest-v5-20260823``: the harvest every ts checkpoint of 2026-08-24..09-23 trained
#: on (v5 arrival rosters, 42,650 arrivals), frozen when the 2026-08-22..09-22 download
#: was merged into the live root.
FROZEN_GENERATIONS: tuple[str, ...] = ("harvest-v5-20260823",)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_generation_roots() -> tuple[Path, ...]:
    """The registered frozen roots present on this machine (a clone may hold none).

    A registered root that exists without its marker is a half-frozen generation and
    raises; an absent one is simply not here, and a replay that needs it says so
    (``is_frozen_arrival_manifest`` cannot match its digests).
    """
    roots = []
    for name in FROZEN_GENERATIONS:
        root = OUTPUTS_ROOT / name
        if not root.exists():
            continue
        if not (root / FROZEN_MARKER).is_file():
            raise FileNotFoundError(
                f"frozen harvest generation {root} is registered but has no {FROZEN_MARKER}"
            )
        roots.append(root)
    return tuple(roots)


@functools.lru_cache(maxsize=None)
def _frozen_arrival_digests(roots: tuple[Path, ...]) -> frozenset[str]:
    digests: set[str] = set()
    for root in roots:
        marker = json.loads((root / FROZEN_MARKER).read_text(encoding="utf-8"))
        if marker.get("schema_version") != FROZEN_SCHEMA:
            raise ValueError(f"{root / FROZEN_MARKER} is not a {FROZEN_SCHEMA} marker")
        digests.update(entry["arrival_manifest_sha256"] for entry in marker["airports"].values())
    return frozenset(digests)


def is_frozen_arrival_manifest(manifest_bytes: bytes) -> bool:
    """Is this arrival manifest byte-identical to one a registered generation froze?"""
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    return digest in _frozen_arrival_digests(frozen_generation_roots())


def freeze_generation(root: Path, *, reason: str) -> dict[str, Any]:
    """Write ``FROZEN.json`` for a harvest root that was just moved aside.

    Records, per airport directory (``_AIRPORT_DIR``), the SHA-256 of both manifests. The
    arrival digest is what the loader matches; the tracks digest is the audit record of the
    roster the lead-landing and scene context read on replay (a frozen root is made
    read-only, and the merge verification re-hashes both). Refuses a root that is already
    frozen: the marker is the record of what the bytes WERE.
    """
    marker_path = root / FROZEN_MARKER
    if marker_path.exists():
        raise FileExistsError(f"{marker_path} exists; a generation is frozen once")
    airports: dict[str, dict[str, str]] = {}
    for airport_dir in sorted(
        path for path in root.iterdir() if path.is_dir() and _AIRPORT_DIR.fullmatch(path.name)
    ):
        arrivals = airport_dir / "arrivals" / "manifest.json"
        tracks = airport_dir / "tracks" / "manifest.json"
        if not arrivals.is_file() or not tracks.is_file():
            raise FileNotFoundError(
                f"{airport_dir} lacks arrivals/manifest.json or tracks/manifest.json; "
                "only a complete harvest root can be frozen"
            )
        airports[airport_dir.name] = {
            "arrival_manifest_sha256": _sha256(arrivals),
            "arrival_schema_version": json.loads(arrivals.read_text(encoding="utf-8"))[
                "schema_version"
            ],
            "tracks_manifest_sha256": _sha256(tracks),
        }
    if not airports:
        raise ValueError(f"{root} holds no airport harvest to freeze")
    marker = {
        "schema_version": FROZEN_SCHEMA,
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        "airports": airports,
    }
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trajectory_data_process.harvest.generations")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="write FROZEN.json for a moved-aside root")
    freeze.add_argument("root", type=Path)
    freeze.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    marker = freeze_generation(args.root, reason=args.reason)
    for airport, entry in marker["airports"].items():
        print(f"{airport}: arrivals {entry['arrival_schema_version']} "
              f"{entry['arrival_manifest_sha256'][:12]}…, tracks "
              f"{entry['tracks_manifest_sha256'][:12]}…")
    print(f"frozen -> {args.root / FROZEN_MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
