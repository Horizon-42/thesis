"""A harvest root's staging leftovers — what a harvest killed mid-write leaves behind.

Every rewrite of a harvest stages its output beside the real one and swaps it in: the observed records under
``<ICAO>/approach/`` (`observed.write_observed_records`: a staging copy, then the previous records moved aside), a
reclassification, a merge or a freshness rebuild in the root (``.<ICAO>-reclassify-*``, ``.<ICAO>-merge-*``,
``.<ICAO>-freshness-*``), and the swap of a reclassified or merged ``tracks/`` (the old tree moved aside as
``<ICAO>/.tracks-before-reclassify-*`` / ``.tracks-before-merge-*``, then deleted). A SIGKILL mid-write leaves them.
Readers never see them (they follow the rosters), but they hold disk (the v5 root once held a 170 MB one).

Nothing removes them unasked: there is no lock, so a sweep could delete another run's live staging. Every harvest run
lists the airport's leftovers, and ``--remove-staging-leftovers`` removes them — run it when no other harvest writes the
root. A moved-aside ``tracks/`` is never removed while ``tracks/`` itself is missing: a kill between the swap's two
renames leaves it the only copy of the stored tracks.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from trajectory_data_process.harvest.store import HarvestPaths

#: ``approach/``'s staged records and the records moved aside while the staging is swapped in.
RECORDS_STAGING_PREFIX = ".records-staging-"
RECORDS_PREVIOUS_PREFIX = ".records-previous-"
#: ``<ICAO>/``'s old ``tracks/``, moved aside while a reclassified or merged one is swapped in.
TRACKS_BACKUP_PREFIX = ".tracks-before-"


def reclassify_prefix(code: str) -> str:
    """A reclassification's temporary root, in the harvest root."""
    return f".{code}-reclassify-"


def merge_prefix(code: str) -> str:
    """A merge's temporary root, in the harvest root."""
    return f".{code}-merge-"


def freshness_prefix(code: str) -> str:
    """A freshness rebuild's temporary root, in the destination root."""
    return f".{code}-freshness-"


def replace_tracks_directory(staged: Path, destination: Path, *, kind: str) -> None:
    """Swap a staged ``tracks/`` in: the old tree is moved aside, the staged one renamed into place,
    then the old one deleted (restored instead if the rename fails)."""
    backup = destination.parent / f"{TRACKS_BACKUP_PREFIX}{kind}-{uuid4().hex}"
    destination.replace(backup)
    try:
        staged.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def staging_leftovers(paths: HarvestPaths) -> list[Path]:
    """The airport's staging leftovers, in path order."""
    found = [path for prefix in (RECORDS_STAGING_PREFIX, RECORDS_PREVIOUS_PREFIX) for path in paths.approach.glob(f"{prefix}*")]
    found += list(paths.airport.glob(f"{TRACKS_BACKUP_PREFIX}*"))
    found += [path for prefix in (reclassify_prefix(paths.code), merge_prefix(paths.code), freshness_prefix(paths.code))
              for path in paths.root.glob(f"{prefix}*")]
    return sorted(found)


def orphaned_tracks_backups(paths: HarvestPaths, leftovers: list[Path]) -> list[Path]:
    """The moved-aside ``tracks/`` trees among ``leftovers`` when ``tracks/`` itself is missing."""
    if paths.tracks.exists():
        return []
    return [path for path in leftovers if path.name.startswith(TRACKS_BACKUP_PREFIX)]


def remove_staging_leftovers(paths: HarvestPaths) -> list[Path]:
    """Remove the airport's staging leftovers; the removed paths. Refuses, removing nothing, while a
    moved-aside ``tracks/`` may be the only copy of the stored tracks."""
    leftovers = staging_leftovers(paths)
    orphaned = orphaned_tracks_backups(paths, leftovers)
    if orphaned:
        raise ValueError(
            f"{paths.tracks} is missing and {orphaned} may be the only copy of the stored tracks: "
            f"restore it (rename it to {paths.tracks.name}/) before removing staging leftovers"
        )
    for path in leftovers:
        shutil.rmtree(path)
    return leftovers
