"""A harvest root's staging leftovers — what a harvest killed mid-write leaves behind.

Every rewrite of a harvest stages its output beside the real one and swaps it in: the observed records under
``<ICAO>/approach/`` (`observed.write_observed_records`: a staging copy, then the previous records moved aside), a
reclassification or a merge in the root (``.<ICAO>-reclassify-*``, ``.<ICAO>-merge-*``). A SIGKILL mid-write leaves them.
Readers never see them (they follow the rosters), but they hold disk (the v5 root once held a 170 MB one).

Nothing removes them unasked: there is no lock, so a sweep could delete another run's live staging. Every harvest run
lists the airport's leftovers, and ``--remove-staging-leftovers`` removes them — run it when no other harvest writes the
root.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from trajectory_data_process.harvest.store import HarvestPaths

#: ``approach/``'s staged records and the records moved aside while the staging is swapped in.
RECORDS_STAGING_PREFIX = ".records-staging-"
RECORDS_PREVIOUS_PREFIX = ".records-previous-"


def reclassify_prefix(code: str) -> str:
    """A reclassification's temporary root, in the harvest root."""
    return f".{code}-reclassify-"


def merge_prefix(code: str) -> str:
    """A merge's temporary root, in the harvest root."""
    return f".{code}-merge-"


def staging_leftovers(paths: HarvestPaths) -> list[Path]:
    """The airport's staging leftovers, in path order."""
    found = [path for prefix in (RECORDS_STAGING_PREFIX, RECORDS_PREVIOUS_PREFIX) for path in paths.approach.glob(f"{prefix}*")]
    found += [path for prefix in (reclassify_prefix(paths.code), merge_prefix(paths.code))
              for path in paths.root.glob(f"{prefix}*")]
    return sorted(found)


def remove_staging_leftovers(paths: HarvestPaths) -> list[Path]:
    """Remove the airport's staging leftovers; the removed paths."""
    leftovers = staging_leftovers(paths)
    for path in leftovers:
        shutil.rmtree(path)
    return leftovers
