"""A killed harvest's staging leftovers are listed by every run and removed only on request."""

from __future__ import annotations

from trajectory_data_process.harvest.__main__ import main
from trajectory_data_process.harvest.staging import (
    RECORDS_PREVIOUS_PREFIX,
    RECORDS_STAGING_PREFIX,
    merge_prefix,
    reclassify_prefix,
    staging_leftovers,
)
from trajectory_data_process.harvest.store import HarvestPaths


def _leftovers(paths: HarvestPaths) -> list:
    made = [
        paths.approach / f"{RECORDS_STAGING_PREFIX}a",
        paths.approach / f"{RECORDS_PREVIOUS_PREFIX}b",
        paths.root / f"{reclassify_prefix(paths.code)}c",
        paths.root / f"{merge_prefix(paths.code)}d",
    ]
    for path in made:
        (path / "records").mkdir(parents=True)
        (path / "records" / "x.json").write_text("{}", encoding="utf-8")
    return sorted(made)


def test_the_leftovers_are_this_airports_staging_and_nothing_live(tmp_path):
    paths = HarvestPaths(tmp_path, "KAAA")
    made = _leftovers(paths)
    (paths.approach / "records").mkdir()
    paths.tracks.mkdir(parents=True)
    _leftovers(HarvestPaths(tmp_path, "KBBB"))

    assert staging_leftovers(paths) == made
    assert staging_leftovers(HarvestPaths(tmp_path / "empty", "KAAA")) == []


def test_the_cli_removes_them_and_leaves_the_live_trees(tmp_path, capsys):
    paths = HarvestPaths(tmp_path, "KAAA")
    made = _leftovers(paths)
    (paths.approach / "records").mkdir()
    paths.tracks.mkdir(parents=True)
    other = _leftovers(HarvestPaths(tmp_path, "KBBB"))

    assert main(["--airport", "KAAA", "--output", str(tmp_path), "--remove-staging-leftovers",
                 "--frontend-data", str(tmp_path / "frontend"), "--adsb-metadata", str(tmp_path / "adsb")]) == 0

    assert not any(path.exists() for path in made)
    assert all(path.exists() for path in other)
    assert (paths.approach / "records").is_dir() and paths.tracks.is_dir()
    assert "removed 4 staging leftover(s)" in capsys.readouterr().out
