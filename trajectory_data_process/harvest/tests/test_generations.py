"""Frozen harvest generations: marker written once, absent roots skipped, half-frozen refused."""

from __future__ import annotations

import hashlib
import json

import pytest

from trajectory_data_process.harvest import generations


def _harvest(root, airport="KAAA"):
    for name in ("arrivals", "tracks"):
        (root / airport / name).mkdir(parents=True)
        (root / airport / name / "manifest.json").write_text(
            json.dumps({"schema_version": f"{name}-vX"}), encoding="utf-8"
        )


def test_freeze_records_every_airports_manifest_digests_once(tmp_path):
    _harvest(tmp_path / "gen")
    marker = generations.freeze_generation(tmp_path / "gen", reason="test")
    arrivals = (tmp_path / "gen" / "KAAA" / "arrivals" / "manifest.json").read_bytes()
    assert marker["airports"]["KAAA"]["arrival_manifest_sha256"] == hashlib.sha256(arrivals).hexdigest()
    assert marker["airports"]["KAAA"]["arrival_schema_version"] == "arrivals-vX"
    assert json.loads((tmp_path / "gen" / generations.FROZEN_MARKER).read_text()) == marker
    with pytest.raises(FileExistsError):
        generations.freeze_generation(tmp_path / "gen", reason="again")


def test_freeze_ignores_what_is_not_an_airport_directory(tmp_path):
    """A killed reclassification leaves ``.KSJC-reclassify-*`` beside the airports."""
    _harvest(tmp_path / "gen")
    (tmp_path / "gen" / ".KSJC-reclassify-abc" / "KSJC" / "tracks").mkdir(parents=True)
    marker = generations.freeze_generation(tmp_path / "gen", reason="test")
    assert list(marker["airports"]) == ["KAAA"]


def test_freeze_refuses_an_incomplete_airport(tmp_path):
    _harvest(tmp_path / "gen")
    (tmp_path / "gen" / "KBBB" / "tracks").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="only a complete harvest root"):
        generations.freeze_generation(tmp_path / "gen", reason="test")


def test_an_absent_generation_is_skipped_and_a_half_frozen_one_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(generations, "OUTPUTS_ROOT", tmp_path)
    monkeypatch.setattr(generations, "FROZEN_GENERATIONS", ("gone", "half"))
    assert generations.frozen_generation_roots() == ()
    _harvest(tmp_path / "half")
    with pytest.raises(FileNotFoundError, match="registered but has no"):
        generations.frozen_generation_roots()
    generations.freeze_generation(tmp_path / "half", reason="test")
    assert generations.frozen_generation_roots() == (tmp_path / "half",)
    arrivals = (tmp_path / "half" / "KAAA" / "arrivals" / "manifest.json").read_bytes()
    assert generations.is_frozen_arrival_manifest(arrivals)
    assert not generations.is_frozen_arrival_manifest(arrivals + b"\n")
