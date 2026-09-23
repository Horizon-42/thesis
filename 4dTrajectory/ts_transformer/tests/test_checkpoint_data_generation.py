"""A checkpoint's arrival manifests are found by the digest it recorded: the live harvest's,
or a frozen generation's -- never by path alone (a merged live root has other bytes)."""

from __future__ import annotations

import hashlib

import pytest

from trajectory_data_process.harvest import generations
from ts_transformer.repo_layout import checkpoint_arrival_manifest, checkpoint_arrival_manifests


def _manifest(root, airport, text):
    path = root / airport / "arrivals" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    (root / airport / "tracks").mkdir()
    (root / airport / "tracks" / "manifest.json").write_text("{}", encoding="utf-8")
    return path


def _payload(**digests):
    return {"data_provenance": {"manifests": [
        {"airport": airport, "arrival_manifest_sha256": digest}
        for airport, digest in digests.items()
    ]}}


def test_a_checkpoint_resolves_to_the_generation_whose_bytes_it_trained_on(tmp_path, monkeypatch):
    monkeypatch.setattr(generations, "OUTPUTS_ROOT", tmp_path)
    monkeypatch.setattr(generations, "FROZEN_GENERATIONS", ("harvest-old",))
    live = tmp_path / "harvest"
    live_krdu = _manifest(live, "KRDU", '{"schema_version": "v", "merged": true}')
    _manifest(live, "KSJC", '{"schema_version": "v", "merged": true, "airport": "KSJC"}')
    old_krdu = _manifest(tmp_path / "harvest-old", "KRDU", '{"schema_version": "v", "merged": false}')
    old_ksjc = _manifest(tmp_path / "harvest-old", "KSJC", '{"schema_version": "v", "merged": false, "airport": "KSJC"}')
    generations.freeze_generation(tmp_path / "harvest-old", reason="test")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731

    old = _payload(KRDU=sha(old_krdu), KSJC=sha(old_ksjc))
    assert checkpoint_arrival_manifests(old, live) == [old_krdu, old_ksjc]
    assert checkpoint_arrival_manifest(_payload(KRDU=sha(live_krdu)), "krdu", live) == live_krdu

    with pytest.raises(ValueError, match="no harvest generation holds the KRDU arrival manifest"):
        checkpoint_arrival_manifests(_payload(KRDU="0" * 64), live)
