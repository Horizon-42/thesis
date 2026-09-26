"""Which procedure's masks a prior speaks under (`prior.masks`, prior design §5.1): the sets and their names, the digest of
the data they read, the record beside a checkpoint and every refusal of it, the free generation's choice, and the speaker
masking with the vocabulary's rules alone unless it is given more."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.experiments.prior_free_generation import procedure_masks_named
from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, RUNWAY, Words
from ts_transformer.prior import masks as masks_module
from ts_transformer.prior.generate import Speaker
from ts_transformer.prior.masks import (
    MASKS_FILE, MASKS_SCHEMA, PROCEDURE_ALTITUDES, SETS, ProcedureMasks, finals_sha256, read_masks, write_masks,
)
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.support import instruction_airport, instruction_spec
from ts_transformer.tests.test_prior import _model as _prior_model, _signals, _two_runways
from ts_transformer.tests.test_prior_procedure import _altitudes, _final

GIT = {"head": "test", "dirty": False}


def _geometries():
    return {"KXXX": instruction_airport()}


@pytest.fixture
def published(monkeypatch):
    """Today's procedures: the fixture airport's one candidate's final (the harvest's data never read)."""
    today = {"KXXX": (_final(),)}
    monkeypatch.setattr(masks_module, "published_procedures", lambda geometries: dict(today))
    return today


def _model_dir(path, weights: bytes = b"weights"):
    path.mkdir()
    (path / "checkpoint.pt").write_bytes(weights)
    return path


def test_the_sets_are_named_once_in_the_code_s_order_and_only_the_altitudes_read_the_finals(published):
    none = ProcedureMasks.none()
    assert none.names == () and none.finals is None and not none.altitudes and none.data_sha256() == {}
    altitudes = ProcedureMasks.build([PROCEDURE_ALTITUDES], _geometries())
    assert altitudes.names == SETS == (PROCEDURE_ALTITUDES,) and altitudes.altitudes and altitudes.finals == published
    assert altitudes.data_sha256() == {PROCEDURE_ALTITUDES: finals_sha256(published)}
    assert ProcedureMasks.build([], _geometries()) == none
    for names in (["procedure-altitudes-v1"], [PROCEDURE_ALTITUDES, PROCEDURE_ALTITUDES]):
        with pytest.raises(ValueError, match="this code implements"):
            ProcedureMasks.build(names, _geometries())
    with pytest.raises(ValueError, match="read the finals"):
        ProcedureMasks((PROCEDURE_ALTITUDES,), None)
    with pytest.raises(ValueError, match="read the finals"):
        ProcedureMasks((), published)
    with pytest.raises(ValueError, match="this code implements"):
        ProcedureMasks(("procedure-altitudes-v1",), None)


def test_the_digest_reads_every_field_of_every_final_whatever_the_airports_order():
    one = {"KXXX": (_final(),)}
    digest = finals_sha256(one)
    assert finals_sha256({"KXXX": (_final(),)}) == digest
    final = _final()
    moved = [_final(decision_m=final.decision_m + 0.5), _final(faf_d_m=final.faf_d_m + 1.0),
             _final(crossing_m=final.crossing_m + 0.1), replace(final, glidepath_tan=final.glidepath_tan * 1.001),
             replace(final, cone=replace(final.cone, course_width_m=final.cone.course_width_m + 1.0)),
             replace(final, candidate=replace(final.candidate, threshold_e_m=1.0))]
    assert all(finals_sha256({"KXXX": (m,)}) != digest for m in moved)
    other = {"KYYY": (_final(crossing_m=200.0),)}
    assert finals_sha256({**one, **other}) == finals_sha256({**other, **one})


def test_the_record_binds_the_masks_to_the_checkpoint_beside_it_and_is_never_overwritten(tmp_path, published):
    directory = _model_dir(tmp_path / "augmented")
    write_masks(directory, ProcedureMasks.build([PROCEDURE_ALTITUDES], _geometries()), writer="test", git=GIT)
    record = json.loads((directory / MASKS_FILE).read_text(encoding="utf-8"))
    assert record["schema"] == MASKS_SCHEMA and record["written_by"] == "test" and record["git"] == GIT
    assert record["sets"] == [{"name": PROCEDURE_ALTITUDES, "data_sha256": finals_sha256(published)}]
    assert read_masks(directory, _geometries()) == ProcedureMasks((PROCEDURE_ALTITUDES,), published)
    with pytest.raises(FileExistsError):
        write_masks(directory, ProcedureMasks.none(), writer="test", git=GIT)
    base = _model_dir(tmp_path / "base")
    write_masks(base, ProcedureMasks.none(), writer="test", git=GIT)
    assert json.loads((base / MASKS_FILE).read_text(encoding="utf-8"))["sets"] == []
    assert read_masks(base, _geometries()) == ProcedureMasks.none()


def test_a_record_that_does_not_hold_on_today_s_code_and_data_is_refused(tmp_path, published, monkeypatch):
    directory = _model_dir(tmp_path / "model")
    with pytest.raises(ValueError, match="records no procedure's masks"):
        read_masks(directory, _geometries())
    write_masks(directory, ProcedureMasks.build([PROCEDURE_ALTITUDES], _geometries()), writer="test", git=GIT)
    path = directory / MASKS_FILE
    good = json.loads(path.read_text(encoding="utf-8"))

    def refused(record, match):
        path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            read_masks(directory, _geometries())

    refused({**good, "schema": "ts-prior-procedure-masks-v0"}, "not ts-prior-procedure-masks-v1")
    refused({**good, "checkpoint_sha256": "0" * 64}, "another checkpoint")
    refused({**good, "sets": [{"name": "procedure-altitudes-v1", "data_sha256": None}]}, "does not implement")
    path.write_text(json.dumps(good), encoding="utf-8")
    (directory / "checkpoint.pt").write_bytes(b"other weights")                 # the weights swapped under the record
    with pytest.raises(ValueError, match="another checkpoint"):
        read_masks(directory, _geometries())
    (directory / "checkpoint.pt").write_bytes(b"weights")
    assert read_masks(directory, _geometries()).altitudes
    # the procedures moved (a new DA): speaking under today's is a decision, not a default
    monkeypatch.setattr(masks_module, "published_procedures", lambda geometries: {"KXXX": (_final(decision_m=500.0),)})
    with pytest.raises(ValueError, match="changed since"):
        read_masks(directory, _geometries())


def test_free_generation_speaks_under_the_model_s_own_masks_unless_told_otherwise(published):
    own = ProcedureMasks((PROCEDURE_ALTITUDES,), published)
    assert procedure_masks_named("own", own, _geometries()) is own
    assert procedure_masks_named("none", own, _geometries()) == ProcedureMasks.none()
    assert procedure_masks_named(PROCEDURE_ALTITUDES, ProcedureMasks.none(), _geometries()) == own
    with pytest.raises(ValueError, match="this code implements"):
        procedure_masks_named("procedure-altitudes-v1", own, _geometries())


def test_the_speaker_masks_by_the_vocabulary_s_rules_alone_unless_given_the_procedure_s():
    words, geometry = Words(instruction_spec()), _two_runways()
    signals = _signals(N_LOOK + 3)
    nowhere = _final(candidate=geometry.candidates[1], crossing_m=110.0, faf_d_m=0.0, decision_m=-1_000.0)

    def speaker(procedure_masks):
        return Speaker(_prior_model(variant="no-context"), [signals], [geometry], None, words, max_rows=N_LOOK + 3,
                       generator=torch.Generator().manual_seed(0), procedure_masks=procedure_masks)

    bare = speaker(ProcedureMasks.none())
    assert list(bare.forbidden) == list(bare.allowed) == [RUNWAY, APPROACH, ANGLE] and bare.procedure == []
    masked = speaker(_altitudes(geometry, _final(), nowhere))
    assert list(masked.forbidden) == list(masked.allowed) == [RUNWAY, APPROACH, ANGLE, ALTITUDE]
    assert [rules.columns for rules in masked.procedure] == [(ALTITUDE, ANGLE)]
    # the vocabulary's rules are the speaker's own either way: the same draws on its columns where the procedure's
    # altitudes remove nothing (the first step's runway and approach are sampled before any altitude word)
    on, off = np.ones(1, dtype=bool), np.zeros(1, dtype=bool)
    assert bare.speak(on, off)[0, [RUNWAY, APPROACH]].tolist() == masked.speak(on, off)[0, [RUNWAY, APPROACH]].tolist()
