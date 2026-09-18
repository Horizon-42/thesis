"""The tokenizer (`manoeuvre/tokenizer.py`, plan §2.4): FSQ rounds to a grid the codes index
exactly and passes a gradient through; the learned tokenizer and the command vocabulary share
one interface; a codebook artefact round-trips, is frozen, and is bound by its sha."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ts_transformer.manoeuvre import tokenizer as tok
from ts_transformer.manoeuvre.segments import segment_offsets_s, segment_rows

DT_S = 2.0


def _segment(segment_s: float = 60.0, *, speed: float = 70.0, rate_dps: float = 0.0,
             descent: float = 0.0, accel: float = 0.0) -> np.ndarray:
    """One segment's rows in the start frame: a turn at ``rate_dps``, a vertical rate and a
    longitudinal acceleration, integrated on the row grid."""
    t = segment_offsets_s(segment_s, DT_S)
    course = np.radians(rate_dps) * t
    v = speed + accel * t
    x = np.concatenate(([0.0], np.cumsum(v[:-1] * np.cos(course[:-1]) * DT_S)))
    y = np.concatenate(([0.0], np.cumsum(v[:-1] * np.sin(course[:-1]) * DT_S)))
    values = np.column_stack([x + 1000.0, y - 500.0, descent * t + 300.0, v * np.cos(course), v * np.sin(course), np.full_like(t, descent)])
    return segment_rows(t, values, 0.0, segment_s, DT_S)


def _state(batch: int) -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.tensor(rng.normal(size=(batch, 6)) * np.array([8000, 8000, 600, 60, 60, 3]), dtype=torch.float32)


# ── FSQ ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("levels", [(4, 4), (8, 4), (4, 4, 4), (8, 4, 4), (4, 4, 4, 4), (5, 3), (7,)])
def test_fsq_rounds_to_its_grid_and_the_codes_index_it_exactly(levels):
    fsq = tok.FSQ(levels)
    assert fsq.code_count == math.prod(levels) == tok.fsq_code_count(levels)
    codes = torch.arange(fsq.code_count)
    z = fsq.codes_to_z(codes)
    assert z.shape == (fsq.code_count, len(levels))
    assert torch.equal(fsq.z_to_codes(z), codes)                # exact inverse on every code
    assert float(z.min()) >= -1.0 and float(z.max()) <= 1.0
    for d, level in enumerate(levels):
        assert len(torch.unique(z[:, d])) == level               # every level of every dimension is a code
    # quantizing pre-activations spread over the bound's active range lands ON the grid and
    # covers the whole code range (every level of every dimension has ≥ ~20 % mass under
    # U(-3, 3), so 20 000 draws reach every one of ≤ 256 codes)
    torch.manual_seed(0)
    zq, cq = fsq.quantize(torch.rand(20_000, len(levels)) * 6.0 - 3.0)
    assert torch.allclose(fsq.codes_to_z(cq), zq)
    assert set(cq.tolist()) == set(range(fsq.code_count))


def test_fsq_passes_a_straight_through_gradient_and_refuses_two_levels():
    fsq = tok.FSQ((4, 4))
    h = torch.zeros(3, 2, requires_grad=True)
    z, _codes = fsq.quantize(h)
    z.sum().backward()
    assert h.grad is not None and bool((h.grad != 0).all())
    with pytest.raises(ValueError, match="levels"):
        tok.FSQ((2, 4))
    with pytest.raises(ValueError, match="levels"):
        tok.FSQ(())
    with pytest.raises(ValueError, match="codes"):
        fsq.codes_to_z(torch.tensor([16]))


# ── the learned tokenizer ────────────────────────────────────────────────────

def test_the_learned_tokenizer_encodes_physical_rows_to_a_code_and_its_grid_vector():
    tokenizer = tok.tokenizer_for("learned", levels=(4, 4), segment_s=60.0, dt_s=DT_S)
    assert tokenizer.rows == 31 and tokenizer.code_count == 16 and tokenizer.z_dim == 2 and tokenizer.trainable
    segment = torch.tensor(np.stack([_segment(rate_dps=1.5), _segment(rate_dps=-1.5), _segment(descent=-3.0)]), dtype=torch.float32)
    z, codes = tokenizer(segment, _state(3))
    assert z.shape == (3, 2) and codes.shape == (3,)
    assert torch.allclose(tokenizer.codes_to_z(codes), z)
    assert bool((codes >= 0).all()) and bool((codes < 16).all())
    # deterministic, and the gradient of z reaches the encoder (joint training's whole point)
    z2, codes2 = tokenizer(segment, _state(3))
    assert torch.equal(codes, codes2) and torch.allclose(z, z2)
    z.sum().backward()
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in tokenizer.encoder.parameters())
    with pytest.raises(ValueError, match="segment rows"):
        tokenizer(segment[:, :30], _state(3))


def test_the_learned_tokenizer_is_small():
    tokenizer = tok.tokenizer_for("learned", levels=(8, 4, 4), segment_s=120.0, dt_s=DT_S)
    assert sum(p.numel() for p in tokenizer.parameters()) < 200_000


# ── the command vocabulary ───────────────────────────────────────────────────

def test_the_command_vocabulary_reads_turns_descents_and_decelerations_by_rule():
    assert tok.COMMAND_VOCABULARY_SIZE == 45
    cases = {
        _segment().tobytes(): (0, 0, 0),                                   # straight, level, hold
        _segment(rate_dps=0.5).tobytes(): (1, 0, 0),                       # 30° left over 60 s
        _segment(rate_dps=-0.5).tobytes(): (2, 0, 0),                      # 30° right
        _segment(rate_dps=1.5).tobytes(): (3, 0, 0),                       # 90° left
        _segment(rate_dps=-2.5).tobytes(): (4, 0, 0),                      # 150° right
        _segment(descent=-3.5).tobytes(): (0, 1, 0),                       # descending
        _segment(descent=2.0).tobytes(): (0, 2, 0),                        # climbing
        _segment(accel=-0.1).tobytes(): (0, 0, 1),                         # decelerating
        _segment(accel=0.1).tobytes(): (0, 0, 2),                          # accelerating
        _segment(rate_dps=-1.0, descent=-3.0, accel=-0.08).tobytes(): (4, 1, 1),
    }
    rows = 31
    for raw, expected in cases.items():
        segment = np.frombuffer(raw, dtype=np.float64).reshape(rows, 6)
        assert tok.command_classes(segment, 60.0) == expected
    # the same segment length changes nothing about a rate class: 120 s at the same descent rate
    assert tok.command_classes(_segment(120.0, descent=-3.5), 120.0)[1] == 1
    assert tok.command_classes(_segment(120.0, descent=-0.5), 120.0)[1] == 0


def test_the_command_vocabulary_shares_the_tokenizer_interface_and_has_no_parameters():
    tokenizer = tok.tokenizer_for("command-vocabulary", levels=(), segment_s=60.0, dt_s=DT_S)
    assert not tokenizer.trainable and tokenizer.code_count == 45 and tokenizer.z_dim == 45
    assert sum(p.numel() for p in tokenizer.parameters()) == 0
    segment = torch.tensor(np.stack([_segment(), _segment(rate_dps=1.5, descent=-3.0)]), dtype=torch.float32)
    z, codes = tokenizer(segment, _state(2))
    assert z.shape == (2, 45) and codes.tolist() == [tok.command_code(0, 0, 0), tok.command_code(3, 1, 0)]
    assert torch.equal(tokenizer.codes_to_z(codes), z) and torch.equal(tokenizer.z_to_codes(z), codes)
    assert tok.command_label(codes[1].item()) == "left-large/descend/hold"
    for code in range(45):
        assert tok.command_label(code).count("/") == 2
    with pytest.raises(ValueError, match="levels"):
        tok.tokenizer_for("command-vocabulary", levels=(4, 4), segment_s=60.0, dt_s=DT_S)
    with pytest.raises(ValueError, match="unknown"):
        tok.tokenizer_for("k-means", levels=(), segment_s=60.0, dt_s=DT_S)


# ── the codebook artefact ────────────────────────────────────────────────────

def test_a_codebook_round_trips_frozen_and_is_bound_by_its_sha(tmp_path):
    torch.manual_seed(1)
    tokenizer = tok.tokenizer_for("learned", levels=(8, 4), segment_s=30.0, dt_s=DT_S)
    segment = np.stack([_segment(30.0, rate_dps=2.0), _segment(30.0, descent=-4.0)]).astype(np.float32)
    state = _state(2).numpy()
    with torch.no_grad():
        z_live, codes_live = tokenizer(torch.from_numpy(segment), torch.from_numpy(state))
    identity = {"schema": "ts-arrival-data-v4-eligible-set", "digest": "b" * 64}
    codebook = tok.write_codebook(
        tmp_path / "cb", tokenizer, segment_s=30.0, dt_s=DT_S, data_identity=identity,
        source={"checkpoint_sha256": "c" * 64, "run": "manoeuvre_tok/K32_s60"},
    )
    assert codebook.kind == "learned" and codebook.levels == (8, 4) and codebook.rows == 16
    assert codebook.segment_s == 30.0 and codebook.code_count == 32 and codebook.z_dim == 2
    assert codebook.data_identity == identity and len(codebook.sha256) == 64
    assert not any(p.requires_grad for p in codebook.tokenizer.parameters())
    codes, z = codebook.encode(segment, state)
    assert codes.tolist() == codes_live.tolist() and np.allclose(z, z_live.numpy())
    single_code, single_z = codebook.encode(segment[1], state[1])
    assert int(single_code) == int(codes[1]) and np.allclose(single_z, z[1])
    # a second load is the same identity; an existing directory is never overwritten
    again = tok.load_codebook(tmp_path / "cb")
    assert again.sha256 == codebook.sha256
    with pytest.raises(FileExistsError):
        tok.write_codebook(tmp_path / "cb", tokenizer, segment_s=30.0, dt_s=DT_S, data_identity={}, source={})
    # a tokenizer sized for another segment cannot be written as this segment's codebook
    with pytest.raises(ValueError, match="rows"):
        tok.write_codebook(tmp_path / "cb2", tokenizer, segment_s=60.0, dt_s=DT_S, data_identity={}, source={})
    # tampered weights are refused on load
    weights = tmp_path / "cb" / tok.CODEBOOK_WEIGHTS
    weights.write_bytes(weights.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        tok.load_codebook(tmp_path / "cb")


def test_a_command_vocabulary_codebook_is_an_artefact_too(tmp_path):
    tokenizer = tok.tokenizer_for("command-vocabulary", levels=(), segment_s=60.0, dt_s=DT_S)
    codebook = tok.write_codebook(tmp_path / "cv", tokenizer, segment_s=60.0, dt_s=DT_S, data_identity={}, source={})
    assert codebook.kind == "command-vocabulary" and codebook.code_count == 45 and codebook.levels == ()
    codes, z = codebook.encode(np.stack([_segment(rate_dps=1.5)]).astype(np.float32), _state(1).numpy())
    assert codes.tolist() == [tok.command_code(3, 0, 0)] and z.shape == (1, 45)
