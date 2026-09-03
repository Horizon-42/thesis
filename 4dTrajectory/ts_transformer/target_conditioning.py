"""Input-only target conditioning: the runway target as DATA the model reads, not geometry.

Under the threshold-anchored frames the model knows its destination only geometrically —
the target IS the chart origin. Under ``airport-enu`` the chart is shared by every runway
of the airport and the target is an ordinary point in it, so the model has no way to tell
which runway it is flying to unless it is told. This module is the telling: five constant
channels appended to the observed history AFTER normalization —

    e_tgt, n_tgt, u_tgt     the target's chart position (``FlightSeries.target_chart``),
                            standardised with the POSITION channels' statistics so that a
                            position token and the target token are commensurate and
                            ``position − target`` is distance-to-go in normalized units
    cos_psi_rwy, sin_psi_rwy  the runway course (math-ENU), already unit scale

broadcast over the ``seq_len`` history steps. They are INPUT-ONLY: ``channels.CHANNELS``
stays the bidirectional contract (``states_from_channels`` would otherwise try to predict
the conditioning, and the tuple indexes the normalizer and the checkpoint), and the model
is only ever asked for those six. iTransformer consumes the extra columns as covariate
variate tokens — the vendored ``x_mark_enc`` path, whose projector output is filtered back
to the state tokens — so the state channels can attend to the target; a channel-
independent backbone (PatchTST) cannot route a conditioning token to a state token at
all, which is why ``TSConfig`` refuses the combination.

Under a threshold-anchored frame ``target_chart`` is identically zero, so the conditioning
collapses to one constant vector per run (``(0 − μ)/σ`` plus the course) — a free control
for the mechanism: if it moves any metric there, the plumbing is wrong, not the science.
"""

from __future__ import annotations

import math

import numpy as np

TARGET_CONDITIONING_NONE = "none"
TARGET_CONDITIONING_CHANNELS = "channels"
TARGET_CONDITIONINGS = (TARGET_CONDITIONING_NONE, TARGET_CONDITIONING_CHANNELS)

# Order is load-bearing like channels.CHANNELS: it is serialised into every checkpoint
# (``input_channels``) and ``train.load_checkpoint`` refuses a mismatch.
CONDITIONING_CHANNELS: tuple[str, ...] = (
    "e_tgt", "n_tgt", "u_tgt", "cos_psi_rwy", "sin_psi_rwy"
)


def conditioning_channel_names(target_conditioning: str) -> tuple[str, ...]:
    """The input-only channels a mode appends after ``channels.CHANNELS``."""
    if target_conditioning == TARGET_CONDITIONING_NONE:
        return ()
    if target_conditioning == TARGET_CONDITIONING_CHANNELS:
        return CONDITIONING_CHANNELS
    raise ValueError(
        f"unknown target_conditioning {target_conditioning!r}; expected one of "
        f"{TARGET_CONDITIONINGS}"
    )


def conditioning_vector(
    target_chart: np.ndarray,
    runway_heading_rad: float,
    *,
    position_mean: np.ndarray,
    position_std: np.ndarray,
) -> np.ndarray:
    """One flight's constant conditioning row, in the model's normalized input space."""
    target_chart = np.asarray(target_chart, dtype=np.float64)
    if target_chart.shape != (3,):
        raise ValueError(f"target_chart must be (e, n, u), got shape {target_chart.shape}")
    normalized = (target_chart - np.asarray(position_mean)) / np.asarray(position_std)
    return np.concatenate([
        normalized,
        [math.cos(runway_heading_rad), math.sin(runway_heading_rad)],
    ]).astype(np.float32)


def conditioned_history(history: np.ndarray, vector: np.ndarray | None) -> np.ndarray:
    """``history[L, C]`` -> ``[L, C + K]`` with the conditioning broadcast over time."""
    if vector is None:
        return history
    if history.ndim != 2:
        raise ValueError(f"history must be [L, C], got shape {history.shape}")
    constant = np.broadcast_to(
        np.asarray(vector, dtype=history.dtype), (len(history), len(vector))
    )
    return np.concatenate([history, constant], axis=1)
