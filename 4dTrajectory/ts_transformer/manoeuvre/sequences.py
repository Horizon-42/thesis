"""A flight as its code sequence (plan §2.5): from the first prediction (the fixed anchor) forward, one
code per full segment through a FROZEN codebook, with the state token at every segment
boundary — the prior's training rows and the protocol-C history.

    x_0            c_1, x_1      c_2, x_2   …   c_T, x_T      landed_fraction
    (the anchor)   (segment 1 and the state at its end) …      (what remains after the last
                                                               full segment, in segments)

* Codes come from `segments.segment_start_times` (FULL segments only) and the codebook's
  encoder over the TRUTH's rows (`segments.truth_segment_rows`) and the state at each start —
  reading the future is the tokenizer's job; the prior is trained to predict what it will say.
* The state token (`state_token`) is the chart row at a boundary as six O(1) features: position
  (e, n, u) in the runway's threshold chart, ground speed, and the course as (cos, sin). P2–P3
  run runway-KNOWN (T1), so the runway's own chart is the frame; P4 moves the token to the
  airport frame when the runway becomes an output (plan §2.5, `data/coordinate_frames.py`).
* ``landed_fraction`` = the truth remaining after the last full segment, in segments ∈ [0, 1):
  the prior's ``LANDED`` position carries it as the arrival time inside the last segment.
* A ROLLED sequence (plan §2.7, the closed-loop training's first step): the input codes and
  states are what the executor FLEW (a protocol-C lockstep's legs tokenised back), the targets
  are the TRUTH's codes at the same absolute time index — ``target_codes`` — and the landing
  label sits where the truth has no full segment left (`rolled_sequence`). A truth sequence's
  targets are its own codes (``target_codes`` None).
* `operational_day_of` / `split_by_operational_day`: the day split (T4, 09Z cut) the runway
  token (P4) and the multi-aircraft layer (P5) are judged on. P2 trains and reads the prior on
  the executor's OWN flight split (the development cohort) so the lockstep pairs the same val
  flights and the prior has seen none of them — the decision is in the plan's §6.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from typing import Sequence

import numpy as np

from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.data.runway_context import operational_day
from ts_transformer.manoeuvre.segments import segment_start_times, state_row, truth_segment_rows
from ts_transformer.manoeuvre.tokenizer import Codebook

#: The state token's features and scales (one definition; the prior's `state_embedding` reads
#: this width): the 25 km slice, its height, the approach speed; the course is a unit vector.
STATE_TOKEN_FEATURES = ("e", "n", "u", "ground_speed", "cos_course", "sin_course")
STATE_TOKEN_SCALE = np.array([20000.0, 20000.0, 2000.0, 100.0, 1.0, 1.0], dtype=np.float32)


def state_token(row: np.ndarray) -> np.ndarray:
    """The six scaled features of one chart row."""
    row = np.asarray(row, dtype=np.float64)
    edot, ndot = row[VELOCITY_IDX[0]], row[VELOCITY_IDX[1]]
    speed = math.hypot(edot, ndot)
    course = math.atan2(ndot, edot) if speed > 0.0 else 0.0
    features = np.array([*row[list(POSITION_IDX)], speed, math.cos(course), math.sin(course)], dtype=np.float32)
    return features / STATE_TOKEN_SCALE


@dataclass(frozen=True)
class CodeSequence:
    """One flight's sequence: ``codes`` ``[T]`` and ``z`` ``[T, Z]`` for the T full segments from
    the anchor, ``states`` ``[T + 1, 6]`` (x_0 at the anchor, x_t at the end of segment t),
    ``start_times`` ``[T]`` on the flight's clock, and what remains."""

    dataset_id: str
    flight_id: str
    anchor: int
    segment_s: float
    start_times: np.ndarray
    codes: np.ndarray
    z: np.ndarray
    states: np.ndarray
    landed_fraction: float
    typecode: str
    runway_course_rad: float
    #: A rolled sequence's targets: the truth's codes by absolute time index (``[T_truth]``);
    #: None = the sequence is the truth's and its targets are its own codes.
    target_codes: np.ndarray | None = None

    def __post_init__(self) -> None:
        count = len(self.start_times)
        if self.codes.shape != (count,) or self.z.shape[0] != count or self.states.shape != (count + 1, len(STATE_TOKEN_FEATURES)):
            raise ValueError(
                f"{self.dataset_id}: {count} segments need [{count}] codes, [{count}, Z] z and "
                f"[{count + 1}, {len(STATE_TOKEN_FEATURES)}] states, got {self.codes.shape}, {self.z.shape}, {self.states.shape}"
            )
        if not 0.0 <= self.landed_fraction < 1.0:
            raise ValueError(f"{self.dataset_id}: landed_fraction is in [0, 1), got {self.landed_fraction!r}")
        if self.target_codes is not None and self.target_codes.ndim != 1:
            raise ValueError(f"{self.dataset_id}: target_codes is a [T] vector")

    @property
    def length(self) -> int:
        return len(self.codes)

    @property
    def targets(self) -> np.ndarray:
        """The next-code targets by position: the truth's codes (time-indexed) for a rolled
        sequence, the sequence's own codes otherwise."""
        return self.codes if self.target_codes is None else self.target_codes

    @property
    def rolled(self) -> bool:
        return self.target_codes is not None


def _boundaries(series: FlightSeries, anchor: int, segment_s: float) -> tuple[np.ndarray, float]:
    starts = segment_start_times(series, anchor, segment_s)
    end = float(series.supervision_times[-1])
    last_end = float(series.times[anchor]) + segment_s * len(starts)
    remaining = max(end - last_end, 0.0)
    if remaining >= segment_s:
        raise ValueError(f"{series.dataset_id}: {remaining:.3f} s remain after the last full segment — the cut missed one")
    return starts, remaining / segment_s


def flight_sequences(
    series: Sequence[FlightSeries], codebook: Codebook, anchor: int, *, batch_size: int = 512
) -> list[CodeSequence]:
    """Every flight's sequence, the segments of all flights encoded in batches of ``batch_size``
    through the codebook (one pass, in cohort order)."""
    segment_s, dt_s = codebook.segment_s, codebook.dt_s
    plans = []
    rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    for item in series:
        starts, landed_fraction = _boundaries(item, anchor, segment_s)
        for start in starts:
            rows.append(truth_segment_rows(item, float(start), segment_s, dt_s).astype(np.float32))
            state_rows.append(state_row(item, float(start)).astype(np.float32))
        plans.append((item, starts, landed_fraction))
    codes = np.empty(len(rows), dtype=np.int64)
    z = np.empty((len(rows), codebook.z_dim), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        chunk_codes, chunk_z = codebook.encode(np.stack(rows[start : start + batch_size]), np.stack(state_rows[start : start + batch_size]))
        codes[start : start + batch_size] = chunk_codes
        z[start : start + batch_size] = chunk_z
    out: list[CodeSequence] = []
    cursor = 0
    for item, starts, landed_fraction in plans:
        count = len(starts)
        boundaries = np.concatenate(([float(item.times[anchor])], starts + segment_s))
        out.append(CodeSequence(
            dataset_id=item.dataset_id, flight_id=item.flight_id, anchor=anchor, segment_s=segment_s,
            start_times=starts, codes=codes[cursor : cursor + count].copy(), z=z[cursor : cursor + count].copy(),
            states=np.stack([state_token(state_row(item, float(time))) for time in boundaries]),
            landed_fraction=landed_fraction, typecode=str(item.scenario.aircraft.code),
            runway_course_rad=float(item.scenario.target.psi),
        ))
        cursor += count
    return out


def flight_sequence(series: FlightSeries, codebook: Codebook, anchor: int) -> CodeSequence:
    return flight_sequences([series], codebook, anchor)[0]


def rolled_sequence(truth: CodeSequence, flown_codes: Sequence[int], flown_states: Sequence[np.ndarray]) -> CodeSequence:
    """The rolled counterpart of ``truth``: the flown codes and boundary states as the input,
    the truth's codes as the time-indexed targets (plan §2.7, v2 §8's decision), the truth's
    landing where its full segments end. ``flown_states`` holds x_0 and one state per flown
    leg; a flight that flew more rounds than the truth has segments is cut to the truth's
    length + 1 positions (past that, every target is the landing)."""
    codes = np.asarray(flown_codes, dtype=np.int64)
    states = np.asarray(flown_states, dtype=np.float32)
    if states.shape[0] != len(codes) + 1:
        raise ValueError(f"{truth.dataset_id}: {len(codes)} flown codes need {len(codes) + 1} boundary states, got {states.shape[0]}")
    keep = min(len(codes), truth.length + 1)
    first = float(truth.start_times[0]) if truth.length else np.nan     # a truth with no full segment has no start
    return CodeSequence(
        dataset_id=truth.dataset_id, flight_id=truth.flight_id, anchor=truth.anchor, segment_s=truth.segment_s,
        start_times=first + truth.segment_s * np.arange(keep, dtype=np.float64),   # the flown rounds' starts
        codes=codes[:keep], z=np.zeros((keep, truth.z.shape[1]), dtype=np.float32), states=states[: keep + 1],
        landed_fraction=truth.landed_fraction, typecode=truth.typecode, runway_course_rad=truth.runway_course_rad,
        target_codes=truth.codes.copy(),
    )


def continuous_targets(
    series: Sequence[FlightSeries], sequences: Sequence[CodeSequence], codebook: Codebook, *, batch_size: int = 512
) -> list[np.ndarray]:
    """The UNROUNDED encoder coordinates ``[T, Z]`` of every sequence's segments (the continuous
    prior's targets, `Codebook.encode_continuous`), one array per sequence, in the same order."""
    segment_s, dt_s = codebook.segment_s, codebook.dt_s
    rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    for item, sequence in zip(series, sequences, strict=True):
        if sequence.dataset_id != item.dataset_id:
            raise ValueError(f"sequence {sequence.dataset_id} is not {item.dataset_id}'s")
        for start in sequence.start_times:
            rows.append(truth_segment_rows(item, float(start), segment_s, dt_s).astype(np.float32))
            state_rows.append(state_row(item, float(start)).astype(np.float32))
    vectors = np.empty((len(rows), codebook.z_dim), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        vectors[start : start + batch_size] = codebook.encode_continuous(
            np.stack(rows[start : start + batch_size]), np.stack(state_rows[start : start + batch_size])
        )
    out, cursor = [], 0
    for sequence in sequences:
        out.append(vectors[cursor : cursor + sequence.length].copy())
        cursor += sequence.length
    return out


# ── the operating-day split (T4) ─────────────────────────────────────────────

def operational_day_of(series: FlightSeries) -> str:
    """The operating day (09Z cut, `data/runway_context.operational_day`) of the flight's landing."""
    raw = str(series.scenario.source["landing_time_utc"])
    landed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if landed.tzinfo is None:
        landed = landed.replace(tzinfo=timezone.utc)
    return operational_day(landed.astimezone(timezone.utc).replace(tzinfo=None))


def split_by_operational_day(
    series: Sequence[FlightSeries], *, val_fraction: float, seed: int
) -> tuple[list[FlightSeries], list[FlightSeries]]:
    """``(train, val)`` with every flight of one operating day on the same side: a day goes to
    val when the sha256 of ``seed:day`` falls under ``val_fraction`` — a pure function of the
    day, never of the flight list's order (the rule `data/splits.split_by_flight` follows)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction is in (0, 1), got {val_fraction!r}")
    train, val = [], []
    for item in series:
        digest = hashlib.sha256(f"{seed}:{operational_day_of(item)}".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        (val if draw < val_fraction else train).append(item)
    return train, val


__all__ = [
    "STATE_TOKEN_FEATURES", "STATE_TOKEN_SCALE", "CodeSequence", "continuous_targets", "flight_sequence",
    "flight_sequences", "operational_day_of", "rolled_sequence", "split_by_operational_day", "state_token",
]
