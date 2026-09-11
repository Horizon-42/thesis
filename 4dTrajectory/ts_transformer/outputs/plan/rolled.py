"""Rolled windows (design v5.2, §9 step 3(e)): the plan head's input windows along a
lockstep flight, each labelled with the truth's instruction in force AT THAT STATE — the
closed-loop training set a receding-horizon head needs, and the sampler that mixes it
into training.

Why: the head is asked every 30 s on the window of its OWN flown rows (`rolled_history`),
and trained on observed windows only it jittered from step to step and sent 29 % of
vectored flights to the time cap (§12.5). Three parts, one contract:

- `record_lockstep` flies a group in lockstep under any policy (the truth's — the ceiling's
  own windows — or a head's, the expert labelling the learner's states) and records, at
  every step, each flight's window and the target vector the truth defines there — the
  truth's policy read at the aircraft's pose (`labels.TruthExpert`, the one definition
  `targets_from_labels` reads at an observed anchor too), as `targets_at` lays it out;
- `RolledWindowTable` is the file (`.npz`, the provenance inside), bound to the window
  contract (L, dt, channels, the target contract) and to the cohort it was flown over
  (`require_cover`: no partial mode);
- `RolledDraw` is the per-flight per-epoch coin that replaces the observed anchor draw
  by one of the flight's rolled windows at `plan_rolled_share`, from the same digest
  discipline as the anchor draw (`dataset.RemainingPathUniformAnchorTrajectoryWindows`),
  salted apart so the draws it passes over are the ones the observed law makes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np
import torch

from ts_transformer.config import TSConfig
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.outputs.plan.extractors import PlanLabels
from ts_transformer.outputs.plan.forecast import (
    LOCKSTEP_S,
    FlightState,
    LegOrder,
    LockstepPolicy,
    RolledFlight,
    fly_lockstep,
    on_final,
    rolled_history,
)
from ts_transformer.outputs.plan.labels import (
    CONTEXT_NEXT_IS_JOIN,
    CONTEXT_ROLLED,
    CONTEXT_SERIES,
    CONTEXT_TARGETS,
    CONTEXT_VALID,
    PLAN_TARGET_CONTRACT,
    TARGETS,
    PlanTargets,
    TruthExpert,
    targets_at,
)

#: The table's schema: the runner stamps it, the loader checks it.
ROLLED_WINDOWS_SCHEMA = "ts-plan-rolled-windows-v1"
#: The draw law's salt (`RolledDraw`): a different law is a different version.
ROLLED_SAMPLING_VERSION = "per-flight-hash-v1-plan-rolled-share"
#: Whose flown states a table holds: the truth policy's (the lockstep oracle: closed-loop
#: windows along the truth's route) or a plan checkpoint's (the head's own states, the
#: truth labelling them — DAgger's round).
POLICY_TRUTH = "truth"
POLICY_MODEL = "model"
POLICIES = (POLICY_TRUTH, POLICY_MODEL)
#: Where a checkpoint records which table taught it (`train.fit_model`).
METADATA_KEY = "plan_rolled_windows"


# ── the truth at a flown state ───────────────────────────────────────────────

@dataclass(frozen=True)
class RolledSample:
    """One step of one flight: the head's input window there and the truth's targets."""

    dataset_id: str
    step: int
    elapsed_s: float
    on_final: bool
    window: np.ndarray        # [L, C] physical channels, float32
    targets: PlanTargets


def record_lockstep(
    states: Sequence[FlightState],
    labels: Sequence[PlanLabels],
    config: TSConfig,
    *,
    policy: LockstepPolicy,
    step_s: float = LOCKSTEP_S,
    time_caps_s: Sequence[float] | None = None,
    device: torch.device | None = None,
) -> tuple[list[RolledFlight], list[RolledSample]]:
    """Fly ``states`` in lockstep under ``policy`` and record, at EVERY step, each flight's
    window (`rolled_history`: the observed track to the anchor continued by the rows flown)
    and the target vector the truth defines at that state — the truth's policy read at
    the aircraft's pose (`labels.TruthExpert`: the instruction in force there, the
    operating parameters from there), as `targets_at` reads it."""
    experts = {
        id(state): TruthExpert(lab, state.series, state.anchor, state.skeleton)
        for state, lab in zip(states, labels, strict=True)
    }
    samples: list[RolledSample] = []

    def recording(active: Sequence[FlightState]) -> Sequence[LegOrder]:
        for state in active:
            window = rolled_history(state.series, state.anchor, state.chunks, config)
            current = state.current
            instruction, operating = experts[id(state)].order_at(current.e, current.n, current.heading_rad, current.speed_mps)
            samples.append(RolledSample(
                dataset_id=state.series.dataset_id, step=int(state.steps), elapsed_s=float(state.elapsed_s),
                on_final=bool(on_final(state)), window=window.astype(np.float32),
                targets=targets_at(operating, instruction, state.current.e, state.current.n, state.skeleton),
            ))
        return policy(active)

    flights = fly_lockstep(states, config, policy=recording, step_s=step_s, time_caps_s=time_caps_s, device=device)
    return flights, samples


# ── the table ─────────────────────────────────────────────────────────────────

@dataclass
class RolledWindowTable:
    """A rolled-window table as loaded: every sample's window and targets, the flights they
    belong to, and the header the generator wrote (the window contract it was flown
    under, the policy, the splits, the checkpoint that named the cohort)."""

    metadata_key: ClassVar[str] = METADATA_KEY

    path: Path
    sha256: str
    header: dict
    windows: np.ndarray        # [S, L, C] float32, physical channels
    targets: np.ndarray        # [S, P] float32
    valid: np.ndarray          # [S, P] float32
    next_is_join: np.ndarray   # [S] float32
    flight: np.ndarray         # [S] int64, an index into ``dataset_ids``
    step: np.ndarray           # [S] int64
    elapsed_s: np.ndarray      # [S] float32
    on_final: np.ndarray       # [S] float32
    dataset_ids: tuple[str, ...]
    _rows_by_flight: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        order = np.lexsort((self.step, self.flight))
        for flight_index, rows in _group_sorted(self.flight[order], order):
            self._rows_by_flight[self.dataset_ids[int(flight_index)]] = rows

    @property
    def samples(self) -> int:
        return int(len(self.flight))

    @property
    def flights(self) -> int:
        return len(self._rows_by_flight)

    @property
    def provenance(self) -> dict[str, object]:
        """What a checkpoint records so it can say which table taught it."""
        return {
            "path": str(self.path), "sha256": self.sha256, "schema": self.header["schema"],
            "policy": self.header["policy"], "lockstep_s": self.header["lockstep_s"],
            "samples": self.samples, "flights": self.flights, "splits": self.header["splits"],
            "checkpoint": self.header["checkpoint"], "extended_from": self.header.get("extended_from"),
        }

    def samples_for(self, dataset_id: str) -> np.ndarray:
        """The flight's sample rows, in step order (`KeyError` for a flight the table
        lacks — `require_cover` has refused such a cohort before anything draws)."""
        return self._rows_by_flight[dataset_id]

    def context_row(self, k: int, series_index: int) -> dict[str, np.ndarray]:
        """Sample ``k`` as the batch context row `PlanContext.row` builds for an observed
        anchor, flagged rolled."""
        return {
            CONTEXT_TARGETS: self.targets[k], CONTEXT_VALID: self.valid[k],
            CONTEXT_NEXT_IS_JOIN: np.array(float(self.next_is_join[k]), dtype=np.float32),
            CONTEXT_SERIES: np.array(int(series_index), dtype=np.int64),
            CONTEXT_ROLLED: np.array(1.0, dtype=np.float32),
        }

    def require_cover(self, series: Sequence[FlightSeries], config: TSConfig, *, what: str) -> None:
        """Refuse unless this table was flown under this run's window contract and covers
        every flight of ``series``. There is no partial mode: a head trained on rolled
        windows for some flights and on observed ones for the rest is a mixture nobody
        asked for, and the epoch's share would report a number about half the cohort.
        The airports come first, as the fitted teacher's check orders them: a table from
        another airport fails every later check and "covers 0 of N" does not name it."""
        airports = {item.airport for item in series}
        foreign = sorted(airports - set(self.header["airports"]))
        if foreign:
            raise ValueError(
                f"{self.path}: the rolled windows were flown over {sorted(self.header['airports'])}, "
                f"{what} covers {sorted(airports)} — {foreign} is outside it"
            )
        contract = {
            "seq_len": int(config.seq_len), "dt_s": float(config.dt_s), "channels": list(config.channels),
            "target_contract": PLAN_TARGET_CONTRACT,
        }
        for key, wanted in contract.items():
            if self.header[key] != wanted:
                raise ValueError(
                    f"{self.path}: the rolled windows carry {key}={self.header[key]!r}, this run "
                    f"{key}={wanted!r} — a window is only the head's input under its own contract"
                )
        missing = [item.dataset_id for item in series if item.dataset_id not in self._rows_by_flight]
        if missing:
            raise ValueError(
                f"{self.path}: the rolled windows cover {len(series) - len(missing)} of {len(series)} flights "
                f"of {what}; first missing: {missing[0]!r} (generate the table over this checkpoint's split)"
            )


def _group_sorted(keys: np.ndarray, rows: np.ndarray):
    """(key, rows) per run of equal ``keys`` (sorted), ``rows`` aligned with ``keys``."""
    if not len(keys):
        return
    edges = np.flatnonzero(np.diff(keys)) + 1
    for start, end in zip(np.concatenate(([0], edges)), np.concatenate((edges, [len(keys)])), strict=True):
        yield keys[start], rows[start:end]


def rolled_table_header(
    config: TSConfig, *, policy: str, lockstep_s: float, airports: Sequence[str], splits: dict[str, int],
    checkpoint: dict[str, str], generated_at: str, wall_s: float,
) -> dict:
    """The header a generator writes: the window contract the table was flown under, and
    what it was flown over."""
    return {
        "schema": ROLLED_WINDOWS_SCHEMA, "policy": policy, "lockstep_s": float(lockstep_s),
        "seq_len": int(config.seq_len), "dt_s": float(config.dt_s), "channels": list(config.channels),
        "target_contract": PLAN_TARGET_CONTRACT, "targets": list(TARGETS),
        "airports": sorted(set(airports)), "splits": dict(splits), "checkpoint": dict(checkpoint),
        "generated_at": generated_at, "wall_s": float(wall_s),
    }


def write_rolled_windows(
    path: str | Path, samples: Sequence[RolledSample], header: dict, *, extend: RolledWindowTable | None = None,
) -> RolledWindowTable:
    """Write a table (one ``.npz``, the header inside) and read it back. With ``extend``,
    that table's samples come first and its provenance is recorded as ``extended_from`` —
    the aggregation a DAgger round trains on. The header's ``splits`` count the flights
    of the NEW samples per split; the extended table keeps its own inside its record."""
    if not samples and extend is None:
        raise ValueError("a rolled-window table needs at least one sample")
    ids: list[str] = [] if extend is None else list(extend.dataset_ids)
    index = {dataset_id: i for i, dataset_id in enumerate(ids)}
    for sample in samples:
        if sample.dataset_id not in index:
            index[sample.dataset_id] = len(ids)
            ids.append(sample.dataset_id)
    new = {
        "windows": np.stack([s.window for s in samples]).astype(np.float32) if samples else np.zeros((0, header["seq_len"], len(header["channels"])), np.float32),
        "targets": np.stack([s.targets.values for s in samples]).astype(np.float32) if samples else np.zeros((0, len(TARGETS)), np.float32),
        "valid": np.stack([s.targets.valid for s in samples]).astype(np.float32) if samples else np.zeros((0, len(TARGETS)), np.float32),
        "next_is_join": np.array([float(s.targets.next_is_join) for s in samples], dtype=np.float32),
        "flight": np.array([index[s.dataset_id] for s in samples], dtype=np.int64),
        "step": np.array([s.step for s in samples], dtype=np.int64),
        "elapsed_s": np.array([s.elapsed_s for s in samples], dtype=np.float32),
        "on_final": np.array([float(s.on_final) for s in samples], dtype=np.float32),
    }
    if extend is not None:
        # the same window contract and the same re-ask period, or the two sets of samples
        # are two different inputs under one name; the cohorts are unioned
        for key in ("seq_len", "dt_s", "channels", "target_contract", "lockstep_s"):
            if extend.header[key] != header[key]:
                raise ValueError(
                    f"{extend.path}: cannot extend a table with {key}={extend.header[key]!r} by one with "
                    f"{key}={header[key]!r}"
                )
        new = {key: np.concatenate([getattr(extend, key), value]) for key, value in new.items()}
        header = {
            **header, "airports": sorted(set(header["airports"]) | set(extend.header["airports"])),
            "extended_from": extend.provenance,
        }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, dataset_ids=np.array(ids, dtype=str), header=np.array(json.dumps(header)), **new)
    tmp.replace(out)
    return load_rolled_windows(out)


def load_rolled_windows(path: str | Path) -> RolledWindowTable:
    """Read a table; the schema is checked here, the per-cohort contract in `require_cover`."""
    table_path = Path(path)
    raw = table_path.read_bytes()
    with np.load(table_path, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
        if header.get("schema") != ROLLED_WINDOWS_SCHEMA:
            raise ValueError(
                f"{table_path}: expected rolled-windows schema {ROLLED_WINDOWS_SCHEMA!r}, got {header.get('schema')!r}"
            )
        arrays = {
            key: np.asarray(data[key]) for key in ("windows", "targets", "valid", "next_is_join", "flight", "step", "elapsed_s", "on_final")
        }
        dataset_ids = tuple(str(x) for x in data["dataset_ids"])
    if arrays["targets"].shape[1:] != (len(TARGETS),):
        raise ValueError(f"{table_path}: targets are {arrays['targets'].shape[1:]} wide, this build has {len(TARGETS)}")
    return RolledWindowTable(
        path=table_path, sha256=hashlib.sha256(raw).hexdigest(), header=header, dataset_ids=dataset_ids, **arrays,
    )


# ── the training draw ─────────────────────────────────────────────────────────

class RolledDraw:
    """The per-flight per-epoch coin (`config.plan_rolled_share`): one digest per flight
    per epoch, bytes 0–8 the coin, bytes 8–16 which of the flight's steps — two
    independent reads of one hash, so which step cannot correlate with whether. Salted
    with `ROLLED_SAMPLING_VERSION`, never with the anchor law's salt: the flights the coin
    passes over draw exactly the anchor the observed law draws at share 0."""

    def __init__(self, table: RolledWindowTable, share: float) -> None:
        if not 0.0 <= share <= 1.0:
            raise ValueError(f"plan_rolled_share must be in [0, 1], got {share!r}")
        self.table = table
        self.share = float(share)

    def sample(self, dataset_id: str, epoch_seed: int) -> int | None:
        """The table row this epoch's draw takes for the flight, or None (the observed
        anchor draw stands)."""
        digest = hashlib.sha256(f"{ROLLED_SAMPLING_VERSION}:{int(epoch_seed)}:{dataset_id}".encode()).digest()
        if int.from_bytes(digest[:8], "big") / 2 ** 64 >= self.share:
            return None
        rows = self.table.samples_for(dataset_id)
        pick = int(int.from_bytes(digest[8:16], "big") / 2 ** 64 * len(rows))
        return int(rows[min(pick, len(rows) - 1)])


__all__ = [
    "METADATA_KEY", "POLICIES", "POLICY_MODEL", "POLICY_TRUTH", "ROLLED_SAMPLING_VERSION", "ROLLED_WINDOWS_SCHEMA",
    "RolledDraw", "RolledSample", "RolledWindowTable", "load_rolled_windows", "record_lockstep",
    "rolled_table_header", "write_rolled_windows",
]
