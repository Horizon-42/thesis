"""The procedure's masks a prior speaks under — a property of the model, recorded beside its checkpoint (prior design
§5.1).

Two kinds of mask take words away when the prior speaks (`generate.Speaker`), kept apart:

- **the vocabulary's rules** — its compatibility rules (`instructions.grammar`), the runway in force not said again,
  the listener's runway lock — belong to the vocabulary and the executor: always on, configured nowhere; a prior is
  bound to its vocabulary by the spec sha `experiments.prior_train.load_prior` checks;
- **the procedure's masks** belong to the post-training stage that trained the model. Each set has a name with a version
  (`SETS`); a change to what a set allows is a new name and the old one is deleted, so a model trained under a set this
  code no longer implements is refused, never spoken under the new rules.

A model's sets are recorded in its directory (`MASKS_FILE`) by the runner that writes its checkpoint (`write_masks`),
bound to that checkpoint by its sha256 and to the data each set reads by a digest (`ProcedureMasks.data_sha256`), and
read back built on today's code and data (`read_masks`) — refused where either moved.

Retired: ``procedure-altitudes-v1`` — the glidepath lower edge alone (rules 1–2), `3cb1fe72` → `181295fc`; post-training
stage 2's runs of schemas v1–v3 were trained under it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.procedure import AltitudeMasks, RunwayProcedure, published_procedures

MASKS_SCHEMA = "ts-prior-procedure-masks-v1"
MASKS_FILE = "procedure_masks.json"
#: The procedure's altitudes: post-training design §3.4, rules 1–5 (`procedure.AltitudeMasks`).
PROCEDURE_ALTITUDES = "procedure-altitudes-v2"
#: Every set this code implements, in the order a model's are listed.
SETS = (PROCEDURE_ALTITUDES,)


def finals_sha256(finals: Mapping[str, Sequence[RunwayProcedure]]) -> str:
    """The digest of the data the procedure's altitudes read: every field of every airport's finals (the candidate's
    threshold, course, elevation and length, the glidepath at the threshold and its slope, the FAF's distance, the LPV
    cone, the DA), airports in name order, each airport's in the pointer's order."""
    data = {code: [asdict(final) for final in finals[code]] for code in sorted(finals)}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProcedureMasks:
    """The procedure's masks a prior speaks under: the sets' names, in `SETS` order (none: the vocabulary's rules
    alone), and what they read — the procedure's altitudes: each airport's candidates' finals, in the pointer's order
    (None without them)."""

    names: tuple[str, ...]
    finals: Mapping[str, tuple[RunwayProcedure, ...]] | None

    def __post_init__(self) -> None:
        if self.names != tuple(s for s in SETS if s in self.names):
            raise ValueError(f"{list(self.names)}: not sets of the procedure's masks this code implements, once each "
                             f"in its order {list(SETS)}")
        if (self.finals is not None) != self.altitudes:
            raise ValueError("the procedure's altitudes read the finals, and only they do")

    @classmethod
    def none(cls) -> ProcedureMasks:
        """The vocabulary's rules alone."""
        return cls((), None)

    @classmethod
    def build(cls, names: Sequence[str], geometries: Mapping[str, AirportGeometry]) -> ProcedureMasks:
        """The sets ``names`` (any order) for the airports ``geometries``, on today's code and data (the procedure's
        altitudes: `procedure.published_procedures`)."""
        unknown = [name for name in names if name not in SETS]
        if unknown or len(set(names)) != len(names):
            raise ValueError(f"{list(names)}: not sets of the procedure's masks this code implements ({list(SETS)}), "
                             f"once each")
        ordered = tuple(s for s in SETS if s in names)
        return cls(ordered, published_procedures(dict(geometries)) if PROCEDURE_ALTITUDES in ordered else None)

    @property
    def altitudes(self) -> bool:
        """Whether the procedure's altitudes are among them."""
        return PROCEDURE_ALTITUDES in self.names

    def data_sha256(self) -> dict[str, str]:
        """Each set's digest of the data it reads (`finals_sha256`)."""
        return {PROCEDURE_ALTITUDES: finals_sha256(self.finals)} if self.altitudes else {}

    def speaking(self, geometries: Sequence[AirportGeometry], words: Words) -> list[AltitudeMasks]:
        """Each set on a batch of flights the prior speaks to (``geometries``: each flight's airport), keeping what it
        tracks row by row."""
        if not self.altitudes:
            return []
        return [AltitudeMasks([self.finals[geometry.code] for geometry in geometries], geometries, words)]


def write_masks(directory: Path, masks: ProcedureMasks, *, writer: str, git: Mapping[str, Any]) -> None:
    """``directory``'s record (`MASKS_FILE`): the procedure's masks its model was trained under (``masks``), bound to the
    checkpoint already written there — never over an existing record."""
    path = directory / MASKS_FILE
    if path.exists():
        raise FileExistsError(f"{path} exists; a model's record is never overwritten")
    write_json_atomic(path, {
        "schema": MASKS_SCHEMA, "checkpoint_sha256": file_sha256(directory / "checkpoint.pt"),
        "sets": [{"name": name, "data_sha256": digest} for name, digest in masks.data_sha256().items()],
        "written_by": writer, "written_utc": utc_now(), "git": dict(git)})


def read_masks(directory: Path, geometries: Mapping[str, AirportGeometry]) -> ProcedureMasks:
    """The procedure's masks ``directory``'s model was trained under, built for the airports ``geometries`` on today's
    code and data — refused: no record, another schema, another checkpoint beside it, a set this code does not implement,
    or data other than the set read when the record was written."""
    path = directory / MASKS_FILE
    if not path.exists():
        raise ValueError(f"{directory} records no procedure's masks ({MASKS_FILE}): which ones it was trained under is "
                         f"not known (prior design §5.1)")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["schema"] != MASKS_SCHEMA:
        raise ValueError(f"{path} is a {record['schema']} file, not {MASKS_SCHEMA}")
    if record["checkpoint_sha256"] != file_sha256(directory / "checkpoint.pt"):
        raise ValueError(f"{path} records another checkpoint than the one beside it")
    recorded = {entry["name"]: entry["data_sha256"] for entry in record["sets"]}
    retired = [name for name in recorded if name not in SETS]
    if retired:
        raise ValueError(f"{directory} was trained under the procedure's masks {retired}, which this code does not "
                         f"implement ({list(SETS)}): it is not spoken under other rules")
    masks = ProcedureMasks.build(list(recorded), geometries)
    moved = [name for name, digest in masks.data_sha256().items() if recorded[name] != digest]
    if moved:
        raise ValueError(f"the data {moved} read changed since {directory} was trained (the procedures, the runway "
                         f"data): speaking under today's is a decision, not a default")
    return masks
