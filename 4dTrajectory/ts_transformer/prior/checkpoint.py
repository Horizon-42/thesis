"""The prior's checkpoint (prior design §7 item 1, §8): a format with its own name, and the identity of the artefact
it was trained on.

The identity (§8 item 1, D21) is the spec sha, the day split, the candidate table, the sha256 of the sentence
files, the landings (D63) and the selection of the sentences (D75); `prior.source.artefact_identity` builds it. A checkpoint opens only for the same identity, compared whole: a
prior of another artefact is refused by what differs, never read with a guess (principle 8). The run (its airports and
its held-out airport, `runs.Run`) and the training configuration are recorded beside it.

`open_prior` opens a prior run as every runner of the prior does (§7 item 1, D106 item 2): its record, its checkpoint
for the artefact's identity and its procedure masks on today's procedure data (§8 item 2). It reads no stored outcome or
mark of a val sentence (D85): the selection's counts are compared for the read splits (`READ_SPLITS`); the val counts —
an outcome is in the val sentence file, whose sha256 is compared, but a mark comes from the val signals (D111) — are
compared by the val readers after the claim of the val read (`source.require_selection_of`). A readout shows the
identity as `readable_identity` gives it: no val count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import torch

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import READ_SPLITS, SPLITS, load_candidates, load_day_split
from ts_transformer.io_utils import file_sha256, utc_now
from ts_transformer.prior.landings import LandingIndex
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.procedure import PROCEDURE_MASKS, procedure_digests
from ts_transformer.prior.source import airport_landings, artefact_identity
from ts_transformer.repo_layout import repo_relative

#: The prior of two-tier v4 (prior design, 2026-10-04): five columns, no airport or row-position embedding, RoPE on
#: seconds, candidate tokens of any number. The checkpoints of the instruction-v3 prior (v3–v5) are not opened.
#: v7 (B8, D75): the identity holds the selection of the sentences (`prior.selection.selection_record`). v8 (B10, B11,
#: D111): the selection leaves out the flights with a faulty observed track and counts them apart.
CHECKPOINT_SCHEMA = "ts-prior-checkpoint-v8"


class Checkpoint(NamedTuple):
    model: Prior
    identity: dict[str, Any]
    run: dict[str, Any]
    train_config: dict[str, Any]


def save_checkpoint(path: Path, model: Prior, state: Mapping[str, torch.Tensor], *, identity: Mapping[str, Any],
                    run: Mapping[str, Any], train_config: Mapping[str, Any]) -> None:
    """Write ``state`` (the weights kept, of ``model``'s shape) to a new file ``path``; never over an existing one."""
    if any(layer.added is not None for layer in model.layers):
        raise ValueError("a prior with added modules is not a prior checkpoint")
    records = {"identity": identity, "run": run, "train_config": train_config}
    for name, record in records.items():
        # plain JSON values only: what `load_checkpoint` opens with ``weights_only`` and compares as it was written
        if json.loads(json.dumps(record)) != dict(record):
            raise ValueError(f"the {name} is not plain JSON values (lists, not tuples): {record}")
    with open(path, "xb") as file:              # never over an existing file
        torch.save({"schema": CHECKPOINT_SCHEMA, "model_config": model.config.to_dict(),
                    "state": {name: value.detach().cpu() for name, value in state.items()},
                    **{name: dict(record) for name, record in records.items()}}, file)


def load_checkpoint(path: Path, identity: Mapping[str, Any], *, counted: Sequence[str] = SPLITS) -> Checkpoint:
    """The prior at ``path`` on the CPU, in eval mode — refused unless it is a `CHECKPOINT_SCHEMA` checkpoint of the
    artefact ``identity``. ``identity`` may count the selection of the splits ``counted`` only
    (`source.artefact_identity`): the stored counts of the others are not compared, their sentence files' sha256 are."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path} is a {payload['schema']!r} checkpoint, not {CHECKPOINT_SCHEMA!r}")
    stored = payload["identity"]
    if set(counted) != set(SPLITS):
        counts = {split: cells for split, cells in stored["selection"]["counts"].items() if split in counted}
        stored = {**stored, "selection": {**stored["selection"], "counts": counts}}
    if stored != dict(identity):
        differ = sorted(key for key in set(stored) | set(identity)
                        if key not in stored or key not in identity or stored[key] != identity[key])
        raise ValueError(f"{path} was trained on another artefact: {differ} differ")
    model = Prior(PriorConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["state"], strict=True)
    return Checkpoint(model.eval(), payload["identity"], payload["run"], payload["train_config"])


def validation_claim(prior_dir: Path, reader: str) -> str | None:
    """The output a reader of the validation days (`claim_validation_read`) claimed for the prior's run, named relative
    to the repository (`repo_layout.repo_relative`; absolute outside it), None when nothing was claimed."""
    path = prior_dir / f"val_read_{reader}.json"
    return json.loads(path.read_text(encoding="utf-8"))["out"] if path.exists() else None


def holds_claim(prior_dir: Path, reader: str, out: Path) -> bool:
    """Whether the prior's run holds a claim by ``reader`` of its val read naming the output ``out`` (named as
    `claim_validation_read` names it)."""
    return validation_claim(prior_dir, reader) == repo_relative(out)


def claim_validation_read(prior_dir: Path, reader: str, out: Path) -> None:
    """The validation days are read once for each stage (D85): a reader of them (``reader``, a runner's name) marks the
    prior's run as read (``val_read_<reader>.json``, created once) before it reads them, and is refused by name when it
    was read already. The output is named relative to the repository (`repo_relative`): the same name from any
    checkout, also once the worktree the readout ran in is gone."""
    path = prior_dir / f"val_read_{reader}.json"
    try:
        with open(path, "x", encoding="utf-8") as stream:
            json.dump({"reader": reader, "out": repo_relative(out), "utc": utc_now()}, stream)
    except FileExistsError:
        raise ValueError(f"{prior_dir}: its validation days were read by {reader} already ({path.name}); they are read "
                         f"once (D85)") from None


def readable_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """``identity`` as a readout shows it (D85): the selection's counts of the read splits (`READ_SPLITS`) alone."""
    counts = {split: cells for split, cells in identity["selection"]["counts"].items() if split in READ_SPLITS}
    return {**identity, "selection": {**identity["selection"], "counts": counts}}


class OpenedPrior(NamedTuple):
    """A prior run as `open_prior` opens it."""

    directory: Path
    config: dict[str, Any]                    # the run's ``config.json``
    checkpoint: Checkpoint
    geometries: dict[str, AirportGeometry]    # the artefact's candidates, by airport
    landings: dict[str, LandingIndex]         # each airport's roster landings (D63)
    interval_s: float
    selection: str                            # the run's selection rule (D75)


def open_prior(prior_dir: Path, instructions: Path, *, procedure_root: Path = DEFAULT_PROCEDURE_ROOT) -> OpenedPrior:
    """A prior run (``prior_train``'s directory) on the artefact ``instructions`` (module docstring): refused unless its
    record is a `CHECKPOINT_SCHEMA` run, its checkpoint is of the artefact's identity at its Δ and selection (the val
    days' outcomes unread, D85), and its procedure masks are `PROCEDURE_MASKS` on today's procedure data (§8 item 2)."""
    config = json.loads((prior_dir / "config.json").read_text(encoding="utf-8"))
    if config["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError(f"{prior_dir}: a {config['schema']!r} prior, not {CHECKPOINT_SCHEMA!r}")
    interval_s = float(config["identity"]["row_interval_s"])
    selection = config["identity"]["selection"]["rule"]
    geometries = load_candidates(instructions)
    landings = airport_landings(geometries, load_day_split(instructions))
    checkpoint = load_checkpoint(prior_dir / "checkpoint.pt",
                                 artefact_identity(instructions, interval_s, landings, selection, counted=READ_SPLITS),
                                 counted=READ_SPLITS)
    masks = json.loads((prior_dir / "procedure_masks.json").read_text(encoding="utf-8"))
    if masks != {"set": PROCEDURE_MASKS, "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt"),
                 "procedure_data": procedure_digests(geometries, root=procedure_root)}:
        raise ValueError(f"{prior_dir}: the prior's procedure masks are not {PROCEDURE_MASKS} on today's procedure data "
                         f"(§8 item 2)")
    return OpenedPrior(prior_dir, config, checkpoint, geometries, landings, interval_s, selection)
