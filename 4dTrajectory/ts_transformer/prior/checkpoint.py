"""The prior's checkpoint (prior design §7 item 1, §8): a format with its own name, and the identity of the artefact
it was trained on.

The identity (§8 item 1, D21) is the spec sha, the day split, the candidate table, the sha256 of the sentence
files, the landings (D63) and the selection of the sentences (D75); `prior.source.artefact_identity` builds it. A checkpoint opens only for the same identity, compared whole: a
prior of another artefact is refused by what differs, never read with a guess (principle 8). The run (its airports and
its held-out airport, `runs.Run`) and the training configuration are recorded beside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import torch

from ts_transformer.prior.model import Prior, PriorConfig

#: The prior of two-tier v4 (prior design, 2026-10-04): five columns, no airport or row-position embedding, RoPE on
#: seconds, candidate tokens of any number. The checkpoints of the instruction-v3 prior (v3–v5) are not opened.
#: v7 (B8, D75): the identity holds the selection of the sentences (`prior.selection.selection_record`).
CHECKPOINT_SCHEMA = "ts-prior-checkpoint-v7"


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


def load_checkpoint(path: Path, identity: Mapping[str, Any]) -> Checkpoint:
    """The prior at ``path`` on the CPU, in eval mode — refused unless it is a `CHECKPOINT_SCHEMA` checkpoint of the
    artefact ``identity``."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path} is a {payload['schema']!r} checkpoint, not {CHECKPOINT_SCHEMA!r}")
    if payload["identity"] != dict(identity):
        stored = payload["identity"]
        differ = sorted(key for key in set(stored) | set(identity)
                        if key not in stored or key not in identity or stored[key] != identity[key])
        raise ValueError(f"{path} was trained on another artefact: {differ} differ")
    model = Prior(PriorConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["state"], strict=True)
    return Checkpoint(model.eval(), payload["identity"], payload["run"], payload["train_config"])
