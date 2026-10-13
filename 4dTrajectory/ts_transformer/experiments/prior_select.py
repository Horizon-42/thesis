"""B5: the choice of cross-validation (prior design §5, §12 B5; D39, D40): the rule fixed before the runs, applied to a
campaign's fold runs (`experiments/prior_campaign.py`) and written as a choice.

- **The score** of an arm (a configuration, the variant `VARIANT` and a seed, on the 5 folds) is the mean, over its
  folds, of the per-step loss on the held-out airport's select days (each fold's ``held_out.json``).
- **The seed scale** is the absolute difference between the scores of configuration A at the two seeds.
- **The configuration** (after the folds of every configuration at the first seed and of A at the second): among the
  configurations whose score is within twice the seed scale of the best, the one with the fewest parameters
  (`choice_configuration.json`).

A fold is read only when it is complete and of the campaign: its held-out airport the fold's; its variant, seed, shape
and training values its arm's (`CONFIGURATIONS`, the rest configuration A's: `PriorConfig`'s and `TrainConfig`'s
defaults); its sentences the selection `landed` (D75); its held-out loss finite; every fold of one arm with one number of
parameters, every run of the campaign of one data identity. No criterion on the readouts of the
folds is applied (D7): they are not read here.

    python run_ts.py prior_select --campaign <a prior_campaign directory>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataclasses import fields

from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.model import PriorConfig
from ts_transformer.prior.train import TrainConfig
from ts_transformer.repo_layout import REPO_ROOT

#: The format of a choice's file.
CHOICE_SCHEMA = "ts-prior-choice-v1"
#: §5's configurations: their values that differ from configuration A (`PriorConfig`'s and `TrainConfig`'s defaults,
#: D40; every head 32 wide); `prior_campaign` gives them to `prior_train` as its flags of the same names.
#: B (d_model 128) and D (A with dropout 0.2, weight decay 0.05) are not run (the user, 2026-10-13): on v12, B scored
#: worst by 21 seed scales and D as A (B5, `choice_configuration.json`).
CONFIGURATIONS = {
    "A": {},
    "C": {"d_model": 256, "heads": 8, "feedforward": 1024},
}
#: The values a fold's run records of its shape and training, each its arm's: configuration A's but those its
#: configuration sets (the seed is the arm's own, checked apart).
SHAPE_FIELDS = ("d_model", "layers", "heads", "feedforward", "dropout", "rope_base")
TRAIN_FIELDS = tuple(f.name for f in fields(TrainConfig) if f.name != "seed")
#: The selection rule every run of the campaign trains under (D75): the one definition (`prior_campaign` reads it).
SELECTION = "landed"
#: The variant every run of the campaign trains (D39): the one definition (`prior_campaign` reads it). The variant
#: `constants` is not run since 2026-10-13 (the user): on v12 it scored worse than `full` by 6 seed scales (B5), and a
#: runway constant is an airport identity (D24).
VARIANT = "full"


def configuration_values(configuration: str) -> dict[str, Any]:
    """A configuration's shape and training values (`SHAPE_FIELDS`, `TRAIN_FIELDS`): configuration A's, those of
    `CONFIGURATIONS` changed."""
    defaults = {f.name: f.default for f in fields(PriorConfig) if f.name in SHAPE_FIELDS}
    defaults.update({f.name: f.default for f in fields(TrainConfig) if f.name in TRAIN_FIELDS})
    unknown = set(CONFIGURATIONS[configuration]) - set(defaults)
    if unknown:
        raise ValueError(f"configuration {configuration} sets {sorted(unknown)}, no value of a run's shape or training")
    return {**defaults, **CONFIGURATIONS[configuration]}


def arm_name(configuration: str, variant: str, seed: int) -> str:
    """A campaign's directory of one arm: its configuration, variant and seed."""
    return f"{configuration}_{variant}_s{seed}"


def arm_score(campaign: Path, configuration: str, variant: str, seed: int, airports: Sequence[str], *,
              smoke: bool = False) -> dict[str, Any]:
    """An arm's score (module docstring) from its folds' ``held_out.json``, with each fold's loss, its parameters and
    the data identity its runs read; refused unless every fold is complete and is the fold it is named, of its arm
    (module docstring), and a smoke run (a sample of the sentences) exactly when the campaign is a smoke."""
    arm = arm_name(configuration, variant, seed)
    expected = configuration_values(configuration)
    folds, parameters, identities = {}, set(), []
    for airport in airports:
        run = campaign / arm / airport
        if not (run / "held_out.json").exists():
            raise ValueError(f"{run}: the fold is not complete (no held_out.json)")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        held_out = json.loads((run / "held_out.json").read_text(encoding="utf-8"))
        if config["run"]["held_out"] != airport or held_out["airport"] != airport:
            raise ValueError(f"{run}: a fold of {config['run']['held_out']}, not {airport}")
        if (config["sample"] is not None) != smoke:
            raise ValueError(f"{run}: a {'formal' if config['sample'] is None else 'smoke'} run in a "
                             f"{'smoke' if smoke else 'formal'} campaign")
        recorded = {**{k: config["model_config"][k] for k in SHAPE_FIELDS},
                    **{k: config["train_config"][k] for k in TRAIN_FIELDS}}
        if (recorded != expected or config["model_config"]["variant"] != variant
                or config["train_config"]["seed"] != seed or config["identity"]["selection"]["rule"] != SELECTION):
            raise ValueError(f"{run}: not a fold of {arm} (variant {config['model_config']['variant']}, seed "
                             f"{config['train_config']['seed']}, {recorded}, selection "
                             f"{config['identity']['selection']['rule']})")
        loss = float(held_out["loss_per_step"])
        if not math.isfinite(loss):
            raise ValueError(f"{run}: the held-out loss is {loss}, not a finite score")
        folds[airport] = loss
        parameters.add(int(config["parameters"]))
        identities.append(config["identity"])
    if len(parameters) != 1:
        raise ValueError(f"{campaign / arm}: its folds have {sorted(parameters)} parameters, not one shape")
    if any(identity != identities[0] for identity in identities):
        raise ValueError(f"{campaign / arm}: its folds read other data")
    return {"score": sum(folds.values()) / len(folds), "folds": folds, "parameters": parameters.pop(),
            "identity": identities[0]}


def seed_scale(scores: Mapping[str, Mapping[str, Any]], seeds: Sequence[int]) -> float:
    """The absolute difference between configuration A's scores at the two seeds."""
    first, second = (scores[arm_name("A", VARIANT, seed)]["score"] for seed in seeds)
    return abs(first - second)


def choose_configuration(scores: Mapping[str, Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    """§5's configuration: the fewest parameters among those within twice the seed scale of the best."""
    scale = seed_scale(scores, seeds)
    candidates = {c: scores[arm_name(c, VARIANT, seeds[0])] for c in CONFIGURATIONS}
    best = min(item["score"] for item in candidates.values())
    within = [c for c in CONFIGURATIONS if candidates[c]["score"] <= best + 2.0 * scale]
    chosen = min(within, key=lambda c: candidates[c]["parameters"])
    return {"seed_scale": scale, "best_score": best, "within": within, "chosen": chosen}


def campaign_record(campaign: Path) -> dict[str, Any]:
    from ts_transformer.experiments.prior_campaign import CAMPAIGN_SCHEMA

    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    if record["schema"] != CAMPAIGN_SCHEMA:
        raise ValueError(f"{campaign} is a {record['schema']} campaign, not {CAMPAIGN_SCHEMA}")
    return record


def select(campaign: Path) -> dict[str, Any]:
    """The choice of the configuration on ``campaign`` (module docstring), as its file holds it."""
    record = campaign_record(campaign)
    seeds, airports = record["seeds"], record["airports"]
    arms = [(c, VARIANT, seeds[0]) for c in CONFIGURATIONS] + [("A", VARIANT, seeds[1])]
    scores = {arm_name(*arm): arm_score(campaign, *arm, airports, smoke=record["smoke"] is not None) for arm in arms}
    identities = [item.pop("identity") for item in scores.values()]
    if any(identity != identities[0] for identity in identities):
        raise ValueError(f"{campaign}: its arms read other data")
    return {"schema": CHOICE_SCHEMA, "step": "configuration", "written_utc": utc_now(), "smoke": record["smoke"],
            "seeds": list(seeds), "airports": list(airports), "arms": scores, **choose_configuration(scores, seeds)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args(argv)
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    out = campaign / "choice_configuration.json"
    if out.exists():
        parser.error(f"{out} exists; a choice is made once")
    choice = select(campaign)
    write_json_atomic(out, choice)
    print(json.dumps({"step": "configuration", "chosen": choice["chosen"], "seed_scale": choice["seed_scale"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
