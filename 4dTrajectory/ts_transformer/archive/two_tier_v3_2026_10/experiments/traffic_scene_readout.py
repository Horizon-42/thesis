"""Multi-aircraft M2's readout (design §6.1, §7): what the scene prior "scene" gains from seeing the traffic, against the
single-aircraft base prior. A readout only (§6.1: recorded, no gate).

On the select days every labelled flight's asked cells are read three times, teacher-forced: by **base** (single aircraft,
the flights of the prior's own split), by **scene** in its scene samples (every aircraft of the sample, with its edge
features; `traffic_scene_data.build_split`) and by **alone** — the same scene prior with each aircraft in a sample of its
own, the attention between aircraft cut (design §7, "交互有没有被用上"). Per cell and column: the negative log-likelihood
of the labelled class, and between scene and alone the KL divergence of the word distributions (how far seeing the
traffic moves them). Refused unless the scene samples ask exactly the cells base's flights ask, each once.

- ``nll_per_step``: each reading's NLL per step over every asked cell (every column; the per-step measure training
  stopped on);
- per read column (speed, heading, clearance) and flag (leader, busy — M1's, `traffic_interaction.step_flags`: every
  arrival of the day counts, the background tracks too long for the model's positions as well), from row
  `traffic_interaction.READ_FROM`: the paired differences scene − base and scene − alone and the KL, their raw means on
  the flagged steps and on the others, each with a 95 % interval from airport × UTC-hour cluster resamples (``paired``;
  resamples holding no step on a side are counted); and for every reading and paired quantity M1's matched gap — flagged
  less the others within phase strata, `traffic_interaction.read_group` (``matched``). Every quantity is resampled with
  the same draws: a generator of its own from `BOOTSTRAP_SEED`, used airport by airport and then for the pool, as M1
  drew — so base's matched gap is M1's to the bit, intervals included.

A smoke prior (trained with ``--limit``) is read and the record says so.

Writes ``scene_readout.json`` into a NEW directory.

    python run_ts.py traffic_scene_readout --scene 4dTrajectory/outputs/POOLED/prior/m2_scene_20260928/scene_s1337 \\
        --base 4dTrajectory/outputs/POOLED/prior/v3_step1_20260924/full_s1337 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --out 4dTrajectory/outputs/POOLED/traffic/scene_readout_<date> --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.experiments.prior_train import (
    PRIOR_CHECKPOINT_SCHEMA, SCENE_CHECKPOINT_SCHEMA, load_prior, roster_digests, roster_record, rosters,
)
from ts_transformer.experiments.traffic_interaction import (
    BOOTSTRAP, BOOTSTRAP_SEED, FLAGS, READ, READ_FROM, SPLIT, STRATA, flight_context, pooled_flights, read_group,
)
from ts_transformer.experiments.traffic_prior_train import scene_batch, scene_batches
from ts_transformer.experiments.traffic_scene_data import Built, build_split, edge_source_sha256
from ts_transformer.instructions.artefact import load_spec
from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.data import VARIANTS, Split, airport_landings, batches, load_split
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene_data import SceneSample
from ts_transformer.prior.train import TrainConfig, batch_logits, to_batch
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-traffic-scene-readout-v1"
READINGS = ("base", "scene", "alone")
#: The paired quantities per cell (module docstring).
PAIRED = ("scene_minus_base", "scene_minus_alone", "kl_scene_alone")


def _log_p(logits: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.log_softmax(column.float(), dim=-1) for column in logits]


def _nll(log_p: Sequence[torch.Tensor], targets: torch.Tensor) -> np.ndarray:
    """``[..., 6]``: the NLL of each column's target class."""
    return torch.stack([-lp.gather(-1, targets[..., c: c + 1])[..., 0] for c, lp in enumerate(log_p)], dim=-1).cpu().numpy()


def _kl(p: Sequence[torch.Tensor], q: Sequence[torch.Tensor]) -> np.ndarray:
    """``[..., 6]``: KL(p ‖ q) per column, from log-probabilities (a class p masks out, at −inf, adds nothing)."""
    def column(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isinf(a), torch.zeros_like(a), a.exp() * (a - b)).sum(dim=-1)

    return torch.stack([column(a, b) for a, b in zip(p, q)], dim=-1).cpu().numpy()


def base_cells(model: Prior, split: Split, tokens: int, device: torch.device) -> list[np.ndarray]:
    """Each flight's NLL per row and column (``[rows, 6]``, NaN where the loss does not ask the cell)."""
    out: list[np.ndarray] = [np.empty(0) for _ in split.flights]
    with torch.inference_mode():
        for indices in batches(split.flights, tokens, None):
            batch = to_batch(split, indices, device)
            nll = _nll(_log_p(batch_logits(model, batch)), batch["targets"])
            for b, i in enumerate(indices):
                flight = split.flights[i]
                out[i] = np.where(flight.asked, nll[b, 0, : flight.rows], np.nan)
    return out


def alone(built: Built, a: int) -> Built:
    """Aircraft ``a`` of ``built`` in a sample of its own: its rows on its own steps, the same loss window."""
    sample, node = built.sample, built.sample.nodes[a]
    first = node.first_step
    lone = SceneSample(sample.airport, (dataclasses.replace(node, first_step=0),), sample.kept(node),
                       sample.loss_from - first, sample.loss_to - first)
    return Built(lone, (built.flights[a],), built.separation)


def scene_cells(model: Prior, built: Sequence[Built], slots: int, tokens: int, device: torch.device
                ) -> dict[str, dict[str, np.ndarray]]:
    """Per flight with a sentence (its dataset id): ``scene`` and ``alone`` NLL and their ``kl`` per row and column
    (``[rows, 6]``, NaN where no sample asks the cell); a cell asked twice is refused."""
    out: dict[str, dict[str, np.ndarray]] = {}
    with torch.inference_mode():
        for indices in scene_batches(built, tokens, None):
            batch = scene_batch(built, indices, slots, device)
            log_p = _log_p(batch_logits(model, batch))
            nll = _nll(log_p, batch["targets"])
            lone = [(b, a) for b, i in enumerate(indices) for a, node in enumerate(built[i].sample.nodes)
                    if built[i].sample.asked(node).any()]
            singles = [alone(built[indices[b]], a) for b, a in lone]
            for group in scene_batches(singles, tokens, None):
                lone_batch = scene_batch(singles, group, slots, device)
                lone_log_p = _log_p(batch_logits(model, lone_batch))
                lone_nll = _nll(lone_log_p, lone_batch["targets"])
                for g, j in enumerate(group):
                    b, a = lone[j]
                    sample = built[indices[b]].sample
                    node = sample.nodes[a]
                    kept, first = sample.kept(node), node.first_step
                    asked = sample.asked(node)
                    kl = _kl([lp[b, a, first: first + kept] for lp in log_p], [lp[g, 0, :kept] for lp in lone_log_p])
                    cells = out.setdefault(node.key, {name: np.full((node.rows, 6), np.nan)
                                                      for name in ("scene", "alone", "kl")})
                    if not np.isnan(cells["scene"][:kept][asked]).all():
                        raise ValueError(f"{node.key}: a cell is asked by two samples")
                    for name, values in (("scene", nll[b, a, first: first + kept]), ("alone", lone_nll[g, 0, :kept]),
                                         ("kl", kl)):
                        cells[name][:kept][asked] = values[asked]
    return out


def nll_per_step(cells: Sequence[np.ndarray]) -> dict[str, Any]:
    """Summed NLL per column over the rows asking any column, and their sum (the training runners' measure)."""
    stacked = np.concatenate(cells)
    steps = int((~np.isnan(stacked)).any(axis=1).sum())
    per_column = np.nansum(stacked, axis=0) / steps
    return {"nll_per_step": float(per_column.sum()), "per_column": dict(zip(COLUMNS, per_column.tolist())),
            "steps": steps}


def paired(flights: Sequence[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    """Per read column and flag: each `PAIRED` quantity's mean on the flagged steps and on the others (read from row
    `READ_FROM`), with 95 % intervals from cluster resamples."""
    names = sorted({f["cluster"] for f in flights})
    cluster = np.concatenate([np.full(len(f["stratum"]), names.index(f["cluster"])) for f in flights])
    rows = np.concatenate([np.arange(len(f["stratum"])) for f in flights])
    weights = rng.multinomial(len(names), np.full(len(names), 1.0 / len(names)), size=BOOTSTRAP)
    out: dict[str, Any] = {"clusters": len(names)}
    values = {name: np.concatenate([f[name] for f in flights]) for name in PAIRED}
    for column in READ:
        read = ~np.isnan(values["scene_minus_base"][:, column]) & (rows >= READ_FROM)
        per_flag: dict[str, Any] = {}
        for flag_name in FLAGS:
            flag = np.concatenate([f[flag_name] for f in flights])
            entry: dict[str, Any] = {}
            for side, where in (("flagged", read & flag), ("other", read & ~flag)):
                entry[side] = {"steps": int(where.sum())}
                for name in PAIRED:
                    sums = np.bincount(cluster[where], values[name][where, column], minlength=len(names))
                    counts = np.bincount(cluster[where], minlength=len(names)).astype(float)
                    draws = (weights @ sums) / np.where(weights @ counts > 0, weights @ counts, np.nan)
                    kept = draws[~np.isnan(draws)]
                    entry[side][name] = {
                        "mean": float(sums.sum() / counts.sum()) if counts.sum() else None,
                        "95": [float(np.percentile(kept, 2.5)), float(np.percentile(kept, 97.5))] if len(kept) else None,
                        "resamples_without_a_step": int(len(draws) - len(kept))}
            per_flag[flag_name] = entry
        out[COLUMNS[column]] = per_flag
    return out


def matched(flights: Sequence[dict[str, Any]], quantity: str, strata: int, rng: np.random.Generator
            ) -> dict[str, Any]:
    """M1's matched gap of ``quantity`` (a reading or a paired quantity): `read_group` on it from row `READ_FROM`, per
    read column and flag."""
    group = [{"nll": np.where(np.arange(len(f["stratum"]))[:, None] >= READ_FROM, f[quantity][:, list(READ)], np.nan),
              "p_change": np.zeros((len(f["stratum"]), len(READ))),
              "changed": np.zeros((len(f["stratum"]), len(READ)), dtype=bool),
              **{k: f[k] for k in (*FLAGS, "stratum", "cluster")}} for f in flights]
    read = read_group(group, strata, rng)
    return {COLUMNS[c]: {flag: read[COLUMNS[c]][flag]["nll"] for flag in FLAGS} for c in READ}


def report(groups: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(per airport, pooled)``: each group's flights, `nll_per_step` of each reading, `paired` and every quantity's
    `matched` gap — each drawn by a generator of its own, airport by airport (in ``groups``' order) and then the pool."""
    pool = pooled_flights(groups)
    airports: dict[str, Any] = {airport: {"flights": len(flights), "nll_per_step": {
        reading: nll_per_step([f[reading] for f in flights]) for reading in READINGS}, "matched": {}}
        for airport, flights in groups.items()}
    pooled: dict[str, Any] = {"flights": len(pool), "nll_per_step": {
        reading: nll_per_step([f[reading] for f in pool]) for reading in READINGS}, "matched": {}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for airport, flights in groups.items():
        airports[airport]["paired"] = paired(flights, rng)
    pooled["paired"] = paired(pool, rng)
    for quantity in (*READINGS, *PAIRED):
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for airport, flights in groups.items():
            airports[airport]["matched"][quantity] = matched(flights, quantity, STRATA, rng)
        pooled["matched"][quantity] = matched(pool, quantity, STRATA * len(groups), rng)
    return airports, pooled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--scene", type=Path, required=True, help="the scene prior's run directory")
    parser.add_argument("--base", type=Path, required=True, help="the base prior's run directory")
    parser.add_argument("--instructions", type=Path, required=True, help="the sentence artefact both read")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    scene_dir, base_dir, instructions, out = (path if path.is_absolute() else REPO_ROOT / path
                                              for path in (args.scene, args.base, args.instructions, args.out))
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    started = time.perf_counter()
    device = torch.device(args.device)
    scene, scene_payload, scene_config, _ = load_prior(scene_dir, instructions)
    base, base_payload, base_config, _ = load_prior(base_dir, instructions)
    if scene_payload["schema"] != SCENE_CHECKPOINT_SCHEMA or base_payload["schema"] != PRIOR_CHECKPOINT_SCHEMA:
        raise SystemExit(f"--scene takes a {SCENE_CHECKPOINT_SCHEMA} prior, --base a {PRIOR_CHECKPOINT_SCHEMA} one")
    variant = base.config.variant
    if scene.config.variant != variant or scene.config.airports != base.config.airports:
        raise SystemExit("the two priors read other variants or airports")
    tracks = rosters(instructions)
    for name, config in (("scene", scene_config), ("base", base_config)):
        if roster_digests(roster_record(tracks)) != roster_digests(config["tracks_rosters"]):
            raise SystemExit(f"the tracks rosters changed since the {name} prior was trained")
    spec = load_spec(instructions)
    landings = airport_landings(instructions, tracks)
    split = load_split(instructions, SPLIT, spec, Words(spec), variant, landings=landings, airports=base.config.airports)
    base.to(device)
    base_nll = base_cells(base, split, TrainConfig(**base_payload["train_config"]).tokens_per_batch, device)
    landings = landings if VARIANTS[variant].landing_context else None
    built, counts = build_split(instructions, SPLIT, spec, scene.config.airports, landings, scene.config.max_rows)
    scene.to(device)
    scene_nll = scene_cells(scene, [b for b in built if b.sample.asks], scene.config.candidate_slots,
                            TrainConfig(**scene_payload["train_config"]).tokens_per_batch, device)
    print(f"{len(split.flights)} {SPLIT} flights, {counts['samples']} samples read, "
          f"{time.perf_counter() - started:.0f}s", flush=True)

    context = flight_context(instructions, spec, SPLIT, [f.dataset_id for f in split.flights])
    groups: dict[str, list[dict[str, Any]]] = {}
    for airport, members in context.items():
        groups[airport] = []
        for k, c in members:
            key, cells = split.flights[k].dataset_id, scene_nll[split.flights[k].dataset_id]
            if not np.array_equal(np.isnan(cells["scene"]), np.isnan(base_nll[k])):
                raise ValueError(f"{key}: the scene samples do not ask the cells base's flight asks")
            groups[airport].append({"base": base_nll[k], "scene": cells["scene"], "alone": cells["alone"],
                                    "scene_minus_base": cells["scene"] - base_nll[k],
                                    "scene_minus_alone": cells["scene"] - cells["alone"], "kl_scene_alone": cells["kl"],
                                    **c})
    if sum(len(g) for g in groups.values()) != len(scene_nll):
        raise ValueError("the scene samples ask flights base's split does not hold")
    airports, pooled = report(groups)
    smoke = {"scene": bool(scene_config["smoke"]), "base": bool(base_config["smoke"])}
    record = {
        "schema": SCHEMA, "written_utc": utc_now(), "split": SPLIT, "read": [COLUMNS[c] for c in READ],
        "read_from_row": READ_FROM, "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED, "unit": "airport × UTC hour"},
        "scene": {"directory": repo_relative(scene_dir), "checkpoint_sha256": file_sha256(scene_dir / "checkpoint.pt"),
                  "edge_source_sha256": edge_source_sha256()},
        "base": {"directory": repo_relative(base_dir), "checkpoint_sha256": file_sha256(base_dir / "checkpoint.pt")},
        "instructions": repo_relative(instructions), "spec_sha256": spec.sha256, "samples": counts, "smoke": smoke,
        "pooled": pooled, "airports": airports, "git": git_state(), "seconds": time.perf_counter() - started,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "scene_readout.json", record)
    if any(smoke.values()):
        print(f"SMOKE priors read: {smoke}")
    for reading in READINGS:
        print(f"{reading:6s} NLL per step {pooled['nll_per_step'][reading]['nll_per_step']:.4f}")
    for column in READ:
        name = COLUMNS[column]
        for flag in FLAGS:
            e = pooled["paired"][name][flag]
            print(f"{name:8s} {flag:6s} scene−base {e['flagged']['scene_minus_base']['mean']:+.4f} "
                  f"{e['flagged']['scene_minus_base']['95']} (others {e['other']['scene_minus_base']['mean']:+.4f}); "
                  f"scene−alone {e['flagged']['scene_minus_alone']['mean']:+.4f} (others "
                  f"{e['other']['scene_minus_alone']['mean']:+.4f}); KL {e['flagged']['kl_scene_alone']['mean']:.4f} "
                  f"(others {e['other']['kl_scene_alone']['mean']:.4f}); matched scene−base "
                  f"{pooled['matched']['scene_minus_base'][name][flag]['difference']:+.4f}")
    print(f"wrote {out / 'scene_readout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
