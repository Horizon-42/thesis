"""Multi-aircraft M1 (design §6.1; prior design §8, "交互"): does the single-aircraft base prior's word choice depend on
traffic it cannot see? A readout only (design §9 item 12), no gate.

On the select days, every labelled flight's sentence is teacher-forced through the prior, and at each step the loss
counts after the first predicted one (from row `READ_FROM` = `prior.scene.N_LOOK` + 1: at `N_LOOK` the prior says every
column from scratch — its NLL is another quantity and a change is certain) the speed, heading and approach columns'
negative log-likelihood of the labelled class and the probability the prior gives a change of word are read. Each step
is flagged:

- **leader** — another aircraft in the scene lands earlier on the same observed runway and is at most `LEADER_RANGE_M`
  closer to its threshold (`prior.scene.leader_gap_m`, the scene census's definition);
- **busy** — at least `BUSY_AIRCRAFT` aircraft in the scene, itself and background arrivals included (prior design §8).

The scene holds the select days' arrivals only (with a sentence, and background: the labeller refused them), each from
its first row to its last sentence row (background: its last row), as the scene census builds it for the training days.

Steps with a leader sit mostly near the final, so a raw comparison mixes traffic with the phase of flight. Each flag is
therefore also read within PHASE strata — established on the final or not (the artefact's capture row) × the straight-line
distance to the threshold in bins of `DISTANCE_BIN_M` up to `LAST_BIN_M`; the pool's strata are airport × phase (airports
differ in both level and traffic) — and the per-stratum differences weighted by the flagged steps' mix (the strata holding
both kinds; the share of flagged steps they cover is reported). The matched difference's 95 % interval comes from a
bootstrap of `BOOTSTRAP` resamples of airport × UTC-hour clusters (flights of one busy hour share their flags; resampling
flights alone would be too narrow). For the changes: the share of steps whose labelled word changes and the prior's mean
probability of a change, per kind — the same matching.

Writes ``interaction.json`` into a NEW directory; CPU.

    python run_ts.py traffic_interaction --prior 4dTrajectory/outputs/POOLED/prior/v3_step1_20260924/full_s1337 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --out 4dTrajectory/outputs/POOLED/traffic/interaction_<date>
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.experiments.prior_train import load_prior, roster_digests, roster_record, rosters, splits
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals, load_spec
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import APPROACH, COLUMNS, HEADING, SPEED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.data import Split, batches
from ts_transformer.prior.scene import LEADER_RANGE_M, N_LOOK, Presence, SceneIndex, leader_gap_m, presence
from ts_transformer.prior.train import TrainConfig, batch_logits, to_batch
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-traffic-interaction-v1"
#: The first row read: after the first predicted step (module docstring).
READ_FROM = N_LOOK + 1
SPLIT = "select"
#: The columns read (design §6.1: speed, heading, clearance words).
READ = (SPEED, HEADING, APPROACH)
#: Prior design §8: "场上是否有 ≥ 3 架飞机".
BUSY_AIRCRAFT = 3
DISTANCE_BIN_M, LAST_BIN_M = 5_000.0, 20_000.0
#: The phase strata: not established / established × the distance bins (the last open-ended).
STRATA = 2 * (int(LAST_BIN_M // DISTANCE_BIN_M) + 1)
BOOTSTRAP, BOOTSTRAP_SEED = 200, 1337
#: A bootstrap cluster: an airport's UTC hour.
CLUSTER_S = 3_600.0
FLAGS = ("leader", "busy")


def step_flags(speaking: Sequence[Presence], background: Sequence[Presence]) -> list[dict[str, np.ndarray]]:
    """For each of ``speaking`` (in order), at each of its rows: leader, busy (module docstring)."""
    index = SceneIndex([*speaking, *background])
    out = []
    for ego in speaking:
        rows_s = ego.times_s
        near = [p for p in index.overlapping(float(rows_s[0]), float(rows_s[-1])) if p.dataset_id != ego.dataset_id]
        present = np.array([(rows_s >= p.start_s) & (rows_s <= p.end_s) for p in near]).reshape(len(near), len(rows_s))
        out.append({"leader": np.isfinite(leader_gap_m(ego, near, rows_s, present)),
                    "busy": present.sum(axis=0) + 1 >= BUSY_AIRCRAFT})
    return out


def phase_strata(ego: Presence, capture_row: int) -> np.ndarray:
    """Each row's phase stratum: established (from the capture row) × the distance bin, as one integer."""
    rows = len(ego.times_s)
    bins = STRATA // 2
    distance = np.minimum(ego.to_threshold_m // DISTANCE_BIN_M, bins - 1).astype(np.int64)
    return (np.arange(rows) >= capture_row).astype(np.int64) * bins + distance


def step_readings(model: Any, split: Split, tokens: int, device: torch.device) -> list[dict[str, np.ndarray]]:
    """Each flight's per-row negative log-likelihood of its labelled class and probability of a change, per read column
    (``[rows, len(READ)]``, NaN where the loss does not count the row), and whether the labelled word changes."""
    out: list[dict[str, np.ndarray]] = [{} for _ in split.flights]
    with torch.inference_mode():
        for indices in batches(split.flights, tokens, None):
            batch = to_batch(split, indices, device)
            logits = batch_logits(model, batch)
            for b, i in enumerate(indices):
                flight = split.flights[i]
                rows = flight.rows
                nll, change = np.full((rows, len(READ)), np.nan), np.full((rows, len(READ)), np.nan)
                for k, column in enumerate(READ):
                    log_p = torch.log_softmax(logits[column][b, 0, :rows], dim=-1)
                    target = batch["targets"][b, 0, :rows, column]
                    asked = flight.asked[:, column] & (np.arange(rows) >= READ_FROM)
                    nll[asked, k] = (-log_p.gather(-1, target[:, None])[:, 0]).cpu().numpy()[asked]
                    change[asked, k] = (1.0 - log_p[:, 0].exp()).cpu().numpy()[asked]
                out[i] = {"nll": nll, "p_change": change, "changed": flight.targets[:, list(READ)] > 0}
    return out


def _matched(sums: np.ndarray, counts: np.ndarray, flagged: int) -> dict[str, float]:
    """From per-stratum sums and counts (``[strata, 2]``: other, flagged): the flagged steps' mean less the others',
    within each stratum holding both, weighted by the flagged steps."""
    both = (counts[:, 0] > 0) & (counts[:, 1] > 0)
    weight = float(counts[both, 1].sum())
    difference = float((counts[both, 1] * (sums[both, 1] / counts[both, 1] - sums[both, 0] / counts[both, 0])).sum())
    return {"difference": difference / weight if weight else float("nan"),
            "flagged_steps_covered": weight / flagged if flagged else float("nan")}


def _per_flight(values: np.ndarray, flag: np.ndarray, stratum: np.ndarray, owner: np.ndarray, flights: int,
                strata: int) -> tuple[np.ndarray, np.ndarray]:
    """``[flights, strata, 2]`` sums and counts of ``values`` by flight, stratum and flag."""
    sums, counts = np.zeros((flights, strata, 2)), np.zeros((flights, strata, 2))
    np.add.at(sums, (owner, stratum, flag.astype(np.int64)), values)
    np.add.at(counts, (owner, stratum, flag.astype(np.int64)), 1.0)
    return sums, counts


def read_group(flights: list[dict[str, Any]], strata: int, rng: np.random.Generator) -> dict[str, Any]:
    """One airport's (or the pool's) numbers over ``strata`` strata: per read column and flag, the raw means, the matched
    differences (`_matched`) and, for the NLL, their cluster-bootstrap 95 % interval (each flight's ``cluster``); the
    same for the changes."""
    def stacked(name: str) -> np.ndarray:
        return np.concatenate([f[name] for f in flights])

    names = sorted({f["cluster"] for f in flights})
    member = np.array([names.index(f["cluster"]) for f in flights])
    owner = np.concatenate([np.full(len(f["stratum"]), member[n]) for n, f in enumerate(flights)])
    stratum = stacked("stratum")
    out: dict[str, Any] = {"flights": len(flights), "clusters": len(names)}
    draws_weights = rng.multinomial(len(names), np.full(len(names), 1.0 / len(names)), size=BOOTSTRAP)
    for k, column in enumerate(READ):
        nll, p_change, changed = stacked("nll")[:, k], stacked("p_change")[:, k], stacked("changed")[:, k]
        counted = ~np.isnan(nll)
        per_flag: dict[str, Any] = {}
        for flag_name in FLAGS:
            flag = stacked(flag_name)[counted]
            s, o = stratum[counted], owner[counted]
            entry: dict[str, Any] = {"steps": int(flag.sum()), "other_steps": int((~flag).sum())}
            for label, values in (("nll", nll[counted]), ("changed", changed[counted].astype(float)),
                                  ("p_change", p_change[counted])):
                sums, counts = _per_flight(values, flag, s, o, len(names), strata)
                entry[label] = {"flagged_mean": float(values[flag].mean()) if flag.any() else None,
                                "other_mean": float(values[~flag].mean()) if (~flag).any() else None,
                                **_matched(sums.sum(axis=0), counts.sum(axis=0), int(flag.sum()))}
                if label == "nll":
                    draws = [_matched(np.tensordot(w, sums, axes=1), np.tensordot(w, counts, axes=1),
                                      int(flag.sum()))["difference"] for w in draws_weights]
                    kept = [d for d in draws if not np.isnan(d)]
                    entry[label]["difference_95"] = ([float(np.percentile(kept, 2.5)), float(np.percentile(kept, 97.5))]
                                                     if kept else None)
                    entry[label]["resamples_without_a_comparison"] = len(draws) - len(kept)
            per_flag[flag_name] = entry
        out[COLUMNS[column]] = per_flag
    return out


def flight_context(instructions: Path, spec: VocabularySpec, split: str, dataset_ids: Sequence[str]
                   ) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """Per airport, each labelled flight of ``split`` (its index in the artefact's sentences, which must be
    ``dataset_ids``) with its rows' flags (`step_flags`, the split's arrivals in the scene), phase strata and bootstrap
    cluster (its airport's UTC hour)."""
    geometries = load_candidates(instructions)
    signals = load_signals(instructions, split)
    sentences = load_sentences(instructions, split, spec)
    offsets = sentences["offsets"]
    spoken = [int(i) for i in sentences["signal_index"]]
    if [signals[i].dataset_id for i in spoken] != list(dataset_ids):
        raise ValueError("the split's flights are not the artefact's sentences in order")
    by_airport: dict[str, list[int]] = defaultdict(list)
    for k, i in enumerate(spoken):
        by_airport[signals[i].airport].append(k)
    background = defaultdict(list)
    for i in sorted(set(range(len(signals))) - set(spoken)):
        flight = signals[i]
        background[flight.airport].append(presence(flight, None, geometries[flight.airport]))
    out: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for airport, members in sorted(by_airport.items()):
        speaking = [presence(signals[spoken[k]], int(offsets[k + 1] - offsets[k]), geometries[airport]) for k in members]
        flags = step_flags(speaking, background[airport])
        out[airport] = [(k, {**flag, "stratum": phase_strata(ego, int(sentences["capture_row"][k])),
                             "cluster": (airport, int(ego.start_s // CLUSTER_S))})
                        for k, ego, flag in zip(members, speaking, flags)]
    return out


def pooled_flights(rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Every airport's flights as one group whose strata are airport × phase (airport ``a``'s phase stratum
    + ``a`` × `STRATA`)."""
    return [{**f, "stratum": f["stratum"] + a * STRATA} for a, group in enumerate(rows.values()) for f in group]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the prior's run directory (the base model)")
    parser.add_argument("--instructions", type=Path, required=True, help="the sentence artefact it reads")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    prior_dir = args.prior if args.prior.is_absolute() else REPO_ROOT / args.prior
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    started = time.perf_counter()
    model, payload, config, _ = load_prior(prior_dir, instructions)
    tracks = rosters(instructions)
    if roster_digests(roster_record(tracks)) != roster_digests(config["tracks_rosters"]):
        raise SystemExit("the tracks rosters changed since the prior was trained")
    device = torch.device(args.device)
    model.to(device)
    spec = load_spec(instructions)
    split = splits(instructions, spec, Words(spec), model.config.variant, tracks, None)[SPLIT]
    readings = step_readings(model, split, TrainConfig(**payload["train_config"]).tokens_per_batch, device)
    print(f"{len(split.flights)} {SPLIT} flights read, {time.perf_counter() - started:.0f}s", flush=True)

    context = flight_context(instructions, spec, SPLIT, [f.dataset_id for f in split.flights])
    rows = {airport: [{**readings[k], **c} for k, c in members] for airport, members in context.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    report = {airport: read_group(group, STRATA, rng) for airport, group in rows.items()}
    pooled = read_group(pooled_flights(rows), STRATA * len(rows), rng)
    payload_out = {
        "schema": SCHEMA, "written_utc": utc_now(), "split": SPLIT, "prior": repo_relative(prior_dir),
        "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt"), "instructions": repo_relative(instructions),
        "spec_sha256": spec.sha256, "read": [COLUMNS[c] for c in READ],
        "flags": {"leader_range_m": LEADER_RANGE_M, "busy_aircraft": BUSY_AIRCRAFT},
        "strata": {"distance_bin_m": DISTANCE_BIN_M, "last_bin_m": LAST_BIN_M, "established": "from the capture row"},
        "read_from_row": READ_FROM,
        "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED, "unit": "airport × UTC hour"},
        "pooled_strata": "airport × phase",
        "pooled": pooled, "airports": report, "git": git_state(), "seconds": time.perf_counter() - started,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "interaction.json", payload_out)
    for column in READ:
        name = COLUMNS[column]
        for flag in FLAGS:
            e = pooled[name][flag]
            print(f"{name:8s} {flag:6s} steps {e['steps']:7d}  NLL {e['nll']['flagged_mean']} vs "
                  f"{e['nll']['other_mean']}, matched {e['nll']['difference']:+.4f} {e['nll']['difference_95']}, "
                  f"changes {e['changed']['difference']:+.4f} (prior {e['p_change']['difference']:+.4f})")
    print(f"wrote {out / 'interaction.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
