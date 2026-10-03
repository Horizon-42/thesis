"""Does post-training move the go-around word? (multi-aircraft design §6.6 step 9.4; the user 2026-10-03, after the small
run: the model never said a go-around itself, so a count of 0 cannot tell whether its probability moved.)

Reads an M4-in-windows run (`traffic_window_reward`, R37) from its own files and, for each training round k ≥ 1, speaks
the round's sentences again exactly as the run spoke them — the round's windows drawn again (`traffic_window_reward.
round_windows`), spoken by the model the run spoke them with (the start for round 1, round k − 1's checkpoint after),
from the round's stream, probed as the run probed (`WindowSpeaking`, in forked processes as the run's speakers: each loop
batch from its own stream) — and refuses unless every sentence is the stored one (``sentences.npz``: window, flight,
sample, every word said, outcome, runway, reward, probed, forced step; the advantages and probe gains computed again).

Then every sentence a probe made say a go-around (inside its counted steps) is read by every model of the run — the
start (round 0) and each round's checkpoint — teacher-forced in its window as the tuner reads it
(`WindowRewardTuner.forced_log_probs`): the probability each model gives the go-around word at the probe's step, under
the masks the sentence was spoken under. That is the word the probe's cross-entropy trains (on the sentences whose probe
gain is above 0, `window_advantages`); this reads whether it moved, on the sentences learned and on the others.

Writes into a NEW directory ``forced.jsonl`` (a row per forced sentence: its round, window, flight, sample, kind, step,
reward, outcome, probe gain, whether it was learned, and each model's log-probability) and ``probe_readout.json``
(`SCHEMA`: the run, the reproduction, and per round of sentences × model the probability's mean, median and quartiles, on
the learned and the other sentences, and how many rose against round 0).

    python run_ts.py traffic_window_probe_readout --run 4dTrajectory/outputs/POOLED/prior/<run>/<seed dir> \\
        --out <new directory> [--workers 3]

On the run's own device (its ``config.json``: CPU and CUDA draw other streams). A round no probe made say a go-around inside
a sentence's counted steps is named and not read.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
import torch

from ts_transformer.autopilot import replay
from ts_transformer.experiments.prior_free_generation import start_altitude_windows
from ts_transformer.experiments.prior_generation_training_export import outputs_path
from ts_transformer.experiments.prior_train import PRIOR_CHECKPOINT_SCHEMA, load_prior, rosters
from ts_transformer.experiments.traffic_rounds import SAMPLING_STREAM, completed_rounds, round_seed
from ts_transformer.experiments.traffic_scene_data import airport_flights, edge_source_sha256
from ts_transformer.experiments.traffic_speaking import with_tracks
from ts_transformer.experiments.traffic_window_augment import busiest
from ts_transformer.experiments.traffic_window_generation import WindowSentences, in_processes
from ts_transformer.experiments.traffic_window_reward import (
    PARENT_GROWTH_GB, SCHEMA as RUN_SCHEMA, SPEAKER_HOST_GB, RoundSource, WindowRound, WindowSpeaking, round_windows,
    window_split,
)
from ts_transformer.experiments.traffic_window_tuner import WindowRewardTuner, window_advantages
from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.artefact import load_candidates, load_spec
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.data import (
    VARIANTS, Split, airport_landings, candidate_table, column_classes, runway_names,
)
from ts_transformer.prior.model import Prior, with_traffic
from ts_transformer.prior.train import RewardConfig
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-traffic-window-probe-readout-v1"
#: What a stored sentence is compared on: ``sentences.npz``'s arrays besides the words (`traffic_window_reward.
#: write_sentences`) and, from ``sentences.json``'s aircraft, what decides which of its rows are read.
COMPARED = ("window", "dataset_id", "sample", "outcome", "runway", "reward", "probed", "given", "forced", "advantage",
            "probe_gain")
FROM_JSON = ("counted", "kind", "starts_in_a_loss")


def run_path(path: str) -> Path:
    """A path a run recorded (absolute, from the checkout it ran in) in this checkout (`outputs_path`)."""
    return REPO_ROOT / outputs_path(path)


def stored_rows(npz: Mapping[str, np.ndarray], aircraft: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``sentences.npz`` as a row per sentence, its words over ``step_offsets``, with ``sentences.json``'s
    `FROM_JSON` fields (its ``aircraft``, in the same order)."""
    offsets = npz["step_offsets"]
    if len(aircraft) != len(npz["window"]):
        raise ValueError(f"sentences.json holds {len(aircraft)} aircraft sentences, sentences.npz {len(npz['window'])}")
    return [{**{name: npz[name][i].item() for name in COMPARED}, **{name: aircraft[i][name] for name in FROM_JSON},
             "said": npz["said"][offsets[i]: offsets[i + 1]]} for i in range(len(npz["window"]))]


def spoken_rows(spoken: WindowSentences, advantages: np.ndarray, gains: np.ndarray) -> list[dict[str, Any]]:
    """A round spoken again as ``sentences.npz`` keeps it (`traffic_window_reward.write_sentences`)."""
    return [{"window": r["window"], "dataset_id": r["dataset_id"], "sample": r["sample"], "outcome": r["outcome"],
             "runway": r["runway"], "reward": r["reward"], "probed": r["probed"],
             "given": r["given"], "forced": -1 if r["forced"] is None else r["forced"], "advantage": float(a),
             "probe_gain": float(g), **{name: r[name] for name in FROM_JSON},
             "said": record.grid[: r["said_steps"]].astype(np.int16)}
            for r, record, a, g in zip(spoken.rows, spoken.records, advantages, gains)]


def require_reproduced(round_number: int, stored: Sequence[Mapping[str, Any]], again: Sequence[Mapping[str, Any]]
                       ) -> None:
    """Every sentence spoken again is the stored one (module docstring), refused at the first that is not."""
    if len(stored) != len(again):
        raise ValueError(f"round {round_number}: {len(stored)} sentences stored, {len(again)} spoken again")
    for i, (a, b) in enumerate(zip(stored, again)):
        differ = [name for name in (*COMPARED, *FROM_JSON) if a[name] != b[name]]
        if not np.array_equal(a["said"], b["said"]):
            differ.append("said")
        if differ:
            raise ValueError(f"round {round_number}, sentence {i} ({a['dataset_id']}, sample {a['sample']}): {differ} "
                             f"differ from the stored — not the run's sentences")


def _cell(entry: Mapping[str, Any], name: str) -> str:
    return "—" if entry["n"] == 0 else f"{entry[name]:.2e}"


def summarise(values: Sequence[float]) -> dict[str, Any]:
    """Probabilities' count, mean, median and quartiles (none: count 0)."""
    if not len(values):
        return {"n": 0}
    p = np.asarray(values)
    return {"n": len(p), "mean": float(p.mean()), "p25": float(np.percentile(p, 25)), "median": float(np.median(p)),
            "p75": float(np.percentile(p, 75))}


def readout(rows: Sequence[Mapping[str, Any]], models: Sequence[str]) -> dict[str, Any]:
    """Per round of sentences, per model: the go-around probability on the learned sentences, the others and all, and
    how many rose against round 0 (the start)."""
    out: dict[str, Any] = {}
    for round_number in sorted({r["round"] for r in rows}):
        mine = [r for r in rows if r["round"] == round_number]
        part: dict[str, Any] = {"sentences": len(mine), "learned": sum(r["learned"] for r in mine)}
        for model in models:
            entry = {}
            for name, group in (("learned", [r for r in mine if r["learned"]]),
                                ("not_learned", [r for r in mine if not r["learned"]]), ("all", mine)):
                entry[name] = {**summarise([math.exp(r["log_p"][model]) for r in group]),
                               "rose_against_round_0": sum(r["log_p"][model] > r["log_p"][models[0]] for r in group)}
            part[model] = entry
        out[f"round_{round_number:02d}_sentences"] = part
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--run", type=Path, required=True, help="an R37 run directory (its config.json, rounds)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--workers", type=int, default=3, help="speaking processes (what is spoken does not depend "
                        "on it)")
    args = parser.parse_args(argv)
    run = args.run if args.run.is_absolute() else REPO_ROOT / args.run
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    if args.workers < 1:
        parser.error("at least one speaking process")
    record = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if record["schema"] != RUN_SCHEMA:
        parser.error(f"{run} is a {record['schema']} run, not {RUN_SCHEMA}")
    if record["probes"]["samples"] == 0:
        parser.error(f"{run} probed no sample: no probe's go-around to read")
    if record["events"]["runs"]:
        parser.error("a run with hard events is not read: its rounds draw event windows from R43 runs")
    if record["edge_source_sha256"] != edge_source_sha256() or tuple(record["edge_features"]) != EDGE_FEATURES:
        parser.error("the run's edge features are not today's code's")
    rounds = list(range(1, completed_rounds(run) + 1))       # the finished rounds (an aborted one is set aside)
    if not rounds:
        parser.error(f"{run} finished no training round")
    started = time.perf_counter()
    git = git_state()
    device = torch.device(record["device"])                 # the run's: CPU and CUDA sample other streams
    prior_dir, base_dir = run_path(record["prior"]["directory"]), run_path(record["base"]["directory"])
    instructions, executor_dir = run_path(record["instructions"]), run_path(record["executor"]["directory"])
    for name, directory in (("prior", prior_dir), ("base", base_dir)):
        if file_sha256(directory / "checkpoint.pt") != record[name]["checkpoint_sha256"]:
            parser.error(f"{directory}'s checkpoint is not the one the run read")
    params, executor_record, words = replay.open_executor(executor_dir, instructions)
    if executor_record["sha256"] != record["executor"]["sha256"]:
        parser.error("the executor spec is not the one the run read")
    spec = load_spec(instructions)
    step_s = spec.step_s
    seed = record["seed"]
    # the run's models, on the CPU until the speaking processes are forked: the start as the run built it, then rounds
    loaded, payload, _, start_masks = load_prior(prior_dir, instructions)
    if payload["schema"] == PRIOR_CHECKPOINT_SCHEMA:
        torch.manual_seed(seed)
        start = with_traffic(loaded, EDGE_FEATURES)
    else:
        start = loaded
    models: dict[str, Prior] = {"round_00": start.eval()}
    for k in rounds:
        models[f"round_{k:02d}"] = load_prior(run / f"round_{k:02d}", instructions)[0].eval()
    base = load_prior(base_dir, instructions)[0].eval()
    variant, airports, slots = start.config.variant, start.config.airports, start.config.candidate_slots
    every_landing = airport_landings(instructions, rosters(instructions))
    landings = every_landing if VARIANTS[variant].landing_context else None
    geometries = load_candidates(instructions)
    table = Split([], airports, candidate_table(geometries, airports, slots), *runway_names(geometries, airports),
                  column_classes(words, slots), variant)
    train_flights, _ = airport_flights(instructions, "train", spec, airports, landings, start.config.max_rows)
    source = RoundSource(instructions, spec, words, with_tracks(instructions, "train", spec, train_flights), airports,
                         params, busiest(instructions, spec, airports, step_s), start.config.max_rows,
                         start_altitude_windows(instructions), seed, record["real_per_airport"],
                         record["augmented_per_airport"], record["commanded"], [], 0)
    probes = record["probes"]
    speaking = WindowSpeaking(words, params, landings, start_masks, record["aircraft_steps"], every_landing,
                              probes["samples"], probes["margin"])
    samples = record["samples"]
    print(f"{len(rounds)} training rounds of {run}; models {list(models)}; {time.perf_counter() - started:.0f}s",
          flush=True)

    # the rounds spoken again, each by the model that spoke it, every sentence checked against the stored one
    free_gb, needed_gb = psutil.virtual_memory().available / 1e9, args.workers * SPEAKER_HOST_GB + PARENT_GROWTH_GB
    if free_gb < needed_gb:
        parser.error(f"{free_gb:.1f} GB of the host's memory free, {needed_gb:.1f} GB needed for {args.workers} "
                     f"speaking processes (R37's SPEAKER_HOST_GB, PARENT_GROWTH_GB): fewer --workers")
    spoken_rounds: dict[int, tuple[WindowRound, WindowSentences, np.ndarray, np.ndarray, np.ndarray]] = {}
    for k in rounds:
        round_, _ = round_windows(source, k)
        speaker = models[f"round_{k - 1:02d}"]
        plan = speaking.plan(round_, samples)
        gc.collect()
        gc.freeze()                                     # the speaking processes share the parent's data, not copy it

        def speak(number: int, round_: WindowRound = round_, speaker: Prior = speaker,
                  plan: list[list[int]] = plan, k: int = k) -> WindowSentences:
            return speaking.speak(speaker.to(device), round_, plan[number], number, samples,
                                  seed=round_seed(seed, k, SAMPLING_STREAM), source="train")

        parts = {number: got for number, got, _ in in_processes(args.workers, list(range(len(plan))), speak)}
        spoken = speaking.assemble(round_, samples, plan, parts)
        advantages, gains, trained = window_advantages(spoken.rows, samples)
        directory = run / f"round_{k:02d}"
        stored = stored_rows(np.load(directory / "sentences.npz"),
                             json.loads((directory / "sentences.json").read_text(encoding="utf-8"))["aircraft"])
        require_reproduced(k, stored, spoken_rows(spoken, advantages, gains))
        spoken_rounds[k] = (round_, spoken, advantages, gains, trained)
        print(f"round {k}: {len(spoken.rows)} sentences spoken again, every one the stored; "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    # every forced sentence read by every model
    config = RewardConfig(**record["optimiser"])
    rows: list[dict[str, Any]] = []
    empty = []
    for k, (round_, spoken, advantages, gains, trained) in spoken_rounds.items():
        forced = np.array([i for i, r in enumerate(spoken.rows) if r["forced"] is not None
                           and r["forced"] < r["counted"]], dtype=np.int64)
        if not len(forced):
            empty.append(k)
            print(f"round {k}: no probe said a go-around inside a sentence's counted steps", flush=True)
            continue
        split, _ = window_split(round_, spoken, advantages, gains, forced, table, landings, step_s)
        # (window sample, place) → the round's sentence: `window_split` lists the samples and places in the rows' order
        places = [(s, place) for s, trained in enumerate(split.trained) for place in trained]
        sentence_of = dict(zip(places, (int(i) for i in forced)))
        if len(places) != len(forced) or any(split.records[s][place] is not spoken.records[i]
                                             for (s, place), i in sentence_of.items()):
            raise ValueError(f"round {k}: the split's sentences are not the forced rows in order")
        log_p: dict[int, dict[str, float]] = {int(i): {} for i in forced}
        for name, model in models.items():
            tuner = WindowRewardTuner(model.to(device), base.to(device), config, device, seed=seed,
                                      traffic_learning_rate=record["traffic_learning_rate"], step_s=step_s,
                                      imitation_weight=probes["imitation_weight"])
            for key, value in tuner.forced_log_probs(split).items():
                log_p[sentence_of[key]][name] = value
            model.to("cpu")
        learned = set(trained.tolist())             # trained on (`window_advantages`) with a probe gain above 0
        for i in forced:
            r = spoken.rows[i]
            rows.append({"round": k, "window": r["window"], "dataset_id": r["dataset_id"], "sample": r["sample"],
                         "kind": r["kind"], "forced": r["forced"], "reward": r["reward"], "outcome": r["outcome"],
                         "probe_gain": float(gains[i]), "learned": bool(int(i) in learned and gains[i] > 0.0),
                         "log_p": log_p[int(i)]})
        print(f"round {k}: {len(forced)} forced sentences read by {list(models)}; "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    out.mkdir(parents=True)
    with (out / "forced.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result = {"schema": SCHEMA, "written_utc": utc_now(), "git": git, "run": repo_relative(run),
              "rounds": rounds, "models": list(models), "device": device.type,
              "reproduced": {f"round_{k:02d}": len(entry[1].rows) for k, entry in spoken_rounds.items()},
              "rounds_without_a_forced_sentence": empty,
              "readout": readout(rows, list(models)), "elapsed_s": time.perf_counter() - started}
    write_json_atomic(out / "probe_readout.json", result, allow_nan=False)
    for part, entry in result["readout"].items():
        print(f"{part}: {entry['sentences']} forced, {entry['learned']} learned")
        for model in models:
            cells = "  ".join(f"{group} p50 {_cell(entry[model][group], 'median')} mean "
                              f"{_cell(entry[model][group], 'mean')} rose {entry[model][group]['rose_against_round_0']}"
                              for group in ("learned", "not_learned"))
            print(f"  {model}: {cells}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
