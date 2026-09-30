"""Multi-aircraft M4's validation readout (design §6.2 "验证集"): the round a finished `traffic_reward` run kept, read once
on a split beside the start (round 0) — the run's own select reading, on other days.

The run's reading exactly (`traffic_reward`'s select readouts): the split's ``per_airport`` flights of each airport drawn
with the run's seed (`autopilot.replay.draw`: their own dynamics), each read in its real scene and in one augmentation
(the run's seed + `SELECT_AUGMENT_OFFSET`: the leader moved, the start moved, a flight inserted, a third each), every
scene spoken to ``samples`` times by the run's speaking processes (`traffic_reward.Speakers`), each side from the run's
select stream — so the start and the kept round speak the same random numbers and compare sentence by sentence
(`traffic_reward.paired_difference`). Per model and side, `traffic_reward.side_readout` (the reward, separation, outcomes,
words, on the real side the ordering) and the teacher-forced NLL on the split's scene samples with the traffic
attention's output against the residual (`traffic_reward.traffic_readout`); beside them the same flights flying their
labelled words and along their records (`traffic_free_generation`: the real scenes).

The run is read by content: its start (``--prior``: the checkpoint's sha256), executor spec (sha256), instruction
artefact (its name under ``4dTrajectory/outputs``) and edge code (`traffic_scene_data.edge_source_sha256`) must be the
run's; the run finished (every round it asked for) and kept a round (``choice.json``). ``--split select`` reads what the
run read: each model's sentences are checked against the run's round-0 and kept-round select readouts (the share said
alike, written — on the GPU a kernel's sums may differ at rounding between two runs); ``--split val`` reads the
validation days once, from a clean tree.

    python run_ts.py traffic_reward_val --run 4dTrajectory/outputs/POOLED/prior/m4_passes_20260929/traffic_s1337 \\
        --prior 4dTrajectory/outputs/POOLED/prior/v3_stage2_clip_20260926/aug_s1337/round_07 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v11_20260927 \\
        --split val --out 4dTrajectory/outputs/POOLED/traffic/m4_passes_val_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ts_transformer.autopilot import replay
from ts_transformer.experiments.prior_augmented_reward import labelled_words
from ts_transformer.experiments.prior_free_generation import start_altitude_windows
from ts_transformer.experiments.prior_train import PRIOR_CHECKPOINT_SCHEMA, load_prior, rosters
from ts_transformer.experiments.traffic_free_generation import labelled_rows, recorded_rows, summary
from ts_transformer.experiments.traffic_reward import (
    SELECT_AUGMENT_OFFSET, SELECT_STREAMS, TRAFFIC_REWARD_SCHEMA, Round, Speakers, Speaking, augmented_round,
    completed_rounds, paired_difference, real_round, round_seed, side_readout, traffic_readout,
)
from ts_transformer.experiments.traffic_reward_readout import artefact_name
from ts_transformer.experiments.traffic_scene_data import airport_flights, edge_source_sha256, split_samples
from ts_transformer.experiments.traffic_speaking import with_tracks
from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.artefact import SPLITS, load_spec
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.data import VARIANTS, airport_landings
from ts_transformer.prior.model import Prior, with_traffic
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-traffic-reward-val-v1"
SIDES = ("real", "augmented")


def run_differences(config: Mapping[str, Any], prior_sha256: str, executor_sha256: str, instructions: Path) -> list[str]:
    """What of the start, the executor, the instruction artefact and the edge code differs from the run's."""
    out = []
    if prior_sha256 != config["prior"]["checkpoint_sha256"]:
        out.append("the start's checkpoint")
    if executor_sha256 != config["executor"]["sha256"]:
        out.append("the executor spec")
    if artefact_name(str(instructions)) != artefact_name(config["instructions"]):
        out.append("the instruction artefact")
    if edge_source_sha256() != config["edge_source_sha256"]:
        out.append("the edge code")
    return out


def sentence_keys(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    """A side's sentences by flight, kind and sample."""
    return {(r["dataset_id"], r["kind"], r["sample"]): r for r in rows}


def rewards(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], float]:
    """The reward of each sentence the reward counts (`traffic_reward.select_rewards`' reading)."""
    return {key: r["reward"] for key, r in sentence_keys(rows).items() if not r["starts_in_a_loss"]}


def said_alike(ours: Sequence[Mapping[str, Any]], theirs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How many of a side's sentences end as the run's own readout ended them: the same outcome, reward and steps."""
    mine, run = sentence_keys(ours), sentence_keys(theirs)
    if mine.keys() != run.keys():
        raise SystemExit("the sentences read are not the run's: another draw of flights or scenes")
    alike = sum((mine[k]["outcome"], mine[k]["reward"], mine[k]["steps_said"])
                == (run[k]["outcome"], run[k]["reward"], run[k]["steps_said"]) for k in mine)
    return {"sentences": len(mine), "alike": alike}


def _no_training_round(key: int) -> Round:
    raise AssertionError("the validation readout speaks no training round")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--run", type=Path, required=True, help="a finished traffic_reward run directory")
    parser.add_argument("--prior", type=Path, required=True, help="the run's start (augmented)")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=[s for s in SPLITS if s != "test"], required=True)
    parser.add_argument("--speakers", type=int, default=4, help="speaking processes (the sentences do not depend on it)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executor flies on CPU")
    parser.add_argument("--out", type=Path, required=True, help="a NEW directory")
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    run, prior_dir, instructions, executor_dir, out = map(
        resolved, (args.run, args.prior, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    if args.split == "val" and git["dirty"]:
        parser.error("the val split is read once, from a clean tree")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config["schema"] != TRAFFIC_REWARD_SCHEMA or config["smoke"]:
        parser.error(f"--run takes a formal {TRAFFIC_REWARD_SCHEMA} run")
    last = completed_rounds(run)
    if config["rounds"] != last:
        parser.error(f"the run asked for rounds 0 … {config['rounds']} and finished 0 … {last}: not finished")
    choice = json.loads((run / "choice.json").read_text(encoding="utf-8"))
    if len(choice["augmented_reward"]) != last + 1:
        parser.error("choice.json does not cover the run's finished rounds")
    kept_round = int(choice["round"])
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor_dir, instructions)
    differs = run_differences(config, file_sha256(prior_dir / "checkpoint.pt"), record["sha256"], instructions)
    if differs:
        parser.error(f"not the run's: {differs}")
    spec = load_spec(instructions)
    step_s, seed = spec.step_s, int(config["seed"])
    per_airport, samples = int(config["select"]["per_airport"]), int(config["select"]["samples"])
    single, payload, start_config, masks = load_prior(prior_dir, instructions)
    if start_config["smoke"] or payload["schema"] != PRIOR_CHECKPOINT_SCHEMA:
        parser.error(f"{prior_dir} is not a single-aircraft prior's formal run")
    torch.manual_seed(seed)                              # the run's start: its traffic attention at zero
    models: dict[str, Prior] = {"start": with_traffic(single, EDGE_FEATURES)}
    kept_dir = run / f"round_{kept_round:02d}"
    if kept_round:
        kept, _, _, kept_masks = load_prior(kept_dir, instructions)
        if kept_masks.names != masks.names:
            parser.error(f"the kept round speaks under {kept_masks.names}, the start under {masks.names}")
        models["kept"] = kept
    model = models["start"]
    variant, airports, slots = model.config.variant, model.config.airports, model.config.candidate_slots
    every_landing = airport_landings(instructions, rosters(instructions))
    landings = every_landing if VARIANTS[variant].landing_context else None
    flights, counts = airport_flights(instructions, args.split, spec, airports, landings, model.config.max_rows)
    scene_airports = with_tracks(instructions, args.split, spec, flights)
    built = [b for b in split_samples(flights, counts, step_s)[0] if b.sample.asks]
    batch = replay.draw(instructions, args.split, spec, words, per_airport=per_airport, seed=seed)
    rounds = {"real": real_round(batch, scene_airports, params, every_landing, step_s)}
    rounds["augmented"], left_out = augmented_round(batch, scene_airports, params, every_landing,
                                                    seed + SELECT_AUGMENT_OFFSET, start_altitude_windows(instructions),
                                                    step_s)
    references = {"labelled": summary(labelled_rows(batch, rounds["real"].scenes, words, params, masks))["labelled"],
                  "recorded": summary(recorded_rows(batch, rounds["real"].scenes, words))["recorded"],
                  "labelled_words_after_first_per_flight": {side: labelled_words(rounds[side].batch) for side in SIDES}}
    print(f"{args.split}: {len(batch.readings)} flights, {len(rounds['augmented'].scenes)} augmented ({left_out} left "
          f"out), {len(built)} scene samples; kept round {kept_round}; {time.perf_counter() - started:.0f}s", flush=True)
    gc.collect()
    gc.freeze()                                          # the speaking processes share the parent's data, not copy it
    speakers = Speakers(args.speakers, _no_training_round, rounds, model, args.device,
                        Speaking(words, params, landings, masks, int(config["aircraft_steps"])))
    device = torch.device(args.device)
    readout: dict[str, Any] = {}
    try:
        for name, speaking in models.items():
            readout[name] = {}
            for side in SIDES:
                spoken, peaks = speakers.speak("select", side, rounds[side], speaking, samples,
                                               seed=round_seed(seed, 0, SELECT_STREAMS[side]), source="scene")
                readout[name][side] = {**side_readout(spoken.rows, rounds[side], samples, step_s, real=side == "real"),
                                       "speaking_gpu_peak_gb": peaks, "flights": spoken.rows}
                print(f"{name} {side}: reward {readout[name][side]['reward']:.4f}, lost separation "
                      f"{readout[name][side]['separation']['lost_separation']:.4f}  "
                      f"[{time.perf_counter() - started:.0f}s]", flush=True)
    finally:
        speakers.close()
    for name, speaking in models.items():
        readout[name]["traffic"] = traffic_readout(speaking.to(device).eval(), built, slots, device)
        speaking.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    paired = {}
    if "kept" in models:
        for side in SIDES:
            difference, error = paired_difference(rewards(readout["start"][side]["flights"]),
                                                  rewards(readout["kept"][side]["flights"]))
            paired[side] = {"reward_difference": difference, "standard_error": error}
    check = None
    if args.split == "select":
        check = {name: {side: said_alike(readout[name][side]["flights"], json.loads(
            (run / f"round_{0 if name == 'start' else kept_round:02d}" / "readout.json").read_text())[side]["flights"])
                        for side in SIDES} for name in models}
    out.mkdir(parents=True)
    write_json_atomic(out / "val_readout.json", {
        "schema": SCHEMA, "written_utc": utc_now(), "git": git, "split": args.split, "run": repo_relative(run),
        "kept_round": kept_round, "kept": None if not kept_round else {
            "directory": repo_relative(kept_dir), "checkpoint_sha256": file_sha256(kept_dir / "checkpoint.pt")},
        "start": {"directory": repo_relative(prior_dir), "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt"),
                  "procedure_masks": list(masks.names)},
        "executor": {"directory": repo_relative(executor_dir), "sha256": record["sha256"]},
        "instructions": repo_relative(instructions), "per_airport": per_airport, "samples": samples, "seed": seed,
        "drawn": batch.drawn, "augment_seed": seed + SELECT_AUGMENT_OFFSET, "augmented_left_out": left_out,
        "scene_samples": len(built), "references": references, "models": readout, "paired": paired,
        "against_the_run": check, "elapsed_s": time.perf_counter() - started})
    for side in SIDES:
        line = "  ".join(f"{name} reward {readout[name][side]['reward']:.4f} lost "
                         f"{readout[name][side]['separation']['lost_separation']:.4f}" for name in models)
        print(f"{side:9s} {line}  paired {paired.get(side)}")
    if check is not None:
        print(f"against the run's own select readouts: {check}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
