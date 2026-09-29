"""Read a finished M4 run (`traffic_reward`, R32) round by round for its readout (multi-aircraft design §7).

Everything comes from the run's own files — ``config.json``, ``history.json``, ``choice.json``, each round's
``readout.json`` (the select flights) and ``sentences.npz`` (the training sentences) — plus, for the real select flights,
the M3 real-scene free-generation directory (`traffic_free_generation`, R31) that flew the same flights on their labelled
words and along their records with the same executor (its sha256) and instruction artefact (its place under
`4dTrajectory/outputs`, where directories are never renamed — a path through a removed worktree still names it) —
refused otherwise, and its records must reproduce the run's own recorded reading of the select flights. The run must be
finished: every round it asked for (``config.rounds``) read, ``history.json`` and ``choice.json`` covering them all.

- **select, per round**: each side's reward, lost separation (both readings), landed and observed-runway shares (the
  run's own numbers); lost separation by approach type (real) and by kind (augmented) from the sentences — a sentence
  starting in a loss it answers for left out, as the run's separation readout leaves it out; the sentences with a
  go-around (said anywhere to the executor's end); the traffic attention's output over the residual stream, per layer;
- **training, per round ≥ 1**: the round's sentence summary and pass (the run's own numbers, traces left out, each
  sweep's kept), and where the reward term's signal comes from — each scene's ``K`` sentences in one of `CLASSES`, per
  kind, and each class's share of the summed |advantage| over all scenes. A scene starting in a loss is not trained on
  and the npz does not mark it: it is counted in its class, and ``differing_scenes`` beside the run's
  ``scenes_with_contrast`` shows how many;
- **the select flights, first round against last**: every round speaks the select scenes with the SAME random streams
  (`traffic_reward`: the select generator is seeded alike every round), so a flight's samples are the same draws each
  round and differ only as the model has moved — rounds are paired, not repeated tries. Reported: the share of (flight,
  sample) pairs whose outcome is the same in every round; the paired change in lost separation from round 0 to the last
  (sentences fixed, newly lost, the net and its standard error ``√(fixed + newly lost) / n``); and in round 0 and in the
  last round the flights grouped by how many of their samples lost separation, each group's share of flights and of
  losses, and on the real side its share of vectored approaches and how often the same flights lost it on their labelled
  words and along their records. A flight with any sentence starting in a loss in any round (the first step's runway word
  decides the runway it is judged against, so it can differ by sample and by round), or whose record or labelled reading
  starts in one, is left out of this part and counted. The last round is not necessarily the kept one (``choice.round``).

Writes ``traffic_reward_readout.json`` into a NEW ``--out``.

    python run_ts.py traffic_reward_readout --run 4dTrajectory/outputs/POOLED/prior/m4_traffic_<date>/traffic_s1337 \\
        --reference 4dTrajectory/outputs/POOLED/traffic/free_generation_20260928 --out <new directory>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ts_transformer.experiments.traffic_free_generation import SCHEMA as FREE_GENERATION_SCHEMA
from ts_transformer.experiments.traffic_free_generation import SOURCES
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
from ts_transformer.experiments.traffic_reward import TRAFFIC_REWARD_SCHEMA, completed_rounds
from ts_transformer.inference.separation import VISUAL
from ts_transformer.instructions.readout import STRATA
from ts_transformer.instructions.words import APPROACH_GO_AROUND, COLUMNS
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-traffic-reward-readout-v1"
SIDES = ("real", "augmented")
RECORDED, LABELLED = SOURCES[3], SOURCES[2]
VECTORED = STRATA[1]
CLASSES = ("all rewarded", "all lost separation", "all failed otherwise", "differing, at least one lost separation",
           "differing, none lost separation")


def _lost(row: Mapping[str, Any]) -> bool:
    return row[VISUAL]["outcome"] == LOST_SEPARATION


def scene_class(outcome: np.ndarray, reward: np.ndarray) -> str:
    """One scene's ``K`` sentences in one of `CLASSES`."""
    lost = outcome == LOST_SEPARATION
    if reward.min() != reward.max():
        return CLASSES[3] if lost.any() else CLASSES[4]
    if reward.min() == 1.0:
        return CLASSES[0]
    return CLASSES[1] if lost.all() else CLASSES[2]


def training_signal(sentences: Mapping[str, np.ndarray], samples: int) -> dict[str, Any]:
    """Where a round's reward-term signal comes from (module docstring), from its ``sentences.npz``; ``samples`` the
    run's sentences per scene."""
    kind, outcome, reward, advantage = (sentences[k] for k in ("kind", "outcome", "reward", "advantage"))
    if len(outcome) != len(kind) * samples:
        raise ValueError(f"{len(outcome)} sentences, not {samples} for each of {len(kind)} scenes")
    counts = {k: Counter() for k in (*sorted(set(kind.tolist())), "all")}
    weight = Counter()
    for j, scene_kind in enumerate(kind.tolist()):
        span = slice(j * samples, (j + 1) * samples)
        name = scene_class(outcome[span], reward[span])
        counts[scene_kind][name] += 1
        counts["all"][name] += 1
        weight[name] += float(np.abs(advantage[span]).sum())
    steps = np.diff(sentences["step_offsets"])
    lost = outcome == LOST_SEPARATION
    total = sum(weight.values())
    return {"scenes": {k: {"scenes": sum(c.values()), **{name: c[name] / sum(c.values()) for name in CLASSES}}
                       for k, c in counts.items()},
            "differing_scenes": counts["all"][CLASSES[3]] + counts["all"][CLASSES[4]],
            "advantage_share": {name: weight[name] / total for name in CLASSES} if total else None,
            "steps_per_sentence": {"lost_separation": float(steps[lost].mean()) if lost.any() else None,
                                   "other": float(steps[~lost].mean())},
            "go_around_steps": int((sentences["said"][:, COLUMNS.index("approach")] == APPROACH_GO_AROUND).sum()),
            "steps": int(len(sentences["said"]))}


def _share_lost(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[bool]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(_lost(row))
    return {name: {"sentences": len(v), "lost_separation": sum(v) / len(v)} for name, v in sorted(groups.items())}


def select_round(summary: Mapping[str, Any], flights: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """One round's select readout (module docstring): ``summary`` its ``history.json`` row, ``flights`` its
    ``readout.json`` rows per side."""
    out: dict[str, Any] = {}
    for side in SIDES:
        rows = [r for r in flights[side] if not r["starts_in_a_loss"]]
        every = summary[side]["free_generation"]["all"]
        out[side] = {"reward": summary[side]["reward"],
                     "lost_separation": summary[side]["separation"]["lost_separation"],
                     "lost_separation_ifr": summary[side]["separation"]["lost_separation_ifr"],
                     "landed": every["outcomes"]["landed"],
                     "landed_on_observed_runway": every["landed_on_observed_runway"],
                     "go_around_sentences": sum(r["go_arounds"] > 0 for r in rows)}
        out[side].update({"by_stratum": _share_lost(rows, "stratum")} if side == "real"
                         else {"by_kind": _share_lost(rows, "kind")})
    out["traffic_output_over_residual"] = summary["traffic"]["traffic_output_over_residual"]
    out["teacher_forced_nll"] = summary["traffic"]["teacher_forced"]["nll_per_step"]
    return out


def groups_by_samples_lost(lost: Mapping[str, Sequence[bool]], facts: Mapping[str, Mapping[str, bool]]
                           ) -> dict[str, Any]:
    """One round's flights grouped by how many of their samples lost separation: ``lost`` each flight's per sample,
    ``facts`` each flight's yes/no facts (the same names for every flight), each group's share of which is reported."""
    samples = {len(v) for v in lost.values()}
    if len(samples) != 1:
        raise ValueError(f"the flights have {sorted(samples)} samples, not one number")
    n_samples = samples.pop()
    total = sum(sum(v) for v in lost.values())
    out = {}
    for k in range(n_samples + 1):
        keys = [key for key, v in lost.items() if sum(v) == k]
        row = {"flights": len(keys), "flight_share": len(keys) / len(lost),
               "loss_share": k * len(keys) / total if total else None}
        if keys:
            row.update({field: sum(facts[key][field] for key in keys) / len(keys) for field in facts[keys[0]]})
        out[f"{k}/{n_samples}"] = row
    return out


def paired_change(first: Mapping[str, Sequence[bool]], last: Mapping[str, Sequence[bool]]) -> dict[str, Any]:
    """Round 0 against the last round on the same draws: sentences fixed, newly lost, the net change of the lost share
    and its paired standard error."""
    pairs = [(a, b) for key in first for a, b in zip(first[key], last[key], strict=True)]
    fixed = sum(a and not b for a, b in pairs)
    newly = sum(b and not a for a, b in pairs)
    return {"sentences": len(pairs), "fixed": fixed, "newly_lost": newly, "net": (newly - fixed) / len(pairs),
            "standard_error": math.sqrt(fixed + newly) / len(pairs)}


def spread(per_round: Sequence[Mapping[str, Sequence[bool]]], starts: set[str], reference_starts: set[str],
           facts: Mapping[str, Mapping[str, bool]]) -> dict[str, Any]:
    """The select flights first round against last (module docstring): ``per_round`` each round's lost/not per flight
    and sample; left out: ``starts`` the flights with a sentence starting in a loss in some round, ``reference_starts``
    those whose record or labelled reading starts in one."""
    kept = [{key: v for key, v in lost.items() if key not in starts | reference_starts} for lost in per_round]
    same = [all(lost[key][s] == kept[0][key][s] for lost in kept)
            for key in kept[0] for s in range(len(kept[0][key]))]
    return {"left_out_starting_in_a_loss": len(starts),
            "left_out_reference_starting_in_a_loss": len(reference_starts - starts), "flights": len(kept[0]),
            "same_in_every_round": sum(same) / len(same), "paired_change": paired_change(kept[0], kept[-1]),
            "first_round": groups_by_samples_lost(kept[0], facts), "last_round": groups_by_samples_lost(kept[-1], facts)}


def _path(path: str) -> Path:
    """A path as the writers take it: relative to the repository unless absolute."""
    return Path(path) if Path(path).is_absolute() else REPO_ROOT / path


def artefact_name(path: str) -> str:
    """An artefact's name: its place under the last ``4dTrajectory/outputs`` of its path (module docstring)."""
    parts = Path(path).parts
    at = [i for i in range(len(parts) - 1) if parts[i:i + 2] == ("4dTrajectory", "outputs")]
    if not at:
        raise SystemExit(f"{path} is not under 4dTrajectory/outputs")
    return Path(*parts[at[-1] + 2:]).as_posix()


def reference_outcomes(directory: Path, config: Mapping[str, Any], real_keys: set[str]
                       ) -> tuple[dict[str, dict[str, bool]], set[str]]:
    """``(each real select flight's loss along its record and on its labelled words, the flights one of whose two
    readings starts in a loss)``, from an M3 real-scene free-generation directory flown by the run's executor on its
    instruction artefact; its records must reproduce the run's recorded reading of these flights."""
    header = json.loads((directory / "free_generation.json").read_text())
    if header["schema"] != FREE_GENERATION_SCHEMA or header["augment_seed"] is not None or header["split"] != "select":
        raise SystemExit(f"--reference takes a {FREE_GENERATION_SCHEMA} directory of real select scenes, not "
                         f"{header['schema']} / augment seed {header['augment_seed']} / {header['split']}")
    if header["executor"]["sha256"] != config["executor"]["sha256"] or \
            artefact_name(header["instructions"]) != artefact_name(config["instructions"]):
        raise SystemExit("--reference was flown by another executor or on another instruction artefact than --run")
    out: dict[str, dict[str, bool]] = {}
    recorded: list[bool] = []
    starting: set[str] = set()
    with (directory / header["flights_file"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["source"] not in (RECORDED, LABELLED):
                continue
            entry = out.setdefault(row["dataset_id"], {})
            if row["source"] in entry:
                raise SystemExit(f"--reference holds {row['dataset_id']} {row['source']} twice")
            entry[row["source"]] = _lost(row)
            if row["starts_in_a_loss"]:
                starting.add(row["dataset_id"])
            elif row["source"] == RECORDED and row["dataset_id"] in real_keys:
                recorded.append(_lost(row))
    missing = sorted(key for key in real_keys if key not in out or set(out[key]) != {RECORDED, LABELLED})
    if missing:
        raise SystemExit(f"{len(missing)} real select flights have no record / labelled reading in --reference, "
                         f"e.g. {missing[:3]}")
    stored = config["select"]["recorded"]
    if len(recorded) != stored["flights"] or not math.isclose(sum(recorded) / len(recorded),
                                                              stored["lost_separation"], rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"--reference's records give {sum(recorded)} / {len(recorded)} lost, the run's recorded reading "
                         f"{stored['lost_separation']:.6f} of {stored['flights']}")
    return {key: out[key] for key in real_keys}, starting & real_keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--run", type=_path, required=True, help="a finished traffic_reward run directory")
    parser.add_argument("--reference", type=_path, required=True,
                        help="the M3 real-scene free-generation directory of the same select flights")
    parser.add_argument("--out", type=_path, required=True, help="a NEW directory")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists; a readout is never overwritten")
    started = time.perf_counter()
    config = json.loads((args.run / "config.json").read_text())
    if config["schema"] != TRAFFIC_REWARD_SCHEMA or config["smoke"]:
        raise SystemExit(f"--run takes a formal {TRAFFIC_REWARD_SCHEMA} run, not {config['schema']} "
                         f"(smoke {config['smoke']})")
    last = completed_rounds(args.run)
    history = json.loads((args.run / "history.json").read_text())["rounds"]
    choice = json.loads((args.run / "choice.json").read_text())
    if config["rounds"] != last:
        raise SystemExit(f"the run asked for rounds 0 … {config['rounds']} and finished 0 … {last}: not finished")
    if [row["round"] for row in history] != list(range(last + 1)) or len(choice["augmented_reward"]) != last + 1:
        raise SystemExit(f"history.json / choice.json cover rounds other than the run's finished 0 … {last}")

    rounds: list[dict[str, Any]] = []
    lost: dict[str, list[dict[str, list[bool]]]] = {side: [] for side in SIDES}
    starts: dict[str, set[str]] = {side: set() for side in SIDES}
    vectored: dict[str, bool] = {}
    for number, summary in enumerate(history):
        directory = args.run / f"round_{number:02d}"
        readout = json.loads((directory / "readout.json").read_text())
        flights = {side: readout[side]["flights"] for side in SIDES}
        row = {"round": number, "select": select_round(summary, flights)}
        if number > 0:
            with np.load(directory / "sentences.npz") as sentences:
                row["training"] = {"sentences": summary["sentences"],
                                   "pass": {k: v for k, v in summary["train_pass"].items()
                                            if k == "sweeps" or not isinstance(v, list)},
                                   "signal": training_signal(sentences, config["samples"])}
        rounds.append(row)
        for side in SIDES:
            this: dict[str, list[bool]] = {}
            for flight in sorted(flights[side], key=lambda r: r["sample"]):
                key = f"{flight['dataset_id']}|{flight['kind']}"
                this.setdefault(key, []).append(_lost(flight))
                if flight["starts_in_a_loss"]:
                    starts[side].add(key)
                if side == "real":
                    vectored[key] = flight["stratum"] == VECTORED
            lost[side].append(this)
        print(f"round {number}: real reward {summary['real']['reward']:.3f} lost "
              f"{summary['real']['separation']['lost_separation']:.3f}, augmented reward "
              f"{summary['augmented']['reward']:.3f} lost {summary['augmented']['separation']['lost_separation']:.3f}",
              flush=True)

    real_keys = {key.split("|")[0] for key in lost["real"][0]}
    references, reference_starts = reference_outcomes(args.reference, config, real_keys)
    spreads = {"real": spread(lost["real"], starts["real"], {f"{key}|real" for key in reference_starts},
                              {key: {"vectored": vectored[key], **references[key.split("|")[0]]}
                               for key in lost["real"][0]}),
               "augmented": spread(lost["augmented"], starts["augmented"], set(),
                                   {key: {} for key in lost["augmented"][0]})}
    for side in SIDES:
        change = spreads[side]["paired_change"]
        print(f"{side}: {spreads[side]['same_in_every_round']:.1%} of sentences the same in every round; round 0 → "
              f"{last}: {change['fixed']} fixed, {change['newly_lost']} newly lost, net {change['net']:+.2%} ± "
              f"{change['standard_error']:.2%}", flush=True)
    args.out.mkdir(parents=True)
    write_json_atomic(args.out / "traffic_reward_readout.json", {
        "schema": SCHEMA, "written_utc": utc_now(), "git": git_state(), "run": repo_relative(args.run),
        "reference": repo_relative(args.reference), "last_round": last,
        "choice": choice, "rounds": rounds, "spread": spreads, "seconds": time.perf_counter() - started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
