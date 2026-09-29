"""Read a finished M4 run (`traffic_reward`, R32) round by round for its readout (multi-aircraft design §7).

Everything comes from the run's own files — ``history.json``, ``choice.json``, each round's ``readout.json`` (the select
flights) and ``sentences.npz`` (the training sentences) — plus, for the real select flights, the M3 real-scene
free-generation directory (`traffic_free_generation`, R31) that flew the same flights on their labelled words and along
their records:

- **select, per round**: each side's reward, lost separation (both readings), landed and observed-runway shares (the
  run's own numbers), and lost separation by approach type (real) and by kind (augmented) from the flights — a flight
  starting in a loss it answers for is left out, as the run's separation readout leaves it out; sentences with a
  go-around; the traffic attention's output over the residual stream, per layer;
- **training, per round ≥ 1**: the round's sentence summary and pass (the run's own numbers), and where the reward
  term's signal comes from — each scene's ``K`` sentences in one of five classes (all rewarded; all lost separation;
  all failed otherwise or mixed failures; rewards differing with a sentence that lost separation; rewards differing
  with none that did), per kind, with each class's share of the summed |advantage|. The scenes starting in a loss (not
  trained on; the npz does not mark them) are counted in their class; their number is printed beside;
- **the select flights over all rounds**: how many of a flight's tries (rounds × samples) lost separation — never,
  sometimes, or in at least `ALMOST_ALWAYS` of them — and, per group, its share of vectored approaches; for the real
  flights also how often the same flight lost it on its labelled words and along its record (M3). The augmented
  scenes are drawn once, the same every round.

Writes ``traffic_reward_readout.json`` into a NEW ``--out``.

    python run_ts.py traffic_reward_readout --run 4dTrajectory/outputs/POOLED/prior/m4_traffic_<date>/traffic_s1337 \\
        --reference 4dTrajectory/outputs/POOLED/traffic/free_generation_20260928 --out <new directory>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ts_transformer.experiments.traffic_free_generation import SCHEMA as FREE_GENERATION_SCHEMA
from ts_transformer.experiments.traffic_reward import TRAFFIC_REWARD_SCHEMA, completed_rounds
from ts_transformer.inference.separation import VISUAL
from ts_transformer.instructions.words import APPROACH_GO_AROUND, COLUMNS
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import git_state, repo_relative

SCHEMA = "ts-traffic-reward-readout-v1"
LOST = "lost_separation"
SIDES = ("real", "augmented")
#: A flight losing separation in at least this share of its tries is "almost always" lost.
ALMOST_ALWAYS = 5 / 6
CLASSES = ("all rewarded", "all lost separation", "all failed otherwise", "differing, one lost separation",
           "differing, none lost separation")


def scene_class(outcome: np.ndarray, reward: np.ndarray) -> str:
    """One scene's ``K`` sentences in one of `CLASSES`."""
    lost = outcome == LOST
    if reward.min() != reward.max():
        return CLASSES[3] if lost.any() else CLASSES[4]
    if reward.min() == 1.0:
        return CLASSES[0]
    return CLASSES[1] if lost.all() else CLASSES[2]


def training_signal(sentences: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Where a round's reward-term signal comes from (module docstring), from its ``sentences.npz``."""
    kind, outcome, reward, advantage = (sentences[k] for k in ("kind", "outcome", "reward", "advantage"))
    samples, rest = divmod(len(outcome), len(kind))
    if rest:
        raise ValueError(f"{len(outcome)} sentences over {len(kind)} scenes")
    counts = {k: Counter() for k in (*sorted(set(kind.tolist())), "all")}
    weight = Counter()
    for j, scene_kind in enumerate(kind.tolist()):
        span = slice(j * samples, (j + 1) * samples)
        name = scene_class(outcome[span], reward[span])
        counts[scene_kind][name] += 1
        counts["all"][name] += 1
        weight[name] += float(np.abs(advantage[span]).sum())
    steps = np.diff(sentences["step_offsets"])
    lost = outcome == LOST
    total = sum(weight.values())
    return {"samples": samples,
            "scenes": {k: {"scenes": sum(c.values()), **{name: c[name] / sum(c.values()) for name in CLASSES}}
                       for k, c in counts.items()},
            "advantage_share": {name: (weight[name] / total if total else 0.0) for name in CLASSES},
            "steps_per_sentence": {"lost_separation": float(steps[lost].mean()) if lost.any() else None,
                                   "other": float(steps[~lost].mean())},
            "go_around_steps": int((sentences["said"][:, COLUMNS.index("approach")] == APPROACH_GO_AROUND).sum()),
            "steps": int(len(sentences["said"]))}


def _share_lost(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[bool]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row[VISUAL]["outcome"] == LOST)
    return {name: {"flights": len(v), "lost_separation": sum(v) / len(v)} for name, v in sorted(groups.items())}


def select_round(summary: Mapping[str, Any], flights: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """One round's select readout (module docstring): ``summary`` its ``history.json`` row, ``flights`` its
    ``readout.json`` flights per side."""
    out: dict[str, Any] = {}
    for side in SIDES:
        rows = [r for r in flights[side] if not r["starts_in_a_loss"]]
        every = summary[side]["free_generation"]["all"]
        out[side] = {"reward": summary[side]["reward"],
                     "lost_separation": summary[side]["separation"]["lost_separation"],
                     "lost_separation_ifr": summary[side]["separation"]["lost_separation_ifr"],
                     "landed": every["outcomes"]["landed"],
                     "landed_on_observed_runway": every["landed_on_observed_runway"],
                     "by_stratum": _share_lost(rows, "stratum"),
                     "go_around_sentences": sum(r["go_arounds"] > 0 for r in flights[side])}
        if side == "augmented":
            out[side]["by_kind"] = _share_lost(rows, "kind")
    out["traffic_output_over_residual"] = summary["traffic"]["traffic_output_over_residual"]
    out["teacher_forced_nll"] = summary["traffic"]["teacher_forced"]["nll_per_step"]
    return out


def concentration(tries: Mapping[str, Sequence[bool]], facts: Mapping[str, Mapping[str, bool]]) -> dict[str, Any]:
    """How the losses spread over flights (module docstring): ``tries`` each flight's lost/not per try, in round order;
    ``facts`` each flight's yes/no facts (the same names for every flight), each group's share of which is reported."""
    per_flight = {key: (sum(v), len(v)) for key, v in tries.items()}
    if len({n for _, n in per_flight.values()}) != 1:
        raise ValueError("the flights were not tried equally often")
    n_tries = next(iter(per_flight.values()))[1]
    groups = {"never": lambda k: k == 0, "sometimes": lambda k: 0 < k < ALMOST_ALWAYS * n_tries,
              "almost_always": lambda k: k >= ALMOST_ALWAYS * n_tries}
    total = sum(k for k, _ in per_flight.values())
    out: dict[str, Any] = {"flights": len(per_flight), "tries": n_tries, "losses": total,
                           "histogram": dict(sorted(Counter(k for k, _ in per_flight.values()).items()))}
    for name, test in groups.items():
        keys = [key for key, (k, _) in per_flight.items() if test(k)]
        row = {"flights": len(keys), "flight_share": len(keys) / len(per_flight),
               "loss_share": sum(per_flight[key][0] for key in keys) / total if total else 0.0}
        if keys:
            row.update({field: sum(facts[key][field] for key in keys) / len(keys) for field in facts[keys[0]]})
        out[name] = row
    return out


def reference_outcomes(directory: Path) -> dict[str, dict[str, bool]]:
    """Each real select flight's loss along its record and on its labelled words, from an M3 real-scene
    free-generation directory."""
    header = json.loads((directory / "free_generation.json").read_text())
    if header["schema"] != FREE_GENERATION_SCHEMA or header["augment_seed"] is not None or header["split"] != "select":
        raise SystemExit(f"--reference takes a {FREE_GENERATION_SCHEMA} directory of real select scenes, not "
                         f"{header['schema']} / augment seed {header['augment_seed']} / {header['split']}")
    out: dict[str, dict[str, bool]] = {}
    with (directory / header["flights_file"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["source"] in ("recorded", "labelled"):
                out.setdefault(row["dataset_id"], {})[row["source"]] = row[VISUAL]["outcome"] == LOST
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--run", type=Path, required=True, help="a finished traffic_reward run directory")
    parser.add_argument("--reference", type=Path, required=True,
                        help="the M3 real-scene free-generation directory of the same select flights")
    parser.add_argument("--out", type=Path, required=True, help="a NEW directory")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists; a readout is never overwritten")
    started = time.perf_counter()
    config = json.loads((args.run / "config.json").read_text())
    if config["schema"] != TRAFFIC_REWARD_SCHEMA:
        raise SystemExit(f"--run takes a {TRAFFIC_REWARD_SCHEMA} run, not {config['schema']}")
    last = completed_rounds(args.run)
    history = json.loads((args.run / "history.json").read_text())["rounds"]
    if [row["round"] for row in history] != list(range(last + 1)):
        raise SystemExit(f"history.json holds rounds {[row['round'] for row in history]}, the run finished 0 … {last}")
    references = reference_outcomes(args.reference)

    rounds, tries, vectored = [], {side: {} for side in SIDES}, {side: {} for side in SIDES}
    for number, summary in enumerate(history):
        directory = args.run / f"round_{number:02d}"
        readout = json.loads((directory / "readout.json").read_text())
        flights = {side: readout[side]["flights"] for side in SIDES}
        row = {"round": number, "select": select_round(summary, flights)}
        if number > 0:
            with np.load(directory / "sentences.npz") as sentences:
                row["training"] = {"sentences": summary["sentences"], "pass": {
                    k: v for k, v in summary["train_pass"].items() if k != "kl_trace"},
                    "signal": training_signal(sentences)}
        rounds.append(row)
        for side in SIDES:
            for flight in flights[side]:
                if not flight["starts_in_a_loss"]:
                    key = f"{flight['dataset_id']}|{flight['kind']}"
                    tries[side].setdefault(key, []).append(flight[VISUAL]["outcome"] == LOST)
                    vectored[side][key] = flight["stratum"] == "vectored"
        print(f"round {number}: real reward {summary['real']['reward']:.3f} lost "
              f"{summary['real']['separation']['lost_separation']:.3f}, augmented reward "
              f"{summary['augmented']['reward']:.3f} lost {summary['augmented']['separation']['lost_separation']:.3f}",
              flush=True)

    real_keys = {key.split("|")[0] for key in tries["real"]}
    missing = sorted(real_keys - {k for k, v in references.items() if set(v) == {"recorded", "labelled"}})
    if missing:
        raise SystemExit(f"{len(missing)} real select flights have no record / labelled reading in --reference, "
                         f"e.g. {missing[:3]}")
    spread = {"real": concentration(tries["real"], {key: {"vectored": vectored["real"][key],
                                                           **references[key.split("|")[0]]} for key in tries["real"]}),
              "augmented": concentration(tries["augmented"], {key: {"vectored": v}
                                                              for key, v in vectored["augmented"].items()})}
    for side in SIDES:
        print(f"{side}: " + ", ".join(f"{g} {spread[side][g]['flight_share']:.1%} of flights hold "
                                      f"{spread[side][g]['loss_share']:.1%} of losses"
                                      for g in ("never", "sometimes", "almost_always")), flush=True)
    args.out.mkdir(parents=True)
    write_json_atomic(args.out / "traffic_reward_readout.json", {
        "schema": SCHEMA, "written_utc": utc_now(), "git": git_state(), "run": repo_relative(args.run),
        "reference": repo_relative(args.reference), "choice": json.loads((args.run / "choice.json").read_text()),
        "almost_always": ALMOST_ALWAYS, "rounds": rounds, "spread": spread,
        "seconds": time.perf_counter() - started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
