"""Read an M4-in-windows run (`traffic_window_reward`, R37) round by round, while it runs or after it ends (multi-aircraft
design §6.6 step 7, 7.6) — from the run's own files only.

The rounds read are the finished ones, 0 … k: a round is finished once its ``readout.json`` is written (the runner writes
it last) and each has its ``history.json`` row; a round directory after them without a readout is running or was cut
short — named, not read. Refused when the finished rounds are not 0 … k or one has no history row.

- **select, per round and side** (the round's ``readout.json``; its history row holds the same numbers): reward, lost
  separation, landed and on the observed runway, the ordering against the record (real windows), the reward per kind
  (augmented windows); the traffic attention's output over the residual stream per layer and the teacher-forced NLL;
- **against round 0, paired**: every round speaks the select windows with the same random streams, so an aircraft
  sentence (window, aircraft, sample) is the same draw in every round and differs only as the model has moved — per
  side, and per kind on the augmented side, the change in reward and in lost separation over the sentences both rounds
  count and its standard error — the round choice's pairs (`traffic_window_reward.select_counted`) and difference
  (`traffic_reward.paired_difference`: √(sentences that changed) / sentences), so on the augmented reward these are
  ``choice.json``'s numbers;
- **who the losses were with**: of the counted sentences that lost separation, the share ended by another commanded
  aircraft and by a replayed one (``ended_with``: the loss that ended it, not every episode it was in);
- **select hard events** (a run with ``--select-events``, multi-aircraft step 8 item 11): the answered aircraft's
  reward, landing in the landing direction, lost separation and go-arounds per round, and against round 0 paired as
  above — read, never in the choice;
- **training, per round ≥ 1** (``history.json``): the round's training sentences (windows, aircraft sentences, trained
  on, starting in a loss, reward and lost separation per kind) and its pass (updates, sentences, the reward term, the KL
  to the reference — mean and largest —, the words outside the clip, the data term, the KL to base at the start);
- **the choice** (``choice.json``, written when an invocation of the runner ends, over the rounds finished then — `run.sh`
  runs one round an invocation, so there after every round): the round kept, the rounds the guards excluded; none
  before the first invocation ends, and one covering fewer rounds than are finished is said to be behind.

Prints the tables; ``--out`` (a new directory) also writes ``traffic_window_reward_readout.json``.

    python run_ts.py traffic_window_reward_readout \\
        --run 4dTrajectory/outputs/POOLED/prior/m4_window_20260930/window_s1337
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
from ts_transformer.experiments.traffic_reward import paired_difference
from ts_transformer.experiments.traffic_window_reward import SCHEMA as RUN_SCHEMA
from ts_transformer.experiments.traffic_window_reward import select_counted
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-traffic-window-reward-readout-v2"
SIDES = ("real", "augmented")


def finished_rounds(run: Path) -> tuple[list[int], list[str]]:
    """``(the finished rounds 0 … k, the round directories after them without a readout)``."""
    numbers = sorted(int(path.name[len("round_"):]) for path in run.glob("round_*")
                     if path.name[len("round_"):].isdigit())
    finished = [n for n in numbers if (run / f"round_{n:02d}" / "readout.json").exists()]
    if finished != list(range(len(finished))):
        raise SystemExit(f"{run}: finished rounds {finished}, not 0 … k")
    unfinished = [f"round_{n:02d}" for n in numbers if n not in finished]
    if not finished:
        raise SystemExit(f"{run} has finished no round yet (round 0's readout comes first)")
    return finished, unfinished


def reward(row: Mapping[str, Any]) -> float:
    return row["reward"]


def lost(row: Mapping[str, Any]) -> float:
    return float(row["outcome"] == LOST_SEPARATION)


def paired(first: Sequence[Mapping[str, Any]], then: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """The change from ``first`` to ``then`` in reward and in lost separation over the sentences both count, and its
    standard error (`paired_difference`)."""
    out = {}
    for name, value in (("reward", reward), ("lost_separation", lost)):
        a, b = select_counted(first, value), select_counted(then, value)
        difference, error = paired_difference(a, b)
        out[name] = {"difference": difference, "standard_error": error, "sentences": len(a.keys() & b.keys())}
    return out


def losses_with(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    """Of the counted sentences that lost separation: how many, and the shares ended by another commanded aircraft
    and by a replayed one (``ended_with``; None without a loss)."""
    losing = [r for r in rows if not r["starts_in_a_loss"] and r["outcome"] == LOST_SEPARATION]
    return {"sentences": len(losing),
            **{partner: sum(r["ended_with"] == partner for r in losing) / len(losing) if losing else None
               for partner in ("commanded", "replayed")}}


def by_kind(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {kind: [r for r in rows if r["kind"] == kind] for kind in sorted({r["kind"] for r in rows})}


def select_round(readout: Mapping[str, Any], first: Mapping[str, Any]) -> dict[str, Any]:
    """One round's select numbers per side (``readout``: its ``readout.json``), paired against round 0's (``first``)."""
    out: dict[str, Any] = {}
    for side in SIDES:
        part, rows, rows_0 = readout[side], readout[side]["aircraft"], first[side]["aircraft"]
        general = part["free_generation"]["all"]
        out[side] = {
            "reward": part["reward"], "lost_separation": part["separation"]["lost_separation"],
            "landed": general["outcomes"]["landed"], "landed_on_observed_runway": general["landed_on_observed_runway"],
            "reward_by_kind": part["reward_by_kind"], "against_round_0": paired(rows_0, rows),
            "losses_with": losses_with(rows)}
        if side == "real":
            out[side]["ordering"] = {k: part["ordering"][k] for k in ("time_ratio", "gap_ratio")}
        else:
            kinds, kinds_0 = by_kind(rows), by_kind(rows_0)
            out[side]["against_round_0_by_kind"] = {kind: paired(kinds_0[kind], kinds[kind]) for kind in kinds}
    if "events" in readout:                # the select days' hard events (R37 `--select-events`): read, never chosen on
        part, rows, rows_0 = readout["events"], readout["events"]["aircraft"], first["events"]["aircraft"]
        out["events"] = {k: part[k] for k in ("sentences", "reward", "landed_here", "lost_separation",
                                              "said_a_go_around")} | {"against_round_0": paired(rows_0, rows)}
    out["traffic"] = {"teacher_forced_nll_per_step": readout["traffic"]["teacher_forced"]["nll_per_step"],
                      "traffic_output_over_residual": readout["traffic"]["traffic_output_over_residual"]}
    return out


def training_round(row: Mapping[str, Any]) -> dict[str, Any]:
    """A training round's sentences and pass (its ``history.json`` row)."""
    sentences, passed = row["sentences"], row["train_pass"]
    return {"sentences": {k: sentences[k] for k in ("windows", "kinds", "aircraft_sentences", "given", "trained_on",
                                                    "starting_in_a_loss", "all", "by_kind", "unprobed", "probed",
                                                    "go_arounds")},
            "pass": {k: passed[k] for k in ("batches", "sentences", "reward_mean", "kl_mean", "kl_max", "clipped_share",
                                            "data_mean", "imitation_mean", "distance_at_start")}}


def read_run(run: Path) -> dict[str, Any]:
    """The run at ``run`` read round by round (module docstring)."""
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config["schema"] != RUN_SCHEMA:
        raise SystemExit(f"{run} is a {config['schema']} run, not {RUN_SCHEMA}")
    finished, unfinished = finished_rounds(run)
    history = {row["round"]: row for row in json.loads((run / "history.json").read_text(encoding="utf-8"))["rounds"]}
    missing = [n for n in finished if n not in history]
    if missing:
        raise SystemExit(f"{run}/history.json holds no row for the finished round(s) {missing}")
    readouts = [json.loads((run / f"round_{n:02d}" / "readout.json").read_text(encoding="utf-8")) for n in finished]
    rounds = [{"round": n, "select": select_round(readouts[n], readouts[0]),
               "training": training_round(history[n]) if n > 0 else None} for n in finished]
    choice_path = run / "choice.json"
    choice = json.loads(choice_path.read_text(encoding="utf-8")) if choice_path.exists() else None
    return {"run": repo_relative(run), "asked_rounds": config["rounds"], "finished_rounds": finished[-1],
            "not_finished": unfinished, "rounds": rounds,
            "choice": None if choice is None else {
                "round": choice["round"], "excluded_by_the_guards": choice["excluded_by_the_guards"],
                "covers_rounds": len(choice["against_round_0"]) - 1, "against_round_0": choice["against_round_0"]}}


def _pp(x: float) -> str:
    return f"{100 * x:+.2f}"


def _number(x: float | None, digits: int = 4) -> str:
    """A number the run leaves None when what it measures did not happen (no landing: no observed-runway share and no
    ordering; no loss: no partner share; no augmented sample trained on: no distance at the start)."""
    return "—" if x is None else f"{x:.{digits}f}"


def tables(read: Mapping[str, Any]) -> list[str]:
    """The readout as text tables."""
    lines = [f"{read['run']}: rounds 0 … {read['finished_rounds']} finished (the last invocation asked for rounds up to "
             f"{read['asked_rounds']})"
             + (f"; not finished: {', '.join(read['not_finished'])}" if read["not_finished"] else ""), ""]
    for side in SIDES:
        lines.append(f"{side} select windows — round: reward, lost separation, landed, on the observed runway"
                     + (", ordering (time, gap)" if side == "real" else "")
                     + "; against round 0: reward, lost separation (pp), ± standard error; losses ended by another "
                     "commanded aircraft")
        for row in read["rounds"]:
            part = row["select"][side]
            change = part["against_round_0"]
            order = part["ordering"] if side == "real" else None
            lines.append(
                f"  {row['round']:>2}  {part['reward']:.4f}  {part['lost_separation']:.4f}  {part['landed']:.4f}  "
                f"{_number(part['landed_on_observed_runway'])}"
                + (f"  ({_number(order['time_ratio'], 3)}, {_number(order['gap_ratio'], 3)})" if side == "real" else "")
                + f"  |  {change['reward']['difference']:+.4f} ± {change['reward']['standard_error']:.4f}"
                f"  {_pp(change['lost_separation']['difference'])} ± {100 * change['lost_separation']['standard_error']:.2f}"
                f"  (n {change['reward']['sentences']})"
                + f"  |  {_number(part['losses_with']['commanded'], 2)}")
            if side == "augmented":
                lines.append("        by kind (reward; against round 0: reward / lost separation pp): " + ", ".join(
                    f"{kind} {part['reward_by_kind'][kind]:.4f}; {v['reward']['difference']:+.4f} ± "
                    f"{v['reward']['standard_error']:.4f} / {_pp(v['lost_separation']['difference'])}"
                    for kind, v in part["against_round_0_by_kind"].items()))
        lines.append("")
    if "events" in read["rounds"][0]["select"]:
        lines.append("select hard events (the answered aircraft, the others given their words) — round: sentences, "
                     "reward, landed here, lost separation, said a go-around; against round 0: reward, lost separation "
                     "(pp), ± standard error")
        for row in read["rounds"]:
            part = row["select"]["events"]
            change = part["against_round_0"]
            lines.append(
                f"  {row['round']:>2}  {part['sentences']}  {part['reward']:.4f}  {part['landed_here']:.4f}  "
                f"{part['lost_separation']:.4f}  {part['said_a_go_around']:.4f}  |  "
                f"{change['reward']['difference']:+.4f} ± {change['reward']['standard_error']:.4f}  "
                f"{_pp(change['lost_separation']['difference'])} ± {100 * change['lost_separation']['standard_error']:.2f}")
        lines.append("")
    lines.append("training — round: windows, aircraft sentences, trained on, starting in a loss, reward; the pass: "
                 "updates, KL mean / max, clipped, data NLL, KL to base at the start (real, augmented); traffic "
                 "attention output / residual per layer; select teacher-forced NLL")
    for row in read["rounds"]:
        traffic = row["select"]["traffic"]
        tail = (f"  |  {' '.join(f'{r:.5f}' for r in traffic['traffic_output_over_residual'])}  "
                f"|  {traffic['teacher_forced_nll_per_step']:.4f}")
        if row["training"] is None:
            lines.append(f"  {row['round']:>2}  (the start){tail}")
            continue
        sentences, passed = row["training"]["sentences"], row["training"]["pass"]
        start = passed["distance_at_start"]
        lines.append(
            f"  {row['round']:>2}  {sentences['windows']}  {sentences['aircraft_sentences']}  {sentences['trained_on']}  "
            f"{sentences['starting_in_a_loss']}  {sentences['all']['reward']:.4f}  |  {passed['batches']}  "
            f"{passed['kl_mean']:.4f} / {passed['kl_max']:.4f}  {passed['clipped_share']:.4f}  {passed['data_mean']:.4f}  "
            + " ".join(_number(start[side]) for side in SIDES) + tail)
    lines.append("")
    choice = read["choice"]
    if choice is None:
        lines.append("no choice yet (choice.json is written when an invocation of the runner ends)")
    else:
        lines.append(f"choice over rounds 0 … {choice['covers_rounds']}: kept round {choice['round']}; the guards "
                     f"excluded {choice['excluded_by_the_guards'] or 'none'}"
                     + (f" — BEHIND the {read['finished_rounds']} rounds finished"
                        if choice["covers_rounds"] < read["finished_rounds"] else ""))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--run", type=Path, required=True, help="the run's --out (its window_s<seed> directory)")
    parser.add_argument("--out", type=Path, help="a NEW directory to write the readout into (printed only without)")
    args = parser.parse_args(argv)
    run = args.run if args.run.is_absolute() else REPO_ROOT / args.run
    out = None if args.out is None else args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out is not None and out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    read = read_run(run)
    print("\n".join(tables(read)))
    if out is not None:
        out.mkdir(parents=True)
        write_json_atomic(out / "traffic_window_reward_readout.json",
                          {"schema": SCHEMA, "written_utc": utc_now(), "git": git_state(), **read})
        print(f"→ {repo_relative(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
