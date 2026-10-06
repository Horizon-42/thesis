"""The readout of a traffic batch: what ``summary.json`` and the ``*_traffic.json`` sidecars say.

    python -m traffic.readout <batch output dir> [...]      (from 4dTrajectory/optimization; an M1 or M2 run)

Counts every window of the roster: solved windows by outcome (a window whose baseline solve failed has a
failed sidecar and counts as ``baseline_failed``; any other failure as ``error``), the re-solves spent, the
landing-time change from the baseline to the last solve, the windows with a loss that counts for the loop
(VISUAL, the commanded aircraft answers for it, MD10 does not exclude it), with an IFR loss it answers
for, and with a VISUAL loss it does not answer for (§5.2 item 4), at the baseline and the end, and the §9 counts: windows starting in a loss (MD10), types without a CWT
category, losses between recorded aircraft near the commanded one, the largest frame error; and the timing of
every scenario, failed ones included (``timing``: the multi-aircraft design §8.5).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]      # for flight_scenarios (python -m from the optimization dir)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_export import EVAL_SUFFIX, STATES_SUFFIX  # noqa: E402
from scenario_batch import sidecar_filename  # noqa: E402
from traffic import M2_MODE, rules  # noqa: E402

#: A slot later than its ETA by more than this is "delayed" (the approach clock's round trip leaves ~1e-7 s).
_DELAY_TOLERANCE_S = 1e-3


def _sidecar(batch_dir: Path, row: dict, suffix: str) -> dict:
    """A summary row's sidecar: a solved row's, or a failed row's failed sidecar (named from its eval record)."""
    states = row["states_file"] or row["eval_file"].removesuffix(EVAL_SUFFIX) + STATES_SUFFIX
    return json.loads((batch_dir / sidecar_filename(states, suffix)).read_text(encoding="utf-8"))


def _spread(values: list[float]) -> dict:
    return ({"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
             "p95": float(np.percentile(values, 95)), "max": float(max(values))} if values else {"n": 0})


def timing(batch_dir: Path, rows: list[dict], suffix: str = "_traffic.json") -> dict:
    """The solve time of every scenario of ``rows`` (failed ones included: their failed sidecars keep the solves they
    timed): per scenario the sum of its solves in wall and process CPU time and their count; the whole window (the
    solves, replays and judge) for the solved ones; the time by kind of solve."""
    wall, cpu, count, failed_attempts, window_wall, window_cpu = [], [], [], 0, [], []
    by_kind: dict[str, dict] = {}
    failed_wall, untimed = [], []
    for row in rows:
        side = _sidecar(batch_dir, row, suffix)
        solves = side["solves"]
        if solves is None:                     # a failure that lost its solve times: counted apart, never as 0 s
            untimed.append(side["flight_key"])
            continue
        wall.append(sum(s["wallS"] for s in solves))
        cpu.append(sum(s["cpuS"] for s in solves))
        count.append(len(solves))
        failed_attempts += sum(not s["ok"] for s in solves)
        for s in solves:
            kind = by_kind.setdefault(s["kind"], {"solves": 0, "failed": 0, "wallS": 0.0, "cpuS": 0.0})
            kind["solves"] += 1
            kind["failed"] += not s["ok"]
            kind["wallS"] += s["wallS"]
            kind["cpuS"] += s["cpuS"]
        if row["status"] == "solved":
            window_wall.append(side["windowWallS"])
            window_cpu.append(side["windowCpuS"])
        else:
            failed_wall.append(wall[-1])
    return {
        "scenarios": len(rows),
        "untimed_scenarios": untimed,
        "solve_wall_s": _spread(wall), "solve_cpu_s": _spread(cpu),
        "solves_per_scenario": dict(sorted(Counter(count).items())),
        "failed_solve_attempts": failed_attempts,
        "by_kind": by_kind,
        # the window of a solved scenario: its solves, replays and judge (an M2 aircraft's ETA solves are not in it)
        "solved_window_wall_s": _spread(window_wall), "solved_window_cpu_s": _spread(window_cpu),
        "unsolved_scenarios_solve_wall_s": _spread(failed_wall),
    }


def readout(batch_dir: Path, sidecar_suffix: str = "_traffic.json") -> dict:
    summary = json.loads((batch_dir / "summary.json").read_text(encoding="utf-8"))
    outcomes, rounds, delta_t, ifr = Counter(), [], [], {"baseline": 0, "final": 0}
    counted, not_answered = {"baseline": 0, "final": 0}, {"baseline": 0, "final": 0}
    starts_in_loss, uncategorised, background, frame_error, no_final = 0, set(), 0, 0.0, set()
    for row in summary["results"]:
        if row["status"] != "solved":
            outcomes["baseline_failed" if row["reason"].startswith("BaselineFailed") else "error"] += 1
            continue
        side = json.loads((batch_dir / sidecar_filename(row["states_file"], sidecar_suffix)).read_text(encoding="utf-8"))
        outcomes[side["outcome"]] += 1
        rounds.append(len(side["rounds"]) - 1)
        kept = side["rounds"][side["kept_round"]]           # MD14: the round whose solve is the record
        delta_t.append(kept["final_time_s"] - side["rounds"][0]["final_time_s"])
        for key, r in (("baseline", side["rounds"][0]), ("final", kept)):
            counted[key] += any(loss[7] for loss in r["losses"])
            ifr[key] += any(loss[6] for loss in side["ifr_losses"][key])
            not_answered[key] += any(not loss[6] for loss in r["losses"])
        starts_in_loss += bool(side["starts_in_loss"])
        uncategorised |= set(side["uncategorised_types"])
        no_final |= set(side["runways_without_faf"])
        background += side["rounds"][0]["background_losses"]
        frame_error = max(frame_error, side["frame_east_error_max"])
    changed = [d for d, r in zip(delta_t, rounds) if r > 0]
    return {
        "windows": summary["total"],
        "outcomes": dict(outcomes),
        "re_solves": dict(Counter(rounds)),
        "landing_time_change_s": (
            {"windows": len(changed), "median": float(np.median(changed)), "min": float(min(changed)),
             "max": float(max(changed))} if changed else {"windows": 0}),
        "windows_with_a_counted_visual_loss": counted,
        "windows_with_an_answered_ifr_loss": ifr,
        "windows_with_a_visual_loss_not_answered_for": not_answered,
        "runways_without_a_coded_final": sorted(no_final),
        "windows_starting_in_a_loss": starts_in_loss,
        "uncategorised_types": sorted(uncategorised),
        "background_losses_at_baseline": background,
        "frame_east_error_max": frame_error,
        "timing": timing(batch_dir, summary["results"], sidecar_suffix),
    }


def blocks_readout(m2_dir: Path) -> dict:
    """An M2 run (its ``summary.json``, mode ``M2_MODE``, with every block): aircraft (those with a dynamics model, the
    ones the block controls) and ``skipped_no_dynamics`` (those without, which stay their records), the schedule's delays, the
    outcomes (a failed solve split by its delay: none, or some; ``slot_failed`` — the slot's fixed-time
    solve on the ETA's IAF failed, no other IAF is tried), and the flown aircraft with a loss left after the
    block's final check — one it answers for, one it does not."""
    summary = json.loads((m2_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["mode"] != M2_MODE:
        raise ValueError(f"{m2_dir}: mode {summary['mode']!r}, not an M2 run")
    outcomes, delays = Counter(), []
    left = {reading: {"answered": 0, "not_answered": 0} for reading in (rules.VISUAL, rules.IFR)}
    aircraft = scheduled = 0
    failed_blocks = [b["label"] for b in summary["blocks"] if "error" in b]
    for block in summary["blocks"]:
        if "error" in block:
            continue
        aircraft += block["aircraft"]
        scheduled += block["scheduled"]
        delay = {slot["flight_key"]: slot["delay_s"] for slot in block["slots"]}
        delays += list(delay.values())
        for key, outcome in block["outcomes"].items():
            if outcome.startswith("BaselineFailed: ETA"):
                outcomes["eta_failed"] += 1
            elif outcome.startswith("BaselineFailed: slot"):
                outcomes["slot_failed"] += 1
            elif outcome == "solve_failed":
                outcomes["solve_failed_" + ("delayed" if delay[key] > _DELAY_TOLERANCE_S else "undelayed")] += 1
            else:
                outcomes[outcome] += 1
        for losses in block["final_losses"].values():
            for reading in left:
                for kind in ("answered", "not_answered"):
                    left[reading][kind] += losses[reading][kind] > 0
    return {
        "blocks": len(summary["blocks"]),
        "failed_blocks": failed_blocks,
        "aircraft": aircraft,
        "skipped_no_dynamics": sum(block["skipped_no_dynamics"] for block in summary["blocks"]),
        "scheduled": scheduled,
        "delay_s": ({"median": float(np.median(delays)), "max": float(max(delays)),
                     "delayed_over_60s": sum(d > 60.0 for d in delays)} if delays else {}),
        "outcomes": dict(outcomes),
        "flown_aircraft_with_a_loss_left_after_the_block": left,
        "timing": timing(m2_dir, summary["results"]),
    }


def main() -> None:
    for arg in sys.argv[1:]:
        print(arg)
        mode = json.loads((Path(arg) / "summary.json").read_text(encoding="utf-8"))["mode"]
        found = blocks_readout(Path(arg)) if mode == M2_MODE else readout(Path(arg))
        print(json.dumps(found, indent=1))


if __name__ == "__main__":
    main()
