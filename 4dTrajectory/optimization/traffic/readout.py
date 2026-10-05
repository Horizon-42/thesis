"""The readout of a traffic batch: what ``summary.json`` and the ``*_traffic.json`` sidecars say.

    python -m traffic.readout <batch output dir> [...]      (from 4dTrajectory/optimization)

Counts every window of the roster: solved windows by outcome (a window whose baseline solve failed has
no sidecar and counts as ``baseline_failed``; any other failure as ``error``), the re-solves spent, the
landing-time change from the baseline to the last solve, the windows with a loss that counts for the loop
(VISUAL, the commanded aircraft answers for it, MD10 does not exclude it), with an IFR loss it answers
for, and with a VISUAL loss it does not answer for (§5.2 item 4), at the baseline and the end, and the §9 counts: windows starting in a loss (MD10), types without a CWT
category, losses between recorded aircraft near the commanded one, the largest frame error.
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

from scenario_batch import sidecar_filename  # noqa: E402


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
        delta_t.append(side["rounds"][-1]["final_time_s"] - side["rounds"][0]["final_time_s"])
        for key, r in (("baseline", side["rounds"][0]), ("final", side["rounds"][-1])):
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
    }


def main() -> None:
    for arg in sys.argv[1:]:
        print(arg)
        print(json.dumps(readout(Path(arg)), indent=1))


if __name__ == "__main__":
    main()
