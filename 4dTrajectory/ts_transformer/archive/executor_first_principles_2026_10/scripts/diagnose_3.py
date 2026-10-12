#!/usr/bin/env python
"""Where the height error at the threshold is made: before or after the observed track ends.

Reads the stored evaluation records only (no flying, nothing written outside --out). For every graded flight of an arm:
  d_end      the observed track's last point, its distance before the threshold along the runway course (m)
  h_obs_end  the observed height above the published path (LTP + TCH, glidepath angle) at that point
  h_fly_end  the flown height above the published path at the same distance before the threshold
  vertical_m the flown height above the path at the threshold (the gate's number, from the arm's report)
  gain       vertical_m − h_fly_end: what the flown stretch after the observed end added
  tail_deg   the flown mean path angle over that stretch; obs_tail_deg the observed one over its last 600 m
A vertical failure is "made after the observed end" when |h_fly_end| ≤ 22 m (the flown path was inside the gate where
the observed track ends) and |vertical_m| > 22 m.
Arms: `new_design_eval` arms A (0) and B1 (1), at Δ 2 and 4 s (the official baseline kept its reports only, no states).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import archive_paths   # noqa: E402  (README §2)
GATE_M = 22.0
M_PER_DEG_LAT = 111_320.0   # diagnostic only: distances of a few km around one threshold
ARMS = {"A_d2": archive_paths.RESULTS / "new_design_eval/d2/arm0_d2", "B1_d2": archive_paths.RESULTS / "new_design_eval/d2/arm1_d2",
        "A_d4": archive_paths.RESULTS / "new_design_eval/d4/arm0_d4", "B1_d4": archive_paths.RESULTS / "new_design_eval/d4/arm1_d4"}


def to_go_and_height(states, target, tch_alt, gpa_deg):
    """The final straight-in leg of ``states`` (walking back from the last state while the distance to go grows and
    the path stays within 500 m of the centreline: a downwind leg also passes the threshold at small distance to go):
    its distance before the threshold along the course, height above the published path and altitude."""
    lat0, lon0, psi = target["lat"], target["lon"], target["psi"]
    k_lon = M_PER_DEG_LAT * math.cos(math.radians(lat0))
    e = np.array([(s["lon"] - lon0) * k_lon for s in states])
    n = np.array([(s["lat"] - lat0) * M_PER_DEG_LAT for s in states])
    to_go = -(e * math.cos(psi) + n * math.sin(psi))
    cross = -e * math.sin(psi) + n * math.cos(psi)
    first = len(states) - 1
    while first > 0 and to_go[first - 1] >= to_go[first] - 1.0 and abs(cross[first - 1]) < 500.0:
        first -= 1
    to_go, alt = to_go[first:], np.array([s["alt"] for s in states[first:]])
    return to_go, alt - (tch_alt + to_go * math.tan(math.radians(gpa_deg))), alt


def flight_row(eval_path: Path, report_row: dict) -> dict | None:
    record = json.loads(eval_path.read_text())
    states = json.loads((eval_path.parent / record["states_ref"]["file"]).read_text())
    target, source = record["target_state"], states["source"]
    gpa = source["published_glidepath_deg"]
    flown, observed = states["predicted_states"], states["observed_states"]
    if not flown or not observed or report_row.get("vertical_m") is None:
        return None
    go_f, h_f, alt_f = to_go_and_height(flown, target, target["alt"], gpa)
    go_o, h_o, alt_o = to_go_and_height(observed, target, target["alt"], gpa)
    d_end = float(go_o[-1])
    if d_end <= 0.0 or go_f.min() > d_end or go_f.max() < d_end:
        h_fly_end = None
    else:
        h_fly_end = float(np.interp(-d_end, -go_f, h_f))   # the final leg's to-go decreases along the flight
    alt_fly_end = None if h_fly_end is None else h_fly_end + target["alt"] + d_end * math.tan(math.radians(gpa))
    vertical = float(report_row["vertical_m"])
    alt_cross = vertical + target["alt"]
    last600 = go_o <= d_end + 600.0
    obs_tail = (math.degrees(math.atan2(float(alt_o[last600][0] - alt_o[-1]), float(go_o[last600][0] - go_o[-1])))
                if last600.sum() >= 2 and go_o[last600][0] > go_o[-1] else None)
    return {"flight_key": report_row["flight_key"], "airport": report_row["airport"], "vertical_m": vertical,
            "vertical_result": report_row["vertical_result"], "d_end": d_end, "h_obs_end": float(h_o[-1]),
            "h_fly_end": h_fly_end, "gain": None if h_fly_end is None else vertical - h_fly_end,
            "tail_deg": (None if h_fly_end is None or d_end < 50.0
                         else math.degrees(math.atan2(alt_fly_end - alt_cross, d_end))),
            "obs_tail_deg": obs_tail, "gpa": gpa, "final_leg_m": float(go_f.max())}


def arm_rows(base: Path) -> list[dict]:
    rows = []
    for report in sorted(glob.glob(str(base / "records" / "*" / "evaluation_report.json"))):
        directory = Path(report).parent
        for row in json.loads(Path(report).read_text())["trajectories"]:
            r = flight_row(directory / row["file"], row)
            if r is not None:
                rows.append(r)
    return rows


def pct(values, qs=(10, 50, 90)):
    values = [v for v in values if v is not None]
    return [round(float(np.percentile(values, q)), 1) for q in qs] if values else None


def summarise(rows: list[dict]) -> dict:
    fails = [r for r in rows if r["vertical_result"] == "fail"]
    after = [r for r in fails if r["h_fly_end"] is not None and abs(r["h_fly_end"]) <= GATE_M]
    shallow = lambda r: r["tail_deg"] is not None and r["tail_deg"] < 2.0
    return {"graded": len(rows), "vertical_fail": len(fails),
            "fail_high": sum(r["vertical_m"] > 0 for r in fails), "fail_low": sum(r["vertical_m"] < 0 for r in fails),
            "fail_made_after_observed_end": len(after),
            "fail_after_end_with_tail_under_2deg": sum(shallow(r) for r in after),
            "fail_observed_itself_outside_at_end": sum(abs(r["h_obs_end"]) > GATE_M for r in fails),
            "all_d_end_p10_p50_p90": pct([r["d_end"] for r in rows]),
            "all_gain_p10_p50_p90": pct([r["gain"] for r in rows]),
            "all_tail_deg_p10_p50_p90": pct([r["tail_deg"] for r in rows]),
            "all_obs_tail_deg_p10_p50_p90": pct([r["obs_tail_deg"] for r in rows]),
            "fails_d_end_p10_p50_p90": pct([r["d_end"] for r in fails]),
            "fails_h_fly_end_p10_p50_p90": pct([r["h_fly_end"] for r in fails]),
            "fails_gain_p10_p50_p90": pct([r["gain"] for r in fails]),
            "fails_tail_deg_p10_p50_p90": pct([r["tail_deg"] for r in fails]),
            "fails_obs_tail_deg_p10_p50_p90": pct([r["obs_tail_deg"] for r in fails]),
            "fails_by_airport": dict(Counter(r["airport"] for r in fails))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    result, per_flight = {}, {}
    for label, base in ARMS.items():
        rows = arm_rows(base)
        per_flight[label] = rows
        result[label] = summarise(rows)
        print(label, json.dumps(result[label]), flush=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=1))
    (args.out / "per_flight.json").write_text(json.dumps(per_flight))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
