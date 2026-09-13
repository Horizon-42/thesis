"""Runway-intent R3 readout: the scheduled plan against the independent forecast and the truth, and the gates of plan §17.3.

Reads each airport's `runway_intent_r3.json` and writes `readout.md` / `readout.json` to the campaign
folder. Every column of one row is on the same flights; the flown columns are on the flights whose
flown rollout crossed the threshold, with the scheduled and independent errors re-read on those.

    python run_ts.py runway_intent_r3_readout \\
        [--campaign-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r3_20260914]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from geokit import NM_M

from ts_transformer.experiments.support import REPO_ROOT

DEFAULT_CAMPAIGN = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments" / "runway_intent_r3_20260914"
AIRPORTS = ["KRDU", "KSJC", "KSTL", "KSMF", "KMSY"]
STRATA = ("all", "busy", "quiet")
#: plan §17.3, pre-registered before the run
KEPT_SHARE = 0.95                 # gate 2: flown consecutive one-runway landings >= S - 10 s
ALL_HOURS_TOLERANCE_S = 5.0       # gate 3: all hours no worse than the independent ETA by more than this
RUNWAY_TOLERANCE = 0.01           # gate 4: runway agreement >= the head's top-1 accuracy - 1 point
R4_RUNWAY_GAP = 0.05              # R4 starts if busy-hour agreement is >= 5 points below quiet


def _f(value: Any, digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:.{digits}f}"


def table(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| airport / hours | n | runway acc indep / sched / causal | abs dt p50 s indep / sched / causal | "
        "order agree close pairs indep / sched (n) | moved: n, abs dt p50 s sched / indep, closer | delay p90 s | flown landed | "
        "abs dt p50 s flown (indep / sched same flights) | flown FDE med m (unassigned) |",
        "|---|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for airport, art in artifacts.items():
        for stratum in STRATA:
            c = art["summary"].get(stratum)
            if c is None:
                continue
            ind, sch, cau = c["independent"], c["scheduled"], c["causal"]
            fl = c.get("flown")
            flown = ("—", "—", "—") if fl is None else (
                f"{100 * fl['landed_share']:.1f}%",
                f"{_f(fl['time_error_s']['p50'])} ({_f(fl['independent_time_error_s_same_flights']['p50'])} / "
                f"{_f(fl['scheduled_time_error_s_same_flights']['p50'])})",
                f"{_f(fl['fde_m_median'])} ({_f(fl['unassigned_fde_m_median'])})",
            )
            lines.append(
                f"| {airport} / {stratum} | {c['flights']} | "
                f"{100 * ind['runway_accuracy']:.1f} / {100 * sch['runway_accuracy']:.1f} / {100 * cau['runway_accuracy']:.1f} | "
                f"{_f(ind['time_error_s']['p50'])} / {_f(sch['time_error_s']['p50'])} / {_f(cau['time_error_s']['p50'])} | "
                f"{_f(100 * ind['order_agreement_close'], 1)} / {_f(100 * sch['order_agreement_close'], 1)} ({c['order_pairs_close']}) | "
                f"{c['moved']['flights']}, {_f(c['moved']['scheduled_time_error_s']['p50'])} / "
                f"{_f(c['moved']['independent_time_error_s']['p50'])}, {_f(100 * c['moved']['scheduled_closer_share'])}% | "
                f"{_f(sch['delay_s']['p90'])} | {flown[0]} | {flown[1]} | {flown[2]} |"
            )
    return lines


def checks_table(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| airport | approach speed m/s | 3 NM s | violations plan / causal / indep / truth(10 s) / flown(10 s) | "
        "one-runway kept truth / flown (all pairs) | kept under 2 minima truth / flown (n) | unlanded | "
        "on-final separation kept / within 0.5 NM | closure absorbed | roster share of visible (busy) |",
        "|---|---:|---:|---|---|---|---:|---|---:|---|",
    ]
    for airport, art in artifacts.items():
        ch, sep = art["checks"], art["separation"]
        flown_all = art["summary"]["all"].get("flown")
        final = ch.get("flown_final_separation")
        lines.append(
            f"| {airport} | {sep['approach_speed_mps']:.1f} | {sep['same_nm'] * NM_M / sep['approach_speed_mps']:.0f} | "
            f"{ch['plan_violations']} / {ch['causal_violations']} / {ch['independent_violations']} / "
            f"{ch['truth_violations_10s']} / {ch.get('flown_violations_10s', '—')} | "
            f"{100 * ch['truth_one_runway_kept_share'][0]:.1f}% / "
            f"{'—' if 'flown_one_runway_kept_share' not in ch else f'{100 * ch['flown_one_runway_kept_share'][0]:.1f}%'} | "
            f"{100 * ch['truth_one_runway_kept_share_close'][0]:.1f}% ({ch['truth_one_runway_kept_share_close'][1]}) / "
            f"{'—' if 'flown_one_runway_kept_share_close' not in ch else f'{100 * ch['flown_one_runway_kept_share_close'][0]:.1f}% ({ch['flown_one_runway_kept_share_close'][1]})'} | "
            f"{ch.get('flown_unlanded', '—')} | "
            f"{'—' if final is None else f'{100 * final['share_kept']:.1f}% / {100 * final['share_within_half_nm']:.1f}%'} | "
            f"{'—' if flown_all is None else f'{100 * flown_all['absorbed_share']:.1f}%'} | "
            f"{100 * ch['roster_share_of_visible_landings']:.0f}% ({100 * ch['roster_share_of_visible_landings_busy']:.0f}%) |"
        )
    return lines


def variants_table(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| airport / rules | truth kept (10 s) | indep violations | runway acc all | abs dt p50 s all (indep) | busy: abs dt p50 s (indep) | "
        "delayed all / busy | moved: abs dt p50 s (indep) |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for airport, art in artifacts.items():
        s = art["summary"]
        rows = {"faa (primary)": None, **art.get("variants", {})}
        for name, v in rows.items():
            if v is None:
                every, busy = s["all"], s.get("busy")
                lines.append(
                    f"| {airport} / {name} | {100 * art['checks']['truth_one_runway_kept_share'][0]:.1f}% | "
                    f"{art['checks']['independent_violations']} | {100 * every['scheduled']['runway_accuracy']:.1f} | "
                    f"{_f(every['scheduled']['time_error_s']['p50'])} ({_f(every['independent']['time_error_s']['p50'])}) | "
                    f"{'—' if busy is None else f'{_f(busy['scheduled']['time_error_s']['p50'])} ({_f(busy['independent']['time_error_s']['p50'])})'} | "
                    f"{100 * every['scheduled']['delay_s']['delayed_share']:.1f}% / "
                    f"{'—' if busy is None else f'{100 * busy['scheduled']['delay_s']['delayed_share']:.1f}%'} | "
                    f"{_f(every['moved']['scheduled_time_error_s']['p50'])} ({_f(every['moved']['independent_time_error_s']['p50'])}) |"
                )
                continue
            every, busy = v["all"], v.get("busy")
            lines.append(
                f"| {airport} / {name} | {100 * v['truth_one_runway_kept_share'][0]:.1f}% | {v['independent_violations']} | "
                f"{100 * every['runway_accuracy']:.1f} | {_f(every['time_error_s']['p50'])} ({_f(s['all']['independent']['time_error_s']['p50'])}) | "
                f"{'—' if busy is None else f'{_f(busy['time_error_s']['p50'])} ({_f(s['busy']['independent']['time_error_s']['p50'])})'} | "
                f"{100 * every['delayed_share']:.1f}% / {'—' if busy is None else f'{100 * busy['delayed_share']:.1f}%'} | "
                f"{_f(every['moved_time_error_s']['p50'])} ({_f(every['moved_independent_time_error_s']['p50'])}) |"
            )
    return lines


def gates(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for airport, art in artifacts.items():
        ch, s = art["checks"], art["summary"]
        busy, quiet, every = s.get("busy"), s.get("quiet"), s["all"]
        g: dict[str, Any] = {
            "1_plan_violations_zero": ch["plan_violations"] == 0,
            "2_flown_kept_share": None if "flown_one_runway_kept_share" not in ch else ch["flown_one_runway_kept_share"][0] >= KEPT_SHARE,
            "3_busy_timing": None if busy is None else busy["scheduled"]["time_error_s"]["p50"] < busy["independent"]["time_error_s"]["p50"],
            "3_all_hours_no_worse": every["scheduled"]["time_error_s"]["p50"] <= every["independent"]["time_error_s"]["p50"] + ALL_HOURS_TOLERANCE_S,
            "4_runway_agreement": every["scheduled"]["runway_accuracy"] >= every["independent"]["runway_accuracy"] - RUNWAY_TOLERANCE,
        }
        # a gate that was not measured (no flight flown, no busy hour) leaves the verdict open, never passed
        g["passed"] = None if any(v is None for v in g.values()) else all(g.values())
        if busy is not None and quiet is not None:
            g["r4_runway_gap"] = quiet["scheduled"]["runway_accuracy"] - busy["scheduled"]["runway_accuracy"]
            g["r4_runway_trigger"] = g["r4_runway_gap"] >= R4_RUNWAY_GAP
            g["r4_timing_ratio_busy_over_quiet"] = busy["scheduled"]["time_error_s"]["p50"] / quiet["scheduled"]["time_error_s"]["p50"]
        out[airport] = g
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-dir", default=str(DEFAULT_CAMPAIGN))
    args = parser.parse_args(argv)
    campaign = Path(args.campaign_dir)
    artifacts = {a: json.loads((campaign / a / "runway_intent_r3.json").read_text(encoding="utf-8"))
                 for a in AIRPORTS if (campaign / a / "runway_intent_r3.json").exists()}
    missing = [a for a in AIRPORTS if a not in artifacts]
    verdicts = gates(artifacts)
    lines = ["# Runway-intent R3 readout", "", f"campaign: `{campaign}`",
             *([] if not missing else ["", f"**missing airports (no artifact): {', '.join(missing)}**"]),
             "", "## Assignment and timing", "",
             *table(artifacts), "", "## Separation and closure", "", *checks_table(artifacts), "",
             "## Other rule sets (plan-only; FCFS by ETA)", "", *variants_table(artifacts), "", "## Gates (plan §17.3)", "",
             "| airport | 1 plan 0 | 2 flown kept | 3 busy | 3 all | 4 runway | passed | R4: busy runway gap | R4: busy/quiet dt |",
             "|---|---|---|---|---|---|---|---:|---:|"]
    for airport, g in verdicts.items():
        mark = {True: "pass", False: "FAIL", None: "—"}
        lines.append(
            f"| {airport} | {mark[g['1_plan_violations_zero']]} | {mark[g['2_flown_kept_share']]} | {mark[g['3_busy_timing']]} | "
            f"{mark[g['3_all_hours_no_worse']]} | {mark[g['4_runway_agreement']]} | {mark[g['passed']]} | "
            f"{_f(100 * g['r4_runway_gap'], 1) if 'r4_runway_gap' in g else '—'} | "
            f"{_f(g['r4_timing_ratio_busy_over_quiet'], 2) if 'r4_timing_ratio_busy_over_quiet' in g else '—'} |"
        )
    lines += ["", "Gate 2 is the pre-registered reading: every consecutive flown one-runway pair, most of them minutes "
              "apart; the pairs a minimum can bind are the \"kept under 2 minima\" column, and a flight that never "
              "crossed its threshold is out of every flown pair (\"unlanded\"). The gated plan is FCFS by ETA, which "
              "is NOT causal (it reads ETAs some flights only have after an earlier flight's anchor); the causal "
              "schedule is the column beside it."]
    text = "\n".join(lines) + "\n"
    print(text)
    (campaign / "readout.md").write_text(text, encoding="utf-8")
    (campaign / "readout.json").write_text(json.dumps({"gates": verdicts, "airports": list(artifacts)}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
