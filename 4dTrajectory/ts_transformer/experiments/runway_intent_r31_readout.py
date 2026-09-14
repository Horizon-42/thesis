"""Runway-intent R3.1 readout: the stochastic schedule's landing-time prediction against the calibrated independent one, and the gates of plan §18.1.

Reads each airport's `runway_intent_r31.json`, writes `readout.md` / `readout.json` to the campaign folder.

    python run_ts.py runway_intent_r31_readout \\
        [--campaign-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r31_20260914]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ts_transformer.experiments.runway_intent_r3 import STRATA
from ts_transformer.experiments.runway_intent_r3_readout import AIRPORTS, fmt
from ts_transformer.experiments.support import REPO_ROOT

DEFAULT_CAMPAIGN = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments" / "runway_intent_r31_20260914"
#: plan §18.1, pre-registered before the run
ALL_HOURS_TOLERANCE_S = 1.0       # gate 2: all hours no worse than the calibrated independent by more than this


def gates(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for airport, art in artifacts.items():
        s = art["summary"]
        busy, every = s.get("busy"), s["all"]
        moved = every["moved"]
        g: dict[str, Any] = {
            "1_busy": None if busy is None else
            busy["stochastic"]["time_error_s"]["p50"] < busy["calibrated"]["time_error_s"]["p50"],
            "2_all_hours": every["stochastic"]["time_error_s"]["p50"]
            <= every["calibrated"]["time_error_s"]["p50"] + ALL_HOURS_TOLERANCE_S,
            "3_moved": None if moved["flights"] == 0 else
            moved["stochastic"]["p50"] <= moved["calibrated"]["p50"],
        }
        # a failed gate fails the airport; otherwise an unmeasured one leaves the verdict open
        g["passed"] = False if any(v is False for v in g.values()) else None if any(v is None for v in g.values()) else True
        out[airport] = g
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-dir", default=str(DEFAULT_CAMPAIGN))
    args = parser.parse_args(argv)
    campaign = Path(args.campaign_dir)
    artifacts = {a: json.loads((campaign / a / "runway_intent_r31.json").read_text(encoding="utf-8"))
                 for a in AIRPORTS if (campaign / a / "runway_intent_r31.json").exists()}
    missing = [a for a in AIRPORTS if a not in artifacts]
    verdicts = gates(artifacts)
    centered = sorted({art["error_model"].get("centered", False) for art in artifacts.values()})
    lines = ["# Runway-intent R3.1 readout", "", f"campaign: `{campaign}`",
             f"error model: {'CENTERED (each stratum median taken out: calibrated = raw ETA)' if centered == [True] else 'as calibrated' if centered == [False] else 'MIXED'}",
             *([] if not missing else ["", f"**missing airports (no artifact): {', '.join(missing)}**"]),
             "", "## Landing-time prediction (|dt| p50 s, signed p50 s in brackets)", "",
             "| airport / hours | n | stochastic | stochastic causal | calibrated independent | raw independent | R3 deterministic | "
             "runway acc stoch / indep | moved: n, stoch / calibrated | q10-q90 coverage | mean P(delayed) |",
             "|---|---:|---|---|---|---|---|---|---|---:|---:|"]
    for airport, art in artifacts.items():
        for stratum in STRATA:
            c = art["summary"].get(stratum)
            if c is None:
                continue

            def cell(name: str) -> str:
                return f"{fmt(c[name]['time_error_s']['p50'], 1)} ({fmt(c[name]['signed_p50_s'], 1)})"

            m = c["moved"]
            lines.append(
                f"| {airport} / {stratum} | {c['flights']} | {cell('stochastic')} | {cell('stochastic_causal')} | {cell('calibrated')} | {cell('independent')} | "
                f"{cell('deterministic')} | {100 * c['stochastic']['runway_accuracy']:.1f} / {100 * c['independent']['runway_accuracy']:.1f} | "
                f"{m['flights']}, {fmt(m['stochastic']['p50'], 1)} / {fmt(m['calibrated']['p50'], 1)} | "
                f"{100 * c['interval_coverage']:.1f}% | {100 * c['p_delayed_mean']:.1f}% |"
            )
    lines += ["", "## Error model (calibration: the expert's own validation split, day_a training days)", "",
              "| airport | calibration flights | tercile edges s | stratum: n, e p10 / p50 / p90 s |", "|---|---:|---|---|"]
    for airport, art in artifacts.items():
        em = art["error_model"]
        strata = "; ".join(f"{name}: {v['n']}, {v['p10']:.0f} / {v['p50']:.0f} / {v['p90']:.0f}" for name, v in em["strata"].items())
        lines.append(f"| {airport} | {em['calibration_flights']} | {', '.join(f'{e:.0f}' for e in em['edges_s'])} | {strata} |")
    lines += ["", "## Gates (plan §18.1)", "", "| airport | 1 busy | 2 all hours (+1 s) | 3 moved | passed |", "|---|---|---|---|---|"]
    mark = {True: "pass", False: "FAIL", None: "—"}
    for airport, g in verdicts.items():
        lines.append(f"| {airport} | {mark[g['1_busy']]} | {mark[g['2_all_hours']]} | {mark[g['3_moved']]} | {mark[g['passed']]} |")
    lines += ["", "Every gate compares the stochastic schedule with the CALIBRATED independent prediction (the same error "
              "model, no interaction), so correcting the ETA's bias is not credited to the interaction."]
    text = "\n".join(lines) + "\n"
    print(text)
    (campaign / "readout.md").write_text(text, encoding="utf-8")
    (campaign / "readout.json").write_text(json.dumps({"gates": verdicts, "airports": list(artifacts)}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
