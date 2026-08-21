"""The stratified arm comparison, tested before it is needed on real results.

It exists to answer "which kind of approach did this dose win on", so the test builds two
arms whose aggregate is a wash and whose strata are opposite, and requires the split to
show that. A tool that only ever runs at result time is a tool debugged at result time.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

TS_DIR = Path(__file__).resolve().parents[1]
DOCS = TS_DIR / "docs"
SCRIPT = DOCS / "compare_control_arms_stratified.py"


def _write_arm(campaign: Path, name: str, straight_ade: float, vectored_ade: float,
               weight: float | None) -> None:
    pred = campaign / f"{name}_pred_val"
    pred.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(60):
        straight = index < 30
        results.append({
            "id": f"FL{index:03d}",
            "runway": "30L",
            "icao24": f"a{index:05x}",
            "landing_time_utc": f"2026-07-07T0{index % 10}:00:00Z",
            "ade_m": straight_ade if straight else vectored_ade,
            "fde_m": (straight_ade if straight else vectored_ade) * 2.0,
            "route_tortuosity": 1.01 if straight else 1.80,
            "established_at_anchor": straight,
        })
    (pred / "summary.json").write_text(json.dumps({"results": results}))
    if weight is not None:
        (campaign / name).mkdir(parents=True, exist_ok=True)
        (campaign / name / "config.json").write_text(
            json.dumps({"control_imitation_loss_weight": weight})
        )


def test_a_wash_in_aggregate_is_split_apart_by_stratum(tmp_path):
    campaign = tmp_path / "campaign"
    # Same aggregate median across the two arms; opposite inside the strata.
    _write_arm(campaign, "low", straight_ade=400.0, vectored_ade=800.0, weight=0.0)
    _write_arm(campaign, "high", straight_ade=300.0, vectored_ade=900.0, weight=64.0)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(campaign)],
        capture_output=True, text=True, check=True,
    )
    out = result.stdout

    assert "60 flights predicted by every arm" in out
    # Columns are ordered by dose, so `low` precedes `high`.
    assert out.strip().splitlines()[-1].index("low") < out.strip().splitlines()[-1].index("high")

    straight = next(l for l in out.splitlines()
                    if "ade_m" in l and "400.0" in l and "300.0" in l)
    vectored = next(l for l in out.splitlines()
                    if "ade_m" in l and "800.0" in l and "900.0" in l)
    # The high dose wins the straight stratum and loses the vectored one — the exact case
    # an aggregate hides.
    assert straight.index("400.0") < straight.index("300.0")
    assert vectored.index("800.0") < vectored.index("900.0")


def test_it_says_so_when_the_covariates_are_missing(tmp_path):
    """Pre-merge artifacts carry no route_tortuosity; that must not read as 'no difference'."""
    campaign = tmp_path / "campaign"
    pred = campaign / "old_pred_val"
    pred.mkdir(parents=True)
    (pred / "summary.json").write_text(json.dumps({
        "results": [{"id": "FL1", "runway": "30L", "icao24": "a1",
                     "landing_time_utc": "z", "ade_m": 400.0}]
    }))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(campaign)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "no arms with difficulty covariates" in result.stdout
