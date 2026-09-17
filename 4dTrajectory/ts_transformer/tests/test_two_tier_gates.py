"""`run_ts.py two_tier_gates` (two-tier feasibility §6, §10.7): G1 / G3 off tracker_lockstep artifacts.

The gates are pre-registered numbers, so what must hold is the READING: which variant, which stratum,
strict or inclusive, the guidance's established share taken over the SAME flights, both seeds required,
and refusals wherever the two sides would not be the same cohort or the same definition.
"""

from __future__ import annotations

import json

import pytest

import ts_transformer.experiments.two_tier_gates as gates
from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED
from ts_transformer.experiments.plan_oracle import POLICY_TRUTH, ROLLING_LOCKSTEP, ROUTE_NEXT, SCHEMA as PLAN_ORACLE_SCHEMA
from ts_transformer.experiments.tracker_lockstep import (
    RESULT_SCHEMA,
    VARIANT_NO_PLAN,
    VARIANT_ONE_SHOT,
    VARIANT_RECEDING,
)

LEADS = ("30", "60")


def _difficulty(kind: str) -> dict:
    return {"route_tortuosity": 1.0 if kind == "straight" else 1.4, "established_at_anchor": False,
            "remaining_path_m": 20_000.0}


def _asks_e(ade: float, *, late_factor: float = 1.0) -> list[dict]:
    """Six 30 s legs: the first three at ``ade / 10``, the later ones ``late_factor`` times that."""
    return [{"ask": k, "lead_s": 30.0, "e_m": ade / 10 * (late_factor if k >= 3 else 1.0)} for k in range(6)]


def _row(kind: str, ade: float, *, flyable: bool = True, established: bool = True, late_factor: float = 1.0) -> dict:
    return {"difficulty": _difficulty(kind), "ade_m": ade, "fde_m": ade, "at": {lead: ade / 2 for lead in LEADS},
            "reference": {"fully_flyable": flyable, "established": established},
            "asks_e_m": _asks_e(ade, late_factor=late_factor)}


def _flights(straight_ade: float, vectored_ade: float, **flags) -> dict[str, dict]:
    rows = {f"s{i}": _row("straight", straight_ade, **flags) for i in range(4)}
    rows.update({f"v{i}": _row("vectored", vectored_ade, **flags) for i in range(4)})
    return rows


def _lockstep(tmp_path, name: str, checkpoints: dict[str, dict[str, dict]], *, source: str = "truth",
              anchor: int = 59, schema: str = RESULT_SCHEMA, split: str = "val", limit: int = 0):
    directory = tmp_path / name
    directory.mkdir()
    payload = {
        "schema": schema,
        "plan": {"split": split, "limit": limit},
        "plan_source": f"{source}:/some/plan/checkpoint.pt" if source == "head" else source,
        "checkpoints": {
            label: {"anchor": anchor, "variants": {v: {"flights": rows} for v, rows in variants.items()}}
            for label, variants in checkpoints.items()
        },
    }
    (directory / "tracker_lockstep.json").write_text(json.dumps(payload))
    return directory


def _oracle(tmp_path, name: str, rows: dict[str, dict], *, anchor: int = 59, flyable_share: float = 1.0, **protocol):
    directory = tmp_path / name
    directory.mkdir()
    artifact = {
        "schema_version": PLAN_ORACLE_SCHEMA, "anchor": anchor, "flights": len(rows),
        "policy": POLICY_TRUTH, "route": ROUTE_NEXT, "rolling": ROLLING_LOCKSTEP, "split": "val", "limit": 0,
        **protocol,
        "summary": {STRATUM_ALL: {"fully_flyable_share": flyable_share}},
        "rows": [
            {"dataset_id": key, "anchor": anchor, "difficulty": row["difficulty"], "prediction": {"ade_m": row["ade_m"]},
             "reference": row["reference"]}
            for key, row in rows.items()
        ],
    }
    (directory / "plan_oracle.json").write_text(json.dumps(artifact))
    return directory


def _guidance(tmp_path, *, vectored_established: bool = True):
    rows = _flights(283.0, 1847.0)
    for key in rows:
        if key.startswith("v") and key in ("v0", "v1") and not vectored_established:
            rows[key]["reference"]["established"] = False
    return _oracle(tmp_path, "guidance", rows)


def _run(tmp_path, *lockstep, gate_arms="s1337,s2024", baseline=None, plan_path=None):
    argv = [arg for path in lockstep for arg in ("--lockstep", str(path))]
    argv += ["--gate-arms", gate_arms, "--out", str(tmp_path / "gates")]
    if baseline is not None:
        argv += ["--baseline", str(baseline)]
    if plan_path is not None:
        argv += ["--plan-path-baseline", str(plan_path)]
    assert gates.main(argv) == 0
    return json.loads((tmp_path / "gates" / "two_tier_gates.json").read_text())


def _variants(straight: float, vectored: float, **flags) -> dict[str, dict]:
    return {VARIANT_RECEDING: _flights(straight, vectored, **flags),
            VARIANT_NO_PLAN: _flights(straight + 50.0, vectored + 400.0),
            VARIANT_ONE_SHOT: _flights(straight + 10.0, vectored + 100.0)}


# ── G1 ─────────────────────────────────────────────────────────────────────────

def test_g1_passes_only_when_both_seeds_pass_every_criterion(tmp_path):
    lockstep = _lockstep(tmp_path, "t1a", {"s1337": _variants(150.0, 900.0), "s2024": _variants(190.0, 999.0),
                                            "p0": _variants(500.0, 3000.0)})
    result = _run(tmp_path, lockstep, baseline=_guidance(tmp_path))
    g1 = result["gates"]["G1"]
    assert g1["pass"] and set(g1["verdicts"]) == {"s1337", "s2024"}      # p0 is read, never judged
    assert "p0" in g1["arms"]
    # the readings beside it: receding against the absent token, the one-shot and the guidance
    minus = g1["arms"]["s1337"]["receding_minus"]
    assert minus[VARIANT_NO_PLAN][STRATUM_VECTORED]["delta_mean_m"] == pytest.approx(-400.0)
    assert minus[VARIANT_ONE_SHOT][STRATUM_STRAIGHT_IN]["delta_mean_m"] == pytest.approx(-10.0)
    assert minus["guidance"][STRATUM_VECTORED]["delta_mean_m"] == pytest.approx(900.0 - 1847.0)
    assert minus["guidance"][STRATUM_VECTORED]["arm_better_share"] == 1.0


@pytest.mark.parametrize("seed_2024, failing", [
    (dict(straight=150.0, vectored=1000.0), "ADE mean vectored"),      # strict: 1000 is not below 1000
    (dict(straight=200.0, vectored=900.0), "ADE mean straight-in"),
    (dict(straight=150.0, vectored=900.0, flyable=False), "fully flyable share, all"),
])
def test_g1_fails_on_one_seed_and_names_the_criterion(tmp_path, seed_2024, failing):
    straight, vectored = seed_2024.pop("straight"), seed_2024.pop("vectored")
    lockstep = _lockstep(tmp_path, "t1a", {"s1337": _variants(150.0, 900.0),
                                            "s2024": _variants(straight, vectored, **seed_2024)})
    g1 = _run(tmp_path, lockstep, baseline=_guidance(tmp_path))["gates"]["G1"]
    assert not g1["pass"] and g1["verdicts"]["s1337"]["pass"] and not g1["verdicts"]["s2024"]["pass"]
    failed = [c["criterion"] for c in g1["verdicts"]["s2024"]["criteria"] if not c["pass"]]
    assert len(failed) == 1 and failed[0].startswith(failing)


def test_g1_established_is_a_fraction_of_the_guidance_on_the_same_flights(tmp_path):
    """The guidance establishes 2 of the 4 vectored flights here, so 0.88 × 0.5 = 0.44 is the bar."""
    receding = _flights(150.0, 900.0)
    for key in ("v0", "v1", "v2"):
        receding[key]["reference"]["established"] = False           # 1 of 4 = 0.25 < 0.44
    lockstep = _lockstep(tmp_path, "t1a", {"s1337": {VARIANT_RECEDING: _flights(150.0, 900.0)},
                                            "s2024": {VARIANT_RECEDING: receding}})
    g1 = _run(tmp_path, lockstep, baseline=_guidance(tmp_path, vectored_established=False))["gates"]["G1"]
    [bar] = [c for c in g1["verdicts"]["s2024"]["criteria"] if c["criterion"].startswith("established share vectored")]
    assert bar["threshold"] == pytest.approx(0.88 * 0.5) and bar["value"] == 0.25 and not bar["pass"]
    assert g1["verdicts"]["s1337"]["pass"]


# ── L1 (two-tier v2 §3) ─────────────────────────────────────────────────────────

def test_l1_reads_a_waypoint_artifact_against_the_guidance_with_the_drift_beside(tmp_path):
    """The truth-waypoint source is judged under L1's own numbers (1500 / 250 m, the guidance's
    established share × 0.88), and the drift reading is REPORTED per variant, never judged."""
    steady = _variants(240.0, 1400.0)
    drifting = {VARIANT_RECEDING: _flights(240.0, 1400.0, late_factor=2.0),
                VARIANT_NO_PLAN: _flights(300.0, 1900.0)}
    lockstep = _lockstep(tmp_path, "l1", {"s1337": steady, "s2024": drifting}, source="truth-waypoints")
    result = _run(tmp_path, lockstep, baseline=_guidance(tmp_path))
    assert set(result["gates"]) == {"L1"}
    l1 = result["gates"]["L1"]
    assert l1["pass"]                                        # 1400 < 1500 and 240 < 250, both seeds
    [vectored] = [c for c in l1["verdicts"]["s1337"]["criteria"] if c["criterion"].startswith("ADE mean vectored")]
    assert vectored["threshold"] == gates.L1_ADE_BELOW_M[STRATUM_VECTORED] == 1500.0
    assert "guidance" in l1["arms"]["s1337"]["receding_minus"]
    steady_drift = l1["arms"]["s1337"]["drift"][VARIANT_RECEDING][STRATUM_VECTORED]
    rising = l1["arms"]["s2024"]["drift"][VARIANT_RECEDING][STRATUM_VECTORED]
    assert steady_drift["ratio"] == pytest.approx(1.0) and steady_drift["rising"] is False
    assert rising["ratio"] == pytest.approx(2.0) and rising["rising"] is True     # 2.0 > 1.5
    assert rising["p50_by_ask_m"]["3"]["p50"] == pytest.approx(280.0) and rising["p50_by_ask_m"]["0"]["n"] == 4
    assert l1["verdicts"]["s2024"]["pass"]                   # the drift never enters the verdict


def test_a_tracker_cohort_inside_the_baselines_is_paired_and_the_dropped_count_stated(tmp_path):
    """A fixed horizon admits only flights with Δ of truth after the anchor, so the tracker may
    hold FEWER flights than the guidance baseline: paired over the tracker's, the gap stated."""
    subset = _variants(240.0, 1400.0)
    for rows in subset.values():
        rows.pop("v3")
    lockstep = _lockstep(tmp_path, "l1", {"s1337": subset, "s2024": _variants(240.0, 1400.0)},
                         source="truth-waypoints")
    l1 = _run(tmp_path, lockstep, baseline=_guidance(tmp_path))["gates"]["L1"]
    assert l1["pass"]
    assert l1["verdicts"]["s1337"]["baseline_flights_not_in_tracker"] == 1
    assert l1["verdicts"]["s2024"]["baseline_flights_not_in_tracker"] == 0
    assert l1["arms"]["s1337"]["receding_minus"]["guidance"][STRATUM_VECTORED]["n"] == 3


def test_l1_fails_at_its_own_thresholds(tmp_path):
    lockstep = _lockstep(tmp_path, "l1", {"s1337": _variants(240.0, 1400.0), "s2024": _variants(250.0, 1400.0)},
                         source="truth-waypoints")
    l1 = _run(tmp_path, lockstep, baseline=_guidance(tmp_path))["gates"]["L1"]
    failed = [c["criterion"] for c in l1["verdicts"]["s2024"]["criteria"] if not c["pass"]]
    assert not l1["pass"] and len(failed) == 1 and failed[0].startswith("ADE mean straight-in")   # strict: 250 is not below 250


# ── G3 ─────────────────────────────────────────────────────────────────────────

def test_g3_reads_a_head_artifact_with_inclusive_ade_and_the_plan_paths_flyable_share_beside(tmp_path):
    head = _lockstep(tmp_path, "t2", {"s1337": _variants(415.0, 2745.0), "s2024": _variants(400.0, 2000.0)},
                     source="head")
    plan_path = _oracle(tmp_path, "plan_path", _flights(773.0, 3143.0), anchor=29, flyable_share=1.0)
    result = _run(tmp_path, head, plan_path=plan_path)
    g3 = result["gates"]["G3"]
    assert set(result["gates"]) == {"G3"} and g3["pass"]            # ≤ 2745 and ≤ 415 are inclusive
    assert "guidance" not in g3["arms"]["s1337"]["receding_minus"]
    [flyable] = [c for c in g3["verdicts"]["s1337"]["criteria"] if c["criterion"].startswith("fully flyable")]
    # the bar is G1's 0.95, not the plan path's 1.000 (§10.7), which is read beside it
    assert flyable["threshold"] == gates.G3_FULLY_FLYABLE_AT_LEAST == 0.95 and "1.000" in flyable["criterion"]


def test_g1_and_g3_artifacts_are_judged_side_by_side(tmp_path):
    truth = _lockstep(tmp_path, "t1a", {"s1337": _variants(150.0, 900.0), "s2024": _variants(150.0, 900.0)})
    head = _lockstep(tmp_path, "t2", {"s1337": _variants(400.0, 2800.0), "s2024": _variants(400.0, 2000.0)},
                     source="head")
    plan_path = _oracle(tmp_path, "plan_path", _flights(773.0, 3143.0), anchor=29)
    result = _run(tmp_path, truth, head, baseline=_guidance(tmp_path), plan_path=plan_path)
    assert result["gates"]["G1"]["pass"] and not result["gates"]["G3"]["pass"]


# ── refusals ───────────────────────────────────────────────────────────────────

def test_the_readout_refuses_another_cohort_anchor_schema_or_a_missing_gate_arm(tmp_path):
    guidance = _guidance(tmp_path)
    fewer = _flights(150.0, 900.0)
    fewer["v9"] = _row("vectored", 900.0)                 # a tracker flight the baseline never flew
    both = {"s1337": _variants(150.0, 900.0), "s2024": _variants(150.0, 900.0)}
    restratified = _flights(150.0, 900.0)
    restratified["s0"]["difficulty"] = _difficulty("vectored")
    only_straight = {f"s{i}": _row("straight", 150.0) for i in range(4)}
    cases = [
        (_lockstep(tmp_path, "cohort", {"s1337": {VARIANT_RECEDING: fewer}, "s2024": _variants(1, 1)}), guidance, "same cohort"),
        (_lockstep(tmp_path, "anchor", both, anchor=29), guidance, "same anchor"),
        (_lockstep(tmp_path, "schema", {"s1337": _variants(1, 1)}, schema="ts-tracker-lockstep-v1"), guidance, "re-run"),
        (_lockstep(tmp_path, "missing", {"s1337": _variants(1, 1)}), guidance, "gate arm"),
        (_lockstep(tmp_path, "smoke", both, limit=100), guidance, "never a smoke run"),
        (_lockstep(tmp_path, "train", both, split="train"), guidance, "never a smoke run"),
        (_lockstep(tmp_path, "strata", {"s1337": {VARIANT_RECEDING: restratified}, "s2024": _variants(1, 1)}),
         guidance, "stratified differently"),
        (_lockstep(tmp_path, "empty", {"s1337": {VARIANT_RECEDING: only_straight}, "s2024": _variants(1, 1)}),
         _oracle(tmp_path, "straight_guidance", only_straight), "judged empty"),
        (_lockstep(tmp_path, "policy", both), _oracle(tmp_path, "model_guidance", _flights(283.0, 1847.0), policy="model"),
         "flying the truth's plan"),
        (_lockstep(tmp_path, "row_anchor", both), _oracle(tmp_path, "a60s_guidance", _flights(283.0, 1847.0), anchor=30),
         "row anchors"),
    ]
    for index, (lockstep, baseline, message) in enumerate(cases):
        with pytest.raises(SystemExit, match=message):
            gates.main(["--lockstep", str(lockstep), "--gate-arms", "s1337,s2024", "--baseline", str(baseline),
                        "--out", str(tmp_path / f"out{index}")])
    with pytest.raises(SystemExit, match="at least 2 seeds"):
        gates.main(["--lockstep", str(_lockstep(tmp_path, "one", both)), "--gate-arms", "s1337",
                    "--baseline", str(guidance), "--out", str(tmp_path / "out_one")])
    stale = tmp_path / "stale_guidance"
    stale.mkdir()
    (stale / "plan_oracle.json").write_text(json.dumps({"schema_version": "ts-plan-oracle-v1", "rows": []}))
    with pytest.raises(SystemExit, match="this readout reads"):
        gates.load_plan_oracle(stale)
    truth = _lockstep(tmp_path, "nobase", {"s1337": _variants(1, 1), "s2024": _variants(1, 1)})
    with pytest.raises(SystemExit, match="needs --baseline"):
        gates.main(["--lockstep", str(truth), "--gate-arms", "s1337,s2024", "--out", str(tmp_path / "out_nobase")])
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        gates.main(["--lockstep", str(truth), "--gate-arms", "s1337", "--baseline", str(guidance),
                    "--out", str(tmp_path / "exists")])


# ── E2E ────────────────────────────────────────────────────────────────────────


def test_e2e_reads_a_segment_head_artifact_by_g3s_criteria_with_e_plan_beside(tmp_path):
    variants = {VARIANT_RECEDING: _flights(400.0, 2700.0), VARIANT_NO_PLAN: _flights(450.0, 3100.0)}
    for rows in variants.values():
        for row in rows.values():
            row["asks_plan_e_m"] = [{"ask": k, "per_waypoint_m": [100.0 + k, 200.0 + k], "e_m": 150.0 + k} for k in range(3)]
    lockstep = _lockstep(tmp_path, "e2e", {"s1337": variants, "s2024": variants}, source="segment-head", anchor=60)
    # a v4 block may carry the floor's exclusion: the gate says the coverage beside the arm, judges the rest
    artifact = json.loads((lockstep / "tracker_lockstep.json").read_text())
    artifact["checkpoints"]["s1337"]["anchor_floor_excluded"] = {"count": 2, "of": 10, "flight_keys": ["a", "b"]}
    (lockstep / "tracker_lockstep.json").write_text(json.dumps(artifact))
    result = _run(tmp_path, lockstep)
    block = result["gates"][gates.GATE_E2E]
    assert block["pass"]
    assert block["arms"]["s1337"]["cohort"] == {"flights": 8, "of": 10, "floor_excluded": 2}
    assert block["arms"]["s2024"]["cohort"]["floor_excluded"] == 0
    assert "2 split flights cannot host the anchor floor" in (tmp_path / "gates" / "two_tier_gates.txt").read_text()
    names = [c["criterion"] for c in block["verdicts"]["s1337"]["criteria"]]
    assert names[0].startswith("ADE mean") and any("fully flyable" in n for n in names) and any("established" in n for n in names)
    cell = block["arms"]["s1337"]["variants"][VARIANT_RECEDING][STRATUM_ALL]
    assert cell["plan_e_by_ask_p50_m"]["0"] == {"n": 8, "p50": 150.0, "per_waypoint_p50": [100.0, 200.0], "per_waypoint_n": [8, 8]}
    assert "e_plan" in (tmp_path / "gates" / "two_tier_gates.txt").read_text()


def test_a_v3_lockstep_artifact_is_still_read_with_no_e_plan(tmp_path):
    """The L1 campaign's artifacts are v3: no `asks_plan_e_m`, no plan-span flag — read, with an empty e_plan."""
    lockstep = _lockstep(tmp_path, "l1v3", {"s1337": _variants(200.0, 1400.0), "s2024": _variants(200.0, 1400.0)},
                         source="truth-waypoints", schema=gates.LOCKSTEP_SCHEMA_V3)
    result = _run(tmp_path, lockstep, baseline=_guidance(tmp_path))
    cell = result["gates"][gates.GATE_L1]["arms"]["s1337"]["variants"][VARIANT_RECEDING][STRATUM_ALL]
    assert cell["plan_e_by_ask_p50_m"] == {}


# ── L2 ─────────────────────────────────────────────────────────────────────────


def _lead_cells(arm_p50: float, reference_p50: float, n: int = 4) -> dict:
    return {lead: {"n": n, "arm_p50_m": arm_p50, "reference_p50_m": reference_p50,
                   "delta_of_p50_m": arm_p50 - reference_p50, "delta_p50_m": arm_p50 - reference_p50,
                   "arm_better_share": 1.0 if arm_p50 < reference_p50 else 0.0, "arm_held": 0, "reference_held": 1}
            for lead in ("60", "120", "180")}


def _segment_readout(tmp_path, name: str, arms: dict[str, dict[str, tuple[float, float]]], *, references=("native32", "state"),
                     split: str = "val", limit: int = 0, schema: str = gates.SEGMENT_READOUT_SCHEMA):
    """``arms[label][reference] = (vectored delta, straight-in delta)`` against a reference p50 of 1000 / 300."""
    directory = tmp_path / name
    directory.mkdir()
    strata_cells = {"n": 4, "at_lead_m": {lead: {"n": 4, "p50": 500.0, "mean": 500.0, "held": 0} for lead in ("60", "120", "180")}}
    checkpoints = {}
    paired = {}
    for label, deltas in arms.items():
        checkpoints[label] = {"checkpoint": f"/ckpt/{label}.pt", "own_fixed_anchor": 60, "seq_len": 61, "prediction_output": "segment-plan",
                              "segment_plan_segments": 10,
                              "sets": {gates.FIXED_SET: {"strata": {s: strata_cells for s in (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED)},
                                                         "plan": None}}}
        paired[label] = {}
        for reference in (*references, gates.CONSTANT_VELOCITY):
            vectored, straight = deltas.get(reference, (-200.0, -10.0))
            paired[label][reference] = {gates.FIXED_SET: {
                STRATUM_ALL: _lead_cells(800.0, 1000.0), STRATUM_VECTORED: _lead_cells(1000.0 + vectored, 1000.0),
                STRATUM_STRAIGHT_IN: _lead_cells(300.0 + straight, 300.0),
            }}
    for reference in (*references, gates.CONSTANT_VELOCITY):
        checkpoints[reference] = {"checkpoint": None if reference == gates.CONSTANT_VELOCITY else f"/ckpt/{reference}.pt",
                                  "own_fixed_anchor": 59, "seq_len": 60, "prediction_output": "state", "segment_plan_segments": None,
                                  "sets": {gates.FIXED_SET: {"strata": {s: strata_cells for s in (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED)}}}}
    payload = {"schema": schema, "plan": {"split": split, "limit": limit, "leads_s": [60.0, 120.0, 180.0], "bins": [gates.FIXED_SET]},
               "anchor": 60, "arms": list(arms), "references": [*references, gates.CONSTANT_VELOCITY],
               "checkpoints": checkpoints, "paired": paired}
    (directory / "segment_plan_readout.json").write_text(json.dumps(payload))
    return directory


def _run_l2(tmp_path, *readouts, gate_arms="A_s1337,A_s2024"):
    out = tmp_path / f"gates_{len(list(tmp_path.glob('gates_*')))}"    # each run its own immutable artifact
    argv = [arg for path in readouts for arg in ("--segment-readout", str(path))]
    argv += ["--gate-arms", gate_arms, "--out", str(out)]
    assert gates.main(argv) == 0
    return json.loads((out / "two_tier_gates.json").read_text())


def test_l2_passes_when_both_seeds_beat_every_reference_by_the_seed_line_and_hold_straight_in(tmp_path):
    readout = _segment_readout(tmp_path, "readout", {
        "A_s1337": {"native32": (-125.0, 0.0), "state": (-130.0, -5.0)},
        "A_s2024": {"native32": (-200.0, -1.0), "state": (-126.0, 0.0)},
    })
    result = _run_l2(tmp_path, readout)
    block = result["gates"][gates.GATE_L2]
    assert block["pass"] and set(block["verdicts"]) == {"A_s1337", "A_s2024"}
    verdict = block["verdicts"]["A_s1337"]
    assert verdict["references"] == ["native32", "state"]      # the constant-velocity floor is never judged
    assert len(verdict["criteria"]) == 2 * 2 * 2                 # references × leads × strata
    assert all(c["pass"] for c in verdict["criteria"])
    assert result["schema"] == gates.SCHEMA and block["baseline"] == ["constant-velocity", "native32", "state"]
    assert set(block["arms"]) == {"A_s1337", "A_s2024"}
    text = (tmp_path / "gates_0" / "two_tier_gates.txt").read_text()
    assert "L2: PASS" in text and gates.CONSTANT_VELOCITY in text and "held past the forecast's end 0 / 1" in text


@pytest.mark.parametrize("deltas, failing", [
    ({"native32": (-124.0, 0.0)}, "vectored p50 at 120 s − native32"),   # short of the seed line against one reference
    ({"state": (-300.0, 1.0)}, "straight-in p50 at 120 s − state"),      # straight-in worse
])
def test_l2_fails_on_one_seed_and_names_the_criterion(tmp_path, deltas, failing):
    readout = _segment_readout(tmp_path, "readout", {"A_s1337": {}, "A_s2024": deltas})
    block = _run_l2(tmp_path, readout)["gates"][gates.GATE_L2]
    assert not block["pass"] and block["verdicts"]["A_s1337"]["pass"] and not block["verdicts"]["A_s2024"]["pass"]
    failed = [c["criterion"] for c in block["verdicts"]["A_s2024"]["criteria"] if not c["pass"]]
    assert failed and all(name.startswith(failing) or name.startswith(failing.replace("120", "180")) for name in failed)


def test_l2_judges_two_readouts_together_only_when_they_agree_on_references_anchor_and_leads(tmp_path):
    one = _segment_readout(tmp_path, "one", {"A_s1337": {}})
    two = _segment_readout(tmp_path, "two", {"A_s2024": {}})
    block = _run_l2(tmp_path, one, two)["gates"][gates.GATE_L2]
    assert block["pass"] and {v["artifact"] for v in block["verdicts"].values()} == {str(one / "segment_plan_readout.json"),
                                                                                       str(two / "segment_plan_readout.json")}
    other = _segment_readout(tmp_path, "other", {"A_s2024": {}}, references=("native32",))
    with pytest.raises(SystemExit, match="one gate reads one reference set"):
        _run_l2(tmp_path, one, other)
    with pytest.raises(SystemExit, match="appears in two readouts"):
        _run_l2(tmp_path, one, _segment_readout(tmp_path, "dup", {"A_s1337": {}, "A_s2024": {}}))


def test_l2_refuses_a_smoke_readout_a_missing_gate_arm_or_a_reference_less_readout(tmp_path):
    with pytest.raises(SystemExit, match="never a smoke run"):
        _run_l2(tmp_path, _segment_readout(tmp_path, "smoke", {"A_s1337": {}, "A_s2024": {}}, limit=10))
    with pytest.raises(SystemExit, match="in no segment-plan readout"):
        _run_l2(tmp_path, _segment_readout(tmp_path, "one", {"A_s1337": {}}))
    with pytest.raises(SystemExit, match="no whole-approach reference"):
        _run_l2(tmp_path, _segment_readout(tmp_path, "cv", {"A_s1337": {}, "A_s2024": {}}, references=()))
    with pytest.raises(SystemExit, match="schema"):
        _run_l2(tmp_path, _segment_readout(tmp_path, "old", {"A_s1337": {}, "A_s2024": {}}, schema="ts-old"))
    parser = gates.build_parser()
    with pytest.raises(SystemExit):
        gates.main(["--gate-arms", "a,b", "--out", str(tmp_path / "nothing")])
