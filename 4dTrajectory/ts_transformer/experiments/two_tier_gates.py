"""Two-tier gates L1 / L2 / G1 / G3 off `tracker_lockstep` and `segment_plan_readout` artifacts, flight by flight.

Two-tier v2 `docs/2026-09-17_two_tier_plan_v2.zh.md` §3 (gate L1, the drift reading) over the feasibility
design's §6 / §10.7 gates (G1 / G3, the abandoned T1a design's, kept for the artifacts that carry them).
A gate is judged per GATE ARM (the checkpoint labels ``--gate-arms`` names, the two seeds) on the
``receding`` variant, and passes only when every gate arm passes every criterion:

* **L1** — an artifact whose plans are the truth's coarse WAYPOINTS (`tracker_lockstep.TruthWaypoints`,
  protocol C): vectored ADE mean < 1500 m AND straight-in < 250 m (better than the rule guidance's
  1847 / 283 m); fully flyable on ≥ 95 % of all flights; established on ≥ 0.88 × the guidance's share per
  stratum over the SAME flights (``--baseline``, as G1). Beside it, never inside it, the DRIFT reading:
  the per-ask step error (`asks_e_m`) pooled per stratum, its p50 over asks ≥ 3 against the p50 over
  asks 0–2 — a ratio over 1.5 is "rising" and triggers S5 (training on the tracker's own states).

* **G1** — an artifact whose plans are the TRUTH's (protocol C): vectored ADE mean < 1000 m AND
  straight-in < 200 m; fully flyable on ≥ 95 % of all flights; established (crossed the threshold on the
  final) on ≥ 0.88 × the rule guidance's share, per stratum, over the SAME flights — the guidance being
  ``--baseline``, the plan oracle flying the truth's plan in lockstep at the same anchor
  (`plan_guidance_20260910/step3d_lockstep_l1`). The two artifacts must cover the same flights at the same
  anchor, and both carry the plan oracle's own `reference_verdicts`, so "established" and "fully flyable"
  are one definition on both sides.
* **G3** — an artifact whose plans are a plan HEAD's (protocol A): vectored ADE mean ≤ 2745 m AND
  straight-in ≤ 415 m (native32's one-shot 2870 / 445 less the seed lines, pre-registered constants);
  fully flyable on ≥ 95 % of all flights; established on ≥ 94 % of all flights. §6 wrote "fully flyable not
  below the plan path"; the plan path's own lockstep is flyable on 1.000 of its flights (the rule guidance
  is flyable by construction), so that bar would demand every flight — set to G1's 0.95 before any T2
  number existed (§10.7), with the plan path's share (``--plan-path-baseline``, another anchor and cohort)
  reported beside it.

* **L2** — off a `segment_plan_readout` artifact (``--segment-readout``, two-tier v2 §4): at the common
  fixed anchor, for every whole-approach reference the readout names (native32, the state arm — the
  built-in constant-velocity extrapolation is a floor reading, shown beside, never judged), at 120 s AND
  180 s: the vectored waypoint p50 over the flights both hold is at least the seed line (125 m) BELOW the
  reference's, and the straight-in p50 is not above it. Both seeds; "the stronger of the references" is
  beating every reference.

Refused: an artifact of a smoke run (``--limit``) or of any split but ``val``; fewer than two gate arms
(the gates are two-seed gates); for G1, a baseline of another schema, protocol (policy ``truth``, route
``next``, rolling ``lockstep``), split, limit, flight set, per-row anchor or stratum membership; a gate
stratum with no flight.

Beside the verdict, for every arm (gate arm or not): each variant's per-stratum ADE and reference shares;
paired per-flight ΔADE of ``receding`` against ``receding-no-plan`` (what the plan buys — risks 2 and 6)
and against ``one-shot`` (what re-asking buys), and, for G1, against the guidance; the displacement p50 at
the lockstep's leads per variant (the drift reading: a ``receding`` error that grows faster with lead than
the ``one-shot``'s is what would make T1c necessary, §10.4 — read, not judged here).

    python run_ts.py two_tier_gates --lockstep <tracker_lockstep dir> [--lockstep <dir> ...] \\
        --gate-arms T1a_plan_p50_s1337,T1a_plan_p50_s2024 \\
        --baseline <plan_guidance_20260910/step3d_lockstep_l1> \\
        [--plan-path-baseline <plan_guidance_20260910/step3g_fan4_top1_lockstep_l1>] --out <dir>
    python run_ts.py two_tier_gates --segment-readout <segment_plan_readout dir> --gate-arms A_s1337,A_s2024 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, strata_masks
from ts_transformer.experiments.plan_oracle import (  # noqa: E402
    POLICY_TRUTH,
    ROLLING_LOCKSTEP,
    ROUTE_NEXT,
    SCHEMA as PLAN_ORACLE_SCHEMA,
)
from ts_transformer.experiments.segment_plan_readout import (
    CONSTANT_VELOCITY,
    FIXED_SET,
    RESULT_SCHEMA as SEGMENT_READOUT_SCHEMA,
)
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.experiments.tracker_lockstep import (
    RESULT_SCHEMA as LOCKSTEP_SCHEMA,
    VARIANT_NO_PLAN,
    VARIANT_ONE_SHOT,
    VARIANT_RECEDING,
    HeadPlans,
    TruthPlans,
    TruthWaypoints,
)
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic

SCHEMA = "ts-two-tier-gates-v3"   # v3 (2026-09-17): gate L2 off the segment-plan readout; v2: gate L1 and the drift reading
GATE_G1 = "G1"
GATE_G3 = "G3"
GATE_L1 = "L1"
GATE_L2 = "L2"
#: The gate an artifact is judged under, by where its plans came from (`tracker_lockstep.plan_source`).
GATE_BY_SOURCE = {TruthPlans.source: GATE_G1, HeadPlans.source: GATE_G3, TruthWaypoints.source: GATE_L1}
GATE_STRATA = (STRATUM_STRAIGHT_IN, STRATUM_VECTORED)
#: The split every gate is read on, and the fewest gate arms (seeds) a gate is judged over.
GATE_SPLIT = "val"
MIN_GATE_ARMS = 2
READ_STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED)


@dataclass(frozen=True)
class TruthGate:
    """A gate on a TRUTH-plan artifact (protocol C): the tracker flies the truth's plan closer than
    the rule guidance does, stays flyable, and gets established nearly as often as the guidance
    over the SAME flights."""

    ade_below_m: dict[str, float]
    fully_flyable_at_least: float
    established_fraction_of_guidance: float


# G1 (feasibility §6, §10.3): the whole-approach tracker under the instruction plan (T1a, abandoned).
G1_ADE_BELOW_M = {STRATUM_VECTORED: 1000.0, STRATUM_STRAIGHT_IN: 200.0}
G1_FULLY_FLYABLE_AT_LEAST = 0.95
G1_ESTABLISHED_FRACTION_OF_GUIDANCE = 0.88
# L1 (two-tier v2 §3): the short-horizon control layer under the truth's coarse waypoints, in
# lockstep — better than the rule guidance (1847 / 283 m), flyable, established ≥ 88 % of it.
L1_ADE_BELOW_M = {STRATUM_VECTORED: 1500.0, STRATUM_STRAIGHT_IN: 250.0}
L1_FULLY_FLYABLE_AT_LEAST = 0.95
L1_ESTABLISHED_FRACTION_OF_GUIDANCE = 0.88
TRUTH_GATES = {
    GATE_G1: TruthGate(G1_ADE_BELOW_M, G1_FULLY_FLYABLE_AT_LEAST, G1_ESTABLISHED_FRACTION_OF_GUIDANCE),
    GATE_L1: TruthGate(L1_ADE_BELOW_M, L1_FULLY_FLYABLE_AT_LEAST, L1_ESTABLISHED_FRACTION_OF_GUIDANCE),
}
# G3 (§6): the two tiers end to end beat native32's one-shot (2870 / 445 m) by the seed lines (125 / 30 m).
G3_ADE_AT_MOST_M = {STRATUM_VECTORED: 2745.0, STRATUM_STRAIGHT_IN: 415.0}
G3_FULLY_FLYABLE_AT_LEAST = 0.95
G3_ESTABLISHED_AT_LEAST = 0.94
# The DRIFT reading (two-tier v2 §3, "我替你选的"): read beside every arm, never part of a gate — it
# decides whether S5 (training on the tracker's own states) is triggered. The per-ask step error
# (`asks_e_m`, the displacement at the end of each leg) is pooled per stratum; "rising" = the p50
# over asks from DRIFT_LATE_FROM_ASK on exceeds DRIFT_RATIO_MAX × the p50 over the asks before it.
DRIFT_LATE_FROM_ASK = 3
DRIFT_RATIO_MAX = 1.5
# L2 (two-tier v2 §4): the plan head's vectored waypoint p50 at these leads beats EVERY whole-approach
# reference by the seed line, and its straight-in p50 is not worse, over the flights both hold at the
# readout's fixed anchor.
L2_LEADS_S = ("120", "180")
L2_SEED_LINE_M = 125.0
L2_STRAIGHT_IN_MAX_WORSE_M = 0.0
L2_SET = FIXED_SET


@dataclass(frozen=True)
class ArmReading:
    """One checkpoint of one lockstep artifact."""

    gate: str
    label: str
    artifact: str
    anchor: int
    variants: dict[str, dict[str, dict]]     # variant -> dataset_id -> flight row


def _p50(values) -> float | None:
    return float(np.median(values)) if len(values) else None


def _mean(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


def load_lockstep(path: Path) -> list[ArmReading]:
    file = path / "tracker_lockstep.json" if path.is_dir() else path
    payload = json.loads(file.read_text(encoding="utf-8"))
    if payload.get("schema") != LOCKSTEP_SCHEMA:
        raise SystemExit(
            f"{file}: schema {payload.get('schema')!r}, this readout needs {LOCKSTEP_SCHEMA!r} (its flight rows "
            "carry the plan oracle's reference verdicts) — re-run tracker_lockstep at this code"
        )
    plan = payload["plan"]
    if plan["limit"] or plan["split"] != GATE_SPLIT:
        raise SystemExit(f"{file}: split {plan['split']!r}, limit {plan['limit']} — a gate reads the whole "
                         f"{GATE_SPLIT!r} split, never a smoke run")
    source = payload["plan_source"].split(":", 1)[0]
    gate = GATE_BY_SOURCE[source]
    return [
        ArmReading(gate=gate, label=label, artifact=str(file), anchor=int(block["anchor"]),
                   variants={name: variant["flights"] for name, variant in block["variants"].items()})
        for label, block in payload["checkpoints"].items()
    ]


def load_segment_readout(path: Path) -> dict:
    file = path / "segment_plan_readout.json" if path.is_dir() else path
    payload = json.loads(file.read_text(encoding="utf-8"))
    if payload.get("schema") != SEGMENT_READOUT_SCHEMA:
        raise SystemExit(f"{file}: schema {payload.get('schema')!r}, this readout needs {SEGMENT_READOUT_SCHEMA!r}")
    plan = payload["plan"]
    if plan["limit"] or plan["split"] != GATE_SPLIT:
        raise SystemExit(f"{file}: split {plan['split']!r}, limit {plan['limit']} — a gate reads the whole "
                         f"{GATE_SPLIT!r} split, never a smoke run")
    missing = [lead for lead in L2_LEADS_S if lead not in {f"{h:g}" for h in plan["leads_s"]}]
    if missing:
        raise SystemExit(f"{file}: the readout has no lead {missing} (leads {plan['leads_s']}); gate L2 reads {L2_LEADS_S}")
    if L2_SET not in payload["plan"]["bins"]:
        raise SystemExit(f"{file}: no {L2_SET!r} anchor set; gate L2 is judged at the fixed anchor")
    payload["path"] = str(file)
    return payload


def load_plan_oracle(path: Path) -> dict:
    file = path / "plan_oracle.json" if path.is_dir() else path
    artifact = json.loads(file.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != PLAN_ORACLE_SCHEMA:
        raise SystemExit(f"{file}: schema {artifact.get('schema_version')!r}, this readout reads {PLAN_ORACLE_SCHEMA!r}")
    artifact["path"] = str(file)
    artifact["rows_by_id"] = {row["dataset_id"]: row for row in artifact.pop("rows")}
    return artifact


def masks_of(rows: dict[str, dict], ids: list[str]) -> dict[str, np.ndarray]:
    return strata_masks({i: rows[i]["difficulty"] for i in ids}, ids)


def variant_block(rows: dict[str, dict]) -> dict:
    ids = sorted(rows)
    masks = masks_of(rows, ids)
    out = {}
    for stratum in READ_STRATA:
        members = [i for i, keep in zip(ids, masks[stratum], strict=True) if keep]
        out[stratum] = {
            "n": len(members),
            "ade_mean_m": _mean([rows[i]["ade_m"] for i in members]),
            "ade_p50_m": _p50([rows[i]["ade_m"] for i in members]),
            "fde_p50_m": _p50([rows[i]["fde_m"] for i in members]),
            "fully_flyable_share": _mean([float(rows[i]["reference"]["fully_flyable"]) for i in members]),
            "established_share": _mean([float(rows[i]["reference"]["established"]) for i in members]),
            "at_lead_p50_m": {
                lead: _p50([rows[i]["at"][lead] for i in members if rows[i]["at"][lead] is not None])
                for lead in (rows[members[0]]["at"] if members else {})
            },
        }
    return out


def paired(arm_rows: dict[str, dict], base_ade: dict[str, float]) -> dict:
    """ΔADE = arm − base per flight over the flights both carry, per stratum (the ARM's strata)."""
    ids = sorted(set(arm_rows) & set(base_ade))
    masks = masks_of(arm_rows, ids)
    out = {}
    for stratum in READ_STRATA:
        delta = np.array([arm_rows[i]["ade_m"] - base_ade[i] for i, keep in zip(ids, masks[stratum], strict=True) if keep])
        out[stratum] = {
            "n": int(delta.size),
            "delta_mean_m": float(delta.mean()) if delta.size else None,
            "delta_p50_m": float(np.median(delta)) if delta.size else None,
            "arm_better_share": float(np.mean(delta < 0.0)) if delta.size else None,
        }
    return out


def criterion(name: str, value: float | None, threshold: float, relation: str) -> dict:
    ok = value is not None and {"<": value < threshold, "<=": value <= threshold, ">=": value >= threshold}[relation]
    return {"criterion": name, "value": value, "threshold": threshold, "relation": relation, "pass": bool(ok)}


def require_same_cohort(arm: ArmReading, baseline: dict) -> None:
    """The guidance baseline is the rule guidance flying the TRUTH's plan in lockstep over the tracker's
    own flights, anchor and strata — or G1 compares two different experiments."""
    receding = arm.variants[VARIANT_RECEDING]
    rows = baseline["rows_by_id"]
    protocol = {"policy": POLICY_TRUTH, "route": ROUTE_NEXT, "rolling": ROLLING_LOCKSTEP, "split": GATE_SPLIT, "limit": 0}
    wrong = {key: baseline.get(key) for key, value in protocol.items() if baseline.get(key) != value}
    if wrong:
        raise SystemExit(f"{baseline['path']}: {wrong} — G1's baseline is the guidance flying the truth's plan in "
                         f"lockstep over the whole {GATE_SPLIT!r} split ({protocol})")
    # every tracker flight must be in the baseline (the pairing is over the tracker's flights); a
    # baseline flight the tracker's cohort dropped (a fixed horizon admits only flights with Δ of
    # truth after the anchor) is stated in the verdict, never silently absent
    missing = sorted(set(receding) - set(rows))
    if missing:
        raise SystemExit(f"{arm.label}: {len(missing)} tracker flights are not in the guidance baseline's "
                         f"{len(rows)} (first: {missing[0]!r}) — a truth gate pairs the same cohort")
    anchors = sorted({int(rows[i]["anchor"]) for i in receding})
    if anchors != [arm.anchor]:
        raise SystemExit(f"{arm.label}: anchor {arm.anchor} against the guidance baseline's row anchors {anchors[:5]} "
                         f"({baseline['path']}) — a truth gate pairs the same flights at the same anchor")
    ids = sorted(receding)
    ours, theirs = masks_of(receding, ids), masks_of(rows, ids)
    moved = {stratum: int(np.sum(ours[stratum] != theirs[stratum])) for stratum in READ_STRATA}
    if any(moved.values()):
        raise SystemExit(f"{arm.label}: flights stratified differently in the tracker and the baseline {moved}")


def baseline_flights_not_in_tracker(arm: ArmReading, baseline: dict) -> int:
    return len(set(baseline["rows_by_id"]) - set(arm.variants[VARIANT_RECEDING]))


def require_members(arm: ArmReading, block: dict) -> None:
    empty = [stratum for stratum in GATE_STRATA if not block[stratum]["n"]]
    if empty:
        raise SystemExit(f"{arm.label} ({arm.artifact}): no flight in {empty}; a gate stratum cannot be judged empty")


def judge_truth_gate(arm: ArmReading, baseline: dict, gate: TruthGate) -> dict:
    require_same_cohort(arm, baseline)
    rows = arm.variants[VARIANT_RECEDING]
    block = variant_block(rows)
    require_members(arm, block)
    ids = sorted(rows)
    masks = masks_of(rows, ids)
    guidance = baseline["rows_by_id"]
    criteria = [
        criterion(f"ADE mean {stratum}", block[stratum]["ade_mean_m"], gate.ade_below_m[stratum], "<")
        for stratum in GATE_STRATA
    ]
    criteria.append(criterion("fully flyable share, all", block[STRATUM_ALL]["fully_flyable_share"],
                              gate.fully_flyable_at_least, ">="))
    for stratum in GATE_STRATA:
        members = [i for i, keep in zip(ids, masks[stratum], strict=True) if keep]
        guidance_share = _mean([float(guidance[i]["reference"]["established"]) for i in members])
        criteria.append(criterion(
            f"established share {stratum} (guidance {guidance_share:.3f} on the same flights)",
            block[stratum]["established_share"], gate.established_fraction_of_guidance * guidance_share, ">=",
        ))
    return {"criteria": criteria, "pass": all(c["pass"] for c in criteria),
            "baseline_flights_not_in_tracker": baseline_flights_not_in_tracker(arm, baseline)}


def judge_g1(arm: ArmReading, baseline: dict) -> dict:
    return judge_truth_gate(arm, baseline, TRUTH_GATES[GATE_G1])


def judge_l1(arm: ArmReading, baseline: dict) -> dict:
    return judge_truth_gate(arm, baseline, TRUTH_GATES[GATE_L1])


def judge_l2(label: str, readout: dict) -> dict:
    """Gate L2 for one plan head off the readout's paired block at the fixed anchor."""
    references = [name for name in readout["references"] if name != CONSTANT_VELOCITY]
    if not references:
        raise SystemExit(f"{readout['path']}: no whole-approach reference beside {CONSTANT_VELOCITY!r}; gate L2 needs one")
    criteria = []
    for reference in references:
        cells = readout["paired"][label][reference][L2_SET]
        for lead in L2_LEADS_S:
            vectored, straight = cells[STRATUM_VECTORED][lead], cells[STRATUM_STRAIGHT_IN][lead]
            if not vectored["n"] or not straight["n"]:
                raise SystemExit(f"{label} − {reference} at {lead} s: an empty gate stratum "
                                 f"(vectored n {vectored['n']}, straight-in n {straight['n']}) cannot be judged")
            criteria.append(criterion(
                f"vectored p50 at {lead} s − {reference}'s over the same {vectored['n']} flights "
                f"({_fmt(vectored['arm_p50_m'])} − {_fmt(vectored['reference_p50_m'])}; held past the forecast's end "
                f"{vectored['arm_held']} / {vectored['reference_held']})",
                vectored["delta_of_p50_m"], -L2_SEED_LINE_M, "<=",
            ))
            criteria.append(criterion(
                f"straight-in p50 at {lead} s − {reference}'s over the same {straight['n']} flights "
                f"({_fmt(straight['arm_p50_m'])} − {_fmt(straight['reference_p50_m'])}; held "
                f"{straight['arm_held']} / {straight['reference_held']})",
                straight["delta_of_p50_m"], L2_STRAIGHT_IN_MAX_WORSE_M, "<=",
            ))
    return {"criteria": criteria, "pass": all(c["pass"] for c in criteria), "references": references}


def l2_arm_readings(label: str, readout: dict) -> dict:
    block = readout["checkpoints"][label]
    return {
        "artifact": readout["path"], "anchor": readout["anchor"], "own_fixed_anchor": block["own_fixed_anchor"],
        "sets": {set_label: {"strata": cell["strata"], "plan": cell.get("plan")} for set_label, cell in block["sets"].items()},
        "paired": readout["paired"].get(label, {}),
    }


def drift_reading(rows: dict[str, dict]) -> dict:
    """The per-ask step error of one variant, per stratum: its p50 by ask, the early / late
    p50s and their ratio, and whether it is RISING (`DRIFT_RATIO_MAX`). A stratum whose flights
    never reach ask `DRIFT_LATE_FROM_ASK` has no late reading and no verdict (None)."""
    ids = sorted(rows)
    masks = masks_of(rows, ids)
    out = {}
    for stratum in READ_STRATA:
        members = [i for i, keep in zip(ids, masks[stratum], strict=True) if keep]
        entries = [entry for i in members for entry in rows[i]["asks_e_m"] if entry["e_m"] is not None]
        by_ask: dict[int, list[float]] = {}
        for entry in entries:
            by_ask.setdefault(int(entry["ask"]), []).append(float(entry["e_m"]))
        early = [entry["e_m"] for entry in entries if entry["ask"] < DRIFT_LATE_FROM_ASK]
        late = [entry["e_m"] for entry in entries if entry["ask"] >= DRIFT_LATE_FROM_ASK]
        early_p50, late_p50 = _p50(early), _p50(late)
        ratio = None if not early_p50 or late_p50 is None else late_p50 / early_p50
        out[stratum] = {
            "p50_by_ask_m": {str(ask): {"n": len(v), "p50": _p50(v)} for ask, v in sorted(by_ask.items())},
            "early_p50_m": early_p50, "late_p50_m": late_p50, "late_from_ask": DRIFT_LATE_FROM_ASK,
            "ratio": ratio, "ratio_max": DRIFT_RATIO_MAX,
            "rising": None if ratio is None else bool(ratio > DRIFT_RATIO_MAX),
        }
    return out


def judge_g3(arm: ArmReading, plan_path: dict | None) -> dict:
    block = variant_block(arm.variants[VARIANT_RECEDING])
    require_members(arm, block)
    criteria = [
        criterion(f"ADE mean {stratum}", block[stratum]["ade_mean_m"], G3_ADE_AT_MOST_M[stratum], "<=")
        for stratum in GATE_STRATA
    ]
    beside = "" if plan_path is None else (
        f" (the plan path's own lockstep: {float(plan_path['summary'][STRATUM_ALL]['fully_flyable_share']):.3f}, "
        f"anchor {plan_path['anchor']}, {plan_path['flights']} flights — read beside, not a pairing)"
    )
    criteria.append(criterion(
        f"fully flyable share, all{beside}", block[STRATUM_ALL]["fully_flyable_share"], G3_FULLY_FLYABLE_AT_LEAST, ">=",
    ))
    criteria.append(criterion("established share, all", block[STRATUM_ALL]["established_share"],
                              G3_ESTABLISHED_AT_LEAST, ">="))
    return {"criteria": criteria, "pass": all(c["pass"] for c in criteria)}


def arm_readings(arm: ArmReading, baseline: dict | None) -> dict:
    receding = arm.variants[VARIANT_RECEDING]
    against = {
        name: {i: row["ade_m"] for i, row in arm.variants[name].items()}
        for name in (VARIANT_NO_PLAN, VARIANT_ONE_SHOT) if name in arm.variants
    }
    if arm.gate in TRUTH_GATES and baseline is not None:
        against["guidance"] = {i: row["prediction"]["ade_m"] for i, row in baseline["rows_by_id"].items()}
    return {
        "artifact": arm.artifact, "anchor": arm.anchor,
        "variants": {name: variant_block(rows) for name, rows in arm.variants.items()},
        "receding_minus": {name: paired(receding, base) for name, base in against.items()},
        # the drift reading of every variant: read, never judged (v2 §3 — it triggers S5)
        "drift": {name: drift_reading(rows) for name, rows in arm.variants.items()},
    }


def build(arms: list[ArmReading], gate_arms: tuple[str, ...], baseline: dict | None, plan_path: dict | None,
          segment_readouts: list[dict] = ()) -> dict:
    if len(set(gate_arms)) < MIN_GATE_ARMS:
        raise SystemExit(f"--gate-arms names {len(set(gate_arms))} arm(s); a gate is judged over at least "
                         f"{MIN_GATE_ARMS} seeds")
    result: dict = {"schema": SCHEMA, "generated_at": utc_now(), "gate_arms": list(gate_arms), "gates": {}}
    if segment_readouts:
        # every readout judged together names the same references, anchor and leads, or the seeds
        # would be judged against different things under one verdict
        first = segment_readouts[0]
        for readout in segment_readouts[1:]:
            for key in ("references", "anchor"):
                if readout[key] != first[key]:
                    raise SystemExit(f"{GATE_L2}: {readout['path']} has {key} {readout[key]!r}, {first['path']} "
                                     f"{first[key]!r}; one gate reads one reference set at one anchor")
            if readout["plan"]["leads_s"] != first["plan"]["leads_s"]:
                raise SystemExit(f"{GATE_L2}: the readouts' leads differ ({readout['plan']['leads_s']} against "
                                 f"{first['plan']['leads_s']})")
        by_label = {}
        for readout in segment_readouts:
            for label in readout["arms"]:
                if label in by_label:
                    raise SystemExit(f"{GATE_L2}: a checkpoint label appears in two readouts: {label!r}")
                by_label[label] = readout
        missing = [label for label in gate_arms if label not in by_label]
        if missing:
            raise SystemExit(f"{GATE_L2}: gate arm(s) {missing} are in no segment-plan readout (have {sorted(by_label)})")
        verdicts = {label: {**judge_l2(label, by_label[label]), "artifact": by_label[label]["path"]} for label in gate_arms}
        result["gates"][GATE_L2] = {
            "pass": all(v["pass"] for v in verdicts.values()),
            "verdicts": verdicts,
            "arms": {label: l2_arm_readings(label, readout) for label, readout in by_label.items()},
            "baseline": sorted({name for readout in segment_readouts for name in readout["references"]}),
        }
    for gate in sorted({arm.gate for arm in arms}):
        members = [arm for arm in arms if arm.gate == gate]
        labels = [arm.label for arm in members]
        if len(set(labels)) != len(labels):
            raise SystemExit(f"{gate}: a checkpoint label appears in two artifacts: {labels}")
        missing = [label for label in gate_arms if label not in labels]
        if missing:
            raise SystemExit(f"{gate}: gate arm(s) {missing} are in no {gate} artifact (have {labels})")
        if gate in TRUTH_GATES and baseline is None:
            raise SystemExit(f"a truth-plan ({gate}) artifact needs --baseline, the guidance's plan-oracle artifact")
        verdicts = {
            arm.label: (judge_truth_gate(arm, baseline, TRUTH_GATES[gate]) if gate in TRUTH_GATES
                        else judge_g3(arm, plan_path))
            for arm in members if arm.label in gate_arms
        }
        result["gates"][gate] = {
            "pass": all(v["pass"] for v in verdicts.values()),
            "verdicts": verdicts,
            "arms": {arm.label: arm_readings(arm, baseline) for arm in members},
            "baseline": baseline["path"] if gate in TRUTH_GATES else (None if plan_path is None else plan_path["path"]),
        }
    return result


def _fmt(value, digits: int = 0) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def render(result: dict) -> str:
    lines = [f"two-tier gates — gate arms {', '.join(result['gate_arms'])} (each must pass every criterion)"]
    for gate, block in result["gates"].items():
        lines += ["", f"══ {gate}: {'PASS' if block['pass'] else 'FAIL'}   (baseline {block['baseline']})"]
        if gate == GATE_L2:
            lines += _render_l2(block)
            continue
        for label, verdict in block["verdicts"].items():
            dropped = verdict.get("baseline_flights_not_in_tracker", 0)
            lines.append(f"  {label}: {'pass' if verdict['pass'] else 'FAIL'}"
                         + (f"   ({dropped} baseline flights are not in the tracker's cohort)" if dropped else ""))
            for c in verdict["criteria"]:
                lines.append(f"    [{'x' if c['pass'] else ' '}] {c['criterion']}: {_fmt(c['value'], 3)} "
                             f"{c['relation']} {_fmt(c['threshold'], 3)}")
        for label, arm in block["arms"].items():
            lines.append(f"  ── {label} ({arm['artifact']}, a0={arm['anchor']})")
            for variant, strata in arm["variants"].items():
                for stratum, cell in strata.items():
                    leads = " ".join(f"{k}s {_fmt(v)}" for k, v in cell["at_lead_p50_m"].items())
                    lines.append(f"     {variant:<17s} {stratum[:30]:<30s} n={cell['n']:>4d} ADE {_fmt(cell['ade_mean_m']):>5}/"
                                 f"{_fmt(cell['ade_p50_m']):>5} FDE50 {_fmt(cell['fde_p50_m']):>5} "
                                 f"flyable {_fmt(cell['fully_flyable_share'], 3)} established {_fmt(cell['established_share'], 3)}"
                                 f" | disp p50 {leads}")
            for name, strata in arm["receding_minus"].items():
                for stratum, cell in strata.items():
                    lines.append(f"     receding − {name:<16s} {stratum[:30]:<30s} n={cell['n']:>4d} ΔADE mean "
                                 f"{_fmt(cell['delta_mean_m']):>6} p50 {_fmt(cell['delta_p50_m']):>6} "
                                 f"receding better {_fmt(cell['arm_better_share'], 3)}")
            for variant, strata in arm["drift"].items():
                for stratum, cell in strata.items():
                    verdict = "—" if cell["rising"] is None else ("RISING" if cell["rising"] else "flat")
                    curve = " ".join(f"k{ask} {_fmt(v['p50'])}" for ask, v in cell["p50_by_ask_m"].items())
                    lines.append(f"     drift {variant:<17s} {stratum[:30]:<30s} early p50 {_fmt(cell['early_p50_m']):>5} "
                                 f"late(≥k{cell['late_from_ask']}) {_fmt(cell['late_p50_m']):>5} ratio "
                                 f"{_fmt(cell['ratio'], 2):>5} (max {cell['ratio_max']:g}) {verdict} | {curve}")
    return "\n".join(lines) + "\n"


def _render_l2(block: dict) -> list[str]:
    lines = []
    for label, verdict in block["verdicts"].items():
        lines.append(f"  {label}: {'pass' if verdict['pass'] else 'FAIL'}   (against {', '.join(verdict['references'])}; "
                     f"{CONSTANT_VELOCITY} read beside, never judged)")
        for c in verdict["criteria"]:
            lines.append(f"    [{'x' if c['pass'] else ' '}] {c['criterion']}: {_fmt(c['value'], 1)} {c['relation']} "
                         f"{_fmt(c['threshold'], 1)}")
    for label, arm in block["arms"].items():
        lines.append(f"  ── {label} ({arm['artifact']}, a0={arm['anchor']}, own fixed anchor {arm['own_fixed_anchor']})")
        for set_label, cell in arm["sets"].items():
            for stratum, s in cell["strata"].items():
                leads = " ".join(f"e({k}) {_fmt(v['p50'])} n{v['n']}" for k, v in s["at_lead_m"].items())
                lines.append(f"     {set_label:<6s} {stratum[:30]:<30s} n={s['n']:>4d} | {leads}")
            if cell.get("plan"):
                for stratum, p in cell["plan"].items():
                    if p is None:
                        continue
                    c = p["arrival_confusion"]
                    lines.append(f"     {set_label:<6s} {stratum[:30]:<30s} plan: covered ADE {_fmt(p['covered_ade_m'])} | "
                                 f"arrives plan/truth {p['plan_arrives_share']:.2f}/{p['truth_arrives_share']:.2f} "
                                 f"(both {c['both']}, plan-only {c['plan_only']}, truth-only {c['truth_only']}) | "
                                 f"|dT| {_fmt(p['arrival_time_error_s']['mean_abs'], 1)} s n{p['arrival_time_error_s']['n']}")
        for reference, sets in arm["paired"].items():
            for set_label, strata in sets.items():
                for stratum, cells in strata.items():
                    parts = " ".join(f"{k}s {_fmt(v['delta_of_p50_m'])} (n{v['n']})" for k, v in cells.items())
                    lines.append(f"     − {reference:<17s} {set_label:<6s} {stratum[:30]:<30s} Δ of p50 {parts}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--lockstep", type=Path, action="append", default=None,
                        help="a tracker_lockstep artifact (repeatable; truth-plan artifacts are judged under G1 / L1, "
                             "head-plan ones under G3)")
    parser.add_argument("--segment-readout", type=Path, action="append", default=None,
                        help="a segment_plan_readout artifact (repeatable): gate L2")
    parser.add_argument("--gate-arms", required=True, help="comma-separated checkpoint labels the gates judge")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="G1: the rule guidance's plan-oracle artifact at the same anchor and cohort")
    parser.add_argument("--plan-path-baseline", type=Path, default=None,
                        help="G3: the plan path's own plan-oracle lockstep artifact, its flyable share read beside")
    parser.add_argument("--out", type=Path, required=True, help="must not exist (immutable artifact)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        raise FileExistsError(f"{out} exists; the gate readout is an immutable artifact")
    gate_arms = tuple(label.strip() for label in args.gate_arms.split(",") if label.strip())
    if not gate_arms:
        parser.error("--gate-arms names no checkpoint")
    if not args.lockstep and not args.segment_readout:
        parser.error("nothing to judge: give --lockstep and/or --segment-readout")
    arms = [arm for path in (args.lockstep or []) for arm in load_lockstep(path)]
    segment_readouts = [load_segment_readout(path) for path in (args.segment_readout or [])]
    for arm in arms:
        if VARIANT_RECEDING not in arm.variants:
            raise SystemExit(f"{arm.label} ({arm.artifact}): no {VARIANT_RECEDING!r} variant, which every gate reads")
    baseline = None if args.baseline is None else load_plan_oracle(args.baseline)
    plan_path = None if args.plan_path_baseline is None else load_plan_oracle(args.plan_path_baseline)
    result = build(arms, gate_arms, baseline, plan_path, segment_readouts)
    result["inputs"] = {
        "lockstep": [{"path": arm_file, "sha256": file_sha256(Path(arm_file))} for arm_file in sorted({arm.artifact for arm in arms})],
        "segment_readouts": [{"path": readout["path"], "sha256": file_sha256(Path(readout["path"]))} for readout in segment_readouts],
        "baseline": None if baseline is None else {"path": baseline["path"], "sha256": file_sha256(Path(baseline["path"]))},
        "plan_path_baseline": None if plan_path is None else {
            "path": plan_path["path"], "sha256": file_sha256(Path(plan_path["path"]))},
    }
    text = render(result)
    out.mkdir(parents=True)
    write_json_atomic(out / "two_tier_gates.json", result)
    (out / "two_tier_gates.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {out / 'two_tier_gates.txt'} and {out / 'two_tier_gates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
