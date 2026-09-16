"""Two-tier gates G1 / G3 off `tracker_lockstep` artifacts, flight by flight against the rule guidance.

Feasibility design `docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §6 (the gates) and §10.7
(this readout, its readings fixed before any T1a number existed). A gate is judged per GATE ARM (the
checkpoint labels ``--gate-arms`` names, the two seeds) on the ``receding`` variant, and passes only when
every gate arm passes every criterion:

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
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, strata_masks
from ts_transformer.experiments.plan_oracle import (
    POLICY_TRUTH,
    ROLLING_LOCKSTEP,
    ROUTE_NEXT,
    SCHEMA as PLAN_ORACLE_SCHEMA,
)
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.experiments.tracker_lockstep import (
    RESULT_SCHEMA as LOCKSTEP_SCHEMA,
    VARIANT_NO_PLAN,
    VARIANT_ONE_SHOT,
    VARIANT_RECEDING,
    HeadPlans,
    TruthPlans,
)
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic

SCHEMA = "ts-two-tier-gates-v1"
GATE_G1 = "G1"
GATE_G3 = "G3"
#: The gate an artifact is judged under, by where its plans came from (`tracker_lockstep.plan_source`).
GATE_BY_SOURCE = {TruthPlans.source: GATE_G1, HeadPlans.source: GATE_G3}
GATE_STRATA = (STRATUM_STRAIGHT_IN, STRATUM_VECTORED)
#: The split every gate is read on, and the fewest gate arms (seeds) a gate is judged over.
GATE_SPLIT = "val"
MIN_GATE_ARMS = 2
READ_STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED)

# G1 (§6, §10.3): the learned tracker flies the truth's plan closer than the rule guidance does.
G1_ADE_BELOW_M = {STRATUM_VECTORED: 1000.0, STRATUM_STRAIGHT_IN: 200.0}
G1_FULLY_FLYABLE_AT_LEAST = 0.95
G1_ESTABLISHED_FRACTION_OF_GUIDANCE = 0.88
# G3 (§6): the two tiers end to end beat native32's one-shot (2870 / 445 m) by the seed lines (125 / 30 m).
G3_ADE_AT_MOST_M = {STRATUM_VECTORED: 2745.0, STRATUM_STRAIGHT_IN: 415.0}
G3_FULLY_FLYABLE_AT_LEAST = 0.95
G3_ESTABLISHED_AT_LEAST = 0.94


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
    if set(receding) != set(rows):
        common = len(set(receding) & set(rows))
        raise SystemExit(f"{arm.label}: {len(receding)} flights against the guidance baseline's "
                         f"{len(rows)} ({common} common) — G1 pairs the same cohort")
    anchors = sorted({int(row["anchor"]) for row in rows.values()})
    if anchors != [arm.anchor]:
        raise SystemExit(f"{arm.label}: anchor {arm.anchor} against the guidance baseline's row anchors {anchors[:5]} "
                         f"({baseline['path']}) — G1 pairs the same flights at the same anchor")
    ids = sorted(receding)
    ours, theirs = masks_of(receding, ids), masks_of(rows, ids)
    moved = {stratum: int(np.sum(ours[stratum] != theirs[stratum])) for stratum in READ_STRATA}
    if any(moved.values()):
        raise SystemExit(f"{arm.label}: flights stratified differently in the tracker and the baseline {moved}")


def require_members(arm: ArmReading, block: dict) -> None:
    empty = [stratum for stratum in GATE_STRATA if not block[stratum]["n"]]
    if empty:
        raise SystemExit(f"{arm.label} ({arm.artifact}): no flight in {empty}; a gate stratum cannot be judged empty")


def judge_g1(arm: ArmReading, baseline: dict) -> dict:
    require_same_cohort(arm, baseline)
    rows = arm.variants[VARIANT_RECEDING]
    block = variant_block(rows)
    require_members(arm, block)
    ids = sorted(rows)
    masks = masks_of(rows, ids)
    guidance = baseline["rows_by_id"]
    criteria = [
        criterion(f"ADE mean {stratum}", block[stratum]["ade_mean_m"], G1_ADE_BELOW_M[stratum], "<")
        for stratum in GATE_STRATA
    ]
    criteria.append(criterion("fully flyable share, all", block[STRATUM_ALL]["fully_flyable_share"],
                              G1_FULLY_FLYABLE_AT_LEAST, ">="))
    for stratum in GATE_STRATA:
        members = [i for i, keep in zip(ids, masks[stratum], strict=True) if keep]
        guidance_share = _mean([float(guidance[i]["reference"]["established"]) for i in members])
        criteria.append(criterion(
            f"established share {stratum} (guidance {guidance_share:.3f} on the same flights)",
            block[stratum]["established_share"], G1_ESTABLISHED_FRACTION_OF_GUIDANCE * guidance_share, ">=",
        ))
    return {"criteria": criteria, "pass": all(c["pass"] for c in criteria)}


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
    if arm.gate == GATE_G1 and baseline is not None:
        against["guidance"] = {i: row["prediction"]["ade_m"] for i, row in baseline["rows_by_id"].items()}
    return {
        "artifact": arm.artifact, "anchor": arm.anchor,
        "variants": {name: variant_block(rows) for name, rows in arm.variants.items()},
        "receding_minus": {name: paired(receding, base) for name, base in against.items()},
    }


def build(arms: list[ArmReading], gate_arms: tuple[str, ...], baseline: dict | None, plan_path: dict | None) -> dict:
    if len(set(gate_arms)) < MIN_GATE_ARMS:
        raise SystemExit(f"--gate-arms names {len(set(gate_arms))} arm(s); a gate is judged over at least "
                         f"{MIN_GATE_ARMS} seeds")
    result: dict = {"schema": SCHEMA, "generated_at": utc_now(), "gate_arms": list(gate_arms), "gates": {}}
    for gate in sorted({arm.gate for arm in arms}):
        members = [arm for arm in arms if arm.gate == gate]
        labels = [arm.label for arm in members]
        if len(set(labels)) != len(labels):
            raise SystemExit(f"{gate}: a checkpoint label appears in two artifacts: {labels}")
        missing = [label for label in gate_arms if label not in labels]
        if missing:
            raise SystemExit(f"{gate}: gate arm(s) {missing} are in no {gate} artifact (have {labels})")
        if gate == GATE_G1 and baseline is None:
            raise SystemExit("a truth-plan (G1) artifact needs --baseline, the guidance's plan-oracle artifact")
        verdicts = {
            arm.label: (judge_g1(arm, baseline) if gate == GATE_G1 else judge_g3(arm, plan_path))
            for arm in members if arm.label in gate_arms
        }
        result["gates"][gate] = {
            "pass": all(v["pass"] for v in verdicts.values()),
            "verdicts": verdicts,
            "arms": {arm.label: arm_readings(arm, baseline) for arm in members},
            "baseline": baseline["path"] if gate == GATE_G1 else (None if plan_path is None else plan_path["path"]),
        }
    return result


def _fmt(value, digits: int = 0) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def render(result: dict) -> str:
    lines = [f"two-tier gates — gate arms {', '.join(result['gate_arms'])} (each must pass every criterion)"]
    for gate, block in result["gates"].items():
        lines += ["", f"══ {gate}: {'PASS' if block['pass'] else 'FAIL'}   (baseline {block['baseline']})"]
        for label, verdict in block["verdicts"].items():
            lines.append(f"  {label}: {'pass' if verdict['pass'] else 'FAIL'}")
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
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--lockstep", type=Path, action="append", required=True,
                        help="a tracker_lockstep artifact (repeatable; truth-plan artifacts are judged under G1, "
                             "head-plan ones under G3)")
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
    arms = [arm for path in args.lockstep for arm in load_lockstep(path)]
    for arm in arms:
        if VARIANT_RECEDING not in arm.variants:
            raise SystemExit(f"{arm.label} ({arm.artifact}): no {VARIANT_RECEDING!r} variant, which every gate reads")
    baseline = None if args.baseline is None else load_plan_oracle(args.baseline)
    plan_path = None if args.plan_path_baseline is None else load_plan_oracle(args.plan_path_baseline)
    result = build(arms, gate_arms, baseline, plan_path)
    result["inputs"] = {
        "lockstep": [{"path": arm_file, "sha256": file_sha256(Path(arm_file))} for arm_file in sorted({arm.artifact for arm in arms})],
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
