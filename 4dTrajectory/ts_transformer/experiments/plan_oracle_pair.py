"""Plan-and-guidance: two plan-oracle artifacts paired flight by flight — an arm against its base, per stratum.

Every `plan_oracle` run writes one row per flight (`plan_oracle.json`); a change to the
lockstep (the order hold, v5.3) or to the head is read as the ARM's rows against the
BASE's on the same flights, never as two summary tables side by side — the cohort is the
same, so the per-flight difference is the measurement and the summary means are the
context. The identity line (`rows differing`, the largest |ΔADE|) is how a refactor that
must not move a trajectory is checked: `--hold-asks 1` against §12.6's artifact.

    python run_ts.py plan_oracle_pair --base s75=<dir> --arm hold2=<dir> --out <dir>

`<dir>` holds `plan_oracle.json` (or names the file). The two artifacts must cover the
same flights; a mismatch is refused with the counts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ts_transformer.data.approach_difficulty import STRATUM_SHORT, strata_masks
from ts_transformer.experiments.plan_oracle import STRATA, held_share
from ts_transformer.io_utils import utc_now, write_json_atomic

SCHEMA = "ts-plan-oracle-pair-v1"
#: A flight whose ADE moved by more than this between the two artifacts is a DIFFERING
#: row: the identity check's ruler (float64 rollouts reproduce to far below it).
IDENTITY_TOLERANCE_M = 1e-3


def _p(values, quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile)) if len(values) else float("nan")


def _nan_mean(values) -> float:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else float("nan")


def load_rows(spec: str) -> tuple[str, dict, dict[str, dict]]:
    """``LABEL=PATH`` → the label, the artifact's header and its rows by flight."""
    label, _, path = spec.partition("=")
    if not label or not path:
        raise SystemExit(f"expected LABEL=PATH, got {spec!r}")
    file = Path(path)
    if file.is_dir():
        file = file / "plan_oracle.json"
    artifact = json.loads(file.read_text(encoding="utf-8"))
    rows = {row["dataset_id"]: row for row in artifact.pop("rows")}
    artifact["path"] = str(file)
    return label, artifact, rows


#: Per-flight values a pair is read on: (key, label, digits, the reader).
FIELDS = (
    ("ade_m", "ADE", 0, lambda r: r["prediction"]["ade_m"]),
    ("fde_m", "FDE", 0, lambda r: r["prediction"]["fde_m"]),
    ("chamfer_m", "chamfer", 0, lambda r: r["geometry"]["chamfer_m"]),
    ("frechet_m", "Fréchet", 0, lambda r: r["geometry"]["frechet_m"]),
    ("abs_dt_s", "|dt| s", 1, lambda r: abs(r["prediction"]["final_time_error_s"])),
    ("established", "established", 3, lambda r: float(r["reference"]["established"])),
    ("fully_flyable", "fully flyable", 3, lambda r: float(r["reference"]["fully_flyable"])),
    ("lateral_violation", "lateral viol.", 3, lambda r: float(r["reference"]["lateral_violation"])),
    ("glidepath_violation", "glidepath viol.", 3, lambda r: float(r["reference"]["glidepath_violation"])),
    ("capped", "rolled flight capped", 3, lambda r: float(bool(r["route"].get("capped_by")))),
    ("turns_incomplete", "turn not completed", 3, lambda r: float(r["route"].get("turns_incomplete", 0) > 0)),
    ("legs", "legs flown", 2, lambda r: float(r["route"].get("legs", 0))),
    ("bank_capped", "bank capped", 3, lambda r: r["hook"]["planBankCappedSteps"]),
    # the order hold's own counts (v5.3): absent on an artifact written before it
    ("orders_held", "orders held (of steps)", 3,
     lambda r: None if "planHeldSteps" not in r["hook"] else held_share(r["hook"])),
    ("order_changes", "order changes / flight", 2,
     lambda r: None if "planOrderChanges" not in r["hook"] else r["hook"]["planOrderChanges"]),
)
#: The fields a paired difference is read on: the median Δ and the share of flights the
#: arm is LOWER on (every one of these is a cost).
PAIRED = ("ade_m", "fde_m", "chamfer_m", "frechet_m", "abs_dt_s")


def pair(base: dict[str, dict], arm: dict[str, dict]) -> dict:
    ids = sorted(base)
    if set(arm) != set(base):
        raise SystemExit(
            f"the two artifacts cover different flights: base {len(base)}, arm {len(arm)}, "
            f"common {len(set(base) & set(arm))} — a pair needs the same cohort"
        )
    covariates = {i: base[i]["difficulty"] for i in ids}
    masks = strata_masks(covariates, ids)
    ade_delta = np.array([arm[i]["prediction"]["ade_m"] - base[i]["prediction"]["ade_m"] for i in ids])
    out = {
        "flights": len(ids),
        "identity": {
            "rows_differing": int(np.sum(np.abs(ade_delta) > IDENTITY_TOLERANCE_M)),
            "max_abs_ade_delta_m": float(np.max(np.abs(ade_delta))) if len(ids) else 0.0,
            "established_differing": int(sum(
                base[i]["reference"]["established"] != arm[i]["reference"]["established"] for i in ids
            )),
        },
        "strata": {},
    }
    for stratum in STRATA:
        members = [i for i, keep in zip(ids, masks[stratum], strict=True) if keep]
        block: dict = {"flights": len(members)}
        for key, _label, _digits, read in FIELDS:
            b = [read(base[i]) for i in members]
            a = [read(arm[i]) for i in members]
            block[key] = {"base": _nan_mean(b), "arm": _nan_mean(a)}
            if key in PAIRED and members:
                delta = np.array([x - y for x, y in zip(a, b, strict=True)], dtype=np.float64)
                block[key]["delta_p50"] = _p(delta, 50)
                block[key]["arm_lower_share"] = float(np.mean(delta < 0.0))
        out["strata"][stratum] = block
    return out


def format_table(result: dict, base_label: str, arm_label: str) -> str:
    strata = [s for s in STRATA if result["strata"][s]["flights"]]
    lines = [
        f"paired per flight, {arm_label} against {base_label}, n = "
        + ", ".join(f"{STRATUM_SHORT[s]} {result['strata'][s]['flights']}" for s in strata),
        f"identity: {result['identity']['rows_differing']} of {result['flights']} rows differ in ADE by over "
        f"{IDENTITY_TOLERANCE_M:g} m (max |ΔADE| {result['identity']['max_abs_ade_delta_m']:.3f} m), "
        f"{result['identity']['established_differing']} differ in `established`",
        "",
        f"{'metric':<26}" + "".join(f"{STRATUM_SHORT[s]:>30}" for s in strata),
        f"{'':<26}" + "".join(f"{'base':>10}{'arm':>10}{'Δp50 (↓%)':>10}" for _ in strata),
    ]
    for key, label, digits, _read in FIELDS:
        cells = []
        for s in strata:
            cell = result["strata"][s][key]
            b, a = cell["base"], cell["arm"]
            text = f"{'n/a' if math.isnan(b) else f'{b:.{digits}f}':>10}{'n/a' if math.isnan(a) else f'{a:.{digits}f}':>10}"
            if "delta_p50" in cell:
                text += f"{cell['delta_p50']:>+6.{digits}f} {cell['arm_lower_share'] * 100.0:>3.0f}"
            else:
                text += f"{'':>10}"
            cells.append(text)
        lines.append(f"{label:<26}" + "".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--base", required=True, metavar="LABEL=PATH", help="the base artifact (a dir or plan_oracle.json)")
    parser.add_argument("--arm", required=True, metavar="LABEL=PATH", help="the arm artifact")
    parser.add_argument("--out", type=Path, default=None, help="where to write pair.json / pair.txt (optional)")
    args = parser.parse_args(argv)
    base_label, base_header, base = load_rows(args.base)
    arm_label, arm_header, arm = load_rows(args.arm)
    result = pair(base, arm)
    text = format_table(result, base_label, arm_label)
    print(text, flush=True)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.out / "pair.json", {
            "schema_version": SCHEMA, "generated_at": utc_now(),
            "base": {"label": base_label, **base_header}, "arm": {"label": arm_label, **arm_header},
            **result,
        })
        (args.out / "pair.txt").write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out / 'pair.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
