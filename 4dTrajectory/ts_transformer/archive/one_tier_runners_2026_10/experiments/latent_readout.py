#!/usr/bin/env python
"""Multimodal readout for a latent control arm (latent-intent design §六 L2, §八).

`predict --latent-samples K [--latent-shuffle]` writes the top-1 records in the arm's
prediction directory, K further decodes under `modes/modeNN/` and the shuffled-latent
diagnostic under `shuffled/`, each a full prediction directory. This runner joins them by
flight and reports, per difficulty stratum:

  top-1 ADE / FDE          the deliverable, comparable with any other arm's readout
  minADE_K / minFDE_K      the best of {top-1, modes} per flight — how much of the error
                           is genuine multimodality the top-1 has to average over
  miss rate                share of flights whose best mode still ends > MISS_FDE_M away
  spread                   per-flight std of the modes' FDE, averaged — are the modes
                           actually different trajectories?
  shuffled ΔADE            ADE when every flight is decoded from ANOTHER flight's latent,
                           minus the top-1 ADE — the posterior-collapse reading: a decoder
                           that ignores z barely moves (≈ 0), one that reads it degrades

Pre-registered (design doc §六 L2 gates 1–2): shuffled ΔADE > 200 m pooled, and minADE_K
clearly below top-1 AND below a same-K random-latent control (a separate arm; this runner
reads whatever directories it is given, so pass that arm as `--control`).

`--history <run>/history.json` adds the KEPT epoch's own latent block (L2.f): the
per-dimension component KL, that KL's mean/variance split, the posterior mean's displacement
from the prior mean in prior sigmas, and both active-unit counts. The displacement is what
the L2.e' arms died on (0.05–0.2 σ against a gate of one — z was the prior's mean wearing
noise) and it is readable at epoch 1, so `--history` works ALONE, before the arm has
predicted anything. A run trained before those diagnostics existed prints the keys it has.

    python run_ts.py latent_readout --history <run>/history.json          # training side
    python run_ts.py latent_readout --arm <pred_dir> [--control <pred_dir>]
                                    [--history <run>/history.json] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ts_transformer.experiments.support import REPO_ROOT

from ts_transformer.data.approach_difficulty import strata_masks  # noqa: E402
# One ruler for the displacement gate and one for an active unit, both defined beside the
# KL they are read against (`outputs/control/latent.py`) rather than restated here.
from ts_transformer.outputs.control.latent import ACTIVE_UNIT_KL_NATS, displacement_verdict  # noqa: E402
from flight_scenarios.identity import summary_row_key  # noqa: E402

MISS_FDE_M = 2_000.0


def scored_rows(pred_dir: Path) -> dict[str, dict]:
    summary = json.loads((pred_dir / "summary.json").read_text())
    return {
        summary_row_key(row): row
        for row in summary["results"]
        if row.get("ade_m") is not None and row.get("fde_m") is not None
    }


def mode_dirs(pred_dir: Path, kind: str = "modes") -> list[Path]:
    """The per-sample prediction directories `predict` wrote under the arm: `modes/` (prior
    samples) or `random/` (the N(0, I) control)."""
    root = pred_dir / kind
    return sorted(root.glob("mode*")) if root.is_dir() else []


def _stack(rows_by_arm: list[dict[str, dict]], keys: list[str], field: str) -> np.ndarray:
    """``[arms, flights]`` of one metric, NaN where an arm lacks the flight."""
    out = np.full((len(rows_by_arm), len(keys)), np.nan)
    for a, rows in enumerate(rows_by_arm):
        for f, key in enumerate(keys):
            row = rows.get(key)
            if row is not None:
                out[a, f] = float(row[field])
    return out


def kept_epoch_latent(history_path: Path) -> dict:
    """The `latent` block of the epoch the checkpoint was taken from.

    The artifact says which epoch that was (`fit_diagnostics.training_objective`), so the
    selection rule is read, never restated here.
    """
    payload = json.loads(history_path.read_text())
    objective = payload["fit_diagnostics"]["training_objective"]
    epoch = int(objective["best_epoch"])
    row = next(item for item in payload["history"] if item["epoch"] == epoch)
    if not row.get("latent"):
        raise SystemExit(f"{history_path} epoch {epoch} has no latent block — not a latent run")
    return {"history": str(history_path), "epoch": epoch,
            "selection_metric": objective["checkpoint_selection_metric"],
            "selection_value": objective["checkpoint_selection_value"],
            **row["latent"]}


def render_latent(block: dict) -> str:
    """One line per diagnostic, in the order a collapse is diagnosed in."""
    order = (
        ("mean_displacement_sigma", "|q mean − p mean| / p sigma (median)", "{:.3f}"),
        ("component_kl_mean_term_nats", "component KL mean term (nats/flight)", "{:.4f}"),
        ("component_kl_variance_term_nats", "component KL variance term (nats/flight)", "{:.4f}"),
        ("component_kl_nats_per_flight", "component KL (nats/flight)", "{:.4f}"),
        ("kl_nats_per_flight", "charged KL (nats/flight)", "{:.4f}"),
        ("beta_effective", "beta effective this epoch", "{:g}"),
        ("active_units", "active units (budget ruler)", "{:.2f}"),
        ("active_units_0p05", f"active units (> {ACTIVE_UNIT_KL_NATS:g} nats)", "{:.2f}"),
    )
    lines = [
        "",
        f"kept epoch {block['epoch']} ({block['selection_metric']}="
        f"{block['selection_value']:.1f}) — {block['history']}",
    ]
    lines += [
        f"  {label:<38s} {form.format(block[key])}"
        for key, label, form in order if key in block
    ]
    if "component_kl_per_dim" in block:
        lines.append("  per-dimension component KL (nats/flight) "
                     + " ".join(f"{value:.3f}" for value in block["component_kl_per_dim"]))
    # The verdict is printed only when the number it judges is there: a run trained before
    # these diagnostics existed has the three old keys and no displacement.
    if "mean_displacement_sigma" in block:
        lines.append("  " + displacement_verdict(block["mean_displacement_sigma"]))
    return "\n".join(lines)


def readout(arm: Path, control: Path | None) -> dict:
    top1 = scored_rows(arm)
    modes = [scored_rows(path) for path in mode_dirs(arm)]
    shuffled = scored_rows(arm / "shuffled") if (arm / "shuffled").is_dir() else None
    keys = sorted(top1)
    if not keys:
        raise SystemExit(f"{arm} has no scored rows")
    masks = strata_masks(top1, keys)
    candidates = [top1, *modes]
    ade = _stack(candidates, keys, "ade_m")            # [1+K, F]
    fde = _stack(candidates, keys, "fde_m")
    shuffled_ade = _stack([shuffled], keys, "ade_m")[0] if shuffled else None
    # The same-K control: the arm's own `random/` modes (latents from N(0, I), decoded by
    # the same checkpoint) — or an external arm's top-1 + modes when one is given.
    control_min_ade = control_min_fde = None
    random_modes = [scored_rows(path) for path in mode_dirs(arm, "random")]
    if control is not None:
        control_rows = [scored_rows(control), *(scored_rows(path) for path in mode_dirs(control))]
    elif random_modes:
        control_rows = [top1, *random_modes]
    else:
        control_rows = []
    if control_rows:
        control_min_ade = np.nanmin(_stack(control_rows, keys, "ade_m"), axis=0)
        control_min_fde = np.nanmin(_stack(control_rows, keys, "fde_m"), axis=0)

    # Coverage is stated, never silent: a mode directory that scored fewer flights than
    # the top-1 would otherwise shrink every nan-reduction's denominator unannounced.
    coverage = {
        "top1": int(np.isfinite(ade[0]).sum()),
        "modes": [int(np.isfinite(ade[1 + k]).sum()) for k in range(len(modes))],
        "shuffled": None if shuffled_ade is None else int(np.isfinite(shuffled_ade).sum()),
        "control": None if not control_rows else int(
            np.isfinite(_stack(control_rows, keys, "ade_m")).all(axis=0).sum()
        ),
    }
    per_stratum = {}
    for stratum, mask in masks.items():
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        block = {
            "n": int(len(idx)),
            "modes": len(modes),
            "min_ade_full_coverage_n": int(np.isfinite(ade[:, idx]).all(axis=0).sum()),
            "shuffled_n": None if shuffled_ade is None else int(np.isfinite(shuffled_ade[idx]).sum()),
            "top1_ade_mean_m": float(np.nanmean(ade[0, idx])),
            "top1_fde_p50_m": float(np.nanmedian(fde[0, idx])),
            "min_ade_mean_m": float(np.nanmean(np.nanmin(ade[:, idx], axis=0))),
            "min_fde_p50_m": float(np.nanmedian(np.nanmin(fde[:, idx], axis=0))),
            "miss_rate": float(np.nanmean(np.nanmin(fde[:, idx], axis=0) > MISS_FDE_M)),
            "mode_fde_spread_m": (
                float(np.nanmean(np.nanstd(fde[1:, idx], axis=0))) if modes else None
            ),
            "shuffled_delta_ade_m": (
                None if shuffled_ade is None
                else float(np.nanmean(shuffled_ade[idx] - ade[0, idx]))
            ),
        }
        if control_min_ade is not None:
            block["control_min_ade_mean_m"] = float(np.nanmean(control_min_ade[idx]))
            block["control_min_fde_p50_m"] = float(np.nanmedian(control_min_fde[idx]))
        per_stratum[stratum] = block
    return {"arm": str(arm),
            "control": str(control) if control is not None else (f"{arm}/random" if random_modes else None),
            # What the control decodes: N(0, I) latents under the arm's own checkpoint (its
            # prior may be far from N(0, I) once trained, especially a mixture) or an
            # external arm's own prior samples — say which, so gate 2 is read correctly.
            "control_kind": (
                "external arm (its prior)" if control is not None
                else ("N(0, I) latents, same checkpoint" if random_modes else None)
            ),
            "control_modes": (len(control_rows) - 1) if control_rows else 0,
            "flights": len(keys), "modes": len(modes), "random_modes": len(random_modes),
            "shuffled": shuffled is not None, "coverage": coverage,
            "miss_fde_m": MISS_FDE_M, "strata": per_stratum}


def render(result: dict) -> str:
    coverage = result["coverage"]
    lines = [
        f"latent readout — {result['arm']} ({result['flights']} flights, {result['modes']} modes"
        f"{', shuffled' if result['shuffled'] else ''})"
        + (f" vs control {result['control']} [{result['control_kind']}, {result['control_modes']} modes]"
           if result["control"] else ""),
        f"coverage: top-1 {coverage['top1']}, modes {coverage['modes']}, "
        f"shuffled {coverage['shuffled']}, control {coverage['control']} of {result['flights']} flights"
        + (" — INCOMPLETE" if any(
            c is not None and c < result["flights"]
            for c in (coverage["top1"], coverage["shuffled"], coverage["control"], *coverage["modes"])
        ) else ""),
        "",
        f"{'stratum':>46s} {'n':>5s} {'top1 ADE':>9s} {'minADE_K':>9s} {'top1 FDE':>9s} "
        f"{'minFDE_K':>9s} {'miss':>6s} {'spread':>7s} {'shufΔADE':>9s} {'ctrl minADE':>11s}",
    ]
    for stratum, b in result["strata"].items():
        ctrl = b.get("control_min_ade_mean_m")
        spread = b["mode_fde_spread_m"]
        shuffled = b["shuffled_delta_ade_m"]
        lines.append(
            f"{stratum:>46s} {b['n']:>5d} {b['top1_ade_mean_m']:>9.1f} {b['min_ade_mean_m']:>9.1f} "
            f"{b['top1_fde_p50_m']:>9.1f} {b['min_fde_p50_m']:>9.1f} {b['miss_rate']:>6.2f} "
            f"{'n/a' if spread is None else f'{spread:.0f}':>7s} "
            f"{'n/a' if shuffled is None else f'{shuffled:+.0f}':>9s} "
            f"{'n/a' if ctrl is None else f'{ctrl:.1f}':>11s}"
        )
    lines.append("")
    lines.append("minADE_K = best of {top-1, modes} per flight; miss = share whose best mode's FDE > "
                 f"{MISS_FDE_M:g} m; spread = per-flight std of the modes' FDE; shufΔADE = ADE decoded "
                 "from another flight's latent minus top-1 (≈0 means the decoder ignores z).")
    if "latent_diagnostics" in result:
        lines.append(render_latent(result["latent_diagnostics"]))
    return "\n".join(lines) + "\n"


def _resolve(path: Path | None) -> Path | None:
    return None if path is None else (path if path.is_absolute() else REPO_ROOT / path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--arm", type=Path, default=None,
                        help="a latent arm's top-1 prediction directory; optional when "
                             "--history is given, which is how the training-side block is "
                             "read at epoch 1, before any prediction exists")
    parser.add_argument("--control", type=Path, default=None,
                        help="an external control arm (top-1 + modes/); default = the arm's own "
                             "random/ modes from `predict --latent-random K`")
    parser.add_argument("--history", type=Path, default=None,
                        help="the arm's training history.json; prints the kept epoch's "
                             "latent diagnostics (per-dimension component KL, its "
                             "mean/variance split, the posterior-mean displacement, both "
                             "active-unit counts)")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.arm is None and args.history is None:
        parser.error("give --arm (the prediction readout), --history (the training-side "
                     "latent block), or both")
    arm, control, history = (_resolve(args.arm), _resolve(args.control), _resolve(args.history))
    if arm is None and control is not None:
        parser.error("--control is a comparison for --arm's modes; it means nothing alone")
    result = readout(arm, control) if arm is not None else {}
    if history is not None:
        result["latent_diagnostics"] = kept_epoch_latent(history)
    text = render(result) if arm is not None else render_latent(result["latent_diagnostics"]) + "\n"
    print(text, end="")
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
