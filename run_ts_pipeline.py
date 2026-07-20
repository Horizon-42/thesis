#!/usr/bin/env python
"""Run the ts_transformer train → predict → {evaluation, comparison CZML} pipeline.

The learned-prediction sibling of ``run_scenario_pipeline.py``: one command runs
the FULL training chain for ``4dTrajectory/ts_transformer`` — both models
(iTransformer, PatchTST) × both horizon modes (window, full), the 2×2 grid the
README's tables compare — per airport, shelling out to the existing CLIs.

Dataset generation and splitting are INSIDE the train step, by design: ``train``
builds the series from the harvest dir (``dataset.select_flight_files`` picks the
``*_arrivals.json`` pattern and excludes ``*_rejected*``), converts the datum,
derives the train/val/test split by ``flight_key`` and persists it in the
checkpoint — ``predict --split test`` then reads the split back from the
checkpoint, so the model is only ever graded on flights it never saw. There is
no separate dataset artifact to build first.

Steps per airport × model × horizon-mode ("cell"):

  1. ts train      landings/<ICAO>/  ─►  4dTrajectory/outputs/<ICAO>/ts_<model>_<mode>/
                                          {checkpoint.pt, history.json}
  2. ts predict    checkpoint + the checkpoint's test split
                                     ─►  4dTrajectory/outputs/<ICAO>/ts_pred_<model>_<mode>/
                                          {*_states.json, *_eval.json, references/,
                                           summary.json, flyability_report.json}
  3. evaluation    eval records      ─►  <pred_dir>/evaluation_report.json   (always)
  4. [eval] evaluation.visualize     ─►  <pred_dir>/evaluation_report.html
  5. [czml] build_scenario_comparison_czml
                                     ─►  aeroviz-4d/public/data/airports/<ICAO>/
                                           comparison/ts_<itr|ptst>_<mode>/
                                          (category "Predicted (<Model>, <mode>)",
                                           purple pred- entities, upserted into
                                           categories.json beside the optimizer's)

Usage:
    # one airport, all 4 cells, evaluation + frontend CZML:
    python run_ts_pipeline.py --airport KRDU
    # one cell only:
    python run_ts_pipeline.py --airport KRDU --models itransformer --modes full
    # every K-airport with harvested arrivals, evaluation only:
    python run_ts_pipeline.py --outputs eval
    # re-predict + republish from existing checkpoints (no retraining):
    python run_ts_pipeline.py --airport KRDU --skip-train
    # preview without running:
    python run_ts_pipeline.py --airport KRDU --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ── Default I/O roots (same layout as run_scenario_pipeline.py) ────────────────
LANDINGS_DIR = REPO_ROOT / "trajectory_data_process" / "outputs" / "landings"
OPT_OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"
COMPARISON_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
TS_SCRIPT = REPO_ROOT / "4dTrajectory" / "ts_transformer" / "__main__.py"
CZML_SCRIPT = REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"

# MUST mirror ts_transformer/config.py MODELS / HORIZON_MODES. The runner shells out
# (import-light: ts_transformer imports torch), so it cannot import them; kept in
# sync by this comment.
MODELS = ("itransformer", "patchtst")
HORIZON_MODES = ("window", "full")

# Frontend category naming — matches the published categories on disk
# (comparison/ts_{itr|ptst}_{window|full}) and the README's table labels.
MODEL_SHORT = {"itransformer": "itr", "patchtst": "ptst"}
MODEL_LABEL = {"itransformer": "iTransformer", "patchtst": "PatchTST"}

OUTPUT_KINDS = ("czml", "eval")


def arrivals_files(airport: str) -> list[Path]:
    airport_dir = LANDINGS_DIR / airport
    return sorted(airport_dir.glob("*_arrivals.json")) if airport_dir.exists() else []


def discover_k_airports() -> list[str]:
    """Every K-prefixed airport under the landings dir with harvested arrivals
    (the training input). US ICAO codes start with 'K'."""
    if not LANDINGS_DIR.exists():
        return []
    return sorted(
        child.name.upper()
        for child in LANDINGS_DIR.iterdir()
        if child.is_dir() and child.name.upper().startswith("K")
        and arrivals_files(child.name.upper())
    )


class Plan:
    """Resolved paths + commands for one airport × model × mode cell (pure data,
    so it can be previewed with --dry-run or asserted in a test)."""

    def __init__(self, airport: str, model: str, mode: str, outputs: tuple[str, ...],
                 *, split: str = "test", epochs: int | None = None,
                 seed: int | None = None, device: str | None = None,
                 aircraft_type: str | None = None) -> None:
        self.airport = airport.strip().upper()
        self.model = model
        self.mode = mode
        self.outputs = outputs
        self.split = split
        self.epochs = epochs
        self.seed = seed
        self.device = device
        # Fallback airframe for unresolvable types — a TSConfig field, recorded in the
        # checkpoint; passed to TRAIN only. Predict deliberately inherits the train-time
        # value from the checkpoint (overriding at predict shifts the ENU frames and
        # gate targets, which the ts CLI warns about).
        self.aircraft_type = aircraft_type

        self.data_dir = LANDINGS_DIR / self.airport
        self.train_dir = OPT_OUTPUTS_ROOT / self.airport / f"ts_{model}_{mode}"
        self.checkpoint = self.train_dir / "checkpoint.pt"
        self.pred_dir = OPT_OUTPUTS_ROOT / self.airport / f"ts_pred_{model}_{mode}"
        self.summary = self.pred_dir / "summary.json"
        self.report = self.pred_dir / "evaluation_report.json"
        self.report_html = self.pred_dir / "evaluation_report.html"

        self.category = f"ts_{MODEL_SHORT[model]}_{mode}"
        self.label = f"Predicted ({MODEL_LABEL[model]}, {mode})"
        self.comparison_dir = (
            COMPARISON_AIRPORTS_ROOT / self.airport / "comparison" / self.category
        )

    def checkpoint_exists(self) -> bool:
        return self.checkpoint.exists()

    def steps(self, *, reuse_checkpoint: bool = False) -> list[tuple[str, list[str]]]:
        """The commands to run; ``reuse_checkpoint`` drops the train step."""
        py = sys.executable
        named: list[tuple[str, list[str]]] = []

        if not reuse_checkpoint:
            train_cmd = [
                py, str(TS_SCRIPT), "train",
                "--data", str(self.data_dir),
                "--airport", self.airport,
                "--model", self.model,
                "--horizon-mode", self.mode,
                "--output-dir", str(self.train_dir),
            ]
            if self.epochs is not None:
                train_cmd += ["--epochs", str(self.epochs)]
            if self.seed is not None:
                train_cmd += ["--seed", str(self.seed)]
            if self.device is not None:
                train_cmd += ["--device", self.device]
            if self.aircraft_type is not None:
                train_cmd += ["--aircraft-type", self.aircraft_type]
            named.append(("train (dataset build + split + fit)", train_cmd))

        predict_cmd = [
            py, str(TS_SCRIPT), "predict",
            "--checkpoint", str(self.checkpoint),
            "--data", str(self.data_dir),
            "--airport", self.airport,
            "--output-dir", str(self.pred_dir),
            "--split", self.split,
        ]
        if self.device is not None:
            predict_cmd += ["--device", self.device]
        named.append((f"predict ({self.split} split)", predict_cmd))

        # The evaluation report always runs (cheap): the eval tail renders it and the
        # CZML tail consumes its verdicts + batch metrics — same wiring as
        # run_scenario_pipeline.py.
        named.append(("evaluation report", [
            py, "-m", "evaluation",
            "--input", str(self.pred_dir),
            "--output", str(self.report),
        ]))

        if "eval" in self.outputs:
            named.append(("evaluation HTML", [
                py, "-m", "evaluation.visualize",
                "--input", str(self.pred_dir),
                "--output", str(self.report_html),
            ]))

        if "czml" in self.outputs:
            named.append(("comparison CZML", [
                py, str(CZML_SCRIPT),
                "--summary", str(self.summary),
                "--output-dir", str(self.comparison_dir),
                "--airport", self.airport,
                "--category", self.category,
                "--category-label", self.label,
                "--evaluation-report", str(self.report),
            ]))

        total = len(named)
        return [(f"{i}/{total} {name}", cmd) for i, (name, cmd) in enumerate(named, 1)]


def run_cell(plan: Plan, *, dry_run: bool, skip_train: bool) -> bool:
    """Run (or preview) one cell. Returns True if it ran / would run."""
    reuse = skip_train and plan.checkpoint_exists()
    mode = "reuse checkpoint" if reuse else "train from scratch"
    print(f"\n━━ {plan.airport}  [{plan.model} · {plan.mode}]  ·  {mode}  "
          f"·  outputs: {', '.join(plan.outputs)}")
    print(f"   data      : {plan.data_dir}")
    print(f"   training  : {plan.train_dir}")
    print(f"   prediction: {plan.pred_dir}")
    if "czml" in plan.outputs:
        print(f"   comparison: {plan.comparison_dir}")

    if not arrivals_files(plan.airport):
        print(f"   ⚠ skip: no *_arrivals.json under {plan.data_dir}")
        return False
    if skip_train and not reuse:
        print("   (no existing checkpoint found → training from scratch)")

    steps = plan.steps(reuse_checkpoint=reuse)
    if dry_run:
        for label, cmd in steps:
            print(f"   [{label}] {' '.join(cmd)}")
        return True

    for label, cmd in steps:
        print(f"\n=== [{plan.airport} · {plan.model} · {plan.mode} · {label}] ===\n"
              f"{' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    print(f"✓ {plan.airport} [{plan.model} · {plan.mode}] done")
    return True


def _parse_csv(raw: str, allowed: tuple[str, ...], flag: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    unknown = [t for t in tokens if t not in allowed]
    if unknown or not tokens:
        raise argparse.ArgumentTypeError(
            f"{flag} takes a comma list from {allowed}, got {raw!r}")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--airport", default=None,
        help="airport ICAO; OMIT to run every K-prefixed airport with harvested arrivals",
    )
    parser.add_argument(
        "--models", type=lambda raw: _parse_csv(raw, MODELS, "--models"),
        default=MODELS, metavar=",".join(MODELS),
        help="which models to run (comma list; default: both)",
    )
    parser.add_argument(
        "--modes", type=lambda raw: _parse_csv(raw, HORIZON_MODES, "--modes"),
        default=HORIZON_MODES, metavar=",".join(HORIZON_MODES),
        help="which horizon modes to run (comma list; default: both)",
    )
    parser.add_argument(
        "--outputs", type=lambda raw: _parse_csv(raw, OUTPUT_KINDS, "--outputs"),
        default=OUTPUT_KINDS, metavar="czml,eval",
        help="which tails to produce from the prediction: 'czml' (frontend comparison "
             "CZML), 'eval' (evaluation report HTML; the JSON report always runs); "
             "default: both",
    )
    parser.add_argument(
        "--split", choices=("test", "val", "train", "all"), default="test",
        help="which of the checkpoint's flight splits to predict (default: test — "
             "the only split the model never saw; the split lives in the checkpoint)",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="training epochs (default: the ts config's own)")
    parser.add_argument("--seed", type=int, default=None,
                        help="training seed (default: the ts config's own; the seed also "
                             "drives the train/val/test split)")
    parser.add_argument("--device", default=None,
                        help='"auto", "cpu", "cuda" (default: the ts CLI\'s auto)')
    parser.add_argument(
        "--aircraft-type", default=None,
        help="fallback airframe for unresolvable types, passed to TRAIN only (predict "
             "inherits the checkpoint's train-time value — overriding at predict shifts "
             "the ENU frames and gate targets)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="reuse an existing checkpoint.pt per cell (skip the train step); cells "
             "without one still train from scratch",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the resolved paths + the commands without running them",
    )
    args = parser.parse_args()

    if args.airport:
        airports = [args.airport.strip().upper()]
    else:
        airports = discover_k_airports()
        if not airports:
            parser.error(f"no K-prefixed airports with *_arrivals.json under {LANDINGS_DIR}")
        print(f"no --airport given → running {len(airports)} K-airport(s): "
              f"{', '.join(airports)}")

    cells = [(airport, model, mode)
             for airport in airports for model in args.models for mode in args.modes]
    print(f"{len(cells)} cell(s): {len(airports)} airport(s) × "
          f"{len(args.models)} model(s) × {len(args.modes)} mode(s)")

    ran = 0
    for airport, model, mode in cells:
        plan = Plan(airport, model, mode, tuple(args.outputs), split=args.split,
                    epochs=args.epochs, seed=args.seed, device=args.device,
                    aircraft_type=args.aircraft_type)
        if run_cell(plan, dry_run=args.dry_run, skip_train=args.skip_train):
            ran += 1

    verb = "previewed" if args.dry_run else "completed"
    print(f"\n✓ {verb} {ran}/{len(cells)} cell(s)  "
          f"[models={','.join(args.models)}, modes={','.join(args.modes)}, "
          f"outputs={','.join(args.outputs)}, split={args.split}, "
          f"skip-train={args.skip_train}]")


if __name__ == "__main__":
    main()
