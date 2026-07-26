#!/usr/bin/env python
"""Run cross-validation -> train -> per-airport prediction/evaluation/CZML.

Two training scopes are explicit:

``per-airport``
    One CV search and checkpoint per airport (the historical organization).

``pooled``
    One CV search and checkpoint over all selected airport manifests, followed by separate
    predictions and publications for each airport's locked split.

The TS command locks outer train/validation/test before cross-validation. CV sees outer-train
only; final training uses outer-validation for early stopping; outer-test is consumed only by
the prediction stage after the final checkpoint exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
OPT_OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"
COMPARISON_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
TS_SCRIPT = REPO_ROOT / "4dTrajectory" / "ts_transformer" / "__main__.py"
CZML_SCRIPT = REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

from config import config_for_mode  # noqa: E402

MODELS = ("itransformer", "patchtst")
HORIZON_MODES = ("window", "full")
TRAINING_MODES = ("per-airport", "pooled")
COORDINATE_FRAMES = ("enu", "runway-aligned")
MODEL_SHORT = {"itransformer": "itr", "patchtst": "ptst"}
MODEL_LABEL = {"itransformer": "iTransformer", "patchtst": "PatchTST"}
OUTPUT_KINDS = ("czml", "eval")

CHECKPOINT_METADATA_NAME = "checkpoint_metadata.json"
CHECKPOINT_METADATA_SCHEMA = "ts-checkpoint-metadata-v2-multi-airport"
CV_RESULTS_NAME = "cv_results.json"
BEST_CONFIG_NAME = "best_config.json"
DEFAULT_POOLED_SAMPLES_PER_EPOCH = 250_000
DEFAULT_CV_SAMPLES_PER_EPOCH = 100_000


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrival_manifest_path(airport: str) -> Path:
    return HARVEST_ROOT / airport.upper() / "arrivals" / "manifest.json"


def discover_k_airports() -> list[str]:
    if not HARVEST_ROOT.exists():
        return []
    return sorted(
        child.name.upper()
        for child in HARVEST_ROOT.iterdir()
        if child.is_dir()
        and child.name.upper().startswith("K")
        and arrival_manifest_path(child.name).exists()
    )


def _manifest_digests(airports: tuple[str, ...]) -> dict[str, str]:
    return {airport: _file_sha256(arrival_manifest_path(airport)) for airport in airports}


def _frame_tag(coordinate_frame: str) -> str:
    return "" if coordinate_frame == "enu" else "_runway_aligned"


class TrainingPlan:
    """One CV/final-training cell, shared by one or more airport predictions."""

    def __init__(
        self,
        airports: tuple[str, ...],
        model: str,
        mode: str,
        *,
        training_mode: str,
        epochs: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        aircraft_type: str | None = None,
        coordinate_frame: str = "enu",
        batch_size: str = "auto",
        samples_per_epoch: int | None = None,
        cv_folds: int = 3,
        cv_trials: int = 4,
        cv_epochs: int = 12,
        cv_patience: int = 4,
        cv_samples_per_epoch: int = DEFAULT_CV_SAMPLES_PER_EPOCH,
        output_dir: str | Path | None = None,
    ) -> None:
        self.airports = tuple(sorted(airport.strip().upper() for airport in airports))
        if not self.airports:
            raise ValueError("TrainingPlan requires at least one airport")
        self.model = model
        self.mode = mode
        self.training_mode = training_mode
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.aircraft_type = aircraft_type
        self.coordinate_frame = coordinate_frame
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.cv_folds = cv_folds
        self.cv_trials = cv_trials
        self.cv_epochs = cv_epochs
        self.cv_patience = cv_patience
        self.cv_samples_per_epoch = cv_samples_per_epoch

        self.data_manifests = tuple(arrival_manifest_path(airport) for airport in self.airports)
        scope = self.airports[0] if training_mode == "per-airport" else "POOLED"
        suffix = _frame_tag(coordinate_frame)
        self.train_dir = (
            Path(output_dir)
            if output_dir is not None
            else OPT_OUTPUTS_ROOT / scope / f"ts_{model}_{mode}{suffix}"
        )
        self.cv_dir = self.train_dir / "cross_validation"
        self.cv_results = self.cv_dir / CV_RESULTS_NAME
        self.best_config = self.cv_dir / BEST_CONFIG_NAME
        self.checkpoint = self.train_dir / "checkpoint.pt"
        self.checkpoint_metadata = self.train_dir / CHECKPOINT_METADATA_NAME

    @property
    def pooled(self) -> bool:
        return self.training_mode == "pooled"

    @property
    def label(self) -> str:
        return "POOLED[" + ",".join(self.airports) + "]" if self.pooled else self.airports[0]

    def _data_args(self) -> list[str]:
        return [token for manifest in self.data_manifests for token in ("--data", str(manifest))]

    def _recipe_args(self, *, cv: bool) -> list[str]:
        args = [
            "--model", self.model,
            "--horizon-mode", self.mode,
            "--coordinate-frame", self.coordinate_frame,
            "--batch-size", self.batch_size,
            "--eval-anchor-policy", "first",
        ]
        if self.seed is not None:
            args += ["--seed", str(self.seed)]
        if self.device is not None:
            args += ["--device", self.device]
        if self.aircraft_type is not None:
            args += ["--aircraft-type", self.aircraft_type]
        if self.pooled:
            args += [
                "--sampling-strategy", "airport-flight-balanced",
                "--samples-per-epoch",
                str(self.cv_samples_per_epoch if cv else (
                    self.samples_per_epoch or DEFAULT_POOLED_SAMPLES_PER_EPOCH
                )),
            ]
        elif cv and self.cv_samples_per_epoch:
            args += ["--samples-per-epoch", str(self.cv_samples_per_epoch)]
        return args

    def checkpoint_reuse_error(self) -> str | None:
        if not self.checkpoint.is_file():
            return f"missing checkpoint {self.checkpoint}"
        if not self.checkpoint_metadata.is_file():
            return f"missing checkpoint metadata {self.checkpoint_metadata}"
        if any(not manifest.is_file() for manifest in self.data_manifests):
            return "one or more arrival manifests are missing"
        try:
            metadata = json.loads(self.checkpoint_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"unreadable checkpoint metadata: {exc}"
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA
        ):
            return "checkpoint metadata has the wrong schema"
        if metadata.get("checkpoint_sha256") != _file_sha256(self.checkpoint):
            return "checkpoint failed SHA-256 validation"
        if metadata.get("arrival_manifests") != _manifest_digests(self.airports):
            return "checkpoint was trained against different arrival manifests"
        return None

    def cv_reuse_error(self) -> str | None:
        if not self.cv_results.is_file() or not self.best_config.is_file():
            return "missing cross-validation results or best_config.json"
        try:
            results = json.loads(self.cv_results.read_text(encoding="utf-8"))
            best = json.loads(self.best_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"unreadable cross-validation artifact: {exc}"
        if (
            not isinstance(results, dict)
            or results.get("schema_version") != "ts-cross-validation-v1"
            or not isinstance(best, dict)
        ):
            return "cross-validation artifact has the wrong schema"
        if results.get("best_overrides") != best:
            return "cross-validation best_config.json disagrees with cv_results.json"
        if results.get("arrival_manifests") != _manifest_digests(self.airports):
            return "cross-validation used different arrival manifests"
        base_config = results.get("base_config")
        expected_config = self._expected_cv_base_config()
        if base_config != expected_config:
            if isinstance(base_config, dict):
                differing = [
                    key for key in expected_config
                    if base_config.get(key) != expected_config[key]
                ]
                detail = differing[0] if differing else "fields"
            else:
                detail = "object"
            return f"cross-validation base_config {detail} does not match current recipe"
        expected_controls = {
            "n_splits": self.cv_folds,
            "max_trials": self.cv_trials,
            "cv_epochs": self.cv_epochs,
            "cv_patience": self.cv_patience,
            "auto_batch_size": self.batch_size == "auto",
        }
        for field, expected in expected_controls.items():
            if results.get(field) != expected:
                return (
                    f"cross-validation {field}={results.get(field)!r} does not match "
                    f"current {expected!r}"
                )
        return None

    def _expected_cv_base_config(self) -> dict[str, object]:
        """Rebuild the exact base TSConfig produced by this plan's CV command."""
        overrides: dict[str, object] = {
            "model": self.model,
            "coordinate_frame": self.coordinate_frame,
            "eval_anchor_policy": "first",
        }
        if self.seed is not None:
            overrides["seed"] = self.seed
        if self.device is not None:
            overrides["device"] = self.device
        if self.aircraft_type is not None:
            overrides["aircraft_type"] = self.aircraft_type
        if self.batch_size != "auto":
            overrides["batch_size"] = int(self.batch_size)
        if self.pooled:
            overrides.update({
                "sampling_strategy": "airport-flight-balanced",
                "train_samples_per_epoch": self.cv_samples_per_epoch,
            })
        elif self.cv_samples_per_epoch:
            overrides["train_samples_per_epoch"] = self.cv_samples_per_epoch
        return config_for_mode(self.mode, **overrides).to_dict()

    def cv_step(self) -> tuple[str, list[str]]:
        """The isolated outer-train CV command for this training cell."""
        py = sys.executable
        return "cross validation (outer-train only)", [
            py, str(TS_SCRIPT), "cross-validate",
            *self._data_args(),
            "--output-dir", str(self.cv_dir),
            *self._recipe_args(cv=True),
            "--folds", str(self.cv_folds),
            "--trials", str(self.cv_trials),
            "--cv-epochs", str(self.cv_epochs),
            "--cv-patience", str(self.cv_patience),
        ]

    def train_step(self, *, use_best_config: bool) -> tuple[str, list[str]]:
        """The final-fit command, optionally consuming this cell's locked CV winner."""
        py = sys.executable
        train_command = [
            py, str(TS_SCRIPT), "train",
            *self._data_args(),
            "--output-dir", str(self.train_dir),
            *self._recipe_args(cv=False),
        ]
        if self.epochs is not None:
            train_command += ["--epochs", str(self.epochs)]
        if use_best_config:
            train_command += ["--config-overrides", str(self.best_config)]
        return "final train (outer-val early stopping)", train_command

    def steps(self, *, skip_cv: bool, reuse_checkpoint: bool) -> list[tuple[str, list[str]]]:
        if reuse_checkpoint:
            return []
        named: list[tuple[str, list[str]]] = []
        cv_available = self.cv_reuse_error() is None
        if not skip_cv:
            named.append(self.cv_step())
        # A CV stage in this run will create it; --skip-cv reuses it only when its manifest
        # provenance still matches. Otherwise final training deliberately uses base defaults.
        named.append(self.train_step(use_best_config=not skip_cv or cv_available))
        return named


class PredictionPlan:
    """Per-airport publication tail for a completed TrainingPlan."""

    def __init__(
        self,
        training: TrainingPlan,
        airport: str,
        outputs: tuple[str, ...],
        *,
        split: str = "test",
        experiment_tag: str | None = None,
    ) -> None:
        self.training = training
        self.airport = airport.upper()
        self.outputs = outputs
        self.split = split
        self.data_manifest = arrival_manifest_path(self.airport)
        scope = "pooled_" if training.pooled else ""
        frame = _frame_tag(training.coordinate_frame)
        tag = f"_{experiment_tag}" if experiment_tag else ""
        stem = f"{scope}{training.model}_{training.mode}{frame}{tag}_{split}"
        self.pred_dir = OPT_OUTPUTS_ROOT / self.airport / f"ts_pred_{stem}"
        self.summary = self.pred_dir / "summary.json"
        self.report = self.pred_dir / "evaluation_report.json"
        self.report_html = self.pred_dir / "evaluation_report.html"
        category_scope = "pooled_" if training.pooled else ""
        self.category = (
            f"ts_{category_scope}{MODEL_SHORT[training.model]}_{training.mode}"
            f"{frame}{tag}_{split}"
        )
        model_label = MODEL_LABEL[training.model]
        pooled_label = "pooled, " if training.pooled else ""
        frame_label = "ENU" if training.coordinate_frame == "enu" else "runway-aligned"
        self.label = (
            f"Predicted ({model_label}, {pooled_label}{training.mode}, "
            f"{frame_label}, {split} split)"
        )
        self.comparison_dir = (
            COMPARISON_AIRPORTS_ROOT / self.airport / "comparison" / self.category
        )

    def steps(self) -> list[tuple[str, list[str]]]:
        py = sys.executable
        named: list[tuple[str, list[str]]] = []
        predict = [
            py, str(TS_SCRIPT), "predict",
            "--checkpoint", str(self.training.checkpoint),
            "--data", str(self.data_manifest),
            "--output-dir", str(self.pred_dir),
            "--split", self.split,
        ]
        if self.training.device is not None:
            predict += ["--device", self.training.device]
        named.append((f"predict ({self.split} split)", predict))
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
        return named


def _run_steps(context: str, steps: list[tuple[str, list[str]]], *, dry_run: bool) -> None:
    total = len(steps)
    for index, (label, command) in enumerate(steps, 1):
        qualified = f"{index}/{total} {label}"
        if dry_run:
            print(f"   [{qualified}] {' '.join(command)}")
            continue
        print(f"\n=== [{context} · {qualified}] ===\n{' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_training(
    plan: TrainingPlan, *, dry_run: bool, skip_cv: bool, skip_train: bool
) -> bool:
    missing = [manifest for manifest in plan.data_manifests if not manifest.exists()]
    if missing:
        print(f"   ⚠ skip {plan.label}: missing {missing[0]}")
        return False
    reuse_error = plan.checkpoint_reuse_error() if skip_train else None
    reuse = skip_train and reuse_error is None
    mode = "reuse checkpoint" if reuse else "train final checkpoint"
    print(f"\n━━ {plan.label} [{plan.model} · {plan.mode} · {plan.coordinate_frame}] · {mode}")
    print(f"   manifests : {len(plan.data_manifests)}")
    print(f"   CV        : {plan.cv_dir}")
    print(f"   training  : {plan.train_dir}")
    if skip_train and not reuse:
        print(f"   (checkpoint not reusable: {reuse_error} → rebuilding)")
    if skip_cv and not reuse and plan.cv_reuse_error() is not None:
        print("   (CV skipped and no reusable CV artifact → base hyperparameters)")
    _run_steps(
        f"{plan.label} · {plan.model} · {plan.mode}",
        plan.steps(skip_cv=skip_cv, reuse_checkpoint=reuse),
        dry_run=dry_run,
    )
    return True


def run_prediction(plan: PredictionPlan, *, dry_run: bool) -> None:
    print(f"\n  ━━ publish {plan.airport}: {plan.pred_dir}")
    _run_steps(
        f"{plan.airport} · {plan.training.model} · {plan.training.mode}",
        plan.steps(),
        dry_run=dry_run,
    )


def _parse_csv(raw: str, allowed: tuple[str, ...], flag: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    unknown = [token for token in tokens if token not in allowed]
    if unknown or not tokens:
        raise argparse.ArgumentTypeError(f"{flag} takes a comma list from {allowed}, got {raw!r}")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default="per-airport")
    parser.add_argument("--airport", default=None,
                        help="optional single airport filter; otherwise discover every K-airport")
    parser.add_argument("--models", type=lambda raw: _parse_csv(raw, MODELS, "--models"),
                        default=MODELS, metavar=",".join(MODELS))
    parser.add_argument("--modes", type=lambda raw: _parse_csv(raw, HORIZON_MODES, "--modes"),
                        default=HORIZON_MODES, metavar=",".join(HORIZON_MODES))
    parser.add_argument("--outputs", type=lambda raw: _parse_csv(raw, OUTPUT_KINDS, "--outputs"),
                        default=OUTPUT_KINDS, metavar="czml,eval")
    parser.add_argument("--split", choices=("test", "val", "train", "all"), default="test")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--aircraft-type", default=None)
    parser.add_argument("--coordinate-frame", choices=COORDINATE_FRAMES, default="enu")
    parser.add_argument("--batch-size", default="auto",
                        help="positive integer or auto (default: actual CUDA training-step probe)")
    parser.add_argument("--samples-per-epoch", type=int, default=None,
                        help=f"pooled final-training budget (default: {DEFAULT_POOLED_SAMPLES_PER_EPOCH})")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-trials", type=int, default=4)
    parser.add_argument("--cv-epochs", type=int, default=12)
    parser.add_argument("--cv-patience", type=int, default=4)
    parser.add_argument("--cv-samples-per-epoch", type=int,
                        default=DEFAULT_CV_SAMPLES_PER_EPOCH)
    parser.add_argument("--skip-cv", action="store_true",
                        help="reuse matching CV artifacts when present, otherwise use base defaults")
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse a checkpoint only when all selected manifest digests match")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_size != "auto":
        try:
            if int(args.batch_size) <= 0:
                raise ValueError
        except ValueError:
            parser.error("--batch-size must be a positive integer or 'auto'")

    if args.airport:
        airports = [args.airport.strip().upper()]
    else:
        airports = discover_k_airports()
        if not airports:
            parser.error(f"no K-prefixed airports with arrivals manifests under {HARVEST_ROOT}")

    scopes = (
        [tuple(airports)]
        if args.training_mode == "pooled"
        else [(airport,) for airport in airports]
    )
    cells = [
        (scope, model, mode)
        for scope in scopes
        for model in args.models
        for mode in args.modes
    ]
    print(f"{len(cells)} training cell(s), mode={args.training_mode}, airports={','.join(airports)}")

    completed = 0
    for scope, model, mode in cells:
        training = TrainingPlan(
            scope,
            model,
            mode,
            training_mode=args.training_mode,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            aircraft_type=args.aircraft_type,
            coordinate_frame=args.coordinate_frame,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            cv_folds=args.cv_folds,
            cv_trials=args.cv_trials,
            cv_epochs=args.cv_epochs,
            cv_patience=args.cv_patience,
            cv_samples_per_epoch=args.cv_samples_per_epoch,
        )
        if not run_training(
            training,
            dry_run=args.dry_run,
            skip_cv=args.skip_cv,
            skip_train=args.skip_train,
        ):
            continue
        for airport in scope:
            run_prediction(
                PredictionPlan(training, airport, tuple(args.outputs), split=args.split),
                dry_run=args.dry_run,
            )
        completed += 1

    verb = "previewed" if args.dry_run else "completed"
    print(f"\n✓ {verb} {completed}/{len(cells)} training cell(s) "
          f"[CV={'skip/reuse' if args.skip_cv else 'run'}, split={args.split}]")


if __name__ == "__main__":
    main()
