#!/usr/bin/env python
"""Leak-free pooled ENU vs runway-aligned coordinate-frame ablation.

The experiment runs two otherwise identical cross-validation searches over outer-train,
checks their split/fold/candidate identities, selects the coordinate frame and hyperparameters
by CV airport-macro loss, and only then trains and evaluates the selected final model. The
losing frame is never trained on outer-validation and is never evaluated on outer-test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_ts_pipeline as pipeline

FRAMES = ("enu", "runway-aligned")
RESULT_SCHEMA = "ts-coordinate-frame-ablation-v5-flight-epoch-airport-macro"
RESULT_NAME = "coordinate_frame_ablation.json"


class AblationContractError(RuntimeError):
    """Raised before final training when the two CV searches are not comparable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def refuse_repeated_test(result_path: Path) -> None:
    """Keep a completed or partially started outer-test evaluation one-shot."""
    if not result_path.is_file():
        return
    try:
        previous = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationContractError(
            f"cannot audit existing ablation result {result_path}: {exc}"
        ) from exc
    if not isinstance(previous, dict) or previous.get("schema_version") != RESULT_SCHEMA:
        raise AblationContractError(
            f"existing ablation result {result_path} has an unsupported schema"
        )
    guard = previous.get("leakage_guard", {})
    if guard.get("outer_test_evaluation_started") or guard.get("outer_test_evaluated"):
        raise AblationContractError(
            "outer-test evaluation already started for this experiment directory; refusing "
            "to expose it again. Preserve this result and use a new --output-dir for a new "
            "preregistered experiment"
        )


def _load_cv(plan: pipeline.TrainingPlan) -> dict[str, Any]:
    try:
        result = json.loads(plan.cv_results.read_text(encoding="utf-8"))
        best_config = json.loads(plan.best_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationContractError(
            f"cannot read {plan.coordinate_frame} CV artifacts under {plan.cv_dir}: {exc}"
        ) from exc
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != pipeline.CV_RESULTS_SCHEMA
    ):
        raise AblationContractError(
            f"{plan.coordinate_frame} CV result has an unsupported schema"
        )
    if result.get("best_overrides") != best_config:
        raise AblationContractError(
            f"{plan.coordinate_frame} best_config.json disagrees with cv_results.json"
        )
    return result


def _fold_signature(result: dict[str, Any]) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for candidate in result.get("candidates", []):
        signature.append({
            "candidate": candidate.get("candidate"),
            "overrides": candidate.get("overrides"),
            "folds": [
                {
                    "fold": fold.get("fold"),
                    "train_flights": fold.get("train_flights"),
                    "validation_flights": fold.get("validation_flights"),
                    "validation_by_airport": fold.get("validation_by_airport"),
                    "validation_split_sha256": fold.get("validation_split_sha256"),
                    "batch_size": fold.get("batch_size"),
                }
                for fold in candidate.get("folds", [])
            ],
        })
    return signature


def _config_without_frame(result: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(result.get("base_config"))
    if not isinstance(config, dict):
        raise AblationContractError("CV result is missing base_config")
    config.pop("coordinate_frame", None)
    return config


def _assert_requested_result(
    plan: pipeline.TrainingPlan,
    result: dict[str, Any],
) -> None:
    guard = result.get("leakage_guard")
    if guard != {
        "search_population": "outer_train_only",
        "outer_validation_used": False,
        "outer_test_used": False,
    }:
        raise AblationContractError(
            f"{plan.coordinate_frame} CV result does not prove outer-train-only isolation"
        )
    expected_manifests = pipeline._manifest_digests(plan.airports)
    if result.get("arrival_manifests") != expected_manifests:
        raise AblationContractError(
            f"{plan.coordinate_frame} CV result does not match the current arrival manifests"
        )
    expected_budget = {
        "n_splits": plan.cv_folds,
        "search_strategy": "exhaustive_grid",
        "tuned_parameters": list(plan.cv_parameters),
        "parameter_grid": pipeline.parameter_grid(plan.cv_parameters),
        "cv_epochs": plan.cv_epochs,
        "cv_patience": plan.cv_patience,
    }
    for field, expected in expected_budget.items():
        if result.get(field) != expected:
            raise AblationContractError(
                f"{plan.coordinate_frame} CV {field}={result.get(field)!r}, expected {expected!r}"
            )
    config = result.get("base_config")
    expected_config = {
        "model": plan.model,
        "n_segments": plan._expected_cv_base_config()["n_segments"],
        "coordinate_frame": plan.coordinate_frame,
        "random_train_anchor": plan.random_train_anchor,
    }
    if plan.seed is not None:
        expected_config["seed"] = plan.seed
    if not isinstance(config, dict) or any(
        config.get(field) != expected for field, expected in expected_config.items()
    ):
        raise AblationContractError(
            f"{plan.coordinate_frame} CV base configuration does not match this experiment"
        )
    score = result.get("best_mean_val_macro_loss")
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise AblationContractError(
            f"{plan.coordinate_frame} CV has no finite best validation loss"
        )


def assert_comparable(
    plans: dict[str, pipeline.TrainingPlan],
    results: dict[str, dict[str, Any]],
) -> None:
    """Refuse selection unless every non-frame experimental control is identical."""
    for frame in FRAMES:
        _assert_requested_result(plans[frame], results[frame])

    enu, aligned = results["enu"], results["runway-aligned"]
    scalar_fields = (
        "selection_metric", "arrival_manifests", "outer_split",
        "n_splits", "search_strategy", "tuned_parameters", "parameter_grid",
        "candidate_count", "cv_epochs", "cv_patience",
    )
    for field in scalar_fields:
        if enu.get(field) != aligned.get(field):
            raise AblationContractError(
                f"CV comparison rejected: {field} differs between coordinate frames"
            )
    if _config_without_frame(enu) != _config_without_frame(aligned):
        raise AblationContractError(
            "CV comparison rejected: base configurations differ beyond coordinate_frame"
        )
    if _fold_signature(enu) != _fold_signature(aligned):
        raise AblationContractError(
            "CV comparison rejected: candidates, folds, sampling batch sizes, or split "
            "digests differ"
        )


def select_winner(results: dict[str, dict[str, Any]]) -> str:
    """Select by CV only; deterministic ties retain the unchanged ENU baseline."""
    return min(
        FRAMES,
        key=lambda frame: (results[frame]["best_mean_val_macro_loss"], FRAMES.index(frame)),
    )


def verify_final_checkpoint(
    plan: pipeline.TrainingPlan,
    cv_result: dict[str, Any],
) -> dict[str, Any]:
    """Prove the final fit rebuilt the same outer split before test is exposed."""
    try:
        metadata = json.loads(plan.checkpoint_metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationContractError(
            f"cannot verify final checkpoint metadata {plan.checkpoint_metadata}: {exc}"
        ) from exc
    expected_splits = {
        "train": cv_result["outer_split"]["train_sha256"],
        "val": cv_result["outer_split"]["validation_sha256"],
        "test": cv_result["outer_split"]["test_sha256"],
    }
    if metadata.get("schema_version") != pipeline.CHECKPOINT_METADATA_SCHEMA:
        raise AblationContractError("final checkpoint metadata has the wrong schema")
    if metadata.get("checkpoint_sha256") != _sha256(plan.checkpoint):
        raise AblationContractError("final checkpoint failed SHA-256 verification")
    if metadata.get("arrival_manifests") != cv_result["arrival_manifests"]:
        raise AblationContractError(
            "arrival manifests changed between CV selection and final training"
        )
    if metadata.get("random_train_anchor") != plan.random_train_anchor:
        raise AblationContractError(
            "final checkpoint anchor policy differs from the selected experiment"
        )
    if metadata.get("split_sha256") != expected_splits:
        raise AblationContractError(
            "final train/validation/test split differs from the selected CV outer split"
        )
    return metadata


def _run(label: str, command: list[str], *, dry_run: bool) -> None:
    print(f"\n=== {label} ===\n{' '.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=pipeline.REPO_ROOT, check=True)


def _parse_airports(raw: str) -> tuple[str, ...]:
    airports = tuple(sorted({token.strip().upper() for token in raw.split(",") if token.strip()}))
    if not airports:
        raise argparse.ArgumentTypeError("--airports requires at least one ICAO code")
    return airports


def _parse_outputs(raw: str) -> tuple[str, ...]:
    outputs = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not outputs or any(output not in pipeline.OUTPUT_KINDS for output in outputs):
        raise argparse.ArgumentTypeError(
            f"--outputs takes a comma list from {pipeline.OUTPUT_KINDS}"
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports", type=_parse_airports, default=None,
                        help="comma-separated fixed airport roster; default: all discovered K-airports")
    parser.add_argument("--model", choices=pipeline.MODELS, default="itransformer")
    parser.add_argument("--n-segments", type=int, default=None,
                        help="base N for normalized progress; CV also tunes N")
    parser.add_argument("--outputs", type=_parse_outputs, default=("eval",),
                        help="test publication outputs: eval, czml, or eval,czml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="final-training epoch cap (early stopping still applies)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--aircraft-type", default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--cv-parameters",
        type=lambda raw: pipeline._parse_csv(
            raw, tuple(pipeline.CV_PARAMETER_GRIDS), "--cv-parameters"
        ),
        default=pipeline.DEFAULT_CV_PARAMETERS,
        metavar=",".join(pipeline.DEFAULT_CV_PARAMETERS),
        help="parameters included in the exhaustive CV grid",
    )
    parser.add_argument("--cv-epochs", type=int, default=pipeline.DEFAULT_CV_EPOCHS)
    parser.add_argument("--cv-patience", type=int, default=pipeline.DEFAULT_CV_PATIENCE)
    parser.add_argument(
        "--random-train-anchor",
        action="store_true",
        help="use one random valid anchor per flight and epoch; default is fixed L-1",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="experiment root; defaults under 4dTrajectory/outputs/POOLED")
    parser.add_argument("--reuse-cv", action="store_true",
                        help="reuse both CV artifacts only after the full contract check")
    parser.add_argument("--skip-test", action="store_true",
                        help="stop after selecting and training; outer-test remains untouched")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_size != "auto":
        try:
            if int(args.batch_size) <= 0:
                raise ValueError
        except ValueError:
            parser.error("--batch-size must be a positive integer or 'auto'")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    for name in ("cv_epochs", "cv_patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")

    airports = args.airports or tuple(pipeline.discover_k_airports())
    if not airports:
        parser.error(f"no K-prefixed airports with arrivals manifests under {pipeline.HARVEST_ROOT}")
    missing = [airport for airport in airports if not pipeline.arrival_manifest_path(airport).is_file()]
    if missing:
        parser.error(f"missing arrivals manifest for {missing[0]}")

    experiment_dir = (args.output_dir or (
        pipeline.OPT_OUTPUTS_ROOT / "POOLED" /
        f"ts_{args.model}_normalized_time_coordinate_frame_ablation"
    )).resolve()
    result_path = experiment_dir / RESULT_NAME
    if not args.dry_run:
        try:
            refuse_repeated_test(result_path)
        except AblationContractError as exc:
            parser.error(str(exc))
    plans = {
        frame: pipeline.TrainingPlan(
            airports,
            args.model,
            training_mode="pooled",
            n_segments=args.n_segments,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            aircraft_type=args.aircraft_type,
            coordinate_frame=frame,
            batch_size=args.batch_size,
            cv_folds=args.cv_folds,
            cv_parameters=args.cv_parameters,
            cv_epochs=args.cv_epochs,
            cv_patience=args.cv_patience,
            random_train_anchor=args.random_train_anchor,
            output_dir=experiment_dir / frame,
        )
        for frame in FRAMES
    }

    print(f"coordinate-frame ablation: model={args.model}, normalized time")
    print(f"airports (locked): {','.join(airports)}")
    print(f"seed={args.seed}, folds={args.cv_folds}, "
          f"CV parameters={','.join(args.cv_parameters)}")
    print(f"experiment: {experiment_dir}")

    if args.dry_run:
        for frame in FRAMES:
            label, command = plans[frame].cv_step()
            _run(f"{frame} · {label}", command, dry_run=True)
        print("\n=== selection gate ===")
        print("compare locked outer split, folds, candidates, budgets, then select by CV loss")
        print("=== selected frame only · final train -> per-airport test evaluation ===")
        return 0

    if not args.reuse_cv:
        for frame in FRAMES:
            label, command = plans[frame].cv_step()
            _run(f"{frame} · {label}", command, dry_run=False)

    results = {frame: _load_cv(plans[frame]) for frame in FRAMES}
    assert_comparable(plans, results)
    winner = select_winner(results)
    selected = plans[winner]
    scores = {
        frame: results[frame]["best_mean_val_macro_loss"] for frame in FRAMES
    }
    decision: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "selected",
        "selected_at": _utc_now(),
        "selection_population": "outer_train_cv_only",
        "selection_metric": results[winner]["selection_metric"],
        "tie_break": "enu baseline wins an exact tie",
        "selected_coordinate_frame": winner,
        "selected_hyperparameters": results[winner]["best_overrides"],
        "cv_scores": scores,
        "controls": {
            "airports": list(airports),
            "model": args.model,
            "prediction_time": "normalized",
            "base_n_segments": args.n_segments,
            "seed": args.seed,
            "folds": args.cv_folds,
            "cv_parameters": list(args.cv_parameters),
            "cv_epochs": args.cv_epochs,
            "cv_patience": args.cv_patience,
            "batch_size": args.batch_size,
            "outer_split": results[winner]["outer_split"],
            "arrival_manifests": results[winner]["arrival_manifests"],
        },
        "cv_artifacts": {
            frame: {
                "path": str(plans[frame].cv_results),
                "sha256": _sha256(plans[frame].cv_results),
            }
            for frame in FRAMES
        },
        "leakage_guard": {
            "outer_validation_used_for_selection": False,
            "outer_test_used_for_selection": False,
            "losing_frame_trained_on_outer_validation": False,
            "outer_test_evaluation_started": False,
            "outer_test_evaluated": False,
        },
    }
    _write_json_atomic(result_path, decision)
    print(f"\n✓ selected {winner} from outer-train CV only: {scores}")

    label, command = selected.train_step(use_best_config=True)
    _run(f"{winner} · {label}", command, dry_run=False)
    checkpoint_metadata = verify_final_checkpoint(selected, results[winner])
    decision["status"] = "trained"
    decision["trained_at"] = _utc_now()
    decision["checkpoint"] = {
        "path": str(selected.checkpoint),
        "sha256": _sha256(selected.checkpoint),
        "split_sha256": checkpoint_metadata["split_sha256"],
    }
    _write_json_atomic(result_path, decision)

    if args.skip_test:
        print(f"✓ final model trained; outer-test remains untouched. Decision: {result_path}")
        return 0

    decision["status"] = "testing"
    decision["test_started_at"] = _utc_now()
    decision["leakage_guard"]["outer_test_evaluation_started"] = True
    _write_json_atomic(result_path, decision)
    test_outputs: dict[str, Any] = {}
    for airport in airports:
        prediction = pipeline.PredictionPlan(
            selected,
            airport,
            tuple(args.outputs),
            split="test",
            experiment_tag="coordinate_frame_ablation",
        )
        pipeline.run_prediction(prediction, dry_run=False)
        test_outputs[airport] = {
            "prediction_dir": str(prediction.pred_dir),
            "evaluation_report": str(prediction.report),
        }
    decision["status"] = "complete"
    decision["tested_at"] = _utc_now()
    decision["test_outputs"] = test_outputs
    decision["leakage_guard"]["outer_test_evaluated"] = True
    _write_json_atomic(result_path, decision)
    print(f"\n✓ ablation complete; test was exposed only after selection: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
