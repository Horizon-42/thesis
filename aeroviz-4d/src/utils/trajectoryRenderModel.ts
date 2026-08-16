/**
 * trajectoryRenderModel.ts
 * ------------------------
 * Pure helpers that decide HOW the (potentially ~1000) trajectory entities render,
 * so the load hooks stay thin and the policy stays unit-testable.
 *
 * Rendering policy (the same for observed tracks and the prediction comparison):
 *   • every SHOWN trajectory draws as a uniform-width coloured path;
 *   • only a capped SUBSET also carries an aircraft glTF model (drawing ~1000 models
 *     is what makes the scene lag), with glTF animation OFF;
 *   • the text label is drawn for the selected/tracked flight only (labels at scale
 *     are the single biggest Cesium cost).
 *
 * `planTrajectoryModels` is the only decision with a choice in it, so it is the part
 * we test; width/animation/label are constants applied directly in the hooks.
 */

import { sampleSubset, type Rng } from "./sampleTrajectories";
import type { ComparisonKind } from "../context/AppContext";
import type { ComparisonStatusLegend } from "./comparisonLegend";
import { OBSERVED_VERDICT_COLORS } from "./observedVerdictColors";

/** Uniform path width for every trajectory (observed + comparison), in px. */
export const TRAJECTORY_PATH_WIDTH = 2;

/**
 * The single source of truth for the prediction-comparison kind colours. Both the legend
 * base colours. Prediction outcomes override the purple fallback with the shared Baseline
 * verdict palette; its checkbox uses a green/red/gray split swatch. The CZML bakes its own
 * colours in, but they vary by category, so the frontend owns the final contract.
 * References are always white. Optimizer paths keep the CZML-baked verdict
 * colour for an off-target group (the builder bakes the simulator/result path bright yellow —
 * `OFF_TARGET_COLOR` — because the marking belongs on the trajectory that missed the
 * target). Keep these RGB values in sync with
 * `python/build_scenario_comparison_czml.py` for the optimizer and lookback base colours.
 */
export const COMPARISON_KIND_COLORS: Record<ComparisonKind, string> = {
  reference: "rgb(235, 235, 235)",
  optimizer: "rgb(255, 140, 0)", // "Optimize states"
  simulator: "rgb(40, 120, 255)", // "Optimize results"
  predicted: "rgb(170, 90, 230)", // fallback when no terminal verdict is available
  // The forecast's input window is purple and faded; unlike the evaluated output, it does
  // not carry pass/fail semantics.
  lookback: "rgb(170, 90, 230)",
};

/**
 * Per-kind path/label alpha. Everything renders at the same opacity as the CZML bakes in
 * (~220/255) except the lookback, which is observed input rather than a result and is faded
 * so a viewer can see at a glance where the model stopped being told and started guessing.
 */
export const COMPARISON_KIND_ALPHA: Record<ComparisonKind, number> = {
  reference: 200 / 255,
  optimizer: 220 / 255,
  simulator: 220 / 255,
  predicted: 225 / 255,
  lookback: 85 / 255,
};

/** Checkbox swatches describe the rendered path. Prediction has outcome colours, not one hue. */
export function comparisonKindSwatch(kind: ComparisonKind): string {
  if (kind !== "predicted") return COMPARISON_KIND_COLORS[kind];
  return `linear-gradient(90deg, ${OBSERVED_VERDICT_COLORS.pass} 0 46%, ` +
    `${OBSERVED_VERDICT_COLORS.fail} 46% 92%, ${OBSERVED_VERDICT_COLORS.undecided} 92% 100%)`;
}

/**
 * Outcome colours that override a kind's normal colour. Prediction uses the exact same
 * pass/fail/undecided palette as Baseline; optimizer replay keeps its established yellow
 * off-target result colour.
 *
 * Keep the optimizer result entry in sync with build_scenario_comparison_czml.py's
 * OFF_TARGET_COLOR.
 */
export const COMPARISON_STATUS_STYLES: Record<
  ComparisonStatusLegend,
  { label: string; color: string; alpha: number }
> = {
  offTargetResult: {
    label: "Off-target optimize result",
    color: "rgb(255, 205, 40)",
    alpha: 235 / 255,
  },
  predictionPass: {
    label: "Prediction pass",
    color: OBSERVED_VERDICT_COLORS.pass,
    alpha: COMPARISON_KIND_ALPHA.predicted,
  },
  predictionFail: {
    label: "Prediction fail",
    color: OBSERVED_VERDICT_COLORS.fail,
    alpha: COMPARISON_KIND_ALPHA.predicted,
  },
  predictionIndeterminate: {
    label: "Prediction indeterminate",
    color: OBSERVED_VERDICT_COLORS.undecided,
    alpha: COMPARISON_KIND_ALPHA.predicted,
  },
};

/** How many shown trajectories carry an aircraft model (the rest are path-only). */
export const DEFAULT_MODEL_BUDGET = 20;

export interface RenderModelPlan {
  /** Ids of the shown entities that should carry an aircraft glTF model. */
  modelIds: Set<string>;
}

/**
 * Choose which of the SHOWN entities get an aircraft model.
 *
 * The selected/tracked flight always gets one (so the followed aircraft is always
 * visible); the remaining slots up to `budget` are filled by a uniform random sample
 * of the other shown entities. When fewer entities are shown than the budget, they all
 * get a model. `budget` is the maximum total (the selected flight counts toward it).
 *
 * Pure → deterministic under a seeded `rng`.
 */
export function planTrajectoryModels(
  shownIds: string[],
  selectedFlightId: string | null,
  budget: number = DEFAULT_MODEL_BUDGET,
  rng: Rng = Math.random,
): RenderModelPlan {
  const modelIds = new Set<string>();

  if (selectedFlightId !== null && shownIds.includes(selectedFlightId)) {
    modelIds.add(selectedFlightId);
  }

  const remaining = budget - modelIds.size;
  if (remaining > 0) {
    const candidates = shownIds.filter((id) => !modelIds.has(id));
    for (const id of sampleSubset(candidates, remaining, rng)) {
      modelIds.add(id);
    }
  }

  return { modelIds };
}
