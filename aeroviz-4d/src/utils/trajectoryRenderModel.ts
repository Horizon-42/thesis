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

/** Uniform path width for every trajectory (observed + comparison), in px. */
export const TRAJECTORY_PATH_WIDTH = 2;

/**
 * The single source of truth for the prediction-comparison kind colours. Both the legend
 * swatches (ControlPanel) and the rendered path/label colours (useComparisonTrajectoryLayer)
 * read this, so they can never drift. The CZML bakes its own colours in, but they vary by
 * category and don't necessarily match the legend, so the frontend overrides each opt/sim
 * path to these. Exceptions keep the CZML-baked verdict colours: the reference always
 * (white / dark-red `FAILED_COLOR` / dark-amber `OFF_TARGET_REF_COLOR`), and every path
 * of an off-target group (the builder bakes the simulator/result path bright yellow —
 * `OFF_TARGET_COLOR` — because the marking belongs on the trajectory that missed the
 * target). Keep these RGB values in sync with
 * `python/build_scenario_comparison_czml.py` (OPTIMIZER_COLOR / SIMULATOR_COLOR /
 * PREDICTION_COLOR / LOOKBACK_COLOR).
 */
export const COMPARISON_KIND_COLORS: Record<ComparisonKind, string> = {
  reference: "rgb(235, 235, 235)",
  optimizer: "rgb(255, 140, 0)", // "Optimize states"
  simulator: "rgb(40, 120, 255)", // "Optimize results"
  predicted: "rgb(170, 90, 230)", // "Predicted" — ts_transformer forecast
  // The forecast's INPUT window, deliberately the same hue as the forecast: it is one
  // continuous track, and what distinguishes the given half from the predicted half is the
  // alpha (COMPARISON_KIND_ALPHA), not the colour.
  lookback: "rgb(170, 90, 230)",
};

/**
 * Per-kind path/label alpha. Everything renders at the same opacity as the CZML bakes in
 * (~220/255) except the lookback, which is observed input rather than a result and is faded
 * so a viewer can see at a glance where the model stopped being told and started guessing.
 */
export const COMPARISON_KIND_ALPHA: Record<ComparisonKind, number> = {
  reference: 0.86,
  optimizer: 0.86,
  simulator: 0.86,
  predicted: 0.86,
  lookback: 0.33,
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
