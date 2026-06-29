/**
 * trajectoryRenderModel.ts
 * ------------------------
 * Pure helpers that decide HOW the (potentially ~1000) trajectory entities render,
 * so the load hooks stay thin and the policy stays unit-testable.
 *
 * Rendering policy (the same for observed tracks and the optimizer comparison):
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

/** Uniform path width for every trajectory (observed + comparison), in px. */
export const TRAJECTORY_PATH_WIDTH = 2;

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
