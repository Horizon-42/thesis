/**
 * comparisonLegend.ts
 * -------------------
 * Builds the prediction-comparison legend from the selected category's committed index.
 * The index is the authority for which paths and outcome colours the category can draw;
 * category names are intentionally not used as schema guesses.
 */

import type { ComparisonIndex } from "../data/airportData";
import type { ComparisonKind } from "../context/AppContext";

export type ComparisonStatusLegend =
  | "offTargetResult"
  | "predictionPass"
  | "predictionFail"
  | "predictionIndeterminate";

/** Optimizer state sequences are intentionally not user-facing comparison paths. */
export type ComparisonLegendKind = Exclude<ComparisonKind, "optimizer">;

export interface ComparisonLegendModel {
  /** User-toggleable path kinds present in the selected category/runway. */
  kinds: ComparisonLegendKind[];
  /** Outcome colours that override the base kind colour in those groups. */
  statuses: ComparisonStatusLegend[];
}

const DISPLAY_KIND_ORDER: ComparisonLegendKind[] = [
  "reference",
  "simulator",
  "predicted",
  "lookback",
];

function entityKind(entityId: string): ComparisonKind | null {
  if (entityId.startsWith("ref-")) return "reference";
  if (entityId.startsWith("opt-")) return "optimizer";
  if (entityId.startsWith("sim-")) return "simulator";
  if (entityId.startsWith("pred-")) return "predicted";
  if (entityId.startsWith("look-")) return "lookback";
  return null;
}

/**
 * Derive the visible legend from the exact groups eligible for the runway selector.
 *
 * Optimizer state sequences (`opt-`) are deliberately omitted: they remain internal
 * solver diagnostics and are currently hidden from the comparison view. A result path
 * (`sim-`) is the user-facing optimized trajectory.
 */
export function buildComparisonLegend(
  index: ComparisonIndex,
  selectedRunway: string | null,
): ComparisonLegendModel {
  const groups = index.groups.filter(
    (group) => selectedRunway === null || group.runway === selectedRunway,
  );
  const availableKinds = new Set<ComparisonKind>();
  let hasOffTargetResult = false;
  let hasPredictionPass = false;
  let hasPredictionFail = false;
  let hasPredictionIndeterminate = false;

  for (const group of groups) {
    const groupKinds = new Set(
      group.entities
        .map(entityKind)
        .filter((kind): kind is ComparisonKind => kind !== null),
    );
    groupKinds.forEach((kind) => availableKinds.add(kind));

    // Optimizer replay paths retain their yellow off-target result convention.
    if (group.status === "offTarget" && groupKinds.has("simulator")) {
      hasOffTargetResult = true;
    }
    if (groupKinds.has("predicted")) {
      if (group.status === "solved") hasPredictionPass = true;
      if (group.status === "offTarget") hasPredictionFail = true;
      if (group.status === "indeterminate") hasPredictionIndeterminate = true;
    }
  }

  const statuses: ComparisonStatusLegend[] = [];
  if (hasOffTargetResult) statuses.push("offTargetResult");
  if (hasPredictionPass) statuses.push("predictionPass");
  if (hasPredictionFail) statuses.push("predictionFail");
  if (hasPredictionIndeterminate) statuses.push("predictionIndeterminate");

  return {
    kinds: DISPLAY_KIND_ORDER.filter((kind) => availableKinds.has(kind)),
    statuses,
  };
}
