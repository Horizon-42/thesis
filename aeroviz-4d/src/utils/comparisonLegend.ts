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
  | "failedReference"
  | "offTargetReference"
  | "offTargetResult";

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
  let hasFailedReference = false;
  let hasOffTargetReference = false;
  let hasOffTargetResult = false;

  for (const group of groups) {
    const groupKinds = new Set(
      group.entities
        .map(entityKind)
        .filter((kind): kind is ComparisonKind => kind !== null),
    );
    groupKinds.forEach((kind) => availableKinds.add(kind));

    if (group.status === "failed" && groupKinds.has("reference")) {
      hasFailedReference = true;
    }
    if (group.status === "offTarget" && groupKinds.has("reference")) {
      hasOffTargetReference = true;
    }
    // Only the simulator/result schema bakes the yellow verdict colour. Prediction paths
    // remain purple even when their index status is offTarget.
    if (group.status === "offTarget" && groupKinds.has("simulator")) {
      hasOffTargetResult = true;
    }
  }

  const statuses: ComparisonStatusLegend[] = [];
  if (hasFailedReference) statuses.push("failedReference");
  if (hasOffTargetReference) statuses.push("offTargetReference");
  if (hasOffTargetResult) statuses.push("offTargetResult");

  return {
    kinds: DISPLAY_KIND_ORDER.filter((kind) => availableKinds.has(kind)),
    statuses,
  };
}
