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
  | "predictionOtherRunway"
  | "predictionFail"
  | "predictionIndeterminate";

/** Optimizer state sequences are intentionally not user-facing comparison paths. */
export type ComparisonLegendKind = Exclude<ComparisonKind, "optimizer">;

/**
 * A traffic category (AV46, AV47): `windows` — M1, one controlled aircraft in recorded traffic per window — or `scene` — M2,
 * every aircraft of a block controlled, the recorded traffic being what is outside the scheduled set.
 */
export type ComparisonTrafficLegend = "windows" | "scene";

export interface ComparisonLegendModel {
  /** User-toggleable path kinds present in the selected category/runway. */
  kinds: ComparisonLegendKind[];
  /** Outcome colours that override the base kind colour in those groups. */
  statuses: ComparisonStatusLegend[];
  /** Set when the index is a traffic category: the legend then names who is controlled and who is not. */
  traffic: ComparisonTrafficLegend | null;
}

const KIND_LABELS: Record<ComparisonLegendKind, string> = {
  reference: "Reference",
  simulator: "Optimize results",
  predicted: "Predicted",
  lookback: "Predictor input",
};

/** In a traffic category the reference is the controlled aircraft's own record, the result its optimized path. */
const TRAFFIC_KIND_LABELS: Record<ComparisonLegendKind, string> = {
  ...KIND_LABELS,
  reference: "Controlled aircraft — its record",
  simulator: "Controlled aircraft — optimized path",
};

export function comparisonKindLabel(kind: ComparisonLegendKind, traffic: ComparisonTrafficLegend | null): string {
  return (traffic === null ? KIND_LABELS : TRAFFIC_KIND_LABELS)[kind];
}

/**
 * In a traffic category the optimized path turns yellow where the aircraft's final state missed the runway-threshold target
 * (the evaluation gates failed, status `offTarget`): said beside the optimized-path entry, not apart under "Outcome colours".
 */
export const TRAFFIC_OFF_TARGET_LABEL = "yellow: the optimized path ended off the runway-threshold target (failed the gates)";

/** The pink aircraft: recorded, and not controlled (an M2 scene shows those outside the scheduled set). */
export function recordedTrafficLabel(traffic: ComparisonTrafficLegend): string {
  return traffic === "scene"
    ? "Recorded traffic — not controlled, outside the scheduled set"
    : "Recorded traffic — not controlled";
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
  // a scene is drawn whole, whatever the runway selector says
  const groups = index.groups.filter(
    (group) => selectedRunway === null || index.scene !== undefined || group.runway === selectedRunway,
  );
  const availableKinds = new Set<ComparisonKind>();
  let hasOffTargetResult = false;
  let hasPredictionPass = false;
  let hasPredictionOtherRunway = false;
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
      if (group.status === "otherRunway") hasPredictionOtherRunway = true;
      if (group.status === "offTarget") hasPredictionFail = true;
      if (group.status === "indeterminate") hasPredictionIndeterminate = true;
    }
  }

  const statuses: ComparisonStatusLegend[] = [];
  if (hasOffTargetResult) statuses.push("offTargetResult");
  if (hasPredictionPass) statuses.push("predictionPass");
  if (hasPredictionOtherRunway) statuses.push("predictionOtherRunway");
  if (hasPredictionFail) statuses.push("predictionFail");
  if (hasPredictionIndeterminate) statuses.push("predictionIndeterminate");

  return {
    kinds: DISPLAY_KIND_ORDER.filter((kind) => availableKinds.has(kind)),
    statuses,
    traffic: index.scene !== undefined ? "scene" : groups.some((group) => group.traffic !== undefined) ? "windows" : null,
  };
}
